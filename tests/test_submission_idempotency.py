from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vfieval.db import Database
from vfieval.server import _create_registered_run, _create_run_from_files
from vfieval.submissions import SubmissionConflict, idempotent_create
from vfieval.video_selection_tokens import VideoSelectionTokenExpired


class SubmissionIdempotencyTests(unittest.TestCase):
    def _database(self, root: Path) -> Database:
        db = Database(root / "vfieval.sqlite")
        db.init()
        return db

    @staticmethod
    def _create(
        db: Database,
        body: dict[str, object],
        calls: list[int],
        resources: dict[int, dict[str, object]],
    ) -> dict[str, object]:
        def create() -> dict[str, object]:
            resource_id = len(calls) + 1
            calls.append(resource_id)
            resource = {"id": resource_id, "name": body["name"]}
            resources[resource_id] = resource
            return resource

        return idempotent_create(
            db,
            scope="test-resource",
            body=body,
            resource_type="test-resource",
            create=create,
            resource_id=lambda resource: int(resource["id"]),
            replay=lambda resource_id: dict(resources[resource_id]),
            mark_replay=lambda resource, replayed, submission_id: {
                **resource,
                "submission_id": submission_id,
                "idempotent_replay": replayed,
            },
        )

    def test_same_submission_and_payload_replays_original_resource(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._database(Path(tmp))
            calls: list[int] = []
            resources: dict[int, dict[str, object]] = {}
            body = {
                "submission_id": "submission-0001",
                "name": "first",
                "preflight_token": "short-lived-one",
            }

            first = self._create(db, body, calls, resources)
            replay = self._create(
                db,
                {**body, "preflight_token": "short-lived-two"},
                calls,
                resources,
            )

            self.assertEqual(calls, [1])
            self.assertEqual(first["id"], replay["id"])
            self.assertFalse(first["idempotent_replay"])
            self.assertTrue(replay["idempotent_replay"])

    def test_same_submission_with_different_payload_is_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._database(Path(tmp))
            calls: list[int] = []
            resources: dict[int, dict[str, object]] = {}
            self._create(
                db,
                {"submission_id": "submission-0002", "name": "first"},
                calls,
                resources,
            )

            with self.assertRaisesRegex(
                SubmissionConflict,
                "different request payload",
            ) as raised:
                self._create(
                    db,
                    {"submission_id": "submission-0002", "name": "changed"},
                    calls,
                    resources,
                )

            self.assertEqual(raised.exception.code, "submission_payload_conflict")
            self.assertEqual(calls, [1])

    def test_failed_creation_releases_reservation_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._database(Path(tmp))
            body = {"submission_id": "submission-0003", "name": "retry"}

            def fail() -> dict[str, object]:
                raise RuntimeError("creation failed")

            with self.assertRaisesRegex(RuntimeError, "creation failed"):
                idempotent_create(
                    db,
                    scope="test-resource",
                    body=body,
                    resource_type="test-resource",
                    create=fail,
                    resource_id=lambda resource: int(resource["id"]),
                    replay=lambda resource_id: {"id": resource_id},
                    mark_replay=lambda resource, _replayed, _submission_id: resource,
                )

            calls: list[int] = []
            resources: dict[int, dict[str, object]] = {}
            created = self._create(db, body, calls, resources)
            self.assertEqual(created["id"], 1)
            self.assertEqual(calls, [1])

    def test_database_migration_creates_submission_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = self._database(Path(tmp))
            tables = {
                str(row["name"])
                for row in db.query(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            self.assertIn("submission_keys", tables)
            columns = {
                str(row["name"])
                for row in db.query("PRAGMA table_info(submission_keys)")
            }
            self.assertIn("lease_expires_at", columns)

    def test_run_creation_helper_replays_one_persisted_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = self._database(root)
            model_id = db.register_model("model", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("dataset", str(root), True)
            body = {
                "submission_id": "run-submit-0001",
                "name": "one-run",
                "height": 8,
                "width": 8,
                "device": "cpu",
                "precision": "fp32",
            }

            first = _create_registered_run(
                db,
                body,
                model_id=model_id,
                dataset_id=dataset_id,
                metrics=[],
            )
            replay = _create_registered_run(
                db,
                body,
                model_id=model_id,
                dataset_id=dataset_id,
                metrics=[],
            )

            self.assertEqual(first["run_id"], replay["run_id"])
            self.assertFalse(first["idempotent_replay"])
            self.assertTrue(replay["idempotent_replay"])
            self.assertEqual(
                int(db.get("SELECT COUNT(*) AS count FROM runs")["count"]),
                1,
            )

    def test_stale_pending_run_reconciles_after_completion_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = self._database(root)
            model_id = db.register_model("model", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("dataset", str(root), True)
            body = {
                "submission_id": "run-crash-0001",
                "name": "crash-window",
                "height": 8,
                "width": 8,
                "device": "cpu",
                "precision": "fp32",
            }

            with (
                patch.object(db, "complete_submission", return_value=False),
                self.assertRaises(SubmissionConflict) as raised,
            ):
                _create_registered_run(
                    db,
                    body,
                    model_id=model_id,
                    dataset_id=dataset_id,
                    metrics=[],
                )

            self.assertEqual(raised.exception.code, "submission_claim_lost")
            pending = db.get_submission("run", "run-crash-0001")
            self.assertEqual(pending["status"], "pending")
            self.assertEqual(
                int(db.get("SELECT COUNT(*) AS count FROM runs")["count"]),
                1,
            )
            created_run = db.query("SELECT id, metadata_json FROM runs")[0]
            self.assertIn("run-crash-0001", str(created_run["metadata_json"]))

            with db.connection() as conn:
                conn.execute(
                    """
                    UPDATE submission_keys
                    SET lease_expires_at = 0, updated_at = 0
                    WHERE scope = 'run' AND submission_id = 'run-crash-0001'
                    """
                )

            recovered = _create_registered_run(
                db,
                body,
                model_id=model_id,
                dataset_id=dataset_id,
                metrics=[],
            )
            self.assertTrue(recovered["idempotent_replay"])
            self.assertEqual(recovered["run_id"], int(created_run["id"]))
            self.assertEqual(
                int(db.get("SELECT COUNT(*) AS count FROM runs")["count"]),
                1,
            )
            self.assertEqual(
                db.get_submission("run", "run-crash-0001")["status"],
                "completed",
            )

    def test_completed_run_replay_does_not_resolve_expired_video_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = self._database(root)
            model_id = db.register_model("model", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("dataset", str(root), True)
            body = {
                "submission_id": "run-video-token-0001",
                "video_group": "group",
                "video_selection_token": "expired-selection-token",
                "height": 8,
                "width": 8,
            }
            prepare_calls = 0

            def prepare(request: dict) -> dict:
                nonlocal prepare_calls
                prepare_calls += 1
                if prepare_calls > 1:
                    raise VideoSelectionTokenExpired("selection expired")
                return request

            def create_once(
                create_db: Database,
                _workspace,
                request: dict,
            ) -> dict:
                run_id = create_db.create_run(
                    "token-run",
                    model_id,
                    dataset_id,
                    8,
                    8,
                    1,
                    "cpu",
                    "fp32",
                    [],
                    metadata=request["_submission_marker"],
                )
                return {"run_id": run_id, "run": create_db.get_run(run_id)}

            with patch(
                "vfieval.server._create_run_from_files_once",
                side_effect=create_once,
            ):
                first = _create_run_from_files(
                    db,
                    object(),
                    body,
                    prepare=prepare,
                )
                replay = _create_run_from_files(
                    db,
                    object(),
                    body,
                    prepare=prepare,
                )

            self.assertEqual(prepare_calls, 1)
            self.assertEqual(first["run_id"], replay["run_id"])
            self.assertFalse(first["idempotent_replay"])
            self.assertTrue(replay["idempotent_replay"])


if __name__ == "__main__":
    unittest.main()
