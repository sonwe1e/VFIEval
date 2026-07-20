from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.media_assets import bind_run_asset, ensure_collection, upsert_asset
from vfieval.run_cleanup import RunCleanupService
from vfieval.worker import _complete_claimed_job


class LifecycleCasRegressionTests(unittest.TestCase):
    def _workspace_db(self, root: Path) -> tuple[WorkspaceConfig, Database, int, int]:
        workspace = WorkspaceConfig.from_root(root / ".vfieval")
        workspace.ensure()
        db = Database(workspace.db_path)
        db.init()
        model_id = db.register_model("dummy", "dummy", None, 4, 4, {})
        dataset_id = db.create_dataset("dataset", str(root), False)
        return workspace, db, model_id, dataset_id

    @staticmethod
    def _insert_legacy_job(
        db: Database,
        kind: str,
        status: str,
        payload: dict[str, object],
    ) -> int:
        now = time.time()
        with db.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO jobs(
                    kind, status, payload_json, created_at, started_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    status,
                    json.dumps(payload, sort_keys=True),
                    now,
                    now if status == "running" else None,
                ),
            )
            return int(cur.lastrowid)

    def test_metric_retry_completion_ignores_failed_historical_wave(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db, model_id, dataset_id = self._workspace_db(Path(tmp))
            run_id = db.create_run(
                "metric-retry",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            self.assertTrue(db.mark_run_started(run_id, "running"))
            first_wave_id = "failed-wave"
            first_jobs = db.publish_metric_wave(
                run_id,
                [
                    {
                        "payload": {
                            "run_id": run_id,
                            "metric_wave_id": first_wave_id,
                            "metric_wave_count": 1,
                            "metric_names": ["lpips_convnext"],
                        }
                    }
                ],
                retry=False,
            )
            self.assertTrue(db.mark_run_started(run_id, "metric_running"))
            first_job = db.claim_next_job("failed-metric", ["metric"])
            self.assertEqual(int(first_job["id"]), first_jobs[0])
            self.assertTrue(
                db.fail_claimed_job_and_run(
                    first_jobs[0],
                    run_id,
                    {"type": "InjectedFailure", "message": "first wave failed"},
                )
            )

            failed_run = db.get_run(run_id)
            self.assertEqual(failed_run["status"], "failed")
            retry_wave_id = "retry-wave"
            retry_jobs = db.publish_metric_wave(
                run_id,
                [
                    {
                        "payload": {
                            "run_id": run_id,
                            "metric_wave_id": retry_wave_id,
                            "metric_wave_count": 1,
                            "metric_names": ["lpips_convnext"],
                        }
                    }
                ],
                retry=True,
                expected_content_revision=int(failed_run["content_revision"]),
            )
            self.assertTrue(db.mark_run_started(run_id, "metric_running"))
            retry_job = db.claim_next_job("retry-metric", ["metric"])
            self.assertEqual(int(retry_job["id"]), retry_jobs[0])
            retry_result = {
                "summary": {
                    "lpips_convnext": {
                        "completed": 1,
                        "mean": 0.1,
                        "value_sum": 0.1,
                    }
                }
            }
            self.assertTrue(
                db.complete_run_metric_wave(
                    run_id,
                    retry_jobs[0],
                    {"lpips_convnext": {"completed": 1, "mean": 0.1}},
                    [],
                    source_job_id=retry_jobs[0],
                    source_job_result=retry_result,
                )
            )
            self.assertEqual(db.get_job(first_jobs[0])["status"], "failed")
            self.assertEqual(db.get_job(retry_jobs[0])["status"], "completed")
            self.assertEqual(db.get_run(run_id)["status"], "completed")

    def test_legacy_associations_are_canceled_and_block_cleanup_while_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db, model_id, dataset_id = self._workspace_db(Path(tmp))
            run_id = db.create_run(
                "legacy-associations",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            direct_running = self._insert_legacy_job(db, "inference", "running", {})
            payload_queued = self._insert_legacy_job(
                db,
                "metric",
                "queued",
                {"run_id": run_id},
            )
            with db.connection() as conn:
                conn.execute(
                    "UPDATE runs SET inference_job_id = ? WHERE id = ?",
                    (direct_running, run_id),
                )

            self.assertTrue(db.fail_run(run_id, {"type": "InjectedFailure"}))
            self.assertEqual(db.get_job(payload_queued)["status"], "canceled")
            self.assertEqual(db.get_job(direct_running)["status"], "running")
            active = RunCleanupService(db, workspace)._active_jobs(run_id)
            self.assertEqual([int(row["job_id"]) for row in active], [direct_running])

    def test_claimed_failure_cancels_payload_only_queued_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db, model_id, dataset_id = self._workspace_db(Path(tmp))
            run_id = db.create_run(
                "claimed-failure",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
            )
            claimed_job_id = int(db.get_run(run_id)["inference_job_id"])
            claimed = db.claim_next_job("inference", ["inference"])
            self.assertEqual(int(claimed["id"]), claimed_job_id)
            payload_queued = self._insert_legacy_job(
                db,
                "metric",
                "queued",
                {"run_id": run_id},
            )

            self.assertTrue(
                db.fail_claimed_job_and_run(
                    claimed_job_id,
                    run_id,
                    {"type": "InjectedFailure"},
                )
            )
            self.assertEqual(db.get_job(claimed_job_id)["status"], "failed")
            self.assertEqual(db.get_job(payload_queued)["status"], "canceled")

    def test_completed_source_handoff_requires_identical_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db, model_id, dataset_id = self._workspace_db(Path(tmp))
            run_id = db.create_run(
                "source-replay",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            source_job_id = db.add_run_job(
                run_id,
                "decode",
                {"run_id": run_id, "dataset_id": dataset_id},
            )
            claimed = db.claim_next_job("decode", ["decode"])
            self.assertEqual(int(claimed["id"]), source_job_id)
            self.assertTrue(db.complete_job(source_job_id, {"samples": 1}))

            with self.assertRaises(RuntimeError):
                db.publish_inference_jobs(
                    run_id,
                    [{"payload": {"run_id": run_id, "dataset_id": dataset_id}}],
                    source_job_id=source_job_id,
                    source_job_result={"samples": 2},
                )
            self.assertEqual(db.list_run_jobs(run_id, "inference"), [])
            self.assertEqual(db.get_run(run_id)["status"], "decoding")

    def test_cancellation_convergence_does_not_cancel_foreign_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db, model_id, dataset_id = self._workspace_db(Path(tmp))
            run_ids: list[int] = []
            job_ids: list[int] = []
            for index in range(2):
                run_id = db.create_run(
                    f"run-{index}",
                    model_id,
                    dataset_id,
                    4,
                    4,
                    1,
                    "cpu",
                    "fp32",
                    [],
                )
                run_ids.append(run_id)
                job_ids.append(int(db.get_run(run_id)["inference_job_id"]))
                claimed = db.claim_next_job(f"worker-{index}", ["inference"])
                self.assertEqual(int(claimed["id"]), job_ids[index])

            self.assertTrue(db.request_run_cancel(run_ids[0]))
            self.assertFalse(db.converge_run_cancellation(run_ids[0], job_ids[1]))
            self.assertEqual(db.get_job(job_ids[0])["status"], "running")
            self.assertEqual(db.get_job(job_ids[1])["status"], "running")
            self.assertTrue(db.converge_run_cancellation(run_ids[0], job_ids[0]))
            self.assertEqual(db.get_job(job_ids[0])["status"], "canceled")
            self.assertEqual(db.get_job(job_ids[1])["status"], "running")

    def test_device_filtered_claim_supports_direct_legacy_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db, model_id, dataset_id = self._workspace_db(Path(tmp))
            run_id = db.create_run(
                "legacy-device",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "npu:0",
                "fp32",
                [],
                create_inference_job=False,
            )
            legacy_job_id = self._insert_legacy_job(
                db,
                "inference",
                "queued",
                {"run_id": run_id, "device": "npu:0"},
            )
            with db.connection() as conn:
                conn.execute(
                    "UPDATE runs SET inference_job_id = ? WHERE id = ?",
                    (legacy_job_id, run_id),
                )

            claimed = db.claim_next_job(
                "legacy-npu-worker",
                ["inference"],
                device_filter="npu:0",
            )
            self.assertIsNotNone(claimed)
            self.assertEqual(int(claimed["id"]), legacy_job_id)

    def test_create_job_rejects_missing_or_terminal_run_without_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db, model_id, dataset_id = self._workspace_db(Path(tmp))
            run_id = db.create_run(
                "terminal-child",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            self.assertTrue(db.fail_run(run_id, {"type": "InjectedFailure"}))
            before = len(db.list_jobs())
            with self.assertRaises(RuntimeError):
                db.create_job("metric", {"run_id": run_id})
            with self.assertRaises(KeyError):
                db.create_job("inference", {"run_id": run_id + 10_000})
            self.assertEqual(len(db.list_jobs()), before)

    def test_create_job_rejects_wrong_phase_and_duplicate_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db, model_id, dataset_id = self._workspace_db(Path(tmp))
            run_id = db.create_run(
                "generic-job-fence",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
            )
            first_metric = db.create_job("metric", {"run_id": run_id})
            with self.assertRaises(RuntimeError):
                db.create_job("metric", {"run_id": run_id})
            self.assertTrue(db.mark_run_started(run_id, "running"))
            with self.assertRaises(RuntimeError):
                db.create_job("inference", {"run_id": run_id})
            self.assertEqual(
                [int(row["job_id"]) for row in db.list_run_jobs(run_id, "metric")],
                [first_metric],
            )

    def test_set_run_metric_job_requires_queued_exclusive_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db, model_id, dataset_id = self._workspace_db(Path(tmp))
            run_ids = [
                db.create_run(
                    f"metric-owner-{index}",
                    model_id,
                    dataset_id,
                    4,
                    4,
                    1,
                    "cpu",
                    "fp32",
                    [],
                    create_inference_job=False,
                )
                for index in range(4)
            ]
            for run_id in run_ids:
                self.assertTrue(db.mark_run_started(run_id, "running"))

            shared_job_id = db.create_job("metric", {"legacy": True})
            self.assertTrue(db.set_run_metric_job(run_ids[0], shared_job_id))
            self.assertFalse(db.set_run_metric_job(run_ids[1], shared_job_id))
            claimed_shared = db.claim_next_job("owned-metric", ["metric"])
            self.assertEqual(int(claimed_shared["id"]), shared_job_id)
            self.assertTrue(db.complete_job(shared_job_id, {"done": True}))

            completed_job_id = db.create_job("metric", {"legacy": "completed"})
            claimed = db.claim_next_job("standalone-metric", ["metric"])
            self.assertEqual(int(claimed["id"]), completed_job_id)
            self.assertTrue(db.complete_job(completed_job_id, {"done": True}))
            self.assertFalse(db.set_run_metric_job(run_ids[2], completed_job_id))
            self.assertEqual(db.get_run(run_ids[2])["status"], "running")

            first_same_run = db.add_run_job(run_ids[3], "metric", {"run_id": run_ids[3]})
            db.add_run_job(run_ids[3], "metric", {"run_id": run_ids[3]})
            self.assertFalse(db.set_run_metric_job(run_ids[3], first_same_run))
            self.assertEqual(db.get_run(run_ids[3])["status"], "running")

    def test_conflicting_completed_source_acknowledgement_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db, model_id, dataset_id = self._workspace_db(Path(tmp))
            run_id = db.create_run(
                "conflicting-source-result",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            source_job_id = db.add_run_job(
                run_id,
                "decode",
                {"run_id": run_id, "dataset_id": dataset_id},
            )
            source_job = db.claim_next_job("decode-source", ["decode"])
            self.assertEqual(int(source_job["id"]), source_job_id)
            published = db.publish_inference_jobs(
                run_id,
                [{"payload": {"run_id": run_id, "dataset_id": dataset_id}}],
                source_job_id=source_job_id,
                source_job_result={"samples": 1},
            )
            self.assertEqual(len(published), 1)
            with self.assertRaises(RuntimeError):
                _complete_claimed_job(db, source_job, {"samples": 2})

    def test_run_failure_does_not_invalidate_foreign_compare_input_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace, db, model_id, dataset_id = self._workspace_db(root)
            run_id = db.create_run(
                "compare-input-invalidation",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                [],
                create_inference_job=False,
            )
            self.assertTrue(db.mark_run_started(run_id, "running"))
            collection = ensure_collection(db, "Run assets", "run-assets-test")
            source_path = root / "source.mp4"
            output_path = root / "output.mp4"
            source_path.write_bytes(b"source")
            output_path.write_bytes(b"output")
            source_asset = upsert_asset(
                db,
                collection_id=int(collection["id"]),
                source_key="run_artifact:1001",
                source_kind="run_artifact",
                media_kind="video",
                role="pred",
                display_name="source prediction",
                original_name=source_path.name,
                storage_path=source_path,
            )
            output_asset = upsert_asset(
                db,
                collection_id=int(collection["id"]),
                source_key="run_artifact:1002",
                source_kind="run_artifact",
                media_kind="video",
                role="pred",
                display_name="compare output",
                original_name=output_path.name,
                storage_path=output_path,
            )
            bind_run_asset(
                db,
                run_id,
                int(source_asset["id"]),
                "pred",
                video_name="clip",
                track_label="source",
                metadata={"input": True},
            )
            bind_run_asset(
                db,
                run_id,
                int(output_asset["id"]),
                "pred",
                video_name="clip",
                track_label="derived",
                metadata={"artifact_id": 1002},
            )

            self.assertTrue(db.fail_run(run_id, {"type": "InjectedFailure"}))
            self.assertEqual(db.get("SELECT state FROM media_assets WHERE id = ?", (source_asset["id"],))["state"], "ready")
            self.assertEqual(db.get("SELECT state FROM media_assets WHERE id = ?", (output_asset["id"],))["state"], "unavailable")

    def test_metric_phase_failures_preserve_published_run_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace, db, model_id, dataset_id = self._workspace_db(root)
            collection = ensure_collection(db, "Metric assets", "metric-assets-test")

            for case, fail_claimed in (("queued", False), ("running", True)):
                with self.subTest(case=case):
                    run_id = db.create_run(
                        f"metric-{case}",
                        model_id,
                        dataset_id,
                        4,
                        4,
                        1,
                        "cpu",
                        "fp32",
                        ["lpips_convnext"],
                        create_inference_job=False,
                    )
                    self.assertTrue(db.mark_run_started(run_id, "running"))
                    output_path = root / f"metric-{case}.mp4"
                    output_path.write_bytes(b"published-output")
                    output_asset = upsert_asset(
                        db,
                        collection_id=int(collection["id"]),
                        source_key=f"run_artifact:metric-phase:{run_id}",
                        source_kind="run_artifact",
                        media_kind="video",
                        role="pred",
                        display_name=f"metric {case}",
                        original_name=output_path.name,
                        storage_path=output_path,
                    )
                    bind_run_asset(
                        db,
                        run_id,
                        int(output_asset["id"]),
                        "pred",
                        video_name="clip",
                        metadata={"artifact_id": run_id},
                    )
                    metric_job_id = db.add_run_job(
                        run_id,
                        "metric",
                        {"run_id": run_id, "metric_names": ["lpips_convnext"]},
                    )
                    self.assertTrue(db.set_run_metric_job(run_id, metric_job_id))
                    if fail_claimed:
                        claimed = db.claim_next_job(f"metric-{case}", ["metric"])
                        self.assertEqual(int(claimed["id"]), metric_job_id)
                        self.assertTrue(db.mark_run_started(run_id, "metric_running"))
                        self.assertTrue(
                            db.fail_claimed_job_and_run(
                                metric_job_id,
                                run_id,
                                {"type": "MetricCrash", "message": "metric failed"},
                            )
                        )
                    else:
                        self.assertEqual(db.get_run(run_id)["status"], "metric_queued")
                        self.assertTrue(
                            db.fail_run(
                                run_id,
                                {"type": "MetricCrash", "message": "metric failed"},
                            )
                        )
                    self.assertEqual(
                        db.get(
                            "SELECT state FROM media_assets WHERE id = ?",
                            (int(output_asset["id"]),),
                        )["state"],
                        "ready",
                    )


if __name__ == "__main__":
    unittest.main()
