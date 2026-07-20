from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.run_cleanup import RunCleanupService, RunPurgePreviewError


def _workspace(tmp: str) -> tuple[WorkspaceConfig, Database]:
    workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
    workspace.ensure()
    db = Database(workspace.db_path)
    db.init()
    return workspace, db


def _run(
    db: Database,
    workspace: WorkspaceConfig,
    name: str,
    *,
    cache_key: str | None = None,
) -> int:
    source = workspace.root / f"source-{name}"
    source.mkdir(parents=True, exist_ok=True)
    dataset_id = db.create_dataset(f"dataset-{name}", str(source), has_gt=True)
    if cache_key is not None:
        cache_path = workspace.root / "decode_cache" / cache_key
        cache_path.mkdir(parents=True, exist_ok=True)
        frame = cache_path / "frame.png"
        if not frame.exists():
            frame.write_bytes(b"cache-bytes")
        db.add_sample(
            dataset_id,
            f"sample-{name}",
            str(frame),
            str(frame),
            str(frame),
            {"cache_key": cache_key},
        )
    model_id = db.upsert_model(f"model-{name}", "dummy", None, 8, 8, {})
    run_id = db.create_run(
        name,
        model_id,
        dataset_id,
        8,
        8,
        1,
        "cpu",
        "fp32",
        [],
        create_inference_job=False,
        metadata={"output_dir": str(workspace.runs_dir / str(db.next_run_id()))},
    )
    with db.connection() as conn:
        conn.execute("UPDATE runs SET status = 'completed' WHERE id = ?", (run_id,))
    return run_id


class RunPurgePreviewTests(unittest.TestCase):
    def test_preview_reports_exact_run_and_deduplicated_cache_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_a = _run(db, workspace, "a", cache_key="shared")
            run_b = _run(db, workspace, "b", cache_key="shared")
            run_c = _run(db, workspace, "c", cache_key="exclusive")
            for run_id, payload in ((run_a, b"aaaa"), (run_b, b"bb"), (run_c, b"ccc")):
                run_dir = workspace.runs_dir / str(run_id)
                run_dir.mkdir(parents=True)
                (run_dir / "artifact.bin").write_bytes(payload)

            service = RunCleanupService(db, workspace)
            preview = service.preview_run_purge("delete_run", [run_c, run_a])

            self.assertEqual(preview["run_ids"], [run_a, run_c])
            self.assertEqual([row["name"] for row in preview["runs"]], ["a", "c"])
            self.assertEqual(preview["summary"]["exclusive_run_bytes"], 7)
            # The shared entry and the selection-exclusive entry are each counted once.
            self.assertEqual(preview["summary"]["referenced_cache_bytes"], 22)
            self.assertEqual(preview["summary"]["shared_cache_bytes"], 11)
            self.assertEqual(preview["summary"]["shared_with_unselected_cache_bytes"], 11)
            self.assertEqual(preview["summary"]["exclusive_cache_bytes"], 11)
            self.assertEqual(preview["summary"]["estimated_reclaimable_bytes"], 7)
            by_id = {row["run_id"]: row for row in preview["runs"]}
            self.assertEqual(by_id[run_a]["bytes"]["shared_cache_bytes"], 11)
            self.assertEqual(by_id[run_c]["bytes"]["exclusive_cache_bytes"], 11)

            consumed = service.consume_run_purge_preview(
                preview["preview_token"],
                request_type="delete_run",
                run_ids=[run_a, run_c],
            )
            self.assertTrue(consumed["validated"])
            self.assertEqual(consumed["run_ids"], [run_a, run_c])
            with self.assertRaisesRegex(RunPurgePreviewError, "already consumed") as raised:
                service.consume_run_purge_preview(
                    preview["preview_token"],
                    request_type="delete_run",
                    run_ids=[run_a, run_c],
                )
            self.assertEqual(raised.exception.code, "missing_preview")

    def test_preview_rejects_operation_and_selection_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_a = _run(db, workspace, "a")
            run_b = _run(db, workspace, "b")
            service = RunCleanupService(db, workspace)

            preview = service.preview_run_purge("delete_run", [run_a])
            with self.assertRaises(RunPurgePreviewError) as raised:
                service.consume_run_purge_preview(
                    preview["preview_token"],
                    request_type="cleanup_artifacts",
                    run_ids=[run_a],
                )
            self.assertEqual(raised.exception.code, "preview_mismatch")

            preview = service.preview_run_purge("delete_run", [run_a])
            with self.assertRaises(RunPurgePreviewError) as raised:
                service.consume_run_purge_preview(
                    preview["preview_token"],
                    request_type="delete_run",
                    run_ids=[run_b],
                )
            self.assertEqual(raised.exception.code, "preview_mismatch")

    def test_preview_rejects_expired_and_stale_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "stale")
            service = RunCleanupService(db, workspace, purge_preview_ttl_seconds=10)

            with patch("vfieval.run_cleanup.utc_ts", return_value=100.0):
                expired = service.preview_run_purge("delete_run", [run_id])
            with patch("vfieval.run_cleanup.utc_ts", return_value=111.0):
                with self.assertRaises(RunPurgePreviewError) as raised:
                    service.consume_run_purge_preview(
                        expired["preview_token"],
                        request_type="delete_run",
                        run_ids=[run_id],
                    )
            self.assertEqual(raised.exception.code, "expired_preview")

            current = service.preview_run_purge("delete_run", [run_id])
            db.bump_run_content_revision(run_id)
            with self.assertRaises(RunPurgePreviewError) as raised:
                service.consume_run_purge_preview(
                    current["preview_token"],
                    request_type="delete_run",
                    run_ids=[run_id],
                )
            self.assertEqual(raised.exception.code, "stale_preview")

    def test_active_delete_preview_tolerates_worker_progress_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "active-delete")
            run_dir = workspace.runs_dir / str(run_id)
            run_dir.mkdir(parents=True)
            (run_dir / "partial.bin").write_bytes(b"before")
            with db.connection() as conn:
                conn.execute(
                    "UPDATE runs SET status = 'running', updated_at = updated_at + 1 WHERE id = ?",
                    (run_id,),
                )

            service = RunCleanupService(db, workspace)
            preview = service.preview_run_purge("delete_run", [run_id])

            # Simulate the worker publishing more output and reporting progress
            # between the confirmation dialog and the DELETE request.
            (run_dir / "partial.bin").write_bytes(b"after-progress")
            with db.connection() as conn:
                conn.execute(
                    "UPDATE runs SET updated_at = updated_at + 1 WHERE id = ?",
                    (run_id,),
                )
            consumed = service.consume_run_purge_preview(
                preview["preview_token"],
                request_type="delete_run",
                run_ids=[run_id],
            )

            self.assertTrue(consumed["validated"])
            self.assertTrue(consumed["state_changed_after_preview"])

    def test_cleanup_preview_surfaces_active_job_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "active")
            now = 123.0
            with db.connection() as conn:
                job = conn.execute(
                    """
                    INSERT INTO jobs(kind, status, payload_json, created_at, started_at)
                    VALUES ('metric', 'running', ?, ?, ?)
                    """,
                    (f'{{"run_id": {run_id}}}', now, now),
                )
                job_id = int(job.lastrowid)
                conn.execute(
                    """
                    INSERT INTO run_jobs(
                        run_id, job_id, role, shard_index, device, metadata_json, created_at
                    ) VALUES (?, ?, 'metric', 0, 'cpu', '{}', ?)
                    """,
                    (run_id, job_id, now),
                )

            preview = RunCleanupService(db, workspace).preview_run_purge(
                "cleanup_artifacts", [run_id]
            )
            row = preview["runs"][0]
            self.assertFalse(row["allowed"])
            self.assertEqual(row["reason"], "active_worker")
            self.assertEqual(row["dependencies"]["active_job_ids"], [job_id])
            self.assertEqual(preview["summary"]["dependencies"]["active_job_ids"], [job_id])


if __name__ == "__main__":
    unittest.main()
