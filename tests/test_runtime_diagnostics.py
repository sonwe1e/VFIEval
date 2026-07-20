from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import MagicMock, patch

from vfieval.catalog_sync import CatalogSyncCoordinator
from vfieval.config import WorkspaceConfig
from vfieval.db import Database, artifact_storage_metadata, utc_ts
from vfieval.evaluations_v2 import ensure_v2_schema
from vfieval.run_cleanup import RunCleanupService
from vfieval.server import _make_handler, _run_detail


def _new_run(
    db: Database,
    root: Path,
    *,
    metadata: dict | None = None,
) -> int:
    model_id = db.register_model("model", "dummy", None, 8, 8, {})
    dataset_id = db.create_dataset("dataset", str(root), True)
    return db.create_run(
        "diagnostic-run",
        model_id,
        dataset_id,
        8,
        8,
        1,
        "cpu",
        "fp32",
        [],
        metadata=metadata or {},
    )


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_precomputed_bulk_sizes_do_not_stat_during_sqlite_flush(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            run_id = _new_run(db, root)
            job_id = int(db.get_run(run_id)["inference_job_id"])
            artifact = root / "frame.png"
            artifact.write_bytes(b"frame")
            stored_metadata = artifact_storage_metadata(artifact, {"sample": "one"})
            original_stat = Path.stat

            def guarded_stat(candidate: Path, *args, **kwargs):
                if candidate == artifact:
                    raise AssertionError("bulk SQLite flush must reuse saver-recorded size")
                return original_stat(candidate, *args, **kwargs)

            with patch.object(Path, "stat", autospec=True, side_effect=guarded_stat):
                db.add_artifacts_bulk(
                    job_id,
                    [
                        {
                            "sample_id": None,
                            "kind": "pred",
                            "path": str(artifact),
                            "mime_type": "image/png",
                            "metadata": stored_metadata,
                        }
                    ],
                )
            self.assertEqual(db.summarize_artifacts(job_id)["storage_bytes"], 5)

    def test_artifact_bytes_are_recorded_once_and_run_detail_uses_summary_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            run_id = _new_run(
                db,
                root,
                metadata={
                    "workload": {"artifact_budget_bytes": 14},
                    "output_dir": str(workspace.runs_dir / "1"),
                },
            )
            run = db.get_run(run_id)
            job_id = int(run["inference_job_id"])
            artifact = root / "artifact.bin"
            preview = root / "artifact-preview.bin"
            artifact.write_bytes(b"12345")
            preview.write_bytes(b"12")
            db.add_artifact(
                job_id,
                None,
                "pred_video",
                str(artifact),
                "application/octet-stream",
                {"preview_path": str(preview)},
            )

            summary = db.summarize_artifacts(job_id)
            self.assertEqual(summary["storage_bytes"], 7)
            self.assertEqual(summary["storage_bytes_by_kind"], {"pred_video": 7})
            self.assertEqual(summary["storage_size_known"], 1)
            self.assertEqual(summary["storage_size_unknown"], 0)
            stored = db.list_artifacts(job_id)[0]["metadata"]
            self.assertEqual(stored["artifact_file_size_bytes"], 5)
            self.assertEqual(stored["preview_file_size_bytes"], 2)

            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertTrue(
                db.complete_run_inference(run_id, {}, summary, "completed")
            )
            original_stat = Path.stat

            def guarded_stat(candidate: Path, *args, **kwargs):
                if candidate in {artifact, preview}:
                    raise AssertionError("Run Detail must not stat artifact files")
                return original_stat(candidate, *args, **kwargs)

            with patch.object(
                Path,
                "stat",
                autospec=True,
                side_effect=guarded_stat,
            ):
                detail = _run_detail(db, run_id)
            diagnostic = detail["artifact_storage"]
            self.assertEqual(diagnostic["predicted_artifact_budget_bytes"], 14)
            self.assertEqual(diagnostic["actual_artifact_bytes"], 7)
            self.assertEqual(diagnostic["actual_minus_predicted_bytes"], -7)
            self.assertEqual(diagnostic["budget_utilization_ratio"], 0.5)
            self.assertEqual(diagnostic["measurement"], "complete")

    def test_health_reports_read_only_catalog_and_cleanup_backlogs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"VFIEVAL_PROJECT_ROOT": tmp},
            clear=False,
        ):
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            ensure_v2_schema(db)
            run_id = _new_run(db, root)
            db.request_run_purge(run_id, "delete_run")
            now = utc_ts()
            with db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO evaluation_purge_requests_v2(
                        campaign_id, status, requested_at, updated_at
                    ) VALUES (?, 'failed', ?, ?)
                    """,
                    (991, now, now),
                )

            sync_calls: list[bool] = []
            coordinator = CatalogSyncCoordinator(
                db,
                workspace,
                sync_callback=lambda _db, _workspace, include_runs: sync_calls.append(
                    bool(include_runs)
                )
                or {"ok": True},
            )
            coordinator.request_sync(include_runs=True)
            completed = coordinator.wait(2)
            self.assertEqual(completed["state"], "completed")
            cleanup_service = MagicMock(spec=RunCleanupService)
            cleanup_service.ensure_backfilled.return_value = {}
            handler = _make_handler(
                db,
                workspace,
                cleanup_service=cleanup_service,
                catalog_sync=coordinator,
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                with urllib.request.urlopen(f"{base_url}/api/health", timeout=10) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            catalog = payload["maintenance"]["catalog"]
            self.assertEqual(catalog["state"], "completed")
            self.assertEqual(catalog["catalog_revision"], 1)
            self.assertIsNotNone(catalog["duration_seconds"])
            self.assertEqual(payload["maintenance"]["run_cleanup"]["backlog"], 1)
            self.assertEqual(
                payload["maintenance"]["run_cleanup"]["by_status"]["requested"],
                1,
            )
            self.assertEqual(payload["maintenance"]["campaign_cleanup"]["backlog"], 1)
            self.assertEqual(
                payload["maintenance"]["campaign_cleanup"]["by_status"]["failed"],
                1,
            )
            self.assertEqual(sync_calls, [True])
            cleanup_service.process_pending.assert_not_called()

    def test_frontend_displays_actual_artifact_bytes_against_budget(self) -> None:
        app_js = (Path(__file__).parents[1] / "src" / "vfieval" / "web" / "app.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("renderArtifactStorageDiagnostic(run.artifact_storage)", app_js)
        self.assertIn("predicted_artifact_budget_bytes", app_js)
        self.assertIn("actual_artifact_bytes", app_js)


if __name__ == "__main__":
    unittest.main()
