from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.job_leases import JobLeaseHeartbeat, JobRecoveryService, recover_stale_jobs
from vfieval.server import _make_handler
from vfieval.worker import WorkerOptions, run_worker


class JobLeaseTests(unittest.TestCase):
    def _workspace_db(self, root: Path) -> tuple[WorkspaceConfig, Database]:
        workspace = WorkspaceConfig.from_root(root / ".vfieval")
        workspace.ensure()
        db = Database(workspace.db_path)
        db.init()
        return workspace, db

    @staticmethod
    def _run(db: Database, root: Path) -> int:
        model_id = db.register_model("dummy", "dummy", None, 4, 4, {})
        dataset_id = db.create_dataset("dataset", str(root), False)
        return db.create_run(
            "lease-run",
            model_id,
            dataset_id,
            4,
            4,
            1,
            "cpu",
            "fp32",
            [],
        )

    @staticmethod
    def _set_heartbeat(db: Database, job_id: int, value: float) -> None:
        with db.connection() as conn:
            conn.execute(
                "UPDATE jobs SET heartbeat_at = ? WHERE id = ?",
                (float(value), int(job_id)),
            )

    def test_existing_database_adds_heartbeat_column_and_recovery_index(self) -> None:
        if sqlite3.sqlite_version_info < (3, 35, 0):
            self.skipTest("SQLite DROP COLUMN is unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db = self._workspace_db(Path(tmp))
            with db.connection() as conn:
                conn.execute("DROP INDEX IF EXISTS idx_jobs_running_heartbeat")
                conn.execute("ALTER TABLE jobs DROP COLUMN heartbeat_at")

            db.init()

            with db.connection() as conn:
                columns = {
                    str(row["name"])
                    for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
                }
                indexes = {
                    str(row["name"])
                    for row in conn.execute("PRAGMA index_list(jobs)").fetchall()
                }
            self.assertIn("heartbeat_at", columns)
            self.assertIn("idx_jobs_running_heartbeat", indexes)

    def test_claim_and_heartbeat_are_worker_fenced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db = self._workspace_db(Path(tmp))
            db.register_worker("owner", "decode", {})
            job_id = db.create_job("decode", {})
            claimed = db.claim_next_job("owner", ["decode"])
            self.assertEqual(int(claimed["id"]), job_id)
            self.assertIsNotNone(claimed["heartbeat_at"])

            self._set_heartbeat(db, job_id, 100.0)
            self.assertFalse(db.heartbeat_job(job_id, "late-owner"))
            self.assertEqual(db.get_job(job_id)["heartbeat_at"], 100.0)
            self.assertTrue(db.heartbeat_job(job_id, "owner"))
            self.assertGreater(float(db.get_job(job_id)["heartbeat_at"]), 100.0)

    def test_http_heartbeat_requires_and_enforces_current_worker_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = self._workspace_db(Path(tmp))
            db.register_worker("owner", "decode", {})
            db.register_worker("wrong-owner", "decode", {})
            job_id = db.create_job("decode", {})
            self.assertEqual(int(db.claim_next_job("owner", ["decode"])["id"]), job_id)
            self._set_heartbeat(db, job_id, 100.0)
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"

            def post(payload: dict[str, object]) -> tuple[int, dict[str, object]]:
                request = urllib.request.Request(
                    f"{base_url}/api/jobs/{job_id}/heartbeat",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=10) as response:
                        return int(response.status), json.loads(response.read().decode("utf-8"))
                except urllib.error.HTTPError as exc:
                    return int(exc.code), json.loads(exc.read().decode("utf-8"))

            try:
                missing_status, missing = post({})
                self.assertEqual(missing_status, 400, missing)
                wrong_status, wrong = post({"worker_id": "wrong-owner"})
                self.assertEqual(wrong_status, 409, wrong)
                self.assertEqual(db.get_job(job_id)["heartbeat_at"], 100.0)
                owner_status, owner = post({"worker_id": "owner"})
                self.assertEqual(owner_status, 200, owner)
                self.assertGreater(float(db.get_job(job_id)["heartbeat_at"]), 100.0)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_heartbeat_thread_renews_until_job_becomes_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db = self._workspace_db(Path(tmp))
            db.register_worker("owner", "decode", {})
            job_id = db.create_job("decode", {})
            self.assertEqual(int(db.claim_next_job("owner", ["decode"])["id"]), job_id)
            self._set_heartbeat(db, job_id, 100.0)

            heartbeat = JobLeaseHeartbeat(
                db,
                job_id,
                "owner",
                interval_seconds=0.05,
            )
            self.assertTrue(heartbeat.start())
            first = float(db.get_job(job_id)["heartbeat_at"])
            time.sleep(0.12)
            second = float(db.get_job(job_id)["heartbeat_at"])
            self.assertGreaterEqual(second, first)
            self.assertTrue(db.complete_job(job_id, {}))
            deadline = time.time() + 1.0
            while not heartbeat.lease_lost and time.time() < deadline:
                time.sleep(0.02)
            self.assertTrue(heartbeat.lease_lost)
            heartbeat.stop()

    def test_worker_wraps_claimed_job_in_heartbeat_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = self._workspace_db(Path(tmp))
            job_id = db.create_job("decode", {})
            events: list[tuple[str, int, str]] = []

            class RecordingHeartbeat:
                def __init__(self, _db: Database, claimed_job_id: int, worker_id: str) -> None:
                    self.job_id = claimed_job_id
                    self.worker_id = worker_id

                def start(self) -> bool:
                    events.append(("start", self.job_id, self.worker_id))
                    return True

                def stop(self) -> None:
                    events.append(("stop", self.job_id, self.worker_id))

            with (
                patch("vfieval.worker.JobLeaseHeartbeat", RecordingHeartbeat),
                patch("vfieval.worker.prepare_worker_device"),
                patch("vfieval.worker.detect_capabilities", return_value={}),
                patch("vfieval.worker.run_decode_job", return_value={}),
                patch("vfieval.worker.RunCleanupService.process_pending", return_value=[]),
            ):
                run_worker(
                    db,
                    workspace,
                    WorkerOptions(role="decode", once=True, worker_id="lease-worker"),
                )

            self.assertEqual(
                events,
                [("start", job_id, "lease-worker"), ("stop", job_id, "lease-worker")],
            )
            self.assertEqual(db.get_job(job_id)["status"], "completed")

    def test_stale_shard_fails_run_and_cancels_only_queued_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace, db = self._workspace_db(root)
            run_id = self._run(db, root)
            lost_job_id = int(db.get_run(run_id)["inference_job_id"])
            running_sibling_id = db.add_run_job(
                run_id,
                "inference",
                {"run_id": run_id, "shard_index": 1, "shard_count": 3},
                shard_index=1,
            )
            queued_sibling_id = db.add_run_job(
                run_id,
                "inference",
                {"run_id": run_id, "shard_index": 2, "shard_count": 3},
                shard_index=2,
            )
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertEqual(
                int(db.claim_next_job("lost-worker", ["inference"])["id"]),
                lost_job_id,
            )
            self.assertEqual(
                int(db.claim_next_job("healthy-worker", ["inference"])["id"]),
                running_sibling_id,
            )
            observed_at = time.time()
            self._set_heartbeat(db, lost_job_id, observed_at - 1000.0)
            self._set_heartbeat(db, running_sibling_id, observed_at)

            recovered = recover_stale_jobs(db, now=observed_at)

            self.assertEqual([row["job_id"] for row in recovered], [lost_job_id])
            lost_job = db.get_job(lost_job_id)
            self.assertEqual(lost_job["status"], "failed")
            self.assertEqual(lost_job["error"]["type"], "WorkerLost")
            self.assertTrue(lost_job["error"]["interrupted"])
            self.assertTrue(lost_job["error"]["retryable"])
            self.assertEqual(lost_job["error"]["lease_timeout_seconds"], 180.0)
            failed_run = db.get_run(run_id)
            self.assertEqual(failed_run["status"], "failed")
            self.assertEqual(failed_run["error"]["type"], "WorkerLost")
            self.assertEqual(failed_run["error"]["job_id"], lost_job_id)
            self.assertEqual(db.get_job(running_sibling_id)["status"], "running")
            queued = db.get_job(queued_sibling_id)
            self.assertEqual(queued["status"], "canceled")
            self.assertEqual(queued["error"]["message"], "sibling shard failed the run")
            self.assertFalse(db.update_job_progress(running_sibling_id, 1, 1))
            self.assertFalse(db.heartbeat_job(lost_job_id, "lost-worker"))
            self.assertEqual(recover_stale_jobs(db, now=observed_at), [])

    def test_fresh_heartbeat_and_completed_job_win_recovery_races(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db = self._workspace_db(Path(tmp))
            observed_at = time.time()

            fresh_id = db.create_job("decode", {})
            self.assertEqual(int(db.claim_next_job("fresh", ["decode"])["id"]), fresh_id)
            self._set_heartbeat(db, fresh_id, observed_at - 1000.0)
            self.assertTrue(db.heartbeat_job(fresh_id, "fresh"))
            self.assertEqual(recover_stale_jobs(db, now=observed_at), [])

            completed_id = db.create_job("decode", {})
            self.assertEqual(
                int(db.claim_next_job("completed", ["decode"])["id"]),
                completed_id,
            )
            self._set_heartbeat(db, completed_id, observed_at - 1000.0)
            self.assertTrue(db.complete_job(completed_id, {"done": True}))
            self.assertEqual(recover_stale_jobs(db, now=observed_at), [])
            self.assertEqual(db.get_job(completed_id)["status"], "completed")

    def test_cancellation_wins_before_stale_worker_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _workspace, db = self._workspace_db(root)
            run_id = self._run(db, root)
            job_id = int(db.get_run(run_id)["inference_job_id"])
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertEqual(int(db.claim_next_job("lost", ["inference"])["id"]), job_id)
            self.assertTrue(db.request_run_cancel(run_id))
            self.assertEqual(db.get_run(run_id)["status"], "cancel_requested")
            observed_at = time.time()
            self._set_heartbeat(db, job_id, observed_at - 1000.0)

            recovered = recover_stale_jobs(db, now=observed_at)

            self.assertEqual(recovered[0]["action"], "canceled")
            self.assertEqual(db.get_job(job_id)["status"], "canceled")
            self.assertEqual(db.get_run(run_id)["status"], "canceled")

    def test_recovery_service_exposes_bounded_health_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db = self._workspace_db(Path(tmp))
            job_id = db.create_job("decode", {})
            self.assertEqual(int(db.claim_next_job("lost", ["decode"])["id"]), job_id)
            observed_at = time.time()
            self._set_heartbeat(db, job_id, observed_at - 1000.0)
            callbacks: list[list[dict[str, object]]] = []
            service = JobRecoveryService(db, on_recovered=callbacks.append)

            recovered = service.run_once(now=observed_at)
            status = service.status(now=observed_at)

            self.assertEqual(recovered[0]["job_id"], job_id)
            self.assertEqual(len(callbacks), 1)
            self.assertEqual(status["recovered_total"], 1)
            self.assertEqual(status["leases"]["running"], 0)
            self.assertIsNone(status["last_error"])


if __name__ == "__main__":
    unittest.main()
