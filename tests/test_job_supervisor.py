from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.orchestration import JobSupervisor, open_worker_admission
from vfieval.server import _retry_run_metrics


class _AliveProcess:
    pid = 4242

    def poll(self):
        return None


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


class JobSupervisorTests(unittest.TestCase):
    def _workspace_db(self, root: Path) -> tuple[WorkspaceConfig, Database]:
        workspace = WorkspaceConfig.from_root(root / ".vfieval")
        workspace.ensure()
        db = Database(workspace.db_path)
        db.init()
        return workspace, db

    def test_startup_scan_consumes_jobs_queued_before_supervisor_started(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = self._workspace_db(Path(tmp))
            first = db.create_job("decode", {})
            second = db.create_job("decode", {})
            supervisor = JobSupervisor(db, workspace, scan_interval=0.05)
            open_worker_admission()
            with (
                patch("vfieval.worker.detect_capabilities", return_value={}) as detect_capabilities,
                patch("vfieval.worker.prepare_worker_device"),
                patch("vfieval.worker.run_decode_job", return_value={}),
                patch("vfieval.worker.RunCleanupService.process_pending", return_value=[]),
            ):
                supervisor.start()
                try:
                    self.assertTrue(
                        _wait_until(
                            lambda: all(
                                db.get_job(job_id)["status"] == "completed"
                                for job_id in (first, second)
                            )
                        )
                    )
                    decode_workers = [
                        worker
                        for worker in db.list_workers()
                        if worker["role"] == "decode"
                    ]
                    self.assertEqual(len(decode_workers), 1)
                    self.assertEqual(detect_capabilities.call_count, 1)
                finally:
                    supervisor.stop()

    def test_startup_scan_starts_a_consumer_for_every_local_job_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = self._workspace_db(Path(tmp))
            db.create_job("decode", {})
            db.create_job("inference", {"device": "cpu"})
            db.create_job("finalize", {})
            db.create_job("metric", {"device": "cpu"})
            supervisor = JobSupervisor(db, workspace, scan_interval=0.05)
            entered: set[str] = set()
            entered_lock = threading.Lock()

            def blocked_worker(_db, _workspace, options) -> None:
                with entered_lock:
                    entered.add(
                        f"{options.role}:{options.device_filter or 'default'}"
                    )
                options.stop_event.wait(timeout=3.0)

            open_worker_admission()
            with patch("vfieval.worker.run_worker", side_effect=blocked_worker):
                supervisor.start()
                try:
                    self.assertTrue(
                        _wait_until(
                            lambda: entered
                            == {
                                "decode:default",
                                "inference:cpu",
                                "finalize:default",
                                "metric:cpu",
                            }
                        ),
                        supervisor.status(),
                    )
                finally:
                    supervisor.stop()

    def test_repeated_wakes_reuse_one_role_device_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = self._workspace_db(Path(tmp))
            db.create_job("decode", {})
            db.create_job("decode", {})
            supervisor = JobSupervisor(db, workspace, scan_interval=0.05)
            entered = threading.Event()
            calls = 0

            def blocked_worker(_db, _workspace, options) -> None:
                nonlocal calls
                calls += 1
                entered.set()
                options.stop_event.wait(timeout=3.0)

            open_worker_admission()
            with patch("vfieval.worker.run_worker", side_effect=blocked_worker):
                supervisor.start()
                try:
                    self.assertTrue(entered.wait(timeout=1.0))
                    for _ in range(5):
                        supervisor.wake()
                        supervisor.run_once()
                    self.assertEqual(calls, 1)
                    self.assertEqual(
                        supervisor.status()["thread_slots"],
                        {"decode:default": True},
                    )
                finally:
                    supervisor.stop()

    def test_accelerator_slot_reuses_one_long_lived_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = self._workspace_db(Path(tmp))
            db.create_job("inference", {"device": "npu:0"})
            supervisor = JobSupervisor(db, workspace, scan_interval=0.05)
            process = _AliveProcess()
            open_worker_admission()
            with patch(
                "vfieval.orchestration._spawn_worker_process",
                return_value=process,
            ) as spawn:
                self.assertEqual(supervisor.run_once(), 1)
                self.assertEqual(supervisor.run_once(), 0)

            spawn.assert_called_once()
            self.assertEqual(
                supervisor.status()["process_slots"],
                {"inference:npu:0": True},
            )

    def test_repeated_multi_npu_runs_do_not_grow_worker_process_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = self._workspace_db(Path(tmp))
            for device in ("npu:0", "npu:1"):
                db.create_job("inference", {"device": device})
            supervisor = JobSupervisor(db, workspace, scan_interval=0.05)
            open_worker_admission()
            with patch(
                "vfieval.orchestration._spawn_worker_process",
                side_effect=[_AliveProcess(), _AliveProcess()],
            ) as spawn:
                self.assertEqual(supervisor.run_once(), 2)
                for device in ("npu:0", "npu:1"):
                    db.create_job("inference", {"device": device})
                for _ in range(5):
                    supervisor.wake()
                    self.assertEqual(supervisor.run_once(), 0)

            self.assertEqual(spawn.call_count, 2)
            self.assertEqual(
                supervisor.status()["process_slots"],
                {
                    "inference:npu:0": True,
                    "inference:npu:1": True,
                },
            )

    def test_metric_retry_is_consumed_by_already_running_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, db = self._workspace_db(root)
            model_id = db.register_model("dummy", "dummy", None, 4, 4, {})
            dataset_id = db.create_dataset("dataset", str(root), False)
            run_id = db.create_run(
                "metric-retry",
                model_id,
                dataset_id,
                4,
                4,
                1,
                "cpu",
                "fp32",
                ["lpips_convnext"],
            )
            inference_job_id = int(db.get_run(run_id)["inference_job_id"])
            db.register_worker("setup", "inference", {})
            self.assertEqual(
                int(db.claim_next_job("setup", ["inference"])["id"]),
                inference_job_id,
            )
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertTrue(db.complete_job(inference_job_id, {"samples": 1}))
            self.assertTrue(
                db.complete_run_inference(
                    run_id,
                    {"samples": 1},
                    {},
                    "completed",
                )
            )

            supervisor = JobSupervisor(db, workspace, scan_interval=0.05)
            open_worker_admission()
            metric_result = {
                "summary": {
                    "lpips_convnext": {
                        "completed": 1,
                        "unavailable": 0,
                        "failed": 0,
                        "skipped": 0,
                        "mean": 0.1,
                        "value_sum": 0.1,
                    }
                }
            }

            def complete_metric_job(worker_db, _workspace, _job_id):
                self.assertTrue(worker_db.mark_run_started(run_id, "metric_running"))
                return metric_result

            with (
                patch("vfieval.worker.detect_capabilities", return_value={}),
                patch("vfieval.worker.prepare_worker_device"),
                patch("vfieval.worker.run_metric_job", side_effect=complete_metric_job),
                patch("vfieval.worker.RunCleanupService.process_pending", return_value=[]),
                patch(
                    "vfieval.pipeline.artifact_integrity.require_metric_retry_integrity",
                    side_effect=lambda _db, _run_id: {
                        "content_revision": int(db.get_run(run_id)["content_revision"])
                    },
                ),
            ):
                supervisor.start()
                try:
                    retry = _retry_run_metrics(db, run_id)
                    metric_job_id = int(retry["metric_job_id"])
                    completed = _wait_until(
                        lambda: db.get_job(metric_job_id)["status"] == "completed"
                        and db.get_run(run_id)["status"] == "completed"
                    )
                    self.assertTrue(
                        completed,
                        (db.get_job(metric_job_id), db.get_run(run_id)),
                    )
                finally:
                    supervisor.stop()


if __name__ == "__main__":
    unittest.main()
