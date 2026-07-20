from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from vfieval.config import WorkspaceConfig
from vfieval.orchestration import (
    _start_local_npu_worker_processes,
    _start_local_worker,
    _spawn_worker_process,
    _tracked_local_worker_thread_count,
    _tracked_worker_process_count,
    open_worker_admission,
    shutdown_worker_processes,
)


class _FakeProcess:
    _next_pid = 1000

    def __init__(self, *, ignore_terminate: bool = False) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.ignore_terminate = ignore_terminate
        self.terminate_calls = 0
        self.kill_calls = 0
        self.returncode: int | None = None
        self._finished = threading.Event()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if timeout is None:
            self._finished.wait()
        elif not self._finished.wait(max(0.0, float(timeout))):
            raise subprocess.TimeoutExpired(["fake-worker"], timeout)
        return int(self.returncode or 0)

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self.ignore_terminate:
            self.finish(-15)

    def kill(self) -> None:
        self.kill_calls += 1
        self.finish(-9)

    def finish(self, returncode: int = 0) -> None:
        self.returncode = int(returncode)
        self._finished.set()


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class OrchestrationProcessRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        shutdown_worker_processes(timeout=0.01)
        open_worker_admission()
        self.assertEqual(_tracked_worker_process_count(), 0)

    def tearDown(self) -> None:
        shutdown_worker_processes(timeout=0.01)
        open_worker_admission()

    def _spawn(self, root: Path, process: _FakeProcess) -> _FakeProcess:
        with patch("vfieval.orchestration.subprocess.Popen", return_value=process) as popen:
            returned = _spawn_worker_process(
                ["python", "-m", "vfieval.cli", "worker"],
                root / "logs" / "worker.log",
            )
        self.assertIs(returned, process)
        popen.assert_called_once()
        return process

    def test_spawn_registers_process_and_natural_exit_removes_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            process = self._spawn(Path(tmp), _FakeProcess())
            self.assertEqual(_tracked_worker_process_count(), 1)

            process.finish(0)

            self.assertTrue(_wait_until(lambda: _tracked_worker_process_count() == 0))

    def test_shutdown_terminates_and_waits_for_owned_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            process = self._spawn(Path(tmp), _FakeProcess())

            summary = shutdown_worker_processes(timeout=0.5)

            self.assertEqual(process.terminate_calls, 1)
            self.assertEqual(process.kill_calls, 0)
            self.assertEqual(summary["tracked"], 1)
            self.assertEqual(summary["terminate_requested"], 1)
            self.assertEqual(summary["kill_requested"], 0)
            self.assertEqual(summary["remaining"], 0)

    def test_shutdown_kills_process_that_ignores_graceful_termination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            process = self._spawn(Path(tmp), _FakeProcess(ignore_terminate=True))

            summary = shutdown_worker_processes(timeout=0.02)

            self.assertEqual(process.terminate_calls, 1)
            self.assertEqual(process.kill_calls, 1)
            self.assertEqual(summary["kill_requested"], 1)
            self.assertEqual(summary["remaining"], 0)

    def test_shutdown_is_idempotent_and_ignores_unregistered_processes(self) -> None:
        external_process = _FakeProcess(ignore_terminate=True)

        first = shutdown_worker_processes(timeout=0.01)
        second = shutdown_worker_processes(timeout=0.01)

        self.assertEqual(first["tracked"], 0)
        self.assertEqual(second["tracked"], 0)
        self.assertEqual(external_process.terminate_calls, 0)
        self.assertEqual(external_process.kill_calls, 0)

    def test_local_worker_thread_is_registered_and_removed_on_natural_exit(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def run_blocked_worker(*_args, **_kwargs) -> None:
            entered.set()
            release.wait(timeout=2.0)

        with patch("vfieval.worker.run_worker", side_effect=run_blocked_worker):
            _start_local_worker(object(), object(), role="decode")
            self.assertTrue(entered.wait(timeout=1.0))
            self.assertEqual(_tracked_local_worker_thread_count(), 1)
            release.set()
            self.assertTrue(_wait_until(lambda: _tracked_local_worker_thread_count() == 0))

    def test_shutdown_closes_admission_and_reports_unstoppable_thread(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        calls = 0

        def run_blocked_worker(*_args, **_kwargs) -> None:
            nonlocal calls
            calls += 1
            entered.set()
            release.wait(timeout=2.0)

        with patch("vfieval.worker.run_worker", side_effect=run_blocked_worker):
            _start_local_worker(object(), object(), role="all")
            self.assertTrue(entered.wait(timeout=1.0))

            summary = shutdown_worker_processes(timeout=0.02)
            _start_local_worker(object(), object(), role="all")

            self.assertEqual(summary["threads_tracked"], 1)
            self.assertEqual(summary["threads_remaining"], 1)
            self.assertEqual(calls, 1)
            self.assertEqual(_tracked_local_worker_thread_count(), 1)
            release.set()
            self.assertTrue(_wait_until(lambda: _tracked_local_worker_thread_count() == 0))

    def test_old_worker_late_handoff_cannot_spawn_after_new_generation_opens(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        handoff_finished = threading.Event()
        handoff_processes: list[object] = []
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()

            def run_old_worker(*_args, **_kwargs) -> None:
                entered.set()
                release.wait(timeout=2.0)
                handoff_processes.extend(
                    _start_local_npu_worker_processes(
                        workspace,
                        run_id=17,
                        devices=["npu:0"],
                        start_metric_worker=True,
                    )
                )
                handoff_finished.set()

            with (
                patch("vfieval.worker.run_worker", side_effect=run_old_worker),
                patch("vfieval.orchestration.subprocess.Popen") as popen,
            ):
                _start_local_worker(object(), workspace, role="decode")
                self.assertTrue(entered.wait(timeout=1.0))
                summary = shutdown_worker_processes(timeout=0.02)
                self.assertEqual(summary["threads_remaining"], 1)

                open_worker_admission()
                release.set()
                self.assertTrue(handoff_finished.wait(timeout=1.0))
                self.assertTrue(_wait_until(lambda: _tracked_local_worker_thread_count() == 0))

            self.assertEqual(handoff_processes, [])
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
