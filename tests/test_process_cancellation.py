from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from vfieval.config import WorkspaceConfig
from vfieval.metrics.cgvqm import CgvqmMetric
from vfieval.metrics.vmaf import VmafMetric
from vfieval.pipeline.inference import RunCanceled, _write_mp4_ffmpeg_pipe
from vfieval.process_control import run_cancellable, terminate_process


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.BytesIO()
        self.stderr = io.BytesIO()
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.communicate_calls = 0

    def poll(self):
        return self.returncode

    def communicate(self, input=None, timeout=None):
        self.communicate_calls += 1
        if self.returncode is None:
            raise subprocess.TimeoutExpired(["fake"], timeout or 0)
        return "", b""

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class _StubbornProcess(_FakeProcess):
    def terminate(self) -> None:
        self.terminated = True


class CancellableProcessTests(unittest.TestCase):
    def test_run_cancellable_terminates_child_when_cancel_check_raises(self) -> None:
        process = _FakeProcess()
        checks = 0

        def cancel_check() -> None:
            nonlocal checks
            checks += 1
            if checks >= 2:
                raise RunCanceled("canceled")

        with patch("vfieval.process_control.subprocess.Popen", return_value=process):
            with self.assertRaises(RunCanceled):
                run_cancellable(
                    ["fake"],
                    input_text="payload",
                    timeout=60,
                    cancel_check=cancel_check,
                    poll_interval=0.01,
                )

        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)

    def test_terminate_process_kills_child_that_ignores_terminate(self) -> None:
        process = _StubbornProcess()
        terminate_process(
            process,
            terminate_timeout=0.01,
            kill_timeout=0.01,
        )

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

    def test_real_child_process_is_canceled_promptly(self) -> None:
        checks = 0

        def cancel_check() -> None:
            nonlocal checks
            checks += 1
            if checks >= 2:
                raise RunCanceled("canceled")

        started = time.monotonic()
        with self.assertRaises(RunCanceled):
            run_cancellable(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                timeout=60,
                cancel_check=cancel_check,
                poll_interval=0.02,
            )
        self.assertLess(time.monotonic() - started, 2.0)

    def test_ffmpeg_pipe_terminates_while_frame_feeder_is_active(self) -> None:
        process = _FakeProcess()
        checks = 0

        def cancel_check() -> None:
            nonlocal checks
            checks += 1
            if checks >= 2:
                raise RunCanceled("canceled")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "000001.png"
            frame.write_bytes(b"not-a-real-png-but-the-pipe-does-not-decode-it")
            output = root / "out.mp4"
            with (
                patch("vfieval.ffmpeg_exe.resolve_ffmpeg", return_value="ffmpeg"),
                patch("vfieval.pipeline.inference.subprocess.Popen", return_value=process),
            ):
                with self.assertRaises(RunCanceled):
                    _write_mp4_ffmpeg_pipe(
                        [frame],
                        output,
                        24.0,
                        cancel_check=cancel_check,
                    )

            self.assertTrue(process.terminated)
            self.assertFalse(output.exists())

    def test_vmaf_adapter_forwards_cancellation_to_ffmpeg_lifecycle(self) -> None:
        def cancel_check() -> None:
            return None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            reference = root / "reference.mp4"
            distorted = root / "distorted.mp4"
            reference.write_bytes(b"reference")
            distorted.write_bytes(b"distorted")
            with (
                patch(
                    "vfieval.metrics.vmaf.metric_health",
                    return_value={
                        "available": True,
                        "status": "available",
                        "resolved_executable": "ffmpeg",
                    },
                ),
                patch(
                    "vfieval.metrics.vmaf.run_cancellable",
                    side_effect=RunCanceled("canceled"),
                ) as run,
            ):
                with self.assertRaises(RunCanceled):
                    VmafMetric(workspace, cancel_check=cancel_check).evaluate(
                        reference,
                        distorted,
                        root / "work",
                    )

            self.assertIs(run.call_args.kwargs["cancel_check"], cancel_check)

    def test_cgvqm_adapter_forwards_cancellation_to_driver_lifecycle(self) -> None:
        def cancel_check() -> None:
            return None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            reference = root / "reference.mp4"
            distorted = root / "distorted.mp4"
            reference.write_bytes(b"reference")
            distorted.write_bytes(b"distorted")
            with (
                patch(
                    "vfieval.metrics.cgvqm.metric_health",
                    return_value={
                        "available": True,
                        "status": "available",
                        "driver_command": ["cgvqm-driver"],
                        "video_eval_long_edge": 720,
                    },
                ),
                patch(
                    "vfieval.metrics.cgvqm._prepare_eval_videos",
                    return_value=(reference, distorted, {}),
                ),
                patch(
                    "vfieval.metrics.cgvqm.run_cancellable",
                    side_effect=RunCanceled("canceled"),
                ) as run,
            ):
                with self.assertRaises(RunCanceled):
                    CgvqmMetric(
                        workspace,
                        device="npu:0",
                        cancel_check=cancel_check,
                    ).evaluate(reference, distorted, root / "work")

            self.assertIs(run.call_args.kwargs["cancel_check"], cancel_check)


if __name__ == "__main__":
    unittest.main()
