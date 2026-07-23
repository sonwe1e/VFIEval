from __future__ import annotations

import errno
import io
import json
from pathlib import Path
from types import SimpleNamespace
import socket
import tempfile
import unittest
from unittest.mock import patch

from vfieval.cli import _doctor_exit_code, main
from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.diagnostics import _device_probe, _port_probe, health_snapshot, run_doctor


class DoctorCapabilityTests(unittest.TestCase):
    def test_port_probe_marks_occupied_target_unavailable(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        try:
            host, port = listener.getsockname()
            result = _port_probe(str(host), int(port))
        finally:
            listener.close()

        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["bindable"])
        self.assertEqual(result["host"], "127.0.0.1")
        self.assertEqual(result["port"], port)

    def test_port_probe_keeps_non_conflict_socket_errors_as_failures(self) -> None:
        failing_socket = unittest.mock.MagicMock()
        failing_socket.bind.side_effect = OSError(errno.EINVAL, "invalid bind")
        with patch("vfieval.diagnostics.socket.socket", return_value=failing_socket):
            result = _port_probe("invalid host", 8765)

        self.assertEqual(result["status"], "error")
        failing_socket.close.assert_called_once()

    def test_device_probe_marks_requested_accelerators_unavailable(self) -> None:
        capabilities = {
            "cpu": True,
            "cuda": [{"id": "cuda:0", "name": "GPU"}],
            "npu": [],
        }
        with patch("vfieval.worker.detect_capabilities", return_value=capabilities):
            report = _device_probe(["CUDA:0", "npu:0", "npu:0"])

        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["requested"], ["cuda:0", "npu:0"])
        self.assertEqual(report["missing_targets"], ["npu:0"])

    def test_doctor_marks_missing_libx264_as_capability_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            command_ok = {
                "status": "ok",
                "available": True,
                "path": "ffmpeg",
            }
            metric_health = {
                "metrics": {
                    "vmaf": {"status": "available", "available": True},
                },
            }
            with (
                patch("vfieval.diagnostics._command_probe", return_value=command_ok.copy()),
                patch(
                    "vfieval.diagnostics.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout="no h264 encoder"),
                ),
                patch("vfieval.diagnostics._port_probe", return_value={"status": "ok"}),
                patch(
                    "vfieval.metrics.health.metrics_health",
                    return_value=metric_health,
                ),
                patch(
                    "vfieval.worker.detect_capabilities",
                    return_value={"cpu": True, "cuda": [], "npu": []},
                ),
            ):
                report = run_doctor(db, workspace)

        self.assertEqual(report["checks"]["ffmpeg_encoders"]["status"], "unavailable")
        self.assertIn("ffmpeg_encoders", report["summary"]["unavailable"])
        self.assertEqual(_doctor_exit_code(report), 2)

    def test_doctor_database_and_storage_errors_are_execution_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            command_ok = {
                "status": "ok",
                "available": True,
                "path": "ffmpeg",
            }
            with (
                patch("vfieval.diagnostics._command_probe", return_value=command_ok.copy()),
                patch(
                    "vfieval.diagnostics.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout="libx264"),
                ),
                patch(
                    "vfieval.diagnostics._database_probe",
                    return_value={"status": "error", "reason": "database unavailable"},
                ),
                patch(
                    "vfieval.diagnostics.health_snapshot",
                    return_value={"storage": {"status": "error", "reason": "disk failed"}},
                ),
                patch("vfieval.diagnostics._port_probe", return_value={"status": "ok"}),
                patch(
                    "vfieval.metrics.health.metrics_health",
                    return_value={"metrics": {}},
                ),
                patch(
                    "vfieval.worker.detect_capabilities",
                    return_value={"cpu": True, "cuda": [], "npu": []},
                ),
            ):
                report = run_doctor(db, workspace)

        self.assertIn("database", report["summary"]["errors"])
        self.assertIn("storage", report["summary"]["errors"])
        self.assertEqual(_doctor_exit_code(report), 1)

    def test_cli_doctor_forwards_targets_without_initializing_database(self) -> None:
        report = {
            "ok": False,
            "checks": {"port": {"status": "unavailable"}},
            "summary": {
                "errors": [],
                "unavailable": ["port"],
                "warnings": [],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "missing"
            with (
                patch("vfieval.cli.run_doctor", return_value=report) as doctor,
                patch.object(Database, "init", side_effect=AssertionError("must not migrate")),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                code = main(
                    [
                        "--workspace",
                        str(workspace_root),
                        "doctor",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "9876",
                        "--device",
                        "cuda:0",
                        "--device",
                        "npu:0",
                    ]
                )

        self.assertEqual(code, 2)
        self.assertFalse(workspace_root.exists())
        self.assertEqual(
            doctor.call_args.kwargs,
            {
                "host": "0.0.0.0",
                "port": 9876,
                "target_devices": ["cuda:0", "npu:0"],
            },
        )

    def test_cli_doctor_converts_internal_exception_to_exit_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch(
                    "vfieval.cli.run_doctor",
                    side_effect=RuntimeError("probe crashed"),
                ),
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                code = main(
                    [
                        "--workspace",
                        str(Path(tmp) / "missing"),
                        "doctor",
                        "--json",
                    ]
                )

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["error"]["type"], "RuntimeError")
        self.assertEqual(payload["summary"]["errors"], ["doctor"])

    def test_health_readiness_reports_recovery_service_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            report = health_snapshot(
                db,
                workspace,
                maintenance={
                    "job_recovery": {
                        "running": False,
                        "last_error": "recovery loop stopped",
                    }
                },
            )

        self.assertFalse(report["ready"])
        self.assertIn("job_recovery_failed", report["reasons"])
        self.assertIn("job_recovery_not_running", report["reasons"])


if __name__ == "__main__":
    unittest.main()
