from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from vfieval.cli import main
from vfieval.config import WorkspaceConfig
from vfieval.db import Database


def _metric_health(status: str) -> dict:
    return {
        "asset_root": "<assets>",
        "metrics": {
            "lpips_vit_patch": {
                "status": status,
                "available": status == "available",
            }
        },
    }


class CliReadOnlyTests(unittest.TestCase):
    def test_prepare_metrics_check_only_does_not_create_workspace_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "missing-workspace"
            with (
                patch("vfieval.cli.metrics_health", return_value=_metric_health("available")),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                code = main(
                    [
                        "--workspace",
                        str(workspace_root),
                        "prepare-metrics",
                        "--check-only",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertFalse(workspace_root.exists())

    def test_metric_commands_distinguish_unavailable_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / ".vfieval"
            with (
                patch(
                    "vfieval.cli.metrics_health",
                    return_value=_metric_health("missing_dependency"),
                ),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                self.assertEqual(
                    main(
                        [
                            "--workspace",
                            str(workspace_root),
                            "prepare-metrics",
                            "--check-only",
                        ]
                    ),
                    2,
                )
            with (
                patch(
                    "vfieval.cli.prepare_metric_asset_manifest",
                    return_value={
                        "errors": [{"metric_name": "vmaf", "status": "failed"}],
                        "health": _metric_health("missing_evaluator"),
                    },
                ),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                self.assertEqual(
                    main(["--workspace", str(workspace_root), "prepare-metrics"]),
                    1,
                )

    def test_smoke_metric_returns_two_for_unavailable_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metric = SimpleNamespace(
                evaluate=lambda *_args: SimpleNamespace(
                    status="unavailable",
                    value=None,
                    details={"reason": "missing evaluator"},
                )
            )
            with (
                patch("vfieval.cli.create_metric", return_value=metric),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                code = main(
                    [
                        "--workspace",
                        str(Path(tmp) / ".vfieval"),
                        "smoke-metric",
                        "--metric",
                        "vmaf",
                        "--reference",
                        str(Path(tmp) / "gt.mp4"),
                        "--distorted",
                        str(Path(tmp) / "pred.mp4"),
                        "--work-dir",
                        str(Path(tmp) / "smoke"),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertFalse((Path(tmp) / ".vfieval" / "vfieval.sqlite").exists())

    def test_doctor_does_not_initialize_database_and_reports_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "missing-workspace"
            report = {
                "ok": False,
                "checks": {
                    "ffmpeg": {
                        "status": "unavailable",
                        "reason": "ffmpeg not found",
                    }
                },
                "summary": {
                    "errors": [],
                    "unavailable": ["ffmpeg"],
                    "warnings": [],
                },
            }
            with (
                patch("vfieval.cli.run_doctor", return_value=report),
                patch.object(Database, "init", side_effect=AssertionError("must not migrate")),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                code = main(["--workspace", str(workspace_root), "doctor"])
            self.assertEqual(code, 2)
            self.assertFalse(workspace_root.exists())

    def test_doctor_handles_missing_workspace_without_creating_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "missing-workspace"
            metric_health = {
                "asset_root": "<assets>",
                "metrics": {
                    "vmaf": {"status": "available", "available": True},
                },
            }
            with (
                patch(
                    "vfieval.diagnostics._command_probe",
                    return_value={
                        "status": "unavailable",
                        "available": False,
                        "reason": "not found",
                    },
                ),
                patch(
                    "vfieval.metrics.health.metrics_health",
                    return_value=metric_health,
                ),
                patch(
                    "vfieval.worker.detect_capabilities",
                    return_value={"cpu": True, "cuda": [], "npu": []},
                ),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                code = main(["--workspace", str(workspace_root), "doctor", "--json"])
            self.assertEqual(code, 1)
            self.assertFalse(workspace_root.exists())

    def test_diagnostics_reads_existing_database_without_modifying_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("model", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("dataset", str(root), True)
            run_id = db.create_run(
                "run",
                model_id,
                dataset_id,
                8,
                8,
                1,
                "cpu",
                "fp32",
                [],
            )
            before = workspace.db_path.read_bytes()
            output = root / "diagnostics.zip"
            with (
                patch(
                    "vfieval.diagnostics.run_doctor",
                    return_value={"ok": True, "checks": {}, "summary": {}},
                ),
                patch.object(Database, "init", side_effect=AssertionError("must not migrate")),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                code = main(
                    [
                        "--workspace",
                        str(workspace.root),
                        "diagnostics",
                        "--run-id",
                        str(run_id),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(workspace.db_path.read_bytes(), before)
            with zipfile.ZipFile(output) as archive:
                self.assertIn("selection.json", archive.namelist())


if __name__ == "__main__":
    unittest.main()
