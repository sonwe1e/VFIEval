from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.diagnostics import create_diagnostics_bundle
from vfieval.runtime_logging import (
    close_runtime_logging,
    configure_runtime_logging,
    log_event,
    runtime_logger,
)


class SupportDiagnosticsTests(unittest.TestCase):
    def test_runtime_json_log_redacts_workspace_and_secret_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            log_path = configure_runtime_logging(workspace)
            log_event(
                40,
                "test.failure",
                f"failed below {workspace.root}; Authorization=Bearer-value; Bearer second-value; "
                "/api/blind/CampaignOpaqueABC123/tasks/TaskOpaqueXYZ789/media/left",
                run_id=7,
                public_token="do-not-log-this",
                cookie="session-do-not-log-this",
            )
            for handler in runtime_logger().handlers:
                handler.flush()
            payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(payload["event"], "test.failure")
            self.assertEqual(payload["run_id"], 7)
            self.assertEqual(payload["public_token"], "<redacted>")
            self.assertEqual(payload["cookie"], "<redacted>")
            self.assertIn("<workspace>", payload["message"])
            self.assertNotIn("Bearer-value", payload["message"])
            self.assertNotIn("second-value", payload["message"])
            self.assertNotIn("CampaignOpaqueABC123", payload["message"])
            self.assertNotIn("TaskOpaqueXYZ789", payload["message"])
            self.assertNotIn(str(workspace.root), json.dumps(payload))
            close_runtime_logging()

    def test_sanitized_bundle_contains_selection_doctor_and_log_tail(self) -> None:
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
                metadata={"output_dir": str(workspace.runs_dir / "1"), "api_token": "secret"},
            )
            configure_runtime_logging(workspace)
            log_event(20, "run.created", "created", run_id=run_id)
            for handler in runtime_logger().handlers:
                handler.flush()
            close_runtime_logging()
            with (workspace.root / "logs" / "server.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    '{"path":"/api/blind/LegacyCampaignToken/reviews/LegacyTaskToken"}\n'
                )

            with patch("vfieval.diagnostics.run_doctor", return_value={"ok": True}):
                bundle = create_diagnostics_bundle(db, workspace, run_id=run_id)
            with zipfile.ZipFile(bundle) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {"manifest.json", "doctor.json", "selection.json", "logs/server-tail.jsonl"},
                )
                selection = archive.read("selection.json").decode("utf-8")
                self.assertIn("<workspace>", selection)
                self.assertNotIn("secret", selection)
                self.assertEqual(json.loads(archive.read("manifest.json"))["selection"]["id"], run_id)
                log_tail = archive.read("logs/server-tail.jsonl").decode("utf-8")
                self.assertNotIn("LegacyCampaignToken", log_tail)
                self.assertNotIn("LegacyTaskToken", log_tail)
            close_runtime_logging()

    def test_bundle_requires_exactly_one_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                create_diagnostics_bundle(db, workspace)


if __name__ == "__main__":
    unittest.main()
