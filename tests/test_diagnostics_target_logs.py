from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.diagnostics import create_diagnostics_bundle


class DiagnosticsTargetLogTests(unittest.TestCase):
    def test_bundle_includes_target_run_logs_and_redacts_external_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("model", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("dataset", str(root), True)
            run_id = db.create_run(
                "diagnostics",
                model_id,
                dataset_id,
                8,
                8,
                1,
                "cpu",
                "fp32",
                [],
            )

            external_models = root.parent / "ExternalModels"
            run_logs = workspace.runs_dir / str(run_id) / "logs"
            run_logs.mkdir(parents=True)
            (run_logs / "model_load.log").write_text(
                f"loaded {str(external_models).upper()}\\adapter.py\n",
                encoding="utf-8",
            )
            integrity_dir = run_logs / "artifact_integrity"
            integrity_dir.mkdir()
            (integrity_dir / "1.json").write_text('{"valid":true}', encoding="utf-8")

            global_logs = workspace.root / "logs"
            global_logs.mkdir()
            (global_logs / "server.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"run_id": run_id, "message": "selected-run"}),
                        json.dumps({"run_id": run_id + 99, "message": "other-run"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (global_logs / "worker.jsonl.1").write_text(
                json.dumps({"run_id": run_id, "message": "rotated-target"}) + "\n",
                encoding="utf-8",
            )

            with (
                patch.dict(
                    os.environ,
                    {"VFIEVAL_MODELS_DIR": str(external_models)},
                    clear=False,
                ),
                patch("vfieval.diagnostics.run_doctor", return_value={"ok": True}),
            ):
                bundle = create_diagnostics_bundle(db, workspace, run_id=run_id)

            with zipfile.ZipFile(bundle) as archive:
                names = set(archive.namelist())
                self.assertIn("logs/run/model_load.log", names)
                self.assertIn("logs/run/artifact_integrity/1.json", names)
                self.assertIn("logs/server-tail.jsonl", names)
                self.assertIn("logs/worker.1-tail.jsonl", names)
                model_log = archive.read("logs/run/model_load.log").decode("utf-8")
                server_log = archive.read("logs/server-tail.jsonl").decode("utf-8")
                manifest = json.loads(archive.read("manifest.json"))

            self.assertIn("<workspace>", model_log)
            self.assertNotIn(str(external_models).upper(), model_log)
            self.assertIn("selected-run", server_log)
            self.assertNotIn("other-run", server_log)
            self.assertGreaterEqual(manifest["logs"]["included_files"], 4)
            self.assertLessEqual(
                manifest["logs"]["included_bytes"],
                manifest["logs"]["budget_bytes"],
            )


if __name__ == "__main__":
    unittest.main()
