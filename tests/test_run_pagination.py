from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfieval.config import WorkspaceConfig
from vfieval.db import Database


class RunPaginationTests(unittest.TestCase):
    def test_history_over_one_hundred_runs_is_paged_and_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_a = db.register_model("model-a", "dummy", None, 8, 8, {})
            model_b = db.register_model("model-b", "dummy", None, 8, 8, {})
            dataset = db.create_dataset("dataset", tmp, True)
            for index in range(125):
                db.create_run(
                    name=f"experiment-{index:03d}",
                    model_id=model_a if index % 2 == 0 else model_b,
                    dataset_id=dataset,
                    height=8,
                    width=8,
                    batch_size=1,
                    device="cpu",
                    precision="fp32",
                    metrics=[],
                    metadata={
                        "run_type": "video_compare" if index % 5 == 0 else "model_inference"
                    },
                    create_inference_job=False,
                )

            first = db.list_runs_page(page=1, page_size=50)
            third = db.list_runs_page(page=3, page_size=50)
            compares = db.list_runs_page(
                page=1,
                page_size=100,
                run_type="video_compare",
                model="model-a",
            )
            searched = db.list_runs_page(query="experiment-007")

            self.assertEqual(first["total"], 125)
            self.assertEqual(first["active_total"], 125)
            self.assertEqual(compares["active_total"], 125)
            self.assertEqual(first["page_count"], 3)
            self.assertEqual(len(first["runs"]), 50)
            self.assertEqual(len(third["runs"]), 25)
            self.assertTrue(compares["runs"])
            self.assertTrue(all(row["model_name"] == "model-a" for row in compares["runs"]))
            self.assertTrue(
                all(row["metadata"]["run_type"] == "video_compare" for row in compares["runs"])
            )
            self.assertEqual([row["name"] for row in searched["runs"]], ["experiment-007"])


if __name__ == "__main__":
    unittest.main()
