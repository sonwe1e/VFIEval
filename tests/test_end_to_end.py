from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfieval.config import WorkspaceConfig
from vfieval.datasets import scan_triplet_dataset
from vfieval.db import Database
from vfieval.worker import WorkerOptions, run_worker


class EndToEndTests(unittest.TestCase):
    def test_dummy_inference_and_unavailable_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            for folder in ("img0", "img1", "gt"):
                (dataset_root / folder).mkdir(parents=True)
            for idx in range(2):
                name = f"sample{idx:03d}.png"
                Image.new("RGB", (8, 8), (idx, 0, 0)).save(dataset_root / "img0" / name)
                Image.new("RGB", (8, 8), (0, idx, 0)).save(dataset_root / "img1" / name)
                Image.new("RGB", (8, 8), (0, 0, idx)).save(dataset_root / "gt" / name)

            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 4, 4)
            dataset_id = db.create_dataset("demo", str(dataset_root), has_gt=True)
            self.assertEqual(scan_triplet_dataset(db, dataset_id), 2)

            job_id = db.create_job(
                "inference",
                {
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "height": 4,
                    "width": 4,
                    "batch_size": 2,
                    "device": "cpu",
                    "precision": "fp32",
                    "metrics": ["cgvqm"],
                },
            )

            run_worker(db, workspace, WorkerOptions(role="inference", once=True, worker_id="test-inference"))
            inference_job = db.get_job(job_id)
            self.assertEqual(inference_job["status"], "completed")
            self.assertEqual(inference_job["progress_current"], 2)
            self.assertEqual(len(db.list_artifacts(job_id=job_id, kind="pred")), 2)
            self.assertEqual(len(db.list_artifacts(job_id=job_id, kind="difference")), 2)

            run_worker(db, workspace, WorkerOptions(role="metric", once=True, worker_id="test-metric"))
            metric_jobs = [job for job in db.list_jobs() if job["kind"] == "metric"]
            self.assertEqual(len(metric_jobs), 1)
            self.assertEqual(metric_jobs[0]["status"], "completed")
            results = db.list_metric_results(inference_job_id=job_id)
            self.assertEqual(len(results), 1)
            self.assertTrue(all(row["metric_name"] == "cgvqm" for row in results))
            self.assertTrue(all(row["status"] == "unavailable" for row in results))
            self.assertTrue(all(row["sample_id"] is None for row in results))
            self.assertIn("requires video artifacts", results[0]["details"]["reason"])


if __name__ == "__main__":
    unittest.main()
