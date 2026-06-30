from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vfieval.db import Database


class DbIndexTests(unittest.TestCase):
    def test_v13_sample_video_indices_exist_after_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "vfieval.sqlite")
            db.init()
            conn = db.connect()
            try:
                artifact_indices = {row["name"] for row in conn.execute("PRAGMA index_list(artifacts)").fetchall()}
                metric_indices = {row["name"] for row in conn.execute("PRAGMA index_list(metric_results)").fetchall()}
                run_job_indices = {row["name"] for row in conn.execute("PRAGMA index_list(run_jobs)").fetchall()}
            finally:
                conn.close()

        self.assertIn("idx_artifacts_sample", artifact_indices)
        self.assertIn("idx_metric_results_sample", metric_indices)
        self.assertIn("idx_run_jobs_device", run_job_indices)


if __name__ == "__main__":
    unittest.main()
