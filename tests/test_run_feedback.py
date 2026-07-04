from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v13_test_utils import get_json, make_workspace, post_json, start_server


def _make_run(db, workspace, name: str = "run") -> int:
    model_id = db.upsert_model(f"model-{name}", "dummy", None, 8, 8, {"source": "test"})
    dataset_root = workspace.root / f"dataset-{name}"
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset_id = db.create_dataset(f"dataset-{name}", str(dataset_root), True, metadata={"source": "test"})
    return db.create_run(
        name, model_id, dataset_id, 8, 8, 1, "cpu", "fp32", [],
        metadata={"output_dir": str(workspace.runs_dir / name)},
    )


class RunFeedbackTests(unittest.TestCase):
    def test_feedback_lifecycle_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            run_a = _make_run(db, workspace, "runA")
            run_b = _make_run(db, workspace, "runB")
            server, thread, base_url = start_server(db, workspace)
            try:
                created = post_json(base_url, f"/api/runs/{run_a}/feedback", {"username": "alice", "rating": 4, "issue": "闪烁"})
                self.assertIn("feedback", created)
                post_json(base_url, f"/api/runs/{run_a}/feedback", {"username": "bob", "rating": 2, "issue": ""})
                post_json(base_url, f"/api/runs/{run_b}/feedback", {"username": "alice", "rating": 5, "issue": "很好"})

                # Per-run listing
                detail = get_json(base_url, f"/api/runs/{run_a}")
                self.assertEqual(len(detail["feedback"]), 2)

                # Global stats
                stats = get_json(base_url, "/api/feedback")
                self.assertEqual(stats["total"], 3)
                self.assertEqual(stats["rating_count"], 3)
                self.assertEqual(stats["issue_count"], 2)
                self.assertAlmostEqual(stats["average_rating"], round((4 + 2 + 5) / 3, 2))
                self.assertEqual(stats["rating_distribution"]["4"], 1)
                self.assertEqual(stats["rating_distribution"]["5"], 1)
                by_user = {row["username"]: row for row in stats["by_user"]}
                self.assertEqual(by_user["alice"]["count"], 2)
                self.assertAlmostEqual(by_user["alice"]["average_rating"], 4.5)

                # Delete one entry, scoped to its run
                feedback_id = int(detail["feedback"][0]["id"])
                import urllib.request
                req = urllib.request.Request(f"{base_url}/api/runs/{run_a}/feedback/{feedback_id}", method="DELETE")
                with urllib.request.urlopen(req) as resp:
                    self.assertEqual(resp.status, 200)
                self.assertEqual(get_json(base_url, "/api/feedback")["total"], 2)
            finally:
                from v13_test_utils import stop_server
                stop_server(server, thread)

    def test_feedback_rejects_empty_and_bad_rating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            run_id = _make_run(db, workspace, "run")
            server, thread, base_url = start_server(db, workspace)
            try:
                import urllib.error
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    post_json(base_url, f"/api/runs/{run_id}/feedback", {"username": "x", "rating": None, "issue": ""})
                self.assertEqual(ctx.exception.code, 400)
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    post_json(base_url, f"/api/runs/{run_id}/feedback", {"username": "x", "rating": 9, "issue": "ok"})
                self.assertEqual(ctx.exception.code, 400)
            finally:
                from v13_test_utils import stop_server
                stop_server(server, thread)


if __name__ == "__main__":
    unittest.main()
