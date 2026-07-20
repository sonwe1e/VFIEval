from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v13_test_utils import get_json, make_workspace, post_json, start_server


def _mark_completed(db, run_id: int) -> None:
    """Force a run into a terminal state so cleanup-artifacts is permitted."""
    with db.connection() as conn:
        conn.execute("UPDATE runs SET status = 'completed' WHERE id = ?", (run_id,))


def _make_run(db, workspace, name: str = "run", *, model: str | None = None, checkpoint: str = "", dataset: str | None = None) -> int:
    model_name = model or f"model-{name}"
    model_id = db.upsert_model(model_name, "dummy", None, 8, 8, {"source": "test"})
    dataset_name = dataset or f"dataset-{name}"
    dataset_root = workspace.root / f"dataset-{name}"
    dataset_root.mkdir(parents=True, exist_ok=True)
    # upsert so grouping tests can reuse a dataset name across runs.
    dataset_id = db.upsert_dataset(dataset_name, str(dataset_root), True, metadata={"source": "test"})
    return db.create_run(
        name, model_id, dataset_id, 8, 8, 1, "cpu", "fp32", [],
        metadata={
            "output_dir": str(workspace.runs_dir / name),
            "model_file": model_name,
            "checkpoint": checkpoint,
        },
    )


def _mark_completed(db, run_id: int) -> None:
    """Force a run to a terminal status so cleanup-artifacts is permitted."""
    from vfieval.db import utc_ts

    with db.connection() as conn:
        conn.execute(
            "UPDATE runs SET status = 'completed', updated_at = ? WHERE id = ?",
            (utc_ts(), run_id),
        )


class RunFeedbackTests(unittest.TestCase):
    def test_feedback_lifecycle_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            run_a = _make_run(db, workspace, "runA")
            run_b = _make_run(db, workspace, "runB")
            server, thread, base_url = start_server(db, workspace)
            try:
                created = post_json(base_url, f"/api/runs/{run_a}/feedback", {"username": "alice", "rating": 4, "issue": "闪烁", "video": "clip01.mp4"})
                self.assertIn("feedback", created)
                post_json(base_url, f"/api/runs/{run_a}/feedback", {"username": "bob", "rating": 2, "issue": "", "video": "clip01.mp4"})
                post_json(base_url, f"/api/runs/{run_b}/feedback", {"username": "alice", "rating": 5, "issue": "很好", "video": "clip02.mp4"})

                # Per-run listing
                detail = get_json(base_url, f"/api/runs/{run_a}")
                self.assertEqual(len(detail["feedback"]), 2)

                # Global stats — distribution keys are 0.25-step strings now.
                stats = get_json(base_url, "/api/feedback")
                self.assertEqual(stats["total"], 3)
                self.assertEqual(stats["rating_count"], 3)
                self.assertEqual(stats["issue_count"], 2)
                self.assertAlmostEqual(stats["average_rating"], round((4 + 2 + 5) / 3, 2))
                self.assertEqual(stats["rating_distribution"]["4.00"], 1)
                self.assertEqual(stats["rating_distribution"]["5.00"], 1)
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
                # Off-step ratings (not a multiple of 0.25) are rejected.
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    post_json(base_url, f"/api/runs/{run_id}/feedback", {"username": "x", "rating": 3.3, "issue": "ok", "video": "c.mp4"})
                self.assertEqual(ctx.exception.code, 400)
            finally:
                from v13_test_utils import stop_server
                stop_server(server, thread)

    def test_quarter_step_rating_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            run_id = _make_run(db, workspace, "run")
            server, thread, base_url = start_server(db, workspace)
            try:
                post_json(base_url, f"/api/runs/{run_id}/feedback", {"username": "a", "rating": 3.25, "issue": "", "video": "c.mp4"})
                detail = get_json(base_url, f"/api/runs/{run_id}")
                self.assertAlmostEqual(float(detail["feedback"][0]["rating"]), 3.25)
                stats = get_json(base_url, "/api/feedback")
                self.assertEqual(stats["rating_distribution"]["3.25"], 1)
            finally:
                from v13_test_utils import stop_server
                stop_server(server, thread)

    def test_feedback_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            run_id = _make_run(db, workspace, "run")
            server, thread, base_url = start_server(db, workspace)
            try:
                post_json(base_url, f"/api/runs/{run_id}/feedback", {"username": "a", "rating": 2, "issue": "wrong", "video": "c.mp4"})
                detail = get_json(base_url, f"/api/runs/{run_id}")
                fid = int(detail["feedback"][0]["id"])
                # Correct a mis-scored rating.
                updated = post_json(base_url, f"/api/runs/{run_id}/feedback/{fid}", {"rating": 4.5, "issue": "actually good"})
                row = updated["feedback"][0]
                self.assertAlmostEqual(float(row["rating"]), 4.5)
                self.assertEqual(row["issue"], "actually good")
                # updated_at should advance past created_at.
                self.assertIsNotNone(row["updated_at"])
                # Clearing rating while keeping an issue is allowed.
                cleared = post_json(base_url, f"/api/runs/{run_id}/feedback/{fid}", {"rating": None})
                self.assertIsNone(cleared["feedback"][0]["rating"])
            finally:
                from v13_test_utils import stop_server
                stop_server(server, thread)

    def test_edit_missing_feedback_404(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            run_id = _make_run(db, workspace, "run")
            server, thread, base_url = start_server(db, workspace)
            try:
                import urllib.error
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    post_json(base_url, f"/api/runs/{run_id}/feedback/999", {"rating": 4})
                self.assertEqual(ctx.exception.code, 404)
            finally:
                from v13_test_utils import stop_server
                stop_server(server, thread)

    def test_cleanup_artifacts_drops_feedback(self) -> None:
        # The #26 case: cleaning a run's outputs must remove its orphaned scores.
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            run_id = _make_run(db, workspace, "run")
            # Put the run into a terminal state so cleanup-artifacts is allowed.
            _mark_completed(db, run_id)
            server, thread, base_url = start_server(db, workspace)
            try:
                post_json(base_url, f"/api/runs/{run_id}/feedback", {"username": "a", "rating": 4, "issue": "", "video": "c.mp4"})
                self.assertEqual(get_json(base_url, "/api/feedback")["total"], 1)
                preview = post_json(
                    base_url,
                    "/api/run-purge/preview",
                    {"request_type": "cleanup_artifacts", "run_ids": [run_id]},
                )
                cleanup = post_json(
                    base_url,
                    f"/api/runs/{run_id}/cleanup-artifacts",
                    {"preview_token": preview["preview_token"]},
                )
                deadline = time.time() + 5
                while time.time() < deadline:
                    request = get_json(
                        base_url,
                        f"/api/run-purge-requests/{cleanup['request_id']}",
                    )
                    if request["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.02)
                self.assertEqual(request["status"], "completed", request)
                self.assertEqual(len(get_json(base_url, f"/api/runs/{run_id}")["feedback"]), 0)
                self.assertEqual(get_json(base_url, "/api/feedback")["total"], 0)
            finally:
                from v13_test_utils import stop_server
                stop_server(server, thread)

    def test_deleted_run_excluded_from_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            run_id = _make_run(db, workspace, "run")
            _mark_completed(db, run_id)
            server, thread, base_url = start_server(db, workspace)
            try:
                post_json(base_url, f"/api/runs/{run_id}/feedback", {"username": "a", "rating": 4, "issue": "", "video": "c.mp4"})
                self.assertEqual(get_json(base_url, "/api/feedback")["total"], 1)
                preview = post_json(
                    base_url,
                    "/api/run-purge/preview",
                    {"request_type": "delete_run", "run_ids": [run_id]},
                )
                deletion = post_json(
                    base_url,
                    f"/api/runs/{run_id}/hide",
                    {"preview_token": preview["preview_token"]},
                )
                request = deletion
                for _attempt in range(100):
                    request = get_json(
                        base_url,
                        f"/api/run-purge-requests/{deletion['request_id']}",
                    )
                    if request["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.02)
                self.assertEqual(request["status"], "completed", request)
                self.assertEqual(get_json(base_url, "/api/feedback")["total"], 0)
            finally:
                from v13_test_utils import stop_server
                stop_server(server, thread)

    def test_stats_grouping_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            run_x = _make_run(db, workspace, "runX", model="RIFE", checkpoint="v4.ckpt", dataset="anime")
            run_y = _make_run(db, workspace, "runY", model="RIFE", checkpoint="v4.6.ckpt", dataset="anime")
            server, thread, base_url = start_server(db, workspace)
            try:
                post_json(base_url, f"/api/runs/{run_x}/feedback", {"username": "a", "rating": 3, "issue": "", "video": "clip.mp4"})
                post_json(base_url, f"/api/runs/{run_y}/feedback", {"username": "a", "rating": 5, "issue": "", "video": "clip.mp4"})

                stats = get_json(base_url, "/api/feedback")
                by_video = {row["video"]: row for row in stats["by_video"]}
                self.assertIn("clip.mp4", by_video)
                self.assertEqual(by_video["clip.mp4"]["count"], 2)
                by_ckpt = {(row["model_name"], row["checkpoint"]): row for row in stats["by_checkpoint"]}
                self.assertIn(("RIFE", "v4.ckpt"), by_ckpt)
                self.assertIn(("RIFE", "v4.6.ckpt"), by_ckpt)

                # Filter by checkpoint narrows the population.
                filtered = get_json(base_url, "/api/feedback?model=RIFE&checkpoint=v4.6.ckpt")
                self.assertEqual(filtered["total"], 1)
                self.assertAlmostEqual(filtered["average_rating"], 5.0)

                options = stats["filter_options"]
                self.assertIn("anime", options["datasets"])
                self.assertIn("RIFE", options["models"])
                self.assertIn("v4.ckpt", options["checkpoints"])
            finally:
                from v13_test_utils import stop_server
                stop_server(server, thread)


if __name__ == "__main__":
    unittest.main()
