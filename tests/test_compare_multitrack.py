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

from vfieval.pipeline.inference import run_inference_job

from v13_test_utils import add_completed_pred_run, make_workspace, post_json, start_server, stop_server, write_mp4


class CompareMultitrackTests(unittest.TestCase):
    def test_structured_compare_writes_track_scoped_video_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            gt_path = Path(tmp) / "videos" / "anime" / "clip.mp4"
            pred_a_path = workspace.root / "pred-a.mp4"
            pred_b_path = workspace.root / "pred-b.mp4"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            write_mp4(gt_path, [(0, 0, 0), (20, 0, 0), (40, 0, 0)])
            write_mp4(pred_a_path, [(0, 0, 0), (0, 20, 0), (0, 40, 0)])
            write_mp4(pred_b_path, [(0, 0, 0), (0, 0, 20), (0, 0, 40)])
            run_a = add_completed_pred_run(db, workspace, "ModelA", pred_a_path)
            run_b = add_completed_pred_run(db, workspace, "ModelB", pred_b_path)

            payload = {
                "run_type": "video_compare",
                "reference": {"kind": "video_group", "group": "anime", "video": "clip.mp4"},
                "distorted": [
                    {"kind": "run_artifact", "run_id": run_a, "video": "clip", "label": "ModelA"},
                    {"kind": "run_artifact", "run_id": run_b, "video": "clip", "label": "ModelB"},
                ],
                "metrics": [],
            }

            server, thread, base_url = start_server(db, workspace)
            try:
                preflight = post_json(base_url, "/api/preflight", payload)
                self.assertTrue(preflight["ok"], preflight)
                self.assertEqual(preflight["alignment"]["track_count"], 2)

                with patch("vfieval.server._start_local_inference_worker", return_value=None):
                    created = post_json(base_url, "/api/runs", payload)
                run_id = int(created["run_id"])
                job_id = int(db.get_run(run_id)["inference_job_id"])
                run_inference_job(db, workspace, job_id)

                run = db.get_run(run_id)
                self.assertEqual(run["status"], "completed", run)
                samples = db.list_samples(int(run["dataset_id"]))
                self.assertEqual(len(samples), 6)
                self.assertIn("clip__ModelA__000000", {row["name"] for row in samples})
                self.assertIn("clip__ModelB__000000", {row["name"] for row in samples})

                pred_videos = db.list_run_artifacts(run_id, kind="pred_video")
                diff_videos = db.list_run_artifacts(run_id, kind="diff_video")
                gt_videos = db.list_run_artifacts(run_id, kind="gt_video")
                self.assertEqual(len(pred_videos), 2)
                self.assertEqual(len(diff_videos), 2)
                self.assertEqual(len(gt_videos), 1)
                pred_paths = {Path(row["path"]).relative_to(workspace.runs_dir / str(run_id)).as_posix() for row in pred_videos}
                self.assertIn("videos/clip/ModelA/pred.mp4", pred_paths)
                self.assertIn("videos/clip/ModelB/pred.mp4", pred_paths)
                self.assertEqual(Path(gt_videos[0]["path"]).relative_to(workspace.runs_dir / str(run_id)).as_posix(), "videos/clip/gt.mp4")
                self.assertEqual({row["metadata"]["compare_track_label"] for row in pred_videos}, {"ModelA", "ModelB"})

                pred_sample_artifacts = db.list_artifacts_by_sample(int(samples[0]["id"]), kind="pred")
                self.assertIn(pred_sample_artifacts[0]["metadata"]["compare_track_label"], {"ModelA", "ModelB"})
            finally:
                stop_server(server, thread)


if __name__ == "__main__":
    unittest.main()
