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
from vfieval import compare_inputs

from v13_test_utils import add_completed_pred_run, get_json, make_workspace, post_json, start_server, stop_server, write_mp4


class CompareMultitrackTests(unittest.TestCase):
    def test_video_compare_probe_uses_fast_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, _db = make_workspace(tmp)
            video_path = Path(tmp) / "clip.mp4"
            video_path.write_bytes(b"placeholder")
            with patch(
                "vfieval.compare_inputs.inspect_video",
                return_value={
                    "decodable": True,
                    "frame_count": 3,
                    "width": 8,
                    "height": 6,
                    "fps": 24.0,
                    "duration_seconds": 0.125,
                },
            ) as inspect_video:
                result = compare_inputs.inspect_compare_path(workspace, video_path)

            inspect_video.assert_called_once_with(video_path.resolve(), workspace, exact=False)
            self.assertEqual(result["frame_count"], 3)

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
            run_a = add_completed_pred_run(db, workspace, "ModelA", pred_a_path, source_video_path=gt_path)
            run_b = add_completed_pred_run(db, workspace, "ModelB", pred_b_path, source_video_path=gt_path)

            payload = {
                "run_type": "video_compare",
                "reference": {"kind": "video_group", "group": "anime", "video": "clip.mp4"},
                "distorted": [
                    {"kind": "run_artifact", "run_id": run_a, "video": "clip", "label": "ModelA"},
                    {"kind": "run_artifact", "run_id": run_b, "video": "clip", "label": "ModelB"},
                ],
                "extra_layers": [
                    {"source": "run_artifact", "run_id": run_a, "kinds": ["flowt_0", "mask0"]},
                    {"source": "run_artifact", "run_id": run_b, "kinds": ["flowt_0", "mask0"]},
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
                self.assertEqual(int(db.claim_next_job("compare-multitrack", ["inference"])["id"]), job_id)
                result = run_inference_job(db, workspace, job_id)
                self.assertTrue(db.complete_job(job_id, result.__dict__))

                run = db.get_run(run_id)
                self.assertEqual(run["status"], "completed", run)
                samples = db.list_samples(int(run["dataset_id"]))
                self.assertEqual(len(samples), 6)
                self.assertIn("clip__ModelA__000000", {row["name"] for row in samples})
                self.assertIn("clip__ModelB__000000", {row["name"] for row in samples})

                pred_videos = db.list_run_artifacts(run_id, kind="pred_video")
                diff_videos = db.list_run_artifacts(run_id, kind="diff_video")
                gt_videos = db.list_run_artifacts(run_id, kind="gt_video")
                self.assertEqual(pred_videos, [])
                self.assertEqual(len(diff_videos), 2)
                self.assertEqual(len(gt_videos), 1)
                diff_paths = {Path(row["path"]).relative_to(workspace.runs_dir / str(run_id)).as_posix() for row in diff_videos}
                self.assertIn("videos/clip/ModelA/diff.mp4", diff_paths)
                self.assertIn("videos/clip/ModelB/diff.mp4", diff_paths)
                self.assertEqual(Path(gt_videos[0]["path"]).relative_to(workspace.runs_dir / str(run_id)).as_posix(), "videos/clip/gt.mp4")
                self.assertEqual({row["metadata"]["compare_track_label"] for row in diff_videos}, {"ModelA", "ModelB"})

                pred_sample_artifacts = db.list_artifacts_by_sample(int(samples[0]["id"]), kind="pred")
                self.assertIn(pred_sample_artifacts[0]["metadata"]["compare_track_label"], {"ModelA", "ModelB"})

                model_a_sample = next(row for row in samples if row["name"] == "clip__ModelA__000000")
                sample_detail = get_json(base_url, f"/api/runs/{run_id}/samples/{model_a_sample['id']}")
                layers = sample_detail["compare_layers"]
                self.assertEqual({row["track_label"] for row in layers}, {"ModelA", "ModelB"})
                self.assertIn("flowt_0", {row["kind"] for row in layers})
                self.assertIn("mask0", {row["kind"] for row in layers})
            finally:
                stop_server(server, thread)

    def test_compare_with_mismatched_frame_count_is_rejected(self) -> None:
        # External inputs are never silently truncated or offset.
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            gt_path = Path(tmp) / "videos" / "anime" / "clip-gt.mp4"
            pred_path = workspace.root / "pred.mp4"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            # GT: 12 frames at 16x8  —  the "full" source clip
            gt_colors = [(i * 10, 20, 30) for i in range(12)]
            pred_colors = [(30, i * 10, 50) for i in range(11)]
            write_mp4(gt_path, gt_colors, size=(16, 8))           # 12 frames, res 16x8
            write_mp4(pred_path, pred_colors, size=(8, 8))       # 11 frames, res 8x8
            _gt_probe = __import__("vfieval.file_inputs", fromlist=["inspect_video"]).inspect_video(
                gt_path, workspace, exact=True
            )
            _pred_probe = __import__("vfieval.file_inputs", fromlist=["inspect_video"]).inspect_video(
                pred_path, workspace, exact=True
            )
            self.assertEqual(_gt_probe["frame_count"], 12)
            self.assertEqual(_pred_probe["frame_count"], 11)
            pred_run = add_completed_pred_run(
                db, workspace, "Pred", pred_path, video_name="clip", sample_count=11, size=(8, 8),
                source_video_path=gt_path,
            )

            payload = {
                "run_type": "video_compare",
                "reference": {"kind": "video_group", "group": "anime", "video": "clip-gt.mp4"},
                "distorted": [{"kind": "run_artifact", "run_id": pred_run, "video": "clip", "label": "Pred"}],
                "metrics": [],
            }
            server, thread, base_url = start_server(db, workspace)
            try:
                preflight = post_json(base_url, "/api/preflight", payload)
                self.assertFalse(preflight["ok"], preflight)
                self.assertIn("matching frame counts", preflight["errors"][0]["message"])
            finally:
                stop_server(server, thread)


    def test_compare_with_mismatched_resolution_builds_explicit_alignment_plan(self) -> None:
        # Item-bound model outputs keep strict time but normalize spatial size.
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            gt_path = Path(tmp) / "videos" / "anime" / "clip-gt.mp4"
            pred_path = workspace.root / "pred.mp4"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            # GT 16x8, Pred 8x8: GT is the higher-res side on the W axis.
            write_mp4(gt_path, [(10, 20, 30)] * 3, size=(16, 8))
            write_mp4(pred_path, [(30, 20, 10)] * 3, size=(8, 8))
            pred_run = add_completed_pred_run(
                db, workspace, "PredHigh", pred_path, video_name="clip", sample_count=3, size=(8, 8),
                source_video_path=gt_path,
            )

            payload = {
                "run_type": "video_compare",
                "reference": {"kind": "video_group", "group": "anime", "video": "clip-gt.mp4"},
                "distorted": [{"kind": "run_artifact", "run_id": pred_run, "video": "clip", "label": "Pred"}],
                "metrics": [],
            }
            server, thread, base_url = start_server(db, workspace)
            try:
                preflight = post_json(base_url, "/api/preflight", payload)
                self.assertTrue(preflight["ok"], preflight)
                self.assertEqual(preflight["alignment_plan"]["target"]["width"], 8)
                self.assertEqual(preflight["alignment_plan"]["target"]["height"], 8)
                gt_report = preflight["alignment_plan"]["sources"]["gt"]
                self.assertEqual(gt_report["direction"], "downscale")
                self.assertTrue(gt_report["aspect_changed"])
            finally:
                stop_server(server, thread)


    def test_two_preds_sharing_a_label_do_not_collapse(self) -> None:
        # Regression: two selected preds that resolve to the same track label
        # (e.g. two runs sharing an auto-generated name) used to collide on the
        # sample name / artifact dir and the second silently overwrote the
        # first, so the compare showed one GT and one pred instead of two preds.
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            gt_path = Path(tmp) / "videos" / "anime" / "clip.mp4"
            pred_a_path = workspace.root / "pred-a.mp4"
            pred_b_path = workspace.root / "pred-b.mp4"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            write_mp4(gt_path, [(0, 0, 0), (20, 0, 0), (40, 0, 0)])
            write_mp4(pred_a_path, [(0, 0, 0), (0, 20, 0), (0, 40, 0)])
            write_mp4(pred_b_path, [(0, 0, 0), (0, 0, 20), (0, 0, 40)])
            run_a = add_completed_pred_run(db, workspace, "ModelA", pred_a_path, source_video_path=gt_path)
            run_b = add_completed_pred_run(db, workspace, "ModelB", pred_b_path, source_video_path=gt_path)

            # Both tracks intentionally carry the SAME label.
            payload = {
                "run_type": "video_compare",
                "reference": {"kind": "video_group", "group": "anime", "video": "clip.mp4"},
                "distorted": [
                    {"kind": "run_artifact", "run_id": run_a, "video": "clip", "label": "Model"},
                    {"kind": "run_artifact", "run_id": run_b, "video": "clip", "label": "Model"},
                ],
                "metrics": [],
            }

            server, thread, base_url = start_server(db, workspace)
            try:
                preflight = post_json(base_url, "/api/preflight", payload)
                self.assertTrue(preflight["ok"], preflight)
                with patch("vfieval.server._start_local_inference_worker", return_value=None):
                    created = post_json(base_url, "/api/runs", payload)
                run_id = int(created["run_id"])
                job_id = int(db.get_run(run_id)["inference_job_id"])
                self.assertEqual(int(db.claim_next_job("compare-labels", ["inference"])["id"]), job_id)
                result = run_inference_job(db, workspace, job_id)
                self.assertTrue(db.complete_job(job_id, result.__dict__))

                run = db.get_run(run_id)
                self.assertEqual(run["status"], "completed", run)
                # Two tracks × 3 frames = 6 samples; a collision would drop to 3.
                samples = db.list_samples(int(run["dataset_id"]))
                self.assertEqual(len(samples), 6)
                # Both preds must survive as distinct tracks, while Compare
                # still publishes no reusable pred_video artifact.
                pred_videos = db.list_run_artifacts(run_id, kind="pred_video")
                self.assertEqual(pred_videos, [])
                diff_videos = db.list_run_artifacts(run_id, kind="diff_video")
                self.assertEqual(len(diff_videos), 2)
                labels = {row["metadata"].get("compare_track_label") for row in diff_videos}
                self.assertEqual(len(labels), 2, labels)
            finally:
                stop_server(server, thread)


    def test_source_frame_indices_reconstruct_pred_aligned_gt(self) -> None:
        # Source-clip GT: the pred carries source_frame_indices, so Compare
        # selects exactly those source frames as its GT (head-offset) instead of
        # keeping a per-run gt_video. A 5-frame source clip inferred at step 1
        # yields a 4-frame pred whose frame i approximates source_frames[i+1];
        # so the aligned GT is source_frames[1:5] and every sample's GT must
        # match the source frame at index+1 — never the tail-trimmed source[i].
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            from PIL import Image as _Image
            workspace, db = make_workspace(tmp)
            gt_path = Path(tmp) / "videos" / "anime" / "clip.mp4"
            pred_path = workspace.root / "pred.mp4"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            # Distinct per-frame colors so we can prove which source frame the GT
            # sample resolved to. Source has 5 frames, pred has 4 (N-step, step=1).
            source_colors = [(i * 40, 0, 0) for i in range(5)]
            pred_colors = [(0, (i + 1) * 40, 0) for i in range(4)]
            write_mp4(gt_path, source_colors, size=(8, 8))
            write_mp4(pred_path, pred_colors, size=(8, 8))
            pred_run = add_completed_pred_run(
                db,
                workspace,
                "Pred",
                pred_path,
                video_name="clip",
                sample_count=4,
                size=(8, 8),
                source_video_path=gt_path,
                source_frame_indices=[1, 2, 3, 4],
                frame_step=1,
            )

            payload = {
                "run_type": "video_compare",
                "reference": {"kind": "video_group", "group": "anime", "video": "clip.mp4"},
                "distorted": [{"kind": "run_artifact", "run_id": pred_run, "video": "clip", "label": "Pred"}],
                "metrics": [],
            }
            server, thread, base_url = start_server(db, workspace)
            try:
                preflight = post_json(base_url, "/api/preflight", payload)
                self.assertTrue(preflight["ok"], preflight)
                with patch("vfieval.server._start_local_inference_worker", return_value=None):
                    created = post_json(base_url, "/api/runs", payload)
                run_id = int(created["run_id"])
                job_id = int(db.get_run(run_id)["inference_job_id"])
                self.assertEqual(int(db.claim_next_job("compare-source-indices", ["inference"])["id"]), job_id)
                result = run_inference_job(db, workspace, job_id)
                self.assertTrue(db.complete_job(job_id, result.__dict__))

                run = db.get_run(run_id)
                self.assertEqual(run["status"], "completed", run)
                # 4 pred frames → 4 samples (source_frames[1:5]), not the 5 the
                # raw clip would give under a naive head-of-source pairing.
                samples = db.list_samples(int(run["dataset_id"]))
                self.assertEqual(len(samples), 4)
                # Each sample's GT must be source_frames[index+1]. Verify by
                # color: sample i's GT red channel ≈ (i+1)*40. MP4 is lossy so we
                # allow a small delta — but the correct frame (i+1)*40 and the
                # tail-trim wrong frame i*40 differ by 40, far outside the delta,
                # so this still proves indexed (not tail-trim) selection.
                for sample in samples:
                    frame_index = int((sample["metadata"] or {}).get("frame_index"))
                    with _Image.open(sample["gt_path"]) as img:
                        red = img.convert("RGB").getpixel((0, 0))[0]
                    self.assertAlmostEqual(red, (frame_index + 1) * 40, delta=8, msg=sample["name"])
            finally:
                stop_server(server, thread)

    def test_invalid_source_frame_indices_do_not_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            gt_path = Path(tmp) / "videos" / "anime" / "clip.mp4"
            pred_path = workspace.root / "pred.mp4"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            write_mp4(gt_path, [(0, 0, 0), (20, 0, 0), (40, 0, 0)])
            write_mp4(pred_path, [(0, 1, 0), (20, 1, 0), (40, 1, 0)])
            pred_run = add_completed_pred_run(
                db,
                workspace,
                "Pred",
                pred_path,
                sample_count=3,
                source_video_path=gt_path,
                source_frame_indices=[0, 1, 99],
            )
            payload = {
                "run_type": "video_compare",
                "reference": {"kind": "video_group", "group": "anime", "video": "clip.mp4"},
                "distorted": [{"kind": "run_artifact", "run_id": pred_run, "video": "clip", "label": "Pred"}],
            }
            server, thread, base_url = start_server(db, workspace)
            try:
                preflight = post_json(base_url, "/api/preflight", payload)
                self.assertFalse(preflight["ok"], preflight)
                self.assertIn("outside the reference", preflight["errors"][0]["message"])
            finally:
                stop_server(server, thread)


if __name__ == "__main__":
    unittest.main()
