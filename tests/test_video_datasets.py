from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from vfieval.config import WorkspaceConfig
from vfieval.datasets import scan_dataset
from vfieval.db import Database
from vfieval.worker import WorkerOptions, run_worker


@unittest.skipIf(cv2 is None or np is None, "opencv-python and numpy are required for video dataset tests")
class VideoDatasetTests(unittest.TestCase):
    def test_video_gt_triplets_and_example_model_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "clip.avi"
            _write_video(video_path, frames=5)

            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model(
                "average",
                "examples.models.average_flow_mask:create_model",
                None,
                4,
                4,
            )
            dataset_id = db.create_dataset(
                "video-demo",
                str(video_path),
                has_gt=True,
                source_type="video",
                decode_mode="video_gt_triplets",
                metadata={"frame_step": 1},
            )

            self.assertEqual(scan_dataset(db, workspace, dataset_id), 4)
            dataset = db.get_dataset(dataset_id)
            self.assertEqual(dataset["source_type"], "video")
            self.assertEqual(dataset["video_count"], 1)
            self.assertEqual(dataset["frame_count"], 5)
            self.assertTrue(Path(dataset["decoded_root_path"]).exists())
            samples = db.list_samples(dataset_id)
            # N - frame_step samples (5 - 1 = 4): the interior 3 keep strict GT
            # triples; the last is a clamped boundary sample whose gt_path points
            # at the last source frame so pred/gt videos stay frame-aligned
            # (the last GT is flagged as approximate in metadata).
            self.assertEqual(len(samples), 4)
            self.assertTrue(all(sample["gt_path"] for sample in samples))
            self.assertEqual(
                [bool(sample.get("metadata", {}).get("clamped_boundary")) for sample in samples],
                [False, False, False, True],
            )
            self.assertEqual(samples[0]["metadata"]["decode_mode"], "video_gt_triplets")

            run_id = db.create_run("average-video", model_id, dataset_id, 4, 4, 2, "cpu", "fp32", [])
            run_worker(db, workspace, WorkerOptions(role="inference", once=True, worker_id="video-inference"))

            run = db.get_run(run_id)
            self.assertEqual(run["status"], "completed")
            inference_job_id = int(run["inference_job_id"])
            self.assertEqual(len(db.list_artifacts(job_id=inference_job_id, kind="pred")), 4)
            # pred video is the per-sample pred stitched together (4 frames N-step).
            self.assertEqual(len(db.list_artifacts(job_id=inference_job_id, kind="pred_video")), 1)
            # gt video is still assembled — the boundary sample keeps a gt_path
            # (pointing at the clamped last source frame), so pred/gt videos are
            # frame-aligned despite the last GT being approximate.
            self.assertEqual(len(db.list_artifacts(job_id=inference_job_id, kind="gt_video")), 1)

    def test_video_triplet_count_equals_source_minus_step(self) -> None:
        # Regression test for the "12 frames produced only 10 preds" bug: after the
        # fix, an N-frame source produces N - frame_step samples (one interpolated
        # frame per adjacent pair), with only the last sample being a clamped
        # boundary pair (metadata flag ``clamped_boundary=True``).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "clip.avi"
            _write_video(video_path, frames=12)

            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            dataset_id = db.create_dataset(
                "count-12",
                str(video_path),
                has_gt=True,
                source_type="video",
                decode_mode="video_gt_triplets",
                metadata={"frame_step": 1},
            )
            self.assertEqual(scan_dataset(db, workspace, dataset_id), 11)  # N - step
            samples = db.list_samples(dataset_id)
            self.assertEqual(len(samples), 11)
            self.assertEqual(
                [bool(s.get("metadata", {}).get("clamped_boundary")) for s in samples],
                [False] * 10 + [True],
            )
            # All samples carry a gt path so pred and gt videos stay aligned.
            self.assertTrue(all(s["gt_path"] for s in samples))

    def test_video_pairs_without_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "clip.avi"
            _write_video(video_path, frames=4)

            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            dataset_id = db.create_dataset(
                "video-pairs",
                str(video_path),
                has_gt=False,
                source_type="video",
                decode_mode="video_pairs",
                metadata={"frame_step": 1},
            )

            self.assertEqual(scan_dataset(db, workspace, dataset_id), 3)
            samples = db.list_samples(dataset_id)
            self.assertEqual(len(samples), 3)
            self.assertTrue(all(sample["gt_path"] is None for sample in samples))
            self.assertEqual(samples[0]["metadata"]["decode_mode"], "video_pairs")


def _write_video(path: Path, frames: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (8, 8))
    if not writer.isOpened():
        raise RuntimeError(f"failed to create test video: {path}")
    try:
        for index in range(frames):
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            frame[:, :, 0] = index * 20
            frame[:, :, 1] = 40
            frame[:, :, 2] = 80
            writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    unittest.main()
