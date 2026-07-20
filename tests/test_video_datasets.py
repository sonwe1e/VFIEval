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

            self.assertEqual(scan_dataset(db, workspace, dataset_id), 3)
            dataset = db.get_dataset(dataset_id)
            self.assertEqual(dataset["source_type"], "video")
            self.assertEqual(dataset["video_count"], 1)
            self.assertEqual(dataset["frame_count"], 5)
            self.assertEqual(
                dataset["metadata"]["evaluation_contract"], "midpoint-triplet-v2"
            )
            self.assertEqual(len(dataset["metadata"]["decoder_reports"]), 1)
            decoder_report = dataset["metadata"]["decoder_reports"][0]
            self.assertIn(decoder_report["actual_backend"], {"ffmpeg", "opencv"})
            self.assertTrue(decoder_report["decoder_identity"])
            self.assertTrue(Path(dataset["decoded_root_path"]).exists())
            samples = db.list_samples(dataset_id)
            # midpoint-triplet-v2 publishes only real symmetric triples:
            # N - 2*frame_step = 5 - 2 = 3.
            self.assertEqual(len(samples), 3)
            self.assertTrue(all(sample["gt_path"] for sample in samples))
            self.assertTrue(
                all(not sample.get("metadata", {}).get("clamped_boundary") for sample in samples)
            )
            self.assertEqual(samples[0]["metadata"]["decode_mode"], "video_gt_triplets")
            self.assertEqual(
                samples[0]["metadata"]["evaluation_contract"], "midpoint-triplet-v2"
            )

            run_id = db.create_run("average-video", model_id, dataset_id, 4, 4, 2, "cpu", "fp32", [])
            run_worker(db, workspace, WorkerOptions(role="inference", once=True, worker_id="video-inference"))

            run = db.get_run(run_id)
            self.assertEqual(run["status"], "completed")
            inference_job_id = int(run["inference_job_id"])
            self.assertEqual(len(db.list_artifacts(job_id=inference_job_id, kind="pred")), 3)
            # Pred and GT videos contain only the three trustworthy midpoint samples.
            self.assertEqual(len(db.list_artifacts(job_id=inference_job_id, kind="pred_video")), 1)
            self.assertEqual(len(db.list_artifacts(job_id=inference_job_id, kind="gt_video")), 1)

    def test_video_triplet_count_equals_source_minus_two_steps(self) -> None:
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
            self.assertEqual(scan_dataset(db, workspace, dataset_id), 10)  # N - 2*step
            samples = db.list_samples(dataset_id)
            self.assertEqual(len(samples), 10)
            self.assertTrue(all(not s.get("metadata", {}).get("clamped_boundary") for s in samples))
            self.assertTrue(all(s["gt_path"] for s in samples))

    def test_frame_step_two_keeps_only_symmetric_triplets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "clip.avi"
            _write_video(video_path, frames=8)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            dataset_id = db.create_dataset(
                "step-two",
                str(video_path),
                has_gt=True,
                source_type="video",
                decode_mode="video_gt_triplets",
                metadata={"frame_step": 2},
            )

            self.assertEqual(scan_dataset(db, workspace, dataset_id), 4)
            samples = db.list_samples(dataset_id)
            self.assertEqual(
                [
                    (
                        sample["metadata"]["img0_index"],
                        sample["metadata"]["gt_index"],
                        sample["metadata"]["img1_index"],
                    )
                    for sample in samples
                ],
                [(0, 2, 4), (1, 3, 5), (2, 4, 6), (3, 5, 7)],
            )

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
