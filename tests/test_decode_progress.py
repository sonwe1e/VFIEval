from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.orchestration import create_inference_jobs_for_run
from vfieval.pipeline.decode_runner import run_decode_job
from vfieval.server import _create_run_from_files
from vfieval.worker import WorkerOptions, detect_capabilities, run_worker


class DecodeProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_project_root = os.environ.get("VFIEVAL_PROJECT_ROOT")
        os.environ["VFIEVAL_PROJECT_ROOT"] = str(ROOT)

    def tearDown(self) -> None:
        if self._old_project_root is None:
            os.environ.pop("VFIEVAL_PROJECT_ROOT", None)
        else:
            os.environ["VFIEVAL_PROJECT_ROOT"] = self._old_project_root

    def test_file_run_creation_queues_decode_job_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()

            with (
                patch("vfieval.server.start_decode_worker") as start_decode_worker,
                patch("vfieval.server.scan_dataset") as scan_dataset,
            ):
                created = _create_run_from_files(
                    db,
                    workspace,
                    {
                        "model_file": "test_average.py",
                        "video_group": "test_style",
                        "selected_videos": ["blocks_motion.avi"],
                        "device": "cpu",
                        "precision": "fp32",
                        "max_frames": 4,
                    },
                )

            scan_dataset.assert_not_called()
            start_decode_worker.assert_called_once()
            run = db.get_run(int(created["run_id"]))
            jobs = db.list_run_jobs(int(created["run_id"]))
            self.assertEqual(run["status"], "decoding")
            self.assertEqual([(job["role"], job["status"]) for job in jobs], [("decode", "queued")])
            self.assertEqual(jobs[0]["progress_total"], 4)

    def test_decode_runner_publishes_frame_progress_and_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("video", str(Path(tmp)), True, source_type="video")
            run_id = db.create_run("decode-run", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", [], create_inference_job=False)
            job_id = db.add_run_job(
                run_id,
                "decode",
                {"run_id": run_id, "dataset_id": dataset_id, "total_frames": 4, "decode_backend": "ffmpeg"},
                progress_total=4,
            )

            def fake_scan(_db, _workspace, _dataset_id, progress_callback=None, decode_backend="auto"):
                self.assertEqual(decode_backend, "ffmpeg")
                progress_callback(
                    {
                        "backend": "ffmpeg",
                        "video_name": "clip.mp4",
                        "decoded_frames": 2,
                        "total_frames": 4,
                        "cache_hits": 0,
                        "cache_misses": 1,
                    }
                )
                return 2

            with patch("vfieval.pipeline.decode_runner.scan_dataset", side_effect=fake_scan):
                result = run_decode_job(db, workspace, job_id)

            job = db.get_job(job_id)
            run = db.get_run(run_id)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(job["progress_current"], 2)
            self.assertEqual(job["result"]["backend"], "ffmpeg")
            self.assertEqual(job["result"]["current_video"], "clip.mp4")
            self.assertEqual(run["status"], "decoding")
            self.assertEqual(run["progress_current"], 2)

    def test_decode_worker_schedules_inference_after_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            frame = root / "frame.png"
            Image.new("RGB", (8, 8), (0, 0, 0)).save(frame)
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("decoded", str(root), True)
            db.add_sample(dataset_id, "sample", str(frame), str(frame), str(frame), {"video_file": "clip.mp4"})
            run_id = db.create_run("decode-worker", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", [], create_inference_job=False)
            decode_job_id = db.add_run_job(
                run_id,
                "decode",
                {"run_id": run_id, "dataset_id": dataset_id, "total_frames": 1},
                progress_total=1,
            )

            with (
                patch("vfieval.worker.run_decode_job", return_value={"status": "completed", "samples": 1, "decoded_frames": 1}),
                patch("vfieval.worker.start_workers_for_run") as start_workers,
            ):
                run_worker(db, workspace, WorkerOptions(role="decode", once=True, worker_id="decode-test"))

            self.assertEqual(db.get_job(decode_job_id)["status"], "completed")
            inference_jobs = db.list_run_jobs(run_id, "inference")
            self.assertEqual(len(inference_jobs), 1)
            self.assertEqual(inference_jobs[0]["status"], "queued")
            self.assertEqual(inference_jobs[0]["progress_total"], 1)
            start_workers.assert_called_once_with(db, workspace, run_id)

    def test_multi_device_inference_jobs_are_created_after_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            frame = root / "frame.png"
            Image.new("RGB", (8, 8), (0, 0, 0)).save(frame)
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("decoded", str(root), True)
            db.add_sample(dataset_id, "a", str(frame), str(frame), str(frame), {"video_file": "a.mp4"})
            db.add_sample(dataset_id, "b", str(frame), str(frame), str(frame), {"video_file": "b.mp4"})
            run_id = db.create_run(
                "multi",
                model_id,
                dataset_id,
                8,
                8,
                1,
                "multi_npu",
                "fp32",
                ["lpips_vit_patch"],
                metadata={"execution_mode": "multi_npu", "devices": ["npu:0", "npu:1"]},
                create_inference_job=False,
            )

            create_inference_jobs_for_run(db, run_id)

            jobs = db.list_run_jobs(run_id, "inference")
            self.assertEqual([job["device"] for job in jobs], ["npu:0", "npu:1"])
            self.assertEqual([job["shard_index"] for job in jobs], [0, 1])
            self.assertTrue(all(int(job["progress_total"] or 0) == 1 for job in jobs))

    def test_decode_backend_capabilities_include_missing_reasons(self) -> None:
        with (
            patch("vfieval.worker.shutil.which", return_value=None),
            patch("vfieval.worker._module_available", side_effect=lambda name: name == "cv2"),
            patch("vfieval.worker._module_error", side_effect=lambda name: None if name == "cv2" else "missing"),
        ):
            capabilities = detect_capabilities()

        self.assertTrue(capabilities["decode_backends"]["opencv"]["available"])
        self.assertFalse(capabilities["decode_backends"]["ffmpeg"]["available"])
        self.assertIn("PATH", capabilities["decode_backends"]["ffmpeg"]["error"])


if __name__ == "__main__":
    unittest.main()
