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
from vfieval.file_inputs import decode_cache_key
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
            claimed = db.claim_next_job("decode-test", ["decode"])
            self.assertIsNotNone(claimed)
            self.assertEqual(int(claimed["id"]), job_id)

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
            patch("vfieval.ffmpeg_exe.resolve_ffmpeg", return_value=None),
            patch("vfieval.worker._module_available", side_effect=lambda name: name == "cv2"),
            patch("vfieval.worker._module_error", side_effect=lambda name: None if name == "cv2" else "missing"),
        ):
            capabilities = detect_capabilities()

        self.assertTrue(capabilities["decode_backends"]["opencv"]["available"])
        self.assertFalse(capabilities["decode_backends"]["ffmpeg"]["available"])
        self.assertIn("PATH", capabilities["decode_backends"]["ffmpeg"]["error"])

    def test_second_run_reuses_decode_cache_without_decoding_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "clip.mp4"
            video_path.write_bytes(b"not a real video but stable cache input")
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset(
                "video",
                str(root),
                True,
                source_type="video",
                decode_mode="video_gt_triplets",
                metadata={"frame_step": 1, "selected_videos": ["clip.mp4"]},
            )

            def fake_decode(_video_path, output_dir, _max_frames, decode_backend="auto", progress_callback=None):
                frames = []
                for index in range(5):
                    frame = output_dir / f"{index:06d}.png"
                    Image.new("RGB", (8, 8), (index, 0, 0)).save(frame)
                    frames.append(frame)
                if progress_callback:
                    progress_callback({"event": "video_done", "backend": "opencv", "frames": len(frames)})
                return frames, 24.0, [index / 24.0 for index in range(5)], {
                    "backend": decode_backend,
                    "fallback_reason": None,
                }

            first_run = db.create_run("first", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", [], create_inference_job=False)
            first_job = db.add_run_job(
                first_run,
                "decode",
                {"run_id": first_run, "dataset_id": dataset_id, "total_frames": 5},
                progress_total=5,
            )
            claimed = db.claim_next_job("decode-first", ["decode"])
            self.assertIsNotNone(claimed)
            self.assertEqual(int(claimed["id"]), first_job)
            with patch("vfieval.datasets._decode_video", side_effect=fake_decode) as decode_video:
                first_result = run_decode_job(db, workspace, first_job)
            self.assertEqual(first_result["samples"], 3)  # N - 2*frame_step = 5 - 2 = 3
            decode_video.assert_called_once()

            second_run = db.create_run("second", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", [], create_inference_job=False)
            second_job = db.add_run_job(
                second_run,
                "decode",
                {"run_id": second_run, "dataset_id": dataset_id, "total_frames": 5},
                progress_total=5,
            )
            claimed = db.claim_next_job("decode-second", ["decode"])
            self.assertIsNotNone(claimed)
            self.assertEqual(int(claimed["id"]), second_job)
            with (
                patch("vfieval.datasets._decode_video_ffmpeg") as ffmpeg_decode,
                patch("vfieval.datasets._decode_video_opencv") as opencv_decode,
            ):
                second_result = run_decode_job(db, workspace, second_job)

            ffmpeg_decode.assert_not_called()
            opencv_decode.assert_not_called()
            second_job_row = db.get_job(second_job)
            self.assertEqual(second_result["samples"], 3)  # N - 2*frame_step = 5 - 2 = 3
            self.assertEqual(second_job_row["result"]["phase"], "indexing_cached_frames")
            self.assertEqual(second_job_row["result"]["backend"], "cache")
            self.assertEqual(second_job_row["result"]["cache_hits"], 1)
            self.assertEqual(second_job_row["result"]["cache_misses"], 0)
            self.assertEqual(second_job_row["result"]["manifest_backend"], "ffmpeg")
            self.assertEqual(second_job_row["progress_current"], 5)

    def test_decode_cache_key_ignores_model_runtime_and_metric_choices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            video_path.write_bytes(b"same source video")

            base_key = decode_cache_key(video_path, "video_gt_triplets", 1, 12)
            changed_model = decode_cache_key(video_path, "video_gt_triplets", 1, 12)
            changed_checkpoint = decode_cache_key(video_path, "video_gt_triplets", 1, 12)
            changed_device = decode_cache_key(video_path, "video_gt_triplets", 1, 12)
            changed_metrics = decode_cache_key(video_path, "video_gt_triplets", 1, 12)

            self.assertEqual(base_key, changed_model)
            self.assertEqual(base_key, changed_checkpoint)
            self.assertEqual(base_key, changed_device)
            self.assertEqual(base_key, changed_metrics)

    def test_decode_cache_key_changes_when_sampling_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            video_path.write_bytes(b"same source video")

            base_key = decode_cache_key(video_path, "video_gt_triplets", 1, 12)

            self.assertNotEqual(base_key, decode_cache_key(video_path, "video_gt_triplets", 2, 12))
            self.assertNotEqual(base_key, decode_cache_key(video_path, "video_gt_triplets", 1, 24))

    def test_decode_cache_key_binds_requested_and_actual_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            video_path.write_bytes(b"same source video")

            auto_ffmpeg = decode_cache_key(
                video_path,
                "video_gt_triplets",
                1,
                None,
                requested_backend="auto",
                actual_backend="ffmpeg",
            )
            auto_opencv = decode_cache_key(
                video_path,
                "video_gt_triplets",
                1,
                None,
                requested_backend="auto",
                actual_backend="opencv",
            )
            explicit_opencv = decode_cache_key(
                video_path,
                "video_gt_triplets",
                1,
                None,
                requested_backend="opencv",
                actual_backend="opencv",
            )

            self.assertNotEqual(auto_ffmpeg, auto_opencv)
            self.assertNotEqual(auto_opencv, explicit_opencv)

    def test_run_detail_contains_decode_cache_reuse_copy(self) -> None:
        app_js = (ROOT / "src" / "vfieval" / "web" / "run-detail.js").read_text(encoding="utf-8")

        self.assertIn("Reusing decoded cache", app_js)
        self.assertIn("rebuilding this Run's sample index", app_js)
        self.assertIn("Cache miss", app_js)


if __name__ == "__main__":
    unittest.main()
