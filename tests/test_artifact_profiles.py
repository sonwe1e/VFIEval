from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vfieval.config import WorkspaceConfig
from vfieval.datasets import scan_triplet_dataset
from vfieval.db import Database
from vfieval.orchestration import partition_samples_by_video
from vfieval.pipeline.inference import _NpuSmiSampler
from vfieval.performance import execution_profile_identity, record_execution_profile, recommend_execution_profile
from vfieval.worker import WorkerOptions, run_worker


class ArtifactProfileTests(unittest.TestCase):
    def test_execution_profile_fingerprint_and_fastest_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            models_dir = workspace.root.parent / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            (models_dir / "test_average.py").write_text("class Model:\n    pass\n", encoding="utf-8")
            db = Database(workspace.db_path)
            db.init()
            payload = {
                "model_file": "test_average.py",
                "checkpoint": "none",
                "device": "cpu",
                "devices": ["cpu"],
                "execution_mode": "single",
                "height": 64,
                "width": 96,
                "precision": "fp32",
                "artifact_profile": "benchmark",
            }
            identity = execution_profile_identity(workspace, payload)
            record_execution_profile(db, identity, {"batch_size": 2}, {"steady_state_fps": 10})
            record_execution_profile(db, identity, {"batch_size": 1}, {"steady_state_fps": 5})
            recommended = recommend_execution_profile(db, workspace, payload)
            self.assertEqual(recommended["settings"]["batch_size"], 2)
            self.assertEqual(recommended["device_model"], "CPU")

    def test_npu_smi_usage_parser_is_optional_and_stable(self) -> None:
        parsed = _NpuSmiSampler._parse("AI Core Utilization Rate : 73\nMemory Usage : 41")
        self.assertEqual(parsed, {"aicore_percent": 73.0, "memory_percent": 41.0})

    def test_long_single_video_is_split_into_contiguous_balanced_segments(self) -> None:
        samples = [
            {"id": index + 1, "name": f"clip-{index}", "metadata": {"video_name": "clip", "frame_index": index}}
            for index in range(17)
        ]
        partitions = partition_samples_by_video(samples, ["npu:0", "npu:1", "npu:2", "npu:3"])
        self.assertEqual(len(partitions), 4)
        self.assertEqual(sorted(value for part in partitions for value in part), list(range(1, 18)))
        self.assertLessEqual(max(map(len, partitions)) - min(map(len, partitions)), 1)
        for part in partitions:
            self.assertEqual(part, list(range(min(part), max(part) + 1)))

    def test_completed_deferred_shards_queue_one_finalize_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db, model_id, dataset_id = self._workspace(Path(tmp))
            run_id = db.create_run(
                "multi",
                model_id,
                dataset_id,
                12,
                16,
                1,
                "multi_npu",
                "fp16",
                [],
                metadata={"execution_mode": "multi_npu", "devices": ["npu:0", "npu:1"]},
                create_inference_job=False,
            )
            job_ids = [
                db.add_run_job(
                    run_id,
                    "inference",
                    {"run_id": run_id, "defer_video_finalize": True, "shard_count": 2},
                    progress_total=1,
                    shard_index=index,
                    device=f"npu:{index}",
                )
                for index in range(2)
            ]
            for job_id in job_ids:
                db.complete_job(job_id, {"samples": 1, "performance": {"total_wall_seconds": 1}})
            self.assertTrue(db.maybe_complete_multi_run_inference(run_id))
            self.assertEqual(db.get_run(run_id)["status"], "finalize_queued")
            finalize = db.list_run_jobs(run_id, "finalize")
            self.assertEqual(len(finalize), 1)
            self.assertTrue(db.maybe_complete_multi_run_inference(run_id))
            self.assertEqual(len(db.list_run_jobs(run_id, "finalize")), 1)

    def _workspace(self, root: Path):
        dataset_root = root / "dataset"
        for folder in ("img0", "img1", "gt"):
            (dataset_root / folder).mkdir(parents=True)
        for index in range(3):
            name = f"sample-{index}.png"
            Image.new("RGB", (16, 12), (index * 10, 0, 0)).save(dataset_root / "img0" / name)
            Image.new("RGB", (16, 12), (0, index * 10, 0)).save(dataset_root / "img1" / name)
            Image.new("RGB", (16, 12), (0, 0, index * 10)).save(dataset_root / "gt" / name)
        workspace = WorkspaceConfig.from_root(root / ".vfieval")
        workspace.ensure()
        db = Database(workspace.db_path)
        db.init()
        model_id = db.register_model("dummy", "dummy", None, 12, 16)
        dataset_id = db.create_dataset("profiles", str(dataset_root), has_gt=True)
        self.assertEqual(scan_triplet_dataset(db, dataset_id), 3)
        return workspace, db, model_id, dataset_id

    def _run_profile(self, db, workspace, model_id: int, dataset_id: int, profile: str):
        job_id = db.create_job(
            "inference",
            {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "height": 12,
                "width": 16,
                "batch_size": 2,
                "device": "cpu",
                "precision": "fp32",
                "metrics": [],
                "artifact_profile": profile,
                "benchmark_warmup_batches": 1,
                "benchmark_samples": 2,
                "max_save_inflight": 1,
                "prefetch_workers": 1,
                "save_workers": 1,
            },
        )
        run_worker(db, workspace, WorkerOptions(role="inference", once=True, worker_id=f"profile-{profile}"))
        return db.get_job(job_id)

    def test_profiles_bound_the_save_queue_and_control_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db, model_id, dataset_id = self._workspace(Path(tmp))
            evaluation = self._run_profile(db, workspace, model_id, dataset_id, "evaluation")
            self.assertEqual(evaluation["status"], "completed", evaluation)
            evaluation_kinds = {row["kind"] for row in db.list_artifacts(job_id=evaluation["id"])}
            self.assertTrue({"pred", "gt", "difference"}.issubset(evaluation_kinds))
            self.assertTrue(evaluation_kinds.isdisjoint({"flowt_0", "flowt_1", "mask0", "mask1", "warp0", "warp1", "blend"}))
            self.assertLessEqual(evaluation["result"]["performance"]["save_max_inflight"], 1)
            self.assertLess(
                evaluation["result"]["performance"]["artifact_db_batches"],
                evaluation["result"]["samples"],
            )

            diagnostic = self._run_profile(db, workspace, model_id, dataset_id, "diagnostic")
            self.assertEqual(diagnostic["status"], "completed", diagnostic)
            diagnostic_kinds = {row["kind"] for row in db.list_artifacts(job_id=diagnostic["id"])}
            self.assertTrue(
                {"pred", "gt", "difference", "flowt_0", "flowt_1", "mask0", "mask1", "warp0", "warp1", "blend"}.issubset(diagnostic_kinds)
            )

            benchmark = self._run_profile(db, workspace, model_id, dataset_id, "benchmark")
            self.assertEqual(benchmark["status"], "completed", benchmark)
            self.assertEqual(db.list_artifacts(job_id=benchmark["id"]), [])
            self.assertEqual(benchmark["result"]["samples"], 2)
            self.assertIsNone(benchmark["result"]["output_health"])
            self.assertGreater(benchmark["result"]["performance"]["steady_state_fps"], 0)


if __name__ == "__main__":
    unittest.main()
