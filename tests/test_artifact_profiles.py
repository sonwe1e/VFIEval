from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
import torch

from vfieval.config import WorkspaceConfig
from vfieval.datasets import scan_triplet_dataset
from vfieval.db import Database
from vfieval.orchestration import _create_inference_shards, partition_samples_by_video
from vfieval.pipeline.inference import (
    DEFAULT_VISUALIZE_HEIGHT,
    DEFAULT_VISUALIZE_WIDTH,
    _AsyncSavePipeline,
    _NpuSmiSampler,
    _compose_canonical_chunks,
    _postprocess_chunk_size,
    _resolve_visualize_size,
)
from vfieval.performance import (
    EXECUTION_PROFILE_CONTRACT,
    execution_profile_identity,
    record_execution_profile,
    recommend_execution_profile,
)
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
            self.assertEqual(identity["execution_profile_contract"], EXECUTION_PROFILE_CONTRACT)
            record_execution_profile(db, identity, {"batch_size": 2}, {"steady_state_fps": 10})
            record_execution_profile(db, identity, {"batch_size": 1}, {"steady_state_fps": 5})
            recommended = recommend_execution_profile(db, workspace, payload)
            self.assertEqual(recommended["settings"]["batch_size"], 2)
            self.assertEqual(recommended["device_model"], "CPU")

    def test_npu_smi_usage_parser_is_optional_and_stable(self) -> None:
        parsed = _NpuSmiSampler._parse("AI Core Utilization Rate : 73\nMemory Usage : 41")
        self.assertEqual(parsed, {"aicore_percent": 73.0, "memory_percent": 41.0})

    def test_canonical_postprocess_chunks_by_pixels_and_halves_on_oom(self) -> None:
        self.assertEqual(_postprocess_chunk_size(64, 2160, 3840), 1)
        self.assertEqual(_postprocess_chunk_size(64, 1080, 1920), 4)
        img0 = torch.zeros((4, 3, 2, 2))
        img1 = torch.ones((4, 3, 2, 2))
        outputs = {
            "flowt_0": torch.zeros((4, 2, 2, 2)),
            "flowt_1": torch.zeros((4, 2, 2, 2)),
            "mask0": torch.zeros((4, 1, 2, 2)),
            "mask1": torch.zeros((4, 1, 2, 2)),
        }
        attempted: list[int] = []

        def compose(left, _right, _outputs):
            attempted.append(int(left.shape[0]))
            if int(left.shape[0]) > 1:
                raise RuntimeError("synthetic out of memory")
            return {"pred": left, "warp0": left, "warp1": left, "blend": left}

        with patch("vfieval.pipeline.inference.compose_interpolated", side_effect=compose):
            chunks = list(_compose_canonical_chunks(img0, img1, outputs, initial_chunk_size=4))
        self.assertEqual([(start, end) for start, end, _outputs, _composed in chunks], [(0, 1), (1, 2), (2, 3), (3, 4)])
        self.assertEqual(attempted[:3], [4, 2, 1])

        with patch(
            "vfieval.pipeline.inference.compose_interpolated",
            side_effect=RuntimeError("synthetic out of memory"),
        ):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                list(_compose_canonical_chunks(img0[:1], img1[:1], outputs, initial_chunk_size=1))

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

    def test_benchmark_shards_do_not_defer_nonexistent_video_artifacts(self) -> None:
        db = Mock()
        db.publish_inference_jobs.return_value = [11, 12]
        samples = [
            {"id": 1, "name": "a", "metadata": {"video_name": "a", "frame_index": 0}},
            {"id": 2, "name": "b", "metadata": {"video_name": "b", "frame_index": 0}},
        ]
        common = {
            "db": db,
            "run_id": 1,
            "model_id": 2,
            "dataset_id": 3,
            "height": 8,
            "width": 8,
            "precision": "fp32",
            "metrics": [],
            "devices": ["cuda:0", "cuda:1"],
            "batch_size_per_device": 1,
            "samples": samples,
        }
        _create_inference_shards(**common, artifact_profile="benchmark")
        benchmark_specs = db.publish_inference_jobs.call_args.args[1]
        self.assertTrue(benchmark_specs)
        self.assertTrue(all(not spec["payload"]["defer_video_finalize"] for spec in benchmark_specs))

        db.publish_inference_jobs.reset_mock()
        _create_inference_shards(**common, artifact_profile="evaluation")
        evaluation_specs = db.publish_inference_jobs.call_args.args[1]
        self.assertTrue(all(spec["payload"]["defer_video_finalize"] for spec in evaluation_specs))

    def test_preview_defaults_avoid_upscale_but_explicit_size_is_exact(self) -> None:
        self.assertEqual(
            _resolve_visualize_size({}, 1080, 1920),
            (DEFAULT_VISUALIZE_HEIGHT, DEFAULT_VISUALIZE_WIDTH),
        )
        self.assertEqual(_resolve_visualize_size({}, 360, 640), (360, 640))
        self.assertEqual(
            _resolve_visualize_size(
                {"visualize_height": 720, "visualize_width": 1280},
                360,
                640,
            ),
            (720, 1280),
        )

    def test_diagnostic_native_flow_and_mask_get_target_size_previews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db, _model_id, dataset_id = self._workspace(Path(tmp))
            sample = db.list_samples(dataset_id)[0]
            job_id = db.create_job("inference", {"artifact_profile": "diagnostic"})
            pipeline = _AsyncSavePipeline(
                db=db,
                job_id=job_id,
                run_id=None,
                is_shard=False,
                run_dir=workspace.runs_dir / "diagnostic-preview",
                save_workers=1,
                max_inflight=1,
                artifact_batch_size=16,
                preview_height=12,
                preview_width=16,
            )
            bundle = {
                "pred": torch.zeros((3, 12, 16)),
                "warp0": torch.zeros((3, 12, 16)),
                "warp1": torch.zeros((3, 12, 16)),
                "blend": torch.zeros((3, 12, 16)),
                "mask0": torch.zeros((1, 6, 8)),
                "mask1": torch.zeros((1, 6, 8)),
                "flowt_0": torch.zeros((2, 6, 8)),
                "flowt_1": torch.zeros((2, 6, 8)),
            }
            try:
                pipeline._save_sample(sample, bundle, {}, None)
                pipeline.shutdown()
            except Exception:
                pipeline.shutdown(suppress_errors=True)
                raise

            artifacts = {
                row["kind"]: row for row in db.list_artifacts(job_id=job_id)
            }
            pred = artifacts["pred"]
            self.assertTrue(pred["metadata"]["preview_uses_canonical"])
            self.assertNotIn("preview_path", pred["metadata"])
            for kind in ("flowt_0", "flowt_1", "mask0", "mask1"):
                artifact = artifacts[kind]
                metadata = artifact["metadata"]
                self.assertEqual(
                    (metadata["canonical_height"], metadata["canonical_width"]),
                    (6, 8),
                )
                self.assertFalse(metadata["preview_uses_canonical"])
                with Image.open(artifact["path"]) as canonical:
                    self.assertEqual(canonical.size, (8, 6))
                with Image.open(metadata["preview_path"]) as preview:
                    self.assertEqual(preview.size, (16, 12))

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
            self.assertTrue(db.mark_run_started(run_id, "running"))
            for index, job_id in enumerate(job_ids):
                claimed = db.claim_next_job(f"inference-{index}", ["inference"])
                self.assertEqual(int(claimed["id"]), job_id)
                self.assertTrue(db.complete_job(job_id, {"samples": 1, "performance": {"total_wall_seconds": 1}}))
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
            pred = db.list_artifacts(job_id=evaluation["id"], kind="pred")[0]
            self.assertEqual(pred["metadata"]["artifact_contract"], "canonical-v1")
            self.assertEqual(
                (pred["metadata"]["preview_height"], pred["metadata"]["preview_width"]),
                (12, 16),
            )
            self.assertTrue(pred["metadata"]["preview_uses_canonical"])
            self.assertNotIn("preview_path", pred["metadata"])

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
