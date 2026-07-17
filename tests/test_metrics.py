from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import textwrap
import unittest
import zipfile
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.metrics.feature import ConvNextFeatureMetric, DinoPatchMetric, _resolve_metric_device
from vfieval.metrics import METRIC_NAMES, create_metric
from vfieval.metrics.base import MetricBatchOutOfMemory, MetricResult, MetricUnavailable
from vfieval.metrics.cgvqm import (
    CgvqmMetric,
    CgvqmMetricFailed,
    CgvqmMetricUnavailable,
    _bounded_aligned_size,
    _parse_driver_output,
    _validate_eval_frame_counts,
)
from vfieval.metrics.health import metric_cache_config, metric_health, prepare_metric_asset_manifest
from vfieval.pipeline.metrics_runner import (
    METRIC_CACHE_VERSION,
    _evaluate_with_cache,
    _published_video_metric_inputs,
    _resolve_frame_reference,
    metric_cache_key,
    run_metric_job,
)
from vfieval.pipeline.metric_jobs import create_metric_wave, maybe_complete_metric_wave
from vfieval.pipeline.artifact_integrity import ArtifactIntegrityError


class MetricTests(unittest.TestCase):
    def test_legacy_invalid_paired_gt_uses_only_exact_source_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_path = root / "pred.png"
            source_gt = root / "source-gt.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(pred_path)
            Image.new("RGB", (8, 8), (4, 5, 6)).save(source_gt)
            pred = {
                "id": 1,
                "job_id": 7,
                "sample_id": 11,
                "path": str(pred_path),
                "metadata": {},
            }
            paired = [{"id": 2, "path": str(root / "missing.png"), "metadata": {}}]

            status, reference, details = _resolve_frame_reference(
                pred,
                {"id": 11, "gt_path": str(source_gt)},
                paired,
            )

            self.assertIsNone(status)
            self.assertEqual(reference, source_gt)
            self.assertTrue(details["legacy_fallback"])
            self.assertIn("paired GT artifact is unusable", details["paired_gt_rejected_reason"])

            canonical_pred = {**pred, "metadata": {"artifact_contract": "canonical-v1"}}
            status, reference, details = _resolve_frame_reference(
                canonical_pred,
                {"id": 11, "gt_path": str(source_gt)},
                paired,
            )
            self.assertEqual(status, "unavailable")
            self.assertIsNone(reference)
            self.assertEqual(details["reference_source"], "paired_gt_artifact")

            status, reference, details = _resolve_frame_reference(
                pred,
                {"id": 11, "gt_path": None},
                [{"id": 3, "path": str(source_gt), "metadata": {}}],
            )
            self.assertEqual(status, "skipped")
            self.assertIsNone(reference)
            self.assertEqual(details["reason"], "sample has no ground-truth reference")
            self.assertEqual(details["ignored_unexpected_gt_artifact_ids"], [3])

    def test_no_gt_video_ignores_stray_paired_gt_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "vfieval.sqlite")
            db.init()
            model_id = db.register_model("stray-video-gt", "dummy", None, 4, 4, {})
            dataset_id = db.create_dataset("stray-video-gt", str(root), False)
            frame = root / "frame.png"
            Image.new("RGB", (4, 4), (1, 2, 3)).save(frame)
            sample_id = db.add_sample(
                dataset_id,
                "clip-0001",
                str(frame),
                str(frame),
                None,
                {"source_type": "video", "video_name": "clip"},
            )
            job_id = db.create_job("inference", {"dataset_id": dataset_id})
            pred_path = root / "pred.mp4"
            gt_path = root / "stray-gt.mp4"
            pred_path.write_bytes(b"pred")
            gt_path.write_bytes(b"gt")
            pred_id = db.add_artifact(
                job_id,
                None,
                "pred_video",
                str(pred_path),
                "video/mp4",
                {"video_name": "clip", "artifact_contract": "canonical-v1"},
            )
            gt_id = db.add_artifact(
                job_id,
                None,
                "gt_video",
                str(gt_path),
                "video/mp4",
                {"video_name": "clip", "artifact_contract": "canonical-v1"},
            )
            pred = next(row for row in db.list_artifacts(job_id=job_id) if int(row["id"]) == pred_id)

            inputs = _published_video_metric_inputs(
                db,
                db.get_dataset(dataset_id),
                {sample_id: db.get_sample(sample_id)},
                [job_id],
                [pred],
            )

            self.assertEqual(len(inputs), 1)
            self.assertIsNone(inputs[0]["reference_path"])
            self.assertNotIn("error", inputs[0])
            self.assertEqual(
                inputs[0]["input_details"]["ignored_unexpected_gt_artifact_ids"],
                [gt_id],
            )

    def test_published_video_metric_rejects_nonidentical_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "vfieval.sqlite")
            db.init()
            model_id = db.register_model("strict-video", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("strict-video", str(root), True)
            frame = root / "frame.png"
            Image.new("RGB", (8, 8)).save(frame)
            sample_id = db.add_sample(
                dataset_id,
                "clip-0001",
                str(frame),
                str(frame),
                str(frame),
                {"source_type": "video", "video_name": "clip"},
            )
            job_id = db.create_job("inference", {"dataset_id": dataset_id})
            pred_path = root / "pred.mp4"
            gt_path = root / "gt.mp4"
            pred_path.write_bytes(b"pred")
            gt_path.write_bytes(b"gt")
            pred_id = db.add_artifact(
                job_id,
                None,
                "pred_video",
                str(pred_path),
                "video/mp4",
                {"video_name": "clip", "frames": 2, "width": 8, "height": 8, "fps": 24.0},
            )
            db.add_artifact(
                job_id,
                None,
                "gt_video",
                str(gt_path),
                "video/mp4",
                {"video_name": "clip", "frames": 1, "width": 8, "height": 8, "fps": 24.0},
            )
            pred = next(row for row in db.list_artifacts(job_id=job_id) if int(row["id"]) == pred_id)

            def inspect(path, exact=True):
                return {
                    "decodable": True,
                    "frame_count": 2 if Path(path) == pred_path else 1,
                    "width": 8,
                    "height": 8,
                    "fps": 24.0,
                }

            with patch("vfieval.file_inputs.inspect_video", side_effect=inspect):
                inputs = _published_video_metric_inputs(
                    db,
                    db.get_dataset(dataset_id),
                    {sample_id: db.get_sample(sample_id)},
                    [job_id],
                    [pred],
                )

            self.assertEqual(inputs[0]["input_status"], "unavailable")
            self.assertIn("strict temporal and spatial identity", inputs[0]["error"])
            mismatch = inputs[0]["input_details"]["video_pair_integrity"]["mismatches"]
            self.assertEqual(mismatch["observed_frame_count"], {"pred": 2, "gt": 1})

    def test_registry_excludes_psnr(self) -> None:
        self.assertNotIn("psnr", METRIC_NAMES)
        with self.assertRaisesRegex(ValueError, "unsupported metric"):
            create_metric("psnr")

    def test_vmaf_rejects_image_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reference = tmp_path / "ref.png"
            distorted = tmp_path / "dist.png"
            Image.new("RGB", (4, 4), (0, 0, 0)).save(reference)
            Image.new("RGB", (4, 4), (1, 1, 1)).save(distorted)
            metric = create_metric("vmaf")

            with self.assertRaises(MetricUnavailable):
                metric.evaluate(reference, distorted, tmp_path / "work")

    def test_lpips_metric_device_uses_shared_npu_resolver(self) -> None:
        import torch

        original_device = torch.device
        with patch("vfieval.devices.set_npu_device") as set_npu_device, patch(
            "vfieval.devices.torch.device",
            side_effect=lambda _name: original_device("cpu"),
        ):
            resolved = _resolve_metric_device("npu:1")

        self.assertEqual(resolved, torch.device("cpu"))
        set_npu_device.assert_called_once_with("npu:1")

    def test_lpips_metric_device_failure_keeps_device_in_unavailable_reason(self) -> None:
        with patch(
            "vfieval.metrics.feature.resolve_torch_device",
            side_effect=RuntimeError("torch_npu failed to bind device"),
        ):
            with self.assertRaisesRegex(MetricUnavailable, r"npu:1.*torch_npu failed"):
                _resolve_metric_device("npu:1")

    def test_dino_batch_keeps_one_model_and_one_warmup_per_shape(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = []
            for index in range(2):
                reference = root / f"ref-{index}.png"
                distorted = root / f"dist-{index}.png"
                Image.new("RGB", (8, 8), (index, 10, 20)).save(reference)
                Image.new("RGB", (8, 8), (index + 1, 11, 21)).save(distorted)
                pairs.append((reference, distorted, root / "work"))

            class FakeDino:
                def __init__(self):
                    self.calls = 0

                def forward_features(self, tensor):
                    self.calls += 1
                    return {"x_norm_patchtokens": tensor.flatten(2).transpose(1, 2)}

            model = FakeDino()
            health = {"available": True, "status": "available", "input_size": 14, "backbone": "fake"}
            with patch("vfieval.metrics.feature.metric_health", return_value=health), patch(
                "vfieval.metrics.feature._resolve_metric_device", return_value=torch.device("cpu")
            ), patch("vfieval.metrics.feature._load_dino_model", return_value=model) as load:
                metric = DinoPatchMetric(WorkspaceConfig.from_root(root / ".vfieval"), "cpu")
                first = metric.evaluate_batch(pairs)
                second = metric.evaluate_batch(pairs)

            self.assertEqual(len(first), 2)
            self.assertEqual(len(second), 2)
            load.assert_called_once()
            self.assertEqual(model.calls, 3, "one warmup plus one combined forward per batch")
            self.assertEqual(metric.performance()["warmed_shape_count"], 1)

    def test_convnext_batch_uses_one_combined_forward(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "ref.png"
            distorted = root / "dist.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(reference)
            Image.new("RGB", (8, 8), (4, 5, 6)).save(distorted)

            class FakeConvNext:
                def __init__(self):
                    self.calls = 0

                def __call__(self, tensor):
                    self.calls += 1
                    return [tensor, torch.nn.functional.avg_pool2d(tensor, 2)]

            model = FakeConvNext()
            health = {"available": True, "status": "available", "input_size": 32, "backbone": "fake"}
            with patch("vfieval.metrics.feature.metric_health", return_value=health), patch(
                "vfieval.metrics.feature._resolve_metric_device", return_value=torch.device("cpu")
            ), patch("vfieval.metrics.feature._load_convnext_model", return_value=model) as load:
                metric = ConvNextFeatureMetric(WorkspaceConfig.from_root(root / ".vfieval"), "cpu")
                results = metric.evaluate_batch([(reference, distorted, root / "work")] * 3)

            self.assertEqual(len(results), 3)
            load.assert_called_once()
            self.assertEqual(model.calls, 2, "one warmup plus one combined forward")

    def test_metric_runner_halves_oom_batch_without_cpu_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("frames", str(root), True)
            run_id = db.create_run(
                name="metric-batch",
                model_id=model_id,
                dataset_id=dataset_id,
                height=8,
                width=8,
                batch_size=1,
                device="npu:0",
                precision="fp32",
                metrics=["lpips_convnext"],
                create_inference_job=False,
            )
            inference_job_id = db.add_run_job(
                run_id, "inference", {"run_id": run_id, "dataset_id": dataset_id}, device="npu:0"
            )
            for index in range(4):
                reference = root / f"gt-{index}.png"
                distorted = root / f"pred-{index}.png"
                Image.new("RGB", (8, 8), (index, 0, 0)).save(reference)
                Image.new("RGB", (8, 8), (index + 1, 0, 0)).save(distorted)
                sample_id = db.add_sample(dataset_id, f"sample-{index}", str(reference), str(reference), str(reference), {})
                db.add_artifact(inference_job_id, sample_id, "pred", str(distorted), "image/png", {})
            metric_job_id = db.add_run_job(
                run_id,
                "metric",
                {
                    "run_id": run_id,
                    "inference_job_id": inference_job_id,
                    "dataset_id": dataset_id,
                    "metric_names": ["lpips_convnext"],
                    "metric_device": "npu:0",
                    "metric_batch_size_per_device": 8,
                },
                device="npu:0",
            )
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertTrue(db.set_run_metric_job(run_id, metric_job_id))
            claimed = db.claim_next_job("metric-batch", ["metric"], device_filter="npu:0")
            self.assertEqual(int(claimed["id"]), metric_job_id)

            class FakeMetric:
                def __init__(self):
                    self.calls = []

                def evaluate_batch(self, pairs):
                    self.calls.append(len(pairs))
                    if len(pairs) > 2:
                        raise MetricBatchOutOfMemory("npu:0 out of memory")
                    return [MetricResult("completed", 0.25, {"device": "npu:0"}) for _ in pairs]

            metric = FakeMetric()
            with patch("vfieval.pipeline.metrics_runner.create_metric", return_value=metric) as create:
                result = run_metric_job(db, workspace, metric_job_id)
            self.assertTrue(db.complete_job(metric_job_id, result))

            create.assert_called_once_with("lpips_convnext", workspace, device="npu:0")
            self.assertEqual(metric.calls, [4, 2, 2])
            self.assertEqual(result["summary"]["lpips_convnext"]["completed"], 4)
            self.assertEqual(result["performance"]["lpips_convnext"]["effective_batch_size"], 2)

    def test_frame_metric_uses_same_job_materialized_gt_not_sample_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("paired", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("paired", str(root), True)
            source_gt = root / "source-gt.png"
            canonical_gt = root / "canonical-gt.png"
            pred = root / "pred.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(source_gt)
            Image.new("RGB", (8, 8), (20, 30, 40)).save(canonical_gt)
            Image.new("RGB", (8, 8), (50, 60, 70)).save(pred)
            sample_id = db.add_sample(
                dataset_id, "sample", str(source_gt), str(source_gt), str(source_gt), {}
            )
            run_id = db.create_run(
                "paired", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", ["lpips_convnext"],
                create_inference_job=False,
            )
            inference_job_id = db.add_run_job(
                run_id, "inference", {"run_id": run_id, "dataset_id": dataset_id}
            )
            pred_id = db.add_artifact(
                inference_job_id,
                sample_id,
                "pred",
                str(pred),
                "image/png",
                {"artifact_contract": "canonical-v1"},
            )
            gt_id = db.add_artifact(
                inference_job_id,
                sample_id,
                "gt",
                str(canonical_gt),
                "image/png",
                {"artifact_contract": "canonical-v1"},
            )
            metric_job_id = db.add_run_job(
                run_id,
                "metric",
                {
                    "run_id": run_id,
                    "inference_job_id": inference_job_id,
                    "dataset_id": dataset_id,
                    "metric_names": ["lpips_convnext"],
                    "metric_device": "cpu",
                },
            )
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertTrue(db.set_run_metric_job(run_id, metric_job_id))
            self.assertEqual(int(db.claim_next_job("paired-metric", ["metric"])["id"]), metric_job_id)

            seen: list[tuple[Path, Path]] = []

            class FakeMetric:
                def evaluate_batch(self, pairs):
                    seen.extend((Path(reference), Path(distorted)) for reference, distorted, _work in pairs)
                    return [MetricResult("completed", 0.5, {}) for _ in pairs]

            with patch("vfieval.pipeline.metrics_runner.create_metric", return_value=FakeMetric()):
                result = run_metric_job(db, workspace, metric_job_id)

            self.assertEqual(seen, [(canonical_gt, pred)])
            metric_result = db.list_metric_results(inference_job_id)[0]
            self.assertEqual(metric_result["details"]["reference_source"], "paired_gt_artifact")
            self.assertEqual(metric_result["details"]["reference_artifact_id"], gt_id)
            self.assertNotEqual(pred_id, gt_id)
            self.assertTrue(db.complete_job(metric_job_id, result))

    def test_canonical_frame_metric_never_falls_back_to_raw_sample_gt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("missing-pair", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("missing-pair", str(root), True)
            source_gt = root / "source-gt.png"
            pred = root / "pred.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(source_gt)
            Image.new("RGB", (8, 8), (4, 5, 6)).save(pred)
            sample_id = db.add_sample(
                dataset_id, "sample", str(source_gt), str(source_gt), str(source_gt), {}
            )
            run_id = db.create_run(
                "missing-pair", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", ["lpips_convnext"],
                create_inference_job=False,
            )
            inference_job_id = db.add_run_job(
                run_id, "inference", {"run_id": run_id, "dataset_id": dataset_id}
            )
            db.add_artifact(
                inference_job_id,
                sample_id,
                "pred",
                str(pred),
                "image/png",
                {"artifact_contract": "canonical-v1"},
            )
            metric_job_id = db.add_run_job(
                run_id,
                "metric",
                {
                    "run_id": run_id,
                    "inference_job_id": inference_job_id,
                    "dataset_id": dataset_id,
                    "metric_names": ["lpips_convnext"],
                    "metric_device": "cpu",
                },
            )
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertTrue(db.set_run_metric_job(run_id, metric_job_id))
            self.assertEqual(int(db.claim_next_job("missing-pair", ["metric"])["id"]), metric_job_id)

            with patch("vfieval.pipeline.metrics_runner.create_metric") as create:
                result = run_metric_job(db, workspace, metric_job_id)

            create.assert_not_called()
            self.assertEqual(result["summary"]["lpips_convnext"]["unavailable"], 1)
            metric_result = db.list_metric_results(inference_job_id)[0]
            self.assertIn("no unique materialized GT", metric_result["details"]["reason"])
            self.assertTrue(db.complete_job(metric_job_id, result))

    def test_canonical_video_metric_gt_coverage_is_scoped_to_video_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("video-pairs", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("video-pairs", str(root), True)
            source_frame = root / "source.png"
            source_gt = root / "source-gt.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(source_frame)
            Image.new("RGB", (8, 8), (4, 5, 6)).save(source_gt)
            db.add_sample(
                dataset_id,
                "with-gt",
                str(source_frame),
                str(source_frame),
                str(source_gt),
                {"source_type": "video", "video_name": "with-gt", "frame_index": 0},
            )
            db.add_sample(
                dataset_id,
                "without-gt",
                str(source_frame),
                str(source_frame),
                None,
                {"source_type": "video", "video_name": "without-gt", "frame_index": 0},
            )
            run_id = db.create_run(
                "video-pairs",
                model_id,
                dataset_id,
                8,
                8,
                1,
                "cpu",
                "fp32",
                ["vmaf"],
                create_inference_job=False,
            )
            inference_job_id = db.add_run_job(
                run_id, "inference", {"run_id": run_id, "dataset_id": dataset_id}
            )
            for video_name in ("with-gt", "without-gt"):
                pred_video = root / f"{video_name}-pred.mp4"
                pred_video.write_bytes(b"canonical-pred-video")
                db.add_artifact(
                    inference_job_id,
                    None,
                    "pred_video",
                    str(pred_video),
                    "video/mp4",
                    {
                        "artifact_contract": "canonical-v1",
                        "video_name": video_name,
                        "frames": 1,
                        "width": 8,
                        "height": 8,
                        "fps": 24.0,
                    },
                )
            metric_job_id = db.add_run_job(
                run_id,
                "metric",
                {
                    "run_id": run_id,
                    "inference_job_id": inference_job_id,
                    "dataset_id": dataset_id,
                    "metric_names": ["vmaf"],
                    "metric_device": "cpu",
                },
            )
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertTrue(db.set_run_metric_job(run_id, metric_job_id))
            self.assertEqual(
                int(db.claim_next_job("video-pairs-metric", ["metric"])["id"]),
                metric_job_id,
            )

            with patch("vfieval.pipeline.metrics_runner.sync_run_assets"), patch(
                "vfieval.pipeline.metrics_runner.create_metric"
            ) as create:
                result = run_metric_job(db, workspace, metric_job_id)

            create.assert_not_called()
            self.assertEqual(result["summary"]["vmaf"]["unavailable"], 1)
            self.assertEqual(result["summary"]["vmaf"]["skipped"], 1)
            results = {
                str(row["details"].get("video_name")): row
                for row in db.list_metric_results(inference_job_id)
            }
            self.assertEqual(results["with-gt"]["status"], "unavailable")
            self.assertIn("no canonical gt_video", results["with-gt"]["details"]["reason"])
            self.assertEqual(results["without-gt"]["status"], "skipped")
            self.assertEqual(
                results["without-gt"]["details"]["reason"],
                "video has no ground-truth reference",
            )
            self.assertTrue(db.complete_job(metric_job_id, result))

    def test_multi_device_metric_wave_partitions_frames_and_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("frames", str(Path(tmp)), True)
            sample_ids = []
            for index in range(2):
                source = Path(tmp) / f"sample-{index}.png"
                source.write_bytes(b"source")
                sample_ids.append(
                    db.add_sample(
                        dataset_id,
                        f"sample-{index}",
                        str(source),
                        str(source),
                        str(source),
                        {"source_type": "frames"},
                    )
                )
            run_id = db.create_run(
                name="metric-wave",
                model_id=model_id,
                dataset_id=dataset_id,
                height=8,
                width=8,
                batch_size=1,
                device="multi_npu",
                precision="fp32",
                metrics=["lpips_convnext", "vmaf"],
                metadata={
                    "devices": ["npu:0", "npu:1"],
                    "request": {"metric_batch_size_per_device": 12},
                },
                create_inference_job=False,
            )
            inference_job_ids = []
            for index, device in enumerate(("npu:0", "npu:1")):
                inference_job_ids.append(db.add_run_job(
                    run_id,
                    "inference",
                    {
                        "run_id": run_id,
                        "dataset_id": dataset_id,
                        "sample_ids": [sample_ids[index]],
                        "artifact_profile": "evaluation",
                    },
                    shard_index=index,
                    device=device,
                ))
            self.assertTrue(db.mark_run_started(run_id, "running"))
            for index, (job_id, sample_id) in enumerate(zip(inference_job_ids, sample_ids)):
                claimed = db.claim_next_job(f"inference-{index}", ["inference"])
                self.assertEqual(int(claimed["id"]), job_id)
                for kind in ("pred", "gt", "difference"):
                    path = Path(tmp) / f"{sample_id}-{kind}.png"
                    path.write_bytes(kind.encode("ascii"))
                    db.add_artifact(job_id, sample_id, kind, str(path), "image/png", {})
                self.assertTrue(db.complete_job(job_id, {"samples": 1}))

            wave = create_metric_wave(db, run_id, ["lpips_convnext", "vmaf"], source="test")
            jobs = db.list_run_jobs(run_id, "metric")
            self.assertEqual([row["device"] for row in jobs], ["npu:0", "npu:1"])
            self.assertEqual(jobs[0]["payload"]["metric_names"], ["lpips_convnext", "vmaf"])
            self.assertEqual(jobs[1]["payload"]["metric_names"], ["lpips_convnext"])
            self.assertEqual(jobs[0]["payload"]["sample_ids"], [sample_ids[0]])
            self.assertEqual(jobs[1]["payload"]["sample_ids"], [sample_ids[1]])
            self.assertTrue(all(row["payload"]["metric_batch_size_per_device"] == 12 for row in jobs))

            self.assertTrue(db.mark_run_started(run_id, "metric_running"))
            for index, job in enumerate(jobs):
                claimed = db.claim_next_job(f"metric-{index}", ["metric"])
                self.assertEqual(int(claimed["id"]), int(job["job_id"]))
            first_result = {"summary": {"lpips_convnext": {"completed": 1, "mean": 0.2, "value_sum": 0.2}, "vmaf": {"completed": 1, "mean": 90.0, "value_sum": 90.0}}}
            self.assertFalse(
                maybe_complete_metric_wave(
                    db,
                    run_id,
                    wave["metric_wave_id"],
                    source_job_id=int(jobs[0]["job_id"]),
                    source_job_result=first_result,
                )
            )
            self.assertEqual(db.get_job(int(jobs[0]["job_id"]))["status"], "completed")
            second_result = {"summary": {"lpips_convnext": {"completed": 3, "mean": 0.4, "value_sum": 1.2}}}
            self.assertTrue(
                maybe_complete_metric_wave(
                    db,
                    run_id,
                    wave["metric_wave_id"],
                    source_job_id=int(jobs[1]["job_id"]),
                    source_job_result=second_result,
                )
            )
            self.assertEqual(db.get_job(int(jobs[1]["job_id"]))["status"], "completed")
            completed = db.get_run(run_id)
            self.assertEqual(completed["status"], "completed")
            self.assertAlmostEqual(completed["metric_summary"]["lpips_convnext"]["mean"], 0.35)
            self.assertEqual(completed["metric_summary"]["vmaf"]["completed"], 1)
            revision = completed["content_revision"]
            self.assertFalse(maybe_complete_metric_wave(db, run_id, wave["metric_wave_id"]))
            completed_again = db.get_run(run_id)
            self.assertEqual(completed_again["status"], "completed")
            self.assertEqual(completed_again["content_revision"], revision)

            retry = create_metric_wave(db, run_id, ["lpips_convnext"], source="retry", retry=True)
            retry_jobs = [
                row
                for row in db.list_run_jobs(run_id, "metric")
                if row["payload"].get("metric_wave_id") == retry["metric_wave_id"]
            ]
            self.assertTrue(db.mark_run_started(run_id, "metric_running"))
            for index, job in enumerate(retry_jobs):
                claimed = db.claim_next_job(f"retry-metric-{index}", ["metric"])
                self.assertEqual(int(claimed["id"]), int(job["job_id"]))
            self.assertTrue(db.complete_job(retry_jobs[0]["job_id"], {"summary": {"lpips_convnext": {"completed": 1, "mean": 0.1, "value_sum": 0.1}}}))
            self.assertTrue(db.complete_job(retry_jobs[1]["job_id"], {"summary": {"lpips_convnext": {"completed": 1, "mean": 0.3, "value_sum": 0.3}}}))
            self.assertTrue(maybe_complete_metric_wave(db, run_id, retry["metric_wave_id"]))
            retried = db.get_run(run_id)
            self.assertAlmostEqual(retried["metric_summary"]["lpips_convnext"]["mean"], 0.2)
            self.assertEqual(retried["metric_summary"]["vmaf"]["completed"], 1)

    def test_metric_wave_publication_rolls_back_as_one_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("frames", str(Path(tmp)), False)
            run_id = db.create_run(
                "atomic-wave", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", [], create_inference_job=False
            )
            self.assertTrue(db.mark_run_started(run_id, "running"))

            with self.assertRaises(TypeError):
                db.publish_metric_wave(
                    run_id,
                    [
                        {"payload": {"run_id": run_id, "metric_names": ["lpips_convnext"]}},
                        {
                            "payload": {"run_id": run_id, "metric_names": ["lpips_convnext"]},
                            "metadata": {"not_json": object()},
                        },
                    ],
                    retry=False,
                )

            self.assertEqual(db.list_run_jobs(run_id, "metric"), [])
            self.assertEqual(db.get_run(run_id)["status"], "running")

    def test_metric_retry_rejects_failed_partial_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("frames", str(Path(tmp)), True)
            source = Path(tmp) / "sample.png"
            source.write_bytes(b"source")
            sample_id = db.add_sample(
                dataset_id,
                "sample",
                str(source),
                str(source),
                str(source),
                {"source_type": "frames"},
            )
            run_id = db.create_run(
                "partial", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", ["lpips_convnext"], create_inference_job=False
            )
            job_id = db.add_run_job(
                run_id,
                "inference",
                {
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "sample_ids": [sample_id],
                    "artifact_profile": "evaluation",
                },
                progress_total=1,
            )
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertEqual(int(db.claim_next_job("partial", ["inference"])["id"]), job_id)
            pred = Path(tmp) / "pred.png"
            pred.write_bytes(b"pred")
            db.add_artifact(job_id, sample_id, "pred", str(pred), "image/png", {})
            error = {"message": "GT save failed", "type": "OSError"}
            self.assertTrue(db.fail_job(job_id, error))
            self.assertTrue(db.fail_run(run_id, error))

            with self.assertRaises(ArtifactIntegrityError):
                create_metric_wave(db, run_id, ["lpips_convnext"], source="retry", retry=True)

            self.assertEqual(db.get_run(run_id)["status"], "failed")
            self.assertEqual(db.list_run_jobs(run_id, "metric"), [])

    def test_vmaf_health_finds_project_local_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "vmaf")
            _write_vmaf_manifest(metric_dir, "")
            local_ffmpeg = metric_dir / "ffmpeg.exe"
            local_ffmpeg.write_bytes(b"project-local ffmpeg")

            with patch(
                "vfieval.metrics.health._inspect_ffmpeg_filters",
                return_value={"available": True, "reason": None},
            ):
                health = metric_health(workspace, "vmaf")

            self.assertEqual(health["status"], "available")
            self.assertEqual(health["executable_source"], "project_local")
            self.assertEqual(health["resolved_executable"], str(local_ffmpeg.resolve()))

    def test_cgvqm_python_driver_uses_current_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "cgvqm")
            _write_cgvqm_manifest(metric_dir, command=["python", "driver.py"])

            with patch("vfieval.metrics.health.importlib.util.find_spec", return_value=object()):
                health = metric_health(workspace, "cgvqm")

            expected_python = str(Path(sys.executable).resolve())
            self.assertEqual(health["status"], "available")
            self.assertEqual(health["resolved_executable"], expected_python)
            self.assertEqual(health["executable_source"], "current_python")
            self.assertEqual(health["driver_command"][0], expected_python)

    def test_cgvqm_eval_size_is_patch_aligned(self) -> None:
        width, height = _bounded_aligned_size(1792, 832, 720, 4)
        self.assertEqual((width, height), (720, 332))
        self.assertEqual(width % 4, 0)
        self.assertEqual(height % 4, 0)

        width, height = _bounded_aligned_size(96, 64, 720, 4)
        self.assertEqual((width, height), (96, 64))

    def test_cgvqm_protocol_parser_uses_last_valid_json_and_bounds_diagnostics(self) -> None:
        payload = _parse_driver_output(
            "warning before protocol\n"
            '{"status":"completed","value":0.25}\n'
            "third-party noise\n"
            '{"status":"completed","value":0.75,"details":{"source":"last"}}\n',
            "driver warning",
        )
        self.assertEqual(payload["value"], 0.75)
        self.assertEqual(payload["details"]["source"], "last")
        self.assertIn("warning before protocol", payload["details"]["driver_nonprotocol_stdout"])
        self.assertIn("third-party noise", payload["details"]["driver_nonprotocol_stdout"])
        self.assertEqual(payload["details"]["driver_stderr"], "driver warning")
        self.assertEqual(payload["details"]["driver_returncode"], 0)

    def test_cgvqm_protocol_parser_distinguishes_empty_bad_json_and_nonzero_exit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no JSON protocol output"):
            _parse_driver_output("", "", returncode=0)
        with self.assertRaisesRegex(RuntimeError, "no valid JSON protocol object"):
            _parse_driver_output("not-json", "decoder warning", returncode=0)
        with self.assertRaisesRegex(RuntimeError, "exited with code 7"):
            _parse_driver_output(
                '{"status":"failed","details":{"reason":"model crashed"}}',
                "trace tail",
                returncode=7,
            )

    def test_cgvqm_frame_count_mismatch_preserves_actual_counts(self) -> None:
        with self.assertRaises(CgvqmMetricUnavailable) as caught:
            _validate_eval_frame_counts(12, 11)
        self.assertEqual(caught.exception.details["eval_reference_frame_count"], 12)
        self.assertEqual(caught.exception.details["eval_distorted_frame_count"], 11)

    def test_cgvqm_timeout_preserves_bounded_driver_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            reference = root / "ref.mp4"
            distorted = root / "dist.mp4"
            reference.write_bytes(b"ref")
            distorted.write_bytes(b"dist")
            health = {
                "available": True,
                "status": "available",
                "reason": None,
                "driver_command": ["fake-cgvqm"],
                "video_eval_long_edge": 720,
                "manifest_path": str(root / "manifest.json"),
                "implementation_mode": "cgvqm_wrapper",
                "env": {},
            }
            timeout = subprocess.TimeoutExpired(
                ["fake-cgvqm"],
                600,
                output="driver started",
                stderr="device still busy",
            )
            with patch("vfieval.metrics.cgvqm.metric_health", return_value=health), patch(
                "vfieval.metrics.cgvqm._prepare_eval_videos",
                return_value=(
                    reference,
                    distorted,
                    {
                        "eval_reference_frame_count": 12,
                        "eval_distorted_frame_count": 12,
                    },
                ),
            ), patch("vfieval.metrics.cgvqm.subprocess.run", side_effect=timeout):
                with self.assertRaises(CgvqmMetricFailed) as caught:
                    CgvqmMetric(workspace, device="npu:0").evaluate(
                        reference, distorted, root / "work"
                    )
            self.assertTrue(caught.exception.details["driver_timed_out"])
            self.assertEqual(caught.exception.details["driver_stdout"], "driver started")
            self.assertEqual(caught.exception.details["driver_stderr"], "device still busy")
            self.assertEqual(caught.exception.details["eval_reference_frame_count"], 12)

    def test_metric_cache_key_uses_file_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            reference = tmp_path / "ref.png"
            distorted = tmp_path / "dist.png"
            Image.new("RGB", (4, 4), (0, 0, 0)).save(reference)
            Image.new("RGB", (4, 4), (1, 1, 1)).save(distorted)

            key1 = metric_cache_key("cgvqm", reference, distorted, {})
            key1_config = metric_cache_key("cgvqm", reference, distorted, {"adapter_version": "x"})
            Image.new("RGB", (5, 5), (1, 1, 1)).save(distorted)
            key2 = metric_cache_key("cgvqm", reference, distorted, {})

            self.assertNotEqual(key1, key2)
            self.assertNotEqual(key1, key1_config)

    def test_metric_retry_bypasses_cached_unavailable_and_replaces_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            reference = root / "ref.mp4"
            distorted = root / "dist.mp4"
            reference.write_bytes(b"reference")
            distorted.write_bytes(b"distorted")
            cache_config = {"adapter": "test"}
            cache_key = metric_cache_key(
                "cgvqm",
                reference,
                distorted,
                {
                    "cache_version": METRIC_CACHE_VERSION,
                    "metric": cache_config,
                    "metric_device": "npu:0",
                },
            )
            db.set_metric_cache(
                cache_key,
                "cgvqm",
                "unavailable",
                None,
                {"reason": "temporary device failure"},
            )

            class RecoveredMetric:
                calls = 0

                def evaluate(self, _reference, _distorted, _work_dir):
                    self.calls += 1
                    return MetricResult("completed", 0.33, {"recovered": True})

            metric = RecoveredMetric()
            with patch("vfieval.pipeline.metrics_runner.create_metric", return_value=metric):
                status, value, details = _evaluate_with_cache(
                    db,
                    workspace,
                    "cgvqm",
                    reference,
                    distorted,
                    None,
                    cache_config,
                    metric_device="npu:0",
                    retry=True,
                )
            self.assertEqual(metric.calls, 1)
            self.assertEqual((status, value), ("completed", 0.33))
            self.assertTrue(details["recovered"])
            self.assertEqual(db.get_metric_cache(cache_key)["status"], "completed")

    def test_metric_cache_config_changes_when_manifest_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "lpips_vit_patch")
            _write_fake_driver(metric_dir)
            _write_manifest(metric_dir, "lpips_vit_patch")

            config1 = metric_cache_config(workspace, "lpips_vit_patch")
            manifest_path = metric_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["asset_version"] = "v3"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            config2 = metric_cache_config(workspace, "lpips_vit_patch")

            self.assertNotEqual(config1["manifest_path"], config2["manifest_path"])

    def test_metric_cache_config_changes_when_feature_weights_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "lpips_convnext")
            weights_path = metric_dir / "weights.pth"
            weights_path.write_bytes(b"weights-v1")
            _write_feature_manifest(metric_dir, "lpips_convnext", weights_path.name)

            config1 = metric_cache_config(workspace, "lpips_convnext")
            weights_path.write_bytes(b"weights-v2")
            config2 = metric_cache_config(workspace, "lpips_convnext")

            self.assertNotEqual(config1["weights_path"], config2["weights_path"])

    def test_prepare_metric_asset_manifest_includes_vmaf_override_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()

            result = prepare_metric_asset_manifest(workspace, downloader=_fake_metric_downloader)
            vmaf_manifest = workspace.root.parent / "set" / "metrics" / "vmaf" / "manifest.json"
            data = json.loads(vmaf_manifest.read_text(encoding="utf-8"))

            self.assertIn(str(vmaf_manifest), result["prepared"])
            self.assertEqual(data["metric_name"], "vmaf")
            self.assertIn("ffmpeg_path", data)
            self.assertNotIn("status", data)

    def test_feature_metric_health_requires_local_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()

            vit = metric_health(workspace, "lpips_vit_patch")
            convnext = metric_health(workspace, "lpips_convnext")

            self.assertEqual(vit["status"], "missing_weights")
            self.assertEqual(vit["implementation_mode"], "dinov2_feature_distance")
            self.assertEqual(vit["backbone"], "dinov2_vits14_reg")
            self.assertEqual(convnext["status"], "missing_weights")
            self.assertEqual(convnext["implementation_mode"], "convnext_feature_distance")
            self.assertEqual(convnext["backbone"], "convnextv2_tiny.fcmae_ft_in22k_in1k")

    def test_prepare_metric_asset_manifest_writes_feature_metric_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()

            result = prepare_metric_asset_manifest(workspace, downloader=_fake_metric_downloader)
            vit_manifest = workspace.root.parent / "set" / "metrics" / "lpips_vit_patch" / "manifest.json"
            convnext_manifest = workspace.root.parent / "set" / "metrics" / "lpips_convnext" / "manifest.json"
            cgvqm_manifest = workspace.root.parent / "set" / "metrics" / "cgvqm" / "manifest.json"

            vit = json.loads(vit_manifest.read_text(encoding="utf-8"))
            convnext = json.loads(convnext_manifest.read_text(encoding="utf-8"))
            cgvqm = json.loads(cgvqm_manifest.read_text(encoding="utf-8"))
            self.assertFalse(result["errors"])
            self.assertEqual(vit["backbone"], "dinov2_vits14_reg")
            self.assertEqual(vit["device_policy"], "require_run_device")
            self.assertEqual(vit["input_size"], 518)
            self.assertEqual(vit["pad_multiple"], 14)
            self.assertTrue((vit_manifest.parent / "dinov2").is_dir())
            self.assertTrue((vit_manifest.parent / "dinov2_vits14_reg.pth").is_file())
            self.assertEqual(convnext["backbone"], "convnextv2_tiny.fcmae_ft_in22k_in1k")
            self.assertEqual(convnext["weights_path"], "model.safetensors")
            self.assertEqual(convnext["input_size"], 288)
            self.assertEqual(convnext["pad_multiple"], 32)
            self.assertTrue((convnext_manifest.parent / "model.safetensors").is_file())
            self.assertEqual(cgvqm["implementation_mode"], "cgvqm_wrapper")
            self.assertEqual(cgvqm["video_eval_long_edge"], 720)
            self.assertEqual(cgvqm["weights_path"], "run_cgvqm_vfieval.py")
            self.assertIn("driver", cgvqm)
            self.assertTrue((cgvqm_manifest.parent / "CGVQM").is_dir())
            self.assertTrue((cgvqm_manifest.parent / "run_cgvqm_vfieval.py").is_file())

    def test_generated_cgvqm_wrapper_calls_intellabs_run_cgvqm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            result = prepare_metric_asset_manifest(workspace, downloader=_fake_metric_downloader)
            self.assertFalse(result["errors"])
            metric_dir = workspace.root.parent / "set" / "metrics" / "cgvqm"
            wrapper = metric_dir / "run_cgvqm_vfieval.py"
            wrapper_source = wrapper.read_text(encoding="utf-8")
            checked_in_wrapper = ROOT / "set" / "metrics" / "cgvqm" / "run_cgvqm_vfieval.py"
            self.assertEqual(wrapper_source, checked_in_wrapper.read_text(encoding="utf-8"))
            self.assertIn("def _prepare_device", wrapper_source)
            self.assertIn("set_device(index)", wrapper_source)
            payload = {
                "metric_name": "cgvqm",
                "reference": str((Path(tmp) / "ref.mp4").resolve()),
                "distorted": str((Path(tmp) / "dist.mp4").resolve()),
                "repo_dir": str((metric_dir / "CGVQM").resolve()),
                "device": "cpu",
            }

            completed = subprocess.run(
                [sys.executable, str(wrapper)],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(output["status"], "completed")
            self.assertEqual(output["value"], 0.5)
            self.assertEqual(output["details"]["entrypoint"], "run_cgvqm")
            self.assertEqual(output["details"]["patch_pool"], "max")
            self.assertEqual(output["details"]["patch_scale"], 4)
            self.assertEqual(output["details"]["driver_stdout"], "WARNING: fake third-party stdout")
            self.assertFalse(output["details"]["driver_stdout_truncated"])

    def test_prepare_metrics_check_only_does_not_create_metric_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()

            from vfieval.metrics.health import metrics_health

            health = metrics_health(workspace)

            self.assertIn("lpips_vit_patch", health["metrics"])
            self.assertFalse((workspace.root.parent / "set").exists())

    def test_prepare_metric_download_failure_leaves_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()

            def failing_downloader(_url: str, _target: Path) -> None:
                raise RuntimeError("network blocked")

            result = prepare_metric_asset_manifest(workspace, downloader=failing_downloader)
            vit_manifest = workspace.root.parent / "set" / "metrics" / "lpips_vit_patch" / "manifest.json"

            self.assertFalse(vit_manifest.exists())
            self.assertTrue(result["errors"])
            self.assertEqual(result["health"]["metrics"]["lpips_vit_patch"]["status"], "missing_weights")

    def test_cgvqm_wrapper_metric_completes_from_driver_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "cgvqm")
            _write_fake_driver(metric_dir, value=0.42)
            _write_cgvqm_manifest(metric_dir)
            reference = tmp_path / "ref.mp4"
            distorted = tmp_path / "dist.mp4"
            _write_metric_video(reference, (0, 0, 0))
            _write_metric_video(distorted, (1, 1, 1))

            with patch("vfieval.metrics.health.importlib.util.find_spec", return_value=object()):
                metric = create_metric("cgvqm", workspace, device="cpu")
                result = metric.evaluate(reference, distorted, tmp_path / "work")

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.value, 0.42)
            self.assertEqual(result.details["manifest_path"], str((metric_dir / "manifest.json").resolve()))
            self.assertEqual(result.details["metric_name"], "cgvqm")
            self.assertEqual(result.details["patch_scale"], 4)

    def test_cgvqm_wrapper_metric_maps_unavailable_stdout_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "cgvqm")
            _write_fake_driver(metric_dir, mode="unavailable")
            _write_cgvqm_manifest(metric_dir)
            reference = tmp_path / "ref.mp4"
            distorted = tmp_path / "dist.mp4"
            _write_metric_video(reference, (0, 0, 0))
            _write_metric_video(distorted, (1, 1, 1))

            with patch("vfieval.metrics.health.importlib.util.find_spec", return_value=object()):
                metric = create_metric("cgvqm", workspace, device="cpu")
                with self.assertRaisesRegex(MetricUnavailable, "driver reported unavailable"):
                    metric.evaluate(reference, distorted, tmp_path / "work")

    def test_cgvqm_wrapper_metric_maps_failed_stdout_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "cgvqm")
            _write_fake_driver(metric_dir, mode="failed")
            _write_cgvqm_manifest(metric_dir)
            reference = tmp_path / "ref.mp4"
            distorted = tmp_path / "dist.mp4"
            _write_metric_video(reference, (0, 0, 0))
            _write_metric_video(distorted, (1, 1, 1))

            with patch("vfieval.metrics.health.importlib.util.find_spec", return_value=object()):
                metric = create_metric("cgvqm", workspace, device="cpu")
                with self.assertRaisesRegex(RuntimeError, "driver reported failed"):
                    metric.evaluate(reference, distorted, tmp_path / "work")

    def test_feature_and_cgvqm_health_maps_missing_assets_and_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()

            missing_manifest = metric_health(workspace, "lpips_vit_patch")
            self.assertEqual(missing_manifest["status"], "missing_weights")

            metric_dir = _metric_dir(workspace, "lpips_convnext")
            _write_feature_manifest(metric_dir, "lpips_convnext", "missing.bin")
            missing_weight = metric_health(workspace, "lpips_convnext")
            self.assertEqual(missing_weight["status"], "missing_weights")
            self.assertIn("missing.bin", missing_weight["reason"])

            cgvqm_dir = _metric_dir(workspace, "cgvqm")
            _write_cgvqm_manifest(cgvqm_dir, command=["missing-driver.exe"])
            with patch("vfieval.metrics.health.importlib.util.find_spec", return_value=object()):
                missing_driver = metric_health(workspace, "cgvqm")
            self.assertEqual(missing_driver["status"], "missing_evaluator")
            self.assertIn("driver executable", missing_driver["reason"])

    def test_vmaf_manifest_overrides_ffmpeg_path_for_health_and_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "vmaf")
            ffmpeg_cmd = _write_fake_ffmpeg(metric_dir, value=93.5)
            _write_vmaf_manifest(metric_dir, ffmpeg_cmd.name)
            reference = tmp_path / "ref.mp4"
            distorted = tmp_path / "dist.mp4"
            reference.write_bytes(b"ref")
            distorted.write_bytes(b"dist")

            health = metric_health(workspace, "vmaf")
            self.assertEqual(health["status"], "available")
            self.assertEqual(health["implementation_mode"], "ffmpeg_libvmaf")
            self.assertEqual(health["executable_source"], "manifest")
            self.assertEqual(health["resolved_executable"], str(ffmpeg_cmd.resolve()))

            metric = create_metric("vmaf", workspace)
            result = metric.evaluate(reference, distorted, tmp_path / "work")

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.value, 93.5)
            self.assertEqual(result.details["resolved_executable"], str(ffmpeg_cmd.resolve()))

    def test_unavailable_metric_cache_invalidates_when_metric_setup_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            reference = tmp_path / "ref.png"
            distorted = tmp_path / "dist.png"
            Image.new("RGB", (4, 4), (0, 0, 0)).save(reference)
            Image.new("RGB", (4, 4), (1, 1, 1)).save(distorted)

            calls: list[str] = []

            class FakeMetric:
                def evaluate(self, _reference: Path, _distorted: Path, _work_dir: Path):
                    calls.append("evaluate")
                    if len(calls) == 1:
                        raise MetricUnavailable("missing weights")
                    return type("Result", (), {"status": "completed", "value": 0.5, "details": {"source": "fake"}})()

            unavailable_config = {"status": "missing_weights", "asset_version": "v1"}
            available_config = {"status": "available", "asset_version": "v2"}

            with patch("vfieval.pipeline.metrics_runner.create_metric", return_value=FakeMetric()):
                first = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_vit_patch",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config=unavailable_config,
                )
                second = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_vit_patch",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config=available_config,
                )
                third = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_vit_patch",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config=available_config,
                )

            self.assertEqual(first[0], "unavailable")
            self.assertEqual(second[0], "completed")
            self.assertEqual(third[0], "completed")
            self.assertEqual(third[2]["cached"], True)
            self.assertEqual(calls, ["evaluate", "evaluate"])

    def test_evaluate_with_cache_records_metric_device_and_unavailable_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            reference, distorted = _write_image_pair(tmp_path)

            class DeviceFailMetric:
                def evaluate(self, _reference: Path, _distorted: Path, _work_dir: Path):
                    raise MetricUnavailable("metric device cuda:7 failed warmup: out of memory")

            with patch("vfieval.pipeline.metrics_runner.create_metric", return_value=DeviceFailMetric()) as factory:
                status, value, details = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_convnext",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config={"status": "available"},
                    metric_device="cuda:7",
                )

            self.assertEqual(status, "unavailable")
            self.assertIsNone(value)
            self.assertIn("cuda:7", details["reason"])
            factory.assert_called_once_with("lpips_convnext", workspace, device="cuda:7")

    def test_feature_metric_completed_result_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            reference, distorted = _write_image_pair(tmp_path)
            calls: list[str] = []

            class FakeMetric:
                def evaluate(self, _reference: Path, _distorted: Path, _work_dir: Path):
                    calls.append("evaluate")
                    return MetricResult("completed", 0.125, {"source": "fake"})

            with patch("vfieval.pipeline.metrics_runner.create_metric", return_value=FakeMetric()):
                first = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_vit_patch",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config={"status": "available", "backbone": "fake"},
                    metric_device="cpu",
                )
                second = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_vit_patch",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config={"status": "available", "backbone": "fake"},
                    metric_device="cpu",
                )

            self.assertEqual(first[0], "completed")
            self.assertEqual(first[1], 0.125)
            self.assertEqual(second[2]["cached"], True)
            self.assertEqual(calls, ["evaluate"])

    def test_metric_cache_isolated_by_alignment_plan_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            reference, distorted = _write_image_pair(tmp_path)
            calls: list[str] = []

            class FakeMetric:
                def evaluate(self, _reference: Path, _distorted: Path, _work_dir: Path):
                    calls.append("evaluate")
                    return MetricResult("completed", 0.125, {"source": "fake"})

            with patch("vfieval.pipeline.metrics_runner.create_metric", return_value=FakeMetric()):
                first = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_vit_patch",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config={"status": "available", "backbone": "fake"},
                    alignment_context={"plan_fingerprint": "alignment-a"},
                )
                cached = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_vit_patch",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config={"status": "available", "backbone": "fake"},
                    alignment_context={"plan_fingerprint": "alignment-a"},
                )
                changed_plan = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="lpips_vit_patch",
                    reference_path=reference,
                    distorted_path=distorted,
                    sample_id=1,
                    cache_config={"status": "available", "backbone": "fake"},
                    alignment_context={"plan_fingerprint": "alignment-b"},
                )

            self.assertEqual(first[0], "completed")
            self.assertTrue(cached[2]["cached"])
            self.assertEqual(changed_plan[0], "completed")
            self.assertEqual(calls, ["evaluate", "evaluate"])

    def test_item_compare_video_metrics_use_private_aligned_cache_for_each_track(self) -> None:
        """Item Compare video metrics must not depend on a reusable pred_video."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("compare-metric", "dummy", None, 4, 4, {})
            plan = {
                "fingerprint": "item-compare-alignment-v1",
                "target": {"width": 4, "height": 4, "source_slot": "pred_a"},
                "temporal": {"frame_count": 2, "fps": 24.0},
            }
            dataset_id = db.create_dataset(
                "item-compare-metric",
                str(tmp_path),
                has_gt=True,
                source_type="compare",
                decode_mode="compare",
                metadata={"alignment_plan": plan},
            )
            run_id = db.create_run(
                name="item-compare-metric",
                model_id=model_id,
                dataset_id=dataset_id,
                height=4,
                width=4,
                batch_size=1,
                device="cpu",
                precision="fp32",
                metrics=["vmaf", "cgvqm"],
                metadata={
                    "run_type": "video_compare",
                    "media_item_id": 101,
                    "publish_compare_pred_video": False,
                    "request": {"media_item_id": 101, "publish_compare_pred_video": False},
                },
                create_inference_job=False,
            )
            inference_job_id = db.add_run_job(
                run_id,
                "inference",
                {"run_id": run_id, "dataset_id": dataset_id},
                progress_total=4,
            )
            for track_index, (track_key, track_label, color) in enumerate(
                [("pred_a", "Method A", (32, 64, 96)), ("pred_b", "Method B", (128, 160, 192))]
            ):
                for frame_index in range(2):
                    gt_path = tmp_path / f"gt-{frame_index}.png"
                    pred_path = tmp_path / f"pred-{track_key}-{frame_index}.png"
                    if not gt_path.exists():
                        Image.new("RGB", (4, 4), (frame_index, 12, 24)).save(gt_path)
                    Image.new("RGB", (4, 4), color).save(pred_path)
                    sample_id = db.add_sample(
                        dataset_id,
                        f"clip__{track_key}__{frame_index:06d}",
                        str(gt_path),
                        str(pred_path),
                        str(gt_path),
                        {
                            "source_type": "compare",
                            "video_name": "clip",
                            "compare_group": "clip",
                            "compare_track_index": track_index,
                            "compare_track_key": track_key,
                            "compare_track_label": track_label,
                            "frame_index": frame_index,
                            "sample_index": frame_index,
                            "fps": 24.0,
                            "alignment_fingerprint": plan["fingerprint"],
                        },
                    )
                    db.add_artifact(inference_job_id, sample_id, "gt", str(gt_path), "image/png", {})
                    db.add_artifact(inference_job_id, sample_id, "pred", str(pred_path), "image/png", {})

            metric_job_id = db.add_run_job(
                run_id,
                "metric",
                {
                    "run_id": run_id,
                    "inference_job_id": inference_job_id,
                    "dataset_id": dataset_id,
                    "metric_names": ["vmaf", "cgvqm"],
                    "metric_device": "cpu",
                },
                progress_total=4,
            )
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertTrue(db.set_run_metric_job(run_id, metric_job_id))
            claimed = db.claim_next_job("item-compare-metric", ["metric"])
            self.assertEqual(int(claimed["id"]), metric_job_id)
            encoded: list[tuple[list[Path], Path, float]] = []
            evaluated: list[tuple[Path, Path]] = []

            def fake_write_mp4(frame_paths: list[Path], output: Path, fps: float) -> None:
                encoded.append((list(frame_paths), output, fps))
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"private metric input")

            class FakeVideoMetric:
                def evaluate(self, reference: Path, distorted: Path, _work_dir: Path):
                    evaluated.append((reference, distorted))
                    return MetricResult("completed", 91.25, {"implementation": "fake-video"})

            with patch("vfieval.pipeline.inference._write_mp4", side_effect=fake_write_mp4), patch(
                "vfieval.pipeline.metrics_runner.create_metric", return_value=FakeVideoMetric()
            ):
                result = run_metric_job(db, workspace, metric_job_id)
            self.assertTrue(db.complete_job(metric_job_id, result))

            self.assertEqual(result["summary"]["vmaf"]["completed"], 2)
            self.assertEqual(result["summary"]["cgvqm"]["completed"], 2)
            self.assertEqual(len(encoded), 3, "one shared GT plus one Pred cache video per track")
            self.assertEqual(len(evaluated), 4)
            self.assertTrue(
                all(path.parent == workspace.root / "compare_cache" and path.suffix == ".mp4" for pair in evaluated for path in pair)
            )
            self.assertEqual(db.list_artifacts(inference_job_id, kind="pred_video"), [])
            results = db.list_metric_results(inference_job_id)
            self.assertEqual({row["metric_name"] for row in results}, {"vmaf", "cgvqm"})
            self.assertEqual({row["details"]["compare_track_label"] for row in results}, {"Method A", "Method B"})
            self.assertTrue(all(row["details"]["video_input"] == "aligned_compare_cache" for row in results))
            self.assertTrue(
                all(row["details"]["alignment_fingerprint"] == plan["fingerprint"] for row in results)
            )
            cache_entries = [
                entry
                for entry in db.list_cache_entries()
                if entry.get("metadata", {}).get("purpose") == "item_compare_video_metric_input"
            ]
            self.assertEqual(len(cache_entries), 3)
            self.assertTrue(all(Path(entry["storage_path"]).is_file() for entry in cache_entries))
            self.assertTrue(
                all(entry["metadata"]["alignment_fingerprint"] == plan["fingerprint"] for entry in cache_entries)
            )

    def test_cli_smoke_metric_runs_manifest_driver_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            workspace = WorkspaceConfig.from_root(tmp_path / ".vfieval")
            workspace.ensure()
            metric_dir = _metric_dir(workspace, "cgvqm")
            _write_fake_driver(metric_dir, value=0.77)
            _write_cgvqm_manifest(metric_dir)
            reference = tmp_path / "ref.mp4"
            distorted = tmp_path / "dist.mp4"
            _write_metric_video(reference, (0, 0, 0))
            _write_metric_video(distorted, (1, 1, 1))
            (tmp_path / "av.py").write_text("", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(tmp_path) + os.pathsep + str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "vfieval.cli",
                    "--workspace",
                    str(workspace.root),
                    "smoke-metric",
                    "--metric",
                    "cgvqm",
                    "--reference",
                    str(reference),
                    "--distorted",
                    str(distorted),
                ],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["value"], 0.77)
            self.assertEqual(payload["details"]["metric_name"], "cgvqm")

    def test_vmaf_real_execution_is_conditional_on_local_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            health = metric_health(workspace, "vmaf")
            if not health["available"]:
                self.assertEqual(health["status"], "missing_evaluator")
                return

            video = next((ROOT / "videos" / "test_style").glob("*.*"))
            metric = create_metric("vmaf", workspace)
            result = metric.evaluate(video, video, workspace.tmp_dir / "real-vmaf")

            self.assertEqual(result.status, "completed")
            self.assertIsInstance(result.value, float)


def _metric_dir(workspace: WorkspaceConfig, metric_name: str) -> Path:
    metric_dir = workspace.root.parent / "set" / "metrics" / metric_name
    metric_dir.mkdir(parents=True, exist_ok=True)
    return metric_dir


def _write_image_pair(tmp_path: Path) -> tuple[Path, Path]:
    reference = tmp_path / "ref.png"
    distorted = tmp_path / "dist.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(reference)
    Image.new("RGB", (4, 4), (1, 1, 1)).save(distorted)
    return reference, distorted


def _write_metric_video(path: Path, color: tuple[int, int, int]) -> None:
    from vfieval.pipeline.inference import _write_mp4

    frame_dir = path.parent / f"{path.stem}-frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for index in range(3):
        frame = frame_dir / f"{index:06d}.png"
        Image.new("RGB", (8, 8), color).save(frame)
        frames.append(frame)
    _write_mp4(frames, path, 5.0)


def _write_manifest(
    metric_dir: Path,
    metric_name: str,
    *,
    input_mode: str = "sample_pair",
    command: list[str] | None = None,
    required_files: list[str] | None = None,
    env: dict[str, str] | None = None,
    create_required_files: bool = True,
) -> Path:
    manifest_path = metric_dir / "manifest.json"
    driver_command = command or [sys.executable, "driver.py"]
    required = required_files or ["weights.bin"]
    payload = {
        "metric_name": metric_name,
        "asset_version": "v2",
        "input_mode": input_mode,
        "driver": {"command": driver_command},
        "required_files": required,
        "env": env or {},
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if create_required_files:
        for rel_path in required:
            asset_path = metric_dir / rel_path
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            if not asset_path.exists():
                asset_path.write_bytes(b"asset")
    return manifest_path


def _write_feature_manifest(metric_dir: Path, metric_name: str, weights_path: str) -> Path:
    if metric_name == "lpips_vit_patch":
        implementation_mode = "dinov2_feature_distance"
        backbone = "dinov2_vits14_reg"
        input_size = 518
        repo_dir = "dinov2"
    else:
        implementation_mode = "convnext_feature_distance"
        backbone = "convnextv2_tiny.fcmae_ft_in22k_in1k"
        input_size = 288
        repo_dir = None
    payload = {
        "metric_name": metric_name,
        "asset_version": "v2",
        "implementation_mode": implementation_mode,
        "backbone": backbone,
        "weights_path": weights_path,
        "device_policy": "require_run_device",
        "input_size": input_size,
    }
    if repo_dir is not None:
        payload["repo_dir"] = repo_dir
    manifest_path = metric_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _write_cgvqm_manifest(metric_dir: Path, command: list[str] | None = None) -> Path:
    (metric_dir / "CGVQM").mkdir(parents=True, exist_ok=True)
    (metric_dir / "weights").mkdir(parents=True, exist_ok=True)
    manifest_path = metric_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "metric_name": "cgvqm",
                "asset_version": "v2",
                "implementation_mode": "cgvqm_wrapper",
                "repo_dir": "CGVQM",
                "weights_path": "weights",
                "device_policy": "require_run_device",
                "driver": {"command": command or [sys.executable, "driver.py"]},
                "env": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_vmaf_manifest(metric_dir: Path, ffmpeg_path: str) -> Path:
    manifest_path = metric_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "metric_name": "vmaf",
                "asset_version": "v2",
                "ffmpeg_path": ffmpeg_path,
                "notes": "test override",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_fake_driver(metric_dir: Path, *, mode: str = "completed", value: float = 0.25) -> Path:
    driver_path = metric_dir / "driver.py"
    driver_path.write_text(
        textwrap.dedent(
            f"""
            import json
            import sys

            payload = json.load(sys.stdin)
            result = {{
                "status": {mode!r},
                "value": {value!r} if {mode!r} == "completed" else None,
                "details": {{
                    "metric_name": payload["metric_name"],
                    "manifest_path": payload["manifest_path"],
                    "patch_scale": payload.get("patch_scale"),
                    "reason": "driver reported {mode}" if {mode!r} != "completed" else "",
                }},
            }}
            print(json.dumps(result))
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return driver_path


def _write_fake_ffmpeg(metric_dir: Path, *, value: float) -> Path:
    script_path = metric_dir / "fake_ffmpeg.py"
    script_path.write_text(
        textwrap.dedent(
            f"""
            import json
            import sys

            args = sys.argv[1:]
            if "-version" in args:
                print("ffmpeg version fake-vmaf")
                raise SystemExit(0)
            if "-filters" in args:
                print(" ... libvmaf ... ")
                raise SystemExit(0)
            if "-lavfi" in args:
                spec = args[args.index("-lavfi") + 1]
                log_path = spec.split("log_path=", 1)[1].strip("'").replace("\\:", ":")
                with open(log_path, "w", encoding="utf-8") as handle:
                    json.dump({{"pooled_metrics": {{"vmaf": {{"mean": {value!r}}}}}}}, handle)
                print("fake ffmpeg vmaf run")
                raise SystemExit(0)
            print("unexpected args: " + " ".join(args), file=sys.stderr)
            raise SystemExit(1)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    wrapper_path = metric_dir / "ffmpeg.cmd"
    wrapper_path.write_text(
        f'@echo off\r\n"{sys.executable}" "%~dp0fake_ffmpeg.py" %*\r\n',
        encoding="utf-8",
    )
    return wrapper_path


def _fake_metric_downloader(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if url.endswith(".zip"):
        top = "repo-main"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr(f"{top}/cgvqm.py", _fake_cgvqm_module())
            archive.writestr(f"{top}/hubconf.py", "def dinov2_vits14_reg(*args, **kwargs):\n    return None\n")
        return
    target.write_bytes(f"downloaded from {url}".encode("utf-8"))


def _fake_cgvqm_module() -> str:
    return textwrap.dedent(
        """
        class CGVQM_TYPE:
            CGVQM_2 = "cgvqm-2"

        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def item(self):
                return 0.5

        def run_cgvqm(test_vid_path, ref_vid_path, cgvqm_type=None, device='cpu', patch_pool='max', patch_scale=4):
            print('WARNING: fake third-party stdout')
            if patch_pool != 'max' or patch_scale != 4:
                raise RuntimeError('bad wrapper defaults')
            if cgvqm_type != CGVQM_TYPE.CGVQM_2:
                raise RuntimeError('bad cgvqm type')
            if not str(test_vid_path).endswith('dist.mp4') or not str(ref_vid_path).endswith('ref.mp4'):
                raise RuntimeError('wrong argument order')
            return FakeTensor(), 'emap'

        def compute_cgvqm(reference, distorted, device='cpu'):
            return 0.25
        """
    ).strip() + "\n"


if __name__ == "__main__":
    unittest.main()
