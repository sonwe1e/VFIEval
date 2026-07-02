from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v13_test_utils import add_completed_pred_run, make_workspace, post_json, start_server, stop_server, write_mp4

from vfieval.config import WorkspaceConfig
from vfieval.datasets import scan_compare_dataset, scan_triplet_dataset
from vfieval.db import Database
from vfieval.devices import list_npu_devices, normalize_device_name, npu_unavailable_reason, set_npu_device, supported_precisions
from vfieval.file_inputs import decode_cache_dir, list_checkpoints, list_model_files, list_video_group_videos, list_video_groups, preflight_run
from vfieval.job_errors import describe_job_failure
from vfieval.metrics.health import metric_assets_dir, metric_health, metrics_health
from vfieval.models import load_flow_mask_model
from vfieval.pipeline.inference import run_inference_job
from vfieval.pipeline.postprocess import validate_model_outputs
from vfieval.server import _make_handler, _partition_samples_by_video, _resolve_execution_devices, _run_metric_summary, _run_timeline, _worker_process_command
from vfieval.worker import WorkerOptions, detect_capabilities, run_worker


class V3FileFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_project_root = os.environ.get("VFIEVAL_PROJECT_ROOT")
        os.environ["VFIEVAL_PROJECT_ROOT"] = str(ROOT)

    def tearDown(self) -> None:
        if self._old_project_root is None:
            os.environ.pop("VFIEVAL_PROJECT_ROOT", None)
        else:
            os.environ["VFIEVAL_PROJECT_ROOT"] = self._old_project_root

    def test_model_file_loader_accepts_dict_tuple_and_extra_outputs(self) -> None:
        import torch
        from vfieval.models.loader import normalize_infer_output

        img0 = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        img1 = torch.ones((1, 3, 8, 8), dtype=torch.float32)
        dict_model = load_flow_mask_model(f"file:{ROOT / 'models' / 'test_dict_return.py'}")
        dict_outputs = dict_model.predict(img0, img1, 0.5)
        validate_model_outputs(dict_outputs, img0)
        self.assertIn("debug_rgb", dict_outputs)

        tuple_model = load_flow_mask_model(f"file:{ROOT / 'models' / 'test_tuple_return.py'}")
        tuple_outputs = tuple_model.predict(img0, img1, 0.5)
        validate_model_outputs(tuple_outputs, img0)

        bad_model = load_flow_mask_model(f"file:{ROOT / 'models' / 'test_bad_shape.py'}")
        with self.assertRaisesRegex(ValueError, "flowt_0"):
            validate_model_outputs(bad_model.predict(img0, img1, 0.5), img0)
        with self.assertRaisesRegex(TypeError, "expected exactly 4 items"):
            normalize_infer_output((torch.zeros(1), torch.zeros(1)))

    def test_model_loader_moves_user_model_and_inner_module_to_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "portable_model.py"
            model_path.write_text(
                """
class Inner:
    def __init__(self):
        self.calls = []

    def to(self, device):
        self.calls.append(("to", str(device)))
        return self

    def eval(self):
        self.calls.append(("eval", None))
        return self


class Model:
    def __init__(self, checkpoint_path=None, device="cpu", metadata=None):
        self.calls = [("init", str(device))]
        self.net = Inner()

    def to(self, device):
        self.calls.append(("to", str(device)))
        return self

    def eval(self):
        self.calls.append(("eval", None))
        return self

    def infer(self, img0, img1):
        batch, _channels, height, width = img0.shape
        flow = img0.new_zeros((batch, 2, height, width))
        mask = img0.new_zeros((batch, 1, height, width))
        return flow, flow, mask, mask
""",
                encoding="utf-8",
            )

            adapter = load_flow_mask_model(f"file:{model_path}", device="npu:3")
            model = adapter._infer.__self__

        self.assertIn(("init", "npu:3"), model.calls)
        self.assertIn(("to", "npu:3"), model.calls)
        self.assertIn(("eval", None), model.calls)
        self.assertIn(("to", "npu:3"), model.net.calls)
        self.assertIn(("eval", None), model.net.calls)

    def test_model_loader_surfaces_constructor_type_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "bad_init_model.py"
            model_path.write_text(
                """
class Model:
    def __init__(self, checkpoint_path=None, device="cpu", metadata=None):
        raise TypeError("checkpoint tensor layout is invalid")

    def infer(self, img0, img1):
        raise AssertionError("not reached")
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TypeError, "checkpoint tensor layout is invalid"):
                load_flow_mask_model(f"file:{model_path}", device="cpu")

    def test_portable_checkpoint_helper_loads_on_cpu_then_moves_module(self) -> None:
        import torch

        from vfieval.models.utils import load_state_dict_portable

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pth"
            source = torch.nn.Linear(2, 1)
            target = torch.nn.Linear(2, 1)
            torch.save({"state_dict": source.state_dict()}, checkpoint_path)

            load_state_dict_portable(target, checkpoint_path, device="cpu")

        for name, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[name]))

    def test_discovery_and_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()

            self.assertIn("test_average.py", [row["name"] for row in list_model_files(workspace)])
            checkpoints = list_checkpoints(workspace, "test_checkpoint.py")
            self.assertIn("test_checkpoint/latest.pth", [row["relative_path"] for row in checkpoints])
            video_groups = list_video_groups(workspace)
            self.assertIn("test_style", [row["name"] for row in video_groups])
            test_style = next(row for row in video_groups if row["name"] == "test_style")
            first_video = test_style["videos"][0]["name"]
            self.assertIn("test_4k", [row["name"] for row in video_groups])
            test_4k = next(row for row in video_groups if row["name"] == "test_4k")
            self.assertEqual(test_4k["video_count"], 1)
            self.assertEqual(test_4k["videos"][0]["width"], 3840)
            self.assertEqual(test_4k["videos"][0]["height"], 2160)
            four_k_video = test_4k["videos"][0]["name"]
            self.assertIn(
                test_style["videos"][0]["frame_count_source"],
                {"container", "exact", "estimated", "ffprobe_nb_frames", "ffprobe_duration"},
            )
            detailed_videos = list_video_group_videos(workspace, "test_style")
            self.assertIn("valid_triplets", detailed_videos["videos"][0])
            self.assertIn("cache_status", detailed_videos["videos"][0])

            ok = preflight_run(
                db,
                workspace,
                {
                    "model_file": "test_average.py",
                    "video_group": "test_style",
                    "device": "cpu",
                    "precision": "fp32",
                    "max_frames": 4,
                },
            )
            self.assertTrue(ok["ok"], ok)
            self.assertGreater(ok["video_group"]["triplets"], 0)

            checkpoint_ok = preflight_run(
                db,
                workspace,
                {
                    "model_file": "test_checkpoint.py",
                    "checkpoint": "auto",
                    "video_group": "test_style",
                    "device": "cpu",
                    "precision": "fp32",
                    "max_frames": 4,
                },
            )
            self.assertTrue(checkpoint_ok["ok"], checkpoint_ok)
            self.assertTrue(checkpoint_ok["model"]["checkpoint"].endswith("latest.pth"))

            selected = preflight_run(
                db,
                workspace,
                {
                    "model_file": "test_average.py",
                    "video_group": "test_style",
                    "selected_videos": [first_video],
                    "device": "cpu",
                    "precision": "fp32",
                    "max_frames": 4,
                },
            )
            self.assertTrue(selected["ok"], selected)
            self.assertEqual(selected["video_group"]["video_count"], 1)
            self.assertEqual(selected["video_group"]["selected_videos"], [first_video])
            self.assertEqual(selected["video_group"]["videos"][0]["frame_count_source"], "exact")

            four_k = preflight_run(
                db,
                workspace,
                {
                    "model_file": "test_average.py",
                    "video_group": "test_4k",
                    "selected_videos": [four_k_video],
                    "device": "cpu",
                    "precision": "fp32",
                    "resolution_mode": "original",
                    "max_frames": 4,
                },
            )
            self.assertTrue(four_k["ok"], four_k)
            self.assertEqual(four_k["video_group"]["video_count"], 1)
            self.assertEqual(four_k["video_group"]["selected_videos"], [four_k_video])
            self.assertEqual(four_k["video_group"]["videos"][0]["width"], 3840)
            self.assertEqual(four_k["video_group"]["videos"][0]["height"], 2160)
            self.assertEqual(four_k["resolution"]["mode"], "original")
            self.assertEqual(four_k["resolution"]["width"], 3840)
            self.assertEqual(four_k["resolution"]["height"], 2160)

            empty_selection = preflight_run(
                db,
                workspace,
                {
                    "model_file": "test_average.py",
                    "video_group": "test_style",
                    "selected_videos": [],
                    "device": "cpu",
                    "precision": "fp32",
                },
            )
            self.assertFalse(empty_selection["ok"])
            self.assertIn("至少选择一个视频", empty_selection["errors"][0]["message"])

            bad_model = preflight_run(
                db,
                workspace,
                {
                    "model_file": "test_bad_shape.py",
                    "video_group": "test_style",
                    "device": "cpu",
                    "precision": "fp32",
                    "max_frames": 4,
                },
            )
            self.assertFalse(bad_model["ok"])
            self.assertIn("模型检查失败", bad_model["errors"][0]["title"])

            short_video = preflight_run(
                db,
                workspace,
                {
                    "model_file": "test_average.py",
                    "video_group": "test_edge_cases",
                    "device": "cpu",
                    "precision": "fp32",
                    "max_frames": 2,
                },
            )
            self.assertFalse(short_video["ok"])
            self.assertIn("帧数不足", short_video["errors"][0]["message"])

    def test_preflight_dry_run_executes_platform_postprocess(self) -> None:
        import torch

        class FakeModel:
            def predict(self, img0, img1, t):
                batch, _channels, height, width = img0.shape
                return {
                    "flowt_0": torch.zeros((batch, 2, height, width), dtype=img0.dtype, device=img0.device),
                    "flowt_1": torch.zeros((batch, 2, height, width), dtype=img0.dtype, device=img0.device),
                    "mask0": torch.zeros((batch, 1, height, width), dtype=img0.dtype, device=img0.device),
                    "mask1": torch.zeros((batch, 1, height, width), dtype=img0.dtype, device=img0.device),
                }

        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            observed: dict[str, object] = {}

            def fake_compose(img0, img1, outputs):
                observed["called"] = True
                observed["device"] = str(img0.device)
                observed["dtype"] = str(img0.dtype)
                observed["img_shape"] = tuple(img0.shape)
                observed["flow_device"] = str(outputs["flowt_0"].device)
                observed["flow_shape"] = tuple(outputs["flowt_0"].shape)
                observed["mask_dtype"] = str(outputs["mask0"].dtype)
                return {
                    "warp0": img0,
                    "warp1": img1,
                    "mask0": outputs["mask0"],
                    "mask1": outputs["mask1"],
                    "blend": img0,
                    "pred": img1,
                }

            with (
                patch("vfieval.file_inputs.load_flow_mask_model", return_value=FakeModel()),
                patch("vfieval.file_inputs.compose_interpolated", side_effect=fake_compose),
            ):
                result = preflight_run(
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

            self.assertTrue(result["ok"], result)
            self.assertTrue(observed.get("called"))
            self.assertEqual(observed["device"], "cpu")
            self.assertEqual(observed["flow_device"], "cpu")
            self.assertEqual(observed["dtype"], "torch.float32")
            self.assertEqual(observed["img_shape"], (1, 3, 128, 128))
            self.assertEqual(observed["flow_shape"], (1, 2, 128, 128))
            self.assertEqual(observed["mask_dtype"], "torch.float32")

    def test_api_file_run_generates_video_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                group = next(row for row in list_video_groups(workspace) if row["name"] == "test_style")
                selected_video = group["videos"][0]["name"]
                video_payload = _get(base_url, "/api/video-groups/test_style/videos")
                self.assertIn("valid_triplets", video_payload["videos"][0])
                self.assertIn("cache_status", video_payload["videos"][0])
                self.assertIn("thumbnail_url", video_payload["videos"][0])
                paged = _get(base_url, "/api/video-groups/test_style/videos?page=1&page_size=2&q=motion")
                self.assertLessEqual(len(paged["videos"]), 2)
                self.assertIn("all_video_names", paged)
                checkpoint_payload = _get(base_url, "/api/checkpoints?model_file=test_checkpoint.py")
                self.assertIn("test_checkpoint/latest.pth", [row["relative_path"] for row in checkpoint_payload])
                devices_payload = _get(base_url, "/api/devices")
                self.assertIn("cpu", devices_payload)
                preflight = _post(
                    base_url,
                    "/api/preflight",
                    {
                        "model_file": "test_average.py",
                        "video_group": "test_style",
                        "selected_videos": [selected_video],
                        "device": "cpu",
                        "precision": "fp32",
                        "max_frames": 4,
                    },
                )
                self.assertTrue(preflight["ok"])
                created = _post(
                    base_url,
                    "/api/runs",
                    {
                        "model_file": "test_average.py",
                        "video_group": "test_style",
                        "selected_videos": [selected_video],
                        "device": "cpu",
                        "precision": "fp32",
                        "batch_size": 2,
                        "max_frames": 4,
                    },
                )
                run_id = created["run_id"]
                run = _wait_for_run(base_url, run_id)
                self.assertEqual(run["status"], "completed", run)
                output_health = run["result"]["output_health"]
                self.assertEqual(output_health["samples"], 2)
                self.assertTrue(output_health["flow_flat"])
                self.assertTrue(output_health["mask_flat"])
                self.assertTrue(output_health["warnings"])
                self.assertIn("flowt_0", output_health["stats"])
                self.assertIn("mask0", output_health["stats"])
                self.assertTrue((Path(run["result"]["output_dir"]) / "logs" / "output_health.log").is_file())
                artifacts = _get(base_url, f"/api/runs/{run_id}/artifacts")
                kinds = [artifact["kind"] for artifact in artifacts]
                self.assertIn("pred_video", kinds)
                self.assertIn("gt_video", kinds)
                self.assertIn("diff_video", kinds)
                self.assertEqual(kinds.count("pred_video"), 1)
                pred_video = next(artifact for artifact in artifacts if artifact["kind"] == "pred_video")
                range_request = urllib.request.Request(
                    f"{base_url}/api/files/{pred_video['id']}",
                    headers={"Range": "bytes=0-15"},
                )
                with urllib.request.urlopen(range_request, timeout=30) as response:
                    partial = response.read()
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.headers["Accept-Ranges"], "bytes")
                    self.assertTrue(response.headers["Content-Range"].startswith("bytes 0-15/"))
                self.assertEqual(len(partial), 16)
                self.assertTrue(Path(run["metadata"]["output_dir"]).exists())
                self.assertEqual(run["metadata"]["selected_videos"], [selected_video])
                self.assertIn("reference_key", run["metadata"])
                self.assertEqual(run["metadata"]["reference_config"]["video_group"], "test_style")
                self.assertEqual(run["metadata"]["reference_config"]["selected_videos"], [selected_video])
                self.assertEqual(run["metadata"]["reference_config"]["height"], run["height"])
                self.assertEqual(run["metadata"]["reference_config"]["width"], run["width"])
                self.assertIn("jobs", _get(base_url, f"/api/runs/{run_id}"))

                metric_job_id = db.create_job("metric", {"test": True})
                samples = db.list_samples(int(run["dataset_id"]))
                sample = samples[0]
                second_sample = samples[1]
                manifest_path = decode_cache_dir(workspace, sample["metadata"]["cache_key"]) / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for key in ("video_name", "video_file", "width", "height", "duration_seconds", "valid_triplets", "decode_status"):
                    self.assertIn(key, manifest)
                db.add_metric_result(
                    job_id=metric_job_id,
                    inference_job_id=int(run["inference_job_id"]),
                    sample_id=int(sample["id"]),
                    metric_name="lpips_vit_patch",
                    status="completed",
                    value=0.25,
                    details={},
                )
                db.add_metric_result(
                    job_id=metric_job_id,
                    inference_job_id=int(run["inference_job_id"]),
                    sample_id=int(second_sample["id"]),
                    metric_name="lpips_vit_patch",
                    status="unavailable",
                    value=None,
                    details={"reason": "missing evaluator"},
                )
                db.add_metric_result(
                    job_id=metric_job_id,
                    inference_job_id=int(run["inference_job_id"]),
                    sample_id=int(second_sample["id"]),
                    metric_name="lpips_convnext",
                    status="skipped",
                    value=None,
                    details={"reason": "no ground truth"},
                )
                db.add_metric_result(
                    job_id=metric_job_id,
                    inference_job_id=int(run["inference_job_id"]),
                    sample_id=None,
                    metric_name="vmaf",
                    status="completed",
                    value=91.5,
                    details={"video_name": Path(selected_video).stem},
                )
                timeline = _get(base_url, f"/api/runs/{run_id}/timeline")
                self.assertEqual(len(timeline["videos"]), 1)
                self.assertEqual(len(timeline["videos"][0]["samples"]), 2)
                first_sample = timeline["videos"][0]["samples"][0]
                self.assertNotIn("artifacts", first_sample)
                self.assertNotIn("sample_files", first_sample)
                self.assertTrue(first_sample["has_artifacts"])
                self.assertIn("lpips_vit_patch", first_sample["metrics"])
                self.assertEqual(first_sample["metric_status"]["completed"], 1)
                sample_detail = _get(base_url, f"/api/runs/{run_id}/samples/{first_sample['sample_id']}")
                self.assertIn("pred", sample_detail["artifacts"])
                self.assertIn("preview_url", sample_detail["artifacts"]["pred"])
                self.assertIn("gt", sample_detail["artifacts"])
                self.assertIn("gt", sample_detail["sample_files"])
                self.assertEqual(sample_detail["frame_index"], first_sample["frame_index"])
                second_timeline_sample = timeline["videos"][0]["samples"][1]
                self.assertEqual(second_timeline_sample["metrics"]["lpips_vit_patch"]["status"], "unavailable")
                self.assertEqual(second_timeline_sample["metrics"]["lpips_convnext"]["status"], "skipped")
                self.assertNotIn("vmaf", first_sample["metrics"])
                self.assertIn("vmaf", timeline["videos"][0]["video_metrics"])
                self.assertEqual(timeline["videos"][0]["metric_summary"]["lpips_vit_patch"]["unavailable"], 1)
                self.assertEqual(timeline["videos"][0]["worst_samples"]["lpips_vit_patch"][0]["sample_id"], int(sample["id"]))

                summary = _get(base_url, f"/api/runs/{run_id}/metric-summary")
                self.assertEqual(summary["metrics"]["lpips_vit_patch"]["completed"], 1)
                self.assertEqual(summary["metrics"]["lpips_vit_patch"]["unavailable"], 1)
                self.assertEqual(summary["metrics"]["lpips_vit_patch"]["mean"], 0.25)
                self.assertEqual(summary["metrics"]["lpips_vit_patch"]["worst_sample_id"], int(sample["id"]))

                run_videos = _get(base_url, f"/api/runs/{run_id}/videos")
                self.assertEqual(run_videos["videos"][0]["sample_count"], 2)
                run_video_name = run_videos["videos"][0]["video_name"]
                video_timeline = _get(base_url, f"/api/runs/{run_id}/videos/{urllib.parse.quote(run_video_name)}/timeline?bucket_count=2&window_size=1")
                self.assertEqual(len(video_timeline["samples"]), 1)
                self.assertEqual(len(video_timeline["overview"]), 2)
                self.assertIn("vmaf", video_timeline["video_metrics"])
                self.assertEqual(video_timeline["video_metrics"]["vmaf"]["status"], "completed")
                compare = _get(
                    base_url,
                    f"/api/compare/samples?run_id={run_id}&video_name={urllib.parse.quote(run_video_name)}&frame_index={first_sample['frame_index']}",
                )
                self.assertTrue(compare["compatible"])
                self.assertIsNotNone(compare["samples"][0]["sample"])

                health = _get(base_url, "/api/metrics/health")
                self.assertIn("lpips_vit_patch", health["metrics"])
                self.assertIn("setup_summary", health["metrics"]["lpips_vit_patch"])
                self.assertEqual(health["metrics"]["lpips_vit_patch"]["implementation_mode"], "dinov2_feature_distance")
                self.assertEqual(health["metrics"]["lpips_vit_patch"]["backbone"], "dinov2_vits14_reg")
                self.assertTrue(health["metrics"]["lpips_vit_patch"]["manifest_path"].endswith("lpips_vit_patch\\manifest.json"))
                self.assertEqual(health["metrics"]["lpips_vit_patch"]["device_policy"], "require_run_device")
                self.assertEqual(health["metrics"]["lpips_vit_patch"]["eval_resolution"], {"mode": "max_edge", "value": 518})
                self.assertTrue(health["metrics"]["lpips_vit_patch"]["auto_download"])
                self.assertIn("input_mode", health["metrics"]["vmaf"])
                self.assertEqual(health["metrics"]["vmaf"]["implementation_mode"], "ffmpeg_libvmaf")
                self.assertIn("resolved_executable", health["metrics"]["vmaf"])
                worker = _post(
                    base_url,
                    "/api/workers/register",
                    {"worker_id": "remote-test", "role": "inference", "capabilities": {"cpu": True}},
                )
                self.assertEqual(worker["worker"]["id"], "remote-test")
                heartbeat = _post(base_url, f"/api/jobs/{int(run['inference_job_id'])}/heartbeat", {"worker_id": "remote-test"})
                self.assertEqual(heartbeat["status"], "heartbeat")

                cleanup = _post(base_url, f"/api/runs/{run_id}/cleanup-artifacts", {})
                self.assertTrue(cleanup["artifact_cleaned"])
                self.assertFalse(Path(cleanup["output_dir"]).exists())
                cleaned_run = _get(base_url, f"/api/runs/{run_id}")
                self.assertEqual(cleaned_run["artifact_summary"]["total"], 0)
                self.assertIsNotNone(cleaned_run["artifact_cleaned_at"])
                cleaned_videos = _get(base_url, f"/api/runs/{run_id}/videos")
                cleaned_video_name = cleaned_videos["videos"][0]["video_name"]
                cleaned_timeline = _get(base_url, f"/api/runs/{run_id}/videos/{urllib.parse.quote(cleaned_video_name)}/timeline")
                self.assertFalse(cleaned_timeline["samples"][0]["has_artifacts"])

                hidden = _delete(base_url, f"/api/runs/{run_id}")
                self.assertTrue(hidden["deleted"])
                visible_runs = _get(base_url, "/api/runs")
                self.assertNotIn(run_id, [int(item["id"]) for item in visible_runs])
                all_runs = _get(base_url, "/api/runs?include_deleted=1")
                self.assertIn(run_id, [int(item["id"]) for item in all_runs])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_server_marks_static_and_json_responses_uncacheable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urllib.request.urlopen(f"{base_url}/", timeout=30) as response:
                    self.assertEqual(response.headers.get("Cache-Control"), "no-store")
                    self.assertIn("text/html", response.headers.get("Content-Type", ""))
                with urllib.request.urlopen(f"{base_url}/app.js", timeout=30) as response:
                    self.assertEqual(response.headers.get("Cache-Control"), "no-store")
                    self.assertIn("javascript", response.headers.get("Content-Type", ""))
                with urllib.request.urlopen(f"{base_url}/api/devices", timeout=30) as response:
                    self.assertEqual(response.headers.get("Cache-Control"), "no-store")
                    self.assertIn("application/json", response.headers.get("Content-Type", ""))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_server_rejects_oversized_json_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                request = urllib.request.Request(
                    f"{base_url}/api/preflight",
                    data=b'{"payload":"too large"}',
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with (
                    patch("vfieval.server.MAX_REQUEST_BODY_BYTES", 8),
                    self.assertRaises(urllib.error.HTTPError) as raised,
                ):
                    urllib.request.urlopen(request, timeout=30)
                self.assertEqual(raised.exception.code, 413)
                error_payload = json.loads(raised.exception.read().decode("utf-8"))
                self.assertEqual(error_payload["error"]["type"], "RequestBodyTooLarge")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_server_hides_internal_exception_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                def explode(*_args, **_kwargs):
                    raise RuntimeError("secret internal path D:/Documents/VFIEval")

                db.list_runs = explode
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(f"{base_url}/api/runs", timeout=30)
                self.assertEqual(raised.exception.code, 500)
                error_payload = json.loads(raised.exception.read().decode("utf-8"))
                self.assertEqual(error_payload["error"]["type"], "InternalServerError")
                self.assertEqual(error_payload["error"]["message"], "internal server error")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_api_file_run_with_metric_records_unavailable_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous_metric_assets = os.environ.get("VFIEVAL_METRIC_ASSETS_DIR")
            os.environ["VFIEVAL_METRIC_ASSETS_DIR"] = str(Path(tmp) / "missing_metrics")
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                group = next(row for row in list_video_groups(workspace) if row["name"] == "test_style")
                selected_video = group["videos"][0]["name"]
                created = _post(
                    base_url,
                    "/api/runs",
                    {
                        "model_file": "test_average.py",
                        "video_group": "test_style",
                        "selected_videos": [selected_video],
                        "device": "cpu",
                        "precision": "fp32",
                        "batch_size": 2,
                        "max_frames": 4,
                        "metrics": ["lpips_vit_patch"],
                    },
                )
                run_id = created["run_id"]
                run = _wait_for_run(base_url, run_id)
                self.assertEqual(run["status"], "completed", run)
                self.assertIsNotNone(run["metric_job_id"])
                self.assertEqual(run["metadata"]["metric_health"]["lpips_vit_patch"]["status"], "missing_weights")
                self.assertTrue(run["metadata"]["metric_health"]["lpips_vit_patch"]["weights_path"].endswith("lpips_vit_patch\\dinov2_vits14_reg.pth"))
                self.assertEqual(run["metadata"]["metric_health"]["lpips_vit_patch"]["implementation_mode"], "dinov2_feature_distance")
                self.assertTrue(run["metadata"]["metric_health"]["lpips_vit_patch"]["manifest_path"].endswith("lpips_vit_patch\\manifest.json"))

                summary = _get(base_url, f"/api/runs/{run_id}/metric-summary")
                self.assertEqual(summary["metrics"]["lpips_vit_patch"]["unavailable"], 2)
                self.assertIn("lpips_vit_patch metric is missing_weights", summary["metrics"]["lpips_vit_patch"]["reasons"][0])

                timeline = _get(base_url, f"/api/runs/{run_id}/timeline")
                statuses = [
                    sample["metrics"]["lpips_vit_patch"]["status"]
                    for sample in timeline["videos"][0]["samples"]
                ]
                self.assertEqual(statuses, ["unavailable", "unavailable"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                if previous_metric_assets is None:
                    os.environ.pop("VFIEVAL_METRIC_ASSETS_DIR", None)
                else:
                    os.environ["VFIEVAL_METRIC_ASSETS_DIR"] = previous_metric_assets

    def test_metric_job_payload_inherits_resolved_inference_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "dataset"
            for folder in ("img0", "img1", "gt"):
                (dataset_root / folder).mkdir(parents=True)
            Image.new("RGB", (8, 8), (10, 20, 30)).save(dataset_root / "img0" / "sample000.png")
            Image.new("RGB", (8, 8), (30, 20, 10)).save(dataset_root / "img1" / "sample000.png")
            Image.new("RGB", (8, 8), (20, 10, 30)).save(dataset_root / "gt" / "sample000.png")

            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("metric-device", str(dataset_root), True)
            self.assertEqual(scan_triplet_dataset(db, dataset_id), 1)
            run_id = db.create_run(
                name="metric-device",
                model_id=model_id,
                dataset_id=dataset_id,
                height=8,
                width=8,
                batch_size=1,
                device="cpu",
                precision="fp32",
                metrics=["lpips_vit_patch"],
            )

            run_worker(db, workspace, WorkerOptions(role="inference", once=True, worker_id="metric-device-inference"))
            run = db.get_run(run_id)
            metric_job = db.get_job(int(run["metric_job_id"]))

            self.assertEqual(metric_job["payload"]["metric_device"], "cpu")

    def test_reopening_run_detail_endpoints_does_not_enqueue_new_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                group = next(row for row in list_video_groups(workspace) if row["name"] == "test_style")
                selected_video = group["videos"][0]["name"]
                created = _post(
                    base_url,
                    "/api/runs",
                    {
                        "model_file": "test_average.py",
                        "video_group": "test_style",
                        "selected_videos": [selected_video],
                        "device": "cpu",
                        "precision": "fp32",
                        "batch_size": 2,
                        "max_frames": 4,
                        "metrics": ["lpips_vit_patch"],
                    },
                )
                run_id = created["run_id"]
                run = _wait_for_run(base_url, run_id)
                self.assertEqual(run["status"], "completed", run)
                self.assertIsNotNone(run["metric_job_id"])

                initial_updated_at = run["updated_at"]
                initial_jobs = [
                    (job["id"], job["kind"], job["status"], json.dumps(job.get("payload") or {}, sort_keys=True))
                    for job in db.list_jobs(limit=50)
                ]
                initial_metrics = _get(base_url, f"/api/runs/{run_id}/metrics")
                videos = _get(base_url, f"/api/runs/{run_id}/videos")
                video_name = videos["videos"][0]["video_name"]
                video_timeline = _get(
                    base_url,
                    f"/api/runs/{run_id}/videos/{urllib.parse.quote(video_name)}/timeline?bucket_count=2&window_size=2",
                )
                sample_id = int(video_timeline["samples"][0]["sample_id"])

                for _ in range(2):
                    _get(base_url, f"/api/runs/{run_id}")
                    _get(base_url, f"/api/runs/{run_id}/metric-summary")
                    _get(base_url, f"/api/runs/{run_id}/timeline")
                    _get(base_url, f"/api/runs/{run_id}/videos")
                    _get(
                        base_url,
                        f"/api/runs/{run_id}/videos/{urllib.parse.quote(video_name)}/timeline?bucket_count=2&window_size=2",
                    )
                    _get(base_url, f"/api/runs/{run_id}/samples/{sample_id}")

                reopened = _get(base_url, f"/api/runs/{run_id}")
                self.assertEqual(reopened["status"], "completed")
                self.assertEqual(reopened["metric_job_id"], run["metric_job_id"])
                self.assertEqual(reopened["updated_at"], initial_updated_at)
                self.assertEqual(
                    [
                        (job["id"], job["kind"], job["status"], json.dumps(job.get("payload") or {}, sort_keys=True))
                        for job in db.list_jobs(limit=50)
                    ],
                    initial_jobs,
                )
                self.assertEqual(_get(base_url, f"/api/runs/{run_id}/metrics"), initial_metrics)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_run_videos_endpoint_supports_pagination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()

            frame = root / "frame.png"
            Image.new("RGB", (8, 8), (32, 64, 96)).save(frame)
            model_id = db.register_model("pagination-model", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("pagination-dataset", str(root), has_gt=True)
            for index, video_name in enumerate(["alpha.mp4", "beta.mp4", "gamma.mp4"]):
                stem = Path(video_name).stem
                db.add_sample(
                    dataset_id,
                    f"{stem}-sample",
                    str(frame),
                    str(frame),
                    str(frame),
                    {
                        "video_file": video_name,
                        "video_name": stem,
                        "frame_index": 1,
                        "sample_index": index,
                        "fps": 24.0,
                    },
                )
            run_id = db.create_run(
                name="pagination-run",
                model_id=model_id,
                dataset_id=dataset_id,
                height=8,
                width=8,
                batch_size=1,
                device="cpu",
                precision="fp32",
                metrics=[],
            )

            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                first_page = _get(base_url, f"/api/runs/{run_id}/videos?page=1&page_size=1")
                second_page = _get(base_url, f"/api/runs/{run_id}/videos?page=2&page_size=1")
                third_page = _get(base_url, f"/api/runs/{run_id}/videos?page=3&page_size=1")

                self.assertEqual(first_page["filtered_count"], 3)
                self.assertEqual(first_page["total_pages"], 3)
                self.assertEqual(first_page["videos"][0]["video_file"], "alpha.mp4")
                self.assertEqual(second_page["videos"][0]["video_file"], "beta.mp4")
                self.assertEqual(third_page["videos"][0]["video_file"], "gamma.mp4")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_sample_preview_variant_is_downscaled_while_original_stays_full_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            for folder in ("img0", "img1", "gt"):
                (dataset_root / folder).mkdir(parents=True)
            size = (2048, 1024)
            Image.new("RGB", size, (16, 32, 48)).save(dataset_root / "img0" / "sample000.png")
            Image.new("RGB", size, (48, 32, 16)).save(dataset_root / "img1" / "sample000.png")
            Image.new("RGB", size, (24, 24, 24)).save(dataset_root / "gt" / "sample000.png")

            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, size[1], size[0], {})
            dataset_id = db.create_dataset("preview-demo", str(dataset_root), has_gt=True)
            self.assertEqual(scan_triplet_dataset(db, dataset_id), 1)
            run_id = db.create_run(
                name="preview-demo",
                model_id=model_id,
                dataset_id=dataset_id,
                height=size[1],
                width=size[0],
                batch_size=1,
                device="cpu",
                precision="fp32",
                metrics=[],
                metadata={"visualize_height": size[1], "visualize_width": size[0]},
            )

            run_worker(db, workspace, WorkerOptions(role="inference", once=True, worker_id="preview-test-worker"))
            run = db.get_run(run_id)
            self.assertEqual(run["status"], "completed", run)

            sample_id = int(db.list_samples(dataset_id)[0]["id"])
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                sample_detail = _get(base_url, f"/api/runs/{run_id}/samples/{sample_id}")
                pred = sample_detail["artifacts"]["pred"]
                self.assertTrue(pred["has_preview"])

                with urllib.request.urlopen(f"{base_url}{pred['preview_url']}", timeout=30) as response:
                    preview_bytes = response.read()
                with Image.open(io.BytesIO(preview_bytes)) as preview_image:
                    self.assertLessEqual(max(preview_image.size), 512)

                with urllib.request.urlopen(f"{base_url}{pred['original_url']}", timeout=30) as response:
                    original_bytes = response.read()
                with Image.open(io.BytesIO(original_bytes)) as original_image:
                    self.assertEqual(original_image.size, size)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_frontend_binds_sample_images_to_preview_urls_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urllib.request.urlopen(f"{base_url}/app.js", timeout=30) as response:
                    app_js = response.read().decode("utf-8")
                self.assertIn("const url = artifact.preview_url || artifact.original_url;", app_js)
                self.assertIn("const href = artifact.original_url || url;", app_js)
                self.assertIn('<img src="${escapeHtml(url)}" alt="${escapeHtml(label)}" loading="lazy">', app_js)
                self.assertIn('href="${escapeHtml(item.original_url || `/api/files/${item.id}`)}"', app_js)
                self.assertIn('src="${escapeHtml(item.preview_url || `/api/files/${item.id}`)}"', app_js)
                self.assertIn('<video controls playsinline preload="metadata" src="${escapeHtml(url)}"', app_js)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_frontend_renders_global_metric_environment_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urllib.request.urlopen(f"{base_url}/", timeout=30) as response:
                    index_html = response.read().decode("utf-8")
                with urllib.request.urlopen(f"{base_url}/app.js", timeout=30) as response:
                    app_js = response.read().decode("utf-8")

                self.assertIn('id="metric-environment"', index_html)
                self.assertIn("function renderMetricEnvironmentPanel()", app_js)
                self.assertIn('const container = $("metric-environment");', app_js)
                self.assertIn("renderMetricEnvironmentPanel();", app_js)
                self.assertIn('state.metricHealth.asset_root || "set/metrics"', app_js)
                self.assertIn("function renderPortableMetricHealthTable(rowsByName)", app_js)
                self.assertIn("<details class=\"metric-health-details\">", app_js)
                self.assertIn("renderMetricHealthSummary", app_js)
                self.assertIn("renderPortableMetricHealthTable(rowsByName)", app_js)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_frontend_renders_output_health_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urllib.request.urlopen(f"{base_url}/app.js", timeout=30) as response:
                    app_js = response.read().decode("utf-8")

                self.assertIn("function renderOutputHealthReport(run)", app_js)
                self.assertIn("run?.result?.output_health", app_js)
                self.assertIn("Output health", app_js)
                self.assertIn("renderOutputHealthReport(run)", app_js)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_frontend_run_detail_uses_paged_video_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urllib.request.urlopen(f"{base_url}/app.js", timeout=30) as response:
                    app_js = response.read().decode("utf-8")

                self.assertIn("async function loadRunVideosPage(runId, page = 1)", app_js)
                self.assertIn("/api/runs/${runId}/videos?page=${page}&page_size=20", app_js)
                self.assertIn("renderRunVideosPager()", app_js)
                self.assertIn('data-run-videos-page="${Number(page.page || 1) - 1}"', app_js)
                self.assertIn('data-run-videos-page="${Number(page.page || 1) + 1}"', app_js)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_frontend_keeps_extra_artifacts_collapsed_until_expanded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                group = next(row for row in list_video_groups(workspace) if row["name"] == "test_style")
                selected_video = group["videos"][0]["name"]
                created = _post(
                    base_url,
                    "/api/runs",
                    {
                        "model_file": "test_dict_return.py",
                        "video_group": "test_style",
                        "selected_videos": [selected_video],
                        "device": "cpu",
                        "precision": "fp32",
                        "batch_size": 1,
                        "max_frames": 4,
                    },
                )
                run_id = created["run_id"]
                run = _wait_for_run(base_url, run_id)
                self.assertEqual(run["status"], "completed", run)

                sample_id = int(db.list_samples(int(db.get_run(run_id)["dataset_id"]))[0]["id"])
                sample_detail = _get(base_url, f"/api/runs/{run_id}/samples/{sample_id}")
                self.assertTrue(sample_detail["extra_artifacts"])
                self.assertIn("preview_url", sample_detail["extra_artifacts"][0])
                self.assertTrue(sample_detail["extra_artifacts"][0]["preview_url"].startswith("/api/files/"))

                with urllib.request.urlopen(f"{base_url}/app.js", timeout=30) as response:
                    app_js = response.read().decode("utf-8")
                self.assertIn("expandedExtraArtifactsBySample", app_js)
                self.assertIn('data-extra-toggle="${escapeHtml(sample.sample_id)}"', app_js)
                self.assertIn('展开后按需加载 extra_* 预览。', app_js)
                self.assertIn('state.expandedExtraArtifactsBySample[sample.sample_id]', app_js)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_frontend_shows_cleaned_artifact_notice_without_loading_sample_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urllib.request.urlopen(f"{base_url}/app.js", timeout=30) as response:
                    app_js = response.read().decode("utf-8")
                self.assertIn("sample.has_artifacts === false", app_js)
                self.assertIn("这个 Run 的产物已清理；如需重新查看预览，请重试重新生成。", app_js)
                self.assertIn("renderCleanedArtifactsNotice", app_js)
                self.assertIn("!detail && sample.has_artifacts !== false", app_js)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_api_file_run_records_cgvqm_as_video_only_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous_metric_assets = os.environ.get("VFIEVAL_METRIC_ASSETS_DIR")
            os.environ["VFIEVAL_METRIC_ASSETS_DIR"] = str(Path(tmp) / "missing_metrics")
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                group = next(row for row in list_video_groups(workspace) if row["name"] == "test_style")
                selected_video = group["videos"][0]["name"]
                created = _post(
                    base_url,
                    "/api/runs",
                    {
                        "model_file": "test_average.py",
                        "video_group": "test_style",
                        "selected_videos": [selected_video],
                        "device": "cpu",
                        "precision": "fp32",
                        "batch_size": 2,
                        "max_frames": 4,
                        "metrics": ["cgvqm"],
                    },
                )
                run_id = created["run_id"]
                run = _wait_for_run(base_url, run_id)
                self.assertEqual(run["status"], "completed", run)

                summary = _get(base_url, f"/api/runs/{run_id}/metric-summary")
                self.assertEqual(summary["metrics"]["cgvqm"]["unavailable"], 1)
                self.assertIn("cgvqm evaluator is missing_weights", summary["metrics"]["cgvqm"]["reasons"][0])

                timeline = _get(base_url, f"/api/runs/{run_id}/timeline")
                self.assertNotIn("cgvqm", timeline["videos"][0]["samples"][0]["metrics"])
                self.assertIn("cgvqm", timeline["videos"][0]["video_metrics"])
                self.assertEqual(timeline["videos"][0]["video_metrics"]["cgvqm"]["status"], "unavailable")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                if previous_metric_assets is None:
                    os.environ.pop("VFIEVAL_METRIC_ASSETS_DIR", None)
                else:
                    os.environ["VFIEVAL_METRIC_ASSETS_DIR"] = previous_metric_assets

    def test_run_detail_api_exposes_human_readable_failed_run_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "dataset"
            for folder in ("img0", "img1", "gt"):
                (dataset_root / folder).mkdir(parents=True)
            Image.new("RGB", (8, 8), (32, 16, 8)).save(dataset_root / "img0" / "sample000.png")
            Image.new("RGB", (8, 8), (8, 16, 32)).save(dataset_root / "img1" / "sample000.png")
            Image.new("RGB", (8, 8), (16, 16, 16)).save(dataset_root / "gt" / "sample000.png")

            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                model = _post(
                    base_url,
                    "/api/models",
                    {
                        "name": "failing-model",
                        "adapter": f"file:{ROOT / 'models' / 'test_exception.py'}",
                        "input_height": 8,
                        "input_width": 8,
                    },
                )
                dataset = _post(
                    base_url,
                    "/api/datasets",
                    {
                        "name": "failing-dataset",
                        "root_path": str(dataset_root),
                        "has_gt": True,
                    },
                )
                _post(base_url, f"/api/datasets/{dataset['dataset_id']}/scan", {})
                created = _post(
                    base_url,
                    "/api/runs",
                    {
                        "model_id": model["model_id"],
                        "dataset_id": dataset["dataset_id"],
                        "height": 8,
                        "width": 8,
                        "batch_size": 1,
                        "device": "cpu",
                        "precision": "fp32",
                    },
                )
                run_id = created["run_id"]

                run_worker(db, workspace, WorkerOptions(role="inference", once=True, worker_id="failing-ui-worker"))

                failed = _get(base_url, f"/api/runs/{run_id}")
                self.assertEqual(failed["status"], "failed", failed)
                self.assertIn("intentional test model exception", failed["error"]["message"])
                self.assertEqual(failed["error"]["type"], "RuntimeError")
                self.assertEqual(len(failed["jobs"]), 1)
                self.assertEqual(failed["jobs"][0]["status"], "failed")
                self.assertIn("intentional test model exception", failed["jobs"][0]["error"]["message"])
                visible_runs = _get(base_url, "/api/runs")
                listed = next(item for item in visible_runs if int(item["id"]) == run_id)
                self.assertEqual(listed["status"], "failed")
                self.assertIn("intentional test model exception", listed["error"]["message"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_no_gt_samples_are_reported_as_skipped_for_full_reference_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "dataset"
            for folder in ("img0", "img1"):
                (dataset_root / folder).mkdir(parents=True)
            Image.new("RGB", (8, 8), (12, 34, 56)).save(dataset_root / "img0" / "sample000.png")
            Image.new("RGB", (8, 8), (56, 34, 12)).save(dataset_root / "img1" / "sample000.png")

            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("no-gt", str(dataset_root), has_gt=False)
            self.assertEqual(scan_triplet_dataset(db, dataset_id), 1)
            run_id = db.create_run(
                name="no-gt-run",
                model_id=model_id,
                dataset_id=dataset_id,
                height=8,
                width=8,
                batch_size=1,
                device="cpu",
                precision="fp32",
                metrics=["lpips_vit_patch"],
            )

            run_worker(db, workspace, WorkerOptions(role="inference", once=True, worker_id="no-gt-inference"))
            run_worker(db, workspace, WorkerOptions(role="metric", once=True, worker_id="no-gt-metric"))

            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                run = _get(base_url, f"/api/runs/{run_id}")
                self.assertEqual(run["status"], "completed", run)
                timeline = _get(base_url, f"/api/runs/{run_id}/timeline")
                metric = timeline["videos"][0]["samples"][0]["metrics"]["lpips_vit_patch"]
                self.assertEqual(metric["status"], "skipped")
                self.assertEqual(metric["details"]["reason"], "sample has no ground-truth reference")

                summary = _get(base_url, f"/api/runs/{run_id}/metric-summary")
                self.assertEqual(summary["metrics"]["lpips_vit_patch"]["skipped"], 1)
                self.assertIn("sample has no ground-truth reference", summary["metrics"]["lpips_vit_patch"]["reasons"][0])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_metric_summary_and_worst_samples_follow_lpips_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()

            frame_a = Path(tmp) / "a.png"
            frame_b = Path(tmp) / "b.png"
            Image.new("RGB", (8, 8), (10, 20, 30)).save(frame_a)
            Image.new("RGB", (8, 8), (30, 20, 10)).save(frame_b)

            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("direction-demo", str(Path(tmp)), True)
            sample_a = db.add_sample(
                dataset_id,
                "sample-a",
                str(frame_a),
                str(frame_a),
                str(frame_a),
                {"video_name": "clip", "video_file": "clip.mp4", "frame_index": 0, "sample_index": 0, "fps": 24.0},
            )
            sample_b = db.add_sample(
                dataset_id,
                "sample-b",
                str(frame_b),
                str(frame_b),
                str(frame_b),
                {"video_name": "clip", "video_file": "clip.mp4", "frame_index": 1, "sample_index": 1, "fps": 24.0},
            )
            run_id = db.create_run(
                name="direction-demo",
                model_id=model_id,
                dataset_id=dataset_id,
                height=8,
                width=8,
                batch_size=1,
                device="cpu",
                precision="fp32",
                metrics=["lpips_vit_patch"],
                create_inference_job=False,
            )
            inference_job_id = db.add_run_job(run_id, "inference", {"run_id": run_id}, progress_total=2)
            metric_job_id = db.add_run_job(
                run_id,
                "metric",
                {"run_id": run_id, "dataset_id": dataset_id, "metric_names": ["lpips_vit_patch"]},
                progress_total=2,
            )
            db.mark_run_started(run_id, "metric_running")
            db.add_metric_result(metric_job_id, inference_job_id, sample_a, "lpips_vit_patch", "completed", 0.10, {})
            db.add_metric_result(metric_job_id, inference_job_id, sample_b, "lpips_vit_patch", "completed", 0.80, {})
            db.complete_job(inference_job_id, {"samples": 2})
            db.complete_job(metric_job_id, {"summary": {"lpips_vit_patch": {"completed": 2}}})
            db.complete_run_metrics(run_id, {"lpips_vit_patch": {"completed": 2, "mean": 0.45}})

            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                summary = _get(base_url, f"/api/runs/{run_id}/metric-summary")
                self.assertEqual(summary["metrics"]["lpips_vit_patch"]["direction"], "lower_is_better")
                self.assertEqual(summary["metrics"]["lpips_vit_patch"]["worst_sample_id"], int(sample_b))
                self.assertEqual(summary["metrics"]["lpips_vit_patch"]["worst_value"], 0.80)

                timeline = _get(base_url, f"/api/runs/{run_id}/timeline")
                worst = timeline["videos"][0]["worst_samples"]["lpips_vit_patch"]
                self.assertEqual([row["sample_id"] for row in worst], [int(sample_b), int(sample_a)])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_delete_running_run_hides_it_and_cancels_worker_at_boundary(self) -> None:
        import torch

        class SlowModel:
            def predict(self, img0, img1, t):
                time.sleep(0.12)
                batch, _channels, height, width = img0.shape
                flow = torch.zeros((batch, 2, height, width), dtype=img0.dtype, device=img0.device)
                mask = torch.zeros((batch, 1, height, width), dtype=img0.dtype, device=img0.device)
                return {
                    "flowt_0": flow,
                    "flowt_1": flow,
                    "mask0": mask,
                    "mask1": mask,
                }

        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                group = next(row for row in list_video_groups(workspace) if row["name"] == "test_style")
                selected_video = group["videos"][0]["name"]
                with patch("vfieval.pipeline.inference.load_flow_mask_model", return_value=SlowModel()):
                    created = _post(
                        base_url,
                        "/api/runs",
                        {
                            "model_file": "test_average.py",
                            "video_group": "test_style",
                            "selected_videos": [selected_video],
                            "device": "cpu",
                            "precision": "fp32",
                            "batch_size": 1,
                        },
                    )
                    run_id = created["run_id"]

                    deadline = time.time() + 10
                    running_seen = False
                    while time.time() < deadline:
                        run = _get(base_url, f"/api/runs/{run_id}")
                        if run["status"] == "running":
                            running_seen = True
                            break
                        time.sleep(0.05)
                    self.assertTrue(running_seen, "run never entered running state before delete")

                    deleted = _delete(base_url, f"/api/runs/{run_id}")
                    self.assertTrue(deleted["deleted"])

                    visible_runs = _get(base_url, "/api/runs")
                    self.assertNotIn(run_id, [int(item["id"]) for item in visible_runs])

                    canceled_run = _wait_for_run(base_url, run_id)
                    self.assertEqual(canceled_run["status"], "canceled", canceled_run)

                    all_runs = _get(base_url, "/api/runs?include_deleted=1")
                    self.assertIn(run_id, [int(item["id"]) for item in all_runs])

                    inference_job = db.get_job(int(canceled_run["inference_job_id"]))
                    self.assertEqual(inference_job["status"], "canceled")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_cleanup_artifacts_rejects_running_run(self) -> None:
        import torch

        class SlowModel:
            def predict(self, img0, img1, t):
                time.sleep(0.12)
                batch, _channels, height, width = img0.shape
                flow = torch.zeros((batch, 2, height, width), dtype=img0.dtype, device=img0.device)
                mask = torch.zeros((batch, 1, height, width), dtype=img0.dtype, device=img0.device)
                return {
                    "flowt_0": flow,
                    "flowt_1": flow,
                    "mask0": mask,
                    "mask1": mask,
                }

        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                group = next(row for row in list_video_groups(workspace) if row["name"] == "test_style")
                selected_video = group["videos"][0]["name"]
                with patch("vfieval.pipeline.inference.load_flow_mask_model", return_value=SlowModel()):
                    created = _post(
                        base_url,
                        "/api/runs",
                        {
                            "model_file": "test_average.py",
                            "video_group": "test_style",
                            "selected_videos": [selected_video],
                            "device": "cpu",
                            "precision": "fp32",
                            "batch_size": 1,
                        },
                    )
                    run_id = created["run_id"]

                    deadline = time.time() + 10
                    running_run = None
                    while time.time() < deadline:
                        run = _get(base_url, f"/api/runs/{run_id}")
                        if run["status"] == "running":
                            running_run = run
                            break
                        time.sleep(0.05)
                    self.assertIsNotNone(running_run, "run never entered running state before cleanup attempt")

                    output_dir = Path(str(running_run["metadata"]["output_dir"]))
                    self.assertTrue(output_dir.exists())
                    request = urllib.request.Request(
                        f"{base_url}/api/runs/{run_id}/cleanup-artifacts",
                        data=b"{}",
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with self.assertRaises(urllib.error.HTTPError) as error_context:
                        urllib.request.urlopen(request, timeout=30)
                    self.assertEqual(error_context.exception.code, 400)
                    payload = json.loads(error_context.exception.read().decode("utf-8"))
                    self.assertIn("completed, failed, or canceled", payload["error"]["message"])
                    self.assertTrue(output_dir.exists())
                    self.assertIsNone(db.get_run(run_id).get("artifact_cleaned_at"))

                    deleted = _delete(base_url, f"/api/runs/{run_id}")
                    self.assertTrue(deleted["deleted"])
                    canceled_run = _wait_for_run(base_url, run_id)
                    self.assertEqual(canceled_run["status"], "canceled", canceled_run)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_cleanup_artifacts_only_removes_current_run_directory_even_if_metadata_output_dir_is_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()

            source_dir = root / "source_videos"
            source_dir.mkdir(parents=True, exist_ok=True)
            source_marker = source_dir / "keep.txt"
            source_marker.write_text("keep", encoding="utf-8")

            model_id = db.register_model("cleanup-model", "dummy", None, 16, 16, {})
            dataset_id = db.create_dataset("cleanup-dataset", str(source_dir), has_gt=False)
            run_id = db.create_run(
                name="cleanup-safety",
                model_id=model_id,
                dataset_id=dataset_id,
                height=16,
                width=16,
                batch_size=1,
                device="cpu",
                precision="fp32",
                metrics=[],
                metadata={"output_dir": str(workspace.runs_dir)},
                create_inference_job=False,
            )
            db.complete_run_inference(run_id, {"output_dir": str(workspace.runs_dir)}, {"total": 1}, "completed")

            run_dir = workspace.runs_dir / str(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "artifact.txt").write_text("artifact", encoding="utf-8")

            sibling_dir = workspace.runs_dir / "999"
            sibling_dir.mkdir(parents=True, exist_ok=True)
            sibling_marker = sibling_dir / "keep.txt"
            sibling_marker.write_text("sibling", encoding="utf-8")

            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                cleanup = _post(base_url, f"/api/runs/{run_id}/cleanup-artifacts", {})
                self.assertTrue(cleanup["artifact_cleaned"])
                self.assertEqual(Path(cleanup["output_dir"]), run_dir)
                self.assertFalse(run_dir.exists())
                self.assertTrue(sibling_dir.exists())
                self.assertEqual(sibling_marker.read_text(encoding="utf-8"), "sibling")
                self.assertTrue(source_dir.exists())
                self.assertEqual(source_marker.read_text(encoding="utf-8"), "keep")
                self.assertIsNotNone(db.get_run(run_id).get("artifact_cleaned_at"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_api_video_compare_run_generates_gt_pred_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            gt_path = Path(tmp) / "videos" / "anime" / "clip.mp4"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            write_mp4(gt_path, [(index * 20, 0, 0) for index in range(3)])
            pred_path = workspace.root / "pred.mp4"
            write_mp4(pred_path, [(0, index * 20, 0) for index in range(3)])
            run_a = add_completed_pred_run(db, workspace, "ModelA", pred_path)

            payload = {
                "run_type": "video_compare",
                "reference": {"kind": "video_group", "group": "anime", "video": "clip.mp4"},
                "distorted": [{"kind": "run_artifact", "run_id": run_a, "video": "clip", "label": "ModelA"}],
                "align_mode": "strict",
            }

            server, thread, base_url = start_server(db, workspace)
            try:
                preflight = post_json(base_url, "/api/preflight", payload)
                self.assertTrue(preflight["ok"], preflight)
                self.assertEqual(preflight["alignment"]["frame_count"], 3)

                with patch("vfieval.server._start_local_inference_worker", return_value=None):
                    created = post_json(base_url, "/api/runs", payload)
                run_id = int(created["run_id"])
                job_id = int(db.get_run(run_id)["inference_job_id"])
                run_inference_job(db, workspace, job_id)

                run = db.get_run(run_id)
                self.assertEqual(run["status"], "completed", run)
                self.assertEqual(run["metadata"]["run_type"], "video_compare")

                run_videos = _get(base_url, f"/api/runs/{run_id}/videos")
                self.assertEqual(len(run_videos["videos"]), 1)
                self.assertEqual(run_videos["videos"][0]["sample_count"], 3)
                self.assertIn("pred_video", run_videos["videos"][0]["video_artifacts"])

                timeline = _get(base_url, f"/api/runs/{run_id}/timeline")
                sample = timeline["videos"][0]["samples"][0]
                sample_detail = _get(base_url, f"/api/runs/{run_id}/samples/{sample['sample_id']}")
                self.assertIn("gt", sample_detail["artifacts"])
                self.assertIn("pred", sample_detail["artifacts"])
                self.assertIn("difference", sample_detail["artifacts"])
                self.assertNotIn("flowt_0", sample_detail["artifacts"])
                self.assertNotIn("mask0", sample_detail["artifacts"])
            finally:
                stop_server(server, thread)

    def test_video_compare_run_bypasses_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            gt_path = Path(tmp) / "videos" / "anime" / "clip.mp4"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            write_mp4(gt_path, [(index * 20, 0, 0) for index in range(3)])
            pred_path = workspace.root / "pred.mp4"
            write_mp4(pred_path, [(0, index * 20, 0) for index in range(3)])
            run_a = add_completed_pred_run(db, workspace, "ModelA", pred_path)

            payload = {
                "run_type": "video_compare",
                "reference": {"kind": "video_group", "group": "anime", "video": "clip.mp4"},
                "distorted": [{"kind": "run_artifact", "run_id": run_a, "video": "clip", "label": "ModelA"}],
                "align_mode": "strict",
            }

            server, thread, base_url = start_server(db, workspace)
            try:
                with patch("vfieval.server._start_local_inference_worker", return_value=None):
                    created = post_json(base_url, "/api/runs", payload)

                run_id = int(created["run_id"])
                job_id = int(db.get_run(run_id)["inference_job_id"])
                with patch(
                    "vfieval.pipeline.inference.load_flow_mask_model",
                    side_effect=AssertionError("video_compare should not load model"),
                ) as load_model:
                    result = run_inference_job(db, workspace, job_id)

                self.assertEqual(result.samples, 3)
                self.assertFalse(load_model.called)

                run = db.get_run(run_id)
                self.assertEqual(run["status"], "completed", run)
                self.assertEqual(run["metadata"]["run_type"], "video_compare")

                timeline = _get(base_url, f"/api/runs/{run_id}/timeline")
                sample = timeline["videos"][0]["samples"][0]
                sample_detail = _get(base_url, f"/api/runs/{run_id}/samples/{sample['sample_id']}")
                self.assertIn("gt", sample_detail["artifacts"])
                self.assertIn("pred", sample_detail["artifacts"])
                self.assertIn("difference", sample_detail["artifacts"])
                self.assertNotIn("flowt_0", sample_detail["artifacts"])
                self.assertNotIn("mask0", sample_detail["artifacts"])
                self.assertNotIn("warp0", sample_detail["artifacts"])
                self.assertNotIn("blend", sample_detail["artifacts"])
            finally:
                stop_server(server, thread)

    def test_video_compare_preflight_rejects_mismatched_frame_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            gt_path = Path(tmp) / "videos" / "anime" / "clip.mp4"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            write_mp4(gt_path, [(index * 20, 0, 0) for index in range(3)])
            pred_path = workspace.root / "pred-short.mp4"
            write_mp4(pred_path, [(0, index * 20, 0) for index in range(2)])
            run_a = add_completed_pred_run(db, workspace, "ModelA", pred_path, sample_count=2)

            result = preflight_run(
                db,
                workspace,
                {
                    "run_type": "video_compare",
                    "reference": {"kind": "video_group", "group": "anime", "video": "clip.mp4"},
                    "distorted": [{"kind": "run_artifact", "run_id": run_a, "video": "clip", "label": "ModelA"}],
                    "align_mode": "strict",
                },
            )
            self.assertFalse(result["ok"])
            self.assertIn("matching frame counts", result["errors"][0]["message"])

    def test_video_compare_rejects_raw_string_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            source_video = next((ROOT / "videos" / "test_style").glob("*.avi"))
            result = preflight_run(
                db,
                workspace,
                {
                    "run_type": "video_compare",
                    "reference": str(source_video),
                    "distorted": str(source_video),
                    "align_mode": "strict",
                },
            )
            self.assertFalse(result["ok"])
            self.assertIn("descriptor must be an object", result["errors"][0]["message"])

    def test_compare_dataset_rejects_mismatched_fps_metadata(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            reference_dir = Path(tmp) / "reference_frames"
            distorted_dir = Path(tmp) / "distorted_frames"
            reference_dir.mkdir()
            distorted_dir.mkdir()
            reference_frames = []
            distorted_frames = []
            for index in range(2):
                reference_path = reference_dir / f"{index:06d}.png"
                distorted_path = distorted_dir / f"{index:06d}.png"
                Image.new("RGB", (16, 16), (index * 20, 0, 0)).save(reference_path)
                Image.new("RGB", (16, 16), (0, index * 20, 0)).save(distorted_path)
                reference_frames.append(reference_path)
                distorted_frames.append(distorted_path)

            dataset_id = db.create_dataset(
                "compare-fps",
                str(reference_dir),
                True,
                source_type="compare",
                decode_mode="compare",
                metadata={
                    "reference_path": str(reference_dir),
                    "distorted_path": str(distorted_dir),
                    "align_mode": "strict",
                },
            )
            with patch(
                "vfieval.datasets._load_compare_source_frames",
                side_effect=[
                    (reference_frames, 24.0, [0.0, 0.125]),
                    (distorted_frames, 30.0, [0.0, 1.0 / 30.0]),
                ],
            ):
                with self.assertRaisesRegex(ValueError, "matching fps metadata"):
                    scan_compare_dataset(db, workspace, dataset_id)

    def test_compare_dataset_rejects_mismatched_frame_timestamps(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            reference_dir = Path(tmp) / "reference_frames"
            distorted_dir = Path(tmp) / "distorted_frames"
            reference_dir.mkdir()
            distorted_dir.mkdir()
            reference_frames = []
            distorted_frames = []
            for index in range(2):
                reference_path = reference_dir / f"{index:06d}.png"
                distorted_path = distorted_dir / f"{index:06d}.png"
                Image.new("RGB", (16, 16), (index * 20, 0, 0)).save(reference_path)
                Image.new("RGB", (16, 16), (0, index * 20, 0)).save(distorted_path)
                reference_frames.append(reference_path)
                distorted_frames.append(distorted_path)

            dataset_id = db.create_dataset(
                "compare-timestamps",
                str(reference_dir),
                True,
                source_type="compare",
                decode_mode="compare",
                metadata={
                    "reference_path": str(reference_dir),
                    "distorted_path": str(distorted_dir),
                    "align_mode": "strict",
                },
            )
            with patch(
                "vfieval.datasets._load_compare_source_frames",
                side_effect=[
                    (reference_frames, 24.0, [0.0, 0.125]),
                    (distorted_frames, 24.0, [0.0, 0.250]),
                ],
            ):
                with self.assertRaisesRegex(ValueError, "matching frame timestamps"):
                    scan_compare_dataset(db, workspace, dataset_id)

    def test_video_compare_preflight_rejects_mismatched_decoded_fps(self) -> None:
        fake_frames = [Path("ref000.png"), Path("ref001.png")]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            gt_path = Path(tmp) / "videos" / "anime" / "clip.mp4"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            write_mp4(gt_path, [(index * 20, 0, 0) for index in range(2)])
            pred_path = workspace.root / "pred.mp4"
            write_mp4(pred_path, [(0, index * 20, 0) for index in range(2)])
            run_a = add_completed_pred_run(db, workspace, "ModelA", pred_path, sample_count=2)
            with patch(
                "vfieval.datasets._load_compare_source_frames",
                side_effect=[
                    (fake_frames, 24.0, [0.0, 0.125]),
                    (fake_frames, 30.0, [0.0, 1.0 / 30.0]),
                ],
            ):
                result = preflight_run(
                    db,
                    workspace,
                    {
                        "run_type": "video_compare",
                        "reference": {"kind": "video_group", "group": "anime", "video": "clip.mp4"},
                        "distorted": [{"kind": "run_artifact", "run_id": run_a, "video": "clip", "label": "ModelA"}],
                        "align_mode": "strict",
                    },
                )
            self.assertFalse(result["ok"])
            self.assertIn("matching fps metadata", result["errors"][0]["message"])

    def test_video_compare_preflight_rejects_mismatched_decoded_timestamps(self) -> None:
        fake_frames = [Path("ref000.png"), Path("ref001.png")]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            gt_path = Path(tmp) / "videos" / "anime" / "clip.mp4"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            write_mp4(gt_path, [(index * 20, 0, 0) for index in range(2)])
            pred_path = workspace.root / "pred.mp4"
            write_mp4(pred_path, [(0, index * 20, 0) for index in range(2)])
            run_a = add_completed_pred_run(db, workspace, "ModelA", pred_path, sample_count=2)
            with patch(
                "vfieval.datasets._load_compare_source_frames",
                side_effect=[
                    (fake_frames, 24.0, [0.0, 0.125]),
                    (fake_frames, 24.0, [0.0, 0.250]),
                ],
            ):
                result = preflight_run(
                    db,
                    workspace,
                    {
                        "run_type": "video_compare",
                        "reference": {"kind": "video_group", "group": "anime", "video": "clip.mp4"},
                        "distorted": [{"kind": "run_artifact", "run_id": run_a, "video": "clip", "label": "ModelA"}],
                        "align_mode": "strict",
                    },
                )
            self.assertFalse(result["ok"])
            self.assertIn("matching frame timestamps", result["errors"][0]["message"])

    def test_metric_timeline_reports_pending_and_running_without_faking_video_level_points(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()

            frame = Path(tmp) / "frame.png"
            Image.new("RGB", (8, 8), (64, 32, 16)).save(frame)

            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("frames", str(Path(tmp)), True)
            sample_id = db.add_sample(dataset_id, "sample-a", str(frame), str(frame), str(frame), {"video_name": "clip", "video_file": "clip.mp4", "frame_index": 0, "sample_index": 0, "fps": 24.0})
            run_id = db.create_run(
                name="metric-state",
                model_id=model_id,
                dataset_id=dataset_id,
                height=8,
                width=8,
                batch_size=1,
                device="cpu",
                precision="fp32",
                metrics=["lpips_vit_patch", "vmaf"],
                create_inference_job=False,
            )

            metric_job_id = db.add_run_job(
                run_id,
                "metric",
                {"run_id": run_id, "dataset_id": dataset_id, "metric_names": ["lpips_vit_patch", "vmaf"]},
                progress_total=1,
            )
            db.set_run_metric_job(run_id, metric_job_id)

            queued_timeline = _run_timeline(db, run_id)
            queued_sample = queued_timeline["videos"][0]["samples"][0]
            self.assertEqual(queued_sample["metrics"]["lpips_vit_patch"]["status"], "pending")
            self.assertNotIn("vmaf", queued_sample["metrics"])
            self.assertEqual(queued_timeline["videos"][0]["video_metrics"]["vmaf"]["status"], "pending")
            queued_summary = _run_metric_summary(db, run_id)
            self.assertEqual(queued_summary["metrics"]["lpips_vit_patch"]["pending"], 1)
            self.assertEqual(queued_summary["metrics"]["vmaf"]["pending"], 1)

            db.register_worker("metric-worker", "metric", {"cpu": True})
            db.claim_next_job("metric-worker", ["metric"])
            db.mark_run_started(run_id, "metric_running")

            running_timeline = _run_timeline(db, run_id)
            running_sample = running_timeline["videos"][0]["samples"][0]
            self.assertEqual(running_sample["metrics"]["lpips_vit_patch"]["status"], "running")
            self.assertNotIn("vmaf", running_sample["metrics"])
            self.assertEqual(running_timeline["videos"][0]["video_metrics"]["vmaf"]["status"], "running")
            running_summary = _run_metric_summary(db, run_id)
            self.assertEqual(running_summary["metrics"]["lpips_vit_patch"]["running"], 1)
            self.assertEqual(running_summary["metrics"]["vmaf"]["running"], 1)
            self.assertEqual(int(queued_sample["sample_id"]), int(sample_id))

    def test_partition_samples_keeps_video_groups_together(self) -> None:
        samples = [
            {"id": 1, "metadata": {"video_file": "a.mp4"}},
            {"id": 2, "metadata": {"video_file": "a.mp4"}},
            {"id": 3, "metadata": {"video_file": "b.mp4"}},
            {"id": 4, "metadata": {"video_file": "c.mp4"}},
        ]
        partitions = _partition_samples_by_video(samples, ["cuda:0", "cuda:1"])
        flattened = sorted(sample_id for shard in partitions for sample_id in shard)
        self.assertEqual(flattened, [1, 2, 3, 4])
        shard_by_sample = {sample_id: index for index, shard in enumerate(partitions) for sample_id in shard}
        self.assertEqual(shard_by_sample[1], shard_by_sample[2])

    def test_npu_capabilities_and_multi_npu_device_resolution(self) -> None:
        fake_npus = [{"id": f"npu:{index}", "name": f"Ascend {index}", "index": index} for index in range(8)]
        with patch("vfieval.worker.list_npu_devices", return_value=fake_npus):
            capabilities = detect_capabilities()
        self.assertEqual([row["id"] for row in capabilities["npu"]], [f"npu:{index}" for index in range(8)])
        self.assertEqual(capabilities["precision_support"]["npu"], ["fp32", "fp16"])

        with patch("vfieval.server.detect_capabilities", return_value={"npu": fake_npus, "cuda": [], "cpu": True}):
            self.assertEqual(_resolve_execution_devices({"devices": ["npu:2", "npu:7"]}, "multi_npu"), ["npu:2", "npu:7"])
            self.assertEqual(_resolve_execution_devices({}, "multi_npu"), [f"npu:{index}" for index in range(8)])

    def test_npu_discovery_does_not_fabricate_device_without_count_probe(self) -> None:
        import torch

        class FakeNpu:
            def is_available(self) -> bool:
                return True

        with (
            patch("vfieval.devices.npu_module", return_value=object()),
            patch.object(torch, "npu", FakeNpu(), create=True),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            self.assertEqual(list_npu_devices(), [])
            self.assertEqual(npu_unavailable_reason(), "torch_npu is installed but no NPU devices were reported")
            self.assertEqual(normalize_device_name("auto"), "cpu")

    def test_supported_precisions_only_advertises_npu_bf16_when_probe_succeeds(self) -> None:
        import torch

        class FakeNpu:
            def __init__(self, value: bool) -> None:
                self.value = value

            def is_bf16_supported(self) -> bool:
                return self.value

        with (
            patch("vfieval.devices.npu_is_available", return_value=True),
            patch.object(torch, "npu", FakeNpu(False), create=True),
        ):
            self.assertEqual(supported_precisions("npu"), ["fp32", "fp16"])

        with (
            patch("vfieval.devices.npu_is_available", return_value=True),
            patch.object(torch, "npu", FakeNpu(True), create=True),
        ):
            self.assertEqual(supported_precisions("npu"), ["fp32", "fp16", "bf16"])

    def test_multi_npu_preflight_dry_runs_each_selected_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            fake_npus = [{"id": f"npu:{index}", "name": f"Ascend {index}", "index": index} for index in range(2)]
            called_devices: list[str] = []

            def fake_dry_run(model_path, checkpoint_path, device_name, precision) -> None:
                called_devices.append(str(device_name))
                if str(device_name) == "npu:1":
                    raise RuntimeError("simulated second-device failure")

            with (
                patch("vfieval.file_inputs.list_npu_devices", return_value=fake_npus),
                patch("vfieval.file_inputs._dry_run_model_file", side_effect=fake_dry_run),
            ):
                result = preflight_run(
                    db,
                    workspace,
                    {
                        "model_file": "test_average.py",
                        "video_group": "test_style",
                        "selected_videos": ["blocks_motion.avi"],
                        "execution_mode": "multi_npu",
                        "devices": ["npu:0", "npu:1"],
                        "precision": "fp32",
                        "max_frames": 4,
                    },
                )

            self.assertFalse(result["ok"])
            self.assertEqual(called_devices, ["npu:0", "npu:1"])
            self.assertEqual(result["model"]["tested_devices"], ["npu:0"])
            self.assertIn("npu:1", result["errors"][0]["message"])
            self.assertIn("simulated second-device failure", result["errors"][0]["message"])

    def test_single_npu_preflight_reports_supported_precisions_and_falls_back_from_unsupported_bf16(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            fake_npus = [{"id": "npu:0", "name": "Ascend 0", "index": 0}]

            with (
                patch("vfieval.file_inputs.list_npu_devices", return_value=fake_npus),
                patch("vfieval.file_inputs.supported_precisions", return_value=["fp32", "fp16"]),
                patch("vfieval.file_inputs._dry_run_model_file", return_value=None),
            ):
                result = preflight_run(
                    db,
                    workspace,
                    {
                        "model_file": "test_average.py",
                        "video_group": "test_style",
                        "selected_videos": ["blocks_motion.avi"],
                        "device": "npu:0",
                        "precision": "bf16",
                        "max_frames": 4,
                    },
                )

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["device"]["supported_precisions"], ["fp32", "fp16"])
            self.assertEqual(result["device"]["effective_precision"], "fp32")
            self.assertTrue(result["device"]["warning"])

    def test_single_npu_preflight_surfaces_exact_unavailable_reason_without_cpu_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()

            with (
                patch("vfieval.file_inputs.list_npu_devices", return_value=[]),
                patch("vfieval.file_inputs.npu_unavailable_reason", return_value="torch_npu is not installed"),
                patch("vfieval.file_inputs._dry_run_model_file") as dry_run,
            ):
                result = preflight_run(
                    db,
                    workspace,
                    {
                        "model_file": "test_average.py",
                        "video_group": "test_style",
                        "selected_videos": ["blocks_motion.avi"],
                        "device": "npu:0",
                        "precision": "fp32",
                        "max_frames": 4,
                    },
                )

            self.assertFalse(result["ok"])
            self.assertIn("torch_npu is not installed", result["errors"][0]["message"])
            self.assertEqual(result["model"]["tested_devices"], [])
            dry_run.assert_not_called()

    def test_single_npu_preflight_reports_output_validation_stage_details(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            fake_npus = [{"id": "npu:0", "name": "Ascend 0", "index": 0}]

            class FakeModel:
                def predict(self, img0, img1, _t):
                    return {
                        "flowt_0": torch.zeros((1, 2, 8, 8), dtype=img0.dtype, device=img0.device),
                        "flowt_1": torch.zeros((1, 2, 8, 8), dtype=img0.dtype, device=img0.device),
                        "mask0": torch.zeros((1, 1, 8, 8), dtype=img0.dtype, device=img0.device),
                        "mask1": torch.zeros((1, 1, 8, 8), dtype=img0.dtype, device=img0.device),
                    }

            with (
                patch("vfieval.file_inputs.list_npu_devices", return_value=fake_npus),
                patch("vfieval.file_inputs.resolve_torch_device", return_value=torch.device("cpu")),
                patch("vfieval.file_inputs.load_flow_mask_model", return_value=FakeModel()),
                patch(
                    "vfieval.file_inputs.validate_model_outputs",
                    side_effect=ValueError("Model output field flowt_0 is on CPU, expected npu:0"),
                ),
            ):
                result = preflight_run(
                    db,
                    workspace,
                    {
                        "model_file": "test_average.py",
                        "video_group": "test_style",
                        "selected_videos": ["blocks_motion.avi"],
                        "device": "npu:0",
                        "precision": "fp32",
                        "max_frames": 4,
                    },
                )

            self.assertFalse(result["ok"])
            self.assertEqual(result["model"]["tested_devices"], [])
            self.assertIn("model dry-run failed on npu:0", result["errors"][0]["message"])
            self.assertIn("output_validation", result["errors"][0]["message"])
            self.assertIn("Model output field flowt_0 is on CPU, expected npu:0", result["errors"][0]["message"])

    def test_single_npu_preflight_reports_model_init_guidance_for_checkpoint_device_errors(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            fake_npus = [{"id": "npu:0", "name": "Ascend 0", "index": 0}]

            with (
                patch("vfieval.file_inputs.list_npu_devices", return_value=fake_npus),
                patch("vfieval.file_inputs.resolve_torch_device", return_value=torch.device("cpu")),
                patch(
                    "vfieval.file_inputs.load_flow_mask_model",
                    side_effect=RuntimeError("Expected all tensors to be on the same device"),
                ),
            ):
                result = preflight_run(
                    db,
                    workspace,
                    {
                        "model_file": "test_checkpoint.py",
                        "checkpoint": "auto",
                        "video_group": "test_style",
                        "selected_videos": ["blocks_motion.avi"],
                        "device": "npu:0",
                        "precision": "fp32",
                        "max_frames": 4,
                    },
                )

            self.assertFalse(result["ok"])
            self.assertEqual(result["model"]["tested_devices"], [])
            self.assertIn("model dry-run failed on npu:0", result["errors"][0]["message"])
            self.assertIn("model_init/checkpoint_load", result["errors"][0]["message"])
            self.assertIn("Expected all tensors to be on the same device", result["errors"][0]["message"])
            self.assertIn("map_location='cpu'", result["errors"][0]["message"])
            self.assertIn("model.to(device)", result["errors"][0]["message"])
            self.assertIn("Model(..., device=...)", result["errors"][0]["message"])

    def test_api_multi_npu_run_queues_decode_before_shard_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(db, workspace))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                fake_npus = [{"id": f"npu:{index}", "name": f"Ascend {index}", "index": index} for index in range(2)]
                with (
                    patch("vfieval.file_inputs.list_npu_devices", return_value=fake_npus),
                    patch("vfieval.file_inputs._dry_run_model_file", return_value=None),
                    patch("vfieval.server.start_decode_worker") as start_decode_worker,
                ):
                    created = _post(
                        base_url,
                        "/api/runs",
                        {
                            "model_file": "test_average.py",
                            "video_group": "test_style",
                            "selected_videos": ["blocks_motion.avi", "gradient_motion.avi"],
                            "execution_mode": "multi_npu",
                            "devices": ["npu:0", "npu:1"],
                            "precision": "fp32",
                            "batch_size_per_device": 1,
                            "metrics": ["lpips_vit_patch"],
                        },
                    )

                run = _get(base_url, f"/api/runs/{created['run_id']}")
                self.assertEqual(run["metadata"]["execution_mode"], "multi_npu")
                self.assertEqual(run["metadata"]["devices"], ["npu:0", "npu:1"])
                self.assertEqual(run["status"], "decoding")
                start_decode_worker.assert_called_once()
                decode_jobs = [job for job in run["jobs"] if job["role"] == "decode"]
                self.assertEqual(len(decode_jobs), 1)
                self.assertEqual(decode_jobs[0]["status"], "queued")
                self.assertGreater(int(decode_jobs[0]["progress_total"] or 0), 0)
                inference_jobs = [job for job in run["jobs"] if job["role"] == "inference"]
                self.assertEqual(inference_jobs, [])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_npu_worker_claims_only_matching_device_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("frames", str(Path(tmp)), True)
            frame = Path(tmp) / "frame.png"
            frame.write_bytes(b"not-used")
            db.add_sample(dataset_id, "sample-a", str(frame), str(frame), str(frame), {"video_file": "a.mp4"})
            db.add_sample(dataset_id, "sample-b", str(frame), str(frame), str(frame), {"video_file": "b.mp4"})
            run_id = db.create_run(
                name="multi-npu",
                model_id=model_id,
                dataset_id=dataset_id,
                height=8,
                width=8,
                batch_size=1,
                device="multi_npu",
                precision="fp32",
                metrics=[],
                create_inference_job=False,
            )
            job0 = db.add_run_job(run_id, "inference", {"device": "npu:0"}, progress_total=1, shard_index=0, device="npu:0")
            job1 = db.add_run_job(run_id, "inference", {"device": "npu:1"}, progress_total=1, shard_index=1, device="npu:1")
            metric_job = db.add_run_job(run_id, "metric", {"metric_names": ["lpips_vit_patch"]}, progress_total=0)

            self.assertEqual(db.claim_next_job("worker-npu-1", ["inference"], device_filter="npu:1")["id"], job1)
            self.assertEqual(db.claim_next_job("worker-npu-0", ["inference"], device_filter="npu:0")["id"], job0)
            self.assertIsNone(db.claim_next_job("worker-npu-0-again", ["inference"], device_filter="npu:0"))
            self.assertEqual(db.claim_next_job("metric-worker", ["metric"])["id"], metric_job)

    def test_multi_npu_worker_failure_preserves_shard_context_for_run_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("frames", str(Path(tmp)), True)
            frame = Path(tmp) / "frame.png"
            frame.write_bytes(b"not-used")
            db.add_sample(dataset_id, "sample-a", str(frame), str(frame), str(frame), {"video_file": "a.mp4"})
            db.add_sample(dataset_id, "sample-b", str(frame), str(frame), str(frame), {"video_file": "b.mp4"})
            run_id = db.create_run(
                name="multi-npu-failure",
                model_id=model_id,
                dataset_id=dataset_id,
                height=8,
                width=8,
                batch_size=1,
                device="multi_npu",
                precision="fp32",
                metrics=[],
                create_inference_job=False,
            )
            job0 = db.add_run_job(
                run_id,
                "inference",
                {"run_id": run_id, "device": "npu:0", "shard_index": 0, "shard_count": 2},
                progress_total=1,
                shard_index=0,
                device="npu:0",
            )
            job1 = db.add_run_job(
                run_id,
                "inference",
                {"run_id": run_id, "device": "npu:1", "shard_index": 1, "shard_count": 2},
                progress_total=1,
                shard_index=1,
                device="npu:1",
            )
            capabilities = {
                "hostname": "test-host",
                "platform": "test-platform",
                "python": "3.11",
                "pid": 123,
                "cuda": [],
                "npu": [{"id": "npu:0"}, {"id": "npu:1"}],
                "errors": {"npu": None},
                "cpu": True,
                "precision_support": {"cpu": ["fp32"], "cuda": [], "npu": ["fp32"]},
                "metric_support": [],
                "decode_backends": {"opencv": False, "ffmpeg": False},
            }

            with (
                patch("vfieval.worker.prepare_worker_device"),
                patch("vfieval.worker.detect_capabilities", return_value=capabilities),
                patch("vfieval.worker.run_inference_job", side_effect=RuntimeError("synthetic shard failure")),
            ):
                run_worker(
                    db,
                    workspace,
                    WorkerOptions(role="inference", once=True, worker_id="worker-npu-1", device_filter="npu:1"),
                )

            expected_message = describe_job_failure(
                {"kind": "inference", "payload": {"device": "npu:1", "shard_index": 1, "shard_count": 2}},
                "synthetic shard failure",
            )
            failed_job = db.get_job(job1)
            self.assertEqual(failed_job["status"], "failed")
            self.assertEqual(failed_job["error"]["message"], expected_message)
            self.assertEqual(failed_job["error"]["device"], "npu:1")
            self.assertEqual(failed_job["error"]["shard_index"], 1)
            self.assertEqual(failed_job["error"]["shard_count"], 2)
            self.assertEqual(failed_job["error"]["worker_id"], "worker-npu-1")
            self.assertEqual(failed_job["error"]["job_id"], job1)

            failed_run = db.get_run(run_id)
            self.assertEqual(failed_run["status"], "failed")
            self.assertEqual(failed_run["error"]["message"], expected_message)
            self.assertEqual(failed_run["error"]["device"], "npu:1")
            self.assertEqual(failed_run["error"]["shard_index"], 1)
            self.assertEqual(failed_run["error"]["worker_id"], "worker-npu-1")

            db.complete_job(job0, {"samples": 1})
            db.maybe_complete_multi_run_inference(run_id)
            failed_run = db.get_run(run_id)
            self.assertEqual(failed_run["status"], "failed")
            self.assertEqual(failed_run["error"]["message"], expected_message)
            self.assertEqual(failed_run["error"]["device"], "npu:1")
            self.assertEqual(failed_run["error"]["shard_index"], 1)
            self.assertEqual(failed_run["error"]["shard_count"], 2)
            self.assertEqual(failed_run["error"]["worker_id"], "worker-npu-1")
            self.assertEqual(failed_run["error"]["job_id"], job1)

    def test_multi_shard_completion_combines_output_health(self) -> None:
        def health(samples: int, flow_abs_max: float, mask_std: float) -> dict[str, object]:
            return {
                "samples": samples,
                "stats": {
                    "flowt_0": {"abs_mean": flow_abs_max / 2, "abs_max": flow_abs_max, "nan_count": 0},
                    "flowt_1": {"abs_mean": flow_abs_max / 2, "abs_max": flow_abs_max, "nan_count": 0},
                    "mask0": {"mean": 0.5, "std": mask_std, "nan_count": 0},
                    "mask1": {"mean": 0.5, "std": mask_std, "nan_count": 0},
                },
                "warnings": ["flow ~= 0"] if flow_abs_max == 0 else [],
                "flow_flat": flow_abs_max < 1e-4,
                "mask_flat": mask_std < 1e-3,
                "has_nan": False,
            }

        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("frames", str(Path(tmp)), True)
            run_id = db.create_run(
                name="multi-shard-output-health",
                model_id=model_id,
                dataset_id=dataset_id,
                height=8,
                width=8,
                batch_size=1,
                device="multi_cuda",
                precision="fp32",
                metrics=[],
                create_inference_job=False,
            )
            job0 = db.add_run_job(run_id, "inference", {"run_id": run_id}, progress_total=1, shard_index=0, device="cuda:0")
            job1 = db.add_run_job(run_id, "inference", {"run_id": run_id}, progress_total=1, shard_index=1, device="cuda:1")

            db.complete_job(job0, {"samples": 1, "output_health": health(1, 0.0, 0.0)})
            db.complete_job(job1, {"samples": 2, "output_health": health(2, 2.0, 0.25)})

            self.assertTrue(db.maybe_complete_multi_run_inference(run_id))
            run = db.get_run(run_id)
            output_health = run["result"]["output_health"]
            self.assertEqual(output_health["samples"], 3)
            self.assertEqual(output_health["shards"], 2)
            self.assertFalse(output_health["flow_flat"])
            self.assertFalse(output_health["mask_flat"])
            self.assertEqual(output_health["stats"]["flowt_0"]["abs_max"], 2.0)
            self.assertEqual(output_health["stats"]["mask0"]["std"], 0.25)

    def test_fail_run_cancels_queued_sibling_shards_and_stops_running_shard(self) -> None:
        from vfieval.pipeline.inference import RunCanceled, _raise_if_canceled

        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("dummy", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("frames", str(Path(tmp)), True)
            run_id = db.create_run(
                name="multi-shard-fail-cancel",
                model_id=model_id,
                dataset_id=dataset_id,
                height=8,
                width=8,
                batch_size=1,
                device="multi_cuda",
                precision="fp32",
                metrics=[],
                create_inference_job=False,
            )
            failing_job = db.add_run_job(
                run_id,
                "inference",
                {"run_id": run_id, "device": "cuda:0", "shard_index": 0, "shard_count": 2},
                progress_total=1,
                shard_index=0,
                device="cuda:0",
            )
            queued_sibling_job = db.add_run_job(
                run_id,
                "inference",
                {"run_id": run_id, "device": "cuda:1", "shard_index": 1, "shard_count": 2},
                progress_total=1,
                shard_index=1,
                device="cuda:1",
            )

            db.fail_run(run_id, {"message": "synthetic shard failure", "device": "cuda:0"})

            # A sibling shard that hadn't been claimed yet must be canceled so
            # no worker starts it after the run is already terminal.
            queued_job_row = db.get_job(queued_sibling_job)
            self.assertEqual(queued_job_row["status"], "canceled")
            self.assertEqual(queued_job_row["error"]["message"], "sibling shard failed the run")

            # A sibling shard that was already running when the failure landed
            # must notice on its next cancellation check and stop itself.
            running_sibling_job = db.add_run_job(
                run_id,
                "inference",
                {"run_id": run_id, "device": "cuda:2", "shard_index": 2, "shard_count": 3},
                progress_total=1,
                shard_index=2,
                device="cuda:2",
            )
            with self.assertRaises(RunCanceled):
                _raise_if_canceled(db, run_id, running_sibling_job)
            running_job_row = db.get_job(running_sibling_job)
            self.assertEqual(running_job_row["status"], "canceled")
            self.assertEqual(running_job_row["error"]["message"], "sibling shard failed the run")

    def test_set_npu_device_prefers_integer_index(self) -> None:
        import torch

        class FakeNpu:
            def __init__(self) -> None:
                self.calls: list[int | str] = []

            def set_device(self, value) -> None:
                self.calls.append(value)

        fake_npu = FakeNpu()
        with (
            patch("vfieval.devices.npu_module", return_value=object()),
            patch("vfieval.devices.npu_is_available", return_value=True),
            patch("vfieval.devices.list_npu_devices", return_value=[{"id": "npu:3", "name": "Ascend 3", "index": 3}]),
            patch.object(torch, "npu", fake_npu, create=True),
        ):
            set_npu_device("npu:3")
        self.assertEqual(fake_npu.calls, [3])

    def test_run_inference_job_sets_npu_device_before_model_load(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            for folder in ("img0", "img1", "gt"):
                (dataset_root / folder).mkdir(parents=True)
            Image.new("RGB", (8, 8), (0, 0, 0)).save(dataset_root / "img0" / "sample000.png")
            Image.new("RGB", (8, 8), (255, 255, 255)).save(dataset_root / "img1" / "sample000.png")
            Image.new("RGB", (8, 8), (128, 128, 128)).save(dataset_root / "gt" / "sample000.png")

            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("npu-order-model", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("npu-order-dataset", str(dataset_root), has_gt=True)
            self.assertEqual(scan_triplet_dataset(db, dataset_id), 1)
            run_id = db.create_run(
                name="npu-order-run",
                model_id=model_id,
                dataset_id=dataset_id,
                height=8,
                width=8,
                batch_size=1,
                device="npu:0",
                precision="fp32",
                metrics=[],
            )
            job_id = int(db.get_run(run_id)["inference_job_id"])
            events: list[str] = []

            class FakeModel:
                def predict(self, img0, img1, _t):
                    batch, _channels, height, width = img0.shape
                    flow = torch.zeros((batch, 2, height, width), dtype=img0.dtype, device=img0.device)
                    mask = torch.zeros((batch, 1, height, width), dtype=img0.dtype, device=img0.device)
                    return {
                        "flowt_0": flow,
                        "flowt_1": flow,
                        "mask0": mask,
                        "mask1": mask,
                    }

            real_torch_device = torch.device

            def fake_set_npu_device(_device_name) -> None:
                events.append("set_device")

            def fake_load_model(*args, **kwargs):
                events.append("load_model")
                return FakeModel()

            with (
                patch("vfieval.devices.set_npu_device", side_effect=fake_set_npu_device),
                patch("vfieval.devices.torch.device", side_effect=lambda _name: real_torch_device("cpu")),
                patch("vfieval.pipeline.inference.load_flow_mask_model", side_effect=fake_load_model),
            ):
                result = run_inference_job(db, workspace, job_id)

            self.assertEqual(result.samples, 1)
            self.assertGreaterEqual(len(events), 2)
            self.assertEqual(events[:2], ["set_device", "load_model"])

    def test_metric_assets_default_to_project_set_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            with patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": str(Path(tmp)), "VFIEVAL_METRIC_ASSETS_DIR": ""}, clear=False):
                asset_root = metric_assets_dir(workspace)
                health = metrics_health(workspace)
            self.assertEqual(asset_root, Path(tmp) / "set" / "metrics")
            self.assertEqual(health["asset_root"], str(Path(tmp) / "set" / "metrics"))
            self.assertEqual(health["metrics"]["lpips_vit_patch"]["status"], "missing_weights")
            self.assertFalse(health["metrics"]["lpips_vit_patch"]["available"])
            self.assertTrue(health["metrics"]["lpips_vit_patch"]["weights_path"].endswith("lpips_vit_patch\\dinov2_vits14_reg.pth"))
            self.assertTrue(health["metrics"]["lpips_vit_patch"]["manifest_path"].endswith("lpips_vit_patch\\manifest.json"))

    def test_metric_health_uses_user_facing_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            with patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": str(Path(tmp)), "VFIEVAL_METRIC_ASSETS_DIR": ""}, clear=False):
                lpips = metric_health(workspace, "lpips_vit_patch")
                with patch("vfieval.metrics.health.shutil.which", return_value=None):
                    vmaf = metric_health(workspace, "vmaf")
                cgvqm = metric_health(workspace, "cgvqm")
        self.assertEqual(lpips["status"], "missing_weights")
        self.assertTrue(lpips["weights_path"].endswith("lpips_vit_patch\\dinov2_vits14_reg.pth"))
        self.assertEqual(lpips["input_mode"], "sample_pair")
        self.assertEqual(lpips["implementation_mode"], "dinov2_feature_distance")
        self.assertIn("DINOv2", lpips["setup_summary"])
        self.assertEqual(lpips["device_policy"], "require_run_device")
        self.assertEqual(vmaf["status"], "missing_evaluator")
        self.assertFalse(vmaf["available"])
        self.assertEqual(vmaf["input_mode"], "video_only")
        self.assertIn("libvmaf", vmaf["setup_summary"])
        self.assertTrue(any(item["kind"] == "ffmpeg_filter" for item in vmaf["setup_requirements"]))
        self.assertEqual(vmaf["implementation_mode"], "ffmpeg_libvmaf")
        self.assertEqual(cgvqm["input_mode"], "video_only")

    def test_metric_health_reports_missing_evaluator_when_driver_command_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            project_root = Path(tmp)
            manifest_path = project_root / "set" / "metrics" / "cgvqm" / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            (manifest_path.parent / "weights").mkdir()
            (manifest_path.parent / "CGVQM").mkdir()
            manifest_path.write_text(
                json.dumps(
                    {
                        "metric_name": "cgvqm",
                        "asset_version": "v2",
                        "implementation_mode": "cgvqm_wrapper",
                        "repo_dir": "CGVQM",
                        "weights_path": "weights",
                        "device_policy": "require_run_device",
                        "driver": {"command": ["missing-driver.exe"]},
                        "env": {},
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": str(project_root), "VFIEVAL_METRIC_ASSETS_DIR": ""}, clear=False):
                with patch("vfieval.metrics.health.importlib.util.find_spec", return_value=object()):
                    cgvqm = metric_health(workspace, "cgvqm")
            self.assertEqual(cgvqm["status"], "missing_evaluator")
            self.assertFalse(cgvqm["available"])
            self.assertIn("driver executable", cgvqm["reason"])

    def test_worker_process_command_includes_device_filter_and_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            command = _worker_process_command(
                workspace,
                role="inference",
                device_filter="npu:0",
                worker_id="worker-npu-0",
                once=True,
                idle_timeout=120.0,
            )
        self.assertIn("--device-filter", command)
        self.assertIn("npu:0", command)
        self.assertIn("--idle-timeout", command)
        self.assertIn("120.0", command)
        self.assertIn("--once", command)


def _post(base_url: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _delete(base_url: str, path: str) -> dict:
    request = urllib.request.Request(f"{base_url}{path}", method="DELETE")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_run(base_url: str, run_id: int) -> dict:
    deadline = time.time() + 30
    while time.time() < deadline:
        run = _get(base_url, f"/api/runs/{run_id}")
        if run["status"] in {"completed", "failed", "canceled"}:
            return run
        time.sleep(0.25)
    raise AssertionError(f"run {run_id} did not finish")


if __name__ == "__main__":
    unittest.main()
