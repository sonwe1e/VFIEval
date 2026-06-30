from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.file_inputs import decode_cache_dir, list_checkpoints, list_model_files, list_video_group_videos, list_video_groups, preflight_run
from vfieval.models import load_flow_mask_model
from vfieval.pipeline.postprocess import validate_model_outputs
from vfieval.server import _make_handler, _partition_samples_by_video


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
                artifacts = _get(base_url, f"/api/runs/{run_id}/artifacts")
                kinds = [artifact["kind"] for artifact in artifacts]
                self.assertIn("pred_video", kinds)
                self.assertIn("gt_video", kinds)
                self.assertIn("diff_video", kinds)
                self.assertEqual(kinds.count("pred_video"), 1)
                self.assertTrue(Path(run["metadata"]["output_dir"]).exists())
                self.assertEqual(run["metadata"]["selected_videos"], [selected_video])
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
                compare = _get(
                    base_url,
                    f"/api/compare/samples?run_id={run_id}&video_name={urllib.parse.quote(run_video_name)}&frame_index={first_sample['frame_index']}",
                )
                self.assertTrue(compare["compatible"])
                self.assertIsNotNone(compare["samples"][0]["sample"])

                health = _get(base_url, "/api/metrics/health")
                self.assertIn("lpips_vit_patch", health["metrics"])
                worker = _post(
                    base_url,
                    "/api/workers/register",
                    {"worker_id": "remote-test", "role": "inference", "capabilities": {"cpu": True}},
                )
                self.assertEqual(worker["worker"]["id"], "remote-test")
                heartbeat = _post(base_url, f"/api/jobs/{int(run['inference_job_id'])}/heartbeat", {"worker_id": "remote-test"})
                self.assertEqual(heartbeat["status"], "heartbeat")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_api_file_run_with_metric_records_unavailable_summary(self) -> None:
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

                summary = _get(base_url, f"/api/runs/{run_id}/metric-summary")
                self.assertEqual(summary["metrics"]["lpips_vit_patch"]["unavailable"], 2)
                self.assertIn("native adapter is missing_assets", summary["metrics"]["lpips_vit_patch"]["reasons"][0])

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
