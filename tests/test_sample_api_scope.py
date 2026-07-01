from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v13_test_utils import get_json, make_workspace, start_server, stop_server


class SampleApiScopeTests(unittest.TestCase):
    def test_sample_and_video_timeline_endpoints_do_not_materialize_full_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            model_id = db.upsert_model("scope-model", "dummy", None, 8, 8, {})
            dataset_root = Path(tmp) / "dataset"
            for folder in ("img0", "img1", "gt"):
                (dataset_root / folder).mkdir(parents=True, exist_ok=True)
            for index in range(2):
                for folder, color in (("img0", (0, 0, 0)), ("img1", (20, 0, 0)), ("gt", (10, 0, 0))):
                    Image.new("RGB", (8, 8), color).save(dataset_root / folder / f"{index:06d}.png")
            dataset_id = db.create_dataset("scope-dataset", str(dataset_root), True)
            sample_ids = []
            for index in range(2):
                sample_ids.append(
                    db.add_sample(
                        dataset_id,
                        f"clip_{index:06d}",
                        str(dataset_root / "img0" / f"{index:06d}.png"),
                        str(dataset_root / "img1" / f"{index:06d}.png"),
                        str(dataset_root / "gt" / f"{index:06d}.png"),
                        {"video_name": "clip", "video_file": "clip.mp4", "frame_index": index, "sample_index": index, "fps": 5.0},
                    )
                )
            run_id = db.create_run("scope-run", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", [])
            job_id = int(db.get_run(run_id)["inference_job_id"])
            pred_path = dataset_root / "pred.png"
            Image.new("RGB", (8, 8), (20, 0, 0)).save(pred_path)
            for sample_id in sample_ids:
                db.add_artifact(job_id, sample_id, "pred", str(pred_path), "image/png", {"sample": f"sample-{sample_id}"})
            db.complete_run_inference(run_id, {"output_dir": str(workspace.runs_dir / str(run_id))}, db.summarize_artifacts(job_id), "completed")

            server, thread, base_url = start_server(db, workspace)
            try:
                with patch("vfieval.server._run_timeline", side_effect=AssertionError("full timeline should not load")):
                    sample_payload = get_json(base_url, f"/api/runs/{run_id}/samples/{sample_ids[0]}")
                    video_payload = get_json(base_url, f"/api/runs/{run_id}/videos/clip/timeline")
                self.assertIn("pred", sample_payload["artifacts"])
                self.assertEqual(video_payload["video_name"], "clip")
                self.assertEqual(video_payload["sample_count"], 2)
            finally:
                stop_server(server, thread)

    def test_video_timeline_uses_batched_sample_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            model_id = db.upsert_model("scope-model", "dummy", None, 8, 8, {})
            dataset_root = Path(tmp) / "dataset"
            for folder in ("img0", "img1", "gt"):
                (dataset_root / folder).mkdir(parents=True, exist_ok=True)
            for index in range(20):
                for folder, color in (("img0", (0, 0, 0)), ("img1", (20, 0, 0)), ("gt", (10, 0, 0))):
                    Image.new("RGB", (8, 8), color).save(dataset_root / folder / f"{index:06d}.png")
            dataset_id = db.create_dataset("scope-dataset", str(dataset_root), True)
            sample_ids = []
            for index in range(20):
                sample_ids.append(
                    db.add_sample(
                        dataset_id,
                        f"clip_{index:06d}",
                        str(dataset_root / "img0" / f"{index:06d}.png"),
                        str(dataset_root / "img1" / f"{index:06d}.png"),
                        str(dataset_root / "gt" / f"{index:06d}.png"),
                        {"video_name": "clip", "video_file": "clip.mp4", "frame_index": index, "sample_index": index, "fps": 5.0},
                    )
                )
            run_id = db.create_run("scope-run", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", ["lpips_vit_patch"])
            inference_job_id = int(db.get_run(run_id)["inference_job_id"])
            metric_job_id = db.create_job("metric", {"run_id": run_id, "inference_job_id": inference_job_id, "dataset_id": dataset_id})
            pred_path = dataset_root / "pred.png"
            Image.new("RGB", (8, 8), (20, 0, 0)).save(pred_path)
            for index, sample_id in enumerate(sample_ids):
                db.add_artifact(inference_job_id, sample_id, "pred", str(pred_path), "image/png", {"sample": f"sample-{sample_id}"})
                db.add_metric_result(metric_job_id, inference_job_id, sample_id, "lpips_vit_patch", "completed", float(index), {})
            db.complete_run_inference(run_id, {"output_dir": str(workspace.runs_dir / str(run_id))}, db.summarize_artifacts(inference_job_id), "completed")

            queries: list[str] = []
            original_connect = db.connect

            def traced_connect():
                conn = original_connect()
                conn.set_trace_callback(queries.append)
                return conn

            server, thread, base_url = start_server(db, workspace)
            try:
                with patch.object(db, "connect", side_effect=traced_connect):
                    payload = get_json(base_url, f"/api/runs/{run_id}/videos/clip/timeline?window_size=5")
                select_count = sum(1 for query in queries if query.lstrip().upper().startswith("SELECT"))
                per_sample_artifact_queries = [
                    query for query in queries
                    if "FROM artifacts" in query and "sample_id =" in query and "sample_id IN" not in query
                ]
                artifact_window_queries = [
                    query for query in queries
                    if "FROM artifacts" in query and "sample_id IN" in query
                ]
                self.assertEqual(len(payload["samples"]), 5)
                self.assertLessEqual(select_count, 16)
                self.assertEqual(per_sample_artifact_queries, [])
                self.assertEqual(len(artifact_window_queries), 1)
                artifact_query = artifact_window_queries[0]
                for sample_id in sample_ids[:5]:
                    self.assertRegex(artifact_query, rf"(?<!\d){sample_id}(?!\d)")
                for sample_id in sample_ids[5:]:
                    self.assertNotRegex(artifact_query, rf"(?<!\d){sample_id}(?!\d)")
            finally:
                stop_server(server, thread)


if __name__ == "__main__":
    unittest.main()
