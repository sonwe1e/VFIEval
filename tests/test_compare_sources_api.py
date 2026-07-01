from __future__ import annotations

import os
import sys
import tempfile
import urllib.error
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v13_test_utils import add_completed_pred_run, get_json, make_workspace, start_server, stop_server, write_mp4


class CompareSourcesApiTests(unittest.TestCase):
    def test_video_groups_summary_skips_video_summaries_and_thumbnails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            gt_path = Path(tmp) / "videos" / "anime" / "clip.mp4"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            write_mp4(gt_path, [(0, 0, 0), (20, 0, 0), (40, 0, 0)])

            server, thread, base_url = start_server(db, workspace)
            try:
                with patch("vfieval.file_inputs.ensure_video_thumbnail", side_effect=AssertionError("summary should not build thumbnails")):
                    payload = get_json(base_url, "/api/video-groups?summary=1")
                self.assertEqual(payload[0]["name"], "anime")
                self.assertEqual(payload[0]["video_count"], 1)
                self.assertNotIn("videos", payload[0])
            finally:
                stop_server(server, thread)

    def test_compare_sources_are_server_resident_and_reject_path_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            gt_path = Path(tmp) / "videos" / "anime" / "clip.mp4"
            pred_path = workspace.root / "pred-a.mp4"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            write_mp4(gt_path, [(0, 0, 0), (20, 0, 0), (40, 0, 0)])
            write_mp4(pred_path, [(0, 0, 0), (0, 20, 0), (0, 40, 0)])
            run_id = add_completed_pred_run(db, workspace, "ModelA", pred_path)

            server, thread, base_url = start_server(db, workspace)
            try:
                gt_sources = get_json(base_url, "/api/compare-sources/gt")["sources"]
                self.assertEqual(gt_sources[0]["group"], "anime")
                self.assertEqual(gt_sources[0]["video"], "clip.mp4")

                pred_sources = get_json(base_url, "/api/compare-sources/pred")["sources"]
                self.assertEqual(pred_sources[0]["run_id"], run_id)
                self.assertEqual(pred_sources[0]["video"], "clip")
                self.assertGreater(pred_sources[0]["artifact_id"], 0)

                flow_sources = get_json(base_url, f"/api/compare-sources/flow?run_id={run_id}&video=clip")["sources"]
                mask_sources = get_json(base_url, f"/api/compare-sources/mask?run_id={run_id}&video=clip")["sources"]
                self.assertIn("flowt_0", {row["kind"] for row in flow_sources})
                self.assertIn("mask0", {row["kind"] for row in mask_sources})

                with self.assertRaises(urllib.error.HTTPError) as raised:
                    get_json(base_url, "/api/compare-sources/pred?path=C:/unsafe.mp4")
                self.assertEqual(raised.exception.code, 400)
            finally:
                stop_server(server, thread)

    def test_compare_sources_support_pagination_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            gt_dir = Path(tmp) / "videos" / "anime"
            gt_dir.mkdir(parents=True, exist_ok=True)
            write_mp4(gt_dir / "clip-a.mp4", [(0, 0, 0), (20, 0, 0), (40, 0, 0)])
            write_mp4(gt_dir / "clip-b.mp4", [(0, 0, 0), (0, 20, 0), (0, 40, 0)])
            pred_a = workspace.root / "pred-a.mp4"
            pred_b = workspace.root / "pred-b.mp4"
            write_mp4(pred_a, [(0, 0, 0), (20, 0, 0), (40, 0, 0)])
            write_mp4(pred_b, [(0, 0, 0), (0, 20, 0), (0, 40, 0)])
            add_completed_pred_run(db, workspace, "ModelA", pred_a, video_name="clip-a")
            run_b = add_completed_pred_run(db, workspace, "ModelB", pred_b, video_name="clip-b")

            server, thread, base_url = start_server(db, workspace)
            try:
                gt_payload = get_json(base_url, "/api/compare-sources/gt?group=anime&q=clip-b&page=1&page_size=1")
                self.assertEqual(gt_payload["filtered_count"], 1)
                self.assertEqual(gt_payload["sources"][0]["video"], "clip-b.mp4")

                pred_payload = get_json(base_url, "/api/compare-sources/pred?q=ModelB&video=clip-b&page=1&page_size=1")
                self.assertEqual(pred_payload["filtered_count"], 1)
                self.assertEqual(pred_payload["sources"][0]["run_id"], run_b)
                self.assertEqual(pred_payload["sources"][0]["video"], "clip-b")
            finally:
                stop_server(server, thread)


if __name__ == "__main__":
    unittest.main()
