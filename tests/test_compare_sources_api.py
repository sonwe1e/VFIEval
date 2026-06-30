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


if __name__ == "__main__":
    unittest.main()
