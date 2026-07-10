from __future__ import annotations

import tempfile
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vfieval.evaluations import (
    add_candidate,
    campaign_analysis,
    campaign_export,
    campaign_export_csv,
    close_campaign,
    create_campaign,
    next_task,
    presentation_for,
    publish_campaign,
    submit_vote,
    upsert_evaluator,
)
from vfieval.media_assets import create_collection, get_asset, soft_delete_asset, upsert_asset

from v13_test_utils import make_workspace, write_mp4


class EvaluationCampaignTests(unittest.TestCase):
    def _asset(self, db, collection_id: int, path: Path, role: str, name: str, model: str = "", checkpoint: str = ""):
        return upsert_asset(
            db,
            collection_id=collection_id,
            source_key=f"upload:{name}",
            source_kind="upload",
            media_kind="video",
            role=role,
            display_name=name,
            original_name=path.name,
            storage_path=path,
            size_bytes=path.stat().st_size,
            frame_count=3,
            width=8,
            height=8,
            fps=5,
            provenance={"external": True, "model_name": model, "checkpoint": checkpoint},
        )

    def test_blind_task_hides_identity_balances_sides_and_ranks_votes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection = create_collection(db, "Blind set")
            root = workspace.media_dir / "blind" / "assets"
            root.mkdir(parents=True)
            gt_path = write_mp4(root / "gt.mp4", [(0, 0, 0), (20, 0, 0), (40, 0, 0)])
            a_path = write_mp4(root / "a.mp4", [(0, 2, 0), (20, 2, 0), (40, 2, 0)])
            b_path = write_mp4(root / "b.mp4", [(0, 0, 2), (20, 0, 2), (40, 0, 2)])
            gt = self._asset(db, collection["id"], gt_path, "gt", "GT")
            pred_a = self._asset(db, collection["id"], a_path, "pred", "Candidate A", "RIFE", "a.ckpt")
            pred_b = self._asset(db, collection["id"], b_path, "pred", "Candidate B", "EMA", "b.ckpt")

            campaign = create_campaign(db, {"name": "Blind", "target_votes": 2, "seed": 17})
            add_candidate(
                db,
                workspace,
                campaign["id"],
                {"reference_asset_id": gt["id"], "asset_id": pred_a["id"], "video_name": "clip"},
            )
            add_candidate(
                db,
                workspace,
                campaign["id"],
                {"reference_asset_id": gt["id"], "asset_id": pred_b["id"], "video_name": "clip"},
            )
            published = publish_campaign(db, workspace, campaign["id"])
            self.assertEqual(published["tasks"], 1)

            for evaluator_id in ("browser-1", "browser-2"):
                upsert_evaluator(db, {"evaluator_id": evaluator_id, "display_name": evaluator_id})
                task_payload = next_task(db, campaign["id"], evaluator_id)
                task = task_payload["task"]
                self.assertNotIn("asset_id", task)
                self.assertNotIn("model", task)
                self.assertNotIn("checkpoint", task)
                self.assertIn("/media/left", task["left_url"])
                stable = presentation_for(db, task["id"], evaluator_id)
                self.assertEqual(stable, presentation_for(db, task["id"], evaluator_id))
                choice = "left" if stable["left_asset_id"] == pred_a["id"] else "right"
                vote = submit_vote(
                    db,
                    task["id"],
                    evaluator_id,
                    {
                        "choice": choice,
                        "reasons": ["temporal_stability"],
                        "confidence": "high",
                        "note": "clear preference",
                    },
                )
                self.assertEqual(vote["preferred_asset_id"], pred_a["id"])

            self.assertTrue(next_task(db, campaign["id"], "browser-1")["complete"])
            analysis = campaign_analysis(db, campaign["id"], bootstrap_samples=100)
            self.assertFalse(analysis["coverage"]["provisional"])
            self.assertEqual(analysis["human"]["vote_count"], 2)
            self.assertEqual(analysis["human"]["ranking"][0]["asset_id"], pred_a["id"])
            self.assertEqual(analysis["quality_reasons"]["temporal_stability"], 2)
            exported = campaign_export(db, campaign["id"])
            self.assertEqual(len(exported["votes"]), 2)
            self.assertIn(b"row_type", campaign_export_csv(exported))

            filtered = campaign_analysis(
                db,
                campaign["id"],
                bootstrap_samples=0,
                filters={"video": "clip", "evaluator_id": "browser-1", "model": "RIFE"},
            )
            self.assertEqual(filtered["human"]["vote_count"], 1)
            self.assertEqual(filtered["filters"]["model"], "RIFE")

            closed = close_campaign(db, campaign["id"])
            self.assertEqual(closed["status"], "closed")
            with self.assertRaisesRegex(ValueError, "not published"):
                next_task(db, campaign["id"], "browser-1")

    def test_vote_can_be_modified_without_duplicate_row_and_protected_asset_bytes_remain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection = create_collection(db, "Blind set")
            root = workspace.media_dir / "blind" / "protected"
            root.mkdir(parents=True)
            gt = self._asset(db, collection["id"], write_mp4(root / "gt.mp4", [(0, 0, 0)] * 3), "gt", "GT")
            a = self._asset(db, collection["id"], write_mp4(root / "a.mp4", [(1, 0, 0)] * 3), "pred", "A")
            b = self._asset(db, collection["id"], write_mp4(root / "b.mp4", [(0, 1, 0)] * 3), "pred", "B")
            campaign = create_campaign(db, {"name": "Modify", "target_votes": 1, "seed": 1})
            for asset in (a, b):
                add_candidate(
                    db,
                    workspace,
                    campaign["id"],
                    {"reference_asset_id": gt["id"], "asset_id": asset["id"], "video_name": "clip"},
                )
            publish_campaign(db, workspace, campaign["id"])
            upsert_evaluator(db, {"evaluator_id": "browser", "display_name": "Evaluator"})
            task = next_task(db, campaign["id"], "browser")["task"]
            first = submit_vote(db, task["id"], "browser", {"choice": "tie"})
            second = submit_vote(db, task["id"], "browser", {"choice": "left", "confidence": "medium"})
            self.assertEqual(first["id"], second["id"])
            count = db.get("SELECT COUNT(*) AS count FROM evaluation_votes WHERE task_id = ?", (task["id"],))
            self.assertEqual(count["count"], 1)

            deleted = soft_delete_asset(db, workspace, a["id"])
            self.assertTrue(deleted["protected_by_evaluation"])
            self.assertFalse(deleted["content_removed"])
            self.assertTrue(Path(a["storage_path"]).exists())
            self.assertEqual(get_asset(db, a["id"], include_deleted=True)["state"], "unavailable")

    def test_twenty_lan_evaluators_can_vote_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection = create_collection(db, "Concurrent")
            root = workspace.media_dir / "concurrent" / "assets"
            root.mkdir(parents=True)
            gt = self._asset(db, collection["id"], write_mp4(root / "gt.mp4", [(0, 0, 0)] * 3), "gt", "GT")
            a = self._asset(db, collection["id"], write_mp4(root / "a.mp4", [(1, 0, 0)] * 3), "pred", "A")
            b = self._asset(db, collection["id"], write_mp4(root / "b.mp4", [(0, 1, 0)] * 3), "pred", "B")
            campaign = create_campaign(db, {"name": "LAN", "target_votes": 20, "seed": 7})
            for asset in (a, b):
                add_candidate(
                    db,
                    workspace,
                    campaign["id"],
                    {"reference_asset_id": gt["id"], "asset_id": asset["id"], "video_name": "clip"},
                )
            publish_campaign(db, workspace, campaign["id"])
            task_id = int(db.get("SELECT id FROM evaluation_tasks WHERE campaign_id = ?", (campaign["id"],))["id"])
            evaluator_ids = [f"browser-{index}" for index in range(20)]
            for evaluator_id in evaluator_ids:
                upsert_evaluator(db, {"evaluator_id": evaluator_id, "display_name": evaluator_id})

            def vote(evaluator_id: str):
                return submit_vote(db, task_id, evaluator_id, {"choice": "tie", "confidence": "medium"})

            with ThreadPoolExecutor(max_workers=20) as pool:
                rows = list(pool.map(vote, evaluator_ids))
            self.assertEqual(len({row["id"] for row in rows}), 20)
            analysis = campaign_analysis(db, campaign["id"], bootstrap_samples=0)
            self.assertEqual(analysis["human"]["vote_count"], 20)
            self.assertFalse(analysis["coverage"]["provisional"])


if __name__ == "__main__":
    unittest.main()
