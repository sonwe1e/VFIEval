from __future__ import annotations

import json
import tempfile
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vfieval.evaluations_v2 import (
    EvaluationConflict,
    blind_heartbeat,
    blind_media_asset,
    blind_payload,
    blind_session,
    blind_submit_vote,
    campaign_analysis_v2,
    campaign_export_v2,
    create_campaign_v2,
    get_campaign_v2,
    get_preparation_v2,
    list_run_outputs,
    preview_campaign_v2,
    publish_campaign_v2,
    request_publish_campaign_v2,
    run_pending_preparations,
)
from vfieval.evaluations import add_candidate, create_campaign, publish_campaign
from vfieval.media_assets import (
    create_collection,
    ensure_collection,
    get_asset,
    soft_delete_asset,
    sync_run_assets,
    upsert_asset,
)
from vfieval.media_items import (
    bind_run_source,
    ensure_canonical_gt_item,
    list_item_predictions,
)
from vfieval.run_cleanup import RunCleanupService

from v13_test_utils import add_completed_pred_run, make_workspace, write_mp4


class EvaluationCampaignV2Tests(unittest.TestCase):
    def _upload_asset(self, db, collection_id, path, role, name):
        return upsert_asset(
            db,
            collection_id=collection_id,
            source_key=f"upload:v2:{name}",
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
        )

    def _two_runs(self, workspace, db, *, size=(8, 8)):
        gt_path = write_mp4(
            workspace.runs_dir / "shared" / "gt.mp4",
            [(0, 0, 0), (20, 0, 0), (40, 0, 0)],
            size=size,
        )
        pred_a = write_mp4(
            workspace.runs_dir / "method-a" / "pred.mp4",
            [(0, 2, 0), (20, 2, 0), (40, 2, 0)],
            size=size,
        )
        pred_b = write_mp4(
            workspace.runs_dir / "method-b" / "pred.mp4",
            [(0, 0, 2), (20, 0, 2), (40, 0, 2)],
            size=size,
        )
        run_a = add_completed_pred_run(
            db, workspace, "method-a", pred_a, video_name="clip", gt_video_path=gt_path, size=size
        )
        run_b = add_completed_pred_run(
            db, workspace, "method-b", pred_b, video_name="clip", gt_video_path=gt_path, size=size
        )
        sync_run_assets(db, workspace, run_a)
        sync_run_assets(db, workspace, run_b)
        body = {
            "name": "internal-interpolation-study",
            "public_title": "Interpolation study",
            "target_votes": 1,
            "seed": 17,
            "methods": [
                {"run_id": run_a, "label": "Method Alpha"},
                {"run_id": run_b, "label": "Method Beta"},
            ],
        }
        return run_a, run_b, body, (gt_path, pred_a, pred_b)

    def _item_campaign(
        self,
        workspace,
        db,
        *,
        gt_size=(12, 8),
        pred_a_size=(8, 8),
        pred_b_size=(16, 8),
    ):
        collection = ensure_collection(
            db,
            "videos/test4k",
            "videos-test4k-campaign-v2",
            {"source_kind": "folder", "video_group": "test4k"},
        )
        gt_path = write_mp4(
            workspace.root.parent / "videos" / "test4k" / "clip.mp4",
            [(0, 0, 0), (10, 0, 0), (20, 0, 0), (30, 0, 0), (40, 0, 0)],
            size=gt_size,
            fps=5,
        )
        gt_asset = upsert_asset(
            db,
            collection_id=int(collection["id"]),
            source_key="folder:test4k/clip.mp4",
            source_kind="folder",
            media_kind="video",
            role="gt",
            display_name="clip.mp4",
            original_name="clip.mp4",
            storage_path=gt_path,
            frame_count=5,
            width=gt_size[0],
            height=gt_size[1],
            fps=5,
        )
        item = ensure_canonical_gt_item(db, int(gt_asset["id"]))
        pred_a = write_mp4(
            workspace.runs_dir / "item-method-a" / "pred.mp4",
            [(0, 1, 0), (20, 1, 0), (40, 1, 0)],
            size=pred_a_size,
            fps=5,
        )
        pred_b = write_mp4(
            workspace.runs_dir / "item-method-b" / "pred.mp4",
            [(0, 0, 1), (20, 0, 1), (40, 0, 1)],
            size=pred_b_size,
            fps=5,
        )
        mapping = [0, 2, 4]
        run_a = add_completed_pred_run(
            db,
            workspace,
            "item-method-a",
            pred_a,
            video_name="clip",
            size=pred_a_size,
            source_frame_indices=mapping,
        )
        run_b = add_completed_pred_run(
            db,
            workspace,
            "item-method-b",
            pred_b,
            video_name="clip",
            size=pred_b_size,
            source_frame_indices=mapping,
        )
        for run_id in (run_a, run_b):
            bind_run_source(db, run_id, int(item["id"]), video_name="clip")
            sync_run_assets(db, workspace, run_id)
        self.assertEqual(len(list_item_predictions(db, int(item["id"]))["predictions"]), 2)
        body = {
            "name": "item-campaign",
            "public_title": "Item Campaign",
            "target_votes": 1,
            "media_item_ids": [int(item["id"])],
            "method_a": {"kind": "run", "run_id": run_a},
            "method_b": {"kind": "run", "run_id": run_b},
            "spatial_policy": {
                "mode": "smallest_pred",
                "filter": "lanczos",
                "allow_known_aspect_stretch": True,
            },
        }
        return item, body, (gt_path, pred_a, pred_b)

    def test_item_mode_publish_materializes_alignment_diff_and_frozen_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            item, body, source_paths = self._item_campaign(workspace, db)
            preview = preview_campaign_v2(db, workspace, body)
            self.assertEqual(preview["task_count"], 1)
            plan = preview["items"][0]["alignment_plan"]
            self.assertEqual(plan["target"]["width"], 8)
            self.assertEqual(plan["target"]["height"], 8)
            self.assertEqual(plan["temporal"]["frame_count"], 3)
            self.assertTrue(plan["sources"]["gt"]["aspect_changed"])
            self.assertTrue(plan["sources"]["pred_b"]["aspect_changed"])

            draft = create_campaign_v2(db, workspace, body)
            published = publish_campaign_v2(db, workspace, int(draft["id"]))
            self.assertEqual(published["status"], "published")
            self.assertEqual(published["tasks"], 1)
            package = workspace.evaluations_dir / str(draft["id"])
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            manifest_item = manifest["items"][0]
            self.assertEqual(manifest_item["media_item_id"], int(item["id"]))
            self.assertEqual(manifest_item["alignment_fingerprint"], plan["fingerprint"])
            self.assertEqual(manifest_item["alignment_plan"]["target"]["width"], 8)
            for method in manifest_item["methods"]:
                self.assertTrue((package / method["path"]).is_file())
                self.assertTrue((package / method["diff"]["path"]).is_file())
                self.assertTrue(method["diff"]["sha256"])

            frozen_assets = db.query(
                "SELECT * FROM media_assets WHERE source_kind = 'evaluation_package' ORDER BY id"
            )
            self.assertEqual(len(frozen_assets), 3)
            self.assertTrue(all(int(asset["width"]) == 8 for asset in frozen_assets))
            self.assertTrue(all(int(asset["height"]) == 8 for asset in frozen_assets))
            self.assertTrue(all(int(asset["frame_count"]) == 3 for asset in frozen_assets))
            frozen_members = db.query(
                """
                SELECT member_role, reusable_as_pred, producer_kind, spatial_origin_json
                FROM media_item_members
                WHERE item_id = ? AND member_role IN ('evaluation_gt', 'evaluation_pred')
                ORDER BY id
                """,
                (int(item["id"]),),
            )
            self.assertEqual(
                [row["member_role"] for row in frozen_members],
                ["evaluation_gt", "evaluation_pred", "evaluation_pred"],
            )
            self.assertTrue(all(int(row["reusable_as_pred"]) == 0 for row in frozen_members))
            self.assertTrue(all(row["producer_kind"] == "evaluation_package" for row in frozen_members))
            bindings = db.query(
                "SELECT frozen_member_id FROM evaluation_bindings_v2 WHERE frozen_member_id IS NOT NULL"
            )
            self.assertEqual(len(bindings), 2)
            analysis = campaign_analysis_v2(db, int(draft["id"]), bootstrap_samples=0)
            self.assertEqual(analysis["alignment"]["fingerprints"], [plan["fingerprint"]])

            cleanup = RunCleanupService(db, workspace, cache_grace_seconds=0)
            request = cleanup.request_delete(int(body["method_a"]["run_id"]))
            self.assertEqual(cleanup.process_request(int(request["id"]))["status"], "completed")
            for source in source_paths:
                source.unlink(missing_ok=True)
            for asset in frozen_assets:
                self.assertTrue(Path(str(asset["storage_path"])).exists())

    def test_item_mode_uses_decoded_dimensions_when_probe_dimensions_are_swapped(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _item, body, _source_paths = self._item_campaign(
                workspace,
                db,
                gt_size=(12, 8),
                pred_a_size=(10, 6),
                pred_b_size=(16, 8),
            )
            from vfieval.compare_inputs import inspect_compare_path as real_inspect

            def swapped_inspect(workspace_arg, path):
                observed = real_inspect(workspace_arg, path)
                observed["width"], observed["height"] = (
                    int(observed["height"]),
                    int(observed["width"]),
                )
                return observed

            with patch(
                "vfieval.compare_inputs.inspect_compare_path",
                side_effect=swapped_inspect,
            ):
                preview = preview_campaign_v2(db, workspace, body)
                row = preview["items"][0]
                self.assertEqual(
                    (row["reference"]["width"], row["reference"]["height"]),
                    (12, 8),
                )
                self.assertEqual(
                    (row["methods"]["a"]["width"], row["methods"]["a"]["height"]),
                    (10, 6),
                )
                self.assertEqual(
                    (row["methods"]["b"]["width"], row["methods"]["b"]["height"]),
                    (16, 8),
                )
                plan = row["alignment_plan"]
                self.assertEqual(
                    plan["sources"]["gt"]["original"],
                    {"width": 12, "height": 8},
                )
                self.assertEqual(
                    plan["sources"]["pred_a"]["original"],
                    {"width": 10, "height": 6},
                )
                self.assertEqual(
                    plan["sources"]["pred_b"]["original"],
                    {"width": 16, "height": 8},
                )
                self.assertEqual(
                    (plan["target"]["width"], plan["target"]["height"]),
                    (10, 6),
                )

                draft = create_campaign_v2(db, workspace, body)
                published = publish_campaign_v2(db, workspace, int(draft["id"]))

            self.assertEqual(published["status"], "published")
            self.assertEqual(published["tasks"], 1)
            frozen_assets = db.query(
                "SELECT width, height, storage_path FROM media_assets "
                "WHERE source_kind = 'evaluation_package' ORDER BY id"
            )
            self.assertEqual(len(frozen_assets), 3)
            self.assertTrue(
                all(
                    (int(asset["width"]), int(asset["height"])) == (10, 6)
                    for asset in frozen_assets
                )
            )
            for asset in frozen_assets:
                observed = real_inspect(workspace, Path(str(asset["storage_path"])))
                self.assertEqual((int(observed["width"]), int(observed["height"])), (10, 6))
            manifest = json.loads(
                (workspace.evaluations_dir / str(draft["id"]) / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["items"][0]["alignment_fingerprint"], plan["fingerprint"])

    def test_item_mode_publish_revalidates_strict_frame_mapping_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _item, body, _source_paths = self._item_campaign(workspace, db)
            draft = create_campaign_v2(db, workspace, body)
            bindings = db.query(
                """
                SELECT source_member_id FROM evaluation_bindings_v2 b
                JOIN evaluation_methods_v2 m ON m.id = b.method_id
                WHERE m.campaign_id = ? ORDER BY m.slot
                """,
                (int(draft["id"]),),
            )
            with db.connection() as conn:
                conn.executemany(
                    "UPDATE media_item_members SET temporal_mapping_json = ? WHERE id = ?",
                    [
                        (
                            json.dumps({"source_frame_indices": [0, 1, 2]}),
                            int(binding["source_member_id"]),
                        )
                        for binding in bindings
                    ],
                )
            with self.assertRaisesRegex(ValueError, "new Campaign from a fresh preview"):
                publish_campaign_v2(db, workspace, int(draft["id"]))
            self.assertEqual(get_campaign_v2(db, int(draft["id"]))["status"], "failed")
            self.assertEqual(
                int(db.get("SELECT COUNT(*) AS count FROM evaluation_tasks_v2")["count"]),
                0,
            )
            self.assertEqual(
                int(
                    db.get(
                        "SELECT COUNT(*) AS count FROM media_assets WHERE source_kind = 'evaluation_package'"
                    )["count"]
                ),
                0,
            )
            self.assertFalse((workspace.evaluations_dir / str(draft["id"])).exists())

    def test_item_mode_publish_preserves_frame_sequence_package_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection = ensure_collection(
                db,
                "videos/frame-set",
                "videos-frame-set-campaign-v2",
                {"source_kind": "folder", "video_group": "frame-set"},
            )
            gt_dir = workspace.root.parent / "videos" / "frame-set" / "clip"
            gt_dir.mkdir(parents=True, exist_ok=True)
            for index, value in enumerate((0, 20, 40)):
                Image.new("RGB", (8, 8), (value, 0, 0)).save(gt_dir / f"{index:06d}.png")
            gt_asset = upsert_asset(
                db,
                collection_id=int(collection["id"]),
                source_key="folder:frame-set/clip",
                source_kind="folder",
                media_kind="frame_sequence",
                role="gt",
                display_name="clip",
                original_name="clip",
                storage_path=gt_dir,
                frame_count=3,
                width=8,
                height=8,
                fps=5,
            )
            item = ensure_canonical_gt_item(db, int(gt_asset["id"]))
            runs: list[int] = []
            for slot, color in (("a", (0, 1, 0)), ("b", (0, 0, 1))):
                pred = write_mp4(
                    workspace.runs_dir / f"frame-sequence-{slot}" / "pred.mp4",
                    [color, color, color],
                    size=(8, 8),
                    fps=5,
                )
                run_id = add_completed_pred_run(
                    db,
                    workspace,
                    f"frame-sequence-{slot}",
                    pred,
                    video_name="clip",
                    size=(8, 8),
                    fps=5,
                )
                bind_run_source(db, run_id, int(item["id"]), video_name="clip")
                sync_run_assets(db, workspace, run_id)
                runs.append(run_id)
            body = {
                "name": "frame-sequence-campaign",
                "public_title": "Frame sequence campaign",
                "media_item_ids": [int(item["id"])],
                "method_a": {"kind": "run", "run_id": runs[0]},
                "method_b": {"kind": "run", "run_id": runs[1]},
            }
            published = publish_campaign_v2(
                db,
                workspace,
                int(create_campaign_v2(db, workspace, body)["id"]),
            )
            self.assertEqual(published["status"], "published")
            assets = db.query(
                "SELECT media_kind, storage_path FROM media_assets WHERE source_kind = 'evaluation_package'"
            )
            self.assertEqual(len(assets), 3)
            self.assertTrue(all(row["media_kind"] == "frame_sequence" for row in assets))
            self.assertTrue(
                all(len(list(Path(str(row["storage_path"])).glob("*.png"))) == 3 for row in assets)
            )

    def test_preview_groups_two_run_methods_and_requires_strict_common_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            run_a, run_b, body, _paths = self._two_runs(workspace, db)
            preview = preview_campaign_v2(db, workspace, body)
            self.assertEqual(preview["schema_version"], 2)
            self.assertEqual(preview["task_count"], 1)
            self.assertEqual(preview["videos"][0]["status"], "ready")
            self.assertEqual(preview["videos"][0]["reference"]["frame_count"], 3)
            self.assertEqual(preview["videos"][0]["methods"]["a"]["width"], 8)
            self.assertEqual(preview["videos"][0]["methods"]["b"]["fps"], 5)
            self.assertEqual([row["run_id"] for row in preview["methods"]], [run_a, run_b])

            outputs = list_run_outputs(db)
            self.assertEqual({row["run_id"] for row in outputs}, {run_a, run_b})
            self.assertTrue(all(row["videos"] for row in outputs))

    def test_preview_rejects_actual_stream_fps_when_asset_metadata_matches(self) -> None:
        """The selector must validate the stream, not only catalog metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            gt_path = write_mp4(
                workspace.runs_dir / "timestamp-shared" / "gt.mp4",
                [(0, 0, 0), (20, 0, 0), (40, 0, 0)],
                fps=5,
            )
            pred_a = write_mp4(
                workspace.runs_dir / "timestamp-a" / "pred.mp4",
                [(0, 2, 0), (20, 2, 0), (40, 2, 0)],
                fps=5,
            )
            pred_b = write_mp4(
                workspace.runs_dir / "timestamp-b" / "pred.mp4",
                [(0, 0, 2), (20, 0, 2), (40, 0, 2)],
                fps=10,
            )
            # Deliberately persist matching 5 fps catalog metadata for all
            # three assets.  The decoded timestamps of Pred B still differ.
            run_a = add_completed_pred_run(
                db, workspace, "timestamp-a", pred_a, video_name="clip", gt_video_path=gt_path, fps=5
            )
            run_b = add_completed_pred_run(
                db, workspace, "timestamp-b", pred_b, video_name="clip", gt_video_path=gt_path, fps=5
            )
            sync_run_assets(db, workspace, run_a)
            sync_run_assets(db, workspace, run_b)
            preview = preview_campaign_v2(
                db,
                workspace,
                {
                    "name": "timestamp-mismatch",
                    "public_title": "Timestamp mismatch",
                    "methods": [
                        {"run_id": run_a, "label": "A"},
                        {"run_id": run_b, "label": "B"},
                    ],
                },
            )
            row = preview["videos"][0]
            self.assertEqual(row["status"], "alignment_mismatch")
            self.assertFalse(row["selectable"])
            self.assertIn("fps", " ".join(row["reasons"]).lower())

    def test_preview_uses_observed_fps_when_catalog_metadata_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            gt_path = write_mp4(
                workspace.runs_dir / "observed-fps-shared" / "gt.mp4",
                [(0, 0, 0), (20, 0, 0), (40, 0, 0)],
                fps=5,
            )
            pred_a = write_mp4(
                workspace.runs_dir / "observed-fps-a" / "pred.mp4",
                [(0, 2, 0), (20, 2, 0), (40, 2, 0)],
                fps=5,
            )
            pred_b = write_mp4(
                workspace.runs_dir / "observed-fps-b" / "pred.mp4",
                [(0, 0, 2), (20, 0, 2), (40, 0, 2)],
                fps=10,
            )
            run_a = add_completed_pred_run(
                db, workspace, "observed-fps-a", pred_a, video_name="clip", gt_video_path=gt_path, fps=5
            )
            run_b = add_completed_pred_run(
                db, workspace, "observed-fps-b", pred_b, video_name="clip", gt_video_path=gt_path, fps=5
            )
            sync_run_assets(db, workspace, run_a)
            sync_run_assets(db, workspace, run_b)
            frames = [Path(f"frame-{index}.png") for index in range(3)]
            # Some decoders cannot expose per-frame timestamps.  Strict
            # selection must still use the source stream fps rather than the
            # stale matching metadata persisted above.
            from vfieval.compare_inputs import inspect_compare_path as real_inspect

            def stale_inspect(workspace_arg, path):
                info = real_inspect(workspace_arg, path)
                info["fps"] = 5.0
                return info

            with patch("vfieval.evaluations_v2.inspect_compare_path", side_effect=stale_inspect), patch(
                "vfieval.datasets._load_compare_source_frames",
                side_effect=[
                    (frames, 5.0, [None, None, None]),
                    (frames, 5.0, [None, None, None]),
                    (frames, 5.0, [None, None, None]),
                    (frames, 10.0, [None, None, None]),
                ],
            ):
                preview = preview_campaign_v2(
                    db,
                    workspace,
                    {
                        "name": "observed-fps-mismatch",
                        "public_title": "Observed FPS mismatch",
                        "methods": [
                            {"run_id": run_a, "label": "A"},
                            {"run_id": run_b, "label": "B"},
                        ],
                    },
                )
            row = preview["videos"][0]
            self.assertEqual(row["status"], "alignment_mismatch")
            self.assertFalse(row["selectable"])
            self.assertIn("fps", " ".join(row["reasons"]).lower())

    def test_preview_reports_decoded_timestamp_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            frames = [Path(f"timestamp-frame-{index}.png") for index in range(3)]
            with patch(
                "vfieval.datasets._load_compare_source_frames",
                side_effect=[
                    (frames, 5.0, [0.0, 0.2, 0.4]),
                    (frames, 5.0, [0.0, 0.2, 0.4]),
                    (frames, 5.0, [0.0, 0.2, 0.4]),
                    (frames, 5.0, [0.0, 0.1, 0.4]),
                ],
            ):
                preview = preview_campaign_v2(db, workspace, body)
            row = preview["videos"][0]
            self.assertEqual(row["status"], "alignment_mismatch")
            self.assertFalse(row["selectable"])
            self.assertIn("timestamps", " ".join(row["reasons"]).lower())

    def test_advanced_uploaded_methods_use_explicit_video_and_gt_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection = create_collection(db, "External blind methods")
            root = workspace.media_dir / "external-v2"
            gt = self._upload_asset(
                db,
                collection["id"],
                write_mp4(root / "gt.mp4", [(0, 0, 0)] * 3),
                "gt",
                "external-gt",
            )
            pred_a = self._upload_asset(
                db,
                collection["id"],
                write_mp4(root / "a.mp4", [(1, 0, 0)] * 3),
                "pred",
                "external-a",
            )
            pred_b = self._upload_asset(
                db,
                collection["id"],
                write_mp4(root / "b.mp4", [(0, 1, 0)] * 3),
                "pred",
                "external-b",
            )
            body = {
                "name": "external",
                "public_title": "External study",
                "methods": [
                    {
                        "source_kind": "upload",
                        "label": "External A",
                        "videos": [
                            {
                                "video_name": "clip",
                                "asset_id": pred_a["id"],
                                "reference_asset_id": gt["id"],
                            }
                        ],
                    },
                    {
                        "source_kind": "upload",
                        "label": "External B",
                        "videos": [
                            {
                                "video_name": "clip",
                                "asset_id": pred_b["id"],
                                "reference_asset_id": gt["id"],
                            }
                        ],
                    },
                ],
            }
            preview = preview_campaign_v2(db, workspace, body)
            self.assertEqual(preview["ready_video_names"], ["clip"])
            campaign = publish_campaign_v2(
                db, workspace, create_campaign_v2(db, workspace, body)["id"]
            )
            self.assertEqual(campaign["status"], "published")

    def test_atomic_package_survives_run_media_cleanup_and_blind_api_hides_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, source_paths = self._two_runs(workspace, db)
            campaign = create_campaign_v2(db, workspace, body)
            published = publish_campaign_v2(db, workspace, campaign["id"])
            self.assertEqual(published["status"], "published")
            self.assertEqual(published["tasks"], 1)
            package = workspace.root / "evaluations" / str(campaign["id"])
            self.assertTrue((package / "manifest.json").is_file())
            frozen = db.query(
                "SELECT * FROM media_assets WHERE source_kind = 'evaluation_package' ORDER BY id"
            )
            self.assertEqual(len(frozen), 3)
            self.assertTrue(all(Path(row["storage_path"]).exists() for row in frozen))
            with self.assertRaisesRegex(ValueError, "immutable"):
                soft_delete_asset(db, workspace, int(frozen[0]["id"]))
            self.assertEqual(get_asset(db, int(frozen[0]["id"]))["state"], "ready")

            token = str(published["public_token"])
            public = blind_payload(db, token)
            self.assertEqual(public["campaign"]["title"], "Interpolation study")
            self.assertIsNone(public["task"])
            first = blind_session(
                db,
                token,
                {"evaluator_id": "browser-uuid", "display_name": "Alice"},
                lease_seconds=120,
            )
            task = first["task"]
            self.assertIsNotNone(task)
            assignment_token = str(
                db.get(
                    "SELECT assignment_token FROM evaluation_assignments_v2 WHERE evaluator_id = ?",
                    ("browser-uuid",),
                )["assignment_token"]
            )
            serialized = str(task)
            self.assertNotIn("run_id", serialized)
            self.assertNotIn("asset_id", serialized)
            self.assertNotIn("Method Alpha", serialized)
            self.assertNotIn("Method Beta", serialized)
            again = blind_session(
                db,
                token,
                {"evaluator_id": "browser-uuid", "display_name": "Alice"},
                lease_seconds=120,
            )
            self.assertEqual(again["task"]["token"], task["token"])
            renewed = blind_heartbeat(
                db, token, task["token"], "browser-uuid", lease_seconds=300
            )
            self.assertGreater(renewed["lease_expires_at"], task["lease_expires_at"])
            _asset, media_path = blind_media_asset(
                db, workspace, token, task["token"], "left", assignment_token
            )
            self.assertTrue(media_path.is_relative_to(package))

            with db.connection() as conn:
                conn.execute(
                    "UPDATE evaluation_assignments_v2 SET state = 'expired', lease_expires_at = ? WHERE evaluator_id = ?",
                    (0, "browser-uuid"),
                )
            with self.assertRaisesRegex(ValueError, "lease expired"):
                blind_media_asset(
                    db, workspace, token, task["token"], "left", assignment_token
                )
            with db.connection() as conn:
                conn.execute(
                    "UPDATE evaluation_assignments_v2 SET state = 'leased', lease_expires_at = ? WHERE evaluator_id = ?",
                    (10**12, "browser-uuid"),
                )

            for source in source_paths:
                source.unlink()
            vote = blind_submit_vote(
                db,
                token,
                task["token"],
                "browser-uuid",
                {
                    "choice": "tie",
                    "reasons": ["temporal_stability"],
                    "confidence": "high",
                },
            )
            self.assertTrue(vote["progress"]["complete"])
            self.assertIn("results", vote)
            self.assertEqual(
                {row["label"] for row in vote["results"]["human"]["ranking"]},
                {"Method Alpha", "Method Beta"},
            )
            _asset, surviving_path = blind_media_asset(
                db, workspace, token, task["token"], "reference", assignment_token
            )
            self.assertTrue(surviving_path.exists())

            analysis_a = campaign_analysis_v2(db, campaign["id"], bootstrap_samples=50)
            analysis_b = campaign_analysis_v2(db, campaign["id"], bootstrap_samples=50)
            self.assertEqual(analysis_a, analysis_b)
            self.assertEqual(analysis_a["human"]["ranking"][0]["score"], 0.5)
            self.assertIsNone(analysis_a["combined_score"])
            exported = campaign_export_v2(db, campaign["id"])
            self.assertEqual(len(exported["methods"]), 2)
            self.assertEqual(len(exported["items"]), 1)
            self.assertEqual(len(exported["tasks"]), 1)
            self.assertEqual(len(exported["votes"]), 1)

    def test_frozen_package_is_a_private_snapshot_not_a_hard_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, source_paths = self._two_runs(workspace, db)
            campaign = publish_campaign_v2(
                db, workspace, create_campaign_v2(db, workspace, body)["id"]
            )
            package = workspace.evaluations_dir / str(campaign["id"])
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            frozen_pred = package / manifest["items"][0]["methods"][0]["path"]
            source_pred = source_paths[1]
            self.assertFalse(frozen_pred.samefile(source_pred))
            before = frozen_pred.read_bytes()
            source_pred.write_bytes(b"source bytes changed after publish")
            self.assertEqual(frozen_pred.read_bytes(), before)

    def test_superseded_preparation_claim_cannot_publish_or_clear_new_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            campaign = create_campaign_v2(db, workspace, body)
            request_publish_campaign_v2(db, campaign["id"])
            with db.connection() as conn:
                conn.execute(
                    """
                    UPDATE evaluation_preparations_v2
                    SET state = 'running', claim_token = 'new-owner', updated_at = ?
                    WHERE campaign_id = ?
                    """,
                    (10**12, int(campaign["id"])),
                )
            with self.assertRaises(EvaluationConflict):
                publish_campaign_v2(
                    db,
                    workspace,
                    campaign["id"],
                    claim_token="stale-owner",
                )
            preparation = get_preparation_v2(db, campaign["id"])
            self.assertEqual(preparation["state"], "running")
            self.assertEqual(preparation["claim_token"], "new-owner")
            self.assertEqual(get_campaign_v2(db, campaign["id"])["status"], "preparing")
            self.assertFalse((workspace.evaluations_dir / str(campaign["id"])).exists())

    def test_persistent_preparation_request_can_be_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            campaign = create_campaign_v2(db, workspace, body)
            requested = request_publish_campaign_v2(db, campaign["id"])
            self.assertEqual(requested["campaign"]["status"], "preparing")
            self.assertEqual(requested["preparation"]["state"], "queued")
            completed = run_pending_preparations(db, workspace)
            self.assertEqual(completed, [{"campaign_id": campaign["id"], "status": "published"}])
            self.assertEqual(get_preparation_v2(db, campaign["id"])["state"], "completed")

    def test_run_purge_keeps_published_frozen_campaign_playable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            published = publish_campaign_v2(
                db, workspace, create_campaign_v2(db, workspace, body)["id"]
            )
            service = RunCleanupService(db, workspace, cache_grace_seconds=0)
            request = service.request_delete(run_a)
            completed = service.process_request(int(request["id"]))
            self.assertEqual(completed["status"], "completed")
            self.assertIsNotNone(db.get_run(run_a)["deleted_at"])

            session = blind_session(
                db,
                str(published["public_token"]),
                {"evaluator_id": "purge-browser", "display_name": "Purge Tester"},
            )
            task = session["task"]
            self.assertIsNotNone(task)
            assignment_token = str(
                db.get(
                    "SELECT assignment_token FROM evaluation_assignments_v2 WHERE evaluator_id = ?",
                    ("purge-browser",),
                )["assignment_token"]
            )
            for side in ("reference", "left", "right"):
                asset, path = blind_media_asset(
                    db,
                    workspace,
                    str(published["public_token"]),
                    str(task["token"]),
                    side,
                    assignment_token,
                )
                self.assertEqual(asset["source_kind"], "evaluation_package")
                self.assertTrue(path.exists())

    def test_run_purge_freezes_readable_published_v1_campaign_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            run_a, run_b, _body, _paths = self._two_runs(workspace, db)
            reference_id = int(
                db.get(
                    "SELECT asset_id FROM run_media_assets WHERE run_id = ? AND role = 'gt' LIMIT 1",
                    (run_a,),
                )["asset_id"]
            )
            pred_a_id = int(
                db.get(
                    "SELECT asset_id FROM run_media_assets WHERE run_id = ? AND role = 'pred' LIMIT 1",
                    (run_a,),
                )["asset_id"]
            )
            pred_b_id = int(
                db.get(
                    "SELECT asset_id FROM run_media_assets WHERE run_id = ? AND role = 'pred' LIMIT 1",
                    (run_b,),
                )["asset_id"]
            )
            legacy = create_campaign(db, {"name": "legacy-protected", "target_votes": 1})
            for asset_id in (pred_a_id, pred_b_id):
                add_candidate(
                    db,
                    workspace,
                    int(legacy["id"]),
                    {
                        "reference_asset_id": reference_id,
                        "asset_id": asset_id,
                        "video_name": "clip",
                    },
                )
            self.assertEqual(
                publish_campaign(db, workspace, int(legacy["id"]))["status"],
                "published",
            )

            service = RunCleanupService(db, workspace, cache_grace_seconds=0)
            request = service.request_delete(run_a)
            completed = service.process_request(int(request["id"]))
            self.assertEqual(completed["status"], "completed")
            protected_reference = db.get(
                "SELECT source_kind, storage_path, state FROM media_assets WHERE id = ?",
                (reference_id,),
            )
            protected_pred = db.get(
                "SELECT source_kind, storage_path, state FROM media_assets WHERE id = ?",
                (pred_a_id,),
            )
            for asset in (protected_reference, protected_pred):
                self.assertEqual(asset["source_kind"], "evaluation_package")
                self.assertEqual(asset["state"], "ready")
                self.assertTrue(Path(str(asset["storage_path"])).exists())

    def test_failed_preparation_removes_staging_and_leaves_no_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            campaign = create_campaign_v2(db, workspace, body)
            source_b = db.get(
                """
                SELECT ma.storage_path
                FROM evaluation_bindings_v2 b
                JOIN evaluation_methods_v2 m ON m.id = b.method_id
                JOIN media_assets ma ON ma.id = b.source_asset_id
                WHERE m.campaign_id = ? AND m.slot = 'b'
                """,
                (int(campaign["id"]),),
            )
            Path(str(source_b["storage_path"])).unlink()
            with self.assertRaises(FileNotFoundError):
                publish_campaign_v2(db, workspace, campaign["id"])
            failed = get_campaign_v2(db, campaign["id"])
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["tasks"], 0)
            self.assertFalse((workspace.root / "evaluations" / str(campaign["id"])).exists())
            self.assertFalse(any((workspace.root / "evaluations" / ".staging").iterdir()))

    def test_assignment_leases_limit_concurrent_votes_to_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body, _paths = self._two_runs(workspace, db)
            body["target_votes"] = 2
            campaign = publish_campaign_v2(
                db, workspace, create_campaign_v2(db, workspace, body)["id"]
            )
            token = campaign["public_token"]

            def join(index: int):
                return blind_session(
                    db,
                    token,
                    {"evaluator_id": f"browser-{index}", "display_name": f"User {index}"},
                )

            with ThreadPoolExecutor(max_workers=5) as pool:
                sessions = list(pool.map(join, range(5)))
            assigned = [
                (index, payload["task"])
                for index, payload in enumerate(sessions)
                if payload["task"] is not None
            ]
            self.assertEqual(len(assigned), 2)
            self.assertTrue(all(not payload["progress"]["complete"] for payload in sessions))

            def vote(pair):
                index, task = pair
                return blind_submit_vote(
                    db, token, task["token"], f"browser-{index}", {"choice": "tie"}
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(vote, assigned))
            count = db.get("SELECT COUNT(*) AS count FROM evaluation_votes_v2")
            self.assertEqual(int(count["count"]), 2)


if __name__ == "__main__":
    unittest.main()
