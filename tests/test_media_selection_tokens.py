from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v13_test_utils import get_json, make_workspace, post_json, start_server, stop_server
from vfieval.db import Database
from vfieval.evaluations_v2 import create_campaign_v2, ensure_v2_schema, preview_campaign_v2
from vfieval.media_assets import ensure_collection
from vfieval.selection_tokens import (
    SelectionTokenError,
    SelectionTokenExpired,
    create_selection_snapshot,
    list_methods_for_selection_snapshot,
    resolve_selection_snapshot,
    selection_snapshot_page,
)


class MediaSelectionTokenTests(unittest.TestCase):
    def _seed_items(
        self,
        db: Database,
        workspace,
        count: int,
        *,
        method_keys: tuple[str, ...] = ("method-a",),
    ) -> tuple[dict, list[dict]]:
        collection = ensure_collection(
            db,
            "videos/selection-token",
            "videos-selection-token",
            {"source_kind": "folder", "video_group": "selection-token"},
        )
        pred_collection = ensure_collection(
            db,
            "Selection token predictions",
            "selection-token-predictions",
            {"source_kind": "upload"},
        )
        now = time.time()
        items: list[dict] = []
        with db.connection() as conn:
            for index in range(count):
                name = f"keep-{index:04d}.mp4"
                gt_cur = conn.execute(
                    """
                    INSERT INTO media_assets(
                        collection_id, source_key, source_kind, media_kind, role,
                        display_name, original_name, state, storage_path,
                        frame_count, width, height, fps, created_at, updated_at
                    ) VALUES (?, ?, 'upload', 'video', 'gt', ?, ?, 'ready', ?, 3, 8, 8, 5, ?, ?)
                    """,
                    (
                        int(collection["id"]),
                        f"upload:selection-token-gt:{name}",
                        name,
                        name,
                        str(workspace.root / "selection-token" / name),
                        now,
                        now,
                    ),
                )
                gt_asset_id = int(gt_cur.lastrowid)
                item_cur = conn.execute(
                    """
                    INSERT INTO media_items(
                        collection_id, item_key, canonical_gt_asset_id, display_name,
                        media_kind, state, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'video', 'ready', '{}', ?, ?)
                    """,
                    (
                        int(collection["id"]),
                        f"canonical:selection-token/{name}",
                        gt_asset_id,
                        name,
                        now,
                        now,
                    ),
                )
                item_id = int(item_cur.lastrowid)
                canonical_cur = conn.execute(
                    """
                    INSERT INTO media_item_members(
                        item_id, asset_id, member_role, producer_kind, method_key,
                        reusable_as_pred, temporal_mapping_json, spatial_origin_json,
                        state, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, 'canonical_gt', 'source', '', 0, '{}', '{}',
                              'ready', '{}', ?, ?)
                    """,
                    (item_id, gt_asset_id, now, now),
                )
                prediction_members: dict[str, int] = {}
                for method_key in method_keys:
                    pred_name = f"{method_key}-{index:04d}.mp4"
                    pred_cur = conn.execute(
                        """
                        INSERT INTO media_assets(
                            collection_id, source_key, source_kind, media_kind, role,
                            display_name, original_name, state, storage_path,
                            frame_count, width, height, fps, created_at, updated_at
                        ) VALUES (?, ?, 'upload', 'video', 'pred', ?, ?, 'ready', ?,
                                  3, 8, 8, 5, ?, ?)
                        """,
                        (
                            int(pred_collection["id"]),
                            f"upload:selection-token:{method_key}:{index}",
                            pred_name,
                            pred_name,
                            str(workspace.root / "selection-token-pred" / pred_name),
                            now,
                            now,
                        ),
                    )
                    pred_asset_id = int(pred_cur.lastrowid)
                    member_cur = conn.execute(
                        """
                        INSERT INTO media_item_members(
                            item_id, asset_id, member_role, producer_kind,
                            method_key, reusable_as_pred, temporal_mapping_json,
                            spatial_origin_json, state, metadata_json, created_at, updated_at
                        ) VALUES (?, ?, 'external_pred', 'external', ?, 1, '{}', '{}',
                                  'ready', '{}', ?, ?)
                        """,
                        (item_id, pred_asset_id, method_key, now, now),
                    )
                    prediction_members[method_key] = int(member_cur.lastrowid)
                items.append(
                    {
                        "id": item_id,
                        "gt_asset_id": gt_asset_id,
                        "canonical_member_id": int(canonical_cur.lastrowid),
                        "prediction_members": prediction_members,
                    }
                )
        return collection, items

    def test_http_snapshot_scales_past_sqlite_parameter_limit_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            ensure_v2_schema(db)
            collection, items = self._seed_items(db, workspace, 1001)
            server, thread, base_url = start_server(db, workspace)
            try:
                created = post_json(
                    base_url,
                    "/api/media/item-selections",
                    {"group_id": int(collection["id"]), "q": "keep-"},
                )
                token = str(created["selection_token"])
                self.assertEqual(created["total"], 1001)

                last_page = get_json(
                    base_url,
                    f"/api/media/item-selections/{token}?page=6&page_size=200",
                )
                self.assertEqual(last_page["total"], 1001)
                self.assertEqual(len(last_page["items"]), 1)

                methods = get_json(
                    base_url,
                    f"/api/media/methods?selection_token={token}",
                )
                self.assertEqual(methods["total"], 1001)
                self.assertEqual(methods["methods"][0]["covered_count"], 1001)
                self.assertTrue(methods["methods"][0]["complete"])
                self.assertNotIn("bindings", methods["methods"][0])
            finally:
                stop_server(server, thread)

            reopened = Database(workspace.db_path)
            resolved = resolve_selection_snapshot(reopened, token, require_non_empty=True)
            self.assertEqual(len(resolved["media_item_ids"]), 1001)
            stored = reopened.get(
                "SELECT token_hash FROM media_item_selection_snapshots"
            )
            self.assertIsNotNone(stored)
            self.assertNotEqual(str(stored["token_hash"]), token)

            with reopened.connection() as conn:
                conn.execute(
                    "UPDATE media_items SET state = 'unavailable' WHERE id = ?",
                    (int(items[-1]["id"]),),
                )
            with self.assertRaisesRegex(SelectionTokenError, "no longer ready"):
                resolve_selection_snapshot(reopened, token)

    def test_snapshot_expiry_is_enforced_for_page_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            ensure_v2_schema(db)
            collection, _items = self._seed_items(db, workspace, 1)
            created = create_selection_snapshot(
                db,
                group_id=int(collection["id"]),
                query="keep",
            )
            with db.connection() as conn:
                conn.execute(
                    """
                    UPDATE media_item_selection_snapshots
                    SET expires_at = ?
                    WHERE token_hash = ?
                    """,
                    (
                        time.time() - 1,
                        hashlib.sha256(
                            created["selection_token"].encode("utf-8")
                        ).hexdigest(),
                    ),
                )
            with self.assertRaises(SelectionTokenExpired):
                selection_snapshot_page(db, created["selection_token"])
            with self.assertRaises(SelectionTokenExpired):
                list_methods_for_selection_snapshot(db, created["selection_token"])

    def test_campaign_preview_and_create_accept_the_same_selection_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            ensure_v2_schema(db)
            collection, items = self._seed_items(
                db,
                workspace,
                1,
                method_keys=("method-a", "method-b"),
            )
            token = create_selection_snapshot(
                db,
                group_id=int(collection["id"]),
                query="keep",
            )["selection_token"]
            body = {
                "name": "selection-token-campaign",
                "public_title": "Selection token campaign",
                "target_votes": 1,
                "selection_token": token,
                "method_a": {"kind": "external", "method_key": "method-a"},
                "method_b": {"kind": "external", "method_key": "method-b"},
            }
            item_row = db.get(
                "SELECT * FROM media_items WHERE id = ?",
                (int(items[0]["id"]),),
            )
            assert item_row is not None
            alignment = {
                "fingerprint": "selection-token-alignment",
                "filter": "lanczos",
                "sources": {
                    "gt": {"original": {"width": 8, "height": 8}},
                    "pred_a": {"original": {"width": 8, "height": 8}},
                    "pred_b": {"original": {"width": 8, "height": 8}},
                },
            }
            fake_reference = (
                item_row,
                {"id": int(items[0]["canonical_member_id"])},
                {"id": int(items[0]["gt_asset_id"])},
                workspace.root / "selection-token" / "keep-0000.mp4",
            )
            with (
                patch(
                    "vfieval.evaluations_v2._campaign_item_alignment",
                    return_value=(alignment, {}, []),
                ),
                patch(
                    "vfieval.evaluations_v2.resolve_item_reference",
                    return_value=fake_reference,
                ),
            ):
                preview = preview_campaign_v2(db, workspace, body)
                campaign = create_campaign_v2(db, workspace, body)

            self.assertEqual(preview["task_count"], 1)
            self.assertEqual(preview["group_id"], int(collection["id"]))
            self.assertEqual(len(campaign["items"]), 1)
            config = json.loads(
                db.get(
                    "SELECT config_json FROM evaluation_campaigns_v2 WHERE id = ?",
                    (int(campaign["id"]),),
                )["config_json"]
            )
            self.assertEqual(config["media_item_ids"], [int(items[0]["id"])])


if __name__ == "__main__":
    unittest.main()
