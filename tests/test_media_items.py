from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vfieval.media_assets import (
    bind_run_asset,
    create_collection,
    sync_folder_assets,
    sync_run_assets,
    upsert_asset,
)
from vfieval.media_items import (
    bind_run_source,
    ensure_canonical_gt_item,
    get_media_item,
    list_item_groups,
    list_item_predictions,
    list_media_items,
    list_methods_for_items,
    list_unbound_predictions,
    register_external_prediction,
    register_model_prediction,
)

from v13_test_utils import make_workspace, write_mp4


class MediaItemTests(unittest.TestCase):
    def _asset(
        self,
        db,
        collection_id: int,
        root: Path,
        source_key: str,
        display_name: str,
        *,
        source_kind: str = "folder",
        role: str = "gt",
        content: bytes = b"video",
        provenance: dict | None = None,
    ) -> dict:
        path = root / f"{source_key.replace(':', '-').replace('/', '-')}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return upsert_asset(
            db,
            collection_id=collection_id,
            source_key=source_key,
            source_kind=source_kind,
            media_kind="video",
            role=role,
            display_name=display_name,
            original_name=display_name,
            storage_path=path,
            content_sha256="a" * 64,
            size_bytes=len(content),
            frame_count=3,
            width=16,
            height=8,
            fps=5.0,
            provenance=provenance,
        )

    def _run(self, db, root: Path, name: str, *, run_type: str = "model_inference") -> int:
        model_id = db.upsert_model(f"model-{name}", "dummy", None, 8, 16)
        dataset_root = root / f"dataset-{name}"
        dataset_root.mkdir(parents=True, exist_ok=True)
        dataset_id = db.create_dataset(f"dataset-{name}", str(dataset_root), True)
        run_id = db.create_run(
            name,
            model_id,
            dataset_id,
            8,
            16,
            1,
            "cpu",
            "fp32",
            [],
            metadata={"run_type": run_type},
        )
        with db.connection() as conn:
            conn.execute("UPDATE runs SET status = 'completed' WHERE id = ?", (run_id,))
        return run_id

    def test_gt_assets_create_exact_items_and_grouped_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            first_group = create_collection(db, "test4k", metadata={"source_kind": "folder"})
            second_group = create_collection(db, "other", metadata={"source_kind": "folder"})
            first = self._asset(db, first_group["id"], workspace.root, "folder:test4k/clip.mp4", "clip.mp4")
            second = self._asset(db, second_group["id"], workspace.root, "folder:other/clip.mp4", "clip.mp4")

            first_item = ensure_canonical_gt_item(db, first["id"])
            second_item = ensure_canonical_gt_item(db, second["id"])
            self.assertNotEqual(first_item["id"], second_item["id"])
            self.assertNotEqual(first_item["item_key"], second_item["item_key"])
            self.assertEqual(get_media_item(db, first_item["id"])["canonical_gt_asset_id"], first["id"])

            groups = list_item_groups(db)["groups"]
            self.assertEqual({row["name"] for row in groups}, {"test4k", "other"})
            page = list_media_items(db, first_group["id"], query="clip")
            self.assertEqual(page["total"], 1)
            self.assertEqual(page["items"][0]["id"], first_item["id"])

    def test_folder_sync_keeps_slug_colliding_groups_as_distinct_gt_collections(self) -> None:
        """Folder names that normalize to one slug must not share an Item group."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            first_path = workspace.root.parent / "videos" / "a b" / "clip.mp4"
            write_mp4(first_path, [(0, 0, 0), (20, 0, 0), (40, 0, 0)], size=(8, 8), fps=5)
            second_path = workspace.root.parent / "videos" / "a-b" / "clip.mp4"
            second_path.parent.mkdir(parents=True, exist_ok=True)
            second_path.write_bytes(first_path.read_bytes())

            self.assertEqual(sync_folder_assets(db, workspace), 2)
            assets = db.query(
                """
                SELECT id, collection_id, content_sha256 FROM media_assets
                WHERE source_key IN ('folder:a b/clip.mp4', 'folder:a-b/clip.mp4')
                ORDER BY source_key
                """
            )
            self.assertEqual(len(assets), 2)
            self.assertNotEqual(int(assets[0]["collection_id"]), int(assets[1]["collection_id"]))
            self.assertEqual(assets[0]["content_sha256"], assets[1]["content_sha256"])
            items = db.query(
                """
                SELECT id, collection_id, canonical_gt_asset_id FROM media_items
                WHERE canonical_gt_asset_id IN (?, ?)
                ORDER BY canonical_gt_asset_id
                """,
                (int(assets[0]["id"]), int(assets[1]["id"])),
            )
            self.assertEqual(len(items), 2)
            self.assertNotEqual(int(items[0]["collection_id"]), int(items[1]["collection_id"]))

    def test_canonical_item_key_collision_never_repoints_an_existing_gt_item(self) -> None:
        """A stale Item key must fail closed instead of merging two GT assets."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            first_group = create_collection(db, "first", metadata={"source_kind": "folder"})
            second_group = create_collection(db, "second", metadata={"source_kind": "folder"})
            first = self._asset(
                db,
                first_group["id"],
                workspace.root,
                "folder:first/clip.mp4",
                "clip.mp4",
            )
            first_item = ensure_canonical_gt_item(db, int(first["id"]))
            # Create the physical asset without the automatic canonical-GT
            # hook, then model a damaged legacy Item key. This verifies the
            # synchronizer never silently reassigns Item identity.
            second = self._asset(
                db,
                second_group["id"],
                workspace.root,
                "folder:second/clip.mp4",
                "clip.mp4",
                role="source",
            )
            collision_key = f"canonical:{second['source_key']}"
            with db.connection() as conn:
                conn.execute("UPDATE media_assets SET role = 'gt' WHERE id = ?", (int(second["id"]),))
                conn.execute(
                    "UPDATE media_items SET item_key = ? WHERE id = ?",
                    (collision_key, int(first_item["id"])),
                )

            with self.assertRaisesRegex(ValueError, "refusing to merge distinct GT assets"):
                ensure_canonical_gt_item(db, int(second["id"]))
            self.assertEqual(
                int(get_media_item(db, int(first_item["id"]))["canonical_gt_asset_id"]),
                int(first["id"]),
            )

    def test_model_prediction_cannot_cross_bind_a_multi_source_run(self) -> None:
        """A Pred from video B must not become a reusable member of Item A."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            source_group = create_collection(db, "Sources", metadata={"source_kind": "folder"})
            run_group = create_collection(db, "Run outputs", metadata={"source_kind": "run_artifact"})
            first_gt = self._asset(
                db,
                source_group["id"],
                workspace.root,
                "folder:source/a.mp4",
                "a.mp4",
            )
            second_gt = self._asset(
                db,
                source_group["id"],
                workspace.root,
                "folder:source/b.mp4",
                "b.mp4",
            )
            first_item = ensure_canonical_gt_item(db, int(first_gt["id"]))
            second_item = ensure_canonical_gt_item(db, int(second_gt["id"]))
            run_id = self._run(db, workspace.root, "multi-source")
            bind_run_source(db, run_id, int(first_item["id"]), video_name="a")
            bind_run_source(db, run_id, int(second_item["id"]), video_name="b")
            pred = self._asset(
                db,
                run_group["id"],
                workspace.runs_dir,
                "run_artifact:multi-source-b",
                "b-pred.mp4",
                source_kind="run_artifact",
                role="pred",
            )
            bind_run_asset(db, run_id, int(pred["id"]), "pred", video_name="b")

            with self.assertRaisesRegex(ValueError, "exactly one bound source media item"):
                register_model_prediction(db, run_id, int(first_item["id"]), int(pred["id"]))
            member = register_model_prediction(db, run_id, int(second_item["id"]), int(pred["id"]))
            self.assertEqual(int(member["item_id"]), int(second_item["id"]))

    def test_only_explicit_new_model_or_external_predictions_are_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            source_collection = create_collection(db, "Sources", metadata={"source_kind": "folder"})
            run_collection = create_collection(db, "Run outputs", metadata={"source_kind": "run_artifact"})
            upload_collection = create_collection(db, "Uploads", metadata={"source_kind": "upload"})
            gt = self._asset(
                db,
                source_collection["id"],
                workspace.root,
                "folder:test/clip.mp4",
                "clip.mp4",
                provenance={"video_group": "test", "video": "clip.mp4"},
            )
            item = ensure_canonical_gt_item(db, gt["id"])

            run_id = self._run(db, workspace.root, "new-model")
            bind_run_asset(db, run_id, gt["id"], "source", video_name="clip.mp4", metadata={"input": True})
            bind_run_source(db, run_id, item["id"], video_name="clip.mp4")
            pred = self._asset(
                db,
                run_collection["id"],
                workspace.runs_dir,
                "run_artifact:new-pred",
                "new-pred.mp4",
                source_kind="run_artifact",
                role="pred",
            )
            bind_run_asset(db, run_id, pred["id"], "pred", video_name="clip.mp4")
            member = register_model_prediction(
                db,
                run_id,
                item["id"],
                pred["id"],
                temporal_mapping={"source_frame_indices": [0, 1, 2]},
                spatial_origin={"width": 16, "height": 8},
            )
            self.assertTrue(member["reusable_as_pred"])

            external = self._asset(
                db,
                upload_collection["id"],
                workspace.media_dir,
                "upload:external",
                "external.mp4",
                source_kind="upload",
                role="pred",
            )
            external_member = register_external_prediction(
                db,
                item["id"],
                external["id"],
                method_key="external:method-a",
            )
            self.assertTrue(external_member["reusable_as_pred"])
            predictions = list_item_predictions(db, item["id"])["predictions"]
            self.assertEqual({row["id"] for row in predictions}, {member["id"], external_member["id"]})

            compare_run = self._run(db, workspace.root, "compare", run_type="video_compare")
            bind_run_asset(db, compare_run, gt["id"], "source", video_name="clip.mp4")
            bind_run_source(db, compare_run, item["id"], video_name="clip.mp4")
            bind_run_asset(db, compare_run, pred["id"], "pred", video_name="clip.mp4")
            with self.assertRaisesRegex(ValueError, "model_inference"):
                register_model_prediction(db, compare_run, item["id"], pred["id"])

    def test_catalog_sync_does_not_backfill_legacy_run_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            run_collection = create_collection(db, "Run outputs", metadata={"source_kind": "run_artifact"})
            run_id = self._run(db, workspace.root, "legacy")
            pred = self._asset(
                db,
                run_collection["id"],
                workspace.runs_dir,
                "run_artifact:legacy",
                "legacy.mp4",
                source_kind="run_artifact",
                role="pred",
            )
            bind_run_asset(db, run_id, pred["id"], "pred", video_name="clip.mp4")

            sync_run_assets(db, workspace, run_id)
            self.assertIsNone(db.get("SELECT id FROM media_item_members WHERE asset_id = ?", (pred["id"],)))
            audit = list_unbound_predictions(db)["predictions"]
            self.assertEqual([row["id"] for row in audit], [pred["id"]])
            self.assertEqual(audit[0]["reason"], "legacy_or_missing_item_binding")

    def test_sync_run_assets_publishes_only_when_source_binding_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            source_collection = create_collection(db, "Sources", metadata={"source_kind": "folder"})
            gt = self._asset(
                db,
                source_collection["id"],
                workspace.root,
                "folder:test/clip.mp4",
                "clip.mp4",
                provenance={"video_group": "test", "video": "clip.mp4"},
            )
            item = ensure_canonical_gt_item(db, gt["id"])
            run_id = self._run(db, workspace.root, "published")
            bind_run_asset(db, run_id, gt["id"], "source", video_name="clip.mp4")
            bind_run_source(db, run_id, item["id"], video_name="clip.mp4")

            output = workspace.runs_dir / str(run_id) / "pred.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"pred")
            job_id = int(db.get_run(run_id)["inference_job_id"])
            db.add_artifact(
                job_id,
                None,
                "pred_video",
                str(output),
                "video/mp4",
                {
                    "video_name": "clip",
                    "frames": 3,
                    "width": 16,
                    "height": 8,
                    "fps": 5.0,
                    "source_frame_indices": [0, 1, 2],
                    "source_video_group": "test",
                    "source_video_file": "clip.mp4",
                },
            )
            assets = sync_run_assets(db, workspace, run_id)
            output_asset = next(row for row in assets if row["role"] == "pred")
            predictions = list_item_predictions(db, item["id"])["predictions"]
            self.assertEqual(len(predictions), 1)
            self.assertEqual(predictions[0]["asset_id"], output_asset["id"])
            self.assertEqual(predictions[0]["temporal_mapping"]["source_frame_indices"], [0, 1, 2])

    def test_method_coverage_and_database_reuse_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection = create_collection(db, "Sources", metadata={"source_kind": "folder"})
            upload_collection = create_collection(db, "Uploads", metadata={"source_kind": "upload"})
            items = []
            for name in ("a.mp4", "b.mp4"):
                gt = self._asset(db, collection["id"], workspace.root, f"folder:test/{name}", name)
                items.append(ensure_canonical_gt_item(db, gt["id"]))
            external = self._asset(
                db,
                upload_collection["id"],
                workspace.media_dir,
                "upload:method-a",
                "method-a.mp4",
                source_kind="upload",
                role="pred",
            )
            register_external_prediction(db, items[0]["id"], external["id"], method_key="external:a")
            methods = list_methods_for_items(db, [row["id"] for row in items])["methods"]
            self.assertEqual(len(methods), 1)
            self.assertFalse(methods[0]["complete"])
            self.assertEqual(methods[0]["missing_item_ids"], [items[1]["id"]])

            with self.assertRaises(sqlite3.IntegrityError), db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO media_item_members(
                        item_id, asset_id, member_role, producer_kind,
                        method_key, reusable_as_pred, created_at, updated_at
                    ) VALUES (?, ?, 'compare_snapshot', 'video_compare', 'bad', 1, 1, 1)
                    """,
                    (items[0]["id"], external["id"]),
                )


if __name__ == "__main__":
    unittest.main()
