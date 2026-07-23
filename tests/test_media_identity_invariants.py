from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from vfieval.config import WorkspaceConfig
from vfieval.db import Database, LATEST_SCHEMA_VERSION, utc_ts
from vfieval.media_assets import (
    bind_run_asset,
    ensure_collection,
    upsert_asset,
)
from vfieval.media_items import (
    bind_compare_input,
    bind_run_source,
    ensure_canonical_gt_item,
    register_external_prediction,
    register_model_prediction,
)


TRIGGER_NAMES = (
    "trg_media_item_members_validate_insert",
    "trg_media_item_members_validate_update",
    "trg_media_items_validate_canonical_update",
    "trg_media_assets_validate_member_role_update",
    "trg_run_media_item_bindings_validate_insert",
    "trg_run_media_item_bindings_validate_update",
)


class MediaIdentityInvariantTests(unittest.TestCase):
    def _workspace(self, tmp: str) -> tuple[WorkspaceConfig, Database]:
        workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
        workspace.ensure()
        db = Database(workspace.db_path)
        db.init()
        return workspace, db

    def _run(
        self,
        db: Database,
        workspace: WorkspaceConfig,
        name: str,
        *,
        run_type: str,
    ) -> int:
        model_id = db.upsert_model(f"model-{name}", "dummy", None, 8, 8, {})
        dataset_root = workspace.root / f"dataset-{name}"
        dataset_root.mkdir(parents=True, exist_ok=True)
        dataset_id = db.create_dataset(
            f"dataset-{name}",
            str(dataset_root),
            has_gt=True,
        )
        run_id = db.create_run(
            name,
            model_id,
            dataset_id,
            8,
            8,
            1,
            "cpu",
            "fp32",
            [],
            metadata={"run_type": run_type},
            create_inference_job=False,
        )
        with db.connection() as conn:
            conn.execute(
                "UPDATE runs SET status = 'completed' WHERE id = ?",
                (run_id,),
            )
        return run_id

    def _asset(
        self,
        db: Database,
        workspace: WorkspaceConfig,
        *,
        collection_id: int,
        source_key: str,
        role: str,
        source_kind: str,
    ) -> dict:
        path = workspace.root / "test-media" / f"{source_key.replace(':', '-')}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source_key.encode("utf-8"))
        return upsert_asset(
            db,
            collection_id=collection_id,
            source_key=source_key,
            source_kind=source_kind,
            media_kind="video",
            role=role,
            display_name=path.name,
            original_name=path.name,
            storage_path=path,
            content_sha256=source_key,
            size_bytes=path.stat().st_size,
            frame_count=3,
            width=8,
            height=8,
            fps=24.0,
        )

    def _fixture(self, tmp: str) -> dict:
        workspace, db = self._workspace(tmp)
        collection = ensure_collection(
            db,
            "GT",
            "gt",
            {"source_kind": "folder"},
        )
        gt_a = self._asset(
            db,
            workspace,
            collection_id=int(collection["id"]),
            source_key="folder:a.mp4",
            role="gt",
            source_kind="folder",
        )
        gt_b = self._asset(
            db,
            workspace,
            collection_id=int(collection["id"]),
            source_key="folder:b.mp4",
            role="gt",
            source_kind="folder",
        )
        item_a = ensure_canonical_gt_item(db, int(gt_a["id"]))
        item_b = ensure_canonical_gt_item(db, int(gt_b["id"]))
        canonical_a = db.get(
            """
            SELECT id FROM media_item_members
            WHERE item_id = ? AND member_role = 'canonical_gt'
            """,
            (int(item_a["id"]),),
        )
        canonical_b = db.get(
            """
            SELECT id FROM media_item_members
            WHERE item_id = ? AND member_role = 'canonical_gt'
            """,
            (int(item_b["id"]),),
        )
        assert canonical_a is not None and canonical_b is not None

        source_run = self._run(
            db,
            workspace,
            "source",
            run_type="model_inference",
        )
        pred_collection = ensure_collection(
            db,
            "Run outputs",
            "run-outputs",
            {"source_kind": "run_artifact"},
        )
        pred = self._asset(
            db,
            workspace,
            collection_id=int(pred_collection["id"]),
            source_key=f"run_artifact:{source_run}:pred",
            role="pred",
            source_kind="run_artifact",
        )
        bind_run_source(
            db,
            source_run,
            int(item_a["id"]),
            video_name="a.mp4",
        )
        bind_run_asset(
            db,
            source_run,
            int(pred["id"]),
            "pred",
            video_name="a.mp4",
        )
        pred_member = register_model_prediction(
            db,
            source_run,
            int(item_a["id"]),
            int(pred["id"]),
            metadata={"video_name": "a.mp4"},
        )

        upload_collection = ensure_collection(
            db,
            "Uploads",
            "uploads",
            {"source_kind": "upload"},
        )
        external = self._asset(
            db,
            workspace,
            collection_id=int(upload_collection["id"]),
            source_key="upload:external",
            role="pred",
            source_kind="upload",
        )
        external_member = register_external_prediction(
            db,
            int(item_a["id"]),
            int(external["id"]),
            method_key="external:test",
        )

        compare_run = self._run(
            db,
            workspace,
            "compare",
            run_type="video_compare",
        )
        bind_compare_input(
            db,
            compare_run,
            int(item_a["id"]),
            int(canonical_a["id"]),
            binding_role="compare_gt",
            slot="GT",
        )
        pred_binding = bind_compare_input(
            db,
            compare_run,
            int(item_a["id"]),
            int(pred_member["id"]),
            binding_role="compare_pred",
            slot="A",
        )
        return {
            "workspace": workspace,
            "db": db,
            "collection_id": int(collection["id"]),
            "item_a": item_a,
            "item_b": item_b,
            "gt_a": gt_a,
            "gt_b": gt_b,
            "canonical_a_id": int(canonical_a["id"]),
            "canonical_b_id": int(canonical_b["id"]),
            "source_run": source_run,
            "pred_member": pred_member,
            "external_member": external_member,
            "compare_run": compare_run,
            "pred_binding": pred_binding,
        }

    def test_member_and_canonical_asset_invariants_reject_direct_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(tmp)
            db = fixture["db"]
            now = utc_ts()
            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "member asset role mismatch",
            ):
                with db.connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO media_item_members(
                            item_id, asset_id, member_role, producer_kind,
                            producer_run_id, method_key, reusable_as_pred,
                            temporal_mapping_json, spatial_origin_json, state,
                            metadata_json, created_at, updated_at, deleted_at
                        ) VALUES (
                            ?, ?, 'evaluation_gt', 'evaluation_package',
                            NULL, '', 0, '{}', '{}', 'ready', '{}', ?, ?, NULL
                        )
                        """,
                        (
                            int(fixture["item_a"]["id"]),
                            int(fixture["pred_member"]["asset_id"]),
                            now,
                            now,
                        ),
                    )

            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "canonical GT member asset mismatch",
            ):
                with db.connection() as conn:
                    conn.execute(
                        """
                        UPDATE media_item_members
                        SET asset_id = ?
                        WHERE id = ?
                        """,
                        (
                            int(fixture["gt_b"]["id"]),
                            int(fixture["canonical_a_id"]),
                        ),
                    )

            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "canonical GT item asset mismatch",
            ):
                with db.connection() as conn:
                    conn.execute(
                        """
                        UPDATE media_items
                        SET canonical_gt_asset_id = ?
                        WHERE id = ?
                        """,
                        (
                            int(fixture["gt_b"]["id"]),
                            int(fixture["item_a"]["id"]),
                        ),
                    )

            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "asset role used by member",
            ):
                with db.connection() as conn:
                    conn.execute(
                        "UPDATE media_assets SET role = 'pred' WHERE id = ?",
                        (int(fixture["gt_a"]["id"]),),
                    )

    def test_binding_item_role_slot_and_identity_invariants_reject_direct_sql(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(tmp)
            db = fixture["db"]
            now = utc_ts()
            common = (
                int(fixture["compare_run"]),
                int(fixture["item_a"]["id"]),
                int(fixture["canonical_b_id"]),
                int(fixture["canonical_b_id"]),
                now,
                now,
            )
            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "belongs to another Item",
            ):
                with db.connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO run_media_item_bindings(
                            run_id, item_id, binding_role, slot,
                            original_member_id, active_member_id,
                            metadata_json, created_at, updated_at
                        ) VALUES (?, ?, 'source', 'cross-item', ?, ?, '{}', ?, ?)
                        """,
                        common,
                    )

            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "original binding role mismatch",
            ):
                with db.connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO run_media_item_bindings(
                            run_id, item_id, binding_role, slot,
                            original_member_id, active_member_id,
                            metadata_json, created_at, updated_at
                        ) VALUES (?, ?, 'source', 'wrong-role', ?, ?, '{}', ?, ?)
                        """,
                        (
                            int(fixture["compare_run"]),
                            int(fixture["item_a"]["id"]),
                            int(fixture["pred_member"]["id"]),
                            int(fixture["pred_member"]["id"]),
                            now,
                            now,
                        ),
                    )

            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "binding slot mismatch",
            ):
                with db.connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO run_media_item_bindings(
                            run_id, item_id, binding_role, slot,
                            original_member_id, active_member_id,
                            metadata_json, created_at, updated_at
                        ) VALUES (?, ?, 'pred_output', 'not-empty', ?, ?, '{}', ?, ?)
                        """,
                        (
                            int(fixture["source_run"]),
                            int(fixture["item_a"]["id"]),
                            int(fixture["pred_member"]["id"]),
                            int(fixture["pred_member"]["id"]),
                            now,
                            now,
                        ),
                    )

            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "preserve original identity",
            ):
                with db.connection() as conn:
                    conn.execute(
                        """
                        UPDATE run_media_item_bindings
                        SET active_member_id = ?
                        WHERE id = ?
                        """,
                        (
                            int(fixture["external_member"]["id"]),
                            int(fixture["pred_binding"]["id"]),
                        ),
                    )

    def test_only_snapshot_of_original_compare_member_can_replace_active_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(tmp)
            db = fixture["db"]
            workspace = fixture["workspace"]
            snapshot = self._asset(
                db,
                workspace,
                collection_id=int(fixture["collection_id"]),
                source_key="run_artifact:compare:snapshot",
                role="pred",
                source_kind="run_artifact",
            )
            now = utc_ts()
            with db.connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO media_item_members(
                        item_id, asset_id, member_role, producer_kind,
                        producer_run_id, method_key, reusable_as_pred,
                        temporal_mapping_json, spatial_origin_json, state,
                        metadata_json, created_at, updated_at, deleted_at
                    ) VALUES (
                        ?, ?, 'compare_snapshot', 'video_compare',
                        ?, 'run:test', 0, '{}', '{}', 'ready', ?, ?, ?, NULL
                    )
                    """,
                    (
                        int(fixture["item_a"]["id"]),
                        int(snapshot["id"]),
                        int(fixture["compare_run"]),
                        json.dumps(
                            {"source_member_id": int(fixture["pred_member"]["id"])}
                        ),
                        now,
                        now,
                    ),
                )
                snapshot_member_id = int(cursor.lastrowid)
                conn.execute(
                    """
                    UPDATE run_media_item_bindings
                    SET active_member_id = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        snapshot_member_id,
                        now,
                        int(fixture["pred_binding"]["id"]),
                    ),
                )

            active = db.get(
                "SELECT active_member_id FROM run_media_item_bindings WHERE id = ?",
                (int(fixture["pred_binding"]["id"]),),
            )
            assert active is not None
            self.assertEqual(int(active["active_member_id"]), snapshot_member_id)

            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "active binding member mismatch",
            ):
                with db.connection() as conn:
                    conn.execute(
                        """
                        UPDATE media_item_members
                        SET metadata_json = ?
                        WHERE id = ?
                        """,
                        (
                            json.dumps(
                                {
                                    "source_member_id": int(
                                        fixture["external_member"]["id"]
                                    )
                                }
                            ),
                            snapshot_member_id,
                        ),
                    )

    def test_migration_reports_historical_inconsistency_without_repairing_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(tmp)
            db = fixture["db"]
            with db.connection() as conn:
                for name in TRIGGER_NAMES:
                    conn.execute(f"DROP TRIGGER {name}")
                conn.execute(
                    """
                    UPDATE run_media_item_bindings
                    SET item_id = ?
                    WHERE id = ?
                    """,
                    (
                        int(fixture["item_b"]["id"]),
                        int(fixture["pred_binding"]["id"]),
                    ),
                )
                conn.execute(
                    "DELETE FROM schema_migrations WHERE version = ?",
                    (LATEST_SCHEMA_VERSION,),
                )

            with self.assertRaisesRegex(
                RuntimeError,
                "binding_member_item_mismatch",
            ):
                db.init()

            damaged = db.get(
                "SELECT item_id FROM run_media_item_bindings WHERE id = ?",
                (int(fixture["pred_binding"]["id"]),),
            )
            assert damaged is not None
            self.assertEqual(
                int(damaged["item_id"]),
                int(fixture["item_b"]["id"]),
            )
            self.assertIsNone(
                db.get(
                    "SELECT 1 FROM schema_migrations WHERE version = ?",
                    (LATEST_SCHEMA_VERSION,),
                )
            )

    def test_consistent_historical_rows_upgrade_and_install_all_triggers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._fixture(tmp)
            db = fixture["db"]
            with db.connection() as conn:
                for name in TRIGGER_NAMES:
                    conn.execute(f"DROP TRIGGER {name}")
                conn.execute(
                    "DELETE FROM schema_migrations WHERE version = ?",
                    (LATEST_SCHEMA_VERSION,),
                )

            db.init()

            self.assertIsNotNone(
                db.get(
                    "SELECT 1 FROM schema_migrations WHERE version = ?",
                    (LATEST_SCHEMA_VERSION,),
                )
            )
            trigger_rows = db.query(
                f"""
                SELECT name FROM sqlite_master
                WHERE type = 'trigger'
                  AND name IN ({", ".join("?" for _ in TRIGGER_NAMES)})
                """,
                TRIGGER_NAMES,
            )
            self.assertEqual(
                {str(row["name"]) for row in trigger_rows},
                set(TRIGGER_NAMES),
            )
            binding = db.get(
                """
                SELECT item_id, original_member_id, active_member_id
                FROM run_media_item_bindings
                WHERE id = ?
                """,
                (int(fixture["pred_binding"]["id"]),),
            )
            assert binding is not None
            self.assertEqual(
                int(binding["item_id"]),
                int(fixture["item_a"]["id"]),
            )
            self.assertEqual(
                int(binding["original_member_id"]),
                int(binding["active_member_id"]),
            )


if __name__ == "__main__":
    unittest.main()
