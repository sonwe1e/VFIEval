from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import vfieval.evaluations_v2 as evaluations_v2_module
from vfieval.config import WorkspaceConfig
from vfieval.db import Database, utc_ts
from vfieval.evaluations_v2 import (
    CAMPAIGN_V2_SCHEMA,
    CAMPAIGN_V2_SCHEMA_VERSION,
    ensure_v2_schema,
)


class CampaignV2MigrationTests(unittest.TestCase):
    def test_historical_campaign_vote_fixture_survives_coordinated_upgrade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceConfig.from_root(Path(directory) / ".vfieval")
            db = Database(workspace.db_path)
            db.init()
            with db.connection() as conn:
                conn.executescript(CAMPAIGN_V2_SCHEMA)
                now = utc_ts()
                asset_ids: list[int] = []
                for source_key, role, display_name in (
                    ("historical-gt", "gt", "Historical GT"),
                    ("historical-a", "pred", "Historical Pred A"),
                    ("historical-b", "pred", "Historical Pred B"),
                ):
                    asset_ids.append(
                        int(
                            conn.execute(
                                """
                                INSERT INTO media_assets(
                                    source_key, source_kind, media_kind, role,
                                    display_name, storage_path, created_at, updated_at
                                ) VALUES (?, 'folder', 'video', ?, ?, ?, ?, ?)
                                """,
                                (
                                    source_key,
                                    role,
                                    display_name,
                                    str((Path(directory) / f"{source_key}.mp4").resolve()),
                                    now,
                                    now,
                                ),
                            ).lastrowid
                        )
                    )
                campaign_id = int(
                    conn.execute(
                        """
                        INSERT INTO evaluation_campaigns_v2(
                            public_token, name, public_title, status, target_votes,
                            seed, config_json, created_at, updated_at, published_at
                        ) VALUES (
                            'historical-public-token', 'Historical campaign',
                            'Historical campaign', 'published', 3, 17, '{}', ?, ?, ?
                        )
                        """,
                        (now, now, now),
                    ).lastrowid
                )
                method_ids: list[int] = []
                for slot, label in (("a", "Historical A"), ("b", "Historical B")):
                    method_ids.append(
                        int(
                            conn.execute(
                                """
                                INSERT INTO evaluation_methods_v2(
                                    campaign_id, slot, source_kind, label_snapshot,
                                    source_spec_json, created_at
                                ) VALUES (?, ?, 'upload', ?, '{}', ?)
                                """,
                                (campaign_id, slot, label, now),
                            ).lastrowid
                        )
                    )
                item_id = int(
                    conn.execute(
                        """
                        INSERT INTO evaluation_items_v2(
                            campaign_id, video_name, reference_source_asset_id,
                            alignment_json, created_at
                        ) VALUES (?, 'historical.mp4', ?, '{}', ?)
                        """,
                        (campaign_id, asset_ids[0], now),
                    ).lastrowid
                )
                binding_ids: list[int] = []
                for method_id, asset_id in zip(method_ids, asset_ids[1:], strict=True):
                    binding_ids.append(
                        int(
                            conn.execute(
                                """
                                INSERT INTO evaluation_bindings_v2(
                                    item_id, method_id, source_asset_id, state,
                                    alignment_json, created_at, updated_at
                                ) VALUES (?, ?, ?, 'ready', '{}', ?, ?)
                                """,
                                (item_id, method_id, asset_id, now, now),
                            ).lastrowid
                        )
                    )
                task_id = int(
                    conn.execute(
                        """
                        INSERT INTO evaluation_tasks_v2(
                            task_token, campaign_id, item_id, binding_a_id,
                            binding_b_id, state, created_at
                        ) VALUES (
                            'historical-task', ?, ?, ?, ?, 'ready', ?
                        )
                        """,
                        (campaign_id, item_id, binding_ids[0], binding_ids[1], now),
                    ).lastrowid
                )
                conn.execute(
                    """
                    INSERT INTO evaluators(
                        id, display_name, metadata_json, created_at, updated_at, last_seen_at
                    ) VALUES ('historical-voter', 'Historical voter', '{}', ?, ?, ?)
                    """,
                    (now, now, now),
                )
                assignment_id = int(
                    conn.execute(
                        """
                        INSERT INTO evaluation_assignments_v2(
                            assignment_token, task_id, evaluator_id, state,
                            side_swap, lease_expires_at, created_at, updated_at
                        ) VALUES (
                            'historical-assignment', ?, 'historical-voter',
                            'voted', 0, ?, ?, ?
                        )
                        """,
                        (task_id, now + 3600, now, now),
                    ).lastrowid
                )
                vote_id = int(
                    conn.execute(
                        """
                        INSERT INTO evaluation_votes_v2(
                            task_id, evaluator_id, assignment_id, choice,
                            preferred_method_id, reasons_json, confidence, note,
                            presentation_json, created_at, updated_at
                        ) VALUES (
                            ?, 'historical-voter', ?, 'left', ?, '[]', 'high',
                            'historical note', '{}', ?, ?
                        )
                        """,
                        (task_id, assignment_id, method_ids[0], now, now),
                    ).lastrowid
                )

            ensure_v2_schema(db)

            with db.connection() as conn:
                vote = conn.execute(
                    """
                    SELECT vote.choice, vote.note, campaign.public_token
                    FROM evaluation_votes_v2 vote
                    JOIN evaluation_tasks_v2 task ON task.id = vote.task_id
                    JOIN evaluation_campaigns_v2 campaign ON campaign.id = task.campaign_id
                    WHERE vote.id = ?
                    """,
                    (vote_id,),
                ).fetchone()
                applied = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?",
                    (CAMPAIGN_V2_SCHEMA_VERSION,),
                ).fetchone()
                violations = conn.execute("PRAGMA foreign_key_check").fetchall()

            self.assertEqual(
                (vote["choice"], vote["note"], vote["public_token"]),
                ("left", "historical note", "historical-public-token"),
            )
            self.assertIsNotNone(applied)
            self.assertEqual(violations, [])
            self.assertEqual(
                len(list(workspace.root.glob("backups/*_campaign_v2/vfieval.sqlite"))),
                1,
            )

    def test_historical_media_asset_constraint_preserves_rows_and_foreign_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceConfig.from_root(Path(directory) / ".vfieval")
            db = Database(workspace.db_path)
            db.init()
            with db.connection() as conn:
                conn.execute("PRAGMA foreign_keys=OFF")
                conn.execute("PRAGMA legacy_alter_table=ON")
                conn.executescript(
                    """
                    ALTER TABLE media_assets RENAME TO media_assets_current;
                    CREATE TABLE media_assets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        collection_id INTEGER REFERENCES media_collections(id) ON DELETE SET NULL,
                        source_key TEXT NOT NULL UNIQUE,
                        source_kind TEXT NOT NULL CHECK(source_kind IN (
                            'folder', 'upload', 'run_artifact'
                        )),
                        media_kind TEXT NOT NULL CHECK(media_kind IN ('video', 'frame_sequence')),
                        role TEXT NOT NULL CHECK(role IN ('source', 'gt', 'pred')),
                        display_name TEXT NOT NULL,
                        original_name TEXT NOT NULL DEFAULT '',
                        state TEXT NOT NULL DEFAULT 'ready',
                        content_sha256 TEXT,
                        size_bytes INTEGER NOT NULL DEFAULT 0,
                        storage_path TEXT NOT NULL,
                        mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
                        frame_count INTEGER NOT NULL DEFAULT 0,
                        width INTEGER NOT NULL DEFAULT 0,
                        height INTEGER NOT NULL DEFAULT 0,
                        fps REAL,
                        provenance_json TEXT NOT NULL DEFAULT '{}',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        deleted_at REAL,
                        UNIQUE(collection_id, display_name)
                    );
                    DROP TABLE media_assets_current;
                    """
                )
                now = utc_ts()
                asset_id = int(
                    conn.execute(
                        """
                        INSERT INTO media_assets(
                            source_key, source_kind, media_kind, role, display_name,
                            storage_path, created_at, updated_at
                        ) VALUES ('legacy-source', 'folder', 'video', 'gt', 'Legacy GT', ?, ?, ?)
                        """,
                        (str((Path(directory) / "legacy.mp4").resolve()), now, now),
                    ).lastrowid
                )
                conn.execute("PRAGMA legacy_alter_table=OFF")
                conn.execute("PRAGMA foreign_keys=ON")

            ensure_v2_schema(db)

            with db.connection() as conn:
                asset = conn.execute(
                    "SELECT source_key, role FROM media_assets WHERE id = ?",
                    (asset_id,),
                ).fetchone()
                table_sql = str(
                    conn.execute(
                        "SELECT sql FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'media_assets'"
                    ).fetchone()["sql"]
                )
                violations = conn.execute("PRAGMA foreign_key_check").fetchall()

            self.assertEqual((asset["source_key"], asset["role"]), ("legacy-source", "gt"))
            self.assertIn("'evaluation_package'", table_sql)
            self.assertEqual(violations, [])

    def test_historical_method_slot_schema_is_backed_up_and_upgraded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceConfig.from_root(Path(directory) / ".vfieval")
            db = Database(workspace.db_path)
            db.init()
            with db.connection() as conn:
                conn.executescript(CAMPAIGN_V2_SCHEMA)
                conn.execute("PRAGMA foreign_keys=OFF")
                conn.execute("PRAGMA legacy_alter_table=ON")
                conn.executescript(
                    """
                    ALTER TABLE evaluation_methods_v2
                    RENAME TO evaluation_methods_v2_current;
                    CREATE TABLE evaluation_methods_v2 (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        campaign_id INTEGER NOT NULL
                            REFERENCES evaluation_campaigns_v2(id) ON DELETE CASCADE,
                        slot TEXT NOT NULL CHECK(slot IN ('a', 'b')),
                        source_kind TEXT NOT NULL CHECK(source_kind IN ('run_track', 'upload')),
                        source_run_id INTEGER,
                        source_track_label TEXT NOT NULL DEFAULT '',
                        label_snapshot TEXT NOT NULL,
                        model_snapshot TEXT NOT NULL DEFAULT '',
                        checkpoint_snapshot TEXT NOT NULL DEFAULT '',
                        source_spec_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL,
                        UNIQUE(campaign_id, slot)
                    );
                    DROP TABLE evaluation_methods_v2_current;
                    """
                )
                now = utc_ts()
                campaign_id = int(
                    conn.execute(
                        """
                        INSERT INTO evaluation_campaigns_v2(
                            public_token, name, public_title, status, target_votes,
                            seed, config_json, created_at, updated_at
                        ) VALUES ('legacy-token', 'Legacy', 'Legacy', 'draft', 3, 7, '{}', ?, ?)
                        """,
                        (now, now),
                    ).lastrowid
                )
                conn.execute(
                    """
                    INSERT INTO evaluation_methods_v2(
                        campaign_id, slot, source_kind, source_track_label,
                        label_snapshot, source_spec_json, created_at
                    ) VALUES (?, 'a', 'upload', '', 'Method A', '{}', ?)
                    """,
                    (campaign_id, now),
                )
                conn.execute("PRAGMA legacy_alter_table=OFF")
                conn.execute("PRAGMA foreign_keys=ON")

            ensure_v2_schema(db)

            with db.connection() as conn:
                method = conn.execute(
                    "SELECT slot, label_snapshot FROM evaluation_methods_v2 WHERE campaign_id = ?",
                    (campaign_id,),
                ).fetchone()
                table_sql = str(
                    conn.execute(
                        "SELECT sql FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'evaluation_methods_v2'"
                    ).fetchone()["sql"]
                )
                applied = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?",
                    (CAMPAIGN_V2_SCHEMA_VERSION,),
                ).fetchone()
                violations = conn.execute("PRAGMA foreign_key_check").fetchall()

            self.assertEqual((method["slot"], method["label_snapshot"]), ("a", "Method A"))
            self.assertIn("'c'", table_sql)
            self.assertIsNotNone(applied)
            self.assertEqual(violations, [])
            backups = list(workspace.root.glob("backups/*_campaign_v2/vfieval.sqlite"))
            self.assertEqual(len(backups), 1)

            ensure_v2_schema(db)
            self.assertEqual(
                list(workspace.root.glob("backups/*_campaign_v2/vfieval.sqlite")),
                backups,
            )

    def test_failed_campaign_migration_does_not_record_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceConfig.from_root(Path(directory) / ".vfieval")
            db = Database(workspace.db_path)
            db.init()
            with db.connection() as conn:
                conn.executescript(CAMPAIGN_V2_SCHEMA)

            with patch(
                "vfieval.evaluations_v2._apply_v2_schema",
                side_effect=RuntimeError("injected migration failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected migration failure"):
                    ensure_v2_schema(db)

            with db.connection() as conn:
                applied = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?",
                    (CAMPAIGN_V2_SCHEMA_VERSION,),
                ).fetchone()
            self.assertIsNone(applied)
            self.assertEqual(
                len(list(workspace.root.glob("backups/*_campaign_v2/vfieval.sqlite"))),
                1,
            )

            ensure_v2_schema(db)
            with db.connection() as conn:
                recovered = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?",
                    (CAMPAIGN_V2_SCHEMA_VERSION,),
                ).fetchone()
                violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            self.assertIsNotNone(recovered)
            self.assertEqual(violations, [])

    def test_mid_migration_failure_rolls_back_schema_data_and_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceConfig.from_root(Path(directory) / ".vfieval")
            db = Database(workspace.db_path)
            db.init()
            with db.connection() as conn:
                conn.execute("PRAGMA foreign_keys=OFF")
                conn.execute("PRAGMA legacy_alter_table=ON")
                conn.executescript(
                    """
                    ALTER TABLE media_assets RENAME TO media_assets_current;
                    CREATE TABLE media_assets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        collection_id INTEGER REFERENCES media_collections(id) ON DELETE SET NULL,
                        source_key TEXT NOT NULL UNIQUE,
                        source_kind TEXT NOT NULL CHECK(source_kind IN (
                            'folder', 'upload', 'run_artifact'
                        )),
                        media_kind TEXT NOT NULL CHECK(media_kind IN ('video', 'frame_sequence')),
                        role TEXT NOT NULL CHECK(role IN ('source', 'gt', 'pred')),
                        display_name TEXT NOT NULL,
                        original_name TEXT NOT NULL DEFAULT '',
                        state TEXT NOT NULL DEFAULT 'ready',
                        content_sha256 TEXT,
                        size_bytes INTEGER NOT NULL DEFAULT 0,
                        storage_path TEXT NOT NULL,
                        mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
                        frame_count INTEGER NOT NULL DEFAULT 0,
                        width INTEGER NOT NULL DEFAULT 0,
                        height INTEGER NOT NULL DEFAULT 0,
                        fps REAL,
                        provenance_json TEXT NOT NULL DEFAULT '{}',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        deleted_at REAL,
                        UNIQUE(collection_id, display_name)
                    );
                    DROP TABLE media_assets_current;
                    """
                )
                now = utc_ts()
                asset_id = int(
                    conn.execute(
                        """
                        INSERT INTO media_assets(
                            source_key, source_kind, media_kind, role, display_name,
                            storage_path, created_at, updated_at
                        ) VALUES (
                            'atomic-rollback-source', 'folder', 'video', 'gt',
                            'Atomic rollback source', ?, ?, ?
                        )
                        """,
                        (
                            str((Path(directory) / "atomic-rollback.mp4").resolve()),
                            now,
                            now,
                        ),
                    ).lastrowid
                )
                conn.execute("PRAGMA legacy_alter_table=OFF")
                conn.execute("PRAGMA foreign_keys=ON")
                legacy_sql = str(
                    conn.execute(
                        "SELECT sql FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'media_assets'"
                    ).fetchone()["sql"]
                )
            self.assertNotIn("evaluation_package", legacy_sql)

            def fail_after_first_schema_statement(
                conn: sqlite3.Connection,
                script: str,
            ) -> None:
                statement = next(iter(evaluations_v2_module._iter_sql_statements(script)))
                conn.execute(statement)
                raise RuntimeError("injected mid-migration failure")

            with patch(
                "vfieval.evaluations_v2._execute_schema_statements",
                side_effect=fail_after_first_schema_statement,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "injected mid-migration failure",
                ):
                    ensure_v2_schema(db)

            with db.connection() as conn:
                rolled_back_sql = str(
                    conn.execute(
                        "SELECT sql FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'media_assets'"
                    ).fetchone()["sql"]
                )
                asset = conn.execute(
                    "SELECT source_key FROM media_assets WHERE id = ?",
                    (asset_id,),
                ).fetchone()
                campaign_table = conn.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'evaluation_campaigns_v2'"
                ).fetchone()
                temporary_tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ("
                    "'media_assets_before_evaluation_v2', "
                    "'evaluation_methods_v2_slots_upgrade'"
                    ")"
                ).fetchall()
                applied = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?",
                    (CAMPAIGN_V2_SCHEMA_VERSION,),
                ).fetchone()

            self.assertEqual(rolled_back_sql, legacy_sql)
            self.assertEqual(asset["source_key"], "atomic-rollback-source")
            self.assertIsNone(campaign_table)
            self.assertEqual(temporary_tables, [])
            self.assertIsNone(applied)

            backups = list(workspace.root.glob("backups/*_campaign_v2/vfieval.sqlite"))
            self.assertEqual(len(backups), 1)
            with Database(backups[0]).connection() as backup_conn:
                backup_asset = backup_conn.execute(
                    "SELECT source_key FROM media_assets WHERE id = ?",
                    (asset_id,),
                ).fetchone()
                backup_sql = str(
                    backup_conn.execute(
                        "SELECT sql FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'media_assets'"
                    ).fetchone()["sql"]
                )
            self.assertEqual(backup_asset["source_key"], "atomic-rollback-source")
            self.assertEqual(backup_sql, legacy_sql)

            ensure_v2_schema(db)
            with db.connection() as conn:
                upgraded_sql = str(
                    conn.execute(
                        "SELECT sql FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'media_assets'"
                    ).fetchone()["sql"]
                )
                recovered_asset = conn.execute(
                    "SELECT source_key FROM media_assets WHERE id = ?",
                    (asset_id,),
                ).fetchone()
                recovered = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?",
                    (CAMPAIGN_V2_SCHEMA_VERSION,),
                ).fetchone()
                violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            self.assertIn("evaluation_package", upgraded_sql)
            self.assertEqual(recovered_asset["source_key"], "atomic-rollback-source")
            self.assertIsNotNone(recovered)
            self.assertEqual(violations, [])

    def test_recreated_database_at_same_path_is_not_skipped_by_process_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceConfig.from_root(Path(directory) / ".vfieval")
            workspace.ensure()
            first = Database(workspace.db_path)
            first.init()
            ensure_v2_schema(first)

            workspace.db_path.unlink()
            second = Database(workspace.db_path)
            second.init()
            ensure_v2_schema(second)

            with second.connection() as conn:
                campaign_table = conn.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'evaluation_campaigns_v2'"
                ).fetchone()
                migration = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?",
                    (CAMPAIGN_V2_SCHEMA_VERSION,),
                ).fetchone()
            self.assertIsNotNone(campaign_table)
            self.assertIsNotNone(migration)


if __name__ == "__main__":
    unittest.main()
