from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v13_test_utils import get_json, post_json
from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.run_cleanup import RunCleanupService
from vfieval.server import _make_handler
from vfieval.video_selection_tokens import (
    VideoSelectionTokenError,
    VideoSelectionTokenExpired,
    create_video_selection_snapshot,
    ensure_video_selection_schema,
    expand_video_selection_payload,
    resolve_video_selection_snapshot,
    video_selection_membership,
    video_selection_snapshot_page,
)


class _StaticCatalog:
    def status(self) -> dict:
        return {"state": "idle", "ready": True}


class VideoSelectionTokenTests(unittest.TestCase):
    def _workspace_with_videos(
        self,
        tmp: str,
        count: int,
        *,
        group: str = "bulk",
    ) -> tuple[WorkspaceConfig, Database, Path]:
        project_root = Path(tmp)
        workspace = WorkspaceConfig.from_root(project_root / ".vfieval")
        workspace.ensure()
        db = Database(workspace.db_path)
        db.init()
        video_dir = project_root / "videos" / group
        video_dir.mkdir(parents=True)
        now = time.time()
        with db.connection() as conn:
            collection_id = int(
                conn.execute(
                    """
                    INSERT INTO media_collections(
                        name, slug, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        f"videos/{group}",
                        f"videos-{group}",
                        json.dumps(
                            {"source_kind": "folder", "video_group": group}
                        ),
                        now,
                        now,
                    ),
                ).lastrowid
            )
            for index in range(count):
                name = f"clip-{index:04d}.mp4"
                path = video_dir / name
                content = f"video-{index}".encode("utf-8")
                path.write_bytes(content)
                stat_result = path.stat()
                conn.execute(
                    """
                    INSERT INTO media_assets(
                        collection_id, source_key, source_kind, media_kind,
                        role, display_name, original_name, state,
                        content_sha256, size_bytes, storage_path, mime_type,
                        frame_count, width, height, fps, metadata_json,
                        created_at, updated_at
                    ) VALUES (
                        ?, ?, 'folder', 'video', 'gt', ?, ?, 'ready',
                        ?, ?, ?, 'video/mp4', 3, 8, 8, 5, ?, ?, ?
                    )
                    """,
                    (
                        collection_id,
                        f"folder:{group}/{name}",
                        name,
                        name,
                        hashlib.sha256(content).hexdigest(),
                        len(content),
                        str(path.resolve()),
                        json.dumps(
                            {"source_mtime_ns": int(stat_result.st_mtime_ns)}
                        ),
                        now,
                        now,
                    ),
                )
        return workspace, db, video_dir

    def test_thousand_item_snapshot_is_persistent_and_pages_only_visible_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"VFIEVAL_PROJECT_ROOT": tmp},
            clear=False,
        ):
            workspace, db, video_dir = self._workspace_with_videos(tmp, 1001)
            created = create_video_selection_snapshot(
                db,
                workspace,
                video_groups=["bulk"],
            )
            token = str(created["video_selection_token"])
            self.assertEqual(created["total"], 1001)
            stored = db.get(
                "SELECT token_hash FROM video_selection_snapshots"
            )
            self.assertIsNotNone(stored)
            self.assertNotEqual(str(stored["token_hash"]), token)

            from vfieval import video_selection_tokens as module

            with patch.object(
                module,
                "_validate_stored_entry",
                wraps=module._validate_stored_entry,
            ) as validate:
                page = video_selection_snapshot_page(
                    db,
                    workspace,
                    token,
                    page=6,
                    page_size=200,
                )
            self.assertEqual(page["total"], 1001)
            self.assertEqual(len(page["videos"]), 1)
            self.assertEqual(validate.call_count, 1)

            visible = video_selection_membership(
                db,
                token,
                video_group="bulk",
                video_names=["clip-0000.mp4", "clip-1000.mp4"],
            )
            self.assertEqual(
                visible["selected_names"],
                {"clip-0000.mp4", "clip-1000.mp4"},
            )

            reopened = Database(workspace.db_path)
            resolved = resolve_video_selection_snapshot(
                reopened,
                workspace,
                token,
                require_non_empty=True,
            )
            self.assertEqual(len(resolved["selected_videos"]), 1001)

            (video_dir / "clip-1000.mp4").write_bytes(b"changed-content")
            with self.assertRaisesRegex(
                VideoSelectionTokenError,
                "content changed|stale",
            ):
                resolve_video_selection_snapshot(reopened, workspace, token)

    def test_single_toggle_does_not_stat_the_entire_base_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"VFIEVAL_PROJECT_ROOT": tmp},
            clear=False,
        ):
            workspace, db, _video_dir = self._workspace_with_videos(tmp, 1001)
            token = create_video_selection_snapshot(
                db,
                workspace,
                video_groups=["bulk"],
            )["video_selection_token"]
            with patch(
                "vfieval.video_selection_tokens._validate_stored_entry",
                side_effect=AssertionError("full snapshot validation is forbidden"),
            ) as validate:
                changed = create_video_selection_snapshot(
                    db,
                    workspace,
                    base_selection_token=token,
                    operation="remove",
                    video_group="bulk",
                    video_names=["clip-0500.mp4"],
                )
            self.assertEqual(validate.call_count, 0)
            self.assertEqual(changed["total"], 1000)

    def test_expiry_conflicts_and_same_path_database_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"VFIEVAL_PROJECT_ROOT": tmp},
            clear=False,
        ):
            workspace, db, _video_dir = self._workspace_with_videos(tmp, 2)
            created = create_video_selection_snapshot(
                db,
                workspace,
                video_groups=["bulk"],
            )
            token = str(created["video_selection_token"])
            expanded = expand_video_selection_payload(
                db,
                workspace,
                {
                    "video_group": "bulk",
                    "video_selection_token": token,
                },
            )
            self.assertEqual(
                expanded["selected_videos"],
                ["clip-0000.mp4", "clip-0001.mp4"],
            )
            self.assertNotIn("video_selection_token", expanded)
            with self.assertRaisesRegex(ValueError, "either"):
                expand_video_selection_payload(
                    db,
                    workspace,
                    {
                        "video_group": "bulk",
                        "video_selection_token": token,
                        "selected_videos": ["clip-0000.mp4"],
                    },
                )
            with self.assertRaisesRegex(
                VideoSelectionTokenError,
                "does not match",
            ):
                expand_video_selection_payload(
                    db,
                    workspace,
                    {
                        "video_group": "other",
                        "video_selection_token": token,
                    },
                )

            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            with db.connection() as conn:
                conn.execute(
                    """
                    UPDATE video_selection_snapshots
                    SET expires_at = ?
                    WHERE token_hash = ?
                    """,
                    (time.time() - 1, digest),
                )
            with self.assertRaises(VideoSelectionTokenExpired):
                resolve_video_selection_snapshot(db, workspace, token)

            with db.connection() as conn:
                conn.execute("DROP TABLE video_selection_snapshot_items")
                conn.execute("DROP TABLE video_selection_snapshots")
            reopened = Database(workspace.db_path)
            ensure_video_selection_schema(reopened)
            self.assertIsNotNone(
                reopened.get(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table' AND name = 'video_selection_snapshots'
                    """
                )
            )

    def test_http_api_never_returns_the_complete_name_or_asset_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"VFIEVAL_PROJECT_ROOT": tmp},
            clear=False,
        ):
            workspace, db, _video_dir = self._workspace_with_videos(tmp, 12)
            cleanup = RunCleanupService(db, workspace)
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                _make_handler(
                    db,
                    workspace,
                    cleanup_service=cleanup,
                    catalog_sync=_StaticCatalog(),
                ),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                created = post_json(
                    base_url,
                    "/api/video-selections",
                    {"video_groups": ["bulk"]},
                )
                token = str(created["video_selection_token"])
                page = get_json(
                    base_url,
                    "/api/video-groups/bulk/videos"
                    f"?page=1&page_size=5&video_selection_token={token}",
                )
                self.assertEqual(page["video_count"], 12)
                self.assertEqual(len(page["videos"]), 5)
                self.assertTrue(all(row["selected"] for row in page["videos"]))
                self.assertNotIn("all_video_names", page)
                self.assertNotIn("filtered_video_names", page)
                self.assertNotIn("asset_ids", page)

                digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
                with db.connection() as conn:
                    conn.execute(
                        """
                        UPDATE video_selection_snapshots
                        SET expires_at = ?
                        WHERE token_hash = ?
                        """,
                        (time.time() - 1, digest),
                    )
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(
                        f"{base_url}/api/video-selections/{token}",
                        timeout=10,
                    )
                self.assertEqual(raised.exception.code, 410)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
