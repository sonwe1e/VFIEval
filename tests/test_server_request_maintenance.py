from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import MagicMock, patch

from vfieval.catalog_sync import CatalogSyncCoordinator
from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.media_assets import ensure_collection, upsert_asset
from vfieval.run_cleanup import RunCleanupService
from vfieval.server import _make_handler


class ServerRequestMaintenanceTests(unittest.TestCase):
    def test_video_group_gets_are_catalog_snapshots_and_thumbnail_is_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"VFIEVAL_PROJECT_ROOT": tmp},
            clear=False,
        ):
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            collection = ensure_collection(
                db,
                "videos/anime",
                "videos-anime",
                {"source_kind": "folder", "video_group": "anime"},
            )
            upsert_asset(
                db,
                collection_id=int(collection["id"]),
                source_key="folder:anime/clip.mp4",
                source_kind="folder",
                media_kind="video",
                role="gt",
                display_name="clip.mp4",
                original_name="clip.mp4",
                storage_path=Path(tmp) / "videos" / "anime" / "clip.mp4",
                state="ready",
                content_sha256="a" * 64,
                size_bytes=123,
                frame_count=7,
                width=1920,
                height=1080,
                fps=24.0,
                provenance={"video_group": "anime", "video": "clip.mp4"},
                metadata={"duration_seconds": 7 / 24, "frame_count_source": "container"},
            )
            coordinator = CatalogSyncCoordinator(
                db,
                workspace,
                sync_callback=lambda *_args: {"ok": True},
            )
            cleanup_service = MagicMock(spec=RunCleanupService)
            cleanup_service.ensure_backfilled.return_value = {}
            handler = _make_handler(
                db,
                workspace,
                cleanup_service=cleanup_service,
                catalog_sync=coordinator,
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with patch(
                    "vfieval.file_inputs.inspect_video",
                    side_effect=AssertionError("catalog GET must not inspect source video"),
                ), patch(
                    "vfieval.file_inputs.file_sha256",
                    side_effect=AssertionError("catalog GET must not hash source video"),
                ), patch(
                    "vfieval.file_inputs.ensure_video_thumbnail",
                    side_effect=AssertionError("catalog GET must not generate thumbnails"),
                ):
                    with urllib.request.urlopen(
                        f"{base_url}/api/video-groups?summary=1", timeout=10
                    ) as response:
                        groups = json.loads(response.read().decode("utf-8"))
                    with urllib.request.urlopen(
                        f"{base_url}/api/video-groups/anime/videos", timeout=10
                    ) as response:
                        page = json.loads(response.read().decode("utf-8"))
                self.assertEqual(groups[0]["video_count"], 1)
                self.assertEqual(page["videos"][0]["frame_count"], 7)
                self.assertEqual(
                    page["videos"][0]["thumbnail_url"],
                    f"/api/media/assets/{page['videos'][0]['asset_id']}/thumbnail",
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_get_static_media_and_runs_do_not_process_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"VFIEVAL_PROJECT_ROOT": tmp},
            clear=False,
        ):
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            cleanup_service = MagicMock(spec=RunCleanupService)
            cleanup_service.ensure_backfilled.return_value = {}

            handler = _make_handler(db, workspace, cleanup_service=cleanup_service)
            cleanup_service.process_pending.assert_not_called()
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                for path in (
                    "/api/health",
                    "/app.js",
                    "/api/media/assets?page=1&page_size=1",
                    "/api/runs",
                ):
                    with self.subTest(path=path):
                        with urllib.request.urlopen(f"{base_url}{path}", timeout=10) as response:
                            self.assertEqual(response.status, 200)
                            response.read()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            cleanup_service.process_pending.assert_not_called()

    def test_catalog_sync_metric_refresh_and_run_paging_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"VFIEVAL_PROJECT_ROOT": tmp},
            clear=False,
        ):
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("model", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("dataset", tmp, True)
            for index in range(3):
                db.create_run(
                    f"run-{index}",
                    model_id,
                    dataset_id,
                    8,
                    8,
                    1,
                    "cpu",
                    "fp32",
                    [],
                )
            sync_calls: list[bool] = []
            coordinator = CatalogSyncCoordinator(
                db,
                workspace,
                sync_callback=lambda _db, _workspace, include_runs: sync_calls.append(
                    bool(include_runs)
                )
                or {"ok": True},
            )
            cleanup_service = MagicMock(spec=RunCleanupService)
            cleanup_service.ensure_backfilled.return_value = {}
            with patch(
                "vfieval.server.metrics_health",
                return_value={"asset_root": "metrics", "metrics": {}},
            ) as health:
                handler = _make_handler(
                    db,
                    workspace,
                    cleanup_service=cleanup_service,
                    catalog_sync=coordinator,
                )
                server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                try:
                    with urllib.request.urlopen(
                        f"{base_url}/api/media/collections", timeout=10
                    ) as response:
                        response.read()
                    self.assertEqual(sync_calls, [])

                    request = urllib.request.Request(
                        f"{base_url}/api/media/sync",
                        data=b'{"include_runs":true}',
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=10) as response:
                        self.assertEqual(response.status, 202)
                        response.read()
                    completed = coordinator.wait(2)
                    self.assertEqual(completed["state"], "completed")
                    self.assertEqual(sync_calls, [True])

                    with urllib.request.urlopen(
                        f"{base_url}/api/media/sync/status", timeout=10
                    ) as response:
                        status = json.loads(response.read().decode("utf-8"))
                    self.assertEqual(status["catalog_revision"], 1)

                    with urllib.request.urlopen(
                        f"{base_url}/api/metrics/health?refresh=1", timeout=10
                    ) as response:
                        response.read()
                    health.assert_called_once_with(workspace, refresh=True)

                    with urllib.request.urlopen(
                        f"{base_url}/api/runs?page=2&page_size=2&q=run", timeout=10
                    ) as response:
                        page = json.loads(response.read().decode("utf-8"))
                    self.assertEqual(page["page"], 2)
                    self.assertEqual(page["page_size"], 2)
                    self.assertEqual(page["total"], 3)
                    self.assertEqual(len(page["runs"]), 1)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
