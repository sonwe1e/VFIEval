from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from vfieval.catalog_sync import CatalogSyncCoordinator
from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.diagnostics import health_snapshot
from vfieval.run_cleanup import (
    CACHE_CATALOG_COORDINATION_VERSION,
    CacheCoordinationUnavailable,
    RunCleanupService,
)
from vfieval.server import _make_handler


def _workspace(tmp: str) -> tuple[WorkspaceConfig, Database]:
    workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
    workspace.ensure()
    db = Database(workspace.db_path)
    db.init()
    return workspace, db


class CacheCoordinationTests(unittest.TestCase):
    def test_background_scan_does_not_block_and_gates_gc_until_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            service = RunCleanupService(db, workspace)
            entered = threading.Event()
            release = threading.Event()

            def slow_scan() -> dict[str, int]:
                entered.set()
                self.assertTrue(release.wait(5))
                return {"physical_entries": 0, "run_refs": 0, "released_refs": 0}

            with patch.object(service, "_scan_cache_catalog", side_effect=slow_scan):
                started_at = time.monotonic()
                status = service.start_cache_coordination()
                self.assertLess(time.monotonic() - started_at, 0.5)
                self.assertEqual(status["state"], "running")
                self.assertTrue(entered.wait(2))

                with self.assertRaises(CacheCoordinationUnavailable):
                    service.gc_preview()
                with self.assertRaises(CacheCoordinationUnavailable):
                    service.garbage_collect(confirmed=True)

                release.set()
                completed = service.wait_for_cache_coordination(5)

            self.assertTrue(completed["ready"])
            self.assertEqual(completed["state"], "ready")
            self.assertFalse(completed["skipped"])
            marker = db.get(
                "SELECT version FROM schema_migrations WHERE version = ?",
                (CACHE_CATALOG_COORDINATION_VERSION,),
            )
            self.assertIsNotNone(marker)
            self.assertEqual(service.gc_preview()["summary"]["entries"], 0)

    def test_failed_scan_is_diagnostic_and_retryable_without_writing_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            service = RunCleanupService(db, workspace)
            reports = [
                OSError("scan unavailable"),
                {"physical_entries": 0, "run_refs": 0, "released_refs": 0},
            ]

            with patch.object(service, "_scan_cache_catalog", side_effect=reports):
                service.start_cache_coordination()
                failed = service.wait_for_cache_coordination(5)
                self.assertEqual(failed["state"], "failed")
                self.assertFalse(failed["ready"])
                self.assertEqual(failed["error"]["type"], "OSError")
                self.assertIsNotNone(failed["next_retry_at"])
                self.assertIsNone(
                    db.get(
                        "SELECT 1 FROM schema_migrations WHERE version = ?",
                        (CACHE_CATALOG_COORDINATION_VERSION,),
                    )
                )
                health = health_snapshot(
                    db,
                    workspace,
                    maintenance={"cache_coordination": failed},
                )
                self.assertIn("cache_coordination_failed", health["reasons"])
                self.assertFalse(health["ready"])

                retrying = service.retry_cache_coordination()
                self.assertEqual(retrying["state"], "running")
                completed = service.wait_for_cache_coordination(5)

            self.assertEqual(completed["state"], "ready")
            self.assertEqual(completed["attempt"], 2)
            self.assertIsNotNone(
                db.get(
                    "SELECT 1 FROM schema_migrations WHERE version = ?",
                    (CACHE_CATALOG_COORDINATION_VERSION,),
                )
            )

    def test_persistent_marker_skips_same_coordination_version_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            first = RunCleanupService(db, workspace)
            first.ensure_backfilled()

            restarted = RunCleanupService(db, workspace)
            with patch.object(
                restarted,
                "_scan_cache_catalog",
                side_effect=AssertionError("persisted version must skip the historical scan"),
            ):
                restarted.start_cache_coordination()
                status = restarted.wait_for_cache_coordination(5)

            self.assertTrue(status["ready"])
            self.assertTrue(status["skipped"])
            self.assertEqual(status["report"]["physical_entries"], 0)

    def test_handler_and_health_do_not_wait_for_scan_and_gc_returns_503(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            service = RunCleanupService(db, workspace)
            entered = threading.Event()
            release = threading.Event()

            def slow_scan() -> dict[str, int]:
                entered.set()
                self.assertTrue(release.wait(5))
                return {"physical_entries": 0, "run_refs": 0, "released_refs": 0}

            catalog = CatalogSyncCoordinator(
                db,
                workspace,
                sync_callback=lambda *_args: {"ok": True},
            )
            with patch.object(service, "_scan_cache_catalog", side_effect=slow_scan):
                started_at = time.monotonic()
                handler = _make_handler(
                    db,
                    workspace,
                    cleanup_service=service,
                    catalog_sync=catalog,
                )
                self.assertLess(time.monotonic() - started_at, 0.5)
                self.assertTrue(entered.wait(2))

                server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                try:
                    with urllib.request.urlopen(f"{base_url}/api/health", timeout=5) as response:
                        health = json.loads(response.read().decode("utf-8"))
                    self.assertEqual(health["maintenance"]["cache_coordination"]["state"], "running")
                    self.assertTrue(health["live"])

                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(
                            f"{base_url}/api/storage/gc/preview",
                            timeout=5,
                        )
                    self.assertEqual(caught.exception.code, 503)
                    error = json.loads(caught.exception.read().decode("utf-8"))
                    self.assertEqual(error["error"]["code"], "cache_catalog_not_ready")
                finally:
                    release.set()
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

                completed = service.wait_for_cache_coordination(5)
                self.assertTrue(completed["ready"])


if __name__ == "__main__":
    unittest.main()
