from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vfieval.catalog_sync import CatalogSyncCoordinator
from vfieval.config import WorkspaceConfig
from vfieval.db import Database


class CatalogSyncCoordinatorTests(unittest.TestCase):
    def _workspace(self, root: Path) -> tuple[WorkspaceConfig, Database]:
        workspace = WorkspaceConfig.from_root(root / ".vfieval")
        workspace.ensure()
        db = Database(workspace.db_path)
        db.init()
        return workspace, db

    def test_concurrent_requests_join_one_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = self._workspace(Path(tmp))
            entered = threading.Event()
            release = threading.Event()
            calls: list[bool] = []

            def sync(_db, _workspace, include_runs):
                calls.append(bool(include_runs))
                entered.set()
                self.assertTrue(release.wait(2))
                return {"files": 2}

            coordinator = CatalogSyncCoordinator(db, workspace, sync_callback=sync)
            first = coordinator.request_sync()
            self.assertEqual(first["state"], "requested")
            self.assertTrue(entered.wait(2))
            joined = coordinator.request_sync()
            self.assertTrue(joined["joined_existing"])
            release.set()
            completed = coordinator.wait(2)

            self.assertEqual(calls, [False])
            self.assertEqual(completed["state"], "completed")
            self.assertEqual(completed["catalog_revision"], 1)
            self.assertEqual(completed["report"], {"files": 2})
            self.assertIsInstance(completed["duration_seconds"], float)
            self.assertGreaterEqual(completed["duration_seconds"], 0.0)

    def test_default_sync_completes_and_reports_canonical_item_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = self._workspace(Path(tmp))
            coordinator = CatalogSyncCoordinator(db, workspace)

            coordinator.request_sync(include_runs=False)
            completed = coordinator.wait(5)

            self.assertEqual(completed["state"], "completed", completed)
            self.assertEqual(completed["report"]["canonical_items"], 0)
            self.assertEqual(completed["report"]["catalog"]["media_items"], 0)

    def test_failure_is_visible_and_next_request_can_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = self._workspace(Path(tmp))
            attempts = 0

            def sync(_db, _workspace, _include_runs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("catalog failed")
                return {"ok": True}

            coordinator = CatalogSyncCoordinator(db, workspace, sync_callback=sync)
            coordinator.request_sync()
            failed = coordinator.wait(2)
            self.assertEqual(failed["state"], "failed")
            self.assertEqual(failed["error"]["type"], "RuntimeError")
            self.assertEqual(failed["catalog_revision"], 0)
            self.assertIsInstance(failed["duration_seconds"], float)

            coordinator.request_sync(include_runs=True)
            completed = coordinator.wait(2)
            self.assertEqual(completed["state"], "completed")
            self.assertEqual(completed["catalog_revision"], 1)

    def test_running_folder_sync_is_upgraded_when_run_assets_are_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = self._workspace(Path(tmp))
            entered = threading.Event()
            release = threading.Event()
            calls: list[bool] = []

            def sync(_db, _workspace, include_runs):
                calls.append(bool(include_runs))
                if not include_runs:
                    entered.set()
                    self.assertTrue(release.wait(2))
                return {"include_runs": bool(include_runs)}

            coordinator = CatalogSyncCoordinator(db, workspace, sync_callback=sync)
            coordinator.request_sync(include_runs=False)
            self.assertTrue(entered.wait(2))
            joined = coordinator.request_sync(include_runs=True)
            self.assertTrue(joined["joined_existing"])
            release.set()
            completed = coordinator.wait(2)

            self.assertEqual(calls, [False, True])
            self.assertEqual(completed["state"], "completed")
            self.assertEqual(completed["report"], {"include_runs": True})


if __name__ == "__main__":
    unittest.main()
