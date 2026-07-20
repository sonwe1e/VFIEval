from __future__ import annotations

from collections import namedtuple
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.storage_budget import (
    StorageCapacityError,
    campaign_requested_bytes,
    ensure_storage_capacity,
    storage_capacity,
)


DiskUsage = namedtuple("DiskUsage", "total used free")


class StorageBudgetTests(unittest.TestCase):
    def test_campaign_budget_accounts_for_low_bitrate_4k_decode_expansion(self) -> None:
        class AssetRows:
            def query(self, _sql, _params):
                return [
                    {
                        "id": 1,
                        "size_bytes": 20 * 1024**2,
                        "frame_count": 300,
                        "width": 3840,
                        "height": 2160,
                    }
                ]

        requested = campaign_requested_bytes(AssetRows(), 7)  # type: ignore[arg-type]
        raw_rgb = 300 * 3840 * 2160 * 3
        self.assertGreaterEqual(requested, (raw_rgb * 11 + 4) // 5)
        self.assertGreater(requested, 20 * 1024**2 * 100)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"VFIEVAL_MIN_FREE_BYTES": "0"},
            clear=False,
        ):
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            with (
                patch("vfieval.storage_budget._active_run_reservations", return_value=0),
                patch("vfieval.storage_budget._active_upload_reservations", return_value=0),
                patch("vfieval.storage_budget._active_campaign_reservations", return_value=0),
                patch(
                    "vfieval.storage_budget.shutil.disk_usage",
                    return_value=DiskUsage(100 * 1024**3, 95 * 1024**3, 5 * 1024**3),
                ),
            ):
                with self.assertRaises(StorageCapacityError):
                    ensure_storage_capacity(db, workspace, requested_bytes=requested)

    def test_capacity_includes_active_run_and_requested_reservations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"VFIEVAL_MIN_FREE_BYTES": "0"}, clear=False
        ):
            root = Path(tmp)
            workspace = WorkspaceConfig.from_root(root / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("model", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("dataset", str(root), True)
            db.create_run(
                "active",
                model_id,
                dataset_id,
                8,
                8,
                1,
                "cpu",
                "fp32",
                [],
                metadata={"workload": {"artifact_budget_bytes": 100}},
            )
            with patch("vfieval.storage_budget.shutil.disk_usage", return_value=DiskUsage(1000, 0, 1000)):
                capacity = storage_capacity(db, workspace, requested_bytes=200)
            self.assertEqual(capacity["reservation_breakdown"]["active_runs"], 100)
            self.assertEqual(capacity["requested_bytes"], 200)
            self.assertEqual(capacity["safety_margin_bytes"], 20)
            self.assertEqual(capacity["remaining_after_request_bytes"], 700)
            self.assertTrue(capacity["sufficient"])

    def test_capacity_error_exposes_actionable_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"VFIEVAL_MIN_FREE_BYTES": "100"}, clear=False
        ):
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            with patch("vfieval.storage_budget.shutil.disk_usage", return_value=DiskUsage(1000, 800, 200)):
                with self.assertRaises(StorageCapacityError) as raised:
                    ensure_storage_capacity(db, workspace, requested_bytes=150)
            payload = raised.exception.public_payload()["error"]
            self.assertEqual(payload["type"], "StorageCapacityError")
            self.assertFalse(payload["capacity"]["sufficient"])
            self.assertEqual(payload["capacity"]["required_free_bytes"], 250)


if __name__ == "__main__":
    unittest.main()
