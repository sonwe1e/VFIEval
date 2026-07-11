from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.media_assets import bind_run_asset, ensure_collection, upsert_asset
from vfieval.media_items import (
    bind_compare_input,
    bind_run_source,
    ensure_canonical_gt_item,
    register_model_prediction,
)
from vfieval.run_cleanup import RunCleanupService


def _workspace(tmp: str) -> tuple[WorkspaceConfig, Database]:
    workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
    workspace.ensure()
    db = Database(workspace.db_path)
    db.init()
    return workspace, db


def _run(
    db: Database,
    workspace: WorkspaceConfig,
    name: str,
    run_type: str,
) -> int:
    model_id = db.upsert_model(f"model-{name}", "dummy", None, 8, 8, {})
    source = workspace.root / f"dataset-{name}"
    source.mkdir(parents=True, exist_ok=True)
    dataset_id = db.create_dataset(f"dataset-{name}", str(source), has_gt=True)
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
        metadata={"run_type": run_type, "output_dir": str(workspace.runs_dir / "pending")},
        create_inference_job=False,
    )
    with db.connection() as conn:
        conn.execute(
            "UPDATE runs SET status = 'completed', metadata_json = json_set(metadata_json, '$.output_dir', ?) WHERE id = ?",
            (str(workspace.runs_dir / str(run_id)), run_id),
        )
    return run_id


def _linked_compare_fixture(
    db: Database,
    workspace: WorkspaceConfig,
) -> tuple[int, int, int, Path]:
    group = ensure_collection(
        db,
        "test4k",
        "folder-test4k",
        {"source_kind": "folder", "group": "test4k"},
    )
    gt_path = workspace.root.parent / "videos" / "test4k" / "clip.mp4"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_bytes(b"canonical-gt")
    gt_asset = upsert_asset(
        db,
        collection_id=int(group["id"]),
        source_key="folder:test4k/clip.mp4",
        source_kind="folder",
        media_kind="video",
        role="gt",
        display_name="clip.mp4",
        original_name="clip.mp4",
        storage_path=gt_path,
        mime_type="video/mp4",
        content_sha256="gt",
        size_bytes=gt_path.stat().st_size,
        frame_count=3,
        width=8,
        height=8,
        fps=24.0,
    )
    item = ensure_canonical_gt_item(db, int(gt_asset["id"]))
    canonical_member = db.get(
        "SELECT id FROM media_item_members WHERE item_id = ? AND member_role = 'canonical_gt'",
        (int(item["id"]),),
    )
    assert canonical_member is not None

    source_run_id = _run(db, workspace, "source", "model_inference")
    source_run_dir = workspace.runs_dir / str(source_run_id)
    pred_path = source_run_dir / "videos" / "clip" / "pred.mp4"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_bytes(b"protected-pred-bytes")
    source_collection = ensure_collection(
        db,
        f"Run {source_run_id}",
        f"run-{source_run_id}",
        {"source_kind": "run_artifact", "run_id": source_run_id},
    )
    pred_asset = upsert_asset(
        db,
        collection_id=int(source_collection["id"]),
        source_key=f"run_artifact:test-pred:{source_run_id}",
        source_kind="run_artifact",
        media_kind="video",
        role="pred",
        display_name="clip source pred",
        original_name="pred.mp4",
        storage_path=pred_path,
        mime_type="video/mp4",
        content_sha256="pred",
        size_bytes=pred_path.stat().st_size,
        frame_count=3,
        width=8,
        height=8,
        fps=24.0,
        provenance={"run_id": source_run_id, "video_name": "clip.mp4"},
    )
    bind_run_asset(
        db,
        source_run_id,
        int(pred_asset["id"]),
        "pred",
        video_name="clip.mp4",
    )
    bind_run_source(db, source_run_id, int(item["id"]), video_name="clip.mp4")
    pred_member = register_model_prediction(
        db,
        source_run_id,
        int(item["id"]),
        int(pred_asset["id"]),
        temporal_mapping={"source_frame_indices": [1, 3, 5], "fps": 24.0},
        spatial_origin={"width": 8, "height": 8, "resolution_mode": "native"},
        metadata={"video_name": "clip.mp4"},
    )

    compare_run_id = _run(db, workspace, "compare", "video_compare")
    (workspace.runs_dir / str(compare_run_id)).mkdir(parents=True, exist_ok=True)
    bind_compare_input(
        db,
        compare_run_id,
        int(item["id"]),
        int(canonical_member["id"]),
        binding_role="compare_gt",
        slot="GT",
        metadata={"video_name": "clip.mp4"},
    )
    bind_compare_input(
        db,
        compare_run_id,
        int(item["id"]),
        int(pred_member["id"]),
        binding_role="compare_pred",
        slot="A",
        metadata={"video_name": "clip.mp4"},
    )
    return source_run_id, compare_run_id, int(pred_member["id"]), pred_path


class CompareSnapshotCleanupTests(unittest.TestCase):
    def test_source_delete_freezes_compare_input_before_removing_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            source_run_id, compare_run_id, source_member_id, pred_path = _linked_compare_fixture(
                db, workspace
            )
            service = RunCleanupService(db, workspace, cache_grace_seconds=0)

            service.request_delete(source_run_id)
            completed = service.process_pending()[0]
            self.assertEqual(completed["status"], "completed")
            self.assertFalse(pred_path.exists())
            self.assertIsNotNone(db.get_run(source_run_id)["deleted_at"])
            report = completed["report"]["compare_input_snapshots"]
            self.assertEqual(report["protected"], 1)
            self.assertEqual(report["compare_run_ids"], [compare_run_id])

            binding = db.get(
                "SELECT * FROM run_media_item_bindings WHERE run_id = ? AND binding_role = 'compare_pred'",
                (compare_run_id,),
            )
            assert binding is not None
            self.assertEqual(int(binding["original_member_id"]), source_member_id)
            self.assertNotEqual(int(binding["active_member_id"]), source_member_id)
            snapshot = db.get(
                """
                SELECT mim.*, a.storage_path, a.source_kind, a.state AS asset_state
                FROM media_item_members mim
                JOIN media_assets a ON a.id = mim.asset_id
                WHERE mim.id = ?
                """,
                (int(binding["active_member_id"]),),
            )
            assert snapshot is not None
            self.assertEqual(snapshot["member_role"], "compare_snapshot")
            self.assertEqual(int(snapshot["reusable_as_pred"]), 0)
            self.assertEqual(snapshot["source_kind"], "run_artifact")
            self.assertEqual(snapshot["asset_state"], "ready")
            snapshot_asset_id = int(snapshot["asset_id"])
            snapshot_path = Path(str(snapshot["storage_path"]))
            self.assertTrue(snapshot_path.is_file())
            self.assertEqual(snapshot_path.read_bytes(), b"protected-pred-bytes")
            self.assertEqual(
                snapshot_path.parent.parent,
                workspace.runs_dir / str(compare_run_id) / "inputs" / "A",
            )
            linked = db.get(
                "SELECT 1 FROM run_media_assets WHERE run_id = ? AND asset_id = ? AND role = 'pred'",
                (compare_run_id, int(snapshot["asset_id"])),
            )
            self.assertIsNotNone(linked)

            # The snapshot is Compare-owned and disappears with that Compare,
            # while deleting the original source never made it unavailable.
            service.request_delete(compare_run_id)
            self.assertEqual(service.process_pending()[0]["status"], "completed")
            self.assertFalse(snapshot_path.exists())
            snapshot_asset = db.get("SELECT state FROM media_assets WHERE id = ?", (snapshot_asset_id,))
            assert snapshot_asset is not None
            self.assertEqual(snapshot_asset["state"], "unavailable")

    def test_snapshot_failure_keeps_source_and_retry_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            source_run_id, compare_run_id, source_member_id, pred_path = _linked_compare_fixture(
                db, workspace
            )
            service = RunCleanupService(db, workspace, cache_grace_seconds=0)
            request = service.request_delete(source_run_id)

            with patch(
                "vfieval.run_cleanup._materialize_snapshot",
                side_effect=PermissionError("snapshot destination is busy"),
            ):
                failed = service.process_request(int(request["id"]))
            self.assertEqual(failed["status"], "failed")
            self.assertIsNone(db.get_run(source_run_id)["deleted_at"])
            self.assertIsNone(db.get_run(source_run_id)["artifact_cleaned_at"])
            self.assertTrue(pred_path.exists())
            binding = db.get(
                "SELECT original_member_id, active_member_id FROM run_media_item_bindings WHERE run_id = ? AND binding_role = 'compare_pred'",
                (compare_run_id,),
            )
            assert binding is not None
            self.assertEqual(int(binding["original_member_id"]), source_member_id)
            self.assertEqual(int(binding["active_member_id"]), source_member_id)

            retried = service.request_delete(source_run_id)
            self.assertEqual(retried["status"], "requested")
            completed = service.process_pending()[0]
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["report"]["compare_input_snapshots"]["protected"], 1)
            self.assertFalse(pred_path.exists())

    def test_untrusted_source_path_blocks_purge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            source_run_id, compare_run_id, source_member_id, pred_path = _linked_compare_fixture(
                db, workspace
            )
            outside = workspace.root / "outside-pred.mp4"
            outside.write_bytes(pred_path.read_bytes())
            source_asset = db.get(
                "SELECT asset_id FROM media_item_members WHERE id = ?",
                (source_member_id,),
            )
            assert source_asset is not None
            with db.connection() as conn:
                conn.execute(
                    "UPDATE media_assets SET storage_path = ? WHERE id = ?",
                    (str(outside), int(source_asset["asset_id"])),
                )

            service = RunCleanupService(db, workspace, cache_grace_seconds=0)
            service.request_delete(source_run_id)
            failed = service.process_pending()[0]
            self.assertEqual(failed["status"], "failed")
            self.assertIn("trusted Run directory", failed["error"]["message"])
            self.assertTrue(pred_path.exists())
            self.assertIsNone(db.get_run(source_run_id)["deleted_at"])
            binding = db.get(
                "SELECT active_member_id FROM run_media_item_bindings WHERE run_id = ? AND binding_role = 'compare_pred'",
                (compare_run_id,),
            )
            assert binding is not None
            self.assertEqual(int(binding["active_member_id"]), source_member_id)


if __name__ == "__main__":
    unittest.main()
