from __future__ import annotations

import tempfile
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.media_assets import bind_run_asset, create_collection, upsert_asset
from vfieval.media_items import (
    bind_compare_input,
    bind_run_source,
    ensure_canonical_gt_item,
    register_model_prediction,
)
from vfieval.run_cleanup import RunCleanupService, register_run_cache_refs
from vfieval.server import _make_handler


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
    dataset_id: int | None = None,
    *,
    run_type: str = "model_inference",
) -> int:
    model_id = db.upsert_model(f"model-{name}", "dummy", None, 8, 8, {})
    if dataset_id is None:
        source = workspace.root / f"source-{name}"
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
        create_inference_job=False,
        metadata={
            "output_dir": str(workspace.runs_dir / str(db.next_run_id())),
            "run_type": run_type,
        },
    )
    with db.connection() as conn:
        conn.execute("UPDATE runs SET status = 'completed' WHERE id = ?", (run_id,))
    return run_id


class RunCleanupTests(unittest.TestCase):
    def _make_compare_dependency(
        self,
        workspace: WorkspaceConfig,
        db: Database,
        *,
        source_exists: bool = True,
    ) -> dict:
        source_collection = create_collection(db, "Snapshot Sources", slug="snapshot-sources")
        gt_path = workspace.media_dir / "snapshot-gt.mp4"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_bytes(b"ground-truth")
        gt_asset = upsert_asset(
            db,
            collection_id=int(source_collection["id"]),
            source_key="folder:snapshot/clip.mp4",
            source_kind="folder",
            media_kind="video",
            role="gt",
            display_name="clip.mp4",
            original_name="clip.mp4",
            storage_path=gt_path,
            content_sha256="g" * 64,
            size_bytes=gt_path.stat().st_size,
            frame_count=3,
            width=16,
            height=8,
            fps=5.0,
        )
        item = ensure_canonical_gt_item(db, int(gt_asset["id"]))
        source_run_id = _run(db, workspace, "snapshot-source")
        bind_run_asset(db, source_run_id, int(gt_asset["id"]), "source", video_name="clip.mp4")
        bind_run_source(db, source_run_id, int(item["id"]), video_name="clip.mp4")

        source_path = workspace.runs_dir / str(source_run_id) / "videos" / "clip" / "pred.mp4"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        if source_exists:
            source_path.write_bytes(b"private prediction bytes")
        pred_asset = upsert_asset(
            db,
            collection_id=None,
            source_key=f"run_artifact:snapshot-source:{source_run_id}",
            source_kind="run_artifact",
            media_kind="video",
            role="pred",
            display_name="snapshot source pred",
            original_name="pred.mp4",
            storage_path=source_path,
            content_sha256="p" * 64,
            size_bytes=len(b"private prediction bytes"),
            frame_count=3,
            width=16,
            height=8,
            fps=5.0,
        )
        bind_run_asset(
            db,
            source_run_id,
            int(pred_asset["id"]),
            "pred",
            video_name="clip.mp4",
            track_label="Model A",
        )
        source_member = register_model_prediction(
            db,
            source_run_id,
            int(item["id"]),
            int(pred_asset["id"]),
            method_key="model:snapshot-source",
            temporal_mapping={"source_frame_indices": [0, 1, 2]},
            spatial_origin={"width": 16, "height": 8},
            metadata={"video_name": "clip.mp4"},
        )

        compare_run_id = _run(db, workspace, "snapshot-compare", run_type="video_compare")
        binding = bind_compare_input(
            db,
            compare_run_id,
            int(item["id"]),
            int(source_member["id"]),
            binding_role="compare_pred",
            slot="pred_a",
            metadata={"track_label": "Model A", "video_name": "clip.mp4"},
        )
        bind_run_asset(
            db,
            compare_run_id,
            int(pred_asset["id"]),
            "pred",
            video_name="clip.mp4",
            track_label="Model A",
            metadata={"input": True},
        )
        return {
            "item_id": int(item["id"]),
            "source_run_id": source_run_id,
            "source_path": source_path,
            "source_asset_id": int(pred_asset["id"]),
            "source_member_id": int(source_member["id"]),
            "compare_run_id": compare_run_id,
            "binding_id": int(binding["id"]),
        }

    def test_cleanup_snapshots_dependent_compare_input_before_source_purge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            dependency = self._make_compare_dependency(workspace, db)
            service = RunCleanupService(db, workspace, cache_grace_seconds=0)

            request = service.request_artifact_cleanup(dependency["source_run_id"])
            completed = service.process_request(int(request["id"]))
            self.assertEqual(completed["status"], "completed")
            snapshot_report = completed["report"]["compare_input_snapshots"]
            self.assertEqual(snapshot_report["protected"], 1)

            binding = db.get(
                "SELECT * FROM run_media_item_bindings WHERE id = ?",
                (dependency["binding_id"],),
            )
            assert binding is not None
            self.assertEqual(int(binding["original_member_id"]), dependency["source_member_id"])
            self.assertNotEqual(int(binding["active_member_id"]), dependency["source_member_id"])
            snapshot_member = db.get(
                "SELECT * FROM media_item_members WHERE id = ?",
                (int(binding["active_member_id"]),),
            )
            assert snapshot_member is not None
            self.assertEqual(snapshot_member["member_role"], "compare_snapshot")
            self.assertEqual(snapshot_member["producer_kind"], "video_compare")
            self.assertEqual(int(snapshot_member["producer_run_id"]), dependency["compare_run_id"])
            self.assertEqual(int(snapshot_member["reusable_as_pred"]), 0)

            snapshot_asset = db.get(
                "SELECT * FROM media_assets WHERE id = ?",
                (int(snapshot_member["asset_id"]),),
            )
            assert snapshot_asset is not None
            snapshot_path = Path(str(snapshot_asset["storage_path"]))
            self.assertTrue(snapshot_path.is_file())
            self.assertEqual(snapshot_path.read_bytes(), b"private prediction bytes")
            self.assertTrue(
                str(snapshot_path.resolve()).startswith(
                    str((workspace.runs_dir / str(dependency["compare_run_id"])).resolve())
                )
            )
            self.assertFalse((workspace.runs_dir / str(dependency["source_run_id"])).exists())
            source_asset = db.get("SELECT state FROM media_assets WHERE id = ?", (dependency["source_asset_id"],))
            assert source_asset is not None
            self.assertEqual(source_asset["state"], "unavailable")
            bound_snapshot = db.get(
                """
                SELECT 1 AS found FROM run_media_assets
                WHERE run_id = ? AND asset_id = ? AND role = 'pred'
                """,
                (dependency["compare_run_id"], int(snapshot_member["asset_id"])),
            )
            self.assertIsNotNone(bound_snapshot)

    def test_snapshot_failure_blocks_source_cleanup_and_keeps_binding_live(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            dependency = self._make_compare_dependency(workspace, db, source_exists=False)
            service = RunCleanupService(db, workspace, cache_grace_seconds=0)

            request = service.request_artifact_cleanup(dependency["source_run_id"])
            failed = service.process_request(int(request["id"]))
            self.assertEqual(failed["status"], "failed")
            self.assertIn("unavailable", failed["error"]["message"])
            source_run = db.get_run(dependency["source_run_id"])
            self.assertIsNone(source_run["artifact_cleaned_at"])
            self.assertTrue((workspace.runs_dir / str(dependency["source_run_id"])).exists())
            binding = db.get(
                "SELECT active_member_id FROM run_media_item_bindings WHERE id = ?",
                (dependency["binding_id"],),
            )
            assert binding is not None
            self.assertEqual(int(binding["active_member_id"]), dependency["source_member_id"])

    def test_compare_snapshot_is_a_private_copy_not_a_hard_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            dependency = self._make_compare_dependency(workspace, db)
            service = RunCleanupService(db, workspace)

            report = service._prepare_compare_input_snapshots(dependency["source_run_id"])
            snapshot_asset_id = int(report["snapshots"][0]["snapshot_asset_id"])
            snapshot = db.get("SELECT storage_path FROM media_assets WHERE id = ?", (snapshot_asset_id,))
            assert snapshot is not None
            snapshot_path = Path(str(snapshot["storage_path"]))
            snapshot_path.write_bytes(b"snapshot-only mutation")
            self.assertEqual(dependency["source_path"].read_bytes(), b"private prediction bytes")

    def test_snapshot_member_and_binding_transition_rollback_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            dependency = self._make_compare_dependency(workspace, db)
            service = RunCleanupService(db, workspace)

            with patch.object(
                service,
                "_bind_compare_snapshot_asset",
                side_effect=RuntimeError("injected snapshot binding failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected snapshot binding failure"):
                    service._prepare_compare_input_snapshots(dependency["source_run_id"])

            binding = db.get(
                "SELECT active_member_id FROM run_media_item_bindings WHERE id = ?",
                (dependency["binding_id"],),
            )
            assert binding is not None
            self.assertEqual(int(binding["active_member_id"]), dependency["source_member_id"])
            count = db.get(
                """
                SELECT COUNT(*) AS count FROM media_item_members
                WHERE item_id = ? AND member_role = 'compare_snapshot'
                """,
                (dependency["item_id"],),
            )
            assert count is not None
            self.assertEqual(int(count["count"]), 0)
            inputs_root = workspace.runs_dir / str(dependency["compare_run_id"]) / "inputs"
            self.assertFalse(
                any(path.name.startswith("snapshot-") for path in inputs_root.rglob("snapshot-*"))
            )

    def test_multiple_snapshot_switches_roll_back_as_one_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            first = self._make_compare_dependency(workspace, db)
            second_compare_run_id = _run(
                db,
                workspace,
                "snapshot-compare-second",
                run_type="video_compare",
            )
            second_binding = bind_compare_input(
                db,
                second_compare_run_id,
                first["item_id"],
                first["source_member_id"],
                binding_role="compare_pred",
                slot="pred_b",
                metadata={"track_label": "Model A", "video_name": "clip.mp4"},
            )
            service = RunCleanupService(db, workspace)
            original_bind = service._bind_compare_snapshot_asset
            calls = 0
            request = service.request_artifact_cleanup(first["source_run_id"])

            def fail_after_first_switch(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected second snapshot binding failure")
                return original_bind(*args, **kwargs)

            with patch.object(
                service,
                "_bind_compare_snapshot_asset",
                side_effect=fail_after_first_switch,
            ):
                failed = service.process_request(int(request["id"]))
            self.assertEqual(failed["status"], "failed")
            self.assertIn("second snapshot binding failure", failed["error"]["message"])

            for binding_id in (first["binding_id"], int(second_binding["id"])):
                binding = db.get(
                    "SELECT active_member_id FROM run_media_item_bindings WHERE id = ?",
                    (binding_id,),
                )
                assert binding is not None
                self.assertEqual(int(binding["active_member_id"]), first["source_member_id"])
            snapshot_count = db.get(
                "SELECT COUNT(*) AS count FROM media_item_members WHERE member_role = 'compare_snapshot'"
            )
            assert snapshot_count is not None
            self.assertEqual(int(snapshot_count["count"]), 0)
            self.assertTrue(first["source_path"].exists())
            self.assertIsNone(db.get_run(first["source_run_id"])["artifact_cleaned_at"])
            for compare_run_id in (first["compare_run_id"], second_compare_run_id):
                inputs_root = workspace.runs_dir / str(compare_run_id) / "inputs"
                if inputs_root.exists():
                    self.assertFalse(
                        any(path.name.startswith("snapshot-") for path in inputs_root.rglob("snapshot-*"))
                    )

    def test_delete_is_persistent_and_only_hides_after_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "one")
            run_dir = workspace.runs_dir / str(run_id)
            run_dir.mkdir(parents=True)
            (run_dir / "artifact.bin").write_bytes(b"payload")
            service = RunCleanupService(db, workspace, cache_grace_seconds=0)

            requested = service.request_delete(run_id)
            self.assertEqual(requested["status"], "requested")
            self.assertIsNone(db.get_run(run_id)["deleted_at"])
            self.assertTrue(run_dir.exists())

            processed = service.process_pending()
            self.assertEqual(processed[0]["status"], "completed")
            run = db.get_run(run_id)
            self.assertIsNotNone(run["artifact_cleaned_at"])
            self.assertIsNotNone(run["deleted_at"])
            self.assertEqual(int(run["content_revision"]), 1)
            self.assertFalse(run_dir.exists())

    def test_running_delete_waits_for_worker_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "running")
            job_id = db.add_run_job(run_id, "inference", {"run_id": run_id})
            with db.connection() as conn:
                conn.execute(
                    "UPDATE runs SET status = 'running', finished_at = NULL WHERE id = ?",
                    (run_id,),
                )
                conn.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (job_id,))
            service = RunCleanupService(db, workspace)

            request = service.request_delete(run_id)
            self.assertEqual(request["status"], "canceling")
            self.assertIsNone(db.get_run(run_id)["deleted_at"])
            self.assertEqual(service.process_pending()[0]["status"], "canceling")

            db.cancel_job(job_id)
            db.cancel_run(run_id)
            completed = service.process_pending()[0]
            self.assertEqual(completed["status"], "completed")
            self.assertIsNotNone(db.get_run(run_id)["deleted_at"])

    def test_shared_cache_waits_for_last_reference_and_active_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            source = workspace.root / "source"
            source.mkdir()
            dataset_id = db.create_dataset("shared", str(source), has_gt=True)
            cache_key = "a" * 64
            cache_dir = workspace.root / "decode_cache" / cache_key
            cache_dir.mkdir(parents=True)
            frame = cache_dir / "000001.png"
            frame.write_bytes(b"frame")
            db.add_sample(
                dataset_id,
                "sample",
                str(frame),
                str(frame),
                str(frame),
                {"cache_key": cache_key},
            )
            run_a = _run(db, workspace, "a", dataset_id)
            run_b = _run(db, workspace, "b", dataset_id)
            service = RunCleanupService(db, workspace, cache_grace_seconds=0)
            service.backfill_cache_catalog()

            service.request_delete(run_a)
            service.process_pending()
            preview = service.gc_preview()
            cache = next(row for row in preview["caches"] if row["cache_key"] == cache_key)
            self.assertFalse(cache["eligible"])
            self.assertEqual(cache["active_run_refs"], 1)

            service.request_delete(run_b)
            service.process_pending()
            entry = db.get_cache_entry("decode_cache", cache_key)
            assert entry is not None
            db.acquire_cache_lease(int(entry["id"]), "test", ttl_seconds=60)
            preview = service.gc_preview()
            cache = next(row for row in preview["caches"] if row["cache_key"] == cache_key)
            self.assertEqual(cache["reason"], "active_lease")

            db.release_cache_lease(int(entry["id"]), "test")
            collected = service.garbage_collect(confirmed=True, entry_ids=[int(entry["id"])])
            self.assertEqual(len(collected["deleted_caches"]), 1)
            self.assertFalse(cache_dir.exists())

    def test_cache_gc_claim_fences_late_references_and_leases(self) -> None:
        """The destructive cache claim must beat a stale preview atomically."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "gc-claim")
            cache_key = "b" * 64
            cache_dir = workspace.root / "decode_cache" / cache_key
            cache_dir.mkdir(parents=True)
            (cache_dir / "frame.png").write_bytes(b"frame")
            service = RunCleanupService(db, workspace, cache_grace_seconds=0)
            service.backfill_cache_catalog()
            entry = db.get_cache_entry("decode_cache", cache_key)
            assert entry is not None
            entry_id = int(entry["id"])
            with db.connection() as conn:
                conn.execute("UPDATE cache_entries SET gc_after = 0 WHERE id = ?", (entry_id,))

            # A reference published after preview but before the destructive
            # transition makes the conditional claim fail.
            db.replace_run_cache_refs(run_id, [entry_id], grace_seconds=0)
            self.assertIsNone(db.claim_cache_entry_for_gc(entry_id))
            db.release_run_cache_refs(run_id, grace_seconds=0)

            claimed = db.claim_cache_entry_for_gc(entry_id)
            self.assertIsNotNone(claimed)
            with self.assertRaises(RuntimeError):
                db.acquire_cache_lease(entry_id, "late-lease", ttl_seconds=60)
            with self.assertRaises(RuntimeError):
                db.replace_run_cache_refs(run_id, [entry_id], grace_seconds=0)
            db.mark_cache_entry_state(entry_id, "ready")

    def test_decode_partial_cache_is_previewed_and_protected_by_base_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            cache_key = "c" * 64
            cache_dir = workspace.root / "decode_cache" / cache_key
            partial_dir = workspace.root / "decode_cache" / f"{cache_key}.partial"
            cache_dir.mkdir(parents=True)
            partial_dir.mkdir(parents=True)
            (cache_dir / "frame.png").write_bytes(b"cached")
            (partial_dir / "frame.png").write_bytes(b"partial")
            run_id = _run(db, workspace, "partial-ref")
            dataset_id = int(db.get_run(run_id)["dataset_id"])
            db.add_sample(
                dataset_id,
                "partial-cache-sample",
                str(cache_dir / "frame.png"),
                str(cache_dir / "frame.png"),
                str(cache_dir / "frame.png"),
                {"cache_key": cache_key},
            )
            service = RunCleanupService(db, workspace, cache_grace_seconds=0)
            service.ensure_backfilled()

            base = db.get_cache_entry("decode_cache", cache_key)
            partial = db.get_cache_entry("decode_cache", f"{cache_key}.partial")
            assert base is not None and partial is not None
            partial_id = int(partial["id"])
            self.assertTrue(partial["metadata"].get("partial"))
            with db.connection() as conn:
                conn.execute("UPDATE cache_entries SET gc_after = 0 WHERE id = ?", (partial_id,))

            preview = service.gc_preview(entry_ids=[partial_id])
            row = preview["caches"][0]
            self.assertEqual(row["reason"], "referenced_by_active_runs")
            self.assertFalse(row["eligible"])
            self.assertIsNone(db.claim_cache_entry_for_gc(partial_id))
            db.release_run_cache_refs(run_id, grace_seconds=0)

            db.acquire_cache_lease(int(base["id"]), "decode-in-progress", ttl_seconds=60)
            preview = service.gc_preview(entry_ids=[partial_id])
            row = preview["caches"][0]
            self.assertEqual(row["reason"], "active_lease")
            self.assertFalse(row["eligible"])
            self.assertIsNone(db.claim_cache_entry_for_gc(partial_id))

            db.release_cache_lease(int(base["id"]), "decode-in-progress")
            collected = service.garbage_collect(confirmed=True, entry_ids=[partial_id])
            self.assertEqual(len(collected["deleted_caches"]), 1)
            self.assertFalse(partial_dir.exists())

    def test_failed_purge_can_be_retried_without_hiding_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "retry")
            run_dir = workspace.runs_dir / str(run_id)
            run_dir.mkdir(parents=True)
            (run_dir / "keep").write_text("x", encoding="utf-8")
            service = RunCleanupService(db, workspace)
            request = service.request_delete(run_id)

            with patch("vfieval.run_cleanup.shutil.rmtree", side_effect=PermissionError("busy")):
                failed = service.process_request(int(request["id"]))
            self.assertEqual(failed["status"], "failed")
            self.assertIsNone(db.get_run(run_id)["deleted_at"])
            self.assertTrue(run_dir.exists())

            retried = service.request_delete(run_id)
            self.assertEqual(retried["status"], "requested")
            service.process_pending()
            self.assertIsNotNone(db.get_run(run_id)["deleted_at"])
            self.assertFalse(run_dir.exists())

    def test_gc_preview_reports_and_cleans_legacy_deleted_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "legacy")
            run_dir = workspace.runs_dir / str(run_id)
            run_dir.mkdir(parents=True)
            (run_dir / "old.bin").write_bytes(b"old")
            db.soft_delete_run(run_id)
            service = RunCleanupService(db, workspace, cache_grace_seconds=0)
            service.ensure_backfilled()

            preview = service.gc_preview()
            row = next(item for item in preview["runs"] if item["run_id"] == run_id)
            self.assertTrue(row["eligible"])
            self.assertGreaterEqual(preview["summary"]["run_bytes"], 3)

            result = service.garbage_collect(confirmed=True, run_ids=[run_id])
            self.assertEqual(len(result["deleted_runs"]), 1)
            self.assertFalse(run_dir.exists())
            self.assertIsNotNone(db.get_run(run_id)["artifact_cleaned_at"])

    def test_register_run_cache_refs_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "empty")
            first = register_run_cache_refs(db, workspace, run_id)
            second = register_run_cache_refs(db, workspace, run_id)
            self.assertEqual(first["total"], 0)
            self.assertEqual(second["total"], 0)

    def test_terminal_failure_and_cancel_invalidate_result_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "revision")
            with db.connection() as conn:
                conn.execute(
                    "UPDATE runs SET status = 'running', finished_at = NULL WHERE id = ?",
                    (run_id,),
                )
            self.assertEqual(db.get_run(run_id)["content_revision"], 0)
            self.assertTrue(db.fail_run(run_id, {"message": "failed"}))
            self.assertEqual(db.get_run(run_id)["content_revision"], 1)
            self.assertFalse(db.cancel_run(run_id, {"message": "canceled"}))
            self.assertEqual(db.get_run(run_id)["content_revision"], 1)

            canceled_id = _run(db, workspace, "canceled-revision")
            with db.connection() as conn:
                conn.execute(
                    "UPDATE runs SET status = 'running', finished_at = NULL WHERE id = ?",
                    (canceled_id,),
                )
            self.assertTrue(db.cancel_run(canceled_id, {"message": "canceled"}))
            self.assertEqual(db.get_run(canceled_id)["content_revision"], 1)

    def test_queued_cancel_and_artifact_publication_invalidate_result_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "published-artifacts")
            job_id = db.add_run_job(run_id, "inference", {"run_id": run_id})

            db.add_artifacts_bulk(
                job_id,
                [
                    {
                        "sample_id": None,
                        "kind": "pred",
                        "path": "published.png",
                        "mime_type": "image/png",
                    }
                ],
            )
            self.assertEqual(db.get_run(run_id)["content_revision"], 1)

            db.add_artifact(job_id, None, "diff", "published-diff.png", "image/png")
            self.assertEqual(db.get_run(run_id)["content_revision"], 2)

            with db.connection() as conn:
                conn.execute("UPDATE runs SET status = 'queued' WHERE id = ?", (run_id,))
            db.request_run_cancel(run_id)
            canceled = db.get_run(run_id)
            self.assertEqual(canceled["status"], "canceled")
            self.assertEqual(canceled["content_revision"], 3)

    def test_legacy_artifact_and_metric_publication_bump_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            model_id = db.upsert_model("legacy-model", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("legacy-dataset", str(workspace.root / "source"), has_gt=True)
            run_id = db.create_run(
                "legacy-run",
                model_id,
                dataset_id,
                8,
                8,
                1,
                "cpu",
                "fp32",
                [],
            )
            inference_job_id = int(db.get_run(run_id)["inference_job_id"])
            # Pre-run_jobs databases retain only the direct run job link.
            with db.connection() as conn:
                conn.execute("DELETE FROM run_jobs WHERE run_id = ?", (run_id,))
            self.assertEqual(db.run_inference_job_ids(run_id), [inference_job_id])

            db.add_artifact(inference_job_id, None, "pred", str(workspace.root / "pred.png"), "image/png")
            self.assertEqual(db.get_run(run_id)["content_revision"], 1)
            metric_job_id = db.create_job("metric", {"run_id": run_id})
            db.add_metric_result(
                metric_job_id,
                inference_job_id,
                None,
                "lpips_vit_patch",
                "completed",
                0.2,
                {},
            )
            self.assertEqual(db.get_run(run_id)["content_revision"], 2)

    def test_cleanup_request_prevents_late_job_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "cleanup-fence")
            db.add_run_job(run_id, "inference", {"run_id": run_id})
            # This models a scheduler racing a persisted cleanup request after
            # the normal queued-job cancellation boundary.
            db.request_run_purge(run_id, "cleanup_artifacts")
            self.assertIsNone(db.claim_next_job("worker", ["inference"]))

    def test_purge_claim_token_heartbeats_and_fences_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "purge-token")
            request = db.request_run_purge(run_id, "delete_run")
            token = "claim-token"
            self.assertTrue(db.claim_run_purge_request(int(request["id"]), token))
            self.assertTrue(db.heartbeat_run_purge_request(int(request["id"]), token))

            completed = db.update_run_purge_request(
                int(request["id"]),
                "completed",
                report={"artifact_cleaned": True},
                expected_claim_token=token,
            )
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["claim_token"], "")
            with self.assertRaises(RuntimeError):
                db.update_run_purge_request(
                    int(request["id"]),
                    "failed",
                    error={"message": "stale worker"},
                    expected_claim_token=token,
                )

    def test_cleanup_loop_stops_without_leaking_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            service = RunCleanupService(db, workspace)
            stop = threading.Event()
            thread = threading.Thread(target=service.run_forever, args=(stop, 0.01), daemon=True)
            thread.start()
            stop.set()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

    def test_cleanup_loop_processes_pending_without_http_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "background-loop")
            run_dir = workspace.runs_dir / str(run_id)
            run_dir.mkdir(parents=True)
            (run_dir / "artifact").write_bytes(b"x")

            service = RunCleanupService(db, workspace)
            request = service.request_delete(run_id)
            stop = threading.Event()
            thread = threading.Thread(
                target=service.run_forever,
                args=(stop, 0.01),
                daemon=True,
            )
            thread.start()
            try:
                poll_wait = threading.Event()
                for _ in range(200):
                    current = db.get_run_purge_request_by_id(int(request["id"]))
                    if current["status"] in {"completed", "failed"}:
                        break
                    poll_wait.wait(0.01)
                else:
                    self.fail("background cleanup loop did not process the purge request")
            finally:
                stop.set()
                thread.join(timeout=2)

            self.assertFalse(thread.is_alive())
            self.assertEqual(current["status"], "completed", current)
            self.assertFalse(run_dir.exists())

    def test_delete_and_storage_gc_http_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = _workspace(tmp)
            run_id = _run(db, workspace, "http")
            run_dir = workspace.runs_dir / str(run_id)
            run_dir.mkdir(parents=True)
            (run_dir / "artifact").write_bytes(b"x")
            service = RunCleanupService(db, workspace)
            cleanup_stop = threading.Event()
            cleanup_thread = threading.Thread(
                target=service.run_forever,
                args=(cleanup_stop, 0.01),
                daemon=True,
            )
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                _make_handler(db, workspace, cleanup_service=service),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            cleanup_thread.start()
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                missing_preview = urllib.request.Request(
                    f"{base}/api/runs/{run_id}",
                    method="DELETE",
                )
                with self.assertRaises(urllib.error.HTTPError) as missing_context:
                    urllib.request.urlopen(missing_preview, timeout=10)
                self.assertEqual(missing_context.exception.code, 409)
                missing_payload = json.loads(
                    missing_context.exception.read().decode("utf-8")
                )
                self.assertEqual(missing_payload["error"]["code"], "missing_preview")
                preview_request = urllib.request.Request(
                    f"{base}/api/run-purge/preview",
                    data=json.dumps(
                        {"request_type": "delete_run", "run_ids": [run_id]}
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(preview_request, timeout=10) as response:
                    run_preview = json.loads(response.read().decode("utf-8"))
                request = urllib.request.Request(
                    f"{base}/api/runs/{run_id}?preview_token={run_preview['preview_token']}",
                    method="DELETE",
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    self.assertEqual(response.status, 202)
                    payload = json.loads(response.read().decode("utf-8"))
                self.assertFalse(payload["deleted"])
                poll_wait = threading.Event()
                for _ in range(200):
                    with urllib.request.urlopen(
                        f"{base}/api/run-purge-requests/{payload['request_id']}", timeout=10
                    ) as response:
                        purge = json.loads(response.read().decode("utf-8"))
                    if purge["status"] in {"completed", "failed"}:
                        break
                    poll_wait.wait(0.01)
                else:
                    self.fail("background cleanup loop did not finish the HTTP purge request")
                self.assertEqual(purge["status"], "completed")

                with urllib.request.urlopen(f"{base}/api/storage/gc/preview", timeout=10) as response:
                    preview = json.loads(response.read().decode("utf-8"))
                self.assertIn("run_bytes", preview["summary"])
                self.assertIn("cache_bytes", preview["summary"])
                self.assertTrue(preview["preview_token"])
                gc_request = urllib.request.Request(
                    f"{base}/api/storage/gc",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as context:
                    urllib.request.urlopen(gc_request, timeout=10)
                self.assertEqual(context.exception.code, 400)
                missing_preview_request = urllib.request.Request(
                    f"{base}/api/storage/gc",
                    data=json.dumps({"confirm": True}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as context:
                    urllib.request.urlopen(missing_preview_request, timeout=10)
                self.assertEqual(context.exception.code, 400)
                confirmed_request = urllib.request.Request(
                    f"{base}/api/storage/gc",
                    data=json.dumps(
                        {"confirm": True, "preview_token": preview["preview_token"]}
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(confirmed_request, timeout=10) as response:
                    self.assertEqual(response.status, 200)
            finally:
                cleanup_stop.set()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                cleanup_thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
