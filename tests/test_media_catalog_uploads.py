from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import unittest
import urllib.request
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vfieval.media_assets import (
    create_collection,
    get_asset,
    list_assets,
    list_folder_group_videos,
    soft_delete_asset,
    source_assets_to_video_payload,
    sync_folder_assets,
    sync_run_assets,
    upsert_asset,
)
import vfieval.media_assets as media_assets_module
from vfieval.pipeline.inference import run_inference_job
from vfieval.media_items import ensure_canonical_gt_item, register_external_prediction
from vfieval.uploads import (
    complete_upload_session,
    create_upload_session,
    receive_upload_part,
)

from v13_test_utils import get_json, make_workspace, post_json, start_server, stop_server, write_mp4


def _png_bytes(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _frame_zip(entries: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, content in entries:
            bundle.writestr(name, content)
    return buffer.getvalue()


class MediaCatalogUploadTests(unittest.TestCase):
    def test_folder_catalog_triplets_use_only_real_symmetric_centers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection = create_collection(
                db,
                "Folder Group",
                metadata={"source_kind": "folder", "video_group": "group-a"},
            )
            upsert_asset(
                db,
                collection_id=collection["id"],
                source_key="folder:group-a/clip.mp4",
                source_kind="folder",
                media_kind="video",
                role="gt",
                display_name="clip.mp4",
                original_name="clip.mp4",
                storage_path=workspace.root.parent / "videos" / "group-a" / "clip.mp4",
                frame_count=10,
                width=16,
                height=8,
                fps=24,
            )

            full = list_folder_group_videos(db, "group-a", frame_step=2)
            limited = list_folder_group_videos(db, "group-a", frame_step=2, max_frames=7)

            self.assertEqual(full["videos"][0]["valid_triplets"], 6)
            self.assertEqual(limited["videos"][0]["valid_triplets"], 3)

    def test_run_asset_publication_failure_invalidates_unbound_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            model_id = db.register_model("asset-failure", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("asset-failure", tmp, False)
            run_id = db.create_run(
                "asset-failure", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", []
            )
            job_id = int(db.get_run(run_id)["inference_job_id"])
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertEqual(int(db.claim_next_job("asset-failure", ["inference"])["id"]), job_id)
            video_path = write_mp4(
                workspace.runs_dir / str(run_id) / "videos" / "clip" / "pred.mp4",
                [(0, 0, 0), (10, 0, 0)],
            )
            db.add_artifact(
                job_id,
                None,
                "pred_video",
                str(video_path),
                "video/mp4",
                {"video_name": "clip", "frames": 2, "width": 8, "height": 8, "fps": 5},
            )
            result = {"samples": 2}
            self.assertTrue(
                db.complete_run_inference(
                    run_id,
                    result,
                    db.summarize_run_artifacts(run_id),
                    "completed",
                    source_job_id=job_id,
                    source_job_result=result,
                )
            )

            with patch(
                "vfieval.media_assets.bind_run_asset",
                side_effect=RuntimeError("injected binding failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected binding failure"):
                    sync_run_assets(db, workspace, run_id)

            assets = db.query(
                "SELECT state FROM media_assets WHERE source_kind = 'run_artifact'"
            )
            self.assertTrue(assets)
            self.assertEqual({row["state"] for row in assets}, {"unavailable"})
            self.assertEqual(
                int(db.get("SELECT COUNT(*) AS count FROM run_media_assets WHERE run_id = ?", (run_id,))["count"]),
                0,
            )

    def test_run_asset_publication_rechecks_canonical_video_before_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            model_id = db.register_model("asset-toctou", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("asset-toctou", tmp, False)
            run_id = db.create_run(
                "asset-toctou", model_id, dataset_id, 8, 8, 1, "cpu", "fp32", []
            )
            job_id = int(db.get_run(run_id)["inference_job_id"])
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertEqual(int(db.claim_next_job("asset-toctou", ["inference"])["id"]), job_id)
            video_path = write_mp4(
                workspace.runs_dir / str(run_id) / "videos" / "clip" / "pred.mp4",
                [(0, 0, 0), (10, 0, 0)],
            )
            db.add_artifact(
                job_id,
                None,
                "pred_video",
                str(video_path),
                "video/mp4",
                {"video_name": "clip", "frames": 2, "width": 8, "height": 8, "fps": 5},
            )
            result = {"samples": 2}
            self.assertTrue(
                db.complete_run_inference(
                    run_id,
                    result,
                    db.summarize_run_artifacts(run_id),
                    "completed",
                    source_job_id=job_id,
                    source_job_result=result,
                )
            )

            original_bind = media_assets_module.bind_run_asset

            def bind_then_remove(*args, **kwargs):
                original_bind(*args, **kwargs)
                self.assertEqual(get_asset(db, int(args[2]))["state"], "unavailable")
                video_path.unlink()

            with patch("vfieval.media_assets.bind_run_asset", side_effect=bind_then_remove):
                with self.assertRaisesRegex(ValueError, "missing or empty"):
                    sync_run_assets(db, workspace, run_id)

            assets = db.query(
                "SELECT state FROM media_assets WHERE source_kind = 'run_artifact'"
            )
            self.assertTrue(assets)
            self.assertEqual({row["state"] for row in assets}, {"unavailable"})

    def test_run_asset_publication_rejects_missing_required_video_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            model_id = db.register_model("asset-row-toctou", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("asset-row-toctou", tmp, False)
            db.add_sample(
                dataset_id,
                "clip_000000",
                str(Path(tmp) / "img0.png"),
                str(Path(tmp) / "img1.png"),
                None,
                {"source_type": "video", "video_name": "clip", "frame_index": 0},
            )
            run_id = db.create_run(
                "asset-row-toctou",
                model_id,
                dataset_id,
                8,
                8,
                1,
                "cpu",
                "fp32",
                [],
                metadata={
                    "artifact_contract": "canonical-v1",
                    "request": {"artifact_contract": "canonical-v1"},
                },
            )
            job_id = int(db.get_run(run_id)["inference_job_id"])
            self.assertTrue(db.mark_run_started(run_id, "running"))
            self.assertEqual(int(db.claim_next_job("asset-row-toctou", ["inference"])["id"]), job_id)
            video_path = write_mp4(
                workspace.runs_dir / str(run_id) / "videos" / "clip" / "pred.mp4",
                [(0, 0, 0), (10, 0, 0)],
            )
            artifact_id = db.add_artifact(
                job_id,
                None,
                "pred_video",
                str(video_path),
                "video/mp4",
                {"video_name": "clip", "frames": 2, "width": 8, "height": 8, "fps": 5},
            )
            result = {"samples": 1}
            self.assertTrue(
                db.complete_run_inference(
                    run_id,
                    result,
                    db.summarize_run_artifacts(run_id),
                    "completed",
                    source_job_id=job_id,
                    source_job_result=result,
                )
            )
            with db.connection() as conn:
                conn.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))

            with self.assertRaisesRegex(ValueError, "artifact contract changed"):
                sync_run_assets(db, workspace, run_id)
            self.assertEqual(
                db.get("SELECT COUNT(*) AS count FROM media_assets")["count"],
                0,
            )

    def test_schema_migration_backup_is_created_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            with db.connection() as conn:
                conn.execute("DELETE FROM schema_migrations")
            db.init()
            backups = list(workspace.backups_dir.glob("*/vfieval.sqlite"))
            self.assertEqual(len(backups), 1)
            db.init()
            self.assertEqual(list(workspace.backups_dir.glob("*/vfieval.sqlite")), backups)

    def test_folder_assets_backfill_idempotently_and_drive_source_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VFIEVAL_PROJECT_ROOT": tmp}, clear=False):
            workspace, db = make_workspace(tmp)
            source = Path(tmp) / "videos" / "demo" / "clip.mp4"
            source.parent.mkdir(parents=True)
            write_mp4(source, [(0, 0, 0), (20, 0, 0), (40, 0, 0)])

            self.assertEqual(sync_folder_assets(db, workspace), 1)
            first = list_assets(db, source_kind="folder")["assets"]
            self.assertEqual(len(first[0]["content_sha256"]), 64)
            self.assertEqual(sync_folder_assets(db, workspace), 1)
            second = list_assets(db, source_kind="folder")["assets"]
            self.assertEqual([row["id"] for row in first], [row["id"] for row in second])

            converted = source_assets_to_video_payload(
                db,
                workspace,
                {"source_assets": [{"asset_id": first[0]["id"]}], "model_file": "test_average.py"},
            )
            self.assertEqual(converted["video_group"], "demo")
            self.assertEqual(converted["selected_videos"], ["clip.mp4"])

    def test_frame_zip_upload_is_resumable_idempotent_and_hash_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection = create_collection(db, "External")
            payload = _frame_zip(
                [
                    ("frames/0001.png", _png_bytes((8, 6), (10, 20, 30))),
                    ("frames/0002.png", _png_bytes((8, 6), (30, 20, 10))),
                ]
            )
            digest = hashlib.sha256(payload).hexdigest()
            session = create_upload_session(
                db,
                workspace,
                {
                    "collection_id": collection["id"],
                    "role": "gt",
                    "media_kind": "frame_sequence",
                    "display_name": "external-gt",
                    "original_name": "frames.zip",
                    "size_bytes": len(payload),
                    "sha256": digest,
                    "fps": 24,
                },
            )
            uploaded = receive_upload_part(
                db,
                workspace,
                session["id"],
                0,
                payload,
                offset_bytes=0,
                sha256=digest,
            )
            self.assertEqual(uploaded["received_bytes"], len(payload))
            repeated = receive_upload_part(
                db,
                workspace,
                session["id"],
                0,
                payload,
                offset_bytes=0,
                sha256=digest,
            )
            self.assertEqual(repeated["part_count"], 1)

            completed = complete_upload_session(db, workspace, session["id"])
            asset = get_asset(db, completed["asset_id"])
            self.assertEqual(asset["frame_count"], 2)
            self.assertEqual((asset["width"], asset["height"]), (8, 6))
            self.assertEqual(asset["fps"], 24)
            self.assertTrue(Path(asset["storage_path"]).is_dir())
            self.assertEqual(complete_upload_session(db, workspace, session["id"])["asset_id"], asset["id"])

    def test_upload_sessions_reject_internal_collections_but_allow_sources_and_uploads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collections = {
                "Sources": create_collection(db, "Sources", metadata={"source_kind": "folder"}),
                "Uploads": create_collection(db, "Uploads", metadata={"source_kind": "upload"}),
                "Run artifacts": create_collection(
                    db,
                    "Run artifacts",
                    metadata={"source_kind": "run_artifact"},
                ),
                "Evaluation package": create_collection(
                    db,
                    "Evaluation package",
                    metadata={"source_kind": "evaluation_package"},
                ),
            }
            body = {
                "role": "pred",
                "media_kind": "video",
                "original_name": "pred.mp4",
                "size_bytes": 1,
                "sha256": "a" * 64,
            }

            for name in ("Sources", "Uploads"):
                session = create_upload_session(
                    db,
                    workspace,
                    {**body, "collection_id": collections[name]["id"], "display_name": f"{name} Pred"},
                )
                self.assertEqual(session["collection_id"], collections[name]["id"])
                self.assertEqual(session["state"], "uploading")

            for name in ("Run artifacts", "Evaluation package"):
                with self.subTest(collection=name):
                    with self.assertRaisesRegex(ValueError, "user-managed Collection"):
                        create_upload_session(
                            db,
                            workspace,
                            {
                                **body,
                                "collection_id": collections[name]["id"],
                                "display_name": f"{name} Pred",
                            },
                        )
            self.assertEqual(db.get("SELECT COUNT(*) AS count FROM upload_sessions")["count"], 2)

    def test_upload_rejects_bad_hash_mixed_dimensions_and_traversal(self) -> None:
        cases = [
            _frame_zip(
                [
                    ("0001.png", _png_bytes((8, 8), (0, 0, 0))),
                    ("0002.png", _png_bytes((9, 8), (0, 0, 0))),
                ]
            ),
            _frame_zip([("../escape.png", _png_bytes((8, 8), (0, 0, 0)))]),
        ]
        for index, payload in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp:
                workspace, db = make_workspace(tmp)
                collection = create_collection(db, "External")
                digest = hashlib.sha256(payload).hexdigest()
                session = create_upload_session(
                    db,
                    workspace,
                    {
                        "collection_id": collection["id"],
                        "role": "pred",
                        "media_kind": "frame_sequence",
                        "display_name": f"bad-{index}",
                        "original_name": "bad.zip",
                        "size_bytes": len(payload),
                        "sha256": digest,
                        "fps": 24,
                    },
                )
                with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                    receive_upload_part(
                        db,
                        workspace,
                        session["id"],
                        0,
                        payload,
                        offset_bytes=0,
                        sha256="0" * 64,
                    )
                receive_upload_part(
                    db,
                    workspace,
                    session["id"],
                    0,
                    payload,
                    offset_bytes=0,
                    sha256=digest,
                )
                with self.assertRaises(ValueError):
                    complete_upload_session(db, workspace, session["id"])

    def test_unreferenced_upload_soft_delete_removes_only_asset_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection = create_collection(db, "External")
            asset_dir = workspace.media_dir / "external" / "asset-1"
            asset_dir.mkdir(parents=True)
            source = asset_dir / "source.mp4"
            source.write_bytes(b"test")
            from vfieval.media_assets import upsert_asset

            asset = upsert_asset(
                db,
                collection_id=collection["id"],
                source_key="upload:test",
                source_kind="upload",
                media_kind="video",
                role="pred",
                display_name="pred",
                original_name="pred.mp4",
                storage_path=source,
            )
            deleted = soft_delete_asset(db, workspace, asset["id"])
            self.assertTrue(deleted["content_removed"])
            self.assertFalse(asset_dir.exists())
            self.assertTrue(workspace.media_dir.exists())
            self.assertEqual(get_asset(db, asset["id"], include_deleted=True)["state"], "deleted")

    def test_media_content_supports_http_byte_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection = create_collection(db, "External")
            asset_dir = workspace.media_dir / "external" / "asset-range"
            asset_dir.mkdir(parents=True)
            source = asset_dir / "source.mp4"
            source.write_bytes(b"0123456789")
            from vfieval.media_assets import upsert_asset

            asset = upsert_asset(
                db,
                collection_id=collection["id"],
                source_key="upload:range",
                source_kind="upload",
                media_kind="video",
                role="pred",
                display_name="range",
                original_name="range.mp4",
                storage_path=source,
            )
            server, thread, base_url = start_server(db, workspace)
            try:
                request = urllib.request.Request(
                    f"{base_url}/api/media/assets/{asset['id']}/content",
                    headers={"Range": "bytes=2-5"},
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.headers.get("Accept-Ranges"), "bytes")
                    self.assertEqual(response.read(), b"2345")
            finally:
                stop_server(server, thread)

    def test_upload_http_api_accepts_out_of_order_safe_parts_and_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            payload = _frame_zip([("0001.png", _png_bytes((8, 8), (1, 2, 3)))])
            digest = hashlib.sha256(payload).hexdigest()
            server, thread, base_url = start_server(db, workspace)
            try:
                collection = post_json(base_url, "/api/media/collections", {"name": "HTTP"})["collection"]
                session = post_json(
                    base_url,
                    "/api/uploads",
                    {
                        "collection_id": collection["id"],
                        "role": "gt",
                        "media_kind": "frame_sequence",
                        "display_name": "HTTP GT",
                        "original_name": "frames.zip",
                        "total_size": len(payload),
                        "sha256": digest,
                        "fps": 12,
                    },
                )["upload"]
                request = urllib.request.Request(
                    f"{base_url}/api/uploads/{session['id']}/parts/0",
                    data=payload,
                    method="PUT",
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Content-Range": f"bytes 0-{len(payload) - 1}/{len(payload)}",
                        "X-Chunk-SHA256": digest,
                    },
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    self.assertEqual(response.status, 200)
                completed = post_json(base_url, f"/api/uploads/{session['id']}/complete", {})
                self.assertGreater(completed["asset_id"], 0)
                assets = get_json(base_url, f"/api/media/assets?collection_id={collection['id']}")
                self.assertEqual([row["display_name"] for row in assets["assets"]], ["HTTP GT"])
            finally:
                stop_server(server, thread)

    def test_media_asset_compare_records_aligned_gt_and_provenance_relations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            collection = create_collection(db, "Compare assets")
            paths = []
            for name, colors in (
                ("gt", [(0, 0, 0), (20, 0, 0), (40, 0, 0)]),
                ("pred", [(0, 1, 0), (20, 1, 0), (40, 1, 0)]),
            ):
                folder = workspace.media_dir / "compare" / name
                folder.mkdir(parents=True)
                paths.append(write_mp4(folder / f"{name}.mp4", colors))
            gt = upsert_asset(
                db,
                collection_id=collection["id"],
                source_key="upload:compare-gt",
                source_kind="upload",
                media_kind="video",
                role="gt",
                display_name="GT",
                original_name="gt.mp4",
                storage_path=paths[0],
                frame_count=3,
                width=8,
                height=8,
                fps=5,
            )
            pred = upsert_asset(
                db,
                collection_id=collection["id"],
                source_key="upload:compare-pred",
                source_kind="upload",
                media_kind="video",
                role="pred",
                display_name="Pred",
                original_name="pred.mp4",
                storage_path=paths[1],
                frame_count=3,
                width=8,
                height=8,
                fps=5,
            )
            item = ensure_canonical_gt_item(db, int(gt["id"]))
            member = register_external_prediction(
                db,
                int(item["id"]),
                int(pred["id"]),
                method_key="external-pred",
                temporal_mapping={"source_frame_indices": [0, 1, 2], "fps": 5},
                spatial_origin={"width": 8, "height": 8},
                aspect_stretch_confirmed=False,
            )
            server, thread, base_url = start_server(db, workspace)
            try:
                with patch("vfieval.server._start_local_inference_worker", return_value=None):
                    created = post_json(
                        base_url,
                        "/api/runs",
                        {
                            "run_type": "video_compare",
                            "media_item_id": int(item["id"]),
                            "pred_member_ids": [int(member["id"])],
                            "spatial_policy": {"mode": "smallest_pred", "filter": "lanczos"},
                            "metrics": [],
                        },
                )
                run_id = int(created["run_id"])
                job_id = int(db.get_run(run_id)["inference_job_id"])
                self.assertEqual(int(db.claim_next_job("catalog-compare", ["inference"])["id"]), job_id)
                result = run_inference_job(db, workspace, job_id)
                self.assertTrue(db.complete_job(job_id, result.__dict__))
                assets = sync_run_assets(db, workspace, run_id)
                aligned = next(row for row in assets if row["role"] == "gt")
                self.assertTrue(aligned["metadata"]["aligned_gt"])
                relations = db.query(
                    "SELECT relation_type FROM media_asset_relations WHERE parent_asset_id = ? AND child_asset_id = ?",
                    (gt["id"], aligned["id"]),
                )
                self.assertEqual({row["relation_type"] for row in relations}, {"generated_from", "aligned_gt_of"})
            finally:
                stop_server(server, thread)


if __name__ == "__main__":
    unittest.main()
