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
    soft_delete_asset,
    source_assets_to_video_payload,
    sync_folder_assets,
    sync_run_assets,
    upsert_asset,
)
from vfieval.pipeline.inference import run_inference_job
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
            server, thread, base_url = start_server(db, workspace)
            try:
                with patch("vfieval.server._start_local_inference_worker", return_value=None):
                    created = post_json(
                        base_url,
                        "/api/runs",
                        {
                            "run_type": "video_compare",
                            "reference": {"kind": "media_asset", "asset_id": gt["id"]},
                            "distorted": [{"kind": "media_asset", "asset_id": pred["id"], "label": "Pred"}],
                            "metrics": [],
                        },
                    )
                run_id = int(created["run_id"])
                run_inference_job(db, workspace, int(db.get_run(run_id)["inference_job_id"]))
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
