from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from vfieval.compare_inputs import IMAGE_SUFFIXES
from vfieval.config import WorkspaceConfig
from vfieval.db import Database, utc_ts
from vfieval.file_inputs import VIDEO_SUFFIXES
from vfieval.media_assets import get_collection, inspect_uploaded_video, slugify, upsert_asset


UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024
UPLOAD_MAX_BYTES = int(os.getenv("VFIEVAL_UPLOAD_MAX_BYTES", str(50 * 1024**3)))
UPLOAD_COLLECTION_QUOTA_BYTES = int(os.getenv("VFIEVAL_COLLECTION_QUOTA_BYTES", str(500 * 1024**3)))
UPLOAD_MAX_FILES = int(os.getenv("VFIEVAL_UPLOAD_MAX_FILES", "200000"))
UPLOAD_STALE_SECONDS = int(os.getenv("VFIEVAL_UPLOAD_STALE_SECONDS", str(24 * 60 * 60)))


def _json(data: Any) -> str:
    return json.dumps(data if data is not None else {}, sort_keys=True, ensure_ascii=False)


def _loads(text: str | None) -> Any:
    return json.loads(text) if text else {}


def _valid_sha256(value: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError("sha256 must be a 64-character hexadecimal digest")
    return text


def create_upload_session(db: Database, workspace: WorkspaceConfig, body: dict[str, Any]) -> dict[str, Any]:
    collection_id = int(body.get("collection_id") or 0)
    collection = get_collection(db, collection_id)
    role = str(body.get("role") or "").strip()
    media_kind = str(body.get("media_kind") or "video").strip()
    display_name = str(body.get("display_name") or "").strip()
    original_name = Path(str(body.get("original_name") or "").strip()).name
    expected_size = int(body.get("size_bytes") or body.get("total_size") or 0)
    expected_sha256 = _valid_sha256(str(body.get("sha256") or ""))
    fps_raw = body.get("fps")
    fps = float(fps_raw) if fps_raw not in {None, ""} else None
    if role not in {"gt", "pred"}:
        raise ValueError("upload role must be gt or pred")
    if media_kind not in {"video", "frame_sequence"}:
        raise ValueError("upload media_kind must be video or frame_sequence")
    if not display_name or not original_name:
        raise ValueError("display_name and original_name are required")
    if expected_size <= 0 or expected_size > UPLOAD_MAX_BYTES:
        raise ValueError(f"upload size must be between 1 and {UPLOAD_MAX_BYTES} bytes")
    usage = db.get(
        """
        SELECT
          COALESCE((SELECT SUM(size_bytes) FROM media_assets
                    WHERE collection_id = ? AND source_kind = 'upload' AND deleted_at IS NULL), 0)
          + COALESCE((SELECT SUM(expected_size) FROM upload_sessions
                      WHERE collection_id = ? AND state IN ('uploading', 'assembling', 'validating')), 0)
          AS bytes
        """,
        (collection_id, collection_id),
    )
    if int((usage or {}).get("bytes") or 0) + expected_size > UPLOAD_COLLECTION_QUOTA_BYTES:
        raise ValueError(f"collection upload quota of {UPLOAD_COLLECTION_QUOTA_BYTES} bytes would be exceeded")
    if media_kind == "video" and Path(original_name).suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("uploaded video has an unsupported file extension")
    if media_kind == "frame_sequence":
        if Path(original_name).suffix.lower() != ".zip":
            raise ValueError("frame_sequence uploads must be ZIP files")
        if fps is None or fps <= 0:
            raise ValueError("frame_sequence uploads require a positive fps")
    duplicate = db.get(
        """
        SELECT id FROM media_assets
        WHERE collection_id = ? AND display_name = ? AND deleted_at IS NULL
        """,
        (collection_id, display_name),
    )
    if duplicate is not None:
        raise FileExistsError("asset display_name already exists in this collection")
    active = db.get(
        """
        SELECT id FROM upload_sessions
        WHERE collection_id = ? AND display_name = ? AND state IN ('uploading', 'assembling', 'validating')
        """,
        (collection_id, display_name),
    )
    if active is not None:
        raise FileExistsError("an active upload already uses this display_name")
    upload_id = uuid.uuid4().hex
    now = utc_ts()
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO upload_sessions(
                id, collection_id, role, media_kind, display_name, original_name,
                expected_size, expected_sha256, fps, chunk_size, state,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'uploading', ?, ?, ?)
            """,
            (
                upload_id,
                collection_id,
                role,
                media_kind,
                display_name[:240],
                original_name[:500],
                expected_size,
                expected_sha256,
                fps,
                UPLOAD_CHUNK_SIZE,
                _json({"collection_slug": collection["slug"]}),
                now,
                now,
            ),
        )
    (workspace.uploads_dir / upload_id).mkdir(parents=True, exist_ok=False)
    return get_upload_session(db, upload_id)


def get_upload_session(db: Database, upload_id: str) -> dict[str, Any]:
    row = db.get("SELECT * FROM upload_sessions WHERE id = ?", (str(upload_id),))
    if row is None:
        raise KeyError(f"upload session {upload_id} not found")
    row["metadata"] = _loads(row.pop("metadata_json", None))
    row["error"] = _loads(row.pop("error_json", None))
    parts = db.query(
        "SELECT part_index, offset_bytes, size_bytes, sha256 FROM upload_parts WHERE upload_id = ? ORDER BY part_index",
        (str(upload_id),),
    )
    row["parts"] = parts
    row["part_count"] = len(parts)
    row["expected_parts"] = math.ceil(int(row["expected_size"]) / int(row["chunk_size"]))
    return row


def receive_upload_part(
    db: Database,
    workspace: WorkspaceConfig,
    upload_id: str,
    part_index: int,
    data: bytes,
    *,
    offset_bytes: int,
    sha256: str,
) -> dict[str, Any]:
    session = get_upload_session(db, upload_id)
    if session["state"] != "uploading":
        raise ValueError(f"upload session is not accepting parts: {session['state']}")
    part_index = int(part_index)
    if part_index < 0:
        raise ValueError("part index must be non-negative")
    expected_offset = part_index * int(session["chunk_size"])
    if int(offset_bytes) != expected_offset:
        raise ValueError(f"part offset must be {expected_offset}")
    expected_length = min(int(session["chunk_size"]), int(session["expected_size"]) - expected_offset)
    if expected_length <= 0 or len(data) != expected_length:
        raise ValueError(f"part size must be {expected_length} bytes")
    digest = hashlib.sha256(data).hexdigest()
    if digest != _valid_sha256(sha256):
        raise ValueError("upload part sha256 mismatch")
    existing = db.get(
        "SELECT * FROM upload_parts WHERE upload_id = ? AND part_index = ?",
        (upload_id, part_index),
    )
    if existing is not None:
        if int(existing["size_bytes"]) == len(data) and str(existing["sha256"]) == digest:
            return get_upload_session(db, upload_id)
        raise ValueError("part index already exists with different content")
    upload_dir = (workspace.uploads_dir / upload_id).resolve()
    _ensure_within(upload_dir, workspace.uploads_dir.resolve())
    upload_dir.mkdir(parents=True, exist_ok=True)
    part_path = upload_dir / f"{part_index:08d}.part"
    with part_path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    now = utc_ts()
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO upload_parts(upload_id, part_index, offset_bytes, size_bytes, sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (upload_id, part_index, expected_offset, len(data), digest, now),
        )
        conn.execute(
            """
            UPDATE upload_sessions
            SET received_bytes = (
                    SELECT COALESCE(SUM(size_bytes), 0) FROM upload_parts WHERE upload_id = ?
                ), updated_at = ?
            WHERE id = ?
            """,
            (upload_id, now, upload_id),
        )
    return get_upload_session(db, upload_id)


def complete_upload_session(db: Database, workspace: WorkspaceConfig, upload_id: str) -> dict[str, Any]:
    session = get_upload_session(db, upload_id)
    if session["state"] == "completed":
        asset_id = int((session.get("metadata") or {}).get("asset_id") or 0)
        return {"upload": session, "asset_id": asset_id}
    if session["state"] != "uploading":
        raise ValueError(f"upload session cannot be completed: {session['state']}")
    expected_parts = int(session["expected_parts"])
    indices = [int(part["part_index"]) for part in session["parts"]]
    if indices != list(range(expected_parts)) or int(session["received_bytes"]) != int(session["expected_size"]):
        raise ValueError("upload is incomplete")
    _set_state(db, upload_id, "assembling")
    upload_dir = (workspace.uploads_dir / upload_id).resolve()
    assembled = upload_dir / "assembled.bin"
    digest = hashlib.sha256()
    final_dir: Path | None = None
    try:
        with assembled.open("xb") as output:
            for index in range(expected_parts):
                part_path = upload_dir / f"{index:08d}.part"
                with part_path.open("rb") as source:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != str(session["expected_sha256"]):
            raise ValueError("complete upload sha256 mismatch")
        _set_state(db, upload_id, "validating")
        collection = get_collection(db, int(session["collection_id"]))
        asset_uuid = uuid.uuid4().hex
        final_dir = (workspace.media_dir / slugify(collection["slug"]) / asset_uuid).resolve()
        _ensure_within(final_dir, workspace.media_dir.resolve())
        final_dir.mkdir(parents=True, exist_ok=False)
        if session["media_kind"] == "video":
            suffix = Path(str(session["original_name"])).suffix.lower()
            target = final_dir / f"source{suffix}"
            shutil.move(str(assembled), str(target))
            info = inspect_uploaded_video(workspace, target)
            storage_path = target
            mime_type = str(info.get("mime_type") or "video/mp4")
            frame_count = int(info.get("frame_count") or 0)
            width = int(info.get("width") or 0)
            height = int(info.get("height") or 0)
            fps = info.get("fps")
            media_metadata = {
                "duration_seconds": info.get("duration_seconds"),
                "metadata_source": info.get("metadata_source"),
            }
        else:
            frames_dir = final_dir / "frames"
            frame_info = _extract_frame_zip(assembled, frames_dir, int(session["expected_size"]))
            storage_path = frames_dir
            mime_type = "application/x-vfieval-frame-sequence"
            frame_count = int(frame_info["frame_count"])
            width = int(frame_info["width"])
            height = int(frame_info["height"])
            fps = float(session["fps"])
            media_metadata = {"frame_names": frame_info["frame_names"], "archive_sha256": digest.hexdigest()}
            assembled.unlink(missing_ok=True)
        asset = upsert_asset(
            db,
            collection_id=int(session["collection_id"]),
            source_key=f"upload:{upload_id}",
            source_kind="upload",
            media_kind=str(session["media_kind"]),
            role=str(session["role"]),
            display_name=str(session["display_name"]),
            original_name=str(session["original_name"]),
            storage_path=storage_path,
            mime_type=mime_type,
            state="ready",
            content_sha256=digest.hexdigest(),
            size_bytes=int(session["expected_size"]),
            frame_count=frame_count,
            width=width,
            height=height,
            fps=fps,
            provenance={"upload_id": upload_id, "external": True},
            metadata=media_metadata,
        )
        now = utc_ts()
        metadata = dict(session.get("metadata") or {})
        metadata["asset_id"] = int(asset["id"])
        with db.connection() as conn:
            conn.execute(
                """
                UPDATE upload_sessions
                SET state = 'completed', metadata_json = ?, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (_json(metadata), now, now, upload_id),
            )
        _remove_parts(upload_dir)
        return {"upload": get_upload_session(db, upload_id), "asset": asset, "asset_id": int(asset["id"])}
    except Exception as exc:
        if final_dir is not None and final_dir.exists():
            _ensure_within(final_dir, workspace.media_dir.resolve())
            shutil.rmtree(final_dir)
        _set_state(db, upload_id, "failed", {"type": type(exc).__name__, "message": str(exc)[:1000]})
        if isinstance(exc, (zipfile.BadZipFile, EOFError)):
            raise ValueError("uploaded frame ZIP is invalid") from exc
        raise


def delete_upload_session(db: Database, workspace: WorkspaceConfig, upload_id: str) -> bool:
    session = get_upload_session(db, upload_id)
    if session["state"] == "completed":
        raise ValueError("completed uploads must be managed through their media asset")
    upload_dir = (workspace.uploads_dir / upload_id).resolve()
    _ensure_within(upload_dir, workspace.uploads_dir.resolve())
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    with db.connection() as conn:
        conn.execute("DELETE FROM upload_sessions WHERE id = ?", (upload_id,))
    return True


def cleanup_stale_uploads(db: Database, workspace: WorkspaceConfig) -> list[str]:
    cutoff = utc_ts() - UPLOAD_STALE_SECONDS
    rows = db.query(
        """
        SELECT id FROM upload_sessions
        WHERE state IN ('uploading', 'failed') AND updated_at < ?
        """,
        (cutoff,),
    )
    deleted: list[str] = []
    for row in rows:
        upload_id = str(row["id"])
        try:
            delete_upload_session(db, workspace, upload_id)
            deleted.append(upload_id)
        except (KeyError, ValueError):
            continue
    return deleted


def _set_state(db: Database, upload_id: str, state: str, error: dict[str, Any] | None = None) -> None:
    with db.connection() as conn:
        conn.execute(
            "UPDATE upload_sessions SET state = ?, error_json = ?, updated_at = ? WHERE id = ?",
            (state, _json(error), utc_ts(), upload_id),
        )


def _remove_parts(upload_dir: Path) -> None:
    for path in upload_dir.glob("*.part"):
        path.unlink(missing_ok=True)


def _ensure_within(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("upload path resolved outside the managed upload root") from exc


def _zip_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _safe_zip_member(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename.replace("\\", "/")
    path = PurePosixPath(name)
    drive_like = bool(path.parts and len(path.parts[0]) == 2 and path.parts[0][1] == ":")
    if path.is_absolute() or drive_like or ".." in path.parts or any(part in {"", "."} for part in path.parts):
        raise ValueError(f"unsafe ZIP member path: {info.filename}")
    if _zip_is_symlink(info):
        raise ValueError(f"ZIP symbolic links are not allowed: {info.filename}")
    return path


def _extract_frame_zip(archive: Path, target_dir: Path, compressed_size: int) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        files = [info for info in bundle.infolist() if not info.is_dir()]
        if not files or len(files) > UPLOAD_MAX_FILES:
            raise ValueError(f"frame ZIP must contain between 1 and {UPLOAD_MAX_FILES} files")
        total_uncompressed = sum(int(info.file_size) for info in files)
        if total_uncompressed > UPLOAD_MAX_BYTES:
            raise ValueError("frame ZIP uncompressed size exceeds upload limit")
        if compressed_size > 0 and total_uncompressed > min(UPLOAD_MAX_BYTES, compressed_size * 200):
            raise ValueError("frame ZIP expansion ratio is unsafe")
        safe_files: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        for info in files:
            safe = _safe_zip_member(info)
            if safe.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"frame ZIP contains unsupported file: {info.filename}")
            safe_files.append((info, safe))
        safe_files.sort(key=lambda item: _natural_key(item[1].as_posix()))
        frame_names: list[str] = []
        expected_size: tuple[int, int] | None = None
        for index, (info, safe) in enumerate(safe_files):
            suffix = safe.suffix.lower()
            output = target_dir / f"{index:08d}{suffix}"
            with bundle.open(info, "r") as source, output.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            try:
                with Image.open(output) as image:
                    image.verify()
                with Image.open(output) as image:
                    size = image.size
            except Exception as exc:
                raise ValueError(f"frame ZIP contains an invalid image: {info.filename}") from exc
            if expected_size is None:
                expected_size = size
            elif size != expected_size:
                raise ValueError("frame ZIP contains mixed frame dimensions")
            frame_names.append(output.name)
    if expected_size is None:
        raise ValueError("frame ZIP contains no images")
    manifest = {
        "frame_count": len(frame_names),
        "width": int(expected_size[0]),
        "height": int(expected_size[1]),
        "frame_names": frame_names,
    }
    (target_dir.parent / "manifest.json").write_text(_json(manifest), encoding="utf-8")
    return manifest


def _natural_key(value: str) -> list[Any]:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]
