from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import shutil
from pathlib import Path
from typing import Any

from vfieval.config import WorkspaceConfig
from vfieval.db import Database, utc_ts
from vfieval.file_inputs import VIDEO_SUFFIXES, inspect_video, list_video_groups, videos_dir


MEDIA_ROLES = {"source", "gt", "pred"}
MEDIA_KINDS = {"video", "frame_sequence"}
SOURCE_KINDS = {"folder", "upload", "run_artifact", "evaluation_package"}
MEDIA_STATES = {"ready", "unavailable", "deleted", "invalid"}
CANONICAL_VIDEO_ARTIFACT_KINDS = ("pred_video", "gt_video", "diff_video")


def _json(data: Any) -> str:
    return json.dumps(data if data is not None else {}, sort_keys=True, ensure_ascii=False)


def _loads(text: str | None) -> Any:
    return json.loads(text) if text else {}


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-.").lower()
    if slug:
        return slug[:80]
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def _file_sha256(db: Database, source_key: str, path: Path) -> str:
    stat_result = path.stat()
    existing = db.get(
        "SELECT content_sha256, metadata_json, size_bytes FROM media_assets WHERE source_key = ?",
        (str(source_key),),
    )
    if existing is not None and existing.get("content_sha256"):
        metadata = _loads(existing.get("metadata_json"))
        if (
            int(existing.get("size_bytes") or 0) == int(stat_result.st_size)
            and int(metadata.get("source_mtime_ns") or 0) == int(stat_result.st_mtime_ns)
        ):
            return str(existing["content_sha256"])
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_collection(
    db: Database,
    name: str,
    slug: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = str(name or "").strip()
    if not name:
        raise ValueError("collection name is required")
    resolved_slug = slugify(slug or name)
    now = utc_ts()
    with db.connection() as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO media_collections(name, slug, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name[:200], resolved_slug, _json(metadata), now, now),
            )
        except Exception as exc:
            if "UNIQUE constraint failed" in str(exc):
                raise ValueError("collection name or slug already exists") from exc
            raise
        collection_id = int(cur.lastrowid)
    return get_collection(db, collection_id)


def ensure_collection(
    db: Database,
    name: str,
    slug: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = db.get("SELECT * FROM media_collections WHERE slug = ?", (slugify(slug),))
    if row is None:
        return create_collection(db, name, slug, metadata)
    if metadata is not None:
        with db.connection() as conn:
            conn.execute(
                "UPDATE media_collections SET name = ?, metadata_json = ?, updated_at = ? WHERE id = ?",
                (str(name)[:200], _json(metadata), utc_ts(), int(row["id"])),
            )
    return get_collection(db, int(row["id"]))


def get_collection(db: Database, collection_id: int) -> dict[str, Any]:
    row = db.get("SELECT * FROM media_collections WHERE id = ?", (int(collection_id),))
    if row is None:
        raise KeyError(f"media collection {collection_id} not found")
    row["metadata"] = _loads(row.pop("metadata_json", None))
    return row


def list_collections(db: Database, *, include_internal: bool = False) -> list[dict[str, Any]]:
    rows = db.query(
        """
        SELECT c.*, COUNT(a.id) AS asset_count
        FROM media_collections c
        LEFT JOIN media_assets a ON a.collection_id = c.id AND a.deleted_at IS NULL
        GROUP BY c.id
        ORDER BY c.name, c.id
        """
    )
    for row in rows:
        row["metadata"] = _loads(row.pop("metadata_json", None))
        row["asset_count"] = int(row.get("asset_count") or 0)
    if not include_internal:
        rows = [
            row
            for row in rows
            if str((row.get("metadata") or {}).get("source_kind") or "")
            not in {"run_artifact", "evaluation_package"}
        ]
    return rows


def upsert_asset(
    db: Database,
    *,
    collection_id: int | None,
    source_key: str,
    source_kind: str,
    media_kind: str,
    role: str,
    display_name: str,
    original_name: str,
    storage_path: str | Path,
    mime_type: str | None = None,
    state: str = "ready",
    content_sha256: str | None = None,
    size_bytes: int = 0,
    frame_count: int = 0,
    width: int = 0,
    height: int = 0,
    fps: float | None = None,
    provenance: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"unsupported media source_kind: {source_kind}")
    if media_kind not in MEDIA_KINDS:
        raise ValueError(f"unsupported media_kind: {media_kind}")
    if role not in MEDIA_ROLES:
        raise ValueError(f"unsupported media role: {role}")
    if state not in MEDIA_STATES:
        raise ValueError(f"unsupported media state: {state}")
    source_key = str(source_key or "").strip()
    display_name = str(display_name or "").strip()
    if not source_key or not display_name:
        raise ValueError("source_key and display_name are required")
    path = Path(storage_path).resolve()
    guessed_mime = mime_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    now = utc_ts()
    existing = db.get("SELECT id FROM media_assets WHERE source_key = ?", (source_key,))
    values = (
        collection_id,
        source_kind,
        media_kind,
        role,
        display_name[:240],
        str(original_name or "")[:500],
        state,
        content_sha256,
        int(size_bytes or 0),
        str(path),
        guessed_mime,
        int(frame_count or 0),
        int(width or 0),
        int(height or 0),
        float(fps) if fps not in {None, ""} else None,
        _json(provenance),
        _json(metadata),
        now,
    )
    with db.connection() as conn:
        if existing is None:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO media_assets(
                        collection_id, source_key, source_kind, media_kind, role,
                        display_name, original_name, state, content_sha256,
                        size_bytes, storage_path, mime_type, frame_count, width,
                        height, fps, provenance_json, metadata_json, created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (values[0], source_key, *values[1:-1], now, values[-1]),
                )
            except Exception as exc:
                if "media_assets.collection_id, media_assets.display_name" in str(exc):
                    raise ValueError("asset display_name already exists in this collection") from exc
                raise
            asset_id = int(cur.lastrowid)
        else:
            asset_id = int(existing["id"])
            conn.execute(
                """
                UPDATE media_assets
                SET collection_id = ?, source_kind = ?, media_kind = ?, role = ?,
                    display_name = ?, original_name = ?, state = ?, content_sha256 = ?,
                    size_bytes = ?, storage_path = ?, mime_type = ?, frame_count = ?,
                    width = ?, height = ?, fps = ?,
                    provenance_json = CASE WHEN provenance_json = '{}' THEN ? ELSE provenance_json END,
                    metadata_json = ?,
                    updated_at = ?, deleted_at = CASE WHEN ? = 'deleted' THEN COALESCE(deleted_at, ?) ELSE NULL END
                WHERE id = ?
                """,
                (*values, state, now, asset_id),
            )
    asset = get_asset(db, asset_id, include_deleted=True)
    if source_kind in {"folder", "upload"} and role == "gt":
        # Media Item identity is created only for authoritative GT sources.
        # Run outputs deliberately require an explicit post-upgrade binding and
        # are never inferred here from names, hashes, or legacy provenance.
        from vfieval.media_items import ensure_canonical_gt_item

        ensure_canonical_gt_item(db, asset_id)
    return asset


def _decode_asset(row: dict[str, Any]) -> dict[str, Any]:
    row["provenance"] = _loads(row.pop("provenance_json", None))
    row["metadata"] = _loads(row.pop("metadata_json", None))
    row["size_bytes"] = int(row.get("size_bytes") or 0)
    row["frame_count"] = int(row.get("frame_count") or 0)
    row["width"] = int(row.get("width") or 0)
    row["height"] = int(row.get("height") or 0)
    return row


def _object(value: Any) -> dict[str, Any]:
    """Decode persisted JSON defensively for provenance-policy checks."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        decoded = _loads(value)
        return dict(decoded) if isinstance(decoded, dict) else {}
    return {}


def _run_is_video_compare(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    metadata = _object(row.get("metadata"))
    if not metadata:
        metadata = _object(row.get("metadata_json"))
    return str(metadata.get("run_type") or "model_inference") == "video_compare"


def is_compare_derived_asset(db: Database, asset: dict[str, Any]) -> bool:
    """Whether a managed Run artifact is owned by a video-Compare Run.

    A model Pred is also bound to a Compare Run as an *input*, so a bare
    ``run_media_assets`` association is not enough to classify it as derived.
    We first use the output asset's provenance (or immutable snapshot marker),
    then only fall back to a non-input binding owned by a Compare Run.  This
    keeps model/external sources reusable while preventing Compare outputs and
    cleanup snapshots from becoming new Compare inputs through legacy asset
    descriptors.
    """
    if str(asset.get("source_kind") or "") != "run_artifact":
        return False

    provenance = _object(asset.get("provenance"))
    if not provenance:
        provenance = _object(asset.get("provenance_json"))
    metadata = _object(asset.get("metadata"))
    if not metadata:
        metadata = _object(asset.get("metadata_json"))
    if bool(provenance.get("compare_snapshot")) or bool(metadata.get("compare_snapshot")):
        return True
    if bool(provenance.get("video_compare_derived")) or bool(metadata.get("video_compare_derived")):
        return True

    for candidate in (provenance.get("run_id"), provenance.get("compare_run_id")):
        try:
            owner_run_id = int(candidate)
        except (TypeError, ValueError):
            continue
        try:
            if _run_is_video_compare(db.get_run(owner_run_id)):
                return True
        except KeyError:
            # An incomplete/stale provenance record must not make a normal
            # catalog lookup fail. The authoritative binding fallback below
            # still catches a known Compare-owned asset.
            continue

    asset_id = asset.get("id")
    try:
        normalized_asset_id = int(asset_id)
    except (TypeError, ValueError):
        return False
    bindings = db.query(
        """
        SELECT r.metadata_json AS run_metadata_json, rma.metadata_json AS binding_metadata_json
        FROM run_media_assets rma
        JOIN runs r ON r.id = rma.run_id
        WHERE rma.asset_id = ?
        """,
        (normalized_asset_id,),
    )
    for binding in bindings:
        if not _run_is_video_compare({"metadata_json": binding.get("run_metadata_json")}):
            continue
        binding_metadata = _object(binding.get("binding_metadata_json"))
        if not bool(binding_metadata.get("input")):
            return True
    return False


def get_asset(db: Database, asset_id: int, include_deleted: bool = False) -> dict[str, Any]:
    clause = "" if include_deleted else " AND a.deleted_at IS NULL"
    row = db.get(
        f"""
        SELECT a.*, c.name AS collection_name, c.slug AS collection_slug
        FROM media_assets a
        LEFT JOIN media_collections c ON c.id = a.collection_id
        WHERE a.id = ?{clause}
        """,
        (int(asset_id),),
    )
    if row is None:
        raise KeyError(f"media asset {asset_id} not found")
    return _decode_asset(row)


def list_assets(
    db: Database,
    *,
    collection_id: int | None = None,
    role: str | None = None,
    source_kind: str | None = None,
    source_kinds: list[str] | tuple[str, ...] | None = None,
    valid_run_outputs: bool = False,
    state: str | None = "ready",
    query: str = "",
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    clauses = ["a.deleted_at IS NULL"]
    params: list[Any] = []
    if collection_id is not None:
        clauses.append("a.collection_id = ?")
        params.append(int(collection_id))
    if role:
        clauses.append("a.role = ?")
        params.append(str(role))
    if source_kind:
        clauses.append("a.source_kind = ?")
        params.append(str(source_kind))
    elif source_kinds:
        normalized_source_kinds = [str(value) for value in source_kinds if str(value)]
        if normalized_source_kinds:
            clauses.append(
                f"a.source_kind IN ({','.join('?' for _value in normalized_source_kinds)})"
            )
            params.extend(normalized_source_kinds)
    if valid_run_outputs:
        clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM run_media_assets rma
                JOIN runs r ON r.id = rma.run_id
                WHERE rma.asset_id = a.id
                  AND rma.role = 'pred'
                  AND r.status IN ('completed', 'metric_queued', 'metric_running')
                  AND r.deleted_at IS NULL
                  AND r.artifact_cleaned_at IS NULL
                  AND COALESCE(json_extract(r.metadata_json, '$.run_type'), 'model_inference') = 'model_inference'
            )
            """
        )
    if state:
        clauses.append("a.state = ?")
        params.append(str(state))
    if query:
        clauses.append("(a.display_name LIKE ? OR a.original_name LIKE ?)")
        needle = f"%{query}%"
        params.extend([needle, needle])
    where = " AND ".join(clauses)
    count = db.get(f"SELECT COUNT(*) AS count FROM media_assets a WHERE {where}", params)
    total = int((count or {}).get("count") or 0)
    page_size = min(200, max(1, int(page_size)))
    page = max(1, int(page))
    offset = (page - 1) * page_size
    rows = db.query(
        f"""
        SELECT a.*, c.name AS collection_name, c.slug AS collection_slug
        FROM media_assets a
        LEFT JOIN media_collections c ON c.id = a.collection_id
        WHERE {where}
        ORDER BY a.display_name, a.id
        LIMIT ? OFFSET ?
        """,
        (*params, page_size, offset),
    )
    return {
        "assets": [_decode_asset(row) for row in rows],
        "page": page,
        "page_size": page_size,
        "total": total,
        "page_count": max(1, (total + page_size - 1) // page_size),
    }


def media_audit(db: Database) -> dict[str, Any]:
    rows = db.query(
        """
        SELECT a.*, c.name AS collection_name, c.slug AS collection_slug
        FROM media_assets a
        LEFT JOIN media_collections c ON c.id = a.collection_id
        WHERE a.state != 'ready' OR a.deleted_at IS NOT NULL
        ORDER BY a.updated_at DESC, a.id DESC
        """
    )
    decoded = [_decode_asset(row) for row in rows]
    return {
        "assets": decoded,
        "total": len(decoded),
        "by_state": {
            state: sum(1 for row in decoded if row.get("state") == state)
            for state in sorted({str(row.get("state") or "unknown") for row in decoded})
        },
    }


def add_relation(
    db: Database,
    parent_asset_id: int,
    child_asset_id: int,
    relation_type: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO media_asset_relations(
                parent_asset_id, child_asset_id, relation_type, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(parent_asset_id, child_asset_id, relation_type)
            DO UPDATE SET metadata_json = excluded.metadata_json
            """,
            (int(parent_asset_id), int(child_asset_id), str(relation_type), _json(metadata), utc_ts()),
        )


def bind_run_asset(
    db: Database,
    run_id: int,
    asset_id: int,
    role: str,
    *,
    video_name: str = "",
    track_label: str = "",
    model_name: str = "",
    checkpoint: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO run_media_assets(
                run_id, asset_id, role, video_name, track_label,
                model_name, checkpoint, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, asset_id, role, video_name, track_label)
            DO UPDATE SET model_name = excluded.model_name,
                          checkpoint = excluded.checkpoint,
                          metadata_json = excluded.metadata_json
            """,
            (
                int(run_id), int(asset_id), str(role), str(video_name), str(track_label),
                str(model_name), str(checkpoint), _json(metadata), utc_ts(),
            ),
        )


def run_asset_pair(
    db: Database,
    run_id: int,
    video_name: str,
    track_label: str = "",
) -> tuple[int | None, int | None]:
    reference = db.get(
        """
        SELECT asset_id FROM run_media_assets
        WHERE run_id = ? AND role = 'gt' AND video_name = ?
        ORDER BY CASE WHEN json_extract(metadata_json, '$.input') = 1 THEN 1 ELSE 0 END, asset_id DESC
        LIMIT 1
        """,
        (int(run_id), str(video_name)),
    )
    distorted = db.get(
        """
        SELECT asset_id FROM run_media_assets
        WHERE run_id = ? AND role = 'pred' AND video_name = ?
          AND (? = '' OR track_label = ?)
        ORDER BY CASE WHEN json_extract(metadata_json, '$.input') = 1 THEN 1 ELSE 0 END, asset_id DESC
        LIMIT 1
        """,
        (int(run_id), str(video_name), str(track_label), str(track_label)),
    )
    return (
        int(reference["asset_id"]) if reference is not None else None,
        int(distorted["asset_id"]) if distorted is not None else None,
    )


def bind_metric_result(
    db: Database,
    metric_result_id: int,
    reference_asset_id: int | None,
    distorted_asset_id: int | None,
    *,
    video_name: str = "",
    track_label: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO metric_asset_bindings(
                metric_result_id, reference_asset_id, distorted_asset_id,
                video_name, track_label, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(metric_result_id) DO UPDATE SET
                reference_asset_id = excluded.reference_asset_id,
                distorted_asset_id = excluded.distorted_asset_id,
                video_name = excluded.video_name,
                track_label = excluded.track_label,
                metadata_json = excluded.metadata_json
            """,
            (
                int(metric_result_id), reference_asset_id, distorted_asset_id,
                str(video_name), str(track_label), _json(metadata), utc_ts(),
            ),
        )


def run_asset_pair(
    db: Database,
    run_id: int,
    video_name: str,
    track_label: str = "",
) -> tuple[int | None, int | None]:
    reference = db.get(
        """
        SELECT asset_id FROM run_media_assets
        WHERE run_id = ? AND role = 'gt' AND video_name = ?
        ORDER BY CASE WHEN json_extract(metadata_json, '$.input') = 1 THEN 1 ELSE 0 END, asset_id DESC
        LIMIT 1
        """,
        (int(run_id), str(video_name)),
    )
    distorted = db.get(
        """
        SELECT asset_id FROM run_media_assets
        WHERE run_id = ? AND role = 'pred' AND video_name = ?
          AND (? = '' OR track_label = ?)
        ORDER BY CASE WHEN json_extract(metadata_json, '$.input') = 1 THEN 1 ELSE 0 END, asset_id DESC
        LIMIT 1
        """,
        (int(run_id), str(video_name), str(track_label), str(track_label)),
    )
    return (
        int(reference["asset_id"]) if reference is not None else None,
        int(distorted["asset_id"]) if distorted is not None else None,
    )


def bind_metric_result(
    db: Database,
    metric_result_id: int,
    reference_asset_id: int | None,
    distorted_asset_id: int | None,
    *,
    video_name: str = "",
    track_label: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO metric_asset_bindings(
                metric_result_id, reference_asset_id, distorted_asset_id,
                video_name, track_label, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(metric_result_id) DO UPDATE SET
                reference_asset_id = excluded.reference_asset_id,
                distorted_asset_id = excluded.distorted_asset_id,
                video_name = excluded.video_name,
                track_label = excluded.track_label,
                metadata_json = excluded.metadata_json
            """,
            (
                int(metric_result_id), reference_asset_id, distorted_asset_id,
                str(video_name), str(track_label), _json(metadata), utc_ts(),
            ),
        )


def _folder_collection_slug(db: Database, group_name: str) -> str:
    """Return a stable, collision-safe Collection slug for ``videos/<group>``.

    ``slugify`` intentionally normalizes punctuation, so two valid folder
    names such as ``"a b"`` and ``"a-b"`` share the same readable slug.  A
    folder Collection is part of a canonical GT Item's identity boundary; in
    that case reusing the first Collection would silently put distinct source
    folders in one GT picker group.  Retain the historical readable slug when
    it already belongs to this exact folder, otherwise add a deterministic
    hash of the unnormalized group name.
    """
    base = f"videos-{slugify(group_name)}"

    def is_this_folder(row: dict[str, Any]) -> bool:
        metadata = _loads(row.get("metadata_json"))
        if not isinstance(metadata, dict):
            return False
        return (
            str(metadata.get("source_kind") or "") == "folder"
            and str(metadata.get("video_group") or "") == str(group_name)
        )

    existing = db.get("SELECT * FROM media_collections WHERE slug = ?", (base,))
    if existing is None or is_this_folder(existing):
        return base

    digest = hashlib.sha256(str(group_name).encode("utf-8")).hexdigest()
    for length in (10, 16, 24, 64):
        candidate = f"{base}-{digest[:length]}"
        collision = db.get("SELECT * FROM media_collections WHERE slug = ?", (candidate,))
        if collision is None or is_this_folder(collision):
            return candidate
    raise RuntimeError(f"could not allocate a distinct Collection slug for videos/{group_name}")


def sync_folder_assets(db: Database, workspace: WorkspaceConfig) -> int:
    seen: set[str] = set()
    count = 0
    for group in list_video_groups(workspace, include_videos=True):
        group_name = str(group["name"])
        collection = ensure_collection(
            db,
            f"videos/{group_name}",
            _folder_collection_slug(db, group_name),
            {"source_kind": "folder", "video_group": group_name},
        )
        for video in group.get("videos") or []:
            path = (Path(str(group["path"])) / str(video["name"])).resolve()
            source_key = f"folder:{group_name}/{path.name}"
            seen.add(source_key)
            state = "ready" if path.is_file() else "unavailable"
            source_stat = path.stat() if path.is_file() else None
            upsert_asset(
                db,
                collection_id=int(collection["id"]),
                source_key=source_key,
                source_kind="folder",
                media_kind="video",
                role="gt",
                display_name=path.name,
                original_name=path.name,
                storage_path=path,
                state=state,
                content_sha256=_file_sha256(db, source_key, path) if path.is_file() else None,
                size_bytes=int(video.get("size_bytes") or (path.stat().st_size if path.exists() else 0)),
                frame_count=int(video.get("frame_count") or 0),
                width=int(video.get("width") or 0),
                height=int(video.get("height") or 0),
                fps=video.get("fps"),
                provenance={"video_group": group_name, "video": path.name},
                metadata={
                    "duration_seconds": video.get("duration_seconds"),
                    "source_mtime_ns": int(source_stat.st_mtime_ns) if source_stat else None,
                },
            )
            count += 1
    rows = db.query("SELECT id, source_key, storage_path FROM media_assets WHERE source_kind = 'folder'")
    with db.connection() as conn:
        for row in rows:
            if row["source_key"] not in seen or not Path(str(row["storage_path"])).exists():
                conn.execute(
                    "UPDATE media_assets SET state = 'unavailable', updated_at = ? WHERE id = ?",
                    (utc_ts(), int(row["id"])),
                )
    from vfieval.media_items import sync_canonical_gt_items

    sync_canonical_gt_items(db)
    return count


def folder_asset_id_map(db: Database, group_name: str) -> dict[str, int]:
    prefix = f"folder:{str(group_name)}/"
    rows = db.query(
        """
        SELECT id, source_key FROM media_assets
        WHERE source_kind = 'folder' AND state = 'ready' AND deleted_at IS NULL
          AND source_key LIKE ?
        """,
        (f"{prefix}%",),
    )
    return {
        str(row["source_key"])[len(prefix):]: int(row["id"])
        for row in rows
        if str(row["source_key"]).startswith(prefix)
    }


def sync_run_assets(
    db: Database,
    workspace: WorkspaceConfig,
    run_id: int,
    *,
    allow_running: bool = False,
) -> list[dict[str, Any]]:
    try:
        return _sync_run_assets(
            db,
            workspace,
            int(run_id),
            allow_running=allow_running,
        )
    except Exception:
        # ``upsert_asset`` and ``bind_run_asset`` intentionally remain small,
        # reusable transactions. If anything fails between them, invalidate
        # both bound assets and artifact-derived assets that have not yet been
        # bound so no partial publication remains catalog-visible.
        db.invalidate_run_media_assets(int(run_id))
        raise


def _run_video_artifacts(db: Database, run_id: int) -> dict[str, list[dict[str, Any]]]:
    return {
        kind: db.list_run_artifacts(int(run_id), kind=kind)
        for kind in CANONICAL_VIDEO_ARTIFACT_KINDS
    }


def _expected_video_artifact_identities(
    db: Database,
    run: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
) -> set[tuple[str, str, str | None]] | None:
    """Derive the canonical video publication contract from source samples.

    New decoded-video and Compare samples carry explicit semantic metadata.
    Legacy frame datasets do not, so ``None`` preserves their readable
    compatibility surface instead of guessing requirements from file names.
    """

    metadata = dict(run.get("metadata") or {})
    request = metadata.get("request") or {}
    if not isinstance(request, dict):
        request = {}
    canonical_required = str(
        metadata.get("artifact_contract") or request.get("artifact_contract") or ""
    ) == "canonical-v1"
    inference_job_ids = db.run_inference_job_ids(int(run["id"]))
    if not canonical_required:
        canonical_required = any(
            str((artifact.get("metadata") or {}).get("artifact_contract") or "") == "canonical-v1"
            for artifact in (artifacts or [])
        )
    if not canonical_required:
        canonical_required = any(
            str((db.get_job(int(job_id)).get("payload") or {}).get("artifact_contract") or "")
            == "canonical-v1"
            for job_id in inference_job_ids
        )
    if not canonical_required:
        return None
    profile = str(metadata.get("artifact_profile") or request.get("artifact_profile") or "evaluation")
    if profile == "benchmark":
        return set()
    samples = db.list_samples(int(run["dataset_id"]))
    selected_sample_ids: set[int] = set()
    has_explicit_selection = False
    for job_id in inference_job_ids:
        payload = db.get_job(int(job_id)).get("payload") or {}
        if payload.get("sample_ids") is None:
            continue
        has_explicit_selection = True
        selected_sample_ids.update(int(sample_id) for sample_id in payload.get("sample_ids") or [])
    if has_explicit_selection:
        samples = [sample for sample in samples if int(sample["id"]) in selected_sample_ids]
    video_samples = [
        sample
        for sample in samples
        if str((sample.get("metadata") or {}).get("source_type") or "") in {"video", "compare"}
        or bool((sample.get("metadata") or {}).get("compare_group"))
    ]
    if not video_samples:
        return None

    publish_pred_video = bool(
        metadata.get(
            "publish_compare_pred_video",
            request.get("publish_compare_pred_video", True),
        )
    )
    by_video: dict[str, list[dict[str, Any]]] = {}
    for sample in video_samples:
        sample_metadata = sample.get("metadata") or {}
        video_name = str(
            sample_metadata.get("video_name")
            or sample_metadata.get("compare_group")
            or "video"
        )
        by_video.setdefault(video_name, []).append(sample)

    expected: set[tuple[str, str, str | None]] = set()
    for video_name, group in by_video.items():
        is_compare = any(
            str((sample.get("metadata") or {}).get("source_type") or "") == "compare"
            or bool((sample.get("metadata") or {}).get("compare_group"))
            for sample in group
        )
        if is_compare:
            track_keys = {
                str((sample.get("metadata") or {}).get("compare_track_key"))
                for sample in group
                if (sample.get("metadata") or {}).get("compare_track_key") not in {None, ""}
            }
            identities = sorted(track_keys) if track_keys else [None]
            for track_key in identities:
                if publish_pred_video:
                    expected.add(("pred_video", video_name, track_key))
                expected.add(("diff_video", video_name, track_key))
            expected.add(("gt_video", video_name, None))
            continue

        expected.add(("pred_video", video_name, None))
        if any(str(sample.get("gt_path") or "").strip() for sample in group):
            expected.add(("gt_video", video_name, None))
            expected.add(("diff_video", video_name, None))
    return expected


def _require_expected_video_artifacts(
    artifacts: list[dict[str, Any]],
    expected: set[tuple[str, str, str | None]] | None,
) -> None:
    if expected is None:
        return
    actual = []
    for artifact in artifacts:
        metadata = artifact.get("metadata") or {}
        track_value = metadata.get("compare_track_key")
        actual.append(
            (
                str(artifact.get("kind") or ""),
                str(metadata.get("video_name") or ""),
                str(track_value) if track_value not in {None, ""} else None,
            )
        )
    if len(actual) != len(set(actual)) or set(actual) != expected:
        missing = sorted(expected - set(actual), key=str)
        unexpected = sorted(set(actual) - expected, key=str)
        raise ValueError(
            "canonical video artifact contract changed before publication: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _require_materialized_video_artifacts(artifacts: list[dict[str, Any]]) -> None:
    for artifact in artifacts:
        path = Path(str(artifact.get("path") or "")).resolve()
        try:
            valid = path.is_file() and path.stat().st_size > 0
        except OSError:
            valid = False
        if not valid:
            raise ValueError(
                f"canonical {artifact.get('kind') or 'video'} artifact "
                f"{int(artifact['id'])} is missing or empty"
            )


def _stage_existing_video_assets(db: Database, artifact_ids: list[int]) -> None:
    """Hide any prior publication before rebuilding all output bindings."""

    ids = sorted({int(artifact_id) for artifact_id in artifact_ids})
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    source_keys = [f"run_artifact:{artifact_id}" for artifact_id in ids]
    with db.connection() as conn:
        conn.execute(
            f"""
            UPDATE media_assets
            SET state = 'unavailable', updated_at = ?
            WHERE source_kind = 'run_artifact'
              AND source_key IN ({placeholders})
            """,
            (utc_ts(), *source_keys),
        )


def _activate_video_assets(
    db: Database,
    run_id: int,
    artifacts: list[dict[str, Any]],
    asset_ids: list[int],
    allowed_statuses: set[str],
) -> bool:
    """Validate the publication snapshot and make all staged assets visible.

    Holding ``BEGIN IMMEDIATE`` makes the final filesystem/row check and the
    Run-state fence linearize with cancellation and cleanup transactions.
    """

    expected_artifacts = {int(artifact["id"]): artifact for artifact in artifacts}
    staged_asset_ids = sorted({int(asset_id) for asset_id in asset_ids})
    if not staged_asset_ids:
        return True
    asset_placeholders = ",".join("?" for _ in staged_asset_ids)
    with db.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        run_row = conn.execute(
            "SELECT status, deleted_at, artifact_cleaned_at FROM runs WHERE id = ?",
            (int(run_id),),
        ).fetchone()
        if (
            run_row is None
            or run_row["deleted_at"] is not None
            or run_row["artifact_cleaned_at"] is not None
            or str(run_row["status"] or "") not in allowed_statuses
        ):
            return False

        rows = conn.execute(
            """
            SELECT DISTINCT a.id, a.kind, a.path, a.metadata_json
            FROM artifacts a
            WHERE a.kind IN ('pred_video', 'gt_video', 'diff_video')
              AND a.job_id IN (
                  SELECT job_id FROM run_jobs
                  WHERE run_id = ? AND role = 'inference'
                  UNION
                  SELECT inference_job_id FROM runs
                  WHERE id = ? AND inference_job_id IS NOT NULL
              )
            ORDER BY a.id
            """,
            (int(run_id), int(run_id)),
        ).fetchall()
        if {int(row["id"]) for row in rows} != set(expected_artifacts):
            raise ValueError("canonical video artifact set changed during media publication")
        for row in rows:
            expected = expected_artifacts[int(row["id"])]
            if (
                str(row["kind"] or "") != str(expected.get("kind") or "")
                or Path(str(row["path"] or "")).resolve()
                != Path(str(expected.get("path") or "")).resolve()
                or _loads(row["metadata_json"]) != dict(expected.get("metadata") or {})
            ):
                raise ValueError("canonical video artifact identity changed during media publication")
        _require_materialized_video_artifacts([dict(row) for row in rows])

        updated = conn.execute(
            f"""
            UPDATE media_assets
            SET state = 'ready', updated_at = ?
            WHERE source_kind = 'run_artifact'
              AND deleted_at IS NULL
              AND id IN ({asset_placeholders})
            """,
            (utc_ts(), *staged_asset_ids),
        )
        if int(updated.rowcount) != len(staged_asset_ids):
            raise ValueError("one or more staged Run video assets are unavailable")
    return True


def _sync_run_assets(
    db: Database,
    workspace: WorkspaceConfig,
    run_id: int,
    *,
    allow_running: bool = False,
) -> list[dict[str, Any]]:
    run = db.get_run(int(run_id))
    if run.get("deleted_at") is not None or run.get("artifact_cleaned_at") is not None:
        db.invalidate_run_media_assets(int(run_id))
        return []
    # Publication is a post-integrity phase. A late worker callback must never
    # catalog outputs for a Run that cancellation/failure already fenced.
    allowed_statuses = {"finalizing", "metric_queued", "metric_running", "completed"}
    if allow_running:
        allowed_statuses.add("running")
    if str(run.get("status") or "") not in allowed_statuses:
        return []
    metadata = run.get("metadata") or {}
    video_artifacts = _run_video_artifacts(db, int(run_id))
    all_video_artifacts = [
        artifact
        for kind in CANONICAL_VIDEO_ARTIFACT_KINDS
        for artifact in video_artifacts[kind]
    ]
    expected_video_identities = _expected_video_artifact_identities(
        db,
        run,
        all_video_artifacts,
    )
    _require_expected_video_artifacts(all_video_artifacts, expected_video_identities)
    if not all_video_artifacts:
        return []
    _require_materialized_video_artifacts(all_video_artifacts)
    artifact_groups = [
        (kind, role, video_artifacts[kind])
        for kind, role in (("pred_video", "pred"), ("gt_video", "gt"))
    ]
    if not any(artifacts for _kind, _role, artifacts in artifact_groups):
        raise ValueError("Run has canonical video artifacts but no publishable Pred or GT video")
    _stage_existing_video_assets(
        db,
        [int(artifact["id"]) for artifact in all_video_artifacts],
    )
    collection = ensure_collection(
        db,
        f"Run {run_id} · {run.get('name') or ''}",
        f"run-{int(run_id)}",
        {"source_kind": "run_artifact", "run_id": int(run_id)},
    )
    is_compare = str(metadata.get("run_type") or "") == "video_compare"
    reference_asset_id = int(metadata.get("reference_asset_id") or 0)
    input_pred_assets = {
        str(row.get("track_label") or ""): int(row["asset_id"])
        for row in db.query(
            """
            SELECT rma.asset_id, rma.track_label
            FROM run_media_assets rma
            WHERE rma.run_id = ? AND rma.role = 'pred'
              AND json_extract(rma.metadata_json, '$.input') = 1
            """,
            (int(run_id),),
        )
    }
    staged: list[dict[str, Any]] = []
    for kind, role, artifacts in artifact_groups:
        for artifact in artifacts:
            artifact_meta = artifact.get("metadata") or {}
            path = Path(str(artifact.get("path") or "")).resolve()
            path_stat = path.stat()
            video_name = str(artifact_meta.get("video_name") or path.stem)
            track_label = str(artifact_meta.get("compare_track_label") or "")
            display = f"{video_name} - {track_label or run.get('name') or f'Run {run_id}'} - {role.upper()}"
            asset_metadata = dict(artifact_meta)
            asset_metadata["source_mtime_ns"] = int(path_stat.st_mtime_ns)
            provenance = {
                "run_id": int(run_id),
                "artifact_id": int(artifact["id"]),
                "artifact_kind": kind,
                "video_name": video_name,
                "track_label": track_label,
                "model_name": metadata.get("model_file") or run.get("model_name"),
                "checkpoint": metadata.get("checkpoint"),
            }
            if is_compare:
                # Catalog entries remain useful for viewing a legacy Compare
                # result, but must carry an explicit non-reusable origin.
                provenance["video_compare_derived"] = True
                asset_metadata["video_compare_derived"] = True
            if is_compare and kind == "gt_video" and reference_asset_id:
                provenance.update({"aligned_gt": True, "source_asset_id": reference_asset_id})
                asset_metadata.update({"aligned_gt": True, "source_asset_id": reference_asset_id})
            asset = upsert_asset(
                db,
                collection_id=int(collection["id"]),
                source_key=f"run_artifact:{int(artifact['id'])}",
                source_kind="run_artifact",
                media_kind="video",
                role=role,
                display_name=display,
                original_name=path.name,
                storage_path=path,
                mime_type=artifact.get("mime_type"),
                state="unavailable",
                content_sha256=_file_sha256(db, f"run_artifact:{int(artifact['id'])}", path),
                size_bytes=path_stat.st_size,
                frame_count=int(artifact_meta.get("frames") or 0),
                width=int(artifact_meta.get("width") or 0),
                height=int(artifact_meta.get("height") or 0),
                fps=artifact_meta.get("fps"),
                provenance=provenance,
                metadata=asset_metadata,
            )
            bind_run_asset(
                db,
                int(run_id),
                int(asset["id"]),
                role,
                video_name=video_name,
                track_label=track_label,
                model_name=str(metadata.get("model_file") or run.get("model_name") or ""),
                checkpoint=str(metadata.get("checkpoint") or ""),
                metadata={"artifact_id": int(artifact["id"]), "artifact_kind": kind},
            )
            staged.append(
                {
                    "asset_id": int(asset["id"]),
                    "artifact": artifact,
                    "artifact_meta": artifact_meta,
                    "kind": kind,
                    "video_name": video_name,
                    "track_label": track_label,
                }
            )

    # Re-read the Run-scoped rows so a deletion or replacement after the
    # integrity check cannot publish a successful subset.
    current_video_artifacts = _run_video_artifacts(db, int(run_id))
    current_all_video_artifacts = [
        artifact
        for kind in CANONICAL_VIDEO_ARTIFACT_KINDS
        for artifact in current_video_artifacts[kind]
    ]
    _require_expected_video_artifacts(current_all_video_artifacts, expected_video_identities)
    if {int(row["id"]) for row in current_all_video_artifacts} != {
        int(row["id"]) for row in all_video_artifacts
    }:
        raise ValueError("canonical video artifact set changed during media publication")
    _require_materialized_video_artifacts(current_all_video_artifacts)
    if not _activate_video_assets(
        db,
        int(run_id),
        current_all_video_artifacts,
        [int(row["asset_id"]) for row in staged],
        allowed_statuses,
    ):
        with db.connection() as conn:
            asset_ids = [int(row["asset_id"]) for row in staged]
            placeholders = ",".join("?" for _ in asset_ids)
            conn.execute(
                f"UPDATE media_assets SET state = 'unavailable', updated_at = ? WHERE id IN ({placeholders})",
                (utc_ts(), *asset_ids),
            )
        return []

    result: list[dict[str, Any]] = []
    for staged_row in staged:
        asset = get_asset(db, int(staged_row["asset_id"]))
        artifact = staged_row["artifact"]
        artifact_meta = staged_row["artifact_meta"]
        kind = str(staged_row["kind"])
        video_name = str(staged_row["video_name"])
        track_label = str(staged_row["track_label"])
        if asset.get("state") != "ready":
            raise ValueError(f"published media asset {int(asset['id'])} is not ready")
        if not is_compare and kind == "pred_video":
            # Only Runs that were explicitly bound to a canonical Item at
            # creation time can publish a reusable prediction. This is the
            # upgrade boundary: catalog sync never guesses bindings for
            # legacy Run outputs from names, labels, or file hashes.
            from vfieval.media_items import find_run_source_item, register_model_prediction

            source_item = find_run_source_item(
                db,
                int(run_id),
                video_name,
                source_video_group=str(artifact_meta.get("source_video_group") or ""),
                source_video_file=str(artifact_meta.get("source_video_file") or ""),
            )
            if source_item is not None:
                request_metadata = metadata.get("request") or {}
                if not isinstance(request_metadata, dict):
                    request_metadata = {}
                temporal_mapping = {
                    key: artifact_meta[key]
                    for key in (
                        "source_frame_indices",
                        "frame_step",
                        "fps",
                        "timestamps",
                        "source_timestamps",
                    )
                    if artifact_meta.get(key) is not None
                }
                spatial_origin = {
                    "width": int(artifact_meta.get("width") or 0),
                    "height": int(artifact_meta.get("height") or 0),
                    "source_width": artifact_meta.get("source_width"),
                    "source_height": artifact_meta.get("source_height"),
                    "resolution_mode": request_metadata.get("resolution_mode"),
                    "artifact_contract": artifact_meta.get("artifact_contract"),
                }
                register_model_prediction(
                    db,
                    int(run_id),
                    int(source_item["id"]),
                    int(asset["id"]),
                    temporal_mapping=temporal_mapping,
                    spatial_origin=spatial_origin,
                    metadata={
                        "artifact_id": int(artifact["id"]),
                        "artifact_kind": kind,
                        "video_name": video_name,
                    },
                )
        if is_compare and reference_asset_id:
            if kind == "gt_video":
                add_relation(db, reference_asset_id, int(asset["id"]), "generated_from")
                add_relation(db, reference_asset_id, int(asset["id"]), "aligned_gt_of")
            elif kind == "pred_video":
                add_relation(db, reference_asset_id, int(asset["id"]), "pred_of")
                source_pred_id = input_pred_assets.get(track_label)
                if source_pred_id:
                    add_relation(db, source_pred_id, int(asset["id"]), "generated_from")
        result.append(asset)
    inference_job_ids = db.run_inference_job_ids(int(run_id))
    unbound_metrics: list[dict[str, Any]] = []
    if inference_job_ids:
        placeholders = ",".join("?" for _ in inference_job_ids)
        unbound_metrics = db.query(
            f"""
            SELECT mr.* FROM metric_results mr
            WHERE mr.inference_job_id IN ({placeholders})
              AND NOT EXISTS (
                  SELECT 1 FROM metric_asset_bindings mab
                  WHERE mab.metric_result_id = mr.id
              )
            ORDER BY mr.id
            """,
            inference_job_ids,
        )
        for metric in unbound_metrics:
            metric["details"] = _loads(metric.pop("details_json", None))
    for metric in unbound_metrics:
        details = metric.get("details") or {}
        sample = db.get_sample(int(metric["sample_id"])) if metric.get("sample_id") is not None else None
        sample_metadata = (sample or {}).get("metadata") or {}
        video_name = str(
            details.get("video_name")
            or sample_metadata.get("video_name")
            or sample_metadata.get("video_file")
            or ""
        )
        track_label = str(
            details.get("compare_track_label")
            or sample_metadata.get("compare_track_label")
            or ""
        )
        reference_asset_id, distorted_asset_id = run_asset_pair(db, int(run_id), video_name, track_label)
        if reference_asset_id is None and distorted_asset_id is None:
            continue
        bind_metric_result(
            db,
            int(metric["id"]),
            reference_asset_id,
            distorted_asset_id,
            video_name=video_name,
            track_label=track_label,
            metadata={"backfilled": True},
        )
    final_run = db.get_run(int(run_id))
    if (
        final_run.get("deleted_at") is not None
        or final_run.get("artifact_cleaned_at") is not None
        or str(final_run.get("status") or "") not in allowed_statuses
    ):
        db.invalidate_run_media_assets(int(run_id))
        return []
    return result


def sync_catalog(db: Database, workspace: WorkspaceConfig, include_runs: bool = True) -> dict[str, int]:
    folders = sync_folder_assets(db, workspace)
    from vfieval.media_items import sync_canonical_gt_items

    item_sync = sync_canonical_gt_items(db)
    run_assets = 0
    if include_runs:
        for run in db.list_runs(limit=10000, include_deleted=False):
            if run.get("artifact_cleaned_at") is not None:
                db.invalidate_run_media_assets(int(run["id"]))
                continue
            try:
                run_assets += len(sync_run_assets(db, workspace, int(run["id"])))
            except (KeyError, ValueError):
                continue
    return {
        "folder_assets": folders,
        "run_assets": run_assets,
        "media_items": int(item_sync["total"]),
    }


def resolve_asset_path(
    db: Database,
    workspace: WorkspaceConfig,
    asset_id: int,
    *,
    role: str | None = None,
) -> tuple[dict[str, Any], Path]:
    asset = get_asset(db, int(asset_id))
    if asset["state"] != "ready":
        raise ValueError(f"media asset {asset_id} is not ready: {asset['state']}")
    if role == "reference" and asset["role"] not in {"source", "gt"}:
        raise ValueError("compare reference media asset must have role gt or source")
    if role == "distorted" and asset["role"] != "pred":
        raise ValueError("compare distorted media asset must have role pred")
    path = Path(str(asset["storage_path"])).resolve()
    if not path.exists():
        with db.connection() as conn:
            conn.execute(
                "UPDATE media_assets SET state = 'unavailable', updated_at = ? WHERE id = ?",
                (utc_ts(), int(asset_id)),
            )
        raise FileNotFoundError(f"media asset content is unavailable: {asset_id}")
    allowed = {
        "folder": videos_dir(workspace).resolve(),
        "upload": workspace.media_dir.resolve(),
        "run_artifact": workspace.runs_dir.resolve(),
        "evaluation_package": workspace.evaluations_dir.resolve(),
    }[str(asset["source_kind"])]
    try:
        path.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("media asset resolved outside its managed storage root") from exc
    return asset, path


def soft_delete_asset(db: Database, workspace: WorkspaceConfig, asset_id: int) -> dict[str, Any]:
    asset = get_asset(db, int(asset_id), include_deleted=True)
    if asset.get("deleted_at") is not None:
        return asset
    # Frozen evaluation packages are the only playback source for published
    # blind Campaigns after a source Run is cleaned.  They are immutable by
    # contract: even a catalog-level soft delete must not mark them unavailable
    # or make participant playback/history disappear.
    if asset["source_kind"] == "evaluation_package":
        raise ValueError("frozen evaluation package media is immutable and cannot be deleted")
    references = db.get(
        """
        SELECT
          (SELECT COUNT(*) FROM evaluation_candidates
           WHERE reference_asset_id = ? OR asset_id = ?) AS candidates,
          (SELECT COUNT(*) FROM evaluation_tasks WHERE reference_asset_id = ?) AS tasks,
          (SELECT COUNT(*) FROM evaluation_votes WHERE preferred_asset_id = ?) AS votes
        """,
        (int(asset_id), int(asset_id), int(asset_id), int(asset_id)),
    ) or {}
    protected = any(int(references.get(key) or 0) > 0 for key in ("candidates", "tasks", "votes"))
    content_removed = False
    if asset["source_kind"] == "upload" and not protected:
        path = Path(str(asset["storage_path"])).resolve()
        media_root = workspace.media_dir.resolve()
        try:
            relative = path.relative_to(media_root)
        except ValueError as exc:
            raise ValueError("upload asset path is outside managed media storage") from exc
        if len(relative.parts) < 2:
            raise ValueError("refusing to remove an upload outside an asset directory")
        asset_root = path if path.is_dir() else path.parent
        try:
            asset_root.relative_to(media_root)
        except ValueError as exc:
            raise ValueError("upload asset directory is outside managed media storage") from exc
        if asset_root.exists():
            shutil.rmtree(asset_root)
            content_removed = True
    now = utc_ts()
    with db.connection() as conn:
        conn.execute(
            """
            UPDATE media_assets
            SET state = ?, deleted_at = ?, updated_at = ?
            WHERE id = ?
            """,
            ("unavailable" if protected else "deleted", now, now, int(asset_id)),
        )
    if asset["source_kind"] in {"folder", "upload"} and asset["role"] == "gt":
        from vfieval.media_items import ensure_canonical_gt_item

        ensure_canonical_gt_item(db, int(asset_id))
    result = get_asset(db, int(asset_id), include_deleted=True)
    result["protected_by_evaluation"] = protected
    result["content_removed"] = content_removed
    return result


def source_assets_to_video_payload(db: Database, workspace: WorkspaceConfig, payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("source_assets")
    if not raw:
        return payload
    if not isinstance(raw, list):
        raise ValueError("source_assets must be a list")
    entries: list[tuple[str, str]] = []
    for descriptor in raw:
        if not isinstance(descriptor, dict) or descriptor.get("asset_id") in {None, ""}:
            raise ValueError("each source_assets entry requires asset_id")
        asset, path = resolve_asset_path(db, workspace, int(descriptor["asset_id"]))
        if asset["source_kind"] != "folder" or path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError("model inference source_assets currently require videos/ folder assets")
        provenance = asset.get("provenance") or {}
        group = str(provenance.get("video_group") or "")
        video = str(provenance.get("video") or path.name)
        if not group:
            raise ValueError(f"folder media asset {asset['id']} is missing video_group provenance")
        entries.append((group, video))
    groups = list(dict.fromkeys(group for group, _video in entries))
    converted = dict(payload)
    converted["video_groups"] = groups
    converted["video_group"] = groups[0]
    converted["selected_videos"] = [video if len(groups) == 1 else f"{group}/{video}" for group, video in entries]
    return converted


def inspect_uploaded_video(workspace: WorkspaceConfig, path: Path) -> dict[str, Any]:
    info = inspect_video(path, workspace, exact=True)
    if not info.get("decodable"):
        raise ValueError(str(info.get("error") or "uploaded video is not decodable"))
    return info
