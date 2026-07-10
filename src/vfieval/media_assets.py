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
SOURCE_KINDS = {"folder", "upload", "run_artifact"}
MEDIA_STATES = {"ready", "unavailable", "deleted", "invalid"}


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


def list_collections(db: Database) -> list[dict[str, Any]]:
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
    return get_asset(db, asset_id, include_deleted=True)


def _decode_asset(row: dict[str, Any]) -> dict[str, Any]:
    row["provenance"] = _loads(row.pop("provenance_json", None))
    row["metadata"] = _loads(row.pop("metadata_json", None))
    row["size_bytes"] = int(row.get("size_bytes") or 0)
    row["frame_count"] = int(row.get("frame_count") or 0)
    row["width"] = int(row.get("width") or 0)
    row["height"] = int(row.get("height") or 0)
    return row


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


def sync_folder_assets(db: Database, workspace: WorkspaceConfig) -> int:
    seen: set[str] = set()
    count = 0
    for group in list_video_groups(workspace, include_videos=True):
        group_name = str(group["name"])
        collection = ensure_collection(
            db,
            f"videos/{group_name}",
            f"videos-{slugify(group_name)}",
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


def sync_run_assets(db: Database, workspace: WorkspaceConfig, run_id: int) -> list[dict[str, Any]]:
    run = db.get_run(int(run_id))
    metadata = run.get("metadata") or {}
    artifact_groups = [
        (kind, role, db.list_run_artifacts(int(run_id), kind=kind))
        for kind, role in (("pred_video", "pred"), ("gt_video", "gt"))
    ]
    if not any(artifacts for _kind, _role, artifacts in artifact_groups):
        return []
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
            JOIN media_assets ma ON ma.id = rma.asset_id
            WHERE rma.run_id = ? AND rma.role = 'pred' AND ma.source_kind != 'run_artifact'
            """,
            (int(run_id),),
        )
    }
    result: list[dict[str, Any]] = []
    for kind, role, artifacts in artifact_groups:
        for artifact in artifacts:
            artifact_meta = artifact.get("metadata") or {}
            path = Path(str(artifact.get("path") or "")).resolve()
            video_name = str(artifact_meta.get("video_name") or path.stem)
            track_label = str(artifact_meta.get("compare_track_label") or "")
            display = f"{video_name} - {track_label or run.get('name') or f'Run {run_id}'} - {role.upper()}"
            state = "ready" if path.is_file() and run.get("artifact_cleaned_at") is None else "unavailable"
            asset_metadata = dict(artifact_meta)
            if path.is_file():
                asset_metadata["source_mtime_ns"] = int(path.stat().st_mtime_ns)
            provenance = {
                "run_id": int(run_id),
                "artifact_id": int(artifact["id"]),
                "artifact_kind": kind,
                "video_name": video_name,
                "track_label": track_label,
                "model_name": metadata.get("model_file") or run.get("model_name"),
                "checkpoint": metadata.get("checkpoint"),
            }
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
                state=state,
                content_sha256=_file_sha256(db, f"run_artifact:{int(artifact['id'])}", path) if path.is_file() else None,
                size_bytes=path.stat().st_size if path.is_file() else 0,
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
    return result


def sync_catalog(db: Database, workspace: WorkspaceConfig, include_runs: bool = True) -> dict[str, int]:
    folders = sync_folder_assets(db, workspace)
    run_assets = 0
    if include_runs:
        for run in db.list_runs(limit=10000, include_deleted=True):
            try:
                run_assets += len(sync_run_assets(db, workspace, int(run["id"])))
            except (KeyError, ValueError):
                continue
    return {"folder_assets": folders, "run_assets": run_assets}


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
