from __future__ import annotations

import hashlib
import json
import secrets
import threading
import weakref
from pathlib import Path
from typing import Any

from vfieval.config import WorkspaceConfig
from vfieval.db import Database, utc_ts
from vfieval.file_inputs import VIDEO_SUFFIXES, videos_dir


VIDEO_SELECTION_SCHEMA_VERSION = "video-selection-v1"
VIDEO_SELECTION_MIGRATION_VERSION = "2026-07-video-selection-snapshots-v1"
VIDEO_SELECTION_TOKEN_TTL_SECONDS = 24 * 60 * 60
_TOKEN_MIN_LENGTH = 32
_TOKEN_MAX_LENGTH = 128
_MAX_QUERY_LENGTH = 500
_MAX_EXPLICIT_NAMES = 200
_SCHEMA_READY: weakref.WeakSet[Database] = weakref.WeakSet()
_SCHEMA_LOCK = threading.Lock()


class VideoSelectionTokenError(ValueError):
    """A persisted folder-video selection can no longer be used safely."""


class VideoSelectionTokenExpired(VideoSelectionTokenError):
    """A folder-video selection exists, but its bounded lifetime elapsed."""


def ensure_video_selection_schema(db: Database) -> None:
    """Install the independent, versioned folder-video selection store."""

    if db in _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if db in _SCHEMA_READY:
            return
        with db.connection() as conn:
            conn.executescript(
                """
            CREATE TABLE IF NOT EXISTS video_selection_snapshots (
                token_hash TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                groups_json TEXT NOT NULL,
                item_count INTEGER NOT NULL CHECK(item_count >= 0),
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS video_selection_snapshot_items (
                token_hash TEXT NOT NULL
                    REFERENCES video_selection_snapshots(token_hash) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                asset_id INTEGER NOT NULL,
                video_group TEXT NOT NULL,
                video_name TEXT NOT NULL,
                source_key TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                content_sha256 TEXT,
                size_bytes INTEGER NOT NULL,
                source_mtime_ns INTEGER NOT NULL,
                PRIMARY KEY(token_hash, ordinal),
                UNIQUE(token_hash, video_group, video_name)
            );

            CREATE INDEX IF NOT EXISTS idx_video_selection_snapshots_expiry
            ON video_selection_snapshots(expires_at);

            CREATE INDEX IF NOT EXISTS idx_video_selection_items_lookup
            ON video_selection_snapshot_items(token_hash, video_group, video_name);
            """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO schema_migrations(version, applied_at)
                VALUES (?, ?)
                """,
                (VIDEO_SELECTION_MIGRATION_VERSION, utc_ts()),
            )
        _SCHEMA_READY.add(db)


def _token_hash(token: str) -> str:
    normalized = str(token or "").strip()
    if not (_TOKEN_MIN_LENGTH <= len(normalized) <= _TOKEN_MAX_LENGTH):
        raise VideoSelectionTokenError("video_selection_token is invalid")
    if any(not (character.isalnum() or character in {"-", "_"}) for character in normalized):
        raise VideoSelectionTokenError("video_selection_token is invalid")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_groups(raw_groups: Any) -> list[str]:
    if isinstance(raw_groups, str):
        raw_groups = [raw_groups]
    if not isinstance(raw_groups, list):
        raise ValueError("video_groups must be a non-empty list")
    groups: list[str] = []
    for raw in raw_groups:
        group = str(raw or "").strip()
        if not group or Path(group).name != group:
            raise ValueError("video group must be one folder name under videos/")
        if group not in groups:
            groups.append(group)
    if not groups:
        raise ValueError("at least one video group is required")
    return groups


def _normalize_query(query: Any) -> str:
    normalized = str(query or "").strip()
    if len(normalized) > _MAX_QUERY_LENGTH:
        raise ValueError(f"q must be at most {_MAX_QUERY_LENGTH} characters")
    return normalized


def _folder_asset_rows(
    db: Database,
    workspace: WorkspaceConfig,
    groups: list[str],
    *,
    query: str = "",
    explicit_names: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    root = videos_dir(workspace).resolve()
    needle = _normalize_query(query)
    rows: list[dict[str, Any]] = []
    for group in groups:
        group_rows = db.query(
            """
            SELECT
                a.id AS asset_id,
                a.display_name AS video_name,
                a.source_key,
                a.storage_path,
                a.content_sha256,
                a.size_bytes,
                a.metadata_json
            FROM media_assets a
            JOIN media_collections c ON c.id = a.collection_id
            WHERE a.source_kind = 'folder'
              AND a.media_kind = 'video'
              AND a.role = 'gt'
              AND a.state = 'ready'
              AND a.deleted_at IS NULL
              AND json_extract(c.metadata_json, '$.source_kind') = 'folder'
              AND json_extract(c.metadata_json, '$.video_group') = ?
              AND (? = '' OR lower(a.display_name) LIKE '%' || lower(?) || '%')
            ORDER BY lower(a.display_name), a.id
            """,
            (group, needle, needle),
        )
        requested = explicit_names.get(group) if explicit_names is not None else None
        seen: set[str] = set()
        for row in group_rows:
            name = str(row.get("video_name") or "")
            if requested is not None and name not in requested:
                continue
            entry = _current_entry(root, group, row)
            rows.append(entry)
            seen.add(name)
        if requested is not None:
            missing = sorted(requested - seen)
            if missing:
                raise VideoSelectionTokenError(
                    f"selected videos are no longer available in videos/{group}: "
                    + ", ".join(missing[:5])
                )
    return rows


def _current_entry(root: Path, group: str, row: dict[str, Any]) -> dict[str, Any]:
    name = str(row.get("video_name") or "").strip()
    if (
        not name
        or Path(name).name != name
        or Path(name).suffix.lower() not in VIDEO_SUFFIXES
    ):
        raise VideoSelectionTokenError(
            f"catalog contains an invalid video name for videos/{group}"
        )
    expected = (root / group / name).resolve()
    try:
        expected.relative_to(root)
    except ValueError as exc:
        raise VideoSelectionTokenError(
            f"catalog video resolved outside videos/: {group}/{name}"
        ) from exc
    catalog_path = Path(str(row.get("storage_path") or "")).resolve()
    if catalog_path != expected:
        raise VideoSelectionTokenError(
            f"catalog path changed for videos/{group}/{name}; refresh the video catalog"
        )
    if not expected.is_file():
        raise VideoSelectionTokenError(
            f"selected video is no longer available: {group}/{name}"
        )
    stat_result = expected.stat()
    metadata = json.loads(str(row.get("metadata_json") or "{}"))
    catalog_size = int(row.get("size_bytes") or 0)
    catalog_mtime = int(metadata.get("source_mtime_ns") or 0)
    if catalog_size != int(stat_result.st_size) or (
        catalog_mtime > 0 and catalog_mtime != int(stat_result.st_mtime_ns)
    ):
        raise VideoSelectionTokenError(
            f"video content changed after the last catalog sync: {group}/{name}; "
            "refresh the video catalog"
        )
    return {
        "asset_id": int(row["asset_id"]),
        "video_group": group,
        "video_name": name,
        "source_key": str(row.get("source_key") or ""),
        "storage_path": str(expected),
        "content_sha256": (
            str(row["content_sha256"]) if row.get("content_sha256") else None
        ),
        "size_bytes": int(stat_result.st_size),
        "source_mtime_ns": int(stat_result.st_mtime_ns),
    }


def _write_snapshot(
    db: Database,
    *,
    groups: list[str],
    entries: list[dict[str, Any]],
    ttl_seconds: float,
) -> dict[str, Any]:
    ttl = min(7 * 24 * 60 * 60, max(60.0, float(ttl_seconds)))
    token = secrets.token_urlsafe(32)
    digest = _token_hash(token)
    created_at = utc_ts()
    expires_at = created_at + ttl
    group_order = {name: index for index, name in enumerate(groups)}
    ordered = sorted(
        entries,
        key=lambda row: (
            group_order[str(row["video_group"])],
            str(row["video_name"]).lower(),
            int(row["asset_id"]),
        ),
    )
    with db.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM video_selection_snapshots WHERE expires_at <= ?",
            (created_at,),
        )
        conn.execute(
            """
            INSERT INTO video_selection_snapshots(
                token_hash, schema_version, groups_json, item_count,
                created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                digest,
                VIDEO_SELECTION_SCHEMA_VERSION,
                json.dumps(groups, ensure_ascii=False, separators=(",", ":")),
                len(ordered),
                created_at,
                expires_at,
            ),
        )
        conn.executemany(
            """
            INSERT INTO video_selection_snapshot_items(
                token_hash, ordinal, asset_id, video_group, video_name,
                source_key, storage_path, content_sha256, size_bytes,
                source_mtime_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    digest,
                    ordinal,
                    int(entry["asset_id"]),
                    str(entry["video_group"]),
                    str(entry["video_name"]),
                    str(entry["source_key"]),
                    str(entry["storage_path"]),
                    entry.get("content_sha256"),
                    int(entry["size_bytes"]),
                    int(entry["source_mtime_ns"]),
                )
                for ordinal, entry in enumerate(ordered)
            ),
        )
    return _public_snapshot(
        token=token,
        groups=groups,
        entries=ordered,
        created_at=created_at,
        expires_at=expires_at,
    )


def _public_snapshot(
    *,
    token: str,
    groups: list[str],
    entries: list[dict[str, Any]],
    created_at: float,
    expires_at: float,
) -> dict[str, Any]:
    group_counts = {group: 0 for group in groups}
    for entry in entries:
        group_counts[str(entry["video_group"])] += 1
    return {
        "video_selection_token": token,
        "schema_version": VIDEO_SELECTION_SCHEMA_VERSION,
        "video_groups": list(groups),
        "total": len(entries),
        "group_counts": group_counts,
        "created_at": created_at,
        "expires_at": expires_at,
    }


def _snapshot_row(db: Database, token: str) -> tuple[str, dict[str, Any]]:
    ensure_video_selection_schema(db)
    digest = _token_hash(token)
    row = db.get(
        """
        SELECT token_hash, schema_version, groups_json, item_count,
               created_at, expires_at
        FROM video_selection_snapshots
        WHERE token_hash = ?
        """,
        (digest,),
    )
    if row is None:
        raise VideoSelectionTokenError(
            "video_selection_token was not found; select the video groups again"
        )
    if str(row.get("schema_version") or "") != VIDEO_SELECTION_SCHEMA_VERSION:
        raise VideoSelectionTokenError(
            "video_selection_token uses an unsupported schema version"
        )
    if float(row["expires_at"]) <= utc_ts():
        raise VideoSelectionTokenExpired(
            "video_selection_token expired; select the video groups again"
        )
    return digest, row


def _stored_entries(
    db: Database,
    digest: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    paging = ""
    params: list[Any] = [digest]
    if limit is not None:
        paging = " LIMIT ? OFFSET ?"
        params.extend([int(limit), max(0, int(offset))])
    return db.query(
        f"""
        SELECT
            vsi.*,
            a.id AS current_asset_id,
            a.source_key AS current_source_key,
            a.storage_path AS current_storage_path,
            a.content_sha256 AS current_content_sha256,
            a.size_bytes AS current_size_bytes,
            a.metadata_json AS current_metadata_json,
            a.state AS current_state,
            a.deleted_at AS current_deleted_at,
            a.source_kind AS current_source_kind,
            a.media_kind AS current_media_kind,
            a.role AS current_role,
            json_extract(c.metadata_json, '$.source_kind') AS collection_source_kind,
            json_extract(c.metadata_json, '$.video_group') AS current_video_group
        FROM video_selection_snapshot_items vsi
        LEFT JOIN media_assets a ON a.id = vsi.asset_id
        LEFT JOIN media_collections c ON c.id = a.collection_id
        WHERE vsi.token_hash = ?
        ORDER BY vsi.ordinal
        {paging}
        """,
        tuple(params),
    )


def _snapshot_items(db: Database, digest: str) -> list[dict[str, Any]]:
    return db.query(
        """
        SELECT asset_id, video_group, video_name, source_key, storage_path,
               content_sha256, size_bytes, source_mtime_ns
        FROM video_selection_snapshot_items
        WHERE token_hash = ?
        ORDER BY ordinal
        """,
        (digest,),
    )


def _validate_stored_entry(
    workspace: WorkspaceConfig,
    row: dict[str, Any],
) -> dict[str, Any]:
    group = str(row["video_group"])
    name = str(row["video_name"])
    current_sha = row.get("current_content_sha256")
    if (
        row.get("current_asset_id") is None
        or str(row.get("current_state") or "") != "ready"
        or row.get("current_deleted_at") is not None
        or str(row.get("current_source_kind") or "") != "folder"
        or str(row.get("current_media_kind") or "") != "video"
        or str(row.get("current_role") or "") != "gt"
        or str(row.get("collection_source_kind") or "") != "folder"
        or str(row.get("current_video_group") or "") != group
        or str(row.get("current_source_key") or "") != str(row["source_key"])
        or str(row.get("current_storage_path") or "") != str(row["storage_path"])
        or (str(current_sha) if current_sha else None)
        != (str(row["content_sha256"]) if row.get("content_sha256") else None)
        or int(row.get("current_size_bytes") or 0) != int(row["size_bytes"])
    ):
        raise VideoSelectionTokenError(
            f"video selection is stale: {group}/{name}; refresh the video catalog"
        )
    root = videos_dir(workspace).resolve()
    expected = (root / group / name).resolve()
    try:
        expected.relative_to(root)
    except ValueError as exc:
        raise VideoSelectionTokenError(
            f"video selection resolved outside videos/: {group}/{name}"
        ) from exc
    if (
        Path(str(row["storage_path"])).resolve() != expected
        or not expected.is_file()
    ):
        raise VideoSelectionTokenError(
            f"selected video is no longer available: {group}/{name}"
        )
    stat_result = expected.stat()
    if (
        int(stat_result.st_size) != int(row["size_bytes"])
        or int(stat_result.st_mtime_ns) != int(row["source_mtime_ns"])
    ):
        raise VideoSelectionTokenError(
            f"selected video content changed: {group}/{name}; "
            "refresh the video catalog and select it again"
        )
    return {
        "asset_id": int(row["asset_id"]),
        "video_group": group,
        "video_name": name,
        "source_key": str(row["source_key"]),
        "storage_path": str(row["storage_path"]),
        "content_sha256": (
            str(row["content_sha256"]) if row.get("content_sha256") else None
        ),
        "size_bytes": int(row["size_bytes"]),
        "source_mtime_ns": int(row["source_mtime_ns"]),
    }


def resolve_video_selection_snapshot(
    db: Database,
    workspace: WorkspaceConfig,
    token: str,
    *,
    require_non_empty: bool = False,
) -> dict[str, Any]:
    """Resolve a token to exact names and reject catalog/path/content drift."""

    digest, snapshot = _snapshot_row(db, token)
    groups = _normalize_groups(json.loads(str(snapshot["groups_json"])))
    stored = _stored_entries(db, digest)
    if len(stored) != int(snapshot["item_count"]):
        raise VideoSelectionTokenError(
            "video_selection_token is incomplete; select the video groups again"
        )
    entries = [_validate_stored_entry(workspace, row) for row in stored]
    if require_non_empty and not entries:
        raise VideoSelectionTokenError("video_selection_token contains no videos")
    multi_group = len(groups) > 1
    selected_videos = [
        (
            f"{entry['video_group']}/{entry['video_name']}"
            if multi_group
            else str(entry["video_name"])
        )
        for entry in entries
    ]
    return {
        "token_hash": digest,
        "schema_version": VIDEO_SELECTION_SCHEMA_VERSION,
        "video_groups": groups,
        "total": len(entries),
        "entries": entries,
        "selected_videos": selected_videos,
        "created_at": float(snapshot["created_at"]),
        "expires_at": float(snapshot["expires_at"]),
    }


def create_video_selection_snapshot(
    db: Database,
    workspace: WorkspaceConfig,
    *,
    video_groups: Any = None,
    query: Any = "",
    base_selection_token: str = "",
    operation: str = "",
    video_group: str = "",
    video_names: Any = None,
    ttl_seconds: float = VIDEO_SELECTION_TOKEN_TTL_SECONDS,
) -> dict[str, Any]:
    """Create or immutably mutate a folder-video selection snapshot."""

    ensure_video_selection_schema(db)
    base_token = str(base_selection_token or "").strip()
    normalized_operation = str(operation or "").strip().lower()
    if not base_token:
        if normalized_operation:
            raise ValueError("operation requires base_selection_token")
        groups = _normalize_groups(video_groups)
        entries = _folder_asset_rows(
            db,
            workspace,
            groups,
            query=_normalize_query(query),
        )
        return _write_snapshot(
            db,
            groups=groups,
            entries=entries,
            ttl_seconds=ttl_seconds,
        )

    if normalized_operation not in {
        "add",
        "remove",
        "toggle",
        "add_filtered",
        "remove_filtered",
        "toggle_filtered",
    }:
        raise ValueError("unsupported video selection operation")
    base_digest, base_snapshot = _snapshot_row(db, base_token)
    groups = _normalize_groups(json.loads(str(base_snapshot["groups_json"])))
    group = str(video_group or "").strip()
    if group not in groups:
        raise ValueError("video_group is not part of the selection snapshot")

    selected = {
        (str(entry["video_group"]), str(entry["video_name"])): dict(entry)
        for entry in _snapshot_items(db, base_digest)
    }
    if len(selected) != int(base_snapshot["item_count"]):
        raise VideoSelectionTokenError(
            "video_selection_token is incomplete; select the video groups again"
        )
    if normalized_operation.endswith("_filtered"):
        candidates = _folder_asset_rows(
            db,
            workspace,
            [group],
            query=_normalize_query(query),
        )
    else:
        if isinstance(video_names, str):
            names = [video_names]
        elif isinstance(video_names, list):
            names = [str(value) for value in video_names]
        else:
            raise ValueError("video_names must be a non-empty list")
        names = list(dict.fromkeys(name.strip() for name in names if name.strip()))
        if not names or len(names) > _MAX_EXPLICIT_NAMES:
            raise ValueError(
                f"video_names must contain 1 to {_MAX_EXPLICIT_NAMES} names"
            )
        for name in names:
            if Path(name).name != name:
                raise ValueError(f"video selection accepts file names only: {name}")
        candidates = _folder_asset_rows(
            db,
            workspace,
            [group],
            explicit_names={group: set(names)},
        )

    action = normalized_operation.removesuffix("_filtered")
    for entry in candidates:
        key = (str(entry["video_group"]), str(entry["video_name"]))
        if action == "add":
            selected[key] = entry
        elif action == "remove":
            selected.pop(key, None)
        elif key in selected:
            selected.pop(key)
        else:
            selected[key] = entry
    return _write_snapshot(
        db,
        groups=groups,
        entries=list(selected.values()),
        ttl_seconds=ttl_seconds,
    )


def video_selection_snapshot_page(
    db: Database,
    workspace: WorkspaceConfig,
    token: str,
    *,
    page: int = 1,
    page_size: int = 100,
) -> dict[str, Any]:
    digest, snapshot = _snapshot_row(db, token)
    groups = _normalize_groups(json.loads(str(snapshot["groups_json"])))
    normalized_page = max(1, int(page))
    normalized_page_size = min(200, max(1, int(page_size)))
    start = (normalized_page - 1) * normalized_page_size
    entries = [
        _validate_stored_entry(workspace, row)
        for row in _stored_entries(
            db,
            digest,
            limit=normalized_page_size,
            offset=start,
        )
    ]
    group_counts = {group: 0 for group in groups}
    for row in db.query(
        """
        SELECT video_group, COUNT(*) AS count
        FROM video_selection_snapshot_items
        WHERE token_hash = ?
        GROUP BY video_group
        """,
        (digest,),
    ):
        group_counts[str(row["video_group"])] = int(row["count"])
    total = int(snapshot["item_count"])
    return {
        "video_selection_token": str(token),
        "schema_version": VIDEO_SELECTION_SCHEMA_VERSION,
        "video_groups": groups,
        "total": total,
        "group_counts": group_counts,
        "videos": [
            {
                "video_group": str(entry["video_group"]),
                "name": str(entry["video_name"]),
            }
            for entry in entries
        ],
        "page": normalized_page,
        "page_size": normalized_page_size,
        "page_count": max(
            1,
            (total + normalized_page_size - 1) // normalized_page_size,
        ),
        "created_at": float(snapshot["created_at"]),
        "expires_at": float(snapshot["expires_at"]),
    }


def video_selection_membership(
    db: Database,
    token: str,
    *,
    video_group: str,
    video_names: list[str],
) -> dict[str, Any]:
    """Return membership only for one visible page, never the full name list."""

    digest, snapshot = _snapshot_row(db, token)
    groups = _normalize_groups(json.loads(str(snapshot["groups_json"])))
    group = str(video_group or "").strip()
    if group not in groups:
        raise VideoSelectionTokenError(
            "video_selection_token does not include this video group"
        )
    names = list(dict.fromkeys(str(name) for name in video_names))
    selected: set[str] = set()
    if names:
        placeholders = ",".join("?" for _ in names)
        rows = db.query(
            f"""
            SELECT video_name
            FROM video_selection_snapshot_items
            WHERE token_hash = ? AND video_group = ?
              AND video_name IN ({placeholders})
            """,
            (digest, group, *names),
        )
        selected = {str(row["video_name"]) for row in rows}
    group_count_row = db.get(
        """
        SELECT COUNT(*) AS count
        FROM video_selection_snapshot_items
        WHERE token_hash = ? AND video_group = ?
        """,
        (digest, group),
    )
    return {
        "video_selection_token": str(token),
        "total": int(snapshot["item_count"]),
        "group_count": int((group_count_row or {}).get("count") or 0),
        "selected_names": selected,
    }


def expand_video_selection_payload(
    db: Database,
    workspace: WorkspaceConfig,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Expand an opaque token to legacy-compatible exact server-owned names."""

    token = str(payload.get("video_selection_token") or "").strip()
    if not token:
        return payload
    if payload.get("selected_videos") is not None:
        raise ValueError(
            "provide either video_selection_token or selected_videos, not both"
        )
    if payload.get("source_assets"):
        raise ValueError(
            "video_selection_token cannot be combined with source_assets"
        )
    snapshot = resolve_video_selection_snapshot(
        db,
        workspace,
        token,
        require_non_empty=True,
    )
    groups = list(snapshot["video_groups"])
    supplied_groups = payload.get("video_groups")
    if supplied_groups is None:
        supplied_group = str(payload.get("video_group") or "").strip()
        supplied = [supplied_group] if supplied_group else []
    else:
        supplied = _normalize_groups(supplied_groups)
    if supplied and supplied != groups:
        raise VideoSelectionTokenError(
            "video_selection_token does not match the requested video groups"
        )
    expanded = dict(payload)
    expanded.pop("video_selection_token", None)
    expanded["selected_videos"] = list(snapshot["selected_videos"])
    if len(groups) == 1:
        expanded["video_group"] = groups[0]
        expanded.pop("video_groups", None)
    else:
        expanded["video_groups"] = groups
        expanded.pop("video_group", None)
    return expanded
