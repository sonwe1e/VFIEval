from __future__ import annotations

import hashlib
import secrets
from typing import Any

from vfieval.db import Database, utc_ts
from vfieval.media_items import _decode_item


SELECTION_TOKEN_TTL_SECONDS = 24 * 60 * 60
_TOKEN_MIN_LENGTH = 32
_TOKEN_MAX_LENGTH = 128


class SelectionTokenError(ValueError):
    """A persisted media selection can no longer be used safely."""


class SelectionTokenExpired(SelectionTokenError):
    """A media selection exists, but its bounded lifetime has elapsed."""


def _token_hash(token: str) -> str:
    normalized = str(token or "").strip()
    if not (_TOKEN_MIN_LENGTH <= len(normalized) <= _TOKEN_MAX_LENGTH):
        raise SelectionTokenError("selection_token is invalid")
    if any(not (character.isalnum() or character in {"-", "_"}) for character in normalized):
        raise SelectionTokenError("selection_token is invalid")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _selection_filter(query: str) -> tuple[str, list[Any]]:
    clauses = [
        "mi.collection_id = ?",
        "mi.state = 'ready'",
        "mi.deleted_at IS NULL",
        "a.state = 'ready'",
        "a.deleted_at IS NULL",
        "a.source_kind IN ('folder', 'upload')",
        "a.role = 'gt'",
    ]
    params: list[Any] = []
    normalized_query = str(query or "").strip()
    if normalized_query:
        clauses.append("(mi.display_name LIKE ? OR a.original_name LIKE ?)")
        needle = f"%{normalized_query}%"
        params.extend([needle, needle])
    return " AND ".join(clauses), params


def create_selection_snapshot(
    db: Database,
    *,
    group_id: int,
    query: str = "",
    ttl_seconds: float = SELECTION_TOKEN_TTL_SECONDS,
) -> dict[str, Any]:
    """Freeze one Item-first filter result without exposing every Item to the browser."""

    try:
        normalized_group_id = int(group_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("group_id must be a positive integer") from exc
    if normalized_group_id <= 0:
        raise ValueError("group_id must be a positive integer")
    normalized_query = str(query or "").strip()
    if len(normalized_query) > 500:
        raise ValueError("q must be at most 500 characters")
    ttl = min(7 * 24 * 60 * 60, max(60.0, float(ttl_seconds)))
    token = secrets.token_urlsafe(32)
    digest = _token_hash(token)
    created_at = utc_ts()
    expires_at = created_at + ttl
    where, filter_params = _selection_filter(normalized_query)
    with db.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        collection = conn.execute(
            "SELECT id FROM media_collections WHERE id = ?",
            (normalized_group_id,),
        ).fetchone()
        if collection is None:
            raise KeyError(f"media collection {normalized_group_id} not found")
        item_rows = conn.execute(
            f"""
            SELECT mi.id
            FROM media_items mi
            JOIN media_assets a ON a.id = mi.canonical_gt_asset_id
            WHERE {where}
            ORDER BY mi.display_name, mi.id
            """,
            (normalized_group_id, *filter_params),
        ).fetchall()
        conn.execute(
            "DELETE FROM media_item_selection_snapshots WHERE expires_at <= ?",
            (created_at,),
        )
        conn.execute(
            """
            INSERT INTO media_item_selection_snapshots(
                token_hash, collection_id, query_text, item_count, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                digest,
                normalized_group_id,
                normalized_query,
                len(item_rows),
                created_at,
                expires_at,
            ),
        )
        conn.executemany(
            """
            INSERT INTO media_item_selection_snapshot_items(
                token_hash, ordinal, media_item_id
            ) VALUES (?, ?, ?)
            """,
            (
                (digest, ordinal, int(row["id"]))
                for ordinal, row in enumerate(item_rows)
            ),
        )
    return {
        "selection_token": token,
        "group_id": normalized_group_id,
        "q": normalized_query,
        "total": len(item_rows),
        "created_at": created_at,
        "expires_at": expires_at,
    }


def _snapshot_row(db: Database, token: str) -> tuple[str, dict[str, Any]]:
    digest = _token_hash(token)
    row = db.get(
        """
        SELECT token_hash, collection_id, query_text, item_count, created_at, expires_at
        FROM media_item_selection_snapshots
        WHERE token_hash = ?
        """,
        (digest,),
    )
    if row is None:
        raise SelectionTokenError("selection_token was not found; select the filtered Items again")
    if float(row["expires_at"]) <= utc_ts():
        raise SelectionTokenExpired("selection_token expired; select the filtered Items again")
    return digest, row


def resolve_selection_snapshot(
    db: Database,
    token: str,
    *,
    require_non_empty: bool = False,
) -> dict[str, Any]:
    """Resolve and revalidate every snapshotted Item using a JOIN, never a large IN list."""

    digest, snapshot = _snapshot_row(db, token)
    rows = db.query(
        """
        SELECT ssi.media_item_id
        FROM media_item_selection_snapshot_items ssi
        JOIN media_items mi ON mi.id = ssi.media_item_id
        JOIN media_assets a ON a.id = mi.canonical_gt_asset_id
        WHERE ssi.token_hash = ?
          AND mi.collection_id = ?
          AND mi.state = 'ready' AND mi.deleted_at IS NULL
          AND a.state = 'ready' AND a.deleted_at IS NULL
          AND a.source_kind IN ('folder', 'upload') AND a.role = 'gt'
        ORDER BY ssi.ordinal
        """,
        (digest, int(snapshot["collection_id"])),
    )
    expected = int(snapshot["item_count"])
    if len(rows) != expected:
        raise SelectionTokenError(
            "selection_token contains an Item that is no longer ready or no longer belongs "
            "to the original GT Collection; select the filtered Items again"
        )
    if require_non_empty and not rows:
        raise SelectionTokenError("selection_token contains no Media Items")
    return {
        "token_hash": digest,
        "group_id": int(snapshot["collection_id"]),
        "q": str(snapshot["query_text"] or ""),
        "total": expected,
        "created_at": float(snapshot["created_at"]),
        "expires_at": float(snapshot["expires_at"]),
        "media_item_ids": [int(row["media_item_id"]) for row in rows],
    }


def selection_snapshot_page(
    db: Database,
    token: str,
    *,
    page: int = 1,
    page_size: int = 100,
) -> dict[str, Any]:
    snapshot = resolve_selection_snapshot(db, token)
    normalized_page = max(1, int(page))
    normalized_page_size = min(200, max(1, int(page_size)))
    rows = db.query(
        """
        SELECT mi.*, c.name AS collection_name, c.slug AS collection_slug,
               a.source_key AS canonical_source_key,
               a.source_kind AS canonical_source_kind,
               a.display_name AS canonical_display_name,
               a.storage_path AS canonical_storage_path,
               a.state AS canonical_asset_state,
               a.frame_count, a.width, a.height, a.fps, a.size_bytes,
               a.content_sha256 AS canonical_content_sha256
        FROM media_item_selection_snapshot_items ssi
        JOIN media_items mi ON mi.id = ssi.media_item_id
        JOIN media_collections c ON c.id = mi.collection_id
        JOIN media_assets a ON a.id = mi.canonical_gt_asset_id
        WHERE ssi.token_hash = ?
        ORDER BY ssi.ordinal
        LIMIT ? OFFSET ?
        """,
        (
            str(snapshot["token_hash"]),
            normalized_page_size,
            (normalized_page - 1) * normalized_page_size,
        ),
    )
    total = int(snapshot["total"])
    return {
        "selection_token": str(token),
        "group_id": int(snapshot["group_id"]),
        "q": str(snapshot["q"]),
        "items": [_decode_item(row) for row in rows],
        "page": normalized_page,
        "page_size": normalized_page_size,
        "total": total,
        "page_count": max(1, (total + normalized_page_size - 1) // normalized_page_size),
        "created_at": float(snapshot["created_at"]),
        "expires_at": float(snapshot["expires_at"]),
    }


def list_methods_for_selection_snapshot(
    db: Database,
    token: str,
) -> dict[str, Any]:
    """Return coverage summaries without serializing every Item/member binding."""

    snapshot = resolve_selection_snapshot(db, token)
    rows = db.query(
        """
        SELECT
            CASE WHEN mim.producer_run_id IS NULL THEN 'external' ELSE 'run' END AS kind,
            mim.producer_run_id AS run_id,
            CASE
                WHEN mim.producer_run_id IS NULL THEN mim.method_key
                ELSE ''
            END AS method_key,
            COALESCE(MAX(r.name), MAX(mim.method_key), 'Pred') AS label,
            COUNT(DISTINCT ssi.media_item_id) AS covered_count
        FROM media_item_selection_snapshot_items ssi
        JOIN media_item_members mim ON mim.item_id = ssi.media_item_id
        JOIN media_assets a ON a.id = mim.asset_id
        LEFT JOIN runs r ON r.id = mim.producer_run_id
        WHERE ssi.token_hash = ?
          AND mim.reusable_as_pred = 1
          AND mim.state = 'ready' AND mim.deleted_at IS NULL
          AND a.state = 'ready' AND a.deleted_at IS NULL
          AND (
            (
                mim.member_role = 'external_pred'
                AND mim.producer_kind = 'external'
                AND mim.producer_run_id IS NULL
                AND a.source_kind = 'upload'
            )
            OR
            (
                mim.member_role = 'model_pred'
                AND mim.producer_kind = 'model_inference'
                AND mim.producer_run_id IS NOT NULL
                AND a.source_kind = 'run_artifact'
                AND r.status IN ('completed', 'metric_queued', 'metric_running')
                AND r.deleted_at IS NULL
                AND r.artifact_cleaned_at IS NULL
                AND COALESCE(
                    json_extract(r.metadata_json, '$.run_type'),
                    'model_inference'
                ) = 'model_inference'
            )
          )
        GROUP BY kind, mim.producer_run_id,
                 CASE WHEN mim.producer_run_id IS NULL THEN mim.method_key ELSE '' END
        ORDER BY covered_count DESC, label, method_key
        """,
        (str(snapshot["token_hash"]),),
    )
    total = int(snapshot["total"])
    methods: list[dict[str, Any]] = []
    for row in rows:
        covered = int(row.get("covered_count") or 0)
        methods.append(
            {
                "kind": str(row["kind"]),
                "run_id": int(row["run_id"]) if row.get("run_id") is not None else None,
                "method_key": str(row.get("method_key") or ""),
                "label": str(row.get("label") or row.get("method_key") or "Pred"),
                "covered_count": covered,
                "missing_count": max(0, total - covered),
                "total_items": total,
                "complete": covered == total,
            }
        )
    return {
        "selection_token": str(token),
        "group_id": int(snapshot["group_id"]),
        "total": total,
        "methods": methods,
    }
