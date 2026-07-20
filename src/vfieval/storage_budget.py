from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from vfieval.config import WorkspaceConfig
from vfieval.db import Database


DEFAULT_MIN_FREE_BYTES = 2 * 1024**3
CAMPAIGN_RAW_BUDGET_NUMERATOR = 11
CAMPAIGN_RAW_BUDGET_DENOMINATOR = 5
CAMPAIGN_UNKNOWN_ASSET_OVERHEAD_BYTES = 64 * 1024**2


class StorageCapacityError(ValueError):
    def __init__(self, capacity: dict[str, Any]):
        self.capacity = capacity
        shortfall = max(0, int(capacity["required_free_bytes"]) - int(capacity["free_bytes"]))
        super().__init__(
            "insufficient workspace storage: "
            f"need {shortfall} more bytes after active reservations and the safety margin"
        )

    def public_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "type": type(self).__name__,
                "message": str(self),
                "capacity": self.capacity,
            }
        }


def _loads(value: Any) -> dict[str, Any]:
    try:
        decoded = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _active_run_reservations(db: Database) -> int:
    rows = db.query(
        """
        SELECT metadata_json
        FROM runs
        WHERE deleted_at IS NULL
          AND status IN (
            'queued', 'decoding', 'running', 'finalize_queued', 'finalizing',
            'metric_queued', 'metric_running', 'cancel_requested'
          )
        """
    )
    return sum(
        max(0, int((_loads(row.get("metadata_json")).get("workload") or {}).get("artifact_budget_bytes") or 0))
        for row in rows
    )


def _active_upload_reservations(db: Database) -> int:
    try:
        row = db.get(
            """
            SELECT COALESCE(SUM(expected_size), 0) AS bytes
            FROM upload_sessions
            WHERE state IN ('uploading', 'assembling', 'validating')
            """
        )
    except sqlite3.Error:
        return 0
    return max(0, int((row or {}).get("bytes") or 0))


def _active_campaign_reservations(db: Database) -> int:
    """Reserve decode caches plus frozen outputs for active Campaigns."""

    try:
        rows = db.query(
            """
            WITH selected_assets AS (
                SELECT p.campaign_id, i.reference_source_asset_id AS asset_id
                FROM evaluation_preparations_v2 p
                JOIN evaluation_items_v2 i ON i.campaign_id = p.campaign_id
                WHERE p.state IN ('queued', 'running')
                UNION
                SELECT p.campaign_id, b.source_asset_id AS asset_id
                FROM evaluation_preparations_v2 p
                JOIN evaluation_items_v2 i ON i.campaign_id = p.campaign_id
                JOIN evaluation_bindings_v2 b ON b.item_id = i.id
                WHERE p.state IN ('queued', 'running')
            )
            SELECT selected.campaign_id, ma.id, ma.size_bytes, ma.frame_count,
                   ma.width, ma.height
            FROM selected_assets selected
            JOIN media_assets ma ON ma.id = selected.asset_id
            """
        )
    except sqlite3.Error:
        return 0
    return sum(_campaign_asset_materialization_budget(row) for row in rows)


def _campaign_asset_materialization_budget(asset: dict[str, Any]) -> int:
    """Conservatively budget PNG decode cache and one frozen GT/Pred output.

    Source bytes already consume real free space and are not the main risk:
    low-bitrate 4K inputs can expand by orders of magnitude when decoded to
    lossless PNG frames.  The 2.2x raw-RGB allowance covers that cache, the
    CRF18 frozen stream, encoder/container overhead, and staging metadata.
    """

    source_bytes = max(0, int(asset.get("size_bytes") or 0))
    frame_count = max(0, int(asset.get("frame_count") or 0))
    width = max(0, int(asset.get("width") or 0))
    height = max(0, int(asset.get("height") or 0))
    raw_rgb_bytes = frame_count * width * height * 3
    if raw_rgb_bytes > 0:
        raw_budget = (
            raw_rgb_bytes * CAMPAIGN_RAW_BUDGET_NUMERATOR
            + CAMPAIGN_RAW_BUDGET_DENOMINATOR
            - 1
        ) // CAMPAIGN_RAW_BUDGET_DENOMINATOR
        return max(source_bytes * 2, raw_budget)
    return source_bytes * 3 + CAMPAIGN_UNKNOWN_ASSET_OVERHEAD_BYTES


def campaign_requested_bytes(db: Database, campaign_id: int) -> int:
    """Estimate decode-cache, staging, and frozen output bytes for a Campaign."""

    try:
        rows = db.query(
            """
            WITH selected_assets AS (
                SELECT reference_source_asset_id AS asset_id
                FROM evaluation_items_v2
                WHERE campaign_id = ?
                UNION
                SELECT b.source_asset_id AS asset_id
                FROM evaluation_items_v2 i
                JOIN evaluation_bindings_v2 b ON b.item_id = i.id
                WHERE i.campaign_id = ?
            )
            SELECT ma.id, ma.size_bytes, ma.frame_count, ma.width, ma.height
            FROM selected_assets selected
            JOIN media_assets ma ON ma.id = selected.asset_id
            """,
            (int(campaign_id), int(campaign_id)),
        )
    except sqlite3.Error:
        return 0
    return sum(_campaign_asset_materialization_budget(row) for row in rows)


def storage_capacity(
    db: Database,
    workspace: WorkspaceConfig,
    *,
    requested_bytes: int = 0,
) -> dict[str, Any]:
    requested = max(0, int(requested_bytes or 0))
    usage = shutil.disk_usage(workspace.root)
    run_bytes = _active_run_reservations(db)
    upload_bytes = _active_upload_reservations(db)
    campaign_bytes = _active_campaign_reservations(db)
    reserved = run_bytes + upload_bytes + campaign_bytes
    configured_floor = max(
        0,
        int(os.getenv("VFIEVAL_MIN_FREE_BYTES", str(DEFAULT_MIN_FREE_BYTES))),
    )
    safety_margin = max(configured_floor, int(usage.total * 0.02))
    required_free = reserved + requested + safety_margin
    remaining = int(usage.free) - reserved - requested
    return {
        "contract": "storage-capacity-v1",
        "workspace_device": str(Path(workspace.root).anchor),
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
        "requested_bytes": requested,
        "reserved_bytes": reserved,
        "reservation_breakdown": {
            "active_runs": run_bytes,
            "active_uploads": upload_bytes,
            "active_campaigns": campaign_bytes,
        },
        "safety_margin_bytes": safety_margin,
        "required_free_bytes": required_free,
        "remaining_after_request_bytes": remaining,
        "sufficient": int(usage.free) >= required_free,
    }


def ensure_storage_capacity(
    db: Database,
    workspace: WorkspaceConfig,
    *,
    requested_bytes: int,
) -> dict[str, Any]:
    capacity = storage_capacity(db, workspace, requested_bytes=requested_bytes)
    if not capacity["sufficient"]:
        raise StorageCapacityError(capacity)
    return capacity
