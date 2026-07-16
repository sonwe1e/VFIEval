from __future__ import annotations

import hashlib
import json
import math
import os
import random
import secrets
import shutil
import statistics
import threading
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from PIL import Image, ImageChops

from vfieval.compare_inputs import (
    inspect_compare_path,
    resolve_compare_descriptor,
    validate_strict_alignment,
    validate_strict_decoded_alignment,
)
from vfieval.alignment import materialize_frame_sets, plan_alignment, validate_temporal_alignment
from vfieval.config import WorkspaceConfig
from vfieval.db import Database, utc_ts
from vfieval.evaluations import CONFIDENCE_VALUES, METRIC_DIRECTIONS, QUALITY_REASONS
from vfieval.media_assets import get_asset, resolve_asset_path, slugify, sync_run_assets
from vfieval.media_items import (
    get_media_item,
    list_item_predictions,
    list_methods_for_items,
    resolve_item_member,
    resolve_item_reference,
)


CAMPAIGN_V2_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_campaigns_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    public_token TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    public_title TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'draft', 'preparing', 'published', 'failed', 'closed', 'archived'
    )),
    target_votes INTEGER NOT NULL DEFAULT 3,
    seed INTEGER NOT NULL,
    vote_revision INTEGER NOT NULL DEFAULT 0,
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    published_at REAL,
    closed_at REAL,
    archived_at REAL
);

CREATE TABLE IF NOT EXISTS evaluation_methods_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL REFERENCES evaluation_campaigns_v2(id) ON DELETE CASCADE,
    slot TEXT NOT NULL CHECK(slot IN ('a', 'b')),
    source_kind TEXT NOT NULL CHECK(source_kind IN ('run_track', 'upload')),
    source_run_id INTEGER,
    source_track_label TEXT NOT NULL DEFAULT '',
    label_snapshot TEXT NOT NULL,
    model_snapshot TEXT NOT NULL DEFAULT '',
    checkpoint_snapshot TEXT NOT NULL DEFAULT '',
    source_spec_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    UNIQUE(campaign_id, slot)
);

CREATE TABLE IF NOT EXISTS evaluation_items_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL REFERENCES evaluation_campaigns_v2(id) ON DELETE CASCADE,
    video_name TEXT NOT NULL,
    reference_source_asset_id INTEGER NOT NULL REFERENCES media_assets(id) ON DELETE RESTRICT,
    frozen_reference_asset_id INTEGER REFERENCES media_assets(id) ON DELETE RESTRICT,
    alignment_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    UNIQUE(campaign_id, video_name)
);

CREATE TABLE IF NOT EXISTS evaluation_bindings_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES evaluation_items_v2(id) ON DELETE CASCADE,
    method_id INTEGER NOT NULL REFERENCES evaluation_methods_v2(id) ON DELETE CASCADE,
    source_asset_id INTEGER NOT NULL REFERENCES media_assets(id) ON DELETE RESTRICT,
    frozen_asset_id INTEGER REFERENCES media_assets(id) ON DELETE RESTRICT,
    state TEXT NOT NULL DEFAULT 'selected' CHECK(state IN (
        'selected', 'validating', 'ready', 'invalid'
    )),
    alignment_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(item_id, method_id)
);

CREATE TABLE IF NOT EXISTS evaluation_preparations_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL UNIQUE REFERENCES evaluation_campaigns_v2(id) ON DELETE CASCADE,
    state TEXT NOT NULL CHECK(state IN ('queued', 'running', 'completed', 'failed')),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    staging_path TEXT NOT NULL DEFAULT '',
    final_path TEXT NOT NULL DEFAULT '',
    claim_token TEXT NOT NULL DEFAULT '',
    report_json TEXT NOT NULL DEFAULT '{}',
    error_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_tasks_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_token TEXT NOT NULL UNIQUE,
    campaign_id INTEGER NOT NULL REFERENCES evaluation_campaigns_v2(id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL REFERENCES evaluation_items_v2(id) ON DELETE CASCADE,
    binding_a_id INTEGER NOT NULL REFERENCES evaluation_bindings_v2(id) ON DELETE RESTRICT,
    binding_b_id INTEGER NOT NULL REFERENCES evaluation_bindings_v2(id) ON DELETE RESTRICT,
    state TEXT NOT NULL DEFAULT 'ready' CHECK(state IN ('ready', 'closed')),
    created_at REAL NOT NULL,
    UNIQUE(campaign_id, item_id)
);

CREATE TABLE IF NOT EXISTS evaluation_assignments_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment_token TEXT NOT NULL UNIQUE,
    task_id INTEGER NOT NULL REFERENCES evaluation_tasks_v2(id) ON DELETE CASCADE,
    evaluator_id TEXT NOT NULL REFERENCES evaluators(id) ON DELETE CASCADE,
    state TEXT NOT NULL CHECK(state IN ('leased', 'voted', 'expired')),
    side_swap INTEGER NOT NULL DEFAULT 0,
    lease_expires_at REAL NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(task_id, evaluator_id)
);

CREATE TABLE IF NOT EXISTS evaluation_votes_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL REFERENCES evaluation_tasks_v2(id) ON DELETE CASCADE,
    evaluator_id TEXT NOT NULL REFERENCES evaluators(id) ON DELETE CASCADE,
    assignment_id INTEGER NOT NULL REFERENCES evaluation_assignments_v2(id) ON DELETE RESTRICT,
    choice TEXT NOT NULL CHECK(choice IN ('left', 'right', 'tie')),
    preferred_method_id INTEGER REFERENCES evaluation_methods_v2(id) ON DELETE SET NULL,
    reasons_json TEXT NOT NULL DEFAULT '[]',
    confidence TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    duration_ms INTEGER,
    presentation_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(task_id, evaluator_id)
);

CREATE TABLE IF NOT EXISTS evaluation_analysis_cache_v2 (
    campaign_id INTEGER NOT NULL REFERENCES evaluation_campaigns_v2(id) ON DELETE CASCADE,
    cache_key TEXT NOT NULL,
    vote_revision INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(campaign_id, cache_key)
);

CREATE INDEX IF NOT EXISTS idx_eval_methods_v2_run
ON evaluation_methods_v2(source_run_id, source_track_label);
CREATE INDEX IF NOT EXISTS idx_eval_items_v2_campaign
ON evaluation_items_v2(campaign_id, video_name);
CREATE INDEX IF NOT EXISTS idx_eval_bindings_v2_item
ON evaluation_bindings_v2(item_id, method_id, state);
CREATE INDEX IF NOT EXISTS idx_eval_tasks_v2_campaign
ON evaluation_tasks_v2(campaign_id, state);
CREATE INDEX IF NOT EXISTS idx_eval_assignments_v2_lease
ON evaluation_assignments_v2(state, lease_expires_at, task_id);
CREATE INDEX IF NOT EXISTS idx_eval_votes_v2_task
ON evaluation_votes_v2(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_eval_votes_v2_evaluator
ON evaluation_votes_v2(evaluator_id, created_at);
"""


class EvaluationConflict(ValueError):
    """The campaign is valid, but concurrent state made this action stale."""


PREPARATION_CLAIM_STALE_SECONDS = 10 * 60


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, ensure_ascii=False)


def _loads(value: str | None, default: Any | None = None) -> Any:
    if not value:
        return {} if default is None else default
    return json.loads(value)


def ensure_v2_schema(db: Database) -> None:
    """Install V2 tables without modifying or guessing at legacy campaign rows."""
    _ensure_evaluation_package_media_kind(db)
    with db.connection() as conn:
        conn.executescript(CAMPAIGN_V2_SCHEMA)
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(evaluation_campaigns_v2)").fetchall()
        }
        if "name" not in columns:
            conn.execute("ALTER TABLE evaluation_campaigns_v2 ADD COLUMN name TEXT NOT NULL DEFAULT ''")
            conn.execute(
                "UPDATE evaluation_campaigns_v2 SET name = public_title WHERE name = ''"
            )
        preparation_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(evaluation_preparations_v2)").fetchall()
        }
        if "claim_token" not in preparation_columns:
            conn.execute(
                "ALTER TABLE evaluation_preparations_v2 ADD COLUMN claim_token TEXT NOT NULL DEFAULT ''"
            )
        item_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(evaluation_items_v2)").fetchall()
        }
        if "media_item_id" not in item_columns:
            conn.execute("ALTER TABLE evaluation_items_v2 ADD COLUMN media_item_id INTEGER")
        binding_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(evaluation_bindings_v2)").fetchall()
        }
        if "source_member_id" not in binding_columns:
            conn.execute("ALTER TABLE evaluation_bindings_v2 ADD COLUMN source_member_id INTEGER")
        if "frozen_member_id" not in binding_columns:
            conn.execute("ALTER TABLE evaluation_bindings_v2 ADD COLUMN frozen_member_id INTEGER")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_items_v2_media_item "
            "ON evaluation_items_v2(media_item_id, campaign_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_bindings_v2_source_member "
            "ON evaluation_bindings_v2(source_member_id, item_id)"
        )


def _ensure_evaluation_package_media_kind(db: Database) -> None:
    """Expand the media identity CHECK while retaining ids and foreign keys.

    SQLite cannot alter CHECK constraints in place.  The migration uses its
    documented legacy rename mode so child foreign keys continue to reference
    the replacement ``media_assets`` table rather than the temporary name.
    """
    conn = db.connect()
    try:
        sql_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'media_assets'"
        ).fetchone()
        if sql_row is None or "evaluation_package" in str(sql_row["sql"] or ""):
            return
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("PRAGMA legacy_alter_table=ON")
        conn.execute("BEGIN EXCLUSIVE")
        sql_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'media_assets'"
        ).fetchone()
        if sql_row is not None and "evaluation_package" not in str(sql_row["sql"] or ""):
            conn.execute("ALTER TABLE media_assets RENAME TO media_assets_before_evaluation_v2")
            conn.execute(
                """
                CREATE TABLE media_assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_id INTEGER REFERENCES media_collections(id) ON DELETE SET NULL,
                    source_key TEXT NOT NULL UNIQUE,
                    source_kind TEXT NOT NULL CHECK(source_kind IN (
                        'folder', 'upload', 'run_artifact', 'evaluation_package'
                    )),
                    media_kind TEXT NOT NULL CHECK(media_kind IN ('video', 'frame_sequence')),
                    role TEXT NOT NULL CHECK(role IN ('source', 'gt', 'pred')),
                    display_name TEXT NOT NULL,
                    original_name TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'ready',
                    content_sha256 TEXT,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    storage_path TEXT NOT NULL,
                    mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
                    frame_count INTEGER NOT NULL DEFAULT 0,
                    width INTEGER NOT NULL DEFAULT 0,
                    height INTEGER NOT NULL DEFAULT 0,
                    fps REAL,
                    provenance_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    deleted_at REAL,
                    UNIQUE(collection_id, display_name)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO media_assets(
                    id, collection_id, source_key, source_kind, media_kind, role,
                    display_name, original_name, state, content_sha256, size_bytes,
                    storage_path, mime_type, frame_count, width, height, fps,
                    provenance_json, metadata_json, created_at, updated_at, deleted_at
                )
                SELECT id, collection_id, source_key, source_kind, media_kind, role,
                       display_name, original_name, state, content_sha256, size_bytes,
                       storage_path, mime_type, frame_count, width, height, fps,
                       provenance_json, metadata_json, created_at, updated_at, deleted_at
                FROM media_assets_before_evaluation_v2
                """
            )
            conn.execute("DROP TABLE media_assets_before_evaluation_v2")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_media_assets_catalog
                ON media_assets(state, role, source_kind, collection_id, display_name)
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_assets_hash ON media_assets(content_sha256)"
            )
        conn.commit()
        conn.execute("PRAGMA legacy_alter_table=OFF")
        conn.execute("PRAGMA foreign_keys=ON")
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError("media_assets evaluation-package migration broke foreign keys")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _token() -> str:
    return secrets.token_urlsafe(24)


def _video_key(value: str) -> str:
    # Preserve the actual selected file identity. Stemming/casefolding silently
    # conflates e.g. clip.mp4 and clip.mov, which makes the common-video matrix
    # drop or mis-pair outputs.
    return Path(str(value or "").strip()).name


def _asset_from_row(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    def field(name: str) -> Any:
        return row.get(f"{prefix}{name}")

    return {
        "id": int(field("id")),
        "source_key": str(field("source_key") or ""),
        "source_kind": str(field("source_kind") or ""),
        "media_kind": str(field("media_kind") or "video"),
        "role": str(field("role") or ""),
        "display_name": str(field("display_name") or ""),
        "state": str(field("state") or ""),
        "content_sha256": field("content_sha256"),
        "size_bytes": int(field("size_bytes") or 0),
        "storage_path": str(field("storage_path") or ""),
        "mime_type": str(field("mime_type") or "application/octet-stream"),
        "frame_count": int(field("frame_count") or 0),
        "width": int(field("width") or 0),
        "height": int(field("height") or 0),
        "fps": float(field("fps")) if field("fps") is not None else None,
        "provenance": _loads(field("provenance_json")),
        "metadata": _loads(field("metadata_json")),
        "deleted_at": field("deleted_at"),
    }


def _asset_info(asset: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_count": int(asset.get("frame_count") or 0),
        "width": int(asset.get("width") or 0),
        "height": int(asset.get("height") or 0),
        "fps": asset.get("fps"),
    }


def list_run_outputs(db: Database) -> list[dict[str, Any]]:
    """Return valid Run -> video -> track output groups for V2 selectors."""
    rows = db.query(
        """
        SELECT r.id AS run_id, r.name AS run_name, r.created_at AS run_created_at,
               rma.video_name, rma.track_label, rma.model_name, rma.checkpoint,
               ma.id AS asset_id, ma.display_name, ma.frame_count, ma.width,
               ma.height, ma.fps, ma.created_at AS asset_created_at
        FROM run_media_assets rma
        JOIN runs r ON r.id = rma.run_id
        JOIN media_assets ma ON ma.id = rma.asset_id
        WHERE rma.role = 'pred'
          AND r.status = 'completed'
          AND r.deleted_at IS NULL
          AND r.artifact_cleaned_at IS NULL
          AND ma.state = 'ready'
          AND ma.deleted_at IS NULL
          AND ma.source_kind = 'run_artifact'
        ORDER BY r.created_at DESC, r.id DESC, rma.video_name, rma.track_label, ma.id DESC
        """
    )
    runs: dict[int, dict[str, Any]] = {}
    seen: set[tuple[int, str, str]] = set()
    for row in rows:
        key = (int(row["run_id"]), str(row["video_name"]), str(row["track_label"] or ""))
        if key in seen:
            continue
        seen.add(key)
        run = runs.setdefault(
            int(row["run_id"]),
            {
                "run_id": int(row["run_id"]),
                "run_name": str(row["run_name"]),
                "created_at": row["run_created_at"],
                "_videos": {},
            },
        )
        video_name = str(row["video_name"])
        video = run["_videos"].setdefault(video_name, {"video_name": video_name, "tracks": []})
        video["tracks"].append(
            {
                "track_label": str(row["track_label"] or ""),
                "asset_id": int(row["asset_id"]),
                "model_name": str(row["model_name"] or ""),
                "checkpoint": str(row["checkpoint"] or ""),
                "frame_count": int(row["frame_count"] or 0),
                "width": int(row["width"] or 0),
                "height": int(row["height"] or 0),
                "fps": row["fps"],
            }
        )
    result: list[dict[str, Any]] = []
    for run in runs.values():
        videos = list(run.pop("_videos").values())
        run["videos"] = videos
        run["video_count"] = len(videos)
        run["track_count"] = len(
            {
                str(track.get("track_label") or "")
                for video in videos
                for track in video["tracks"]
            }
        )
        result.append(run)
    return result


def _normalize_method_specs(body: dict[str, Any]) -> list[dict[str, Any]]:
    methods = body.get("methods") or body.get("pred_methods") or []
    if not methods and (body.get("method_a") or body.get("method_b")):
        methods = [body.get("method_a"), body.get("method_b")]
    if not isinstance(methods, list) or len(methods) != 2:
        raise ValueError("Campaign V2 requires exactly two Pred methods")
    normalized: list[dict[str, Any]] = []
    for slot, raw in zip(("a", "b"), methods):
        if not isinstance(raw, dict):
            raise ValueError("each Campaign V2 method must be an object")
        source_kind = str(raw.get("source_kind") or "run_track")
        if source_kind not in {"run_track", "upload"}:
            raise ValueError("Campaign V2 method source_kind must be run_track or upload")
        run_id: int | None = None
        if source_kind == "run_track":
            try:
                run_id = int(raw.get("run_id"))
            except (TypeError, ValueError) as exc:
                raise ValueError("each Run/Track Campaign V2 method requires run_id") from exc
            if run_id <= 0:
                raise ValueError("each Run/Track Campaign V2 method requires a positive run_id")
        else:
            videos = raw.get("videos") or []
            if not isinstance(videos, list) or not videos:
                raise ValueError("an uploaded Campaign V2 method requires a non-empty videos list")
            if not str(raw.get("label") or "").strip():
                raise ValueError("an uploaded Campaign V2 method requires label")
        normalized.append(
            {
                "slot": slot,
                "source_kind": source_kind,
                "run_id": run_id,
                "track_label": str(
                    raw.get("source_track_label")
                    or raw.get("compare_track_label")
                    or raw.get("track_label")
                    or ""
                ).strip(),
                "label": str(raw.get("label") or raw.get("track_label") or "").strip(),
                "raw": dict(raw),
            }
        )
    identities = {
        (
            row["source_kind"],
            row["run_id"],
            row["track_label"],
            tuple(
                sorted(
                    int(video.get("asset_id") or 0)
                    for video in (row["raw"].get("videos") or [])
                    if isinstance(video, dict)
                )
            ),
        )
        for row in normalized
    }
    if len(identities) != 2:
        raise ValueError("Campaign V2 methods must refer to two distinct Pred sources")
    return normalized


def _method_output_rows(db: Database, workspace: WorkspaceConfig, method: dict[str, Any]) -> dict[str, Any]:
    if method["source_kind"] == "upload":
        outputs: dict[str, dict[str, Any]] = {}
        references: dict[str, dict[str, Any]] = {}
        display_names: dict[str, str] = {}
        for descriptor in method["raw"].get("videos") or []:
            if not isinstance(descriptor, dict):
                raise ValueError("uploaded Campaign V2 video descriptors must be objects")
            video_name = str(descriptor.get("video_name") or descriptor.get("video") or "").strip()
            if not video_name:
                raise ValueError("uploaded Campaign V2 video descriptor requires video_name")
            try:
                pred_id = int(descriptor.get("asset_id"))
                reference_id = int(descriptor.get("reference_asset_id"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "uploaded Campaign V2 videos require asset_id and reference_asset_id"
                ) from exc
            pred = get_asset(db, pred_id)
            reference = get_asset(db, reference_id)
            if pred["source_kind"] != "upload" or pred["role"] != "pred" or pred["state"] != "ready":
                raise ValueError(f"uploaded Pred media asset {pred_id} is not ready")
            if (
                reference["source_kind"] not in {"folder", "upload"}
                or reference["role"] not in {"source", "gt"}
                or reference["state"] != "ready"
            ):
                raise ValueError(f"uploaded method reference media asset {reference_id} is not ready")
            key = _video_key(video_name)
            if key in outputs:
                raise ValueError(f"uploaded Campaign V2 method repeats video {video_name}")
            outputs[key] = pred
            references[key] = reference
            display_names[key] = video_name
        return {
            "method": {
                **method,
                "label": str(method["label"]),
                "run_name": "",
                "model_name": str(method["raw"].get("model_name") or ""),
                "checkpoint": str(method["raw"].get("checkpoint") or ""),
            },
            "outputs": outputs,
            "references": references,
            "display_names": display_names,
        }
    run = db.get_run(int(method["run_id"]))
    if run.get("deleted_at") is not None or run.get("artifact_cleaned_at") is not None:
        raise ValueError(f"Run {method['run_id']} output is unavailable")
    if str(run.get("status") or "") != "completed":
        raise ValueError(f"Run {method['run_id']} is not completed")
    sync_run_assets(db, workspace, int(method["run_id"]))
    tracks = db.query(
        """
        SELECT DISTINCT track_label FROM run_media_assets
        WHERE run_id = ? AND role = 'pred'
        ORDER BY track_label
        """,
        (int(method["run_id"]),),
    )
    available_tracks = [str(row["track_label"] or "") for row in tracks]
    requested_track = str(method["track_label"] or "")
    if not requested_track and len(available_tracks) > 1:
        raise ValueError(f"Run {method['run_id']} has multiple tracks; track_label is required")
    if requested_track not in available_tracks:
        if len(available_tracks) == 1 and not requested_track:
            requested_track = available_tracks[0]
        else:
            raise ValueError(
                f"Run {method['run_id']} has no track {requested_track or '<default>'}"
            )
    method["track_label"] = requested_track
    pred_rows = db.query(
        """
        SELECT rma.video_name AS bound_video_name, ma.* FROM run_media_assets rma
        JOIN media_assets ma ON ma.id = rma.asset_id
        WHERE rma.run_id = ? AND rma.role = 'pred' AND rma.track_label = ?
          AND ma.state = 'ready' AND ma.deleted_at IS NULL
        ORDER BY CASE WHEN ma.source_kind = 'run_artifact' THEN 0 ELSE 1 END, ma.id DESC
        """,
        (int(method["run_id"]), requested_track),
    )
    outputs: dict[str, dict[str, Any]] = {}
    display_names: dict[str, str] = {}
    for row in pred_rows:
        asset = _asset_from_row(row)
        provenance = asset.get("provenance") or {}
        video_name = str(
            row.get("bound_video_name")
            or provenance.get("video_name")
            or asset["display_name"]
        )
        key = _video_key(video_name)
        if not key:
            continue
        if key in outputs:
            raise ValueError(
                f"Run {method['run_id']} has ambiguous duplicate Pred video identity {video_name}"
            )
        outputs[key] = asset
        display_names[key] = video_name
    gt_rows = db.query(
        """
        SELECT rma.video_name AS bound_video_name, rma.role AS bound_role, ma.*
        FROM run_media_assets rma
        JOIN media_assets ma ON ma.id = rma.asset_id
        WHERE rma.run_id = ? AND rma.role IN ('gt', 'source')
          AND ma.state = 'ready' AND ma.deleted_at IS NULL
        ORDER BY CASE rma.role WHEN 'gt' THEN 0 ELSE 1 END,
                 CASE WHEN ma.source_kind IN ('folder', 'upload') THEN 0 ELSE 1 END,
                 ma.id DESC
        """,
        (int(method["run_id"]),),
    )
    references: dict[str, dict[str, Any]] = {}
    for row in gt_rows:
        key = _video_key(str(row.get("bound_video_name") or row.get("display_name") or ""))
        if not key:
            continue
        asset = _asset_from_row(row)
        existing = references.get(key)
        if existing is not None:
            if not _same_reference(existing, asset):
                raise ValueError(
                    f"Run {method['run_id']} has conflicting GT assets for video {key}"
                )
            continue
        references[key] = asset
    metadata = run.get("metadata") or {}
    label = method["label"] or requested_track or str(run.get("name") or f"Run {method['run_id']}")
    return {
        "method": {
            **method,
            "label": label,
            "run_name": str(run.get("name") or ""),
            "model_name": str(metadata.get("model_file") or run.get("model_name") or ""),
            "checkpoint": str(metadata.get("checkpoint") or ""),
        },
        "outputs": outputs,
        "references": references,
        "display_names": display_names,
    }


def _same_reference(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if int(left["id"]) == int(right["id"]):
        return True
    left_hash = str(left.get("content_sha256") or "")
    right_hash = str(right.get("content_sha256") or "")
    if left_hash and right_hash:
        return left_hash == right_hash
    return Path(str(left.get("storage_path") or "")).resolve() == Path(
        str(right.get("storage_path") or "")
    ).resolve()


def _shallow_alignment(reference: dict[str, Any], distorted: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        return "ready", validate_strict_alignment(_asset_info(reference), _asset_info(distorted))
    except ValueError as exc:
        return "alignment_mismatch", {"error": str(exc)}


def _preview_pair_alignment(
    db: Database,
    workspace: WorkspaceConfig,
    reference: dict[str, Any],
    distorted: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Deep-validate a selectable matrix pair, including decoded timestamps."""
    state, alignment = _shallow_alignment(reference, distorted)
    if state != "ready":
        return state, alignment
    try:
        reference_asset, reference_path = _managed_source_asset(
            db, workspace, int(reference["id"]), "reference"
        )
        distorted_asset, distorted_path = _managed_source_asset(
            db, workspace, int(distorted["id"]), "distorted"
        )
        return "ready", _deep_validate_pair(
            db,
            workspace,
            reference_asset,
            reference_path,
            distorted_asset,
            distorted_path,
            "evaluation_v2_validation",
        )
    except Exception as exc:
        return "alignment_mismatch", {"error": str(exc)}


def _normalize_item_method_specs(body: dict[str, Any]) -> list[dict[str, Any]]:
    raw_methods = body.get("methods")
    if raw_methods is None:
        raw_methods = [body.get("method_a"), body.get("method_b")]
    if not isinstance(raw_methods, list) or len(raw_methods) != 2:
        raise ValueError("Campaign V2 requires exactly two methods")
    methods: list[dict[str, Any]] = []
    for slot, raw in zip(("a", "b"), raw_methods):
        if not isinstance(raw, dict):
            raise ValueError("each Campaign V2 method must be an object")
        kind = str(raw.get("kind") or raw.get("source_kind") or "run").strip().lower()
        if kind in {"run", "run_track"}:
            try:
                run_id = int(raw.get("run_id"))
            except (TypeError, ValueError) as exc:
                raise ValueError("each Run method requires a positive run_id") from exc
            if run_id <= 0:
                raise ValueError("each Run method requires a positive run_id")
            methods.append(
                {
                    "slot": slot,
                    "kind": "run",
                    "source_kind": "run_track",
                    "run_id": run_id,
                    "method_key": str(raw.get("method_key") or raw.get("track_label") or ""),
                    "label": str(raw.get("label") or "").strip(),
                    "raw": dict(raw),
                }
            )
        elif kind in {"external", "upload"}:
            method_key = str(raw.get("method_key") or "").strip()
            if not method_key:
                raise ValueError("external Campaign method requires method_key")
            methods.append(
                {
                    "slot": slot,
                    "kind": "external",
                    "source_kind": "upload",
                    "run_id": None,
                    "method_key": method_key,
                    "label": str(raw.get("label") or method_key).strip(),
                    "raw": dict(raw),
                }
            )
        else:
            raise ValueError("Campaign method kind must be run or external")
    identities = {
        (method["kind"], method.get("run_id"), method.get("method_key"))
        for method in methods
    }
    if len(identities) != 2:
        raise ValueError("Campaign methods A and B must be distinct")
    return methods


def _item_method_prediction(
    db: Database,
    item_id: int,
    method: dict[str, Any],
) -> dict[str, Any] | None:
    predictions = list_item_predictions(db, int(item_id))["predictions"]
    if method["kind"] == "run":
        candidates = [
            prediction
            for prediction in predictions
            if prediction.get("producer_run_id") is not None
            and int(prediction["producer_run_id"]) == int(method["run_id"])
        ]
    else:
        candidates = [
            prediction
            for prediction in predictions
            if prediction.get("producer_run_id") is None
            and str(prediction.get("method_key") or "") == str(method["method_key"])
        ]
    requested_key = str(method.get("method_key") or "")
    if requested_key:
        keyed = [
            prediction
            for prediction in candidates
            if str(prediction.get("method_key") or "") == requested_key
        ]
        if keyed:
            candidates = keyed
    if not candidates:
        return None
    candidates.sort(key=lambda prediction: int(prediction["id"]), reverse=True)
    return candidates[0]


def _campaign_item_alignment(
    db: Database,
    workspace: WorkspaceConfig,
    item_id: int,
    predictions: list[dict[str, Any]],
    spatial_policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    from vfieval.datasets import _load_compare_source_frames

    def available_timestamps(values: list[float | None] | None) -> list[float] | None:
        if not values or any(value is None for value in values):
            return None
        return [float(value) for value in values if value is not None]

    def decoded_dimensions(frame_paths: list[Path], label: str) -> tuple[int, int]:
        if not frame_paths:
            raise ValueError(f"Campaign V2 {label} has no decoded frames")
        with Image.open(frame_paths[0]) as image:
            width, height = image.size
        return int(width), int(height)

    reference = resolve_compare_descriptor(
        workspace,
        db,
        {"kind": "media_item", "item_id": int(item_id)},
        role="reference",
    )
    resolved_predictions = [
        resolve_compare_descriptor(
            workspace,
            db,
            {"kind": "media_item_member", "member_id": int(prediction["id"])},
            role="distorted",
        )
        for prediction in predictions
    ]
    reference_frames, reference_fps, reference_timestamps = _load_compare_source_frames(
        db,
        workspace,
        Path(str(reference["path"])),
        f"campaign_item_{item_id}_gt",
    )
    reference_width, reference_height = decoded_dimensions(reference_frames, "GT")
    reference.update(
        {
            "slot": "gt",
            "frame_count": len(reference_frames),
            "width": reference_width,
            "height": reference_height,
            "fps": reference_fps if reference_fps is not None else reference.get("fps"),
            "timestamps": available_timestamps(reference_timestamps),
            "frame_paths": reference_frames,
        }
    )
    decoded: list[dict[str, Any]] = []
    for index, prediction in enumerate(resolved_predictions):
        _frames, fps, timestamps = _load_compare_source_frames(
            db,
            workspace,
            Path(str(prediction["path"])),
            f"campaign_item_{item_id}_pred_{index}",
        )
        decoded_width, decoded_height = decoded_dimensions(
            _frames,
            f"Pred {chr(ord('A') + index)}",
        )
        temporal_mapping = prediction.get("temporal_mapping") or {}
        mapping = prediction.get("source_frame_indices")
        if mapping is None and isinstance(temporal_mapping, dict):
            mapping = temporal_mapping.get("source_frame_indices")
        preserved_timestamps = None
        if isinstance(temporal_mapping, dict):
            preserved_timestamps = temporal_mapping.get("timestamps") or temporal_mapping.get(
                "source_timestamps"
            )
        source = dict(prediction)
        source.update(
            {
                "slot": f"pred_{chr(ord('a') + index)}",
                "frame_count": len(_frames),
                "width": decoded_width,
                "height": decoded_height,
                "fps": fps if fps is not None else prediction.get("fps"),
                "timestamps": (
                    preserved_timestamps
                    if mapping is not None
                    else available_timestamps(timestamps)
                ),
                "source_frame_indices": mapping,
                "frame_paths": _frames,
            }
        )
        decoded.append(source)
    temporal = validate_temporal_alignment(reference, decoded)
    plan = plan_alignment(
        reference,
        decoded,
        spatial_policy=spatial_policy,
        temporal_summary=temporal,
    )
    return plan, reference, decoded


def _preview_item_campaign_v2(
    db: Database,
    workspace: WorkspaceConfig,
    body: dict[str, Any],
) -> dict[str, Any]:
    raw_item_ids = body.get("media_item_ids")
    if not isinstance(raw_item_ids, list) or not raw_item_ids:
        raise ValueError("Campaign V2 requires a non-empty media_item_ids list")
    try:
        item_ids = list(dict.fromkeys(int(value) for value in raw_item_ids))
    except (TypeError, ValueError) as exc:
        raise ValueError("media_item_ids must contain positive integers") from exc
    if any(value <= 0 for value in item_ids):
        raise ValueError("media_item_ids must contain positive integers")
    items = [get_media_item(db, item_id) for item_id in item_ids]
    collection_ids = {int(item["collection_id"]) for item in items}
    if len(collection_ids) != 1:
        raise ValueError("one Campaign may select media items from only one GT group")
    methods = _normalize_item_method_specs(body)
    spatial_policy = dict(body.get("spatial_policy") or {})
    spatial_policy.setdefault("mode", "smallest_pred")
    spatial_policy.setdefault("filter", "lanczos")
    spatial_policy.setdefault("allow_known_aspect_stretch", True)

    method_matrix = list_methods_for_items(db, item_ids)
    rows: list[dict[str, Any]] = []
    for item in items:
        item_id = int(item["id"])
        selected_predictions = [
            _item_method_prediction(db, item_id, method) for method in methods
        ]
        reasons: list[str] = []
        for method, prediction in zip(methods, selected_predictions):
            if prediction is None:
                reasons.append(
                    f"{method.get('label') or method.get('run_id') or method.get('method_key')} 缺少该 Item"
                )
        plan: dict[str, Any] = {}
        if not reasons:
            try:
                plan, _reference, _predictions = _campaign_item_alignment(
                    db,
                    workspace,
                    item_id,
                    [prediction for prediction in selected_predictions if prediction is not None],
                    spatial_policy,
                )
            except Exception as exc:
                reasons.append(str(exc))
        method_payload: dict[str, Any] = {}
        for method, prediction in zip(methods, selected_predictions):
            slot = str(method["slot"])
            alignment = (plan.get("sources") or {}).get(f"pred_{slot}") or {}
            original = alignment.get("original") or {}
            method_payload[slot] = {
                "label": str(
                    method.get("label")
                    or (prediction or {}).get("run_name")
                    or method.get("method_key")
                    or f"Method {slot.upper()}"
                ),
                "member_id": int(prediction["id"]) if prediction else None,
                "asset_id": int(prediction["asset_id"]) if prediction else None,
                "run_id": prediction.get("producer_run_id") if prediction else method.get("run_id"),
                "frame_count": int((prediction or {}).get("frame_count") or 0),
                "width": int(original.get("width") or (prediction or {}).get("width") or 0),
                "height": int(original.get("height") or (prediction or {}).get("height") or 0),
                "fps": (prediction or {}).get("fps"),
                "alignment": alignment,
            }
        reference_alignment = (plan.get("sources") or {}).get("gt") or {}
        reference_original = reference_alignment.get("original") or {}
        row = {
            "media_item_id": item_id,
            "item_id": item_id,
            "video_name": str(item["display_name"]),
            "video_key": str(item["item_key"]),
            "collection_id": int(item["collection_id"]),
            "status": "ready" if not reasons else "missing_or_misaligned",
            "selectable": not reasons,
            "reasons": reasons,
            "reference_asset_id": int(item["canonical_gt_asset_id"]),
            "reference_member_id": int(resolve_item_reference(db, workspace, item_id)[1]["id"]),
            "reference": {
                "asset_id": int(item["canonical_gt_asset_id"]),
                "display_name": str(item["display_name"]),
                "frame_count": int(item.get("frame_count") or 0),
                "width": int(reference_original.get("width") or item.get("width") or 0),
                "height": int(reference_original.get("height") or item.get("height") or 0),
                "fps": item.get("fps"),
                "media_kind": str(item.get("media_kind") or "video"),
                "alignment": reference_alignment,
            },
            "methods": method_payload,
            "alignment_plan": plan,
            "alignment_fingerprint": plan.get("fingerprint"),
        }
        rows.append(row)

    normalized_methods: list[dict[str, Any]] = []
    for method in methods:
        run = db.get_run(int(method["run_id"])) if method.get("run_id") else None
        normalized_methods.append(
            {
                **method,
                "label": str(method.get("label") or (run or {}).get("name") or method.get("method_key") or ""),
                "run_name": str((run or {}).get("name") or ""),
                "model_name": str(((run or {}).get("metadata") or {}).get("model_file") or ""),
                "checkpoint": str(((run or {}).get("metadata") or {}).get("checkpoint") or ""),
                "track_label": str(method.get("method_key") or ""),
            }
        )
    ready = [row for row in rows if row["selectable"]]
    return {
        "schema_version": 2,
        "item_mode": True,
        "group_id": next(iter(collection_ids)),
        "methods": normalized_methods,
        "items": rows,
        "videos": rows,
        "coverage": method_matrix,
        "ready_media_item_ids": [row["media_item_id"] for row in ready],
        "ready_video_names": [row["video_name"] for row in ready],
        "task_count": len(ready),
        "spatial_policy": spatial_policy,
    }


def preview_campaign_v2(db: Database, workspace: WorkspaceConfig, body: dict[str, Any]) -> dict[str, Any]:
    ensure_v2_schema(db)
    if body.get("media_item_ids") is not None:
        return _preview_item_campaign_v2(db, workspace, body)
    methods = _normalize_method_specs(body)
    left = _method_output_rows(db, workspace, methods[0])
    right = _method_output_rows(db, workspace, methods[1])
    keys = sorted(set(left["outputs"]) | set(right["outputs"]))
    rows: list[dict[str, Any]] = []
    for key in keys:
        pred_a = left["outputs"].get(key)
        pred_b = right["outputs"].get(key)
        ref_a = left["references"].get(key)
        ref_b = right["references"].get(key)
        display_name = left["display_names"].get(key) or right["display_names"].get(key) or key
        status = "ready"
        reasons: list[str] = []
        alignments: dict[str, Any] = {}
        if pred_a is None:
            status, reasons = "missing", [f"{left['method']['label']} 缺少 Pred"]
        if pred_b is None:
            status = "missing"
            reasons.append(f"{right['method']['label']} 缺少 Pred")
        if ref_a is None or ref_b is None:
            status = "missing_gt"
            reasons.append("至少一个 Run 缺少该视频的规范 GT")
        elif not _same_reference(ref_a, ref_b):
            status = "gt_conflict"
            reasons.append("两个 Run 的 GT 内容不一致")
        if status == "ready":
            media_kinds = {
                str(ref_a.get("media_kind") or "video"),
                str(pred_a.get("media_kind") or "video"),
                str(pred_b.get("media_kind") or "video"),
            }
            if len(media_kinds) != 1:
                status = "alignment_mismatch"
                reasons.append("GT、Pred A、Pred B 的 media_kind 必须一致")
            else:
                state_a, alignment_a = _preview_pair_alignment(db, workspace, ref_a, pred_a)
                state_b, alignment_b = _preview_pair_alignment(db, workspace, ref_a, pred_b)
                alignments = {"a": alignment_a, "b": alignment_b}
                if state_a != "ready" or state_b != "ready":
                    status = "alignment_mismatch"
                    for slot, state, alignment in (
                        ("A", state_a, alignment_a),
                        ("B", state_b, alignment_b),
                    ):
                        if state != "ready":
                            reasons.append(f"方法 {slot}: {alignment['error']}")
        rows.append(
            {
                "video_name": display_name,
                "video_key": key,
                "status": status,
                "selectable": status == "ready",
                "media_kind": str(ref_a.get("media_kind") or "") if status == "ready" else None,
                "reasons": reasons,
                "reference_asset_id": int(ref_a["id"]) if ref_a else None,
                "reference": (
                    {
                        "asset_id": int(ref_a["id"]),
                        "display_name": str(ref_a.get("display_name") or display_name),
                        "frame_count": int(ref_a.get("frame_count") or 0),
                        "width": int(ref_a.get("width") or 0),
                        "height": int(ref_a.get("height") or 0),
                        "fps": ref_a.get("fps"),
                        "media_kind": str(ref_a.get("media_kind") or "video"),
                    }
                    if ref_a
                    else None
                ),
                "methods": {
                    "a": {
                        "label": str(left["method"]["label"]),
                        "asset_id": int(pred_a["id"]) if pred_a else None,
                        "frame_count": int(pred_a.get("frame_count") or 0) if pred_a else 0,
                        "width": int(pred_a.get("width") or 0) if pred_a else 0,
                        "height": int(pred_a.get("height") or 0) if pred_a else 0,
                        "fps": pred_a.get("fps") if pred_a else None,
                        "media_kind": str(pred_a.get("media_kind") or "") if pred_a else None,
                        "alignment": alignments.get("a"),
                    },
                    "b": {
                        "label": str(right["method"]["label"]),
                        "asset_id": int(pred_b["id"]) if pred_b else None,
                        "frame_count": int(pred_b.get("frame_count") or 0) if pred_b else 0,
                        "width": int(pred_b.get("width") or 0) if pred_b else 0,
                        "height": int(pred_b.get("height") or 0) if pred_b else 0,
                        "fps": pred_b.get("fps") if pred_b else None,
                        "media_kind": str(pred_b.get("media_kind") or "") if pred_b else None,
                        "alignment": alignments.get("b"),
                    },
                },
            }
        )
    ready = [row for row in rows if row["selectable"]]
    return {
        "schema_version": 2,
        "methods": [left["method"], right["method"]],
        "videos": rows,
        "ready_video_names": [row["video_name"] for row in ready],
        "task_count": len(ready),
    }


def _create_item_campaign_v2(
    db: Database,
    workspace: WorkspaceConfig,
    body: dict[str, Any],
) -> dict[str, Any]:
    preview = _preview_item_campaign_v2(db, workspace, body)
    name = str(body.get("name") or "").strip()
    if not name:
        raise ValueError("Campaign V2 internal name is required")
    title = str(body.get("public_title") or "").strip()
    if not title:
        raise ValueError("Campaign V2 public_title is required")
    target_votes = int(body.get("target_votes") or 3)
    if target_votes < 1 or target_votes > 1000:
        raise ValueError("target_votes must be between 1 and 1000")
    requested_ids = [int(value) for value in body.get("media_item_ids") or []]
    rows_by_id = {int(row["media_item_id"]): row for row in preview["items"]}
    selected_rows = [rows_by_id[item_id] for item_id in requested_ids if item_id in rows_by_id]
    if len(selected_rows) != len(set(requested_ids)):
        raise ValueError("one or more selected media items were not found")
    invalid = [row for row in selected_rows if not row["selectable"]]
    if invalid:
        details = "; ".join(
            f"{row['video_name']}: {', '.join(row['reasons'])}" for row in invalid
        )
        raise ValueError(
            "every selected media item must be covered and aligned; unselect items instead of skipping: "
            + details
        )
    now = utc_ts()
    seed = int(
        body.get("seed")
        if body.get("seed") is not None
        else random.SystemRandom().randrange(2**31)
    )
    config = dict(body.get("metadata") or {})
    config.update(
        {
            "item_mode": True,
            "group_id": int(preview["group_id"]),
            "media_item_ids": requested_ids,
            "spatial_policy": preview["spatial_policy"],
        }
    )
    with db.connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO evaluation_campaigns_v2(
                public_token, name, public_title, status, target_votes, seed,
                config_json, created_at, updated_at
            ) VALUES (?, ?, ?, 'draft', ?, ?, ?, ?, ?)
            """,
            (_token(), name[:240], title[:240], target_votes, seed, _json(config), now, now),
        )
        campaign_id = int(cur.lastrowid)
        method_ids: dict[str, int] = {}
        for method in preview["methods"]:
            method_cur = conn.execute(
                """
                INSERT INTO evaluation_methods_v2(
                    campaign_id, slot, source_kind, source_run_id, source_track_label,
                    label_snapshot, model_snapshot, checkpoint_snapshot,
                    source_spec_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    str(method["slot"]),
                    str(method["source_kind"]),
                    int(method["run_id"]) if method.get("run_id") is not None else None,
                    str(method.get("method_key") or ""),
                    str(method.get("label") or "")[:240],
                    str(method.get("model_name") or "")[:240],
                    str(method.get("checkpoint") or "")[:500],
                    _json(method.get("raw") or {}),
                    now,
                ),
            )
            method_ids[str(method["slot"])] = int(method_cur.lastrowid)
        for row in selected_rows:
            item_cur = conn.execute(
                """
                INSERT INTO evaluation_items_v2(
                    campaign_id, video_name, reference_source_asset_id,
                    alignment_json, created_at, media_item_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    str(row["video_name"])[:500],
                    int(row["reference_asset_id"]),
                    _json(row["alignment_plan"]),
                    now,
                    int(row["media_item_id"]),
                ),
            )
            evaluation_item_id = int(item_cur.lastrowid)
            for slot in ("a", "b"):
                method_row = row["methods"][slot]
                conn.execute(
                    """
                    INSERT INTO evaluation_bindings_v2(
                        item_id, method_id, source_asset_id, state,
                        alignment_json, created_at, updated_at, source_member_id
                    ) VALUES (?, ?, ?, 'selected', ?, ?, ?, ?)
                    """,
                    (
                        evaluation_item_id,
                        method_ids[slot],
                        int(method_row["asset_id"]),
                        _json(row["alignment_plan"]),
                        now,
                        now,
                        int(method_row["member_id"]),
                    ),
                )
    return get_campaign_v2(db, campaign_id)


def create_campaign_v2(db: Database, workspace: WorkspaceConfig, body: dict[str, Any]) -> dict[str, Any]:
    ensure_v2_schema(db)
    if body.get("media_item_ids") is not None:
        return _create_item_campaign_v2(db, workspace, body)
    preview = preview_campaign_v2(db, workspace, body)
    name = str(body.get("name") or "").strip()
    if not name:
        raise ValueError("Campaign V2 internal name is required")
    title = str(body.get("public_title") or "").strip()
    if not title:
        raise ValueError("Campaign V2 public_title is required")
    target_votes = int(body.get("target_votes") or 3)
    if target_votes < 1 or target_votes > 1000:
        raise ValueError("target_votes must be between 1 and 1000")
    selected_names = body.get("selected_videos") or body.get("video_names")
    if selected_names is None:
        selected_keys = {row["video_key"] for row in preview["videos"] if row["selectable"]}
    else:
        if not isinstance(selected_names, list):
            raise ValueError("selected_videos must be a list")
        selected_keys = {_video_key(str(value)) for value in selected_names}
    selected_rows = [row for row in preview["videos"] if row["video_key"] in selected_keys]
    if not selected_rows:
        raise ValueError("Campaign V2 requires at least one selected, aligned video")
    invalid = [row for row in selected_rows if not row["selectable"]]
    if invalid:
        details = "; ".join(f"{row['video_name']}: {', '.join(row['reasons'])}" for row in invalid)
        raise ValueError(f"selected Campaign V2 videos are not publishable: {details}")
    missing = selected_keys - {row["video_key"] for row in selected_rows}
    if missing:
        raise ValueError(f"selected Campaign V2 videos were not found: {', '.join(sorted(missing))}")
    now = utc_ts()
    seed = int(body.get("seed") if body.get("seed") is not None else random.SystemRandom().randrange(2**31))
    with db.connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO evaluation_campaigns_v2(
                public_token, name, public_title, status, target_votes, seed,
                config_json, created_at, updated_at
            ) VALUES (?, ?, ?, 'draft', ?, ?, ?, ?, ?)
            """,
            (
                _token(), name[:240], title[:240], target_votes, seed,
                _json(body.get("metadata") or {}), now, now,
            ),
        )
        campaign_id = int(cur.lastrowid)
        method_ids: dict[str, int] = {}
        for method in preview["methods"]:
            method_cur = conn.execute(
                """
                INSERT INTO evaluation_methods_v2(
                    campaign_id, slot, source_kind, source_run_id, source_track_label,
                    label_snapshot, model_snapshot, checkpoint_snapshot,
                    source_spec_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    method["slot"],
                    str(method["source_kind"]),
                    int(method["run_id"]) if method.get("run_id") is not None else None,
                    str(method["track_label"]),
                    str(method["label"])[:240],
                    str(method.get("model_name") or "")[:240],
                    str(method.get("checkpoint") or "")[:500],
                    _json(method.get("raw") or {}),
                    now,
                ),
            )
            method_ids[str(method["slot"])] = int(method_cur.lastrowid)
        for row in selected_rows:
            item_cur = conn.execute(
                """
                INSERT INTO evaluation_items_v2(
                    campaign_id, video_name, reference_source_asset_id,
                    alignment_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    str(row["video_name"])[:500],
                    int(row["reference_asset_id"]),
                    _json({"strict": True}),
                    now,
                ),
            )
            item_id = int(item_cur.lastrowid)
            for slot in ("a", "b"):
                conn.execute(
                    """
                    INSERT INTO evaluation_bindings_v2(
                        item_id, method_id, source_asset_id, state,
                        alignment_json, created_at, updated_at
                    ) VALUES (?, ?, ?, 'selected', ?, ?, ?)
                    """,
                    (
                        item_id,
                        method_ids[slot],
                        int(row["methods"][slot]["asset_id"]),
                        _json(row["methods"][slot].get("alignment") or {}),
                        now,
                        now,
                    ),
                )
    return get_campaign_v2(db, campaign_id)


def _decode_json_fields(row: dict[str, Any], names: Iterable[str]) -> dict[str, Any]:
    for name in names:
        row[name.removesuffix("_json")] = _loads(row.pop(name, None))
    return row


def get_campaign_v2(db: Database, campaign_id: int) -> dict[str, Any]:
    ensure_v2_schema(db)
    row = db.get("SELECT * FROM evaluation_campaigns_v2 WHERE id = ?", (int(campaign_id),))
    if row is None:
        raise KeyError(f"evaluation campaign V2 {campaign_id} not found")
    _decode_json_fields(row, ("config_json",))
    methods = db.query(
        "SELECT * FROM evaluation_methods_v2 WHERE campaign_id = ? ORDER BY slot",
        (int(campaign_id),),
    )
    for method in methods:
        _decode_json_fields(method, ("source_spec_json",))
    items = db.query(
        "SELECT * FROM evaluation_items_v2 WHERE campaign_id = ? ORDER BY video_name, id",
        (int(campaign_id),),
    )
    for item in items:
        _decode_json_fields(item, ("alignment_json",))
        bindings = db.query(
            """
            SELECT b.*, m.slot, m.label_snapshot
            FROM evaluation_bindings_v2 b
            JOIN evaluation_methods_v2 m ON m.id = b.method_id
            WHERE b.item_id = ? ORDER BY m.slot
            """,
            (int(item["id"]),),
        )
        for binding in bindings:
            _decode_json_fields(binding, ("alignment_json",))
        item["bindings"] = bindings
    counts = db.get(
        """
        SELECT (SELECT COUNT(*) FROM evaluation_tasks_v2 WHERE campaign_id = ?) AS tasks,
               (SELECT COUNT(*) FROM evaluation_votes_v2 v
                JOIN evaluation_tasks_v2 t ON t.id = v.task_id
                WHERE t.campaign_id = ?) AS votes
        """,
        (int(campaign_id), int(campaign_id)),
    ) or {}
    row.update(
        {
            "schema_version": 2,
            "campaign_key": f"v2:{int(campaign_id)}",
            "methods": methods,
            "items": items,
            "tasks": int(counts.get("tasks") or 0),
            "votes": int(counts.get("votes") or 0),
            "item_count": len(items),
            "task_count": int(counts.get("tasks") or 0),
            "vote_count": int(counts.get("votes") or 0),
            "share_token": str(row["public_token"]),
            "share_url": f"/evaluate/{row['public_token']}",
            "read_only": row["status"] in {"published", "closed", "archived"},
        }
    )
    return row


def list_campaigns_v2(db: Database) -> list[dict[str, Any]]:
    ensure_v2_schema(db)
    return [
        get_campaign_v2(db, int(row["id"]))
        for row in db.query("SELECT id FROM evaluation_campaigns_v2 ORDER BY id DESC")
    ]


def get_preparation_v2(db: Database, campaign_id: int) -> dict[str, Any] | None:
    ensure_v2_schema(db)
    row = db.get(
        "SELECT * FROM evaluation_preparations_v2 WHERE campaign_id = ?",
        (int(campaign_id),),
    )
    if row is None:
        return None
    return _decode_json_fields(row, ("report_json", "error_json"))


def _claim_direct_preparation(db: Database, campaign_id: int) -> str | None:
    """Start one owned preparation attempt for an explicit synchronous publish."""
    token = uuid.uuid4().hex
    now = utc_ts()
    with db.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        campaign = conn.execute(
            "SELECT status FROM evaluation_campaigns_v2 WHERE id = ?", (int(campaign_id),)
        ).fetchone()
        if campaign is None:
            raise KeyError("Campaign V2 not found")
        status = str(campaign["status"])
        if status == "published":
            return None
        if status not in {"draft", "failed", "preparing"}:
            raise ValueError("only a draft, failed, or queued Campaign V2 can be published")
        preparation = conn.execute(
            "SELECT state FROM evaluation_preparations_v2 WHERE campaign_id = ?",
            (int(campaign_id),),
        ).fetchone()
        if status == "preparing" and preparation is not None and preparation["state"] == "running":
            raise EvaluationConflict("Campaign V2 preparation is already claimed by another worker")
        conn.execute(
            "UPDATE evaluation_campaigns_v2 SET status = 'preparing', updated_at = ? WHERE id = ?",
            (now, int(campaign_id)),
        )
        if preparation is None:
            conn.execute(
                """
                INSERT INTO evaluation_preparations_v2(
                    campaign_id, state, attempt_count, claim_token, report_json,
                    error_json, created_at, started_at, updated_at
                ) VALUES (?, 'running', 1, ?, '{}', '{}', ?, ?, ?)
                """,
                (int(campaign_id), token, now, now, now),
            )
        else:
            conn.execute(
                """
                UPDATE evaluation_preparations_v2
                SET state = 'running', attempt_count = attempt_count + 1,
                    claim_token = ?, report_json = '{}', error_json = '{}',
                    started_at = ?, completed_at = NULL, updated_at = ?
                WHERE campaign_id = ?
                """,
                (token, now, now, int(campaign_id)),
            )
    return token


def _claim_pending_preparation(
    db: Database,
    campaign_id: int,
    *,
    stale_before: float,
) -> str | None:
    """Atomically take a queued or stale durable preparation claim."""
    token = uuid.uuid4().hex
    now = utc_ts()
    with db.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT p.state, p.updated_at, c.status
            FROM evaluation_preparations_v2 p
            JOIN evaluation_campaigns_v2 c ON c.id = p.campaign_id
            WHERE p.campaign_id = ?
            """,
            (int(campaign_id),),
        ).fetchone()
        if row is None or row["status"] != "preparing":
            return None
        if row["state"] != "queued" and not (
            row["state"] == "running" and float(row["updated_at"]) < float(stale_before)
        ):
            return None
        cursor = conn.execute(
            """
            UPDATE evaluation_preparations_v2
            SET state = 'running', attempt_count = attempt_count + 1,
                claim_token = ?, report_json = '{}', error_json = '{}',
                started_at = ?, completed_at = NULL, updated_at = ?
            WHERE campaign_id = ?
              AND (state = 'queued' OR (state = 'running' AND updated_at < ?))
            """,
            (token, now, now, int(campaign_id), float(stale_before)),
        )
        if cursor.rowcount != 1:
            return None
    return token


def _require_preparation_claim(
    db: Database,
    campaign_id: int,
    claim_token: str,
    *,
    ownership_lost: threading.Event | None = None,
) -> None:
    if ownership_lost is not None and ownership_lost.is_set():
        raise EvaluationConflict("Campaign V2 preparation claim was superseded")
    row = db.get(
        """
        SELECT p.id
        FROM evaluation_preparations_v2 p
        JOIN evaluation_campaigns_v2 c ON c.id = p.campaign_id
        WHERE p.campaign_id = ? AND p.state = 'running' AND p.claim_token = ?
          AND c.status = 'preparing'
        """,
        (int(campaign_id), str(claim_token)),
    )
    if row is None:
        raise EvaluationConflict("Campaign V2 preparation claim was superseded")


def _update_preparation_progress(
    db: Database,
    campaign_id: int,
    claim_token: str,
    report: dict[str, Any],
) -> None:
    with db.connection() as conn:
        cursor = conn.execute(
            """
            UPDATE evaluation_preparations_v2
            SET report_json = ?, updated_at = ?
            WHERE campaign_id = ? AND state = 'running' AND claim_token = ?
            """,
            (_json(report), utc_ts(), int(campaign_id), str(claim_token)),
        )
        if cursor.rowcount != 1:
            raise EvaluationConflict("Campaign V2 preparation claim was superseded")


@contextmanager
def _preparation_claim_heartbeat(
    db: Database,
    campaign_id: int,
    claim_token: str,
) -> Iterable[threading.Event]:
    """Renew an owned preparation claim while decoding/freezing may take time."""
    stop = threading.Event()
    lost = threading.Event()

    def renew() -> None:
        interval = max(1.0, min(30.0, PREPARATION_CLAIM_STALE_SECONDS / 4.0))
        while not stop.wait(interval):
            try:
                with db.connection() as conn:
                    cursor = conn.execute(
                        """
                        UPDATE evaluation_preparations_v2
                        SET updated_at = ?
                        WHERE campaign_id = ? AND state = 'running' AND claim_token = ?
                        """,
                        (utc_ts(), int(campaign_id), str(claim_token)),
                    )
                if cursor.rowcount != 1:
                    lost.set()
                    return
            except Exception:
                # SQLite may be transiently busy; another renewal happens well
                # before the stale-claim timeout.
                continue

    thread = threading.Thread(target=renew, name="vfieval-evaluation-preparation", daemon=True)
    thread.start()
    try:
        yield lost
    finally:
        stop.set()
        thread.join(timeout=2.0)


def request_publish_campaign_v2(db: Database, campaign_id: int) -> dict[str, Any]:
    """Persist a publish request for a background preparation runner."""
    campaign = get_campaign_v2(db, int(campaign_id))
    if campaign["status"] == "published":
        return {"campaign": campaign, "preparation": get_preparation_v2(db, int(campaign_id))}
    if campaign["status"] == "preparing":
        return {"campaign": campaign, "preparation": get_preparation_v2(db, int(campaign_id))}
    if campaign["status"] not in {"draft", "failed"}:
        raise ValueError("only a draft or failed Campaign V2 can be queued for publication")
    now = utc_ts()
    final_root = (db.db_path.parent / "evaluations" / str(int(campaign_id))).resolve()
    with db.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE evaluation_campaigns_v2 SET status = 'preparing', updated_at = ? WHERE id = ?",
            (now, int(campaign_id)),
        )
        conn.execute(
            """
            INSERT INTO evaluation_preparations_v2(
                campaign_id, state, attempt_count, staging_path, final_path, claim_token,
                report_json, error_json, created_at, updated_at
            ) VALUES (?, 'queued', 0, '', ?, '', '{}', '{}', ?, ?)
            ON CONFLICT(campaign_id) DO UPDATE SET
                state = 'queued', staging_path = '', final_path = excluded.final_path,
                claim_token = '', report_json = '{}', error_json = '{}', completed_at = NULL,
                updated_at = excluded.updated_at
            """,
            (int(campaign_id), str(final_root), now, now),
        )
    return {
        "campaign": get_campaign_v2(db, int(campaign_id)),
        "preparation": get_preparation_v2(db, int(campaign_id)),
    }


def run_pending_preparations(
    db: Database,
    workspace: WorkspaceConfig,
    *,
    limit: int = 4,
    stale_after_seconds: int = 600,
) -> list[dict[str, Any]]:
    """Claim queued (or restart-stale) preparations and finish them idempotently."""
    ensure_v2_schema(db)
    cutoff = utc_ts() - max(60, int(stale_after_seconds))
    candidates = db.query(
        """
        SELECT p.campaign_id
        FROM evaluation_preparations_v2 p
        JOIN evaluation_campaigns_v2 c ON c.id = p.campaign_id
        WHERE c.status = 'preparing'
          AND (p.state = 'queued' OR (p.state = 'running' AND p.updated_at < ?))
        ORDER BY p.updated_at, p.id
        LIMIT ?
        """,
        (cutoff, max(1, min(100, int(limit)))),
    )
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        campaign_id = int(candidate["campaign_id"])
        claim_token = _claim_pending_preparation(
            db,
            campaign_id,
            stale_before=cutoff,
        )
        if claim_token is None:
            continue
        try:
            campaign = publish_campaign_v2(
                db,
                workspace,
                campaign_id,
                claim_token=claim_token,
            )
            results.append({"campaign_id": campaign_id, "status": campaign["status"]})
        except Exception as exc:
            results.append(
                {
                    "campaign_id": campaign_id,
                    "status": "failed",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
    return results


def _managed_source_asset(
    db: Database,
    workspace: WorkspaceConfig,
    asset_id: int,
    role: str,
) -> tuple[dict[str, Any], Path]:
    asset, path = resolve_asset_path(db, workspace, int(asset_id), role=role)
    if asset["source_kind"] not in {"folder", "upload", "run_artifact"}:
        raise ValueError(f"unsupported Campaign V2 source kind: {asset['source_kind']}")
    return asset, path


def _deep_validate_pair(
    db: Database,
    workspace: WorkspaceConfig,
    reference_asset: dict[str, Any],
    reference_path: Path,
    distorted_asset: dict[str, Any],
    distorted_path: Path,
    cache_tag: str,
) -> dict[str, Any]:
    if str(reference_asset.get("media_kind")) != str(distorted_asset.get("media_kind")):
        raise ValueError("strict blind evaluation requires matching media_kind")
    reference_info = inspect_compare_path(workspace, reference_path)
    distorted_info = inspect_compare_path(workspace, distorted_path)
    # Stream inspection is authoritative when it can report fps.  Catalog
    # metadata is only a fallback for frame directories, which have no stream
    # fps of their own.  Letting catalog metadata overwrite the inspected fps
    # would allow stale but matching rows to hide a real source mismatch.
    if reference_info.get("fps") is None and reference_asset.get("fps") is not None:
        reference_info["fps"] = float(reference_asset["fps"])
    if distorted_info.get("fps") is None and distorted_asset.get("fps") is not None:
        distorted_info["fps"] = float(distorted_asset["fps"])
    alignment = validate_strict_alignment(reference_info, distorted_info)
    from vfieval.datasets import _load_compare_source_frames

    reference_frames, decoded_reference_fps, reference_timestamps = _load_compare_source_frames(
        db, workspace, reference_path, f"{cache_tag}_reference"
    )
    distorted_frames, decoded_distorted_fps, distorted_timestamps = _load_compare_source_frames(
        db, workspace, distorted_path, f"{cache_tag}_distorted"
    )
    decoded_reference_alignment_fps = (
        float(decoded_reference_fps)
        if decoded_reference_fps is not None
        else (
            float(reference_asset["fps"])
            if reference_asset.get("fps") is not None
            else None
        )
    )
    decoded_distorted_alignment_fps = (
        float(decoded_distorted_fps)
        if decoded_distorted_fps is not None
        else (
            float(distorted_asset["fps"])
            if distorted_asset.get("fps") is not None
            else None
        )
    )
    validate_strict_decoded_alignment(
        reference_frames,
        distorted_frames,
        decoded_reference_alignment_fps,
        decoded_distorted_alignment_fps,
        reference_timestamps,
        distorted_timestamps,
    )
    return alignment


def _clone_or_copy_file(source: Path, target: Path) -> None:
    """Create an immutable package snapshot without sharing a writable inode.

    A raw hard link is cheap but unsuitable here: an in-place edit to a source
    artifact would silently edit the supposedly frozen evaluation package too.
    On Linux filesystems that support copy-on-write reflinks, use one; otherwise
    make an ordinary private copy.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    cloned = False
    if os.name != "nt":
        try:
            import fcntl  # Linux-only optional acceleration.

            # FICLONE from linux/fs.h.  It creates a distinct inode with shared
            # copy-on-write extents, so later source writes cannot alter target.
            with source.open("rb") as source_handle, target.open("xb") as target_handle:
                fcntl.ioctl(target_handle.fileno(), 0x40049409, source_handle.fileno())
            shutil.copystat(source, target, follow_symlinks=False)
            cloned = True
        except (ImportError, OSError):
            try:
                target.unlink()
            except FileNotFoundError:
                pass
    if not cloned:
        shutil.copy2(source, target, follow_symlinks=False)


def _clone_managed_path(source: Path, target: Path) -> None:
    if source.is_symlink():
        raise ValueError("Campaign V2 refuses to freeze symlink media")
    if source.is_file():
        _clone_or_copy_file(source, target)
        return
    if not source.is_dir():
        raise FileNotFoundError(f"Campaign V2 source is unavailable: {source}")
    target.mkdir(parents=True, exist_ok=False)
    for child in sorted(source.rglob("*")):
        if child.is_symlink():
            raise ValueError("Campaign V2 refuses to freeze symlink media")
        relative = child.relative_to(source)
        destination = target / relative
        if child.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif child.is_file():
            _clone_or_copy_file(child, destination)


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(child for child in path.rglob("*") if child.is_file())
    for child in files:
        relative = child.name if path.is_file() else child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _frozen_target(directory: Path, role: str, source: Path) -> Path:
    suffix = source.suffix if source.is_file() else ""
    return directory / f"{role}{suffix}"


def _write_private_png_sequence(frame_paths: list[Path], target: Path) -> list[Path]:
    """Materialize private RGB PNG bytes, never links into a mutable cache/source."""
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Campaign V2 package target already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    outputs: list[Path] = []
    for index, source in enumerate(frame_paths):
        destination = target / f"{index:06d}.png"
        with Image.open(source) as image:
            image.convert("RGB").save(destination, format="PNG")
        outputs.append(destination)
    if not outputs:
        raise ValueError("Campaign V2 cannot freeze an empty frame set")
    return outputs


def _write_difference_sequence(
    reference_frames: list[Path],
    prediction_frames: list[Path],
    target: Path,
) -> list[Path]:
    if len(reference_frames) != len(prediction_frames) or not reference_frames:
        raise ValueError("Campaign V2 Diff requires equal, non-empty aligned frame sets")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Campaign V2 package target already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    outputs: list[Path] = []
    for index, (reference, prediction) in enumerate(zip(reference_frames, prediction_frames)):
        destination = target / f"{index:06d}.png"
        with Image.open(reference) as left, Image.open(prediction) as right:
            left_rgb = left.convert("RGB")
            right_rgb = right.convert("RGB")
            if left_rgb.size != right_rgb.size:
                raise ValueError("Campaign V2 Diff received spatially unaligned frames")
            ImageChops.difference(left_rgb, right_rgb).save(destination, format="PNG")
        outputs.append(destination)
    return outputs


def _write_package_media(
    workspace: WorkspaceConfig,
    frame_paths: list[Path],
    target: Path,
    *,
    media_kind: str,
    fps: float | None,
    expected_width: int,
    expected_height: int,
) -> Path:
    """Write one normalized package medium and verify its exact dimensions/count."""
    if media_kind == "frame_sequence":
        outputs = _write_private_png_sequence(frame_paths, target)
        with Image.open(outputs[0]) as image:
            actual_size = image.size
        if actual_size != (int(expected_width), int(expected_height)):
            raise ValueError(
                "Campaign V2 normalized frame dimensions changed while freezing: "
                f"expected {expected_width}x{expected_height}, got {actual_size[0]}x{actual_size[1]}"
            )
        return target
    if media_kind != "video":
        raise ValueError(f"unsupported Campaign V2 package media_kind: {media_kind}")

    temporary_frames = target.parent / f".{target.stem}-{uuid.uuid4().hex}-frames"
    try:
        private_frames = _write_private_png_sequence(frame_paths, temporary_frames)
        from vfieval.pipeline.inference import _write_mp4

        _write_mp4(private_frames, target, float(fps or 24.0))
    finally:
        if temporary_frames.exists():
            shutil.rmtree(temporary_frames)
    observed = inspect_compare_path(workspace, target)
    actual_size = (int(observed.get("width") or 0), int(observed.get("height") or 0))
    if actual_size != (int(expected_width), int(expected_height)):
        raise ValueError(
            "Campaign V2 normalized video dimensions changed while encoding: "
            f"expected {expected_width}x{expected_height}, got {actual_size[0]}x{actual_size[1]}"
        )
    if int(observed.get("frame_count") or 0) != len(frame_paths):
        raise ValueError(
            "Campaign V2 normalized video frame count changed while encoding: "
            f"expected {len(frame_paths)}, got {int(observed.get('frame_count') or 0)}"
        )
    return target


def _normalized_asset_snapshot(
    source_asset: dict[str, Any],
    *,
    media_kind: str,
    frame_count: int,
    width: int,
    height: int,
    fps: float | None,
) -> dict[str, Any]:
    snapshot = dict(source_asset)
    snapshot.update(
        {
            "media_kind": media_kind,
            "frame_count": int(frame_count),
            "width": int(width),
            "height": int(height),
            "fps": float(fps) if fps is not None else None,
            "mime_type": "video/mp4" if media_kind == "video" else "application/x-directory",
        }
    )
    return snapshot


def _stage_item_campaign_media(
    db: Database,
    workspace: WorkspaceConfig,
    campaign: dict[str, Any],
    item: dict[str, Any],
    video_dir: Path,
    staging: Path,
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    """Materialize one Media Item GT/A/B package from its immutable AlignmentPlan."""
    media_item_id = int(item.get("media_item_id") or 0)
    if media_item_id <= 0:
        raise ValueError("Campaign V2 item-mode row is missing media_item_id")
    semantic_item, reference_member, reference_asset, _reference_path = resolve_item_reference(
        db, workspace, media_item_id
    )
    if int(reference_asset["id"]) != int(item["reference_source_asset_id"]):
        raise ValueError("Campaign V2 canonical GT changed after the draft was created")

    by_slot = {str(binding["slot"]): binding for binding in item["bindings"]}
    if set(by_slot) != {"a", "b"}:
        raise ValueError("Campaign V2 item-mode publishing requires exactly Method A and B")
    members: dict[str, dict[str, Any]] = {}
    source_assets: dict[str, dict[str, Any]] = {}
    for slot in ("a", "b"):
        binding = by_slot[slot]
        source_member_id = int(binding.get("source_member_id") or 0)
        if source_member_id <= 0:
            raise ValueError(f"Campaign V2 Method {slot.upper()} is missing source_member_id")
        member_item, member, asset, _path = resolve_item_member(
            db,
            workspace,
            source_member_id,
            require_reusable=True,
        )
        if int(member_item["id"]) != media_item_id:
            raise ValueError("Campaign V2 source prediction no longer belongs to the selected Item")
        if int(asset["id"]) != int(binding["source_asset_id"]):
            raise ValueError("Campaign V2 source prediction changed after the draft was created")
        members[slot] = member
        source_assets[slot] = asset

    spatial_policy = dict((campaign.get("config") or {}).get("spatial_policy") or {})
    recomputed_plan, reference, decoded_predictions = _campaign_item_alignment(
        db,
        workspace,
        media_item_id,
        [members["a"], members["b"]],
        spatial_policy,
    )
    stored_plan = dict(item.get("alignment") or {})
    if not stored_plan.get("fingerprint"):
        raise ValueError("Campaign V2 item-mode draft has no AlignmentPlan fingerprint")
    if str(recomputed_plan["fingerprint"]) != str(stored_plan["fingerprint"]):
        raise ValueError(
            "Campaign V2 AlignmentPlan changed after the draft was created; "
            "create a new Campaign from a fresh preview"
        )
    for slot in ("a", "b"):
        binding_plan = dict(by_slot[slot].get("alignment") or {})
        if str(binding_plan.get("fingerprint") or "") != str(stored_plan["fingerprint"]):
            raise ValueError(f"Campaign V2 Method {slot.upper()} AlignmentPlan is stale")

    reference_frames = [Path(path) for path in reference.get("frame_paths") or []]
    prediction_frames = {
        slot: [Path(path) for path in decoded.get("frame_paths") or []]
        for slot, decoded in zip(("a", "b"), decoded_predictions)
    }
    mappings = [decoded.get("source_frame_indices") for decoded in decoded_predictions]
    selected_reference_indices = list(range(len(reference_frames)))
    if mappings[0] is not None:
        mapping = [int(value) for value in mappings[0]]
        selected_reference_indices = mapping
        reference_frames = [reference_frames[index] for index in mapping]
    sources = {
        "gt": reference_frames,
        "pred_a": prediction_frames["a"],
        "pred_b": prediction_frames["b"],
    }
    aligned = materialize_frame_sets(db, workspace, stored_plan, sources)
    target = stored_plan["target"]
    target_width = int(target["width"])
    target_height = int(target["height"])
    frame_count = int((stored_plan.get("temporal") or {}).get("frame_count") or 0)
    if frame_count <= 0:
        raise ValueError("Campaign V2 AlignmentPlan has no aligned frames")
    fps_value = (stored_plan.get("temporal") or {}).get("fps")
    fps = float(fps_value) if fps_value is not None else None
    media_kind = str(semantic_item.get("media_kind") or "video")
    suffix = ".mp4" if media_kind == "video" else ""
    video_dir.mkdir(parents=True, exist_ok=False)
    output_paths: dict[str, Path] = {}
    for slot, role in (("gt", "reference"), ("pred_a", "method-a"), ("pred_b", "method-b")):
        output_paths[slot] = _write_package_media(
            workspace,
            aligned[slot],
            video_dir / f"{role}{suffix}",
            media_kind=media_kind,
            fps=fps,
            expected_width=target_width,
            expected_height=target_height,
        )
    diff_paths: dict[str, Path] = {}
    for slot in ("a", "b"):
        temporary_diff = video_dir / f".diff-{slot}-{uuid.uuid4().hex}-frames"
        try:
            diff_frames = _write_difference_sequence(
                aligned["gt"], aligned[f"pred_{slot}"], temporary_diff
            )
            diff_paths[slot] = _write_package_media(
                workspace,
                diff_frames,
                video_dir / f"diff-{slot}{suffix}",
                media_kind=media_kind,
                fps=fps,
                expected_width=target_width,
                expected_height=target_height,
            )
        finally:
            if temporary_diff.exists():
                shutil.rmtree(temporary_diff)

    evaluation_item_id = int(item["id"])
    normalized_reference = _normalized_asset_snapshot(
        reference_asset,
        media_kind=media_kind,
        frame_count=frame_count,
        width=target_width,
        height=target_height,
        fps=fps,
    )
    frozen: dict[tuple[int, str], dict[str, Any]] = {
        (evaluation_item_id, "reference"): {
            "asset": normalized_reference,
            "path": output_paths["gt"],
            "digest": _path_sha256(output_paths["gt"]),
            "alignment": stored_plan,
            "source_member": reference_member,
            "media_item_id": media_item_id,
        }
    }
    methods_manifest: list[dict[str, Any]] = []
    for slot in ("a", "b"):
        binding = by_slot[slot]
        path = output_paths[f"pred_{slot}"]
        diff_path = diff_paths[slot]
        normalized_asset = _normalized_asset_snapshot(
            source_assets[slot],
            media_kind=media_kind,
            frame_count=frame_count,
            width=target_width,
            height=target_height,
            fps=fps,
        )
        frozen[(evaluation_item_id, slot)] = {
            "asset": normalized_asset,
            "path": path,
            "digest": _path_sha256(path),
            "alignment": stored_plan,
            "source_member": members[slot],
            "binding_id": int(binding["id"]),
            "media_item_id": media_item_id,
            "slot_report": dict((stored_plan.get("sources") or {}).get(f"pred_{slot}") or {}),
        }
        methods_manifest.append(
            {
                "slot": slot,
                "source_member_id": int(members[slot]["id"]),
                "path": path.relative_to(staging).as_posix(),
                "sha256": _path_sha256(path),
                "size_bytes": _path_size(path),
                "diff": {
                    "path": diff_path.relative_to(staging).as_posix(),
                    "sha256": _path_sha256(diff_path),
                    "size_bytes": _path_size(diff_path),
                },
                "temporal_mapping": dict(members[slot].get("temporal_mapping") or {}),
                "transform": dict((stored_plan.get("sources") or {}).get(f"pred_{slot}") or {}),
            }
        )
    reference_path = output_paths["gt"]
    manifest_item = {
        "item_id": evaluation_item_id,
        "media_item_id": media_item_id,
        "video_name": str(item["video_name"]),
        "media_kind": media_kind,
        "alignment_fingerprint": str(stored_plan["fingerprint"]),
        "alignment_plan": stored_plan,
        "temporal_materialization": {
            "source_frame_indices": selected_reference_indices,
            "frame_count": frame_count,
            "fps": fps,
        },
        "reference": {
            "source_member_id": int(reference_member["id"]),
            "path": reference_path.relative_to(staging).as_posix(),
            "sha256": _path_sha256(reference_path),
            "size_bytes": _path_size(reference_path),
            "transform": dict((stored_plan.get("sources") or {}).get("gt") or {}),
        },
        "methods": methods_manifest,
    }
    return frozen, manifest_item


def _register_frozen_member(
    conn: Any,
    *,
    media_item_id: int,
    asset_id: int,
    member_role: str,
    campaign_id: int,
    evaluation_item_id: int,
    slot: str,
    source_member: dict[str, Any],
    alignment_plan: dict[str, Any],
) -> int:
    if member_role not in {"evaluation_gt", "evaluation_pred"}:
        raise ValueError("Campaign package member role must be evaluation_gt or evaluation_pred")
    now = utc_ts()
    source_mapping = dict(source_member.get("temporal_mapping") or {})
    temporal = {
        **dict(alignment_plan.get("temporal") or {}),
        "source_frame_indices": source_mapping.get("source_frame_indices"),
        "alignment_fingerprint": alignment_plan.get("fingerprint"),
    }
    spatial = {
        **dict((alignment_plan.get("sources") or {}).get("gt" if slot == "reference" else f"pred_{slot}") or {}),
        "alignment_fingerprint": alignment_plan.get("fingerprint"),
    }
    metadata = {
        "immutable": True,
        "campaign_id": int(campaign_id),
        "evaluation_item_id": int(evaluation_item_id),
        "slot": str(slot),
        "source_member_id": int(source_member["id"]),
        "alignment_fingerprint": alignment_plan.get("fingerprint"),
    }
    method_key = "" if slot == "reference" else str(source_member.get("method_key") or "")
    conn.execute(
        """
        INSERT INTO media_item_members(
            item_id, asset_id, member_role, producer_kind, producer_run_id,
            method_key, reusable_as_pred, temporal_mapping_json,
            spatial_origin_json, state, metadata_json,
            created_at, updated_at, deleted_at
        ) VALUES (?, ?, ?, 'evaluation_package', NULL, ?, 0, ?, ?, 'ready', ?, ?, ?, NULL)
        ON CONFLICT(item_id, asset_id, member_role) DO UPDATE SET
            producer_kind = 'evaluation_package', producer_run_id = NULL,
            method_key = excluded.method_key, reusable_as_pred = 0,
            temporal_mapping_json = excluded.temporal_mapping_json,
            spatial_origin_json = excluded.spatial_origin_json,
            state = 'ready', metadata_json = excluded.metadata_json,
            updated_at = excluded.updated_at, deleted_at = NULL
        """,
        (
            int(media_item_id),
            int(asset_id),
            member_role,
            method_key,
            _json(temporal),
            _json(spatial),
            _json(metadata),
            now,
            now,
        ),
    )
    row = conn.execute(
        "SELECT id FROM media_item_members WHERE item_id = ? AND asset_id = ? AND member_role = ?",
        (int(media_item_id), int(asset_id), member_role),
    ).fetchone()
    if row is None:
        raise RuntimeError("Campaign V2 failed to register its frozen Media Item member")
    return int(row["id"])


def _register_frozen_asset(
    conn: Any,
    *,
    collection_id: int,
    campaign_id: int,
    item_id: int,
    slot: str,
    source_asset: dict[str, Any],
    path: Path,
    digest: str,
    display_name: str,
) -> int:
    source_key = f"evaluation_package:{campaign_id}:{item_id}:{slot}"
    role = "gt" if slot == "reference" else "pred"
    now = utc_ts()
    values = (
        collection_id,
        source_key,
        "evaluation_package",
        str(source_asset.get("media_kind") or "video"),
        role,
        display_name[:240],
        path.name,
        "ready",
        digest,
        _path_size(path),
        str(path.resolve()),
        str(source_asset.get("mime_type") or "application/octet-stream"),
        int(source_asset.get("frame_count") or 0),
        int(source_asset.get("width") or 0),
        int(source_asset.get("height") or 0),
        source_asset.get("fps"),
        _json(
            {
                "campaign_id": campaign_id,
                "item_id": item_id,
                "slot": slot,
                "source_asset_id": int(source_asset["id"]),
            }
        ),
        _json({"immutable": True, "evaluation_package": True}),
        now,
        now,
    )
    try:
        conn.execute(
            """
            INSERT INTO media_assets(
                collection_id, source_key, source_kind, media_kind, role,
                display_name, original_name, state, content_sha256, size_bytes,
                storage_path, mime_type, frame_count, width, height, fps,
                provenance_json, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET
                collection_id = excluded.collection_id,
                source_kind = excluded.source_kind,
                media_kind = excluded.media_kind,
                role = excluded.role,
                display_name = excluded.display_name,
                state = 'ready',
                content_sha256 = excluded.content_sha256,
                size_bytes = excluded.size_bytes,
                storage_path = excluded.storage_path,
                mime_type = excluded.mime_type,
                frame_count = excluded.frame_count,
                width = excluded.width,
                height = excluded.height,
                fps = excluded.fps,
                provenance_json = excluded.provenance_json,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at,
                deleted_at = NULL
            """,
            values,
        )
    except Exception as exc:
        if "CHECK constraint failed" in str(exc):
            raise RuntimeError(
                "media_assets schema must allow source_kind='evaluation_package' before publishing Campaign V2"
            ) from exc
        raise
    row = conn.execute("SELECT id FROM media_assets WHERE source_key = ?", (source_key,)).fetchone()
    assert row is not None
    return int(row["id"])


def _stage_legacy_campaign_media(
    db: Database,
    workspace: WorkspaceConfig,
    campaign_id: int,
    item: dict[str, Any],
    video_dir: Path,
    staging: Path,
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    """Retain the exact asset-mode freeze path for pre-Item Campaign drafts."""
    reference_asset, reference_path = _managed_source_asset(
        db, workspace, int(item["reference_source_asset_id"]), "reference"
    )
    reference_target = _frozen_target(video_dir, "reference", reference_path)
    _clone_managed_path(reference_path, reference_target)
    staged_bindings: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for binding in item["bindings"]:
        slot = str(binding["slot"])
        distorted_asset, distorted_path = _managed_source_asset(
            db, workspace, int(binding["source_asset_id"]), "distorted"
        )
        distorted_target = _frozen_target(video_dir, f"method-{slot}", distorted_path)
        _clone_managed_path(distorted_path, distorted_target)
        staged_bindings.append((binding, distorted_asset, distorted_target))

    item_id = int(item["id"])
    frozen: dict[tuple[int, str], dict[str, Any]] = {}
    bindings_manifest: list[dict[str, Any]] = []
    for binding, distorted_asset, distorted_target in staged_bindings:
        slot = str(binding["slot"])
        try:
            alignment = _deep_validate_pair(
                db,
                workspace,
                reference_asset,
                reference_target,
                distorted_asset,
                distorted_target,
                f"evaluation_v2_{campaign_id}_{item_id}_{slot}",
            )
        except Exception as exc:
            raise ValueError(f"{item['video_name']} / {binding['label_snapshot']}: {exc}") from exc
        digest = _path_sha256(distorted_target)
        frozen[(item_id, slot)] = {
            "asset": distorted_asset,
            "path": distorted_target,
            "digest": digest,
            "alignment": alignment,
            "binding_id": int(binding["id"]),
        }
        bindings_manifest.append(
            {
                "slot": slot,
                "path": distorted_target.relative_to(staging).as_posix(),
                "sha256": digest,
                "size_bytes": _path_size(distorted_target),
                "alignment": alignment,
            }
        )
    reference_digest = _path_sha256(reference_target)
    frozen[(item_id, "reference")] = {
        "asset": reference_asset,
        "path": reference_target,
        "digest": reference_digest,
    }
    manifest_item = {
        "item_id": item_id,
        "video_name": str(item["video_name"]),
        "reference": {
            "path": reference_target.relative_to(staging).as_posix(),
            "sha256": reference_digest,
            "size_bytes": _path_size(reference_target),
        },
        "methods": bindings_manifest,
    }
    return frozen, manifest_item


def publish_campaign_v2(
    db: Database,
    workspace: WorkspaceConfig,
    campaign_id: int,
    *,
    claim_token: str | None = None,
) -> dict[str, Any]:
    """Freeze and publish a two-method Campaign V2 under a durable claim.

    The package is copied into a private staging directory before *any* deep
    validation.  This makes the manifest a snapshot of the bytes that were
    actually validated, rather than a promise about source files that could
    change between validation and publication.
    """
    ensure_v2_schema(db)
    if claim_token is None:
        claim_token = _claim_direct_preparation(db, int(campaign_id))
        if claim_token is None:
            return get_campaign_v2(db, int(campaign_id))
    claim_token = str(claim_token)
    if not claim_token:
        raise ValueError("Campaign V2 publication requires a preparation claim")
    _require_preparation_claim(db, int(campaign_id), claim_token)
    campaign = get_campaign_v2(db, int(campaign_id))
    if campaign["status"] == "published":
        return campaign

    started = utc_ts()
    evaluations_root = workspace.evaluations_dir.resolve()
    staging_root = (evaluations_root / ".staging").resolve()
    final_root = (evaluations_root / str(int(campaign_id))).resolve()
    staging = (staging_root / f"{int(campaign_id)}-{uuid.uuid4().hex}").resolve()
    if staging.parent != staging_root or final_root.parent != evaluations_root:
        raise ValueError("invalid Campaign V2 evaluation package path")
    staging_root.mkdir(parents=True, exist_ok=True)
    item_total = len(campaign["items"])
    initial_report = {
        "phase": "validating_and_freezing",
        "current": 0,
        "total": item_total,
    }
    with db.connection() as conn:
        cursor = conn.execute(
            """
            UPDATE evaluation_preparations_v2
            SET staging_path = ?, final_path = ?, report_json = ?, error_json = '{}',
                updated_at = ?
            WHERE campaign_id = ? AND state = 'running' AND claim_token = ?
            """,
            (
                str(staging),
                str(final_root),
                _json(initial_report),
                started,
                int(campaign_id),
                claim_token,
            ),
        )
        if cursor.rowcount != 1:
            raise EvaluationConflict("Campaign V2 preparation claim was superseded")

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "campaign_id": int(campaign_id),
        "created_at": started,
        "items": [],
    }
    moved = False
    try:
        with _preparation_claim_heartbeat(db, int(campaign_id), claim_token) as ownership_lost:
            _require_preparation_claim(
                db, int(campaign_id), claim_token, ownership_lost=ownership_lost
            )
            staging.mkdir(parents=True, exist_ok=False)
            frozen: dict[tuple[int, str], dict[str, Any]] = {}
            for item_index, item in enumerate(campaign["items"], start=1):
                _require_preparation_claim(
                    db, int(campaign_id), claim_token, ownership_lost=ownership_lost
                )
                video_dir = staging / f"{int(item['id'])}-{slugify(str(item['video_name']))}"
                if item.get("media_item_id") is not None:
                    item_frozen, manifest_item = _stage_item_campaign_media(
                        db,
                        workspace,
                        campaign,
                        item,
                        video_dir,
                        staging,
                    )
                else:
                    item_frozen, manifest_item = _stage_legacy_campaign_media(
                        db,
                        workspace,
                        int(campaign_id),
                        item,
                        video_dir,
                        staging,
                    )
                frozen.update(item_frozen)
                manifest["items"].append(manifest_item)
                _update_preparation_progress(
                    db,
                    int(campaign_id),
                    claim_token,
                    {
                        "phase": "validating_and_freezing",
                        "current": item_index,
                        "total": item_total,
                    },
                )

            _require_preparation_claim(
                db, int(campaign_id), claim_token, ownership_lost=ownership_lost
            )
            manifest_path = staging / "manifest.json"
            manifest_path.write_text(_json(manifest), encoding="utf-8")

            # Final filesystem replacement and metadata publication are guarded
            # by one SQLite write transaction.  A stale worker can never move or
            # delete a package after a newer worker has replaced its claim.
            collection_slug = f"evaluation-package-{int(campaign_id)}"
            with db.connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                owner = conn.execute(
                    """
                    SELECT p.id
                    FROM evaluation_preparations_v2 p
                    JOIN evaluation_campaigns_v2 c ON c.id = p.campaign_id
                    WHERE p.campaign_id = ? AND p.state = 'running' AND p.claim_token = ?
                      AND c.status = 'preparing'
                    """,
                    (int(campaign_id), claim_token),
                ).fetchone()
                if owner is None:
                    raise EvaluationConflict("Campaign V2 preparation claim was superseded")
                if final_root.exists() or final_root.is_symlink():
                    if final_root.is_symlink():
                        raise ValueError("Campaign V2 refuses to replace a symlink package path")
                    shutil.rmtree(final_root)
                staging.replace(final_root)
                moved = True
                collection = conn.execute(
                    "SELECT id FROM media_collections WHERE slug = ?", (collection_slug,)
                ).fetchone()
                if collection is None:
                    collection_cur = conn.execute(
                        """
                        INSERT INTO media_collections(name, slug, metadata_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            f"Evaluation Package {int(campaign_id)}",
                            collection_slug,
                            _json(
                                {
                                    "source_kind": "evaluation_package",
                                    "campaign_id": int(campaign_id),
                                }
                            ),
                            started,
                            started,
                        ),
                    )
                    collection_id = int(collection_cur.lastrowid)
                else:
                    collection_id = int(collection["id"])
                for item in campaign["items"]:
                    item_id = int(item["id"])
                    item_mode = item.get("media_item_id") is not None
                    reference = frozen[(item_id, "reference")]
                    reference_path = final_root / reference["path"].relative_to(staging)
                    reference_id = _register_frozen_asset(
                        conn,
                        collection_id=collection_id,
                        campaign_id=int(campaign_id),
                        item_id=item_id,
                        slot="reference",
                        source_asset=reference["asset"],
                        path=reference_path,
                        digest=reference["digest"],
                        display_name=f"{item['video_name']} / GT",
                    )
                    if item_mode:
                        alignment_payload = dict(reference["alignment"])
                        _register_frozen_member(
                            conn,
                            media_item_id=int(item["media_item_id"]),
                            asset_id=reference_id,
                            member_role="evaluation_gt",
                            campaign_id=int(campaign_id),
                            evaluation_item_id=item_id,
                            slot="reference",
                            source_member=reference["source_member"],
                            alignment_plan=alignment_payload,
                        )
                    else:
                        alignment_payload = {
                            slot: frozen[(item_id, slot)]["alignment"] for slot in ("a", "b")
                        }
                    conn.execute(
                        """
                        UPDATE evaluation_items_v2
                        SET frozen_reference_asset_id = ?, alignment_json = ?
                        WHERE id = ?
                        """,
                        (reference_id, _json(alignment_payload), item_id),
                    )
                    binding_ids: dict[str, int] = {}
                    for slot in ("a", "b"):
                        payload = frozen[(item_id, slot)]
                        frozen_path = final_root / payload["path"].relative_to(staging)
                        asset_id = _register_frozen_asset(
                            conn,
                            collection_id=collection_id,
                            campaign_id=int(campaign_id),
                            item_id=item_id,
                            slot=slot,
                            source_asset=payload["asset"],
                            path=frozen_path,
                            digest=payload["digest"],
                            display_name=f"{item['video_name']} / Method {slot.upper()}",
                        )
                        if item_mode:
                            frozen_member_id = _register_frozen_member(
                                conn,
                                media_item_id=int(item["media_item_id"]),
                                asset_id=asset_id,
                                member_role="evaluation_pred",
                                campaign_id=int(campaign_id),
                                evaluation_item_id=item_id,
                                slot=slot,
                                source_member=payload["source_member"],
                                alignment_plan=payload["alignment"],
                            )
                            conn.execute(
                                """
                                UPDATE evaluation_bindings_v2
                                SET frozen_asset_id = ?, frozen_member_id = ?, state = 'ready',
                                    alignment_json = ?, updated_at = ?
                                WHERE id = ?
                                """,
                                (
                                    asset_id,
                                    frozen_member_id,
                                    _json(payload["alignment"]),
                                    utc_ts(),
                                    int(payload["binding_id"]),
                                ),
                            )
                        else:
                            conn.execute(
                                """
                                UPDATE evaluation_bindings_v2
                                SET frozen_asset_id = ?, state = 'ready', alignment_json = ?, updated_at = ?
                                WHERE id = ?
                                """,
                                (
                                    asset_id,
                                    _json(payload["alignment"]),
                                    utc_ts(),
                                    int(payload["binding_id"]),
                                ),
                            )
                        binding_ids[slot] = int(payload["binding_id"])
                    conn.execute(
                        """
                        INSERT INTO evaluation_tasks_v2(
                            task_token, campaign_id, item_id, binding_a_id,
                            binding_b_id, state, created_at
                        ) VALUES (?, ?, ?, ?, ?, 'ready', ?)
                        """,
                        (_token(), int(campaign_id), item_id, binding_ids["a"], binding_ids["b"], started),
                    )
                conn.execute(
                    "DELETE FROM evaluation_analysis_cache_v2 WHERE campaign_id = ?",
                    (int(campaign_id),),
                )
                completed = utc_ts()
                campaign_cursor = conn.execute(
                    """
                    UPDATE evaluation_campaigns_v2
                    SET status = 'published', published_at = ?, updated_at = ?
                    WHERE id = ? AND status = 'preparing'
                    """,
                    (completed, completed, int(campaign_id)),
                )
                if campaign_cursor.rowcount != 1:
                    raise EvaluationConflict("Campaign V2 preparation claim was superseded")
                preparation_cursor = conn.execute(
                    """
                    UPDATE evaluation_preparations_v2
                    SET state = 'completed', claim_token = '', report_json = ?, error_json = '{}',
                        completed_at = ?, updated_at = ?
                    WHERE campaign_id = ? AND state = 'running' AND claim_token = ?
                    """,
                    (
                        _json(
                            {
                                "phase": "completed",
                                "current": item_total,
                                "total": item_total,
                                "manifest_path": str(final_root / "manifest.json"),
                                "items": len(manifest["items"]),
                                "size_bytes": _path_size(final_root),
                            }
                        ),
                        completed,
                        completed,
                        int(campaign_id),
                        claim_token,
                    ),
                )
                if preparation_cursor.rowcount != 1:
                    raise EvaluationConflict("Campaign V2 preparation claim was superseded")
        return get_campaign_v2(db, int(campaign_id))
    except Exception as exc:
        try:
            if staging.exists() or staging.is_symlink():
                if staging.is_symlink():
                    staging.unlink()
                else:
                    shutil.rmtree(staging)
        except OSError:
            # The durable error below still gives the operator a retryable
            # report; a later request will use a fresh staging directory.
            pass
        failed = utc_ts()
        with db.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            owner = conn.execute(
                """
                SELECT p.id
                FROM evaluation_preparations_v2 p
                JOIN evaluation_campaigns_v2 c ON c.id = p.campaign_id
                WHERE p.campaign_id = ? AND p.state = 'running' AND p.claim_token = ?
                  AND c.status = 'preparing'
                """,
                (int(campaign_id), claim_token),
            ).fetchone()
            if owner is not None:
                if moved and (final_root.exists() or final_root.is_symlink()):
                    if final_root.is_symlink():
                        final_root.unlink()
                    else:
                        shutil.rmtree(final_root)
                conn.execute(
                    """
                    UPDATE evaluation_campaigns_v2
                    SET status = 'failed', updated_at = ?
                    WHERE id = ? AND status = 'preparing'
                    """,
                    (failed, int(campaign_id)),
                )
                conn.execute(
                    """
                    UPDATE evaluation_preparations_v2
                    SET state = 'failed', claim_token = '', error_json = ?,
                        completed_at = ?, updated_at = ?
                    WHERE campaign_id = ? AND state = 'running' AND claim_token = ?
                    """,
                    (
                        _json({"message": str(exc), "type": type(exc).__name__}),
                        failed,
                        failed,
                        int(campaign_id),
                        claim_token,
                    ),
                )
        raise


def close_campaign_v2(db: Database, campaign_id: int) -> dict[str, Any]:
    campaign = get_campaign_v2(db, int(campaign_id))
    if campaign["status"] == "closed":
        return campaign
    if campaign["status"] != "published":
        raise ValueError("only a published Campaign V2 can be closed")
    now = utc_ts()
    with db.connection() as conn:
        conn.execute(
            "UPDATE evaluation_campaigns_v2 SET status = 'closed', closed_at = ?, updated_at = ? WHERE id = ?",
            (now, now, int(campaign_id)),
        )
        conn.execute(
            "UPDATE evaluation_tasks_v2 SET state = 'closed' WHERE campaign_id = ?",
            (int(campaign_id),),
        )
    return get_campaign_v2(db, int(campaign_id))


def archive_campaign_v2(db: Database, campaign_id: int) -> dict[str, Any]:
    campaign = get_campaign_v2(db, int(campaign_id))
    if campaign["status"] == "archived":
        return campaign
    if campaign["status"] == "preparing":
        raise ValueError("a preparing Campaign V2 cannot be archived")
    now = utc_ts()
    with db.connection() as conn:
        conn.execute(
            "UPDATE evaluation_campaigns_v2 SET status = 'archived', archived_at = ?, updated_at = ? WHERE id = ?",
            (now, now, int(campaign_id)),
        )
        conn.execute(
            "UPDATE evaluation_tasks_v2 SET state = 'closed' WHERE campaign_id = ?",
            (int(campaign_id),),
        )
    return get_campaign_v2(db, int(campaign_id))


def _campaign_by_token(db: Database, token: str) -> dict[str, Any]:
    ensure_v2_schema(db)
    row = db.get("SELECT * FROM evaluation_campaigns_v2 WHERE public_token = ?", (str(token),))
    if row is None:
        raise KeyError("blind evaluation campaign not found")
    return _decode_json_fields(row, ("config_json",))


def _upsert_blind_evaluator(db: Database, evaluator_id: str, display_name: str) -> None:
    evaluator_id = str(evaluator_id or "").strip()
    display_name = str(display_name or "").strip()
    if not evaluator_id or len(evaluator_id) > 128:
        raise ValueError("a stable evaluator_id is required")
    if not display_name:
        raise ValueError("evaluator display_name is required")
    now = utc_ts()
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO evaluators(id, display_name, metadata_json, created_at, updated_at, last_seen_at)
            VALUES (?, ?, '{}', ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                display_name = excluded.display_name,
                updated_at = excluded.updated_at,
                last_seen_at = excluded.last_seen_at
            """,
            (evaluator_id, display_name[:120], now, now, now),
        )


def blind_session(
    db: Database,
    campaign_token: str,
    body: dict[str, Any],
    *,
    lease_seconds: int = 300,
) -> dict[str, Any]:
    campaign = _campaign_by_token(db, campaign_token)
    if campaign["status"] not in {"published", "closed"}:
        raise ValueError("blind evaluation campaign is not available")
    evaluator_id = str(body.get("evaluator_id") or body.get("browser_uuid") or "").strip()
    display_name = str(body.get("display_name") or "").strip()
    _upsert_blind_evaluator(db, evaluator_id, display_name)
    return blind_payload(db, campaign_token, evaluator_id, lease_seconds=lease_seconds)


def _progress(db: Database, campaign_id: int, evaluator_id: str) -> dict[str, Any]:
    rows = db.query(
        """
        SELECT t.id,
               EXISTS(SELECT 1 FROM evaluation_votes_v2 mine
                      WHERE mine.task_id = t.id AND mine.evaluator_id = ?) AS mine,
               (SELECT COUNT(*) FROM evaluation_votes_v2 all_votes
                WHERE all_votes.task_id = t.id) AS votes
        FROM evaluation_tasks_v2 t
        WHERE t.campaign_id = ?
        ORDER BY t.id
        """,
        (str(evaluator_id), int(campaign_id)),
    )
    campaign = db.get(
        "SELECT target_votes FROM evaluation_campaigns_v2 WHERE id = ?", (int(campaign_id),)
    ) or {"target_votes": 1}
    target = int(campaign["target_votes"])
    own = sum(1 for row in rows if int(row["mine"] or 0))
    remaining = sum(
        1 for row in rows if not int(row["mine"] or 0) and int(row["votes"] or 0) < target
    )
    return {
        "completed": own,
        "total": own + remaining,
        "campaign_tasks": len(rows),
        "remaining": remaining,
        "complete": remaining == 0,
    }


def _stable_swap(seed: int, task_token: str, evaluator_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{task_token}:{evaluator_id}".encode("utf-8")).digest()
    return int(bool(digest[0] & 1))


def _lease_assignment(
    db: Database,
    campaign: dict[str, Any],
    evaluator_id: str,
    lease_seconds: int,
) -> dict[str, Any] | None:
    now = utc_ts()
    expires = now + max(30, min(3600, int(lease_seconds)))
    with db.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            UPDATE evaluation_assignments_v2
            SET state = 'expired', updated_at = ?
            WHERE state = 'leased' AND lease_expires_at <= ?
            """,
            (now, now),
        )
        existing = conn.execute(
            """
            SELECT a.*, t.task_token FROM evaluation_assignments_v2 a
            JOIN evaluation_tasks_v2 t ON t.id = a.task_id
            WHERE t.campaign_id = ? AND a.evaluator_id = ? AND a.state = 'leased'
              AND a.lease_expires_at > ?
            ORDER BY a.id LIMIT 1
            """,
            (int(campaign["id"]), str(evaluator_id), now),
        ).fetchone()
        if existing is not None:
            conn.execute(
                "UPDATE evaluation_assignments_v2 SET lease_expires_at = ?, updated_at = ? WHERE id = ?",
                (expires, now, int(existing["id"])),
            )
            return {**dict(existing), "lease_expires_at": expires}
        task = conn.execute(
            """
            SELECT t.id, t.task_token
            FROM evaluation_tasks_v2 t
            WHERE t.campaign_id = ? AND t.state = 'ready'
              AND NOT EXISTS (
                  SELECT 1 FROM evaluation_votes_v2 mine
                  WHERE mine.task_id = t.id AND mine.evaluator_id = ?
              )
              AND (
                  SELECT COUNT(*) FROM evaluation_votes_v2 votes WHERE votes.task_id = t.id
              ) < ?
              AND (
                  (SELECT COUNT(*) FROM evaluation_votes_v2 votes WHERE votes.task_id = t.id)
                  +
                  (SELECT COUNT(*) FROM evaluation_assignments_v2 leases
                   WHERE leases.task_id = t.id AND leases.state = 'leased'
                     AND leases.lease_expires_at > ?)
              ) < ?
            ORDER BY
              (SELECT COUNT(*) FROM evaluation_votes_v2 votes WHERE votes.task_id = t.id),
              t.id
            LIMIT 1
            """,
            (
                int(campaign["id"]),
                str(evaluator_id),
                int(campaign["target_votes"]),
                now,
                int(campaign["target_votes"]),
            ),
        ).fetchone()
        if task is None:
            return None
        swap = _stable_swap(int(campaign["seed"]), str(task["task_token"]), str(evaluator_id))
        old = conn.execute(
            "SELECT id, assignment_token FROM evaluation_assignments_v2 WHERE task_id = ? AND evaluator_id = ?",
            (int(task["id"]), str(evaluator_id)),
        ).fetchone()
        if old is None:
            assignment_token = _token()
            cur = conn.execute(
                """
                INSERT INTO evaluation_assignments_v2(
                    assignment_token, task_id, evaluator_id, state, side_swap,
                    lease_expires_at, created_at, updated_at
                ) VALUES (?, ?, ?, 'leased', ?, ?, ?, ?)
                """,
                (assignment_token, int(task["id"]), str(evaluator_id), swap, expires, now, now),
            )
            assignment_id = int(cur.lastrowid)
        else:
            assignment_id = int(old["id"])
            assignment_token = str(old["assignment_token"])
            conn.execute(
                """
                UPDATE evaluation_assignments_v2
                SET state = 'leased', side_swap = ?, lease_expires_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (swap, expires, now, assignment_id),
            )
        return {
            "id": assignment_id,
            "assignment_token": assignment_token,
            "task_id": int(task["id"]),
            "task_token": str(task["task_token"]),
            "evaluator_id": str(evaluator_id),
            "state": "leased",
            "side_swap": swap,
            "lease_expires_at": expires,
        }


def _task_payload(
    db: Database,
    campaign_token: str,
    assignment: dict[str, Any],
) -> dict[str, Any]:
    row = db.get(
        """
        SELECT t.task_token, i.video_name, i.frozen_reference_asset_id,
               ba.frozen_asset_id AS asset_a_id, bb.frozen_asset_id AS asset_b_id,
               gra.media_kind AS reference_media_kind,
               gaa.media_kind AS media_a_kind, gba.media_kind AS media_b_kind,
               MIN(gra.frame_count, gaa.frame_count, gba.frame_count) AS frame_count
        FROM evaluation_tasks_v2 t
        JOIN evaluation_items_v2 i ON i.id = t.item_id
        JOIN evaluation_bindings_v2 ba ON ba.id = t.binding_a_id
        JOIN evaluation_bindings_v2 bb ON bb.id = t.binding_b_id
        JOIN media_assets gra ON gra.id = i.frozen_reference_asset_id
        JOIN media_assets gaa ON gaa.id = ba.frozen_asset_id
        JOIN media_assets gba ON gba.id = bb.frozen_asset_id
        WHERE t.id = ?
        """,
        (int(assignment["task_id"]),),
    )
    if row is None:
        raise KeyError("blind evaluation task not found")
    task_token = str(row["task_token"])
    assignment_query = quote(str(assignment["assignment_token"]), safe="")
    base = f"/api/blind/{campaign_token}/tasks/{task_token}/media"
    swap = bool(int(assignment["side_swap"] or 0))
    return {
        "token": task_token,
        "video_name": str(row["video_name"]),
        "reference_url": f"{base}/reference?assignment={assignment_query}",
        "left_url": f"{base}/left?assignment={assignment_query}",
        "right_url": f"{base}/right?assignment={assignment_query}",
        "reference_media_kind": str(row["reference_media_kind"]),
        "left_media_kind": str(row["media_b_kind"] if swap else row["media_a_kind"]),
        "right_media_kind": str(row["media_a_kind"] if swap else row["media_b_kind"]),
        "frame_count": int(row["frame_count"] or 0),
        "quality_reasons": sorted(QUALITY_REASONS),
        "lease_expires_at": float(assignment["lease_expires_at"]),
    }


def _participant_results(analysis: dict[str, Any]) -> dict[str, Any]:
    def clean(section: dict[str, Any]) -> dict[str, Any]:
        return {
            "vote_count": int(section.get("vote_count") or 0),
            "ranking": [
                {"label": row["label"], "score": row["score"], "ci95": row["ci95"]}
                for row in section.get("ranking") or []
            ],
        }

    return {
        "coverage": analysis["coverage"],
        "human": clean(analysis["human"]),
        "by_video": {video: clean(payload) for video, payload in analysis["by_video"].items()},
    }


def blind_payload(
    db: Database,
    campaign_token: str,
    evaluator_id: str = "",
    *,
    lease_seconds: int = 300,
) -> dict[str, Any]:
    campaign = _campaign_by_token(db, campaign_token)
    if not str(evaluator_id or "").strip():
        task_count = db.get(
            "SELECT COUNT(*) AS count FROM evaluation_tasks_v2 WHERE campaign_id = ?",
            (int(campaign["id"]),),
        ) or {"count": 0}
        return {
            "campaign": {
                "title": str(campaign["public_title"]),
                "status": str(campaign["status"]),
            },
            "progress": {
                "completed": 0,
                "total": int(task_count["count"] or 0),
                "campaign_tasks": int(task_count["count"] or 0),
                "remaining": int(task_count["count"] or 0),
                "complete": False,
                "waiting": False,
            },
            "task": None,
        }
    evaluator = db.get("SELECT id, display_name FROM evaluators WHERE id = ?", (str(evaluator_id),))
    if evaluator is None:
        raise KeyError("blind evaluator session not found")
    assignment = None
    if campaign["status"] == "published":
        assignment = _lease_assignment(db, campaign, str(evaluator_id), lease_seconds)
    progress = _progress(db, int(campaign["id"]), str(evaluator_id))
    if campaign["status"] == "closed":
        progress.update(
            {
                "total": int(progress["completed"]),
                "remaining": 0,
                "complete": True,
            }
        )
    task = _task_payload(db, campaign_token, assignment) if assignment else None
    waiting = bool(progress["remaining"] and task is None and campaign["status"] == "published")
    response: dict[str, Any] = {
        "campaign": {
            "title": str(campaign["public_title"]),
            "status": str(campaign["status"]),
        },
        "evaluator": {"display_name": str(evaluator["display_name"])},
        "progress": {**progress, "waiting": waiting},
        "task": task,
    }
    if progress["complete"]:
        response["results"] = _participant_results(
            campaign_analysis_v2(db, int(campaign["id"]), bootstrap_samples=200)
        )
    return response


def blind_public_payload(db: Database, campaign_token: str) -> dict[str, Any]:
    """Public campaign intro with no evaluator, method, Run, task, or asset identity."""
    return blind_payload(db, campaign_token, "")


def blind_media_asset(
    db: Database,
    workspace: WorkspaceConfig,
    campaign_token: str,
    task_token: str,
    side: str,
    assignment_token: str,
) -> tuple[dict[str, Any], Path]:
    campaign = _campaign_by_token(db, campaign_token)
    row = db.get(
        """
        SELECT a.side_swap, a.state, a.lease_expires_at,
               i.frozen_reference_asset_id,
               ba.frozen_asset_id AS asset_a_id,
               bb.frozen_asset_id AS asset_b_id
        FROM evaluation_tasks_v2 t
        JOIN evaluation_assignments_v2 a ON a.task_id = t.id
        JOIN evaluation_items_v2 i ON i.id = t.item_id
        JOIN evaluation_bindings_v2 ba ON ba.id = t.binding_a_id
        JOIN evaluation_bindings_v2 bb ON bb.id = t.binding_b_id
        WHERE t.campaign_id = ? AND t.task_token = ? AND a.assignment_token = ?
        """,
        (int(campaign["id"]), str(task_token), str(assignment_token)),
    )
    if row is None:
        raise KeyError("blind media assignment not found")
    if row["state"] == "expired" or (
        row["state"] == "leased" and float(row["lease_expires_at"]) <= utc_ts()
    ):
        raise EvaluationConflict("blind media assignment lease expired")
    swap = bool(int(row["side_swap"] or 0))
    mapping = {
        "reference": int(row["frozen_reference_asset_id"]),
        "left": int(row["asset_b_id"] if swap else row["asset_a_id"]),
        "right": int(row["asset_a_id"] if swap else row["asset_b_id"]),
    }
    if side not in mapping:
        raise ValueError("blind media side must be reference, left, or right")
    asset = get_asset(db, mapping[side])
    path = Path(str(asset["storage_path"])).resolve()
    evaluations_root = workspace.evaluations_dir.resolve()
    try:
        path.relative_to(evaluations_root)
    except ValueError as exc:
        raise ValueError("blind media resolved outside immutable evaluation packages") from exc
    if not path.exists() or asset["state"] != "ready":
        raise FileNotFoundError("blind evaluation media is unavailable")
    return asset, path


def blind_heartbeat(
    db: Database,
    campaign_token: str,
    task_token: str,
    evaluator_id: str,
    *,
    lease_seconds: int = 300,
) -> dict[str, Any]:
    """Renew the evaluator's active assignment without revealing task identity."""
    campaign = _campaign_by_token(db, campaign_token)
    if campaign["status"] != "published":
        raise ValueError("blind evaluation campaign is not accepting lease renewals")
    now = utc_ts()
    expires = now + max(30, min(3600, int(lease_seconds)))
    with db.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        assignment = conn.execute(
            """
            SELECT a.id, a.state, a.lease_expires_at
            FROM evaluation_assignments_v2 a
            JOIN evaluation_tasks_v2 t ON t.id = a.task_id
            WHERE t.campaign_id = ? AND t.task_token = ? AND a.evaluator_id = ?
            """,
            (int(campaign["id"]), str(task_token), str(evaluator_id)),
        ).fetchone()
        if assignment is None:
            raise EvaluationConflict("blind evaluation task is not assigned to this evaluator")
        if assignment["state"] != "leased" or float(assignment["lease_expires_at"]) <= now:
            raise EvaluationConflict("blind evaluation assignment lease expired")
        conn.execute(
            "UPDATE evaluation_assignments_v2 SET lease_expires_at = ?, updated_at = ? WHERE id = ?",
            (expires, now, int(assignment["id"])),
        )
    return {"ok": True, "lease_expires_at": expires}


def blind_submit_vote(
    db: Database,
    campaign_token: str,
    task_token: str,
    evaluator_id: str,
    body: dict[str, Any],
    *,
    lease_seconds: int = 300,
) -> dict[str, Any]:
    campaign = _campaign_by_token(db, campaign_token)
    if campaign["status"] != "published":
        raise ValueError("blind evaluation campaign is not accepting votes")
    choice = str(body.get("choice") or "").strip()
    if choice not in {"left", "right", "tie"}:
        raise ValueError("vote choice must be left, right, or tie")
    reasons = list(dict.fromkeys(str(value) for value in (body.get("reasons") or [])))
    if any(reason not in QUALITY_REASONS for reason in reasons):
        raise ValueError("vote contains an unsupported quality reason")
    confidence = str(body.get("confidence") or "").strip()
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError("confidence must be low, medium, high, or blank")
    note = str(body.get("note") or "").strip()[:4000]
    duration_raw = body.get("duration_ms")
    duration = max(0, int(duration_raw)) if duration_raw not in {None, ""} else None
    now = utc_ts()
    with db.connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        assignment = conn.execute(
            """
            SELECT a.*, t.id AS task_id, t.binding_a_id, t.binding_b_id,
                   ba.method_id AS method_a_id, bb.method_id AS method_b_id
            FROM evaluation_tasks_v2 t
            JOIN evaluation_assignments_v2 a ON a.task_id = t.id
            JOIN evaluation_bindings_v2 ba ON ba.id = t.binding_a_id
            JOIN evaluation_bindings_v2 bb ON bb.id = t.binding_b_id
            WHERE t.campaign_id = ? AND t.task_token = ? AND a.evaluator_id = ?
            """,
            (int(campaign["id"]), str(task_token), str(evaluator_id)),
        ).fetchone()
        if assignment is None:
            raise EvaluationConflict("blind evaluation task is not assigned to this evaluator")
        existing = conn.execute(
            "SELECT id FROM evaluation_votes_v2 WHERE task_id = ? AND evaluator_id = ?",
            (int(assignment["task_id"]), str(evaluator_id)),
        ).fetchone()
        if existing is None:
            if assignment["state"] != "leased" or float(assignment["lease_expires_at"]) <= now:
                raise EvaluationConflict("blind evaluation assignment lease expired")
            vote_count = int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM evaluation_votes_v2 WHERE task_id = ?",
                    (int(assignment["task_id"]),),
                ).fetchone()["count"]
            )
            if vote_count >= int(campaign["target_votes"]):
                conn.execute(
                    "UPDATE evaluation_assignments_v2 SET state = 'expired', updated_at = ? WHERE id = ?",
                    (now, int(assignment["id"])),
                )
                raise EvaluationConflict("blind evaluation task already reached its target vote count")
        swap = bool(int(assignment["side_swap"] or 0))
        left_method = int(assignment["method_b_id"] if swap else assignment["method_a_id"])
        right_method = int(assignment["method_a_id"] if swap else assignment["method_b_id"])
        preferred_method = left_method if choice == "left" else right_method if choice == "right" else None
        presentation = {"swapped": swap, "choice": choice}
        conn.execute(
            """
            INSERT INTO evaluation_votes_v2(
                task_id, evaluator_id, assignment_id, choice, preferred_method_id,
                reasons_json, confidence, note, duration_ms, presentation_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id, evaluator_id) DO UPDATE SET
                choice = excluded.choice,
                preferred_method_id = excluded.preferred_method_id,
                reasons_json = excluded.reasons_json,
                confidence = excluded.confidence,
                note = excluded.note,
                duration_ms = excluded.duration_ms,
                presentation_json = excluded.presentation_json,
                updated_at = excluded.updated_at
            """,
            (
                int(assignment["task_id"]),
                str(evaluator_id),
                int(assignment["id"]),
                choice,
                preferred_method,
                _json(reasons),
                confidence,
                note,
                duration,
                _json(presentation),
                now,
                now,
            ),
        )
        conn.execute(
            "UPDATE evaluation_assignments_v2 SET state = 'voted', updated_at = ? WHERE id = ?",
            (now, int(assignment["id"])),
        )
        conn.execute(
            "UPDATE evaluation_campaigns_v2 SET vote_revision = vote_revision + 1, updated_at = ? WHERE id = ?",
            (now, int(campaign["id"])),
        )
    payload = blind_payload(
        db, campaign_token, str(evaluator_id), lease_seconds=lease_seconds
    )
    return {
        "ok": True,
        "progress": payload["progress"],
        "next_task": payload["task"],
        **({"results": payload["results"]} if "results" in payload else {}),
    }


def _bradley_terry(method_ids: list[int], observations: list[dict[str, Any]]) -> dict[int, float]:
    if not method_ids:
        return {}
    ability = {method_id: 1.0 for method_id in method_ids}
    wins = {method_id: 1e-9 for method_id in method_ids}
    comparisons: Counter[tuple[int, int]] = Counter()
    for row in observations:
        a, b = int(row["a"]), int(row["b"])
        wins[a] += float(row["score_a"])
        wins[b] += float(row["score_b"])
        comparisons[(min(a, b), max(a, b))] += 1
    for _ in range(200):
        updated: dict[int, float] = {}
        for method_id in method_ids:
            denominator = 0.0
            for other in method_ids:
                if method_id == other:
                    continue
                count = comparisons[(min(method_id, other), max(method_id, other))]
                if count:
                    denominator += count / max(ability[method_id] + ability[other], 1e-12)
            updated[method_id] = wins[method_id] / denominator if denominator else ability[method_id]
        geometric = math.exp(
            sum(math.log(max(value, 1e-12)) for value in updated.values()) / len(updated)
        )
        updated = {key: max(value / geometric, 1e-12) for key, value in updated.items()}
        delta = max(abs(math.log(updated[key]) - math.log(ability[key])) for key in method_ids)
        ability = updated
        if delta < 1e-10:
            break
    total = sum(ability.values()) or 1.0
    return {method_id: ability[method_id] / total for method_id in method_ids}


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    position = probability * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def _rank_methods(
    method_ids: list[int],
    methods: dict[int, dict[str, Any]],
    observations: list[dict[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    scores = _bradley_terry(method_ids, observations)
    intervals = {method_id: (scores[method_id], scores[method_id]) for method_id in method_ids}
    if observations and bootstrap_samples:
        rng = random.Random(seed)
        samples: dict[int, list[float]] = {method_id: [] for method_id in method_ids}
        for _ in range(bootstrap_samples):
            resampled = [observations[rng.randrange(len(observations))] for _row in observations]
            result = _bradley_terry(method_ids, resampled)
            for method_id in method_ids:
                samples[method_id].append(result[method_id])
        for method_id, values in samples.items():
            values.sort()
            intervals[method_id] = (_percentile(values, 0.025), _percentile(values, 0.975))
    ranking = []
    for method_id in method_ids:
        method = methods[method_id]
        low, high = intervals[method_id]
        ranking.append(
            {
                "method_id": method_id,
                "slot": method["slot"],
                "label": method["label_snapshot"],
                "model_name": method["model_snapshot"],
                "checkpoint": method["checkpoint_snapshot"],
                "score": round(scores[method_id], 6),
                "ci95": [round(low, 6), round(high, 6)],
            }
        )
    ranking.sort(key=lambda value: (-float(value["score"]), str(value["label"])))
    pair = {"votes": len(observations), "wins_a": 0.0, "wins_b": 0.0, "ties": 0}
    for row in observations:
        method_a = methods[int(row["a"])]
        score_a = float(row["score_a"])
        if method_a["slot"] == "a":
            pair["wins_a"] += score_a
            pair["wins_b"] += 1.0 - score_a
        else:
            pair["wins_a"] += 1.0 - score_a
            pair["wins_b"] += score_a
        if score_a == 0.5:
            pair["ties"] += 1
    return {"vote_count": len(observations), "ranking": ranking, "head_to_head": pair}


def _objective_by_method(
    db: Database,
    bindings: list[dict[str, Any]],
    methods: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    asset_to_binding = {
        int(row["source_asset_id"]): (int(row["method_id"]), str(row.get("video_name") or ""))
        for row in bindings
    }
    if not asset_to_binding:
        return {"metrics": [], "by_video": {}}
    placeholders = ",".join("?" for _ in asset_to_binding)
    rows = db.query(
        f"""
        SELECT mr.metric_name, mr.status, mr.value, mab.distorted_asset_id
        FROM metric_asset_bindings mab
        JOIN metric_results mr ON mr.id = mab.metric_result_id
        WHERE mab.distorted_asset_id IN ({placeholders})
        """,
        tuple(sorted(asset_to_binding)),
    )
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    statuses: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    video_grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    video_statuses: dict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
    for row in rows:
        method_id, video_name = asset_to_binding[int(row["distorted_asset_id"])]
        key = (str(row["metric_name"]), method_id)
        video_key = (video_name, str(row["metric_name"]), method_id)
        statuses[key][str(row["status"])] += 1
        video_statuses[video_key][str(row["status"])] += 1
        if row["status"] == "completed" and row.get("value") is not None:
            grouped[key].append(float(row["value"]))
            video_grouped[video_key].append(float(row["value"]))
    def summary(
        name: str,
        method_id: int,
        values: list[float],
        counts: Counter[str],
    ) -> dict[str, Any]:
        values = sorted(values)
        return {
            "metric_name": name,
            "direction": METRIC_DIRECTIONS.get(name, "lower_is_better"),
            "method_id": method_id,
            "method_label": methods[method_id]["label_snapshot"],
            "status_counts": dict(counts),
            "count": len(values),
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p10": _percentile(values, 0.1) if values else None,
            "p90": _percentile(values, 0.9) if values else None,
        }
    metrics = []
    for name, method_id in sorted(set(grouped) | set(statuses)):
        metrics.append(
            summary(name, method_id, grouped.get((name, method_id), []), statuses[(name, method_id)])
        )
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for video_name, name, method_id in sorted(set(video_grouped) | set(video_statuses)):
        by_video[video_name].append(
            summary(
                name,
                method_id,
                video_grouped.get((video_name, name, method_id), []),
                video_statuses[(video_name, name, method_id)],
            )
        )
    return {"metrics": metrics, "by_video": dict(by_video)}


def campaign_analysis_v2(
    db: Database,
    campaign_id: int,
    *,
    bootstrap_samples: int = 1000,
    video: str = "",
) -> dict[str, Any]:
    campaign = get_campaign_v2(db, int(campaign_id))
    samples = max(0, min(5000, int(bootstrap_samples)))
    cache_key = _json({"bootstrap_samples": samples, "video": str(video or "")})
    cached = db.get(
        """
        SELECT payload_json FROM evaluation_analysis_cache_v2
        WHERE campaign_id = ? AND cache_key = ? AND vote_revision = ?
        """,
        (int(campaign_id), cache_key, int(campaign["vote_revision"])),
    )
    if cached is not None:
        return _loads(cached["payload_json"])
    methods = {int(row["id"]): row for row in campaign["methods"]}
    method_ids = sorted(methods)
    clauses = ["t.campaign_id = ?"]
    params: list[Any] = [int(campaign_id)]
    if video:
        clauses.append("i.video_name = ?")
        params.append(str(video))
    tasks = db.query(
        f"""
        SELECT t.id, i.video_name, ba.method_id AS method_a_id,
               bb.method_id AS method_b_id
        FROM evaluation_tasks_v2 t
        JOIN evaluation_items_v2 i ON i.id = t.item_id
        JOIN evaluation_bindings_v2 ba ON ba.id = t.binding_a_id
        JOIN evaluation_bindings_v2 bb ON bb.id = t.binding_b_id
        WHERE {' AND '.join(clauses)} ORDER BY t.id
        """,
        params,
    )
    task_by_id = {int(row["id"]): row for row in tasks}
    if task_by_id:
        placeholders = ",".join("?" for _ in task_by_id)
        votes = db.query(
            f"SELECT * FROM evaluation_votes_v2 WHERE task_id IN ({placeholders}) ORDER BY id",
            tuple(task_by_id),
        )
    else:
        votes = []
    observations: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for vote_row in votes:
        task = task_by_id[int(vote_row["task_id"])]
        method_a = int(task["method_a_id"])
        method_b = int(task["method_b_id"])
        preferred = vote_row.get("preferred_method_id")
        if preferred is None:
            score_a, score_b = 0.5, 0.5
        elif int(preferred) == method_a:
            score_a, score_b = 1.0, 0.0
        else:
            score_a, score_b = 0.0, 1.0
        observations.append(
            {
                "a": method_a,
                "b": method_b,
                "score_a": score_a,
                "score_b": score_b,
                "video_name": str(task["video_name"]),
            }
        )
        for reason in _loads(vote_row.get("reasons_json"), []):
            reason_counts[str(reason)] += 1
    human = _rank_methods(
        method_ids,
        methods,
        observations,
        seed=int(campaign["seed"]),
        bootstrap_samples=samples,
    )
    videos = sorted({str(row["video_name"]) for row in tasks})
    by_video = {
        video_name: _rank_methods(
            method_ids,
            methods,
            [row for row in observations if row["video_name"] == video_name],
            seed=int(campaign["seed"])
            ^ int(hashlib.sha256(video_name.encode("utf-8")).hexdigest()[:8], 16),
            bootstrap_samples=min(samples, 1000),
        )
        for video_name in videos
    }
    counts = Counter(int(row["task_id"]) for row in votes)
    completed_tasks = sum(
        1 for task in tasks if counts[int(task["id"])] >= int(campaign["target_votes"])
    )
    bindings = [
        {**binding, "video_name": str(item["video_name"])}
        for item in campaign["items"]
        for binding in item["bindings"]
    ]
    alignment_items = [
        {
            "evaluation_item_id": int(item["id"]),
            "media_item_id": (
                int(item["media_item_id"]) if item.get("media_item_id") is not None else None
            ),
            "video_name": str(item["video_name"]),
            "fingerprint": str((item.get("alignment") or {}).get("fingerprint") or ""),
            "plan": item.get("alignment") or {},
        }
        for item in campaign["items"]
        if (item.get("alignment") or {}).get("fingerprint")
    ]
    result = {
        "schema_version": 2,
        "campaign": {
            "id": int(campaign_id),
            "public_title": campaign["public_title"],
            "status": campaign["status"],
            "methods": campaign["methods"],
        },
        "coverage": {
            "tasks": len(tasks),
            "completed_tasks": completed_tasks,
            "target_votes_per_task": int(campaign["target_votes"]),
            "complete": bool(tasks) and completed_tasks == len(tasks),
            "provisional": not tasks or completed_tasks != len(tasks),
        },
        "human": human,
        "by_video": by_video,
        "quality_reasons": dict(sorted(reason_counts.items())),
        "objective": _objective_by_method(db, bindings, methods),
        "alignment": {
            "items": alignment_items,
            "fingerprints": [row["fingerprint"] for row in alignment_items],
        },
        "combined_score": None,
    }
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO evaluation_analysis_cache_v2(
                campaign_id, cache_key, vote_revision, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(campaign_id, cache_key) DO UPDATE SET
                vote_revision = excluded.vote_revision,
                payload_json = excluded.payload_json,
                created_at = excluded.created_at
            """,
            (int(campaign_id), cache_key, int(campaign["vote_revision"]), _json(result), utc_ts()),
        )
    return result


def campaign_export_v2(db: Database, campaign_id: int) -> dict[str, Any]:
    """Return the complete organizer-owned V2 record as portable JSON."""
    campaign = get_campaign_v2(db, int(campaign_id))
    tasks = db.query(
        """
        SELECT t.*, i.video_name,
               ma.slot AS method_a_slot, ma.label_snapshot AS method_a_label,
               mb.slot AS method_b_slot, mb.label_snapshot AS method_b_label
        FROM evaluation_tasks_v2 t
        JOIN evaluation_items_v2 i ON i.id = t.item_id
        JOIN evaluation_bindings_v2 ba ON ba.id = t.binding_a_id
        JOIN evaluation_bindings_v2 bb ON bb.id = t.binding_b_id
        JOIN evaluation_methods_v2 ma ON ma.id = ba.method_id
        JOIN evaluation_methods_v2 mb ON mb.id = bb.method_id
        WHERE t.campaign_id = ? ORDER BY t.id
        """,
        (int(campaign_id),),
    )
    votes = db.query(
        """
        SELECT v.*, e.display_name AS evaluator_name, t.task_token, i.video_name,
               m.slot AS preferred_method_slot,
               m.label_snapshot AS preferred_method_label
        FROM evaluation_votes_v2 v
        JOIN evaluation_tasks_v2 t ON t.id = v.task_id
        JOIN evaluation_items_v2 i ON i.id = t.item_id
        JOIN evaluators e ON e.id = v.evaluator_id
        LEFT JOIN evaluation_methods_v2 m ON m.id = v.preferred_method_id
        WHERE t.campaign_id = ? ORDER BY v.id
        """,
        (int(campaign_id),),
    )
    for vote in votes:
        _decode_json_fields(vote, ("reasons_json", "presentation_json"))
    return {
        "schema_version": 2,
        "campaign": campaign,
        "methods": campaign["methods"],
        "items": campaign["items"],
        "tasks": tasks,
        "votes": votes,
        "preparation": get_preparation_v2(db, int(campaign_id)),
        "analysis": campaign_analysis_v2(db, int(campaign_id)),
    }


def legacy_campaigns_readonly(db: Database) -> list[dict[str, Any]]:
    rows = db.query("SELECT * FROM evaluation_campaigns ORDER BY id DESC")
    for row in rows:
        metadata = _loads(row.pop("metadata_json", None))
        row.update(
            {
                "schema_version": 1,
                "campaign_key": f"v1:{int(row['id'])}",
                "metadata": metadata,
                "read_only": True,
                "archived": bool(metadata.get("archived_at")),
                "allowed_actions": ["export"] if metadata.get("archived_at") else ["export", "archive"],
            }
        )
    return rows


def archive_legacy_campaign(db: Database, campaign_id: int) -> dict[str, Any]:
    row = db.get("SELECT * FROM evaluation_campaigns WHERE id = ?", (int(campaign_id),))
    if row is None:
        raise KeyError(f"legacy evaluation campaign {campaign_id} not found")
    metadata = _loads(row.get("metadata_json"))
    metadata["archived_at"] = metadata.get("archived_at") or utc_ts()
    with db.connection() as conn:
        conn.execute(
            "UPDATE evaluation_campaigns SET metadata_json = ?, updated_at = ? WHERE id = ?",
            (_json(metadata), utc_ts(), int(campaign_id)),
        )
    return next(row for row in legacy_campaigns_readonly(db) if int(row["id"]) == int(campaign_id))


def discard_empty_legacy_draft(db: Database, campaign_id: int) -> None:
    row = db.get(
        """
        SELECT ec.status, COUNT(v.id) AS votes
        FROM evaluation_campaigns ec
        LEFT JOIN evaluation_tasks t ON t.campaign_id = ec.id
        LEFT JOIN evaluation_votes v ON v.task_id = t.id
        WHERE ec.id = ? GROUP BY ec.id
        """,
        (int(campaign_id),),
    )
    if row is None:
        raise KeyError(f"legacy evaluation campaign {campaign_id} not found")
    if row["status"] != "draft" or int(row["votes"] or 0):
        raise ValueError("only a legacy draft without votes can be discarded")
    with db.connection() as conn:
        conn.execute("DELETE FROM evaluation_campaigns WHERE id = ?", (int(campaign_id),))


def _legacy_assets_for_run(db: Database, run_id: int) -> list[dict[str, Any]]:
    return db.query(
        """
        SELECT DISTINCT ec.id AS campaign_id, ec.status AS campaign_status, ma.*
        FROM evaluation_campaigns ec
        JOIN evaluation_candidates c ON c.campaign_id = ec.id
        JOIN media_assets ma ON ma.id IN (c.reference_asset_id, c.asset_id)
        JOIN run_media_assets rma ON rma.asset_id = ma.id
        WHERE rma.run_id = ? AND ec.status IN ('published', 'closed')
        ORDER BY ec.id, ma.id
        """,
        (int(run_id),),
    )


def protect_campaign_media_for_run(
    db: Database,
    workspace: WorkspaceConfig,
    run_id: int,
) -> dict[str, Any]:
    """Freeze legacy references and verify V2 packages before a Run purge."""
    ensure_v2_schema(db)
    preparing = db.query(
        """
        SELECT DISTINCT c.id FROM evaluation_campaigns_v2 c
        JOIN evaluation_methods_v2 m ON m.campaign_id = c.id
        WHERE m.source_run_id = ? AND c.status = 'preparing'
        """,
        (int(run_id),),
    )
    if preparing:
        raise ValueError(
            "Run is referenced by a Campaign V2 preparation: "
            + ", ".join(str(row["id"]) for row in preparing)
        )
    v2_rows = db.query(
        """
        SELECT DISTINCT c.id FROM evaluation_campaigns_v2 c
        JOIN evaluation_methods_v2 m ON m.campaign_id = c.id
        WHERE m.source_run_id = ? AND c.status IN ('published', 'closed', 'archived')
        """,
        (int(run_id),),
    )
    for row in v2_rows:
        package = (workspace.evaluations_dir / str(int(row["id"]))).resolve()
        if not (package / "manifest.json").is_file():
            raise ValueError(f"Campaign V2 {row['id']} evaluation package is incomplete")
        assets = db.query(
            """
            SELECT frozen_reference_asset_id AS asset_id
            FROM evaluation_items_v2 WHERE campaign_id = ?
            UNION ALL
            SELECT b.frozen_asset_id AS asset_id
            FROM evaluation_bindings_v2 b
            JOIN evaluation_items_v2 i ON i.id = b.item_id
            WHERE i.campaign_id = ?
            """,
            (int(row["id"]), int(row["id"])),
        )
        if not assets or any(asset.get("asset_id") is None for asset in assets):
            raise ValueError(f"Campaign V2 {row['id']} has incomplete frozen asset bindings")
        for asset_row in assets:
            asset = get_asset(db, int(asset_row["asset_id"]), include_deleted=True)
            path = Path(str(asset["storage_path"])).resolve()
            try:
                path.relative_to(package)
            except ValueError as exc:
                raise ValueError(
                    f"Campaign V2 {row['id']} frozen asset escaped its evaluation package"
                ) from exc
            if asset["state"] != "ready" or asset.get("deleted_at") is not None or not path.exists():
                raise ValueError(
                    f"Campaign V2 {row['id']} frozen asset {asset['id']} is unavailable"
                )
    legacy_rows = _legacy_assets_for_run(db, int(run_id))
    migrated: list[int] = []
    for row in legacy_rows:
        asset = _asset_from_row(row)
        source = Path(str(asset["storage_path"])).resolve()
        if not source.exists():
            raise FileNotFoundError(
                f"legacy Campaign {row['campaign_id']} source asset {asset['id']} is unavailable"
            )
        campaign_root = (
            workspace.evaluations_dir / f"legacy-{int(row['campaign_id'])}"
        ).resolve()
        target = _frozen_target(campaign_root, f"asset-{int(asset['id'])}", source)
        if not target.exists():
            campaign_root.mkdir(parents=True, exist_ok=True)
            _clone_managed_path(source, target)
        digest = _path_sha256(target)
        with db.connection() as conn:
            try:
                conn.execute(
                    """
                    UPDATE media_assets
                    SET source_kind = 'evaluation_package', storage_path = ?,
                        content_sha256 = ?, size_bytes = ?, state = 'ready',
                        metadata_json = ?, updated_at = ?, deleted_at = NULL
                    WHERE id = ?
                    """,
                    (
                        str(target),
                        digest,
                        _path_size(target),
                        _json({**(asset.get("metadata") or {}), "legacy_campaign_frozen": True}),
                        utc_ts(),
                        int(asset["id"]),
                    ),
                )
            except Exception as exc:
                if "CHECK constraint failed" in str(exc):
                    raise RuntimeError(
                        "media_assets schema must allow evaluation_package before legacy campaign protection"
                    ) from exc
                raise
        migrated.append(int(asset["id"]))
    return {
        "run_id": int(run_id),
        "v2_campaign_ids": [int(row["id"]) for row in v2_rows],
        "legacy_frozen_asset_ids": migrated,
    }
