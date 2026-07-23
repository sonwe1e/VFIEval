from __future__ import annotations

from contextlib import contextmanager
import json
import shutil
import sqlite3
import stat as stat_module
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from vfieval.job_errors import enrich_job_error


def utc_ts() -> float:
    return time.time()


def _json(data: Any) -> str:
    return json.dumps(data if data is not None else {}, sort_keys=True)


def _loads(text: str | None) -> Any:
    if not text:
        return {}
    return json.loads(text)


def artifact_storage_metadata(
    path: str | Path,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Capture artifact bytes once, at the existing publication boundary.

    Run Detail can then compare the estimate with stored summary data without
    walking the Run directory.  Bulk artifact publication happens on the save
    workers; video artifacts are published by the finalize worker after encode.
    """

    result = dict(metadata or {})
    if "storage_bytes" in result and "storage_size_complete" in result:
        return result
    candidates: list[tuple[str, str]] = []
    canonical = str(path or "").strip()
    if canonical:
        candidates.append(("artifact_file_size_bytes", canonical))
    preview = str(result.get("preview_path") or "").strip()
    if preview and preview != canonical:
        candidates.append(("preview_file_size_bytes", preview))

    total = 0
    complete = True
    for field, candidate_text in candidates:
        try:
            file_stat = Path(candidate_text).stat()
            if not stat_module.S_ISREG(file_stat.st_mode):
                raise OSError("artifact path is not a regular file")
        except OSError:
            complete = False
            continue
        size_bytes = max(0, int(file_stat.st_size))
        result[field] = size_bytes
        total += size_bytes
    result["storage_bytes"] = total
    result["storage_size_complete"] = complete
    return result


LATEST_SCHEMA_VERSION = "2026-07-job-heartbeat-leases"


RUN_TERMINAL_STATUSES = {"completed", "failed", "canceled"}
RUN_NON_PROGRESS_STATUSES = RUN_TERMINAL_STATUSES | {"cancel_requested"}
JOB_TERMINAL_STATUSES = {"completed", "failed", "canceled"}

# State changes are intentionally centralized here. Runs and Jobs follow the
# guarded lifecycle contract; progress publication never changes lifecycle
# state and terminal states are absorbing.
RUN_START_SOURCES: dict[str, tuple[str, ...]] = {
    "decoding": ("queued", "decoding"),
    "running": ("queued", "running"),
    "finalizing": ("finalize_queued", "finalizing"),
    "metric_running": ("metric_queued", "metric_running"),
}
RUN_INFERENCE_COMPLETION_SOURCES: dict[str, tuple[str, ...]] = {
    "completed": ("running", "finalizing"),
    "finalize_queued": ("running",),
    "metric_queued": ("running", "finalizing"),
}


def _rating_key(score: float) -> str:
    """Canonical 0.25-step histogram key, e.g. 3.0 -> "3.00", 3.25 -> "3.25"."""
    return f"{round(float(score) * 4) / 4:.2f}"


def _combine_output_health(reports: Iterable[dict[str, Any] | None]) -> dict[str, Any] | None:
    valid = [report for report in reports if isinstance(report, dict)]
    if not valid:
        return None
    total_samples = sum(int(report.get("samples") or 0) for report in valid)
    weight_total = sum(int(report.get("samples") or 0) or 1 for report in valid)
    stats: dict[str, dict[str, float | int]] = {}
    for name in ("flowt_0", "flowt_1"):
        weighted_abs_mean = 0.0
        abs_max = 0.0
        nan_count = 0
        for report in valid:
            shard_samples = int(report.get("samples") or 0) or 1
            shard_stats = ((report.get("stats") or {}).get(name) or {})
            weighted_abs_mean += float(shard_stats.get("abs_mean") or 0.0) * shard_samples
            abs_max = max(abs_max, float(shard_stats.get("abs_max") or 0.0))
            nan_count += int(shard_stats.get("nan_count") or 0)
        stats[name] = {
            "abs_mean": float(weighted_abs_mean / weight_total),
            "abs_max": float(abs_max),
            "nan_count": nan_count,
        }
    for name in ("mask0", "mask1"):
        weighted_mean = 0.0
        std = 0.0
        nan_count = 0
        for report in valid:
            shard_samples = int(report.get("samples") or 0) or 1
            shard_stats = ((report.get("stats") or {}).get(name) or {})
            weighted_mean += float(shard_stats.get("mean") or 0.0) * shard_samples
            std = max(std, float(shard_stats.get("std") or 0.0))
            nan_count += int(shard_stats.get("nan_count") or 0)
        stats[name] = {
            "mean": float(weighted_mean / weight_total),
            "std": float(std),
            "nan_count": nan_count,
        }
    flow_flat = all(float(stats[name]["abs_max"]) < 1e-4 for name in ("flowt_0", "flowt_1"))
    mask_flat = all(float(stats[name]["std"]) < 1e-3 for name in ("mask0", "mask1"))
    has_nan = any(int(stats[name]["nan_count"]) > 0 for name in stats)
    warnings: list[str] = []
    seen: set[str] = set()
    for report in valid:
        for warning in report.get("warnings") or []:
            warning_text = str(warning)
            if warning_text in seen:
                continue
            seen.add(warning_text)
            warnings.append(warning_text)
    return {
        "stats": stats,
        "warnings": warnings,
        "flow_flat": flow_flat,
        "mask_flat": mask_flat,
        "has_nan": has_nan,
        "samples": total_samples,
        "shards": len(valid),
    }


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    adapter TEXT NOT NULL,
    checkpoint_path TEXT,
    input_height INTEGER NOT NULL,
    input_width INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    root_path TEXT NOT NULL,
    has_gt INTEGER NOT NULL DEFAULT 1,
    source_type TEXT NOT NULL DEFAULT 'frames',
    decode_mode TEXT NOT NULL DEFAULT 'frames',
    decoded_root_path TEXT,
    video_count INTEGER NOT NULL DEFAULT 0,
    frame_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    img0_path TEXT NOT NULL,
    img1_path TEXT NOT NULL,
    gt_path TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    UNIQUE(dataset_id, name)
);

CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    capabilities_json TEXT NOT NULL DEFAULT '{}',
    last_seen_at REAL NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    worker_id TEXT,
    progress_current INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0,
    result_json TEXT NOT NULL DEFAULT '{}',
    error_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    started_at REAL,
    heartbeat_at REAL,
    finished_at REAL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_kind ON jobs(status, kind, created_at);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    sample_id INTEGER REFERENCES samples(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_job_kind ON artifacts(job_id, kind);
CREATE INDEX IF NOT EXISTS idx_artifacts_sample ON artifacts(sample_id, kind);

CREATE TABLE IF NOT EXISTS metric_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    inference_job_id INTEGER NOT NULL,
    sample_id INTEGER REFERENCES samples(id) ON DELETE SET NULL,
    metric_name TEXT NOT NULL,
    status TEXT NOT NULL,
    value REAL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metric_results_inference ON metric_results(inference_job_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_metric_results_sample ON metric_results(sample_id, metric_name);

CREATE TABLE IF NOT EXISTS metric_cache (
    cache_key TEXT PRIMARY KEY,
    metric_name TEXT NOT NULL,
    status TEXT NOT NULL,
    value REAL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE RESTRICT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
    height INTEGER NOT NULL,
    width INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    device TEXT NOT NULL,
    precision TEXT NOT NULL,
    metrics_json TEXT NOT NULL DEFAULT '[]',
    inference_job_id INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
    metric_job_id INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    progress_current INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0,
    artifact_summary_json TEXT NOT NULL DEFAULT '{}',
    metric_summary_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    error_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    content_revision INTEGER NOT NULL DEFAULT 0,
    deleted_at REAL,
    artifact_cleaned_at REAL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_status_created ON runs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_jobs ON runs(inference_job_id, metric_job_id);

CREATE TABLE IF NOT EXISTS run_purge_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    request_type TEXT NOT NULL CHECK(request_type IN ('delete_run', 'cleanup_artifacts')),
    status TEXT NOT NULL CHECK(status IN ('requested', 'canceling', 'purging', 'failed', 'completed')),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    reclaimed_bytes INTEGER NOT NULL DEFAULT 0,
    report_json TEXT NOT NULL DEFAULT '{}',
    error_json TEXT NOT NULL DEFAULT '{}',
    claim_token TEXT NOT NULL DEFAULT '',
    requested_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    updated_at REAL NOT NULL,
    UNIQUE(run_id, request_type)
);

CREATE INDEX IF NOT EXISTS idx_run_purge_requests_status
ON run_purge_requests(status, updated_at);

CREATE TABLE IF NOT EXISTS cache_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_type TEXT NOT NULL CHECK(cache_type IN ('decode_cache', 'compare_cache')),
    cache_key TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'ready' CHECK(state IN ('ready', 'missing', 'deleting', 'deleted', 'failed')),
    size_bytes INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    last_used_at REAL NOT NULL,
    gc_after REAL,
    deleted_at REAL,
    updated_at REAL NOT NULL,
    UNIQUE(cache_type, cache_key)
);

CREATE INDEX IF NOT EXISTS idx_cache_entries_gc
ON cache_entries(state, gc_after, cache_type);

CREATE TABLE IF NOT EXISTS run_cache_refs (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    cache_entry_id INTEGER NOT NULL REFERENCES cache_entries(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    released_at REAL,
    PRIMARY KEY(run_id, cache_entry_id)
);

CREATE INDEX IF NOT EXISTS idx_run_cache_refs_entry
ON run_cache_refs(cache_entry_id, run_id);

CREATE TABLE IF NOT EXISTS cache_leases (
    cache_entry_id INTEGER NOT NULL REFERENCES cache_entries(id) ON DELETE CASCADE,
    lease_id TEXT NOT NULL,
    expires_at REAL NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY(cache_entry_id, lease_id)
);

CREATE INDEX IF NOT EXISTS idx_cache_leases_expiry
ON cache_leases(expires_at, cache_entry_id);

CREATE TABLE IF NOT EXISTS decode_cache_build_locks (
    cache_key TEXT PRIMARY KEY,
    owner_token TEXT NOT NULL,
    expires_at REAL NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decode_cache_build_locks_expiry
ON decode_cache_build_locks(expires_at, cache_key);

CREATE TABLE IF NOT EXISTS run_jobs (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    shard_index INTEGER NOT NULL DEFAULT 0,
    device TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY(run_id, job_id)
);

CREATE INDEX IF NOT EXISTS idx_run_jobs_run_role ON run_jobs(run_id, role, shard_index);
CREATE INDEX IF NOT EXISTS idx_run_jobs_job ON run_jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_run_jobs_device ON run_jobs(device);

CREATE TABLE IF NOT EXISTS run_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    username TEXT NOT NULL DEFAULT '',
    rating REAL,
    issue TEXT NOT NULL DEFAULT '',
    video TEXT NOT NULL DEFAULT '',
    track_label TEXT NOT NULL DEFAULT '',
    model_name TEXT NOT NULL DEFAULT '',
    checkpoint TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL
);

CREATE INDEX IF NOT EXISTS idx_run_feedback_run ON run_feedback(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_run_feedback_video ON run_feedback(video, model_name, checkpoint);
CREATE INDEX IF NOT EXISTS idx_run_feedback_filters ON run_feedback(model_name, checkpoint, video, run_id);
CREATE INDEX IF NOT EXISTS idx_run_feedback_user ON run_feedback(username, created_at);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS media_collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    slug TEXT NOT NULL UNIQUE,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS media_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER REFERENCES media_collections(id) ON DELETE SET NULL,
    source_key TEXT NOT NULL UNIQUE,
    source_kind TEXT NOT NULL CHECK(source_kind IN ('folder', 'upload', 'run_artifact', 'evaluation_package')),
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
);

CREATE INDEX IF NOT EXISTS idx_media_assets_catalog
ON media_assets(state, role, source_kind, collection_id, display_name);
CREATE INDEX IF NOT EXISTS idx_media_assets_hash ON media_assets(content_sha256);

CREATE TABLE IF NOT EXISTS media_asset_relations (
    parent_asset_id INTEGER NOT NULL REFERENCES media_assets(id) ON DELETE CASCADE,
    child_asset_id INTEGER NOT NULL REFERENCES media_assets(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY(parent_asset_id, child_asset_id, relation_type)
);

CREATE TABLE IF NOT EXISTS run_media_assets (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    asset_id INTEGER NOT NULL REFERENCES media_assets(id) ON DELETE RESTRICT,
    role TEXT NOT NULL,
    video_name TEXT NOT NULL DEFAULT '',
    track_label TEXT NOT NULL DEFAULT '',
    model_name TEXT NOT NULL DEFAULT '',
    checkpoint TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY(run_id, asset_id, role, video_name, track_label)
);

CREATE INDEX IF NOT EXISTS idx_run_media_assets_run
ON run_media_assets(run_id, role, video_name, track_label);
CREATE INDEX IF NOT EXISTS idx_run_media_assets_asset ON run_media_assets(asset_id);

CREATE TABLE IF NOT EXISTS media_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES media_collections(id) ON DELETE RESTRICT,
    item_key TEXT NOT NULL UNIQUE,
    canonical_gt_asset_id INTEGER NOT NULL UNIQUE REFERENCES media_assets(id) ON DELETE RESTRICT,
    display_name TEXT NOT NULL,
    media_kind TEXT NOT NULL CHECK(media_kind IN ('video', 'frame_sequence')),
    state TEXT NOT NULL DEFAULT 'ready' CHECK(state IN ('ready', 'unavailable', 'deleted')),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    deleted_at REAL,
    UNIQUE(collection_id, display_name)
);

CREATE INDEX IF NOT EXISTS idx_media_items_group
ON media_items(collection_id, state, display_name, id);

CREATE TABLE IF NOT EXISTS media_item_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES media_items(id) ON DELETE RESTRICT,
    asset_id INTEGER NOT NULL REFERENCES media_assets(id) ON DELETE RESTRICT,
    member_role TEXT NOT NULL CHECK(member_role IN (
        'canonical_gt', 'model_pred', 'external_pred', 'compare_snapshot',
        'evaluation_gt', 'evaluation_pred'
    )),
    producer_kind TEXT NOT NULL CHECK(producer_kind IN (
        'source', 'model_inference', 'external', 'video_compare', 'evaluation_package'
    )),
    producer_run_id INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    method_key TEXT NOT NULL DEFAULT '',
    reusable_as_pred INTEGER NOT NULL DEFAULT 0 CHECK(reusable_as_pred IN (0, 1)),
    temporal_mapping_json TEXT NOT NULL DEFAULT '{}',
    spatial_origin_json TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL DEFAULT 'ready' CHECK(state IN ('ready', 'unavailable', 'deleted')),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    deleted_at REAL,
    UNIQUE(item_id, asset_id, member_role),
    UNIQUE(asset_id, member_role),
    CHECK(
        reusable_as_pred = 0
        OR (member_role = 'model_pred' AND producer_kind = 'model_inference')
        OR (member_role = 'external_pred' AND producer_kind = 'external')
    ),
    CHECK(member_role != 'canonical_gt' OR (producer_kind = 'source' AND producer_run_id IS NULL)),
    CHECK(member_role != 'model_pred' OR (producer_kind = 'model_inference' AND producer_run_id IS NOT NULL)),
    CHECK(member_role != 'external_pred' OR (producer_kind = 'external' AND producer_run_id IS NULL)),
    CHECK(member_role != 'compare_snapshot' OR producer_kind = 'video_compare'),
    CHECK(member_role NOT IN ('evaluation_gt', 'evaluation_pred') OR producer_kind = 'evaluation_package')
);

CREATE INDEX IF NOT EXISTS idx_media_item_members_item
ON media_item_members(item_id, reusable_as_pred, state, member_role, method_key);
CREATE INDEX IF NOT EXISTS idx_media_item_members_producer
ON media_item_members(producer_run_id, member_role, state);
CREATE INDEX IF NOT EXISTS idx_media_item_members_asset
ON media_item_members(asset_id, item_id, member_role);

CREATE TABLE IF NOT EXISTS run_media_item_bindings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL REFERENCES media_items(id) ON DELETE RESTRICT,
    binding_role TEXT NOT NULL CHECK(binding_role IN (
        'source', 'pred_output', 'compare_gt', 'compare_pred'
    )),
    slot TEXT NOT NULL DEFAULT '',
    original_member_id INTEGER NOT NULL REFERENCES media_item_members(id) ON DELETE RESTRICT,
    active_member_id INTEGER NOT NULL REFERENCES media_item_members(id) ON DELETE RESTRICT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(run_id, item_id, binding_role, slot)
);

CREATE INDEX IF NOT EXISTS idx_run_media_item_bindings_run
ON run_media_item_bindings(run_id, binding_role, item_id, slot);
CREATE INDEX IF NOT EXISTS idx_run_media_item_bindings_active
ON run_media_item_bindings(active_member_id, binding_role, run_id);
CREATE INDEX IF NOT EXISTS idx_run_media_item_bindings_original
ON run_media_item_bindings(original_member_id, binding_role, run_id);

CREATE TABLE IF NOT EXISTS metric_asset_bindings (
    metric_result_id INTEGER PRIMARY KEY REFERENCES metric_results(id) ON DELETE CASCADE,
    reference_asset_id INTEGER REFERENCES media_assets(id) ON DELETE SET NULL,
    distorted_asset_id INTEGER REFERENCES media_assets(id) ON DELETE SET NULL,
    video_name TEXT NOT NULL DEFAULT '',
    track_label TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metric_asset_bindings_pair
ON metric_asset_bindings(reference_asset_id, distorted_asset_id, video_name);

CREATE TABLE IF NOT EXISTS upload_sessions (
    id TEXT PRIMARY KEY,
    collection_id INTEGER NOT NULL REFERENCES media_collections(id) ON DELETE RESTRICT,
    role TEXT NOT NULL CHECK(role IN ('gt', 'pred')),
    media_kind TEXT NOT NULL CHECK(media_kind IN ('video', 'frame_sequence')),
    display_name TEXT NOT NULL,
    original_name TEXT NOT NULL,
    expected_size INTEGER NOT NULL,
    expected_sha256 TEXT NOT NULL,
    fps REAL,
    chunk_size INTEGER NOT NULL,
    state TEXT NOT NULL,
    received_bytes INTEGER NOT NULL DEFAULT 0,
    error_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    completed_at REAL
);

CREATE INDEX IF NOT EXISTS idx_upload_sessions_state ON upload_sessions(state, updated_at);

CREATE TABLE IF NOT EXISTS upload_parts (
    upload_id TEXT NOT NULL REFERENCES upload_sessions(id) ON DELETE CASCADE,
    part_index INTEGER NOT NULL,
    offset_bytes INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(upload_id, part_index)
);

CREATE TABLE IF NOT EXISTS execution_profiles (
    fingerprint TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    checkpoint TEXT NOT NULL DEFAULT '',
    device_kind TEXT NOT NULL,
    device_model TEXT NOT NULL DEFAULT '',
    device_count INTEGER NOT NULL,
    height INTEGER NOT NULL,
    width INTEGER NOT NULL,
    precision TEXT NOT NULL,
    artifact_profile TEXT NOT NULL,
    settings_json TEXT NOT NULL DEFAULT '{}',
    performance_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluators (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_seen_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_campaigns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    campaign_type TEXT NOT NULL CHECK(campaign_type IN ('campaign', 'adhoc')),
    status TEXT NOT NULL CHECK(status IN ('draft', 'published', 'closed')),
    target_votes INTEGER NOT NULL DEFAULT 3,
    seed INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL REFERENCES evaluation_campaigns(id) ON DELETE CASCADE,
    reference_asset_id INTEGER NOT NULL REFERENCES media_assets(id) ON DELETE RESTRICT,
    asset_id INTEGER NOT NULL REFERENCES media_assets(id) ON DELETE RESTRICT,
    video_name TEXT NOT NULL,
    label_snapshot TEXT NOT NULL,
    model_snapshot TEXT NOT NULL DEFAULT '',
    checkpoint_snapshot TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    UNIQUE(campaign_id, reference_asset_id, asset_id, video_name)
);

CREATE INDEX IF NOT EXISTS idx_evaluation_candidates_campaign
ON evaluation_candidates(campaign_id, video_name);

CREATE TABLE IF NOT EXISTS evaluation_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER REFERENCES evaluation_campaigns(id) ON DELETE CASCADE,
    reference_asset_id INTEGER NOT NULL REFERENCES media_assets(id) ON DELETE RESTRICT,
    candidate_a_id INTEGER NOT NULL REFERENCES evaluation_candidates(id) ON DELETE CASCADE,
    candidate_b_id INTEGER NOT NULL REFERENCES evaluation_candidates(id) ON DELETE CASCADE,
    video_name TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'ready',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    CHECK(candidate_a_id < candidate_b_id),
    UNIQUE(campaign_id, reference_asset_id, candidate_a_id, candidate_b_id, video_name)
);

CREATE INDEX IF NOT EXISTS idx_evaluation_tasks_campaign
ON evaluation_tasks(campaign_id, video_name, state);

CREATE TABLE IF NOT EXISTS evaluation_votes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL REFERENCES evaluation_tasks(id) ON DELETE CASCADE,
    evaluator_id TEXT NOT NULL REFERENCES evaluators(id) ON DELETE CASCADE,
    choice TEXT NOT NULL CHECK(choice IN ('left', 'right', 'tie')),
    preferred_asset_id INTEGER REFERENCES media_assets(id) ON DELETE SET NULL,
    reasons_json TEXT NOT NULL DEFAULT '[]',
    confidence TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    duration_ms INTEGER,
    presentation_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(task_id, evaluator_id)
);

CREATE INDEX IF NOT EXISTS idx_evaluation_votes_task ON evaluation_votes(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_evaluation_votes_evaluator ON evaluation_votes(evaluator_id, created_at);
"""


class Database:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self.connection() as conn:
            self._backup_before_upgrade(conn)
            conn.executescript(SCHEMA)
            self._migrate(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (LATEST_SCHEMA_VERSION, utc_ts()),
            )
            now = utc_ts()
            conn.execute(
                """
                INSERT OR IGNORE INTO experiments(name, description, metadata_json, created_at)
                VALUES ('Default', 'Default experiment', '{}', ?)
                """,
                (now,),
            )

    def _backup_before_upgrade(self, conn: sqlite3.Connection) -> None:
        has_runs = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'runs'"
        ).fetchone()
        if has_runs is None:
            return
        has_migrations = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
        ).fetchone()
        if has_migrations is not None:
            applied = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version = ?",
                (LATEST_SCHEMA_VERSION,),
            ).fetchone()
            if applied is not None:
                return
        conn.execute("PRAGMA wal_checkpoint(FULL)")
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        backup_dir = self.db_path.parent / "backups" / stamp
        backup_dir.mkdir(parents=True, exist_ok=True)
        target = backup_dir / self.db_path.name
        if self.db_path.exists() and not target.exists():
            shutil.copy2(self.db_path, target)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_jobs (
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                shard_index INTEGER NOT NULL DEFAULT 0,
                device TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                PRIMARY KEY(run_id, job_id)
            );
            CREATE INDEX IF NOT EXISTS idx_run_jobs_run_role ON run_jobs(run_id, role, shard_index);
            CREATE INDEX IF NOT EXISTS idx_run_jobs_job ON run_jobs(job_id);
            CREATE INDEX IF NOT EXISTS idx_run_jobs_device ON run_jobs(device);
            CREATE INDEX IF NOT EXISTS idx_artifacts_sample ON artifacts(sample_id, kind);
            CREATE INDEX IF NOT EXISTS idx_metric_results_sample ON metric_results(sample_id, metric_name);
            CREATE TABLE IF NOT EXISTS run_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                username TEXT NOT NULL DEFAULT '',
                rating REAL,
                issue TEXT NOT NULL DEFAULT '',
                video TEXT NOT NULL DEFAULT '',
                track_label TEXT NOT NULL DEFAULT '',
                model_name TEXT NOT NULL DEFAULT '',
                checkpoint TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_run_feedback_run ON run_feedback(run_id, created_at);
            CREATE TABLE IF NOT EXISTS run_purge_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                request_type TEXT NOT NULL CHECK(request_type IN ('delete_run', 'cleanup_artifacts')),
                status TEXT NOT NULL CHECK(status IN ('requested', 'canceling', 'purging', 'failed', 'completed')),
                attempt_count INTEGER NOT NULL DEFAULT 0,
                reclaimed_bytes INTEGER NOT NULL DEFAULT 0,
                report_json TEXT NOT NULL DEFAULT '{}',
                error_json TEXT NOT NULL DEFAULT '{}',
                claim_token TEXT NOT NULL DEFAULT '',
                requested_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                updated_at REAL NOT NULL,
                UNIQUE(run_id, request_type)
            );
            CREATE INDEX IF NOT EXISTS idx_run_purge_requests_status
            ON run_purge_requests(status, updated_at);
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_type TEXT NOT NULL CHECK(cache_type IN ('decode_cache', 'compare_cache')),
                cache_key TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'ready' CHECK(state IN ('ready', 'missing', 'deleting', 'deleted', 'failed')),
                size_bytes INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                last_used_at REAL NOT NULL,
                gc_after REAL,
                deleted_at REAL,
                updated_at REAL NOT NULL,
                UNIQUE(cache_type, cache_key)
            );
            CREATE INDEX IF NOT EXISTS idx_cache_entries_gc
            ON cache_entries(state, gc_after, cache_type);
            CREATE TABLE IF NOT EXISTS run_cache_refs (
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                cache_entry_id INTEGER NOT NULL REFERENCES cache_entries(id) ON DELETE CASCADE,
                created_at REAL NOT NULL,
                released_at REAL,
                PRIMARY KEY(run_id, cache_entry_id)
            );
            CREATE INDEX IF NOT EXISTS idx_run_cache_refs_entry
            ON run_cache_refs(cache_entry_id, run_id);
            CREATE TABLE IF NOT EXISTS cache_leases (
                cache_entry_id INTEGER NOT NULL REFERENCES cache_entries(id) ON DELETE CASCADE,
                lease_id TEXT NOT NULL,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(cache_entry_id, lease_id)
            );
            CREATE INDEX IF NOT EXISTS idx_cache_leases_expiry
            ON cache_leases(expires_at, cache_entry_id);
            CREATE TABLE IF NOT EXISTS decode_cache_build_locks (
                cache_key TEXT PRIMARY KEY,
                owner_token TEXT NOT NULL,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_decode_cache_build_locks_expiry
            ON decode_cache_build_locks(expires_at, cache_key);
            """
        )
        feedback_columns = {row["name"] for row in conn.execute("PRAGMA table_info(run_feedback)").fetchall()}
        for name, definition in {
            "video": "TEXT NOT NULL DEFAULT ''",
            "track_label": "TEXT NOT NULL DEFAULT ''",
            "model_name": "TEXT NOT NULL DEFAULT ''",
            "checkpoint": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "REAL",
        }.items():
            if name not in feedback_columns:
                conn.execute(f"ALTER TABLE run_feedback ADD COLUMN {name} {definition}")
        # Create after the ALTER so the columns exist on pre-existing tables.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_run_feedback_video ON run_feedback(video, model_name, checkpoint)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_run_feedback_filters ON run_feedback(model_name, checkpoint, video, run_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_run_feedback_user ON run_feedback(username, created_at)"
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(datasets)").fetchall()}
        dataset_columns = {
            "source_type": "TEXT NOT NULL DEFAULT 'frames'",
            "decode_mode": "TEXT NOT NULL DEFAULT 'frames'",
            "decoded_root_path": "TEXT",
            "video_count": "INTEGER NOT NULL DEFAULT 0",
            "frame_count": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, definition in dataset_columns.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE datasets ADD COLUMN {name} {definition}")
        run_columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        for name, definition in {
            "content_revision": "INTEGER NOT NULL DEFAULT 0",
            "deleted_at": "REAL",
            "artifact_cleaned_at": "REAL",
        }.items():
            if name not in run_columns:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")
        job_columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        if "heartbeat_at" not in job_columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN heartbeat_at REAL")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_running_heartbeat "
            "ON jobs(status, heartbeat_at, started_at, id)"
        )
        cache_ref_columns = {row["name"] for row in conn.execute("PRAGMA table_info(run_cache_refs)").fetchall()}
        if "released_at" not in cache_ref_columns:
            conn.execute("ALTER TABLE run_cache_refs ADD COLUMN released_at REAL")
        purge_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(run_purge_requests)").fetchall()
        }
        if "claim_token" not in purge_columns:
            conn.execute("ALTER TABLE run_purge_requests ADD COLUMN claim_token TEXT NOT NULL DEFAULT ''")
        profile_columns = {row["name"] for row in conn.execute("PRAGMA table_info(execution_profiles)").fetchall()}
        if "device_model" not in profile_columns:
            conn.execute("ALTER TABLE execution_profiles ADD COLUMN device_model TEXT NOT NULL DEFAULT ''")

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def get(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def register_model(
        self,
        name: str,
        adapter: str,
        checkpoint_path: str | None,
        input_height: int,
        input_width: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO models(name, adapter, checkpoint_path, input_height, input_width, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (name, adapter, checkpoint_path, input_height, input_width, _json(metadata), now),
            )
            return int(cur.lastrowid)

    def upsert_model(
        self,
        name: str,
        adapter: str,
        checkpoint_path: str | None,
        input_height: int,
        input_width: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        existing = self.get("SELECT id FROM models WHERE name = ?", (name,))
        now = utc_ts()
        if existing:
            model_id = int(existing["id"])
            with self.connection() as conn:
                conn.execute(
                    """
                    UPDATE models
                    SET adapter = ?,
                        checkpoint_path = ?,
                        input_height = ?,
                        input_width = ?,
                        metadata_json = ?
                    WHERE id = ?
                    """,
                    (adapter, checkpoint_path, input_height, input_width, _json(metadata), model_id),
                )
            return model_id
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO models(name, adapter, checkpoint_path, input_height, input_width, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (name, adapter, checkpoint_path, input_height, input_width, _json(metadata), now),
            )
            return int(cur.lastrowid)

    def list_models(self) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM models ORDER BY id")
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def get_model(self, model_id: int) -> dict[str, Any]:
        row = self.get("SELECT * FROM models WHERE id = ?", (model_id,))
        if row is None:
            raise KeyError(f"model {model_id} not found")
        row["metadata"] = _loads(row.pop("metadata_json"))
        return row

    def create_dataset(
        self,
        name: str,
        root_path: str,
        has_gt: bool,
        source_type: str = "frames",
        decode_mode: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        source_type = source_type or "frames"
        decode_mode = decode_mode or ("frames" if source_type == "frames" else ("video_gt_triplets" if has_gt else "video_pairs"))
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO datasets(
                    name, root_path, has_gt, source_type, decode_mode, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    str(Path(root_path).resolve()),
                    int(has_gt),
                    source_type,
                    decode_mode,
                    _json(metadata),
                    now,
                ),
            )
            return int(cur.lastrowid)

    def upsert_dataset(
        self,
        name: str,
        root_path: str,
        has_gt: bool,
        source_type: str = "frames",
        decode_mode: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        source_type = source_type or "frames"
        decode_mode = decode_mode or ("frames" if source_type == "frames" else ("video_gt_triplets" if has_gt else "video_pairs"))
        existing = self.get("SELECT id FROM datasets WHERE name = ?", (name,))
        if existing:
            dataset_id = int(existing["id"])
            with self.connection() as conn:
                conn.execute(
                    """
                    UPDATE datasets
                    SET root_path = ?,
                        has_gt = ?,
                        source_type = ?,
                        decode_mode = ?,
                        metadata_json = ?
                    WHERE id = ?
                    """,
                    (
                        str(Path(root_path).resolve()),
                        int(has_gt),
                        source_type,
                        decode_mode,
                        _json(metadata),
                        dataset_id,
                    ),
                )
            return dataset_id
        return self.create_dataset(name, root_path, has_gt, source_type, decode_mode, metadata)

    def list_datasets(self) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM datasets ORDER BY id")
        for row in rows:
            row["has_gt"] = bool(row["has_gt"])
            row["metadata"] = _loads(row.pop("metadata_json"))
            row["sample_count"] = self.count_samples(int(row["id"]))
        return rows

    def get_dataset(self, dataset_id: int) -> dict[str, Any]:
        row = self.get("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        if row is None:
            raise KeyError(f"dataset {dataset_id} not found")
        row["has_gt"] = bool(row["has_gt"])
        row["metadata"] = _loads(row.pop("metadata_json"))
        return row

    def update_dataset_scan_info(
        self,
        dataset_id: int,
        decoded_root_path: str | None,
        video_count: int,
        frame_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        dataset = self.get_dataset(dataset_id)
        merged_metadata = {**(dataset.get("metadata") or {}), **(metadata or {})}
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE datasets
                SET decoded_root_path = ?,
                    video_count = ?,
                    frame_count = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    str(Path(decoded_root_path).resolve()) if decoded_root_path else None,
                    int(video_count),
                    int(frame_count),
                    _json(merged_metadata),
                    dataset_id,
                ),
            )

    def add_sample(
        self,
        dataset_id: int,
        name: str,
        img0_path: str,
        img1_path: str,
        gt_path: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT OR REPLACE INTO samples(dataset_id, name, img0_path, img1_path, gt_path, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    name,
                    str(Path(img0_path).resolve()),
                    str(Path(img1_path).resolve()),
                    str(Path(gt_path).resolve()) if gt_path else None,
                    _json(metadata),
                    now,
                ),
            )
            return int(cur.lastrowid)

    def batch_add_samples(
        self,
        dataset_id: int,
        rows: list[dict[str, Any]],
    ) -> None:
        """Insert multiple samples in one transaction.  Each row must have
        ``name``, ``img0_path``, ``img1_path``, and optionally ``gt_path``
        and ``metadata``.  Uses INSERT OR REPLACE so re-scanning is safe."""
        if not rows:
            return
        now = utc_ts()
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO samples
                    (dataset_id, name, img0_path, img1_path, gt_path, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        dataset_id,
                        row["name"],
                        str(Path(row["img0_path"]).resolve()),
                        str(Path(row["img1_path"]).resolve()),
                        str(Path(row["gt_path"]).resolve()) if row.get("gt_path") else None,
                        _json(row.get("metadata")),
                        now,
                    )
                    for row in rows
                ],
            )

    def list_samples(self, dataset_id: int) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM samples WHERE dataset_id = ? ORDER BY name", (dataset_id,))
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def clear_samples(self, dataset_id: int) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM samples WHERE dataset_id = ?", (dataset_id,))

    def count_samples(self, dataset_id: int) -> int:
        row = self.get("SELECT COUNT(*) AS count FROM samples WHERE dataset_id = ?", (dataset_id,))
        return int(row["count"] if row else 0)

    def get_sample(self, sample_id: int) -> dict[str, Any]:
        row = self.get("SELECT * FROM samples WHERE id = ?", (sample_id,))
        if row is None:
            raise KeyError(f"sample {sample_id} not found")
        row["metadata"] = _loads(row.pop("metadata_json"))
        return row

    def list_samples_by_video(self, run_id: int, video_name: str) -> list[dict[str, Any]]:
        run = self.get_run(run_id)
        dataset_id = int(run["dataset_id"])
        video_text = str(video_name or "")
        if not video_text:
            return []
        patterns = [
            f'%"video_name": "{video_text}"%',
            f'%"video_file": "{video_text}"%',
            f'%"compare_group": "{video_text}"%',
        ]
        name_patterns = [
            f"{video_text}__%",
            f"%_{video_text}_%",
        ]
        clauses = ["metadata_json LIKE ?" for _ in patterns] + ["name LIKE ?" for _ in name_patterns]
        rows = self.query(
            f"""
            SELECT *
            FROM samples
            WHERE dataset_id = ?
              AND ({' OR '.join(clauses)})
            ORDER BY name
            """,
            [dataset_id, *patterns, *name_patterns],
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def find_sample_by_video_frame(
        self,
        run_id: int,
        video_name: str,
        frame_index: int,
        track_label: str | None = None,
    ) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        dataset_id = int(run["dataset_id"])
        video_text = str(video_name or "")
        params: list[Any] = [dataset_id, video_text, video_text, int(frame_index), int(frame_index)]
        track_clause = ""
        if track_label:
            track_clause = """
              AND (
                json_extract(metadata_json, '$.compare_track_label') IS NULL
                OR json_extract(metadata_json, '$.compare_track_label') = ?
              )
            """
            params.append(str(track_label))
        rows = self.query(
            f"""
            SELECT *
            FROM samples
            WHERE dataset_id = ?
              AND (
                json_extract(metadata_json, '$.video_name') = ?
                OR json_extract(metadata_json, '$.video_file') = ?
              )
              AND (
                CAST(json_extract(metadata_json, '$.frame_index') AS INTEGER) = ?
                OR CAST(json_extract(metadata_json, '$.sample_index') AS INTEGER) = ?
              )
              {track_clause}
            ORDER BY id
            LIMIT 1
            """,
            params,
        )
        if not rows:
            return None
        row = rows[0]
        row["metadata"] = _loads(row.pop("metadata_json"))
        return row

    def list_run_video_summaries(self, run_id: int, query: str = "") -> list[dict[str, Any]]:
        run = self.get_run(run_id)
        dataset_id = int(run["dataset_id"])
        rows = self.query(
            """
            SELECT
                COALESCE(
                    json_extract(metadata_json, '$.video_name'),
                    json_extract(metadata_json, '$.video_file'),
                    'frames'
                ) AS video_name,
                COALESCE(
                    json_extract(metadata_json, '$.video_file'),
                    json_extract(metadata_json, '$.video_name'),
                    'frames'
                ) AS video_file,
                AVG(CAST(COALESCE(json_extract(metadata_json, '$.fps'), 0) AS REAL)) AS fps,
                COUNT(*) AS sample_count
            FROM samples
            WHERE dataset_id = ?
            GROUP BY video_name, video_file
            ORDER BY video_file, video_name
            """,
            (dataset_id,),
        )
        normalized_query = str(query or "").strip().lower()
        result = []
        for row in rows:
            video_name = str(row.get("video_name") or "frames")
            video_file = str(row.get("video_file") or video_name)
            if normalized_query and normalized_query not in video_name.lower() and normalized_query not in video_file.lower():
                continue
            result.append(
                {
                    "video_name": video_name,
                    "video_file": video_file,
                    "fps": float(row.get("fps") or 0.0),
                    "sample_count": int(row.get("sample_count") or 0),
                }
            )
        return result

    def create_experiment(
        self,
        name: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO experiments(name, description, metadata_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, description, _json(metadata), now),
            )
            return int(cur.lastrowid)

    def get_default_experiment_id(self) -> int:
        row = self.get("SELECT id FROM experiments WHERE name = 'Default'")
        if row is not None:
            return int(row["id"])
        return self.create_experiment("Default", "Default experiment")

    def list_experiments(self) -> list[dict[str, Any]]:
        rows = self.query(
            """
            SELECT e.*, COUNT(r.id) AS run_count
            FROM experiments e
            LEFT JOIN runs r ON r.experiment_id = e.id
            GROUP BY e.id
            ORDER BY e.id
            """
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def get_experiment(self, experiment_id: int) -> dict[str, Any]:
        row = self.get("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        if row is None:
            raise KeyError(f"experiment {experiment_id} not found")
        row["metadata"] = _loads(row.pop("metadata_json"))
        return row

    def create_run(
        self,
        name: str,
        model_id: int,
        dataset_id: int,
        height: int,
        width: int,
        batch_size: int,
        device: str,
        precision: str,
        metrics: list[str],
        experiment_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        create_inference_job: bool = True,
    ) -> int:
        experiment_id = experiment_id or self.get_default_experiment_id()
        now = utc_ts()
        progress_total = self.count_samples(dataset_id)
        with self.connection() as conn:
            run_cur = conn.execute(
                """
                INSERT INTO runs(
                    experiment_id, name, model_id, dataset_id, height, width, batch_size,
                    device, precision, metrics_json, status, progress_total,
                    metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    name,
                    model_id,
                    dataset_id,
                    height,
                    width,
                    batch_size,
                    device,
                    precision,
                    _json(metrics),
                    progress_total,
                    _json(metadata),
                    now,
                    now,
                ),
            )
            run_id = int(run_cur.lastrowid)
            if not create_inference_job:
                return run_id
            payload = {
                "run_id": run_id,
                "model_id": model_id,
                "dataset_id": dataset_id,
                "height": height,
                "width": width,
                "batch_size": batch_size,
                "device": device,
                "precision": precision,
                "metrics": metrics,
            }
            meta = metadata or {}
            if meta.get("visualize_height") is not None:
                payload["visualize_height"] = meta.get("visualize_height")
            if meta.get("visualize_width") is not None:
                payload["visualize_width"] = meta.get("visualize_width")
            job_cur = conn.execute(
                """
                INSERT INTO jobs(kind, status, payload_json, progress_total, created_at)
                VALUES ('inference', 'queued', ?, ?, ?)
                """,
                (_json(payload), progress_total, now),
            )
            inference_job_id = int(job_cur.lastrowid)
            conn.execute(
                """
                UPDATE runs
                SET inference_job_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (inference_job_id, now, run_id),
            )
            conn.execute(
                """
                INSERT INTO run_jobs(run_id, job_id, role, shard_index, device, metadata_json, created_at)
                VALUES (?, ?, 'inference', 0, ?, '{}', ?)
                """,
                (run_id, inference_job_id, device, now),
            )
            return run_id

    def add_run_job(
        self,
        run_id: int,
        role: str,
        payload: dict[str, Any],
        progress_total: int = 0,
        shard_index: int = 0,
        device: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO jobs(kind, status, payload_json, progress_total, created_at)
                VALUES (?, 'queued', ?, ?, ?)
                """,
                (role, _json(payload), int(progress_total), now),
            )
            job_id = int(cur.lastrowid)
            conn.execute(
                """
                INSERT INTO run_jobs(run_id, job_id, role, shard_index, device, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, job_id, role, int(shard_index), device, _json(metadata), now),
            )
            if role == "inference":
                conn.execute(
                    """
                    UPDATE runs
                    SET inference_job_id = COALESCE(inference_job_id, ?), updated_at = ?
                    WHERE id = ?
                    """,
                    (job_id, now, run_id),
                )
            elif role == "decode":
                conn.execute(
                    """
                    UPDATE runs
                    SET status = 'decoding', updated_at = ?
                    WHERE id = ? AND status = 'queued'
                    """,
                    (now, run_id),
                )
            return job_id

    @staticmethod
    def _source_job_status(
        conn: sqlite3.Connection,
        run_id: int,
        source_job_id: int | None,
        allowed_kinds: Iterable[str],
    ) -> str | None:
        """Validate that an optional phase-producing Job belongs to this Run."""
        if source_job_id is None:
            return None
        kinds = tuple(str(kind) for kind in allowed_kinds)
        if not kinds:
            raise ValueError("source Job validation requires at least one kind")
        placeholders = ",".join("?" for _ in kinds)
        row = conn.execute(
            f"""
            SELECT j.status
            FROM jobs j
            WHERE j.id = ? AND j.kind IN ({placeholders})
              AND (
                    CAST(json_extract(j.payload_json, '$.run_id') AS INTEGER) = ?
                    OR EXISTS (
                        SELECT 1 FROM run_jobs rj
                        WHERE rj.run_id = ? AND rj.job_id = j.id
                    )
                    OR EXISTS (
                        SELECT 1 FROM runs r
                        WHERE r.id = ?
                          AND (r.inference_job_id = j.id OR r.metric_job_id = j.id)
                    )
              )
            """,
            (int(source_job_id), *kinds, int(run_id), int(run_id), int(run_id)),
        ).fetchone()
        if row is None:
            return ""
        return str(row["status"] or "")

    @staticmethod
    def _complete_source_job_in_transaction(
        conn: sqlite3.Connection,
        source_job_id: int | None,
        source_status: str | None,
        source_job_result: dict[str, Any] | None,
        now: float,
    ) -> bool:
        if source_job_id is None:
            return True
        if source_status == "completed":
            row = conn.execute(
                "SELECT result_json FROM jobs WHERE id = ?",
                (int(source_job_id),),
            ).fetchone()
            return bool(
                row is not None
                and _loads(row["result_json"]) == _loads(_json(source_job_result))
            )
        if source_status != "running":
            return False
        cur = conn.execute(
            """
            UPDATE jobs
            SET status = 'completed', result_json = ?, finished_at = ?
            WHERE id = ? AND status = 'running'
            """,
            (_json(source_job_result), now, int(source_job_id)),
        )
        return bool(cur.rowcount)

    def publish_inference_jobs(
        self,
        run_id: int,
        job_specs: Iterable[dict[str, Any]],
        *,
        source_job_id: int | None = None,
        source_job_result: dict[str, Any] | None = None,
    ) -> list[int]:
        """Atomically publish the decode -> inference handoff.

        A cancellation that commits first prevents every new inference job;
        publication that commits first exposes the complete job set and the
        queued Run state together. Repeated calls return the original set.
        """
        specs = list(job_specs)
        if not specs:
            raise ValueError("inference handoff requires at least one job")
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute(
                """
                SELECT status, inference_job_id, device,
                       deleted_at, artifact_cleaned_at
                FROM runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"run {run_id} not found")
            source_status = self._source_job_status(
                conn,
                run_id,
                source_job_id,
                ("decode",),
            )
            if source_job_id is not None and source_status not in {"running", "completed"}:
                return []
            purge_pending = conn.execute(
                """
                SELECT 1
                FROM run_purge_requests
                WHERE run_id = ?
                  AND status IN ('requested', 'canceling', 'purging')
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            existing = conn.execute(
                """
                SELECT rj.job_id, j.kind, j.status
                FROM run_jobs rj
                JOIN jobs j ON j.id = rj.job_id
                WHERE rj.run_id = ? AND rj.role = 'inference'
                ORDER BY rj.shard_index, rj.job_id
                """,
                (run_id,),
            ).fetchall()
            if existing:
                if (
                    str(run["status"]) not in {"queued", "decoding", "running"}
                    or run["deleted_at"] is not None
                    or run["artifact_cleaned_at"] is not None
                    or purge_pending is not None
                    or any(
                        str(row["kind"]) != "inference"
                        or str(row["status"]) not in {"queued", "running"}
                        for row in existing
                    )
                ):
                    return []
                if not self._complete_source_job_in_transaction(
                    conn,
                    source_job_id,
                    source_status,
                    source_job_result,
                    now,
                ):
                    raise RuntimeError(f"decode Job {source_job_id} rejected inference handoff completion")
                conn.execute(
                    "UPDATE runs SET status = 'queued', updated_at = ? WHERE id = ? AND status = 'decoding'",
                    (now, run_id),
                )
                return [int(row["job_id"]) for row in existing]

            # Historical Runs may point directly at their inference Job
            # without a corresponding ``run_jobs`` row. Treat a coherent,
            # active direct link as the already-published handoff instead of
            # creating a second inference Job. Backfill the binding so device
            # claims and all newer lifecycle checks see the same association.
            direct_job = None
            if run["inference_job_id"] is not None:
                direct_job = conn.execute(
                    "SELECT id, kind, status FROM jobs WHERE id = ?",
                    (int(run["inference_job_id"]),),
                ).fetchone()
            if direct_job is not None and (
                str(direct_job["kind"]) == "inference"
                and str(direct_job["status"]) in {"queued", "running"}
            ):
                if (
                    str(run["status"]) not in {"queued", "decoding", "running"}
                    or run["deleted_at"] is not None
                    or run["artifact_cleaned_at"] is not None
                    or purge_pending is not None
                ):
                    return []
                conn.execute(
                    """
                    INSERT OR IGNORE INTO run_jobs(
                        run_id, job_id, role, shard_index, device,
                        metadata_json, created_at
                    )
                    VALUES (?, ?, 'inference', 0, ?, '{}', ?)
                    """,
                    (run_id, int(direct_job["id"]), run["device"], now),
                )
                if not self._complete_source_job_in_transaction(
                    conn,
                    source_job_id,
                    source_status,
                    source_job_result,
                    now,
                ):
                    raise RuntimeError(f"decode Job {source_job_id} rejected inference handoff completion")
                conn.execute(
                    "UPDATE runs SET status = 'queued', updated_at = ? WHERE id = ? AND status = 'decoding'",
                    (now, run_id),
                )
                return [int(direct_job["id"])]

            if run["inference_job_id"] is not None:
                # A historical direct link is authoritative even when its Job
                # is no longer active. Never create a second inference Job for
                # the same Run from a partially migrated state.
                return []

            if (
                str(run["status"]) not in {"queued", "decoding"}
                or run["deleted_at"] is not None
                or run["artifact_cleaned_at"] is not None
                or purge_pending is not None
            ):
                return []

            job_ids: list[int] = []
            progress_total = 0
            for spec in specs:
                payload = dict(spec.get("payload") or {})
                payload.setdefault("run_id", run_id)
                total = int(spec.get("progress_total") or 0)
                cur = conn.execute(
                    """
                    INSERT INTO jobs(kind, status, payload_json, progress_total, created_at)
                    VALUES ('inference', 'queued', ?, ?, ?)
                    """,
                    (_json(payload), total, now),
                )
                job_id = int(cur.lastrowid)
                job_ids.append(job_id)
                progress_total += total
                conn.execute(
                    """
                    INSERT INTO run_jobs(
                        run_id, job_id, role, shard_index, device, metadata_json, created_at
                    )
                    VALUES (?, ?, 'inference', ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        job_id,
                        int(spec.get("shard_index") or 0),
                        spec.get("device"),
                        _json(spec.get("metadata") or {}),
                        now,
                    ),
                )

            cur = conn.execute(
                """
                UPDATE runs
                SET inference_job_id = COALESCE(inference_job_id, ?),
                    status = 'queued', progress_current = 0,
                    progress_total = ?, updated_at = ?
                WHERE id = ? AND status IN ('queued', 'decoding')
                """,
                (job_ids[0], progress_total, now, run_id),
            )
            if cur.rowcount != 1:
                raise RuntimeError(f"run {run_id} rejected inference job publication")
            if not self._complete_source_job_in_transaction(
                conn,
                source_job_id,
                source_status,
                source_job_result,
                now,
            ):
                raise RuntimeError(f"decode Job {source_job_id} rejected inference handoff completion")
            return job_ids

    def list_run_jobs(self, run_id: int, role: str | None = None) -> list[dict[str, Any]]:
        params: list[Any] = [run_id]
        clause = ""
        if role is not None:
            clause = " AND rj.role = ?"
            params.append(role)
        rows = self.query(
            f"""
            SELECT rj.*, j.kind, j.status, j.payload_json, j.worker_id,
                   j.progress_current, j.progress_total, j.result_json, j.error_json,
                   j.started_at, j.heartbeat_at, j.finished_at
            FROM run_jobs rj
            JOIN jobs j ON j.id = rj.job_id
            WHERE rj.run_id = ?{clause}
            ORDER BY rj.role, rj.shard_index, rj.job_id
            """,
            params,
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
            row["payload"] = _loads(row.pop("payload_json"))
            row["result"] = _loads(row.pop("result_json"))
            row["error"] = _loads(row.pop("error_json"))
        return rows

    def next_run_id(self) -> int:
        row = self.get("SELECT seq FROM sqlite_sequence WHERE name = 'runs'")
        if row is not None:
            return int(row["seq"]) + 1
        row = self.get("SELECT MAX(id) AS max_id FROM runs")
        max_id = int(row["max_id"] or 0) if row else 0
        return max_id + 1

    def list_runs(self, limit: int = 100, include_deleted: bool = False) -> list[dict[str, Any]]:
        deleted_clause = "" if include_deleted else "WHERE r.deleted_at IS NULL"
        rows = self.query(
            f"""
            SELECT
                r.*,
                e.name AS experiment_name,
                m.name AS model_name,
                d.name AS dataset_name
            FROM runs r
            LEFT JOIN experiments e ON e.id = r.experiment_id
            JOIN models m ON m.id = r.model_id
            JOIN datasets d ON d.id = r.dataset_id
            {deleted_clause}
            ORDER BY r.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        for row in rows:
            self._decode_run(row)
            row["purge_request"] = self.get_run_purge_request(int(row["id"]))
        return rows

    def list_runs_page(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        query: str = "",
        status: str = "",
        run_type: str = "",
        model: str = "",
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        page = max(1, int(page or 1))
        page_size = max(1, min(200, int(page_size or 50)))
        clauses = [] if include_deleted else ["r.deleted_at IS NULL"]
        params: list[Any] = []
        query = str(query or "").strip()
        if query:
            pattern = f"%{query}%"
            clauses.append(
                "(r.name LIKE ? OR m.name LIKE ? OR d.name LIKE ? "
                "OR CAST(r.id AS TEXT) = ?)"
            )
            params.extend((pattern, pattern, pattern, query.lstrip("#")))
        status = str(status or "").strip()
        if status:
            clauses.append("r.status = ?")
            params.append(status)
        run_type = str(run_type or "").strip()
        if run_type:
            clauses.append(
                "COALESCE(json_extract(r.metadata_json, '$.run_type'), 'model_inference') = ?"
            )
            params.append(run_type)
        model = str(model or "").strip()
        if model:
            clauses.append("m.name = ?")
            params.append(model)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total_row = self.get(
            f"""
            SELECT COUNT(*) AS total
            FROM runs r
            JOIN models m ON m.id = r.model_id
            JOIN datasets d ON d.id = r.dataset_id
            {where}
            """,
            tuple(params),
        ) or {"total": 0}
        total = int(total_row["total"] or 0)
        active_row = self.get(
            """
            SELECT COUNT(*) AS total
            FROM runs r
            WHERE r.deleted_at IS NULL
              AND (
                    r.status NOT IN ('completed', 'failed', 'canceled')
                    OR EXISTS (
                        SELECT 1
                        FROM run_purge_requests pr
                        WHERE pr.run_id = r.id
                          AND pr.status IN ('requested', 'canceling', 'purging')
                    )
                  )
            """
        ) or {"total": 0}
        page_count = max(1, (total + page_size - 1) // page_size)
        page = min(page, page_count)
        rows = self.query(
            f"""
            SELECT
                r.*,
                e.name AS experiment_name,
                m.name AS model_name,
                d.name AS dataset_name
            FROM runs r
            LEFT JOIN experiments e ON e.id = r.experiment_id
            JOIN models m ON m.id = r.model_id
            JOIN datasets d ON d.id = r.dataset_id
            {where}
            ORDER BY r.id DESC
            LIMIT ? OFFSET ?
            """,
            tuple([*params, page_size, (page - 1) * page_size]),
        )
        for row in rows:
            self._decode_run(row)
            row["purge_request"] = self.get_run_purge_request(int(row["id"]))
        return {
            "runs": rows,
            "page": page,
            "page_size": page_size,
            "page_count": page_count,
            "total": total,
            # This deliberately ignores the current page and filters so the UI
            # keeps its fast poll while background work exists elsewhere.
            "active_total": int(active_row["total"] or 0),
            "query": query,
            "filters": {
                "status": status,
                "run_type": run_type,
                "model": model,
            },
        }

    def list_run_associated_jobs(
        self,
        run_id: int,
        statuses: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List Jobs linked through current bindings or legacy Run identity.

        Historical databases may lack ``run_jobs`` rows while still linking a
        Job through ``runs.inference_job_id``, ``runs.metric_job_id``, or the
        Job payload's ``run_id``.  Lifecycle and cleanup decisions must see
        the same complete association set as cancellation and Job CAS fences.
        """
        requested = tuple(sorted({str(status) for status in (statuses or [])}))
        status_clause = ""
        params: list[Any] = [int(run_id), int(run_id), int(run_id)]
        if requested:
            placeholders = ",".join("?" for _ in requested)
            status_clause = f" AND j.status IN ({placeholders})"
            params.extend(requested)
        rows = self.query(
            f"""
            SELECT DISTINCT j.*, j.id AS job_id
            FROM jobs j
            JOIN runs r ON r.id = ?
            WHERE (
                    EXISTS (
                        SELECT 1 FROM run_jobs rj
                        WHERE rj.run_id = ? AND rj.job_id = j.id
                    )
                    OR j.id = r.inference_job_id
                    OR j.id = r.metric_job_id
                    OR CAST(json_extract(j.payload_json, '$.run_id') AS INTEGER) = ?
                  )
                  {status_clause}
            ORDER BY j.created_at, j.id
            """,
            params,
        )
        for row in rows:
            self._decode_job(row)
        return rows

    def get_run(self, run_id: int) -> dict[str, Any]:
        row = self.get(
            """
            SELECT
                r.*,
                e.name AS experiment_name,
                m.name AS model_name,
                d.name AS dataset_name
            FROM runs r
            LEFT JOIN experiments e ON e.id = r.experiment_id
            JOIN models m ON m.id = r.model_id
            JOIN datasets d ON d.id = r.dataset_id
            WHERE r.id = ?
            """,
            (run_id,),
        )
        if row is None:
            raise KeyError(f"run {run_id} not found")
        self._decode_run(row)
        row["purge_request"] = self.get_run_purge_request(run_id)
        return row

    def get_run_by_job(self, job_id: int) -> dict[str, Any] | None:
        row = self.get(
            """
            SELECT
                r.*,
                e.name AS experiment_name,
                m.name AS model_name,
                d.name AS dataset_name
            FROM runs r
            LEFT JOIN experiments e ON e.id = r.experiment_id
            JOIN models m ON m.id = r.model_id
            JOIN datasets d ON d.id = r.dataset_id
            LEFT JOIN run_jobs rj ON rj.run_id = r.id
            WHERE r.inference_job_id = ? OR r.metric_job_id = ?
               OR rj.job_id = ?
            """,
            (job_id, job_id, job_id),
        )
        if row is None:
            return None
        self._decode_run(row)
        row["purge_request"] = self.get_run_purge_request(int(row["id"]))
        return row

    def update_run_progress(
        self,
        run_id: int,
        current: int,
        total: int | None = None,
        status: str | None = None,
    ) -> bool:
        """Publish progress without changing the Run lifecycle state.

        ``status`` remains accepted for source compatibility, but lifecycle
        changes must use one of the guarded transition methods.  This prevents
        late save/metric callbacks from reviving a canceled or terminal Run.
        """
        del status
        now = utc_ts()
        with self.connection() as conn:
            if total is None:
                cur = conn.execute(
                    """
                    UPDATE runs
                    SET progress_current = ?, updated_at = ?
                    WHERE id = ?
                      AND status NOT IN ('completed', 'failed', 'cancel_requested', 'canceled')
                    """,
                    (int(current), now, run_id),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE runs
                    SET progress_current = ?, progress_total = ?, updated_at = ?
                    WHERE id = ?
                      AND status NOT IN ('completed', 'failed', 'cancel_requested', 'canceled')
                    """,
                    (int(current), int(total), now, run_id),
                )
            return bool(cur.rowcount)

    def merge_run_result(self, run_id: int, patch: dict[str, Any]) -> bool:
        """Merge diagnostic result details without changing lifecycle state."""
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status, result_json FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"run {run_id} not found")
            if str(row["status"]) in RUN_NON_PROGRESS_STATUSES:
                return False
            result = dict(_loads(row["result_json"]) or {})
            result.update(dict(patch or {}))
            cur = conn.execute(
                """
                UPDATE runs
                SET result_json = ?, content_revision = content_revision + 1,
                    updated_at = ?
                WHERE id = ?
                  AND status NOT IN ('completed', 'failed', 'cancel_requested', 'canceled')
                """,
                (_json(result), now, run_id),
            )
            return bool(cur.rowcount)

    def mark_run_started(self, run_id: int, status: str = "running") -> bool:
        sources = RUN_START_SOURCES.get(status)
        if sources is None:
            raise ValueError(f"unsupported active Run status: {status}")
        placeholders = ",".join("?" for _ in sources)
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                f"""
                UPDATE runs
                SET status = ?, started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE id = ? AND status IN ({placeholders})
                """,
                (status, now, now, run_id, *sources),
            )
            return bool(cur.rowcount)

    def set_run_metric_job(self, run_id: int, metric_job_id: int) -> bool:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute(
                """
                SELECT status, deleted_at, artifact_cleaned_at
                FROM runs
                WHERE id = ?
                """,
                (int(run_id),),
            ).fetchone()
            if run is None:
                raise KeyError(f"run {run_id} not found")
            if (
                str(run["status"] or "") not in {"running", "finalizing"}
                or run["deleted_at"] is not None
                or run["artifact_cleaned_at"] is not None
            ):
                return False
            purge_pending = conn.execute(
                """
                SELECT 1
                FROM run_purge_requests
                WHERE run_id = ?
                  AND status IN ('requested', 'canceling', 'purging')
                LIMIT 1
                """,
                (int(run_id),),
            ).fetchone()
            if purge_pending is not None:
                return False
            job = conn.execute(
                "SELECT kind, status, payload_json FROM jobs WHERE id = ?",
                (metric_job_id,),
            ).fetchone()
            if job is None:
                raise KeyError(f"job {metric_job_id} not found")
            if str(job["kind"]) != "metric":
                raise ValueError(f"job {metric_job_id} is not a metric job")
            if str(job["status"] or "") != "queued":
                return False
            payload = _loads(job["payload_json"])
            payload_run_id = payload.get("run_id") if isinstance(payload, dict) else None
            if payload_run_id is not None and int(payload_run_id) != int(run_id):
                return False
            foreign_owner = conn.execute(
                """
                SELECT 1
                FROM runs owner
                WHERE owner.id != ?
                  AND (
                      owner.inference_job_id = ?
                      OR owner.metric_job_id = ?
                      OR EXISTS (
                          SELECT 1 FROM run_jobs rj
                          WHERE rj.job_id = ? AND rj.run_id = owner.id
                      )
                  )
                LIMIT 1
                """,
                (int(run_id), int(metric_job_id), int(metric_job_id), int(metric_job_id)),
            ).fetchone()
            if foreign_owner is not None:
                return False
            other_active_metric = conn.execute(
                """
                SELECT 1
                FROM jobs active
                JOIN runs owner ON owner.id = ?
                WHERE active.id != ?
                  AND active.kind = 'metric'
                  AND active.status IN ('queued', 'running')
                  AND (
                      active.id = owner.metric_job_id
                      OR EXISTS (
                          SELECT 1 FROM run_jobs rj
                          WHERE rj.run_id = owner.id AND rj.job_id = active.id
                      )
                      OR CAST(json_extract(active.payload_json, '$.run_id') AS INTEGER) = owner.id
                  )
                LIMIT 1
                """,
                (int(run_id), int(metric_job_id)),
            ).fetchone()
            if other_active_metric is not None:
                return False
            conn.execute(
                """
                INSERT OR IGNORE INTO run_jobs(run_id, job_id, role, shard_index, device, metadata_json, created_at)
                VALUES (?, ?, 'metric', 0, NULL, '{}', ?)
                """,
                (run_id, metric_job_id, now),
            )
            cur = conn.execute(
                """
                UPDATE runs
                SET metric_job_id = ?, status = 'metric_queued', updated_at = ?
                WHERE id = ?
                  AND status IN ('running', 'finalizing')
                """,
                (metric_job_id, now, run_id),
            )
            if cur.rowcount:
                return True

            # A cancellation can win after metric jobs are constructed but
            # before the leader is published. Fence every queued job in that
            # wave so it cannot remain as an orphaned claimable job.
            wave_id = str(payload.get("metric_wave_id") or "")
            if wave_id:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'canceled', error_json = ?, finished_at = ?
                    WHERE status = 'queued'
                      AND id IN (
                          SELECT rj.job_id
                          FROM run_jobs rj
                          JOIN jobs j ON j.id = rj.job_id
                          WHERE rj.run_id = ? AND rj.role = 'metric'
                            AND json_extract(j.payload_json, '$.metric_wave_id') = ?
                      )
                    """,
                    (
                        _json({"message": "Run state rejected metric publication", "type": "RunCanceled"}),
                        now,
                        run_id,
                        wave_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'canceled', error_json = ?, finished_at = ?
                    WHERE id = ? AND status = 'queued'
                    """,
                    (
                        _json({"message": "Run state rejected metric publication", "type": "RunCanceled"}),
                        now,
                        metric_job_id,
                    ),
                )
            return False

    def publish_metric_wave(
        self,
        run_id: int,
        job_specs: Iterable[dict[str, Any]],
        *,
        retry: bool,
        result: dict[str, Any] | None = None,
        artifact_summary: dict[str, Any] | None = None,
        expected_content_revision: int | None = None,
        source_job_id: int | None = None,
        source_job_result: dict[str, Any] | None = None,
    ) -> list[int]:
        """Atomically publish a complete metric wave and its Run phase.

        Jobs are not visible to workers until every shard, its Run binding,
        the leader id, and ``metric_queued`` commit together. A cancellation
        that commits first therefore leaves no orphaned claimable jobs.
        """
        specs = list(job_specs)
        if not specs:
            raise ValueError("metric wave requires at least one job")
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute(
                """
                SELECT status, metric_job_id, result_json,
                       artifact_summary_json, content_revision,
                       deleted_at, artifact_cleaned_at
                FROM runs WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"run {run_id} not found")
            source_job_status = self._source_job_status(
                conn,
                run_id,
                source_job_id,
                ("inference", "finalize"),
            )
            if source_job_id is not None and source_job_status not in {"running", "completed"}:
                return []
            allowed = {"completed", "failed"} if retry else {"running", "finalizing"}
            source_status = str(run["status"])
            if source_status not in allowed:
                return []
            if retry:
                if (
                    expected_content_revision is None
                    or int(run["content_revision"] or 0) != int(expected_content_revision)
                    or run["deleted_at"] is not None
                    or run["artifact_cleaned_at"] is not None
                ):
                    return []
                purge = conn.execute(
                    """
                    SELECT 1 FROM run_purge_requests
                    WHERE run_id = ? AND status IN ('requested', 'canceling', 'purging')
                    LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()
                if purge is not None:
                    return []
                incomplete = conn.execute(
                    """
                    SELECT 1
                    FROM run_jobs rj
                    JOIN jobs j ON j.id = rj.job_id
                    WHERE rj.run_id = ?
                      AND rj.role IN ('inference', 'finalize')
                      AND j.status != 'completed'
                    LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()
                if incomplete is not None:
                    return []
            active = conn.execute(
                """
                SELECT 1
                FROM run_jobs rj
                JOIN jobs j ON j.id = rj.job_id
                WHERE rj.run_id = ? AND rj.role = 'metric'
                  AND j.status IN ('queued', 'running')
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            if active is None and run["metric_job_id"] is not None:
                active = conn.execute(
                    """
                    SELECT 1
                    FROM jobs
                    WHERE id = ? AND kind = 'metric'
                      AND status IN ('queued', 'running')
                    LIMIT 1
                    """,
                    (int(run["metric_job_id"]),),
                ).fetchone()
            if active is not None:
                raise ValueError("Run already has an active metric evaluation")

            job_ids: list[int] = []
            progress_total = 0
            for spec in specs:
                payload = dict(spec.get("payload") or {})
                payload.setdefault("run_id", int(run_id))
                payload["retry"] = bool(retry)
                total = int(spec.get("progress_total") or 0)
                cur = conn.execute(
                    """
                    INSERT INTO jobs(kind, status, payload_json, progress_total, created_at)
                    VALUES ('metric', 'queued', ?, ?, ?)
                    """,
                    (_json(payload), total, now),
                )
                job_id = int(cur.lastrowid)
                job_ids.append(job_id)
                progress_total += total
                conn.execute(
                    """
                    INSERT INTO run_jobs(
                        run_id, job_id, role, shard_index, device, metadata_json, created_at
                    )
                    VALUES (?, ?, 'metric', ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        job_id,
                        int(spec.get("shard_index") or 0),
                        spec.get("device"),
                        _json(spec.get("metadata") or {}),
                        now,
                    ),
                )

            result_json = _json(result) if result is not None else str(run["result_json"] or "{}")
            artifact_json = (
                _json(artifact_summary)
                if artifact_summary is not None
                else str(run["artifact_summary_json"] or "{}")
            )
            cur = conn.execute(
                """
                UPDATE runs
                SET metric_job_id = ?, status = 'metric_queued',
                    result_json = ?, artifact_summary_json = ?,
                    progress_current = 0, progress_total = ?,
                    error_json = CASE WHEN ? THEN '{}' ELSE error_json END,
                    finished_at = NULL,
                    content_revision = content_revision + 1,
                    updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    job_ids[0],
                    result_json,
                    artifact_json,
                    progress_total,
                    1 if retry else 0,
                    now,
                    run_id,
                    source_status,
                ),
            )
            if cur.rowcount != 1:
                raise RuntimeError(f"run {run_id} rejected metric wave publication")
            if not self._complete_source_job_in_transaction(
                conn,
                source_job_id,
                source_job_status,
                source_job_result,
                now,
            ):
                raise RuntimeError(f"source Job {source_job_id} rejected metric wave completion")
            return job_ids

    def complete_run_inference(
        self,
        run_id: int,
        result: dict[str, Any],
        artifact_summary: dict[str, Any],
        status: str,
        *,
        source_job_id: int | None = None,
        source_job_result: dict[str, Any] | None = None,
    ) -> bool:
        sources = RUN_INFERENCE_COMPLETION_SOURCES.get(status)
        if sources is None:
            raise ValueError(f"unsupported inference completion status: {status}")
        placeholders = ",".join("?" for _ in sources)
        now = utc_ts()
        finished_at = now if status == "completed" else None
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            source_job_status = self._source_job_status(
                conn,
                run_id,
                source_job_id,
                ("inference", "finalize"),
            )
            if source_job_id is not None and source_job_status not in {"running", "completed"}:
                return False
            cur = conn.execute(
                f"""
                UPDATE runs
                SET status = ?,
                    result_json = ?,
                    artifact_summary_json = ?,
                    content_revision = content_revision + 1,
                    finished_at = COALESCE(?, finished_at),
                    updated_at = ?
                WHERE id = ? AND status IN ({placeholders})
                """,
                (status, _json(result), _json(artifact_summary), finished_at, now, run_id, *sources),
            )
            if not cur.rowcount:
                return False
            if not self._complete_source_job_in_transaction(
                conn,
                source_job_id,
                source_job_status,
                source_job_result,
                now,
            ):
                conn.rollback()
                return False
            return True

    def complete_run_metrics(
        self,
        run_id: int,
        metric_summary: dict[str, Any],
        *,
        source_job_id: int | None = None,
        source_job_result: dict[str, Any] | None = None,
    ) -> bool:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            source_job_status = self._source_job_status(
                conn,
                run_id,
                source_job_id,
                ("metric",),
            )
            if source_job_id is not None and source_job_status not in {"running", "completed"}:
                return False
            cur = conn.execute(
                """
                UPDATE runs
                SET status = 'completed',
                    metric_summary_json = ?,
                    content_revision = content_revision + 1,
                    finished_at = ?,
                    updated_at = ?
                WHERE id = ? AND status = 'metric_running'
                """,
                (_json(metric_summary), now, now, run_id),
            )
            if not cur.rowcount:
                return False
            if not self._complete_source_job_in_transaction(
                conn,
                source_job_id,
                source_job_status,
                source_job_result,
                now,
            ):
                conn.rollback()
                return False
            return True

    def complete_run_metric_wave(
        self,
        run_id: int,
        leader_job_id: int,
        metric_summary: dict[str, Any],
        performance: list[dict[str, Any]],
        *,
        source_job_id: int | None = None,
        source_job_result: dict[str, Any] | None = None,
    ) -> bool:
        """Publish a metric wave once even when shards finish concurrently."""
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            source_job_status = self._source_job_status(
                conn,
                run_id,
                source_job_id,
                ("metric",),
            )
            if source_job_id is not None and source_job_status not in {"running", "completed"}:
                return False
            leader = conn.execute(
                "SELECT kind, payload_json FROM jobs WHERE id = ?",
                (int(leader_job_id),),
            ).fetchone()
            if leader is None or str(leader["kind"] or "") != "metric":
                return False
            leader_payload = _loads(leader["payload_json"])
            wave_id = str(leader_payload.get("metric_wave_id") or "")
            if not wave_id:
                return False
            if source_job_id is not None:
                source = conn.execute(
                    "SELECT payload_json FROM jobs WHERE id = ?",
                    (int(source_job_id),),
                ).fetchone()
                source_wave_id = str(
                    (_loads(source["payload_json"]) if source is not None else {}).get("metric_wave_id")
                    or ""
                )
                if source_wave_id != wave_id:
                    return False
            if source_job_id is not None and source_job_status == "running":
                other_incomplete = conn.execute(
                    """
                    SELECT 1
                    FROM run_jobs rj
                    JOIN jobs j ON j.id = rj.job_id
                    WHERE rj.run_id = ? AND rj.role = 'metric'
                      AND json_extract(j.payload_json, '$.metric_wave_id') = ?
                      AND j.id != ? AND j.status != 'completed'
                    LIMIT 1
                    """,
                    (run_id, wave_id, int(source_job_id)),
                ).fetchone()
                if other_incomplete is not None:
                    return False
            row = conn.execute("SELECT result_json FROM runs WHERE id = ?", (run_id,)).fetchone()
            result = _loads(row["result_json"]) if row is not None else {}
            result["metric_performance"] = performance
            cur = conn.execute(
                """
                UPDATE runs
                SET status = 'completed', metric_summary_json = ?, result_json = ?,
                    content_revision = content_revision + 1, finished_at = ?, updated_at = ?
                WHERE id = ? AND metric_job_id = ?
                  AND status = 'metric_running'
                """,
                (_json(metric_summary), _json(result), now, now, run_id, leader_job_id),
            )
            if not cur.rowcount:
                return False
            if not self._complete_source_job_in_transaction(
                conn,
                source_job_id,
                source_job_status,
                source_job_result,
                now,
            ):
                conn.rollback()
                return False
            return True

    def update_run_metric_wave_progress(
        self,
        run_id: int,
        leader_job_id: int,
        current: int,
        total: int,
    ) -> bool:
        """Update active-wave progress without regressing a completed Run."""
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                UPDATE runs
                SET progress_current = ?, progress_total = ?, updated_at = ?
                WHERE id = ? AND metric_job_id = ?
                  AND status = 'metric_running'
                """,
                (int(current), int(total), now, run_id, leader_job_id),
            )
            return bool(cur.rowcount)

    def bump_run_content_revision(self, run_id: int) -> int:
        """Invalidate all client-side result caches for a Run.

        The revision is deliberately monotonic and updated in SQLite together
        with the state change that made cached result payloads stale.
        """
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE runs
                SET content_revision = content_revision + 1,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, run_id),
            )
            row = conn.execute("SELECT content_revision FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run {run_id} not found")
        return int(row["content_revision"])

    def fail_run(self, run_id: int, error: dict[str, Any]) -> bool:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            previous = conn.execute(
                "SELECT status FROM runs WHERE id = ?",
                (int(run_id),),
            ).fetchone()
            metric_phase = previous is not None and str(previous["status"] or "") in {
                "metric_queued",
                "metric_running",
            }
            cur = conn.execute(
                """
                UPDATE runs
                SET status = 'failed',
                    error_json = ?,
                    content_revision = content_revision + 1,
                    finished_at = ?,
                    updated_at = ?
                WHERE id = ?
                  AND status NOT IN ('completed', 'failed', 'cancel_requested', 'canceled')
                """,
                (_json(error), now, now, run_id),
            )
            if not cur.rowcount:
                return False
            # Cancel sibling shard jobs that have not started yet so a worker
            # never claims them once the run is already known to have failed.
            # Already-running shards notice via the run-status check each
            # batch (_raise_if_canceled) and stop themselves.
            conn.execute(
                """
                UPDATE jobs
                SET status = 'canceled',
                    error_json = ?,
                    finished_at = ?
                WHERE status = 'queued'
                  AND (
                      id IN (SELECT job_id FROM run_jobs WHERE run_id = ?)
                      OR id = (SELECT inference_job_id FROM runs WHERE id = ?)
                      OR id = (SELECT metric_job_id FROM runs WHERE id = ?)
                      OR CAST(json_extract(payload_json, '$.run_id') AS INTEGER) = ?
                  )
                """,
                (
                    _json({"message": "sibling shard failed the run"}),
                    now,
                    run_id,
                    run_id,
                    run_id,
                    run_id,
                ),
            )
            if not metric_phase:
                conn.execute(
                    """
                    UPDATE media_assets
                    SET state = 'unavailable', updated_at = ?
                    WHERE source_kind = 'run_artifact'
                      AND id IN (
                          SELECT asset_id FROM run_media_assets
                          WHERE run_id = ?
                            AND (
                                COALESCE(json_extract(metadata_json, '$.input'), 0) != 1
                                OR COALESCE(json_extract(metadata_json, '$.compare_snapshot'), 0) = 1
                            )
                      )
                    """,
                    (now, run_id),
                )
            return True

    def fail_claimed_job_and_run(
        self,
        job_id: int,
        run_id: int,
        error: dict[str, Any],
    ) -> bool:
        """Atomically fail an active Run and its currently claimed Job.

        Cancellation and failure are linearized by the ``BEGIN IMMEDIATE``
        writer lock. If cancellation committed first, this method changes
        neither object and the caller must converge cancellation instead.
        """
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            associated_job = conn.execute(
                """
                SELECT j.status, r.status AS run_status
                FROM jobs j
                JOIN runs r ON r.id = ?
                WHERE j.id = ?
                  AND (
                      EXISTS (
                          SELECT 1 FROM run_jobs rj
                          WHERE rj.run_id = r.id AND rj.job_id = j.id
                      )
                      OR j.id = r.inference_job_id
                      OR j.id = r.metric_job_id
                      OR CAST(json_extract(j.payload_json, '$.run_id') AS INTEGER) = r.id
                  )
                """,
                (int(run_id), int(job_id)),
            ).fetchone()
            if associated_job is None or str(associated_job["status"] or "") != "running":
                return False
            metric_phase = str(associated_job["run_status"] or "") in {
                "metric_queued",
                "metric_running",
            }
            cur = conn.execute(
                """
                UPDATE runs
                SET status = 'failed', error_json = ?,
                    content_revision = content_revision + 1,
                    finished_at = ?, updated_at = ?
                WHERE id = ?
                  AND status NOT IN ('completed', 'failed', 'cancel_requested', 'canceled')
                """,
                (_json(error), now, now, int(run_id)),
            )
            if cur.rowcount != 1:
                return False
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed', error_json = ?, finished_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (_json(error), now, int(job_id)),
            )
            conn.execute(
                """
                UPDATE jobs
                SET status = 'canceled', error_json = ?, finished_at = ?
                WHERE status = 'queued'
                  AND (
                      id IN (SELECT job_id FROM run_jobs WHERE run_id = ?)
                      OR id = (SELECT inference_job_id FROM runs WHERE id = ?)
                      OR id = (SELECT metric_job_id FROM runs WHERE id = ?)
                      OR CAST(json_extract(payload_json, '$.run_id') AS INTEGER) = ?
                  )
                """,
                (
                    _json({"message": "sibling shard failed the run", "type": "RunCanceled"}),
                    now,
                    int(run_id),
                    int(run_id),
                    int(run_id),
                    int(run_id),
                ),
            )
            if not metric_phase:
                conn.execute(
                    """
                    UPDATE media_assets
                    SET state = 'unavailable', updated_at = ?
                    WHERE source_kind = 'run_artifact'
                      AND id IN (
                          SELECT asset_id FROM run_media_assets
                          WHERE run_id = ?
                            AND (
                                COALESCE(json_extract(metadata_json, '$.input'), 0) != 1
                                OR COALESCE(json_extract(metadata_json, '$.compare_snapshot'), 0) = 1
                            )
                      )
                    """,
                    (now, int(run_id)),
                )
            return True

    def request_run_cancel(self, run_id: int) -> bool:
        """Atomically fence new work and request/finish Run cancellation."""
        now = utc_ts()
        error = {"message": "User canceled the Run", "type": "RunCanceled"}
        with self.connection() as conn:
            # A reserved writer lock makes job observation, queued-job
            # cancellation, and the Run transition one linearizable action.
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute(
                "SELECT status, inference_job_id, metric_job_id FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"run {run_id} not found")
            if str(run["status"]) in RUN_TERMINAL_STATUSES:
                return False

            conn.execute(
                """
                UPDATE jobs
                SET status = 'canceled', error_json = ?, finished_at = ?
                WHERE status = 'queued'
                  AND (
                      id IN (SELECT job_id FROM run_jobs WHERE run_id = ?)
                      OR id = ? OR id = ?
                      OR CAST(json_extract(payload_json, '$.run_id') AS INTEGER) = ?
                  )
                """,
                (
                    _json(error),
                    now,
                    run_id,
                    run["inference_job_id"],
                    run["metric_job_id"],
                    run_id,
                ),
            )
            running = conn.execute(
                """
                SELECT 1
                FROM jobs
                WHERE status = 'running'
                  AND (
                      id IN (SELECT job_id FROM run_jobs WHERE run_id = ?)
                      OR id = ? OR id = ?
                      OR CAST(json_extract(payload_json, '$.run_id') AS INTEGER) = ?
                  )
                LIMIT 1
                """,
                (run_id, run["inference_job_id"], run["metric_job_id"], run_id),
            ).fetchone()
            if running is None:
                cur = conn.execute(
                    """
                    UPDATE runs
                    SET status = 'canceled', error_json = ?,
                        content_revision = content_revision + 1,
                        finished_at = ?, updated_at = ?
                    WHERE id = ?
                      AND status NOT IN ('completed', 'failed', 'canceled')
                    """,
                    (_json(error), now, now, run_id),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE runs
                    SET status = 'cancel_requested', updated_at = ?
                    WHERE id = ?
                      AND status NOT IN ('completed', 'failed', 'canceled')
                    """,
                    (now, run_id),
                )
            if cur.rowcount:
                conn.execute(
                    """
                    UPDATE media_assets
                    SET state = 'unavailable', updated_at = ?
                    WHERE source_kind = 'run_artifact'
                      AND id IN (
                          SELECT asset_id FROM run_media_assets
                          WHERE run_id = ?
                            AND (
                                COALESCE(json_extract(metadata_json, '$.input'), 0) != 1
                                OR COALESCE(json_extract(metadata_json, '$.compare_snapshot'), 0) = 1
                            )
                      )
                    """,
                    (now, run_id),
                )
            return bool(cur.rowcount)

    def cancel_run(self, run_id: int, error: dict[str, Any] | None = None) -> bool:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                """
                UPDATE runs
                SET status = 'canceled',
                    error_json = ?,
                    content_revision = content_revision + 1,
                    finished_at = ?,
                    updated_at = ?
                WHERE id = ?
                  AND status NOT IN ('completed', 'failed', 'canceled')
                """,
                (_json(error or {"message": "Run canceled", "type": "RunCanceled"}), now, now, run_id),
            )
            if cur.rowcount:
                conn.execute(
                    """
                    UPDATE media_assets
                    SET state = 'unavailable', updated_at = ?
                    WHERE source_kind = 'run_artifact'
                      AND id IN (
                          SELECT asset_id FROM run_media_assets
                          WHERE run_id = ?
                            AND (
                                COALESCE(json_extract(metadata_json, '$.input'), 0) != 1
                                OR COALESCE(json_extract(metadata_json, '$.compare_snapshot'), 0) = 1
                            )
                      )
                    """,
                    (now, run_id),
                )
            return bool(cur.rowcount)

    def converge_run_cancellation(self, run_id: int, job_id: int | None = None) -> bool:
        """Stop active work after cancellation wins a Run state race."""
        now = utc_ts()
        error = {"message": "User canceled the Run", "type": "RunCanceled"}
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute(
                "SELECT status, inference_job_id, metric_job_id FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"run {run_id} not found")
            if str(run["status"]) not in {"cancel_requested", "canceled"}:
                return False

            associated_job_id: int | None = None
            if job_id is not None:
                associated = conn.execute(
                    """
                    SELECT 1
                    FROM jobs j
                    WHERE j.id = ?
                      AND (
                          j.id IN (SELECT job_id FROM run_jobs WHERE run_id = ?)
                          OR j.id = ? OR j.id = ?
                          OR CAST(json_extract(j.payload_json, '$.run_id') AS INTEGER) = ?
                      )
                    LIMIT 1
                    """,
                    (
                        int(job_id),
                        int(run_id),
                        run["inference_job_id"],
                        run["metric_job_id"],
                        int(run_id),
                    ),
                ).fetchone()
                if associated is not None:
                    associated_job_id = int(job_id)

            conn.execute(
                """
                UPDATE jobs
                SET status = 'canceled', error_json = ?, finished_at = ?
                WHERE (
                        status = 'queued'
                        OR (status = 'running' AND id = ?)
                      )
                      AND (
                      id IN (SELECT job_id FROM run_jobs WHERE run_id = ?)
                      OR id = ? OR id = ?
                      OR CAST(json_extract(payload_json, '$.run_id') AS INTEGER) = ?
                  )
                """,
                (
                    _json(error), now, associated_job_id, run_id,
                    run["inference_job_id"], run["metric_job_id"], run_id,
                ),
            )
            running = conn.execute(
                """
                SELECT 1 FROM jobs
                WHERE status = 'running'
                  AND (
                      id IN (SELECT job_id FROM run_jobs WHERE run_id = ?)
                      OR id = ? OR id = ?
                      OR CAST(json_extract(payload_json, '$.run_id') AS INTEGER) = ?
                  )
                LIMIT 1
                """,
                (run_id, run["inference_job_id"], run["metric_job_id"], run_id),
            ).fetchone()
            transitioned = False
            if running is None and str(run["status"]) == "cancel_requested":
                cur = conn.execute(
                    """
                    UPDATE runs
                    SET status = 'canceled', error_json = ?,
                        content_revision = content_revision + 1,
                        finished_at = ?, updated_at = ?
                    WHERE id = ? AND status = 'cancel_requested'
                    """,
                    (_json(error), now, now, run_id),
                )
                transitioned = bool(cur.rowcount)
            conn.execute(
                """
                UPDATE media_assets
                SET state = 'unavailable', updated_at = ?
                WHERE source_kind = 'run_artifact'
                  AND id IN (
                      SELECT asset_id FROM run_media_assets
                      WHERE run_id = ?
                        AND (
                            COALESCE(json_extract(metadata_json, '$.input'), 0) != 1
                            OR COALESCE(json_extract(metadata_json, '$.compare_snapshot'), 0) = 1
                        )
                  )
                """,
                (now, run_id),
            )
            return transitioned or str(run["status"]) == "canceled"

    def cancel_job(self, job_id: int, error: dict[str, Any] | None = None) -> bool:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'canceled',
                    error_json = ?,
                    finished_at = ?
                WHERE id = ? AND status IN ('queued', 'running')
                """,
                (_json(error or {"message": "Job canceled", "type": "RunCanceled"}), now, job_id),
            )
            return bool(cur.rowcount)

    def cancel_queued_run_jobs(self, run_id: int, reason: str = "Run cleanup requested") -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'canceled',
                    error_json = ?,
                    finished_at = ?
                WHERE status = 'queued'
                  AND (
                      id IN (SELECT job_id FROM run_jobs WHERE run_id = ?)
                      OR id = (SELECT inference_job_id FROM runs WHERE id = ?)
                      OR id = (SELECT metric_job_id FROM runs WHERE id = ?)
                      OR CAST(json_extract(payload_json, '$.run_id') AS INTEGER) = ?
                  )
                """,
                (
                    _json({"message": reason, "type": "RunCanceled"}),
                    now,
                    run_id,
                    run_id,
                    run_id,
                    run_id,
                ),
            )
            return int(cur.rowcount)

    def soft_delete_run(self, run_id: int) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE runs
                SET deleted_at = COALESCE(deleted_at, ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, run_id),
            )

    def rename_run(self, run_id: int, name: str) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE runs
                SET name = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (name, now, run_id),
            )

    def add_run_feedback(
        self,
        run_id: int,
        username: str,
        rating: float | None,
        issue: str,
        video: str = "",
        track_label: str = "",
        model_name: str = "",
        checkpoint: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO run_feedback(
                    run_id, username, rating, issue,
                    video, track_label, model_name, checkpoint,
                    metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    username,
                    float(rating) if rating is not None else None,
                    issue,
                    video,
                    track_label,
                    model_name,
                    checkpoint,
                    _json(metadata),
                    now,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def update_run_feedback(
        self,
        run_id: int,
        feedback_id: int,
        *,
        username: str | None = None,
        rating: float | None = None,
        issue: str | None = None,
        clear_rating: bool = False,
    ) -> bool:
        """Patch an existing feedback row. Only provided fields change.

        ``clear_rating=True`` explicitly nulls the rating (distinct from leaving
        ``rating`` as ``None`` to mean "unchanged").
        """
        sets: list[str] = []
        params: list[Any] = []
        if username is not None:
            sets.append("username = ?")
            params.append(username)
        if issue is not None:
            sets.append("issue = ?")
            params.append(issue)
        if clear_rating:
            sets.append("rating = NULL")
        elif rating is not None:
            sets.append("rating = ?")
            params.append(float(rating))
        if not sets:
            return False
        sets.append("updated_at = ?")
        params.append(utc_ts())
        params.extend([feedback_id, run_id])
        with self.connection() as conn:
            cur = conn.execute(
                f"UPDATE run_feedback SET {', '.join(sets)} WHERE id = ? AND run_id = ?",
                tuple(params),
            )
            return cur.rowcount > 0

    def list_run_feedback(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT * FROM run_feedback WHERE run_id = ? ORDER BY created_at DESC, id DESC",
            (run_id,),
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json", None))
        return rows

    def delete_run_feedback(self, run_id: int, feedback_id: int) -> bool:
        with self.connection() as conn:
            cur = conn.execute(
                "DELETE FROM run_feedback WHERE id = ? AND run_id = ?",
                (feedback_id, run_id),
            )
            return cur.rowcount > 0

    def list_all_feedback(self, limit: int = 1000, include_deleted: bool = False) -> list[dict[str, Any]]:
        """All feedback joined to its run + dataset context, newest first.

        Runs whose records were deleted are excluded by default so the stats tab
        only reflects live evaluations (matches the #26 "delete run → drop its
        feedback from stats" expectation). Cleaning artifacts cascades the delete
        at the DB level, but a soft-deleted run keeps its rows, so filter here.
        """
        deleted_clause = "" if include_deleted else "WHERE r.deleted_at IS NULL"
        rows = self.query(
            f"""
            SELECT
                f.*,
                r.name AS run_name,
                r.status AS run_status,
                r.deleted_at AS run_deleted_at,
                d.name AS dataset_name
            FROM run_feedback f
            LEFT JOIN runs r ON r.id = f.run_id
            LEFT JOIN datasets d ON d.id = r.dataset_id
            {deleted_clause}
            ORDER BY f.created_at DESC, f.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json", None))
        return rows

    def feedback_stats(
        self,
        *,
        dataset: str | None = None,
        model_name: str | None = None,
        checkpoint: str | None = None,
        video: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate feedback overall and grouped by user/run/video/model/checkpoint.

        Optional filters narrow the population before aggregation so the caller
        can ask e.g. "only this dataset" or "only this checkpoint". Ratings use a
        0.25 step, so the distribution histogram spans the 17 quarter-values from
        1.00 to 5.00.
        """
        return self._feedback_stats_sql(
            dataset=dataset,
            model_name=model_name,
            checkpoint=checkpoint,
            video=video,
        )

    def _feedback_sql_scope(
        self,
        *,
        dataset: str | None = None,
        model_name: str | None = None,
        checkpoint: str | None = None,
        video: str | None = None,
    ) -> tuple[str, list[Any]]:
        clauses = ["r.deleted_at IS NULL"]
        params: list[Any] = []
        for value, expression in (
            (dataset, "d.name"),
            (model_name, "f.model_name"),
            (checkpoint, "f.checkpoint"),
            (video, "f.video"),
        ):
            if value:
                clauses.append(f"{expression} = ?")
                params.append(str(value))
        return " AND ".join(clauses), params

    def _feedback_stats_sql(
        self,
        *,
        dataset: str | None = None,
        model_name: str | None = None,
        checkpoint: str | None = None,
        video: str | None = None,
    ) -> dict[str, Any]:
        where, params = self._feedback_sql_scope(
            dataset=dataset,
            model_name=model_name,
            checkpoint=checkpoint,
            video=video,
        )
        source = "run_feedback f JOIN runs r ON r.id = f.run_id JOIN datasets d ON d.id = r.dataset_id"
        summary = self.get(
            f"""
            SELECT COUNT(*) AS total,
                   COUNT(f.rating) AS rating_count,
                   ROUND(AVG(f.rating), 2) AS average_rating,
                   SUM(CASE WHEN TRIM(f.issue) != '' THEN 1 ELSE 0 END) AS issue_count
            FROM {source}
            WHERE {where}
            """,
            params,
        ) or {}
        distribution = {_rating_key(step / 4): 0 for step in range(4, 21)}
        for row in self.query(
            f"""
            SELECT printf('%.2f', f.rating) AS rating_key, COUNT(*) AS count
            FROM {source}
            WHERE {where} AND f.rating IS NOT NULL
            GROUP BY printf('%.2f', f.rating)
            """,
            params,
        ):
            if row["rating_key"] in distribution:
                distribution[row["rating_key"]] = int(row["count"])

        def grouped(
            fields: list[tuple[str, str]],
            *,
            extra: str = "",
            order: str,
        ) -> list[dict[str, Any]]:
            select_fields = ", ".join(f"{expression} AS {alias}" for alias, expression in fields)
            group_fields = ", ".join(expression for _alias, expression in fields)
            scoped = f"{where}{extra}"
            rows = self.query(
                f"""
                SELECT {select_fields}, COUNT(*) AS count,
                       COUNT(f.rating) AS rating_count,
                       ROUND(AVG(f.rating), 2) AS average_rating,
                       SUM(CASE WHEN TRIM(f.issue) != '' THEN 1 ELSE 0 END) AS issues
                FROM {source}
                WHERE {scoped}
                GROUP BY {group_fields}
                ORDER BY {order}
                """,
                params,
            )
            rating_rows = self.query(
                f"""
                SELECT {select_fields}, printf('%.2f', f.rating) AS rating_key, COUNT(*) AS count
                FROM {source}
                WHERE {scoped} AND f.rating IS NOT NULL
                GROUP BY {group_fields}, printf('%.2f', f.rating)
                """,
                params,
            )
            histograms: dict[tuple[Any, ...], dict[str, int]] = {}
            for rating_row in rating_rows:
                key = tuple(rating_row[alias] for alias, _expression in fields)
                histogram = histograms.setdefault(key, {_rating_key(step / 4): 0 for step in range(4, 21)})
                if rating_row["rating_key"] in histogram:
                    histogram[rating_row["rating_key"]] = int(rating_row["count"])
            for row in rows:
                key = tuple(row[alias] for alias, _expression in fields)
                row["rating_distribution"] = histograms.get(
                    key, {_rating_key(step / 4): 0 for step in range(4, 21)}
                )
                row["count"] = int(row["count"] or 0)
                row["rating_count"] = int(row["rating_count"] or 0)
                row["issues"] = int(row["issues"] or 0)
            return rows

        username = "CASE WHEN TRIM(f.username) = '' THEN '匿名' ELSE f.username END"
        checkpoint_expr = "CASE WHEN TRIM(f.checkpoint) = '' THEN '-' ELSE f.checkpoint END"
        return {
            "total": int(summary.get("total") or 0),
            "rating_count": int(summary.get("rating_count") or 0),
            "average_rating": summary.get("average_rating"),
            "issue_count": int(summary.get("issue_count") or 0),
            "rating_distribution": distribution,
            "by_user": grouped([("username", username)], order="count DESC, username"),
            "by_run": grouped(
                [("run_id", "f.run_id"), ("run_name", "r.name")], order="f.run_id DESC"
            ),
            "by_video": grouped(
                [("video", "f.video")], extra=" AND TRIM(f.video) != ''", order="count DESC, f.video"
            ),
            "by_model": grouped(
                [("model_name", "f.model_name")],
                extra=" AND TRIM(f.model_name) != ''",
                order="count DESC, f.model_name",
            ),
            "by_checkpoint": grouped(
                [("model_name", "f.model_name"), ("checkpoint", checkpoint_expr)],
                extra=" AND TRIM(f.model_name) != ''",
                order=f"f.model_name, {checkpoint_expr}",
            ),
            "by_model_checkpoint": grouped(
                [
                    ("model_name", "f.model_name"),
                    ("checkpoint", checkpoint_expr),
                    ("video", "f.video"),
                ],
                extra=" AND TRIM(f.model_name) != ''",
                order=f"f.video, f.model_name, {checkpoint_expr}",
            ),
        }

    def feedback_filter_options(self) -> dict[str, list[str]]:
        """Distinct datasets/models/checkpoints/videos present in live feedback.

        Powers the stats-tab filter dropdowns without the frontend having to
        derive them from the full entry list.
        """
        source = "run_feedback f JOIN runs r ON r.id = f.run_id JOIN datasets d ON d.id = r.dataset_id"

        def values(expression: str) -> list[str]:
            rows = self.query(
                f"""
                SELECT DISTINCT {expression} AS value
                FROM {source}
                WHERE r.deleted_at IS NULL AND TRIM({expression}) != ''
                ORDER BY value
                """
            )
            return [str(row["value"]) for row in rows]

        return {
            "datasets": values("d.name"),
            "models": values("f.model_name"),
            "checkpoints": values("f.checkpoint"),
            "videos": values("f.video"),
        }

    def list_recent_feedback(
        self,
        *,
        limit: int = 100,
        dataset: str | None = None,
        model_name: str | None = None,
        checkpoint: str | None = None,
        video: str | None = None,
    ) -> list[dict[str, Any]]:
        where, params = self._feedback_sql_scope(
            dataset=dataset,
            model_name=model_name,
            checkpoint=checkpoint,
            video=video,
        )
        rows = self.query(
            f"""
            SELECT f.*, r.name AS run_name, r.status AS run_status,
                   r.deleted_at AS run_deleted_at, d.name AS dataset_name
            FROM run_feedback f
            JOIN runs r ON r.id = f.run_id
            JOIN datasets d ON d.id = r.dataset_id
            WHERE {where}
            ORDER BY f.created_at DESC, f.id DESC
            LIMIT ?
            """,
            (*params, min(1000, max(1, int(limit)))),
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json", None))
        return rows

    def mark_run_artifacts_cleaned(self, run_id: int) -> None:
        now = utc_ts()
        job_ids = self.run_inference_job_ids(run_id)
        with self.connection() as conn:
            if job_ids:
                placeholders = ",".join("?" for _ in job_ids)
                conn.execute(f"DELETE FROM artifacts WHERE job_id IN ({placeholders})", tuple(job_ids))
            # Feedback describes the visual results; once those are cleaned the
            # scores are orphaned (the #26 case: results deleted but ratings
            # lingered), so drop them alongside the artifacts.
            conn.execute("DELETE FROM run_feedback WHERE run_id = ?", (run_id,))
            conn.execute(
                """
                UPDATE media_assets
                SET state = 'unavailable', updated_at = ?
                WHERE source_kind = 'run_artifact'
                  AND id IN (
                      SELECT asset_id FROM run_media_assets
                      WHERE run_id = ?
                        AND (
                            COALESCE(json_extract(metadata_json, '$.input'), 0) != 1
                            OR COALESCE(json_extract(metadata_json, '$.compare_snapshot'), 0) = 1
                        )
                  )
                """,
                (now, run_id),
            )
            conn.execute(
                """
                UPDATE runs
                SET artifact_cleaned_at = COALESCE(artifact_cleaned_at, ?),
                    artifact_summary_json = ?,
                    content_revision = content_revision + 1,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    now,
                    _json(
                        {
                            "total": 0,
                            "by_kind": {},
                            "storage_bytes": 0,
                            "storage_bytes_by_kind": {},
                            "storage_size_known": 0,
                            "storage_size_unknown": 0,
                        }
                    ),
                    now,
                    run_id,
                ),
            )

    def mark_run_deleted_after_purge(self, run_id: int) -> None:
        """Hide a Run only after its managed output cleanup has succeeded."""
        now = utc_ts()
        with self.connection() as conn:
            row = conn.execute(
                "SELECT artifact_cleaned_at FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"run {run_id} not found")
            if row["artifact_cleaned_at"] is None:
                raise ValueError("run artifacts must be cleaned before the run can be deleted")
            conn.execute(
                """
                UPDATE runs
                SET deleted_at = COALESCE(deleted_at, ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, run_id),
            )

    def request_run_purge(self, run_id: int, request_type: str) -> dict[str, Any]:
        if request_type not in {"delete_run", "cleanup_artifacts"}:
            raise ValueError("request_type must be delete_run or cleanup_artifacts")
        now = utc_ts()
        with self.connection() as conn:
            if conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
                raise KeyError(f"run {run_id} not found")
            row = conn.execute(
                "SELECT * FROM run_purge_requests WHERE run_id = ? AND request_type = ?",
                (run_id, request_type),
            ).fetchone()
            if row is None:
                cur = conn.execute(
                    """
                    INSERT INTO run_purge_requests(
                        run_id, request_type, status, requested_at, updated_at
                    )
                    VALUES (?, ?, 'requested', ?, ?)
                    """,
                    (run_id, request_type, now, now),
                )
                request_id = int(cur.lastrowid)
            else:
                request_id = int(row["id"])
                if str(row["status"]) == "failed":
                    conn.execute(
                        """
                        UPDATE run_purge_requests
                        SET status = 'requested',
                            report_json = '{}',
                            error_json = '{}',
                            claim_token = '',
                            reclaimed_bytes = 0,
                            requested_at = ?,
                            started_at = NULL,
                            completed_at = NULL,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (now, now, request_id),
                    )
        return self.get_run_purge_request_by_id(request_id)

    def get_run_purge_request_by_id(self, request_id: int) -> dict[str, Any]:
        row = self.get("SELECT * FROM run_purge_requests WHERE id = ?", (request_id,))
        if row is None:
            raise KeyError(f"run purge request {request_id} not found")
        self._decode_run_purge_request(row)
        return row

    def get_run_purge_request(
        self,
        run_id: int,
        request_type: str | None = None,
    ) -> dict[str, Any] | None:
        if request_type is None:
            row = self.get(
                "SELECT * FROM run_purge_requests WHERE run_id = ? ORDER BY id DESC LIMIT 1",
                (run_id,),
            )
        else:
            row = self.get(
                "SELECT * FROM run_purge_requests WHERE run_id = ? AND request_type = ?",
                (run_id, request_type),
            )
        if row is not None:
            self._decode_run_purge_request(row)
        return row

    def list_pending_run_purge_requests(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.query(
            """
            SELECT * FROM run_purge_requests
            WHERE status IN ('requested', 'canceling')
            ORDER BY requested_at, id
            LIMIT ?
            """,
            (min(1000, max(1, int(limit))),),
        )
        for row in rows:
            self._decode_run_purge_request(row)
        return rows

    def cleanup_backlog_counts(self) -> dict[str, Any]:
        """Return durable cleanup queue counts without claiming or retrying work."""

        run_statuses = ("requested", "canceling", "purging", "failed")
        campaign_statuses = ("requested", "running", "failed")
        with self.connection() as conn:
            run_rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM run_purge_requests
                WHERE status != 'completed'
                GROUP BY status
                """
            ).fetchall()
            campaign_table = conn.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'evaluation_purge_requests_v2'
                """
            ).fetchone()
            campaign_rows = (
                conn.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM evaluation_purge_requests_v2
                    WHERE status != 'completed'
                    GROUP BY status
                    """
                ).fetchall()
                if campaign_table is not None
                else []
            )

        run_by_status = {status: 0 for status in run_statuses}
        for row in run_rows:
            run_by_status[str(row["status"])] = int(row["count"] or 0)
        campaign_by_status = {status: 0 for status in campaign_statuses}
        for row in campaign_rows:
            campaign_by_status[str(row["status"])] = int(row["count"] or 0)
        return {
            "run_cleanup": {
                "backlog": sum(run_by_status.values()),
                "by_status": run_by_status,
            },
            "campaign_cleanup": {
                "backlog": sum(campaign_by_status.values()),
                "by_status": campaign_by_status,
            },
        }

    def recover_stale_run_purge_requests(self, stale_before: float) -> int:
        """Return abandoned purge claims to the durable queue after a crash."""
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                UPDATE run_purge_requests
                SET status = 'requested',
                    error_json = ?,
                    claim_token = '',
                    updated_at = ?
                WHERE status = 'purging' AND updated_at < ?
                """,
                (
                    _json({"type": "InterruptedPurge", "message": "resuming an interrupted purge request"}),
                    now,
                    float(stale_before),
                ),
            )
            return int(cur.rowcount)

    def update_run_purge_request(
        self,
        request_id: int,
        status: str,
        *,
        report: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        reclaimed_bytes: int | None = None,
        expected_claim_token: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"requested", "canceling", "purging", "failed", "completed"}:
            raise ValueError(f"unsupported purge status: {status}")
        now = utc_ts()
        started = now if status == "purging" else None
        completed = now if status == "completed" else None
        where = "WHERE id = ?"
        params: list[Any] = [
            status,
            _json(report) if report is not None else None,
            _json(error) if error is not None else None,
            int(reclaimed_bytes) if reclaimed_bytes is not None else None,
            started,
            started,
            completed,
            completed,
            status,
            now,
        ]
        if expected_claim_token is not None:
            where += " AND claim_token = ?"
        params.append(request_id)
        if expected_claim_token is not None:
            params.append(str(expected_claim_token))
        with self.connection() as conn:
            cur = conn.execute(
                f"""
                UPDATE run_purge_requests
                SET status = ?,
                    report_json = COALESCE(?, report_json),
                    error_json = COALESCE(?, error_json),
                    reclaimed_bytes = COALESCE(?, reclaimed_bytes),
                    started_at = CASE WHEN ? IS NULL THEN started_at ELSE COALESCE(started_at, ?) END,
                    completed_at = CASE WHEN ? IS NULL THEN completed_at ELSE ? END,
                    attempt_count = attempt_count + CASE WHEN ? = 'purging' THEN 1 ELSE 0 END,
                    claim_token = CASE WHEN ? IN ('completed', 'failed') THEN '' ELSE claim_token END,
                    updated_at = ?
                {where}
                """,
                (*params[:9], status, *params[9:]),
            )
            if cur.rowcount != 1:
                if expected_claim_token is not None:
                    raise RuntimeError(f"run purge request {request_id} claim was lost")
                raise KeyError(f"run purge request {request_id} not found")
        return self.get_run_purge_request_by_id(request_id)

    def claim_run_purge_request(self, request_id: int, claim_token: str = "") -> bool:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                UPDATE run_purge_requests
                SET status = 'purging',
                    started_at = COALESCE(started_at, ?),
                    attempt_count = attempt_count + 1,
                    error_json = '{}',
                    claim_token = ?,
                    updated_at = ?
                WHERE id = ? AND status IN ('requested', 'canceling')
                """,
                (now, str(claim_token), now, request_id),
            )
            return cur.rowcount == 1

    def heartbeat_run_purge_request(self, request_id: int, claim_token: str) -> bool:
        """Renew a destructive purge claim while filesystem work is in flight."""
        with self.connection() as conn:
            cur = conn.execute(
                """
                UPDATE run_purge_requests
                SET updated_at = ?
                WHERE id = ? AND status = 'purging' AND claim_token = ?
                """,
                (utc_ts(), int(request_id), str(claim_token)),
            )
            return cur.rowcount == 1

    def upsert_cache_entry(
        self,
        cache_type: str,
        cache_key: str,
        storage_path: str | Path,
        *,
        state: str = "ready",
        size_bytes: int = 0,
        metadata: dict[str, Any] | None = None,
        last_used_at: float | None = None,
        gc_after: float | None = None,
    ) -> dict[str, Any]:
        if cache_type not in {"decode_cache", "compare_cache"}:
            raise ValueError("cache_type must be decode_cache or compare_cache")
        if state not in {"ready", "missing", "deleting", "deleted", "failed"}:
            raise ValueError(f"unsupported cache state: {state}")
        key = str(cache_key).strip()
        if not key:
            raise ValueError("cache_key must not be empty")
        now = utc_ts()
        used_at = float(last_used_at if last_used_at is not None else now)
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO cache_entries(
                    cache_type, cache_key, storage_path, state, size_bytes,
                    metadata_json, created_at, last_used_at, gc_after, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_type, cache_key) DO UPDATE SET
                    storage_path = excluded.storage_path,
                    state = CASE
                        WHEN cache_entries.state = 'deleting' THEN cache_entries.state
                        ELSE excluded.state
                    END,
                    size_bytes = excluded.size_bytes,
                    metadata_json = excluded.metadata_json,
                    last_used_at = MAX(cache_entries.last_used_at, excluded.last_used_at),
                    gc_after = CASE
                        WHEN EXISTS (
                            SELECT 1 FROM run_cache_refs
                            WHERE cache_entry_id = cache_entries.id AND released_at IS NULL
                        ) THEN NULL
                        ELSE COALESCE(cache_entries.gc_after, excluded.gc_after)
                    END,
                    deleted_at = CASE WHEN excluded.state = 'deleted' THEN cache_entries.deleted_at ELSE NULL END,
                    updated_at = excluded.updated_at
                """,
                (
                    cache_type,
                    key,
                    str(Path(storage_path).resolve()),
                    state,
                    max(0, int(size_bytes)),
                    _json(metadata),
                    used_at,
                    used_at,
                    gc_after,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM cache_entries WHERE cache_type = ? AND cache_key = ?",
                (cache_type, key),
            ).fetchone()
        assert row is not None
        result = dict(row)
        self._decode_cache_entry(result)
        return result

    def get_cache_entry(self, cache_type: str, cache_key: str) -> dict[str, Any] | None:
        row = self.get(
            "SELECT * FROM cache_entries WHERE cache_type = ? AND cache_key = ?",
            (cache_type, cache_key),
        )
        if row is not None:
            self._decode_cache_entry(row)
        return row

    def list_cache_entries(self, include_deleted: bool = False) -> list[dict[str, Any]]:
        clause = "" if include_deleted else "WHERE deleted_at IS NULL AND state != 'deleted'"
        rows = self.query(
            f"SELECT * FROM cache_entries {clause} ORDER BY cache_type, cache_key"
        )
        for row in rows:
            self._decode_cache_entry(row)
        return rows

    def replace_run_cache_refs(
        self,
        run_id: int,
        cache_entry_ids: Iterable[int],
        *,
        grace_seconds: float = 600.0,
    ) -> dict[str, int]:
        entry_ids = sorted({int(entry_id) for entry_id in cache_entry_ids})
        now = utc_ts()
        gc_after = now + max(0.0, float(grace_seconds))
        with self.connection() as conn:
            if conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
                raise KeyError(f"run {run_id} not found")
            if entry_ids:
                placeholders = ",".join("?" for _ in entry_ids)
                deleting = conn.execute(
                    f"SELECT id FROM cache_entries WHERE id IN ({placeholders}) AND state = 'deleting'",
                    tuple(entry_ids),
                ).fetchone()
                if deleting is not None:
                    raise RuntimeError(
                        f"cache entry {int(deleting['id'])} is being garbage-collected; retry registration"
                    )
            previous = {
                int(row["cache_entry_id"])
                for row in conn.execute(
                    "SELECT cache_entry_id FROM run_cache_refs WHERE run_id = ? AND released_at IS NULL",
                    (run_id,),
                ).fetchall()
            }
            wanted = set(entry_ids)
            for entry_id in wanted:
                conn.execute(
                    """
                    INSERT INTO run_cache_refs(run_id, cache_entry_id, created_at, released_at)
                    VALUES (?, ?, ?, NULL)
                    ON CONFLICT(run_id, cache_entry_id) DO UPDATE SET released_at = NULL
                    """,
                    (run_id, entry_id, now),
                )
                conn.execute(
                    """
                    UPDATE cache_entries
                    SET last_used_at = ?, gc_after = NULL, deleted_at = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (now, now, entry_id),
                )
            released = previous - wanted
            if released:
                placeholders = ",".join("?" for _ in released)
                conn.execute(
                    f"UPDATE run_cache_refs SET released_at = ? WHERE run_id = ? AND cache_entry_id IN ({placeholders})",
                    (now, run_id, *sorted(released)),
                )
                conn.execute(
                    f"""
                    UPDATE cache_entries
                    SET gc_after = COALESCE(gc_after, ?), updated_at = ?
                    WHERE id IN ({placeholders})
                      AND NOT EXISTS (
                          SELECT 1 FROM run_cache_refs rcr
                          WHERE rcr.cache_entry_id = cache_entries.id AND rcr.released_at IS NULL
                      )
                    """,
                    (gc_after, now, *sorted(released)),
                )
        return {"added": len(wanted - previous), "released": len(released), "total": len(wanted)}

    def release_run_cache_refs(self, run_id: int, *, grace_seconds: float = 600.0) -> list[int]:
        now = utc_ts()
        gc_after = now + max(0.0, float(grace_seconds))
        with self.connection() as conn:
            entry_ids = [
                int(row["cache_entry_id"])
                for row in conn.execute(
                    "SELECT cache_entry_id FROM run_cache_refs WHERE run_id = ? AND released_at IS NULL",
                    (run_id,),
                ).fetchall()
            ]
            conn.execute(
                "UPDATE run_cache_refs SET released_at = ? WHERE run_id = ? AND released_at IS NULL",
                (now, run_id),
            )
            if entry_ids:
                placeholders = ",".join("?" for _ in entry_ids)
                conn.execute(
                    f"""
                    UPDATE cache_entries
                    SET gc_after = COALESCE(gc_after, ?), updated_at = ?
                    WHERE id IN ({placeholders})
                      AND NOT EXISTS (
                          SELECT 1 FROM run_cache_refs rcr
                          WHERE rcr.cache_entry_id = cache_entries.id AND rcr.released_at IS NULL
                      )
                    """,
                    (gc_after, now, *entry_ids),
                )
        return entry_ids

    def acquire_cache_lease(self, cache_entry_id: int, lease_id: str, ttl_seconds: float = 21600.0) -> None:
        now = utc_ts()
        expires_at = now + max(1.0, float(ttl_seconds))
        with self.connection() as conn:
            conn.execute("DELETE FROM cache_leases WHERE expires_at <= ?", (now,))
            row = conn.execute(
                "SELECT state, deleted_at FROM cache_entries WHERE id = ?",
                (int(cache_entry_id),),
            ).fetchone()
            if row is None:
                raise KeyError(f"cache entry {cache_entry_id} not found")
            if row["state"] == "deleting":
                raise RuntimeError(f"cache entry {cache_entry_id} is being garbage-collected; retry acquisition")
            conn.execute(
                """
                INSERT INTO cache_leases(cache_entry_id, lease_id, expires_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_entry_id, lease_id) DO UPDATE SET
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (cache_entry_id, str(lease_id), expires_at, now, now),
            )

    def release_cache_lease(self, cache_entry_id: int, lease_id: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "DELETE FROM cache_leases WHERE cache_entry_id = ? AND lease_id = ?",
                (cache_entry_id, str(lease_id)),
            )

    def claim_decode_cache_build_lock(
        self,
        cache_key: str,
        owner_token: str,
        *,
        ttl_seconds: float = 5 * 60,
    ) -> bool:
        """Atomically acquire or take over one decode-cache build lock.

        This is deliberately separate from ``cache_leases``: a cache lease
        prevents storage GC from removing an in-use entry, while this lock
        makes one process the sole producer for a cache key.
        """
        key = str(cache_key).strip()
        owner = str(owner_token).strip()
        if not key:
            raise ValueError("decode cache build lock requires a cache key")
        if not owner:
            raise ValueError("decode cache build lock requires an owner token")
        now = utc_ts()
        expires_at = now + max(0.01, float(ttl_seconds))
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM decode_cache_build_locks WHERE expires_at <= ?", (now,))
            row = conn.execute(
                "SELECT owner_token FROM decode_cache_build_locks WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO decode_cache_build_locks(
                        cache_key, owner_token, expires_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (key, owner, expires_at, now, now),
                )
                return True
            if str(row["owner_token"]) != owner:
                return False
            conn.execute(
                """
                UPDATE decode_cache_build_locks
                SET expires_at = ?, updated_at = ?
                WHERE cache_key = ? AND owner_token = ?
                """,
                (expires_at, now, key, owner),
            )
            return True

    def renew_decode_cache_build_lock(
        self,
        cache_key: str,
        owner_token: str,
        *,
        ttl_seconds: float = 5 * 60,
    ) -> bool:
        """Extend a lock only when this producer still owns an unexpired row."""
        key = str(cache_key).strip()
        owner = str(owner_token).strip()
        if not key or not owner:
            return False
        now = utc_ts()
        expires_at = now + max(0.01, float(ttl_seconds))
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                """
                UPDATE decode_cache_build_locks
                SET expires_at = ?, updated_at = ?
                WHERE cache_key = ? AND owner_token = ? AND expires_at > ?
                """,
                (expires_at, now, key, owner, now),
            )
            return cur.rowcount == 1

    def owns_decode_cache_build_lock(self, cache_key: str, owner_token: str) -> bool:
        """Return whether an unexpired build lock still belongs to ``owner_token``."""
        key = str(cache_key).strip()
        owner = str(owner_token).strip()
        if not key or not owner:
            return False
        now = utc_ts()
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM decode_cache_build_locks
                WHERE cache_key = ? AND owner_token = ? AND expires_at > ?
                """,
                (key, owner, now),
            ).fetchone()
        return row is not None

    def release_decode_cache_build_lock(self, cache_key: str, owner_token: str) -> None:
        """Release only the caller's build lock; never clear a new owner's row."""
        with self.connection() as conn:
            conn.execute(
                "DELETE FROM decode_cache_build_locks WHERE cache_key = ? AND owner_token = ?",
                (str(cache_key).strip(), str(owner_token).strip()),
            )

    @contextmanager
    def decode_cache_build_publish_guard(
        self,
        cache_key: str,
        owner_token: str,
        *,
        ttl_seconds: float = 5 * 60,
    ) -> Iterator[None]:
        """Fence the filesystem publish step for one decode-cache producer.

        The guard holds SQLite's writer lock only while the completed private
        staging directory is being published.  It asserts the lock is still
        owned and unexpired, refreshes its expiry, then removes the build lock
        only after the caller exits successfully.  A former producer therefore
        cannot publish after another process has taken over its expired lock.
        """
        key = str(cache_key).strip()
        owner = str(owner_token).strip()
        if not key or not owner:
            raise ValueError("decode cache publish requires a cache key and owner token")
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = utc_ts()
            expires_at = now + max(0.01, float(ttl_seconds))
            cur = conn.execute(
                """
                UPDATE decode_cache_build_locks
                SET expires_at = ?, updated_at = ?
                WHERE cache_key = ? AND owner_token = ? AND expires_at > ?
                """,
                (expires_at, now, key, owner, now),
            )
            if cur.rowcount != 1:
                conn.rollback()
                raise RuntimeError("decode cache build lock was lost before publish; retry the operation")
            try:
                yield
            except BaseException:
                conn.rollback()
                raise
            else:
                conn.execute(
                    "DELETE FROM decode_cache_build_locks WHERE cache_key = ? AND owner_token = ?",
                    (key, owner),
                )
                conn.commit()
        finally:
            conn.close()

    def cache_gc_inventory(self) -> list[dict[str, Any]]:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("DELETE FROM cache_leases WHERE expires_at <= ?", (now,))
            rows = conn.execute(
                """
                SELECT ce.*,
                       COUNT(DISTINCT rcr.run_id) AS active_run_refs,
                       COUNT(DISTINCT cl.lease_id) AS active_leases
                FROM cache_entries ce
                LEFT JOIN run_cache_refs rcr ON rcr.cache_entry_id = ce.id AND rcr.released_at IS NULL
                LEFT JOIN cache_leases cl ON cl.cache_entry_id = ce.id AND cl.expires_at > ?
                WHERE ce.deleted_at IS NULL AND ce.state != 'deleted'
                GROUP BY ce.id
                ORDER BY ce.cache_type, ce.cache_key
                """,
                (now,),
            ).fetchall()
        result = [dict(row) for row in rows]
        for row in result:
            self._decode_cache_entry(row)
        return result

    def claim_cache_entry_for_gc(self, cache_entry_id: int) -> dict[str, Any] | None:
        """Atomically reserve an otherwise-unreferenced cache for deletion.

        Preview is advisory.  This conditional transition is the destructive
        boundary: a new Run reference or lease added after preview makes the
        claim fail instead of allowing GC to remove an in-use cache.
        """
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM cache_leases WHERE expires_at <= ?", (now,))
            entry = conn.execute(
                "SELECT cache_type, cache_key FROM cache_entries WHERE id = ?",
                (int(cache_entry_id),),
            ).fetchone()
            if entry is None:
                conn.commit()
                return None

            # A decode writes into ``<key>.partial`` while holding the lease
            # for its eventual ``<key>`` cache entry.  Treat that base entry's
            # activity as protection for the partial directory too; otherwise
            # a concurrent catalog/GC pass can remove live decoder output.
            companion_clause = ""
            companion_params: list[Any] = []
            cache_type = str(entry["cache_type"])
            cache_key = str(entry["cache_key"])
            if cache_type == "decode_cache" and cache_key.endswith(".partial"):
                base = conn.execute(
                    "SELECT id FROM cache_entries WHERE cache_type = ? AND cache_key = ?",
                    (cache_type, cache_key[: -len(".partial")]),
                ).fetchone()
                if base is not None:
                    base_id = int(base["id"])
                    companion_clause = """
                      AND NOT EXISTS (
                          SELECT 1 FROM cache_entries base
                          WHERE base.id = ? AND base.state = 'deleting'
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM run_cache_refs rcr
                          WHERE rcr.cache_entry_id = ?
                            AND rcr.released_at IS NULL
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM cache_leases cl
                          WHERE cl.cache_entry_id = ?
                            AND cl.expires_at > ?
                      )
                    """
                    companion_params = [base_id, base_id, base_id, now]
            cur = conn.execute(
                f"""
                UPDATE cache_entries
                SET state = 'deleting', updated_at = ?
                WHERE id = ?
                  AND deleted_at IS NULL
                  AND state IN ('ready', 'missing', 'failed')
                  AND gc_after IS NOT NULL AND gc_after <= ?
                  AND NOT EXISTS (
                      SELECT 1 FROM run_cache_refs rcr
                      WHERE rcr.cache_entry_id = cache_entries.id
                        AND rcr.released_at IS NULL
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM cache_leases cl
                      WHERE cl.cache_entry_id = cache_entries.id
                        AND cl.expires_at > ?
                  )
                {companion_clause}
                """,
                (now, int(cache_entry_id), now, now, *companion_params),
            )
            if cur.rowcount != 1:
                conn.commit()
                return None
            row = conn.execute("SELECT * FROM cache_entries WHERE id = ?", (int(cache_entry_id),)).fetchone()
            conn.commit()
        assert row is not None
        result = dict(row)
        self._decode_cache_entry(result)
        return result

    def mark_cache_entry_state(
        self,
        cache_entry_id: int,
        state: str,
        *,
        size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if state not in {"ready", "missing", "deleting", "deleted", "failed"}:
            raise ValueError(f"unsupported cache state: {state}")
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE cache_entries
                SET state = ?,
                    size_bytes = COALESCE(?, size_bytes),
                    metadata_json = COALESCE(?, metadata_json),
                    deleted_at = CASE WHEN ? = 'deleted' THEN ? ELSE deleted_at END,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    state,
                    int(size_bytes) if size_bytes is not None else None,
                    _json(metadata) if metadata is not None else None,
                    state,
                    now,
                    now,
                    cache_entry_id,
                ),
            )

    def invalidate_run_media_assets(self, run_id: int) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                UPDATE media_assets
                SET state = 'unavailable', updated_at = ?
                WHERE source_kind = 'run_artifact'
                  AND (
                      id IN (
                          SELECT asset_id
                          FROM run_media_assets
                          WHERE run_id = ?
                            AND (
                                COALESCE(json_extract(metadata_json, '$.input'), 0) != 1
                                OR COALESCE(json_extract(metadata_json, '$.compare_snapshot'), 0) = 1
                            )
                      )
                      OR source_key IN (
                          SELECT 'run_artifact:' || a.id
                          FROM artifacts a
                          WHERE a.job_id IN (
                              SELECT job_id
                              FROM run_jobs
                              WHERE run_id = ?
                              UNION
                              SELECT inference_job_id
                              FROM runs
                              WHERE id = ? AND inference_job_id IS NOT NULL
                              UNION
                              SELECT metric_job_id
                              FROM runs
                              WHERE id = ? AND metric_job_id IS NOT NULL
                          )
                      )
                  )
                """,
                (now, run_id, run_id, run_id, run_id),
            )
            return int(cur.rowcount)

    def summarize_artifacts(self, job_id: int) -> dict[str, Any]:
        rows = self.query(
            """
            SELECT kind,
                   COUNT(*) AS count,
                   SUM(CAST(COALESCE(json_extract(metadata_json, '$.storage_bytes'), 0) AS INTEGER))
                       AS storage_bytes,
                   SUM(
                       CASE
                           WHEN COALESCE(
                               json_extract(metadata_json, '$.storage_size_complete'),
                               0
                           ) = 1 THEN 1
                           ELSE 0
                       END
                   ) AS storage_size_known
            FROM artifacts
            WHERE job_id = ?
            GROUP BY kind
            ORDER BY kind
            """,
            (job_id,),
        )
        by_kind = {row["kind"]: int(row["count"]) for row in rows}
        storage_bytes_by_kind = {
            row["kind"]: int(row["storage_bytes"] or 0) for row in rows
        }
        storage_size_known = sum(int(row["storage_size_known"] or 0) for row in rows)
        total = sum(by_kind.values())
        return {
            "total": total,
            "by_kind": by_kind,
            "storage_bytes": sum(storage_bytes_by_kind.values()),
            "storage_bytes_by_kind": storage_bytes_by_kind,
            "storage_size_known": storage_size_known,
            "storage_size_unknown": max(0, total - storage_size_known),
        }

    def run_inference_job_ids(self, run_id: int) -> list[int]:
        job_ids = [int(row["job_id"]) for row in self.list_run_jobs(run_id, "inference")]
        if job_ids:
            return job_ids
        run = self.get_run(run_id)
        return [int(run["inference_job_id"])] if run.get("inference_job_id") is not None else []

    def update_run_progress_from_jobs(self, run_id: int, status: str | None = None) -> bool:
        rows = self.list_run_jobs(run_id, "inference")
        current = sum(int(row.get("progress_current") or 0) for row in rows)
        total = sum(int(row.get("progress_total") or 0) for row in rows)
        return self.update_run_progress(run_id, current, total, status)

    def summarize_run_artifacts(self, run_id: int) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        storage_bytes_by_kind: dict[str, int] = {}
        storage_size_known = 0
        storage_size_unknown = 0
        for job_id in self.run_inference_job_ids(run_id):
            summary = self.summarize_artifacts(job_id)
            for kind, count in (summary.get("by_kind") or {}).items():
                by_kind[kind] = by_kind.get(kind, 0) + int(count)
            for kind, size_bytes in (summary.get("storage_bytes_by_kind") or {}).items():
                storage_bytes_by_kind[kind] = (
                    storage_bytes_by_kind.get(kind, 0) + int(size_bytes)
                )
            storage_size_known += int(summary.get("storage_size_known") or 0)
            storage_size_unknown += int(summary.get("storage_size_unknown") or 0)
        return {
            "total": sum(by_kind.values()),
            "by_kind": by_kind,
            "storage_bytes": sum(storage_bytes_by_kind.values()),
            "storage_bytes_by_kind": storage_bytes_by_kind,
            "storage_size_known": storage_size_known,
            "storage_size_unknown": storage_size_unknown,
        }

    def maybe_complete_multi_run_inference(
        self,
        run_id: int,
        *,
        source_job_id: int | None = None,
        source_job_result: dict[str, Any] | None = None,
    ) -> bool:
        run = self.get_run(run_id)
        if run["status"] in {"cancel_requested", "canceled"}:
            self.converge_run_cancellation(run_id)
            return False
        if run["status"] in {"completed", "failed"}:
            return False
        jobs = self.list_run_jobs(run_id, "inference")
        if not jobs:
            return False
        source_original_status: str | None = None
        if source_job_id is not None:
            source_row = next(
                (row for row in jobs if int(row["job_id"]) == int(source_job_id)),
                None,
            )
            if source_row is None:
                return False
            source_original_status = str(source_row.get("status") or "")
            if source_original_status not in {"running", "completed"}:
                return False
            if source_original_status == "running":
                source_row["status"] = "completed"
                source_row["result"] = dict(source_job_result or {})
        if any(job["status"] == "failed" for job in jobs):
            failed = next(job for job in jobs if job["status"] == "failed")
            self.fail_run(run_id, enrich_job_error(failed, failed.get("error") or {}))
            return True
        if any(job["status"] == "canceled" for job in jobs):
            self.cancel_run(run_id, {"message": "inference shard canceled"})
            return True
        if any(job["status"] != "completed" for job in jobs):
            if source_job_id is not None and source_original_status == "running":
                if not self.complete_job(int(source_job_id), source_job_result):
                    return False
            self.update_run_progress_from_jobs(run_id)
            return False

        output_health = _combine_output_health((job.get("result") or {}).get("output_health") for job in jobs)
        result = {
            "samples": sum(int(job.get("result", {}).get("samples") or 0) for job in jobs),
            "output_dir": (run.get("metadata") or {}).get("output_dir"),
            "shards": [
                {
                    "job_id": int(job["job_id"]),
                    "device": job.get("device"),
                    "status": job["status"],
                    "result": job.get("result") or {},
                }
                for job in jobs
            ],
        }
        if output_health is not None:
            result["output_health"] = output_health
        artifact_summary = self.summarize_run_artifacts(run_id)
        if any(bool((job.get("payload") or {}).get("defer_video_finalize")) for job in jobs):
            advanced = self.queue_run_finalize(
                run_id,
                result,
                artifact_summary,
                [int(job["job_id"]) for job in jobs],
                source_job_id=source_job_id,
                source_job_result=source_job_result,
            )
            if not advanced and source_job_id is not None and source_original_status == "running":
                raise RuntimeError(f"run {run_id} rejected atomic finalize handoff")
            return advanced
        metrics = list(run.get("metrics") or [])
        if metrics:
            from vfieval.pipeline.metric_jobs import create_metric_wave

            try:
                create_metric_wave(
                    self,
                    run_id,
                    metrics,
                    source=str((run.get("metadata") or {}).get("execution_mode") or "multi"),
                    result=result,
                    artifact_summary=artifact_summary,
                    source_job_id=source_job_id,
                    source_job_result=source_job_result,
                )
            except ValueError:
                if source_job_id is not None and source_original_status == "running":
                    raise RuntimeError(f"run {run_id} rejected atomic metric handoff")
                return False
        else:
            advanced = self.complete_run_inference(
                run_id,
                result,
                artifact_summary,
                "completed",
                source_job_id=source_job_id,
                source_job_result=source_job_result,
            )
            if not advanced and source_job_id is not None and source_original_status == "running":
                raise RuntimeError(f"run {run_id} rejected atomic inference completion")
            return advanced
        return True

    def queue_run_finalize(
        self,
        run_id: int,
        result: dict[str, Any],
        artifact_summary: dict[str, Any],
        inference_job_ids: Iterable[int],
        *,
        source_job_id: int | None = None,
        source_job_result: dict[str, Any] | None = None,
    ) -> bool:
        """Atomically publish a finalize Job and the ``finalize_queued`` state."""
        source_ids = sorted({int(job_id) for job_id in inference_job_ids})
        if not source_ids:
            return False
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute(
                """
                SELECT status, deleted_at, artifact_cleaned_at
                FROM runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"run {run_id} not found")
            source_job_status = self._source_job_status(
                conn,
                run_id,
                source_job_id,
                ("inference",),
            )
            if (
                source_job_id is not None
                and (
                    int(source_job_id) not in source_ids
                    or source_job_status not in {"running", "completed"}
                )
            ):
                return False
            if (
                run["deleted_at"] is not None
                or run["artifact_cleaned_at"] is not None
                or conn.execute(
                    """
                    SELECT 1
                    FROM run_purge_requests
                    WHERE run_id = ?
                      AND status IN ('requested', 'canceling', 'purging')
                    LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()
                is not None
            ):
                return False
            existing_rows = conn.execute(
                """
                SELECT rj.job_id, j.kind, j.status, j.payload_json
                FROM run_jobs rj
                JOIN jobs j ON j.id = rj.job_id
                WHERE rj.run_id = ? AND rj.role = 'finalize'
                ORDER BY rj.job_id
                """,
                (run_id,),
            ).fetchall()
            existing = existing_rows[0] if len(existing_rows) == 1 else None
            if existing_rows:
                existing_payload = _loads(existing["payload_json"]) if existing is not None else {}
                try:
                    existing_source_ids = sorted(
                        {
                            int(job_id)
                            for job_id in (
                                existing_payload.get("inference_job_ids")
                                if isinstance(existing_payload, dict)
                                else []
                            )
                            or []
                        }
                    )
                except (TypeError, ValueError):
                    return False
                if (
                    existing is None
                    or str(existing["kind"]) != "finalize"
                    or str(existing["status"]) not in {"queued", "running"}
                    or existing_source_ids != source_ids
                ):
                    return False

            run_status = str(run["status"])
            if run_status not in {"running", "finalize_queued", "finalizing"}:
                return False

            inference_rows = conn.execute(
                """
                SELECT rj.job_id, j.kind, j.status
                FROM run_jobs rj
                JOIN jobs j ON j.id = rj.job_id
                WHERE rj.run_id = ? AND rj.role = 'inference'
                ORDER BY rj.job_id
                """,
                (run_id,),
            ).fetchall()
            if (
                [int(row["job_id"]) for row in inference_rows] != source_ids
                or any(
                    str(row["kind"]) != "inference"
                    or (
                        str(row["status"]) != "completed"
                        and not (
                            source_job_id is not None
                            and int(row["job_id"]) == int(source_job_id)
                            and str(row["status"]) == "running"
                        )
                    )
                    for row in inference_rows
                )
            ):
                return False

            if existing is not None:
                job_status = str(existing["status"])
                if run_status == "finalize_queued" and job_status in {"queued", "running"}:
                    if not self._complete_source_job_in_transaction(
                        conn,
                        source_job_id,
                        source_job_status,
                        source_job_result,
                        now,
                    ):
                        raise RuntimeError(f"source Job {source_job_id} rejected finalize handoff completion")
                    return True
                if run_status == "finalizing" and job_status == "running":
                    if not self._complete_source_job_in_transaction(
                        conn,
                        source_job_id,
                        source_job_status,
                        source_job_result,
                        now,
                    ):
                        raise RuntimeError(f"source Job {source_job_id} rejected finalize handoff completion")
                    return True
                if run_status != "running" or job_status != "queued":
                    return False

            if existing is None:
                job_cur = conn.execute(
                    """
                    INSERT INTO jobs(kind, status, payload_json, progress_total, created_at)
                    VALUES ('finalize', 'queued', ?, 1, ?)
                    """,
                    (
                        _json({"run_id": run_id, "inference_job_ids": source_ids}),
                        now,
                    ),
                )
                finalize_job_id = int(job_cur.lastrowid)
                conn.execute(
                    """
                    INSERT INTO run_jobs(
                        run_id, job_id, role, shard_index, device, metadata_json, created_at
                    )
                    VALUES (?, ?, 'finalize', 0, NULL, ?, ?)
                    """,
                    (run_id, finalize_job_id, _json({"source": "multi_device"}), now),
                )

            cur = conn.execute(
                """
                UPDATE runs
                SET status = 'finalize_queued', result_json = ?,
                    artifact_summary_json = ?,
                    content_revision = content_revision + 1,
                    updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (_json(result), _json(artifact_summary), now, run_id),
            )
            if not cur.rowcount:
                return False
            if not self._complete_source_job_in_transaction(
                conn,
                source_job_id,
                source_job_status,
                source_job_result,
                now,
            ):
                raise RuntimeError(f"source Job {source_job_id} rejected finalize handoff completion")
            return True

    def list_run_artifacts(self, run_id: int, kind: str | None = None) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        for job_id in self.run_inference_job_ids(run_id):
            artifacts.extend(self.list_artifacts(job_id=int(job_id), kind=kind))
        return artifacts

    def list_run_video_artifacts(
        self,
        run_id: int,
        video_name: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        job_ids = self.run_inference_job_ids(run_id)
        if not job_ids:
            return []
        placeholders = ",".join("?" for _ in job_ids)
        clauses = ["job_id IN (" + placeholders + ")", "sample_id IS NULL"]
        params: list[Any] = list(job_ids)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if video_name is not None:
            clauses.append("json_extract(metadata_json, '$.video_name') = ?")
            params.append(str(video_name))
        rows = self.query(
            f"""
            SELECT *
            FROM artifacts
            WHERE {' AND '.join(clauses)}
            ORDER BY kind, id
            """,
            params,
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def list_run_metrics(self, run_id: int) -> list[dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        for job_id in self.run_inference_job_ids(run_id):
            metrics.extend(self.list_metric_results(inference_job_id=int(job_id)))
        return metrics

    def list_run_video_metrics(self, run_id: int, video_name: str | None = None) -> list[dict[str, Any]]:
        job_ids = self.run_inference_job_ids(run_id)
        if not job_ids:
            return []
        placeholders = ",".join("?" for _ in job_ids)
        rows = self.query(
            f"""
            SELECT *
            FROM metric_results
            WHERE sample_id IS NULL
              AND inference_job_id IN ({placeholders})
            ORDER BY metric_name, id
            """,
            job_ids,
        )
        result = []
        for row in rows:
            row["details"] = _loads(row.pop("details_json"))
            if video_name is not None and str((row.get("details") or {}).get("video_name") or "") != str(video_name):
                continue
            result.append(row)
        return result

    def list_run_samples(self, run_id: int) -> list[dict[str, Any]]:
        run = self.get_run(run_id)
        samples = self.list_samples(int(run["dataset_id"]))
        artifacts = self.list_run_artifacts(run_id)
        metrics = self.list_run_metrics(run_id)
        artifacts_by_sample: dict[int, dict[str, list[dict[str, Any]]]] = {}
        for artifact in artifacts:
            sample_id = artifact.get("sample_id")
            if sample_id is None:
                continue
            artifacts_by_sample.setdefault(int(sample_id), {}).setdefault(artifact["kind"], []).append(artifact)
        metrics_by_sample: dict[int, list[dict[str, Any]]] = {}
        for metric in metrics:
            sample_id = metric.get("sample_id")
            if sample_id is None:
                continue
            metrics_by_sample.setdefault(int(sample_id), []).append(metric)
        for sample in samples:
            sample_id = int(sample["id"])
            sample["artifacts"] = artifacts_by_sample.get(sample_id, {})
            sample["metrics"] = metrics_by_sample.get(sample_id, [])
        return samples

    def register_worker(self, worker_id: str, role: str, capabilities: dict[str, Any] | None = None) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO workers(id, role, capabilities_json, last_seen_at, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    role=excluded.role,
                    capabilities_json=excluded.capabilities_json,
                    last_seen_at=excluded.last_seen_at
                """,
                (worker_id, role, _json(capabilities), now, now),
            )

    def list_workers(self) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM workers ORDER BY last_seen_at DESC")
        for row in rows:
            row["capabilities"] = _loads(row.pop("capabilities_json"))
        return rows

    def get_worker(self, worker_id: str) -> dict[str, Any]:
        row = self.get("SELECT * FROM workers WHERE id = ?", (worker_id,))
        if row is None:
            raise KeyError(f"worker {worker_id} not found")
        row["capabilities"] = _loads(row.pop("capabilities_json"))
        return row

    def touch_worker(self, worker_id: str, capabilities: dict[str, Any] | None = None) -> None:
        worker = self.get_worker(worker_id)
        self.register_worker(worker_id, worker["role"], capabilities or worker.get("capabilities") or {})

    def create_job(self, kind: str, payload: dict[str, Any], progress_total: int = 0) -> int:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run_id = payload.get("run_id")
            run = None
            if run_id is not None:
                run = conn.execute(
                    """
                    SELECT status, deleted_at, artifact_cleaned_at
                    FROM runs
                    WHERE id = ?
                    """,
                    (int(run_id),),
                ).fetchone()
                if run is None:
                    raise KeyError(f"run {run_id} not found")
                purge_pending = conn.execute(
                    """
                    SELECT 1
                    FROM run_purge_requests
                    WHERE run_id = ?
                      AND status IN ('requested', 'canceling', 'purging')
                    LIMIT 1
                    """,
                    (int(run_id),),
                ).fetchone()
                if (
                    str(run["status"] or "") in RUN_NON_PROGRESS_STATUSES
                    or run["deleted_at"] is not None
                    or run["artifact_cleaned_at"] is not None
                    or purge_pending is not None
                ):
                    raise RuntimeError(f"run {run_id} rejects new {kind} Job publication")
                allowed_statuses = {
                    "decode": {"queued"},
                    "inference": {"queued"},
                    # Legacy callers may prepare one metric Job before the
                    # inference/finalize completion CAS publishes it. It stays
                    # unclaimable until ``set_run_metric_job`` advances the Run.
                    "metric": {"queued", "running", "finalizing"},
                }
                if kind not in allowed_statuses:
                    raise ValueError(f"run-bound generic Job kind is not supported: {kind}")
                if str(run["status"] or "") not in allowed_statuses[kind]:
                    raise RuntimeError(
                        f"run {run_id} in {run['status']} rejects new {kind} Job publication"
                    )
                active_same_role = conn.execute(
                    """
                    SELECT 1
                    FROM jobs active
                    JOIN runs owner ON owner.id = ?
                    WHERE active.kind = ?
                      AND active.status IN ('queued', 'running')
                      AND (
                          EXISTS (
                              SELECT 1 FROM run_jobs rj
                              WHERE rj.run_id = owner.id AND rj.job_id = active.id
                          )
                          OR (? = 'inference' AND active.id = owner.inference_job_id)
                          OR (? = 'metric' AND active.id = owner.metric_job_id)
                          OR CAST(json_extract(active.payload_json, '$.run_id') AS INTEGER) = owner.id
                      )
                    LIMIT 1
                    """,
                    (int(run_id), str(kind), str(kind), str(kind)),
                ).fetchone()
                if active_same_role is not None:
                    raise RuntimeError(f"run {run_id} already has an active {kind} Job")
            cur = conn.execute(
                """
                INSERT INTO jobs(kind, status, payload_json, progress_total, created_at)
                VALUES (?, 'queued', ?, ?, ?)
                """,
                (kind, _json(payload), progress_total, now),
            )
            job_id = int(cur.lastrowid)
            if run_id is not None:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO run_jobs(
                        run_id, job_id, role, shard_index, device, metadata_json, created_at
                    )
                    VALUES (?, ?, ?, 0, ?, '{}', ?)
                    """,
                    (
                        int(run_id),
                        job_id,
                        kind,
                        payload.get("device") or payload.get("metric_device"),
                        now,
                    ),
                )
                if kind == "inference":
                    conn.execute(
                        """
                        UPDATE runs
                        SET inference_job_id = COALESCE(inference_job_id, ?), updated_at = ?
                        WHERE id = ?
                        """,
                        (job_id, now, int(run_id)),
                    )
                elif kind == "decode":
                    conn.execute(
                        """
                        UPDATE runs
                        SET status = 'decoding', updated_at = ?
                        WHERE id = ? AND status = 'queued'
                        """,
                        (now, int(run_id)),
                    )
            return job_id

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
        for row in rows:
            self._decode_job(row)
        return rows

    def get_job(self, job_id: int) -> dict[str, Any]:
        row = self.get("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row is None:
            raise KeyError(f"job {job_id} not found")
        self._decode_job(row)
        return row

    def claim_next_job(self, worker_id: str, kinds: list[str], device_filter: str | None = None) -> dict[str, Any] | None:
        if not kinds:
            return None
        placeholders = ",".join("?" for _ in kinds)
        now = utc_ts()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if device_filter:
                row = conn.execute(
                    f"""
                    SELECT j.*
                    FROM jobs j
                    JOIN runs r ON (
                        r.inference_job_id = j.id
                        OR r.metric_job_id = j.id
                        OR CAST(json_extract(j.payload_json, '$.run_id') AS INTEGER) = r.id
                        OR EXISTS (
                            SELECT 1 FROM run_jobs linked
                            WHERE linked.run_id = r.id AND linked.job_id = j.id
                        )
                    )
                    LEFT JOIN run_jobs rj
                      ON rj.run_id = r.id AND rj.job_id = j.id
                    WHERE j.status = 'queued'
                      AND j.kind IN ({placeholders})
                      AND (
                          COALESCE(
                              rj.device,
                              json_extract(j.payload_json, '$.metric_device'),
                              json_extract(j.payload_json, '$.device'),
                              r.device
                          ) = ?
                          OR (
                              j.kind = 'finalize'
                              AND COALESCE(
                                  rj.device,
                                  json_extract(j.payload_json, '$.metric_device'),
                                  json_extract(j.payload_json, '$.device'),
                                  r.device
                              ) IS NULL
                          )
                      )
                      AND r.deleted_at IS NULL
                      AND r.artifact_cleaned_at IS NULL
                      AND (
                          (j.kind = 'decode' AND r.status IN ('queued', 'decoding'))
                          OR (j.kind = 'inference' AND r.status IN ('queued', 'running'))
                          OR (j.kind = 'finalize' AND r.status IN ('finalize_queued', 'finalizing'))
                          OR (j.kind = 'metric' AND r.status IN ('metric_queued', 'metric_running'))
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM run_purge_requests pr
                          WHERE pr.run_id = r.id
                            AND pr.status IN ('requested', 'canceling', 'purging')
                      )
                    ORDER BY j.created_at, j.id
                    LIMIT 1
                    """,
                    (*tuple(kinds), device_filter),
                ).fetchone()
            else:
                row = conn.execute(
                    f"""
                    SELECT j.* FROM jobs j
                    WHERE j.status = 'queued' AND j.kind IN ({placeholders})
                      AND NOT EXISTS (
                          SELECT 1
                          FROM run_jobs rj
                          JOIN runs r ON r.id = rj.run_id
                          WHERE rj.job_id = j.id
                            AND (
                                r.deleted_at IS NOT NULL
                                OR r.artifact_cleaned_at IS NOT NULL
                                OR NOT (
                                    (j.kind = 'decode' AND r.status IN ('queued', 'decoding'))
                                    OR (j.kind = 'inference' AND r.status IN ('queued', 'running'))
                                    OR (j.kind = 'finalize' AND r.status IN ('finalize_queued', 'finalizing'))
                                    OR (j.kind = 'metric' AND r.status IN ('metric_queued', 'metric_running'))
                                )
                                OR EXISTS (
                                    SELECT 1 FROM run_purge_requests pr
                                    WHERE pr.run_id = r.id
                                      AND pr.status IN ('requested', 'canceling', 'purging')
                                )
                            )
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM runs r
                          WHERE (
                                r.inference_job_id = j.id
                                OR r.metric_job_id = j.id
                                OR CAST(json_extract(j.payload_json, '$.run_id') AS INTEGER) = r.id
                            )
                            AND (
                                r.deleted_at IS NOT NULL
                                OR r.artifact_cleaned_at IS NOT NULL
                                OR NOT (
                                    (j.kind = 'decode' AND r.status IN ('queued', 'decoding'))
                                    OR (j.kind = 'inference' AND r.status IN ('queued', 'running'))
                                    OR (j.kind = 'finalize' AND r.status IN ('finalize_queued', 'finalizing'))
                                    OR (j.kind = 'metric' AND r.status IN ('metric_queued', 'metric_running'))
                                )
                                OR EXISTS (
                                    SELECT 1 FROM run_purge_requests pr
                                    WHERE pr.run_id = r.id
                                      AND pr.status IN ('requested', 'canceling', 'purging')
                                )
                            )
                      )
                    ORDER BY created_at, id
                    LIMIT 1
                    """,
                    tuple(kinds),
                ).fetchone()
            if row is None:
                conn.commit()
                return None
            if device_filter and str(row["kind"]) == "metric":
                conn.execute(
                    "UPDATE run_jobs SET device = COALESCE(device, ?) WHERE job_id = ? AND role = 'metric'",
                    (device_filter, int(row["id"])),
                )
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'running', worker_id = ?, started_at = ?, heartbeat_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (worker_id, now, now, int(row["id"])),
            )
            if cur.rowcount != 1:
                conn.commit()
                return None
            conn.commit()
        claimed = self.get_job(int(row["id"]))
        return claimed

    def update_job_progress(
        self,
        job_id: int,
        current: int,
        total: int | None = None,
        result: dict[str, Any] | None = None,
    ) -> bool:
        now = utc_ts()
        assignments = ["progress_current = ?", "heartbeat_at = ?"]
        params: list[Any] = [int(current), now]
        if total is not None:
            assignments.append("progress_total = ?")
            params.append(int(total))
        if result is not None:
            assignments.append("result_json = ?")
            params.append(_json(result))
        params.append(int(job_id))
        with self.connection() as conn:
            cur = conn.execute(
                f"""
                UPDATE jobs
                SET {', '.join(assignments)}
                WHERE id = ? AND status = 'running'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM run_jobs rj
                      JOIN runs r ON r.id = rj.run_id
                      WHERE rj.job_id = jobs.id
                        AND r.status IN ('failed', 'cancel_requested', 'canceled')
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM runs r
                      WHERE (
                            r.inference_job_id = jobs.id
                            OR r.metric_job_id = jobs.id
                            OR CAST(json_extract(jobs.payload_json, '$.run_id') AS INTEGER) = r.id
                        )
                        AND r.status IN ('failed', 'cancel_requested', 'canceled')
                  )
                """,
                params,
            )
            return bool(cur.rowcount)

    def complete_job(self, job_id: int, result: dict[str, Any] | None = None) -> bool:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'completed', result_json = ?, finished_at = ?
                WHERE id = ? AND status = 'running'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM run_jobs rj
                      JOIN runs r ON r.id = rj.run_id
                      WHERE rj.job_id = jobs.id
                        AND r.status IN ('failed', 'cancel_requested', 'canceled')
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM runs r
                      WHERE (
                            r.inference_job_id = jobs.id
                            OR r.metric_job_id = jobs.id
                            OR CAST(json_extract(jobs.payload_json, '$.run_id') AS INTEGER) = r.id
                        )
                        AND r.status IN ('failed', 'cancel_requested', 'canceled')
                  )
                """,
                (_json(result), now, job_id),
            )
            if cur.rowcount:
                return True
            # Phase handoffs can complete the source Job in the same
            # transaction as the Run transition. Accept only an identical
            # replay as idempotent; conflicting or terminal callbacks remain
            # rejected and cannot overwrite the stored result.
            row = conn.execute(
                "SELECT status, result_json FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            return bool(
                row is not None
                and str(row["status"] or "") == "completed"
                and _loads(row["result_json"]) == _loads(_json(result))
            )

    def fail_job(self, job_id: int, error: dict[str, Any]) -> bool:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'failed', error_json = ?, finished_at = ?
                WHERE id = ? AND status = 'running'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM run_jobs rj
                      JOIN runs r ON r.id = rj.run_id
                      WHERE rj.job_id = jobs.id
                        AND r.status IN ('cancel_requested', 'canceled')
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM runs r
                      WHERE (
                            r.inference_job_id = jobs.id
                            OR r.metric_job_id = jobs.id
                            OR CAST(json_extract(jobs.payload_json, '$.run_id') AS INTEGER) = r.id
                        )
                        AND r.status IN ('cancel_requested', 'canceled')
                  )
                """,
                (_json(error), now, job_id),
            )
            return bool(cur.rowcount)

    def heartbeat_job(self, job_id: int, worker_id: str) -> bool:
        """Renew a running Job lease owned by ``worker_id``.

        Worker ownership is a fencing token: a late heartbeat from an old
        process cannot revive or extend a Job that has been recovered or
        claimed by another worker.
        """
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET started_at = COALESCE(started_at, ?), heartbeat_at = ?
                WHERE id = ? AND status = 'running' AND worker_id = ?
                """,
                (now, now, int(job_id), str(worker_id)),
            )
            if cur.rowcount:
                conn.execute(
                    "UPDATE workers SET last_seen_at = ? WHERE id = ?",
                    (now, str(worker_id)),
                )
            return bool(cur.rowcount)

    def touch_job(self, job_id: int) -> bool:
        """Compatibility heartbeat for the existing HTTP worker endpoint.

        New in-process workers use :meth:`heartbeat_job` so lease renewal is
        owner-fenced.  The legacy endpoint authenticates the worker separately
        and therefore retains this job-id-only surface until that API changes.
        """
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET started_at = COALESCE(started_at, ?), heartbeat_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (now, now, int(job_id)),
            )
            return bool(cur.rowcount)

    def job_lease_summary(
        self,
        stale_before: float | None = None,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Return a cheap SQLite-only summary for runtime health surfaces."""
        observed_at = utc_ts() if now is None else float(now)
        cutoff = observed_at - 90.0 if stale_before is None else float(stale_before)
        row = self.get(
            """
            SELECT
                COUNT(*) AS running,
                COALESCE(SUM(
                    CASE WHEN COALESCE(heartbeat_at, started_at, created_at) < ?
                         THEN 1 ELSE 0 END
                ), 0) AS stale,
                MIN(COALESCE(heartbeat_at, started_at, created_at)) AS oldest_heartbeat_at
            FROM jobs
            WHERE status = 'running'
            """,
            (cutoff,),
        ) or {}
        running = int(row.get("running") or 0)
        stale = int(row.get("stale") or 0)
        oldest = row.get("oldest_heartbeat_at")
        return {
            "running": running,
            "fresh": max(0, running - stale),
            "stale": stale,
            "oldest_heartbeat_at": float(oldest) if oldest is not None else None,
            "oldest_heartbeat_age_seconds": (
                max(0.0, observed_at - float(oldest)) if oldest is not None else None
            ),
        }

    def recover_stale_jobs(
        self,
        stale_before: float,
        *,
        recovered_at: float | None = None,
        lease_timeout_seconds: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fence stale running Jobs and safely fail their active Runs.

        The stale check is repeated under ``BEGIN IMMEDIATE``.  A heartbeat
        that commits before the recovery writer lock therefore wins, while a
        late worker is fenced by the Job's terminal status.  Run failure,
        root Job failure, queued-sibling cancellation, and media invalidation
        are committed together.
        """
        cutoff = float(stale_before)
        now = utc_ts() if recovered_at is None else float(recovered_at)
        batch_limit = max(1, min(int(limit), 1000))
        candidate_rows = self.query(
            """
            SELECT id
            FROM jobs
            WHERE status = 'running'
              AND COALESCE(heartbeat_at, started_at, created_at) < ?
            ORDER BY COALESCE(heartbeat_at, started_at, created_at), id
            LIMIT ?
            """,
            (cutoff, batch_limit),
        )
        recovered: list[dict[str, Any]] = []
        for candidate in candidate_rows:
            job_id = int(candidate["id"])
            with self.connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """
                    SELECT *
                    FROM jobs
                    WHERE id = ? AND status = 'running'
                      AND COALESCE(heartbeat_at, started_at, created_at) < ?
                    """,
                    (job_id, cutoff),
                ).fetchone()
                if row is None:
                    continue

                job = dict(row)
                job["payload"] = _loads(job.get("payload_json"))
                run_rows = conn.execute(
                    """
                    SELECT DISTINCT r.id, r.status, r.inference_job_id,
                                    r.metric_job_id
                    FROM runs r
                    JOIN jobs j ON j.id = ?
                    LEFT JOIN run_jobs rj
                      ON rj.run_id = r.id AND rj.job_id = j.id
                    WHERE rj.job_id IS NOT NULL
                       OR r.inference_job_id = j.id
                       OR r.metric_job_id = j.id
                       OR CAST(json_extract(j.payload_json, '$.run_id') AS INTEGER) = r.id
                    ORDER BY r.id
                    """,
                    (job_id,),
                ).fetchall()
                active_runs = [
                    run_row
                    for run_row in run_rows
                    if str(run_row["status"] or "") not in RUN_NON_PROGRESS_STATUSES
                ]
                last_heartbeat = (
                    row["heartbeat_at"]
                    if row["heartbeat_at"] is not None
                    else row["started_at"]
                    if row["started_at"] is not None
                    else row["created_at"]
                )
                kind = str(row["kind"] or "job")
                error = enrich_job_error(
                    job,
                    {
                        "type": "WorkerLost",
                        "code": "worker_lost",
                        "message": (
                            f"{kind.capitalize()} worker stopped reporting heartbeat; "
                            "the Job was interrupted safely. Check the worker and device, "
                            "then retry the Run."
                        ),
                        "interrupted": True,
                        "retryable": True,
                        "last_heartbeat_at": float(last_heartbeat),
                        "recovered_at": now,
                    },
                )
                if lease_timeout_seconds is not None:
                    error["lease_timeout_seconds"] = float(lease_timeout_seconds)
                if len(run_rows) == 1 and error.get("run_id") is None:
                    error["run_id"] = int(run_rows[0]["id"])

                if active_runs:
                    job_cur = conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'failed', error_json = ?, finished_at = ?
                        WHERE id = ? AND status = 'running'
                          AND COALESCE(heartbeat_at, started_at, created_at) < ?
                        """,
                        (_json(error), now, job_id, cutoff),
                    )
                    if job_cur.rowcount != 1:
                        continue
                    failed_run_ids: list[int] = []
                    for run_row in active_runs:
                        run_id = int(run_row["id"])
                        metric_phase = str(run_row["status"] or "") in {
                            "metric_queued",
                            "metric_running",
                        }
                        run_error = dict(error)
                        run_error["run_id"] = run_id
                        run_cur = conn.execute(
                            """
                            UPDATE runs
                            SET status = 'failed', error_json = ?,
                                content_revision = content_revision + 1,
                                finished_at = ?, updated_at = ?
                            WHERE id = ?
                              AND status NOT IN (
                                  'completed', 'failed', 'cancel_requested', 'canceled'
                              )
                            """,
                            (_json(run_error), now, now, run_id),
                        )
                        if run_cur.rowcount != 1:
                            continue
                        failed_run_ids.append(run_id)
                        conn.execute(
                            """
                            UPDATE jobs
                            SET status = 'canceled', error_json = ?, finished_at = ?
                            WHERE status = 'queued'
                              AND (
                                  id IN (SELECT job_id FROM run_jobs WHERE run_id = ?)
                                  OR id = ? OR id = ?
                                  OR CAST(json_extract(payload_json, '$.run_id') AS INTEGER) = ?
                              )
                            """,
                            (
                                _json(
                                    {
                                        "message": "sibling shard failed the run",
                                        "type": "RunCanceled",
                                    }
                                ),
                                now,
                                run_id,
                                run_row["inference_job_id"],
                                run_row["metric_job_id"],
                                run_id,
                            ),
                        )
                        if not metric_phase:
                            conn.execute(
                                """
                                UPDATE media_assets
                                SET state = 'unavailable', updated_at = ?
                                WHERE source_kind = 'run_artifact'
                                  AND id IN (
                                      SELECT asset_id FROM run_media_assets
                                      WHERE run_id = ?
                                        AND (
                                            COALESCE(json_extract(metadata_json, '$.input'), 0) != 1
                                            OR COALESCE(
                                                json_extract(metadata_json, '$.compare_snapshot'), 0
                                            ) = 1
                                        )
                                  )
                                """,
                                (now, run_id),
                            )
                    self._converge_recovered_cancel_requests(conn, run_rows, now)
                    recovered.append(
                        {
                            "job_id": job_id,
                            "run_ids": failed_run_ids,
                            "action": "failed",
                            "error": error,
                        }
                    )
                    continue

                terminal_run_statuses = {str(run_row["status"] or "") for run_row in run_rows}
                if terminal_run_statuses & {"cancel_requested", "canceled"}:
                    terminal_error = {
                        "message": "Run cancellation completed while its worker was unavailable",
                        "type": "RunCanceled",
                    }
                elif terminal_run_statuses:
                    terminal_error = {
                        "message": "Run already reached a terminal state",
                        "type": "RunCanceled",
                    }
                else:
                    terminal_error = error
                terminal_status = "canceled" if run_rows else "failed"
                cur = conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, error_json = ?, finished_at = ?
                    WHERE id = ? AND status = 'running'
                      AND COALESCE(heartbeat_at, started_at, created_at) < ?
                    """,
                    (terminal_status, _json(terminal_error), now, job_id, cutoff),
                )
                if cur.rowcount != 1:
                    continue
                self._converge_recovered_cancel_requests(conn, run_rows, now)
                recovered.append(
                    {
                        "job_id": job_id,
                        "run_ids": [int(run_row["id"]) for run_row in run_rows],
                        "action": terminal_status,
                        "error": terminal_error,
                    }
                )
        return recovered

    @staticmethod
    def _converge_recovered_cancel_requests(
        conn: sqlite3.Connection,
        run_rows: Iterable[sqlite3.Row],
        now: float,
    ) -> None:
        """Finish cancellation after recovery fenced the last running Job."""
        cancel_error = {"message": "User canceled the Run", "type": "RunCanceled"}
        for run_row in run_rows:
            if str(run_row["status"] or "") != "cancel_requested":
                continue
            run_id = int(run_row["id"])
            running = conn.execute(
                """
                SELECT 1
                FROM jobs j
                WHERE j.status = 'running'
                  AND (
                      j.id IN (SELECT job_id FROM run_jobs WHERE run_id = ?)
                      OR j.id = ? OR j.id = ?
                      OR CAST(json_extract(j.payload_json, '$.run_id') AS INTEGER) = ?
                  )
                LIMIT 1
                """,
                (
                    run_id,
                    run_row["inference_job_id"],
                    run_row["metric_job_id"],
                    run_id,
                ),
            ).fetchone()
            if running is None:
                conn.execute(
                    """
                    UPDATE runs
                    SET status = 'canceled', error_json = ?,
                        content_revision = content_revision + 1,
                        finished_at = ?, updated_at = ?
                    WHERE id = ? AND status = 'cancel_requested'
                    """,
                    (_json(cancel_error), now, now, run_id),
                )

    def add_artifact(
        self,
        job_id: int,
        sample_id: int | None,
        kind: str,
        path: str,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_ts()
        stored_metadata = artifact_storage_metadata(path, metadata)
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO artifacts(job_id, sample_id, kind, path, mime_type, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    sample_id,
                    kind,
                    str(Path(path).resolve()),
                    mime_type,
                    _json(stored_metadata),
                    now,
                ),
            )
            self._bump_run_revision_for_result_publish(conn, (int(job_id),), now)
            return int(cur.lastrowid)

    def add_artifacts_bulk(self, job_id: int, records: Iterable[dict[str, Any]]) -> list[int]:
        rows = [
            {
                **dict(record),
                "metadata": artifact_storage_metadata(
                    str(record.get("path") or ""),
                    record.get("metadata") or {},
                ),
            }
            for record in records
        ]
        if not rows:
            return []
        now = utc_ts()
        ids: list[int] = []
        with self.connection() as conn:
            for record in rows:
                cur = conn.execute(
                    """
                    INSERT INTO artifacts(job_id, sample_id, kind, path, mime_type, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(job_id),
                        record.get("sample_id"),
                        str(record["kind"]),
                        str(record.get("path") or ""),
                        str(record.get("mime_type") or "application/octet-stream"),
                        _json(record.get("metadata") or {}),
                        now,
                    ),
                )
                ids.append(int(cur.lastrowid))
            # Artifact batches are the publication boundary for lazy Run Detail
            # views. Bump once per committed batch (rather than per image) so
            # clients can invalidate scoped caches without turning the save pool
            # into a stream of SQLite writes.
            self._bump_run_revision_for_result_publish(conn, (int(job_id),), now)
        return ids

    @staticmethod
    def _bump_run_revision_for_result_publish(
        conn: sqlite3.Connection,
        job_ids: Iterable[int],
        now: float,
    ) -> None:
        """Mark result payloads stale when a linked job publishes results.

        A job may be a standalone legacy job, so this intentionally becomes a
        no-op unless it belongs to a Run.  Both modern ``run_jobs`` rows and the
        older ``runs.inference_job_id`` / ``metric_job_id`` links are checked.
        Keeping this update in the insert transaction means a client can never
        observe a newer revision that points at an uncommitted result batch.
        """
        ids = sorted({int(job_id) for job_id in job_ids})
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        conn.execute(
            f"""
            UPDATE runs
            SET content_revision = content_revision + 1,
                updated_at = ?
            WHERE id IN (
                SELECT run_id FROM run_jobs WHERE job_id IN ({placeholders})
                UNION
                SELECT id FROM runs
                WHERE inference_job_id IN ({placeholders})
                   OR metric_job_id IN ({placeholders})
            )
            """,
            (now, *ids, *ids, *ids),
        )

    def list_artifacts(self, job_id: int | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if job_id is not None:
            clauses.append("job_id = ?")
            params.append(job_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.query(f"SELECT * FROM artifacts{where} ORDER BY id", params)
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def list_artifacts_by_sample(self, sample_id: int, kind: str | None = None) -> list[dict[str, Any]]:
        clauses = ["sample_id = ?"]
        params: list[Any] = [sample_id]
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        rows = self.query(
            f"SELECT * FROM artifacts WHERE {' AND '.join(clauses)} ORDER BY kind, id",
            params,
        )
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def list_artifacts_by_samples(
        self,
        sample_ids: Iterable[int],
        job_ids: Iterable[int] | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        ids = [int(sample_id) for sample_id in sample_ids]
        if not ids:
            return []
        job_filter: list[int] | None = None
        if job_ids is not None:
            job_filter = [int(job_id) for job_id in job_ids]
            if not job_filter:
                return []
        fixed_param_count = (len(job_filter) if job_filter is not None else 0) + (1 if kind is not None else 0)
        chunk_size = max(1, 900 - fixed_param_count)
        rows: list[dict[str, Any]] = []
        for offset in range(0, len(ids), chunk_size):
            chunk = ids[offset : offset + chunk_size]
            clauses = [f"sample_id IN ({','.join('?' for _ in chunk)})"]
            params: list[Any] = list(chunk)
            if job_filter is not None:
                clauses.append(f"job_id IN ({','.join('?' for _ in job_filter)})")
                params.extend(job_filter)
            if kind is not None:
                clauses.append("kind = ?")
                params.append(kind)
            rows.extend(
                self.query(
                    f"SELECT * FROM artifacts WHERE {' AND '.join(clauses)} ORDER BY sample_id, kind, id",
                    params,
                )
            )
        rows.sort(key=lambda row: (int(row["sample_id"]), str(row["kind"]), int(row["id"])))
        for row in rows:
            row["metadata"] = _loads(row.pop("metadata_json"))
        return rows

    def add_metric_result(
        self,
        job_id: int,
        inference_job_id: int,
        sample_id: int | None,
        metric_name: str,
        status: str,
        value: float | None,
        details: dict[str, Any] | None = None,
    ) -> int:
        now = utc_ts()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO metric_results(job_id, inference_job_id, sample_id, metric_name, status, value, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, inference_job_id, sample_id, metric_name, status, value, _json(details), now),
            )
            self._bump_run_revision_for_result_publish(
                conn,
                (int(job_id), int(inference_job_id)),
                now,
            )
            return int(cur.lastrowid)

    def list_metric_results(self, inference_job_id: int | None = None) -> list[dict[str, Any]]:
        if inference_job_id is None:
            rows = self.query("SELECT * FROM metric_results ORDER BY id")
        else:
            rows = self.query(
                "SELECT * FROM metric_results WHERE inference_job_id = ? ORDER BY id",
                (inference_job_id,),
            )
        for row in rows:
            row["details"] = _loads(row.pop("details_json"))
        return rows

    def list_metrics_by_sample(self, sample_id: int, metric_name: str | None = None) -> list[dict[str, Any]]:
        clauses = ["sample_id = ?"]
        params: list[Any] = [sample_id]
        if metric_name is not None:
            clauses.append("metric_name = ?")
            params.append(metric_name)
        rows = self.query(
            f"SELECT * FROM metric_results WHERE {' AND '.join(clauses)} ORDER BY metric_name, id",
            params,
        )
        for row in rows:
            row["details"] = _loads(row.pop("details_json"))
        return rows

    def list_metrics_by_samples(
        self,
        sample_ids: Iterable[int],
        inference_job_ids: Iterable[int] | None = None,
        metric_name: str | None = None,
    ) -> list[dict[str, Any]]:
        ids = [int(sample_id) for sample_id in sample_ids]
        if not ids:
            return []
        job_filter: list[int] | None = None
        if inference_job_ids is not None:
            job_filter = [int(job_id) for job_id in inference_job_ids]
            if not job_filter:
                return []
        fixed_param_count = (len(job_filter) if job_filter is not None else 0) + (1 if metric_name is not None else 0)
        chunk_size = max(1, 900 - fixed_param_count)
        rows: list[dict[str, Any]] = []
        for offset in range(0, len(ids), chunk_size):
            chunk = ids[offset : offset + chunk_size]
            clauses = [f"sample_id IN ({','.join('?' for _ in chunk)})"]
            params: list[Any] = list(chunk)
            if job_filter is not None:
                clauses.append(f"inference_job_id IN ({','.join('?' for _ in job_filter)})")
                params.extend(job_filter)
            if metric_name is not None:
                clauses.append("metric_name = ?")
                params.append(metric_name)
            rows.extend(
                self.query(
                    f"SELECT * FROM metric_results WHERE {' AND '.join(clauses)} ORDER BY sample_id, metric_name, id",
                    params,
                )
            )
        rows.sort(key=lambda row: (int(row["sample_id"]), str(row["metric_name"]), int(row["id"])))
        for row in rows:
            row["details"] = _loads(row.pop("details_json"))
        return rows

    def get_metric_cache(self, cache_key: str) -> dict[str, Any] | None:
        row = self.get("SELECT * FROM metric_cache WHERE cache_key = ?", (cache_key,))
        if row is None:
            return None
        row["details"] = _loads(row.pop("details_json"))
        return row

    def set_metric_cache(
        self,
        cache_key: str,
        metric_name: str,
        status: str,
        value: float | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        now = utc_ts()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO metric_cache(cache_key, metric_name, status, value, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    metric_name=excluded.metric_name,
                    status=excluded.status,
                    value=excluded.value,
                    details_json=excluded.details_json,
                    created_at=excluded.created_at
                """,
                (cache_key, metric_name, status, value, _json(details), now),
            )

    @staticmethod
    def _decode_job(row: dict[str, Any]) -> None:
        row["payload"] = _loads(row.pop("payload_json"))
        row["result"] = _loads(row.pop("result_json"))
        row["error"] = _loads(row.pop("error_json"))

    @staticmethod
    def _decode_run(row: dict[str, Any]) -> None:
        row["metrics"] = _loads(row.pop("metrics_json"))
        row["artifact_summary"] = _loads(row.pop("artifact_summary_json"))
        row["metric_summary"] = _loads(row.pop("metric_summary_json"))
        row["result"] = _loads(row.pop("result_json"))
        row["error"] = _loads(row.pop("error_json"))
        row["metadata"] = _loads(row.pop("metadata_json"))

    @staticmethod
    def _decode_run_purge_request(row: dict[str, Any]) -> None:
        row["report"] = _loads(row.pop("report_json", None))
        row["error"] = _loads(row.pop("error_json", None))

    @staticmethod
    def _decode_cache_entry(row: dict[str, Any]) -> None:
        row["metadata"] = _loads(row.pop("metadata_json", None))
