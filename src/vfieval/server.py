from __future__ import annotations

import errno
import hashlib
import json
import logging
import mimetypes
import os
import re
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote, unquote, urlparse

from vfieval.api_validation import pagination_params, query_int, query_value
from vfieval.catalog_sync import CatalogSyncCoordinator
from vfieval.config import WorkspaceConfig
from vfieval.datasets import _sample_token, scan_dataset
from vfieval.db import Database
from vfieval.diagnostics import release_info
from vfieval.file_inputs import (
    DECODE_STRATEGY_VERSION,
    MIDPOINT_TRIPLET_CONTRACT,
    VIDEO_SUFFIXES,
    checkpoints_dir,
    ensure_video_thumbnail,
    list_checkpoints,
    list_model_files,
    models_dir,
    normalize_decode_backend,
    normalize_device_precision,
    preflight_run,
    resolve_checkpoint,
    resolve_model_file,
    resolve_run_dimensions,
    resolve_video_group,
    resolve_video_selection,
    thumbnail_path,
    video_thumbnail_key,
    video_summary,
    videos_dir,
)
from vfieval.metrics import METRIC_NAMES
from vfieval.metrics.health import metrics_health
from vfieval.input_identity import (
    InputIdentityChanged,
    assert_input_identity_files_available,
    assert_input_identity_matches,
    build_checkpoint_identity,
    build_file_identity,
    build_file_identity_from_values,
    build_run_input_identity,
    build_source_identity,
    compare_input_identities,
    normalize_request_identity,
    resolved_checkpoint_relative_path,
    validate_input_identity,
)
from vfieval.job_leases import JobRecoveryService
from vfieval.job_api import (
    JobApiError,
    claim_job_request,
    create_job_request,
    heartbeat_job_request,
    job_callback_request,
    register_worker_request,
)
from vfieval.media_assets import (
    bind_run_asset,
    create_collection,
    ensure_folder_asset,
    get_asset,
    list_assets,
    list_collections,
    list_folder_group_videos,
    list_folder_groups,
    media_audit,
    resolve_asset_path,
    soft_delete_asset,
    source_assets_to_video_payload,
    sync_run_assets,
)
from vfieval.media_items import (
    bind_compare_input,
    bind_run_source,
    ensure_canonical_gt_item,
    get_media_item,
    list_item_groups,
    list_item_predictions,
    list_media_items,
    list_methods_for_items,
    list_unbound_predictions,
    register_external_prediction,
    resolve_media_item_compare,
    resolve_item_member,
)
from vfieval.orchestration import (
    JobSupervisor,
    open_worker_admission,
    shutdown_worker_processes,
    start_decode_worker,
    start_workers_for_run,
    wake_job_supervisor,
)
from vfieval.run_cleanup import (
    CacheCoordinationUnavailable,
    RunCleanupService,
    RunPurgePreviewError,
    register_run_cache_refs,
)
from vfieval.runtime_logging import (
    close_runtime_logging,
    configure_runtime_logging,
    log_event,
    runtime_logger,
)
from vfieval.runtime_api import runtime_health_payload
from vfieval.storage_budget import (
    StorageCapacityError,
    ensure_storage_capacity,
    storage_capacity,
)
from vfieval.selection_tokens import (
    SelectionTokenExpired,
    create_selection_snapshot,
    list_methods_for_selection_snapshot,
    selection_snapshot_page,
)
from vfieval.submissions import (
    SubmissionConflict,
    idempotent_create,
    submission_fingerprint,
)
from vfieval.video_selection_tokens import (
    VideoSelectionTokenExpired,
    create_video_selection_snapshot,
    ensure_video_selection_schema,
    expand_video_selection_payload,
    video_selection_membership,
    video_selection_snapshot_page,
)
from vfieval.pipeline.inference import DEFAULT_VISUALIZE_HEIGHT, DEFAULT_VISUALIZE_WIDTH
from vfieval.performance import recommend_execution_profile
from vfieval.worker import detect_capabilities
from vfieval.uploads import (
    UPLOAD_CHUNK_SIZE,
    complete_upload_session,
    cleanup_stale_uploads,
    create_upload_session,
    delete_upload_session,
    get_upload_session,
    receive_upload_part,
)

from vfieval.evaluations import (
    analysis_csv,
    campaign_analysis,
    campaign_export,
    campaign_export_csv,
    get_campaign,
    list_candidates,
)
from vfieval.evaluations_v2 import (
    EvaluationConflict,
    archive_campaign_v2,
    archive_legacy_campaign,
    blind_heartbeat,
    blind_media_asset,
    blind_payload,
    blind_review_task,
    blind_reviews,
    blind_session,
    blind_submit_vote,
    campaign_analysis_v2,
    campaign_export_v2,
    campaign_objective_curve_v2,
    close_campaign_v2,
    create_campaign_v2,
    delete_campaign_v2,
    discard_empty_legacy_draft,
    ensure_v2_schema,
    get_campaign_v2,
    get_preparation_v2,
    list_campaign_summaries_page,
    list_campaign_purge_requests_v2,
    preview_campaign_v2,
    process_campaign_purge_requests_v2,
    request_publish_campaign_v2,
    run_pending_preparations,
)
from vfieval.workload import estimate_workload, workload_confirmation_scope_fingerprint


CANONICAL_ARTIFACT_CONTRACT = "canonical-v1"
TERMINAL_RUN_STATUSES = {"completed", "failed", "canceled"}
COMPARE_SOURCE_RUN_STATUSES = {"completed", "metric_queued", "metric_running"}
MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024

_CLIENT_DISCONNECT_ERRNOS = frozenset(
    {
        errno.EPIPE,
        errno.ECONNABORTED,
        errno.ECONNRESET,
        errno.ENOTCONN,
        errno.ESHUTDOWN,
    }
)
_CLIENT_DISCONNECT_WINERRORS = frozenset(
    {
        109,  # ERROR_BROKEN_PIPE
        10053,  # WSAECONNABORTED
        10054,  # WSAECONNRESET
        10057,  # WSAENOTCONN
        10058,  # WSAESHUTDOWN
    }
)
_CLONE_RISK_ACK_TTL_SECONDS = 600.0
_CLONE_RISK_ACK_LOCK = threading.Lock()
_CLONE_RISK_ACK_CHALLENGES: dict[tuple[str, int, str], dict[str, Any]] = {}


class _RequestBodyTooLarge(ValueError):
    pass


class _WorkloadRiskConfirmationRequired(ValueError):
    def __init__(self, workload: dict[str, Any]):
        self.workload = dict(workload)
        super().__init__("high-risk workload requires explicit confirmation")

    def public_payload(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__.lstrip("_"),
            "message": str(self),
            "workload": self.workload,
        }


class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Thread-per-request server with an explicit concurrency ceiling."""

    daemon_threads = True
    block_on_close = True
    request_queue_size = 128

    def __init__(self, server_address, request_handler_class, *, max_threads: int = 64):
        self._request_slots = threading.BoundedSemaphore(max(1, int(max_threads)))
        super().__init__(server_address, request_handler_class)

    def get_request(self):
        request, client_address = super().get_request()
        request.settimeout(120.0)
        return request, client_address

    def process_request(self, request, client_address) -> None:
        if not self._request_slots.acquire(blocking=False):
            payload = b'{"error":{"type":"ServerBusy","message":"server request capacity reached; retry shortly"}}'
            response = (
                b"HTTP/1.1 503 Service Unavailable\r\n"
                b"Content-Type: application/json; charset=utf-8\r\n"
                b"Cache-Control: no-store\r\n"
                b"Retry-After: 2\r\n"
                + f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode("ascii")
                + payload
            )
            try:
                request.sendall(response)
            except OSError:
                pass
            self.shutdown_request(request)
            log_event(
                logging.WARNING,
                "http.capacity_rejected",
                "request rejected because the HTTP worker limit was reached",
                client=str(client_address[0]) if client_address else "",
            )
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._request_slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()

    def handle_error(self, request, client_address) -> None:
        runtime_logger().exception(
            "uncaught HTTP request error",
            extra={
                "event": "http.uncaught_error",
                "details": {"client": str(client_address[0]) if client_address else ""},
            },
        )


def _is_client_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
        return True
    if not isinstance(exc, OSError):
        return False
    error_number = getattr(exc, "errno", None)
    winerror = getattr(exc, "winerror", None)
    return (
        error_number in _CLIENT_DISCONNECT_ERRNOS
        or error_number in _CLIENT_DISCONNECT_WINERRORS
        or winerror in _CLIENT_DISCONNECT_WINERRORS
    )


def run_server(db: Database, workspace: WorkspaceConfig, host: str, port: int) -> None:
    open_worker_admission()
    log_path = configure_runtime_logging(workspace)
    cleanup_stale_uploads(db, workspace)
    ensure_v2_schema(db)
    ensure_video_selection_schema(db)
    cleanup_service = RunCleanupService(db, workspace)
    catalog_sync = CatalogSyncCoordinator(db, workspace)
    preparation_stop = threading.Event()
    preparation_wake = threading.Event()
    job_recovery = JobRecoveryService(
        db,
        on_recovered=lambda rows: _log_recovered_jobs(rows),
    )
    job_supervisor = JobSupervisor(db, workspace)
    handler = _make_handler(
        db,
        workspace,
        cleanup_service=cleanup_service,
        catalog_sync=catalog_sync,
        preparation_wake_event=preparation_wake,
        job_recovery=job_recovery,
        job_supervisor=job_supervisor,
    )
    try:
        max_http_threads = max(1, int(os.getenv("VFIEVAL_HTTP_MAX_THREADS", "64")))
    except ValueError:
        max_http_threads = 64
    server = _BoundedThreadingHTTPServer((host, port), handler, max_threads=max_http_threads)
    cleanup_stop = threading.Event()
    cleanup_thread = threading.Thread(
        target=cleanup_service.run_forever,
        args=(cleanup_stop,),
        name="vfieval-run-cleanup",
        daemon=True,
    )
    cleanup_thread.start()
    preparation_thread = threading.Thread(
        target=_run_evaluation_preparations_forever,
        args=(db, workspace, preparation_stop, preparation_wake),
        name="vfieval-evaluation-preparation",
        daemon=True,
    )
    preparation_thread.start()
    job_recovery.start()
    job_supervisor.start()
    catalog_sync.request_sync(include_runs=True)
    build = release_info()
    log_event(
        logging.INFO,
        "server.started",
        f"VFIEval listening on http://{host}:{port}",
        host=host,
        port=port,
        max_http_threads=max_http_threads,
        log_path=str(log_path),
        build_id=build["build_id"],
        version=build["version"],
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_stop.set()
        preparation_stop.set()
        preparation_wake.set()
        job_supervisor.stop(timeout=5)
        job_recovery.stop(timeout=5)
        cleanup_thread.join(timeout=5)
        preparation_thread.join(timeout=30)
        if preparation_thread.is_alive():
            log_event(
                logging.WARNING,
                "campaign.preparation_shutdown_timeout",
                "Campaign preparation did not reach a cancellation checkpoint before shutdown timeout",
            )
        catalog_sync.wait(timeout=5)
        worker_shutdown = shutdown_worker_processes(timeout=5)
        if (
            worker_shutdown["tracked"]
            or worker_shutdown["remaining"]
            or worker_shutdown["threads_tracked"]
            or worker_shutdown["threads_remaining"]
        ):
            log_event(
                logging.INFO
                if not (worker_shutdown["remaining"] or worker_shutdown["threads_remaining"])
                else logging.WARNING,
                "worker_processes.stopped",
                "Local accelerator worker processes were stopped with the server",
                **worker_shutdown,
            )
        server.server_close()
        log_event(logging.INFO, "server.stopped", "VFIEval server stopped")
        close_runtime_logging()


def _log_recovered_jobs(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        error = row.get("error") if isinstance(row.get("error"), dict) else {}
        log_event(
            logging.ERROR,
            "job.worker_lost",
            "A stale worker Job was fenced and its Run failed safely",
            job_id=row.get("job_id"),
            run_ids=row.get("run_ids"),
            action=row.get("action"),
            error_type=error.get("type"),
            last_heartbeat_at=error.get("last_heartbeat_at"),
        )


def _run_evaluation_preparations_forever(
    db: Database,
    workspace: WorkspaceConfig,
    stop_event: threading.Event,
    wake_event: threading.Event | None = None,
) -> None:
    while not stop_event.is_set():
        try:
            results = run_pending_preparations(
                db,
                workspace,
                limit=1,
                cancel_check=stop_event.is_set,
            )
            _log_evaluation_preparation_results(db, workspace, results)
        except Exception as exc:
            runtime_logger().exception(
                "evaluation preparation loop failed",
                extra={"event": "campaign.preparation_loop_failed", "details": {"error_type": type(exc).__name__}},
            )
        try:
            process_campaign_purge_requests_v2(db, workspace, limit=20)
        except Exception as exc:
            runtime_logger().exception(
                "evaluation cleanup loop failed",
                extra={"event": "campaign.cleanup_loop_failed", "details": {"error_type": type(exc).__name__}},
            )
        if wake_event is None:
            stop_event.wait(1.0)
        else:
            wake_event.wait(1.0)
            wake_event.clear()


def _log_evaluation_preparation_results(
    db: Database,
    workspace: WorkspaceConfig,
    results: list[dict[str, Any]],
) -> None:
    """Log one sanitized terminal line per claimed Campaign preparation."""

    for result in results:
        try:
            campaign_id = int(result.get("campaign_id") or 0)
        except (TypeError, ValueError):
            continue
        if campaign_id <= 0:
            continue
        status = str(result.get("status") or "").strip().lower()
        if status == "published":
            log_event(
                logging.INFO,
                "campaign.preparation_published",
                "Campaign preparation published",
                campaign_id=campaign_id,
            )
            continue
        if status != "failed":
            continue
        error = result.get("error") if isinstance(result.get("error"), dict) else {}
        error_type = re.sub(
            r"[^A-Za-z0-9_.-]",
            "",
            str(error.get("type") or "Error"),
        )[:80] or "Error"
        message = _sanitize_evaluation_preparation_log_message(
            db,
            workspace,
            campaign_id,
            error.get("message"),
        )
        log_event(
            logging.ERROR,
            "campaign.preparation_failed",
            message,
            campaign_id=campaign_id,
            error_type=error_type,
        )


def _sanitize_evaluation_preparation_log_message(
    db: Database,
    workspace: WorkspaceConfig,
    campaign_id: int,
    value: Any,
) -> str:
    message = str(value or "unknown error")
    sensitive_values: list[tuple[str, str]] = []
    for root in sorted(
        {workspace.root.resolve(), workspace.root.parent.resolve()},
        key=lambda path: len(str(path)),
        reverse=True,
    ):
        sensitive_values.append((str(root), "<path>"))
    try:
        rows = db.query(
            """
            SELECT c.public_token, m.label_snapshot, m.source_track_label
            FROM evaluation_campaigns_v2 c
            LEFT JOIN evaluation_methods_v2 m ON m.campaign_id = c.id
            WHERE c.id = ?
            """,
            (int(campaign_id),),
        )
    except Exception:
        rows = []
    for row in rows:
        sensitive_values.extend(
            (
                (str(row.get("public_token") or ""), "<token>"),
                (str(row.get("label_snapshot") or ""), "<method>"),
                (str(row.get("source_track_label") or ""), "<method>"),
            )
        )
    for raw, replacement in sensitive_values:
        if not raw:
            continue
        for variant in {raw, raw.replace("\\", "/"), raw.replace("/", "\\")}:
            if variant:
                pattern = re.escape(variant)
                if variant[0].isalnum() or variant[0] == "_":
                    pattern = rf"(?<![A-Za-z0-9_]){pattern}"
                if replacement == "<path>":
                    pattern = rf"{pattern}(?:[\\/][^\r\n,;:()]*)*"
                if variant[-1].isalnum() or variant[-1] == "_":
                    pattern = rf"{pattern}(?![A-Za-z0-9_])"
                message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
    message = re.sub(r"https?://\S+", "<url>", message, flags=re.IGNORECASE)
    message = re.sub(
        r"([\"'])(?:[A-Za-z]:[\\/]|/)[^\"']+\1",
        '"<path>"',
        message,
    )
    message = re.sub(
        r"(?i)(?<![A-Za-z0-9_])(?:[A-Z]:[\\/]|\\\\)[^\s,;]+",
        "<path>",
        message,
    )
    message = re.sub(
        r"(?<![A-Za-z0-9_])/(?:[^/\s]+/)+[^\s,;]*",
        "<path>",
        message,
    )
    message = re.sub(
        r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{24,}(?![A-Za-z0-9_-])",
        "<redacted>",
        message,
    )
    message = " ".join(message.split())
    if len(message) > 1200:
        message = message[:1197] + "..."
    return message or "unknown error"


def _make_handler(
    db: Database,
    workspace: WorkspaceConfig,
    cleanup_service: RunCleanupService | None = None,
    catalog_sync: CatalogSyncCoordinator | None = None,
    preparation_wake_event: threading.Event | None = None,
    job_recovery: JobRecoveryService | None = None,
    job_supervisor: JobSupervisor | None = None,
):
    # Schema/backfill is cheap and idempotent. Catalog reconciliation itself is
    # owned by the coordinator: production starts it in the background and all
    # catalog GET routes remain read-only SQLite snapshots.
    ensure_v2_schema(db)
    ensure_video_selection_schema(db)
    if catalog_sync is None:
        # Direct handler construction is primarily used by deterministic test
        # servers. Production passes a coordinator and schedules this work in
        # the background from ``run_server``.
        catalog_sync = CatalogSyncCoordinator(db, workspace)
        catalog_sync.request_sync(include_runs=True)
        catalog_sync.wait()
    owns_cleanup_service = cleanup_service is None
    cleanup_service = cleanup_service or RunCleanupService(db, workspace)
    if owns_cleanup_service:
        # Direct handler construction is used by embedded/test servers that do
        # not own the production cleanup-loop lifecycle. Finish the tiny local
        # coordination synchronously so no daemon thread can retain the SQLite
        # file after that server is discarded.
        cleanup_service.backfill_cache_catalog()
    else:
        cleanup_service.start_cache_coordination()
    preflight_cache_lock = threading.Lock()
    preflight_cache: dict[str, dict[str, Any]] = {}

    def store_preflight(request: dict[str, Any], result: dict[str, Any]) -> str:
        token = uuid.uuid4().hex
        now = time.time()
        physical_inputs = _physical_inputs_from_deep_preflight(request, result)
        if physical_inputs is not None:
            result["input_fingerprint"] = hashlib.sha256(
                json.dumps(
                    physical_inputs,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        with preflight_cache_lock:
            expired = [
                key
                for key, row in preflight_cache.items()
                if now - float(row.get("created_at") or 0.0) > 600.0
            ]
            for key in expired:
                preflight_cache.pop(key, None)
            preflight_cache[token] = {
                "created_at": now,
                "request_fingerprint": _preflight_request_fingerprint(request),
                "result": json.loads(json.dumps(result, default=str)),
                "physical_inputs": physical_inputs,
            }
        return token

    def cached_preflight(request: dict[str, Any]) -> dict[str, Any] | None:
        token = str(request.get("preflight_token") or "").strip()
        if not token:
            return None
        with preflight_cache_lock:
            row = preflight_cache.get(token)
        if row is None or time.time() - float(row.get("created_at") or 0.0) > 600.0:
            raise ValueError("preflight token expired; run preflight again")
        if row.get("request_fingerprint") != _preflight_request_fingerprint(request):
            raise ValueError("Run configuration changed after preflight; run preflight again")
        physical_inputs = row.get("physical_inputs")
        if isinstance(physical_inputs, dict):
            request["_preflight_physical_inputs"] = _revalidate_preflight_physical_inputs(
                workspace,
                request,
                physical_inputs,
            )
        result = json.loads(json.dumps(row["result"], default=str))
        result["preflight_cache_hit"] = True
        return result

    class VFIEvalHandler(BaseHTTPRequestHandler):
        server_version = "VFIEval/0.1"

        def handle_one_request(self) -> None:
            self.request_id = uuid.uuid4().hex[:16]
            try:
                super().handle_one_request()
            except OSError as exc:
                if not self._ignore_client_disconnect(exc):
                    raise

        def _ignore_client_disconnect(self, exc: BaseException) -> bool:
            if not _is_client_disconnect(exc):
                return False
            self.close_connection = True
            log_event(
                logging.INFO,
                "http.client_disconnected",
                "client disconnected while the server was writing a response",
                request_id=getattr(self, "request_id", ""),
                method=getattr(self, "command", ""),
                path=urlparse(getattr(self, "path", "")).path,
                client=str(self.client_address[0]) if getattr(self, "client_address", None) else "",
                error_type=type(exc).__name__,
                errno=getattr(exc, "errno", None),
                winerror=getattr(exc, "winerror", None),
            )
            return True

        def _client_io(self, operation: Any, *args: Any) -> bool:
            try:
                operation(*args)
            except OSError as exc:
                if not self._ignore_client_disconnect(exc):
                    raise
                return False
            return True

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)
                if path == "/":
                    return self._send_static("index.html")
                if re.fullmatch(r"/evaluate/[A-Za-z0-9_-]+", path):
                    return self._send_static("blind.html")
                if path in {
                    "/app.js",
                    "/shared.js",
                    "/compare.js",
                    "/run-detail.js",
                    "/media.js",
                    "/styles.css",
                    "/studio.js",
                    "/studio.css",
                    "/blind.js",
                    "/blind.css",
                }:
                    return self._send_static(path.lstrip("/"))
                if path == "/api/health":
                    return self._json(
                        runtime_health_payload(
                            db,
                            workspace,
                            catalog_sync=catalog_sync,
                            cleanup_service=cleanup_service,
                            job_recovery=job_recovery,
                            job_supervisor=job_supervisor,
                        )
                    )
                if path == "/api/storage/gc/preview":
                    return self._json(
                        cleanup_service.gc_preview(
                            _query_int_values(query, "entry_id") or None,
                            _query_int_values(query, "run_id") or None,
                        )
                    )
                match = re.fullmatch(r"/api/run-purge-requests/(\d+)", path)
                if match:
                    return self._json(db.get_run_purge_request_by_id(int(match.group(1))))
                if path == "/api/dashboard":
                    return self._json(_dashboard(db))
                if path == "/api/model-files":
                    return self._json(list_model_files(workspace))
                if path == "/api/checkpoints":
                    return self._json(list_checkpoints(workspace, query.get("model_file", [None])[0]))
                if path == "/api/devices":
                    return self._json(detect_capabilities())
                if path == "/api/media/sync/status":
                    return self._json(catalog_sync.status())
                if path == "/api/media/collections":
                    return self._json({"collections": list_collections(db)})
                if path == "/api/media/assets":
                    collection_id = _optional_int(query.get("collection_id", [None])[0])
                    requested_source_kind = query.get("source_kind", [None])[0] or None
                    return self._json(
                        list_assets(
                            db,
                            collection_id=collection_id,
                            role=query.get("role", [None])[0] or None,
                            source_kind=requested_source_kind,
                            valid_run_outputs=requested_source_kind == "run_artifact",
                            state=query.get("state", ["ready"])[0] or None,
                            query=query.get("q", [""])[0],
                            page=int(query.get("page", ["1"])[0] or 1),
                            page_size=int(query.get("page_size", ["50"])[0] or 50),
                        )
                    )
                if path == "/api/media/sources":
                    return self._json(
                        list_assets(
                            db,
                            role=query.get("role", [None])[0] or None,
                            source_kinds=["folder", "upload"],
                            state=query.get("state", ["ready"])[0] or None,
                            query=query.get("q", [""])[0],
                            page=int(query.get("page", ["1"])[0] or 1),
                            page_size=int(query.get("page_size", ["50"])[0] or 50),
                        )
                    )
                if path == "/api/media/audit":
                    return self._json(media_audit(db))
                if path == "/api/media/run-outputs":
                    return self._json({"runs": _bound_run_outputs(db)})
                if path == "/api/media/item-groups":
                    if query.get("role", ["gt"])[0] not in {"", "gt"}:
                        raise ValueError("media item groups currently support role=gt only")
                    return self._json(list_item_groups(db))
                if path == "/api/media/items":
                    group_id = _optional_int(query.get("group_id", [None])[0])
                    if group_id is None:
                        raise ValueError("group_id is required")
                    return self._json(
                        list_media_items(
                            db,
                            group_id,
                            query=query.get("q", [""])[0],
                            page=int(query.get("page", ["1"])[0] or 1),
                            page_size=int(query.get("page_size", ["50"])[0] or 50),
                        )
                    )
                match = re.fullmatch(
                    r"/api/media/item-selections/([A-Za-z0-9_-]{32,128})",
                    path,
                )
                if match:
                    try:
                        return self._json(
                            selection_snapshot_page(
                                db,
                                match.group(1),
                                page=int(query.get("page", ["1"])[0] or 1),
                                page_size=int(query.get("page_size", ["100"])[0] or 100),
                            )
                        )
                    except SelectionTokenExpired as exc:
                        return self._error(HTTPStatus.GONE, str(exc), type(exc).__name__)
                match = re.fullmatch(r"/api/media/items/(\d+)/predictions", path)
                if match:
                    return self._json(list_item_predictions(db, int(match.group(1))))
                if path == "/api/media/methods":
                    item_ids = _query_int_values(query, "item_id")
                    selection_token = str(query.get("selection_token", [""])[0] or "").strip()
                    if item_ids and selection_token:
                        raise ValueError("provide either item_id or selection_token, not both")
                    if selection_token:
                        return self._json(
                            list_methods_for_selection_snapshot(db, selection_token)
                        )
                    if not item_ids:
                        raise ValueError("at least one item_id or selection_token is required")
                    return self._json(list_methods_for_items(db, item_ids))
                if path == "/api/media/unbound-predictions":
                    return self._json(
                        list_unbound_predictions(
                            db,
                            page=int(query.get("page", ["1"])[0] or 1),
                            page_size=int(query.get("page_size", ["50"])[0] or 50),
                        )
                    )
                match = re.fullmatch(r"/api/media/assets/(\d+)(?:/(content))?", path)
                if match:
                    asset_id = int(match.group(1))
                    if match.group(2) == "content":
                        _asset, media_path = resolve_asset_path(db, workspace, asset_id)
                        if media_path.is_dir():
                            files = sorted(child for child in media_path.iterdir() if child.is_file())
                            if not files:
                                return self._error(HTTPStatus.NOT_FOUND, "frame sequence is empty")
                            media_path = files[0]
                        return self._send_file(media_path)
                    return self._json(get_asset(db, asset_id))
                match = re.fullmatch(r"/api/uploads/([a-f0-9]{32})", path)
                if match:
                    return self._json(get_upload_session(db, match.group(1)))
                if path == "/api/evaluation-campaigns":
                    return self._json(
                        list_campaign_summaries_page(
                            db,
                            page=int(query.get("page", ["1"])[0] or 1),
                            page_size=int(query.get("page_size", ["50"])[0] or 50),
                            query=query.get("q", [""])[0],
                            status=query.get("status", [""])[0],
                        )
                    )
                if path == "/api/evaluation-cleanup-requests":
                    include_completed = query.get("include_completed", ["0"])[0] in {
                        "1",
                        "true",
                        "yes",
                    }
                    return self._json(
                        {
                            "requests": list_campaign_purge_requests_v2(
                                db,
                                include_completed=include_completed,
                            )
                        }
                    )
                match = re.fullmatch(
                    r"/api/evaluation-campaigns/v2/(\d+)(?:/(analysis|export|preparation|objective-curve))?",
                    path,
                )
                if match:
                    campaign_id = int(match.group(1))
                    section = match.group(2)
                    if section == "analysis":
                        return self._json(campaign_analysis_v2(db, campaign_id))
                    if section == "objective-curve":
                        item_id = _optional_int(query.get("item_id", [None])[0])
                        if item_id is None:
                            return self._error(HTTPStatus.BAD_REQUEST, "item_id is required")
                        return self._json(
                            campaign_objective_curve_v2(
                                db,
                                campaign_id,
                                item_id,
                                str(query.get("metric_name", [""])[0] or ""),
                            )
                        )
                    if section == "preparation":
                        return self._json(
                            {
                                "campaign_id": campaign_id,
                                "preparation": get_preparation_v2(db, campaign_id),
                            }
                        )
                    if section == "export":
                        payload = json.dumps(campaign_export_v2(db, campaign_id), indent=2, ensure_ascii=False, default=str).encode("utf-8")
                        return self._send_bytes(
                            payload,
                            "application/json; charset=utf-8",
                            f"campaign-v2-{campaign_id}-export.json",
                        )
                    return self._json(_evaluation_campaign_v2_payload(db, campaign_id))
                match = re.fullmatch(r"/api/blind/([A-Za-z0-9_-]+)(?:/tasks/([A-Za-z0-9_-]+)/media/(reference|left|right))?", path)
                review_match = re.fullmatch(
                    r"/api/blind/([A-Za-z0-9_-]+)/reviews(?:/([A-Za-z0-9_-]+))?",
                    path,
                )
                if review_match:
                    token, task_token = review_match.groups()
                    evaluator_id = str(query.get("evaluator_id", [""])[0]).strip()
                    if not evaluator_id:
                        return self._error(HTTPStatus.BAD_REQUEST, "evaluator_id is required")
                    if task_token:
                        return self._json(
                            blind_review_task(db, token, task_token, evaluator_id)
                        )
                    return self._json(blind_reviews(db, token, evaluator_id))
                if match:
                    token, task_token, side = match.groups()
                    if task_token and side:
                        assignment_token = str(query.get("assignment", [""])[0]).strip()
                        if not assignment_token:
                            return self._error(HTTPStatus.BAD_REQUEST, "opaque assignment token is required")
                        _asset, media_path = blind_media_asset(
                            db, workspace, token, task_token, side, assignment_token
                        )
                        if media_path.is_dir():
                            files = sorted(child for child in media_path.iterdir() if child.is_file())
                            if not files:
                                return self._error(HTTPStatus.NOT_FOUND, "frame sequence is empty")
                            frame_index = int(query.get("frame", ["0"])[0] or 0)
                            if frame_index < 0 or frame_index >= len(files):
                                return self._error(
                                    HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                                    "frame index is out of range",
                                )
                            media_path = files[frame_index]
                        return self._send_file(media_path, cache_control="no-store")
                    evaluator_id = str(query.get("evaluator_id", [""])[0]).strip()
                    return self._json(blind_payload(db, token, evaluator_id))
                match = re.fullmatch(r"/api/evaluation-campaigns/(\d+)(?:/(candidates|next|analysis|export))?", path)
                if match:
                    campaign_id = int(match.group(1))
                    section = match.group(2)
                    if section == "candidates":
                        return self._json({"campaign_id": campaign_id, "candidates": list_candidates(db, campaign_id)})
                    if section == "next":
                        return self._error(
                            HTTPStatus.GONE,
                            "legacy Campaign V1 is read-only; participants must use an opaque Campaign V2 /evaluate URL",
                        )
                    if section == "analysis":
                        analysis = campaign_analysis(
                            db,
                            campaign_id,
                            bootstrap_samples=int(query.get("bootstrap_samples", ["1000"])[0] or 1000),
                            filters={
                                "video": query.get("video", [""])[0],
                                "model": query.get("model", [""])[0],
                                "checkpoint": query.get("checkpoint", [""])[0],
                                "collection_id": query.get("collection_id", [""])[0],
                                "evaluator_id": query.get("evaluator_id", [""])[0],
                            },
                        )
                        if query.get("format", ["json"])[0] == "csv":
                            return self._send_bytes(
                                analysis_csv(analysis),
                                "text/csv; charset=utf-8",
                                f"campaign-{campaign_id}-analysis.csv",
                            )
                        return self._json(analysis)
                    if section == "export":
                        exported = campaign_export(db, campaign_id)
                        if query.get("format", ["json"])[0] == "csv":
                            return self._send_bytes(
                                campaign_export_csv(exported),
                                "text/csv; charset=utf-8",
                                f"campaign-{campaign_id}-export.csv",
                            )
                        return self._json(exported)
                    return self._json(_legacy_evaluation_campaign_payload(db, campaign_id))
                match = re.fullmatch(r"/api/evaluation-tasks/(\d+)/media/(reference|left|right)", path)
                if match:
                    return self._error(
                        HTTPStatus.GONE,
                        "legacy Campaign V1 is read-only; participant media is available only through opaque Campaign V2 URLs",
                    )
                if path == "/api/video-groups":
                    summary = query.get("summary", ["0"])[0] in {"1", "true", "yes"}
                    groups = list_folder_groups(db)
                    if not summary:
                        for group in groups:
                            group["videos"] = list_folder_group_videos(
                                db,
                                str(group["name"]),
                                page=1,
                                page_size=200,
                            )["videos"]
                    return self._json(groups)
                match = re.fullmatch(
                    r"/api/video-selections/([A-Za-z0-9_-]{32,128})",
                    path,
                )
                if match:
                    try:
                        page, page_size = pagination_params(
                            query,
                            default_page_size=100,
                        )
                        return self._json(
                            video_selection_snapshot_page(
                                db,
                                workspace,
                                match.group(1),
                                page=page,
                                page_size=page_size,
                            )
                        )
                    except VideoSelectionTokenExpired as exc:
                        return self._error(
                            HTTPStatus.GONE,
                            str(exc),
                            type(exc).__name__,
                        )
                match = re.fullmatch(r"/api/video-groups/([^/]+)/videos", path)
                if match:
                    group_name = unquote(match.group(1))
                    page, page_size = pagination_params(
                        query,
                        default_page_size=50,
                    )
                    frame_step = query_int(
                        query,
                        "frame_step",
                        default=1,
                        minimum=1,
                        clamp=True,
                    )
                    max_frames = _optional_int(
                        query_value(query, "max_frames", "")
                    )
                    payload = list_folder_group_videos(
                        db,
                        group_name,
                        frame_step=frame_step,
                        max_frames=max_frames,
                        page=page,
                        page_size=page_size,
                        query=query_value(query, "q", ""),
                        sort=query_value(query, "sort", "name"),
                    )
                    selection_token = str(
                        query.get("video_selection_token", [""])[0] or ""
                    ).strip()
                    if selection_token:
                        try:
                            membership = video_selection_membership(
                                db,
                                selection_token,
                                video_group=group_name,
                                video_names=[
                                    str(row.get("name") or "")
                                    for row in payload.get("videos") or []
                                ],
                            )
                        except VideoSelectionTokenExpired as exc:
                            return self._error(
                                HTTPStatus.GONE,
                                str(exc),
                                type(exc).__name__,
                            )
                        selected_names = membership.pop("selected_names")
                        for row in payload.get("videos") or []:
                            row["selected"] = (
                                str(row.get("name") or "") in selected_names
                            )
                        payload["selection"] = membership
                    return self._json(payload)
                match = re.fullmatch(r"/api/media/assets/(\d+)/thumbnail", path)
                if match:
                    asset, source_path = resolve_asset_path(db, workspace, int(match.group(1)))
                    if asset.get("source_kind") != "folder" or source_path.suffix.lower() not in VIDEO_SUFFIXES:
                        return self._error(HTTPStatus.NOT_FOUND, "folder video thumbnail not found")
                    key = video_thumbnail_key(source_path)
                    generated = ensure_video_thumbnail(workspace, source_path, key)
                    if generated is None:
                        return self._error(HTTPStatus.NOT_FOUND, "video thumbnail could not be generated")
                    return self._send_file(generated, cache_control="public, max-age=31536000, immutable")
                match = re.fullmatch(r"/api/video-thumbnails/([a-f0-9]{64})", path)
                if match:
                    return self._send_file(thumbnail_path(workspace, match.group(1)))
                if path == "/api/metrics/health":
                    refresh = query.get("refresh", ["0"])[0] in {"1", "true", "yes"}
                    return self._json(metrics_health(workspace, refresh=refresh))
                if path == "/api/models":
                    return self._json(db.list_models())
                if path == "/api/datasets":
                    return self._json(db.list_datasets())
                if path.startswith("/api/datasets/") and path.endswith("/samples"):
                    dataset_id = int(path.split("/")[3])
                    return self._json(db.list_samples(dataset_id))
                if path == "/api/experiments":
                    return self._json(db.list_experiments())
                if path == "/api/runs":
                    if any(
                        key in query
                        for key in ("page", "page_size", "q", "status", "run_type", "model")
                    ):
                        return self._json(
                            db.list_runs_page(
                                page=int(query.get("page", ["1"])[0] or 1),
                                page_size=int(query.get("page_size", ["50"])[0] or 50),
                                query=query.get("q", [""])[0],
                                status=query.get("status", [""])[0],
                                run_type=query.get("run_type", [""])[0],
                                model=query.get("model", [""])[0],
                                include_deleted=query.get("include_deleted", ["0"])[0]
                                in {"1", "true", "yes"},
                            )
                        )
                    return self._json(
                        db.list_runs(
                            limit=int(query.get("limit", ["100"])[0]),
                            include_deleted=query.get("include_deleted", ["0"])[0] in {"1", "true", "yes"},
                        )
                    )
                if path == "/api/feedback":
                    return self._json(
                        _feedback_overview(
                            db,
                            dataset=(query.get("dataset", [None])[0] or None),
                            model_name=(query.get("model", [None])[0] or None),
                            checkpoint=(query.get("checkpoint", [None])[0] or None),
                            video=(query.get("video", [None])[0] or None),
                        )
                    )
                match = re.fullmatch(r"/api/runs/(\d+)/feedback", path)
                if match:
                    run_id = int(match.group(1))
                    return self._json({"run_id": run_id, "feedback": db.list_run_feedback(run_id)})
                match = re.fullmatch(r"/api/runs/(\d+)/compare-inputs", path)
                if match:
                    return self._json(_compare_inputs_payload(db, int(match.group(1))))
                match = re.fullmatch(r"/api/runs/(\d+)/compare-inputs/([^/]+)/media", path)
                if match:
                    run_id = int(match.group(1))
                    slot = unquote(match.group(2))
                    media_path = _compare_input_media(
                        db,
                        workspace,
                        run_id,
                        slot,
                        variant=str(query.get("variant", ["original"])[0] or "original"),
                    )
                    if media_path.is_dir():
                        files = sorted(child for child in media_path.iterdir() if child.is_file())
                        if not files:
                            return self._error(HTTPStatus.NOT_FOUND, "frame sequence is empty")
                        frame_index = int(query.get("frame", ["0"])[0] or 0)
                        if frame_index < 0 or frame_index >= len(files):
                            return self._error(
                                HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                                "frame index is out of range",
                            )
                        media_path = files[frame_index]
                    return self._send_file(media_path, cache_control="no-store")
                match = re.fullmatch(r"/api/runs/(\d+)/samples/(\d+)", path)
                if match:
                    return self._json(_run_sample_payload(db, int(match.group(1)), int(match.group(2))))
                match = re.fullmatch(r"/api/runs/(\d+)/videos", path)
                if match:
                    return self._json(
                        _run_videos(
                            db,
                            int(match.group(1)),
                            page=int(query.get("page", ["1"])[0] or 1),
                            page_size=int(query.get("page_size", ["50"])[0] or 50),
                            q=query.get("q", [""])[0],
                        )
                    )
                match = re.fullmatch(r"/api/runs/(\d+)/videos/(.+)/timeline", path)
                if match:
                    return self._json(
                        _run_video_timeline(
                            db,
                            int(match.group(1)),
                            unquote(match.group(2)),
                            metric=query.get("metric", [None])[0],
                            bucket_count=int(query.get("bucket_count", ["120"])[0] or 120),
                            window_start=int(query.get("window_start", ["0"])[0] or 0),
                            window_size=int(query.get("window_size", ["300"])[0] or 300),
                        )
                    )
                match = re.fullmatch(r"/api/runs/(\d+)(?:/(samples|artifacts|metrics|timeline|metric-summary))?", path)
                if match:
                    run_id = int(match.group(1))
                    section = match.group(2)
                    if section == "samples":
                        return self._json(db.list_run_samples(run_id))
                    if section == "artifacts":
                        kind = query.get("kind", [None])[0]
                        return self._json(db.list_run_artifacts(run_id, kind=kind))
                    if section == "metrics":
                        return self._json(db.list_run_metrics(run_id))
                    if section == "timeline":
                        return self._json(_run_timeline(db, run_id), headers={"X-Deprecated": "use /api/runs/{id}/videos"})
                    if section == "metric-summary":
                        return self._json(_run_metric_summary(db, run_id))
                    return self._json(_run_detail(db, run_id))
                if path == "/api/jobs":
                    return self._json(db.list_jobs(limit=int(query.get("limit", ["100"])[0])))
                if path == "/api/workers":
                    return self._json(db.list_workers())
                if path == "/api/artifacts":
                    job_id = _optional_int(query.get("job_id", [None])[0])
                    kind = query.get("kind", [None])[0]
                    return self._json(db.list_artifacts(job_id=job_id, kind=kind))
                if path == "/api/metrics":
                    inference_job_id = _optional_int(query.get("inference_job_id", [None])[0])
                    return self._json(db.list_metric_results(inference_job_id=inference_job_id))
                match = re.fullmatch(r"/api/compare-sources/(gt|pred|flow|mask)", path)
                if match:
                    if "path" in query:
                        return self._error(HTTPStatus.BAD_REQUEST, "compare source APIs do not accept client-supplied paths")
                    return self._json(_compare_sources(db, workspace, match.group(1), query))
                if path == "/api/compare":
                    return self._json(_compare(db, query))
                if path == "/api/compare/samples":
                    return self._json(_compare_samples(db, query))
                if path.startswith("/api/files/"):
                    artifact_id = int(path.rsplit("/", 1)[-1])
                    return self._send_artifact(artifact_id, query.get("variant", ["original"])[0])
                match = re.fullmatch(r"/api/sample-files/(\d+)/(img0|img1|gt)", path)
                if match:
                    return self._send_sample_file(int(match.group(1)), match.group(2))
                self._error(HTTPStatus.NOT_FOUND, "not found")
            except CacheCoordinationUnavailable as exc:
                self._json(
                    {
                        "error": {
                            "type": type(exc).__name__,
                            "code": exc.code,
                            "message": str(exc),
                        }
                    },
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
            except KeyError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc), type(exc).__name__)
            except FileNotFoundError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc), type(exc).__name__)
            except EvaluationConflict as exc:
                self._error(HTTPStatus.CONFLICT, str(exc), type(exc).__name__)
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
            except Exception as exc:
                self._error_internal(exc)

        def do_POST(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                try:
                    body = self._read_json()
                except _RequestBodyTooLarge as exc:
                    return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(exc), "RequestBodyTooLarge")
                if path == "/api/run-purge/preview":
                    raw_ids = body.get("run_ids") or []
                    if not isinstance(raw_ids, list):
                        return self._error(HTTPStatus.BAD_REQUEST, "run_ids must be a list")
                    preview = cleanup_service.preview_run_purge(
                        str(body.get("request_type") or ""),
                        raw_ids,
                    )
                    return self._json(preview)
                if path == "/api/storage/gc":
                    try:
                        result = cleanup_service.garbage_collect(
                            confirmed=body.get("confirm") is True,
                            entry_ids=body.get("entry_ids") if isinstance(body.get("entry_ids"), list) else None,
                            run_ids=body.get("run_ids") if isinstance(body.get("run_ids"), list) else None,
                            preview_token=str(body.get("preview_token") or ""),
                            require_preview_token=True,
                        )
                    except CacheCoordinationUnavailable as exc:
                        return self._json(
                            {
                                "error": {
                                    "type": type(exc).__name__,
                                    "code": exc.code,
                                    "message": str(exc),
                                }
                            },
                            status=HTTPStatus.SERVICE_UNAVAILABLE,
                        )
                    except ValueError as exc:
                        return self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
                    return self._json(result)
                if path == "/api/media/sync":
                    requested = catalog_sync.request_sync(
                        include_runs=body.get("include_runs", True) is not False
                    )
                    return self._json(requested, status=HTTPStatus.ACCEPTED)
                if path == "/api/video-selections":
                    try:
                        selection = create_video_selection_snapshot(
                            db,
                            workspace,
                            video_groups=body.get("video_groups"),
                            query=body.get("q") or "",
                            base_selection_token=str(
                                body.get("base_selection_token") or ""
                            ),
                            operation=str(body.get("operation") or ""),
                            video_group=str(body.get("video_group") or ""),
                            video_names=body.get("video_names"),
                        )
                    except VideoSelectionTokenExpired as exc:
                        return self._error(
                            HTTPStatus.GONE,
                            str(exc),
                            type(exc).__name__,
                        )
                    return self._json(selection, status=HTTPStatus.CREATED)
                if path == "/api/media/item-selections":
                    selection = create_selection_snapshot(
                        db,
                        group_id=body.get("group_id"),
                        query=str(body.get("q") or ""),
                    )
                    return self._json(selection, status=HTTPStatus.CREATED)
                if path == "/api/media/collections":
                    try:
                        collection = create_collection(
                            db,
                            str(body.get("name") or ""),
                            str(body.get("slug") or "") or None,
                            body.get("metadata") or {},
                        )
                    except ValueError as exc:
                        return self._error(HTTPStatus.CONFLICT, str(exc), type(exc).__name__)
                    return self._json({"collection": collection}, status=HTTPStatus.CREATED)
                match = re.fullmatch(r"/api/media/items/(\d+)/external-predictions", path)
                if match:
                    try:
                        member = register_external_prediction(
                            db,
                            int(match.group(1)),
                            int(body.get("asset_id")),
                            method_key=str(body.get("method_key") or ""),
                            temporal_mapping=(
                                body.get("temporal_mapping")
                                if isinstance(body.get("temporal_mapping"), dict)
                                else None
                            ),
                            spatial_origin=(
                                body.get("spatial_origin")
                                if isinstance(body.get("spatial_origin"), dict)
                                else None
                            ),
                            aspect_stretch_confirmed=body.get("aspect_stretch_confirmed") is True,
                            metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
                        )
                    except (TypeError, ValueError) as exc:
                        return self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
                    return self._json({"member": member}, status=HTTPStatus.CREATED)
                if path == "/api/uploads":
                    try:
                        upload = create_upload_session(db, workspace, body)
                    except FileExistsError as exc:
                        return self._error(HTTPStatus.CONFLICT, str(exc), type(exc).__name__)
                    except StorageCapacityError as exc:
                        return self._json(exc.public_payload(), status=HTTPStatus.INSUFFICIENT_STORAGE)
                    except ValueError as exc:
                        return self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
                    return self._json(
                        {"upload": upload, "chunk_size": UPLOAD_CHUNK_SIZE},
                        status=HTTPStatus.CREATED,
                    )
                match = re.fullmatch(r"/api/uploads/([a-f0-9]{32})/complete", path)
                if match:
                    try:
                        completed = complete_upload_session(db, workspace, match.group(1))
                    except ValueError as exc:
                        return self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
                    return self._json(completed)
                if path == "/api/evaluators/session":
                    return self._error(
                        HTTPStatus.GONE,
                        "legacy Campaign V1 is read-only; create evaluator sessions through an opaque Campaign V2 URL",
                    )
                if path == "/api/evaluation-campaigns/v2/preview":
                    try:
                        return self._json(preview_campaign_v2(db, workspace, body))
                    except ValueError as exc:
                        return self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
                if path == "/api/evaluation-campaigns/v2":
                    try:
                        campaign = create_campaign_v2(db, workspace, body)
                    except SubmissionConflict as exc:
                        return self._json(
                            {
                                "error": {
                                    "type": type(exc).__name__,
                                    "code": exc.code,
                                    "message": str(exc),
                                }
                            },
                            status=HTTPStatus.CONFLICT,
                        )
                    except ValueError as exc:
                        return self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
                    return self._json({"campaign": campaign}, status=HTTPStatus.CREATED)
                match = re.fullmatch(r"/api/evaluation-campaigns/v2/(\d+)/(publish|close|archive)", path)
                if match:
                    campaign_id = int(match.group(1))
                    action = match.group(2)
                    try:
                        if action == "publish":
                            result = request_publish_campaign_v2(db, campaign_id, workspace)
                            if preparation_wake_event is not None:
                                preparation_wake_event.set()
                            else:
                                # Embedded/test servers do not own the resident
                                # preparation loop, so retain an asynchronous pump.
                                threading.Thread(
                                    target=run_pending_preparations,
                                    args=(db, workspace),
                                    kwargs={"limit": 1},
                                    name=f"vfieval-evaluation-{campaign_id}",
                                    daemon=True,
                                ).start()
                            return self._json(result, status=HTTPStatus.ACCEPTED)
                        if action == "close":
                            campaign = close_campaign_v2(db, campaign_id)
                        else:
                            campaign = archive_campaign_v2(db, campaign_id)
                        return self._json({"campaign": campaign})
                    except EvaluationConflict as exc:
                        return self._error(HTTPStatus.CONFLICT, str(exc), type(exc).__name__)
                    except StorageCapacityError as exc:
                        return self._json(exc.public_payload(), status=HTTPStatus.INSUFFICIENT_STORAGE)
                    except ValueError as exc:
                        return self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
                match = re.fullmatch(r"/api/evaluation-cleanup-requests/(\d+)/retry", path)
                if match:
                    request_id = int(match.group(1))
                    existing = db.get(
                        "SELECT id FROM evaluation_purge_requests_v2 WHERE id = ?",
                        (request_id,),
                    )
                    if existing is None:
                        return self._error(HTTPStatus.NOT_FOUND, "cleanup request not found")
                    processed = process_campaign_purge_requests_v2(db, workspace, limit=20)
                    request = next(
                        (row for row in processed if int(row.get("id") or 0) == request_id),
                        next(
                            (
                                row
                                for row in list_campaign_purge_requests_v2(
                                    db,
                                    include_completed=True,
                                )
                                if int(row.get("id") or 0) == request_id
                            ),
                            None,
                        ),
                    )
                    return self._json({"request": request})
                match = re.fullmatch(r"/api/evaluation-campaigns/(\d+)/(archive|discard)", path)
                if match:
                    campaign_id = int(match.group(1))
                    try:
                        if match.group(2) == "archive":
                            return self._json({"campaign": archive_legacy_campaign(db, campaign_id)})
                        discard_empty_legacy_draft(db, campaign_id)
                        return self._json({"campaign_id": campaign_id, "discarded": True})
                    except ValueError as exc:
                        return self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
                match = re.fullmatch(r"/api/blind/([A-Za-z0-9_-]+)/session", path)
                if match:
                    try:
                        return self._json(blind_session(db, match.group(1), body), status=HTTPStatus.CREATED)
                    except ValueError as exc:
                        return self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
                match = re.fullmatch(r"/api/blind/([A-Za-z0-9_-]+)/tasks/([A-Za-z0-9_-]+)/(vote|heartbeat)", path)
                if match:
                    token, task_token, action = match.groups()
                    evaluator_id = str(body.get("evaluator_id") or "").strip()
                    if not evaluator_id:
                        return self._error(HTTPStatus.BAD_REQUEST, "evaluator_id is required")
                    try:
                        if action == "heartbeat":
                            return self._json(blind_heartbeat(db, token, task_token, evaluator_id))
                        return self._json(
                            blind_submit_vote(db, token, task_token, evaluator_id, body)
                        )
                    except EvaluationConflict as exc:
                        return self._error(HTTPStatus.CONFLICT, str(exc), type(exc).__name__)
                    except ValueError as exc:
                        return self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
                if path == "/api/evaluation-campaigns":
                    return self._error(
                        HTTPStatus.GONE,
                        "legacy Campaign V1 is read-only; create Campaigns with /api/evaluation-campaigns/v2",
                    )
                match = re.fullmatch(r"/api/evaluation-campaigns/(\d+)/(candidates|publish|close)", path)
                if match:
                    return self._error(
                        HTTPStatus.GONE,
                        "legacy Campaign V1 is read-only; use Campaign V2 for candidate, publish, and close operations",
                    )
                if path == "/api/evaluation-tasks/adhoc":
                    return self._error(
                        HTTPStatus.GONE,
                        "legacy Campaign V1 is read-only; ad-hoc tasks are no longer available",
                    )
                match = re.fullmatch(r"/api/evaluation-tasks/(\d+)/votes", path)
                if match:
                    return self._error(
                        HTTPStatus.GONE,
                        "legacy Campaign V1 is read-only; votes are accepted only through opaque Campaign V2 task URLs",
                    )
                if path == "/api/models":
                    adapter = body["adapter"]
                    if not adapter.startswith("file:") and adapter != "dummy":
                        return self._error(HTTPStatus.BAD_REQUEST, "adapter must start with 'file:' or be 'dummy'")
                    if adapter.startswith("file:"):
                        adapter_path = Path(adapter.removeprefix("file:")).resolve()
                        allowed_dir = models_dir(workspace)
                        if not _is_relative_to(adapter_path, allowed_dir):
                            return self._error(HTTPStatus.BAD_REQUEST, "adapter file must be inside models/ directory")
                    model_id = db.register_model(
                        name=body["name"],
                        adapter=adapter,
                        checkpoint_path=body.get("checkpoint_path"),
                        input_height=int(body["input_height"]),
                        input_width=int(body["input_width"]),
                        metadata=body.get("metadata") or {},
                    )
                    return self._json({"model_id": model_id}, status=HTTPStatus.CREATED)
                if path == "/api/datasets":
                    source_type = body.get("source_type", "frames")
                    decode_mode = body.get("decode_mode")
                    metadata = body.get("metadata") or {}
                    for key in ("frame_step", "max_frames", "video_glob"):
                        if key in body and body[key] not in {None, ""}:
                            metadata[key] = body[key]
                    dataset_id = db.create_dataset(
                        name=body["name"],
                        root_path=body["root_path"],
                        has_gt=bool(body.get("has_gt", True)),
                        source_type=source_type,
                        decode_mode=decode_mode,
                        metadata=metadata,
                    )
                    return self._json({"dataset_id": dataset_id}, status=HTTPStatus.CREATED)
                if path == "/api/experiments":
                    experiment_id = db.create_experiment(
                        name=body["name"],
                        description=body.get("description", ""),
                        metadata=body.get("metadata") or {},
                    )
                    return self._json({"experiment_id": experiment_id}, status=HTTPStatus.CREATED)
                if path == "/api/workers/register":
                    return self._json(register_worker_request(db, body))
                if path in {"/api/preflight", "/api/preflight/quick"}:
                    if path == "/api/preflight/quick":
                        body = dict(body)
                        body["preflight_level"] = "quick"
                    body = expand_video_selection_payload(db, workspace, body)
                    prepared = source_assets_to_video_payload(db, workspace, body)
                    prepared = _prepare_media_item_compare_payload(db, workspace, prepared)
                    preflight_level = str(prepared.get("preflight_level") or "deep").strip().lower()
                    if preflight_level not in {"quick", "deep"}:
                        return self._error(
                            HTTPStatus.BAD_REQUEST,
                            "preflight_level must be quick or deep",
                        )
                    if (
                        preflight_level == "quick"
                        and str(prepared.get("run_type") or "model_inference") == "video_compare"
                    ):
                        return self._error(
                            HTTPStatus.BAD_REQUEST,
                            "quick preflight is available only for model-inference Runs",
                        )
                    prepared["preflight_level"] = preflight_level
                    result = preflight_run(db, workspace, prepared)
                    result["preflight_cache_hit"] = False
                    _attach_workload_estimate(db, workspace, result, prepared)
                    profile_request = dict(prepared)
                    profile_request.update(
                        {
                            "height": int((result.get("resolution") or {}).get("height") or prepared.get("height") or 0),
                            "width": int((result.get("resolution") or {}).get("width") or prepared.get("width") or 0),
                            "precision": str((result.get("device") or {}).get("effective_precision") or prepared.get("precision") or "fp32"),
                        }
                    )
                    result["execution_profile"] = recommend_execution_profile(db, workspace, profile_request)
                    if preflight_level == "deep" and bool(result.get("ok")):
                        result["preflight_token"] = store_preflight(prepared, result)
                    return self._json(result)
                if path == "/api/runs":
                    # Even Runs without a detailed workload estimate must retain
                    # the configured emergency free-space margin.
                    ensure_storage_capacity(db, workspace, requested_bytes=0)

                    def prepare_submitted_run(request: dict) -> dict:
                        prepared = expand_video_selection_payload(
                            db,
                            workspace,
                            request,
                        )
                        prepared = source_assets_to_video_payload(
                            db,
                            workspace,
                            prepared,
                        )
                        prepared = _prepare_media_item_compare_payload(
                            db,
                            workspace,
                            prepared,
                        )
                        # These fields are server-owned trust material. A
                        # client may present only the opaque token; validated
                        # signatures are injected after the submission claim.
                        prepared.pop("_preflight_result", None)
                        prepared.pop("_preflight_physical_inputs", None)
                        stored_preflight = cached_preflight(prepared)
                        if stored_preflight is not None:
                            prepared["_preflight_result"] = stored_preflight
                        return prepared

                    run_type = str(body.get("run_type") or "")
                    if run_type == "video_compare":
                        created = _create_video_compare_run(
                            db,
                            workspace,
                            body,
                            prepare=prepare_submitted_run,
                        )
                        return self._json(created, status=HTTPStatus.CREATED)
                    if body.get("model_file") or body.get("video_group") or body.get("source_assets"):
                        created = _create_run_from_files(
                            db,
                            workspace,
                            body,
                            prepare=prepare_submitted_run,
                        )
                        return self._json(created, status=HTTPStatus.CREATED)
                    metrics = list(body.get("metrics") or [])
                    unsupported = [name for name in metrics if name not in METRIC_NAMES]
                    if unsupported:
                        return self._error(HTTPStatus.BAD_REQUEST, f"unsupported metrics: {', '.join(unsupported)}")
                    model_id = int(body["model_id"])
                    dataset_id = int(body["dataset_id"])
                    created = _create_registered_run(
                        db,
                        body,
                        model_id=model_id,
                        dataset_id=dataset_id,
                        metrics=metrics,
                    )
                    wake_job_supervisor(db)
                    return self._json(created, status=HTTPStatus.CREATED)
                match = re.fullmatch(r"/api/datasets/(\d+)/scan", path)
                if match:
                    dataset_id = int(match.group(1))
                    samples = scan_dataset(db, workspace, dataset_id)
                    return self._json({"dataset_id": dataset_id, "samples": samples})
                match = re.fullmatch(r"/api/runs/(\d+)/(cancel|retry|clone)", path)
                if match:
                    run_id = int(match.group(1))
                    action = match.group(2)
                    if action == "cancel":
                        db.request_run_cancel(run_id)
                        return self._json({"run_id": run_id, "run": db.get_run(run_id)})
                    created = (
                        _retry_run(db, workspace, run_id)
                        if action == "retry"
                        else _clone_run(
                            db,
                            workspace,
                            run_id,
                            risk_ack_fingerprint=body.get("risk_ack_fingerprint"),
                        )
                    )
                    return self._json(created, status=HTTPStatus.CREATED)
                match = re.fullmatch(r"/api/runs/(\d+)/(hide|cleanup-artifacts)", path)
                if match:
                    run_id = int(match.group(1))
                    action = match.group(2)
                    if action == "hide":
                        cleanup_service.consume_run_purge_preview(
                            body.get("preview_token"),
                            request_type="delete_run",
                            run_ids=[run_id],
                        )
                        request = cleanup_service.request_delete(run_id)
                        return self._json(_purge_response(request), status=HTTPStatus.ACCEPTED)
                    try:
                        cleanup_service.assert_artifact_cleanup_eligible(run_id)
                        cleanup_service.consume_run_purge_preview(
                            body.get("preview_token"),
                            request_type="cleanup_artifacts",
                            run_ids=[run_id],
                        )
                        request = cleanup_service.request_artifact_cleanup(run_id)
                    except RunPurgePreviewError:
                        raise
                    except ValueError as exc:
                        return self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
                    return self._json(_purge_response(request), status=HTTPStatus.ACCEPTED)
                match = re.fullmatch(r"/api/runs/(\d+)/metrics/retry", path)
                if match:
                    run_id = int(match.group(1))
                    retry = _retry_run_metrics(db, run_id)
                    return self._json(retry, status=HTTPStatus.CREATED)
                match = re.fullmatch(r"/api/runs/(\d+)/rename", path)
                if match:
                    run_id = int(match.group(1))
                    name = str(body.get("name") or "").strip()
                    if not name:
                        return self._error(HTTPStatus.BAD_REQUEST, "name must not be empty")
                    db.rename_run(run_id, name)
                    return self._json({"run_id": run_id, "run": db.get_run(run_id)})
                match = re.fullmatch(r"/api/runs/(\d+)/feedback", path)
                if match:
                    run_id = int(match.group(1))
                    try:
                        created = _create_run_feedback(db, run_id, body)
                    except ValueError as exc:
                        return self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
                    return self._json(created, status=HTTPStatus.CREATED)
                match = re.fullmatch(r"/api/runs/(\d+)/feedback/(\d+)", path)
                if match:
                    run_id = int(match.group(1))
                    feedback_id = int(match.group(2))
                    try:
                        updated = _update_run_feedback(db, run_id, feedback_id, body)
                    except KeyError:
                        return self._error(HTTPStatus.NOT_FOUND, "feedback not found")
                    except ValueError as exc:
                        return self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
                    return self._json(updated)
                if path == "/api/runs/batch-delete":
                    raw_ids = body.get("run_ids") or []
                    if not isinstance(raw_ids, list) or not raw_ids:
                        return self._error(HTTPStatus.BAD_REQUEST, "run_ids must be a non-empty list")
                    cleanup_service.consume_run_purge_preview(
                        body.get("preview_token"),
                        request_type="delete_run",
                        run_ids=raw_ids,
                    )
                    accepted = []
                    failures = []
                    for raw in raw_ids:
                        try:
                            run_id = int(raw)
                            accepted.append(_purge_response(cleanup_service.request_delete(run_id)))
                        except Exception as exc:
                            failures.append(
                                {
                                    "run_id": raw,
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                }
                            )
                    return self._json(
                        {
                            "requests": accepted,
                            "accepted": [int(row["run_id"]) for row in accepted],
                            "deleted": [int(row["run_id"]) for row in accepted if row["deleted"]],
                            "failures": failures,
                            "count": len(accepted),
                        },
                        status=HTTPStatus.ACCEPTED,
                    )
                if path == "/api/jobs":
                    payload, status = create_job_request(db, body)
                    return self._json(payload, status=status)
                if path == "/api/jobs/claim":
                    return self._json(claim_job_request(db, body))
                match = re.fullmatch(r"/api/jobs/(\d+)/(complete|fail|progress)", path)
                if match:
                    return self._json(
                        job_callback_request(
                            db,
                            int(match.group(1)),
                            match.group(2),
                            body,
                        )
                    )
                match = re.fullmatch(r"/api/jobs/(\d+)/heartbeat", path)
                if match:
                    return self._json(
                        heartbeat_job_request(
                            db,
                            int(match.group(1)),
                            body,
                        )
                    )
                self._error(HTTPStatus.NOT_FOUND, "not found")
            except JobApiError as exc:
                self._error(exc.status, str(exc), exc.code)
            except SubmissionConflict as exc:
                self._json(
                    {
                        "error": {
                            "type": type(exc).__name__,
                            "code": exc.code,
                            "message": str(exc),
                        }
                    },
                    status=HTTPStatus.CONFLICT,
                )
            except CacheCoordinationUnavailable as exc:
                self._json(
                    {
                        "error": {
                            "type": type(exc).__name__,
                            "code": exc.code,
                            "message": str(exc),
                        }
                    },
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
            except EvaluationConflict as exc:
                self._error(HTTPStatus.CONFLICT, str(exc), type(exc).__name__)
            except RunPurgePreviewError as exc:
                self._json(
                    {
                        "error": {
                            "type": type(exc).__name__,
                            "code": exc.code,
                            "message": str(exc),
                        }
                    },
                    status=HTTPStatus.CONFLICT,
                )
            except InputIdentityChanged as exc:
                self._json(exc.public_payload(), status=HTTPStatus.CONFLICT)
            except _WorkloadRiskConfirmationRequired as exc:
                self._json(exc.public_payload(), status=HTTPStatus.CONFLICT)
            except StorageCapacityError as exc:
                self._json(exc.public_payload(), status=HTTPStatus.INSUFFICIENT_STORAGE)
            except VideoSelectionTokenExpired as exc:
                self._error(HTTPStatus.GONE, str(exc), type(exc).__name__)
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
            except KeyError as exc:
                self._error(HTTPStatus.BAD_REQUEST, f"missing field {exc}")
            except Exception as exc:
                self._error_internal(exc)

        def do_PUT(self) -> None:
            try:
                parsed = urlparse(self.path)
                match = re.fullmatch(r"/api/uploads/([a-f0-9]{32})/parts/(\d+)", parsed.path)
                if not match:
                    return self._error(HTTPStatus.NOT_FOUND, "not found")
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length <= 0 or length > UPLOAD_CHUNK_SIZE:
                    return self._error(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        f"upload part must contain between 1 and {UPLOAD_CHUNK_SIZE} bytes",
                    )
                content_range = str(self.headers.get("Content-Range") or "")
                range_match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if not range_match:
                    return self._error(HTTPStatus.BAD_REQUEST, "valid Content-Range header is required")
                start, end, total = (int(value) for value in range_match.groups())
                if end - start + 1 != length:
                    return self._error(HTTPStatus.BAD_REQUEST, "Content-Range does not match Content-Length")
                session = get_upload_session(db, match.group(1))
                if total != int(session["expected_size"]):
                    return self._error(HTTPStatus.BAD_REQUEST, "Content-Range total does not match upload size")
                digest = str(self.headers.get("X-Chunk-SHA256") or "").strip()
                data = self.rfile.read(length)
                uploaded = receive_upload_part(
                    db,
                    workspace,
                    match.group(1),
                    int(match.group(2)),
                    data,
                    offset_bytes=start,
                    sha256=digest,
                )
                return self._json({"upload": uploaded})
            except KeyError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc), type(exc).__name__)
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
            except Exception as exc:
                self._error_internal(exc)

        def do_DELETE(self) -> None:
            try:
                parsed = urlparse(self.path)
                campaign_match = re.fullmatch(r"/api/evaluation-campaigns/v2/(\d+)", parsed.path)
                if campaign_match:
                    try:
                        body = self._read_json()
                    except _RequestBodyTooLarge as exc:
                        return self._error(
                            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                            str(exc),
                            "RequestBodyTooLarge",
                        )
                    if not isinstance(body, dict):
                        return self._error(
                            HTTPStatus.BAD_REQUEST,
                            "Campaign V2 delete body must be a JSON object",
                        )
                    result = delete_campaign_v2(
                        db,
                        workspace,
                        int(campaign_match.group(1)),
                        confirmed=body.get("confirm") is True,
                        destructive_confirmed=body.get("confirm_destructive") is True,
                    )
                    return self._json(result)
                asset_match = re.fullmatch(r"/api/media/assets/(\d+)", parsed.path)
                if asset_match:
                    asset = soft_delete_asset(db, workspace, int(asset_match.group(1)))
                    return self._json({"asset": asset, "deleted": True})
                upload_match = re.fullmatch(r"/api/uploads/([a-f0-9]{32})", parsed.path)
                if upload_match:
                    delete_upload_session(db, workspace, upload_match.group(1))
                    return self._json({"upload_id": upload_match.group(1), "deleted": True})
                match = re.fullmatch(r"/api/runs/(\d+)/feedback/(\d+)", parsed.path)
                if match:
                    run_id = int(match.group(1))
                    feedback_id = int(match.group(2))
                    removed = db.delete_run_feedback(run_id, feedback_id)
                    if not removed:
                        return self._error(HTTPStatus.NOT_FOUND, "feedback not found")
                    return self._json({"run_id": run_id, "feedback_id": feedback_id, "deleted": True})
                match = re.fullmatch(r"/api/runs/(\d+)", parsed.path)
                if not match:
                    return self._error(HTTPStatus.NOT_FOUND, "not found")
                run_id = int(match.group(1))
                query = parse_qs(parsed.query)
                cleanup_service.consume_run_purge_preview(
                    query.get("preview_token", [""])[0],
                    request_type="delete_run",
                    run_ids=[run_id],
                )
                request = cleanup_service.request_delete(run_id)
                return self._json(_purge_response(request), status=HTTPStatus.ACCEPTED)
            except CacheCoordinationUnavailable as exc:
                self._json(
                    {
                        "error": {
                            "type": type(exc).__name__,
                            "code": exc.code,
                            "message": str(exc),
                        }
                    },
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
            except KeyError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc), type(exc).__name__)
            except EvaluationConflict as exc:
                self._error(HTTPStatus.CONFLICT, str(exc), type(exc).__name__)
            except RunPurgePreviewError as exc:
                self._json(
                    {
                        "error": {
                            "type": type(exc).__name__,
                            "code": exc.code,
                            "message": str(exc),
                        }
                    },
                    status=HTTPStatus.CONFLICT,
                )
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
            except Exception as exc:
                self._error_internal(exc)

        def log_message(self, fmt: str, *args) -> None:
            log_event(
                logging.INFO,
                "http.access",
                fmt % args,
                request_id=getattr(self, "request_id", ""),
                method=getattr(self, "command", ""),
                path=urlparse(getattr(self, "path", "")).path,
                client=str(self.client_address[0]) if getattr(self, "client_address", None) else "",
            )

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            if length > MAX_REQUEST_BODY_BYTES:
                raise _RequestBodyTooLarge(
                    f"request body of {length} bytes exceeds the {MAX_REQUEST_BODY_BYTES}-byte limit"
                )
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _json(self, data, status: HTTPStatus = HTTPStatus.OK, headers: dict[str, str] | None = None) -> None:
            payload = json.dumps(data, indent=2, default=str).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Request-ID", getattr(self, "request_id", ""))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            if not self._client_io(self.end_headers):
                return
            self._client_io(self.wfile.write, payload)

        def _send_bytes(self, payload: bytes, content_type: str, filename: str | None = None) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Request-ID", getattr(self, "request_id", ""))
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            if not self._client_io(self.end_headers):
                return
            self._client_io(self.wfile.write, payload)

        def _error(self, status: HTTPStatus, message: str, error_type: str = "Error") -> None:
            self._json({"error": {"type": error_type, "message": message}}, status=status)

        def _error_internal(self, exc: Exception) -> None:
            # Log full detail server-side; never echo raw exception text (which can
            # contain internal paths/stack info) back to the client.
            support_id = uuid.uuid4().hex[:12]
            runtime_logger().error(
                "internal request error",
                exc_info=(type(exc), exc, exc.__traceback__),
                extra={
                    "event": "http.internal_error",
                    "details": {
                        "support_id": support_id,
                        "request_id": getattr(self, "request_id", ""),
                        "method": getattr(self, "command", ""),
                        "path": urlparse(getattr(self, "path", "")).path,
                        "error_type": type(exc).__name__,
                    },
                },
            )
            self._json(
                {
                    "error": {
                        "type": "InternalServerError",
                        "message": "internal server error",
                        "support_id": support_id,
                        "request_id": getattr(self, "request_id", ""),
                    }
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        def _send_static(self, name: str) -> None:
            path = Path(__file__).parent / "web" / name
            if not path.exists():
                return self._error(HTTPStatus.NOT_FOUND, f"static file {name} not found")
            self._send_file(path, cache_control="no-store")

        def _send_artifact(self, artifact_id: int, variant: str = "original") -> None:
            artifact = db.get("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
            if artifact is None:
                return self._error(HTTPStatus.NOT_FOUND, "artifact not found")
            workspace_root = workspace.root.resolve()
            canonical_path = Path(artifact["path"]).resolve()
            preview_path = _materialized_preview_path(artifact)
            if (
                variant == "preview"
                and preview_path is not None
                and _is_relative_to(preview_path, workspace_root)
            ):
                path = preview_path
            else:
                path = canonical_path
            if not _is_relative_to(path, workspace_root):
                return self._error(HTTPStatus.FORBIDDEN, "artifact path is outside workspace")
            self._send_file(path)

        def _send_sample_file(self, sample_id: int, slot: str) -> None:
            sample = db.get_sample(sample_id)
            key = f"{slot}_path"
            path_text = sample.get(key)
            if not path_text:
                return self._error(HTTPStatus.NOT_FOUND, f"sample has no {slot} file")
            path = Path(path_text).resolve()
            workspace_root = workspace.root.resolve()
            project = Path(__file__).resolve().parents[2]
            if not _is_relative_to(path, workspace_root) and not _is_relative_to(path, project):
                return self._error(HTTPStatus.FORBIDDEN, "sample file path is outside allowed directories")
            self._send_file(path)

        def _send_file(self, path: Path, cache_control: str | None = None) -> None:
            if not path.exists() or not path.is_file():
                return self._error(HTTPStatus.NOT_FOUND, f"file not found: {path}")
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            file_size = path.stat().st_size
            byte_range = _parse_range_header(self.headers.get("Range"), file_size)
            if byte_range == "invalid":
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("X-Request-ID", getattr(self, "request_id", ""))
                self._client_io(self.end_headers)
                return
            if byte_range is None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("X-Request-ID", getattr(self, "request_id", ""))
                if cache_control:
                    self.send_header("Cache-Control", cache_control)
                if not self._client_io(self.end_headers):
                    return
                with path.open("rb") as handle:
                    while True:
                        chunk = handle.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        if not self._client_io(self.wfile.write, chunk):
                            return
            else:
                start, end = byte_range
                content_length = end - start + 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(content_length))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("X-Request-ID", getattr(self, "request_id", ""))
                if cache_control:
                    self.send_header("Cache-Control", cache_control)
                if not self._client_io(self.end_headers):
                    return
                with path.open("rb") as handle:
                    handle.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk = handle.read(min(remaining, 4 * 1024 * 1024))
                        if not chunk:
                            break
                        if not self._client_io(self.wfile.write, chunk):
                            return
                        remaining -= len(chunk)

    return VFIEvalHandler


def _parse_range_header(range_header: str | None, file_size: int) -> tuple[int, int] | str | None:
    if not range_header:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match or file_size < 0:
        return "invalid"
    start_text, end_text = match.groups()
    if not start_text and not end_text:
        return "invalid"
    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            return "invalid"
        start = max(0, file_size - suffix_length)
        end = file_size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    if start < 0 or end < start or start >= file_size:
        return "invalid"
    return start, min(end, file_size - 1)


def _compare(db: Database, query: dict[str, list[str]]) -> dict:
    run_ids = [
        int(part)
        for raw in query.get("run_id", [])
        for part in raw.split(",")
        if part.strip()
    ]
    if run_ids:
        runs = [_run_compare_payload(db, run_id) for run_id in run_ids]
        keys = [item["compare_key"] for item in runs]
        return {"compatible": len({json.dumps(key, sort_keys=True) for key in keys}) <= 1, "runs": runs}

    inference_ids = [
        int(part)
        for raw in query.get("inference_job_id", [])
        for part in raw.split(",")
        if part.strip()
    ]
    if not inference_ids:
        return {"runs": []}
    runs = []
    for inference_job_id in inference_ids:
        job = db.get_job(inference_job_id)
        metrics = db.list_metric_results(inference_job_id=inference_job_id)
        by_metric: dict[str, list[float]] = {}
        for row in metrics:
            if row["status"] == "completed" and row["value"] is not None:
                by_metric.setdefault(row["metric_name"], []).append(float(row["value"]))
        aggregate = {
            metric: sum(values) / len(values)
            for metric, values in by_metric.items()
            if values
        }
        runs.append({"job": job, "metrics": aggregate})
    return {"runs": runs}


def _compare_sources(db: Database, workspace: WorkspaceConfig, source_type: str, query: dict[str, list[str]]) -> dict:
    if source_type == "gt":
        return _compare_gt_sources(db, workspace, query)
    if source_type == "pred":
        run_id = _optional_int(query.get("run_id", [None])[0])
        return _compare_pred_sources(db, workspace, query, run_id)
    if source_type in {"flow", "mask"}:
        run_id = _optional_int(query.get("run_id", [None])[0])
        if run_id is None:
            raise ValueError(f"/api/compare-sources/{source_type} requires run_id")
        video = query.get("video", [None])[0]
        return {"sources": _compare_layer_sources(db, run_id, source_type, video)}
    raise ValueError(f"unsupported compare source type: {source_type}")


def _compare_gt_sources(db: Database, workspace: WorkspaceConfig, query: dict[str, list[str]]) -> dict:
    group_filter = str(query.get("group", [""])[0] or "").strip()
    text_filter = str(query.get("q", [""])[0] or "").strip().lower()
    page, page_size = _source_pagination(query)

    # GT is a property of the source clip, not of each run: a clip inferred by
    # two runs used to surface two byte-identical "Run GT" cards. Compare now
    # lists each ``videos/`` clip once and reconstructs the pred-aligned GT from
    # the source clip using the pred's recorded ``source_frame_indices`` (see
    # datasets._select_track_reference). So the picker enumerates source clips
    # only — one card per clip. video_summary decode is deferred to the
    # requested page so listing stays cheap.
    candidates: list[tuple[str, Path]] = []
    root = videos_dir(workspace)
    if root.exists():
        for folder in sorted(path for path in root.iterdir() if path.is_dir()):
            if group_filter and folder.name != group_filter:
                continue
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                    continue
                searchable = f"{folder.name}/{path.name}".lower()
                if text_filter and text_filter not in searchable:
                    continue
                candidates.append((folder.name, path))
    total = len(candidates)
    start = (page - 1) * page_size
    rows: list[dict[str, object]] = []
    for group_name, path in candidates[start : start + page_size]:
        video = video_summary(workspace, path, exact=False)
        rows.append(
            {
                "kind": "video_group",
                "group": group_name,
                "video": video["name"],
                "video_name": Path(str(video["name"])).stem,
                "path": str(path.resolve()),
                "frame_count": int(video.get("frame_count") or 0),
                "width": int(video.get("width") or 0),
                "height": int(video.get("height") or 0),
                "fps": video.get("fps"),
                "duration_seconds": video.get("duration_seconds"),
                "decodable": bool(video.get("decodable")),
                "thumbnail_url": video.get("thumbnail_url"),
            }
        )
    return _source_page_payload(rows, page, page_size, total, query=query)


def _compare_pred_sources(
    db: Database,
    workspace: WorkspaceConfig,
    query: dict[str, list[str]],
    run_id: int | None = None,
) -> dict:
    rows: list[dict[str, object]] = []
    text_filter = str(query.get("q", [""])[0] or "").strip().lower()
    video_filter = str(query.get("video", [""])[0] or "").strip()
    item_filter = _optional_int(query.get("item_id", [None])[0])
    item_rows = (
        [{"id": item_filter}]
        if item_filter is not None
        else db.query(
            """
            SELECT DISTINCT item_id AS id FROM media_item_members
            WHERE reusable_as_pred = 1 AND state = 'ready' AND deleted_at IS NULL
            ORDER BY item_id
            """
        )
    )
    for item_row in item_rows:
        payload = list_item_predictions(db, int(item_row["id"]))
        item = payload["item"]
        for prediction in payload["predictions"]:
            prediction_run_id = prediction.get("producer_run_id")
            # ``list_item_predictions`` already enforces this contract, but
            # keep the legacy source endpoint defensive if a hand-edited DB
            # row claims to be reusable. Compare-derived output is viewable in
            # its Run Detail, never a new Compare source.
            run_metadata = prediction.get("run_metadata") or {}
            if (
                prediction_run_id is not None
                and isinstance(run_metadata, dict)
                and str(run_metadata.get("run_type") or "model_inference") == "video_compare"
            ):
                continue
            if run_id is not None and int(prediction_run_id or 0) != int(run_id):
                continue
            row = {
                "kind": "media_item_member",
                "item_id": int(item["id"]),
                "member_id": int(prediction["id"]),
                "asset_id": int(prediction["asset_id"]),
                "run_id": int(prediction_run_id) if prediction_run_id is not None else None,
                "run_name": prediction.get("run_name"),
                "video": str(item.get("display_name") or ""),
                "video_name": str(item.get("display_name") or ""),
                "artifact_id": (prediction.get("metadata") or {}).get("artifact_id"),
                "frame_count": int(prediction.get("frame_count") or 0),
                "width": int(prediction.get("width") or 0),
                "height": int(prediction.get("height") or 0),
                "fps": prediction.get("fps"),
                "method_key": prediction.get("method_key"),
                "member_role": prediction.get("member_role"),
                "reusable_as_pred": True,
            }
            searchable = (
                f"{row.get('run_name') or ''} {row.get('video') or ''} "
                f"{row.get('method_key') or ''}"
            ).lower()
            if text_filter and text_filter not in searchable:
                continue
            if video_filter and str(row.get("video") or "") != video_filter and Path(
                str(row.get("video") or "")
            ).stem != video_filter:
                continue
            rows.append(row)
    rows = sorted(
        rows,
        key=lambda item: (
            str(item.get("video") or ""),
            str(item.get("run_name") or ""),
            int(item.get("member_id") or 0),
        ),
    )
    page, page_size = _source_pagination(query)
    total = len(rows)
    start = (page - 1) * page_size
    return _source_page_payload(rows[start : start + page_size], page, page_size, total, query=query)


def _source_pagination(query: dict[str, list[str]]) -> tuple[int, int]:
    page = max(1, int(query.get("page", ["1"])[0] or 1))
    page_size_raw = query.get("page_size", [None])[0]
    if page_size_raw in {None, ""}:
        return page, 10000
    return page, min(200, max(1, int(page_size_raw)))


def _source_page_payload(
    rows: list[dict[str, object]],
    page: int,
    page_size: int,
    total: int,
    query: dict[str, list[str]],
) -> dict:
    return {
        "sources": rows,
        "page": page,
        "page_size": page_size,
        "filtered_count": total,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "query": query.get("q", [""])[0],
    }


def _compare_layer_sources(db: Database, run_id: int, source_type: str, video_name: str | None = None) -> list[dict[str, object]]:
    run = db.get_run(run_id)
    if run.get("deleted_at") is not None or run.get("artifact_cleaned_at") is not None:
        return []
    if str((run.get("metadata") or {}).get("run_type") or "model_inference") == "video_compare":
        return []
    reusable = db.get(
        """
        SELECT 1 AS present FROM media_item_members
        WHERE producer_run_id = ? AND member_role = 'model_pred'
          AND reusable_as_pred = 1 AND state = 'ready' AND deleted_at IS NULL
        LIMIT 1
        """,
        (int(run_id),),
    )
    if reusable is None:
        return []
    kinds = ["mask0", "mask1"] if source_type == "mask" else ["flowt_0", "flowt_1", "warp0", "warp1", "blend"]
    groups: dict[tuple[str, str, str], dict[str, object]] = {}
    sample_cache: dict[int, dict[str, Any]] = {}
    for kind in kinds:
        for artifact in db.list_run_artifacts(run_id, kind=kind):
            sample_id = artifact.get("sample_id")
            if sample_id is None:
                continue
            sample = sample_cache.get(int(sample_id))
            if sample is None:
                sample = db.get_sample(int(sample_id))
                sample_cache[int(sample_id)] = sample
            sample_meta = sample.get("metadata") or {}
            artifact_meta = artifact.get("metadata") or {}
            current_video = str(sample_meta.get("video_name") or sample_meta.get("video_file") or "frames")
            if video_name and video_name not in {current_video, str(sample_meta.get("video_file") or "")}:
                continue
            track_label = str(artifact_meta.get("compare_track_label") or sample_meta.get("compare_track_label") or run.get("name") or f"run-{run_id}")
            key = (current_video, track_label, kind)
            row = groups.setdefault(
                key,
                {
                    "run_id": run_id,
                    "run_name": run.get("name"),
                    "video": current_video,
                    "kind": kind,
                    "track_label": track_label,
                    "sample_count": 0,
                    "artifact_ids": [],
                },
            )
            row["sample_count"] = int(row["sample_count"]) + 1
            if len(row["artifact_ids"]) < 5:
                row["artifact_ids"].append(int(artifact["id"]))
    return sorted(groups.values(), key=lambda item: (str(item["video"]), str(item["track_label"]), str(item["kind"])))


def _compare_samples(db: Database, query: dict[str, list[str]]) -> dict:
    run_ids = [
        int(part)
        for raw in query.get("run_id", [])
        for part in raw.split(",")
        if part.strip()
    ]
    video_name = query.get("video_name", [""])[0]
    frame_index = _optional_int(query.get("frame_index", [None])[0])
    if not run_ids or not video_name or frame_index is None:
        raise ValueError("compare samples requires run_id, video_name, and frame_index")
    runs = [_run_compare_payload(db, run_id) for run_id in run_ids]
    compatible = len({json.dumps(item["compare_key"], sort_keys=True) for item in runs}) <= 1
    samples = []
    for run in runs:
        row = _find_run_sample_by_frame(db, int(run["run"]["id"]), video_name, frame_index)
        samples.append(
            {
                "run_id": int(run["run"]["id"]),
                "run_name": run["run"]["name"],
                "sample": _run_sample_payload(db, int(run["run"]["id"]), int(row["id"])) if row else None,
            }
        )
    return {"compatible": compatible, "video_name": video_name, "frame_index": frame_index, "samples": samples}


def _bound_run_outputs(db: Database) -> list[dict[str, Any]]:
    """Compatibility view backed only by reusable, Item-bound predictions.

    Historical Run artifacts and Compare-derived media intentionally never
    appear here.  New clients should use the GT-first Item endpoints instead.
    """
    rows = db.query(
        """
        SELECT r.id AS run_id, r.name AS run_name, r.created_at AS run_created_at,
               mi.id AS item_id, mi.display_name AS video_name,
               mim.id AS member_id, mim.method_key,
               ma.id AS asset_id, ma.display_name, ma.frame_count, ma.width,
               ma.height, ma.fps
        FROM media_item_members mim
        JOIN media_items mi ON mi.id = mim.item_id
        JOIN media_assets ma ON ma.id = mim.asset_id
        JOIN runs r ON r.id = mim.producer_run_id
        WHERE mim.member_role = 'model_pred'
          AND mim.producer_kind = 'model_inference'
          AND mim.reusable_as_pred = 1
          AND mim.state = 'ready' AND mim.deleted_at IS NULL
          AND mi.state = 'ready' AND mi.deleted_at IS NULL
          AND ma.state = 'ready' AND ma.deleted_at IS NULL
          AND r.status IN ('completed', 'metric_queued', 'metric_running')
          AND r.deleted_at IS NULL AND r.artifact_cleaned_at IS NULL
          AND COALESCE(json_extract(r.metadata_json, '$.run_type'), 'model_inference') = 'model_inference'
        ORDER BY r.created_at DESC, r.id DESC, mi.display_name, mim.id DESC
        """
    )
    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        run_id = int(row["run_id"])
        run = grouped.setdefault(
            run_id,
            {
                "run_id": run_id,
                "run_name": str(row["run_name"] or f"Run {run_id}"),
                "created_at": row["run_created_at"],
                "videos": [],
            },
        )
        run["videos"].append(
            {
                "item_id": int(row["item_id"]),
                "video_name": str(row["video_name"]),
                "tracks": [
                    {
                        "member_id": int(row["member_id"]),
                        "asset_id": int(row["asset_id"]),
                        "track_label": str(row["method_key"] or row["run_name"] or ""),
                        "display_name": str(row["display_name"] or ""),
                        "frame_count": int(row["frame_count"] or 0),
                        "width": int(row["width"] or 0),
                        "height": int(row["height"] or 0),
                        "fps": row["fps"],
                    }
                ],
            }
        )
    for run in grouped.values():
        run["video_count"] = len(run["videos"])
        run["track_count"] = sum(len(video["tracks"]) for video in run["videos"])
    return list(grouped.values())


def _compare_input_binding_rows(db: Database, run_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = db.get_run(int(run_id))
    if str((run.get("metadata") or {}).get("run_type") or "") != "video_compare":
        raise ValueError(f"Run {run_id} is not a video_compare Run")
    rows = db.query(
        """
        SELECT b.*, mi.display_name AS item_display_name,
               active.member_role, active.producer_kind,
               active.producer_run_id, active.method_key,
               active.reusable_as_pred, active.asset_id,
               original.asset_id AS original_asset_id,
               ma.display_name AS asset_display_name,
               ma.media_kind, ma.frame_count, ma.width, ma.height, ma.fps,
               ma.source_kind, ma.state AS asset_state, ma.deleted_at AS asset_deleted_at
        FROM run_media_item_bindings b
        JOIN media_items mi ON mi.id = b.item_id
        JOIN media_item_members active ON active.id = b.active_member_id
        JOIN media_item_members original ON original.id = b.original_member_id
        JOIN media_assets ma ON ma.id = active.asset_id
        WHERE b.run_id = ? AND b.binding_role IN ('compare_gt', 'compare_pred')
        ORDER BY CASE b.binding_role WHEN 'compare_gt' THEN 0 ELSE 1 END, b.slot, b.id
        """,
        (int(run_id),),
    )
    if not rows:
        raise ValueError("this Compare Run has no Item-bound inputs")
    for row in rows:
        row["metadata"] = json.loads(row.pop("metadata_json") or "{}")
        row["reusable_as_pred"] = bool(row.get("reusable_as_pred"))
    return run, rows


def _compare_inputs_payload(db: Database, run_id: int) -> dict[str, Any]:
    run, rows = _compare_input_binding_rows(db, int(run_id))
    plan = dict((run.get("metadata") or {}).get("alignment_plan") or {})
    reports = plan.get("sources") or {}
    inputs: list[dict[str, Any]] = []
    for row in rows:
        slot = str(row["slot"] or ("gt" if row["binding_role"] == "compare_gt" else "pred"))
        inputs.append(
            {
                "slot": slot,
                "role": "gt" if row["binding_role"] == "compare_gt" else "pred",
                "item_id": int(row["item_id"]),
                "item_display_name": str(row["item_display_name"] or ""),
                "original_member_id": int(row["original_member_id"]),
                "active_member_id": int(row["active_member_id"]),
                "snapshot_active": int(row["active_member_id"]) != int(row["original_member_id"]),
                "member_role": str(row["member_role"] or ""),
                "producer_kind": str(row["producer_kind"] or ""),
                "method_key": str(row["method_key"] or ""),
                "display_name": str(row["asset_display_name"] or ""),
                "media_kind": str(row["media_kind"] or "video"),
                "frame_count": int(row["frame_count"] or 0),
                "width": int(row["width"] or 0),
                "height": int(row["height"] or 0),
                "fps": row["fps"],
                "alignment": reports.get(slot) or {},
                "original_url": f"/api/runs/{int(run_id)}/compare-inputs/{quote(slot)}/media?variant=original",
                "aligned_url": f"/api/runs/{int(run_id)}/compare-inputs/{quote(slot)}/media?variant=aligned",
            }
        )
    return {
        "run_id": int(run_id),
        "media_item_id": int(rows[0]["item_id"]),
        "alignment_plan": plan,
        "inputs": inputs,
    }


def _compare_input_media(
    db: Database,
    workspace: WorkspaceConfig,
    run_id: int,
    slot: str,
    *,
    variant: str,
) -> Path:
    if variant not in {"original", "aligned"}:
        raise ValueError("compare input media variant must be original or aligned")
    run, rows = _compare_input_binding_rows(db, int(run_id))
    binding = next((row for row in rows if str(row["slot"]) == str(slot)), None)
    if binding is None:
        raise KeyError(f"Compare input slot not found: {slot}")
    _item, _member, _asset, original_path = resolve_item_member(
        db,
        workspace,
        int(binding["active_member_id"]),
        require_reusable=False,
    )
    if variant == "original":
        return original_path

    plan = dict((run.get("metadata") or {}).get("alignment_plan") or {})
    if not plan or not plan.get("fingerprint"):
        raise ValueError("Compare Run has no Alignment Plan")
    frame_paths, fps = _compare_aligned_frame_paths(db, run, binding)
    if not frame_paths:
        raise FileNotFoundError(f"Compare input {slot} has no aligned frames")
    if str(binding.get("media_kind") or "video") == "frame_sequence":
        # The HTTP route accepts an optional frame index for original
        # directories.  The current detail UI requests one representative
        # aligned frame, so return the first already-materialized sample.
        return frame_paths[0]
    signature_rows = []
    for path in frame_paths:
        stat = path.stat()
        signature_rows.append((path.as_posix(), int(stat.st_size), int(stat.st_mtime_ns)))
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "run_id": int(run_id),
                "slot": str(slot),
                "active_member_id": int(binding["active_member_id"]),
                "alignment_fingerprint": str(plan["fingerprint"]),
                "frames": signature_rows,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    # Cache entries are intentionally direct children of compare_cache so the
    # shared cache catalog/lease service can validate and GC them safely.
    cache_root = (workspace.root / "compare_cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    output = (cache_root / f"{cache_key}.mp4").resolve()
    if output.parent != cache_root:
        raise ValueError("invalid Compare aligned media cache path")
    from vfieval.pipeline.inference import _write_mp4
    from vfieval.run_cleanup import CACHE_GRACE_SECONDS, cache_lease

    with cache_lease(db, workspace, "compare_cache", cache_key, output):
        if not output.is_file() or output.stat().st_size <= 0:
            temporary = output.with_name(f"{output.stem}.{uuid.uuid4().hex}.tmp.mp4")
            try:
                _write_mp4(frame_paths, temporary, fps or 24.0)
                os.replace(temporary, output)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        now = time.time()
        db.upsert_cache_entry(
            "compare_cache",
            cache_key,
            output,
            state="ready",
            size_bytes=int(output.stat().st_size),
            metadata={
                "run_id": int(run_id),
                "slot": str(slot),
                "alignment_fingerprint": str(plan["fingerprint"]),
            },
            last_used_at=now,
            gc_after=now + CACHE_GRACE_SECONDS,
        )
    return output


def _compare_aligned_frame_paths(
    db: Database,
    run: dict[str, Any],
    binding: dict[str, Any],
) -> tuple[list[Path], float]:
    samples = db.list_samples(int(run["dataset_id"]))
    binding_metadata = binding.get("metadata") or {}
    track_label = str(binding_metadata.get("track_label") or "")
    is_gt = str(binding["binding_role"]) == "compare_gt"
    selected: list[tuple[int, dict[str, Any]]] = []
    for sample in samples:
        metadata = sample.get("metadata") or {}
        if is_gt:
            if int(metadata.get("compare_track_index") or 0) != 0:
                continue
        elif track_label and str(metadata.get("compare_track_label") or "") != track_label:
            continue
        elif not track_label:
            expected_index = max(0, ord(str(binding["slot"])[-1:].lower() or "a") - ord("a"))
            if int(metadata.get("compare_track_index") or 0) != expected_index:
                continue
        selected.append((int(metadata.get("frame_index") or 0), sample))
    selected.sort(key=lambda pair: pair[0])
    paths: list[Path] = []
    fps = 0.0
    run_job_ids = set(db.run_inference_job_ids(int(run["id"])))
    for _index, sample in selected:
        metadata = sample.get("metadata") or {}
        fps = fps or float(metadata.get("fps") or 0.0)
        kind = "gt" if is_gt else "pred"
        artifact = next(
            (
                row
                for row in db.list_artifacts_by_sample(int(sample["id"]))
                if int(row.get("job_id") or 0) in run_job_ids and row.get("kind") == kind
            ),
            None,
        )
        candidate = Path(str(artifact["path"])) if artifact else Path(
            str(sample.get("gt_path") if is_gt else sample.get("img1_path"))
        )
        if not candidate.is_file():
            raise FileNotFoundError(f"aligned Compare frame is unavailable: {candidate}")
        paths.append(candidate.resolve())
    return paths, fps or float(binding.get("fps") or 24.0)


def _find_run_sample_by_frame(db: Database, run_id: int, video_name: str, frame_index: int) -> dict | None:
    run = db.get_run(run_id)
    for sample in db.list_samples(int(run["dataset_id"])):
        metadata = sample.get("metadata") or {}
        names = {
            str(metadata.get("video_name") or ""),
            str(metadata.get("video_file") or ""),
            str(Path(str(metadata.get("video_file") or "")).stem),
        }
        current_frame = int(metadata.get("frame_index") or metadata.get("sample_index") or 0)
        if video_name in names and current_frame == frame_index:
            return sample
    return None


def _create_registered_run(
    db: Database,
    body: dict,
    *,
    model_id: int,
    dataset_id: int,
    metrics: list[str],
) -> dict:
    submitted_body = dict(body)
    creation_body = {
        **submitted_body,
        "_submission_marker": _submission_marker(submitted_body),
    }

    def create() -> dict:
        default_name = f"model-{model_id}-dataset-{dataset_id}"
        run_id = db.create_run(
            name=creation_body.get("name") or default_name,
            experiment_id=_optional_int(creation_body.get("experiment_id")),
            model_id=model_id,
            dataset_id=dataset_id,
            height=int(creation_body["height"]),
            width=int(creation_body["width"]),
            batch_size=int(creation_body.get("batch_size", 1)),
            device=creation_body.get("device", "auto"),
            precision=creation_body.get("precision", "fp32"),
            metrics=metrics,
            metadata=_metadata_with_submission(
                creation_body.get("metadata") or {},
                creation_body,
            ),
        )
        return {"run_id": run_id, "run": db.get_run(run_id)}

    return idempotent_create(
        db,
        scope="run",
        body=submitted_body,
        resource_type="run",
        create=create,
        resource_id=lambda result: int(result["run_id"]),
        replay=lambda run_id: {"run_id": run_id, "run": db.get_run(run_id)},
        mark_replay=_mark_submission_result,
        recover=lambda submission_id, fingerprint: _recover_run_submission(
            db,
            submission_id,
            fingerprint,
        ),
    )


def _create_run_from_files_once(db: Database, workspace: WorkspaceConfig, body: dict) -> dict:
    artifact_profile = str(body.get("artifact_profile") or "evaluation")
    if artifact_profile not in {"evaluation", "diagnostic", "benchmark"}:
        raise ValueError("artifact_profile must be evaluation, diagnostic, or benchmark")
    cached = body.get("_preflight_result")
    preflight_payload = dict(body)
    if "_exact_checkpoint" in body:
        preflight_payload["checkpoint"] = body.get("_exact_checkpoint")
    preflight = (
        json.loads(json.dumps(cached, default=str))
        if isinstance(cached, dict)
        else preflight_run(db, workspace, preflight_payload)
    )
    workload = preflight.get("workload") or _attach_workload_estimate(db, workspace, preflight, body)
    if not preflight["ok"]:
        raise ValueError(_preflight_error_message(preflight))
    acknowledged_scope = str(
        body.get("_workload_risk_ack_scope_fingerprint") or ""
    ).strip()
    current_scope = (
        workload_confirmation_scope_fingerprint(workload)
        if workload is not None
        else ""
    )
    if (
        workload is not None
        and workload.get("risk_level") == "high"
        and body.get("_confirm_current_workload") is not True
        and acknowledged_scope != current_scope
        and str(body.get("risk_ack_fingerprint") or "")
        != str(workload.get("risk_fingerprint") or "")
    ):
        raise _WorkloadRiskConfirmationRequired(workload)
    if workload is not None:
        workload["storage_capacity"] = ensure_storage_capacity(
            db,
            workspace,
            requested_bytes=int(workload.get("artifact_budget_bytes") or 0),
        )
    metrics = list(body.get("metrics") or [])
    if artifact_profile == "benchmark" and metrics:
        raise ValueError("benchmark artifact_profile does not run metrics")
    unsupported = [name for name in metrics if name not in METRIC_NAMES]
    if unsupported:
        raise ValueError(f"unsupported metrics: {', '.join(unsupported)}")
    metric_batch_size = _optional_int(body.get("metric_batch_size_per_device"))
    if metric_batch_size is not None and metric_batch_size <= 0:
        raise ValueError("metric_batch_size_per_device must be a positive integer")

    model_path = resolve_model_file(workspace, str(body["model_file"]))
    selection = resolve_video_selection(workspace, body)
    groups = selection["groups"]
    multi_group = selection["multi_group"]
    # Multi-group runs root the dataset at videos/ and carry group-qualified
    # "group/file" selections so same-named clips never collide; single-group
    # runs keep rooting at the group folder with bare file names so existing
    # datasets, caches, and reference keys are byte-for-byte unchanged.
    if multi_group:
        dataset_root = str(videos_dir(workspace))
        group_label = " + ".join(groups)
    else:
        dataset_root = str(resolve_video_group(workspace, selection["primary_group"]))
        group_label = selection["primary_group"]
    frame_step = max(1, int(body.get("frame_step") or 1))
    max_frames = _optional_int(body.get("max_frames"))
    decode_backend = normalize_decode_backend(body.get("decode_backend") or "auto")
    video_infos = preflight.get("video_group", {}).get("videos", [])
    selected_videos = [str(name) for name in selection["selected_videos"]]
    height, width = resolve_run_dimensions(body, video_infos)
    visualize_height, visualize_width = _resolve_visualize_dimensions(body, height, width)
    execution_mode = str(body.get("execution_mode") or "single")
    devices = _resolve_execution_devices(body, execution_mode)
    requested_device = str(body.get("device") or "auto")
    is_multi = execution_mode in {"multi_cuda", "multi_npu"}
    device, precision = normalize_device_precision(devices[0] if is_multi else requested_device, str(body.get("precision") or "fp32"))
    checkpoint_selector = body.get("_exact_checkpoint", body.get("checkpoint"))
    checkpoint_path = resolve_checkpoint(workspace, checkpoint_selector, model_path.name)
    checkpoint_relative = _checkpoint_relative(workspace, checkpoint_path)
    model_record_base = model_path.name if checkpoint_relative is None else f"{model_path.name} [{checkpoint_relative}]"
    reference_config = _reference_config(
        video_group=group_label,
        selected_videos=selected_videos,
        frame_step=frame_step,
        max_frames=max_frames,
        resolution_mode=str(body.get("resolution_mode") or "original"),
        height=height,
        width=width,
    )
    reference_config["decode_backend"] = decode_backend
    reference_config["evaluation_contract"] = MIDPOINT_TRIPLET_CONTRACT
    trusted_preflight_inputs = body.get("_preflight_physical_inputs")
    if not isinstance(trusted_preflight_inputs, dict) and not isinstance(cached, dict):
        try:
            trusted_preflight_inputs = _physical_inputs_from_deep_preflight(
                preflight_payload,
                preflight,
            )
        except (TypeError, ValueError):
            # Tests and legacy internal callers may provide a synthetic
            # preflight without physical signatures.  They receive the normal
            # hashing path below; opaque HTTP tokens always carry signatures.
            trusted_preflight_inputs = None
    trusted_source_files = {
        str(row.get("relative_path") or ""): row
        for row in (
            (trusted_preflight_inputs or {}).get("sources")
            if isinstance(trusted_preflight_inputs, dict)
            else []
        )
        if isinstance(row, dict)
    }
    preflight_video_infos = {
        (str(row.get("group") or ""), str(row.get("video_file") or "")): row
        for row in video_infos
        if isinstance(row, dict)
    }
    # Bind every selected source clip to its exact canonical Item before the
    # Run starts.  This is deliberately based on the resolved group/file
    # selection, never a stem/hash guess, so a later pred can be reusable only
    # for the GT Item it actually came from.
    source_bindings: list[dict[str, Any]] = []
    for selected_video in selected_videos:
        if multi_group:
            group_name, separator, file_name = selected_video.partition("/")
            if not separator or not group_name or not file_name:
                raise ValueError(f"invalid qualified source video selection: {selected_video}")
            timeline_name = f"{group_name}/{Path(file_name).stem}"
        else:
            group_name = groups[0]
            file_name = selected_video
            timeline_name = Path(file_name).stem
        asset = ensure_folder_asset(
            db,
            workspace,
            group_name,
            file_name,
            media_info=preflight_video_infos.get((group_name, file_name)),
            trusted_content_identity=trusted_source_files.get(f"{group_name}/{file_name}"),
        )
        asset_id = int(asset["id"])
        item = ensure_canonical_gt_item(db, int(asset_id))
        source_bindings.append(
            {
                "asset_id": int(asset_id),
                "item_id": int(item["id"]),
                "video_name": timeline_name,
                "group": group_name,
                "file_name": file_name,
            }
        )
    request_metadata = {
        "run_type": "model_inference",
        "artifact_contract": CANONICAL_ARTIFACT_CONTRACT,
        "model_file": model_path.name,
        "video_group": group_label,
        "video_groups": groups,
        "resolution_mode": body.get("resolution_mode") or "original",
        "height": height,
        "width": width,
        "visualize_height": visualize_height,
        "visualize_width": visualize_width,
        "batch_size": int(body.get("batch_size") or 1),
        "batch_size_per_device": int(body.get("batch_size_per_device") or body.get("batch_size") or 1),
        "metric_batch_size_per_device": metric_batch_size,
        "device": body.get("device") or "auto",
        "devices": devices,
        "execution_mode": execution_mode,
        "precision": body.get("precision") or "fp32",
        "checkpoint": body.get("checkpoint") or "none",
        "frame_step": frame_step,
        "max_frames": max_frames,
        "decode_backend": decode_backend,
        "evaluation_contract": MIDPOINT_TRIPLET_CONTRACT,
        "selected_videos": selected_videos,
        "source_assets": body.get("source_assets") or [],
        "metrics": metrics,
        "artifact_profile": artifact_profile,
        "prefetch_workers": body.get("prefetch_workers"),
        "save_workers": body.get("save_workers"),
        "max_save_inflight": body.get("max_save_inflight"),
        "artifact_db_batch_size": body.get("artifact_db_batch_size"),
        "sample_npu_smi": body.get("sample_npu_smi", True),
        "benchmark_warmup_batches": int(body.get("benchmark_warmup_batches") or 10),
        "benchmark_samples": int(body.get("benchmark_samples") or 200),
    }
    input_identity = _build_model_run_input_identity(
        db,
        workspace,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        checkpoint_requested=body.get("checkpoint") or "none",
        source_bindings=source_bindings,
        request=request_metadata,
        refresh_source_hashes=body.get("input_identity") is not None,
        trusted_preflight_inputs=(
            trusted_preflight_inputs
            if isinstance(trusted_preflight_inputs, dict)
            else None
        ),
    )
    expected_identity = body.get("input_identity")
    if expected_identity is not None:
        validate_input_identity(expected_identity)
        assert_input_identity_matches(expected_identity, input_identity)
    clone_identity_comparison = None
    clone_source_identity = body.get("_clone_input_identity")
    if isinstance(clone_source_identity, dict):
        validate_input_identity(clone_source_identity)
        clone_identity_comparison = compare_input_identities(
            clone_source_identity,
            input_identity,
        )
    reference_config = {
        **reference_config,
        "source_identities": [
            {
                "item_id": int(row["item_id"]),
                "asset_id": int(row["asset_id"]),
                "qualified_name": str(row["qualified_name"]),
                "content": dict(row["content"]),
            }
            for row in input_identity["sources"]
        ],
    }
    reference_key = _reference_key(reference_config)
    physical_model_fingerprint = _reference_key(
        {
            "model": input_identity["model"],
            "checkpoint": input_identity["checkpoint"].get("resolved"),
        }
    )
    model_record_name = f"{model_record_base} @{physical_model_fingerprint[:12]}"
    model_id = db.upsert_model(
        name=model_record_name,
        adapter=f"file:{model_path}",
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        input_height=height,
        input_width=width,
        metadata={
            "source": "file",
            "model_file": model_path.name,
            "model_path": str(model_path),
            "checkpoint": checkpoint_relative,
            "contract": "Model.infer(img0, img1)",
            "physical_fingerprint": physical_model_fingerprint,
            "model_identity": input_identity["model"],
            "checkpoint_identity": input_identity["checkpoint"],
        },
    )
    dataset_id = db.upsert_dataset(
        name=f"video:{group_label}:{reference_key[:20]}",
        root_path=dataset_root,
        has_gt=True,
        source_type="video",
        decode_mode="video_gt_triplets",
        metadata={
            "source": "folder",
            "video_group": group_label,
            "video_groups": groups,
            "multi_group": multi_group,
            "frame_step": frame_step,
            "max_frames": max_frames,
            "decode_backend": decode_backend,
            "evaluation_contract": MIDPOINT_TRIPLET_CONTRACT,
            "video_glob": "*",
            "selected_videos": selected_videos,
            "source_assets": body.get("source_assets") or [],
            "reference_key": reference_key,
            "source_identities": reference_config["source_identities"],
        },
    )
    metadata = {
        "run_type": "model_inference",
        "source": "folder_flow",
        "request": request_metadata,
        "model_file": model_path.name,
        "artifact_contract": CANONICAL_ARTIFACT_CONTRACT,
        "evaluation_contract": MIDPOINT_TRIPLET_CONTRACT,
        "checkpoint": checkpoint_relative,
        "video_group": group_label,
        "video_groups": groups,
        "multi_group": multi_group,
        "execution_mode": execution_mode,
        "devices": devices,
        "visualize_height": visualize_height,
        "visualize_width": visualize_width,
        "npu_devices": devices if execution_mode == "multi_npu" else [],
        "reference_key": reference_key,
        "reference_config": reference_config,
        "worker_launch": _worker_launch_metadata(execution_mode, devices, bool(metrics)),
        "selected_videos": selected_videos,
        "metric_health": preflight.get("metrics", {}).get("health", {}),
        "preflight": preflight,
        "artifact_profile": artifact_profile,
        "workload": workload,
        "input_identity": input_identity,
    }
    metadata.update(_submission_marker_from_creation_body(body))
    if body.get("retry_of_run_id") is not None:
        metadata["retry_of_run_id"] = int(body["retry_of_run_id"])
    if body.get("clone_of_run_id") is not None:
        metadata["clone_of_run_id"] = int(body["clone_of_run_id"])
        metadata["clone_identity_comparison"] = clone_identity_comparison or {}
    name = body.get("name") or _default_run_name(model_path, checkpoint_relative, body.get("checkpoint"), group_label)
    output_dir = str(workspace.runs_dir / str(db.next_run_id()))
    run_id = db.create_run(
        name=name,
        model_id=model_id,
        dataset_id=dataset_id,
        height=height,
        width=width,
        batch_size=int(body.get("batch_size_per_device") or body.get("batch_size") or 1),
        device=execution_mode if is_multi else device,
        precision=precision,
        metrics=metrics,
        metadata={**metadata, "output_dir": output_dir},
        create_inference_job=False,
    )
    for source_binding in source_bindings:
        bind_run_asset(
            db,
            run_id,
            int(source_binding["asset_id"]),
            "source",
            video_name=str(source_binding["video_name"]),
            model_name=model_path.name,
            checkpoint=str(checkpoint_relative or ""),
            metadata={
                "input": True,
                "video_group": source_binding["group"],
                "video_file": source_binding["file_name"],
            },
        )
        bind_run_source(
            db,
            run_id,
            int(source_binding["item_id"]),
            video_name=str(source_binding["video_name"]),
            metadata={
                "video_group": source_binding["group"],
                "video_file": source_binding["file_name"],
            },
        )
    total_decode_frames = _decode_progress_total(video_infos, max_frames)
    db.add_run_job(
        run_id,
        "decode",
        {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "video_group": group_label,
            "video_groups": groups,
            "selected_videos": selected_videos,
            "video_count": len(selected_videos),
            "total_frames": total_decode_frames,
            "decode_backend": decode_backend,
        },
        progress_total=total_decode_frames,
        metadata={"phase": "decode"},
    )
    if not db.update_run_progress(run_id, 0, total_decode_frames):
        raise RuntimeError(f"run {run_id} rejected decode progress initialization")
    start_decode_worker(db, workspace)
    return {"run_id": run_id, "run": db.get_run(run_id), "preflight": preflight}


def _create_run_from_files(
    db: Database,
    workspace: WorkspaceConfig,
    body: dict,
    *,
    prepare: Callable[[dict], dict] | None = None,
) -> dict:
    submitted_body = dict(body)

    def create() -> dict:
        creation_body = (
            prepare(dict(submitted_body))
            if prepare is not None
            else expand_video_selection_payload(db, workspace, dict(submitted_body))
        )
        creation_body["_submission_marker"] = _submission_marker(submitted_body)
        return _create_run_from_files_once(db, workspace, creation_body)

    return idempotent_create(
        db,
        scope="run",
        body=submitted_body,
        resource_type="run",
        create=create,
        resource_id=lambda result: int(result["run_id"]),
        replay=lambda run_id: {"run_id": run_id, "run": db.get_run(run_id)},
        mark_replay=_mark_submission_result,
        recover=lambda submission_id, fingerprint: _recover_run_submission(
            db,
            submission_id,
            fingerprint,
        ),
    )


def _dedupe_track_labels(distorted_tracks: list[dict[str, Any]]) -> list[str]:
    """Return per-track labels that stay distinct after `_sample_token`.

    Track labels drive sample names (`{video}__{label}__{frame}`) and per-track
    artifact directories, so two tracks whose labels collapse to the same
    sanitized token would overwrite each other. Any track whose token has
    already been used (or whose label is blank) is suffixed with its 1-based
    position so every track keeps a unique, human-readable label.
    """
    labels: list[str] = []
    seen: set[str] = set()
    for index, track in enumerate(distorted_tracks):
        base = str(track.get("track_label") or track.get("label") or "").strip()
        if not base:
            base = f"pred{index + 1}"
        label = base
        # If the sanitized token is taken, suffix with the 1-based position. The
        # position is unique per track, so one bump always resolves the clash.
        if _sample_token(label) in seen:
            label = f"{base}#{index + 1}"
        seen.add(_sample_token(label))
        labels.append(label)
    return labels


def _prepare_media_item_compare_payload(
    db: Database,
    workspace: WorkspaceConfig,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Validate and translate the GT-first Compare request to descriptors.

    The public request contains only opaque Item/Member ids.  Filesystem paths
    are resolved later by ``compare_inputs`` and never accepted from clients.
    Legacy structured descriptors remain readable through the old path.
    """
    if str(body.get("run_type") or "") != "video_compare":
        return body
    raw_reference = body.get("reference")
    raw_distorted = body.get("distorted")
    raw_descriptors = raw_distorted if isinstance(raw_distorted, list) else [raw_distorted]
    if any(
        isinstance(descriptor, dict) and "path" in descriptor
        for descriptor in [raw_reference, *raw_descriptors]
    ):
        raise ValueError("Compare descriptors must not include client-supplied paths")
    item_value = body.get("media_item_id")
    member_values = body.get("pred_member_ids")
    if item_value in {None, ""} and member_values is None:
        inferred = _infer_legacy_compare_item_ids(db, workspace, body)
        if inferred is None:
            # Descriptor compatibility is deliberately an adapter onto the
            # Item contract, not a second source-selection path.  Letting an
            # unresolved legacy body continue would send it through the old
            # preflight branch, where an unbound historical artifact (or a
            # Compare result) could bypass canonical GT identity checks.
            raise ValueError(
                "legacy Compare descriptors must resolve to one canonical media item "
                "and one or two reusable prediction members"
            )
        item_value, member_values = inferred
    try:
        item_id = int(item_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("video_compare requires a positive media_item_id") from exc
    if item_id <= 0:
        raise ValueError("video_compare requires a positive media_item_id")
    if not isinstance(member_values, list) or not 1 <= len(member_values) <= 2:
        raise ValueError("video_compare requires one or two pred_member_ids")
    try:
        member_ids = [int(value) for value in member_values]
    except (TypeError, ValueError) as exc:
        raise ValueError("pred_member_ids must contain positive integers") from exc
    if any(value <= 0 for value in member_ids) or len(set(member_ids)) != len(member_ids):
        raise ValueError("pred_member_ids must contain one or two distinct positive integers")

    # This is the authoritative Item resolver: it enforces exact canonical-GT
    # identity, trusted managed paths, reusable-role constraints, and rejects
    # Compare/evaluation/snapshot sources even if malformed DB rows claim they
    # are reusable.  Keep the HTTP layer deliberately descriptor-only.
    resolved = resolve_media_item_compare(db, workspace, item_id, member_ids)
    canonical_member_id = int(resolved["reference"]["member_id"])
    members = list(resolved["members"])

    prepared = dict(body)
    prepared["media_item_id"] = item_id
    prepared["pred_member_ids"] = member_ids
    prepared["reference"] = {"kind": "media_item", "item_id": item_id}
    prepared["distorted"] = [
        {
            "kind": "media_item_member",
            "member_id": int(member["id"]),
            "label": str(member.get("run_name") or member.get("method_key") or f"Pred {index + 1}"),
        }
        for index, member in enumerate(members)
    ]
    prepared["align_mode"] = "strict"
    prepared["media_item_compare"] = True
    prepared["canonical_gt_member_id"] = canonical_member_id
    spatial_policy = dict(prepared.get("spatial_policy") or {})
    spatial_policy.setdefault("mode", "smallest_pred")
    spatial_policy.setdefault("filter", "lanczos")
    spatial_policy.setdefault("allow_known_aspect_stretch", True)
    prepared["spatial_policy"] = spatial_policy
    return prepared


def _infer_legacy_compare_item_ids(
    db: Database,
    workspace: WorkspaceConfig,
    body: dict[str, Any],
) -> tuple[int, list[int]] | None:
    """Resolve compatibility descriptors to exact Item/Member identities.

    This is intentionally a lookup, not a historical backfill: an old Run
    artifact without a trustworthy Item member remains unselectable.
    """
    reference = body.get("reference")
    distorted = body.get("distorted")
    if not isinstance(reference, dict):
        return None
    descriptors = distorted if isinstance(distorted, list) else [distorted]
    if not descriptors or any(not isinstance(descriptor, dict) for descriptor in descriptors):
        return None
    reference_kind = str(reference.get("kind") or "")
    if reference_kind == "media_item":
        reference_item_id = int(reference.get("item_id") or 0)
        get_media_item(db, reference_item_id)
    elif reference_kind == "media_asset":
        asset_id = int(reference.get("asset_id") or 0)
        item = db.get(
            """
            SELECT id FROM media_items
            WHERE canonical_gt_asset_id = ? AND state = 'ready' AND deleted_at IS NULL
            """,
            (asset_id,),
        )
        if item is None:
            raise ValueError("legacy Compare GT does not resolve to a canonical media item")
        reference_item_id = int(item["id"])
    elif reference_kind == "video_group":
        group = str(reference.get("group") or "").strip()
        video = str(reference.get("video") or "").strip()
        asset = db.get(
            """
            SELECT id FROM media_assets
            WHERE source_key = ? AND state = 'ready' AND deleted_at IS NULL
            """,
            (f"folder:{group}/{video}",),
        )
        if asset is None:
            raise ValueError("legacy Compare GT does not resolve to a canonical media item")
        reference_item_id = int(ensure_canonical_gt_item(db, int(asset["id"]))["id"])
    else:
        return None

    member_ids: list[int] = []
    for descriptor in descriptors:
        kind = str(descriptor.get("kind") or "")
        if kind == "media_item_member":
            member_id = int(descriptor.get("member_id") or 0)
        elif kind == "run_artifact":
            run_id = int(descriptor.get("run_id") or 0)
            source_run = db.get_run(run_id)
            if str((source_run.get("metadata") or {}).get("run_type") or "model_inference") == "video_compare":
                raise ValueError("legacy Compare cannot reuse artifacts produced by a video_compare Run")
            artifact_id = _optional_int(descriptor.get("artifact_id"))
            artifact_kind = str(descriptor.get("artifact_kind") or "pred_video")
            artifacts = db.list_run_artifacts(run_id, kind=artifact_kind)
            if artifact_id is None:
                video = str(descriptor.get("video") or "")
                matches = [
                    artifact
                    for artifact in artifacts
                    if not video
                    or str((artifact.get("metadata") or {}).get("video_name") or Path(str(artifact["path"])).stem)
                    in {video, Path(video).stem}
                ]
                if len(matches) != 1:
                    raise ValueError("legacy Run artifact descriptor is ambiguous or unavailable")
                artifact_id = int(matches[0]["id"])
            elif not any(int(artifact["id"]) == artifact_id for artifact in artifacts):
                raise ValueError("legacy Run artifact descriptor does not belong to its declared Run")
            media = db.get(
                "SELECT id FROM media_assets WHERE source_key = ? AND state = 'ready' AND deleted_at IS NULL",
                (f"run_artifact:{artifact_id}",),
            )
            if media is None:
                sync_run_assets(db, workspace, run_id)
                media = db.get(
                    "SELECT id FROM media_assets WHERE source_key = ? AND state = 'ready' AND deleted_at IS NULL",
                    (f"run_artifact:{artifact_id}",),
                )
            member_id = _legacy_reusable_member_id(
                db,
                int(media["id"]) if media is not None else None,
                source_name="legacy Run Pred",
            )
            if member_id is None:
                raise ValueError(
                    "legacy Run Pred has no trustworthy media item binding; create a new bound inference Run"
                )
        elif kind == "media_asset":
            member_id = _legacy_reusable_member_id(
                db,
                _optional_int(descriptor.get("asset_id")),
                source_name="legacy Pred asset",
            )
            if member_id is None:
                raise ValueError("legacy Pred asset has no trustworthy media item binding")
        else:
            return None
        member_ids.append(member_id)
    return reference_item_id, member_ids


def _legacy_reusable_member_id(
    db: Database,
    asset_id: int | None,
    *,
    source_name: str,
) -> int | None:
    """Return the sole reusable Item member for a legacy Pred asset.

    Legacy descriptors carry a physical asset/run id, not a semantic Item id.
    Ambiguous bindings must not be resolved by row order: choosing one would
    silently reinterpret a historical asset as a different GT Item.
    """
    if asset_id is None or int(asset_id) <= 0:
        return None
    rows = db.query(
        """
        SELECT id FROM media_item_members
        WHERE asset_id = ? AND reusable_as_pred = 1
          AND state = 'ready' AND deleted_at IS NULL
        ORDER BY id
        """,
        (int(asset_id),),
    )
    if len(rows) > 1:
        raise ValueError(f"{source_name} resolves to multiple media item bindings")
    return int(rows[0]["id"]) if rows else None


def _create_video_compare_run_once(db: Database, workspace: WorkspaceConfig, body: dict) -> dict:
    body = _prepare_media_item_compare_payload(db, workspace, dict(body))
    payload = dict(body)
    payload["run_type"] = "video_compare"
    preflight = preflight_run(db, workspace, payload)
    if not preflight["ok"]:
        raise ValueError(_preflight_error_message(preflight))
    metrics = list(body.get("metrics") or [])
    unsupported = [name for name in metrics if name not in METRIC_NAMES]
    if unsupported:
        raise ValueError(f"unsupported metrics: {', '.join(unsupported)}")

    reference = dict(preflight.get("reference") or {})
    distorted_tracks = [dict(track) for track in (preflight.get("distorted_tracks") or [])]
    if not distorted_tracks and preflight.get("distorted"):
        distorted_tracks = [dict(preflight["distorted"])]
    reference_path = str(reference.get("path") or "")
    distorted_path = str(distorted_tracks[0].get("path") if distorted_tracks else "")
    # External tracks use their exact strict dimensions. Platform-owned indexed
    # tracks use the inference resolution for generated aligned GT frames.
    alignment = preflight.get("alignment") or {}
    alignment_plan = dict(preflight.get("alignment_plan") or {})
    is_item_compare = bool(body.get("media_item_compare") or body.get("media_item_id") is not None)
    width = int(alignment.get("width") or 0)
    height = int(alignment.get("height") or 0)
    target_width = int(alignment.get("target_width") or 0)
    target_height = int(alignment.get("target_height") or 0)
    run_height = target_height or height
    run_width = target_width or width
    visualize_height, visualize_width = _resolve_visualize_dimensions(
        body,
        run_height,
        run_width,
    )
    video_name = Path(reference_path).stem
    # Track labels become sample-name tokens (`{video}__{label}__{frame}`) and
    # per-track artifact directories (`videos/{video}/{label}/`). Two selected
    # preds can carry the same label — e.g. two runs sharing an auto-generated
    # name, or labels that only differ in characters `_sample_token` collapses.
    # Left unqualified they collide on `UNIQUE(dataset_id, name)` and the later
    # track silently overwrites the earlier one (multiple preds appear as one).
    # Disambiguate on the sanitized token so every track keeps a distinct label.
    unique_labels = _dedupe_track_labels(distorted_tracks)
    compare_tracks = [
        {
            "distorted_path": str(track.get("path") or ""),
            "asset_id": track.get("asset_id"),
            "track_label": unique_labels[index],
            "track_run_id": track.get("run_id") or track.get("track_run_id"),
            "artifact_id": track.get("artifact_id"),
            "video_name": track.get("video_name") or track.get("video"),
            "width": int(track.get("width") or 0),
            "height": int(track.get("height") or 0),
            "needs_downscale": bool((int(track.get("width") or 0), int(track.get("height") or 0)) != (target_width, target_height)),
            # Source-clip mapping lets Compare reconstruct a Pred-aligned GT.
            # Legacy tracks without it must already satisfy exact strict alignment.
            "source_video_path": track.get("source_video_path"),
            "source_frame_indices": track.get("source_frame_indices"),
            "frame_step": track.get("frame_step"),
            "member_id": track.get("member_id"),
            "item_id": track.get("item_id"),
            "alignment_slot": track.get("alignment_slot") or f"pred_{chr(ord('a') + index)}",
        }
        for index, track in enumerate(distorted_tracks)
    ]
    reference_needs_downscale = bool((width, height) != (target_width, target_height))
    effective_frame_count = int(alignment.get("frame_count") or 0)
    reference_config = {
        "run_type": "video_compare",
        "reference_path": reference_path,
        "distorted_tracks": compare_tracks,
        "align_mode": "strict",
        "frame_count": effective_frame_count,
        "width": width,
        "height": height,
        "target_width": target_width,
        "target_height": target_height,
        "reference_needs_downscale": reference_needs_downscale,
    }
    reference_key = _reference_key(reference_config)
    compare_tag = reference_key[:12]
    model_id = db.upsert_model(
        name="video_compare",
        adapter="dummy",
        checkpoint_path=None,
        # Record the exact validated or platform-generated aligned resolution.
        input_height=target_height or height,
        input_width=target_width or width,
        metadata={"source": "compare", "run_type": "video_compare"},
    )
    dataset_id = db.upsert_dataset(
        name=f"compare:{compare_tag}",
        root_path=reference_path,
        has_gt=True,
        source_type="compare",
        decode_mode="compare",
        metadata={
            "reference_path": reference_path,
            "distorted_path": distorted_path,
            "compare_tracks": compare_tracks,
            "align_mode": "strict",
            "compare_tag": compare_tag,
            "video_name": video_name,
            # Exact validated or platform-generated aligned resolution.
            "compare_target_width": target_width,
            "compare_target_height": target_height,
            "reference_needs_downscale": reference_needs_downscale,
            # Exact post-selection frame count; no implicit truncation occurs.
            "compare_effective_frame_count": effective_frame_count,
            **({"alignment_plan": alignment_plan} if alignment_plan else {}),
        },
    )
    samples = scan_dataset(db, workspace, dataset_id)
    if samples <= 0:
        raise ValueError("compare inputs did not produce any aligned frames")

    metadata = {
        "run_type": "video_compare",
        "artifact_contract": CANONICAL_ARTIFACT_CONTRACT,
        "source": "direct_compare",
        "reference_path": reference_path,
        "reference_asset_id": reference.get("asset_id"),
        "distorted_path": distorted_path,
        "distorted_tracks": compare_tracks,
        "align_mode": "strict",
        "reference_key": reference_key,
        "reference_config": reference_config,
        "compare_target_width": target_width,
        "compare_target_height": target_height,
        "visualize_height": visualize_height,
        "visualize_width": visualize_width,
        "reference_needs_downscale": reference_needs_downscale,
        "compare_effective_frame_count": effective_frame_count,
        "media_item_id": int(body["media_item_id"]) if is_item_compare else None,
        "pred_member_ids": [int(value) for value in (body.get("pred_member_ids") or [])],
        "alignment_plan": alignment_plan,
        "publish_compare_pred_video": not is_item_compare,
        "metric_health": preflight.get("metrics", {}).get("health", {}),
        "request": {
            "run_type": "video_compare",
            "artifact_contract": CANONICAL_ARTIFACT_CONTRACT,
            "reference": body.get("reference"),
            "distorted": body.get("distorted"),
            "extra_layers": body.get("extra_layers") if "extra_layers" in body else None,
            "align_mode": "strict",
            "metrics": metrics,
            "visualize_height": visualize_height,
            "visualize_width": visualize_width,
            "media_item_id": int(body["media_item_id"]) if is_item_compare else None,
            "pred_member_ids": [int(value) for value in (body.get("pred_member_ids") or [])],
            "spatial_policy": body.get("spatial_policy") or {},
            "publish_compare_pred_video": not is_item_compare,
        },
        "preflight": preflight,
    }
    metadata.update(_submission_marker_from_creation_body(body))
    if body.get("retry_of_run_id") is not None:
        metadata["retry_of_run_id"] = int(body["retry_of_run_id"])
    run_id = db.create_run(
        name=body.get("name") or f"compare / {Path(reference_path).stem}",
        model_id=model_id,
        dataset_id=dataset_id,
        height=run_height,
        width=run_width,
        batch_size=1,
        device="cpu",
        precision="fp32",
        metrics=metrics,
        metadata={**metadata, "output_dir": str(workspace.runs_dir / str(db.next_run_id()))},
    )
    register_run_cache_refs(db, workspace, run_id)
    if reference.get("asset_id") is not None:
        from vfieval.media_assets import bind_run_asset

        bind_run_asset(
            db,
            run_id,
            int(reference["asset_id"]),
            "gt",
            video_name=video_name,
            metadata={"input": True, "descriptor_kind": reference.get("descriptor_kind")},
        )
        for track in compare_tracks:
            if track.get("asset_id") is None:
                continue
            bind_run_asset(
                db,
                run_id,
                int(track["asset_id"]),
                "pred",
                video_name=video_name,
                track_label=str(track.get("track_label") or ""),
                metadata={"input": True},
            )
    if is_item_compare:
        item_id = int(body["media_item_id"])
        canonical_member_id = int(
            body.get("canonical_gt_member_id")
            or reference.get("member_id")
            or 0
        )
        if canonical_member_id <= 0:
            raise ValueError("Item Compare canonical GT member is missing")
        bind_compare_input(
            db,
            run_id,
            item_id,
            canonical_member_id,
            binding_role="compare_gt",
            slot="gt",
            metadata={"alignment_fingerprint": alignment_plan.get("fingerprint")},
        )
        for index, track in enumerate(compare_tracks):
            member_id = int(track.get("member_id") or 0)
            if member_id <= 0:
                raise ValueError("Item Compare prediction member is missing")
            bind_compare_input(
                db,
                run_id,
                item_id,
                member_id,
                binding_role="compare_pred",
                slot=str(track.get("alignment_slot") or f"pred_{chr(ord('a') + index)}"),
                metadata={
                    "track_label": str(track.get("track_label") or ""),
                    "alignment_fingerprint": alignment_plan.get("fingerprint"),
                },
            )
    _start_local_inference_worker(db, workspace, run_id=run_id)
    return {"run_id": run_id, "run": db.get_run(run_id), "preflight": preflight}


def _create_video_compare_run(
    db: Database,
    workspace: WorkspaceConfig,
    body: dict,
    *,
    prepare: Callable[[dict], dict] | None = None,
) -> dict:
    submitted_body = dict(body)

    def create() -> dict:
        creation_body = (
            prepare(dict(submitted_body))
            if prepare is not None
            else dict(submitted_body)
        )
        creation_body["_submission_marker"] = _submission_marker(submitted_body)
        return _create_video_compare_run_once(db, workspace, creation_body)

    return idempotent_create(
        db,
        scope="run",
        body=submitted_body,
        resource_type="run",
        create=create,
        resource_id=lambda result: int(result["run_id"]),
        replay=lambda run_id: {"run_id": run_id, "run": db.get_run(run_id)},
        mark_replay=_mark_submission_result,
        recover=lambda submission_id, fingerprint: _recover_run_submission(
            db,
            submission_id,
            fingerprint,
        ),
    )


def _submission_marker(body: dict) -> dict[str, str]:
    submission_id = str(body.get("submission_id") or "").strip()
    if not submission_id:
        return {}
    return {
        "submission_id": submission_id,
        "submission_fingerprint": submission_fingerprint(body),
    }


def _submission_marker_from_creation_body(body: dict) -> dict[str, str]:
    marker = body.get("_submission_marker")
    if not isinstance(marker, dict):
        return {}
    submission_id = str(marker.get("submission_id") or "").strip()
    fingerprint = str(marker.get("submission_fingerprint") or "").strip()
    if not submission_id or not fingerprint:
        return {}
    return {
        "submission_id": submission_id,
        "submission_fingerprint": fingerprint,
    }


def _metadata_with_submission(metadata: dict, body: dict) -> dict:
    return {
        **dict(metadata),
        **_submission_marker_from_creation_body(body),
    }


def _recover_run_submission(
    db: Database,
    submission_id: str,
    fingerprint: str,
) -> dict | None:
    rows = db.query(
        """
        SELECT id
        FROM runs
        WHERE json_extract(metadata_json, '$.submission_id') = ?
          AND json_extract(metadata_json, '$.submission_fingerprint') = ?
        ORDER BY id
        LIMIT 2
        """,
        (str(submission_id), str(fingerprint)),
    )
    if not rows:
        return None
    if len(rows) > 1:
        raise SubmissionConflict(
            "multiple Runs match the stale submission reservation",
            code="submission_recovery_ambiguous",
        )
    run_id = int(rows[0]["id"])
    return {"run_id": run_id, "run": db.get_run(run_id)}


def _mark_submission_result(result: dict, replayed: bool, submission_id: str) -> dict:
    return {
        **result,
        "submission_id": submission_id,
        "idempotent_replay": bool(replayed),
    }


def _normalize_preflight_file_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("deep preflight file identity is missing")
    return build_file_identity_from_values(
        relative_path=str(value.get("relative_path") or ""),
        display_path=str(value.get("display_path") or ""),
        size_bytes=value.get("size_bytes"),
        mtime_ns=value.get("mtime_ns"),
        sha256=str(value.get("sha256") or ""),
    )


def _physical_inputs_from_deep_preflight(
    request: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract only portable, content-bound trust material from deep preflight."""

    if (
        str(result.get("run_type") or request.get("run_type") or "model_inference")
        != "model_inference"
        or str(result.get("preflight_level") or request.get("preflight_level") or "deep")
        != "deep"
        or not bool(result.get("ok"))
    ):
        return None
    model = result.get("model") or {}
    videos = (result.get("video_group") or {}).get("videos") or []
    model_identity = _normalize_preflight_file_identity(model.get("physical_identity"))
    checkpoint_identity = model.get("checkpoint_identity")
    normalized_checkpoint = (
        _normalize_preflight_file_identity(checkpoint_identity)
        if checkpoint_identity is not None
        else None
    )
    normalized_sources = []
    for video in videos:
        if not isinstance(video, dict):
            raise ValueError("deep preflight video identity is invalid")
        normalized_sources.append(
            _normalize_preflight_file_identity(video.get("physical_identity"))
        )
    if not normalized_sources:
        raise ValueError("deep preflight contains no source identities")
    return {
        "schema": "deep-preflight-physical-inputs-v1",
        "model": model_identity,
        "checkpoint": normalized_checkpoint,
        "sources": normalized_sources,
    }


def _revalidate_preflight_physical_inputs(
    workspace: WorkspaceConfig,
    request: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Re-hash token-bound files and reject any stat, content, or selection drift."""

    if str(expected.get("schema") or "") != "deep-preflight-physical-inputs-v1":
        raise ValueError("preflight token input identity is invalid; run preflight again")
    expected_model = _normalize_preflight_file_identity(expected.get("model"))
    expected_checkpoint_raw = expected.get("checkpoint")
    expected_checkpoint = (
        _normalize_preflight_file_identity(expected_checkpoint_raw)
        if expected_checkpoint_raw is not None
        else None
    )
    expected_sources = [
        _normalize_preflight_file_identity(row)
        for row in (expected.get("sources") or [])
    ]
    try:
        model_path = resolve_model_file(workspace, str(request.get("model_file") or ""))
        checkpoint_path = resolve_checkpoint(
            workspace,
            request.get("_exact_checkpoint", request.get("checkpoint")),
            model_path.name,
        )
        selection = resolve_video_selection(workspace, request)
        source_paths = [Path(str(entry["path"])).resolve() for entry in selection["entries"]]

        if model_path.relative_to(models_dir(workspace)).as_posix() != expected_model["relative_path"]:
            raise ValueError("resolved model changed")
        if (checkpoint_path is None) != (expected_checkpoint is None):
            raise ValueError("resolved checkpoint changed")
        if checkpoint_path is not None and expected_checkpoint is not None:
            if (
                checkpoint_path.relative_to(checkpoints_dir(workspace)).as_posix()
                != expected_checkpoint["relative_path"]
            ):
                raise ValueError("resolved checkpoint changed")
        source_relatives = [
            path.relative_to(videos_dir(workspace)).as_posix()
            for path in source_paths
        ]
        if source_relatives != [row["relative_path"] for row in expected_sources]:
            raise ValueError("selected source videos changed")

        current_model = build_file_identity(
            model_path,
            trusted_root=models_dir(workspace),
            display_path=expected_model["display_path"],
        )
        current_checkpoint = (
            build_file_identity(
                checkpoint_path,
                trusted_root=checkpoints_dir(workspace),
                display_path=expected_checkpoint["display_path"],
            )
            if checkpoint_path is not None and expected_checkpoint is not None
            else None
        )
        current_sources = [
            build_file_identity(
                path,
                trusted_root=videos_dir(workspace),
                display_path=expected_row["display_path"],
            )
            for path, expected_row in zip(source_paths, expected_sources, strict=True)
        ]
    except Exception as exc:
        raise ValueError(
            f"Run inputs changed after preflight; run preflight again ({exc})"
        ) from exc

    current = {
        "schema": "deep-preflight-physical-inputs-v1",
        "model": current_model,
        "checkpoint": current_checkpoint,
        "sources": current_sources,
    }
    if current != {
        "schema": "deep-preflight-physical-inputs-v1",
        "model": expected_model,
        "checkpoint": expected_checkpoint,
        "sources": expected_sources,
    }:
        raise ValueError("Run inputs changed after preflight; run preflight again")
    return current


def _preflight_request_fingerprint(request: dict[str, Any]) -> str:
    identity_request = dict(request)
    identity_request.pop("preflight_level", None)
    normalized = normalize_request_identity(identity_request)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _attach_workload_estimate(
    db: Database,
    workspace: WorkspaceConfig,
    preflight: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any] | None:
    if str(preflight.get("run_type") or request.get("run_type") or "model_inference") != "model_inference":
        return None
    resolution = preflight.get("resolution") or {}
    height = int(resolution.get("height") or request.get("height") or 0)
    width = int(resolution.get("width") or request.get("width") or 0)
    if height <= 0 or width <= 0:
        return None
    device_info = preflight.get("device") or {}
    effective_devices = [
        str(value)
        for value in (device_info.get("effective_devices") or [])
        if str(value or "").strip()
    ]
    if not effective_devices:
        effective_devices = [str(device_info.get("effective_device") or request.get("device") or "cpu")]
    capabilities = detect_capabilities()
    memory_by_device = {
        str(row.get("id") or ""): int(row["memory_bytes"])
        for kind in ("cuda", "npu")
        for row in (capabilities.get(kind) or [])
        if isinstance(row, dict) and row.get("memory_bytes")
    }
    known_device_memory = [
        memory_by_device[device]
        for device in effective_devices
        if device in memory_by_device
    ]
    device_memory = min(known_device_memory) if known_device_memory else None
    workload = estimate_workload(
        device=effective_devices[0],
        precision=str(device_info.get("effective_precision") or request.get("precision") or "fp32"),
        device_memory_bytes=device_memory,
        host_available_memory_bytes=_host_available_memory_bytes(),
        batch_size_per_device=int(
            request.get("batch_size_per_device") or request.get("batch_size") or 1
        ),
        height=height,
        width=width,
        sample_count=int((preflight.get("video_group") or {}).get("triplets") or 0),
        artifact_profile=str(request.get("artifact_profile") or "evaluation"),
        prefetch_depth=3,
    )
    capacity = storage_capacity(
        db,
        workspace,
        requested_bytes=int(workload.get("artifact_budget_bytes") or 0),
    )
    workload["storage_capacity"] = capacity
    if not capacity["sufficient"]:
        workload["risk_level"] = "high"
        workload.setdefault("risk_reasons", []).append(
            {
                "code": "insufficient_workspace_storage",
                "free_bytes": int(capacity["free_bytes"]),
                "required_free_bytes": int(capacity["required_free_bytes"]),
            }
        )
    preflight["workload"] = workload
    if workload["risk_level"] == "high" and not any(
        row.get("type") == "WorkloadRisk"
        for row in (preflight.get("warnings") or [])
        if isinstance(row, dict)
    ):
        preflight.setdefault("warnings", []).append(
            {
                "title": "High resource workload",
                "message": "Review memory and artifact estimates before publishing this Run.",
                "type": "WorkloadRisk",
                "details": {"reasons": workload["risk_reasons"]},
            }
        )
    return workload


def _host_available_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, OSError, ValueError):
        return None


def _trusted_preflight_sha256(
    value: Any,
    path: Path,
    trusted_root: Path,
) -> str | None:
    """Reuse a server-validated digest only while its exact stat still matches."""

    if not isinstance(value, dict):
        return None
    try:
        identity = _normalize_preflight_file_identity(value)
        relative = path.resolve().relative_to(trusted_root.resolve()).as_posix()
        stat_result = path.stat()
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(
            "Run inputs changed after preflight; run preflight again"
        ) from exc
    if (
        relative != identity["relative_path"]
        or int(stat_result.st_size) != int(identity["size_bytes"])
        or int(stat_result.st_mtime_ns) != int(identity["mtime_ns"])
    ):
        raise ValueError("Run inputs changed after preflight; run preflight again")
    return str(identity["sha256"])


def _build_model_run_input_identity(
    db: Database,
    workspace: WorkspaceConfig,
    *,
    model_path: Path,
    checkpoint_path: Path | None,
    checkpoint_requested: Any,
    source_bindings: list[dict[str, Any]],
    request: dict[str, Any],
    refresh_source_hashes: bool,
    trusted_preflight_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trusted_preflight_inputs = (
        trusted_preflight_inputs
        if isinstance(trusted_preflight_inputs, dict)
        and trusted_preflight_inputs.get("schema") == "deep-preflight-physical-inputs-v1"
        else {}
    )
    trusted_model_sha = _trusted_preflight_sha256(
        trusted_preflight_inputs.get("model"),
        model_path,
        models_dir(workspace),
    )
    model_identity = build_file_identity(
        model_path,
        trusted_root=models_dir(workspace),
        display_path=model_path.name,
        content_sha256=trusted_model_sha,
    )
    checkpoint_file_identity = None
    if checkpoint_path is not None:
        trusted_checkpoint_sha = _trusted_preflight_sha256(
            trusted_preflight_inputs.get("checkpoint"),
            checkpoint_path,
            checkpoints_dir(workspace),
        )
        checkpoint_file_identity = build_file_identity(
            checkpoint_path,
            trusted_root=checkpoints_dir(workspace),
            display_path=checkpoint_path.relative_to(checkpoints_dir(workspace)).as_posix(),
            content_sha256=trusted_checkpoint_sha,
        )
    source_identities: list[dict[str, Any]] = []
    trusted_videos = videos_dir(workspace)
    trusted_sources = {
        str(row.get("relative_path") or ""): row
        for row in (trusted_preflight_inputs.get("sources") or [])
        if isinstance(row, dict)
    }
    for binding in source_bindings:
        asset = get_asset(db, int(binding["asset_id"]))
        source_path = Path(str(asset.get("storage_path") or "")).resolve()
        qualified_name = f"{binding['group']}/{binding['file_name']}"
        try:
            catalog_relative = source_path.relative_to(trusted_videos).as_posix()
        except ValueError as exc:
            raise ValueError("cataloged source media is outside videos/") from exc
        if catalog_relative != qualified_name:
            raise ValueError(
                f"catalog source identity mismatch: expected {qualified_name}, got {catalog_relative}"
            )
        stat_result = source_path.stat()
        asset_metadata = asset.get("metadata") or {}
        catalog_digest = None
        if (
            not refresh_source_hashes
            and int(asset.get("size_bytes") or 0) == int(stat_result.st_size)
            and int(asset_metadata.get("source_mtime_ns") or 0) == int(stat_result.st_mtime_ns)
        ):
            catalog_digest = str(asset.get("content_sha256") or "") or None
        file_identity = build_file_identity(
            source_path,
            trusted_root=trusted_videos,
            display_path=qualified_name,
            content_sha256=(
                _trusted_preflight_sha256(
                    trusted_sources.get(catalog_relative),
                    source_path,
                    trusted_videos,
                )
                or catalog_digest
            ),
        )
        source_identities.append(
            build_source_identity(
                item_id=int(binding["item_id"]),
                asset_id=int(binding["asset_id"]),
                qualified_name=qualified_name,
                file_identity=file_identity,
            )
        )
    return build_run_input_identity(
        model=model_identity,
        checkpoint=build_checkpoint_identity(
            checkpoint_requested,
            resolved=checkpoint_file_identity,
        ),
        sources=source_identities,
        request=request,
    )


def _retry_run(db: Database, workspace: WorkspaceConfig, run_id: int) -> dict:
    run = db.get_run(run_id)
    metadata = run.get("metadata") or {}
    request = dict(metadata.get("request") or {})
    if not request:
        raise ValueError("this Run does not contain a retryable file-input request")
    request["retry_of_run_id"] = run_id
    request["name"] = f"{run['name']} retry"
    if str(request.get("run_type") or "model_inference") == "video_compare":
        return _create_video_compare_run(db, workspace, request)
    identity = metadata.get("input_identity")
    if not isinstance(identity, dict):
        raise ValueError("this legacy Run has no exact input identity; use Clone instead")
    validate_input_identity(identity)
    assert_input_identity_files_available(
        identity,
        models_root=models_dir(workspace),
        checkpoints_root=checkpoints_dir(workspace),
        videos_root=videos_dir(workspace),
    )
    request["input_identity"] = identity
    request["_exact_checkpoint"] = resolved_checkpoint_relative_path(identity) or "none"
    # Clicking Retry explicitly confirms repeating the stored workload. Exact
    # identity validation below still fences model, checkpoint, and media drift.
    request["_confirm_current_workload"] = True
    return _create_run_from_files(db, workspace, request)


def _clone_run(
    db: Database,
    workspace: WorkspaceConfig,
    run_id: int,
    *,
    risk_ack_fingerprint: Any = None,
) -> dict:
    run = db.get_run(run_id)
    source_metadata = run.get("metadata") or {}
    request = dict(source_metadata.get("request") or {})
    if not request:
        raise ValueError("this Run does not contain a cloneable file-input request")
    request.pop("retry_of_run_id", None)
    request["clone_of_run_id"] = run_id
    request["name"] = f"{run['name']} clone"
    if str(request.get("run_type") or "model_inference") == "video_compare":
        return _create_video_compare_run(db, workspace, request)
    if isinstance(source_metadata.get("input_identity"), dict):
        request["_clone_input_identity"] = source_metadata["input_identity"]
    request_fingerprint = _preflight_request_fingerprint(request)
    acknowledged = str(risk_ack_fingerprint or "").strip()
    if risk_ack_fingerprint is not None and not isinstance(risk_ack_fingerprint, str):
        raise ValueError("risk_ack_fingerprint must be a string")
    acknowledged_scope = None
    if acknowledged:
        acknowledged_scope = _consume_clone_risk_ack(
            workspace,
            run_id,
            acknowledged,
            request_fingerprint,
        )
    if acknowledged_scope:
        # The acknowledgement binds the exact workload returned by the prior
        # 409. _create_run_from_files still requires its stable execution scope
        # to match after re-reading current files and device availability.
        request["_workload_risk_ack_scope_fingerprint"] = acknowledged_scope
    try:
        return _create_run_from_files(db, workspace, request)
    except _WorkloadRiskConfirmationRequired as exc:
        challenge = str(exc.workload.get("risk_fingerprint") or "").strip()
        if challenge:
            _remember_clone_risk_ack(
                workspace,
                run_id,
                challenge,
                request_fingerprint,
                workload_confirmation_scope_fingerprint(exc.workload),
            )
        raise


def _clone_risk_ack_scope(workspace: WorkspaceConfig) -> str:
    return str(workspace.db_path.resolve())


def _remember_clone_risk_ack(
    workspace: WorkspaceConfig,
    run_id: int,
    risk_fingerprint: str,
    request_fingerprint: str,
    workload_scope_fingerprint: str,
) -> None:
    now = time.monotonic()
    key = (_clone_risk_ack_scope(workspace), int(run_id), str(risk_fingerprint))
    with _CLONE_RISK_ACK_LOCK:
        expired = [
            row_key
            for row_key, row in _CLONE_RISK_ACK_CHALLENGES.items()
            if float(row.get("expires_at") or 0.0) <= now
        ]
        for row_key in expired:
            _CLONE_RISK_ACK_CHALLENGES.pop(row_key, None)
        _CLONE_RISK_ACK_CHALLENGES[key] = {
            "expires_at": now + _CLONE_RISK_ACK_TTL_SECONDS,
            "request_fingerprint": str(request_fingerprint),
            "workload_scope_fingerprint": str(workload_scope_fingerprint),
        }


def _consume_clone_risk_ack(
    workspace: WorkspaceConfig,
    run_id: int,
    risk_fingerprint: str,
    request_fingerprint: str,
) -> str | None:
    now = time.monotonic()
    key = (_clone_risk_ack_scope(workspace), int(run_id), str(risk_fingerprint))
    with _CLONE_RISK_ACK_LOCK:
        row = _CLONE_RISK_ACK_CHALLENGES.get(key)
        if row is None:
            return None
        if (
            float(row.get("expires_at") or 0.0) <= now
            or str(row.get("request_fingerprint") or "") != str(request_fingerprint)
        ):
            _CLONE_RISK_ACK_CHALLENGES.pop(key, None)
            return None
        _CLONE_RISK_ACK_CHALLENGES.pop(key, None)
        scope_fingerprint = str(row.get("workload_scope_fingerprint") or "")
        return scope_fingerprint or None


def _reference_config(
    video_group: str,
    selected_videos: list[str],
    frame_step: int,
    max_frames: int | None,
    resolution_mode: str,
    height: int,
    width: int,
) -> dict:
    return {
        "video_group": video_group,
        "selected_videos": list(selected_videos),
        "frame_step": int(frame_step),
        "max_frames": max_frames,
        "resolution_mode": resolution_mode,
        "height": int(height),
        "width": int(width),
        "decode_strategy": DECODE_STRATEGY_VERSION,
    }


def _reference_key(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _worker_launch_metadata(execution_mode: str, devices: list[str], has_metrics: bool) -> dict:
    if execution_mode == "multi_npu":
        return {
            "mode": "auto_process",
            "inference_workers": [{"role": "inference", "device_filter": device} for device in devices],
            "metric_worker": bool(has_metrics),
        }
    if execution_mode == "multi_cuda":
        return {
            "mode": "local_threads",
            "inference_workers": [{"role": "all", "device_filter": None} for _device in devices],
            "metric_worker": False,
        }
    return {"mode": "local_thread", "inference_workers": [{"role": "all", "device_filter": None}], "metric_worker": False}


def _cleanup_run_artifacts(db: Database, workspace: WorkspaceConfig, run_id: int) -> dict:
    request = RunCleanupService(db, workspace).request_artifact_cleanup(run_id)
    if request.get("status") != "completed":
        raise ValueError((request.get("error") or {}).get("message") or "artifact cleanup did not complete")
    return dict(request.get("report") or {})


def _checkpoint_relative(workspace: WorkspaceConfig, checkpoint_path: Path | None) -> str | None:
    if checkpoint_path is None:
        return None
    from vfieval.file_inputs import checkpoints_dir

    return checkpoint_path.resolve().relative_to(checkpoints_dir(workspace)).as_posix()


def _default_run_name(
    model_path: Path,
    checkpoint_relative: str | None,
    checkpoint_request: Any,
    video_group: str,
) -> str:
    """Compose the default run name as model-checkpoint-videogroup.

    The checkpoint segment prefers the resolved weight file stem; it falls back
    to the raw request ("auto"/"none") so a run created without a checkpoint
    still reads clearly.
    """
    if checkpoint_relative:
        checkpoint_label = Path(checkpoint_relative).stem
    else:
        request = str(checkpoint_request or "none").strip().lower()
        checkpoint_label = request if request in {"auto", "none"} else "none"
    return f"{model_path.stem}-{checkpoint_label}-{video_group}"


def _resolve_visualize_dimensions(body: dict, height: int, width: int) -> tuple[int, int]:
    """Resolve the requested preview resolution.

    Canonical artifacts always use the Run's output resolution.  These values
    affect only optional LANCZOS previews and therefore do not alter inference,
    metrics, video publication, Compare, or Campaign inputs.
    """
    raw_h = body.get("visualize_height")
    raw_w = body.get("visualize_width")
    vis_h = int(raw_h) if raw_h else min(DEFAULT_VISUALIZE_HEIGHT, int(height))
    vis_w = int(raw_w) if raw_w else min(DEFAULT_VISUALIZE_WIDTH, int(width))
    if vis_h <= 0:
        vis_h = min(DEFAULT_VISUALIZE_HEIGHT, int(height))
    if vis_w <= 0:
        vis_w = min(DEFAULT_VISUALIZE_WIDTH, int(width))
    # Explicit preview dimensions are exact, including intentional upscaling;
    # they never feed canonical composition or metric inputs.
    return vis_h, vis_w


def _resolve_execution_devices(body: dict, execution_mode: str) -> list[str]:
    if execution_mode == "single":
        return [str(body.get("device") or "auto")]
    if execution_mode not in {"multi_cuda", "multi_npu"}:
        raise ValueError("execution_mode must be single, multi_cuda, or multi_npu")
    kind = "cuda" if execution_mode == "multi_cuda" else "npu"
    raw_devices = body.get("devices") or []
    devices = [str(device) for device in raw_devices if str(device).startswith(f"{kind}:")]
    if not devices:
        capabilities = detect_capabilities()
        devices = [str(row["id"]) for row in capabilities.get(kind, [])]
    if not devices:
        raise ValueError(f"{execution_mode} requires at least one {kind.upper()} device")
    return devices


def _create_inference_shards(
    db: Database,
    run_id: int,
    model_id: int,
    dataset_id: int,
    height: int,
    width: int,
    precision: str,
    metrics: list[str],
    devices: list[str],
    batch_size_per_device: int,
) -> None:
    samples = db.list_samples(dataset_id)
    partitions = _partition_samples_by_video(samples, devices)
    job_specs: list[dict] = []
    for shard_index, device in enumerate(devices):
        sample_ids = partitions[shard_index]
        if not sample_ids:
            continue
        payload = {
            "run_id": run_id,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "height": height,
            "width": width,
            "batch_size": batch_size_per_device,
            "device": device,
            "precision": precision,
            "metrics": [],
            "sample_ids": sample_ids,
            "shard_index": shard_index,
            "shard_count": len(devices),
        }
        job_specs.append(
            {
                "payload": payload,
                "progress_total": len(sample_ids),
                "shard_index": shard_index,
                "device": device,
                "metadata": {"metrics_after_all_shards": metrics},
            }
        )
    if not db.publish_inference_jobs(run_id, job_specs):
        raise ValueError(f"Run {run_id} rejected inference shard publication")


def _partition_samples_by_video(samples: list[dict], devices: list[str]) -> list[list[int]]:
    grouped: dict[str, list[int]] = {}
    for sample in samples:
        metadata = sample.get("metadata") or {}
        key = str(metadata.get("video_file") or metadata.get("video_name") or sample.get("name"))
        grouped.setdefault(key, []).append(int(sample["id"]))
    partitions: list[list[int]] = [[] for _ in devices]
    loads = [0 for _ in devices]
    for _key, sample_ids in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        shard_index = min(range(len(devices)), key=lambda index: loads[index])
        partitions[shard_index].extend(sample_ids)
        loads[shard_index] += len(sample_ids)
    return partitions


def _preflight_error_message(preflight: dict) -> str:
    messages = [f"{item.get('title')}: {item.get('message')}" for item in preflight.get("errors", [])]
    return "；".join(messages) or "预检查失败"


def _start_local_inference_worker(
    db: Database,
    workspace: WorkspaceConfig,
    count: int = 1,
    *,
    run_id: int | None = None,
) -> None:
    """Compatibility entry point routed through the shared Job supervisor."""
    if run_id is not None:
        start_workers_for_run(db, workspace, int(run_id))
        return
    if wake_job_supervisor(db):
        return
    raise RuntimeError("a run_id is required when no Job supervisor is active")


def _selection_hash(selected_videos: list[str], frame_step: int, max_frames: int | None) -> str:
    data = {
        "selected_videos": selected_videos,
        "frame_step": frame_step,
        "max_frames": max_frames,
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _decode_progress_total(video_infos: list[dict], max_frames: int | None) -> int:
    total = 0
    for info in video_infos:
        frame_count = int(info.get("frame_count") or 0)
        total += min(frame_count, int(max_frames)) if max_frames else frame_count
    return total


CORE_TIMELINE_ARTIFACTS = {
    "gt",
    "pred",
    "difference",
    "flowt_0",
    "flowt_1",
    "mask0",
    "mask1",
    "warp0",
    "warp1",
    "blend",
}

VIDEO_TIMELINE_ARTIFACTS = {"pred_video", "gt_video", "diff_video"}
COMPARE_LAYER_ARTIFACTS = ("flowt_0", "flowt_1", "mask0", "mask1", "warp0", "warp1", "blend")
METRIC_DIRECTIONS = {
    "vmaf": "higher_is_better",
    "lpips_vit_patch": "lower_is_better",
    "lpips_convnext": "lower_is_better",
    "cgvqm": "lower_is_better",
}
METRIC_TIMELINE_SUPPORT = {
    "lpips_vit_patch": True,
    "lpips_convnext": True,
    "vmaf": False,
    "cgvqm": False,
}
METRIC_STATUSES = ("pending", "running", "completed", "unavailable", "failed", "skipped", "missing")


def _run_detail(db: Database, run_id: int) -> dict:
    run = db.get_run(run_id)
    run["jobs"] = db.list_run_jobs(run_id)
    run["feedback"] = db.list_run_feedback(run_id)
    run["artifact_storage"] = _artifact_storage_diagnostic(run)
    return run


def _artifact_storage_diagnostic(run: dict[str, Any]) -> dict[str, Any]:
    """Compare stored preflight budget with the completion-time byte summary."""

    workload = (run.get("metadata") or {}).get("workload") or {}
    summary = run.get("artifact_summary") or {}
    predicted_raw = workload.get("artifact_budget_bytes")
    try:
        predicted = max(0, int(predicted_raw)) if predicted_raw is not None else None
    except (TypeError, ValueError):
        predicted = None
    has_actual = "storage_bytes" in summary
    try:
        actual = max(0, int(summary.get("storage_bytes") or 0)) if has_actual else None
    except (TypeError, ValueError):
        actual = None
        has_actual = False
    try:
        unknown = max(0, int(summary.get("storage_size_unknown") or 0))
    except (TypeError, ValueError):
        unknown = 0
    try:
        known = max(0, int(summary.get("storage_size_known") or 0))
    except (TypeError, ValueError):
        known = 0
    if not has_actual:
        measurement = "unavailable"
    elif unknown:
        measurement = "partial"
    elif run.get("artifact_cleaned_at") is not None:
        measurement = "cleaned"
    elif str(run.get("status") or "") in TERMINAL_RUN_STATUSES:
        measurement = "complete"
    else:
        measurement = "in_progress"
    comparable = predicted is not None and predicted > 0 and actual is not None
    return {
        "predicted_artifact_budget_bytes": predicted,
        "actual_artifact_bytes": actual,
        "actual_size_known_artifacts": known,
        "actual_size_unknown_artifacts": unknown,
        "measurement": measurement,
        "actual_minus_predicted_bytes": (
            int(actual - predicted) if comparable else None
        ),
        "budget_utilization_ratio": (
            float(actual / predicted) if comparable else None
        ),
        "source": "artifact_summary",
    }


def _parse_feedback_rating(rating_raw: Any) -> float | None:
    """Parse a rating on the 1.00–5.00 scale in 0.25 steps.

    Returns ``None`` when no rating was supplied (blank/None). Raises
    ``ValueError`` for out-of-range values or ones that do not fall on a quarter
    step, so a slip like 3.3 is rejected rather than silently rounded.
    """
    if rating_raw in {None, ""}:
        return None
    try:
        rating = float(rating_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("rating must be a number between 1 and 5 in steps of 0.25") from exc
    if rating < 1 or rating > 5:
        raise ValueError("rating must be between 1 and 5")
    if abs(rating * 4 - round(rating * 4)) > 1e-6:
        raise ValueError("rating must be in steps of 0.25")
    return round(round(rating * 4) / 4, 2)


def _feedback_context(db: Database, run_id: int, body: dict) -> dict[str, str]:
    """Resolve the (video, track, model, checkpoint) a feedback row describes.

    The client sends the video and — for compare runs — the pred track label it
    is scoring. The model/checkpoint come from the client when a track is picked
    (compare tracks each carry their own weight) and otherwise fall back to the
    run's own model metadata, so a plain inference run still records what it ran.
    """
    run = db.get_run(run_id)
    metadata = run.get("metadata") or {}
    video = str(body.get("video") or "").strip()[:200]
    track_label = str(body.get("track_label") or "").strip()[:200]
    model_name = str(body.get("model_name") or "").strip()
    checkpoint = str(body.get("checkpoint") or "").strip()
    # Compare runs stitch several source runs together, one per pred track, each
    # with its own weight. When the client names the track it is scoring, chase
    # that track's source run so the row records the real model/checkpoint rather
    # than the synthetic "video_compare" model the compare run itself carries.
    if not model_name or not checkpoint:
        track_run_id = body.get("track_run_id")
        if track_run_id in (None, "") and track_label:
            for track in metadata.get("distorted_tracks") or []:
                if str(track.get("track_label") or "").strip() == track_label:
                    track_run_id = track.get("track_run_id") or track.get("run_id")
                    break
        if track_run_id not in (None, ""):
            try:
                source_run = db.get_run(int(track_run_id))
            except (KeyError, ValueError, TypeError):
                source_run = None
            if source_run is not None:
                source_meta = source_run.get("metadata") or {}
                if not model_name:
                    model_name = str(source_meta.get("model_file") or source_run.get("model_name") or "").strip()
                if not checkpoint:
                    checkpoint = str(source_meta.get("checkpoint") or "").strip()
    if not model_name:
        model_name = str(metadata.get("model_file") or run.get("model_name") or "").strip()
    if not checkpoint:
        checkpoint = str(metadata.get("checkpoint") or "").strip()
    return {
        "video": video,
        "track_label": track_label,
        "model_name": model_name[:200],
        "checkpoint": checkpoint[:200],
    }


def _create_run_feedback(db: Database, run_id: int, body: dict) -> dict:
    """Validate and persist one feedback entry for a run.

    A submission needs at least a rating or a non-empty issue — a blank form
    should not create a row. Rating, when present, must fall on the 0.25-step
    1–5 scale. The entry also records the video/track/model/checkpoint it scores
    so the stats tab can group by content rather than only by run.
    """
    db.get_run(run_id)  # 404s via the caller if the run does not exist.
    username = str(body.get("username") or "").strip()[:120]
    issue = str(body.get("issue") or "").strip()
    rating = _parse_feedback_rating(body.get("rating"))
    if rating is None and not issue:
        raise ValueError("feedback requires a rating or an issue")
    context = _feedback_context(db, run_id, body)
    feedback_id = db.add_run_feedback(
        run_id,
        username,
        rating,
        issue,
        video=context["video"],
        track_label=context["track_label"],
        model_name=context["model_name"],
        checkpoint=context["checkpoint"],
    )
    return {
        "run_id": run_id,
        "feedback_id": feedback_id,
        "feedback": db.list_run_feedback(run_id),
    }


def _update_run_feedback(db: Database, run_id: int, feedback_id: int, body: dict) -> dict:
    """Patch an existing feedback entry so a mis-scored review can be corrected.

    Only the fields present in the body change. ``rating`` may be set to a new
    0.25-step value, or explicitly cleared by sending ``null``/``""``. The result
    must still carry a rating or an issue, matching create-time validation.
    """
    existing = {int(row["id"]): row for row in db.list_run_feedback(run_id)}
    current = existing.get(int(feedback_id))
    if current is None:
        raise KeyError("feedback not found")

    kwargs: dict[str, Any] = {}
    if "username" in body:
        kwargs["username"] = str(body.get("username") or "").strip()[:120]
    if "issue" in body:
        kwargs["issue"] = str(body.get("issue") or "").strip()

    rating_present = "rating" in body
    if rating_present:
        if body.get("rating") in {None, ""}:
            kwargs["clear_rating"] = True
        else:
            kwargs["rating"] = _parse_feedback_rating(body.get("rating"))

    # Guard against a patch that empties the row of both signals.
    final_rating = (
        None if kwargs.get("clear_rating")
        else kwargs.get("rating", current.get("rating"))
    )
    final_issue = kwargs.get("issue", current.get("issue") or "")
    if final_rating is None and not str(final_issue).strip():
        raise ValueError("feedback requires a rating or an issue")

    if not db.update_run_feedback(run_id, feedback_id, **kwargs):
        raise ValueError("no feedback fields to update")
    return {
        "run_id": run_id,
        "feedback_id": feedback_id,
        "feedback": db.list_run_feedback(run_id),
    }


def _feedback_overview(
    db: Database,
    *,
    dataset: str | None = None,
    model_name: str | None = None,
    checkpoint: str | None = None,
    video: str | None = None,
) -> dict:
    """Aggregate run feedback for the statistics tab.

    Wraps ``db.feedback_stats`` (overall + per-user/run/video/model/checkpoint
    rollups) and adds the 0.25-step rating distribution, filter options, and a
    recent-entries feed. The filter arguments narrow the population (dataset,
    model, checkpoint, video) before aggregation.
    """
    stats = db.feedback_stats(
        dataset=dataset or None,
        model_name=model_name or None,
        checkpoint=checkpoint or None,
        video=video or None,
    )
    active = {"dataset": dataset, "model_name": model_name, "checkpoint": checkpoint, "video": video}
    stats["filters"] = {k: v for k, v in active.items() if v}
    stats["filter_options"] = db.feedback_filter_options()

    stats["recent"] = db.list_recent_feedback(
        limit=100,
        dataset=dataset or None,
        model_name=model_name or None,
        checkpoint=checkpoint or None,
        video=video or None,
    )
    return stats


def _run_timeline(db: Database, run_id: int) -> dict:
    run = db.get_run(run_id)
    samples = db.list_samples(int(run["dataset_id"]))
    artifacts = db.list_run_artifacts(run_id)
    metrics = _latest_metric_rows(db.list_run_metrics(run_id))
    metric_job_status = _metric_job_status(db, run)
    requested_sample_metrics = _requested_sample_metrics(run)
    requested_video_metrics = _requested_video_metrics(run)

    samples_with_artifacts: set[int] = set()
    sample_errors: dict[int, dict[str, str]] = {}
    video_artifacts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        kind = artifact["kind"]
        sample_id = artifact.get("sample_id")
        if sample_id is None:
            if kind in VIDEO_TIMELINE_ARTIFACTS:
                name = str(artifact.get("metadata", {}).get("video_name") or "video")
                video_artifacts.setdefault(name, {})[kind] = int(artifact["id"])
            continue
        if kind == "sample_error":
            meta = artifact.get("metadata") or {}
            sample_errors[int(sample_id)] = {"error_type": meta.get("error_type", ""), "message": meta.get("message", "")}
            continue
        samples_with_artifacts.add(int(sample_id))

    metrics_by_sample: dict[int, dict[str, dict[str, object]]] = {}
    metric_status_by_sample: dict[int, dict[str, int]] = {}
    video_metric_summaries: dict[str, dict[str, dict[str, object]]] = {}
    for metric in metrics:
        metric_name = metric["metric_name"]
        value = metric.get("value")
        payload = {"status": metric["status"], "value": value, "details": metric.get("details") or {}}
        sample_id = metric.get("sample_id")
        if sample_id is not None:
            sample_key = int(sample_id)
            metrics_by_sample.setdefault(sample_key, {})[metric_name] = payload
            metric_status_by_sample.setdefault(sample_key, {status: 0 for status in METRIC_STATUSES})
            metric_status_by_sample[sample_key][metric["status"]] = metric_status_by_sample[sample_key].get(metric["status"], 0) + 1
        else:
            video_name = str((metric.get("details") or {}).get("video_name") or "video")
            video_metric_summaries.setdefault(video_name, {})[metric_name] = payload

    groups: dict[str, dict[str, object]] = {}
    summary = _run_metric_summary(db, run_id)
    for sample in samples:
        metadata = sample.get("metadata") or {}
        video_name = str(metadata.get("video_name") or metadata.get("video_file") or "frames")
        group = groups.setdefault(
            video_name,
            {
                "video_name": video_name,
                "video_file": metadata.get("video_file") or video_name,
                "fps": float(metadata.get("fps") or 0.0),
                "samples": [],
                "video_artifacts": video_artifacts.get(video_name, {}),
                "video_metrics": video_metric_summaries.get(video_name, {}),
                "metric_summary": {},
                "worst_samples": {},
            },
        )
        sample_id = int(sample["id"])
        timestamps = metadata.get("timestamps") or {}
        sample_metrics = _sample_metrics_with_defaults(
            metrics_by_sample.get(sample_id, {}),
            requested_sample_metrics,
            bool(sample.get("gt_path")),
            str(run.get("status") or ""),
            metric_job_status,
        )
        sample_entry = {
                "sample_id": sample_id,
                "sample_name": sample["name"],
                "frame_index": int(metadata.get("frame_index") or metadata.get("sample_index") or 0),
                "sample_index": int(metadata.get("sample_index") or 0),
                "timestamp": timestamps.get("gt") if isinstance(timestamps, dict) else None,
                "has_artifacts": sample_id in samples_with_artifacts,
                "has_gt": bool(sample.get("gt_path")),
                "metrics": sample_metrics,
                "metric_status": _metric_status_counts(sample_metrics),
            }
        if sample_id in sample_errors:
            sample_entry["error"] = sample_errors[sample_id]
        group["samples"].append(sample_entry)

    videos = []
    for group in groups.values():
        group["samples"] = sorted(group["samples"], key=lambda item: (item["frame_index"], item["sample_index"]))
        group["video_metrics"] = _video_metrics_with_defaults(
            actual_metrics=group.get("video_metrics") or {},
            requested_video_metrics=requested_video_metrics,
            run_status=str(run.get("status") or ""),
            metric_job_status=metric_job_status,
        )
        group["metric_summary"] = _metric_summary_for_video(group["samples"], summary.get("metrics", {}))
        group["worst_samples"] = _worst_samples_for_video(group["samples"])
        videos.append(group)
    videos.sort(key=lambda item: str(item.get("video_file") or item.get("video_name")))
    return {"run_id": run_id, "metric_summary": summary, "videos": videos}


def _run_videos(db: Database, run_id: int, page: int = 1, page_size: int = 50, q: str = "") -> dict:
    run = db.get_run(run_id)
    summaries = db.list_run_video_summaries(run_id, q)
    requested_video_metrics = _requested_video_metrics(run)
    metric_job_status = _metric_job_status(db, run)
    page_size = min(200, max(1, int(page_size or 50)))
    page = max(1, int(page or 1))
    total = len(summaries)
    start = (page - 1) * page_size
    videos = []
    for summary in summaries[start : start + page_size]:
        video_name = str(summary["video_name"])
        actual_video_metrics = _video_metric_payloads(db, run_id, video_name)
        video_artifacts = _video_artifacts_for_video(db, run_id, video_name)
        videos.append(
            {
                "video_name": video_name,
                "video_file": summary.get("video_file") or video_name,
                "fps": summary.get("fps"),
                "sample_count": int(summary.get("sample_count") or 0),
                "video_artifacts": _video_artifact_map_from_rows(video_artifacts),
                "video_artifact_tracks": _video_artifact_tracks_from_rows(video_artifacts),
                "video_metrics": _video_metrics_with_defaults(
                    actual_metrics=actual_video_metrics,
                    requested_video_metrics=requested_video_metrics,
                    run_status=str(run.get("status") or ""),
                    metric_job_status=metric_job_status,
                ),
                "metric_summary": {},
                "worst_samples": {},
            }
        )
    return {
        "run_id": run_id,
        "page": page,
        "page_size": page_size,
        "filtered_count": total,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "videos": videos,
    }


def _run_video_timeline(
    db: Database,
    run_id: int,
    video_name: str,
    metric: str | None = None,
    bucket_count: int = 120,
    window_start: int = 0,
    window_size: int = 300,
) -> dict:
    run = db.get_run(run_id)
    sample_rows = sorted(db.list_samples_by_video(run_id, video_name), key=_timeline_sample_sort_key)
    if not sample_rows:
        raise ValueError(f"video not found in run timeline: {video_name}")
    job_ids = set(db.run_inference_job_ids(run_id))
    requested_sample_metrics = _requested_sample_metrics(run)
    requested_video_metrics = _requested_video_metrics(run)
    metric_job_status = _metric_job_status(db, run)
    metric_entries = _sample_metric_entries(db, run, sample_rows, job_ids, requested_sample_metrics, metric_job_status)
    metric_name = metric or _first_metric_name(metric_entries)
    bucket_count = min(1000, max(1, int(bucket_count or 120)))
    window_size = min(2000, max(1, int(window_size or 300)))
    window_start = max(0, min(int(window_start or 0), max(0, len(sample_rows) - 1)))
    window_rows = sample_rows[window_start : window_start + window_size]
    window_samples = _sample_timeline_entries(db, run, window_rows, job_ids, requested_sample_metrics, metric_job_status)
    first_meta = (sample_rows[0].get("metadata") if sample_rows else {}) or {}
    current_video_name = str(first_meta.get("video_name") or video_name)
    video_file = str(first_meta.get("video_file") or current_video_name)
    actual_video_metrics = _video_metric_payloads(db, run_id, current_video_name)
    video_artifacts = _video_artifacts_for_video(db, run_id, current_video_name)
    return {
        "run_id": run_id,
        "video_name": current_video_name,
        "video_file": video_file,
        "fps": float(first_meta.get("fps") or 0.0),
        "metric": metric_name,
        "sample_count": len(sample_rows),
        "window_start": window_start,
        "window_size": window_size,
        "overview": _timeline_buckets(metric_entries, metric_name, bucket_count),
        "samples": window_samples,
        "video_artifacts": _video_artifact_map_from_rows(video_artifacts),
        "video_artifact_tracks": _video_artifact_tracks_from_rows(video_artifacts),
        "video_metrics": _video_metrics_with_defaults(
            actual_metrics=actual_video_metrics,
            requested_video_metrics=requested_video_metrics,
            run_status=str(run.get("status") or ""),
            metric_job_status=metric_job_status,
        ),
        "metric_summary": _metric_summary_for_video(
            metric_entries,
            {name: _empty_metric_summary(name) for name in run.get("metrics") or []},
        ),
        "worst_samples": _worst_samples_for_video(metric_entries),
    }


def _timeline_sample_sort_key(sample: dict) -> tuple[int, str, int, int]:
    metadata = sample.get("metadata") or {}
    return (
        int(metadata.get("frame_index") or metadata.get("sample_index") or 0),
        str(metadata.get("compare_track_label") or ""),
        int(metadata.get("sample_index") or 0),
        int(sample.get("id") or 0),
    )


def _build_video_payload(db: Database, run: dict[str, object], samples: list[dict], requested_video_name: str) -> dict:
    job_ids = set(db.run_inference_job_ids(int(run["id"])))
    requested_sample_metrics = _requested_sample_metrics(run)
    requested_video_metrics = _requested_video_metrics(run)
    metric_job_status = _metric_job_status(db, run)
    entries = _sample_timeline_entries(db, run, samples, job_ids, requested_sample_metrics, metric_job_status)
    entries.sort(
        key=lambda item: (
            int(item.get("frame_index") or 0),
            str(item.get("track_label") or ""),
            int(item.get("sample_index") or 0),
        )
    )
    first_meta = (samples[0].get("metadata") if samples else {}) or {}
    video_name = str(first_meta.get("video_name") or requested_video_name)
    video_file = str(first_meta.get("video_file") or video_name)
    actual_video_metrics = _video_metric_payloads(db, int(run["id"]), video_name)
    video_artifacts = _video_artifacts_for_video(db, int(run["id"]), video_name)
    video_metrics = _video_metrics_with_defaults(
        actual_metrics=actual_video_metrics,
        requested_video_metrics=requested_video_metrics,
        run_status=str(run.get("status") or ""),
        metric_job_status=metric_job_status,
    )
    return {
        "video_name": video_name,
        "video_file": video_file,
        "fps": float(first_meta.get("fps") or 0.0),
        "samples": entries,
        "video_artifacts": _video_artifact_map_from_rows(video_artifacts),
        "video_artifact_tracks": _video_artifact_tracks_from_rows(video_artifacts),
        "video_metrics": video_metrics,
        "metric_summary": _metric_summary_for_video(entries, {name: _empty_metric_summary(name) for name in run.get("metrics") or []}),
        "worst_samples": _worst_samples_for_video(entries),
    }


def _sample_timeline_entry(
    db: Database,
    run: dict[str, object],
    sample: dict,
    job_ids: set[int],
    requested_sample_metrics: list[str],
    metric_job_status: str | None,
) -> dict[str, object]:
    artifact_rows = [row for row in db.list_artifacts_by_sample(int(sample["id"])) if int(row["job_id"]) in job_ids]
    metric_rows = _latest_metric_rows(
        [row for row in db.list_metrics_by_sample(int(sample["id"])) if int(row["inference_job_id"]) in job_ids]
    )
    return _sample_timeline_entry_from_rows(
        run,
        sample,
        artifact_rows,
        metric_rows,
        requested_sample_metrics,
        metric_job_status,
    )


def _sample_timeline_entries(
    db: Database,
    run: dict[str, object],
    samples: list[dict],
    job_ids: set[int],
    requested_sample_metrics: list[str],
    metric_job_status: str | None,
) -> list[dict[str, object]]:
    sample_ids = [int(sample["id"]) for sample in samples]
    artifacts_by_sample: dict[int, list[dict]] = {}
    for artifact in db.list_artifacts_by_samples(sample_ids, job_ids=job_ids):
        artifacts_by_sample.setdefault(int(artifact["sample_id"]), []).append(artifact)
    metrics_by_sample: dict[int, list[dict]] = {}
    for metric in _latest_metric_rows(db.list_metrics_by_samples(sample_ids, inference_job_ids=job_ids)):
        metrics_by_sample.setdefault(int(metric["sample_id"]), []).append(metric)
    return [
        _sample_timeline_entry_from_rows(
            run,
            sample,
            artifacts_by_sample.get(int(sample["id"]), []),
            metrics_by_sample.get(int(sample["id"]), []),
            requested_sample_metrics,
            metric_job_status,
        )
        for sample in samples
    ]


def _sample_metric_entries(
    db: Database,
    run: dict[str, object],
    samples: list[dict],
    job_ids: set[int],
    requested_sample_metrics: list[str],
    metric_job_status: str | None,
) -> list[dict[str, object]]:
    sample_ids = [int(sample["id"]) for sample in samples]
    metrics_by_sample: dict[int, list[dict]] = {}
    for metric in _latest_metric_rows(db.list_metrics_by_samples(sample_ids, inference_job_ids=job_ids)):
        metrics_by_sample.setdefault(int(metric["sample_id"]), []).append(metric)
    return [
        _sample_timeline_entry_from_rows(
            run,
            sample,
            [],
            metrics_by_sample.get(int(sample["id"]), []),
            requested_sample_metrics,
            metric_job_status,
        )
        for sample in samples
    ]


def _sample_timeline_entry_from_rows(
    run: dict[str, object],
    sample: dict,
    artifact_rows: list[dict],
    metric_rows: list[dict],
    requested_sample_metrics: list[str],
    metric_job_status: str | None,
) -> dict[str, object]:
    sample_id = int(sample["id"])
    sample_errors: list[dict[str, str]] = []
    has_artifacts = False
    for artifact in artifact_rows:
        kind = artifact["kind"]
        if kind == "sample_error":
            meta = artifact.get("metadata") or {}
            sample_errors.append({"error_type": meta.get("error_type", ""), "message": meta.get("message", "")})
        else:
            has_artifacts = True
    actual_metrics = {
        row["metric_name"]: {
            "status": row["status"],
            "value": row.get("value"),
            "details": row.get("details") or {},
        }
        for row in metric_rows
    }
    sample_metrics = _sample_metrics_with_defaults(
        actual_metrics,
        requested_sample_metrics,
        bool(sample.get("gt_path")),
        str(run.get("status") or ""),
        metric_job_status,
    )
    metadata = sample.get("metadata") or {}
    timestamps = metadata.get("timestamps") or {}
    entry: dict[str, object] = {
        "sample_id": sample_id,
        "sample_name": sample["name"],
        "frame_index": int(metadata.get("frame_index") or metadata.get("sample_index") or 0),
        "sample_index": int(metadata.get("sample_index") or 0),
        "timestamp": timestamps.get("gt") if isinstance(timestamps, dict) else None,
        "has_artifacts": has_artifacts,
        "has_gt": bool(sample.get("gt_path")),
        "metrics": sample_metrics,
        "metric_status": _metric_status_counts(sample_metrics),
        "track_label": metadata.get("compare_track_label"),
        "track_index": metadata.get("compare_track_index"),
    }
    if sample_errors:
        entry["error"] = sample_errors[-1]
    return entry


def _video_metric_payloads(db: Database, run_id: int, video_name: str) -> dict[str, dict[str, object]]:
    rows = _latest_metric_rows(db.list_run_video_metrics(run_id, video_name))
    return {
        row["metric_name"]: {
            "status": row["status"],
            "value": row.get("value"),
            "details": row.get("details") or {},
        }
        for row in rows
    }


def _video_artifact_map(db: Database, run_id: int, video_name: str) -> dict[str, int]:
    return _video_artifact_map_from_rows(_video_artifacts_for_video(db, run_id, video_name))


def _video_artifact_map_from_rows(artifacts: list[dict]) -> dict[str, int]:
    result: dict[str, int] = {}
    for artifact in artifacts:
        result.setdefault(artifact["kind"], int(artifact["id"]))
    return result


def _video_artifact_tracks(db: Database, run_id: int, video_name: str) -> list[dict[str, object]]:
    return _video_artifact_tracks_from_rows(_video_artifacts_for_video(db, run_id, video_name))


def _video_artifact_tracks_from_rows(artifacts: list[dict]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for artifact in artifacts:
        artifact_id = int(artifact["id"])
        has_preview = _materialized_preview_path(artifact) is not None
        preview_url = (
            f"/api/files/{artifact_id}?variant=preview"
            if has_preview
            else f"/api/files/{artifact_id}"
        )
        result.append({
            "id": int(artifact["id"]),
            "kind": artifact["kind"],
            "url": preview_url,
            "preview_url": preview_url,
            "original_url": f"/api/files/{artifact_id}",
            "has_preview": has_preview,
            "track_label": (artifact.get("metadata") or {}).get("compare_track_label"),
            "track_run_id": (artifact.get("metadata") or {}).get("compare_track_run_id"),
        })
    return result


def _video_artifacts_for_video(db: Database, run_id: int, video_name: str) -> list[dict]:
    return [
        artifact
        for artifact in db.list_run_video_artifacts(run_id, video_name=video_name)
        if artifact["kind"] in VIDEO_TIMELINE_ARTIFACTS
    ]


def _compare_layer_payloads(db: Database, run: dict[str, object], sample: dict) -> list[dict[str, object]]:
    metadata = sample.get("metadata") or {}
    if metadata.get("source_type") != "compare":
        return []
    video_name = str(metadata.get("video_name") or metadata.get("compare_group") or "")
    frame_index = int(metadata.get("frame_index") or metadata.get("sample_index") or 0)
    track_rows = _compare_track_rows(run, metadata)
    requested_layers = _requested_compare_layers(run)
    if requested_layers is not None and not requested_layers:
        return []
    layers: list[dict[str, object]] = []
    for track in track_rows:
        source_run_id = track.get("track_run_id")
        if source_run_id in {None, ""}:
            continue
        track_label = str(track.get("track_label") or f"run-{source_run_id}")
        source_sample = db.find_sample_by_video_frame(int(source_run_id), video_name, frame_index)
        if source_sample is None:
            continue
        source_job_ids = db.run_inference_job_ids(int(source_run_id))
        if not source_job_ids:
            continue
        allowed_kinds = requested_layers.get(int(source_run_id), set()) if requested_layers is not None else None
        for artifact in db.list_artifacts_by_samples(
            [int(source_sample["id"])],
            job_ids=source_job_ids,
        ):
            kind = str(artifact["kind"])
            if kind not in COMPARE_LAYER_ARTIFACTS:
                continue
            if allowed_kinds is not None and kind not in allowed_kinds:
                continue
            layers.append(
                {
                    "kind": kind,
                    "group": _compare_layer_group(kind),
                    "track_label": track_label,
                    "track_run_id": int(source_run_id),
                    "source_sample_id": int(source_sample["id"]),
                    "artifact": _artifact_payload(artifact),
                }
            )
    return sorted(layers, key=lambda item: (str(item["group"]), str(item["kind"]), str(item["track_label"])))


def _compare_track_rows(run: dict[str, object], sample_metadata: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for track in (run.get("metadata") or {}).get("distorted_tracks") or []:
        rows.append(
            {
                "track_label": track.get("track_label") or track.get("label"),
                "track_run_id": track.get("track_run_id") or track.get("run_id"),
            }
        )
    if not rows and sample_metadata.get("compare_track_run_id") is not None:
        rows.append(
            {
                "track_label": sample_metadata.get("compare_track_label"),
                "track_run_id": sample_metadata.get("compare_track_run_id"),
            }
        )
    return rows


def _requested_compare_layers(run: dict[str, object]) -> dict[int, set[str]] | None:
    request = (run.get("metadata") or {}).get("request") or {}
    if "extra_layers" not in request or request.get("extra_layers") is None:
        return None
    result: dict[int, set[str]] = {}
    for layer in request.get("extra_layers") or []:
        if str(layer.get("source") or "run_artifact") != "run_artifact":
            continue
        run_id = layer.get("run_id")
        if run_id in {None, ""}:
            continue
        kinds = {str(kind) for kind in (layer.get("kinds") or []) if str(kind) in COMPARE_LAYER_ARTIFACTS}
        result[int(run_id)] = kinds
    return result


def _compare_layer_group(kind: str) -> str:
    if kind.startswith("flow"):
        return "flow"
    if kind.startswith("mask"):
        return "mask"
    return "warp"


def _find_timeline_video(videos: list[dict], video_name: str) -> dict:
    for video in videos:
        if video_name in {str(video.get("video_name")), str(video.get("video_file"))}:
            return video
    raise ValueError(f"video not found in run timeline: {video_name}")


def _first_metric_name(samples: list[dict]) -> str | None:
    for sample in samples:
        names = sorted((sample.get("metrics") or {}).keys())
        if names:
            return names[0]
    return None


def _timeline_buckets(samples: list[dict], metric_name: str | None, bucket_count: int) -> list[dict]:
    if not samples:
        return []
    bucket_count = min(bucket_count, len(samples))
    buckets = []
    for bucket_index in range(bucket_count):
        start = bucket_index * len(samples) // bucket_count
        end = (bucket_index + 1) * len(samples) // bucket_count
        rows = samples[start:end]
        values = []
        status_count = {status: 0 for status in METRIC_STATUSES}
        worst_sample_id = None
        worst_value = None
        for sample in rows:
            metric = (sample.get("metrics") or {}).get(metric_name) if metric_name else None
            status = metric.get("status") if metric else "missing"
            status_count[status] = status_count.get(status, 0) + 1
            value = metric.get("value") if metric else None
            if status == "completed" and value is not None:
                numeric = float(value)
                values.append(numeric)
                if _is_worse(str(metric_name), numeric, worst_value):
                    worst_value = numeric
                    worst_sample_id = sample.get("sample_id")
        buckets.append(
            {
                "bucket_index": bucket_index,
                "start_index": start,
                "end_index": max(start, end - 1),
                "frame_start": rows[0]["frame_index"],
                "frame_end": rows[-1]["frame_index"],
                "count": len(rows),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": (sum(values) / len(values)) if values else None,
                "status_count": status_count,
                "worst_sample_id": worst_sample_id,
                "worst_value": worst_value,
            }
        )
    return buckets


def _materialized_preview_path(artifact: dict) -> Path | None:
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        try:
            metadata = json.loads(artifact.get("metadata_json") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
    path_value = metadata.get("preview_path")
    if not path_value:
        return None
    try:
        path = Path(str(path_value)).resolve()
        if path.is_file() and path.stat().st_size > 0:
            return path
    except OSError:
        pass
    return None


def _artifact_payload(artifact: dict) -> dict[str, object]:
    artifact_id = int(artifact["id"])
    has_preview = _materialized_preview_path(artifact) is not None
    return {
        "id": artifact_id,
        "original": artifact_id,
        "original_url": f"/api/files/{artifact_id}",
        "preview_url": f"/api/files/{artifact_id}?variant=preview" if has_preview else f"/api/files/{artifact_id}",
        "has_preview": has_preview,
    }


def _run_sample_payload(db: Database, run_id: int, sample_id: int) -> dict:
    run = db.get_run(run_id)
    sample = db.get_sample(sample_id)
    if sample is None or int(sample["dataset_id"]) != int(run["dataset_id"]):
        raise ValueError("sample does not belong to this run")

    artifacts: dict[str, object] = {}
    extra_artifacts: list[dict[str, object]] = []
    job_ids = set(db.run_inference_job_ids(run_id))
    for artifact in db.list_artifacts_by_sample(sample_id):
        if int(artifact["job_id"]) not in job_ids:
            continue
        kind = artifact["kind"]
        if kind in CORE_TIMELINE_ARTIFACTS:
            artifacts[kind] = _artifact_payload(artifact)
        elif kind.startswith("extra_"):
            extra_artifacts.append({"id": int(artifact["id"]), "kind": kind, **_artifact_payload(artifact)})

    metric_rows = _latest_metric_rows(
        [row for row in db.list_metrics_by_sample(sample_id) if int(row["inference_job_id"]) in job_ids]
    )
    actual_metrics = {
        row["metric_name"]: {
            "status": row["status"],
            "value": row.get("value"),
            "details": row.get("details") or {},
        }
        for row in metric_rows
    }
    metrics = _sample_metrics_with_defaults(
        actual_metrics,
        _requested_sample_metrics(run),
        bool(sample.get("gt_path")),
        str(run.get("status") or ""),
        _metric_job_status(db, run),
    )
    metric_status = _metric_status_counts(metrics)

    metadata = sample.get("metadata") or {}
    timestamps = metadata.get("timestamps") or {}
    return {
        "sample_id": sample_id,
        "sample_name": sample["name"],
        "frame_index": int(metadata.get("frame_index") or metadata.get("sample_index") or 0),
        "sample_index": int(metadata.get("sample_index") or 0),
        "timestamp": timestamps.get("gt") if isinstance(timestamps, dict) else None,
        "metadata": metadata,
        "track_label": metadata.get("compare_track_label"),
        "track_index": metadata.get("compare_track_index"),
        "artifacts": artifacts,
        "extra_artifacts": extra_artifacts,
        "compare_layers": _compare_layer_payloads(db, run, sample),
        "sample_files": {
            "img0": f"/api/sample-files/{sample_id}/img0",
            "img1": f"/api/sample-files/{sample_id}/img1",
            "gt": f"/api/sample-files/{sample_id}/gt" if sample.get("gt_path") else None,
        },
        "metrics": metrics,
        "metric_status": metric_status,
    }


def _run_metric_summary(db: Database, run_id: int) -> dict:
    run = db.get_run(run_id)
    requested_metrics = list(run.get("metrics") or [])
    metrics = _latest_metric_rows(db.list_run_metrics(run_id))
    samples = db.list_samples(int(run["dataset_id"]))
    metric_job_status = _metric_job_status(db, run)
    requested_video_metrics = _requested_video_metrics(run)
    actual_sample_metric_keys = {
        (int(row["sample_id"]), row["metric_name"])
        for row in metrics
        if row.get("sample_id") is not None
    }
    actual_video_metric_keys = {
        (str((row.get("details") or {}).get("video_name") or "video"), row["metric_name"])
        for row in metrics
        if row.get("sample_id") is None
    }
    summary: dict[str, dict[str, object]] = {
        name: _empty_metric_summary(name)
        for name in requested_metrics
    }
    for metric in metrics:
        name = metric["metric_name"]
        row = summary.setdefault(name, _empty_metric_summary(name))
        status = metric["status"]
        row[status] = int(row.get(status, 0)) + 1
        value = metric.get("value")
        sample_id = metric.get("sample_id")
        if status == "completed" and value is not None:
            values = row.setdefault("_values", [])
            values.append(float(value))
            if sample_id is not None and _is_worse(name, float(value), row.get("worst_value")):
                row["worst_value"] = float(value)
                row["worst_sample_id"] = int(sample_id)
        elif status in {"unavailable", "failed", "skipped"}:
            reasons = row.setdefault("reasons", [])
            details = metric.get("details") or {}
            reason = details.get("reason") or details.get("type") or status
            if reason not in reasons:
                reasons.append(reason)
    for name in _requested_sample_metrics(run):
        row = summary.setdefault(name, _empty_metric_summary(name))
        for sample in samples:
            sample_id = int(sample["id"])
            if (sample_id, name) in actual_sample_metric_keys:
                continue
            default_metric = _default_sample_metric_payload(
                has_gt=bool(sample.get("gt_path")),
                run_status=str(run.get("status") or ""),
                metric_job_status=metric_job_status,
            )
            status = default_metric["status"]
            row[status] = int(row.get(status, 0)) + 1
            if status in {"failed", "skipped", "missing"}:
                reasons = row.setdefault("reasons", [])
                reason = (default_metric.get("details") or {}).get("reason") or status
                if reason not in reasons:
                    reasons.append(reason)
    for name in requested_video_metrics:
        row = summary.setdefault(name, _empty_metric_summary(name))
        for video_name in _video_metric_target_names(samples):
            if (video_name, name) in actual_video_metric_keys:
                continue
            default_metric = _default_video_metric_payload(
                run_status=str(run.get("status") or ""),
                metric_job_status=metric_job_status,
            )
            status = default_metric["status"]
            row[status] = int(row.get(status, 0)) + 1
            if status in {"failed", "skipped", "missing"}:
                reasons = row.setdefault("reasons", [])
                reason = (default_metric.get("details") or {}).get("reason") or status
                if reason not in reasons:
                    reasons.append(reason)
    for row in summary.values():
        values = list(row.pop("_values", []))
        if values:
            row["mean"] = sum(values) / len(values)
            row["min"] = min(values)
            row["max"] = max(values)
        else:
            row["mean"] = None
            row["min"] = None
            row["max"] = None
    return {"run_id": run_id, "metrics": summary}


def _empty_metric_summary(name: str) -> dict[str, object]:
    return {
        "metric_name": name,
        "direction": METRIC_DIRECTIONS.get(name, "lower_is_better"),
        "pending": 0,
        "running": 0,
        "completed": 0,
        "unavailable": 0,
        "failed": 0,
        "skipped": 0,
        "missing": 0,
        "mean": None,
        "min": None,
        "max": None,
        "worst_sample_id": None,
        "worst_value": None,
        "reasons": [],
    }


def _latest_metric_rows(metrics: list[dict]) -> list[dict]:
    latest: dict[tuple[object, ...], dict] = {}
    for row in metrics:
        details = row.get("details") or {}
        sample_id = row.get("sample_id")
        if sample_id is not None:
            key = ("sample", int(sample_id), row["metric_name"])
        else:
            track_identity = (
                details.get("compare_track_label")
                or details.get("compare_track_key")
                or details.get("compare_track_run_id")
            )
            key = (
                "video",
                row["metric_name"],
                details.get("video_name"),
                track_identity,
            )
        current = latest.get(key)
        if current is None or int(row.get("id") or 0) > int(current.get("id") or 0):
            latest[key] = row
    return list(latest.values())


def _is_worse(metric_name: str, value: float, current: object) -> bool:
    if current is None:
        return True
    current_value = float(current)
    if METRIC_DIRECTIONS.get(metric_name) == "higher_is_better":
        return value < current_value
    return value > current_value


def _metric_summary_for_video(samples: list[dict[str, object]], global_summary: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    metric_names = set(global_summary)
    for sample in samples:
        metric_names.update((sample.get("metrics") or {}).keys())
    summary = {name: _empty_metric_summary(name) for name in sorted(metric_names)}
    for sample in samples:
        sample_id = int(sample["sample_id"])
        for name, metric in (sample.get("metrics") or {}).items():
            row = summary.setdefault(name, _empty_metric_summary(name))
            status = metric.get("status")
            row[status] = int(row.get(status, 0)) + 1
            value = metric.get("value")
            if status == "completed" and value is not None:
                values = row.setdefault("_values", [])
                values.append(float(value))
                if _is_worse(name, float(value), row.get("worst_value")):
                    row["worst_value"] = float(value)
                    row["worst_sample_id"] = sample_id
            elif status in {"unavailable", "failed", "skipped"}:
                reasons = row.setdefault("reasons", [])
                details = metric.get("details") or {}
                reason = details.get("reason") or details.get("type") or status
                if reason not in reasons:
                    reasons.append(reason)
    for row in summary.values():
        values = list(row.pop("_values", []))
        row["mean"] = sum(values) / len(values) if values else None
        row["min"] = min(values) if values else None
        row["max"] = max(values) if values else None
    return summary


def _requested_sample_metrics(run: dict[str, object]) -> list[str]:
    return [
        name
        for name in list(run.get("metrics") or [])
        if METRIC_TIMELINE_SUPPORT.get(str(name), False)
    ]


def _requested_video_metrics(run: dict[str, object]) -> list[str]:
    return [
        name
        for name in list(run.get("metrics") or [])
        if not METRIC_TIMELINE_SUPPORT.get(str(name), False)
    ]


def _metric_job_status(db: Database, run: dict[str, object]) -> str | None:
    try:
        from vfieval.pipeline.metric_jobs import metric_wave_status

        return metric_wave_status(db, run)
    except Exception:
        return None


def _default_sample_metric_payload(
    has_gt: bool,
    run_status: str,
    metric_job_status: str | None,
) -> dict[str, object]:
    if not has_gt:
        return {"status": "skipped", "value": None, "details": {"reason": "sample has no ground-truth reference"}}
    if metric_job_status == "running" or run_status == "metric_running":
        return {"status": "running", "value": None, "details": {"reason": "metric evaluation is running"}}
    if metric_job_status == "queued" or run_status in {"queued", "running", "metric_queued"}:
        return {"status": "pending", "value": None, "details": {"reason": "metric evaluation has not started"}}
    return {"status": "missing", "value": None, "details": {"reason": "metric result is not available"}}


def _default_video_metric_payload(
    run_status: str,
    metric_job_status: str | None,
) -> dict[str, object]:
    if metric_job_status == "running" or run_status == "metric_running":
        return {"status": "running", "value": None, "details": {"reason": "video-level metric evaluation is running"}}
    if metric_job_status == "queued" or run_status in {"queued", "running", "metric_queued"}:
        return {"status": "pending", "value": None, "details": {"reason": "video-level metric evaluation has not started"}}
    return {"status": "missing", "value": None, "details": {"reason": "video-level metric result is not available"}}


def _sample_metrics_with_defaults(
    actual_metrics: dict[str, dict[str, object]],
    requested_sample_metrics: list[str],
    has_gt: bool,
    run_status: str,
    metric_job_status: str | None,
) -> dict[str, dict[str, object]]:
    metrics = dict(actual_metrics)
    for name in requested_sample_metrics:
        if name not in metrics:
            metrics[name] = _default_sample_metric_payload(has_gt, run_status, metric_job_status)
    return metrics


def _video_metrics_with_defaults(
    actual_metrics: dict[str, dict[str, object]],
    requested_video_metrics: list[str],
    run_status: str,
    metric_job_status: str | None,
) -> dict[str, dict[str, object]]:
    metrics = dict(actual_metrics)
    for name in requested_video_metrics:
        if name not in metrics:
            metrics[name] = _default_video_metric_payload(run_status, metric_job_status)
    return metrics


def _metric_status_counts(metrics: dict[str, dict[str, object]]) -> dict[str, int]:
    counts = {status: 0 for status in METRIC_STATUSES}
    for metric in metrics.values():
        status = str(metric.get("status") or "missing")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _worst_samples_for_video(samples: list[dict[str, object]], limit: int = 8) -> dict[str, list[dict[str, object]]]:
    by_metric: dict[str, list[dict[str, object]]] = {}
    for sample in samples:
        for name, metric in (sample.get("metrics") or {}).items():
            if metric.get("status") != "completed" or metric.get("value") is None:
                continue
            by_metric.setdefault(name, []).append(
                {
                    "sample_id": sample["sample_id"],
                    "sample_name": sample["sample_name"],
                    "frame_index": sample["frame_index"],
                    "timestamp": sample.get("timestamp"),
                    "value": float(metric["value"]),
                    "status": metric["status"],
                    "reason": (metric.get("details") or {}).get("reason"),
                }
            )
    result = {}
    for name, rows in by_metric.items():
        reverse = METRIC_DIRECTIONS.get(name) != "higher_is_better"
        result[name] = sorted(rows, key=lambda item: float(item["value"]), reverse=reverse)[:limit]
    return result


def _video_metric_target_names(samples: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        metadata = sample.get("metadata") or {}
        video_name = str(metadata.get("video_name") or metadata.get("video_file") or "frames")
        if video_name not in seen:
            seen.add(video_name)
            names.append(video_name)
    return names


def _retry_run_metrics(db: Database, run_id: int) -> dict:
    run = db.get_run(run_id)
    inference_job_ids = db.run_inference_job_ids(run_id)
    if not inference_job_ids:
        raise ValueError("Run has no inference job")
    metric_rows = db.list_run_metrics(run_id)
    failed_names = sorted(
        {
            row["metric_name"]
            for row in _latest_metric_rows(metric_rows)
            if row["status"] in {"failed", "unavailable"}
        }
    )
    if not failed_names and not metric_rows:
        failed_names = list(run.get("metrics") or [])
    if not failed_names:
        raise ValueError("Run has no failed or unavailable metrics to retry")
    from vfieval.pipeline.metric_jobs import create_metric_wave

    wave = create_metric_wave(db, run_id, failed_names, source="retry", retry=True)
    wake_job_supervisor(db)
    return wave


def _dashboard(db: Database) -> dict:
    runs = db.list_runs(limit=500)
    workers = db.list_workers()
    active_statuses = {"queued", "running", "finalize_queued", "finalizing", "metric_queued", "metric_running"}
    now = time.time()
    healthy_workers = [worker for worker in workers if now - float(worker["last_seen_at"]) < 120.0]
    metric_unavailable = 0
    for run in runs:
        for summary in run.get("metric_summary", {}).values():
            metric_unavailable += int(summary.get("unavailable", 0))

    completed = [run for run in runs if run["status"] == "completed"]
    recent_model_fps = [
        float(run.get("result", {}).get("model_fps", 0.0))
        for run in completed[:20]
        if run.get("result", {}).get("model_fps") is not None
    ]
    return {
        "active_runs": sum(1 for run in runs if run["status"] in active_statuses),
        "failed_runs": sum(1 for run in runs if run["status"] == "failed"),
        "completed_runs": len(completed),
        "workers": len(workers),
        "healthy_workers": len(healthy_workers),
        "metric_unavailable": metric_unavailable,
        "recent_model_fps": sum(recent_model_fps) / len(recent_model_fps) if recent_model_fps else None,
    }


def _run_compare_payload(db: Database, run_id: int) -> dict:
    run = db.get_run(run_id)
    metrics = db.list_run_metrics(run_id)
    by_metric: dict[str, list[float]] = {}
    status_counts: dict[str, int] = {}
    for row in metrics:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        if row["status"] == "completed" and row["value"] is not None:
            by_metric.setdefault(row["metric_name"], []).append(float(row["value"]))
    aggregate = {
        metric: sum(values) / len(values)
        for metric, values in by_metric.items()
        if values
    }
    return {
        "run": run,
        "compare_key": _compare_key(run),
        "metrics": aggregate,
        "metric_status_counts": status_counts,
    }


def _compare_key(run: dict) -> dict:
    request = (run.get("metadata") or {}).get("request") or {}
    return {
        "video_group": request.get("video_group") or (run.get("metadata") or {}).get("video_group"),
        "selected_videos": sorted(request.get("selected_videos") or (run.get("metadata") or {}).get("selected_videos") or []),
        "frame_step": int(request.get("frame_step") or 1),
        "max_frames": request.get("max_frames"),
        "height": int(run.get("height") or 0),
        "width": int(run.get("width") or 0),
        "has_gt": True,
    }


def _optional_int(value) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _query_int_values(query: dict[str, list[str]], key: str) -> list[int]:
    return [
        int(part)
        for raw in query.get(key, [])
        for part in str(raw).split(",")
        if part.strip()
    ]


def _purge_response(request: dict[str, Any]) -> dict[str, Any]:
    report = dict(request.get("report") or {})
    completed = str(request.get("status") or "") == "completed"
    delete_run = str(request.get("request_type") or "") == "delete_run"
    return {
        **report,
        "run_id": int(request["run_id"]),
        "request_id": int(request["id"]),
        "purge_status": request.get("status"),
        "deleting": not completed and delete_run,
        "deleted": completed and delete_run,
        "artifact_cleaned": completed and bool(report.get("artifact_cleaned")),
        "request": request,
    }


def _evaluation_campaign_v2_payload(db: Database, campaign_id: int) -> dict[str, Any]:
    campaign = get_campaign_v2(db, int(campaign_id))
    preparation_row = get_preparation_v2(db, int(campaign_id))
    preparation: dict[str, Any] = {}
    if preparation_row is not None:
        report = dict(preparation_row.get("report") or {})
        error = dict(preparation_row.get("error") or {})
        state = str(preparation_row.get("state") or "")
        total = int(report.get("total") or campaign.get("item_count") or 0)
        current_default = total if state == "completed" else 0
        preparation = {
            "state": state,
            "phase": report.get("phase") or state,
            "current": int(report.get("current") or current_default),
            "total": total,
            "attempt_count": int(preparation_row.get("attempt_count") or 0),
            "error": error,
            "report": report,
            "updated_at": preparation_row.get("updated_at"),
            "completed_at": preparation_row.get("completed_at"),
        }
        for optional_field in (
            "stage",
            "item_index",
            "item_name",
            "frame_current",
            "frame_total",
            "overall_fraction",
            "pipeline",
            "timings",
        ):
            if optional_field in report:
                preparation[optional_field] = report[optional_field]
        campaign["preparation_status"] = state
        if error:
            campaign["preparation_error"] = error
    analysis = None
    if campaign.get("status") in {"published", "closed", "archived"}:
        analysis = campaign_analysis_v2(db, int(campaign_id), bootstrap_samples=200)
    return {
        "campaign": campaign,
        "preparation": preparation,
        "coverage": {
            "items": int(campaign.get("item_count") or 0),
            "tasks": int(campaign.get("task_count") or 0),
            "votes": int(campaign.get("vote_count") or 0),
        },
        "analysis": analysis,
        "share_url": campaign.get("share_url"),
    }


def _legacy_evaluation_campaign_payload(db: Database, campaign_id: int) -> dict[str, Any]:
    campaign = get_campaign(db, int(campaign_id))
    metadata = dict(campaign.get("metadata") or {})
    campaign.update(
        {
            "schema_version": 1,
            "campaign_key": f"v1:{int(campaign_id)}",
            "public_title": metadata.get("public_title") or campaign.get("name"),
            "archived": bool(metadata.get("archived_at")),
            "read_only": True,
            "item_count": int(campaign.get("tasks") or 0),
            "task_count": int(campaign.get("tasks") or 0),
            "vote_count": int(campaign.get("votes") or 0),
        }
    )
    return campaign


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
