from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from vfieval.config import WorkspaceConfig
from vfieval.datasets import scan_dataset
from vfieval.db import Database
from vfieval.file_inputs import (
    DECODE_STRATEGY_VERSION,
    inspect_video,
    list_checkpoints,
    list_video_group_videos,
    list_model_files,
    list_video_groups,
    models_dir,
    normalize_device_precision,
    preflight_run,
    resolve_checkpoint,
    resolve_model_file,
    resolve_run_dimensions,
    resolve_video_group,
    thumbnail_path,
    videos_dir,
)
from vfieval.metrics import METRIC_NAMES
from vfieval.metrics.health import metrics_health
from vfieval.worker import WorkerOptions, detect_capabilities, run_worker


TERMINAL_RUN_STATUSES = {"completed", "failed", "canceled"}
COMPARE_SOURCE_RUN_STATUSES = {"completed", "metric_queued", "metric_running"}


def run_server(db: Database, workspace: WorkspaceConfig, host: str, port: int) -> None:
    handler = _make_handler(db, workspace)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"VFIEval listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def _make_handler(db: Database, workspace: WorkspaceConfig):
    class VFIEvalHandler(BaseHTTPRequestHandler):
        server_version = "VFIEval/0.1"

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)
                if path == "/":
                    return self._send_static("index.html")
                if path in {"/app.js", "/styles.css"}:
                    return self._send_static(path.lstrip("/"))
                if path == "/api/health":
                    return self._json({"ok": True, "metrics": list(METRIC_NAMES)})
                if path == "/api/dashboard":
                    return self._json(_dashboard(db))
                if path == "/api/model-files":
                    return self._json(list_model_files(workspace))
                if path == "/api/checkpoints":
                    return self._json(list_checkpoints(workspace, query.get("model_file", [None])[0]))
                if path == "/api/devices":
                    return self._json(detect_capabilities())
                if path == "/api/video-groups":
                    return self._json(list_video_groups(workspace))
                match = re.fullmatch(r"/api/video-groups/([^/]+)/videos", path)
                if match:
                    frame_step = max(1, int(query.get("frame_step", ["1"])[0] or 1))
                    max_frames = _optional_int(query.get("max_frames", [None])[0])
                    return self._json(
                        list_video_group_videos(
                            workspace,
                            unquote(match.group(1)),
                            frame_step,
                            max_frames,
                            page=int(query.get("page", ["1"])[0] or 1),
                            page_size=int(query.get("page_size", ["50"])[0] or 50),
                            query=query.get("q", [""])[0],
                            sort=query.get("sort", ["name"])[0],
                        )
                    )
                match = re.fullmatch(r"/api/video-thumbnails/([a-f0-9]{64})", path)
                if match:
                    return self._send_file(thumbnail_path(workspace, match.group(1)))
                if path == "/api/metrics/health":
                    return self._json(metrics_health(workspace))
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
                    return self._json(
                        db.list_runs(
                            limit=int(query.get("limit", ["100"])[0]),
                            include_deleted=query.get("include_deleted", ["0"])[0] in {"1", "true", "yes"},
                        )
                    )
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
            except Exception as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), type(exc).__name__)

        def do_POST(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                body = self._read_json()
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
                    worker_id = body["worker_id"]
                    role = body.get("role", "remote")
                    db.register_worker(worker_id, role, body.get("capabilities") or {})
                    return self._json({"worker_id": worker_id, "worker": db.get_worker(worker_id)})
                if path == "/api/preflight":
                    return self._json(preflight_run(db, workspace, body))
                if path == "/api/runs":
                    run_type = str(body.get("run_type") or "")
                    if run_type == "video_compare" or body.get("reference") or body.get("distorted"):
                        created = _create_video_compare_run(db, workspace, body)
                        return self._json(created, status=HTTPStatus.CREATED)
                    if body.get("model_file") or body.get("video_group"):
                        created = _create_run_from_files(db, workspace, body)
                        return self._json(created, status=HTTPStatus.CREATED)
                    metrics = list(body.get("metrics") or [])
                    unsupported = [name for name in metrics if name not in METRIC_NAMES]
                    if unsupported:
                        return self._error(HTTPStatus.BAD_REQUEST, f"unsupported metrics: {', '.join(unsupported)}")
                    model_id = int(body["model_id"])
                    dataset_id = int(body["dataset_id"])
                    default_name = f"model-{model_id}-dataset-{dataset_id}"
                    run_id = db.create_run(
                        name=body.get("name") or default_name,
                        experiment_id=_optional_int(body.get("experiment_id")),
                        model_id=model_id,
                        dataset_id=dataset_id,
                        height=int(body["height"]),
                        width=int(body["width"]),
                        batch_size=int(body.get("batch_size", 1)),
                        device=body.get("device", "auto"),
                        precision=body.get("precision", "fp32"),
                        metrics=metrics,
                        metadata=body.get("metadata") or {},
                    )
                    return self._json({"run_id": run_id, "run": db.get_run(run_id)}, status=HTTPStatus.CREATED)
                match = re.fullmatch(r"/api/datasets/(\d+)/scan", path)
                if match:
                    dataset_id = int(match.group(1))
                    samples = scan_dataset(db, workspace, dataset_id)
                    return self._json({"dataset_id": dataset_id, "samples": samples})
                match = re.fullmatch(r"/api/runs/(\d+)/(cancel|retry)", path)
                if match:
                    run_id = int(match.group(1))
                    action = match.group(2)
                    if action == "cancel":
                        db.request_run_cancel(run_id)
                        return self._json({"run_id": run_id, "run": db.get_run(run_id)})
                    retry = _retry_run(db, workspace, run_id)
                    return self._json(retry, status=HTTPStatus.CREATED)
                match = re.fullmatch(r"/api/runs/(\d+)/(hide|cleanup-artifacts)", path)
                if match:
                    run_id = int(match.group(1))
                    action = match.group(2)
                    if action == "hide":
                        if db.get_run(run_id)["status"] not in TERMINAL_RUN_STATUSES:
                            db.request_run_cancel(run_id)
                        db.soft_delete_run(run_id)
                        return self._json({"run_id": run_id, "deleted": True})
                    try:
                        cleaned = _cleanup_run_artifacts(db, workspace, run_id)
                    except ValueError as exc:
                        return self._error(HTTPStatus.BAD_REQUEST, str(exc), type(exc).__name__)
                    return self._json(cleaned)
                match = re.fullmatch(r"/api/runs/(\d+)/metrics/retry", path)
                if match:
                    run_id = int(match.group(1))
                    retry = _retry_run_metrics(db, run_id)
                    return self._json(retry, status=HTTPStatus.CREATED)
                if path == "/api/jobs":
                    kind = body["kind"]
                    if kind not in {"inference", "metric"}:
                        return self._error(HTTPStatus.BAD_REQUEST, "kind must be inference or metric")
                    job_id = db.create_job(kind, body.get("payload") or {})
                    return self._json({"job_id": job_id, "kind": kind}, status=HTTPStatus.CREATED)
                if path == "/api/jobs/claim":
                    worker_id = body["worker_id"]
                    role = body.get("role", "remote")
                    kinds = list(body.get("kinds") or [])
                    db.register_worker(worker_id, role, body.get("capabilities") or {})
                    job = db.claim_next_job(worker_id, kinds)
                    return self._json({"job": job})
                match = re.fullmatch(r"/api/jobs/(\d+)/(complete|fail|progress)", path)
                if match:
                    job_id = int(match.group(1))
                    action = match.group(2)
                    if action == "complete":
                        db.complete_job(job_id, body.get("result") or {})
                    elif action == "fail":
                        db.fail_job(job_id, body.get("error") or {})
                    else:
                        db.update_job_progress(job_id, int(body.get("current", 0)), body.get("total"))
                    return self._json({"job_id": job_id, "status": action})
                match = re.fullmatch(r"/api/jobs/(\d+)/heartbeat", path)
                if match:
                    job_id = int(match.group(1))
                    worker_id = body.get("worker_id")
                    if worker_id:
                        db.touch_worker(str(worker_id), body.get("capabilities") or None)
                    db.touch_job(job_id)
                    return self._json({"job_id": job_id, "status": "heartbeat"})
                self._error(HTTPStatus.NOT_FOUND, "not found")
            except KeyError as exc:
                self._error(HTTPStatus.BAD_REQUEST, f"missing field {exc}")
            except Exception as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), type(exc).__name__)

        def do_DELETE(self) -> None:
            try:
                parsed = urlparse(self.path)
                match = re.fullmatch(r"/api/runs/(\d+)", parsed.path)
                if not match:
                    return self._error(HTTPStatus.NOT_FOUND, "not found")
                run_id = int(match.group(1))
                if db.get_run(run_id)["status"] not in TERMINAL_RUN_STATUSES:
                    db.request_run_cancel(run_id)
                db.soft_delete_run(run_id)
                return self._json({"run_id": run_id, "deleted": True})
            except Exception as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), type(exc).__name__)

        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _json(self, data, status: HTTPStatus = HTTPStatus.OK, headers: dict[str, str] | None = None) -> None:
            payload = json.dumps(data, indent=2, default=str).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, status: HTTPStatus, message: str, error_type: str = "Error") -> None:
            self._json({"error": {"type": error_type, "message": message}}, status=status)

        def _send_static(self, name: str) -> None:
            path = Path(__file__).parent / "web" / name
            if not path.exists():
                return self._error(HTTPStatus.NOT_FOUND, f"static file {name} not found")
            self._send_file(path, cache_control="no-store")

        def _send_artifact(self, artifact_id: int, variant: str = "original") -> None:
            artifact = db.get("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
            if artifact is None:
                return self._error(HTTPStatus.NOT_FOUND, "artifact not found")
            metadata = json.loads(artifact.get("metadata_json") or "{}")
            if variant == "preview" and metadata.get("preview_path"):
                path = Path(metadata["preview_path"]).resolve()
            else:
                path = Path(artifact["path"]).resolve()
            workspace_root = workspace.root.resolve()
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
                self.end_headers()
                return
            if byte_range is None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                if cache_control:
                    self.send_header("Cache-Control", cache_control)
                self.end_headers()
                with path.open("rb") as handle:
                    while True:
                        chunk = handle.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            else:
                start, end = byte_range
                content_length = end - start + 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(content_length))
                self.send_header("Accept-Ranges", "bytes")
                if cache_control:
                    self.send_header("Cache-Control", cache_control)
                self.end_headers()
                with path.open("rb") as handle:
                    handle.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk = handle.read(min(remaining, 4 * 1024 * 1024))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
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
        return {"sources": _compare_gt_sources(workspace)}
    if source_type == "pred":
        run_id = _optional_int(query.get("run_id", [None])[0])
        return {"sources": _compare_pred_sources(db, workspace, run_id)}
    if source_type in {"flow", "mask"}:
        run_id = _optional_int(query.get("run_id", [None])[0])
        if run_id is None:
            raise ValueError(f"/api/compare-sources/{source_type} requires run_id")
        video = query.get("video", [None])[0]
        return {"sources": _compare_layer_sources(db, run_id, source_type, video)}
    raise ValueError(f"unsupported compare source type: {source_type}")


def _compare_gt_sources(workspace: WorkspaceConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    root = videos_dir(workspace)
    for group in list_video_groups(workspace):
        group_name = str(group["name"])
        for video in group.get("videos") or []:
            path = (root / group_name / str(video["name"])).resolve()
            rows.append(
                {
                    "kind": "video_group",
                    "group": group_name,
                    "video": video["name"],
                    "video_name": Path(str(video["name"])).stem,
                    "path": str(path),
                    "frame_count": int(video.get("frame_count") or 0),
                    "width": int(video.get("width") or 0),
                    "height": int(video.get("height") or 0),
                    "fps": video.get("fps"),
                    "duration_seconds": video.get("duration_seconds"),
                    "decodable": bool(video.get("decodable")),
                    "thumbnail_url": video.get("thumbnail_url"),
                }
            )
    return rows


def _compare_pred_sources(db: Database, workspace: WorkspaceConfig, run_id: int | None = None) -> list[dict[str, object]]:
    runs = [db.get_run(run_id)] if run_id is not None else db.list_runs(limit=10000)
    rows: list[dict[str, object]] = []
    for run in runs:
        if run.get("deleted_at") is not None or run.get("artifact_cleaned_at") is not None:
            continue
        if str(run.get("status") or "") not in COMPARE_SOURCE_RUN_STATUSES:
            continue
        for artifact in db.list_run_artifacts(int(run["id"]), kind="pred_video"):
            metadata = artifact.get("metadata") or {}
            path = Path(str(artifact["path"])).resolve()
            row = {
                "kind": "run_artifact",
                "run_id": int(run["id"]),
                "run_name": run.get("name"),
                "video": metadata.get("video_name") or path.stem,
                "artifact_id": int(artifact["id"]),
                "frame_count": int(metadata.get("frames") or 0),
                "width": _optional_int(metadata.get("width")),
                "height": _optional_int(metadata.get("height")),
                "fps": metadata.get("fps"),
                "created_at": artifact.get("created_at"),
                "compare_track_label": metadata.get("compare_track_label"),
                "compare_track_run_id": metadata.get("compare_track_run_id"),
            }
            if (not row["frame_count"] or not row["width"] or not row["height"]) and path.exists():
                try:
                    info = inspect_video(path, workspace, exact=True)
                    row.update(
                        {
                            "frame_count": int(row["frame_count"] or info.get("frame_count") or 0),
                            "width": int(row["width"] or info.get("width") or 0),
                            "height": int(row["height"] or info.get("height") or 0),
                            "fps": row["fps"] or info.get("fps"),
                        }
                    )
                except Exception:
                    pass
            rows.append(row)
    return sorted(rows, key=lambda item: (str(item.get("video") or ""), str(item.get("run_name") or ""), int(item["artifact_id"])))


def _compare_layer_sources(db: Database, run_id: int, source_type: str, video_name: str | None = None) -> list[dict[str, object]]:
    run = db.get_run(run_id)
    if run.get("deleted_at") is not None or run.get("artifact_cleaned_at") is not None:
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


def _create_run_from_files(db: Database, workspace: WorkspaceConfig, body: dict) -> dict:
    preflight = preflight_run(db, workspace, body)
    if not preflight["ok"]:
        raise ValueError(_preflight_error_message(preflight))
    metrics = list(body.get("metrics") or [])
    unsupported = [name for name in metrics if name not in METRIC_NAMES]
    if unsupported:
        raise ValueError(f"unsupported metrics: {', '.join(unsupported)}")

    model_path = resolve_model_file(workspace, str(body["model_file"]))
    video_folder = resolve_video_group(workspace, str(body["video_group"]))
    frame_step = max(1, int(body.get("frame_step") or 1))
    max_frames = _optional_int(body.get("max_frames"))
    video_infos = preflight.get("video_group", {}).get("videos", [])
    selected_videos = [str(name) for name in preflight.get("video_group", {}).get("selected_videos", [])]
    height, width = resolve_run_dimensions(body, video_infos)
    execution_mode = str(body.get("execution_mode") or "single")
    devices = _resolve_execution_devices(body, execution_mode)
    requested_device = str(body.get("device") or "auto")
    is_multi = execution_mode in {"multi_cuda", "multi_npu"}
    device, precision = normalize_device_precision(devices[0] if is_multi else requested_device, str(body.get("precision") or "auto"))
    checkpoint_path = resolve_checkpoint(workspace, body.get("checkpoint"), model_path.name)
    checkpoint_relative = _checkpoint_relative(workspace, checkpoint_path)
    selection_hash = _selection_hash(selected_videos, frame_step, max_frames)
    model_record_name = model_path.name if checkpoint_relative is None else f"{model_path.name} [{checkpoint_relative}]"
    reference_config = _reference_config(
        video_group=video_folder.name,
        selected_videos=selected_videos,
        frame_step=frame_step,
        max_frames=max_frames,
        resolution_mode=str(body.get("resolution_mode") or "original"),
        height=height,
        width=width,
    )
    reference_key = _reference_key(reference_config)

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
        },
    )
    dataset_id = db.upsert_dataset(
        name=f"video:{video_folder.name}:{selection_hash}",
        root_path=str(video_folder),
        has_gt=True,
        source_type="video",
        decode_mode="video_gt_triplets",
        metadata={
            "source": "folder",
            "video_group": video_folder.name,
            "frame_step": frame_step,
            "max_frames": max_frames,
            "video_glob": "*",
            "selected_videos": selected_videos,
        },
    )
    samples = scan_dataset(db, workspace, dataset_id)
    if samples <= 0:
        raise ValueError("视频集没有生成可推理 triplets")

    metadata = {
        "run_type": "model_inference",
        "source": "folder_flow",
        "request": {
            "run_type": "model_inference",
            "model_file": model_path.name,
            "video_group": video_folder.name,
            "resolution_mode": body.get("resolution_mode") or "original",
            "height": height,
            "width": width,
            "batch_size": int(body.get("batch_size") or 1),
            "batch_size_per_device": int(body.get("batch_size_per_device") or body.get("batch_size") or 1),
            "device": body.get("device") or "auto",
            "devices": devices,
            "execution_mode": execution_mode,
            "precision": body.get("precision") or "auto",
            "checkpoint": body.get("checkpoint") or "none",
            "frame_step": frame_step,
            "max_frames": max_frames,
            "selected_videos": selected_videos,
            "metrics": metrics,
        },
        "model_file": model_path.name,
        "checkpoint": checkpoint_relative,
        "video_group": video_folder.name,
        "execution_mode": execution_mode,
        "devices": devices,
        "npu_devices": devices if execution_mode == "multi_npu" else [],
        "reference_key": reference_key,
        "reference_config": reference_config,
        "worker_launch": _worker_launch_metadata(execution_mode, devices, bool(metrics)),
        "selected_videos": selected_videos,
        "metric_health": preflight.get("metrics", {}).get("health", {}),
        "preflight": preflight,
    }
    if body.get("retry_of_run_id") is not None:
        metadata["retry_of_run_id"] = int(body["retry_of_run_id"])
    name = body.get("name") or f"{model_path.stem} / {video_folder.name}"
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
        metadata={**metadata, "output_dir": str(workspace.runs_dir / str(db.next_run_id()))},
        create_inference_job=not is_multi,
    )
    if is_multi:
        _create_inference_shards(db, run_id, model_id, dataset_id, height, width, precision, metrics, devices, int(body.get("batch_size_per_device") or body.get("batch_size") or 1))
        if execution_mode == "multi_npu":
            _start_local_npu_worker_processes(workspace, run_id, devices, start_metric_worker=bool(metrics))
        else:
            _start_local_inference_worker(db, workspace, count=len(devices))
    else:
        _start_local_inference_worker(db, workspace)
    return {"run_id": run_id, "run": db.get_run(run_id), "preflight": preflight}


def _create_video_compare_run(db: Database, workspace: WorkspaceConfig, body: dict) -> dict:
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
    width = int(preflight.get("alignment", {}).get("width") or 0)
    height = int(preflight.get("alignment", {}).get("height") or 0)
    video_name = Path(reference_path).stem
    compare_tracks = [
        {
            "distorted_path": str(track.get("path") or ""),
            "track_label": str(track.get("track_label") or track.get("label") or f"pred{index + 1}"),
            "track_run_id": track.get("run_id") or track.get("track_run_id"),
            "artifact_id": track.get("artifact_id"),
            "video_name": track.get("video_name") or track.get("video"),
        }
        for index, track in enumerate(distorted_tracks)
    ]
    reference_config = {
        "run_type": "video_compare",
        "reference_path": reference_path,
        "distorted_tracks": compare_tracks,
        "align_mode": "strict",
        "frame_count": int(preflight.get("alignment", {}).get("frame_count") or 0),
        "width": width,
        "height": height,
    }
    reference_key = _reference_key(reference_config)
    compare_tag = reference_key[:12]
    model_id = db.upsert_model(
        name="video_compare",
        adapter="dummy",
        checkpoint_path=None,
        input_height=height,
        input_width=width,
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
        },
    )
    samples = scan_dataset(db, workspace, dataset_id)
    if samples <= 0:
        raise ValueError("compare inputs did not produce any aligned frames")

    metadata = {
        "run_type": "video_compare",
        "source": "direct_compare",
        "reference_path": reference_path,
        "distorted_path": distorted_path,
        "distorted_tracks": compare_tracks,
        "align_mode": "strict",
        "reference_key": reference_key,
        "reference_config": reference_config,
        "metric_health": preflight.get("metrics", {}).get("health", {}),
        "request": {
            "run_type": "video_compare",
            "reference": body.get("reference"),
            "distorted": body.get("distorted"),
            "extra_layers": body.get("extra_layers") if "extra_layers" in body else None,
            "align_mode": "strict",
            "metrics": metrics,
        },
        "preflight": preflight,
    }
    if body.get("retry_of_run_id") is not None:
        metadata["retry_of_run_id"] = int(body["retry_of_run_id"])
    run_id = db.create_run(
        name=body.get("name") or f"compare / {Path(reference_path).stem}",
        model_id=model_id,
        dataset_id=dataset_id,
        height=height,
        width=width,
        batch_size=1,
        device="cpu",
        precision="fp32",
        metrics=metrics,
        metadata={**metadata, "output_dir": str(workspace.runs_dir / str(db.next_run_id()))},
    )
    _start_local_inference_worker(db, workspace)
    return {"run_id": run_id, "run": db.get_run(run_id), "preflight": preflight}


def _retry_run(db: Database, workspace: WorkspaceConfig, run_id: int) -> dict:
    run = db.get_run(run_id)
    request = dict((run.get("metadata") or {}).get("request") or {})
    if not request:
        raise ValueError("这个 Run 没有可重试的文件夹入口配置")
    request["retry_of_run_id"] = run_id
    request["name"] = f"{run['name']} retry"
    if str(request.get("run_type") or "model_inference") == "video_compare":
        return _create_video_compare_run(db, workspace, request)
    return _create_run_from_files(db, workspace, request)


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
    run = db.get_run(run_id)
    if str(run.get("status") or "") not in TERMINAL_RUN_STATUSES:
        raise ValueError("cleanup-artifacts is only allowed after a run is completed, failed, or canceled")
    run_dir = (workspace.runs_dir / str(run_id)).resolve()
    runs_root = workspace.runs_dir.resolve()
    if not _is_relative_to(run_dir, runs_root):
        raise ValueError("run output directory is outside workspace runs directory")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    db.mark_run_artifacts_cleaned(run_id)
    return {"run_id": run_id, "artifact_cleaned": True, "output_dir": str(run_dir)}


def _checkpoint_relative(workspace: WorkspaceConfig, checkpoint_path: Path | None) -> str | None:
    if checkpoint_path is None:
        return None
    from vfieval.file_inputs import checkpoints_dir

    return checkpoint_path.resolve().relative_to(checkpoints_dir(workspace)).as_posix()


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
        db.add_run_job(
            run_id,
            "inference",
            payload,
            progress_total=len(sample_ids),
            shard_index=shard_index,
            device=device,
            metadata={"metrics_after_all_shards": metrics},
        )
    db.update_run_progress_from_jobs(run_id, "queued")


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


def _start_local_inference_worker(db: Database, workspace: WorkspaceConfig, count: int = 1) -> None:
    def _target(index: int) -> None:
        run_worker(
            db,
            workspace,
            WorkerOptions(role="all", once=True, worker_id=f"local-ui-worker-{index}"),
        )

    for index in range(max(1, int(count))):
        threading.Thread(target=_target, args=(index,), daemon=True).start()


def _start_local_npu_worker_processes(
    workspace: WorkspaceConfig,
    run_id: int,
    devices: list[str],
    start_metric_worker: bool = False,
) -> list[subprocess.Popen]:
    processes = []
    logs_dir = workspace.runs_dir / str(run_id) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for index, device in enumerate(devices):
        command = _worker_process_command(
            workspace,
            role="inference",
            device_filter=device,
            worker_id=f"local-npu-{run_id}-{index}-{device.replace(':', '-')}",
            once=True,
            idle_timeout=None,
        )
        processes.append(_spawn_worker_process(command, logs_dir / f"worker-{device.replace(':', '-')}.log"))
    if start_metric_worker:
        command = _worker_process_command(
            workspace,
            role="metric",
            device_filter=None,
            worker_id=f"local-metric-{run_id}",
            once=False,
            idle_timeout=86400.0,
        )
        processes.append(_spawn_worker_process(command, logs_dir / "worker-metric.log"))
    return processes


def _worker_process_command(
    workspace: WorkspaceConfig,
    role: str,
    device_filter: str | None = None,
    worker_id: str | None = None,
    once: bool = False,
    idle_timeout: float | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vfieval.cli",
        "--workspace",
        str(workspace.root),
        "worker",
        "--role",
        role,
        "--poll-interval",
        "1",
    ]
    if once:
        command.append("--once")
    if worker_id:
        command.extend(["--worker-id", worker_id])
    if device_filter:
        command.extend(["--device-filter", device_filter])
    if idle_timeout is not None:
        command.extend(["--idle-timeout", str(float(idle_timeout))])
    return command


def _spawn_worker_process(command: list[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    src_root = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_root if not existing_pythonpath else f"{src_root}{os.pathsep}{existing_pythonpath}"
    log_handle = log_path.open("ab")
    try:
        return subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    finally:
        log_handle.close()


def _selection_hash(selected_videos: list[str], frame_step: int, max_frames: int | None) -> str:
    data = {
        "selected_videos": selected_videos,
        "frame_step": frame_step,
        "max_frames": max_frames,
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:12]


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
    return run


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
    page_size = min(200, max(1, int(page_size or 50)))
    page = max(1, int(page or 1))
    total = len(summaries)
    start = (page - 1) * page_size
    videos = []
    for summary in summaries[start : start + page_size]:
        samples = db.list_samples_by_video(run_id, str(summary["video_name"]))
        video_payload = _build_video_payload(db, run, samples, str(summary["video_name"]))
        videos.append(
            {
                "video_name": video_payload.get("video_name") or summary["video_name"],
                "video_file": video_payload.get("video_file") or summary.get("video_file"),
                "fps": video_payload.get("fps") or summary.get("fps"),
                "sample_count": int(summary.get("sample_count") or len(video_payload.get("samples") or [])),
                "video_artifacts": video_payload.get("video_artifacts") or {},
                "video_artifact_tracks": video_payload.get("video_artifact_tracks") or [],
                "video_metrics": video_payload.get("video_metrics") or {},
                "metric_summary": video_payload.get("metric_summary") or {},
                "worst_samples": video_payload.get("worst_samples") or {},
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
    sample_rows = db.list_samples_by_video(run_id, video_name)
    if not sample_rows:
        raise ValueError(f"video not found in run timeline: {video_name}")
    video = _build_video_payload(db, run, sample_rows, video_name)
    samples = list(video.get("samples") or [])
    metric_name = metric or _first_metric_name(samples)
    bucket_count = min(1000, max(1, int(bucket_count or 120)))
    window_size = min(2000, max(1, int(window_size or 300)))
    window_start = max(0, min(int(window_start or 0), max(0, len(samples) - 1)))
    window_samples = samples[window_start : window_start + window_size]
    return {
        "run_id": run_id,
        "video_name": video.get("video_name"),
        "video_file": video.get("video_file"),
        "fps": video.get("fps"),
        "metric": metric_name,
        "sample_count": len(samples),
        "window_start": window_start,
        "window_size": window_size,
        "overview": _timeline_buckets(samples, metric_name, bucket_count),
        "samples": window_samples,
        "video_artifacts": video.get("video_artifacts") or {},
        "video_artifact_tracks": video.get("video_artifact_tracks") or [],
        "video_metrics": video.get("video_metrics") or {},
        "metric_summary": video.get("metric_summary") or {},
        "worst_samples": video.get("worst_samples") or {},
    }


def _build_video_payload(db: Database, run: dict[str, object], samples: list[dict], requested_video_name: str) -> dict:
    job_ids = set(db.run_inference_job_ids(int(run["id"])))
    requested_sample_metrics = _requested_sample_metrics(run)
    requested_video_metrics = _requested_video_metrics(run)
    metric_job_status = _metric_job_status(db, run)
    entries = [
        _sample_timeline_entry(db, run, sample, job_ids, requested_sample_metrics, metric_job_status)
        for sample in samples
    ]
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
        "video_artifacts": _video_artifact_map(db, int(run["id"]), video_name),
        "video_artifact_tracks": _video_artifact_tracks(db, int(run["id"]), video_name),
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
    sample_id = int(sample["id"])
    artifact_rows = [row for row in db.list_artifacts_by_sample(sample_id) if int(row["job_id"]) in job_ids]
    sample_errors: list[dict[str, str]] = []
    has_artifacts = False
    for artifact in artifact_rows:
        kind = artifact["kind"]
        if kind == "sample_error":
            meta = artifact.get("metadata") or {}
            sample_errors.append({"error_type": meta.get("error_type", ""), "message": meta.get("message", "")})
        else:
            has_artifacts = True
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
    result: dict[str, int] = {}
    for artifact in _video_artifacts_for_video(db, run_id, video_name):
        result.setdefault(artifact["kind"], int(artifact["id"]))
    return result


def _video_artifact_tracks(db: Database, run_id: int, video_name: str) -> list[dict[str, object]]:
    return [
        {
            "id": int(artifact["id"]),
            "kind": artifact["kind"],
            "url": f"/api/files/{int(artifact['id'])}",
            "track_label": (artifact.get("metadata") or {}).get("compare_track_label"),
            "track_run_id": (artifact.get("metadata") or {}).get("compare_track_run_id"),
        }
        for artifact in _video_artifacts_for_video(db, run_id, video_name)
    ]


def _video_artifacts_for_video(db: Database, run_id: int, video_name: str) -> list[dict]:
    rows = []
    for kind in VIDEO_TIMELINE_ARTIFACTS:
        rows.extend(db.list_run_video_artifacts(run_id, video_name=video_name, kind=kind))
    return rows


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
        allowed_kinds = requested_layers.get(int(source_run_id), set()) if requested_layers is not None else None
        for artifact in db.list_artifacts_by_sample(int(source_sample["id"])):
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


def _artifact_payload(artifact: dict) -> dict[str, object]:
    artifact_id = int(artifact["id"])
    has_preview = bool((artifact.get("metadata") or {}).get("preview_path"))
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
        video_name = details.get("video_name")
        key = (
            row.get("sample_id"),
            row["metric_name"],
            video_name,
            details.get("compare_track_label"),
            details.get("compare_track_run_id"),
        )
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
    metric_job_id = run.get("metric_job_id")
    if metric_job_id is None:
        return None
    try:
        return str(db.get_job(int(metric_job_id)).get("status") or "")
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
    failed_names = sorted(
        {
            row["metric_name"]
            for row in db.list_run_metrics(run_id)
            if row["status"] in {"failed", "unavailable"}
        }
    )
    if not failed_names:
        failed_names = list(run.get("metrics") or [])
    if not failed_names:
        raise ValueError("Run has no metrics to retry")
    job_id = db.create_job(
        "metric",
        {
            "run_id": run_id,
            "inference_job_id": int(inference_job_ids[0]),
            "inference_job_ids": inference_job_ids,
            "dataset_id": int(run["dataset_id"]),
            "metric_names": failed_names,
            "retry": True,
        },
    )
    db.set_run_metric_job(run_id, job_id)
    return {"run_id": run_id, "metric_job_id": job_id, "metric_names": failed_names}


def _dashboard(db: Database) -> dict:
    runs = db.list_runs(limit=500)
    workers = db.list_workers()
    active_statuses = {"queued", "running", "metric_queued", "metric_running"}
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


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
