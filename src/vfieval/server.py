from __future__ import annotations

import hashlib
import json
import mimetypes
import re
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
    list_video_group_videos,
    list_model_files,
    list_video_groups,
    normalize_device_precision,
    preflight_run,
    resolve_model_file,
    resolve_run_dimensions,
    resolve_video_group,
    thumbnail_path,
)
from vfieval.metrics import METRIC_NAMES
from vfieval.metrics.health import metrics_health
from vfieval.worker import WorkerOptions, run_worker


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
                    return self._json(db.list_runs(limit=int(query.get("limit", ["100"])[0])))
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
                        return self._json(_run_timeline(db, run_id))
                    if section == "metric-summary":
                        return self._json(_run_metric_summary(db, run_id))
                    return self._json(db.get_run(run_id))
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
                if path == "/api/compare":
                    return self._json(_compare(db, query))
                if path == "/api/compare/samples":
                    return self._json(_compare_samples(db, query))
                if path.startswith("/api/files/"):
                    artifact_id = int(path.rsplit("/", 1)[-1])
                    return self._send_artifact(artifact_id)
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
                    model_id = db.register_model(
                        name=body["name"],
                        adapter=body["adapter"],
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

        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _json(self, data, status: HTTPStatus = HTTPStatus.OK) -> None:
            payload = json.dumps(data, indent=2, default=str).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, status: HTTPStatus, message: str, error_type: str = "Error") -> None:
            self._json({"error": {"type": error_type, "message": message}}, status=status)

        def _send_static(self, name: str) -> None:
            path = Path(__file__).parent / "web" / name
            if not path.exists():
                return self._error(HTTPStatus.NOT_FOUND, f"static file {name} not found")
            self._send_file(path)

        def _send_artifact(self, artifact_id: int) -> None:
            artifact = db.get("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
            if artifact is None:
                return self._error(HTTPStatus.NOT_FOUND, "artifact not found")
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
            self._send_file(Path(path_text).resolve())

        def _send_file(self, path: Path) -> None:
            if not path.exists() or not path.is_file():
                return self._error(HTTPStatus.NOT_FOUND, f"file not found: {path}")
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return VFIEvalHandler


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
    device, precision = normalize_device_precision(str(body.get("device") or "auto"), str(body.get("precision") or "auto"))
    selection_hash = _selection_hash(selected_videos, frame_step, max_frames)

    model_id = db.upsert_model(
        name=model_path.name,
        adapter=f"file:{model_path}",
        checkpoint_path=None,
        input_height=height,
        input_width=width,
        metadata={
            "source": "file",
            "model_file": model_path.name,
            "model_path": str(model_path),
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
        "source": "folder_flow",
        "request": {
            "model_file": model_path.name,
            "video_group": video_folder.name,
            "resolution_mode": body.get("resolution_mode") or "original",
            "height": height,
            "width": width,
            "batch_size": int(body.get("batch_size") or 1),
            "device": body.get("device") or "auto",
            "precision": body.get("precision") or "auto",
            "frame_step": frame_step,
            "max_frames": max_frames,
            "selected_videos": selected_videos,
            "metrics": metrics,
        },
        "model_file": model_path.name,
        "video_group": video_folder.name,
        "selected_videos": selected_videos,
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
        batch_size=int(body.get("batch_size") or 1),
        device=device,
        precision=precision,
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
    return _create_run_from_files(db, workspace, request)


def _preflight_error_message(preflight: dict) -> str:
    messages = [f"{item.get('title')}: {item.get('message')}" for item in preflight.get("errors", [])]
    return "；".join(messages) or "预检查失败"


def _start_local_inference_worker(db: Database, workspace: WorkspaceConfig) -> None:
    def _target() -> None:
        run_worker(
            db,
            workspace,
            WorkerOptions(role="all", once=True, worker_id="local-ui-worker"),
        )

    threading.Thread(target=_target, daemon=True).start()


def _selection_hash(selected_videos: list[str], frame_step: int, max_frames: int | None) -> str:
    data = {
        "selected_videos": selected_videos,
        "frame_step": frame_step,
        "max_frames": max_frames,
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:12]


CORE_TIMELINE_ARTIFACTS = {
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
METRIC_DIRECTIONS = {
    "vmaf": "higher_is_better",
    "lpips_vit_patch": "lower_is_better",
    "lpips_convnext": "lower_is_better",
    "cgvqm": "lower_is_better",
}
METRIC_STATUSES = ("completed", "unavailable", "failed", "skipped")


def _run_timeline(db: Database, run_id: int) -> dict:
    run = db.get_run(run_id)
    samples = db.list_samples(int(run["dataset_id"]))
    artifacts = db.list_run_artifacts(run_id)
    metrics = _latest_metric_rows(db.list_run_metrics(run_id))

    samples_with_artifacts: set[int] = set()
    video_artifacts: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        kind = artifact["kind"]
        sample_id = artifact.get("sample_id")
        if sample_id is None:
            if kind in VIDEO_TIMELINE_ARTIFACTS:
                name = str(artifact.get("metadata", {}).get("video_name") or "video")
                video_artifacts.setdefault(name, {})[kind] = int(artifact["id"])
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
        group["samples"].append(
            {
                "sample_id": sample_id,
                "sample_name": sample["name"],
                "frame_index": int(metadata.get("frame_index") or metadata.get("sample_index") or 0),
                "sample_index": int(metadata.get("sample_index") or 0),
                "timestamp": timestamps.get("gt") if isinstance(timestamps, dict) else None,
                "has_artifacts": sample_id in samples_with_artifacts,
                "has_gt": bool(sample.get("gt_path")),
                "metrics": metrics_by_sample.get(sample_id, {}),
                "metric_status": metric_status_by_sample.get(sample_id, {status: 0 for status in METRIC_STATUSES}),
            }
        )

    videos = []
    for group in groups.values():
        group["samples"] = sorted(group["samples"], key=lambda item: (item["frame_index"], item["sample_index"]))
        group["metric_summary"] = _metric_summary_for_video(group["samples"], summary.get("metrics", {}))
        group["worst_samples"] = _worst_samples_for_video(group["samples"])
        videos.append(group)
    videos.sort(key=lambda item: str(item.get("video_file") or item.get("video_name")))
    return {"run_id": run_id, "metric_summary": summary, "videos": videos}


def _run_videos(db: Database, run_id: int, page: int = 1, page_size: int = 50, q: str = "") -> dict:
    timeline = _run_timeline(db, run_id)
    query = q.strip().lower()
    videos = [
        {
            "video_name": video["video_name"],
            "video_file": video.get("video_file") or video["video_name"],
            "fps": video.get("fps"),
            "sample_count": len(video.get("samples") or []),
            "video_artifacts": video.get("video_artifacts") or {},
            "video_metrics": video.get("video_metrics") or {},
            "metric_summary": video.get("metric_summary") or {},
            "worst_samples": video.get("worst_samples") or {},
        }
        for video in timeline.get("videos", [])
        if not query
        or query in str(video.get("video_name") or "").lower()
        or query in str(video.get("video_file") or "").lower()
    ]
    page_size = min(200, max(1, int(page_size or 50)))
    page = max(1, int(page or 1))
    total = len(videos)
    start = (page - 1) * page_size
    return {
        "run_id": run_id,
        "page": page,
        "page_size": page_size,
        "filtered_count": total,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "videos": videos[start : start + page_size],
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
    timeline = _run_timeline(db, run_id)
    video = _find_timeline_video(timeline.get("videos", []), video_name)
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
        "video_metrics": video.get("video_metrics") or {},
        "metric_summary": video.get("metric_summary") or {},
        "worst_samples": video.get("worst_samples") or {},
    }


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


def _run_sample_payload(db: Database, run_id: int, sample_id: int) -> dict:
    run = db.get_run(run_id)
    sample = db.get_sample(sample_id)
    if sample is None or int(sample["dataset_id"]) != int(run["dataset_id"]):
        raise ValueError("sample does not belong to this run")

    artifacts: dict[str, int] = {}
    extra_artifacts: list[dict[str, object]] = []
    for artifact in db.list_run_artifacts(run_id):
        if artifact.get("sample_id") != sample_id:
            continue
        kind = artifact["kind"]
        if kind in CORE_TIMELINE_ARTIFACTS:
            artifacts[kind] = int(artifact["id"])
        elif kind.startswith("extra_"):
            extra_artifacts.append({"id": int(artifact["id"]), "kind": kind})

    metric_rows = _latest_metric_rows(db.list_run_metrics(run_id))
    metrics = {
        row["metric_name"]: {
            "status": row["status"],
            "value": row.get("value"),
            "details": row.get("details") or {},
        }
        for row in metric_rows
        if row.get("sample_id") == sample_id
    }
    metric_status = {status: 0 for status in METRIC_STATUSES}
    for metric in metrics.values():
        status = str(metric.get("status"))
        metric_status[status] = metric_status.get(status, 0) + 1

    metadata = sample.get("metadata") or {}
    timestamps = metadata.get("timestamps") or {}
    return {
        "sample_id": sample_id,
        "sample_name": sample["name"],
        "frame_index": int(metadata.get("frame_index") or metadata.get("sample_index") or 0),
        "sample_index": int(metadata.get("sample_index") or 0),
        "timestamp": timestamps.get("gt") if isinstance(timestamps, dict) else None,
        "metadata": metadata,
        "artifacts": artifacts,
        "extra_artifacts": extra_artifacts,
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
        "completed": 0,
        "unavailable": 0,
        "failed": 0,
        "skipped": 0,
        "mean": None,
        "min": None,
        "max": None,
        "worst_sample_id": None,
        "worst_value": None,
        "reasons": [],
    }


def _latest_metric_rows(metrics: list[dict]) -> list[dict]:
    latest: dict[tuple[object, str, object], dict] = {}
    for row in metrics:
        details = row.get("details") or {}
        video_name = details.get("video_name")
        key = (row.get("sample_id"), row["metric_name"], video_name)
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


def _retry_run_metrics(db: Database, run_id: int) -> dict:
    run = db.get_run(run_id)
    inference_job_id = run.get("inference_job_id")
    if inference_job_id is None:
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
            "inference_job_id": int(inference_job_id),
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
