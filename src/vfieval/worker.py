from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import time
import traceback
from dataclasses import dataclass

import torch

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.devices import list_npu_devices, npu_unavailable_reason, prepare_worker_device, supported_precisions
from vfieval.job_errors import describe_job_failure, enrich_job_error
from vfieval.orchestration import create_inference_jobs_for_run, start_workers_for_run
from vfieval.pipeline.decode_runner import run_decode_job
from vfieval.pipeline.inference import RunCanceled, run_inference_job
from vfieval.pipeline.finalize_runner import run_finalize_job
from vfieval.pipeline.metrics_runner import run_metric_job
from vfieval.run_cleanup import RunCleanupService
from vfieval.metrics import METRIC_NAMES


ROLE_KINDS = {
    "decode": ["decode"],
    "inference": ["inference"],
    "metric": ["metric"],
    "finalize": ["finalize"],
    "all": ["decode", "inference", "finalize", "metric"],
}


@dataclass(frozen=True)
class WorkerOptions:
    role: str = "all"
    once: bool = False
    poll_interval: float = 5.0
    worker_id: str | None = None
    device_filter: str | None = None
    idle_timeout: float | None = None


def detect_capabilities() -> dict[str, object]:
    cuda_devices = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            cuda_devices.append(
                {
                    "id": f"cuda:{idx}",
                    "name": torch.cuda.get_device_name(idx),
                    "memory_bytes": torch.cuda.get_device_properties(idx).total_memory,
                }
            )
    npu_devices = list_npu_devices()
    npu_error = npu_unavailable_reason()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "cuda": cuda_devices,
        "npu": npu_devices,
        "errors": {
            "npu": npu_error,
        },
        "cpu": True,
        "precision_support": {
            "cpu": supported_precisions("cpu"),
            "cuda": supported_precisions("cuda", available=bool(cuda_devices)),
            "npu": supported_precisions("npu", available=bool(npu_devices)),
        },
        "metric_support": list(METRIC_NAMES),
        "decode_backends": _decode_backend_capabilities(),
    }


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _module_error(name: str) -> str | None:
    try:
        __import__(name)
        return None
    except ImportError as exc:
        return str(exc)


def _decode_backend_capabilities() -> dict[str, dict[str, object]]:
    ffmpeg = shutil.which("ffmpeg")
    ffmpeg_version = None
    ffmpeg_error = None
    if ffmpeg:
        try:
            completed = subprocess.run(
                [ffmpeg, "-version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            first_line = (completed.stdout or completed.stderr or "").splitlines()
            ffmpeg_version = first_line[0] if first_line else None
            if completed.returncode != 0:
                ffmpeg_error = completed.stderr.strip() or "ffmpeg -version failed"
        except Exception as exc:
            ffmpeg_error = str(exc)
    return {
        "opencv": {
            "available": _module_available("cv2"),
            "error": _module_error("cv2"),
        },
        "ffmpeg": {
            "available": bool(ffmpeg) and ffmpeg_error is None,
            "path": ffmpeg,
            "version": ffmpeg_version,
            "error": ffmpeg_error or (None if ffmpeg else "ffmpeg is not on PATH"),
        },
    }


def run_worker(db: Database, workspace: WorkspaceConfig, options: WorkerOptions) -> None:
    if options.role not in ROLE_KINDS:
        raise ValueError(f"role must be one of {sorted(ROLE_KINDS)}")
    prepare_worker_device(options.device_filter)
    suffix = f"-{options.device_filter}" if options.device_filter else ""
    worker_id = options.worker_id or f"{socket.gethostname()}-{options.role}{suffix}-{os.getpid()}"
    capabilities = detect_capabilities()
    if options.device_filter:
        capabilities["device_filter"] = options.device_filter
    db.register_worker(worker_id, options.role, capabilities)
    last_activity = time.time()

    while True:
        capabilities = detect_capabilities()
        if options.device_filter:
            capabilities["device_filter"] = options.device_filter
        db.register_worker(worker_id, options.role, capabilities)
        job = db.claim_next_job(worker_id, ROLE_KINDS[options.role], device_filter=options.device_filter)
        if job is None:
            if options.once:
                return
            if options.idle_timeout is not None and time.time() - last_activity >= options.idle_timeout:
                return
            time.sleep(options.poll_interval)
            continue
        last_activity = time.time()

        try:
            if job["kind"] == "decode":
                result = run_decode_job(db, workspace, int(job["id"]))
                if db.get_job(int(job["id"]))["status"] != "canceled":
                    db.complete_job(int(job["id"]), result)
                    run_id = job.get("payload", {}).get("run_id")
                    if run_id is not None:
                        create_inference_jobs_for_run(db, int(run_id))
                        start_workers_for_run(db, workspace, int(run_id))
            elif job["kind"] == "inference":
                result = run_inference_job(db, workspace, int(job["id"]))
                if db.get_job(int(job["id"]))["status"] != "canceled":
                    db.complete_job(int(job["id"]), result.__dict__)
                    run_id = job.get("payload", {}).get("run_id")
                    if run_id is not None:
                        from vfieval.media_assets import sync_run_assets

                        sync_run_assets(db, workspace, int(run_id))
                        if result.performance:
                            from vfieval.performance import execution_profile_identity, record_execution_profile

                            run = db.get_run(int(run_id))
                            request = dict((run.get("metadata") or {}).get("request") or {})
                            request.update(
                                {
                                    "height": int(run.get("height") or 0),
                                    "width": int(run.get("width") or 0),
                                    "artifact_profile": result.performance.get("artifact_profile") or request.get("artifact_profile"),
                                    "device_model": result.performance.get("device_name") or "",
                                }
                            )
                            try:
                                identity = execution_profile_identity(workspace, request)
                                record_execution_profile(
                                    db,
                                    identity,
                                    {
                                        "batch_size": int(job.get("payload", {}).get("batch_size") or 1),
                                        "prefetch_workers": result.performance.get("prefetch_workers"),
                                        "save_workers": result.performance.get("save_workers"),
                                        "max_save_inflight": job.get("payload", {}).get("max_save_inflight"),
                                    },
                                    result.performance,
                                )
                            except Exception:
                                pass
                    if run_id is not None and int(job.get("payload", {}).get("shard_count") or 1) > 1:
                        db.maybe_complete_multi_run_inference(int(run_id))
            elif job["kind"] == "metric":
                result = run_metric_job(db, workspace, int(job["id"]))
                db.complete_job(int(job["id"]), result)
            elif job["kind"] == "finalize":
                result = run_finalize_job(db, workspace, int(job["id"]))
                db.complete_job(int(job["id"]), result)
            else:
                raise ValueError(f"unsupported job kind {job['kind']}")
        except RunCanceled:
            if options.once:
                return
        except Exception as exc:
            payload = job.get("payload") or {}
            run_id = payload.get("run_id")
            error = enrich_job_error(
                job,
                {
                    "message": describe_job_failure(job, exc),
                    "type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
            db.fail_job(
                int(job["id"]),
                error,
            )
            if run_id is not None:
                db.fail_run(int(run_id), error)
            if options.once:
                return
        finally:
            # A delete request may have been waiting for this exact worker
            # boundary. Persistent purge state lets any worker safely resume it;
            # the SQLite claim prevents two workers from deleting concurrently.
            try:
                RunCleanupService(db, workspace).process_pending(limit=20)
            except Exception as exc:
                print(f"run cleanup coordinator failed: {type(exc).__name__}: {exc}")
