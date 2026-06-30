from __future__ import annotations

import os
import platform
import shutil
import socket
import time
import traceback
from dataclasses import dataclass

import torch

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.pipeline.inference import RunCanceled, run_inference_job
from vfieval.pipeline.metrics_runner import run_metric_job
from vfieval.metrics import METRIC_NAMES


ROLE_KINDS = {
    "inference": ["inference"],
    "metric": ["metric"],
    "all": ["inference", "metric"],
}


@dataclass(frozen=True)
class WorkerOptions:
    role: str = "all"
    once: bool = False
    poll_interval: float = 5.0
    worker_id: str | None = None


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
    npu_devices = []
    try:
        import torch_npu  # noqa: F401

        npu_devices.append({"id": "npu:0", "name": "Ascend NPU"})
    except ImportError:
        pass
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "cuda": cuda_devices,
        "npu": npu_devices,
        "cpu": True,
        "precision_support": {
            "cpu": ["fp32"],
            "cuda": ["fp32", "fp16", "bf16"] if cuda_devices else [],
            "npu": ["fp32", "fp16", "bf16"] if npu_devices else [],
        },
        "metric_support": list(METRIC_NAMES),
        "decode_backends": {
            "opencv": _module_available("cv2"),
            "ffmpeg": shutil.which("ffmpeg") is not None,
        },
    }


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def run_worker(db: Database, workspace: WorkspaceConfig, options: WorkerOptions) -> None:
    if options.role not in ROLE_KINDS:
        raise ValueError(f"role must be one of {sorted(ROLE_KINDS)}")
    worker_id = options.worker_id or f"{socket.gethostname()}-{options.role}-{os.getpid()}"
    db.register_worker(worker_id, options.role, detect_capabilities())

    while True:
        db.register_worker(worker_id, options.role, detect_capabilities())
        job = db.claim_next_job(worker_id, ROLE_KINDS[options.role])
        if job is None:
            if options.once:
                return
            time.sleep(options.poll_interval)
            continue

        try:
            if job["kind"] == "inference":
                result = run_inference_job(db, workspace, int(job["id"]))
                if db.get_job(int(job["id"]))["status"] != "canceled":
                    db.complete_job(int(job["id"]), result.__dict__)
            elif job["kind"] == "metric":
                result = run_metric_job(db, workspace, int(job["id"]))
                db.complete_job(int(job["id"]), result)
            else:
                raise ValueError(f"unsupported job kind {job['kind']}")
        except RunCanceled:
            if options.once:
                return
        except Exception as exc:
            payload = job.get("payload") or {}
            run_id = payload.get("run_id")
            error = {
                "message": str(exc),
                "type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
            db.fail_job(
                int(job["id"]),
                error,
            )
            if run_id is not None:
                db.fail_run(int(run_id), error)
            if options.once:
                return
