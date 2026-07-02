from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from vfieval.config import WorkspaceConfig
from vfieval.db import Database


def create_inference_jobs_for_run(db: Database, run_id: int) -> list[int]:
    existing = db.list_run_jobs(run_id, "inference")
    if existing:
        return [int(row["job_id"]) for row in existing]

    run = db.get_run(run_id)
    metadata = run.get("metadata") or {}
    dataset_id = int(run["dataset_id"])
    model_id = int(run["model_id"])
    height = int(run["height"])
    width = int(run["width"])
    precision = str(run["precision"])
    metrics = list(run.get("metrics") or [])
    execution_mode = str(metadata.get("execution_mode") or run.get("device") or "single")
    devices = [str(device) for device in (metadata.get("devices") or [run.get("device") or "cpu"])]
    batch_size = int(run.get("batch_size") or 1)
    visualize_height = metadata.get("visualize_height")
    visualize_width = metadata.get("visualize_width")
    samples = db.list_samples(dataset_id)
    if not samples:
        raise ValueError("decoded dataset has no samples")

    if execution_mode in {"multi_cuda", "multi_npu"}:
        job_ids = _create_inference_shards(
            db,
            run_id=run_id,
            model_id=model_id,
            dataset_id=dataset_id,
            height=height,
            width=width,
            precision=precision,
            metrics=metrics,
            devices=devices,
            batch_size_per_device=batch_size,
            samples=samples,
            visualize_height=visualize_height,
            visualize_width=visualize_width,
        )
    else:
        payload = {
            "run_id": run_id,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "height": height,
            "width": width,
            "batch_size": batch_size,
            "device": str(run.get("device") or "cpu"),
            "precision": precision,
            "metrics": metrics,
            "visualize_height": visualize_height,
            "visualize_width": visualize_width,
        }
        job_ids = [
            db.add_run_job(
                run_id,
                "inference",
                payload,
                progress_total=len(samples),
                shard_index=0,
                device=str(run.get("device") or "cpu"),
            )
        ]
    db.update_run_progress_from_jobs(run_id, "queued")
    return job_ids


def start_decode_worker(db: Database, workspace: WorkspaceConfig) -> None:
    _start_local_worker(db, workspace, role="decode", count=1)


def start_workers_for_run(db: Database, workspace: WorkspaceConfig, run_id: int) -> list[subprocess.Popen]:
    run = db.get_run(run_id)
    metadata = run.get("metadata") or {}
    execution_mode = str(metadata.get("execution_mode") or run.get("device") or "single")
    devices = [str(device) for device in (metadata.get("devices") or [run.get("device") or "cpu"])]
    if execution_mode == "multi_npu":
        return _start_local_npu_worker_processes(
            workspace,
            run_id,
            devices,
            start_metric_worker=bool(run.get("metrics")),
        )
    if execution_mode == "multi_cuda":
        _start_local_worker(db, workspace, role="all", count=max(1, len(devices)))
        return []
    _start_local_worker(db, workspace, role="all", count=1)
    return []


def partition_samples_by_video(samples: list[dict[str, Any]], devices: list[str]) -> list[list[int]]:
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


def worker_process_command(
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
    samples: list[dict[str, Any]],
    visualize_height: int | None = None,
    visualize_width: int | None = None,
) -> list[int]:
    partitions = partition_samples_by_video(samples, devices)
    job_ids: list[int] = []
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
            "visualize_height": visualize_height,
            "visualize_width": visualize_width,
        }
        job_ids.append(
            db.add_run_job(
                run_id,
                "inference",
                payload,
                progress_total=len(sample_ids),
                shard_index=shard_index,
                device=device,
                metadata={"metrics_after_all_shards": metrics},
            )
        )
    if not job_ids:
        raise ValueError("decoded dataset did not produce any non-empty inference shards")
    return job_ids


def _start_local_worker(db: Database, workspace: WorkspaceConfig, role: str, count: int = 1) -> None:
    def _target(index: int) -> None:
        from vfieval.worker import WorkerOptions, run_worker

        run_worker(
            db,
            workspace,
            WorkerOptions(role=role, once=True, worker_id=f"local-ui-{role}-worker-{index}"),
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
        command = worker_process_command(
            workspace,
            role="inference",
            device_filter=device,
            worker_id=f"local-npu-{run_id}-{index}-{device.replace(':', '-')}",
            once=True,
            idle_timeout=None,
        )
        processes.append(_spawn_worker_process(command, logs_dir / f"worker-{device.replace(':', '-')}.log"))
    if start_metric_worker:
        command = worker_process_command(
            workspace,
            role="metric",
            device_filter=None,
            worker_id=f"local-metric-{run_id}",
            once=False,
            idle_timeout=86400.0,
        )
        processes.append(_spawn_worker_process(command, logs_dir / "worker-metric.log"))
    return processes


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
