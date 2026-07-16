from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from vfieval.config import WorkspaceConfig
from vfieval.db import Database


def create_inference_jobs_for_run(
    db: Database,
    run_id: int,
    *,
    source_job_id: int | None = None,
    source_job_result: dict[str, Any] | None = None,
) -> list[int]:
    existing = db.list_run_jobs(run_id, "inference")
    if existing and source_job_id is None:
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
    request = metadata.get("request") or {}
    artifact_profile = str(metadata.get("artifact_profile") or request.get("artifact_profile") or "evaluation")
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
            artifact_profile=artifact_profile,
            prefetch_workers=request.get("prefetch_workers"),
            save_workers=request.get("save_workers"),
            max_save_inflight=request.get("max_save_inflight"),
            artifact_db_batch_size=request.get("artifact_db_batch_size"),
            sample_npu_smi=request.get("sample_npu_smi", True),
            benchmark_warmup_batches=request.get("benchmark_warmup_batches"),
            benchmark_samples=request.get("benchmark_samples"),
            source_job_id=source_job_id,
            source_job_result=source_job_result,
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
            "artifact_profile": artifact_profile,
            "prefetch_workers": request.get("prefetch_workers"),
            "save_workers": request.get("save_workers"),
            "max_save_inflight": request.get("max_save_inflight"),
            "artifact_db_batch_size": request.get("artifact_db_batch_size"),
            "sample_npu_smi": request.get("sample_npu_smi", True),
            "benchmark_warmup_batches": request.get("benchmark_warmup_batches"),
            "benchmark_samples": request.get("benchmark_samples"),
        }
        job_ids = db.publish_inference_jobs(
            run_id,
            [
                {
                    "payload": payload,
                    "progress_total": len(samples),
                    "shard_index": 0,
                    "device": str(run.get("device") or "cpu"),
                }
            ],
            source_job_id=source_job_id,
            source_job_result=source_job_result,
        )
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
    if not devices:
        return []
    total = sum(len(values) for values in grouped.values())
    average = total / max(1, len(devices))
    split_needed = len(grouped) < len(devices) or any(len(values) > average * 1.25 for values in grouped.values())
    units: list[list[int]] = []
    for _key, sample_ids in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        if not split_needed:
            units.append(sample_ids)
            continue
        desired = max(1, min(len(sample_ids), round(len(sample_ids) / max(1.0, average))))
        desired = max(desired, min(len(devices), len(sample_ids)) if len(grouped) == 1 else desired)
        base, remainder = divmod(len(sample_ids), desired)
        offset = 0
        for segment_index in range(desired):
            segment_size = base + (1 if segment_index < remainder else 0)
            if segment_size <= 0:
                continue
            units.append(sample_ids[offset : offset + segment_size])
            offset += segment_size
    partitions: list[list[int]] = [[] for _ in devices]
    loads = [0 for _ in devices]
    for sample_ids in sorted(units, key=len, reverse=True):
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
    artifact_profile: str = "evaluation",
    prefetch_workers: int | None = None,
    save_workers: int | None = None,
    max_save_inflight: int | None = None,
    artifact_db_batch_size: int | None = None,
    sample_npu_smi: bool = True,
    benchmark_warmup_batches: int | None = None,
    benchmark_samples: int | None = None,
    source_job_id: int | None = None,
    source_job_result: dict[str, Any] | None = None,
) -> list[int]:
    partitions = partition_samples_by_video(samples, devices)
    job_specs: list[dict[str, Any]] = []
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
            "artifact_profile": artifact_profile,
            "prefetch_workers": prefetch_workers,
            "save_workers": save_workers,
            "max_save_inflight": max_save_inflight,
            "artifact_db_batch_size": artifact_db_batch_size,
            "sample_npu_smi": bool(sample_npu_smi),
            "benchmark_warmup_batches": benchmark_warmup_batches,
            "benchmark_samples": benchmark_samples,
            # Benchmark performs canonical post-processing but deliberately
            # publishes no artifacts. It therefore has no shard video
            # manifests for a finalize job to merge.
            "defer_video_finalize": artifact_profile != "benchmark",
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
    if not job_specs:
        raise ValueError("decoded dataset did not produce any non-empty inference shards")
    return db.publish_inference_jobs(
        run_id,
        job_specs,
        source_job_id=source_job_id,
        source_job_result=source_job_result,
    )


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
    finalize_command = worker_process_command(
        workspace,
        role="finalize",
        device_filter=None,
        worker_id=f"local-finalize-{run_id}",
        once=False,
        idle_timeout=3600.0,
    )
    processes.append(_spawn_worker_process(finalize_command, logs_dir / "worker-finalize.log"))
    if start_metric_worker:
        for index, metric_device in enumerate(devices or [None]):
            command = worker_process_command(
                workspace,
                role="metric",
                device_filter=metric_device,
                worker_id=f"local-metric-{run_id}-{index}",
                once=False,
                idle_timeout=86400.0,
            )
            suffix = str(metric_device or "default").replace(":", "-")
            processes.append(_spawn_worker_process(command, logs_dir / f"worker-metric-{suffix}.log"))
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
