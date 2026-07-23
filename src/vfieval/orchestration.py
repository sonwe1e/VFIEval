from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from vfieval.config import WorkspaceConfig
from vfieval.db import Database


_WORKER_PROCESSES: dict[int, subprocess.Popen] = {}
_LOCAL_WORKER_THREADS: dict[int, threading.Thread] = {}
_WORKER_LIFECYCLE_LOCK = threading.RLock()
_WORKER_PROCESS_SHUTDOWN_LOCK = threading.Lock()
_WORKER_ADMISSION_OPEN = True
_WORKER_LIFECYCLE_GENERATION = 0
_WORKER_LIFECYCLE_STOP_EVENT = threading.Event()
_WORKER_THREAD_CONTEXT = threading.local()
_JOB_SUPERVISORS: dict[str, "JobSupervisor"] = {}


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
    if wake_job_supervisor(db):
        return
    _start_local_worker(db, workspace, role="decode", count=1)


def start_workers_for_run(db: Database, workspace: WorkspaceConfig, run_id: int) -> list[subprocess.Popen]:
    supervisor = active_job_supervisor(db)
    if supervisor is not None:
        supervisor.wake()
        return []
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


class JobSupervisor:
    """Maintain one reusable local Worker per queued role/device slot."""

    def __init__(
        self,
        db: Database,
        workspace: WorkspaceConfig,
        *,
        scan_interval: float = 1.0,
    ) -> None:
        self.db = db
        self.workspace = workspace
        self.scan_interval = max(0.1, float(scan_interval))
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._lock = threading.RLock()
        self._threads: dict[tuple[str, str], tuple[threading.Thread, threading.Event]] = {}
        self._processes: dict[tuple[str, str], subprocess.Popen] = {}
        self._coordinator: threading.Thread | None = None
        self._started = False
        self._last_scan_at: float | None = None
        self._last_error: str | None = None

    @property
    def registry_key(self) -> str:
        return str(self.db.db_path.resolve())

    def start(self) -> None:
        with self._lock:
            if self._started:
                self.wake()
                return
            self._stop_event.clear()
            self._wake_event.clear()
            self._started = True
            with _WORKER_LIFECYCLE_LOCK:
                existing = _JOB_SUPERVISORS.get(self.registry_key)
                if existing is not None and existing is not self:
                    self._started = False
                    raise RuntimeError(
                        f"Job supervisor already active for {self.registry_key}"
                    )
                _JOB_SUPERVISORS[self.registry_key] = self
            self._coordinator = threading.Thread(
                target=self._run,
                name="vfieval-job-supervisor",
                daemon=True,
            )
            self._coordinator.start()
        self.wake()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
            self._stop_event.set()
            self._wake_event.set()
            for _thread, wake_event in self._threads.values():
                wake_event.set()
            coordinator = self._coordinator
            threads = [thread for thread, _wake in self._threads.values()]
            processes = list(self._processes.values())
            with _WORKER_LIFECYCLE_LOCK:
                if _JOB_SUPERVISORS.get(self.registry_key) is self:
                    _JOB_SUPERVISORS.pop(self.registry_key, None)
        deadline = time.monotonic() + max(0.0, float(timeout))
        if coordinator is not None and coordinator is not threading.current_thread():
            coordinator.join(timeout=max(0.0, deadline - time.monotonic()))
        for thread in threads:
            if thread is threading.current_thread():
                continue
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in processes:
            if _process_exited(process):
                continue
            try:
                process.terminate()
            except (ChildProcessError, ProcessLookupError):
                continue
            except Exception:
                pass
        for process in processes:
            if _process_exited(process):
                continue
            _wait_for_worker_process(
                process,
                max(0.0, deadline - time.monotonic()),
            )
        with self._lock:
            self._coordinator = None
            self._threads = {
                key: slot for key, slot in self._threads.items() if slot[0].is_alive()
            }
            self._processes = {
                key: process
                for key, process in self._processes.items()
                if not _process_exited(process)
            }

    def wake(self) -> None:
        self._wake_event.set()
        with self._lock:
            for _thread, wake_event in self._threads.values():
                wake_event.set()

    def run_once(self) -> int:
        started = 0
        try:
            requirements = self.db.queued_job_requirements()
            for requirement in requirements:
                role = str(requirement["role"])
                raw_device = requirement.get("device")
                device = str(raw_device) if raw_device is not None else ""
                if role in {"decode", "finalize"}:
                    device = ""
                if self._ensure_slot(role, device):
                    started += 1
            self._last_error = None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
        self._last_scan_at = time.time()
        return started

    def status(self) -> dict[str, Any]:
        with self._lock:
            thread_slots = {
                f"{role}:{device or 'default'}": thread.is_alive()
                for (role, device), (thread, _wake) in self._threads.items()
            }
            process_slots = {
                f"{role}:{device or 'default'}": not _process_exited(process)
                for (role, device), process in self._processes.items()
            }
        return {
            "running": self._started,
            "thread_slots": thread_slots,
            "process_slots": process_slots,
            "last_scan_at": self._last_scan_at,
            "last_error": self._last_error,
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.run_once()
            self._wake_event.wait(self.scan_interval)
            self._wake_event.clear()

    def _ensure_slot(self, role: str, device: str) -> bool:
        key = (role, device)
        accelerator = device.startswith(("cuda:", "npu:"))
        with self._lock:
            if accelerator:
                existing = self._processes.get(key)
                if existing is not None and not _process_exited(existing):
                    return False
                if existing is not None:
                    self._processes.pop(key, None)
                command = worker_process_command(
                    self.workspace,
                    role=role,
                    device_filter=device,
                    worker_id=self._worker_id(role, device),
                    once=False,
                    idle_timeout=None,
                )
                log_path = (
                    self.workspace.root
                    / "logs"
                    / "workers"
                    / f"{role}-{device.replace(':', '-')}.log"
                )
                try:
                    self._processes[key] = _spawn_worker_process(command, log_path)
                except _WorkerAdmissionClosed:
                    return False
                return True

            existing_thread = self._threads.get(key)
            if existing_thread is not None and existing_thread[0].is_alive():
                existing_thread[1].set()
                return False
            worker_wake = threading.Event()
            thread = threading.Thread(
                target=self._run_thread_worker,
                args=(key, worker_wake),
                name=f"vfieval-supervisor-{role}-{device or 'default'}",
                daemon=True,
            )
            self._threads[key] = (thread, worker_wake)
            thread.start()
            return True

    def _run_thread_worker(
        self,
        key: tuple[str, str],
        wake_event: threading.Event,
    ) -> None:
        role, device = key
        try:
            from vfieval.worker import WorkerOptions, run_worker

            run_worker(
                self.db,
                self.workspace,
                WorkerOptions(
                    role=role,
                    once=False,
                    poll_interval=self.scan_interval,
                    worker_id=self._worker_id(role, device),
                    device_filter=device or None,
                    stop_event=self._stop_event,
                    wake_event=wake_event,
                ),
            )
        finally:
            with self._lock:
                current = self._threads.get(key)
                if current is not None and current[0] is threading.current_thread():
                    self._threads.pop(key, None)

    @staticmethod
    def _worker_id(role: str, device: str) -> str:
        suffix = (device or "default").replace(":", "-")
        return f"local-supervisor-{role}-{suffix}-{os.getpid()}"


def active_job_supervisor(db: Database) -> JobSupervisor | None:
    key = str(db.db_path.resolve())
    with _WORKER_LIFECYCLE_LOCK:
        supervisor = _JOB_SUPERVISORS.get(key)
    return supervisor


def wake_job_supervisor(db: Database) -> bool:
    supervisor = active_job_supervisor(db)
    if supervisor is None:
        return False
    supervisor.wake()
    return True


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


class _WorkerAdmissionClosed(RuntimeError):
    pass


def _admitted_worker_generation() -> tuple[int, threading.Event] | None:
    with _WORKER_LIFECYCLE_LOCK:
        caller_generation = getattr(_WORKER_THREAD_CONTEXT, "generation", None)
        if not _WORKER_ADMISSION_OPEN:
            return None
        if caller_generation is not None and int(caller_generation) != _WORKER_LIFECYCLE_GENERATION:
            return None
        return _WORKER_LIFECYCLE_GENERATION, _WORKER_LIFECYCLE_STOP_EVENT


def open_worker_admission() -> int:
    """Open a fresh local-worker generation for a new server lifecycle."""
    global _WORKER_ADMISSION_OPEN
    global _WORKER_LIFECYCLE_GENERATION
    global _WORKER_LIFECYCLE_STOP_EVENT
    with _WORKER_LIFECYCLE_LOCK:
        if _WORKER_ADMISSION_OPEN:
            return _WORKER_LIFECYCLE_GENERATION
        _WORKER_LIFECYCLE_GENERATION += 1
        _WORKER_LIFECYCLE_STOP_EVENT = threading.Event()
        _WORKER_ADMISSION_OPEN = True
        return _WORKER_LIFECYCLE_GENERATION


def _unregister_local_worker_thread(thread: threading.Thread) -> None:
    with _WORKER_LIFECYCLE_LOCK:
        key = id(thread)
        if _LOCAL_WORKER_THREADS.get(key) is thread:
            _LOCAL_WORKER_THREADS.pop(key, None)


def _local_worker_thread_snapshot() -> list[threading.Thread]:
    with _WORKER_LIFECYCLE_LOCK:
        return list(_LOCAL_WORKER_THREADS.values())


def _tracked_local_worker_thread_count() -> int:
    with _WORKER_LIFECYCLE_LOCK:
        return len(_LOCAL_WORKER_THREADS)


def _run_registered_local_worker(
    db: Database,
    workspace: WorkspaceConfig,
    role: str,
    index: int,
    generation: int,
    stop_event: threading.Event,
) -> None:
    thread = threading.current_thread()
    _WORKER_THREAD_CONTEXT.generation = generation
    try:
        if stop_event.is_set():
            return
        from vfieval.worker import WorkerOptions, run_worker

        if stop_event.is_set():
            return
        run_worker(
            db,
            workspace,
            WorkerOptions(role=role, once=True, worker_id=f"local-ui-{role}-worker-{index}"),
        )
    finally:
        _WORKER_THREAD_CONTEXT.__dict__.pop("generation", None)
        _unregister_local_worker_thread(thread)


def _start_local_worker(db: Database, workspace: WorkspaceConfig, role: str, count: int = 1) -> None:
    for index in range(max(1, int(count))):
        with _WORKER_LIFECYCLE_LOCK:
            admission = _admitted_worker_generation()
            if admission is None:
                return
            generation, stop_event = admission
            thread = threading.Thread(
                target=_run_registered_local_worker,
                args=(db, workspace, role, index, generation, stop_event),
                name=f"vfieval-local-{role}-{index}",
                daemon=True,
            )
            _LOCAL_WORKER_THREADS[id(thread)] = thread
            try:
                # Starting under the lifecycle lock removes the register/start
                # gap where shutdown could otherwise try to join an unstarted
                # thread and then let it escape afterward.
                thread.start()
            except Exception:
                _LOCAL_WORKER_THREADS.pop(id(thread), None)
                raise


def _start_local_npu_worker_processes(
    workspace: WorkspaceConfig,
    run_id: int,
    devices: list[str],
    start_metric_worker: bool = False,
) -> list[subprocess.Popen]:
    if _admitted_worker_generation() is None:
        return []
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
        try:
            processes.append(_spawn_worker_process(command, logs_dir / f"worker-{device.replace(':', '-')}.log"))
        except _WorkerAdmissionClosed:
            return processes
    finalize_command = worker_process_command(
        workspace,
        role="finalize",
        device_filter=None,
        worker_id=f"local-finalize-{run_id}",
        once=False,
        idle_timeout=3600.0,
    )
    try:
        processes.append(_spawn_worker_process(finalize_command, logs_dir / "worker-finalize.log"))
    except _WorkerAdmissionClosed:
        return processes
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
            try:
                processes.append(_spawn_worker_process(command, logs_dir / f"worker-metric-{suffix}.log"))
            except _WorkerAdmissionClosed:
                return processes
    return processes


def _worker_process_snapshot() -> list[subprocess.Popen]:
    with _WORKER_LIFECYCLE_LOCK:
        return list(_WORKER_PROCESSES.values())


def _tracked_worker_process_count() -> int:
    with _WORKER_LIFECYCLE_LOCK:
        return len(_WORKER_PROCESSES)


def _unregister_worker_process(process: subprocess.Popen) -> None:
    with _WORKER_LIFECYCLE_LOCK:
        key = id(process)
        if _WORKER_PROCESSES.get(key) is process:
            _WORKER_PROCESSES.pop(key, None)


def _watch_worker_process(process: subprocess.Popen) -> None:
    try:
        process.wait()
    except Exception:
        # A transient wait failure must not make a live owned child invisible
        # to explicit shutdown. Poll only removes a process confirmed dead.
        if _process_exited(process):
            _unregister_worker_process(process)
    else:
        _unregister_worker_process(process)


def _process_exited(process: subprocess.Popen) -> bool:
    try:
        return process.poll() is not None
    except Exception:
        return False


def _wait_for_worker_process(process: subprocess.Popen, timeout: float) -> bool:
    try:
        process.wait(timeout=max(0.0, float(timeout)))
    except subprocess.TimeoutExpired:
        return False
    except (ChildProcessError, ProcessLookupError):
        pass
    except Exception:
        if not _process_exited(process):
            return False
    _unregister_worker_process(process)
    return True


def _stop_process_spawned_during_shutdown(process: subprocess.Popen) -> None:
    """Fence a child whose ``Popen`` overlapped a completed shutdown pass."""
    if _process_exited(process):
        _unregister_worker_process(process)
        return
    try:
        process.terminate()
    except Exception:
        pass
    if _wait_for_worker_process(process, 0.25):
        return
    try:
        process.kill()
    except Exception:
        return
    _wait_for_worker_process(process, 0.75)


def _register_worker_process(
    process: subprocess.Popen,
    *,
    spawn_generation: int,
) -> subprocess.Popen:
    with _WORKER_LIFECYCLE_LOCK:
        _WORKER_PROCESSES[id(process)] = process
        overlapped_shutdown = (
            not _WORKER_ADMISSION_OPEN
            or spawn_generation != _WORKER_LIFECYCLE_GENERATION
        )
    watcher = threading.Thread(
        target=_watch_worker_process,
        args=(process,),
        name=f"vfieval-worker-process-{getattr(process, 'pid', id(process))}",
        daemon=True,
    )
    watcher.start()
    if overlapped_shutdown:
        _stop_process_spawned_during_shutdown(process)
    return process


def shutdown_worker_processes(timeout: float = 5.0) -> dict[str, int]:
    """Close local-worker admission and stop workers owned by this module.

    Calls are serialized and idempotent. Registered local worker threads get a
    shared stop signal and are joined within the same grace deadline. Because
    ``run_worker`` has no external stop parameter, a thread already executing
    a Job may outlive that deadline, but its generation remains fenced from
    all later handoffs. Every tracked subprocess receives ``terminate`` first;
    processes still alive after the deadline receive ``kill``.
    """
    timeout_seconds = max(0.0, float(timeout))
    with _WORKER_PROCESS_SHUTDOWN_LOCK:
        global _WORKER_ADMISSION_OPEN
        global _WORKER_LIFECYCLE_GENERATION
        with _WORKER_LIFECYCLE_LOCK:
            if _WORKER_ADMISSION_OPEN:
                _WORKER_ADMISSION_OPEN = False
                _WORKER_LIFECYCLE_GENERATION += 1
            _WORKER_LIFECYCLE_STOP_EVENT.set()
            tracked_at_start = len(_WORKER_PROCESSES)
            threads_tracked_at_start = len(_LOCAL_WORKER_THREADS)
        terminate_requested = 0
        kill_requested = 0
        targets = _worker_process_snapshot()
        deadline = time.monotonic() + timeout_seconds
        for process in targets:
            if _process_exited(process):
                _unregister_worker_process(process)
                continue
            try:
                process.terminate()
                terminate_requested += 1
            except (ChildProcessError, ProcessLookupError):
                _unregister_worker_process(process)
            except Exception:
                # Keep the process registered so the kill pass can retry.
                pass

        for thread in _local_worker_thread_snapshot():
            if thread is threading.current_thread():
                continue
            if not thread.is_alive():
                _unregister_local_worker_thread(thread)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            thread.join(timeout=remaining)
            if not thread.is_alive():
                _unregister_local_worker_thread(thread)

        for process in targets:
            if _process_exited(process):
                _unregister_worker_process(process)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            _wait_for_worker_process(process, remaining)

        # Snapshot again so children whose Popen overlapped admission closure
        # cannot escape the hard-stop pass.
        survivors = _worker_process_snapshot()
        for process in survivors:
            if _process_exited(process):
                _unregister_worker_process(process)
                continue
            try:
                process.kill()
                kill_requested += 1
            except (ChildProcessError, ProcessLookupError):
                _unregister_worker_process(process)
            except Exception:
                pass

        kill_deadline = time.monotonic() + min(1.0, max(0.1, timeout_seconds))
        for process in survivors:
            if _process_exited(process):
                _unregister_worker_process(process)
                continue
            remaining = kill_deadline - time.monotonic()
            if remaining <= 0.0:
                break
            _wait_for_worker_process(process, remaining)

        with _WORKER_LIFECYCLE_LOCK:
            for thread in list(_LOCAL_WORKER_THREADS.values()):
                if not thread.is_alive():
                    _LOCAL_WORKER_THREADS.pop(id(thread), None)
            remaining_count = len(_WORKER_PROCESSES)
            threads_remaining = len(_LOCAL_WORKER_THREADS)
        return {
            "tracked": tracked_at_start,
            "terminate_requested": terminate_requested,
            "kill_requested": kill_requested,
            "remaining": remaining_count,
            "threads_tracked": threads_tracked_at_start,
            "threads_remaining": threads_remaining,
        }


def _spawn_worker_process(command: list[str], log_path: Path) -> subprocess.Popen:
    admission = _admitted_worker_generation()
    if admission is None:
        raise _WorkerAdmissionClosed("local worker admission is closed")
    spawn_generation, _stop_event = admission
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    src_root = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_root if not existing_pythonpath else f"{src_root}{os.pathsep}{existing_pythonpath}"
    log_handle = log_path.open("ab")
    try:
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    finally:
        log_handle.close()
    return _register_worker_process(process, spawn_generation=spawn_generation)


def _shutdown_worker_processes_at_exit() -> None:
    try:
        shutdown_worker_processes(timeout=2.0)
    except Exception:
        pass


atexit.register(_shutdown_worker_processes_at_exit)
