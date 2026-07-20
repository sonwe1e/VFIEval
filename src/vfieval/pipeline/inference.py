from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image, ImageChops

from vfieval.config import WorkspaceConfig
from vfieval.db import Database, artifact_storage_metadata
from vfieval.devices import autocast_context, resolve_torch_device, tune_for_inference
from vfieval.models import load_flow_mask_model
from vfieval.pipeline.artifact_integrity import (
    ArtifactIntegrityError,
    merge_integrity_reports,
    require_job_artifact_integrity,
    validate_job_artifact_integrity,
    validate_video_artifact_integrity,
    write_integrity_report,
)
from vfieval.pipeline.io import batch_tensors, load_rgb_tensor, resize_batch
from vfieval.pipeline.postprocess import (
    compose_interpolated,
)
from vfieval.pipeline.visualize import save_difference, save_extra_tensor, save_preview_image, save_rgb_tensor


VALID_PRECISIONS = {"fp32", "fp16", "bf16"}
ARTIFACT_PROFILES = {"evaluation", "diagnostic", "benchmark"}
DEFAULT_VISUALIZE_HEIGHT = 832
DEFAULT_VISUALIZE_WIDTH = 1792
ARTIFACT_CONTRACT = "canonical-v1"
POSTPROCESS_MAX_PIXELS = 8_388_608
CORE_OUTPUTS = {"flowt_0", "flowt_1", "mask0", "mask1"}


class RunCanceled(RuntimeError):
    pass


class CoreArtifactSaveError(RuntimeError):
    """A required per-sample artifact could not be written or registered."""

    def __init__(self, sample_id: int, sample_name: str, cause: BaseException):
        self.sample_id = int(sample_id)
        self.sample_name = str(sample_name)
        self.cause = cause
        super().__init__(f"failed to publish core artifacts for sample {sample_name} ({sample_id}): {cause}")


class ArtifactSaveAggregateError(RuntimeError):
    """All bounded async save work drained, with one or more failures."""

    def __init__(self, failures: list[BaseException]):
        self.failures = tuple(failures)
        first = failures[0]
        super().__init__(
            f"{len(failures)} artifact save operation(s) failed; first error: "
            f"{type(first).__name__}: {first}"
        )


@dataclass(frozen=True)
class InferenceJobResult:
    samples: int
    output_dir: str
    decode_fps: float
    model_fps: float
    postprocess_fps: float
    save_fps: float
    model_load: dict[str, Any] | None = None
    output_health: dict[str, Any] | None = None
    prefetch_wait_seconds: float = 0.0
    save_backlog_seconds: float = 0.0
    performance: dict[str, Any] | None = None
    artifact_integrity: dict[str, Any] | None = None


class _DeviceEventTimings:
    """Collect asynchronous CUDA/NPU event durations with one final sync."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._module = None
        self._pairs: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        if device.type == "cuda" and hasattr(torch, "cuda"):
            self._module = torch.cuda
        elif device.type == "npu" and hasattr(torch, "npu"):
            self._module = torch.npu

    def start(self) -> Any | None:
        if self._module is None or not hasattr(self._module, "Event"):
            return None
        try:
            event = self._module.Event(enable_timing=True)
            event.record()
            return event
        except Exception:
            self._module = None
            return None

    def stop(self, stage: str, start: Any | None) -> None:
        if self._module is None or start is None:
            return
        try:
            end = self._module.Event(enable_timing=True)
            end.record()
            self._pairs[stage].append((start, end))
        except Exception:
            self._module = None
            self._pairs.clear()

    def result(self) -> dict[str, float]:
        if self._module is None or not self._pairs:
            return {}
        try:
            self._module.synchronize()
            return {
                stage: sum(float(start.elapsed_time(end)) for start, end in pairs) / 1000.0
                for stage, pairs in self._pairs.items()
            }
        except Exception:
            return {}


class _NpuSmiSampler:
    """Best-effort low-rate Ascend utilization sampling; failures stay optional."""

    def __init__(self, device: torch.device, enabled: bool = True) -> None:
        self._command = shutil.which("npu-smi") if enabled and device.type == "npu" else None
        self._device_index = int(device.index or 0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, float]] = []

    def start(self) -> None:
        if self._command is None:
            return
        self._thread = threading.Thread(target=self._run, name="vfieval-npu-smi", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def result(self) -> dict[str, Any] | None:
        if not self._samples:
            return None
        keys = sorted({key for sample in self._samples for key in sample})
        return {
            "sample_count": len(self._samples),
            "averages": {
                key: sum(sample[key] for sample in self._samples if key in sample)
                / sum(1 for sample in self._samples if key in sample)
                for key in keys
            },
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [str(self._command), "info", "-t", "usages", "-i", str(self._device_index)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=4,
                )
                if result.returncode == 0:
                    sample = self._parse(result.stdout)
                    if sample:
                        self._samples.append(sample)
            except Exception:
                return
            self._stop.wait(1.0)

    @staticmethod
    def _parse(text: str) -> dict[str, float]:
        sample: dict[str, float] = {}
        for line in str(text).splitlines():
            lowered = line.lower()
            values = [float(value) for value in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", line)]
            if not values:
                continue
            if "aicore" in lowered or "ai core" in lowered or "utilization rate" in lowered:
                sample["aicore_percent"] = values[-1]
            elif "memory" in lowered and ("usage" in lowered or "utilization" in lowered):
                sample["memory_percent"] = values[-1]
        return sample


def _extract_model_load_report(model: Any) -> dict[str, Any] | None:
    for candidate in (
        model,
        getattr(model, "_infer", None),
        getattr(model, "model", None),
        getattr(model, "net", None),
        getattr(model, "network", None),
        getattr(model, "module", None),
    ):
        if candidate is None:
            continue
        try:
            report = getattr(candidate, "_last_load_report", None)
        except Exception:
            report = None
        if isinstance(report, dict):
            return report
        owner = getattr(candidate, "__self__", None)
        if owner is not None:
            try:
                owner_report = getattr(owner, "_last_load_report", None)
            except Exception:
                owner_report = None
            if isinstance(owner_report, dict):
                return owner_report
    return None


def _write_model_load_log(run_dir: Path, report: dict[str, Any]) -> None:
    if not report:
        return
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"checkpoint: {report.get('checkpoint_path')}",
        f"matched: {report.get('matched')} / {report.get('total_in_checkpoint')}",
        f"missing_keys: {len(report.get('missing_keys') or [])}",
        f"unexpected_keys: {len(report.get('unexpected_keys') or [])}",
    ]
    for key in list(report.get("missing_keys") or [])[:100]:
        lines.append(f"  - missing: {key}")
    for key in list(report.get("unexpected_keys") or [])[:100]:
        lines.append(f"  - unexpected: {key}")
    (logs_dir / "model_load.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


class _OutputHealthAccumulator:
    def __init__(self) -> None:
        self._flow: dict[str, dict[str, float]] = {
            name: {"count": 0.0, "abs_sum": 0.0, "abs_max": 0.0, "nan_count": 0.0}
            for name in ("flowt_0", "flowt_1")
        }
        self._mask: dict[str, dict[str, float]] = {
            name: {"count": 0.0, "sum": 0.0, "sum_sq": 0.0, "nan_count": 0.0}
            for name in ("mask0", "mask1")
        }
        self._samples = 0

    def update(self, bundle_cpu: dict[str, torch.Tensor]) -> None:
        pred = bundle_cpu.get("pred")
        if isinstance(pred, torch.Tensor) and pred.ndim > 0:
            self._samples += int(pred.shape[0])
        for name in ("flowt_0", "flowt_1"):
            self._update_flow(name, bundle_cpu.get(name))
        for name in ("mask0", "mask1"):
            self._update_mask(name, bundle_cpu.get(name))

    def _update_flow(self, name: str, tensor: torch.Tensor | None) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        values = tensor.detach().float()
        stats = self._flow[name]
        stats["nan_count"] += float(torch.isnan(values).sum().item())
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return
        abs_values = finite.abs()
        stats["count"] += float(abs_values.numel())
        stats["abs_sum"] += float(abs_values.sum().item())
        stats["abs_max"] = max(stats["abs_max"], float(abs_values.max().item()))

    def _update_mask(self, name: str, tensor: torch.Tensor | None) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        values = tensor.detach().float()
        stats = self._mask[name]
        stats["nan_count"] += float(torch.isnan(values).sum().item())
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return
        stats["count"] += float(finite.numel())
        stats["sum"] += float(finite.sum().item())
        stats["sum_sq"] += float((finite * finite).sum().item())

    def to_dict(self) -> dict[str, Any]:
        stats: dict[str, dict[str, float | int]] = {}
        for name, raw in self._flow.items():
            count = int(raw["count"])
            stats[name] = {
                "abs_mean": float(raw["abs_sum"] / count) if count else 0.0,
                "abs_max": float(raw["abs_max"]),
                "nan_count": int(raw["nan_count"]),
            }
        for name, raw in self._mask.items():
            count = int(raw["count"])
            mean = float(raw["sum"] / count) if count else 0.0
            variance = max(0.0, float(raw["sum_sq"] / count) - mean * mean) if count else 0.0
            stats[name] = {
                "mean": mean,
                "std": float(variance ** 0.5),
                "nan_count": int(raw["nan_count"]),
            }
        flow_flat = all(float(stats[name]["abs_max"]) < 1e-4 for name in ("flowt_0", "flowt_1"))
        mask_flat = all(float(stats[name]["std"]) < 1e-3 for name in ("mask0", "mask1"))
        has_nan = any(int(stats[name]["nan_count"]) > 0 for name in stats)
        warnings: list[str] = []
        if has_nan:
            warnings.append("model output contains NaN on real inference frames")
        if flow_flat and mask_flat:
            warnings.append(
                "flow ~= 0 and mask ~= constant on real inference frames; checkpoint may be loaded but model outputs are semantically empty"
            )
        return {
            "stats": stats,
            "warnings": warnings,
            "flow_flat": flow_flat,
            "mask_flat": mask_flat,
            "has_nan": has_nan,
            "samples": self._samples,
        }


def _write_output_health_log(run_dir: Path, report: dict[str, Any]) -> None:
    if not report:
        return
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"samples: {report.get('samples', 0)}"]
    for name in ("flowt_0", "flowt_1", "mask0", "mask1"):
        stats = (report.get("stats") or {}).get(name) or {}
        if "abs_max" in stats:
            lines.append(
                f"{name}: abs_mean={stats.get('abs_mean', 0.0):.8g} abs_max={stats.get('abs_max', 0.0):.8g} nan_count={stats.get('nan_count', 0)}"
            )
        else:
            lines.append(
                f"{name}: mean={stats.get('mean', 0.0):.8g} std={stats.get('std', 0.0):.8g} nan_count={stats.get('nan_count', 0)}"
            )
    for warning in report.get("warnings") or []:
        lines.append(f"warning: {warning}")
    (logs_dir / "output_health.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sanitize_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return clean or "sample"


def _unique_sanitized_tokens(entries: list[tuple[str, str]]) -> dict[str, str]:
    """Return stable path components without changing non-colliding names."""

    normalized = [(str(identity), sanitize_name(str(label))) for identity, label in entries]
    counts = defaultdict(int)
    for _identity, base in normalized:
        counts[base] += 1

    resolved = {
        identity: base
        for identity, base in normalized
        if counts[base] == 1
    }
    used = set(resolved.values())
    for identity, base in sorted(normalized, key=lambda item: (item[1], item[0])):
        if counts[base] == 1:
            continue
        digest = hashlib.sha256(f"{identity}\0{base}".encode("utf-8")).hexdigest()
        digest_length = 12
        candidate = f"{base}__{digest[:digest_length]}"
        while candidate in used and digest_length < len(digest):
            digest_length += 4
            candidate = f"{base}__{digest[:digest_length]}"
        suffix = 2
        unique_candidate = candidate
        while unique_candidate in used:
            unique_candidate = f"{candidate}_{suffix}"
            suffix += 1
        resolved[identity] = unique_candidate
        used.add(unique_candidate)
    return resolved


def resolve_device(device_name: str) -> torch.device:
    return resolve_torch_device(device_name)


def _autocast_context(device: torch.device, precision: str):
    return autocast_context(device, precision)


def _artifact_mime(kind: str) -> str:
    if kind.endswith("video"):
        return "video/mp4"
    return "image/png"


def run_inference_job(db: Database, workspace: WorkspaceConfig, job_id: int) -> InferenceJobResult:
    total_wall_start = time.perf_counter()
    job = db.get_job(job_id)
    payload = job["payload"]
    model_id = int(payload["model_id"])
    dataset_id = int(payload["dataset_id"])
    height = int(payload.get("height") or payload.get("input_height") or 0)
    width = int(payload.get("width") or payload.get("input_width") or 0)
    batch_size = int(payload.get("batch_size", 1))
    device = resolve_device(str(payload.get("device", "auto")))
    precision = str(payload.get("precision", "fp32"))
    artifact_profile = str(payload.get("artifact_profile") or "evaluation")
    metric_names = list(payload.get("metrics", []))
    run_id = int(payload["run_id"]) if payload.get("run_id") is not None else None
    shard_count = int(payload.get("shard_count") or 1)
    is_shard = shard_count > 1
    run = db.get_run(run_id) if run_id is not None else None
    run_metadata = dict((run or {}).get("metadata") or {})

    if precision == "auto":
        precision = "fp16" if device.type in {"cuda", "npu"} else "fp32"
    if precision in {"fp16", "bf16"} and device.type not in {"cuda", "npu"}:
        precision = "fp32"
    if precision not in VALID_PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(VALID_PRECISIONS)}, got {precision}")
    if artifact_profile not in ARTIFACT_PROFILES:
        raise ValueError(f"artifact_profile must be one of {sorted(ARTIFACT_PROFILES)}")
    if artifact_profile == "benchmark" and metric_names:
        raise ValueError("benchmark artifact_profile does not run metrics")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    visualize_height, visualize_width = _resolve_visualize_size(payload, height, width)

    samples = db.list_samples(dataset_id)
    sample_ids = payload.get("sample_ids")
    if sample_ids is not None:
        allowed = {int(sample_id) for sample_id in sample_ids}
        samples = [sample for sample in samples if int(sample["id"]) in allowed]
    if artifact_profile == "benchmark":
        samples = samples[: max(1, int(payload.get("benchmark_samples") or 200))]
    if not samples:
        raise ValueError(f"dataset {dataset_id} has no samples")

    if str(run_metadata.get("run_type") or payload.get("run_type") or "model_inference") == "video_compare":
        return _run_video_compare_job(
            db=db,
            workspace=workspace,
            job_id=job_id,
            run_id=run_id,
            job=job,
            run=run,
            samples=samples,
            metric_names=metric_names,
            is_shard=is_shard,
            dataset_id=dataset_id,
        )

    model_row = db.get_model(model_id)
    run_dir = workspace.runs_dir / (str(run_id) if run_id is not None else f"inference_{job_id:06d}")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(run_dir, job, model_row, db.get_dataset(dataset_id) if dataset_id else None)

    _require_progress_cas(db, run_id, job_id, db.update_job_progress(job_id, 0, len(samples)), "Job")
    if run_id is not None:
        if not db.mark_run_started(run_id, "running"):
            _raise_if_canceled(db, run_id, job_id)
            raise RuntimeError(f"run {run_id} rejected inference start from status {db.get_run(run_id)['status']}")
        if is_shard:
            _require_progress_cas(
                db, run_id, job_id, db.update_run_progress_from_jobs(run_id), "Run"
            )
        else:
            _require_progress_cas(
                db, run_id, job_id, db.update_run_progress(run_id, 0, len(samples)), "Run"
            )
    model = load_flow_mask_model(
        adapter=model_row["adapter"],
        checkpoint_path=model_row.get("checkpoint_path"),
        device=str(device),
        metadata=model_row.get("metadata") or {},
    )
    model_load_report = _extract_model_load_report(model)
    tune_for_inference(device)
    startup_seconds = time.perf_counter() - total_wall_start
    _reset_peak_memory(device)
    device_events = _DeviceEventTimings(device)
    npu_smi = _NpuSmiSampler(device, enabled=bool(payload.get("sample_npu_smi", True)))
    npu_smi.start()

    if model_load_report is not None:
        _write_model_load_log(run_dir, model_load_report)

    # Decode and save pools scale with the cores actually available to this
    # shard. NPU shards are independent processes and CUDA shards are threads in
    # one process, so dividing by shard_count keeps the machine-wide thread
    # count near the core count either way. The 48 ceiling is where GIL
    # contention flattens the return on PIL/numpy/torch decode threads (they all
    # release the GIL, but the glue between them does not).
    cores = os.cpu_count() or 8
    per_shard = max(1, cores // max(1, shard_count))
    prefetch_workers = _resolve_pool_size(payload.get("prefetch_workers"), min(2, per_shard), lo=1, hi=2)
    save_workers = _resolve_pool_size(payload.get("save_workers"), min(8, per_shard), lo=1, hi=8)
    save_warp_blend = artifact_profile == "diagnostic" or bool(payload.get("save_warp_blend", False))
    max_save_inflight = int(payload.get("max_save_inflight") or max(2, save_workers * 2))
    pipeline = _AsyncSavePipeline(
        db=db,
        job_id=job_id,
        run_id=run_id,
        is_shard=is_shard,
        run_dir=run_dir,
        decode_cache_root=workspace.root / "decode_cache",
        save_workers=save_workers,
        max_inflight=max_save_inflight,
        artifact_batch_size=int(payload.get("artifact_db_batch_size") or 128),
        preview_height=visualize_height,
        preview_width=visualize_width,
    )

    timing = {"decode": 0.0, "model": 0.0, "post": 0.0, "save": 0.0}
    output_health = _OutputHealthAccumulator()
    direct_processed = 0
    steady_start = time.perf_counter()
    benchmark_warmed = False
    postprocess_chunk_size = _postprocess_chunk_size(batch_size, height, width)

    try:
        for batch_rows, img0_cpu, img1_cpu, gt_cpu_list, prefetch_wait in _iter_prefetched_batches(
            samples=samples,
            batch_size=batch_size,
            height=height,
            width=width,
            has_gt=artifact_profile != "benchmark",
            workers=prefetch_workers,
        ):
            _raise_if_canceled(db, run_id, job_id)
            pipeline.raise_if_failed()
            timing["decode"] += prefetch_wait

            t1 = time.perf_counter()
            model_event = device_events.start()
            img0 = img0_cpu.to(device, non_blocking=True)
            img1 = img1_cpu.to(device, non_blocking=True)
            if artifact_profile == "benchmark" and not benchmark_warmed:
                warmup_batches = max(0, int(payload.get("benchmark_warmup_batches") or 10))
                with torch.no_grad(), _autocast_context(device, precision):
                    for _ in range(warmup_batches):
                        warm_outputs = model.predict(img0, img1, 0.5)
                        for _start, _end, _chunk_outputs, warm_composed in _compose_canonical_chunks(
                            img0,
                            img1,
                            warm_outputs,
                            initial_chunk_size=postprocess_chunk_size,
                        ):
                            del warm_composed
                module = _device_module(device)
                if module is not None:
                    try:
                        module.synchronize()
                    except Exception:
                        pass
                benchmark_warmed = True
                timing = {"decode": 0.0, "model": 0.0, "post": 0.0, "save": 0.0}
                device_events = _DeviceEventTimings(device)
                steady_start = time.perf_counter()
                t1 = time.perf_counter()
                model_event = device_events.start()
            with torch.no_grad(), _autocast_context(device, precision):
                outputs = model.predict(img0, img1, 0.5)
            device_events.stop("transfer_and_model", model_event)
            timing["model"] += time.perf_counter() - t1
            # Surface any save-pool failure that accumulated during the GPU
            # forward pass before we start chunk composition for this batch.
            pipeline.raise_if_failed()

            t2 = time.perf_counter()
            post_event = device_events.start()
            if artifact_profile == "benchmark":
                for _start, _end, _chunk_outputs, composed in _compose_canonical_chunks(
                    img0,
                    img1,
                    outputs,
                    initial_chunk_size=postprocess_chunk_size,
                ):
                    del composed
                device_events.stop("postprocess", post_event)
                direct_processed += len(batch_rows)
                timing["post"] += time.perf_counter() - t2
                _require_progress_cas(
                    db, run_id, job_id, db.update_job_progress(job_id, direct_processed), "Job"
                )
                if run_id is not None:
                    if is_shard:
                        _require_progress_cas(
                            db, run_id, job_id, db.update_run_progress_from_jobs(run_id), "Run"
                        )
                    else:
                        _require_progress_cas(
                            db, run_id, job_id, db.update_run_progress(run_id, direct_processed), "Run"
                        )
                continue
            # Pred/warp/blend are canonical output-resolution artifacts.
            # Native flow/mask stay compact; visualize_* only controls previews.
            for start, end, chunk_outputs, composed in _compose_canonical_chunks(
                img0,
                img1,
                outputs,
                initial_chunk_size=postprocess_chunk_size,
            ):
                pipeline.raise_if_failed()
                device_bundle: dict[str, torch.Tensor] = {
                    "pred": composed["pred"],
                    "mask0": torch.sigmoid(chunk_outputs["mask0"]),
                    "mask1": torch.sigmoid(chunk_outputs["mask1"]),
                    "flowt_0": chunk_outputs["flowt_0"],
                    "flowt_1": chunk_outputs["flowt_1"],
                }
                if save_warp_blend:
                    device_bundle.update({name: composed[name] for name in ("warp0", "warp1", "blend")})
                device_extra: dict[str, torch.Tensor] = {}
                if artifact_profile == "diagnostic":
                    for name, tensor in chunk_outputs.items():
                        if name in CORE_OUTPUTS or not isinstance(tensor, torch.Tensor):
                            continue
                        if tensor.ndim > 0 and int(tensor.shape[0]) == end - start:
                            device_extra[name] = tensor
                transferred = _detach_tensors_to_cpu(
                    {**device_bundle, **{f"extra::{name}": tensor for name, tensor in device_extra.items()}}
                )
                health_bundle = {
                    name: transferred[name]
                    for name in ("pred", "mask0", "mask1", "flowt_0", "flowt_1")
                }
                output_health.update(health_bundle)
                bundle_cpu = {"pred": transferred["pred"]}
                if artifact_profile == "diagnostic":
                    bundle_cpu.update(
                        {name: transferred[name] for name in ("mask0", "mask1", "flowt_0", "flowt_1")}
                    )
                if save_warp_blend:
                    bundle_cpu.update({name: transferred[name] for name in ("warp0", "warp1", "blend")})
                extra_cpu = {name: transferred[f"extra::{name}"] for name in device_extra}

                t3 = time.perf_counter()
                pipeline.submit_batch(
                    batch_rows=batch_rows[start:end],
                    bundle_cpu=bundle_cpu,
                    extra_cpu=extra_cpu,
                    gt_cpu_list=gt_cpu_list[start:end],
                )
                timing["save"] += time.perf_counter() - t3
                del composed, transferred
            device_events.stop("postprocess", post_event)
            timing["post"] += time.perf_counter() - t2

        backlog_start = time.perf_counter()
        pipeline.wait_for_all()
        save_backlog_seconds = time.perf_counter() - backlog_start
        pipeline.shutdown()
    except BaseException as exc:
        pipeline.shutdown(suppress_errors=True)
        npu_smi.stop()
        try:
            _log_failed_job_integrity(
                db,
                run_dir,
                job_id,
                [int(sample["id"]) for sample in samples],
                exc,
                phase="core_artifact_pipeline",
            )
        except Exception:
            pass
        raise
    processed = pipeline.processed_count + direct_processed
    video_groups = pipeline.video_groups
    npu_smi.stop()

    expected_sample_ids = [int(sample["id"]) for sample in samples]
    integrity_report = _require_and_log_job_integrity(db, run_dir, job_id, expected_sample_ids)

    if video_groups:
        if is_shard and bool(payload.get("defer_video_finalize")):
            _write_shard_video_manifest(
                run_dir,
                run_id,
                job_id,
                video_groups,
                expected_sample_ids=expected_sample_ids,
                successful_sample_ids=[int(value) for value in integrity_report["successful_sample_ids"]],
                core_artifact_counts=dict(integrity_report["core_artifact_counts"]),
            )
        else:
            _raise_if_canceled(db, run_id, job_id)
            integrity_report = _encode_and_validate_video_artifacts(
                db,
                job_id,
                run_dir,
                video_groups,
                integrity_report,
                expected_sample_ids=expected_sample_ids,
                preview_height=visualize_height,
                preview_width=visualize_width,
            )
            _raise_if_canceled(db, run_id, job_id)

    device_timing = device_events.result()
    steady_seconds = time.perf_counter() - steady_start
    total_wall_seconds = time.perf_counter() - total_wall_start
    performance = {
        "artifact_profile": artifact_profile,
        "artifact_contract": ARTIFACT_CONTRACT,
        "startup_seconds": startup_seconds,
        "steady_state_seconds": steady_seconds,
        "total_wall_seconds": total_wall_seconds,
        "end_to_end_fps": _fps(processed, total_wall_seconds),
        "steady_state_fps": _fps(processed, steady_seconds),
        "prefetch_wait_seconds": timing["decode"],
        "save_backpressure_seconds": pipeline.backpressure_seconds,
        "save_backlog_seconds": save_backlog_seconds,
        "save_max_inflight": pipeline.max_observed_inflight,
        "artifact_db_batches": pipeline.artifact_db_batches,
        "device_seconds": device_timing,
        "device_memory": _peak_memory(device),
        "device_name": _device_name(device),
        "npu_smi": npu_smi.result(),
        "batch_size": batch_size,
        "prefetch_workers": prefetch_workers,
        "save_workers": save_workers,
        "postprocess_chunk_size": postprocess_chunk_size,
    }

    output_health_report = None if artifact_profile == "benchmark" else output_health.to_dict()
    if output_health_report is not None:
        _write_output_health_log(run_dir, output_health_report)

    result = InferenceJobResult(
        samples=processed,
        output_dir=str(run_dir),
        decode_fps=_fps(processed, timing["decode"]),
        model_fps=_fps(processed, timing["model"]),
        postprocess_fps=_fps(processed, timing["post"]),
        save_fps=_fps(processed, timing["save"]),
        model_load=model_load_report,
        output_health=output_health_report,
        prefetch_wait_seconds=timing["decode"],
        save_backlog_seconds=save_backlog_seconds,
        performance=performance,
        artifact_integrity=integrity_report,
    )

    result_dict = dict(result.__dict__)
    if model_load_report is not None:
        result_dict["model_load"] = model_load_report
    artifact_summary = db.summarize_artifacts(job_id)

    if is_shard:
        return result

    if run_id is not None:
        _publish_run_media_strict(
            db,
            workspace,
            run_id,
            job_id,
            run_dir,
            integrity_report,
        )

    if metric_names:
        if run_id is not None:
            from vfieval.pipeline.metric_jobs import create_metric_wave

            try:
                create_metric_wave(
                    db,
                    run_id,
                    metric_names,
                    source="inference",
                    result=result_dict,
                    artifact_summary=artifact_summary,
                    source_job_id=job_id,
                    source_job_result=result_dict,
                )
            except ValueError:
                db.invalidate_run_media_assets(run_id)
                _raise_if_canceled(db, run_id, job_id)
                raise
        else:
            db.create_job(
                "metric",
                {
                    "inference_job_id": job_id,
                    "dataset_id": dataset_id,
                    "metric_names": metric_names,
                    "metric_device": str(device),
                },
            )
    elif run_id is not None:
        if not db.complete_run_inference(
            run_id,
            result_dict,
            artifact_summary,
            "completed",
            source_job_id=job_id,
            source_job_result=result_dict,
        ):
            db.invalidate_run_media_assets(run_id)
            _raise_if_canceled(db, run_id, job_id)
            raise RuntimeError(f"run {run_id} rejected completion from status {db.get_run(run_id)['status']}")

    return result


def _run_video_compare_job(
    db: Database,
    workspace: WorkspaceConfig,
    job_id: int,
    run_id: int | None,
    job: dict[str, Any],
    run: dict[str, Any] | None,
    samples: list[dict[str, Any]],
    metric_names: list[str],
    is_shard: bool,
    dataset_id: int,
) -> InferenceJobResult:
    run_dir = workspace.runs_dir / (str(run_id) if run_id is not None else f"inference_{job_id:06d}")
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset = db.get_dataset(dataset_id) if dataset_id else None
    _write_run_metadata(run_dir, job, db.get_model(int(job["payload"]["model_id"])), dataset)
    alignment_plan = dict(((dataset or {}).get("metadata") or {}).get("alignment_plan") or {})
    if alignment_plan:
        _write_compare_alignment_report(db, job_id, run_dir, alignment_plan)

    run_metadata = (run or {}).get("metadata") or {}
    request_metadata = run_metadata.get("request") or {}
    canonical_height = int((run or {}).get("height") or 0)
    canonical_width = int((run or {}).get("width") or 0)
    preview_height, preview_width = _resolve_visualize_size(
        request_metadata,
        canonical_height,
        canonical_width,
    )
    publish_pred_video = bool(
        run_metadata.get(
            "publish_compare_pred_video",
            request_metadata.get("publish_compare_pred_video", True),
        )
    )

    _require_progress_cas(db, run_id, job_id, db.update_job_progress(job_id, 0, len(samples)), "Job")
    if run_id is not None:
        if not db.mark_run_started(run_id, "running"):
            _raise_if_canceled(db, run_id, job_id)
            raise RuntimeError(f"run {run_id} rejected Compare start from status {db.get_run(run_id)['status']}")
        if is_shard:
            _require_progress_cas(
                db, run_id, job_id, db.update_run_progress_from_jobs(run_id), "Run"
            )
        else:
            _require_progress_cas(
                db, run_id, job_id, db.update_run_progress(run_id, 0, len(samples)), "Run"
            )

    processed = 0
    video_groups: dict[str, dict[str, Any]] = {}
    save_seconds = 0.0
    for row in samples:
        _raise_if_canceled(db, run_id, job_id)
        t0 = time.perf_counter()
        try:
            sample_dir = run_dir / f"{int(row['id']):08d}_{sanitize_name(row['name'])}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            gt_output_path = _copy_compare_image(Path(row["gt_path"]), sample_dir / "gt.png")
            pred_output_path = _copy_compare_image(Path(row["img1_path"]), sample_dir / "pred.png")
            diff_output_path = sample_dir / "difference.png"
            with Image.open(gt_output_path).convert("RGB") as gt_image, Image.open(pred_output_path).convert("RGB") as pred_image:
                ImageChops.difference(pred_image, gt_image).save(diff_output_path)

            with Image.open(pred_output_path) as canonical_image:
                canonical_width, canonical_height = canonical_image.size
            artifact_metadata = {
                "sample": row["name"],
                "artifact_contract": ARTIFACT_CONTRACT,
                "canonical_height": int(canonical_height),
                "canonical_width": int(canonical_width),
                **_compare_track_metadata(row),
            }
            make_preview = (canonical_height, canonical_width) != (preview_height, preview_width)
            db.add_artifacts_bulk(
                job_id,
                [
                    _image_artifact_record(
                        int(row["id"]),
                        kind,
                        path,
                        artifact_metadata,
                        make_preview=make_preview,
                        preview_height=preview_height,
                        preview_width=preview_width,
                    )
                    for kind, path in (
                        ("gt", gt_output_path),
                        ("pred", pred_output_path),
                        ("difference", diff_output_path),
                    )
                ],
            )
            _collect_compare_frame(video_groups, row, gt_output_path, pred_output_path, diff_output_path)
        except RunCanceled:
            raise
        except Exception as exc:
            try:
                _record_sample_error(db, job_id, int(row["id"]), row["name"], exc)
            except Exception:
                pass
            wrapped = CoreArtifactSaveError(int(row["id"]), str(row["name"]), exc)
            try:
                _log_failed_job_integrity(
                    db,
                    run_dir,
                    job_id,
                    [int(sample["id"]) for sample in samples],
                    wrapped,
                    phase="core_artifact_pipeline",
                )
            except Exception:
                pass
            raise wrapped from exc

        processed += 1
        _require_progress_cas(
            db, run_id, job_id, db.update_job_progress(job_id, processed), "Job"
        )
        if run_id is not None:
            if is_shard:
                _require_progress_cas(
                    db, run_id, job_id, db.update_run_progress_from_jobs(run_id), "Run"
                )
            else:
                _require_progress_cas(
                    db, run_id, job_id, db.update_run_progress(run_id, processed), "Run"
                )
        save_seconds += time.perf_counter() - t0

    expected_sample_ids = [int(sample["id"]) for sample in samples]
    integrity_report = _require_and_log_job_integrity(db, run_dir, job_id, expected_sample_ids)
    if video_groups:
        _raise_if_canceled(db, run_id, job_id)
        integrity_report = _encode_and_validate_video_artifacts(
            db,
            job_id,
            run_dir,
            video_groups,
            integrity_report,
            publish_pred_video=publish_pred_video,
            expected_sample_ids=expected_sample_ids,
            preview_height=preview_height,
            preview_width=preview_width,
        )
        _raise_if_canceled(db, run_id, job_id)

    result = InferenceJobResult(
        samples=processed,
        output_dir=str(run_dir),
        decode_fps=0.0,
        model_fps=0.0,
        postprocess_fps=0.0,
        save_fps=_fps(processed, save_seconds),
        artifact_integrity=integrity_report,
    )
    artifact_summary = db.summarize_artifacts(job_id)
    if is_shard:
        return result
    if run_id is not None:
        _publish_run_media_strict(
            db,
            workspace,
            run_id,
            job_id,
            run_dir,
            integrity_report,
        )
    if metric_names:
        if run_id is not None:
            from vfieval.pipeline.metric_jobs import create_metric_wave

            try:
                create_metric_wave(
                    db,
                    run_id,
                    metric_names,
                    source="compare",
                    result=result.__dict__,
                    artifact_summary=artifact_summary,
                    source_job_id=job_id,
                    source_job_result=result.__dict__,
                )
            except ValueError:
                db.invalidate_run_media_assets(run_id)
                _raise_if_canceled(db, run_id, job_id)
                raise
        else:
            db.create_job(
                "metric",
                {
                    "inference_job_id": job_id,
                    "dataset_id": dataset_id,
                    "metric_names": metric_names,
                    "metric_device": str(job.get("payload", {}).get("device") or "cpu"),
                },
            )
    elif run_id is not None:
        if not db.complete_run_inference(
            run_id,
            result.__dict__,
            artifact_summary,
            "completed",
            source_job_id=job_id,
            source_job_result=result.__dict__,
        ):
            db.invalidate_run_media_assets(run_id)
            _raise_if_canceled(db, run_id, job_id)
            raise RuntimeError(f"run {run_id} rejected completion from status {db.get_run(run_id)['status']}")
    return result


def _resolve_visualize_size(payload: dict[str, Any], height: int, width: int) -> tuple[int, int]:
    """Resolution of optional preview derivatives.

    Canonical artifacts always use the inference resolution. Explicit preview
    dimensions are exact; defaults avoid gratuitously upscaling small Runs.
    """
    raw_h = payload.get("visualize_height")
    raw_w = payload.get("visualize_width")
    vis_h = int(raw_h) if raw_h else min(DEFAULT_VISUALIZE_HEIGHT, height)
    vis_w = int(raw_w) if raw_w else min(DEFAULT_VISUALIZE_WIDTH, width)
    if vis_h <= 0:
        vis_h = min(DEFAULT_VISUALIZE_HEIGHT, height)
    if vis_w <= 0:
        vis_w = min(DEFAULT_VISUALIZE_WIDTH, width)
    return vis_h, vis_w


def _resize_chw(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a CHW RGB tensor to (height, width) via bilinear interpolation."""
    if tuple(tensor.shape[-2:]) == (height, width):
        return tensor
    resized = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0)


def _resolve_pool_size(override: Any, per_shard: int, *, lo: int, hi: int) -> int:
    """Resolve a worker-pool size from an optional payload override.

    An explicit payload value wins (still clamped to >= 1). Otherwise the pool
    scales with the cores available to this shard, bounded to [lo, hi].
    """
    if override:
        try:
            return max(1, int(override))
        except (TypeError, ValueError):
            pass
    return max(lo, min(hi, int(per_shard)))


def _postprocess_chunk_size(batch_size: int, height: int, width: int) -> int:
    pixels_per_sample = max(1, int(height) * int(width))
    return max(1, min(int(batch_size), POSTPROCESS_MAX_PIXELS // pixels_per_sample))


def _slice_model_outputs(
    outputs: dict[str, Any],
    start: int,
    end: int,
    batch_size: int,
) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for name, value in outputs.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == int(batch_size):
            sliced[name] = value[start:end]
        else:
            sliced[name] = value
    return sliced


def _is_device_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    return (oom_type is not None and isinstance(exc, oom_type)) or "out of memory" in str(exc).lower()


def _release_device_cache(device: torch.device) -> None:
    module = _device_module(device)
    empty_cache = getattr(module, "empty_cache", None) if module is not None else None
    if callable(empty_cache):
        try:
            empty_cache()
        except Exception:
            pass


def _compose_canonical_chunks(
    img0: torch.Tensor,
    img1: torch.Tensor,
    outputs: dict[str, Any],
    *,
    initial_chunk_size: int,
) -> Iterator[tuple[int, int, dict[str, Any], dict[str, torch.Tensor]]]:
    """Compose output-resolution artifacts with bounded accelerator memory.

    The model batch remains intact. Only warp/blend/pred post-processing is
    sliced, and OOM retries reduce the slice without changing model precision.
    """
    batch_size = int(img0.shape[0])
    chunk_size = max(1, min(batch_size, int(initial_chunk_size)))
    cursor = 0
    while cursor < batch_size:
        end = min(batch_size, cursor + chunk_size)
        chunk_outputs = _slice_model_outputs(outputs, cursor, end, batch_size)
        try:
            composed = compose_interpolated(img0[cursor:end], img1[cursor:end], chunk_outputs)
        except RuntimeError as exc:
            if not _is_device_oom(exc) or chunk_size <= 1:
                raise
            _release_device_cache(img0.device)
            chunk_size = max(1, chunk_size // 2)
            continue
        yield cursor, end, chunk_outputs, composed
        cursor = end


def _detach_tensors_to_cpu(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Transfer a heterogeneous tensor bundle with one device-to-host copy."""
    if not tensors:
        return {}
    names: list[str] = []
    shapes: list[torch.Size] = []
    sizes: list[int] = []
    flattened: list[torch.Tensor] = []
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        detached = tensor.detach()
        names.append(name)
        shapes.append(detached.shape)
        sizes.append(detached.numel())
        flattened.append(detached.reshape(-1))
    if not flattened:
        return {}
    packed = torch.cat(flattened, dim=0) if len(flattened) > 1 else flattened[0]
    packed_cpu = packed.to("cpu")
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for name, shape, size in zip(names, shapes, sizes):
        result[name] = packed_cpu[offset : offset + size].reshape(shape)
        offset += size
    return result


def _fps(count: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return float(count) / seconds


def _device_module(device: torch.device):
    if device.type == "cuda" and hasattr(torch, "cuda"):
        return torch.cuda
    if device.type == "npu" and hasattr(torch, "npu"):
        return torch.npu
    return None


def _reset_peak_memory(device: torch.device) -> None:
    module = _device_module(device)
    if module is None:
        return
    try:
        module.reset_peak_memory_stats(device)
    except Exception:
        try:
            module.reset_peak_memory_stats()
        except Exception:
            pass


def _peak_memory(device: torch.device) -> dict[str, int]:
    module = _device_module(device)
    if module is None:
        return {}
    result: dict[str, int] = {}
    for name in ("max_memory_allocated", "max_memory_reserved"):
        function = getattr(module, name, None)
        if function is None:
            continue
        try:
            result[name] = int(function(device))
        except Exception:
            try:
                result[name] = int(function())
            except Exception:
                continue
    return result


def _device_name(device: torch.device) -> str:
    module = _device_module(device)
    if module is None:
        return "CPU"
    function = getattr(module, "get_device_name", None)
    if function is None:
        return device.type.upper()
    try:
        return str(function(device.index or 0))
    except Exception:
        try:
            return str(function(device))
        except Exception:
            return device.type.upper()


def _load_resized_batch(paths: list[str], device: torch.device, height: int, width: int) -> torch.Tensor:
    tensors = []
    for path in paths:
        tensor = load_rgb_tensor(path, device).unsqueeze(0)
        tensors.append(resize_batch(tensor, height, width)[0])
    return batch_tensors(tensors)


def _image_artifact_record(
    sample_id: int,
    kind: str,
    path: Path,
    metadata: dict[str, Any],
    *,
    make_preview: bool,
    preview_height: int | None = None,
    preview_width: int | None = None,
) -> dict[str, Any]:
    preview_metadata = dict(metadata)
    if preview_height is not None and preview_width is not None:
        preview_metadata.update(
            {
                "preview_height": int(preview_height),
                "preview_width": int(preview_width),
                "preview_resize": "lanczos",
                "preview_uses_canonical": not bool(make_preview),
            }
        )
    if make_preview:
        preview_path = path.parent / "preview" / path.name
        try:
            preview = save_preview_image(
                path,
                preview_path,
                height=preview_height,
                width=preview_width,
            )
            preview_metadata.update({"preview_path": str(preview), "preview_resize": "lanczos"})
            if preview_height is None or preview_width is None:
                preview_metadata["preview_max_edge"] = 512
        except Exception as exc:
            preview_metadata["preview_warning"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
    return {
        "sample_id": int(sample_id),
        "kind": str(kind),
        "path": str(path),
        "mime_type": "image/png",
        # This runs inside the bounded save pool for model inference, so the
        # eventual batched SQLite publication does not add filesystem stats to
        # the accelerator's main compute thread.
        "metadata": artifact_storage_metadata(path, preview_metadata),
    }


def _compare_track_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata") or {}
    result: dict[str, Any] = {}
    for key in (
        "compare_track_label",
        "compare_track_key",
        "compare_track_index",
        "compare_track_run_id",
        "compare_track_artifact_id",
    ):
        if key in metadata and metadata[key] is not None:
            result[key] = metadata[key]
    return result


def _copy_compare_image(source_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path).convert("RGB") as image:
        image.save(output_path)
    return output_path


def _write_run_metadata(run_dir: Path, job: dict[str, Any], model: dict[str, Any], dataset: dict[str, Any] | None) -> None:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "inference.log").write_text("VFIEval inference run started\n", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(job.get("payload") or {}, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "model_info.json").write_text(json.dumps(model, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if dataset is not None:
        (run_dir / "video_group_info.json").write_text(json.dumps(dataset, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _require_and_log_job_integrity(
    db: Database,
    run_dir: Path,
    job_id: int,
    expected_sample_ids: list[int],
) -> dict[str, Any]:
    path = run_dir / "logs" / "artifact_integrity" / f"{int(job_id)}.json"
    try:
        report = require_job_artifact_integrity(
            db,
            int(job_id),
            expected_sample_ids=expected_sample_ids,
        )
    except ArtifactIntegrityError as exc:
        write_integrity_report(path, exc.report)
        _persist_job_integrity_failure(db, int(job_id), exc.report)
        raise
    write_integrity_report(path, report)
    return report


def _persist_job_integrity_failure(
    db: Database,
    job_id: int,
    report: dict[str, Any],
) -> None:
    try:
        job = db.get_job(int(job_id))
        result = dict(job.get("result") or {})
        result["artifact_integrity"] = report
        db.update_job_progress(
            int(job_id),
            int(job.get("progress_current") or 0),
            result=result,
        )
        run_id = (job.get("payload") or {}).get("run_id")
        if run_id is not None:
            db.merge_run_result(int(run_id), {"artifact_integrity": report})
    except Exception:
        pass


def _log_failed_job_integrity(
    db: Database,
    run_dir: Path,
    job_id: int,
    expected_sample_ids: list[int],
    exc: BaseException,
    *,
    phase: str,
) -> dict[str, Any]:
    report = validate_job_artifact_integrity(
        db,
        int(job_id),
        expected_sample_ids=expected_sample_ids,
    )
    report["errors"].append(
        {
            "code": f"{phase}_failed",
            "message": str(exc) or type(exc).__name__,
            "error_type": type(exc).__name__,
        }
    )
    report["error_count"] = len(report["errors"])
    report["valid"] = False
    write_integrity_report(
        run_dir / "logs" / "artifact_integrity" / f"{int(job_id)}.json",
        report,
    )
    _persist_job_integrity_failure(db, int(job_id), report)
    return report


def _encode_and_validate_video_artifacts(
    db: Database,
    job_id: int,
    run_dir: Path,
    video_groups: dict[str, dict[str, Any]],
    job_integrity: dict[str, Any],
    *,
    expected_sample_ids: list[int],
    publish_pred_video: bool = True,
    preview_height: int | None = None,
    preview_width: int | None = None,
) -> dict[str, Any]:
    integrity_path = run_dir / "logs" / "artifact_integrity" / f"{int(job_id)}.json"
    try:
        _write_video_artifacts(
            db,
            job_id,
            run_dir,
            video_groups,
            publish_pred_video=publish_pred_video,
            preview_height=preview_height,
            preview_width=preview_width,
        )
    except Exception as exc:
        report = merge_integrity_reports(
            "job_artifacts",
            [job_integrity],
            job_id=int(job_id),
            phase="video_encoding",
        )
        report["errors"].append(
            {
                "code": "video_encoding_failed",
                "message": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
            }
        )
        report["error_count"] = len(report["errors"])
        report["valid"] = False
        write_integrity_report(integrity_path, report)
        _persist_job_integrity_failure(db, int(job_id), report)
        raise ArtifactIntegrityError(report) from exc

    video_report = validate_video_artifact_integrity(
        db,
        job_id,
        video_groups,
        publish_pred_video=publish_pred_video,
        expected_sample_ids=expected_sample_ids,
    )
    report = merge_integrity_reports(
        "job_artifacts",
        [job_integrity, video_report],
        job_id=int(job_id),
        phase="video_validation",
    )
    write_integrity_report(integrity_path, report)
    if not report["valid"]:
        _persist_job_integrity_failure(db, int(job_id), report)
        raise ArtifactIntegrityError(report)
    return report


def _publish_run_media_strict(
    db: Database,
    workspace: WorkspaceConfig,
    run_id: int,
    job_id: int,
    run_dir: Path,
    integrity_report: dict[str, Any],
) -> None:
    from vfieval.media_assets import sync_run_assets

    try:
        sync_run_assets(db, workspace, int(run_id), allow_running=True)
    except Exception as exc:
        db.invalidate_run_media_assets(int(run_id))
        report = merge_integrity_reports(
            "job_artifacts",
            [integrity_report],
            job_id=int(job_id),
            run_id=int(run_id),
            phase="media_publication",
        )
        report["errors"].append(
            {
                "code": "media_publication_failed",
                "message": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
            }
        )
        report["error_count"] = len(report["errors"])
        report["valid"] = False
        write_integrity_report(
            run_dir / "logs" / "artifact_integrity" / f"{int(job_id)}.json",
            report,
        )
        _persist_job_integrity_failure(db, int(job_id), report)
        raise ArtifactIntegrityError(report) from exc
    _raise_if_canceled(db, int(run_id), int(job_id))


def _write_shard_video_manifest(
    run_dir: Path,
    run_id: int | None,
    job_id: int,
    video_groups: dict[str, dict[str, Any]],
    *,
    expected_sample_ids: list[int],
    successful_sample_ids: list[int],
    core_artifact_counts: dict[str, int],
) -> Path:
    manifest_dir = run_dir / "logs" / "shards"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{int(job_id)}.json"
    path.write_text(
        json.dumps(
            {
                "version": "artifact-shard-v1",
                "run_id": int(run_id) if run_id is not None else None,
                "job_id": int(job_id),
                "expected_sample_ids": [int(value) for value in expected_sample_ids],
                "successful_sample_ids": [int(value) for value in successful_sample_ids],
                "core_artifact_counts": {str(key): int(value) for key, value in core_artifact_counts.items()},
                "video_groups": video_groups,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def _save_extra_outputs(outputs: dict[str, torch.Tensor], sample_dir: Path, index: int) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, tensor in outputs.items():
        if name in CORE_OUTPUTS or not isinstance(tensor, torch.Tensor):
            continue
        try:
            safe_name = sanitize_name(name)
            path = sample_dir / f"extra_{safe_name}.png"
            paths[f"extra_{safe_name}"] = save_extra_tensor(tensor, path, index)
        except Exception:
            # Extra visualizations must not invalidate the core flow/mask contract.
            continue
    return paths


def _record_sample_error(db: Database, job_id: int, sample_id: int, sample_name: str, exc: Exception) -> None:
    error_type = type(exc).__name__
    message = str(exc)[:500]
    db.add_artifact(
        job_id,
        sample_id,
        "sample_error",
        "",
        "application/json",
        {"sample": sample_name, "error_type": error_type, "message": message},
    )


def _require_progress_cas(
    db: Database,
    run_id: int | None,
    job_id: int,
    accepted: bool,
    target: str,
) -> None:
    if accepted:
        return
    _raise_if_canceled(db, run_id, job_id)
    raise RuntimeError(f"{target} rejected inference progress CAS")


def _raise_if_canceled(db: Database, run_id: int | None, job_id: int) -> None:
    if run_id is None:
        return
    run = db.get_run(run_id)
    if run["status"] in {"cancel_requested", "canceled"}:
        raise RunCanceled("用户取消了 Run")
    if run["status"] == "failed":
        # A sibling shard already failed the run (multi_cuda/multi_npu). Stop
        # this shard instead of burning device time toward a run that is
        # already terminal.
        # Keep the Job running until the worker has unwound save pools and
        # other file users; the worker owns the terminal cancellation CAS.
        raise RunCanceled("sibling shard failed the run")


def _collect_video_frame(
    video_groups: dict[str, dict[str, Any]],
    sample: dict[str, Any],
    pred_path: Path,
    diff_path: Path | None,
    gt_path: Path | None = None,
) -> None:
    metadata = sample.get("metadata") or {}
    if metadata.get("source_type") != "video":
        return
    video_key = str(metadata.get("video_path") or metadata.get("video_name") or "video")
    group = video_groups.setdefault(
        video_key,
        {
            "video_name": metadata.get("video_name") or sanitize_name(video_key),
            "fps": float(metadata.get("fps") or 24.0),
            "frames": [],
            # Source-clip identity so Compare can reconstruct a pred-aligned GT
            # from the decode cache instead of storing a per-run GT copy.
            "source_video_path": metadata.get("video_path"),
            "source_video_group": metadata.get("video_group"),
            "source_video_file": metadata.get("video_file"),
        },
    )
    frame_order_value = metadata.get("frame_index")
    if frame_order_value is None:
        frame_order_value = metadata.get("sample_index")
    if frame_order_value is None:
        frame_order_value = len(group["frames"])
    frame_order = int(frame_order_value)
    # Prefer the canonical GT materialized alongside Pred so both video streams
    # have the exact Run output dimensions. Legacy callers may still fall back
    # to the decoded source when no paired artifact was produced.
    resolved_gt = Path(gt_path) if gt_path is not None else (Path(sample["gt_path"]) if sample.get("gt_path") else None)
    group["frames"].append(
        {
            "order": frame_order,
            "sample_id": int(sample["id"]),
            "sample_name": sample["name"],
            "pred_path": Path(pred_path),
            "gt_path": resolved_gt,
            "diff_path": Path(diff_path) if diff_path else None,
            # gt_index is the source-clip frame this pred approximates
            # (pred[i] ≈ source_frames[gt_index]); Compare uses the ordered
            # list of these to head-offset the source clip into an aligned GT.
            "source_frame_index": metadata.get("gt_index"),
            # Encoded Pred videos begin at t=0 even when their first output
            # represents source frame 1. Preserve the semantic GT timestamp
            # alongside the index so Item Compare can validate the mapping
            # without treating that container origin as a temporal offset.
            "source_timestamp": (metadata.get("timestamps") or {}).get("gt"),
        }
    )


def _collect_compare_frame(
    video_groups: dict[str, dict[str, Any]],
    sample: dict[str, Any],
    gt_path: Path,
    pred_path: Path,
    diff_path: Path,
) -> None:
    metadata = sample.get("metadata") or {}
    video_key = str(metadata.get("compare_group") or metadata.get("video_name") or "compare")
    group = video_groups.setdefault(
        video_key,
        {
            "video_name": metadata.get("video_name") or video_key,
            "fps": float(metadata.get("fps") or 24.0),
            "frames": [],
        },
    )
    frame_order_value = metadata.get("frame_index")
    if frame_order_value is None:
        frame_order_value = metadata.get("sample_index")
    if frame_order_value is None:
        frame_order_value = len(group["frames"])
    frame_order = int(frame_order_value)
    group["frames"].append(
        {
            "order": frame_order,
            "sample_id": int(sample["id"]),
            "sample_name": sample["name"],
            "pred_path": Path(pred_path),
            "gt_path": Path(gt_path),
            "diff_path": Path(diff_path),
            "track_label": metadata.get("compare_track_label"),
            "track_key": metadata.get("compare_track_key"),
            "track_run_id": metadata.get("compare_track_run_id"),
            "track_artifact_id": metadata.get("compare_track_artifact_id"),
        }
    )


def _attach_optional_video_preview(
    frame_paths: list[Path],
    canonical_path: Path,
    metadata: dict[str, Any],
    fps: float,
    *,
    preview_height: int | None,
    preview_width: int | None,
) -> None:
    metadata.update({"artifact_contract": ARTIFACT_CONTRACT})
    if preview_height is None or preview_width is None or not frame_paths:
        return
    try:
        with Image.open(frame_paths[0]) as image:
            canonical_width, canonical_height = image.size
        resolved_preview_height = int(preview_height)
        resolved_preview_width = int(preview_width)
        metadata.update(
            {
                "canonical_height": int(canonical_height),
                "canonical_width": int(canonical_width),
                "preview_height": int(resolved_preview_height),
                "preview_width": int(resolved_preview_width),
                "preview_resize": "lanczos",
                "preview_uses_canonical": (canonical_height, canonical_width)
                == (resolved_preview_height, resolved_preview_width),
            }
        )
        if (canonical_height, canonical_width) == (resolved_preview_height, resolved_preview_width):
            return
        preview_path = canonical_path.parent / "preview" / canonical_path.name
        _write_mp4(
            frame_paths,
            preview_path,
            fps,
            target_height=resolved_preview_height,
            target_width=resolved_preview_width,
        )
        metadata.update(
            {
                "preview_path": str(preview_path),
                "preview_height": resolved_preview_height,
                "preview_width": resolved_preview_width,
                "preview_resize": "lanczos",
            }
        )
    except Exception as exc:
        metadata["preview_warning"] = {
            "type": type(exc).__name__,
            "message": str(exc)[:500],
        }


def _write_video_artifacts(
    db: Database,
    job_id: int,
    run_dir: Path,
    video_groups: dict[str, dict[str, Any]],
    *,
    publish_pred_video: bool = True,
    preview_height: int | None = None,
    preview_width: int | None = None,
) -> None:
    directory_names = _unique_sanitized_tokens(
        [
            (str(group_key), str(group.get("video_name") or group_key))
            for group_key, group in video_groups.items()
        ]
    )
    for group_key, group in video_groups.items():
        frames = sorted(group["frames"], key=lambda item: item["order"])
        if not frames:
            continue
        video_name = directory_names[str(group_key)]
        fps = float(group["fps"] or 24.0)
        if any(frame.get("track_label") for frame in frames):
            _write_multitrack_compare_video_artifacts(
                db,
                job_id,
                run_dir,
                group,
                frames,
                video_name,
                fps,
                publish_pred_video=publish_pred_video,
                preview_height=preview_height,
                preview_width=preview_width,
            )
            continue
        video_dir = run_dir / "videos" / video_name
        video_dir.mkdir(parents=True, exist_ok=True)
        pred_frame_paths = [Path(frame["pred_path"]) for frame in frames]
        pred_video_path = None
        if publish_pred_video:
            pred_video_path = video_dir / "pred.mp4"
            _write_mp4(pred_frame_paths, pred_video_path, fps)
            pred_metadata = _video_artifact_metadata(group["video_name"], frames, pred_frame_paths, fps)
            pred_metadata.update(_source_mapping_metadata(group, frames))
            _attach_optional_video_preview(
                pred_frame_paths,
                pred_video_path,
                pred_metadata,
                fps,
                preview_height=preview_height,
                preview_width=preview_width,
            )
            db.add_artifact(
                job_id,
                None,
                "pred_video",
                str(pred_video_path),
                "video/mp4",
                pred_metadata,
            )

        gt_paths = [frame["gt_path"] for frame in frames if frame["gt_path"] is not None]
        if len(gt_paths) == len(frames):
            gt_frame_paths = [Path(path) for path in gt_paths]
            gt_video_path = video_dir / "gt.mp4"
            _write_mp4(gt_frame_paths, gt_video_path, fps)
            gt_metadata = _video_artifact_metadata(group["video_name"], frames, gt_frame_paths, fps)
            _attach_optional_video_preview(
                gt_frame_paths,
                gt_video_path,
                gt_metadata,
                fps,
                preview_height=preview_height,
                preview_width=preview_width,
            )
            db.add_artifact(
                job_id,
                None,
                "gt_video",
                str(gt_video_path),
                "video/mp4",
                gt_metadata,
            )

        diff_paths = [frame["diff_path"] for frame in frames if frame["diff_path"] is not None]
        if len(diff_paths) == len(frames):
            diff_frame_paths = [Path(path) for path in diff_paths]
            diff_video_path = video_dir / "diff.mp4"
            _write_mp4(diff_frame_paths, diff_video_path, fps)
            diff_metadata = _video_artifact_metadata(group["video_name"], frames, diff_frame_paths, fps)
            _attach_optional_video_preview(
                diff_frame_paths,
                diff_video_path,
                diff_metadata,
                fps,
                preview_height=preview_height,
                preview_width=preview_width,
            )
            db.add_artifact(
                job_id,
                None,
                "diff_video",
                str(diff_video_path),
                "video/mp4",
                diff_metadata,
            )

        manifest = {
            "video_name": group["video_name"],
            "fps": fps,
            "frames": len(frames),
            "pred_video": str(pred_video_path.resolve()) if pred_video_path else None,
            "gt_video": str((video_dir / "gt.mp4").resolve()) if (video_dir / "gt.mp4").exists() else None,
            "diff_video": str((video_dir / "diff.mp4").resolve()) if (video_dir / "diff.mp4").exists() else None,
        }
        (video_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_multitrack_compare_video_artifacts(
    db: Database,
    job_id: int,
    run_dir: Path,
    group: dict[str, Any],
    frames: list[dict[str, Any]],
    video_name: str,
    fps: float,
    *,
    publish_pred_video: bool = True,
    preview_height: int | None = None,
    preview_width: int | None = None,
) -> None:
    video_dir = run_dir / "videos" / video_name
    tracks: dict[str, list[dict[str, Any]]] = {}
    gt_by_order: dict[int, Path] = {}
    for frame in frames:
        track_label = str(frame.get("track_label") or "pred")
        track_key = str(frame.get("track_key") or sanitize_name(track_label))
        tracks.setdefault(track_key, []).append(frame)
        if frame.get("gt_path") is not None:
            gt_by_order.setdefault(int(frame["order"]), Path(frame["gt_path"]))

    track_directory_names = _unique_sanitized_tokens(
        [
            (
                track_key,
                str(sorted(track_frames, key=lambda item: item["order"])[0].get("track_label") or track_key),
            )
            for track_key, track_frames in tracks.items()
            if track_frames
        ]
    )

    gt_video_path = None
    ordered_gt = [gt_by_order[index] for index in sorted(gt_by_order)]
    if ordered_gt:
        gt_frame_paths = ordered_gt
        gt_video_path = video_dir / "gt.mp4"
        _write_mp4(gt_frame_paths, gt_video_path, fps)
        gt_metadata = _video_artifact_metadata(group["video_name"], frames, gt_frame_paths, fps)
        _attach_optional_video_preview(
            gt_frame_paths,
            gt_video_path,
            gt_metadata,
            fps,
            preview_height=preview_height,
            preview_width=preview_width,
        )
        db.add_artifact(
            job_id,
            None,
            "gt_video",
            str(gt_video_path),
            "video/mp4",
            gt_metadata,
        )

    manifest_tracks = []
    for track_key, track_frames in sorted(tracks.items()):
        ordered = sorted(track_frames, key=lambda item: item["order"])
        if not ordered:
            continue
        track_label = str(ordered[0].get("track_label") or track_key)
        track_dir = video_dir / track_directory_names[track_key]
        track_dir.mkdir(parents=True, exist_ok=True)
        pred_frame_paths = [Path(frame["pred_path"]) for frame in ordered]
        pred_video_path = None
        if publish_pred_video:
            pred_video_path = track_dir / "pred.mp4"
            _write_mp4(pred_frame_paths, pred_video_path, fps)
        diff_frame_paths = [Path(frame["diff_path"]) for frame in ordered]
        diff_video_path = track_dir / "diff.mp4"
        _write_mp4(diff_frame_paths, diff_video_path, fps)
        track_metadata = {
            "compare_track_label": track_label,
            "compare_track_key": track_key,
            "compare_track_run_id": ordered[0].get("track_run_id"),
            "compare_track_artifact_id": ordered[0].get("track_artifact_id"),
        }
        if pred_video_path is not None:
            pred_metadata = {
                **_video_artifact_metadata(group["video_name"], ordered, pred_frame_paths, fps),
                **track_metadata,
            }
            _attach_optional_video_preview(
                pred_frame_paths,
                pred_video_path,
                pred_metadata,
                fps,
                preview_height=preview_height,
                preview_width=preview_width,
            )
            db.add_artifact(
                job_id,
                None,
                "pred_video",
                str(pred_video_path),
                "video/mp4",
                pred_metadata,
            )
        diff_metadata = {
            **_video_artifact_metadata(group["video_name"], ordered, diff_frame_paths, fps),
            **track_metadata,
        }
        _attach_optional_video_preview(
            diff_frame_paths,
            diff_video_path,
            diff_metadata,
            fps,
            preview_height=preview_height,
            preview_width=preview_width,
        )
        db.add_artifact(
            job_id,
            None,
            "diff_video",
            str(diff_video_path),
            "video/mp4",
            diff_metadata,
        )

        manifest_tracks.append(
            {
                "track_label": track_label,
                "track_key": track_key,
                "frames": len(ordered),
                "pred_video": str(pred_video_path.resolve()) if pred_video_path else None,
                "diff_video": str(diff_video_path.resolve()),
                "compare_track_run_id": ordered[0].get("track_run_id"),
                "compare_track_artifact_id": ordered[0].get("track_artifact_id"),
            }
        )

    manifest = {
        "video_name": group["video_name"],
        "fps": fps,
        "frames": len(ordered_gt) if ordered_gt else 0,
        "gt_video": str(gt_video_path.resolve()) if gt_video_path else None,
        "tracks": manifest_tracks,
    }
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_compare_alignment_report(
    db: Database,
    job_id: int,
    run_dir: Path,
    alignment_plan: dict[str, Any],
) -> Path:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "alignment.json"
    path.write_text(json.dumps(alignment_plan, indent=2, ensure_ascii=False), encoding="utf-8")
    db.add_artifact(
        job_id,
        None,
        "alignment_report",
        str(path),
        "application/json",
        {
            "alignment_fingerprint": alignment_plan.get("fingerprint"),
            "target": alignment_plan.get("target"),
            "filter": alignment_plan.get("filter"),
        },
    )
    return path


def _video_artifact_metadata(
    video_name: str,
    frames: list[dict[str, Any]],
    frame_paths: list[Path],
    fps: float,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"video_name": video_name, "frames": len(frame_paths), "fps": fps}
    if frame_paths:
        try:
            with Image.open(frame_paths[0]) as image:
                width, height = image.size
            metadata.update({"width": int(width), "height": int(height)})
        except Exception:
            pass
    return metadata


def _source_mapping_metadata(
    group: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """Source-clip mapping so Compare can reconstruct a pred-aligned GT.

    ``pred[i] ≈ source_frames[source_frame_indices[i]]``, so Compare can select
    ``source_frames[indices]`` from the raw ``videos/`` clip (decode cache)
    instead of relying on a per-run ``gt.mp4`` copy. The indices are ordered by
    the same frame order used to assemble ``pred.mp4``. Returns an empty dict
    when the source identity or per-frame indices are unavailable (e.g. legacy
    samples), so callers fall back to the stored ``gt_video``.
    """
    source_path = group.get("source_video_path")
    if not source_path:
        return {}
    indices = [frame.get("source_frame_index") for frame in frames]
    if any(index is None for index in indices):
        return {}
    mapping: dict[str, Any] = {
        "source_video_path": str(source_path),
        "source_frame_indices": [int(index) for index in indices],
    }
    timestamps = [frame.get("source_timestamp") for frame in frames]
    if timestamps and all(timestamp is not None for timestamp in timestamps):
        mapping["timestamps"] = [float(timestamp) for timestamp in timestamps]
    if group.get("source_video_group") is not None:
        mapping["source_video_group"] = group.get("source_video_group")
    if group.get("source_video_file") is not None:
        mapping["source_video_file"] = group.get("source_video_file")
    return mapping


def _copy_ordered_frames(frame_paths: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for index, frame_path in enumerate(frame_paths):
        target = output_dir / f"{index:06d}.png"
        shutil.copy2(frame_path, target)
        copied.append(target)
    return copied


def _write_mp4(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    *,
    target_height: int | None = None,
    target_width: int | None = None,
) -> None:
    if not frame_paths:
        raise RuntimeError(f"cannot create video artifact without frames: {output_path}")
    _validate_video_encode_dimensions(
        frame_paths,
        target_height=target_height,
        target_width=target_width,
    )
    if _write_mp4_ffmpeg_pipe(
        frame_paths,
        output_path,
        fps,
        target_height=target_height,
        target_width=target_width,
    ):
        return
    _write_mp4_cv2(
        frame_paths,
        output_path,
        fps,
        target_height=target_height,
        target_width=target_width,
    )


def _validate_video_encode_dimensions(
    frame_paths: list[Path],
    *,
    target_height: int | None,
    target_width: int | None,
) -> tuple[int, int]:
    if (target_height is None) != (target_width is None):
        raise ValueError("video target height and width must be provided together")
    if target_height is not None and target_width is not None:
        height, width = int(target_height), int(target_width)
        if height <= 0 or width <= 0:
            raise ValueError("video target height and width must both be positive")
    else:
        try:
            with Image.open(frame_paths[0]) as image:
                width, height = image.size
        except Exception as exc:
            raise RuntimeError(f"failed to inspect video frame dimensions: {frame_paths[0]}") from exc
        for frame_path in frame_paths[1:]:
            try:
                with Image.open(frame_path) as image:
                    frame_size = image.size
            except Exception as exc:
                raise RuntimeError(f"failed to inspect video frame dimensions: {frame_path}") from exc
            if frame_size != (width, height):
                raise ValueError(
                    "canonical video frames must all have identical dimensions; "
                    f"expected {width}x{height}, found {frame_size[0]}x{frame_size[1]} in {frame_path}"
                )
    if width % 2 or height % 2:
        raise ValueError(
            "browser-compatible H.264/yuv420p requires even dimensions; "
            f"requested {width}x{height}. Canonical video dimensions are exact and will not be padded."
        )
    return int(height), int(width)


def _write_mp4_ffmpeg_pipe(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    *,
    target_height: int | None = None,
    target_width: int | None = None,
) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not frame_paths or any(path.suffix.lower() != ".png" for path in frame_paths):
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    if target_height is not None or target_width is not None:
        if target_height is None or target_width is None or int(target_height) <= 0 or int(target_width) <= 0:
            raise ValueError("video preview height and width must both be positive")
        filters.append(f"scale={int(target_width)}:{int(target_height)}:flags=lanczos")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-framerate",
        str(float(fps)),
        "-vcodec",
        "png",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    if filters:
        pixel_format_index = command.index("-pix_fmt")
        command[pixel_format_index:pixel_format_index] = ["-vf", ",".join(filters)]
    process = None
    stderr: bytes | str | None = b""
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert process.stdin is not None
        stdin = process.stdin
        for path in frame_paths:
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, stdin, length=1024 * 1024)
        stdin.close()
        process.stdin = None
        _, stderr = process.communicate(timeout=600)
        if process.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0:
            return True
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            try:
                process.kill()
                process.communicate(timeout=5)
            except Exception:
                pass
        output_path.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg timed out while encoding browser-compatible H.264 video") from exc
    except Exception as exc:
        stderr = b""
        if process is not None:
            try:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                process.stdin = None
                _, stderr = process.communicate(timeout=10)
            except Exception:
                try:
                    process.kill()
                    _, stderr = process.communicate(timeout=5)
                except Exception:
                    pass
        output_path.unlink(missing_ok=True)
        detail = _ffmpeg_error_detail(stderr) or str(exc)
        raise RuntimeError(f"failed to encode browser-compatible H.264 video with ffmpeg/libx264: {detail}") from exc
    finally:
        if process is not None:
            for stream in (process.stdin, process.stderr):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except Exception:
                        pass
    output_path.unlink(missing_ok=True)
    detail = _ffmpeg_error_detail(stderr)
    raise RuntimeError(
        "failed to encode browser-compatible H.264 video with ffmpeg/libx264"
        + (f": {detail}" if detail else "")
    )


def _ffmpeg_error_detail(stderr: bytes | str | None) -> str:
    if not stderr:
        return ""
    text = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else str(stderr)
    compact = " ".join(text.strip().split())
    return compact[-1200:]


def _write_mp4_cv2(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    *,
    target_height: int | None = None,
    target_width: int | None = None,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("writing video artifacts requires opencv-python (cv2)") from exc

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"failed to read video frame: {frame_paths[0]}")
    source_height, source_width = first.shape[:2]
    height = int(target_height) if target_height is not None else source_height
    width = int(target_width) if target_width is not None else source_width
    if height <= 0 or width <= 0:
        raise ValueError("video preview height and width must both be positive")
    if width % 2 or height % 2:
        raise ValueError(
            "browser-compatible H.264/yuv420p requires even dimensions; "
            f"requested {width}x{height}. Canonical video dimensions are exact and will not be padded."
        )
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            "failed to create a browser-compatible H.264 video artifact; "
            "install ffmpeg with the libx264 encoder or provide OpenCV with AVC support"
        )
    try:
        writer.write(_resize_video_frame(first, width, height))
        for frame_path in frame_paths[1:]:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"failed to read video frame: {frame_path}")
            frame = _resize_video_frame(frame, width, height)
            writer.write(frame)
    finally:
        writer.release()


def _resize_video_frame(frame, width: int, height: int):
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("resizing video frames requires opencv-python (cv2)") from exc
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_LANCZOS4)


def _load_rgb_cpu(path: str, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)
    if tensor.shape[-2:] != (height, width):
        tensor = torch.nn.functional.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.squeeze(0)


def _stack_pinned(tensors: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(tensors, dim=0)
    if torch.cuda.is_available():
        try:
            stacked = stacked.pin_memory()
        except Exception:
            pass
    return stacked


class _FrameDecodeCache:
    """Path-keyed decode pool that submits one task per distinct image.

    Video triplets overlap heavily — img1 of frame N is img0 of frame N+1, and
    a sample's GT is often a neighbouring source frame — so the same PNG would
    otherwise be decoded two or three times. Keying inflight futures by
    (path, height, width) collapses those to a single decode. Every worker
    pulls one image at a time, so a large pool stays saturated instead of a few
    workers each grinding through a whole batch serially.

    The inflight map is a bounded LRU. Eviction only drops entries whose batch
    has already been consumed; a re-decode after eviction is a correctness-safe
    cache miss, never wrong data.
    """

    def __init__(self, pool: ThreadPoolExecutor, height: int, width: int, capacity: int) -> None:
        self._pool = pool
        self._height = height
        self._width = width
        self._capacity = max(1, capacity)
        self._inflight: "OrderedDict[tuple[str, int, int], Future]" = OrderedDict()
        self._lock = threading.Lock()

    def submit(self, path: str) -> Future:
        key = (path, self._height, self._width)
        with self._lock:
            future = self._inflight.get(key)
            if future is not None:
                self._inflight.move_to_end(key)
                return future
            future = self._pool.submit(_load_rgb_cpu, path, self._height, self._width)
            self._inflight[key] = future
            while len(self._inflight) > self._capacity:
                self._inflight.popitem(last=False)
            return future


def _iter_prefetched_batches(
    samples: list[dict[str, Any]],
    batch_size: int,
    height: int,
    width: int,
    has_gt: bool,
    workers: int,
):
    """Yield (batch_rows, img0_cpu, img1_cpu, gt_cpu_list, wait_seconds) tuples.

    Decode runs one task per distinct image on a pool sized to the shard's core
    budget, prefetching up to 3 batches ahead so PIL decode and resize overlap
    device compute. A path-keyed cache collapses the img0/img1/gt overlap in
    video triplets to a single decode. wait_seconds is how long the main loop
    blocked assembling the next batch (waiting on outstanding decode futures).
    """
    if not samples:
        return
    workers = max(1, int(workers or 1))
    max_ahead = 3
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vfi-prefetch")
    # The working set is the distinct images across the batches held in flight
    # (each triplet touches up to 3). Keep generous headroom so overlapping
    # frames survive from one batch to the next.
    cache = _FrameDecodeCache(pool, height, width, capacity=(max_ahead + 1) * batch_size * 3 + batch_size)
    # Each pending entry holds the per-image futures for one batch.
    pending: list[tuple[list[dict[str, Any]], list[Future], list[Future], list[Future | None]]] = []

    def _submit(batch_rows: list[dict[str, Any]]):
        img0_futures = [cache.submit(row["img0_path"]) for row in batch_rows]
        img1_futures = [cache.submit(row["img1_path"]) for row in batch_rows]
        gt_futures: list[Future | None] = []
        if has_gt:
            for row in batch_rows:
                gt_path = row.get("gt_path")
                gt_futures.append(cache.submit(gt_path) if gt_path else None)
        else:
            gt_futures = [None] * len(batch_rows)
        return batch_rows, img0_futures, img1_futures, gt_futures

    clean_exit = False
    try:
        cursor = 0
        while cursor < len(samples) and len(pending) < max_ahead:
            end = min(cursor + batch_size, len(samples))
            pending.append(_submit(samples[cursor:end]))
            cursor = end

        while pending:
            batch_rows, img0_futures, img1_futures, gt_futures = pending.pop(0)
            wait_start = time.perf_counter()
            img0_cpu = _stack_pinned([future.result() for future in img0_futures])
            img1_cpu = _stack_pinned([future.result() for future in img1_futures])
            gt_cpu_list = [future.result() if future is not None else None for future in gt_futures]
            wait_seconds = time.perf_counter() - wait_start

            if cursor < len(samples):
                end = min(cursor + batch_size, len(samples))
                pending.append(_submit(samples[cursor:end]))
                cursor = end

            yield batch_rows, img0_cpu, img1_cpu, gt_cpu_list, wait_seconds
        clean_exit = True
    finally:
        # On normal completion there is nothing left in flight, so waiting is
        # free. On early exit (exception/cancel raised into this generator)
        # cancel outstanding work and don't block on it.
        pool.shutdown(wait=clean_exit, cancel_futures=not clean_exit)


def _materialize_gt_artifact(
    tensor: torch.Tensor,
    destination: Path,
    *,
    source_path: Path,
    source_cache_key: str,
    decode_cache_root: Path | None,
    expected_height: int,
    expected_width: int,
) -> dict[str, Any]:
    """Publish one canonical GT image, reusing immutable decode bytes when safe.

    A hard link keeps the artifact path inside the Run, so integrity checks,
    streaming, and Run deletion retain their existing ownership model.  It
    also remains readable if the cache entry is later garbage-collected.  Any
    uncertainty (unmanaged source, resized inference, non-PNG input, or a
    filesystem without hard-link support) falls back to the established PNG
    encoder.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    cache_source = _trusted_native_decode_gt(
        source_path,
        cache_key=source_cache_key,
        decode_cache_root=decode_cache_root,
        expected_height=expected_height,
        expected_width=expected_width,
    )
    if cache_source is not None:
        try:
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            os.link(cache_source, destination)
            return {
                "storage_mode": "hardlink",
                "storage_source": "decode_cache",
                "deduplicated": True,
                "source_cache_key": source_cache_key,
            }
        except OSError:
            # Cross-volume workspaces and filesystems without hard-link support
            # keep the exact historical behavior.
            pass

    save_rgb_tensor(tensor, destination)
    return {
        "storage_mode": "run_copy",
        "storage_source": "generated",
        "deduplicated": False,
    }


def _trusted_native_decode_gt(
    source_path: Path,
    *,
    cache_key: str,
    decode_cache_root: Path | None,
    expected_height: int,
    expected_width: int,
) -> Path | None:
    if decode_cache_root is None or not cache_key:
        return None
    raw_source = Path(source_path)
    if raw_source.is_symlink():
        return None
    try:
        root = Path(decode_cache_root).resolve(strict=True)
        source = raw_source.resolve(strict=True)
        relative = source.relative_to(root)
    except (OSError, ValueError):
        return None
    if len(relative.parts) != 2 or relative.parts[0] != cache_key:
        return None
    if source.parent.name != cache_key or source.suffix.lower() != ".png" or not source.is_file():
        return None
    try:
        with Image.open(source) as image:
            if image.format != "PNG" or image.size != (int(expected_width), int(expected_height)):
                return None
    except (OSError, ValueError):
        return None
    return source


class _AsyncSavePipeline:
    """Background PNG-encoding + DB-insert pool for inference artifacts.

    Each sqlite3 connection is short-lived and created inside the worker
    thread (via Database.connection()), so no cross-thread lock is needed —
    WAL mode already serializes writes at the file level.
    """

    def __init__(
        self,
        *,
        db: Database,
        job_id: int,
        run_id: int | None,
        is_shard: bool,
        run_dir: Path,
        decode_cache_root: Path | None = None,
        save_workers: int,
        max_inflight: int,
        artifact_batch_size: int,
        preview_height: int,
        preview_width: int,
    ) -> None:
        self._db = db
        self._job_id = job_id
        self._run_id = run_id
        self._is_shard = is_shard
        self._run_dir = run_dir
        self._decode_cache_root = (
            Path(decode_cache_root).resolve() if decode_cache_root is not None else None
        )
        self._pool = ThreadPoolExecutor(max_workers=save_workers, thread_name_prefix="vfi-save")
        self._slots = threading.Semaphore(max(1, int(max_inflight)))
        self._pending: list[Future] = []
        self._lock = threading.Lock()
        self._processed = 0
        self._total = 0
        self._video_groups: dict[str, dict[str, Any]] = {}
        self._last_progress_report = 0
        self._backpressure_seconds = 0.0
        self._max_observed_inflight = 0
        self._artifact_batch_size = max(1, int(artifact_batch_size))
        self._artifact_buffer: list[dict[str, Any]] = []
        self._artifact_db_batches = 0
        self._preview_height = int(preview_height)
        self._preview_width = int(preview_width)
        self._failure_event = threading.Event()
        self._failures: list[BaseException] = []

    @property
    def processed_count(self) -> int:
        return self._processed

    @property
    def video_groups(self) -> dict[str, dict[str, Any]]:
        return self._video_groups

    @property
    def backpressure_seconds(self) -> float:
        return self._backpressure_seconds

    @property
    def max_observed_inflight(self) -> int:
        return self._max_observed_inflight

    @property
    def artifact_db_batches(self) -> int:
        return self._artifact_db_batches

    def submit_batch(
        self,
        *,
        batch_rows: list[dict[str, Any]],
        bundle_cpu: dict[str, torch.Tensor],
        extra_cpu: dict[str, torch.Tensor],
        gt_cpu_list: list[torch.Tensor | None],
    ) -> None:
        self.raise_if_failed()
        self._total += len(batch_rows)
        for idx, row in enumerate(batch_rows):
            wait_start = time.perf_counter()
            self._slots.acquire()
            waited = time.perf_counter() - wait_start
            if self._failure_event.is_set():
                self._slots.release()
                self._cancel_not_started()
                self.raise_if_failed()
            per_sample_bundle = {name: bundle_cpu[name][idx] for name in bundle_cpu}
            per_sample_extra = {name: extra_cpu[name][idx] for name in extra_cpu}
            gt_tensor = gt_cpu_list[idx] if idx < len(gt_cpu_list) else None
            future = self._pool.submit(
                self._save_sample,
                dict(row),
                per_sample_bundle,
                per_sample_extra,
                gt_tensor,
            )
            with self._lock:
                self._backpressure_seconds += waited
                self._pending.append(future)
                self._max_observed_inflight = max(self._max_observed_inflight, len(self._pending))
            future.add_done_callback(self._on_sample_done)

    def raise_if_failed(self) -> None:
        if not self._failure_event.is_set():
            return
        self._cancel_not_started()
        with self._lock:
            failure = self._failures[0] if self._failures else RuntimeError("artifact save pipeline failed")
        raise failure

    def _cancel_not_started(self) -> None:
        with self._lock:
            pending = list(self._pending)
        for future in pending:
            future.cancel()

    def wait_for_all(self) -> None:
        while True:
            with self._lock:
                pending = list(self._pending)
                self._pending = []
            if not pending:
                break
            for future in pending:
                try:
                    future.result()
                except CancelledError:
                    continue
                except BaseException as exc:
                    # The callback records the first actionable exception. Keep
                    # draining so already-running, bounded save work finishes.
                    self._record_failure(exc)
                    continue
        try:
            self._flush_artifact_records()
        except BaseException as exc:
            self._record_failure(exc)
        self._raise_collected_failures()

    def shutdown(self, *, suppress_errors: bool = False) -> None:
        self._pool.shutdown(wait=True)
        try:
            self._flush_artifact_records()
        except BaseException as exc:
            self._record_failure(exc)
            if not suppress_errors:
                raise
        if not suppress_errors:
            self._raise_collected_failures()

    def _raise_collected_failures(self) -> None:
        if not self._failure_event.is_set():
            return
        with self._lock:
            failures = list(self._failures)
        if not failures:
            raise RuntimeError("artifact save pipeline failed")
        if len(failures) == 1:
            raise failures[0]
        raise ArtifactSaveAggregateError(failures) from failures[0]

    def _record_failure(self, exc: BaseException) -> None:
        if isinstance(exc, CancelledError):
            return
        with self._lock:
            if all(existing is not exc for existing in self._failures):
                self._failures.append(exc)
            self._failure_event.set()

    def _save_sample(
        self,
        row: dict[str, Any],
        bundle: dict[str, torch.Tensor],
        extra: dict[str, torch.Tensor],
        gt_tensor: torch.Tensor | None,
    ) -> None:
        job_id = self._job_id
        sample_id = int(row["id"])
        sample_name = row["name"]
        try:
            sample_dir = self._run_dir / f"{sample_id:08d}_{sanitize_name(sample_name)}"
            paths = _save_visual_bundle_from_cpu(bundle, sample_dir)
            artifact_records: list[dict[str, Any]] = []
            pred_h, pred_w = int(bundle["pred"].shape[-2]), int(bundle["pred"].shape[-1])
            base_artifact_metadata = {
                "sample": sample_name,
                "artifact_contract": ARTIFACT_CONTRACT,
            }
            extra_paths: dict[str, Path] = {}
            optional_warnings: list[dict[str, str]] = []
            for name, tensor in extra.items():
                try:
                    safe_name = sanitize_name(name)
                    path = sample_dir / f"extra_{safe_name}.png"
                    save_extra_tensor(tensor, path, index=0)
                    extra_paths[f"extra_{safe_name}"] = path
                except Exception as exc:
                    optional_warnings.append(
                        {
                            "kind": f"extra_{sanitize_name(name)}",
                            "type": type(exc).__name__,
                            "message": str(exc)[:500],
                        }
                    )
                    continue
            if optional_warnings:
                base_artifact_metadata["optional_warnings"] = optional_warnings
            for kind, path in paths.items():
                tensor = bundle[kind]
                artifact_h = int(tensor.shape[-2])
                artifact_w = int(tensor.shape[-1])
                artifact_metadata = {
                    **base_artifact_metadata,
                    "canonical_height": artifact_h,
                    "canonical_width": artifact_w,
                }
                artifact_records.append(
                    _image_artifact_record(
                        sample_id,
                        kind,
                        path,
                        artifact_metadata,
                        make_preview=(artifact_h, artifact_w)
                        != (self._preview_height, self._preview_width),
                        preview_height=self._preview_height,
                        preview_width=self._preview_width,
                    )
                )
            for kind, path in extra_paths.items():
                artifact_records.append(
                    _image_artifact_record(
                        sample_id, kind, path, {"sample": sample_name}, make_preview=False
                    )
                )

            diff_path = None
            gt_path = None
            if gt_tensor is not None and row.get("gt_path"):
                pred_h, pred_w = int(bundle["pred"].shape[-2]), int(bundle["pred"].shape[-1])
                if tuple(gt_tensor.shape[-2:]) != (pred_h, pred_w):
                    gt_tensor = _resize_chw(gt_tensor, pred_h, pred_w)
                gt_path = sample_dir / "gt.png"
                gt_storage = _materialize_gt_artifact(
                    gt_tensor,
                    gt_path,
                    source_path=Path(str(row["gt_path"])),
                    source_cache_key=str((row.get("metadata") or {}).get("cache_key") or ""),
                    decode_cache_root=self._decode_cache_root,
                    expected_height=pred_h,
                    expected_width=pred_w,
                )
                canonical_metadata = {
                    **base_artifact_metadata,
                    "canonical_height": pred_h,
                    "canonical_width": pred_w,
                    **gt_storage,
                }
                make_canonical_preview = (pred_h, pred_w) != (
                    self._preview_height,
                    self._preview_width,
                )
                artifact_records.append(
                    _image_artifact_record(
                        sample_id,
                        "gt",
                        gt_path,
                        canonical_metadata,
                        make_preview=make_canonical_preview,
                        preview_height=self._preview_height,
                        preview_width=self._preview_width,
                    )
                )
                diff_path = sample_dir / "difference.png"
                save_difference(bundle["pred"], gt_tensor, diff_path)
                artifact_records.append(
                    _image_artifact_record(
                        sample_id,
                        "difference",
                        diff_path,
                        canonical_metadata,
                        make_preview=make_canonical_preview,
                        preview_height=self._preview_height,
                        preview_width=self._preview_width,
                    )
                )
            self._queue_artifact_records(artifact_records)
            with self._lock:
                _collect_video_frame(self._video_groups, row, paths["pred"], diff_path, gt_path)
        except Exception as exc:
            try:
                _record_sample_error(self._db, job_id, sample_id, sample_name, exc)
            except Exception:
                pass
            raise CoreArtifactSaveError(sample_id, sample_name, exc) from exc

    def _queue_artifact_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        batch: list[dict[str, Any]] = []
        with self._lock:
            self._artifact_buffer.extend(records)
            if len(self._artifact_buffer) >= self._artifact_batch_size:
                batch = self._artifact_buffer
                self._artifact_buffer = []
        if batch:
            try:
                self._db.add_artifacts_bulk(self._job_id, batch)
            except Exception:
                with self._lock:
                    self._artifact_buffer = batch + self._artifact_buffer
                raise
            with self._lock:
                self._artifact_db_batches += 1

    def _flush_artifact_records(self) -> None:
        with self._lock:
            batch = self._artifact_buffer
            self._artifact_buffer = []
        if not batch:
            return
        self._db.add_artifacts_bulk(self._job_id, batch)
        with self._lock:
            self._artifact_db_batches += 1

    def _on_sample_done(self, future: Future) -> None:
        report_now = False
        processed = 0
        total = 0
        try:
            try:
                exc = future.exception()
            except CancelledError:
                exc = None
            with self._lock:
                if exc is None and not future.cancelled():
                    self._processed += 1
                elif exc is not None:
                    if all(existing is not exc for existing in self._failures):
                        self._failures.append(exc)
                    self._failure_event.set()
                processed = self._processed
                total = self._total
                try:
                    self._pending.remove(future)
                except ValueError:
                    pass
                step = max(1, total // 200)
                report_now = exc is None and (
                    processed == total or processed - self._last_progress_report >= step
                )
                if report_now:
                    self._last_progress_report = processed
        finally:
            self._slots.release()
        if not report_now:
            return
        try:
            self._db.update_job_progress(self._job_id, processed)
            if self._run_id is not None:
                if self._is_shard:
                    self._db.update_run_progress_from_jobs(self._run_id)
                else:
                    self._db.update_run_progress(self._run_id, processed)
        except Exception:
            pass


def _save_visual_bundle_from_cpu(bundle: dict[str, torch.Tensor], sample_dir: Path) -> dict[str, Path]:
    """Save an already-on-CPU bundle. Mirrors save_visual_bundle but skips
    the per-tensor .cpu() calls that visualize.save_visual_bundle does when
    the input still lives on the device."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    from vfieval.pipeline.visualize import save_rgb_tensor, save_mask, save_flow

    # Only the keys present in the bundle are saved: warp0/warp1/blend are
    # omitted unless save_warp_blend was requested, and pred is always present.
    savers = {
        "pred": save_rgb_tensor,
        "warp0": save_rgb_tensor,
        "warp1": save_rgb_tensor,
        "blend": save_rgb_tensor,
        "mask0": save_mask,
        "mask1": save_mask,
        "flowt_0": save_flow,
        "flowt_1": save_flow,
    }
    paths: dict[str, Path] = {}
    for kind, saver in savers.items():
        tensor = bundle.get(kind)
        if tensor is None:
            continue
        path = sample_dir / f"{kind}.png"
        saver(tensor, path)
        paths[kind] = path
    return paths
