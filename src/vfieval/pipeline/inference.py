from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.devices import autocast_context, resolve_torch_device
from vfieval.models import load_flow_mask_model
from vfieval.pipeline.io import batch_tensors, load_rgb_tensor, resize_batch
from vfieval.pipeline.postprocess import (
    compose_interpolated,
    normalize_model_outputs,
)
from vfieval.pipeline.visualize import save_difference, save_extra_tensor, save_preview_image, save_rgb_tensor


VALID_PRECISIONS = {"fp32", "fp16", "bf16"}
DEFAULT_VISUALIZE_HEIGHT = 384
DEFAULT_VISUALIZE_WIDTH = 832
# Above this max edge a downscaled preview thumbnail is worth its extra encode;
# at or below it the saved artifact is already small enough to display directly,
# so skipping the preview removes redundant save-pool work on long videos.
PREVIEW_SKIP_MAX_EDGE = 1024
CORE_OUTPUTS = {"flowt_0", "flowt_1", "mask0", "mask1"}
BUNDLE_KEYS = ("pred", "warp0", "warp1", "blend", "mask0", "mask1", "flowt_0", "flowt_1")


class RunCanceled(RuntimeError):
    pass


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


def resolve_device(device_name: str) -> torch.device:
    return resolve_torch_device(device_name)


def _autocast_context(device: torch.device, precision: str):
    return autocast_context(device, precision)


def _artifact_mime(kind: str) -> str:
    if kind.endswith("video"):
        return "video/mp4"
    return "image/png"


def run_inference_job(db: Database, workspace: WorkspaceConfig, job_id: int) -> InferenceJobResult:
    job = db.get_job(job_id)
    payload = job["payload"]
    model_id = int(payload["model_id"])
    dataset_id = int(payload["dataset_id"])
    height = int(payload.get("height") or payload.get("input_height") or 0)
    width = int(payload.get("width") or payload.get("input_width") or 0)
    batch_size = int(payload.get("batch_size", 1))
    device = resolve_device(str(payload.get("device", "auto")))
    precision = str(payload.get("precision", "fp32"))
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

    db.update_job_progress(job_id, 0, len(samples))
    if run_id is not None:
        db.mark_run_started(run_id, "running")
        if is_shard:
            db.update_run_progress_from_jobs(run_id, "running")
        else:
            db.update_run_progress(run_id, 0, len(samples), "running")
    model = load_flow_mask_model(
        adapter=model_row["adapter"],
        checkpoint_path=model_row.get("checkpoint_path"),
        device=str(device),
        metadata=model_row.get("metadata") or {},
    )
    model_load_report = _extract_model_load_report(model)

    if model_load_report is not None:
        _write_model_load_log(run_dir, model_load_report)

    save_workers = max(1, int(payload.get("save_workers") or min(8, os.cpu_count() or 4)))
    prefetch_workers = max(1, int(payload.get("prefetch_workers") or 2))
    pipeline = _AsyncSavePipeline(
        db=db,
        job_id=job_id,
        run_id=run_id,
        is_shard=is_shard,
        run_dir=run_dir,
        save_workers=save_workers,
    )

    timing = {"decode": 0.0, "model": 0.0, "post": 0.0, "save": 0.0}
    output_health = _OutputHealthAccumulator()

    try:
        for batch_rows, img0_cpu, img1_cpu, gt_cpu_list, prefetch_wait in _iter_prefetched_batches(
            samples=samples,
            batch_size=batch_size,
            height=height,
            width=width,
            has_gt=True,
            workers=prefetch_workers,
        ):
            _raise_if_canceled(db, run_id, job_id)
            timing["decode"] += prefetch_wait

            t1 = time.perf_counter()
            img0 = img0_cpu.to(device, non_blocking=True)
            img1 = img1_cpu.to(device, non_blocking=True)
            with torch.no_grad(), _autocast_context(device, precision):
                outputs = model.predict(img0, img1, 0.5)
            timing["model"] += time.perf_counter() - t1

            t2 = time.perf_counter()
            # Compose at the visualization resolution: downscale the (near
            # full-res) source frames to viz size, upsample the model's low-res
            # flow/mask to match, then warp. Warping sharp sources keeps pred
            # crisp (warping the model's native 208x448 pixels would blur it),
            # while composing at viz res instead of full inference res keeps the
            # on-device work and the PNG payload small.
            img0_viz = _resize_to_device(img0, visualize_height, visualize_width)
            img1_viz = _resize_to_device(img1, visualize_height, visualize_width)
            normalized_viz = normalize_model_outputs(outputs, img0_viz)
            composed_viz = compose_interpolated(img0_viz, img1_viz, normalized_viz)
            bundle_cpu = {
                "pred": composed_viz["pred"].detach().to("cpu"),
                "warp0": composed_viz["warp0"].detach().to("cpu"),
                "warp1": composed_viz["warp1"].detach().to("cpu"),
                "blend": composed_viz["blend"].detach().to("cpu"),
                "mask0": composed_viz["mask0"].detach().to("cpu"),
                "mask1": composed_viz["mask1"].detach().to("cpu"),
                "flowt_0": normalized_viz["flowt_0"].detach().to("cpu"),
                "flowt_1": normalized_viz["flowt_1"].detach().to("cpu"),
            }
            output_health.update(bundle_cpu)
            extra_cpu: dict[str, torch.Tensor] = {}
            for name, tensor in outputs.items():
                if name in CORE_OUTPUTS or not isinstance(tensor, torch.Tensor):
                    continue
                try:
                    extra_cpu[name] = tensor.detach().to("cpu")
                except Exception:
                    continue
            timing["post"] += time.perf_counter() - t2

            t3 = time.perf_counter()
            pipeline.submit_batch(
                batch_rows=batch_rows,
                bundle_cpu=bundle_cpu,
                extra_cpu=extra_cpu,
                gt_cpu_list=gt_cpu_list,
            )
            timing["save"] += time.perf_counter() - t3

        pipeline.wait_for_all()
    except BaseException:
        pipeline.shutdown()
        raise

    processed = pipeline.processed_count
    video_groups = pipeline.video_groups
    pipeline.shutdown()

    if video_groups:
        _write_video_artifacts(db, job_id, run_dir, video_groups)

    output_health_report = output_health.to_dict()
    _write_output_health_log(run_dir, output_health_report)

    result = InferenceJobResult(
        samples=processed,
        output_dir=str(run_dir),
        decode_fps=_fps(processed, timing["decode"]),
        model_fps=_fps(processed, timing["model"]),
        postprocess_fps=_fps(processed, timing["post"]),
        save_fps=_fps(processed, timing["save"]),
        output_health=output_health_report,
    )

    result_dict = dict(result.__dict__)
    if model_load_report is not None:
        result_dict["model_load"] = model_load_report
    artifact_summary = db.summarize_artifacts(job_id)

    if is_shard:
        return result

    if metric_names:
        metric_payload = {
            "inference_job_id": job_id,
            "dataset_id": dataset_id,
            "metric_names": metric_names,
            "metric_device": str(device),
        }
        if run_id is not None:
            metric_payload["run_id"] = run_id
        metric_job_id = db.create_job(
            "metric",
            metric_payload,
        )
        if run_id is not None:
            db.complete_run_inference(run_id, result_dict, artifact_summary, "metric_queued")
            db.set_run_metric_job(run_id, metric_job_id)
    elif run_id is not None:
        db.complete_run_inference(run_id, result_dict, artifact_summary, "completed")

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
    _write_run_metadata(run_dir, job, db.get_model(int(job["payload"]["model_id"])), db.get_dataset(dataset_id) if dataset_id else None)

    db.update_job_progress(job_id, 0, len(samples))
    if run_id is not None:
        db.mark_run_started(run_id, "running")
        if is_shard:
            db.update_run_progress_from_jobs(run_id, "running")
        else:
            db.update_run_progress(run_id, 0, len(samples), "running")

    processed = 0
    video_groups: dict[str, dict[str, Any]] = {}
    save_seconds = 0.0
    for row in samples:
        _raise_if_canceled(db, run_id, job_id)
        t0 = time.perf_counter()
        try:
            sample_dir = run_dir / sanitize_name(row["name"])
            sample_dir.mkdir(parents=True, exist_ok=True)
            gt_output_path = _copy_compare_image(Path(row["gt_path"]), sample_dir / "gt.png")
            pred_output_path = _copy_compare_image(Path(row["img1_path"]), sample_dir / "pred.png")
            diff_output_path = sample_dir / "difference.png"
            save_difference(load_rgb_tensor(pred_output_path), load_rgb_tensor(gt_output_path), diff_output_path)

            artifact_metadata = {"sample": row["name"], **_compare_track_metadata(row)}
            _add_image_artifact_with_preview(db, job_id, int(row["id"]), "gt", gt_output_path, artifact_metadata)
            _add_image_artifact_with_preview(db, job_id, int(row["id"]), "pred", pred_output_path, artifact_metadata)
            _add_image_artifact_with_preview(db, job_id, int(row["id"]), "difference", diff_output_path, artifact_metadata)
            _collect_compare_frame(video_groups, row, gt_output_path, pred_output_path, diff_output_path)
        except RunCanceled:
            raise
        except Exception as exc:
            _record_sample_error(db, job_id, int(row["id"]), row["name"], exc)

        processed += 1
        db.update_job_progress(job_id, processed)
        if run_id is not None:
            if is_shard:
                db.update_run_progress_from_jobs(run_id, "running")
            else:
                db.update_run_progress(run_id, processed)
        save_seconds += time.perf_counter() - t0

    if video_groups:
        _write_video_artifacts(db, job_id, run_dir, video_groups)

    result = InferenceJobResult(
        samples=processed,
        output_dir=str(run_dir),
        decode_fps=0.0,
        model_fps=0.0,
        postprocess_fps=0.0,
        save_fps=_fps(processed, save_seconds),
    )
    artifact_summary = db.summarize_artifacts(job_id)
    if is_shard:
        return result
    if metric_names:
        metric_payload = {
            "inference_job_id": job_id,
            "dataset_id": dataset_id,
            "metric_names": metric_names,
            "metric_device": str((run or {}).get("device") or job.get("payload", {}).get("device") or "cpu"),
        }
        if run_id is not None:
            metric_payload["run_id"] = run_id
        metric_job_id = db.create_job("metric", metric_payload)
        if run_id is not None:
            db.complete_run_inference(run_id, result.__dict__, artifact_summary, "metric_queued")
            db.set_run_metric_job(run_id, metric_job_id)
    elif run_id is not None:
        db.complete_run_inference(run_id, result.__dict__, artifact_summary, "completed")
    return result


def _resolve_visualize_size(payload: dict[str, Any], height: int, width: int) -> tuple[int, int]:
    """Resolution at which visual artifacts (PNGs) are saved.

    Defaults to 832x384 so the save pool encodes small images regardless of the
    full inference resolution. The visualization size is clamped to never exceed
    the inference resolution (upscaling artifacts for display wastes disk and CPU
    without adding information).
    """
    raw_h = payload.get("visualize_height")
    raw_w = payload.get("visualize_width")
    vis_h = int(raw_h) if raw_h else DEFAULT_VISUALIZE_HEIGHT
    vis_w = int(raw_w) if raw_w else DEFAULT_VISUALIZE_WIDTH
    if vis_h <= 0 or vis_w <= 0:
        vis_h, vis_w = DEFAULT_VISUALIZE_HEIGHT, DEFAULT_VISUALIZE_WIDTH
    vis_h = min(vis_h, height)
    vis_w = min(vis_w, width)
    return vis_h, vis_w


def _resize_chw(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a CHW RGB tensor to (height, width) via bilinear interpolation."""
    if tuple(tensor.shape[-2:]) == (height, width):
        return tensor
    resized = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0)


def _resize_to_device(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a BCHW tensor in place on its current device (no host copy)."""
    if tuple(tensor.shape[-2:]) == (height, width):
        return tensor
    return torch.nn.functional.interpolate(
        tensor, size=(height, width), mode="bilinear", align_corners=False
    )


def _fps(count: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return float(count) / seconds


def _load_resized_batch(paths: list[str], device: torch.device, height: int, width: int) -> torch.Tensor:
    tensors = []
    for path in paths:
        tensor = load_rgb_tensor(path, device).unsqueeze(0)
        tensors.append(resize_batch(tensor, height, width)[0])
    return batch_tensors(tensors)


def _add_image_artifact_with_preview(
    db: Database,
    job_id: int,
    sample_id: int,
    kind: str,
    path: Path,
    metadata: dict[str, Any],
    make_preview: bool = True,
) -> int:
    # Previews exist so the UI can render a small thumbnail without fetching a
    # multi-megapixel original. When the artifact itself is already small (the
    # visualization resolution defaults to 832x384, at or below the 512px
    # preview edge), the extra thumbnail encode is pure overhead on the save
    # pool and the UI falls back to the original URL when no preview exists.
    preview_metadata = dict(metadata)
    if make_preview:
        preview_path = path.parent / "preview" / path.name
        try:
            preview = save_preview_image(path, preview_path)
            preview_metadata.update({"preview_path": str(preview), "preview_max_edge": 512})
        except Exception:
            pass
    return db.add_artifact(job_id, sample_id, kind, str(path), "image/png", preview_metadata)


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


def _raise_if_canceled(db: Database, run_id: int | None, job_id: int) -> None:
    if run_id is None:
        return
    run = db.get_run(run_id)
    if run["status"] == "cancel_requested":
        error = {"message": "用户取消了 Run", "type": "RunCanceled"}
        db.cancel_job(job_id, error)
        db.cancel_run(run_id, error)
        raise RunCanceled("用户取消了 Run")
    if run["status"] == "failed":
        # A sibling shard already failed the run (multi_cuda/multi_npu). Stop
        # this shard instead of burning device time toward a run that is
        # already terminal.
        error = {"message": "sibling shard failed the run", "type": "RunCanceled"}
        db.cancel_job(job_id, error)
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
        },
    )
    frame_order = int(metadata.get("frame_index") or metadata.get("sample_index") or len(group["frames"]))
    # Prefer the visualization-resolution GT written alongside pred so pred/gt
    # video frames share dimensions (VMAF requires matched sizes). Fall back to
    # the original decoded GT when no resized copy was produced.
    resolved_gt = Path(gt_path) if gt_path is not None else (Path(sample["gt_path"]) if sample.get("gt_path") else None)
    group["frames"].append(
        {
            "order": frame_order,
            "sample_name": sample["name"],
            "pred_path": Path(pred_path),
            "gt_path": resolved_gt,
            "diff_path": Path(diff_path) if diff_path else None,
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
    frame_order = int(metadata.get("frame_index") or metadata.get("sample_index") or len(group["frames"]))
    group["frames"].append(
        {
            "order": frame_order,
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


def _write_video_artifacts(
    db: Database,
    job_id: int,
    run_dir: Path,
    video_groups: dict[str, dict[str, Any]],
) -> None:
    for group in video_groups.values():
        frames = sorted(group["frames"], key=lambda item: item["order"])
        if not frames:
            continue
        video_name = sanitize_name(str(group["video_name"]))
        fps = float(group["fps"] or 24.0)
        if any(frame.get("track_label") for frame in frames):
            _write_multitrack_compare_video_artifacts(db, job_id, run_dir, group, frames, video_name, fps)
            continue
        video_dir = run_dir / "videos" / video_name
        pred_frames_dir = video_dir / "pred_frames"
        gt_frames_dir = video_dir / "gt_frames"
        diff_frames_dir = video_dir / "diff_frames"
        pred_frame_paths = _copy_ordered_frames([frame["pred_path"] for frame in frames], pred_frames_dir)
        pred_video_path = video_dir / "pred.mp4"
        _write_mp4(pred_frame_paths, pred_video_path, fps)
        pred_metadata = _video_artifact_metadata(group["video_name"], frames, pred_frame_paths, fps)
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
            gt_frame_paths = _copy_ordered_frames(gt_paths, gt_frames_dir)
            gt_video_path = video_dir / "gt.mp4"
            _write_mp4(gt_frame_paths, gt_video_path, fps)
            gt_metadata = _video_artifact_metadata(group["video_name"], frames, gt_frame_paths, fps)
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
            diff_frame_paths = _copy_ordered_frames(diff_paths, diff_frames_dir)
            diff_video_path = video_dir / "diff.mp4"
            _write_mp4(diff_frame_paths, diff_video_path, fps)
            diff_metadata = _video_artifact_metadata(group["video_name"], frames, diff_frame_paths, fps)
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
            "pred_video": str(pred_video_path.resolve()),
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

    gt_video_path = None
    ordered_gt = [gt_by_order[index] for index in sorted(gt_by_order)]
    if ordered_gt:
        gt_frame_paths = _copy_ordered_frames(ordered_gt, video_dir / "gt_frames")
        gt_video_path = video_dir / "gt.mp4"
        _write_mp4(gt_frame_paths, gt_video_path, fps)
        db.add_artifact(
            job_id,
            None,
            "gt_video",
            str(gt_video_path),
            "video/mp4",
            _video_artifact_metadata(group["video_name"], frames, gt_frame_paths, fps),
        )

    manifest_tracks = []
    for track_key, track_frames in sorted(tracks.items()):
        ordered = sorted(track_frames, key=lambda item: item["order"])
        if not ordered:
            continue
        track_label = str(ordered[0].get("track_label") or track_key)
        track_dir = video_dir / sanitize_name(track_label)
        pred_frame_paths = _copy_ordered_frames([Path(frame["pred_path"]) for frame in ordered], track_dir / "pred_frames")
        pred_video_path = track_dir / "pred.mp4"
        _write_mp4(pred_frame_paths, pred_video_path, fps)
        track_metadata = {
            "compare_track_label": track_label,
            "compare_track_key": track_key,
            "compare_track_run_id": ordered[0].get("track_run_id"),
            "compare_track_artifact_id": ordered[0].get("track_artifact_id"),
        }
        db.add_artifact(
            job_id,
            None,
            "pred_video",
            str(pred_video_path),
            "video/mp4",
            {**_video_artifact_metadata(group["video_name"], ordered, pred_frame_paths, fps), **track_metadata},
        )

        diff_video_path = None
        diff_paths = [Path(frame["diff_path"]) for frame in ordered if frame.get("diff_path") is not None]
        if len(diff_paths) == len(ordered):
            diff_frame_paths = _copy_ordered_frames(diff_paths, track_dir / "diff_frames")
            diff_video_path = track_dir / "diff.mp4"
            _write_mp4(diff_frame_paths, diff_video_path, fps)
            db.add_artifact(
                job_id,
                None,
                "diff_video",
                str(diff_video_path),
                "video/mp4",
                {**_video_artifact_metadata(group["video_name"], ordered, diff_frame_paths, fps), **track_metadata},
            )

        manifest_tracks.append(
            {
                "track_label": track_label,
                "track_key": track_key,
                "frames": len(ordered),
                "pred_video": str(pred_video_path.resolve()),
                "diff_video": str(diff_video_path.resolve()) if diff_video_path else None,
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


def _copy_ordered_frames(frame_paths: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for index, frame_path in enumerate(frame_paths):
        target = output_dir / f"{index:06d}.png"
        shutil.copy2(frame_path, target)
        copied.append(target)
    return copied


def _write_mp4(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    frame_dir = frame_paths[0].parent
    if _write_mp4_ffmpeg(frame_dir, output_path, fps):
        return
    _write_mp4_cv2(frame_paths, output_path, fps)


def _write_mp4_ffmpeg(frame_dir: Path, output_path: Path, fps: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frame_dir / "%06d.png"),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(output_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)
    except subprocess.TimeoutExpired:
        return False
    if result.returncode != 0:
        return False
    return output_path.exists()


def _write_mp4_cv2(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("writing video artifacts requires opencv-python (cv2)") from exc

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"failed to read video frame: {frame_paths[0]}")
    height, width = first.shape[:2]
    encode_width = width if width % 2 == 0 else width + 1
    encode_height = height if height % 2 == 0 else height + 1
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (encode_width, encode_height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (encode_width, encode_height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to create video artifact: {output_path}")
    try:
        writer.write(_fit_video_frame(first, encode_width, encode_height))
        for frame_path in frame_paths[1:]:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"failed to read video frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(_fit_video_frame(frame, encode_width, encode_height))
    finally:
        writer.release()


def _fit_video_frame(frame, width: int, height: int):
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("resizing video frames requires opencv-python (cv2)") from exc
    return cv2.copyMakeBorder(frame, 0, height - frame.shape[0], 0, width - frame.shape[1], cv2.BORDER_REPLICATE)


def _load_rgb_cpu(path: str, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)
    if tensor.shape[-2:] != (height, width):
        tensor = torch.nn.functional.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.squeeze(0)


def _load_batch_cpu(paths: list[str], height: int, width: int) -> torch.Tensor:
    tensors = [_load_rgb_cpu(path, height, width) for path in paths]
    stacked = torch.stack(tensors, dim=0)
    if torch.cuda.is_available():
        try:
            stacked = stacked.pin_memory()
        except Exception:
            pass
    return stacked


def _iter_prefetched_batches(
    samples: list[dict[str, Any]],
    batch_size: int,
    height: int,
    width: int,
    has_gt: bool,
    workers: int,
):
    """Yield (batch_rows, img0_cpu, img1_cpu, gt_cpu_list, wait_seconds) tuples.

    Prefetches up to 2 batches ahead on a small CPU pool so PIL decode and
    resize overlap with device compute. wait_seconds is how long the main
    loop blocked waiting for the next batch to arrive.
    """
    if not samples:
        return
    workers = max(1, int(workers or 1))
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vfi-prefetch")
    pending: list[tuple[list[dict[str, Any]], Future, Future, Future | None]] = []
    max_ahead = 2

    def _submit(batch_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Future, Future, Future | None]:
        img0_paths = [row["img0_path"] for row in batch_rows]
        img1_paths = [row["img1_path"] for row in batch_rows]
        img0_future = pool.submit(_load_batch_cpu, img0_paths, height, width)
        img1_future = pool.submit(_load_batch_cpu, img1_paths, height, width)
        gt_future: Future | None = None
        if has_gt and any(row.get("gt_path") for row in batch_rows):
            def _load_gts(paths: list[str | None]) -> list[torch.Tensor | None]:
                loaded: list[torch.Tensor | None] = []
                for path in paths:
                    if not path:
                        loaded.append(None)
                        continue
                    loaded.append(_load_rgb_cpu(path, height, width))
                return loaded

            gt_future = pool.submit(_load_gts, [row.get("gt_path") for row in batch_rows])
        return batch_rows, img0_future, img1_future, gt_future

    clean_exit = False
    try:
        cursor = 0
        while cursor < len(samples) and len(pending) < max_ahead:
            end = min(cursor + batch_size, len(samples))
            pending.append(_submit(samples[cursor:end]))
            cursor = end

        while pending:
            batch_rows, img0_future, img1_future, gt_future = pending.pop(0)
            wait_start = time.perf_counter()
            img0_cpu = img0_future.result()
            img1_cpu = img1_future.result()
            gt_cpu_list = gt_future.result() if gt_future is not None else [None] * len(batch_rows)
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
        save_workers: int,
    ) -> None:
        self._db = db
        self._job_id = job_id
        self._run_id = run_id
        self._is_shard = is_shard
        self._run_dir = run_dir
        self._pool = ThreadPoolExecutor(max_workers=save_workers, thread_name_prefix="vfi-save")
        self._pending: list[Future] = []
        self._lock = threading.Lock()
        self._processed = 0
        self._total = 0
        self._video_groups: dict[str, dict[str, Any]] = {}
        self._last_progress_report = 0

    @property
    def processed_count(self) -> int:
        return self._processed

    @property
    def video_groups(self) -> dict[str, dict[str, Any]]:
        return self._video_groups

    def submit_batch(
        self,
        *,
        batch_rows: list[dict[str, Any]],
        bundle_cpu: dict[str, torch.Tensor],
        extra_cpu: dict[str, torch.Tensor],
        gt_cpu_list: list[torch.Tensor | None],
    ) -> None:
        self._total += len(batch_rows)
        with self._lock:
            for idx, row in enumerate(batch_rows):
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
                future.add_done_callback(self._on_sample_done)
                self._pending.append(future)

    def wait_for_all(self) -> None:
        while True:
            with self._lock:
                pending = list(self._pending)
                self._pending = []
            if not pending:
                return
            for future in pending:
                # _save_sample already catches its own exceptions and records
                # them via _record_sample_error, so a future.exception() here
                # means the save task itself crashed outside that try/except
                # (e.g. a bug in _on_sample_done). Surface those, since they
                # are not otherwise recorded anywhere.
                exc = future.exception()
                if exc is not None:
                    raise exc

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

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
            sample_dir = self._run_dir / sanitize_name(sample_name)
            paths = _save_visual_bundle_from_cpu(bundle, sample_dir)
            # Previews are only worth their extra encode when the artifact is
            # genuinely large. At the default visualization resolution the saved
            # image is already small, so skip the redundant thumbnail — the
            # biggest save-pool cost on long videos. The UI falls back to the
            # original URL when no preview exists.
            pred_h, pred_w = int(bundle["pred"].shape[-2]), int(bundle["pred"].shape[-1])
            make_preview = max(pred_h, pred_w) > PREVIEW_SKIP_MAX_EDGE
            for kind, path in paths.items():
                _add_image_artifact_with_preview(
                    self._db, job_id, sample_id, kind, path, {"sample": sample_name}, make_preview=make_preview
                )
            extra_paths: dict[str, Path] = {}
            for name, tensor in extra.items():
                try:
                    safe_name = sanitize_name(name)
                    path = sample_dir / f"extra_{safe_name}.png"
                    save_extra_tensor(tensor, path, index=0)
                    extra_paths[f"extra_{safe_name}"] = path
                except Exception:
                    continue
            for kind, path in extra_paths.items():
                _add_image_artifact_with_preview(
                    self._db, job_id, sample_id, kind, path, {"sample": sample_name}, make_preview=False
                )

            diff_path = None
            gt_path = None
            if gt_tensor is not None and row.get("gt_path"):
                # pred is saved at the visualization resolution; match GT to it
                # so the difference map and the pred/gt video pair (VMAF input)
                # share dimensions.
                pred_h, pred_w = int(bundle["pred"].shape[-2]), int(bundle["pred"].shape[-1])
                if tuple(gt_tensor.shape[-2:]) != (pred_h, pred_w):
                    gt_tensor = _resize_chw(gt_tensor, pred_h, pred_w)
                gt_path = sample_dir / "gt.png"
                save_rgb_tensor(gt_tensor, gt_path)
                _add_image_artifact_with_preview(
                    self._db, job_id, sample_id, "gt", gt_path, {"sample": sample_name}, make_preview=False
                )
                diff_path = sample_dir / "difference.png"
                save_difference(bundle["pred"], gt_tensor, diff_path)
                _add_image_artifact_with_preview(
                    self._db, job_id, sample_id, "difference", diff_path, {"sample": sample_name}, make_preview=False
                )
            with self._lock:
                _collect_video_frame(self._video_groups, row, paths["pred"], diff_path, gt_path)
        except Exception as exc:
            _record_sample_error(self._db, job_id, sample_id, sample_name, exc)

    def _on_sample_done(self, future: Future) -> None:
        with self._lock:
            self._processed += 1
            processed = self._processed
            total = self._total
            try:
                self._pending.remove(future)
            except ValueError:
                pass
            step = max(1, total // 200)
            report_now = processed == total or processed - self._last_progress_report >= step
            if report_now:
                self._last_progress_report = processed
        if not report_now:
            return
        try:
            self._db.update_job_progress(self._job_id, processed)
            if self._run_id is not None:
                if self._is_shard:
                    self._db.update_run_progress_from_jobs(self._run_id, "running")
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

    paths = {
        "pred": sample_dir / "pred.png",
        "warp0": sample_dir / "warp0.png",
        "warp1": sample_dir / "warp1.png",
        "blend": sample_dir / "blend.png",
        "mask0": sample_dir / "mask0.png",
        "mask1": sample_dir / "mask1.png",
        "flowt_0": sample_dir / "flowt_0.png",
        "flowt_1": sample_dir / "flowt_1.png",
    }
    save_rgb_tensor(bundle["pred"], paths["pred"])
    save_rgb_tensor(bundle["warp0"], paths["warp0"])
    save_rgb_tensor(bundle["warp1"], paths["warp1"])
    save_rgb_tensor(bundle["blend"], paths["blend"])
    save_mask(bundle["mask0"], paths["mask0"])
    save_mask(bundle["mask1"], paths["mask1"])
    save_flow(bundle["flowt_0"], paths["flowt_0"])
    save_flow(bundle["flowt_1"], paths["flowt_1"])
    return paths
