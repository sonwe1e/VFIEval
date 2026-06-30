from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.devices import autocast_context, resolve_torch_device
from vfieval.models import load_flow_mask_model
from vfieval.pipeline.io import batch_tensors, load_rgb_tensor, resize_batch
from vfieval.pipeline.postprocess import compose_interpolated, normalize_model_outputs
from vfieval.pipeline.visualize import save_difference, save_extra_tensor, save_preview_image, save_visual_bundle


VALID_PRECISIONS = {"fp32", "fp16", "bf16"}
CORE_OUTPUTS = {"flowt_0", "flowt_1", "mask0", "mask1"}


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

    db.update_job_progress(job_id, 0, len(samples))
    if run_id is not None:
        db.mark_run_started(run_id, "running")
        if is_shard:
            db.update_run_progress_from_jobs(run_id, "running")
        else:
            db.update_run_progress(run_id, 0, len(samples), "running")
    model_row = db.get_model(model_id)
    model = load_flow_mask_model(
        adapter=model_row["adapter"],
        checkpoint_path=model_row.get("checkpoint_path"),
        device=str(device),
        metadata=model_row.get("metadata") or {},
    )

    run_dir = workspace.runs_dir / (str(run_id) if run_id is not None else f"inference_{job_id:06d}")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(run_dir, job, model_row, db.get_dataset(dataset_id) if dataset_id else None)

    timing = {"decode": 0.0, "model": 0.0, "post": 0.0, "save": 0.0}
    processed = 0
    video_groups: dict[str, dict[str, Any]] = {}

    for batch_start in range(0, len(samples), batch_size):
        _raise_if_canceled(db, run_id, job_id)
        batch_rows = samples[batch_start : batch_start + batch_size]

        t0 = time.perf_counter()
        img0 = _load_resized_batch([row["img0_path"] for row in batch_rows], device, height, width)
        img1 = _load_resized_batch([row["img1_path"] for row in batch_rows], device, height, width)
        timing["decode"] += time.perf_counter() - t0

        t1 = time.perf_counter()
        with torch.no_grad(), _autocast_context(device, precision):
            outputs = model.predict(img0, img1, 0.5)
        normalized_outputs = normalize_model_outputs(outputs, img0)
        timing["model"] += time.perf_counter() - t1

        t2 = time.perf_counter()
        composed = compose_interpolated(img0, img1, normalized_outputs)
        bundle = {**composed, "flowt_0": normalized_outputs["flowt_0"], "flowt_1": normalized_outputs["flowt_1"]}
        timing["post"] += time.perf_counter() - t2

        t3 = time.perf_counter()
        for idx, row in enumerate(batch_rows):
            _raise_if_canceled(db, run_id, job_id)
            try:
                sample_dir = run_dir / sanitize_name(row["name"])
                paths = save_visual_bundle(bundle, sample_dir, idx)
                for kind, path in paths.items():
                    _add_image_artifact_with_preview(db, job_id, int(row["id"]), kind, path, {"sample": row["name"]})
                extra_paths = _save_extra_outputs(outputs, sample_dir, idx)
                for kind, path in extra_paths.items():
                    _add_image_artifact_with_preview(db, job_id, int(row["id"]), kind, path, {"sample": row["name"]})

                diff_path = None
                if row.get("gt_path"):
                    gt = load_rgb_tensor(row["gt_path"], device).unsqueeze(0)
                    gt = resize_batch(gt, height, width)[0]
                    _add_image_artifact_with_preview(db, job_id, int(row["id"]), "gt", Path(row["gt_path"]), {"sample": row["name"]})
                    diff_path = sample_dir / "difference.png"
                    save_difference(composed["pred"][idx], gt, diff_path)
                    _add_image_artifact_with_preview(db, job_id, int(row["id"]), "difference", diff_path, {"sample": row["name"]})
                _collect_video_frame(video_groups, row, paths["pred"], diff_path)
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
        timing["save"] += time.perf_counter() - t3

    if video_groups:
        _write_video_artifacts(db, job_id, run_dir, video_groups)

    result = InferenceJobResult(
        samples=processed,
        output_dir=str(run_dir),
        decode_fps=_fps(processed, timing["decode"]),
        model_fps=_fps(processed, timing["model"]),
        postprocess_fps=_fps(processed, timing["post"]),
        save_fps=_fps(processed, timing["save"]),
    )

    result_dict = result.__dict__
    artifact_summary = db.summarize_artifacts(job_id)

    if is_shard:
        return result

    if metric_names:
        metric_payload = {
            "inference_job_id": job_id,
            "dataset_id": dataset_id,
            "metric_names": metric_names,
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
    db.update_job_progress(job_id, 0, len(samples))
    if run_id is not None:
        db.mark_run_started(run_id, "running")
        if is_shard:
            db.update_run_progress_from_jobs(run_id, "running")
        else:
            db.update_run_progress(run_id, 0, len(samples), "running")

    run_dir = workspace.runs_dir / (str(run_id) if run_id is not None else f"inference_{job_id:06d}")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(run_dir, job, db.get_model(int(job["payload"]["model_id"])), db.get_dataset(dataset_id) if dataset_id else None)

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
) -> int:
    preview_path = path.parent / "preview" / path.name
    preview_metadata = dict(metadata)
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


def _collect_video_frame(
    video_groups: dict[str, dict[str, Any]],
    sample: dict[str, Any],
    pred_path: Path,
    diff_path: Path | None,
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
    group["frames"].append(
        {
            "order": frame_order,
            "sample_name": sample["name"],
            "pred_path": Path(pred_path),
            "gt_path": Path(sample["gt_path"]) if sample.get("gt_path") else None,
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
