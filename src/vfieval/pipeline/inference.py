from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.models import load_flow_mask_model
from vfieval.pipeline.io import batch_tensors, load_rgb_tensor, resize_batch
from vfieval.pipeline.postprocess import compose_interpolated, validate_model_outputs
from vfieval.pipeline.visualize import save_difference, save_extra_tensor, save_visual_bundle


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
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU device requested but torch_npu is not installed") from exc
    return torch.device(device_name)


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32" or device.type not in {"cuda"}:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=True)


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

    if precision == "auto":
        precision = "fp16" if device.type == "cuda" else "fp32"
    if precision in {"fp16", "bf16"} and device.type != "cuda":
        precision = "fp32"
    if precision not in VALID_PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(VALID_PRECISIONS)}, got {precision}")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model_row = db.get_model(model_id)
    samples = db.list_samples(dataset_id)
    if not samples:
        raise ValueError(f"dataset {dataset_id} has no samples")

    db.update_job_progress(job_id, 0, len(samples))
    if run_id is not None:
        db.mark_run_started(run_id, "running")
        db.update_run_progress(run_id, 0, len(samples), "running")
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
        validate_model_outputs(outputs, img0)
        timing["model"] += time.perf_counter() - t1

        t2 = time.perf_counter()
        composed = compose_interpolated(img0, img1, outputs)
        bundle = {**composed, "flowt_0": outputs["flowt_0"], "flowt_1": outputs["flowt_1"]}
        timing["post"] += time.perf_counter() - t2

        t3 = time.perf_counter()
        for idx, row in enumerate(batch_rows):
            _raise_if_canceled(db, run_id, job_id)
            sample_dir = run_dir / sanitize_name(row["name"])
            paths = save_visual_bundle(bundle, sample_dir, idx)
            for kind, path in paths.items():
                db.add_artifact(job_id, int(row["id"]), kind, str(path), _artifact_mime(kind), {"sample": row["name"]})
            extra_paths = _save_extra_outputs(outputs, sample_dir, idx)
            for kind, path in extra_paths.items():
                db.add_artifact(job_id, int(row["id"]), kind, str(path), "image/png", {"sample": row["name"]})

            diff_path = None
            if row.get("gt_path"):
                gt = load_rgb_tensor(row["gt_path"], device).unsqueeze(0)
                gt = resize_batch(gt, height, width)[0]
                diff_path = sample_dir / "difference.png"
                save_difference(composed["pred"][idx], gt, diff_path)
                db.add_artifact(job_id, int(row["id"]), "difference", str(diff_path), "image/png", {"sample": row["name"]})
            _collect_video_frame(video_groups, row, paths["pred"], diff_path)

            processed += 1
            db.update_job_progress(job_id, processed)
            if run_id is not None:
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
        video_dir = run_dir / "videos" / video_name
        pred_frames_dir = video_dir / "pred_frames"
        gt_frames_dir = video_dir / "gt_frames"
        diff_frames_dir = video_dir / "diff_frames"
        pred_frame_paths = _copy_ordered_frames([frame["pred_path"] for frame in frames], pred_frames_dir)
        pred_video_path = video_dir / "pred.mp4"
        _write_mp4(pred_frame_paths, pred_video_path, fps)
        db.add_artifact(
            job_id,
            None,
            "pred_video",
            str(pred_video_path),
            "video/mp4",
            {"video_name": group["video_name"], "frames": len(frames)},
        )

        gt_paths = [frame["gt_path"] for frame in frames if frame["gt_path"] is not None]
        if len(gt_paths) == len(frames):
            gt_frame_paths = _copy_ordered_frames(gt_paths, gt_frames_dir)
            gt_video_path = video_dir / "gt.mp4"
            _write_mp4(gt_frame_paths, gt_video_path, fps)
            db.add_artifact(
                job_id,
                None,
                "gt_video",
                str(gt_video_path),
                "video/mp4",
                {"video_name": group["video_name"], "frames": len(frames)},
            )

        diff_paths = [frame["diff_path"] for frame in frames if frame["diff_path"] is not None]
        if len(diff_paths) == len(frames):
            diff_frame_paths = _copy_ordered_frames(diff_paths, diff_frames_dir)
            diff_video_path = video_dir / "diff.mp4"
            _write_mp4(diff_frame_paths, diff_video_path, fps)
            db.add_artifact(
                job_id,
                None,
                "diff_video",
                str(diff_video_path),
                "video/mp4",
                {"video_name": group["video_name"], "frames": len(frames)},
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


def _copy_ordered_frames(frame_paths: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for index, frame_path in enumerate(frame_paths):
        target = output_dir / f"{index:06d}.png"
        shutil.copy2(frame_path, target)
        copied.append(target)
    return copied


def _write_mp4(frame_paths: list[Path], output_path: Path, fps: float) -> None:
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
