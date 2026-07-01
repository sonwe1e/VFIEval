from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.devices import device_type_name, list_npu_devices, normalize_device_name, npu_unavailable_reason, resolve_torch_device, supported_precisions
from vfieval.metrics.health import metrics_health
from vfieval.models import load_flow_mask_model
from vfieval.pipeline.postprocess import compose_interpolated, validate_model_outputs


VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
CHECKPOINT_SUFFIXES = {".bin", ".ckpt", ".pth", ".pt", ".safetensors"}
DECODE_STRATEGY_VERSION = "opencv-rgb-v1"
VIDEO_INSPECT_VERSION = "ffprobe-opencv-v3"


def project_root(workspace: WorkspaceConfig) -> Path:
    return Path(os.getenv("VFIEVAL_PROJECT_ROOT") or workspace.root.parent).resolve()


def models_dir(workspace: WorkspaceConfig) -> Path:
    return Path(os.getenv("VFIEVAL_MODELS_DIR") or project_root(workspace) / "models").resolve()


def videos_dir(workspace: WorkspaceConfig) -> Path:
    return Path(os.getenv("VFIEVAL_VIDEOS_DIR") or project_root(workspace) / "videos").resolve()


def checkpoints_dir(workspace: WorkspaceConfig) -> Path:
    return Path(os.getenv("VFIEVAL_CHECKPOINTS_DIR") or project_root(workspace) / "checkpoints").resolve()


def list_model_files(workspace: WorkspaceConfig) -> list[dict[str, Any]]:
    root = models_dir(workspace)
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        stat = path.stat()
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return rows


def list_checkpoints(workspace: WorkspaceConfig, model_file: str | None = None) -> list[dict[str, Any]]:
    root = checkpoints_dir(workspace)
    if not root.exists():
        return []
    model_stem = Path(model_file).stem if model_file else None
    search_roots = [root / model_stem] if model_stem else [path for path in sorted(root.iterdir()) if path.is_dir()]
    rows: list[dict[str, Any]] = []
    for folder in search_roots:
        if not folder.exists() or not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in CHECKPOINT_SUFFIXES:
                continue
            stat = path.stat()
            rows.append(
                {
                    "name": path.name,
                    "model": folder.name,
                    "relative_path": path.relative_to(root).as_posix(),
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return sorted(rows, key=lambda item: (str(item["model"]), -int(item["mtime_ns"]), str(item["relative_path"])))


def list_video_groups(workspace: WorkspaceConfig) -> list[dict[str, Any]]:
    root = videos_dir(workspace)
    if not root.exists():
        return []
    groups = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        videos = [_video_summary(workspace, video) for video in _iter_videos(folder)]
        groups.append(
            {
                "name": folder.name,
                "path": str(folder),
                "video_count": len(videos),
                "videos": videos,
            }
        )
    return groups


def list_video_group_videos(
    workspace: WorkspaceConfig,
    video_group: str,
    frame_step: int = 1,
    max_frames: int | None = None,
    page: int = 1,
    page_size: int = 50,
    query: str = "",
    sort: str = "name",
) -> dict[str, Any]:
    folder = resolve_video_group(workspace, video_group)
    all_paths = _iter_videos(folder)
    all_names = [path.name for path in all_paths]
    normalized_query = query.strip().lower()
    paths = [
        path for path in all_paths
        if not normalized_query or normalized_query in path.name.lower()
    ]
    videos = [
        video_summary(workspace, video, frame_step=frame_step, max_frames=max_frames, exact=False)
        for video in paths
    ]
    videos = _sort_video_summaries(videos, sort)
    page_size = min(200, max(1, int(page_size or 50)))
    page = max(1, int(page or 1))
    total = len(videos)
    start = (page - 1) * page_size
    paged = videos[start : start + page_size]
    return {
        "name": folder.name,
        "path": str(folder),
        "video_count": len(all_paths),
        "filtered_count": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "query": query,
        "sort": sort,
        "all_video_names": all_names,
        "videos": paged,
    }


def resolve_model_file(workspace: WorkspaceConfig, model_file: str) -> Path:
    if Path(model_file).name != model_file or Path(model_file).suffix.lower() != ".py":
        raise ValueError("模型文件必须来自 models/ 下的 .py 文件")
    path = (models_dir(workspace) / model_file).resolve()
    _ensure_child(path, models_dir(workspace), "模型文件不在 models/ 目录下")
    if not path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {model_file}")
    return path


def resolve_checkpoint(workspace: WorkspaceConfig, checkpoint: str | None, model_file: str | None = None) -> Path | None:
    if checkpoint in {None, "", "none"}:
        return None
    root = checkpoints_dir(workspace)
    if checkpoint == "auto":
        candidates = list_checkpoints(workspace, model_file)
        return Path(candidates[0]["path"]) if candidates else None
    checkpoint_path = Path(str(checkpoint))
    if checkpoint_path.is_absolute() or ".." in checkpoint_path.parts:
        raise ValueError("checkpoint must be a relative path under checkpoints/")
    path = (root / checkpoint_path).resolve()
    _ensure_child(path, root, "checkpoint is outside checkpoints/")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if path.suffix.lower() not in CHECKPOINT_SUFFIXES:
        raise ValueError(f"unsupported checkpoint file type: {path.name}")
    if model_file:
        model_root = (root / Path(model_file).stem).resolve()
        if model_root.exists():
            _ensure_child(path, model_root, f"checkpoint must be under checkpoints/{Path(model_file).stem}/")
    return path


def resolve_video_group(workspace: WorkspaceConfig, video_group: str) -> Path:
    if Path(video_group).name != video_group:
        raise ValueError("视频集必须是 videos/ 下的一级文件夹")
    path = (videos_dir(workspace) / video_group).resolve()
    _ensure_child(path, videos_dir(workspace), "视频集不在 videos/ 目录下")
    if not path.is_dir():
        raise FileNotFoundError(f"视频集不存在: {video_group}")
    return path


def resolve_selected_videos(workspace: WorkspaceConfig, video_group: str, selected_videos: Any) -> list[Path]:
    folder = resolve_video_group(workspace, video_group)
    all_videos = _iter_videos(folder)
    if selected_videos is None:
        return all_videos
    if isinstance(selected_videos, str):
        selected_names = [part.strip() for part in selected_videos.split(",") if part.strip()]
    else:
        selected_names = [str(item) for item in selected_videos]
    if not selected_names:
        raise ValueError("至少选择一个视频")

    by_name = {path.name: path for path in all_videos}
    resolved: list[Path] = []
    for name in selected_names:
        if Path(name).name != name:
            raise ValueError(f"视频选择只能使用文件名: {name}")
        path = by_name.get(name)
        if path is None:
            raise FileNotFoundError(f"选中的视频不存在或不支持: {name}")
        resolved.append(path)
    return resolved


def preflight_run(db: Database, workspace: WorkspaceConfig, payload: dict[str, Any]) -> dict[str, Any]:
    if str(payload.get("run_type") or "model_inference") == "video_compare":
        from vfieval.compare_inputs import (
            resolve_compare_descriptor,
            validate_strict_alignment,
            validate_strict_decoded_alignment,
        )
        from vfieval.datasets import _load_compare_source_frames

        health = metrics_health(workspace)
        selected_metrics = [str(name) for name in (payload.get("metrics") or [])]
        result: dict[str, Any] = {
            "ok": True,
            "run_type": "video_compare",
            "errors": [],
            "warnings": [],
            "reference": {},
            "distorted": {},
            "distorted_tracks": [],
            "alignment": {"mode": str(payload.get("align_mode") or "strict")},
            "metrics": {
                "requested": selected_metrics,
                "health": {
                    name: health["metrics"][name]
                    for name in selected_metrics
                    if name in health["metrics"]
                },
            },
            "output_dir": str(workspace.runs_dir / str(db.next_run_id())),
        }
        try:
            align_mode = str(payload.get("align_mode") or "strict")
            if align_mode != "strict":
                raise ValueError("video_compare currently only supports strict alignment for external inputs")
            reference = resolve_compare_descriptor(workspace, db, payload.get("reference"), role="reference")
            distorted_payload = payload.get("distorted")
            distorted_descriptors = distorted_payload if isinstance(distorted_payload, list) else [distorted_payload]
            distorted_descriptors = [item for item in distorted_descriptors if item is not None and item != ""]
            if not distorted_descriptors:
                raise ValueError("video_compare requires at least one distorted track")
            reference_frames, reference_fps, reference_timestamps = _load_compare_source_frames(
                workspace,
                Path(str(reference["path"])),
                "compare_reference",
            )
            distorted_tracks = []
            for track_index, descriptor in enumerate(distorted_descriptors):
                distorted = resolve_compare_descriptor(workspace, db, descriptor, role="distorted")
                track_label = str(distorted.get("track_label") or distorted.get("label") or f"pred{track_index + 1}")
                try:
                    validate_strict_alignment(reference, distorted)
                    distorted_frames, distorted_fps, distorted_timestamps = _load_compare_source_frames(
                        workspace,
                        Path(str(distorted["path"])),
                        f"compare_distorted_{track_index}",
                    )
                    validate_strict_decoded_alignment(
                        reference_frames=reference_frames,
                        distorted_frames=distorted_frames,
                        reference_fps=reference_fps,
                        distorted_fps=distorted_fps,
                        reference_timestamps=reference_timestamps,
                        distorted_timestamps=distorted_timestamps,
                    )
                except Exception as exc:
                    raise RuntimeError(f"track {track_label}: {exc}") from exc
                track = dict(distorted)
                track.update(
                    {
                        "track_index": track_index,
                        "track_label": track_label,
                        "frame_count": len(distorted_frames),
                        "fps": distorted_fps,
                    }
                )
                distorted_tracks.append(track)
            result["reference"] = reference
            result["distorted"] = distorted_tracks[0] if distorted_tracks else {}
            result["distorted_tracks"] = distorted_tracks
            result["alignment"] = {
                "mode": "strict",
                "frame_count": len(reference_frames),
                "width": int(reference["width"]),
                "height": int(reference["height"]),
                "fps": reference_fps if reference_fps is not None else (distorted_tracks[0].get("fps") if distorted_tracks else None),
                "track_count": len(distorted_tracks),
                "verified_with": "decoded_frames",
            }
        except Exception as exc:
            result["ok"] = False
            result["errors"].append({"title": "对比输入检查失败", "message": str(exc), "type": type(exc).__name__})
        for name, row in result["metrics"]["health"].items():
            if not row.get("available"):
                result["warnings"].append(
                    {
                        "title": f"metric {name}",
                        "message": row.get("reason") or row.get("status") or "metric is unavailable",
                        "type": "MetricUnavailable",
                    }
                )
        return result

    model_file = str(payload.get("model_file") or "")
    video_group = str(payload.get("video_group") or "")
    frame_step = max(1, int(payload.get("frame_step") or 1))
    max_frames = _optional_positive_int(payload.get("max_frames"))
    device_request = str(payload.get("device") or "auto")
    precision_request = str(payload.get("precision") or "auto")
    checkpoint_request = payload.get("checkpoint")
    execution_mode = str(payload.get("execution_mode") or "single")

    result: dict[str, Any] = {
        "ok": True,
        "run_type": "model_inference",
        "errors": [],
        "warnings": [],
        "model": {},
        "video_group": {},
        "device": {},
        "resolution": {},
        "cache": {},
        "metrics": {},
        "output_dir": str(workspace.runs_dir / str(db.next_run_id())),
    }

    if execution_mode == "multi_cuda":
        device_info = _check_multi_accelerator(payload, precision_request, "cuda")
    elif execution_mode == "multi_npu":
        device_info = _check_multi_accelerator(payload, precision_request, "npu")
    else:
        device_info = _check_device(device_request, precision_request)
    result["device"] = device_info
    if device_info["status"] == "error":
        result["ok"] = False
        result["errors"].append({"title": "设备检查失败", "message": device_info["message"], "type": "DeviceError"})
    if device_info.get("warning"):
        result["warnings"].append(device_info["warning"])

    model_path: Path | None = None
    tested_devices: list[str] = []
    last_diagnostics: dict[str, Any] | None = None
    try:
        model_path = resolve_model_file(workspace, model_file)
        checkpoint_path = resolve_checkpoint(workspace, checkpoint_request, model_path.name)
        for dry_run_device in _dry_run_devices(device_info):
            try:
                last_diagnostics = _dry_run_model_file(
                    model_path,
                    checkpoint_path,
                    dry_run_device,
                    str(device_info.get("effective_precision") or "fp32"),
                )
            except Exception as exc:
                raise RuntimeError(f"model dry-run failed on {dry_run_device}: {exc}") from exc
            tested_devices.append(dry_run_device)
        model_row: dict[str, Any] = {
            "name": model_path.name,
            "path": str(model_path),
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "interface_ok": True,
            "tested_devices": tested_devices,
        }
        if last_diagnostics is not None:
            health = last_diagnostics.get("output_health") or {}
            model_row["output_health"] = health
            model_load = last_diagnostics.get("model_load")
            if model_load is not None:
                model_row["model_load"] = model_load
                missing = list(model_load.get("missing_keys") or [])
                unexpected = list(model_load.get("unexpected_keys") or [])
                if missing or unexpected:
                    result["warnings"].append(
                        {
                            "title": "checkpoint 键不完全匹配",
                            "message": (
                                f"matched {model_load.get('matched')} / "
                                f"{model_load.get('total_in_checkpoint')}, "
                                f"missing {len(missing)}, unexpected {len(unexpected)}"
                            ),
                            "type": "CheckpointLoadReport",
                            "details": {"missing_keys": missing[:20], "unexpected_keys": unexpected[:20]},
                        }
                    )
            for warning_text in health.get("warnings") or []:
                result["warnings"].append(
                    {
                        "title": "模型输出异常",
                        "message": warning_text,
                        "type": "ModelOutputHealth",
                        "details": health.get("stats"),
                    }
                )
        result["model"] = model_row
    except Exception as exc:
        result["ok"] = False
        result["model"] = {"name": model_file, "interface_ok": False, "tested_devices": tested_devices}
        result["errors"].append(_error("模型检查失败", exc))

    video_folder: Path | None = None
    video_infos: list[dict[str, Any]] = []
    try:
        video_folder = resolve_video_group(workspace, video_group)
        video_paths = resolve_selected_videos(workspace, video_group, payload.get("selected_videos"))
        if not video_paths:
            raise FileNotFoundError("视频集中没有支持的视频文件")
        for video_path in video_paths:
            info = inspect_video(video_path, workspace, exact=True)
            info["valid_triplets"] = _valid_triplets(info["frame_count"], frame_step, max_frames)
            info["triplets"] = info["valid_triplets"]
            info["cache_key"] = decode_cache_key(video_path, "video_gt_triplets", frame_step, max_frames)
            info["cache_status"] = decode_cache_status(workspace, info["cache_key"])
            video_infos.append(info)
        short = [info["name"] for info in video_infos if info["triplets"] <= 0]
        bad = [info["name"] for info in video_infos if not info["decodable"]]
        if bad:
            raise RuntimeError(f"视频无法解码: {', '.join(bad)}")
        if short:
            raise RuntimeError(f"视频帧数不足 3 帧: {', '.join(short)}")
        result["video_group"] = {
            "name": video_folder.name,
            "path": str(video_folder),
            "video_count": len(video_infos),
            "selected_videos": [path.name for path in video_paths],
            "frame_count": sum(int(info["frame_count"]) for info in video_infos),
            "duration_seconds": sum(float(info.get("duration_seconds") or 0.0) for info in video_infos),
            "triplets": sum(int(info["triplets"]) for info in video_infos),
            "videos": video_infos,
        }
        statuses = {info["cache_status"] for info in video_infos}
        result["cache"] = {"status": "mixed" if len(statuses) > 1 else next(iter(statuses)), "videos": video_infos}
    except Exception as exc:
        result["ok"] = False
        result["video_group"] = {"name": video_group, "videos": video_infos}
        result["errors"].append(_error("视频集检查失败", exc))

    result["resolution"] = _resolve_preflight_resolution(payload, video_infos)
    health = metrics_health(workspace)
    selected_metrics = [str(name) for name in (payload.get("metrics") or [])]
    metric_rows = {
        name: health["metrics"][name]
        for name in selected_metrics
        if name in health["metrics"]
    }
    result["metrics"] = {
        "requested": selected_metrics,
        "health": metric_rows,
    }
    for name, row in metric_rows.items():
        if not row.get("available"):
            result["warnings"].append(
                {
                    "title": f"metric {name}",
                    "message": row.get("reason") or row.get("status") or "metric is unavailable",
                    "type": "MetricUnavailable",
                }
            )
    return result


def inspect_video(path: Path, workspace: WorkspaceConfig | None = None, exact: bool = True) -> dict[str, Any]:
    cached = _read_video_inspect_cache(path, workspace, exact)
    if cached is not None:
        return cached

    ffprobe = _inspect_video_ffprobe(path)
    opencv = _inspect_video_opencv(path, exact)
    if ffprobe is None and opencv is None:
        result = {
            "name": path.name,
            "path": str(path),
            "decodable": False,
            "error": "无法打开视频；ffprobe 不可用且 OpenCV 解码失败",
            "frame_count": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "duration_seconds": 0.0,
            "metadata_source": "none",
        }
        _write_video_inspect_cache(path, workspace, result, exact)
        return result

    base = dict(ffprobe or opencv or {})
    if opencv:
        if exact:
            base["frame_count"] = int(opencv.get("frame_count") or base.get("frame_count") or 0)
            base["frame_count_source"] = opencv.get("frame_count_source") or "exact"
        else:
            base["container_frame_count"] = int(opencv.get("container_frame_count") or base.get("container_frame_count") or 0)
            if not base.get("frame_count"):
                base["frame_count"] = int(opencv.get("frame_count") or 0)
                base["frame_count_source"] = opencv.get("frame_count_source") or "container"
        base["fps"] = float(base.get("fps") or opencv.get("fps") or 24.0)
        base["width"] = int(base.get("width") or opencv.get("width") or 0)
        base["height"] = int(base.get("height") or opencv.get("height") or 0)
    frame_count = int(base.get("frame_count") or 0)
    fps = float(base.get("fps") or 24.0)
    if not base.get("duration_seconds"):
        base["duration_seconds"] = float(frame_count) / fps if fps > 0 and frame_count > 0 else 0.0
    base.update(
        {
            "name": path.name,
            "path": str(path),
            "decodable": frame_count > 0,
            "error": None if frame_count > 0 else "没有可解码帧",
            "frame_count": frame_count,
            "fps": fps,
            "width": int(base.get("width") or 0),
            "height": int(base.get("height") or 0),
        }
    )
    _write_video_inspect_cache(path, workspace, base, exact)
    return base


def _inspect_video_opencv(path: Path, exact: bool) -> dict[str, Any] | None:
    try:
        import cv2
    except ImportError:
        return None

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    try:
        container_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = _exact_frame_count(capture) if exact else container_frame_count
        frame_count_source = "exact" if exact else "container"
        warning = None
        if exact and frame_count <= 0 and container_frame_count > 0:
            frame_count = container_frame_count
            frame_count_source = "estimated"
            warning = "OpenCV 无法精确计数，已使用容器声明帧数"
        if not exact and container_frame_count <= 0:
            warning = "容器未提供帧数，将在预检查时精确计数"
        duration_seconds = float(frame_count) / fps if fps > 0 and frame_count > 0 else 0.0
        return {
            "frame_count": frame_count,
            "container_frame_count": container_frame_count,
            "frame_count_source": frame_count_source,
            "frame_count_warning": warning,
            "duration_seconds": duration_seconds,
            "fps": fps,
            "width": width,
            "height": height,
            "metadata_source": "opencv",
        }
    finally:
        capture.release()


def _inspect_video_ffprobe(path: Path) -> dict[str, Any] | None:
    executable = shutil.which("ffprobe")
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,codec_name,pix_fmt,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return None
    streams = payload.get("streams") or []
    stream = streams[0] if streams else {}
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate")) or 24.0
    duration = _float_or_none(stream.get("duration")) or _float_or_none((payload.get("format") or {}).get("duration")) or 0.0
    nb_frames = _optional_positive_int(stream.get("nb_frames"))
    frame_count = nb_frames or (round(duration * fps) if duration > 0 and fps > 0 else 0)
    source = "ffprobe_nb_frames" if nb_frames else "ffprobe_duration"
    return {
        "frame_count": int(frame_count),
        "container_frame_count": int(frame_count),
        "frame_count_source": source,
        "frame_count_warning": None if nb_frames else "ffprobe 未提供 nb_frames，已按 duration * fps 估算",
        "duration_seconds": float(duration),
        "fps": float(fps),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "codec": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "metadata_source": "ffprobe",
    }


def decode_cache_key(video_path: Path, decode_mode: str, frame_step: int, max_frames: int | None) -> str:
    stat = video_path.stat()
    data = {
        "path": str(video_path.resolve()).replace("\\", "/"),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(video_path),
        "decode_mode": decode_mode,
        "frame_step": frame_step,
        "max_frames": max_frames,
        "strategy": DECODE_STRATEGY_VERSION,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def decode_cache_dir(workspace: WorkspaceConfig, cache_key: str) -> Path:
    return workspace.root / "decode_cache" / cache_key


def decode_cache_status(workspace: WorkspaceConfig, cache_key: str) -> str:
    cache_dir = decode_cache_dir(workspace, cache_key)
    partial_dir = cache_dir.with_name(cache_dir.name + ".partial")
    manifest = cache_dir / "manifest.json"
    if partial_dir.exists():
        return "需要更新"
    if manifest.exists():
        return "已缓存"
    return "未解码"


def thumbnail_path(workspace: WorkspaceConfig, key: str) -> Path:
    return workspace.root / "video_thumbnails" / f"{key}.webp"


def video_thumbnail_key(path: Path) -> str:
    stat = path.stat()
    data = {
        "path": str(path.resolve()).replace("\\", "/"),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "version": "video-thumb-v1",
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def ensure_video_thumbnail(workspace: WorkspaceConfig, path: Path, key: str) -> Path | None:
    target = thumbnail_path(workspace, key)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        from PIL import Image
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 3:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
        ok, bgr = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = capture.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((320, 320))
        image.save(target, "WEBP", quality=76)
        return target
    finally:
        capture.release()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_device_precision(device: str, precision: str) -> tuple[str, str]:
    device_info = _check_device(device, precision)
    if device_info["status"] == "error":
        raise RuntimeError(device_info["message"])
    return str(device_info["effective_device"]), str(device_info["effective_precision"])


def resolve_run_dimensions(payload: dict[str, Any], video_infos: list[dict[str, Any]]) -> tuple[int, int]:
    resolution_mode = str(payload.get("resolution_mode") or "original")
    if resolution_mode == "custom":
        return int(payload["height"]), int(payload["width"])
    first = next((info for info in video_infos if info.get("width") and info.get("height")), None)
    if first is None:
        return int(payload.get("height") or 256), int(payload.get("width") or 448)
    if resolution_mode == "720p":
        height = 720
        width = max(2, round(float(first["width"]) * height / float(first["height"])))
        return height, width
    return int(first["height"]), int(first["width"])


def _dry_run_model_file(
    model_path: Path,
    checkpoint_path: Path | None = None,
    device_name: str = "cpu",
    precision: str = "fp32",
) -> dict[str, Any]:
    if device_name.startswith("multi_"):
        device_name = "cpu"
    device = resolve_torch_device(device_name)
    dtype = torch.float32
    if precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    try:
        model = load_flow_mask_model(
            f"file:{model_path}",
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
            device=str(device),
            metadata={},
        )
    except Exception as exc:
        raise RuntimeError(_describe_dry_run_failure("model_init/checkpoint_load", exc)) from exc

    try:
        img0 = torch.zeros((1, 3, 8, 8), dtype=dtype, device=device)
        img1 = torch.ones((1, 3, 8, 8), dtype=dtype, device=device)
        with torch.no_grad():
            outputs = model.predict(img0, img1, 0.5)
    except Exception as exc:
        raise RuntimeError(_describe_dry_run_failure("predict", exc)) from exc

    try:
        validate_model_outputs(outputs, img0)
    except Exception as exc:
        raise RuntimeError(_describe_dry_run_failure("output_validation", exc)) from exc

    try:
        compose_interpolated(img0, img1, outputs)
    except Exception as exc:
        raise RuntimeError(_describe_dry_run_failure("postprocess", exc)) from exc

    diagnostics = _diagnose_model_outputs(outputs)
    load_report = _extract_load_report(model)
    return {"output_health": diagnostics, "model_load": load_report}


def _diagnose_model_outputs(outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    stats: dict[str, dict[str, float]] = {}
    for name in ("flowt_0", "flowt_1"):
        flow = outputs[name].detach().float()
        stats[name] = {
            "abs_mean": float(flow.abs().mean().item()),
            "abs_max": float(flow.abs().max().item()),
            "nan_count": int(torch.isnan(flow).sum().item()),
        }
    for name in ("mask0", "mask1"):
        raw = outputs[name].detach().float()
        mask = torch.sigmoid(raw)
        stats[name] = {
            "mean": float(mask.mean().item()),
            "std": float(mask.std().item()),
            "nan_count": int(torch.isnan(raw).sum().item()),
        }
    flow_flat = all(stats[name]["abs_max"] < 1e-4 for name in ("flowt_0", "flowt_1"))
    mask_flat = all(stats[name]["std"] < 1e-3 for name in ("mask0", "mask1"))
    has_nan = any(stats[name]["nan_count"] > 0 for name in stats)
    warnings: list[str] = []
    if has_nan:
        warnings.append("模型输出含 NaN，检查数值稳定性 / autocast 精度设置")
    if flow_flat and mask_flat:
        warnings.append(
            "flow ≈ 0 且 mask ≈ constant，通常意味着 checkpoint 未加载或权重初始化仍是 0。"
            "请确认 Model.__init__ 里读取了 checkpoint_path 并调用了 load_state_dict。"
        )
    return {"stats": stats, "warnings": warnings, "flow_flat": flow_flat, "mask_flat": mask_flat, "has_nan": has_nan}


def _extract_load_report(model: Any) -> dict[str, Any] | None:
    for candidate in (model, getattr(model, "_infer", None), getattr(model, "model", None), getattr(model, "net", None), getattr(model, "network", None), getattr(model, "module", None)):
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


def _describe_dry_run_failure(stage: str, exc: Exception) -> str:
    message = f"{stage}: {exc}"
    hint = _dry_run_failure_hint(stage, str(exc))
    if hint:
        return f"{message}. Hint: {hint}"
    return message


def _dry_run_failure_hint(stage: str, detail: str) -> str | None:
    if stage == "model_init/checkpoint_load":
        return (
            "Ensure Model(..., device=...) uses the requested device, checkpoint restore uses "
            "torch.load(..., map_location='cpu'), and the network is moved with model.to(device)."
        )
    if stage in {"predict", "postprocess"} and "Expected all tensors to be on the same device" in detail:
        return (
            "Model infer likely created tensors on a fixed device or dtype. Create tensors and "
            "buffers from img0.device and img0.dtype instead of hard-coded .cuda(), .to('npu'), or .cpu()."
        )
    return None


def _dry_run_device(device_info: dict[str, Any]) -> str:
    if device_info.get("status") != "ok":
        return "cpu"
    devices = device_info.get("effective_devices")
    if isinstance(devices, list) and devices:
        return str(devices[0])
    return str(device_info.get("effective_device") or "cpu")


def _dry_run_devices(device_info: dict[str, Any]) -> list[str]:
    if device_info.get("status") != "ok":
        return []
    devices = device_info.get("effective_devices")
    if isinstance(devices, list) and devices:
        return [str(device) for device in devices]
    return [str(device_info.get("effective_device") or "cpu")]


def _check_device(device: str, precision: str) -> dict[str, Any]:
    requested_device = device or "auto"
    requested_precision = precision or "auto"
    effective_device = normalize_device_name(requested_device)

    if str(effective_device).startswith("cuda") and not torch.cuda.is_available():
        return {"status": "error", "message": "CUDA is not available", "effective_device": effective_device}
    if str(effective_device).startswith("npu"):
        available = [device["id"] for device in list_npu_devices()]
        reason = None if available else npu_unavailable_reason()
        if reason is not None:
            return {"status": "error", "message": f"NPU unavailable: {reason}", "effective_device": effective_device}
        if effective_device not in available:
            return {"status": "error", "message": f"NPU device does not exist: {effective_device}", "effective_device": effective_device}

    kind = device_type_name(effective_device)
    supported = supported_precisions(kind, available=kind == "cpu" or str(effective_device).startswith(("cuda", "npu")))
    effective_precision = requested_precision
    warning = None
    if requested_precision == "auto":
        effective_precision = "fp16" if str(effective_device).startswith(("cuda", "npu")) else "fp32"
    if effective_precision in {"fp16", "bf16"} and not str(effective_device).startswith(("cuda", "npu")):
        warning = f"{effective_device} does not support {effective_precision} autocast; falling back to fp32"
        effective_precision = "fp32"
    if effective_precision == "bf16" and str(effective_device).startswith("cuda"):
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            warning = "Current CUDA device does not support bf16; falling back to fp32"
            effective_precision = "fp32"
    if str(effective_device).startswith(("cuda", "npu")) and effective_precision not in supported:
        unsupported = effective_precision
        effective_precision = "fp32"
        warning = f"{kind.upper()} does not support {unsupported}; falling back to fp32"
    if not warning and str(effective_device).startswith(("cuda", "npu")) and requested_precision not in {"auto", "fp32"} and requested_precision not in supported and effective_precision == "fp32":
        warning = f"{kind.upper()} does not support {requested_precision}; falling back to fp32"
    return {
        "status": "ok",
        "requested_device": requested_device,
        "requested_precision": requested_precision,
        "effective_device": effective_device,
        "effective_precision": effective_precision,
        "supported_precisions": supported,
        "warning": warning,
    }


def _check_multi_cuda(payload: dict[str, Any], precision: str) -> dict[str, Any]:
    return _check_multi_accelerator(payload, precision, "cuda")


def _check_multi_accelerator(payload: dict[str, Any], precision: str, kind: str) -> dict[str, Any]:
    requested = [str(device) for device in (payload.get("devices") or []) if str(device).startswith(f"{kind}:")]
    unavailable_reason = None
    if kind == "cuda":
        available = [f"cuda:{index}" for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        if not available:
            unavailable_reason = "CUDA is not available"
    elif kind == "npu":
        available = [str(device["id"]) for device in list_npu_devices()]
        unavailable_reason = None if available else npu_unavailable_reason()
    else:
        raise ValueError("kind must be cuda or npu")
    mode = f"multi_{kind}"
    if unavailable_reason is not None:
        return {"status": "error", "message": f"{kind.upper()} unavailable: {unavailable_reason}", "effective_device": mode}
    if not available:
        return {"status": "error", "message": f"{kind.upper()} is not available; cannot enable {mode}", "effective_device": mode}
    devices = requested or available
    invalid = [device for device in devices if device not in available]
    if invalid:
        return {"status": "error", "message": f"{kind.upper()} device does not exist: {', '.join(invalid)}", "effective_device": mode}
    requested_precision = precision or "auto"
    effective_precision = requested_precision
    warning = None
    supported = supported_precisions(kind, available=bool(available))
    if requested_precision == "auto":
        effective_precision = "fp16"
    if kind == "cuda" and effective_precision == "bf16" and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        warning = "Current CUDA device does not support bf16; falling back to fp32"
        effective_precision = "fp32"
    if effective_precision not in supported:
        unsupported = effective_precision
        effective_precision = "fp32"
        warning = f"{kind.upper()} does not support {unsupported}; falling back to fp32"
    if warning and effective_precision == "fp32" and requested_precision != "fp32":
        unsupported_precision = requested_precision if requested_precision != "auto" else "fp16"
        warning = f"{kind.upper()} does not support {unsupported_precision}; falling back to fp32"
    return {
        "status": "ok",
        "requested_device": mode,
        "requested_precision": requested_precision,
        "effective_device": mode,
        "effective_devices": devices,
        "effective_precision": effective_precision,
        "supported_precisions": supported,
        "warning": warning,
    }


def _resolve_preflight_resolution(payload: dict[str, Any], video_infos: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        height, width = resolve_run_dimensions(payload, video_infos)
        return {
            "mode": str(payload.get("resolution_mode") or "original"),
            "height": height,
            "width": width,
        }
    except Exception as exc:
        return {"mode": str(payload.get("resolution_mode") or "original"), "error": str(exc)}


def _video_summary(workspace: WorkspaceConfig, path: Path) -> dict[str, Any]:
    return video_summary(workspace, path, exact=False)


def video_summary(
    workspace: WorkspaceConfig,
    path: Path,
    frame_step: int = 1,
    max_frames: int | None = None,
    exact: bool = False,
) -> dict[str, Any]:
    info = inspect_video(path, workspace, exact=exact)
    cache_key = decode_cache_key(path, "video_gt_triplets", frame_step, max_frames)
    manifest = _read_decode_manifest(workspace, cache_key)
    if manifest:
        info = _merge_manifest_info(info, manifest)
    thumb_key = video_thumbnail_key(path)
    has_thumb = ensure_video_thumbnail(workspace, path, thumb_key) is not None
    return {
        "name": info["name"],
        "frame_count": info["frame_count"],
        "container_frame_count": info.get("container_frame_count", 0),
        "frame_count_source": info.get("frame_count_source", "exact"),
        "frame_count_warning": info.get("frame_count_warning"),
        "duration_seconds": info.get("duration_seconds", 0.0),
        "fps": info["fps"],
        "width": info["width"],
        "height": info["height"],
        "decodable": info["decodable"],
        "error": info["error"],
        "valid_triplets": _valid_triplets(info["frame_count"], frame_step, max_frames),
        "cache_status": decode_cache_status(workspace, cache_key),
        "metadata_source": info.get("metadata_source"),
        "codec": info.get("codec"),
        "pix_fmt": info.get("pix_fmt"),
        "thumbnail_url": f"/api/video-thumbnails/{thumb_key}" if has_thumb else None,
    }


def _iter_videos(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)


def _sort_video_summaries(videos: list[dict[str, Any]], sort: str) -> list[dict[str, Any]]:
    reverse = sort.startswith("-")
    key = sort[1:] if reverse else sort
    supported = {
        "name": lambda item: str(item.get("name") or ""),
        "duration": lambda item: float(item.get("duration_seconds") or 0),
        "frame_count": lambda item: int(item.get("frame_count") or 0),
        "resolution": lambda item: int(item.get("width") or 0) * int(item.get("height") or 0),
        "triplets": lambda item: int(item.get("valid_triplets") or 0),
    }
    return sorted(videos, key=supported.get(key, supported["name"]), reverse=reverse)


def _read_decode_manifest(workspace: WorkspaceConfig, cache_key: str) -> dict[str, Any] | None:
    manifest_path = decode_cache_dir(workspace, cache_key) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _merge_manifest_info(info: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    merged = dict(info)
    frame_count = int(manifest.get("frame_count") or merged.get("frame_count") or 0)
    fps = float(manifest.get("fps") or merged.get("fps") or 24.0)
    merged.update(
        {
            "frame_count": frame_count,
            "container_frame_count": int(merged.get("container_frame_count") or frame_count),
            "frame_count_source": "manifest_exact",
            "frame_count_warning": None,
            "duration_seconds": float(manifest.get("duration_seconds") or (frame_count / fps if fps else 0.0)),
            "fps": fps,
            "width": int(manifest.get("width") or merged.get("width") or 0),
            "height": int(manifest.get("height") or merged.get("height") or 0),
            "decodable": manifest.get("decode_status") == "completed" or frame_count > 0,
            "error": None if frame_count > 0 else merged.get("error"),
            "metadata_source": "manifest_exact",
        }
    )
    return merged


def _exact_frame_count(capture: Any) -> int:
    counted = 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        counted += 1
    return counted


def _video_inspect_cache_path(path: Path, workspace: WorkspaceConfig | None, exact: bool) -> Path | None:
    if workspace is None:
        return None
    stat = path.stat()
    data = {
        "path": str(path.resolve()).replace("\\", "/"),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "version": VIDEO_INSPECT_VERSION,
        "mode": "exact" if exact else "fast",
    }
    key = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
    return workspace.root / "video_meta_cache" / f"{key}.json"


def _read_video_inspect_cache(path: Path, workspace: WorkspaceConfig | None, exact: bool) -> dict[str, Any] | None:
    cache_path = _video_inspect_cache_path(path, workspace, exact)
    if cache_path is None or not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_video_inspect_cache(path: Path, workspace: WorkspaceConfig | None, info: dict[str, Any], exact: bool) -> None:
    cache_path = _video_inspect_cache_path(path, workspace, exact)
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def _ensure_child(path: Path, root: Path, message: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(message) from exc


def _optional_positive_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _valid_triplets(frame_count: int, frame_step: int, max_frames: int | None) -> int:
    effective = min(int(frame_count or 0), max_frames or int(frame_count or 0))
    return max(0, effective - 2 * max(1, int(frame_step or 1)))


def _parse_rate(value: Any) -> float | None:
    if not value or value == "0/0":
        return None
    text = str(value)
    if "/" in text:
        num, den = text.split("/", 1)
        denominator = float(den)
        if denominator == 0:
            return None
        return float(num) / denominator
    parsed = float(text)
    return parsed if parsed > 0 else None


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    parsed = float(value)
    return parsed if parsed > 0 else None


def _error(title: str, exc: Exception) -> dict[str, str]:
    return {"title": title, "message": str(exc), "type": type(exc).__name__}
