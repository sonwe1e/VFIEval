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
from vfieval.models import load_flow_mask_model
from vfieval.pipeline.postprocess import validate_model_outputs


VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
DECODE_STRATEGY_VERSION = "opencv-rgb-v1"
VIDEO_INSPECT_VERSION = "ffprobe-opencv-v3"


def project_root(workspace: WorkspaceConfig) -> Path:
    return Path(os.getenv("VFIEVAL_PROJECT_ROOT") or workspace.root.parent).resolve()


def models_dir(workspace: WorkspaceConfig) -> Path:
    return Path(os.getenv("VFIEVAL_MODELS_DIR") or project_root(workspace) / "models").resolve()


def videos_dir(workspace: WorkspaceConfig) -> Path:
    return Path(os.getenv("VFIEVAL_VIDEOS_DIR") or project_root(workspace) / "videos").resolve()


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
    model_file = str(payload.get("model_file") or "")
    video_group = str(payload.get("video_group") or "")
    frame_step = max(1, int(payload.get("frame_step") or 1))
    max_frames = _optional_positive_int(payload.get("max_frames"))
    device_request = str(payload.get("device") or "auto")
    precision_request = str(payload.get("precision") or "auto")

    result: dict[str, Any] = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "model": {},
        "video_group": {},
        "device": {},
        "resolution": {},
        "cache": {},
        "output_dir": str(workspace.runs_dir / str(db.next_run_id())),
    }

    model_path: Path | None = None
    try:
        model_path = resolve_model_file(workspace, model_file)
        _dry_run_model_file(model_path)
        result["model"] = {"name": model_path.name, "path": str(model_path), "interface_ok": True}
    except Exception as exc:
        result["ok"] = False
        result["model"] = {"name": model_file, "interface_ok": False}
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

    device_info = _check_device(device_request, precision_request)
    result["device"] = device_info
    if device_info["status"] == "error":
        result["ok"] = False
        result["errors"].append({"title": "设备检查失败", "message": device_info["message"], "type": "DeviceError"})
    if device_info.get("warning"):
        result["warnings"].append(device_info["warning"])

    result["resolution"] = _resolve_preflight_resolution(payload, video_infos)
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


def _dry_run_model_file(model_path: Path) -> None:
    model = load_flow_mask_model(f"file:{model_path}", device="cpu", metadata={})
    img0 = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    img1 = torch.ones((1, 3, 8, 8), dtype=torch.float32)
    with torch.no_grad():
        outputs = model.predict(img0, img1, 0.5)
    validate_model_outputs(outputs, img0)


def _check_device(device: str, precision: str) -> dict[str, Any]:
    requested_device = device or "auto"
    requested_precision = precision or "auto"
    effective_device = requested_device
    if requested_device == "auto":
        effective_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda":
        effective_device = "cuda:0"
    if requested_device == "npu":
        effective_device = "npu:0"

    if str(effective_device).startswith("cuda") and not torch.cuda.is_available():
        return {"status": "error", "message": "CUDA 不可用", "effective_device": effective_device}
    if str(effective_device).startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return {"status": "error", "message": "NPU 不可用：未安装 torch_npu", "effective_device": effective_device}

    effective_precision = requested_precision
    warning = None
    if requested_precision == "auto":
        effective_precision = "fp16" if str(effective_device).startswith("cuda") else "fp32"
    if effective_precision in {"fp16", "bf16"} and not str(effective_device).startswith("cuda"):
        warning = f"{effective_device} 不支持 {effective_precision} autocast，已回退 fp32"
        effective_precision = "fp32"
    if effective_precision == "bf16" and str(effective_device).startswith("cuda"):
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            warning = "当前 CUDA 设备不支持 bf16，已回退 fp32"
            effective_precision = "fp32"
    return {
        "status": "ok",
        "requested_device": requested_device,
        "requested_precision": requested_precision,
        "effective_device": effective_device,
        "effective_precision": effective_precision,
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
