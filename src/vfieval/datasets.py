from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from PIL import Image

from vfieval.compare_inputs import (
    compare_video_name,
    image_size,
    list_frame_images,
    resolve_compare_source_path,
    validate_strict_decoded_alignment,
)
from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.file_inputs import VIDEO_SUFFIXES, decode_cache_dir, decode_cache_key


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ProgressCallback = Callable[[dict[str, Any]], None]


def scan_dataset(
    db: Database,
    workspace: WorkspaceConfig,
    dataset_id: int,
    progress_callback: ProgressCallback | None = None,
    decode_backend: str = "auto",
) -> int:
    dataset = db.get_dataset(dataset_id)
    source_type = dataset.get("source_type") or "frames"
    db.clear_samples(dataset_id)
    if source_type == "frames":
        count = scan_triplet_dataset(db, dataset_id)
        db.update_dataset_scan_info(dataset_id, None, video_count=0, frame_count=count)
        return count
    if source_type == "video":
        return scan_video_dataset(db, workspace, dataset_id, progress_callback=progress_callback, decode_backend=decode_backend)
    if source_type == "compare":
        return scan_compare_dataset(db, workspace, dataset_id)
    raise ValueError(f"unsupported dataset source_type: {source_type}")


def scan_triplet_dataset(db: Database, dataset_id: int) -> int:
    dataset = db.get_dataset(dataset_id)
    root = Path(dataset["root_path"])
    img0_dir = root / "img0"
    img1_dir = root / "img1"
    gt_dir = root / "gt"
    if not img0_dir.is_dir() or not img1_dir.is_dir():
        raise FileNotFoundError("dataset scan expects img0/ and img1/ folders")

    added = 0
    img1_by_name = {_key(path): path for path in _iter_images(img1_dir)}
    gt_by_name = {_key(path): path for path in _iter_images(gt_dir)} if gt_dir.is_dir() else {}

    for img0_path in _iter_images(img0_dir):
        key = _key(img0_path)
        img1_path = img1_by_name.get(key)
        if img1_path is None:
            continue
        gt_path = gt_by_name.get(key)
        if dataset["has_gt"] and gt_path is None:
            continue
        db.add_sample(
            dataset_id=dataset_id,
            name=key,
            img0_path=str(img0_path),
            img1_path=str(img1_path),
            gt_path=str(gt_path) if gt_path else None,
        )
        added += 1
    return added


def scan_video_dataset(
    db: Database,
    workspace: WorkspaceConfig,
    dataset_id: int,
    progress_callback: ProgressCallback | None = None,
    decode_backend: str = "auto",
) -> int:
    dataset = db.get_dataset(dataset_id)
    metadata = dataset.get("metadata") or {}
    decode_mode = dataset.get("decode_mode") or ("video_gt_triplets" if dataset["has_gt"] else "video_pairs")
    frame_step = max(1, int(metadata.get("frame_step") or 1))
    max_frames = _optional_positive_int(metadata.get("max_frames"))
    video_glob = str(metadata.get("video_glob") or "*")
    selected_videos = metadata.get("selected_videos")

    videos = _find_videos(Path(dataset["root_path"]), video_glob, selected_videos)
    if not videos:
        raise FileNotFoundError(f"no videos found under {dataset['root_path']}")

    decoded_root = workspace.root / "decode_cache"
    decoded_root.mkdir(parents=True, exist_ok=True)
    added = 0
    decoded_frames = 0
    decode_results: list[tuple[str, list[Path], float, list[float], dict[str, Any]] | None] = [None for _ in videos]
    progress_counts = [0 for _ in videos]
    cache_hits = 0
    cache_misses = 0
    lock = Lock()

    def decode_one(video_index: int, video_path: Path) -> tuple[str, list[Path], float, list[float], dict[str, Any]]:
        cache_key = decode_cache_key(video_path, decode_mode, frame_step, max_frames)
        local_cache_hit = False

        def on_decode_progress(event: dict[str, Any]) -> None:
            nonlocal cache_hits, cache_misses, local_cache_hit
            if progress_callback is None:
                return
            frames_done = int(event.get("frames") or 0)
            with lock:
                progress_counts[video_index] = frames_done
                if event.get("event") == "cache_hit" and not local_cache_hit:
                    local_cache_hit = True
                    cache_hits += 1
                elif event.get("event") == "cache_miss" and not event.get("_cache_miss_counted"):
                    event["_cache_miss_counted"] = True
                    cache_misses += 1
                payload = {
                    **event,
                    "video_index": video_index,
                    "video_count": len(videos),
                    "video_name": video_path.name,
                    "decoded_frames": sum(progress_counts),
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                }
            progress_callback(payload)

        frames, fps, timestamps, decode_info = _decode_video_cached(
            workspace,
            video_path,
            cache_key,
            max_frames,
            decode_mode,
            frame_step,
            progress_callback=on_decode_progress,
            decode_backend=decode_backend,
        )
        return cache_key, frames, fps, timestamps, decode_info

    worker_count = _decode_worker_count(len(videos), metadata)
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(decode_one, index, path): index for index, path in enumerate(videos)}
            for future in as_completed(futures):
                decode_results[futures[future]] = future.result()
    else:
        for video_index, video_path in enumerate(videos):
            decode_results[video_index] = decode_one(video_index, video_path)

    for video_index, video_path in enumerate(videos):
        result = decode_results[video_index]
        if result is None:
            raise RuntimeError(f"video decode did not produce a result: {video_path.name}")
        cache_key, frames, fps, timestamps, decode_info = result
        decoded_frames += len(frames)
        if decode_mode == "video_gt_triplets":
            added += _add_video_triplets(
                db,
                dataset_id,
                video_path,
                video_index,
                frames,
                frame_step,
                fps,
                timestamps,
                cache_key,
            )
        elif decode_mode == "video_pairs":
            added += _add_video_pairs(
                db,
                dataset_id,
                video_path,
                video_index,
                frames,
                frame_step,
                fps,
                timestamps,
                cache_key,
            )
        else:
            raise ValueError(f"unsupported video decode_mode: {decode_mode}")

    db.update_dataset_scan_info(
        dataset_id,
        str(decoded_root),
        video_count=len(videos),
        frame_count=decoded_frames,
        metadata={
            "frame_step": frame_step,
            "max_frames": max_frames,
            "video_glob": video_glob,
            "selected_videos": [path.name for path in videos],
            "decode_mode": decode_mode,
            "decode_backend": decode_backend,
        },
    )
    return added


def scan_compare_dataset(db: Database, workspace: WorkspaceConfig, dataset_id: int) -> int:
    dataset = db.get_dataset(dataset_id)
    metadata = dataset.get("metadata") or {}
    if metadata.get("compare_tracks"):
        return _scan_multitrack_compare_dataset(db, workspace, dataset_id, metadata)

    reference_path = resolve_compare_source_path(workspace, str(metadata.get("reference_path") or ""))
    distorted_path = resolve_compare_source_path(workspace, str(metadata.get("distorted_path") or ""))
    align_mode = str(metadata.get("align_mode") or "strict")
    if align_mode != "strict":
        raise ValueError("compare datasets currently only support strict alignment")

    reference_frames, reference_fps, reference_timestamps = _load_compare_source_frames(workspace, reference_path, "compare_reference")
    distorted_frames, distorted_fps, distorted_timestamps = _load_compare_source_frames(workspace, distorted_path, "compare_distorted")
    validate_strict_decoded_alignment(
        reference_frames=reference_frames,
        distorted_frames=distorted_frames,
        reference_fps=reference_fps,
        distorted_fps=distorted_fps,
        reference_timestamps=reference_timestamps,
        distorted_timestamps=distorted_timestamps,
    )

    compare_name = compare_video_name(reference_path, distorted_path)
    added = 0
    for frame_index, (reference_frame, distorted_frame) in enumerate(zip(reference_frames, distorted_frames)):
        reference_size = image_size(reference_frame)
        distorted_size = image_size(distorted_frame)
        if reference_size != distorted_size:
            raise ValueError(
                "strict compare requires matching frame dimensions: "
                f"{reference_frame.name}={reference_size[0]}x{reference_size[1]} vs "
                f"{distorted_frame.name}={distorted_size[0]}x{distorted_size[1]}"
            )
        name = f"compare_{compare_name}_{frame_index:06d}"
        db.add_sample(
            dataset_id=dataset_id,
            name=name,
            img0_path=str(reference_frame.resolve()),
            img1_path=str(distorted_frame.resolve()),
            gt_path=str(reference_frame.resolve()),
            metadata={
                "source_type": "compare",
                "compare_group": compare_name,
                "video_name": compare_name,
                "video_file": distorted_path.name,
                "reference_path": str(reference_path),
                "distorted_path": str(distorted_path),
                "frame_index": frame_index,
                "sample_index": frame_index,
                "fps": reference_fps or distorted_fps or 24.0,
                "timestamps": {
                    "gt": reference_timestamps[frame_index] if frame_index < len(reference_timestamps) else None,
                    "pred": distorted_timestamps[frame_index] if frame_index < len(distorted_timestamps) else None,
                },
            },
        )
        added += 1

    db.update_dataset_scan_info(
        dataset_id,
        None,
        video_count=1 if added else 0,
        frame_count=added,
        metadata={
            "reference_path": str(reference_path),
            "distorted_path": str(distorted_path),
            "align_mode": align_mode,
            "video_name": compare_name,
            "reference_frame_count": len(reference_frames),
            "distorted_frame_count": len(distorted_frames),
        },
    )
    return added


def _scan_multitrack_compare_dataset(
    db: Database,
    workspace: WorkspaceConfig,
    dataset_id: int,
    metadata: dict[str, Any],
) -> int:
    reference_path = resolve_compare_source_path(workspace, str(metadata.get("reference_path") or ""))
    align_mode = str(metadata.get("align_mode") or "strict")
    if align_mode != "strict":
        raise ValueError("compare datasets currently only support strict alignment")
    tracks = list(metadata.get("compare_tracks") or [])
    if not tracks:
        raise ValueError("multi-track compare dataset requires at least one track")

    reference_frames, reference_fps, reference_timestamps = _load_compare_source_frames(workspace, reference_path, "compare_reference")
    video_name = str(metadata.get("video_name") or reference_path.stem)
    video_token = _sample_token(video_name)
    added = 0
    scanned_tracks: list[dict[str, Any]] = []

    for track_index, track in enumerate(tracks):
        distorted_path = resolve_compare_source_path(workspace, str(track.get("distorted_path") or ""))
        track_label = str(track.get("track_label") or track.get("label") or f"pred{track_index + 1}")
        track_token = _sample_token(track_label)
        distorted_frames, distorted_fps, distorted_timestamps = _load_compare_source_frames(
            workspace,
            distorted_path,
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
        scanned_tracks.append(
            {
                "track_label": track_label,
                "track_key": track_token,
                "track_run_id": track.get("track_run_id"),
                "artifact_id": track.get("artifact_id"),
                "distorted_path": str(distorted_path),
                "frame_count": len(distorted_frames),
            }
        )
        for frame_index, (reference_frame, distorted_frame) in enumerate(zip(reference_frames, distorted_frames)):
            reference_size = image_size(reference_frame)
            distorted_size = image_size(distorted_frame)
            if reference_size != distorted_size:
                raise ValueError(
                    f"track {track_label}: strict compare requires matching frame dimensions: "
                    f"{reference_frame.name}={reference_size[0]}x{reference_size[1]} vs "
                    f"{distorted_frame.name}={distorted_size[0]}x{distorted_size[1]}"
                )
            sample_index = track_index * len(reference_frames) + frame_index
            db.add_sample(
                dataset_id=dataset_id,
                name=f"{video_token}__{track_token}__{frame_index:06d}",
                img0_path=str(reference_frame.resolve()),
                img1_path=str(distorted_frame.resolve()),
                gt_path=str(reference_frame.resolve()),
                metadata={
                    "source_type": "compare",
                    "compare_group": video_name,
                    "video_name": video_name,
                    "video_file": metadata.get("video_file") or reference_path.name,
                    "reference_path": str(reference_path),
                    "distorted_path": str(distorted_path),
                    "frame_index": frame_index,
                    "sample_index": sample_index,
                    "fps": reference_fps or distorted_fps or 24.0,
                    "compare_track_label": track_label,
                    "compare_track_key": track_token,
                    "compare_track_index": track_index,
                    "compare_track_run_id": track.get("track_run_id"),
                    "compare_track_artifact_id": track.get("artifact_id"),
                    "timestamps": {
                        "gt": reference_timestamps[frame_index] if frame_index < len(reference_timestamps) else None,
                        "pred": distorted_timestamps[frame_index] if frame_index < len(distorted_timestamps) else None,
                    },
                },
            )
            added += 1

    db.update_dataset_scan_info(
        dataset_id,
        None,
        video_count=1 if added else 0,
        frame_count=added,
        metadata={
            "reference_path": str(reference_path),
            "align_mode": align_mode,
            "video_name": video_name,
            "reference_frame_count": len(reference_frames),
            "track_count": len(scanned_tracks),
            "compare_tracks": scanned_tracks,
        },
    )
    return added


def _add_video_triplets(
    db: Database,
    dataset_id: int,
    video_path: Path,
    video_index: int,
    frames: list[Path],
    frame_step: int,
    fps: float,
    timestamps: list[float],
    cache_key: str,
) -> int:
    added = 0
    if len(frames) < 2 * frame_step + 1:
        return 0
    for sample_index, img0_index in enumerate(range(0, len(frames) - 2 * frame_step)):
        gt_index = img0_index + frame_step
        img1_index = img0_index + 2 * frame_step
        name = f"{video_index:03d}_{video_path.stem}_{sample_index:06d}"
        db.add_sample(
            dataset_id=dataset_id,
            name=name,
            img0_path=str(frames[img0_index]),
            img1_path=str(frames[img1_index]),
            gt_path=str(frames[gt_index]),
            metadata=_video_sample_metadata(
                video_path,
                "video_gt_triplets",
                video_index,
                sample_index,
                img0_index,
                img1_index,
                gt_index,
                fps,
                timestamps,
                cache_key,
            ),
        )
        added += 1
    return added


def _add_video_pairs(
    db: Database,
    dataset_id: int,
    video_path: Path,
    video_index: int,
    frames: list[Path],
    frame_step: int,
    fps: float,
    timestamps: list[float],
    cache_key: str,
) -> int:
    added = 0
    if len(frames) < frame_step + 1:
        return 0
    for sample_index, img0_index in enumerate(range(0, len(frames) - frame_step)):
        img1_index = img0_index + frame_step
        name = f"{video_index:03d}_{video_path.stem}_{sample_index:06d}"
        db.add_sample(
            dataset_id=dataset_id,
            name=name,
            img0_path=str(frames[img0_index]),
            img1_path=str(frames[img1_index]),
            gt_path=None,
            metadata=_video_sample_metadata(
                video_path,
                "video_pairs",
                video_index,
                sample_index,
                img0_index,
                img1_index,
                None,
                fps,
                timestamps,
                cache_key,
            ),
        )
        added += 1
    return added


def _find_videos(root: Path, video_glob: str, selected_videos: Any = None) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in VIDEO_SUFFIXES else []
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    videos = sorted(
        path
        for path in root.glob(video_glob)
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if selected_videos is None:
        return videos
    selected_names = [str(item) for item in selected_videos]
    if not selected_names:
        raise ValueError("selected_videos must contain at least one video")
    by_name = {path.name: path for path in videos}
    resolved = []
    for name in selected_names:
        if Path(name).name != name:
            raise ValueError(f"selected video must be a file name: {name}")
        path = by_name.get(name)
        if path is None:
            raise FileNotFoundError(f"selected video not found: {name}")
        resolved.append(path)
    return resolved


def _load_compare_source_frames(
    workspace: WorkspaceConfig,
    source_path: Path,
    cache_prefix: str,
) -> tuple[list[Path], float | None, list[float | None]]:
    if source_path.is_file() and source_path.suffix.lower() in VIDEO_SUFFIXES:
        cache_key = decode_cache_key(source_path, cache_prefix, 1, None)
        frames, fps, timestamps, _decode_info = _decode_video_cached(
            workspace,
            source_path,
            cache_key,
            None,
            cache_prefix,
            1,
        )
        return frames, fps, list(timestamps)
    if source_path.is_dir():
        frames = list_frame_images(source_path)
        if not frames:
            raise FileNotFoundError(f"frame directory has no supported images: {source_path}")
        return frames, None, [None for _ in frames]
    raise ValueError(f"unsupported compare source: {source_path}")


def _decode_video_cached(
    workspace: WorkspaceConfig,
    video_path: Path,
    cache_key: str,
    max_frames: int | None,
    decode_mode: str,
    frame_step: int,
    progress_callback: ProgressCallback | None = None,
    decode_backend: str = "auto",
) -> tuple[list[Path], float, list[float], dict[str, Any]]:
    output_dir = decode_cache_dir(workspace, cache_key)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = [Path(path) for path in manifest.get("frames", []) if Path(path).exists()]
        timestamps = [float(value) for value in manifest.get("timestamps", [])]
        if frames and len(timestamps) == len(frames):
            decode_info = {
                "backend": "cache",
                "manifest_backend": manifest.get("decode_backend"),
                "fallback_reason": manifest.get("decode_fallback_reason"),
                "cache_hit": True,
            }
            if progress_callback:
                progress_callback(
                    {
                        "event": "cache_hit",
                        "backend": "cache",
                        "manifest_backend": manifest.get("decode_backend"),
                        "frames": len(frames),
                    }
                )
            return frames, float(manifest.get("fps") or 24.0), timestamps, decode_info

    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    partial_dir.mkdir(parents=True, exist_ok=True)
    try:
        if progress_callback:
            progress_callback({"event": "cache_miss", "backend": decode_backend, "frames": 0})
        frames, fps, timestamps, decode_info = _decode_video(
            video_path,
            partial_dir,
            max_frames,
            decode_backend=decode_backend,
            progress_callback=progress_callback,
        )
        width, height = _frame_size(frames[0]) if frames else (0, 0)
        manifest = {
            "video_name": video_path.stem,
            "video_file": video_path.name,
            "video_path": str(video_path.resolve()),
            "cache_key": cache_key,
            "fps": fps,
            "frames": [str(path.resolve()) for path in frames],
            "timestamps": timestamps,
            "frame_count": len(frames),
            "width": width,
            "height": height,
            "duration_seconds": float(timestamps[-1]) if timestamps else (len(frames) / fps if fps > 0 else 0.0),
            "valid_triplets": max(0, len(frames) - 2 * frame_step) if decode_mode == "video_gt_triplets" else max(0, len(frames) - frame_step),
            "decode_status": "completed",
            "decode_mode": decode_mode,
            "frame_step": frame_step,
            "max_frames": max_frames,
            "decode_backend": decode_info.get("backend"),
            "decode_fallback_reason": decode_info.get("fallback_reason"),
            "color": "RGB",
        }
        (partial_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        partial_dir.rename(output_dir)
        frames = [output_dir / path.name for path in frames]
        return frames, fps, timestamps, decode_info
    except Exception:
        if partial_dir.exists():
            shutil.rmtree(partial_dir)
        raise


def _decode_video(
    video_path: Path,
    output_dir: Path,
    max_frames: int | None,
    decode_backend: str = "auto",
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[Path], float, list[float], dict[str, Any]]:
    requested = str(decode_backend or "auto")
    fallback_reason = None
    if requested in {"auto", "ffmpeg"}:
        try:
            frames, fps, timestamps = _decode_video_ffmpeg(video_path, output_dir, max_frames, progress_callback)
            return frames, fps, timestamps, {"backend": "ffmpeg", "fallback_reason": None}
        except Exception as exc:
            if requested == "ffmpeg":
                raise
            fallback_reason = f"ffmpeg unavailable or failed: {exc}"
            _clear_decoded_frames(output_dir)
            if progress_callback:
                progress_callback({"event": "fallback", "backend": "opencv", "fallback_reason": fallback_reason, "frames": 0})

    frames, fps, timestamps = _decode_video_opencv(video_path, output_dir, max_frames, progress_callback)
    return frames, fps, timestamps, {"backend": "opencv", "fallback_reason": fallback_reason}


def _decode_video_opencv(
    video_path: Path,
    output_dir: Path,
    max_frames: int | None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[Path], float, list[float]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("video dataset scanning requires opencv-python (cv2)") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
    frames: list[Path] = []
    timestamps: list[float] = []
    try:
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break
            ok, bgr = capture.read()
            if not ok:
                break
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_path = output_dir / f"{len(frames):06d}.png"
            Image.fromarray(rgb).save(frame_path)
            frames.append(frame_path)
            timestamps.append(timestamp_ms / 1000.0)
            if progress_callback and (len(frames) == 1 or len(frames) % 10 == 0):
                progress_callback({"event": "frame", "backend": "opencv", "frames": len(frames)})
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"video has no decodable frames: {video_path}")
    if progress_callback:
        progress_callback({"event": "video_done", "backend": "opencv", "frames": len(frames)})
    return frames, fps, timestamps


def _decode_video_ffmpeg(
    video_path: Path,
    output_dir: Path,
    max_frames: int | None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[Path], float, list[float]]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not on PATH")
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "%06d.png"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vsync",
        "0",
    ]
    if max_frames is not None:
        command.extend(["-frames:v", str(int(max_frames))])
    command.append(str(pattern))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    last_count = -1
    while process.poll() is None:
        count = len(list(output_dir.glob("*.png")))
        if progress_callback and count != last_count:
            progress_callback({"event": "frame", "backend": "ffmpeg", "frames": count})
            last_count = count
        time.sleep(0.25)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError((stderr or stdout or "ffmpeg decode failed").strip())
    frames = sorted(output_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames: {video_path}")
    fps = _probe_video_fps(video_path)
    timestamps = [index / fps for index in range(len(frames))]
    if progress_callback:
        progress_callback({"event": "video_done", "backend": "ffmpeg", "frames": len(frames)})
    return frames, fps, timestamps


def _decode_worker_count(video_count: int, metadata: dict[str, Any]) -> int:
    requested = metadata.get("decode_workers")
    if requested in {None, ""}:
        return max(1, min(4, int(video_count or 1)))
    return max(1, min(int(requested), int(video_count or 1)))


def _clear_decoded_frames(output_dir: Path) -> None:
    for path in output_dir.glob("*.png"):
        try:
            path.unlink()
        except OSError:
            pass


def _probe_video_fps(video_path: Path) -> float:
    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps > 0:
                return fps
        finally:
            capture.release()
    except Exception:
        pass
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            completed = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=avg_frame_rate,r_frame_rate",
                    "-of",
                    "json",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            payload = json.loads(completed.stdout or "{}")
            stream = (payload.get("streams") or [{}])[0]
            return _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate")) or 24.0
        except Exception:
            pass
    return 24.0


def _parse_rate(value: Any) -> float | None:
    text = str(value or "")
    if not text or text == "0/0":
        return None
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator or 0)
        if denominator_value == 0:
            return None
        return float(numerator) / denominator_value
    parsed = float(text)
    return parsed if parsed > 0 else None


def _frame_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _video_sample_metadata(
    video_path: Path,
    decode_mode: str,
    video_index: int,
    sample_index: int,
    img0_index: int,
    img1_index: int,
    gt_index: int | None,
    fps: float,
    timestamps: list[float],
    cache_key: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_type": "video",
        "decode_mode": decode_mode,
        "video_index": video_index,
        "video_name": video_path.stem,
        "video_file": video_path.name,
        "video_path": str(video_path.resolve()),
        "sample_index": sample_index,
        "img0_index": img0_index,
        "img1_index": img1_index,
        "fps": fps,
        "cache_key": cache_key,
        "timestamps": {
            "img0": timestamps[img0_index] if img0_index < len(timestamps) else None,
            "img1": timestamps[img1_index] if img1_index < len(timestamps) else None,
        },
    }
    if gt_index is not None:
        metadata["gt_index"] = gt_index
        metadata["frame_index"] = gt_index
        metadata["timestamps"]["gt"] = timestamps[gt_index] if gt_index < len(timestamps) else None
    else:
        metadata["frame_index"] = img0_index
    return metadata


def _sample_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return token.strip("._") or "track"


def _optional_positive_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _iter_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _key(path: Path) -> str:
    return path.stem
