from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.file_inputs import VIDEO_SUFFIXES, decode_cache_dir, decode_cache_key


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def scan_dataset(db: Database, workspace: WorkspaceConfig, dataset_id: int) -> int:
    dataset = db.get_dataset(dataset_id)
    source_type = dataset.get("source_type") or "frames"
    db.clear_samples(dataset_id)
    if source_type == "frames":
        count = scan_triplet_dataset(db, dataset_id)
        db.update_dataset_scan_info(dataset_id, None, video_count=0, frame_count=count)
        return count
    if source_type == "video":
        return scan_video_dataset(db, workspace, dataset_id)
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


def scan_video_dataset(db: Database, workspace: WorkspaceConfig, dataset_id: int) -> int:
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

    for video_index, video_path in enumerate(videos):
        cache_key = decode_cache_key(video_path, decode_mode, frame_step, max_frames)
        frames, fps, timestamps = _decode_video_cached(
            workspace,
            video_path,
            cache_key,
            max_frames,
            decode_mode,
            frame_step,
        )
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


def _decode_video_cached(
    workspace: WorkspaceConfig,
    video_path: Path,
    cache_key: str,
    max_frames: int | None,
    decode_mode: str,
    frame_step: int,
) -> tuple[list[Path], float, list[float]]:
    output_dir = decode_cache_dir(workspace, cache_key)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = [Path(path) for path in manifest.get("frames", []) if Path(path).exists()]
        timestamps = [float(value) for value in manifest.get("timestamps", [])]
        if frames and len(timestamps) == len(frames):
            return frames, float(manifest.get("fps") or 24.0), timestamps

    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    partial_dir.mkdir(parents=True, exist_ok=True)
    try:
        frames, fps, timestamps = _decode_video(video_path, partial_dir, max_frames)
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
            "color": "RGB",
        }
        (partial_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        partial_dir.rename(output_dir)
        frames = [output_dir / path.name for path in frames]
        return frames, fps, timestamps
    except Exception:
        if partial_dir.exists():
            shutil.rmtree(partial_dir)
        raise


def _decode_video(video_path: Path, output_dir: Path, max_frames: int | None) -> tuple[list[Path], float, list[float]]:
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
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"video has no decodable frames: {video_path}")
    return frames, fps, timestamps


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
