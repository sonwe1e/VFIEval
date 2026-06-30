from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from vfieval.config import WorkspaceConfig
from vfieval.file_inputs import VIDEO_SUFFIXES, inspect_video, project_root


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TIMESTAMP_ALIGNMENT_TOLERANCE_SECONDS = 1e-3


def resolve_compare_source_path(workspace: WorkspaceConfig, source: str) -> Path:
    text = str(source or "").strip()
    if not text:
        raise ValueError("compare source path is required")
    path = Path(text)
    if not path.is_absolute():
        path = (project_root(workspace) / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"compare source not found: {path}")
    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        return path
    if path.is_dir():
        return path
    raise ValueError(f"unsupported compare source: {path}")


def inspect_compare_source(workspace: WorkspaceConfig, source: str) -> dict[str, Any]:
    path = resolve_compare_source_path(workspace, source)
    if path.is_file():
        info = inspect_video(path, workspace, exact=True)
        if not info.get("decodable"):
            raise RuntimeError(info.get("error") or f"video is not decodable: {path}")
        return {
            "path": str(path),
            "name": path.name,
            "source_kind": "video",
            "frame_count": int(info.get("frame_count") or 0),
            "width": int(info.get("width") or 0),
            "height": int(info.get("height") or 0),
            "fps": float(info.get("fps") or 0.0) or None,
            "duration_seconds": float(info.get("duration_seconds") or 0.0),
            "metadata_source": info.get("metadata_source"),
            "frame_count_source": info.get("frame_count_source"),
        }

    frames = list_frame_images(path)
    if not frames:
        raise FileNotFoundError(f"frame directory has no supported images: {path}")
    width, height = image_size(frames[0])
    return {
        "path": str(path),
        "name": path.name,
        "source_kind": "frames",
        "frame_count": len(frames),
        "width": width,
        "height": height,
        "fps": None,
        "duration_seconds": None,
        "metadata_source": "frames",
        "frame_count_source": "directory_listing",
    }


def validate_strict_alignment(reference: dict[str, Any], distorted: dict[str, Any]) -> None:
    if int(reference.get("frame_count") or 0) != int(distorted.get("frame_count") or 0):
        raise ValueError(
            "strict compare requires matching frame counts: "
            f"{reference.get('frame_count')} vs {distorted.get('frame_count')}"
        )
    if (int(reference.get("width") or 0), int(reference.get("height") or 0)) != (
        int(distorted.get("width") or 0),
        int(distorted.get("height") or 0),
    ):
        raise ValueError(
            "strict compare requires matching frame dimensions: "
            f"{reference.get('width')}x{reference.get('height')} vs "
            f"{distorted.get('width')}x{distorted.get('height')}"
        )
    reference_fps = reference.get("fps")
    distorted_fps = distorted.get("fps")
    if reference_fps is not None and distorted_fps is not None:
        if abs(float(reference_fps) - float(distorted_fps)) > 1e-6:
            raise ValueError(
                "strict compare requires matching fps metadata: "
                f"{reference_fps} vs {distorted_fps}"
            )


def validate_strict_decoded_alignment(
    reference_frames: list[Path],
    distorted_frames: list[Path],
    reference_fps: float | None,
    distorted_fps: float | None,
    reference_timestamps: list[float | None],
    distorted_timestamps: list[float | None],
) -> None:
    if len(reference_frames) != len(distorted_frames):
        raise ValueError(
            "strict compare requires matching frame counts: "
            f"{len(reference_frames)} vs {len(distorted_frames)}"
        )
    if reference_fps is not None and distorted_fps is not None:
        if abs(float(reference_fps) - float(distorted_fps)) > 1e-6:
            raise ValueError(
                "strict compare requires matching fps metadata: "
                f"{reference_fps} vs {distorted_fps}"
            )
    if _timestamps_available(reference_timestamps) and _timestamps_available(distorted_timestamps):
        if len(reference_timestamps) != len(reference_frames) or len(distorted_timestamps) != len(distorted_frames):
            raise ValueError("strict compare requires timestamp metadata for every decoded frame")
        for frame_index, (reference_ts, distorted_ts) in enumerate(zip(reference_timestamps, distorted_timestamps)):
            if reference_ts is None or distorted_ts is None:
                raise ValueError("strict compare requires timestamp metadata for every decoded frame")
            if abs(float(reference_ts) - float(distorted_ts)) > TIMESTAMP_ALIGNMENT_TOLERANCE_SECONDS:
                raise ValueError(
                    "strict compare requires matching frame timestamps: "
                    f"frame {frame_index} {float(reference_ts):.6f}s vs {float(distorted_ts):.6f}s"
                )


def _timestamps_available(values: list[float | None]) -> bool:
    return any(value is not None for value in values)


def list_frame_images(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"frame directory not found: {path}")
    return sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
    )


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def compare_video_name(reference_path: Path, distorted_path: Path) -> str:
    if reference_path.stem == distorted_path.stem:
        return reference_path.stem
    return f"{reference_path.stem}_vs_{distorted_path.stem}"
