from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.file_inputs import VIDEO_SUFFIXES, inspect_video, project_root, resolve_video_group


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TIMESTAMP_ALIGNMENT_TOLERANCE_SECONDS = 1e-3
COMPARE_SOURCE_RUN_STATUSES = {"completed", "metric_queued", "metric_running"}


def resolve_compare_source_path(workspace: WorkspaceConfig, source: str) -> Path:
    text = str(source or "").strip()
    if not text:
        raise ValueError("compare source path is required")
    path = Path(text)
    if not path.is_absolute():
        path = (project_root(workspace) / path).resolve()
    else:
        path = path.resolve()
    if ".." in path.parts:
        raise ValueError("compare source path must not contain '..'")
    if not path.exists():
        raise FileNotFoundError(f"compare source not found: {path}")
    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        return path
    if path.is_dir():
        return path
    raise ValueError(f"unsupported compare source: {path}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def inspect_compare_path(workspace: WorkspaceConfig, path: Path) -> dict[str, Any]:
    path = path.resolve()
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


def resolve_compare_descriptor(
    workspace: WorkspaceConfig,
    db: Database,
    descriptor: Any,
    role: str | None = None,
) -> dict[str, Any]:
    if not isinstance(descriptor, dict):
        raise ValueError(
            "compare source descriptor must be an object with a 'kind' field; "
            "raw path strings are no longer accepted"
        )

    kind = str(descriptor.get("kind") or "").strip()
    if kind == "video_group":
        if role not in {None, "reference"}:
            raise ValueError("video_group descriptors are only valid for the compare reference")
        info = _resolve_video_group_descriptor(workspace, descriptor)
        info.update({"descriptor_kind": "video_group", "role": "reference"})
        return info
    if kind == "run_artifact":
        if "path" in descriptor:
            raise ValueError("run_artifact descriptors must not include client-supplied paths")
        info = _resolve_run_artifact_descriptor(workspace, db, descriptor)
        info.update({"descriptor_kind": "run_artifact", "role": role or "distorted"})
        return info
    if not kind and "path" in descriptor:
        raise ValueError("structured compare descriptors must use server-resolved sources, not path fields")
    raise ValueError(f"unsupported compare source descriptor kind: {kind or '<missing>'}")


def _resolve_video_group_descriptor(workspace: WorkspaceConfig, descriptor: dict[str, Any]) -> dict[str, Any]:
    group = str(descriptor.get("group") or "").strip()
    video = str(descriptor.get("video") or "").strip()
    if not group or not video:
        raise ValueError("video_group compare descriptor requires group and video")
    if Path(video).name != video:
        raise ValueError("video_group compare descriptor video must be a file name")
    folder = resolve_video_group(workspace, group)
    path = (folder / video).resolve()
    if not _is_relative_to(path, folder.resolve()):
        raise ValueError("video_group compare descriptor resolved outside its video group")
    if not path.exists():
        raise FileNotFoundError(f"compare GT video not found: {group}/{video}")
    info = inspect_compare_path(workspace, path)
    info.update({"group": group, "video": video, "label": descriptor.get("label") or video})
    return info


def _resolve_run_artifact_descriptor(
    workspace: WorkspaceConfig,
    db: Database,
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    run_id = _positive_int(descriptor.get("run_id"), "run_artifact descriptor requires run_id")
    artifact_kind = str(descriptor.get("artifact_kind") or "pred_video")
    run = db.get_run(run_id)
    if run.get("artifact_cleaned_at") is not None:
        raise ValueError(f"run {run_id} artifacts have been cleaned")
    if str(run.get("status") or "") not in COMPARE_SOURCE_RUN_STATUSES:
        raise ValueError(f"run {run_id} is not ready for compare sources: {run.get('status')}")

    artifacts = db.list_run_artifacts(run_id, kind=artifact_kind)
    artifact_id = descriptor.get("artifact_id")
    video = str(descriptor.get("video") or "").strip()
    artifact = None
    if artifact_id not in {None, ""}:
        desired_id = _positive_int(artifact_id, "artifact_id must be a positive integer")
        artifact = next((row for row in artifacts if int(row["id"]) == desired_id), None)
    else:
        artifact = next((row for row in artifacts if _artifact_matches_video(row, video)), None)
    if artifact is None:
        target = f"artifact_id={artifact_id}" if artifact_id not in {None, ""} else f"video={video or '<any>'}"
        raise FileNotFoundError(f"run {run_id} has no {artifact_kind} compare source for {target}")

    path = Path(str(artifact["path"])).resolve()
    workspace_root = workspace.root.resolve()
    if not _is_relative_to(path, workspace_root):
        raise ValueError("run artifact compare source resolved outside the VFIEval workspace")
    info = inspect_compare_path(workspace, path)
    metadata = artifact.get("metadata") or {}
    video_name = str(metadata.get("video_name") or video or path.stem)
    label = str(
        descriptor.get("label")
        or descriptor.get("track_label")
        or metadata.get("compare_track_label")
        or run.get("name")
        or f"run-{run_id}"
    )
    info.update(
        {
            "run_id": run_id,
            "run_name": run.get("name"),
            "artifact_id": int(artifact["id"]),
            "artifact_kind": artifact_kind,
            "video": video_name,
            "video_name": video_name,
            "label": label,
            "track_label": label,
            "track_run_id": run_id,
            "artifact_metadata": metadata,
        }
    )
    return info


def _artifact_matches_video(artifact: dict[str, Any], video: str) -> bool:
    if not video:
        return True
    metadata = artifact.get("metadata") or {}
    candidates = {
        str(metadata.get("video_name") or ""),
        str(metadata.get("video_file") or ""),
        Path(str(artifact.get("path") or "")).stem,
    }
    return video in candidates


def _positive_int(value: Any, message: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if number <= 0:
        raise ValueError(message)
    return number


def validate_strict_alignment(reference: dict[str, Any], distorted: dict[str, Any]) -> dict[str, Any]:
    """Validate reference/distorted alignment for ``strict`` compare.

    Frame counts are allowed to differ (common for VFI comparisons, where GT is
    the source clip with ``N`` frames and Pred with ``N-step`` frames). The
    effective, aligned frame count is ``min(ref_fc, dist_fc)`` — decoded-frame
    lists are trimmed to that length before sample assembly.

    Resolution is allowed to differ too: the higher-resolution side is
    downscaled per-frame to the common ``target_width`` x ``target_height``
    (per-axis min) before evaluation and visualization.

    fps must still match — it is not affected by frame sampling or scaling.

    Returns the resolved alignment metadata including ``effective_frame_count``
    and ``target_width``/``target_height``.
    """
    ref_fc = int(reference.get("frame_count") or 0)
    dist_fc = int(distorted.get("frame_count") or 0)
    # Common frame count is the shorter side. Decoded-frame lists are trimmed to
    # this downstream in validate_strict_decoded_alignment / scan_compare_*.
    effective_frame_count = min(ref_fc, dist_fc) if ref_fc and dist_fc else max(ref_fc, dist_fc)
    reference_fps = reference.get("fps")
    distorted_fps = distorted.get("fps")
    if reference_fps is not None and distorted_fps is not None:
        if abs(float(reference_fps) - float(distorted_fps)) > 1e-6:
            raise ValueError(
                "strict compare requires matching fps metadata: "
                f"{reference_fps} vs {distorted_fps}"
            )
    ref_w = int(reference.get("width") or 0)
    ref_h = int(reference.get("height") or 0)
    dist_w = int(distorted.get("width") or 0)
    dist_h = int(distorted.get("height") or 0)
    # Common target is the smaller of each axis. The higher-resolution side is
    # downscaled to this size before evaluation/visualization; the lower side is
    # never upscaled (downscaling only — preserves detail, matches VMAF's
    # equal-size requirement).
    target_w = min(ref_w, dist_w) if ref_w and dist_w else max(ref_w, dist_w)
    target_h = min(ref_h, dist_h) if ref_h and dist_h else max(ref_h, dist_h)
    return {
        # Original clip counts, useful for the UI badge ("帧数 X ≠ Y → 已对齐Z帧").
        "frame_count": ref_fc,
        "track_frame_count": dist_fc,
        # Length that downstream sample assembly actually uses.
        "effective_frame_count": effective_frame_count,
        "reference_needs_trim": ref_fc > effective_frame_count,
        "distorted_needs_trim": dist_fc > effective_frame_count,
        "width": ref_w,
        "height": ref_h,
        "track_width": dist_w,
        "track_height": dist_h,
        "target_width": target_w,
        "target_height": target_h,
        "reference_needs_downscale": (ref_w, ref_h) != (target_w, target_h),
        "distorted_needs_downscale": (dist_w, dist_h) != (target_w, target_h),
        "fps": reference_fps if reference_fps is not None else distorted_fps,
    }


def validate_strict_decoded_alignment(
    reference_frames: list[Path],
    distorted_frames: list[Path],
    reference_fps: float | None,
    distorted_fps: float | None,
    reference_timestamps: list[float | None],
    distorted_timestamps: list[float | None],
) -> None:
    # GT (N source frames) and Pred (typically N-step inferred frames) rarely
    # share a length. Trim both decoded-frame lists to their common min length
    # so downstream iteration always stays in lockstep. The leading frames are
    # kept — for step=1 this matches the natural "pred[i] ≈ source[i+1]" layout.
    common = min(len(reference_frames), len(distorted_frames))
    if common == 0:
        raise ValueError("strict compare requires at least one aligned frame on each side")
    if len(reference_frames) != len(distorted_frames):
        del reference_frames[common:]
        del distorted_frames[common:]
        # Timestamps frame-indexed with the frames; stay in lockstep.
        if reference_timestamps is not None and len(reference_timestamps) > common:
            del reference_timestamps[common:]
        if distorted_timestamps is not None and len(distorted_timestamps) > common:
            del distorted_timestamps[common:]
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
