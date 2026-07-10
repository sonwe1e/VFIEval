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
    if kind == "media_asset":
        if "path" in descriptor:
            raise ValueError("media_asset descriptors must not include client-supplied paths")
        info = _resolve_media_asset_descriptor(workspace, db, descriptor, role=role)
        info.update({"descriptor_kind": "media_asset", "role": role or "distorted"})
        return info
    if not kind and "path" in descriptor:
        raise ValueError("structured compare descriptors must use server-resolved sources, not path fields")
    raise ValueError(f"unsupported compare source descriptor kind: {kind or '<missing>'}")


def _resolve_media_asset_descriptor(
    workspace: WorkspaceConfig,
    db: Database,
    descriptor: dict[str, Any],
    role: str | None,
) -> dict[str, Any]:
    from vfieval.media_assets import resolve_asset_path

    asset_id = _positive_int(descriptor.get("asset_id"), "media_asset descriptor requires asset_id")
    asset, path = resolve_asset_path(db, workspace, asset_id, role=role)
    info = inspect_compare_path(workspace, path)
    provenance = asset.get("provenance") or {}
    metadata = asset.get("metadata") or {}
    label = str(
        descriptor.get("label")
        or descriptor.get("track_label")
        or asset.get("display_name")
        or f"asset-{asset_id}"
    )
    info.update(
        {
            "asset_id": asset_id,
            "source_kind": asset.get("source_kind"),
            "media_kind": asset.get("media_kind"),
            "asset_role": asset.get("role"),
            "label": label,
            "track_label": label,
            "run_id": provenance.get("run_id"),
            "track_run_id": provenance.get("run_id"),
            "artifact_id": provenance.get("artifact_id"),
            "artifact_kind": provenance.get("artifact_kind"),
            "video": provenance.get("video_name") or provenance.get("video") or path.stem,
            "video_name": provenance.get("video_name") or provenance.get("video") or path.stem,
            "group": provenance.get("video_group"),
            "asset_metadata": metadata,
            "source_video_path": metadata.get("source_video_path"),
            "source_video_group": metadata.get("source_video_group"),
            "source_video_file": metadata.get("source_video_file"),
            "source_frame_indices": metadata.get("source_frame_indices"),
            "frame_step": metadata.get("frame_step"),
        }
    )
    return info


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
            # Source-clip mapping (present on preds produced after the
            # source-clip-GT change). Compare uses these to head-offset the
            # source clip into a pred-aligned GT instead of relying on a
            # per-run gt_video copy. Legacy preds lack them and fall back.
            "source_video_path": metadata.get("source_video_path"),
            "source_video_group": metadata.get("source_video_group"),
            "source_video_file": metadata.get("source_video_file"),
            "source_frame_indices": metadata.get("source_frame_indices"),
            "frame_step": metadata.get("frame_step"),
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

    Frame count, dimensions, FPS, and available decoded timestamps must match.
    Platform-owned ``source_frame_indices`` are resolved before this function;
    external inputs are never trimmed, offset, resized, or normalized.
    """
    ref_fc = int(reference.get("frame_count") or 0)
    dist_fc = int(distorted.get("frame_count") or 0)
    if ref_fc <= 0 or dist_fc <= 0 or ref_fc != dist_fc:
        raise ValueError(f"strict compare requires matching frame counts: {ref_fc} vs {dist_fc}")
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
    if ref_w <= 0 or ref_h <= 0 or (ref_w, ref_h) != (dist_w, dist_h):
        raise ValueError(
            "strict compare requires matching dimensions: "
            f"{ref_w}x{ref_h} vs {dist_w}x{dist_h}"
        )
    return {
        # Original clip counts, useful for the UI badge ("帧数 X ≠ Y → 已对齐Z帧").
        "frame_count": ref_fc,
        "track_frame_count": dist_fc,
        # Length that downstream sample assembly actually uses.
        "effective_frame_count": ref_fc,
        "reference_needs_trim": False,
        "distorted_needs_trim": False,
        "width": ref_w,
        "height": ref_h,
        "track_width": dist_w,
        "track_height": dist_h,
        "target_width": ref_w,
        "target_height": ref_h,
        "reference_needs_downscale": False,
        "distorted_needs_downscale": False,
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
    # The caller may already have selected a platform-owned indexed GT subset;
    # after that selection both decoded sides must match exactly.
    if not reference_frames or not distorted_frames:
        raise ValueError("strict compare requires at least one aligned frame on each side")
    if len(reference_frames) != len(distorted_frames):
        raise ValueError(
            "strict compare requires matching decoded frame counts: "
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
