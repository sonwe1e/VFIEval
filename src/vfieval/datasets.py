from __future__ import annotations

import errno
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable

from PIL import Image

from vfieval.alignment import materialize_aligned_frame
from vfieval.compare_inputs import (
    compare_video_name,
    image_size,
    list_frame_images,
    resolve_compare_source_path,
    validate_strict_decoded_alignment,
)
from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.file_inputs import (
    MIDPOINT_TRIPLET_CONTRACT,
    VIDEO_SUFFIXES,
    decode_backend_candidates,
    decode_backend_identity,
    decode_cache_dir,
    decode_cache_key,
    file_sha256,
    normalize_decode_backend,
)
from vfieval.run_cleanup import cache_lease, decode_cache_build_lock


def _frame_resize_copy(
    db: Database,
    workspace: WorkspaceConfig,
    src_path: Path,
    target_w: int,
    target_h: int,
) -> Path:
    """Resize a platform-owned aligned GT frame to ``(target_w, target_h)`` and cache
    the result under ``<workspace>/compare_cache``, keyed by source path +
    mtime/size + target size. Returns the path to the cached PNG.

    External sources must already match exactly. This helper exists for the
    explicit ``source_frame_indices`` path, where VFIEval generates an aligned
    GT asset at the inference output resolution.
    """
    src_path = src_path.resolve()
    stat = src_path.stat()
    key_data = (
        str(src_path).replace("\\", "/"),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(target_w),
        int(target_h),
    )
    cache_key = hashlib.sha256("|".join(str(item) for item in key_data).encode("utf-8")).hexdigest()
    cache_dir = workspace.root / "compare_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{cache_key}.png"
    with cache_lease(db, workspace, "compare_cache", cache_key, out_path):
        if not out_path.exists():
            with Image.open(src_path).convert("RGB") as image:
                image.resize((target_w, target_h), Image.LANCZOS).save(out_path)
    return out_path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ProgressCallback = Callable[[dict[str, Any]], None]


class DecodeCacheBuildError(RuntimeError):
    """A recoverable decode-cache coordination or publish failure."""


class VideoDecodeBackendError(RuntimeError):
    """The selected decoder could not decode this source video."""


class VideoEntry:
    """A decode target plus the identity it carries into the sample timeline.

    Single-group runs keep the historical identity (``display_name`` = file
    stem, ``video_file`` = file name). Multi-group runs qualify both with the
    group name (e.g. ``anime/clip01`` and ``anime/clip01.mp4``) so a clip that
    exists under several ``videos/`` groups never collides in the run timeline
    or artifact layout.
    """

    __slots__ = ("path", "display_name", "video_file", "group")

    def __init__(self, path: Path, display_name: str, video_file: str, group: str | None = None) -> None:
        self.path = path
        self.display_name = display_name
        self.video_file = video_file
        self.group = group

    @property
    def sample_token(self) -> str:
        return _sample_token(self.display_name)


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

    entries = _resolve_video_entries(Path(dataset["root_path"]), video_glob, selected_videos)
    if not entries:
        raise FileNotFoundError(f"no videos found under {dataset['root_path']}")
    videos = [entry.path for entry in entries]

    decoded_root = workspace.root / "decode_cache"
    decoded_root.mkdir(parents=True, exist_ok=True)
    added = 0
    decoded_frames = 0
    decode_results: list[
        tuple[str, list[Path], float, list[float | None], dict[str, Any]] | None
    ] = [None for _ in videos]
    progress_counts = [0 for _ in videos]
    cache_hits = 0
    cache_misses = 0
    cache_hit_videos: list[str] = []
    cache_miss_videos: list[str] = []
    decoder_reports: list[dict[str, Any]] = []
    lock = Lock()

    def decode_one(
        video_index: int,
        entry: "VideoEntry",
    ) -> tuple[str, list[Path], float, list[float | None], dict[str, Any]]:
        video_path = entry.path
        local_cache_hit = False
        local_cache_miss = False

        def on_decode_progress(event: dict[str, Any]) -> None:
            nonlocal cache_hits, cache_misses, local_cache_hit, local_cache_miss
            if progress_callback is None:
                return
            frames_done = int(event.get("frames") or 0)
            with lock:
                progress_counts[video_index] = frames_done
                if event.get("event") == "cache_hit" and not local_cache_hit:
                    if local_cache_miss:
                        local_cache_miss = False
                        cache_misses = max(0, cache_misses - 1)
                        try:
                            cache_miss_videos.remove(entry.display_name)
                        except ValueError:
                            pass
                    local_cache_hit = True
                    cache_hits += 1
                    cache_hit_videos.append(entry.display_name)
                elif event.get("event") == "cache_miss" and not local_cache_miss:
                    local_cache_miss = True
                    cache_misses += 1
                    cache_miss_videos.append(entry.display_name)
                payload = {
                    **event,
                    "video_index": video_index,
                    "video_count": len(entries),
                    "video_name": entry.display_name,
                    "decoded_frames": sum(progress_counts),
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "cache_hit_videos": list(cache_hit_videos),
                    "cache_miss_videos": list(cache_miss_videos),
                }
            progress_callback(payload)

        cache_key, frames, fps, timestamps, decode_info = _decode_video_with_backend_cache(
            db,
            workspace,
            video_path,
            max_frames,
            decode_mode,
            frame_step,
            progress_callback=on_decode_progress,
            decode_backend=decode_backend,
        )
        return cache_key, frames, fps, timestamps, decode_info

    worker_count = _decode_worker_count(len(entries), metadata)
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(decode_one, index, entry): index for index, entry in enumerate(entries)}
            for future in as_completed(futures):
                decode_results[futures[future]] = future.result()
    else:
        for video_index, entry in enumerate(entries):
            decode_results[video_index] = decode_one(video_index, entry)

    for video_index, entry in enumerate(entries):
        result = decode_results[video_index]
        if result is None:
            raise RuntimeError(f"video decode did not produce a result: {entry.path.name}")
        cache_key, frames, fps, timestamps, decode_info = result
        decoder_reports.append(
            {
                "video_file": entry.video_file,
                "cache_key": cache_key,
                "requested_backend": str(
                    decode_info.get("requested_backend") or decode_backend
                ),
                "actual_backend": str(
                    decode_info.get("manifest_backend")
                    or decode_info.get("backend")
                    or "unknown"
                ),
                "decoder_identity": decode_info.get("decoder_identity"),
                "cache_hit": bool(decode_info.get("cache_hit")),
                "fallback_reason": decode_info.get("fallback_reason"),
                "timestamps_available": bool(
                    decode_info.get("timestamps_available")
                ),
            }
        )
        decoded_frames += len(frames)
        if decode_mode == "video_gt_triplets":
            added += _add_video_triplets(
                db,
                dataset_id,
                entry,
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
                entry,
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
        video_count=len(entries),
        frame_count=decoded_frames,
        metadata={
            "frame_step": frame_step,
            "max_frames": max_frames,
            "video_glob": video_glob,
            "selected_videos": [entry.video_file for entry in entries],
            "decode_mode": decode_mode,
            "decode_backend": decode_backend,
            "decoder_reports": decoder_reports,
            "evaluation_contract": (
                MIDPOINT_TRIPLET_CONTRACT
                if decode_mode == "video_gt_triplets"
                else None
            ),
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

    alignment_plan = metadata.get("alignment_plan")
    if alignment_plan is not None and not isinstance(alignment_plan, dict):
        raise ValueError("compare alignment_plan must be an object")

    # External inputs have already passed exact strict validation. A target may
    # still be present for the platform-owned aligned-GT path.
    target_w = _optional_positive_int(metadata.get("compare_target_width"))
    target_h = _optional_positive_int(metadata.get("compare_target_height"))
    if alignment_plan:
        target = alignment_plan.get("target") or {}
        target_w = _optional_positive_int(target.get("width"))
        target_h = _optional_positive_int(target.get("height"))
    downscale_mode = "none"
    if target_w is not None and target_h is not None and target_w > 0 and target_h > 0:
        # Image-size inspection happens per-frame below; defer the decision on
        # which side to downscale until we see actual decoded sizes.
        downscale_mode = "enabled"

    reference_frames, reference_fps, reference_timestamps = _load_compare_source_frames(db, workspace, reference_path, "compare_reference")
    distorted_frames, distorted_fps, distorted_timestamps = _load_compare_source_frames(db, workspace, distorted_path, "compare_distorted")
    validate_strict_decoded_alignment(
        reference_frames=reference_frames,
        distorted_frames=distorted_frames,
        reference_fps=reference_fps,
        distorted_fps=distorted_fps,
        reference_timestamps=reference_timestamps,
        distorted_timestamps=distorted_timestamps,
    )

    compare_name = compare_video_name(reference_path, distorted_path)
    reference_slot = _alignment_source_slot(alignment_plan, "gt", 0) if alignment_plan else None
    distorted_slot = _alignment_source_slot(alignment_plan, "pred", 0) if alignment_plan else None
    added = 0
    for frame_index, (reference_frame, distorted_frame) in enumerate(zip(reference_frames, distorted_frames)):
        reference_frame, distorted_frame = _align_compare_frames(
            db=db,
            workspace=workspace,
            reference_frame=reference_frame,
            distorted_frame=distorted_frame,
            target_w=target_w,
            target_h=target_h,
            downscale_mode=downscale_mode,
            alignment_plan=alignment_plan,
            reference_slot=reference_slot,
            distorted_slot=distorted_slot,
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
                **({"alignment_fingerprint": alignment_plan.get("fingerprint")} if alignment_plan else {}),
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
            **({"alignment_plan": alignment_plan} if alignment_plan else {}),
            **({"compare_target_width": target_w, "compare_target_height": target_h, "downscale_mode": downscale_mode} if target_w else {}),
        },
    )
    return added


def _align_compare_frames(
    db: Database,
    workspace: WorkspaceConfig,
    reference_frame: Path,
    distorted_frame: Path,
    target_w: int | None,
    target_h: int | None,
    downscale_mode: str,
    alignment_plan: dict[str, Any] | None = None,
    reference_slot: str | None = None,
    distorted_slot: str | None = None,
) -> tuple[Path, Path]:
    """Resolve frame-level dimension alignment for the two compare sides.

    When a platform-owned indexed mapping supplies a target resolution, resize
    the generated aligned GT to that exact resolution. External inputs already
    match, so this function does not normalize external mismatches.
    """
    if alignment_plan is not None:
        if not reference_slot or not distorted_slot:
            raise ValueError("alignment plan is missing GT or Pred source slots")
        return (
            materialize_aligned_frame(db, workspace, alignment_plan, reference_slot, reference_frame),
            materialize_aligned_frame(db, workspace, alignment_plan, distorted_slot, distorted_frame),
        )
    if downscale_mode != "enabled" or target_w is None or target_h is None:
        return reference_frame, distorted_frame
    ref_size = image_size(reference_frame)
    dist_size = image_size(distorted_frame)
    if ref_size == dist_size and ref_size == (target_w, target_h):
        return reference_frame, distorted_frame
    if ref_size != (target_w, target_h):
        reference_frame = _frame_resize_copy(db, workspace, reference_frame, target_w, target_h)
    if dist_size != (target_w, target_h):
        distorted_frame = _frame_resize_copy(db, workspace, distorted_frame, target_w, target_h)
    return reference_frame, distorted_frame


def _select_track_reference(
    reference_frames: list[Path],
    reference_timestamps: list[float | None],
    source_frame_indices: Any,
    distorted_count: int,
) -> tuple[list[Path], list[float | None], str]:
    """Pick a pred-aligned GT subset from the source clip for one track.

    A run's pred approximates ``source_frames[gt_index]`` for each sample, so the
    ordered ``source_frame_indices`` (the recorded ``gt_index`` list) map pred
    frame ``i`` onto the correct source frame. Selecting those frames yields a GT
    that lines up head-to-head with the pred without an implicit offset.

    Returns fresh lists (never the shared ``reference_frames``). Tracks without
    a mapping use strict one-to-one alignment; a present but invalid mapping is
    rejected instead of silently falling back.
    """
    indices = [int(value) for value in source_frame_indices] if source_frame_indices else []
    if not indices:
        return list(reference_frames), list(reference_timestamps), "strict"
    if len(indices) != int(distorted_count):
        raise ValueError("source_frame_indices must contain exactly one index per pred frame")
    if any(index < 0 or index >= len(reference_frames) for index in indices):
        raise ValueError("source_frame_indices contains an index outside the reference clip")
    selected_frames = [reference_frames[index] for index in indices]
    selected_timestamps = [
        reference_timestamps[index] if index < len(reference_timestamps) else None
        for index in indices
    ]
    return selected_frames, selected_timestamps, "indexed"


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

    alignment_plan = metadata.get("alignment_plan")
    if alignment_plan is not None and not isinstance(alignment_plan, dict):
        raise ValueError("compare alignment_plan must be an object")

    # Exact external resolution, or the inference resolution for indexed GT.
    target_w = _optional_positive_int(metadata.get("compare_target_width"))
    target_h = _optional_positive_int(metadata.get("compare_target_height"))
    if alignment_plan:
        target = alignment_plan.get("target") or {}
        target_w = _optional_positive_int(target.get("width"))
        target_h = _optional_positive_int(target.get("height"))
    downscale_mode = "none"
    if target_w is not None and target_h is not None and target_w > 0 and target_h > 0:
        downscale_mode = "enabled"

    reference_frames, reference_fps, reference_timestamps = _load_compare_source_frames(db, workspace, reference_path, "compare_reference")
    video_name = str(metadata.get("video_name") or reference_path.stem)
    video_token = _sample_token(video_name)
    added = 0
    scanned_tracks: list[dict[str, Any]] = []
    reference_slot = _alignment_source_slot(alignment_plan, "gt", 0) if alignment_plan else None

    for track_index, track in enumerate(tracks):
        distorted_path = resolve_compare_source_path(workspace, str(track.get("distorted_path") or ""))
        track_label = str(track.get("track_label") or track.get("label") or f"pred{track_index + 1}")
        track_token = _sample_token(track_label)
        distorted_slot = (
            str(track.get("alignment_slot") or _alignment_source_slot(alignment_plan, "pred", track_index))
            if alignment_plan
            else None
        )
        distorted_frames, distorted_fps, distorted_timestamps = _load_compare_source_frames(
            db,
            workspace,
            distorted_path,
            f"compare_distorted_{track_index}",
        )
        # A pred does not approximate source frame i; it approximates
        # source_frames[gt_index]. When the track carries source_frame_indices,
        # select exactly those source frames as this track's GT so pred[i] lines
        # up with reference[i]. Legacy preds without a mapping remain valid only
        # when they already satisfy the exact strict contract.
        track_reference_frames, track_reference_timestamps, alignment_mode = _select_track_reference(
            reference_frames=reference_frames,
            reference_timestamps=reference_timestamps,
            source_frame_indices=track.get("source_frame_indices"),
            distorted_count=len(distorted_frames),
        )
        if alignment_mode == "strict":
            validate_strict_decoded_alignment(
                reference_frames=track_reference_frames,
                distorted_frames=distorted_frames,
                reference_fps=reference_fps,
                distorted_fps=distorted_fps,
                reference_timestamps=track_reference_timestamps,
                distorted_timestamps=distorted_timestamps,
            )
        else:
            if len(track_reference_frames) != len(distorted_frames):
                raise ValueError(
                    "source_frame_indices must contain exactly one reference frame per pred frame"
                )
            if reference_fps is not None and distorted_fps is not None:
                if abs(float(reference_fps) - float(distorted_fps)) > 1e-6:
                    raise ValueError(
                        "strict compare requires matching fps metadata: "
                        f"{reference_fps} vs {distorted_fps}"
                    )
        scanned_tracks.append(
            {
                "track_label": track_label,
                "track_key": track_token,
                "track_run_id": track.get("track_run_id"),
                "artifact_id": track.get("artifact_id"),
                "distorted_path": str(distorted_path),
                "frame_count": len(distorted_frames),
                "alignment_mode": alignment_mode,
                **({"alignment_slot": distorted_slot} if distorted_slot else {}),
            }
        )
        for frame_index, (reference_frame, distorted_frame) in enumerate(zip(track_reference_frames, distorted_frames)):
            reference_frame, distorted_frame = _align_compare_frames(
                db=db,
                workspace=workspace,
                reference_frame=reference_frame,
                distorted_frame=distorted_frame,
                target_w=target_w,
                target_h=target_h,
                downscale_mode=downscale_mode,
                alignment_plan=alignment_plan,
                reference_slot=reference_slot,
                distorted_slot=distorted_slot,
            )
            sample_index = track_index * len(track_reference_frames) + frame_index
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
                    **({"alignment_fingerprint": alignment_plan.get("fingerprint")} if alignment_plan else {}),
                    "timestamps": {
                        "gt": track_reference_timestamps[frame_index] if frame_index < len(track_reference_timestamps) else None,
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
            **({"alignment_plan": alignment_plan} if alignment_plan else {}),
            **({"compare_target_width": target_w, "compare_target_height": target_h, "downscale_mode": downscale_mode} if target_w else {}),
        },
    )
    return added


def _alignment_source_slot(plan: dict[str, Any] | None, role: str, index: int) -> str:
    if not plan:
        raise ValueError("alignment plan is required")
    slots = [
        str(slot)
        for slot, report in (plan.get("sources") or {}).items()
        if isinstance(report, dict) and str(report.get("role") or "") == role
    ]
    if index < 0 or index >= len(slots):
        raise ValueError(f"alignment plan has no {role} source at index {index}")
    return slots[index]


def _add_video_triplets(
    db: Database,
    dataset_id: int,
    entry: "VideoEntry",
    video_index: int,
    frames: list[Path],
    frame_step: int,
    fps: float,
    timestamps: list[float | None],
    cache_key: str,
) -> int:
    added = 0
    if len(frames) < 2 * frame_step + 1:
        return 0
    # Only publish real symmetric triples. A trailing pair without a distinct
    # midpoint GT is not a full-reference evaluation sample.
    for sample_index, img0_index in enumerate(range(0, len(frames) - 2 * frame_step)):
        gt_index = img0_index + frame_step
        img1_index = img0_index + 2 * frame_step
        name = f"{video_index:03d}_{entry.sample_token}_{sample_index:06d}"
        db.add_sample(
            dataset_id=dataset_id,
            name=name,
            img0_path=str(frames[img0_index]),
            img1_path=str(frames[img1_index]),
            gt_path=str(frames[gt_index]),
            metadata=_video_sample_metadata(
                entry,
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
    entry: "VideoEntry",
    video_index: int,
    frames: list[Path],
    frame_step: int,
    fps: float,
    timestamps: list[float | None],
    cache_key: str,
) -> int:
    added = 0
    if len(frames) < frame_step + 1:
        return 0
    for sample_index, img0_index in enumerate(range(0, len(frames) - frame_step)):
        img1_index = img0_index + frame_step
        name = f"{video_index:03d}_{entry.sample_token}_{sample_index:06d}"
        db.add_sample(
            dataset_id=dataset_id,
            name=name,
            img0_path=str(frames[img0_index]),
            img1_path=str(frames[img1_index]),
            gt_path=None,
            metadata=_video_sample_metadata(
                entry,
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


def _resolve_video_entries(
    root: Path,
    video_glob: str,
    selected_videos: Any = None,
) -> list["VideoEntry"]:
    """Resolve selected videos into decode entries carrying their identity.

    Two shapes are supported:

    - Single-group runs: ``root`` is a ``videos/<group>`` folder and each
      selected entry is a bare file name. Identity stays historical
      (display_name = file stem) so existing runs and caches are unaffected.
    - Multi-group runs: ``root`` is the ``videos/`` directory and each selected
      entry is ``"<group>/<file>"``. Identity is qualified with the group so
      same-named clips across groups never collide.

    ``selected_videos=None`` falls back to the historical directory scan.
    """
    if selected_videos is None:
        return [
            VideoEntry(path, path.stem, path.name, None)
            for path in _find_videos(root, video_glob, None)
        ]
    raw_names = [str(item) for item in selected_videos]
    if not raw_names:
        raise ValueError("selected_videos must contain at least one video")
    multi_group = any("/" in name or "\\" in name for name in raw_names)
    entries: list[VideoEntry] = []
    root_resolved = root.resolve()
    for raw in raw_names:
        normalized = raw.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        if multi_group:
            if len(parts) != 2:
                raise ValueError(f"multi-group video selection must be 'group/file': {raw}")
            group, file_name = parts
            if Path(file_name).name != file_name or Path(group).name != group:
                raise ValueError(f"invalid multi-group video selection: {raw}")
            path = (root / group / file_name).resolve()
            display_name = f"{group}/{Path(file_name).stem}"
            video_file = f"{group}/{file_name}"
            group_name: str | None = group
        else:
            file_name = parts[-1] if parts else raw
            if Path(file_name).name != file_name:
                raise ValueError(f"selected video must be a file name: {raw}")
            path = (root / file_name).resolve()
            display_name = Path(file_name).stem
            video_file = file_name
            group_name = None
        try:
            path.relative_to(root_resolved)
        except ValueError:
            raise ValueError(f"selected video resolved outside the dataset root: {raw}")
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            raise FileNotFoundError(f"selected video not found: {raw}")
        entries.append(VideoEntry(path, display_name, video_file, group_name))
    return entries


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
    db: Database,
    workspace: WorkspaceConfig,
    source_path: Path,
    cache_prefix: str,
) -> tuple[list[Path], float | None, list[float | None]]:
    frames, fps, timestamps, _cache = _load_compare_source_frames_with_cache(
        db,
        workspace,
        source_path,
        cache_prefix,
    )
    return frames, fps, timestamps


def _load_compare_source_frames_with_cache(
    db: Database,
    workspace: WorkspaceConfig,
    source_path: Path,
    cache_prefix: str,
    *,
    trusted_source_signature: dict[str, Any] | None = None,
) -> tuple[
    list[Path],
    float | None,
    list[float | None],
    dict[str, Any] | None,
]:
    """Load Compare frames and expose the already-computed decode cache identity.

    Long-running consumers such as Campaign publication can retain a lease on
    the returned cache directory without hashing the source video a second
    time. Frame-directory inputs have no shared decode cache and return
    ``None`` for the descriptor.
    """
    if source_path.is_file() and source_path.suffix.lower() in VIDEO_SUFFIXES:
        source_path = source_path.resolve()
        stat_before = source_path.stat()
        identity_started = time.monotonic()
        trusted = dict(trusted_source_signature or {})
        trusted_sha256 = str(trusted.get("content_sha256") or "").strip().lower()
        try:
            trusted_size = int(trusted.get("size_bytes"))
            trusted_mtime_ns = int(trusted.get("source_mtime_ns"))
        except (TypeError, ValueError):
            trusted_size = -1
            trusted_mtime_ns = -1
        trusted_identity = bool(
            trusted_sha256
            and trusted_size == int(stat_before.st_size)
            and trusted_mtime_ns == int(stat_before.st_mtime_ns)
        )
        source_sha256 = trusted_sha256 if trusted_identity else file_sha256(source_path)
        stat_after_hash = source_path.stat()
        if (
            int(stat_before.st_size) != int(stat_after_hash.st_size)
            or int(stat_before.st_mtime_ns) != int(stat_after_hash.st_mtime_ns)
        ):
            raise ValueError(
                "compare source video changed while its content signature was being read"
            )
        identity_seconds = time.monotonic() - identity_started
        decode_started = time.monotonic()
        cache_key, frames, fps, timestamps, decode_info = _decode_video_with_backend_cache(
            db,
            workspace,
            source_path,
            None,
            cache_prefix,
            1,
            content_sha256=source_sha256,
        )
        decode_seconds = time.monotonic() - decode_started
        stat_after_decode = source_path.stat()
        if (
            int(stat_after_hash.st_size) != int(stat_after_decode.st_size)
            or int(stat_after_hash.st_mtime_ns) != int(stat_after_decode.st_mtime_ns)
        ):
            raise ValueError("compare source video changed while it was being decoded")
        return (
            frames,
            fps,
            list(timestamps),
            {
                "cache_type": "decode_cache",
                "cache_key": cache_key,
                "path": decode_cache_dir(workspace, cache_key),
                "source_path": source_path,
                "source_sha256": source_sha256,
                "source_size_bytes": int(stat_after_decode.st_size),
                "source_mtime_ns": int(stat_after_decode.st_mtime_ns),
                "source_identity": "trusted_catalog" if trusted_identity else "full_sha256",
                "source_identity_seconds": identity_seconds,
                "source_hash_seconds": 0.0 if trusted_identity else identity_seconds,
                "decode_seconds": decode_seconds,
                "cache_hit": bool(decode_info.get("cache_hit")),
                "decode_backend": (
                    decode_info.get("manifest_backend") or decode_info.get("backend")
                ),
                "decode_backend_request": decode_info.get("requested_backend"),
                "decoder_identity": decode_info.get("decoder_identity"),
                "timestamps_available": bool(decode_info.get("timestamps_available")),
            },
        )
    if source_path.is_dir():
        frames = list_frame_images(source_path)
        if not frames:
            raise FileNotFoundError(f"frame directory has no supported images: {source_path}")
        return frames, None, [None for _ in frames], None
    raise ValueError(f"unsupported compare source: {source_path}")


def _decode_video_with_backend_cache(
    db: Database,
    workspace: WorkspaceConfig,
    video_path: Path,
    max_frames: int | None,
    decode_mode: str,
    frame_step: int,
    progress_callback: ProgressCallback | None = None,
    decode_backend: str = "auto",
    *,
    content_sha256: str | None = None,
) -> tuple[str, list[Path], float, list[float | None], dict[str, Any]]:
    """Resolve an auto request into backend-specific cache identities.

    FFmpeg and OpenCV never publish into the same key. An ``auto`` request
    tries the FFmpeg identity first and only then opens the independently
    keyed OpenCV cache/build path.
    """

    requested = normalize_decode_backend(decode_backend)
    source_sha256 = str(content_sha256 or file_sha256(video_path))
    fallback_reason: str | None = None
    for actual in decode_backend_candidates(requested):
        cache_key = decode_cache_key(
            video_path,
            decode_mode,
            frame_step,
            max_frames,
            content_sha256=source_sha256,
            requested_backend=requested,
            actual_backend=actual,
        )
        try:
            frames, fps, timestamps, decode_info = _decode_video_cached(
                db,
                workspace,
                video_path,
                cache_key,
                max_frames,
                decode_mode,
                frame_step,
                progress_callback=progress_callback,
                decode_backend=actual,
                requested_decode_backend=requested,
                decoder_identity=decode_backend_identity(actual),
                decode_fallback_reason=fallback_reason,
            )
            if fallback_reason and not decode_info.get("fallback_reason"):
                decode_info["fallback_reason"] = fallback_reason
            decode_info["requested_backend"] = requested
            decode_info["decoder_identity"] = decode_backend_identity(actual)
            decode_info.setdefault(
                "timestamps_available",
                bool(timestamps)
                and all(
                    value is not None and math.isfinite(float(value))
                    for value in timestamps
                ),
            )
            return cache_key, frames, fps, timestamps, decode_info
        except DecodeCacheBuildError:
            raise
        except RuntimeError as exc:
            if requested != "auto" or actual != "ffmpeg":
                raise
            fallback_reason = f"ffmpeg unavailable or failed: {exc}"
            if progress_callback:
                progress_callback(
                    {
                        "event": "fallback",
                        "backend": "opencv",
                        "fallback_reason": fallback_reason,
                        "frames": 0,
                    }
                )
    raise RuntimeError("no decode backend is available")


def _decode_video_cached(
    db: Database,
    workspace: WorkspaceConfig,
    video_path: Path,
    cache_key: str,
    max_frames: int | None,
    decode_mode: str,
    frame_step: int,
    progress_callback: ProgressCallback | None = None,
    decode_backend: str = "auto",
    *,
    requested_decode_backend: str | None = None,
    decoder_identity: dict[str, Any] | None = None,
    decode_fallback_reason: str | None = None,
) -> tuple[list[Path], float, list[float | None], dict[str, Any]]:
    output_dir = decode_cache_dir(workspace, cache_key)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Every caller, including Compare/Campaign preflight paths, now holds a
    # normal cache lease while reading or building the entry.  The separate
    # build lock below is intentionally not a GC lease.
    with cache_lease(db, workspace, "decode_cache", cache_key, output_dir):
        cached = _read_valid_decode_cache(output_dir, cache_key)
        if cached is not None:
            return _decode_cache_hit_result(cached, progress_callback)

        last_wait_report = 0.0

        def report_wait() -> None:
            nonlocal last_wait_report
            if progress_callback is None:
                return
            now = time.monotonic()
            if now - last_wait_report < 1.0:
                return
            last_wait_report = now
            progress_callback(
                {
                    "event": "cache_wait",
                    "phase": "waiting_for_cache",
                    "backend": "cache",
                    "frames": 0,
                }
            )

        staging_dir: Path | None = None
        try:
            with decode_cache_build_lock(db, cache_key, on_wait=report_wait) as build_lock:
                # A waiter must always revalidate after it owns the producer
                # slot: the previous producer may have published immediately
                # before releasing its row.
                cached = _read_valid_decode_cache(output_dir, cache_key)
                if cached is not None:
                    return _decode_cache_hit_result(cached, progress_callback)

                # Only the active producer may remove malformed final output,
                # legacy shared partial output, or stale private staging.
                _remove_decode_cache_path_if_present(workspace, output_dir.with_name(output_dir.name + ".partial"))
                _remove_decode_cache_path_if_present(workspace, output_dir)
                staging_root = workspace.tmp_dir / "decode-cache-staging"
                _clear_stale_decode_staging(staging_root, cache_key)
                staging_root.mkdir(parents=True, exist_ok=True)
                staging_dir = staging_root / f"{cache_key}.{uuid.uuid4().hex}.partial"
                staging_dir.mkdir(parents=False, exist_ok=False)

                if progress_callback:
                    progress_callback(
                        {"event": "cache_miss", "phase": "decoding", "backend": decode_backend, "frames": 0}
                    )
                frames, fps, timestamps, decode_info = _decode_video(
                    video_path,
                    staging_dir,
                    max_frames,
                    decode_backend=decode_backend,
                    progress_callback=progress_callback,
                )
                reported_backend = str(decode_info.get("backend") or "")
                if decoder_identity is not None and reported_backend != decode_backend:
                    raise VideoDecodeBackendError(
                        "decoder result backend does not match its cache identity: "
                        f"expected {decode_backend}, got {reported_backend or '<missing>'}"
                    )
                if decode_fallback_reason:
                    decode_info["fallback_reason"] = decode_fallback_reason
                width, height = _frame_size(frames[0]) if frames else (0, 0)
                final_frames = [output_dir / path.name for path in frames]
                available_timestamps = bool(timestamps) and all(
                    value is not None and math.isfinite(float(value)) for value in timestamps
                )
                duration_seconds = (
                    max(0.0, float(timestamps[-1]) - float(timestamps[0]))
                    if available_timestamps
                    else (len(frames) / fps if fps > 0 else 0.0)
                )
                manifest = {
                    "video_name": video_path.stem,
                    "video_file": video_path.name,
                    "video_path": str(video_path.resolve()),
                    "cache_key": cache_key,
                    "fps": fps,
                    "frames": [str(path.resolve()) for path in final_frames],
                    "timestamps": timestamps,
                    "frame_count": len(frames),
                    "width": width,
                    "height": height,
                    "duration_seconds": duration_seconds,
                    "valid_triplets": max(0, len(frames) - 2 * frame_step),
                    "evaluation_contract": (
                        MIDPOINT_TRIPLET_CONTRACT
                        if decode_mode == "video_gt_triplets"
                        else None
                    ),
                    "decode_status": "completed",
                    "decode_mode": decode_mode,
                    "frame_step": frame_step,
                    "max_frames": max_frames,
                    "decode_backend": decode_info.get("backend"),
                    "decode_backend_request": normalize_decode_backend(
                        requested_decode_backend or decode_backend
                    ),
                    "decoder_identity": decoder_identity
                    or {
                        "backend": str(decode_info.get("backend") or decode_backend),
                        "executable": "",
                        "version": "unspecified-direct-cache-call",
                    },
                    "decode_fallback_reason": decode_info.get("fallback_reason"),
                    "timestamps_available": available_timestamps,
                    "timestamps_unavailable_reason": (
                        None
                        if available_timestamps
                        else "decoder did not expose one finite timestamp per decoded frame"
                    ),
                    "color": "RGB",
                    "color_policy": "decoded-rgb-png-v1",
                    "rotation_policy": "backend-native-auto-rotation-v1",
                }
                (staging_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
                )

                if build_lock.lost.is_set():
                    raise DecodeCacheBuildError(
                        "decode cache build ownership was lost; retry the input check"
                    )

                # This short SQLite-fenced region performs the only final-dir
                # mutation. A former producer cannot publish after expiry and
                # takeover, and WinError 183 is handled as a winner cache hit.
                with db.decode_cache_build_publish_guard(
                    cache_key,
                    build_lock.owner_token,
                    ttl_seconds=build_lock.ttl_seconds,
                ):
                    cached = _read_valid_decode_cache(output_dir, cache_key)
                    if cached is not None:
                        return _decode_cache_hit_result(cached, progress_callback)
                    _remove_decode_cache_path_if_present(workspace, output_dir)
                    winner = _publish_decode_staging(staging_dir, output_dir, cache_key)
                    if winner is not None:
                        return _decode_cache_hit_result(winner, progress_callback)
                    staging_dir = None
                    published = _read_valid_decode_cache(output_dir, cache_key)
                    if published is None:
                        raise DecodeCacheBuildError(
                            "decode cache publish produced an invalid manifest; retry the input check"
                        )
                    published_frames, published_fps, published_timestamps, _manifest = published
                    return published_frames, published_fps, published_timestamps, decode_info
        except TimeoutError as exc:
            raise DecodeCacheBuildError(str(exc)) from exc
        except DecodeCacheBuildError:
            raise
        except OSError as exc:
            raise DecodeCacheBuildError(
                f"decode cache build failed; retry the input check ({exc})"
            ) from exc
        finally:
            if staging_dir is not None:
                _remove_private_decode_staging(staging_dir)


def _decode_cache_hit_result(
    cached: tuple[list[Path], float, list[float | None], dict[str, Any]],
    progress_callback: ProgressCallback | None,
) -> tuple[list[Path], float, list[float | None], dict[str, Any]]:
    frames, fps, timestamps, manifest = cached
    decode_info = {
        "backend": "cache",
        "manifest_backend": manifest.get("decode_backend"),
        "fallback_reason": manifest.get("decode_fallback_reason"),
        "requested_backend": manifest.get("decode_backend_request"),
        "decoder_identity": manifest.get("decoder_identity"),
        "timestamps_available": bool(manifest.get("timestamps_available")),
        "cache_hit": True,
    }
    if progress_callback:
        progress_callback(
            {
                "event": "cache_hit",
                "phase": "indexing_cached_frames",
                "backend": "cache",
                "manifest_backend": manifest.get("decode_backend"),
                "frames": len(frames),
            }
        )
    return frames, fps, timestamps, decode_info


def _read_valid_decode_cache(
    output_dir: Path,
    cache_key: str,
) -> tuple[list[Path], float, list[float | None], dict[str, Any]] | None:
    """Return a completed cache only when its manifest and every frame agree."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or str(manifest.get("cache_key") or "") != str(cache_key):
            return None
        raw_frames = manifest.get("frames")
        raw_timestamps = manifest.get("timestamps")
        if not isinstance(raw_frames, list) or not isinstance(raw_timestamps, list) or not raw_frames:
            return None
        if int(manifest.get("frame_count")) != len(raw_frames) or len(raw_timestamps) != len(raw_frames):
            return None
        fps = float(manifest.get("fps") or 24.0)
        if not math.isfinite(fps) or fps <= 0:
            return None
        timestamps = [None if value is None else float(value) for value in raw_timestamps]
        if not all(value is None or math.isfinite(value) for value in timestamps):
            return None
        root = output_dir.resolve()
        frames = [Path(str(value)).resolve() for value in raw_frames]
        if len(set(frames)) != len(frames):
            return None
        for frame in frames:
            try:
                frame.relative_to(root)
            except ValueError:
                return None
            if not frame.is_file():
                return None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return frames, fps, timestamps, manifest


def _remove_decode_cache_path_if_present(workspace: WorkspaceConfig, path: Path) -> None:
    """Remove a final/legacy partial cache path only after a producer claim."""
    root = (workspace.root / "decode_cache").resolve()
    if path.parent.resolve() != root:
        raise ValueError("decode cache cleanup path is outside the managed cache root")
    if path.is_symlink():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _clear_stale_decode_staging(staging_root: Path, cache_key: str) -> None:
    if not staging_root.is_dir():
        return
    prefix = f"{cache_key}."
    for child in staging_root.iterdir():
        if child.name.startswith(prefix) and child.name.endswith(".partial"):
            _remove_private_decode_staging(child)


def _remove_private_decode_staging(path: Path) -> None:
    if path.is_symlink():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _publish_decode_staging(
    staging_dir: Path,
    output_dir: Path,
    cache_key: str,
) -> tuple[list[Path], float, list[float | None], dict[str, Any]] | None:
    """Publish a private staging directory, accepting a valid concurrent winner."""
    try:
        staging_dir.rename(output_dir)
        return None
    except OSError as exc:
        exists_error = isinstance(exc, FileExistsError) or exc.errno == errno.EEXIST or getattr(exc, "winerror", None) == 183
        if not exists_error:
            raise
    # An old-version process may still be publishing without a build lock.
    # Prefer its complete output instead of deleting it or surfacing WinError
    # 183 to the input-check UI.
    winner = _read_valid_decode_cache(output_dir, cache_key)
    if winner is not None:
        return winner
    # The caller holds the SQLite publish fence and has established that the
    # final path is malformed, so this producer may clean it and retry once.
    if output_dir.exists() or output_dir.is_symlink():
        if output_dir.is_symlink():
            output_dir.unlink(missing_ok=True)
        elif output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    try:
        staging_dir.rename(output_dir)
    except OSError as exc:
        raise DecodeCacheBuildError(
            "decode cache publish conflicted with another builder; retry the input check"
        ) from exc
    return None


def _decode_video(
    video_path: Path,
    output_dir: Path,
    max_frames: int | None,
    decode_backend: str = "auto",
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[Path], float, list[float | None], dict[str, Any]]:
    requested = normalize_decode_backend(decode_backend)
    fallback_reason = None
    if requested in {"auto", "ffmpeg"}:
        try:
            frames, fps, timestamps = _decode_video_ffmpeg(video_path, output_dir, max_frames, progress_callback)
            return frames, fps, timestamps, {"backend": "ffmpeg", "fallback_reason": None}
        except Exception as exc:
            if requested == "ffmpeg":
                raise VideoDecodeBackendError(f"ffmpeg decode failed: {exc}") from exc
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
) -> tuple[list[Path], float, list[float | None]]:
    from vfieval.ffmpeg_exe import resolve_ffmpeg

    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not available (set VFIEVAL_VIDEO_FFMPEG or add ffmpeg to PATH)")
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "%06d.png"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "info",
        "-progress",
        "pipe:1",
        "-nostats",
        "-i",
        str(video_path),
        "-vsync",
        "0",
    ]
    if max_frames is not None:
        command.extend(["-frames:v", str(int(max_frames))])
    command.append(str(pattern))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    progress_state = {"frames": 0, "out_time_seconds": 0.0}
    progress_lines: list[str] = []

    def read_progress() -> None:
        assert process.stdout is not None
        last_reported = -1
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            progress_lines.append(line)
            key, separator, value = line.partition("=")
            if not separator:
                continue
            if key == "frame":
                try:
                    progress_state["frames"] = max(0, int(value.strip()))
                except ValueError:
                    continue
                if progress_callback and progress_state["frames"] != last_reported:
                    progress_callback(
                        {
                            "event": "frame",
                            "backend": "ffmpeg",
                            "frames": progress_state["frames"],
                        }
                    )
                    last_reported = progress_state["frames"]
            elif key in {"out_time_us", "out_time_ms"}:
                try:
                    progress_state["out_time_seconds"] = max(
                        progress_state["out_time_seconds"],
                        float(value.strip()) / 1_000_000.0,
                    )
                except ValueError:
                    continue

    progress_thread = Thread(
        target=read_progress,
        name=f"vfieval-ffmpeg-progress-{video_path.stem}",
        daemon=True,
    )
    progress_thread.start()
    stderr = process.stderr.read() if process.stderr is not None else ""
    returncode = process.wait()
    progress_thread.join(timeout=5)
    if process.stdout is not None:
        process.stdout.close()
    if process.stderr is not None:
        process.stderr.close()
    stdout = "\n".join(progress_lines)
    if returncode != 0:
        raise RuntimeError((stderr or stdout or "ffmpeg decode failed").strip())
    frames = sorted(output_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames: {video_path}")
    fps = _ffmpeg_reported_fps(stderr) or _probe_video_fps(video_path)
    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0
    probed_timestamps = _probe_video_frame_timestamps(video_path, max_frames=max_frames)
    timestamps: list[float | None]
    if probed_timestamps is not None and len(probed_timestamps) == len(frames):
        timestamps = list(probed_timestamps)
    else:
        # Frame index / FPS is not evidence of a real presentation timestamp,
        # especially for VFR sources. Preserve unavailability explicitly.
        timestamps = [None for _ in frames]
    if progress_callback:
        progress_callback({"event": "video_done", "backend": "ffmpeg", "frames": len(frames)})
    return frames, fps, timestamps


def _probe_video_frame_timestamps(
    video_path: Path,
    *,
    max_frames: int | None = None,
) -> list[float] | None:
    """Return one real ffprobe PTS per decoded video frame when available."""

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pts_time",
        "-of",
        "json",
    ]
    if max_frames is not None:
        # Allow a small reorder/lookahead margin for codecs with B-frames while
        # avoiding a full-file probe when inference intentionally truncates.
        command.extend(["-read_intervals", f"%+#{int(max_frames) + 32}"])
    command.append(str(video_path))
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return None
    rows = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return None
    if max_frames is not None:
        rows = rows[: int(max_frames)]
    timestamps: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        raw = row.get("best_effort_timestamp_time")
        if raw in {None, "", "N/A"}:
            raw = row.get("pts_time")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        timestamps.append(value)
    return timestamps or None


def _ffmpeg_reported_fps(stderr: str) -> float | None:
    """Read the input stream FPS from the decode process, never its elapsed time."""

    input_report = str(stderr or "").split("Output #0", 1)[0]
    for line in input_report.splitlines():
        if "Stream #" not in line or "Video:" not in line:
            continue
        match = re.search(r",\s*([0-9]+(?:\.[0-9]+)?)\s+fps(?:,|\s)", line)
        if match is None:
            match = re.search(r",\s*([0-9]+(?:\.[0-9]+)?)\s+tbr(?:,|\s)", line)
        if match is None:
            continue
        fps = float(match.group(1))
        if math.isfinite(fps) and fps > 0:
            return fps
    return None


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
    entry: "VideoEntry",
    decode_mode: str,
    video_index: int,
    sample_index: int,
    img0_index: int,
    img1_index: int,
    gt_index: int | None,
    fps: float,
    timestamps: list[float | None],
    cache_key: str,
) -> dict[str, Any]:
    video_path = entry.path
    metadata: dict[str, Any] = {
        "source_type": "video",
        "decode_mode": decode_mode,
        "video_index": video_index,
        "video_name": entry.display_name,
        "video_file": entry.video_file,
        "video_group": entry.group,
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
        metadata["evaluation_contract"] = MIDPOINT_TRIPLET_CONTRACT
    else:
        # video_pairs path: no GT frame exists, report frame position as img0.
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
