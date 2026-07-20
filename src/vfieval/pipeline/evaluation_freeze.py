from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


from vfieval.alignment import materialize_aligned_rgb


SOURCE_SLOTS = ("gt", "pred_a", "pred_b")
OUTPUT_SLOTS = SOURCE_SLOTS
OUTPUT_NAMES = {
    "gt": "reference",
    "pred_a": "method-a",
    "pred_b": "method-b",
}
FREEZE_PIPELINE_VERSION = "campaign-freeze-stream-v3"
FFMPEG_TIMEOUT_SECONDS = 600.0
FFMPEG_PIPE_EXIT_TIMEOUT_SECONDS = 2.0
TIMESTAMP_TOLERANCE_SECONDS = 1e-3
GOP_TARGET_INTERVAL_SECONDS = 1.0
REMUX_FIRST_KEYFRAME_TOLERANCE_SECONDS = 0.1
REMUX_MAX_KEYFRAME_INTERVAL_SECONDS = 2.0
_STREAMING_BACKEND_CACHE: dict[tuple[str, int, int], bool] = {}
_STREAMING_BACKEND_CACHE_LOCK = threading.Lock()

CancelCheck = Callable[[], Any]
ProgressCallback = Callable[[dict[str, Any]], None]


class FreezeError(RuntimeError):
    """Base class for Campaign package materialization failures."""


class FreezeBackendUnavailable(FreezeError):
    """Raised before frame streaming when FFmpeg/libx264 is unavailable."""


class FreezeCancelled(FreezeError):
    """Raised when a caller's cancellation callback requests cancellation."""


class SourceChanged(FreezeError):
    """Raised when a source changes while its package is being built."""


class RemuxError(FreezeError):
    """A compatibility failure which may safely fall back to pixel streaming."""


def _policy_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _video_stability_policy(fps: float) -> dict[str, Any]:
    """Return the deterministic browser-seek policy for a v3 frozen video."""

    resolved_fps = float(fps)
    if not math.isfinite(resolved_fps) or resolved_fps <= 0:
        raise ValueError("Campaign freeze fps must be a positive finite number")
    gop_frames = max(
        1,
        int(math.floor(resolved_fps * GOP_TARGET_INTERVAL_SECONDS + 0.5)),
    )
    payload: dict[str, Any] = {
        "version": "campaign-video-stability-v1",
        "encoding": {
            "codec": "h264",
            "encoder": "libx264",
            "crf": 18,
            "pixel_format": "yuv420p",
            "faststart": True,
            "gop_mode": "fixed_closed",
            "target_interval_seconds": GOP_TARGET_INTERVAL_SECONDS,
            "gop_frames": gop_frames,
            "keyint_min_frames": gop_frames,
            "scene_cut_threshold": 0,
            "open_gop": False,
        },
        "prediction_remux": {
            "first_keyframe_tolerance_seconds": REMUX_FIRST_KEYFRAME_TOLERANCE_SECONDS,
            "max_keyframe_interval_seconds": REMUX_MAX_KEYFRAME_INTERVAL_SECONDS,
            "paired": True,
        },
    }
    payload["fingerprint"] = _policy_fingerprint(payload)
    return payload


def _frame_sequence_stability_policy() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": "campaign-video-stability-v1",
        "encoding": {"applicable": False, "media_kind": "frame_sequence"},
        "prediction_remux": {"applicable": False},
    }
    payload["fingerprint"] = _policy_fingerprint(payload)
    return payload


def freeze_campaign_media(
    plan: Mapping[str, Any],
    sources: Mapping[str, Sequence[str | Path]],
    output_dir: str | Path,
    *,
    media_kind: str,
    fps: float | None,
    source_media: Mapping[str, str | Path] | None = None,
    source_timestamps: Mapping[str, Sequence[float | None]] | None = None,
    expected_source_sha256: Mapping[str, str | None] | None = None,
    source_signatures: Mapping[str, Mapping[str, Any]] | None = None,
    cancel_check: CancelCheck | None = None,
    progress_callback: ProgressCallback | None = None,
    ffmpeg: str | Path | None = None,
    ffprobe: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze one aligned GT/A/B item into private media.

    Video frames are read once per logical source frame and sent directly to
    bounded rawvideo sinks.  Eligible source MP4s may instead be privately
    remuxed, but A and B always take that path as a pair. Frame sequences are
    written directly to their three final PNG directories with no intermediate
    sequence. Diff media was never consumed by Campaign playback and is no
    longer materialized in stream-v2 and later packages.
    """

    started = time.monotonic()
    timings: dict[str, float] = {}
    normalized_sources = _validate_freeze_inputs(plan, sources)
    target = plan.get("target") or {}
    width = int(target.get("width") or 0)
    height = int(target.get("height") or 0)
    temporal = plan.get("temporal") or {}
    frame_count = int(temporal.get("frame_count") or len(normalized_sources["gt"]))
    resolved_fps = float(fps if fps not in {None, 0, 0.0} else temporal.get("fps") or 24.0)
    if not math.isfinite(resolved_fps) or resolved_fps <= 0:
        raise ValueError("Campaign freeze fps must be a positive finite number")

    target_dir = Path(output_dir).resolve()
    if target_dir.exists() and not target_dir.is_dir():
        raise FileExistsError(f"Campaign freeze output is not a directory: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    if media_kind == "frame_sequence":
        result = _freeze_frame_sequence(
            plan,
            normalized_sources,
            target_dir,
            frame_count=frame_count,
            fps=resolved_fps,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )
        result["timings"]["total"] = time.monotonic() - started
        return result
    if media_kind != "video":
        raise ValueError(f"unsupported Campaign freeze media_kind: {media_kind}")
    stability_policy = _video_stability_policy(resolved_fps)
    gop_policy = dict(stability_policy["encoding"])
    if width % 2 or height % 2:
        raise ValueError(
            "browser-compatible H.264/yuv420p requires even dimensions; "
            f"requested {width}x{height}. Canonical dimensions will not be padded."
        )

    ffmpeg_path = _resolve_executable("ffmpeg", ffmpeg)
    ffprobe_path = _resolve_executable("ffprobe", ffprobe)
    backend_started = time.monotonic()
    if not streaming_backend_available(ffmpeg_path, cancel_check=cancel_check):
        raise FreezeBackendUnavailable("Campaign streaming freeze requires FFmpeg with libx264")
    if ffprobe_path is None:
        raise FreezeBackendUnavailable("Campaign streaming freeze requires ffprobe")
    timings["backend_check"] = time.monotonic() - backend_started

    _check_cancel(cancel_check)
    _progress(
        progress_callback,
        stage="probing",
        frame_current=0,
        frame_total=frame_count,
        timings=dict(timings),
        force=True,
    )
    probe_started = time.monotonic()
    eligibility = _collect_remux_eligibility(
        plan,
        source_media or {},
        source_timestamps or {},
        expected_source_sha256=expected_source_sha256 or {},
        fps=resolved_fps,
        ffprobe=ffprobe_path,
        cancel_check=cancel_check,
    )
    timings["probe"] = time.monotonic() - probe_started

    paths = {slot: target_dir / f"{OUTPUT_NAMES[slot]}.mp4" for slot in OUTPUT_SLOTS}
    remuxed: set[str] = set()
    signatures: dict[str, dict[str, Any]] = {}
    remux_started = time.monotonic()
    _progress(
        progress_callback,
        stage="remuxing",
        frame_current=0,
        frame_total=frame_count,
        timings=dict(timings),
        force=True,
    )
    try:
        if eligibility["gt"]["eligible"]:
            signature = _initial_source_signature(
                "gt",
                Path(str(source_media["gt"])).resolve(),
                source_signatures,
                cancel_check=cancel_check,
            )
            _validate_expected_source_digest("gt", signature, expected_source_sha256)
            try:
                remux_video(
                    source_media["gt"],
                    paths["gt"],
                    width=width,
                    height=height,
                    frame_count=frame_count,
                    fps=resolved_fps,
                    ffmpeg=ffmpeg_path,
                    ffprobe=ffprobe_path,
                    stability_policy=stability_policy,
                    cancel_check=cancel_check,
                )
            except RemuxError as exc:
                _assert_source_signature_unchanged(
                    "gt", signature, cancel_check=cancel_check
                )
                eligibility["gt"]["fallback_reason"] = str(exc)
                paths["gt"].unlink(missing_ok=True)
            else:
                signatures["gt"] = signature
                remuxed.add("gt")

        pred_pair_eligible = all(eligibility[slot]["eligible"] for slot in ("pred_a", "pred_b"))
        if pred_pair_eligible:
            pair_signatures = {
                slot: _initial_source_signature(
                    slot,
                    Path(str(source_media[slot])).resolve(),
                    source_signatures,
                    cancel_check=cancel_check,
                )
                for slot in ("pred_a", "pred_b")
            }
            for slot, signature in pair_signatures.items():
                _validate_expected_source_digest(slot, signature, expected_source_sha256)
            try:
                for slot in ("pred_a", "pred_b"):
                    remux_video(
                        source_media[slot],
                        paths[slot],
                        width=width,
                        height=height,
                        frame_count=frame_count,
                        fps=resolved_fps,
                        ffmpeg=ffmpeg_path,
                        ffprobe=ffprobe_path,
                        stability_policy=stability_policy,
                        cancel_check=cancel_check,
                    )
            except RemuxError as exc:
                for slot, signature in pair_signatures.items():
                    _assert_source_signature_unchanged(
                        slot, signature, cancel_check=cancel_check
                    )
                for slot in ("pred_a", "pred_b"):
                    paths[slot].unlink(missing_ok=True)
                    eligibility[slot]["fallback_reason"] = str(exc)
            else:
                signatures.update(pair_signatures)
                remuxed.update(("pred_a", "pred_b"))
    except Exception:
        _remove_outputs(paths.values())
        raise
    timings["remux"] = time.monotonic() - remux_started

    stream_slots = [slot for slot in OUTPUT_SLOTS if slot not in remuxed]
    cpu_count = max(1, int(os.cpu_count() or 1))
    thread_count = (
        max(1, max(1, cpu_count - 1) // len(stream_slots)) if stream_slots else 0
    )
    pipeline = "remux" if not stream_slots else "remux+stream" if remuxed else "streaming"
    sinks: dict[str, _RawVideoSink] = {}
    sink_failures = _SinkFailureState()
    materialize_started = time.monotonic()
    _progress(
        progress_callback,
        stage="materializing",
        frame_current=0,
        frame_total=frame_count,
        pipeline=pipeline,
        timings=dict(timings),
        force=True,
    )
    try:
        for slot in stream_slots:
            _check_cancel(cancel_check)
            sink = _RawVideoSink(
                paths[slot],
                width=width,
                height=height,
                fps=resolved_fps,
                threads=thread_count,
                ffmpeg=ffmpeg_path,
                gop_policy=gop_policy,
                cancel_check=cancel_check,
                failure_state=sink_failures,
            )
            try:
                sink.start()
            except (FileNotFoundError, OSError) as exc:
                sink.abort()
                if not sinks:
                    raise FreezeBackendUnavailable(
                        f"failed to start FFmpeg/libx264: {exc}"
                    ) from exc
                raise FreezeError(
                    f"FFmpeg/libx264 failed after Campaign streaming had started: {exc}"
                ) from exc
            except Exception:
                sink.abort()
                raise
            sinks[slot] = sink

        for index in range(frame_count):
            _check_cancel(cancel_check)
            images = {
                slot: materialize_aligned_rgb(plan, slot, normalized_sources[slot][index])
                for slot in stream_slots
            }
            try:
                for slot, sink in sinks.items():
                    sink.write(images[slot].tobytes())
            finally:
                for image in images.values():
                    image.close()
            _progress(
                progress_callback,
                stage="materializing",
                frame_current=index + 1,
                frame_total=frame_count,
                pipeline=pipeline,
                timings=dict(timings),
                force=(index + 1 == frame_count),
            )

        for sink in sinks.values():
            sink.close_input()
        for sink in sinks.values():
            sink.wait()
    except Exception:
        for sink in sinks.values():
            sink.abort()
        _remove_outputs(paths.values())
        raise
    timings["materialize_encode"] = time.monotonic() - materialize_started

    validation_started = time.monotonic()
    _progress(
        progress_callback,
        stage="validating",
        frame_current=frame_count,
        frame_total=frame_count,
        pipeline=pipeline,
        timings=dict(timings),
        force=True,
    )
    output_probes: dict[str, dict[str, Any]] = {}
    try:
        for slot in stream_slots:
            output_probes[slot] = validate_frozen_video(
                paths[slot],
                width=width,
                height=height,
                frame_count=frame_count,
                fps=resolved_fps,
                ffprobe=ffprobe_path,
                stability_policy=stability_policy,
                cancel_check=cancel_check,
            )
    except Exception:
        _remove_outputs(paths.values())
        raise
    timings["output_validate"] = time.monotonic() - validation_started

    source_stability_started = time.monotonic()
    try:
        for slot, signature in signatures.items():
            _assert_source_signature_unchanged(
                slot, signature, cancel_check=cancel_check
            )
    except Exception:
        _remove_outputs(paths.values())
        raise
    timings["source_stability"] = time.monotonic() - source_stability_started

    hash_started = time.monotonic()
    _progress(
        progress_callback,
        stage="hashing",
        frame_current=frame_count,
        frame_total=frame_count,
        pipeline=pipeline,
        timings=dict(timings),
        force=True,
    )
    try:
        artifacts = {
            slot: _artifact_result(
                paths[slot],
                "remux" if slot in remuxed else "stream",
                cancel_check=cancel_check,
            )
            for slot in OUTPUT_SLOTS
        }
    except Exception:
        _remove_outputs(paths.values())
        raise
    timings["hash"] = time.monotonic() - hash_started
    timings["total"] = time.monotonic() - started
    keyframe_probe = {
        slot: {
            **_keyframe_probe_summary(
                output_probes.get(slot)
                or dict((eligibility.get(slot) or {}).get("probe") or {})
            ),
            "basis": "frozen_output" if slot in output_probes else "remux_source",
        }
        for slot in OUTPUT_SLOTS
    }
    _progress(
        progress_callback,
        stage="completed",
        frame_current=frame_count,
        frame_total=frame_count,
        pipeline=pipeline,
        timings=dict(timings),
        force=True,
    )
    return {
        "version": FREEZE_PIPELINE_VERSION,
        "pipeline": pipeline,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": resolved_fps,
        "encoder_threads": thread_count,
        "artifacts": artifacts,
        "timings": timings,
        "remux": eligibility,
        "stream_slots": stream_slots,
        "remuxed_slots": sorted(remuxed),
        "gop_policy": gop_policy,
        "keyframe_probe": keyframe_probe,
        "stability_policy": stability_policy,
        "stability_policy_fingerprint": str(stability_policy["fingerprint"]),
    }


def _freeze_frame_sequence(
    plan: Mapping[str, Any],
    sources: Mapping[str, list[Path]],
    output_dir: Path,
    *,
    frame_count: int,
    fps: float,
    cancel_check: CancelCheck | None,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    started = time.monotonic()
    timings: dict[str, float] = {}
    target = plan.get("target") or {}
    width = int(target.get("width") or 0)
    height = int(target.get("height") or 0)
    source_signatures: dict[Path, tuple[str, dict[str, Any]]] = {}
    for slot, frame_paths in sources.items():
        for index, frame_path in enumerate(frame_paths):
            resolved = frame_path.resolve()
            if resolved in source_signatures:
                continue
            source_signatures[resolved] = (
                f"{slot} frame {index}",
                _source_signature(resolved, cancel_check=cancel_check),
            )
    timings["source_validate"] = time.monotonic() - started
    materialize_started = time.monotonic()
    paths = {slot: output_dir / OUTPUT_NAMES[slot] for slot in OUTPUT_SLOTS}
    for path in paths.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Campaign package target already exists: {path}")
        path.mkdir(parents=False, exist_ok=False)
    _progress(
        progress_callback,
        stage="materializing",
        frame_current=0,
        frame_total=frame_count,
        pipeline="png_sequence",
        timings=dict(timings),
        force=True,
    )
    try:
        for index in range(frame_count):
            _check_cancel(cancel_check)
            images = {
                slot: materialize_aligned_rgb(plan, slot, sources[slot][index])
                for slot in SOURCE_SLOTS
            }
            try:
                for slot, image in images.items():
                    image.save(paths[slot] / f"{index:06d}.png", format="PNG")
            finally:
                for image in images.values():
                    image.close()
            _progress(
                progress_callback,
                stage="materializing",
                frame_current=index + 1,
                frame_total=frame_count,
                pipeline="png_sequence",
                timings=dict(timings),
                force=(index + 1 == frame_count),
            )
        for label, signature in source_signatures.values():
            _assert_source_signature_unchanged(
                label,
                signature,
                cancel_check=cancel_check,
            )
    except Exception:
        _remove_outputs(paths.values())
        raise
    timings["materialize"] = time.monotonic() - materialize_started
    hash_started = time.monotonic()
    _progress(
        progress_callback,
        stage="hashing",
        frame_current=frame_count,
        frame_total=frame_count,
        pipeline="png_sequence",
        timings=dict(timings),
        force=True,
    )
    try:
        artifacts = {
            slot: _artifact_result(path, "png_sequence", cancel_check=cancel_check)
            for slot, path in paths.items()
        }
        for label, signature in source_signatures.values():
            _assert_source_stat_unchanged(label, signature)
    except Exception:
        _remove_outputs(paths.values())
        raise
    timings["hash"] = time.monotonic() - hash_started
    timings["total"] = time.monotonic() - started
    _progress(
        progress_callback,
        stage="completed",
        frame_current=frame_count,
        frame_total=frame_count,
        pipeline="png_sequence",
        timings=dict(timings),
        force=True,
    )
    stability_policy = _frame_sequence_stability_policy()
    return {
        "version": FREEZE_PIPELINE_VERSION,
        "pipeline": "png_sequence",
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "encoder_threads": 0,
        "artifacts": artifacts,
        "timings": timings,
        "remux": {slot: {"eligible": False, "reasons": ["frame_sequence"]} for slot in SOURCE_SLOTS},
        "stream_slots": list(SOURCE_SLOTS),
        "remuxed_slots": [],
        "gop_policy": dict(stability_policy["encoding"]),
        "keyframe_probe": {},
        "stability_policy": stability_policy,
        "stability_policy_fingerprint": str(stability_policy["fingerprint"]),
    }


def _keyframe_probe_summary(probe: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a bounded, manifest-safe summary of an ffprobe keyframe scan."""

    payload = dict(probe or {})
    keyframes = _finite_timestamps(payload.get("keyframe_timestamps"))
    timestamps = _finite_timestamps(payload.get("timestamps"))
    fps = float(payload.get("fps") or 0.0)
    frame_count = _int_or_none(payload.get("frame_count"))
    probe_mode = str(payload.get("keyframe_probe_mode") or "full_frames")
    first_frame = _float_or_none(payload.get("first_frame_timestamp"))
    if first_frame is None and timestamps:
        first_frame = float(timestamps[0])
    duration = _float_or_none(payload.get("duration_seconds"))
    full_timeline_required = probe_mode != "keyframes_only"
    complete = (
        bool(keyframes)
        and first_frame is not None
        and payload.get("keyframe_probe_complete") is not False
        and (
            not full_timeline_required
            or (
                bool(timestamps)
                and (frame_count in {None, 0} or len(timestamps or []) == frame_count)
            )
        )
    )
    if (timestamps is not None and any(
        right < left for left, right in zip(timestamps, timestamps[1:])
    )) or (keyframes is not None and any(
        right < left for left, right in zip(keyframes, keyframes[1:])
    )):
        complete = False
    first_keyframe = float(keyframes[0]) if keyframes else None
    first_offset = (
        float(first_keyframe) - float(first_frame)
        if first_keyframe is not None and first_frame is not None
        else None
    )
    maximum_interval: float | None = None
    if complete and keyframes is not None and first_frame is not None:
        intervals = [
            max(0.0, float(right) - float(left))
            for left, right in zip(keyframes, keyframes[1:])
        ]
        intervals.append(max(0.0, float(keyframes[0]) - float(first_frame)))
        timeline_end: float | None = None
        if timestamps:
            durations = _finite_timestamps(payload.get("frame_durations")) or []
            final_duration = (
                float(durations[-1])
                if durations and float(durations[-1]) > 0
                else (1.0 / fps if fps > 0 else 0.0)
            )
            timeline_end = float(timestamps[-1]) + final_duration
        elif duration is not None and duration > 0:
            timeline_end = float(first_frame) + float(duration)
        if timeline_end is None or timeline_end + TIMESTAMP_TOLERANCE_SECONDS < keyframes[-1]:
            complete = False
        else:
            intervals.append(max(0.0, timeline_end - float(keyframes[-1])))
            maximum_interval = max(intervals) if intervals else 0.0
    fingerprint_payload = {
        "first_frame": round(float(first_frame), 9) if first_frame is not None else None,
        "duration": round(float(duration), 9) if duration is not None else None,
        "keyframes": [round(float(value), 9) for value in (keyframes or [])],
    }
    return {
        "complete": bool(complete),
        "count": len(keyframes or []),
        "first_frame_seconds": first_frame,
        "first_keyframe_seconds": first_keyframe,
        "first_keyframe_offset_seconds": first_offset,
        # Compatibility alias for the v3 preview implementation.
        "first_seconds": first_keyframe,
        "max_interval_seconds": maximum_interval,
        "timestamps_sha256": _policy_fingerprint(fingerprint_payload),
        "probe_mode": probe_mode,
    }


def _prediction_keyframe_reasons(
    summary: Mapping[str, Any],
    stability_policy: Mapping[str, Any],
) -> list[str]:
    policy = dict(stability_policy.get("prediction_remux") or {})
    if not bool(summary.get("complete")):
        return ["keyframe timestamps are unavailable or incomplete"]
    reasons: list[str] = []
    first_offset = _float_or_none(summary.get("first_keyframe_offset_seconds"))
    if first_offset is None:
        first = _float_or_none(summary.get("first_keyframe_seconds"))
        first_frame = _float_or_none(summary.get("first_frame_seconds"))
        if first is not None and first_frame is not None:
            first_offset = first - first_frame
    first_tolerance = float(
        policy.get("first_keyframe_tolerance_seconds")
        or REMUX_FIRST_KEYFRAME_TOLERANCE_SECONDS
    )
    if (
        first_offset is None
        or abs(first_offset) > first_tolerance + TIMESTAMP_TOLERANCE_SECONDS
    ):
        reasons.append("first keyframe is not near the first frame")
    maximum = _float_or_none(summary.get("max_interval_seconds"))
    maximum_allowed = float(
        policy.get("max_keyframe_interval_seconds")
        or REMUX_MAX_KEYFRAME_INTERVAL_SECONDS
    )
    if maximum is None or maximum > maximum_allowed + TIMESTAMP_TOLERANCE_SECONDS:
        reasons.append(
            f"maximum keyframe interval exceeds {maximum_allowed:.3f} seconds"
        )
    return reasons


def remux_eligibility(
    plan: Mapping[str, Any],
    slot: str,
    source_path: str | Path,
    *,
    timestamps: Sequence[float | None] | None,
    fps: float | None,
    ffprobe: str | Path | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """Return strict, diagnostic remux eligibility for one source slot."""

    reasons: list[str] = []
    path = Path(source_path).resolve()
    if Path(source_path).is_symlink():
        reasons.append("symlink sources are not eligible")
    if not path.is_file():
        return {"eligible": False, "reasons": ["source video is unavailable"], "probe": None}
    report = (plan.get("sources") or {}).get(slot)
    if not isinstance(report, Mapping):
        return {"eligible": False, "reasons": [f"Alignment Plan has no {slot} source"], "probe": None}
    target = plan.get("target") or {}
    width = int(target.get("width") or 0)
    height = int(target.get("height") or 0)
    original = report.get("original") or {}
    original_size = (int(original.get("width") or 0), int(original.get("height") or 0))
    if original_size != (width, height):
        reasons.append("source requires spatial normalization")
    if width % 2 or height % 2:
        reasons.append("target dimensions are not yuv420p-compatible")

    temporal = plan.get("temporal") or {}
    frame_count = int(temporal.get("frame_count") or 0)
    temporal_mode = str(temporal.get("mode") or "")
    if slot == "gt":
        if temporal_mode != "exact":
            reasons.append("GT requires indexed temporal materialization")
        if int(temporal.get("reference_frame_count") or 0) != frame_count:
            reasons.append("reference frame count is not full identity")
        if int(temporal.get("mapping_count") or 0) != frame_count:
            reasons.append("temporal mapping count is not full identity")
        if frame_count > 0 and (
            int(temporal.get("mapping_first") or 0) != 0
            or int(
                temporal.get("mapping_last")
                if temporal.get("mapping_last") is not None
                else -1
            )
            != frame_count - 1
        ):
            reasons.append("temporal mapping endpoints are not full identity")
    else:
        if temporal_mode not in {"exact", "indexed"}:
            reasons.append("prediction temporal mapping is unsupported")
        prediction_counts = list(temporal.get("prediction_frame_counts") or [])
        prediction_index = 0 if slot == "pred_a" else 1
        if (
            prediction_index >= len(prediction_counts)
            or int(prediction_counts[prediction_index] or 0) != frame_count
        ):
            reasons.append("prediction source is not the complete aligned frame sequence")
        if int(temporal.get("mapping_count") or 0) != frame_count:
            reasons.append("prediction temporal mapping count is incomplete")

    expected_timestamps = _finite_timestamps(timestamps)
    if expected_timestamps is None or len(expected_timestamps) != frame_count:
        reasons.append("source timestamps are unavailable or incomplete")
    # Do not enumerate every encoded frame when the immutable plan alone has
    # already disqualified this source (resize, indexed mapping, odd target,
    # missing timestamp diagnostics, or a symlink).
    if reasons:
        return {"eligible": False, "reasons": reasons, "probe": None}

    try:
        # One timestamp scan is sufficient for codec/container diagnostics,
        # exact frame count, CFR, and source timestamp validation.  Avoid a
        # second count_packets scan on every remux candidate.
        probe = probe_video_for_freeze(
            path,
            ffprobe=ffprobe,
            include_timestamps=True,
            cancel_check=cancel_check,
        )
    except FreezeCancelled:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        return {"eligible": False, "reasons": [*reasons, f"ffprobe failed: {exc}"], "probe": None}
    policy_fps = float(
        fps if fps not in {None, 0, 0.0} else temporal.get("fps") or 0.0
    )
    stability_policy = _video_stability_policy(policy_fps if policy_fps > 0 else 1.0)
    keyframe_probe = _keyframe_probe_summary(probe)
    reasons.extend(_prediction_keyframe_reasons(keyframe_probe, stability_policy))
    if str(probe.get("codec") or "").lower() != "h264":
        reasons.append("codec is not H.264")
    if str(probe.get("pix_fmt") or "").lower() != "yuv420p":
        reasons.append("pixel format is not yuv420p")
    if not _zero_rotation(float(probe.get("rotation_degrees") or 0.0)):
        reasons.append("rotation metadata is not zero")
    if (int(probe.get("width") or 0), int(probe.get("height") or 0)) != (width, height):
        reasons.append("display dimensions do not match the Alignment Plan target")
    if int(probe.get("frame_count") or 0) != frame_count:
        reasons.append("container frame count does not match the Alignment Plan")
    expected_fps = float(fps if fps not in {None, 0, 0.0} else temporal.get("fps") or 0.0)
    if expected_fps <= 0 or abs(float(probe.get("fps") or 0.0) - expected_fps) > 1e-6:
        reasons.append("fps does not match the Alignment Plan")
    if not bool(probe.get("cfr")):
        reasons.append("source video is not CFR")
    observed_timestamps = _finite_timestamps(probe.get("timestamps"))
    if observed_timestamps is None or len(observed_timestamps) != frame_count:
        reasons.append("ffprobe timestamps are unavailable or incomplete")
    elif not _relative_timelines_match(expected_timestamps, observed_timestamps):
        reasons.append("source timestamps changed after alignment")
    keyframe_probe["remux_gate_applied"] = True
    keyframe_probe["policy_compliant"] = not _prediction_keyframe_reasons(
        keyframe_probe,
        stability_policy,
    )
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "probe": probe,
        "keyframe_probe": keyframe_probe,
    }


def probe_video_for_freeze(
    path: str | Path,
    *,
    ffprobe: str | Path | None = None,
    include_timestamps: bool = True,
    include_keyframes: bool | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """Probe stream-copy eligibility and final package invariants with ffprobe."""

    source = Path(path).resolve()
    executable = _resolve_executable("ffprobe", ffprobe)
    if executable is None:
        raise RuntimeError("ffprobe is unavailable")
    scan_keyframes = (
        bool(include_timestamps)
        if include_keyframes is None
        else bool(include_keyframes)
    )
    metadata_command = [
        executable,
        "-v",
        "error",
        *([] if include_timestamps else ["-count_packets"]),
        "-show_entries",
        (
            "stream=index,codec_type,codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,"
            "nb_frames,nb_read_packets,start_time,duration:stream_tags=rotate:"
            "stream_side_data=side_data_type,rotation:format=start_time,duration"
        ),
        "-of",
        "json",
        str(source),
    ]
    payload = _run_json(
        metadata_command,
        "ffprobe metadata",
        cancel_check=cancel_check,
    )
    streams = payload.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if not isinstance(video, Mapping):
        raise ValueError(f"video stream not found: {source}")
    timestamps: list[float] = []
    durations: list[float] = []
    keyframe_timestamps: list[float] = []
    if include_timestamps or scan_keyframes:
        keyframes_only = bool(scan_keyframes and not include_timestamps)
        frame_payload = _run_json(
            [
                executable,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                *(["-skip_frame", "nokey"] if keyframes_only else []),
                "-show_frames",
                "-show_entries",
                (
                    "frame=best_effort_timestamp_time,pts_time,pkt_duration_time,"
                    "duration_time,key_frame,pict_type"
                ),
                "-of",
                "json",
                str(source),
            ],
            "ffprobe frame timestamps",
            cancel_check=cancel_check,
        )
        for frame in frame_payload.get("frames") or []:
            timestamp = _float_or_none(
                frame.get("best_effort_timestamp_time")
                if frame.get("best_effort_timestamp_time") is not None
                else frame.get("pts_time")
            )
            if timestamp is not None:
                if include_timestamps:
                    timestamps.append(timestamp)
                try:
                    is_keyframe = int(frame.get("key_frame") or 0) == 1
                except (TypeError, ValueError):
                    is_keyframe = False
                if scan_keyframes and is_keyframe:
                    keyframe_timestamps.append(timestamp)
            if include_timestamps:
                duration = _float_or_none(
                    frame.get("pkt_duration_time")
                    if frame.get("pkt_duration_time") is not None
                    else frame.get("duration_time")
                )
                if duration is not None:
                    durations.append(duration)
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate")) or 0.0
    nominal_fps = _parse_rate(video.get("r_frame_rate")) or fps
    packet_count = _int_or_none(video.get("nb_read_packets"))
    declared_frame_count = _int_or_none(video.get("nb_frames"))
    frame_count = int(
        packet_count
        if packet_count is not None
        else declared_frame_count
        if declared_frame_count is not None
        else len(timestamps)
    )
    coded_width = int(video.get("width") or 0)
    coded_height = int(video.get("height") or 0)
    rotation = _stream_rotation(video)
    width, height = coded_width, coded_height
    if _rotation_swaps_dimensions(rotation):
        width, height = coded_height, coded_width
    format_payload = payload.get("format") or {}
    first_frame_timestamp = (
        float(timestamps[0])
        if timestamps
        else _float_or_none(video.get("start_time"))
    )
    if first_frame_timestamp is None and isinstance(format_payload, Mapping):
        first_frame_timestamp = _float_or_none(format_payload.get("start_time"))
    duration_seconds = _float_or_none(video.get("duration"))
    if duration_seconds is None and isinstance(format_payload, Mapping):
        duration_seconds = _float_or_none(format_payload.get("duration"))
    cfr = _timestamps_are_cfr(timestamps, durations, fps)
    if fps <= 0 or nominal_fps <= 0 or abs(fps - nominal_fps) > 1e-6:
        cfr = False
    return {
        "path": source,
        "codec": video.get("codec_name"),
        "pix_fmt": video.get("pix_fmt"),
        "coded_width": coded_width,
        "coded_height": coded_height,
        "width": width,
        "height": height,
        "rotation_degrees": rotation,
        "frame_count": frame_count,
        "packet_count": packet_count,
        "declared_frame_count": declared_frame_count,
        "fps": fps,
        "timestamps": timestamps,
        "frame_durations": durations,
        "first_frame_timestamp": first_frame_timestamp,
        "duration_seconds": duration_seconds,
        "keyframe_timestamps": keyframe_timestamps,
        "keyframe_probe_complete": bool(scan_keyframes),
        "keyframe_probe_mode": (
            "full_frames"
            if include_timestamps
            else "keyframes_only"
            if scan_keyframes
            else "none"
        ),
        "cfr": cfr,
        "audio_stream_count": sum(1 for item in streams if item.get("codec_type") == "audio"),
        "stream_count": len(streams),
    }


def remux_video(
    source: str | Path,
    target: str | Path,
    *,
    width: int,
    height: int,
    frame_count: int,
    fps: float,
    ffmpeg: str | Path | None = None,
    ffprobe: str | Path | None = None,
    stability_policy: Mapping[str, Any] | None = None,
    cancel_check: CancelCheck | None = None,
) -> Path:
    """Create and validate a private, video-only faststart stream-copy."""

    source_path = Path(source).resolve()
    target_path = Path(target).resolve()
    if Path(source).is_symlink():
        raise RemuxError("Campaign freeze refuses to remux symlink media")
    if target_path.exists() or target_path.is_symlink():
        raise FileExistsError(f"Campaign package target already exists: {target_path}")
    executable = _resolve_executable("ffmpeg", ffmpeg)
    if executable is None:
        raise RemuxError("ffmpeg is unavailable")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "-y",
        "-v",
        "error",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-metadata:s:v:0",
        "rotate=0",
        "-c:v",
        "copy",
        "-movflags",
        "+faststart",
        str(target_path),
    ]
    succeeded = False
    process: subprocess.Popen[bytes] | None = None
    stderr_handle = tempfile.TemporaryFile(mode="w+b")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_handle,
        )
        deadline = time.monotonic() + FFMPEG_TIMEOUT_SECONDS
        while True:
            _check_cancel(cancel_check)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RemuxError("ffmpeg stream-copy timed out")
            try:
                returncode = process.wait(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        stderr_handle.flush()
        stderr_handle.seek(0)
        stderr_text = stderr_handle.read().decode("utf-8", errors="replace")
        if returncode != 0:
            raise RemuxError(
                "ffmpeg stream-copy failed"
                + (f": {_error_detail(stderr_text)}" if stderr_text else "")
            )
        validate_frozen_video(
            target_path,
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps,
            ffprobe=ffprobe,
            stability_policy=stability_policy,
            cancel_check=cancel_check,
        )
        _check_cancel(cancel_check)
        succeeded = True
    except RemuxError:
        raise
    except Exception as exc:
        if isinstance(exc, FreezeCancelled):
            raise
        raise RemuxError(f"remux validation failed: {exc}") from exc
    finally:
        if process is not None:
            _terminate_process(process)
        stderr_handle.close()
        if not succeeded:
            target_path.unlink(missing_ok=True)
    return target_path


def validate_frozen_video(
    path: str | Path,
    *,
    width: int,
    height: int,
    frame_count: int,
    fps: float,
    ffprobe: str | Path | None = None,
    stability_policy: Mapping[str, Any] | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """Validate frozen MP4 structure without decoding all pixels again."""

    target = Path(path).resolve()
    probe = probe_video_for_freeze(
        target,
        ffprobe=ffprobe,
        include_timestamps=False,
        include_keyframes=stability_policy is not None,
        cancel_check=cancel_check,
    )
    errors: list[str] = []
    if str(probe.get("codec") or "").lower() != "h264":
        errors.append("codec is not H.264")
    if str(probe.get("pix_fmt") or "").lower() != "yuv420p":
        errors.append("pixel format is not yuv420p")
    if not _zero_rotation(float(probe.get("rotation_degrees") or 0.0)):
        errors.append("rotation metadata is not zero")
    if (int(probe.get("width") or 0), int(probe.get("height") or 0)) != (int(width), int(height)):
        errors.append(
            f"display dimensions changed: expected {width}x{height}, "
            f"got {probe.get('width')}x{probe.get('height')}"
        )
    packet_count = probe.get("packet_count")
    declared_frame_count = probe.get("declared_frame_count")
    if packet_count is not None and int(packet_count) != int(frame_count):
        errors.append(f"packet count changed: expected {frame_count}, got {int(packet_count)}")
    if declared_frame_count is not None and int(declared_frame_count) != int(frame_count):
        errors.append(
            f"declared frame count changed: expected {frame_count}, got {int(declared_frame_count)}"
        )
    if packet_count is None and declared_frame_count is None:
        if int(probe.get("frame_count") or 0) != int(frame_count):
            errors.append(
                f"decoded frame count changed: expected {frame_count}, "
                f"got {int(probe.get('frame_count') or 0)}"
            )
    if abs(float(probe.get("fps") or 0.0) - float(fps)) > 1e-6:
        errors.append(f"fps changed: expected {fps}, got {probe.get('fps')}")
    if int(probe.get("audio_stream_count") or 0) != 0:
        errors.append("frozen package contains an audio stream")
    if not _mp4_has_faststart(target):
        errors.append("MP4 moov atom is not before media data")
    if stability_policy is not None:
        keyframe_probe = _keyframe_probe_summary(probe)
        keyframe_errors = _prediction_keyframe_reasons(
            keyframe_probe,
            stability_policy,
        )
        errors.extend(f"frozen GOP {reason}" for reason in keyframe_errors)
        probe["keyframe_probe"] = {
            **keyframe_probe,
            "policy_compliant": not keyframe_errors,
        }
    if errors:
        raise ValueError("invalid Campaign frozen video: " + "; ".join(errors))
    return probe


def streaming_backend_available(
    ffmpeg: str | Path | None = None,
    *,
    cancel_check: CancelCheck | None = None,
) -> bool:
    executable = _resolve_executable("ffmpeg", ffmpeg)
    if executable is None:
        return False
    _check_cancel(cancel_check)
    try:
        stat = Path(executable).stat()
        cache_key = (str(Path(executable).resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    except OSError:
        return False
    with _STREAMING_BACKEND_CACHE_LOCK:
        cached = _STREAMING_BACKEND_CACHE.get(cache_key)
        if cached is not None:
            _check_cancel(cancel_check)
            return bool(cached)
        available = _probe_streaming_backend(executable, cancel_check=cancel_check)
        _STREAMING_BACKEND_CACHE[cache_key] = bool(available)
        return bool(available)


def _probe_streaming_backend(
    executable: str,
    *,
    cancel_check: CancelCheck | None = None,
) -> bool:
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [executable, "-hide_banner", "-loglevel", "error", "-h", "encoder=libx264"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 10.0
        while True:
            _check_cancel(cancel_check)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                return process.wait(timeout=min(0.1, remaining)) == 0
            except subprocess.TimeoutExpired:
                continue
    except OSError:
        return False
    finally:
        if process is not None:
            _terminate_process(process)


def _clear_streaming_backend_cache() -> None:
    """Clear process-local capability state for focused tests."""

    with _STREAMING_BACKEND_CACHE_LOCK:
        _STREAMING_BACKEND_CACHE.clear()


class _SinkFailureState:
    """Share the first encoder failure across all bounded sinks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failure: tuple[str, BaseException] | None = None

    def record(self, sink_name: str, error: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = (str(sink_name), error)

    def raise_if_failed(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            sink_name, error = failure
            raise RuntimeError(f"Campaign encoder {sink_name} failed: {error}") from error


class _RawVideoSink:
    """One bounded raw RGB producer/FFmpeg consumer pair."""

    _SENTINEL = object()

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        threads: int,
        ffmpeg: str,
        gop_policy: Mapping[str, Any] | None = None,
        cancel_check: CancelCheck | None = None,
        failure_state: _SinkFailureState | None = None,
    ) -> None:
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.threads = max(1, int(threads))
        self.ffmpeg = str(ffmpeg)
        self.gop_policy = dict(gop_policy or _video_stability_policy(self.fps)["encoding"])
        self.cancel_check = cancel_check
        self.failure_state = failure_state or _SinkFailureState()
        self._queue: queue.Queue[bytes | object] = queue.Queue(maxsize=1)
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._closing = False
        self._aborted = threading.Event()
        self._stderr = tempfile.TemporaryFile(mode="w+b")

    def start(self) -> None:
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(f"Campaign package target already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = self.command()
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
        )
        self._thread = threading.Thread(
            target=self._writer,
            name=f"campaign-freeze-{self.path.stem}",
            daemon=True,
        )
        self._thread.start()

    def command(self) -> list[str]:
        """Build the deterministic v3 libx264 command for tests and diagnostics."""

        gop_frames = max(1, int(self.gop_policy.get("gop_frames") or 1))
        keyint_min = max(
            1,
            int(self.gop_policy.get("keyint_min_frames") or gop_frames),
        )
        scene_cut_threshold = int(self.gop_policy.get("scene_cut_threshold") or 0)
        x264_params = (
            f"keyint={gop_frames}:min-keyint={keyint_min}:"
            f"scenecut={scene_cut_threshold}:open-gop=0"
        )
        return [
            self.ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-an",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-g",
            str(gop_frames),
            "-keyint_min",
            str(keyint_min),
            "-sc_threshold",
            str(scene_cut_threshold),
            "-x264-params",
            x264_params,
            "-pix_fmt",
            "yuv420p",
            "-threads",
            str(self.threads),
            "-movflags",
            "+faststart",
            str(self.path),
        ]

    def write(self, frame: bytes) -> None:
        if len(frame) != self.width * self.height * 3:
            raise ValueError(
                f"rawvideo frame size changed for {self.path.name}: "
                f"expected {self.width * self.height * 3} bytes, got {len(frame)}"
            )
        if self._closing:
            raise RuntimeError("cannot write after closing a Campaign video sink")
        deadline = time.monotonic() + FFMPEG_TIMEOUT_SECONDS
        while True:
            self._raise_error()
            _check_cancel(self.cancel_check)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"FFmpeg input queue timed out for {self.path.name}")
            try:
                self._queue.put(frame, timeout=min(0.1, remaining))
                self._raise_error()
                return
            except queue.Full:
                continue

    def close_input(self) -> None:
        if self._closing:
            return
        self._closing = True
        deadline = time.monotonic() + FFMPEG_TIMEOUT_SECONDS
        while True:
            self._raise_error()
            _check_cancel(self.cancel_check)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"FFmpeg input queue timed out for {self.path.name}")
            try:
                self._queue.put(self._SENTINEL, timeout=min(0.1, remaining))
                return
            except queue.Full:
                continue

    def wait(self) -> None:
        if not self._closing:
            self.close_input()
        if self._thread is None:
            raise RuntimeError("Campaign video sink was not started")
        deadline = time.monotonic() + FFMPEG_TIMEOUT_SECONDS
        while self._thread.is_alive():
            self._raise_error()
            _check_cancel(self.cancel_check)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.abort()
                raise RuntimeError(f"FFmpeg timed out while encoding {self.path.name}")
            self._thread.join(timeout=min(0.1, remaining))
        self._raise_error()
        if not self.path.is_file() or self.path.stat().st_size <= 0:
            raise RuntimeError(f"FFmpeg produced no Campaign video: {self.path.name}")
        self._stderr.close()

    def finish(self) -> None:
        self.close_input()
        self.wait()

    def abort(self) -> None:
        self._aborted.set()
        process = self._process
        if process is not None:
            _terminate_process(process)
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        try:
            self._stderr.close()
        except Exception:
            pass
        self.path.unlink(missing_ok=True)

    def _writer(self) -> None:
        process = self._process
        try:
            if process is None or process.stdin is None:
                raise RuntimeError("Campaign FFmpeg sink did not expose stdin")
            while not self._aborted.is_set():
                item = self._queue.get()
                if item is self._SENTINEL:
                    break
                try:
                    process.stdin.write(item)
                except OSError as exc:
                    if not _is_broken_pipe_error(exc):
                        raise
                    raise self._broken_pipe_failure(process) from exc
            try:
                process.stdin.close()
            except OSError as exc:
                if not _is_broken_pipe_error(exc):
                    raise
                raise self._broken_pipe_failure(process) from exc
            process.stdin = None
            returncode = process.wait(timeout=FFMPEG_TIMEOUT_SECONDS)
            if self._aborted.is_set():
                return
            if returncode != 0:
                raise RuntimeError(
                    f"FFmpeg/libx264 failed for {self.path.name} "
                    f"(exit code {returncode}): {self._stderr_detail()}"
                )
        except BaseException as exc:
            if not self._aborted.is_set():
                self._error = exc
                self.failure_state.record(self.path.name, exc)
            if process is not None:
                _terminate_process(process)

    def _broken_pipe_failure(self, process: subprocess.Popen[bytes]) -> RuntimeError:
        """Turn an encoder pipe closure into a useful child-process diagnostic."""

        returncode = process.poll()
        wait_note = ""
        if returncode is None:
            try:
                returncode = process.wait(timeout=FFMPEG_PIPE_EXIT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                _terminate_process(process)
                returncode = process.poll()
                wait_note = (
                    f"; encoder did not exit within {FFMPEG_PIPE_EXIT_TIMEOUT_SECONDS:g}s "
                    "after closing stdin and was terminated"
                )
            except OSError as exc:
                _terminate_process(process)
                returncode = process.poll()
                wait_note = f"; failed to wait for encoder: {_error_detail(exc)}"

        stderr_detail = self._stderr_detail()
        if stderr_detail == "unknown encoder error":
            stderr_detail = (
                "no FFmpeg stderr was captured; check whether FFmpeg was terminated "
                "externally and verify server memory, disk space, and encoder availability"
            )
        exit_code = "unknown" if returncode is None else str(returncode)
        return RuntimeError(
            f"Campaign FFmpeg sink {self.path.name} closed stdin unexpectedly "
            f"(exit code {exit_code}){wait_note}; stderr: {stderr_detail}"
        )

    def _stderr_detail(self) -> str:
        try:
            self._stderr.flush()
            self._stderr.seek(0)
            return _error_detail(self._stderr.read().decode("utf-8", errors="replace"))
        except Exception:
            return "unknown encoder error"

    def _raise_error(self) -> None:
        if self._error is not None:
            error = self._error
            self._error = None
            raise RuntimeError(str(error)) from error
        self.failure_state.raise_if_failed()


def _validate_freeze_inputs(
    plan: Mapping[str, Any],
    sources: Mapping[str, Sequence[str | Path]],
) -> dict[str, list[Path]]:
    target = plan.get("target") or {}
    width = int(target.get("width") or 0)
    height = int(target.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Campaign freeze Alignment Plan has no target dimensions")
    plan_sources = plan.get("sources") or {}
    if set(plan_sources) != set(SOURCE_SLOTS):
        raise ValueError(f"Campaign freeze Alignment Plan slots changed: {sorted(plan_sources)}")
    if set(sources) != set(SOURCE_SLOTS):
        raise ValueError(f"Campaign freeze source slots changed: {sorted(sources)}")
    normalized = {slot: [Path(path) for path in sources[slot]] for slot in SOURCE_SLOTS}
    counts = {slot: len(paths) for slot, paths in normalized.items()}
    expected_count = int((plan.get("temporal") or {}).get("frame_count") or 0)
    if expected_count <= 0:
        raise ValueError("Campaign freeze Alignment Plan has no aligned frames")
    if any(count != expected_count for count in counts.values()):
        raise ValueError(f"Campaign freeze frame counts do not match the Alignment Plan: {counts}")
    return normalized


def _collect_remux_eligibility(
    plan: Mapping[str, Any],
    source_media: Mapping[str, str | Path],
    timestamps: Mapping[str, Sequence[float | None]],
    *,
    expected_source_sha256: Mapping[str, str | None],
    fps: float,
    ffprobe: str,
    cancel_check: CancelCheck | None = None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    prediction_digests_ready = all(expected_source_sha256.get(slot) for slot in ("pred_a", "pred_b"))
    for slot in SOURCE_SLOTS:
        source = source_media.get(slot)
        if source is None:
            result[slot] = {
                "eligible": False,
                "reasons": ["original source video is unavailable"],
                "probe": None,
            }
            continue
        if not expected_source_sha256.get(slot) or (slot.startswith("pred_") and not prediction_digests_ready):
            reasons = ["trusted source content digest is unavailable"]
            if slot.startswith("pred_") and expected_source_sha256.get(slot):
                reasons = ["paired prediction trusted content digest is unavailable"]
            result[slot] = {"eligible": False, "reasons": reasons, "probe": None}
            continue
        result[slot] = remux_eligibility(
            plan,
            slot,
            source,
            timestamps=timestamps.get(slot),
            fps=fps,
            ffprobe=ffprobe,
            cancel_check=cancel_check,
        )
    if all(result[slot]["eligible"] for slot in ("pred_a", "pred_b")):
        left_timestamps = _finite_timestamps(
            (result["pred_a"].get("probe") or {}).get("timestamps")
        )
        right_timestamps = _finite_timestamps(
            (result["pred_b"].get("probe") or {}).get("timestamps")
        )
        if (
            left_timestamps is None
            or right_timestamps is None
            or not _relative_timelines_match(left_timestamps, right_timestamps)
        ):
            for slot in ("pred_a", "pred_b"):
                result[slot]["eligible"] = False
                result[slot]["reasons"].append(
                    "paired prediction relative timelines do not match"
                )
    if not all(result[slot]["eligible"] for slot in ("pred_a", "pred_b")):
        for slot in ("pred_a", "pred_b"):
            if result[slot]["eligible"]:
                result[slot]["eligible"] = False
                result[slot]["reasons"].append("paired prediction is not remux-eligible")
    return result


def _artifact_result(
    path: Path,
    mode: str,
    *,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    digest, size_bytes = _digest_path(path, cancel_check=cancel_check)
    return {
        "path": path,
        "sha256": digest,
        "size_bytes": size_bytes,
        "mode": mode,
    }


def _digest_path(
    path: Path,
    *,
    cancel_check: CancelCheck | None = None,
) -> tuple[str, int]:
    """Return package-compatible digest and size in one traversal."""

    digest = hashlib.sha256()
    total = 0
    files = [path] if path.is_file() else sorted(child for child in path.rglob("*") if child.is_file())
    for child in files:
        _check_cancel(cancel_check)
        relative = child.name if path.is_file() else child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                _check_cancel(cancel_check)
                digest.update(chunk)
                total += len(chunk)
    return digest.hexdigest(), total


def _source_signature(
    path: Path,
    *,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            _check_cancel(cancel_check)
            digest.update(chunk)
    final_stat = path.stat()
    if (
        int(stat.st_size) != int(final_stat.st_size)
        or int(stat.st_mtime_ns) != int(final_stat.st_mtime_ns)
    ):
        raise SourceChanged("Campaign remux source changed while its content signature was calculated")
    return {
        "path": path,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _initial_source_signature(
    slot: str,
    path: Path,
    provided: Mapping[str, Mapping[str, Any]] | None,
    *,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """Reuse a freshly computed decode-cache signature when it is still current."""

    candidate = provided.get(slot) if provided is not None else None
    if not isinstance(candidate, Mapping):
        return _source_signature(path, cancel_check=cancel_check)
    try:
        candidate_path = Path(str(candidate["path"])).resolve()
        expected_size = int(candidate["size_bytes"])
        expected_mtime = int(candidate["mtime_ns"])
        expected_digest = str(candidate["sha256"])
    except (KeyError, TypeError, ValueError, OSError):
        return _source_signature(path, cancel_check=cancel_check)
    if candidate_path != path or not expected_digest:
        return _source_signature(path, cancel_check=cancel_check)
    _check_cancel(cancel_check)
    stat = path.stat()
    if int(stat.st_size) != expected_size or int(stat.st_mtime_ns) != expected_mtime:
        raise SourceChanged(f"Campaign remux source {slot} changed before freezing")
    return {
        "path": path,
        "size_bytes": expected_size,
        "mtime_ns": expected_mtime,
        "sha256": expected_digest,
    }


def _assert_source_signature_unchanged(
    slot: str,
    expected: Mapping[str, Any],
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    current = _source_signature(Path(expected["path"]), cancel_check=cancel_check)
    for field in ("size_bytes", "mtime_ns", "sha256"):
        if current[field] != expected[field]:
            raise SourceChanged(f"Campaign source {slot} changed while freezing")


def _assert_source_stat_unchanged(slot: str, expected: Mapping[str, Any]) -> None:
    try:
        current = Path(expected["path"]).stat()
    except OSError as exc:
        raise SourceChanged(f"Campaign source {slot} is no longer available") from exc
    if (
        int(current.st_size) != int(expected["size_bytes"])
        or int(current.st_mtime_ns) != int(expected["mtime_ns"])
    ):
        raise SourceChanged(f"Campaign source {slot} changed while freezing")


def _validate_expected_source_digest(
    slot: str,
    signature: Mapping[str, Any],
    expected: Mapping[str, str | None] | None,
) -> None:
    if expected is None or not expected.get(slot):
        return
    if str(expected[slot]).lower() != str(signature["sha256"]).lower():
        raise SourceChanged(f"Campaign remux source {slot} content digest changed before freezing")


def _check_cancel(cancel_check: CancelCheck | None) -> None:
    if cancel_check is None:
        return
    try:
        cancelled = cancel_check()
    except FreezeCancelled:
        raise
    except BaseException as exc:
        raise FreezeCancelled(str(exc) or "Campaign preparation was cancelled") from exc
    if cancelled:
        raise FreezeCancelled("Campaign preparation was cancelled")


def _progress(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is not None:
        callback(dict(payload))


def _resolve_executable(name: str, explicit: str | Path | None) -> str | None:
    if explicit not in {None, ""}:
        return shutil.which(str(explicit))
    return shutil.which(name)


def _run_json(
    command: list[str],
    label: str,
    *,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    process: subprocess.Popen[bytes] | None = None
    with tempfile.TemporaryFile(mode="w+b") as stdout_handle, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_handle:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            deadline = time.monotonic() + FFMPEG_TIMEOUT_SECONDS
            while True:
                _check_cancel(cancel_check)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"{label} timed out")
                try:
                    returncode = process.wait(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            stdout_handle.seek(0)
            stderr_handle.seek(0)
            stdout_text = stdout_handle.read().decode("utf-8", errors="replace")
            stderr_text = stderr_handle.read().decode("utf-8", errors="replace")
        finally:
            if process is not None:
                _terminate_process(process)
    if returncode != 0:
        raise RuntimeError(f"{label} failed: {_error_detail(stderr_text)}")
    try:
        payload = json.loads(stdout_text or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} returned an invalid payload")
    return payload


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    """Close a child process and reap it on every cancellation/error path."""

    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=2)
            except Exception:
                pass
    else:
        try:
            process.wait(timeout=0)
        except Exception:
            pass
    # Never close a live pipe before termination: another thread can be
    # blocked in BufferedWriter.write() while holding the stream lock.
    if process.poll() is not None and process.stdin is not None:
        try:
            process.stdin.close()
        except Exception:
            pass


def _parse_rate(value: Any) -> float:
    text = str(value or "").strip()
    if not text or text in {"0", "0/0"}:
        return 0.0
    try:
        parsed = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return 0.0
    return parsed if math.isfinite(parsed) and parsed > 0 else 0.0


def _float_or_none(value: Any) -> float | None:
    if value in {None, "", "N/A"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _int_or_none(value: Any) -> int | None:
    if value in {None, "", "N/A"}:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _finite_timestamps(values: Any) -> list[float] | None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    parsed: list[float] = []
    for value in values:
        number = _float_or_none(value)
        if number is None:
            return None
        parsed.append(number)
    return parsed


def _relative_timestamps(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    origin = float(values[0])
    return [float(value) - origin for value in values]


def _relative_timelines_match(
    left: Sequence[float],
    right: Sequence[float],
    *,
    tolerance: float = TIMESTAMP_TOLERANCE_SECONDS,
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        abs(left_value - right_value) <= float(tolerance)
        for left_value, right_value in zip(
            _relative_timestamps(left),
            _relative_timestamps(right),
        )
    )


def _timestamps_are_cfr(timestamps: list[float], durations: list[float], fps: float) -> bool:
    if fps <= 0 or not timestamps:
        return False
    period = 1.0 / fps
    tolerance = max(1e-6, min(TIMESTAMP_TOLERANCE_SECONDS, period * 0.05))
    if len(timestamps) > 1 and any(
        abs((right - left) - period) > tolerance
        for left, right in zip(timestamps, timestamps[1:])
    ):
        return False
    if durations and any(abs(value - period) > tolerance for value in durations):
        return False
    return True


def _stream_rotation(stream: Mapping[str, Any]) -> float:
    side_data = stream.get("side_data_list") or []
    if isinstance(side_data, list):
        display_matrices = [
            item
            for item in side_data
            if isinstance(item, Mapping)
            and "display matrix" in str(item.get("side_data_type") or "").lower()
        ]
        for item in [*display_matrices, *side_data]:
            if isinstance(item, Mapping):
                value = _float_or_none(item.get("rotation"))
                if value is not None:
                    return value
    tags = stream.get("tags") or {}
    if isinstance(tags, Mapping):
        value = _float_or_none(tags.get("rotate"))
        if value is not None:
            return value
    return 0.0


def _rotation_swaps_dimensions(rotation: float) -> bool:
    normalized = float(rotation) % 360.0
    return abs(normalized - 90.0) <= 0.5 or abs(normalized - 270.0) <= 0.5


def _zero_rotation(rotation: float) -> bool:
    normalized = float(rotation) % 360.0
    return normalized <= 0.5 or abs(normalized - 360.0) <= 0.5


def _mp4_has_faststart(path: Path) -> bool:
    """Return whether the top-level moov atom appears before mdat."""

    try:
        with path.open("rb") as handle:
            file_size = path.stat().st_size
            position = 0
            while position + 8 <= file_size:
                handle.seek(position)
                header = handle.read(8)
                if len(header) != 8:
                    return False
                size = int.from_bytes(header[:4], "big")
                kind = header[4:8]
                header_size = 8
                if size == 1:
                    extended = handle.read(8)
                    if len(extended) != 8:
                        return False
                    size = int.from_bytes(extended, "big")
                    header_size = 16
                elif size == 0:
                    size = file_size - position
                if size < header_size or position + size > file_size:
                    return False
                if kind == b"moov":
                    return True
                if kind == b"mdat":
                    return False
                position += size
    except OSError:
        return False
    return False


def _remove_outputs(paths: Sequence[Path] | Any) -> None:
    for path in paths:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except OSError:
            pass


def _error_detail(value: Any) -> str:
    compact = " ".join(str(value or "").strip().split())
    return compact[-1200:] or "unknown error"


def _is_broken_pipe_error(error: OSError) -> bool:
    return isinstance(error, BrokenPipeError) or error.errno == errno.EPIPE
