from __future__ import annotations

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

from PIL import Image, ImageChops

from vfieval.alignment import materialize_aligned_rgb


SOURCE_SLOTS = ("gt", "pred_a", "pred_b")
OUTPUT_SLOTS = ("gt", "pred_a", "pred_b", "diff_a", "diff_b")
OUTPUT_NAMES = {
    "gt": "reference",
    "pred_a": "method-a",
    "pred_b": "method-b",
    "diff_a": "diff-a",
    "diff_b": "diff-b",
}
FFMPEG_TIMEOUT_SECONDS = 600.0
TIMESTAMP_TOLERANCE_SECONDS = 1e-3

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
    """Freeze one aligned GT/A/B item and its two Diffs into private media.

    Video frames are read once per logical source frame and sent directly to
    bounded rawvideo sinks.  Eligible source MP4s may instead be privately
    remuxed, but A and B always take that path as a pair.  Frame sequences are
    written directly to their five final PNG directories with no intermediate
    sequence.
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
    if width % 2 or height % 2:
        raise ValueError(
            "browser-compatible H.264/yuv420p requires even dimensions; "
            f"requested {width}x{height}. Canonical dimensions will not be padded."
        )

    ffmpeg_path = _resolve_executable("ffmpeg", ffmpeg)
    ffprobe_path = _resolve_executable("ffprobe", ffprobe)
    if not streaming_backend_available(ffmpeg_path, cancel_check=cancel_check):
        raise FreezeBackendUnavailable("Campaign streaming freeze requires FFmpeg with libx264")
    if ffprobe_path is None:
        raise FreezeBackendUnavailable("Campaign streaming freeze requires ffprobe")

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
    thread_count = max(1, max(1, cpu_count - 1) // max(1, len(stream_slots)))
    pipeline = "remux+stream" if remuxed else "streaming"
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
                for slot in SOURCE_SLOTS
            }
            try:
                diff_a = ImageChops.difference(images["gt"], images["pred_a"])
                diff_b = ImageChops.difference(images["gt"], images["pred_b"])
                try:
                    frame_images = {
                        "gt": images["gt"],
                        "pred_a": images["pred_a"],
                        "pred_b": images["pred_b"],
                        "diff_a": diff_a,
                        "diff_b": diff_b,
                    }
                    for slot, sink in sinks.items():
                        sink.write(frame_images[slot].tobytes())
                finally:
                    diff_a.close()
                    diff_b.close()
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
    try:
        for slot in stream_slots:
            validate_frozen_video(
                paths[slot],
                width=width,
                height=height,
                frame_count=frame_count,
                fps=resolved_fps,
                ffprobe=ffprobe_path,
                cancel_check=cancel_check,
            )
        for slot, signature in signatures.items():
            _assert_source_signature_unchanged(
                slot, signature, cancel_check=cancel_check
            )
    except Exception:
        _remove_outputs(paths.values())
        raise
    timings["validate"] = time.monotonic() - validation_started

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
        "pipeline": pipeline,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": resolved_fps,
        "encoder_threads": thread_count,
        "artifacts": artifacts,
        "timings": timings,
        "remux": eligibility,
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
                diff_a = ImageChops.difference(images["gt"], images["pred_a"])
                diff_b = ImageChops.difference(images["gt"], images["pred_b"])
                try:
                    outputs = {
                        "gt": images["gt"],
                        "pred_a": images["pred_a"],
                        "pred_b": images["pred_b"],
                        "diff_a": diff_a,
                        "diff_b": diff_b,
                    }
                    for slot, image in outputs.items():
                        image.save(paths[slot] / f"{index:06d}.png", format="PNG")
                finally:
                    diff_a.close()
                    diff_b.close()
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
    return {
        "pipeline": "png_sequence",
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "encoder_threads": 0,
        "artifacts": artifacts,
        "timings": timings,
        "remux": {slot: {"eligible": False, "reasons": ["frame_sequence"]} for slot in SOURCE_SLOTS},
    }


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
    if str(temporal.get("mode") or "") != "exact":
        reasons.append("temporal mapping is not full identity")
    if int(temporal.get("reference_frame_count") or 0) != frame_count:
        reasons.append("reference frame count is not full identity")
    if int(temporal.get("mapping_count") or 0) != frame_count:
        reasons.append("temporal mapping count is not full identity")
    if frame_count > 0 and (
        int(temporal.get("mapping_first") or 0) != 0
        or int(temporal.get("mapping_last") if temporal.get("mapping_last") is not None else -1)
        != frame_count - 1
    ):
        reasons.append("temporal mapping endpoints are not full identity")

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
    elif any(
        abs(left - right) > TIMESTAMP_TOLERANCE_SECONDS
        for left, right in zip(expected_timestamps, observed_timestamps)
    ):
        reasons.append("source timestamps changed after alignment")
    return {"eligible": not reasons, "reasons": reasons, "probe": probe}


def probe_video_for_freeze(
    path: str | Path,
    *,
    ffprobe: str | Path | None = None,
    include_timestamps: bool = True,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """Probe stream-copy eligibility and final package invariants with ffprobe."""

    source = Path(path).resolve()
    executable = _resolve_executable("ffprobe", ffprobe)
    if executable is None:
        raise RuntimeError("ffprobe is unavailable")
    metadata_command = [
        executable,
        "-v",
        "error",
        *([] if include_timestamps else ["-count_packets"]),
        "-show_entries",
        (
            "stream=index,codec_type,codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,"
            "nb_frames,nb_read_packets,duration:stream_tags=rotate:"
            "stream_side_data=side_data_type,rotation:format=duration"
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
    if include_timestamps:
        frame_payload = _run_json(
            [
                executable,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time,pts_time,pkt_duration_time,duration_time",
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
                timestamps.append(timestamp)
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
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """Validate frozen MP4 structure without decoding all pixels again."""

    target = Path(path).resolve()
    probe = probe_video_for_freeze(
        target,
        ffprobe=ffprobe,
        include_timestamps=False,
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
        errors.append("frame and packet counts are unavailable")
    if abs(float(probe.get("fps") or 0.0) - float(fps)) > 1e-6:
        errors.append(f"fps changed: expected {fps}, got {probe.get('fps')}")
    if int(probe.get("audio_stream_count") or 0) != 0:
        errors.append("frozen package contains an audio stream")
    if not _mp4_has_faststart(target):
        errors.append("MP4 moov atom is not before media data")
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
        cancel_check: CancelCheck | None = None,
        failure_state: _SinkFailureState | None = None,
    ) -> None:
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.threads = max(1, int(threads))
        self.ffmpeg = str(ffmpeg)
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
        command = [
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
            "-pix_fmt",
            "yuv420p",
            "-threads",
            str(self.threads),
            "-movflags",
            "+faststart",
            str(self.path),
        ]
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
                process.stdin.write(item)
            process.stdin.close()
            process.stdin = None
            returncode = process.wait(timeout=FFMPEG_TIMEOUT_SECONDS)
            if self._aborted.is_set():
                return
            if returncode != 0:
                raise RuntimeError(
                    f"FFmpeg/libx264 failed for {self.path.name}: {self._stderr_detail()}"
                )
        except BaseException as exc:
            if not self._aborted.is_set():
                self._error = exc
                self.failure_state.record(self.path.name, exc)
            if process is not None:
                _terminate_process(process)

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
