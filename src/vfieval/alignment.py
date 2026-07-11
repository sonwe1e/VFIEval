from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.run_cleanup import CACHE_GRACE_SECONDS, cache_lease


ALIGNMENT_PLAN_VERSION = 1
DEFAULT_TIMESTAMP_TOLERANCE_SECONDS = 1e-3
SUPPORTED_SPATIAL_MODE = "smallest_pred"
SUPPORTED_FILTER = "lanczos"


def validate_temporal_alignment(
    reference: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    tolerance_seconds: float = DEFAULT_TIMESTAMP_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    """Validate exact temporal identity and return a fingerprintable summary.

    ``source_frame_indices`` is the only supported non-identity mapping.  When
    one Pred supplies it, every Pred must supply the same ordered mapping.  A
    spatial resize is deliberately irrelevant here: it can never repair a
    frame-count, FPS, timestamp, or source-frame mismatch.
    """

    if not predictions or len(predictions) > 2:
        raise ValueError("alignment requires one or two predictions")
    tolerance = float(tolerance_seconds)
    if tolerance < 0:
        raise ValueError("timestamp tolerance must be non-negative")

    reference_count = _positive_dimension(reference.get("frame_count"), "reference frame_count")
    prediction_counts = [
        _positive_dimension(source.get("frame_count"), f"prediction {index + 1} frame_count")
        for index, source in enumerate(predictions)
    ]
    mappings = [_optional_index_mapping(source.get("source_frame_indices")) for source in predictions]
    has_mapping = any(mapping is not None for mapping in mappings)
    if has_mapping:
        if any(mapping is None for mapping in mappings):
            raise ValueError("all predictions must use the same source_frame_indices mapping")
        expected = mappings[0] or []
        for index, (mapping, count) in enumerate(zip(mappings, prediction_counts)):
            current = mapping or []
            if len(current) != count:
                raise ValueError(
                    f"prediction {index + 1} source_frame_indices must contain exactly one index per frame"
                )
            if any(value < 0 or value >= reference_count for value in current):
                raise ValueError(f"prediction {index + 1} source_frame_indices is outside the reference")
            if current != expected:
                raise ValueError("all predictions must use the same ordered source_frame_indices")
        effective_count = len(expected)
        mapping_mode = "indexed"
        selected_indices = expected
    else:
        if any(count != reference_count for count in prediction_counts):
            raise ValueError(
                "exact temporal alignment requires matching frame counts when source_frame_indices is absent: "
                f"{reference_count} vs {', '.join(str(value) for value in prediction_counts)}"
            )
        effective_count = reference_count
        mapping_mode = "exact"
        selected_indices = list(range(reference_count))

    fps_values = [_optional_fps(reference.get("fps"))] + [_optional_fps(source.get("fps")) for source in predictions]
    present_fps = [value for value in fps_values if value is not None]
    if present_fps and any(abs(value - present_fps[0]) > 1e-6 for value in present_fps[1:]):
        raise ValueError("exact temporal alignment requires matching fps metadata")

    reference_timestamps = _optional_timestamps(reference.get("timestamps"), reference_count, "reference")
    prediction_timestamps = [
        _optional_timestamps(source.get("timestamps"), count, f"prediction {index + 1}")
        for index, (source, count) in enumerate(zip(predictions, prediction_counts))
    ]
    selected_reference_timestamps = (
        [reference_timestamps[index] for index in selected_indices]
        if reference_timestamps is not None
        else None
    )
    timestamp_sets: list[tuple[str, list[float]]] = []
    if selected_reference_timestamps is not None:
        timestamp_sets.append(("reference", selected_reference_timestamps))
    timestamp_sets.extend(
        (f"prediction {index + 1}", values)
        for index, values in enumerate(prediction_timestamps)
        if values is not None
    )
    timestamps_verified = len(timestamp_sets) >= 2
    if timestamps_verified:
        baseline_name, baseline = timestamp_sets[0]
        if len(baseline) != effective_count:
            raise ValueError(f"{baseline_name} timestamp count does not match aligned frames")
        for name, values in timestamp_sets[1:]:
            if len(values) != effective_count:
                raise ValueError(f"{name} timestamp count does not match aligned frames")
            for frame_index, (left, right) in enumerate(zip(baseline, values)):
                if abs(left - right) > tolerance:
                    raise ValueError(
                        "exact temporal alignment requires matching frame timestamps: "
                        f"frame {frame_index} {left:.6f}s vs {right:.6f}s"
                    )

    mapping_sha256 = _json_sha256(selected_indices)
    timestamp_payload = {name: values for name, values in timestamp_sets}
    return {
        "mode": mapping_mode,
        "reference_frame_count": reference_count,
        "frame_count": effective_count,
        "prediction_frame_counts": prediction_counts,
        "mapping_count": len(selected_indices),
        "mapping_first": selected_indices[0] if selected_indices else None,
        "mapping_last": selected_indices[-1] if selected_indices else None,
        "mapping_sha256": mapping_sha256,
        "fps": present_fps[0] if present_fps else None,
        "timestamps_verified": timestamps_verified,
        "timestamps_sha256": _json_sha256(timestamp_payload) if timestamp_payload else None,
        "timestamp_tolerance_seconds": tolerance,
    }


def plan_alignment(
    reference: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    spatial_policy: Mapping[str, Any] | None = None,
    temporal_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic spatial/temporal alignment plan.

    Callers own canonical Media Item validation.  This function owns the
    transformation contract only: one Pred keeps its native size; two Preds
    choose the lexicographically smallest ``(area, max-edge, width, height)``
    size.  Every resize uses LANCZOS and is recorded in the returned report.
    """

    if not predictions or len(predictions) > 2:
        raise ValueError("alignment requires one or two predictions")
    policy = dict(spatial_policy or {})
    mode = str(policy.get("mode") or SUPPORTED_SPATIAL_MODE).strip().lower()
    if mode != SUPPORTED_SPATIAL_MODE:
        raise ValueError(f"unsupported spatial alignment mode: {mode}")
    resize_filter = str(policy.get("filter") or SUPPORTED_FILTER).strip().lower()
    if resize_filter != SUPPORTED_FILTER:
        raise ValueError(f"unsupported spatial alignment filter: {resize_filter}")

    prediction_records = [
        _source_record(source, str(source.get("slot") or f"pred_{chr(ord('a') + index)}"), "pred")
        for index, source in enumerate(predictions)
    ]
    reference_record = _source_record(reference, str(reference.get("slot") or "gt"), "gt")
    slots = [reference_record["slot"], *[record["slot"] for record in prediction_records]]
    if len(set(slots)) != len(slots):
        raise ValueError("alignment source slots must be unique")

    target_source = min(
        prediction_records,
        key=lambda source: (
            source["width"] * source["height"],
            max(source["width"], source["height"]),
            source["width"],
            source["height"],
            source["slot"],
        ),
    )
    target_width = int(target_source["width"])
    target_height = int(target_source["height"])
    allow_known_stretch = bool(policy.get("allow_known_aspect_stretch", True))
    allow_external_stretch = bool(policy.get("allow_external_aspect_stretch", False))

    reports: dict[str, dict[str, Any]] = {}
    for source in [reference_record, *prediction_records]:
        aspect_changed = source["width"] * target_height != target_width * source["height"]
        is_external_pred = source["role"] == "pred" and source["external"]
        explicit_source_permission = bool(source["allow_aspect_stretch"])
        if aspect_changed and is_external_pred and not (explicit_source_permission or allow_external_stretch):
            raise ValueError(
                f"external prediction {source['slot']} changes aspect ratio; explicit confirmation is required"
            )
        if aspect_changed and source["role"] == "pred" and not is_external_pred and not allow_known_stretch:
            raise ValueError(f"prediction {source['slot']} changes aspect ratio but known stretch is disabled")
        reports[source["slot"]] = _source_report(
            source,
            target_width,
            target_height,
            resize_filter,
            aspect_changed=aspect_changed,
            aspect_stretch_authorized=(
                not aspect_changed
                or source["role"] == "gt"
                or explicit_source_permission
                or (is_external_pred and allow_external_stretch)
                or (not is_external_pred and allow_known_stretch)
            ),
        )

    temporal = dict(temporal_summary) if temporal_summary is not None else validate_temporal_alignment(reference, predictions)
    payload: dict[str, Any] = {
        "version": ALIGNMENT_PLAN_VERSION,
        "mode": mode,
        "filter": resize_filter,
        "target": {
            "width": target_width,
            "height": target_height,
            "source_slot": target_source["slot"],
        },
        "sources": reports,
        "temporal": temporal,
    }
    payload["fingerprint"] = _json_sha256(payload)
    return payload


def materialize_aligned_frame(
    db: Database,
    workspace: WorkspaceConfig,
    plan: Mapping[str, Any],
    slot: str,
    source_path: str | Path,
) -> Path:
    """Return an original frame or a rebuildable LANCZOS compare-cache frame."""

    normalized_plan = _validated_plan(plan)
    source_report = normalized_plan["sources"].get(str(slot))
    if not isinstance(source_report, Mapping):
        raise KeyError(f"alignment plan has no source slot: {slot}")
    path = Path(source_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"alignment source frame not found: {path}")
    with Image.open(path) as image:
        actual_width, actual_height = image.size
    original = source_report.get("original") or {}
    expected_size = (int(original.get("width") or 0), int(original.get("height") or 0))
    if (actual_width, actual_height) != expected_size:
        raise ValueError(
            f"alignment source {slot} dimensions changed: "
            f"expected {expected_size[0]}x{expected_size[1]}, got {actual_width}x{actual_height}"
        )
    target = normalized_plan["target"]
    target_size = (int(target["width"]), int(target["height"]))
    if (actual_width, actual_height) == target_size:
        return path

    cache_key = alignment_cache_key(normalized_plan, str(slot), path)
    cache_dir = (workspace.root / "compare_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{cache_key}.png"
    with cache_lease(db, workspace, "compare_cache", cache_key, output):
        if not _valid_cached_image(output, target_size):
            temporary = cache_dir / f"{cache_key}.{uuid.uuid4().hex}.tmp.png"
            try:
                with Image.open(path).convert("RGB") as image:
                    image.resize(target_size, _lanczos()).save(temporary, format="PNG")
                os.replace(temporary, output)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        stat = output.stat()
        db.upsert_cache_entry(
            "compare_cache",
            cache_key,
            output,
            state="ready",
            size_bytes=int(stat.st_size),
            metadata={
                "alignment_fingerprint": normalized_plan["fingerprint"],
                "slot": str(slot),
                "source_path": path.as_posix(),
            },
            last_used_at=time.time(),
            gc_after=time.time() + CACHE_GRACE_SECONDS,
        )
    return output


def materialize_frame_sets(
    db: Database,
    workspace: WorkspaceConfig,
    plan: Mapping[str, Any],
    sources: Mapping[str, Sequence[str | Path]],
) -> dict[str, list[Path]]:
    """Materialize equally-sized aligned frame sets for Compare or Campaign."""

    normalized_plan = _validated_plan(plan)
    expected_slots = set(normalized_plan["sources"])
    provided_slots = {str(slot) for slot in sources}
    if provided_slots != expected_slots:
        missing = sorted(expected_slots - provided_slots)
        extra = sorted(provided_slots - expected_slots)
        raise ValueError(f"alignment frame slots mismatch; missing={missing}, extra={extra}")
    counts = {slot: len(paths) for slot, paths in sources.items()}
    expected_count = int((normalized_plan.get("temporal") or {}).get("frame_count") or 0)
    if expected_count > 0 and any(count != expected_count for count in counts.values()):
        raise ValueError(f"alignment frame counts do not match temporal plan: {counts}")
    if len(set(counts.values())) > 1:
        raise ValueError(f"alignment frame sets must have equal lengths: {counts}")
    return {
        str(slot): [
            materialize_aligned_frame(db, workspace, normalized_plan, str(slot), path)
            for path in paths
        ]
        for slot, paths in sources.items()
    }


def alignment_cache_key(plan: Mapping[str, Any], slot: str, source_path: str | Path) -> str:
    normalized_plan = _validated_plan(plan)
    path = Path(source_path).resolve()
    stat = path.stat()
    return _json_sha256(
        {
            "alignment_fingerprint": normalized_plan["fingerprint"],
            "slot": str(slot),
            "source_path": path.as_posix(),
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
        }
    )


def _source_record(source: Mapping[str, Any], slot: str, role: str) -> dict[str, Any]:
    width = _positive_dimension(source.get("width"), f"{slot} width")
    height = _positive_dimension(source.get("height"), f"{slot} height")
    member_role = str(source.get("member_role") or "")
    producer_kind = str(source.get("producer_kind") or source.get("source_kind") or "")
    spatial_origin = source.get("spatial_origin") or source.get("spatial_origin_json") or {}
    external = bool(
        source.get("external")
        or member_role == "external_pred"
        or producer_kind in {"external", "external_pred", "upload_pred"}
    )
    identity = next(
        (
            f"{field}:{source[field]}"
            for field in ("member_id", "asset_id", "source_key", "id")
            if source.get(field) not in {None, ""}
        ),
        f"slot:{slot}",
    )
    return {
        "slot": slot,
        "role": role,
        "width": width,
        "height": height,
        "identity": identity,
        "member_role": member_role or None,
        "producer_kind": producer_kind or None,
        "external": external,
        "allow_aspect_stretch": bool(source.get("allow_aspect_stretch")),
        "spatial_origin": spatial_origin if isinstance(spatial_origin, Mapping) else {},
    }


def _source_report(
    source: Mapping[str, Any],
    target_width: int,
    target_height: int,
    resize_filter: str,
    *,
    aspect_changed: bool,
    aspect_stretch_authorized: bool,
) -> dict[str, Any]:
    width = int(source["width"])
    height = int(source["height"])
    scale_x = target_width / width
    scale_y = target_height / height
    if target_width == width and target_height == height:
        direction = "none"
    elif target_width >= width and target_height >= height:
        direction = "upscale"
    elif target_width <= width and target_height <= height:
        direction = "downscale"
    else:
        direction = "mixed"
    return {
        "slot": source["slot"],
        "role": source["role"],
        "source_identity": source["identity"],
        "member_role": source["member_role"],
        "producer_kind": source["producer_kind"],
        "original": {"width": width, "height": height},
        "target": {"width": target_width, "height": target_height},
        "scale_x": scale_x,
        "scale_y": scale_y,
        "direction": direction,
        "aspect_changed": bool(aspect_changed),
        "aspect_stretch_authorized": bool(aspect_stretch_authorized),
        "filter": resize_filter,
        "spatial_origin": dict(source["spatial_origin"]),
    }


def _validated_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(plan)
    fingerprint = str(value.get("fingerprint") or "")
    unsigned = {key: item for key, item in value.items() if key != "fingerprint"}
    if not fingerprint or fingerprint != _json_sha256(unsigned):
        raise ValueError("alignment plan fingerprint is missing or invalid")
    target = value.get("target") or {}
    _positive_dimension(target.get("width"), "alignment target width")
    _positive_dimension(target.get("height"), "alignment target height")
    if not isinstance(value.get("sources"), Mapping) or not value["sources"]:
        raise ValueError("alignment plan has no sources")
    return value


def _positive_dimension(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _optional_index_mapping(value: Any) -> list[int] | None:
    if value is None or value == "":
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("source_frame_indices must be an array")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("source_frame_indices must contain integers") from exc


def _optional_fps(value: Any) -> float | None:
    if value in {None, "", 0, 0.0}:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("fps must be a positive finite number")
    return parsed


def _optional_timestamps(value: Any, count: int, label: str) -> list[float] | None:
    if value is None or value == "" or isinstance(value, Mapping):
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} timestamps must be an array")
    if not value:
        return None
    if len(value) != count or any(item is None for item in value):
        raise ValueError(f"{label} timestamps must contain one value per frame")
    try:
        timestamps = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} timestamps must be numeric") from exc
    if any(not math.isfinite(item) for item in timestamps):
        raise ValueError(f"{label} timestamps must be finite")
    return timestamps


def _valid_cached_image(path: Path, expected_size: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.load()
            return image.size == expected_size
    except (OSError, ValueError):
        return False


def _lanczos() -> Any:
    resampling = getattr(Image, "Resampling", Image)
    return resampling.LANCZOS


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
