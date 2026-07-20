from __future__ import annotations

import hashlib
import json
from typing import Any


WORKLOAD_ESTIMATE_CONTRACT = "workload-estimate-v1"
WORKLOAD_CONFIRMATION_SCOPE_CONTRACT = "workload-confirmation-scope-v1"

_CONFIRMATION_SCOPE_EFFECTIVE_KEYS = (
    "device",
    "precision",
    "batch_size_per_device",
    "height",
    "width",
    "sample_count",
    "artifact_profile",
    "prefetch_depth",
)

_PRECISION_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
}
_ARTIFACT_PROFILES = {"evaluation", "diagnostic", "benchmark"}

# The risk thresholds are integer constants so boundary decisions do not
# depend on floating-point rounding.
_DEVICE_MEMORY_RISK_PERCENT = 5
_HOST_MEMORY_RISK_PERCENT = 25
_UNKNOWN_DEVICE_MEMORY_PIXEL_THRESHOLD = 16_000_000

# Artifact planning uses uncompressed RGB-equivalent bytes.  Evaluation keeps
# pred/GT/difference images.  Diagnostic additionally keeps three composed RGB
# images, two RGB flow visualizations, and two single-channel mask images.
_CANONICAL_IMAGE_BYTES_PER_PIXEL = {
    "benchmark": 0,
    "evaluation": 9,
    "diagnostic": 26,
}
_ENCODED_VIDEO_RESERVE_BYTES_PER_PIXEL = {
    "benchmark": 0,
    "evaluation": 9,
    "diagnostic": 9,
}
_ARTIFACT_OVERHEAD_PERCENT = 25


def estimate_workload(
    *,
    device: str,
    precision: str,
    device_memory_bytes: int | None,
    host_available_memory_bytes: int | None,
    batch_size_per_device: int,
    height: int,
    width: int,
    sample_count: int,
    artifact_profile: str,
    prefetch_depth: int,
) -> dict[str, Any]:
    """Return a deterministic, JSON-safe workload and risk estimate.

    The memory values are deliberately lower bounds, not an OOM prediction:
    ``input_tensor_bytes_lower_bound`` covers only the two RGB model inputs,
    and ``prefetch_host_bytes_lower_bound`` covers only float32 decoded input
    pairs.  Model parameters, activations, outputs, GT decode tensors, allocator
    workspace, frame-cache overlap, and save-queue memory are not included.

    ``artifact_budget_bytes`` is a planning reserve.  It budgets canonical
    images at their raw uint8 size, a raw-RGB-equivalent reserve for the three
    encoded video streams, and 25 percent for container/metadata variance.
    Model-specific diagnostic extras are unknowable before the model runs and
    are explicitly excluded in the returned breakdown.
    """

    normalized_device = _normalized_nonempty_string("device", device)
    normalized_precision = _normalized_choice("precision", precision, set(_PRECISION_BYTES))
    normalized_profile = _normalized_choice("artifact_profile", artifact_profile, _ARTIFACT_PROFILES)
    normalized_batch = _integer("batch_size_per_device", batch_size_per_device, minimum=1)
    normalized_height = _integer("height", height, minimum=1)
    normalized_width = _integer("width", width, minimum=1)
    normalized_samples = _integer("sample_count", sample_count, minimum=0)
    normalized_prefetch = _integer("prefetch_depth", prefetch_depth, minimum=0)
    normalized_device_memory = _optional_positive_integer("device_memory_bytes", device_memory_bytes)
    normalized_host_memory = _optional_positive_integer(
        "host_available_memory_bytes", host_available_memory_bytes
    )

    bytes_per_element = _PRECISION_BYTES[normalized_precision]
    batch_pixels = normalized_batch * normalized_height * normalized_width
    rgb_elements = normalized_batch * 3 * normalized_height * normalized_width
    input_tensor_bytes = 2 * rgb_elements * bytes_per_element

    # File decode produces float32 tensors before the requested device dtype is
    # applied.  Keep this estimate independent of model precision.
    prefetch_host_bytes = 2 * rgb_elements * 4 * normalized_prefetch

    artifact_budget, artifact_breakdown = _artifact_budget(
        profile=normalized_profile,
        height=normalized_height,
        width=normalized_width,
        sample_count=normalized_samples,
    )

    reasons: list[dict[str, int | str]] = []
    if normalized_device_memory is not None and (
        input_tensor_bytes * 100 >= normalized_device_memory * _DEVICE_MEMORY_RISK_PERCENT
    ):
        reasons.append(
            {
                "code": "input_pair_device_memory_ge_5_percent",
                "actual_bytes": input_tensor_bytes,
                "available_bytes": normalized_device_memory,
                "threshold_percent": _DEVICE_MEMORY_RISK_PERCENT,
            }
        )
    if normalized_host_memory is not None and (
        prefetch_host_bytes * 100 >= normalized_host_memory * _HOST_MEMORY_RISK_PERCENT
    ):
        reasons.append(
            {
                "code": "prefetch_host_memory_ge_25_percent",
                "actual_bytes": prefetch_host_bytes,
                "available_bytes": normalized_host_memory,
                "threshold_percent": _HOST_MEMORY_RISK_PERCENT,
            }
        )
    if (
        normalized_device_memory is None
        and batch_pixels > _UNKNOWN_DEVICE_MEMORY_PIXEL_THRESHOLD
    ):
        reasons.append(
            {
                "code": "unknown_device_memory_batch_pixels_gt_16000000",
                "batch_pixels_per_device": batch_pixels,
                "threshold_pixels": _UNKNOWN_DEVICE_MEMORY_PIXEL_THRESHOLD,
            }
        )

    effective = {
        "device": normalized_device,
        "device_kind": normalized_device.split(":", 1)[0],
        "precision": normalized_precision,
        "bytes_per_element": bytes_per_element,
        "device_memory_bytes": normalized_device_memory,
        "host_available_memory_bytes": normalized_host_memory,
        "batch_size_per_device": normalized_batch,
        "height": normalized_height,
        "width": normalized_width,
        "sample_count": normalized_samples,
        "artifact_profile": normalized_profile,
        "prefetch_depth": normalized_prefetch,
    }
    fingerprint_payload = {
        "contract": WORKLOAD_ESTIMATE_CONTRACT,
        "effective": effective,
    }
    risk_fingerprint = hashlib.sha256(_canonical_json(fingerprint_payload).encode("utf-8")).hexdigest()

    return {
        "contract": WORKLOAD_ESTIMATE_CONTRACT,
        "effective": effective,
        "batch_pixels_per_device": batch_pixels,
        "input_tensor_bytes_lower_bound": input_tensor_bytes,
        "prefetch_host_bytes_lower_bound": prefetch_host_bytes,
        "device_memory_fraction": _ratio(input_tensor_bytes, normalized_device_memory),
        "host_memory_fraction": _ratio(prefetch_host_bytes, normalized_host_memory),
        "artifact_budget_bytes": artifact_budget,
        "artifact_budget_breakdown": artifact_breakdown,
        "risk_level": "high" if reasons else "normal",
        "risk_reasons": reasons,
        "risk_fingerprint": risk_fingerprint,
    }


def workload_confirmation_scope(workload: dict[str, Any]) -> dict[str, Any]:
    """Return the stable execution scope covered by a risk acknowledgement.

    Available host/device memory is intentionally excluded because it can
    fluctuate between the 409 response and the immediately-following retry.
    Execution-affecting values that can change when current source files are
    replaced, especially dimensions and sample count, remain in the scope.
    """

    if not isinstance(workload, dict):
        raise TypeError("workload must be an object")
    effective = workload.get("effective")
    if not isinstance(effective, dict):
        raise ValueError("workload effective values are missing")
    missing = [key for key in _CONFIRMATION_SCOPE_EFFECTIVE_KEYS if key not in effective]
    if missing:
        raise ValueError(
            "workload confirmation scope is missing: " + ", ".join(missing)
        )
    return {
        "contract": WORKLOAD_CONFIRMATION_SCOPE_CONTRACT,
        "effective": {
            key: effective[key]
            for key in _CONFIRMATION_SCOPE_EFFECTIVE_KEYS
        },
    }


def workload_confirmation_scope_fingerprint(workload: dict[str, Any]) -> str:
    scope = workload_confirmation_scope(workload)
    return hashlib.sha256(_canonical_json(scope).encode("utf-8")).hexdigest()


def _artifact_budget(
    *,
    profile: str,
    height: int,
    width: int,
    sample_count: int,
) -> tuple[int, dict[str, int | bool]]:
    pixels = height * width * sample_count
    canonical_bytes = pixels * _CANONICAL_IMAGE_BYTES_PER_PIXEL[profile]
    encoded_video_bytes = pixels * _ENCODED_VIDEO_RESERVE_BYTES_PER_PIXEL[profile]
    subtotal = canonical_bytes + encoded_video_bytes
    overhead_bytes = (subtotal * _ARTIFACT_OVERHEAD_PERCENT + 99) // 100
    return (
        subtotal + overhead_bytes,
        {
            "canonical_images_bytes": canonical_bytes,
            "encoded_video_reserve_bytes": encoded_video_bytes,
            "overhead_reserve_bytes": overhead_bytes,
            "model_specific_extras_included": False,
        },
    )


def _normalized_nonempty_string(name: str, value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _normalized_choice(name: str, value: Any, allowed: set[str]) -> str:
    normalized = _normalized_nonempty_string(name, value)
    if normalized not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}")
    return normalized


def _integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(value, str) and str(normalized) != value.strip():
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if normalized < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return normalized


def _optional_positive_integer(name: str, value: Any) -> int | None:
    if value is None:
        return None
    return _integer(name, value, minimum=1)


def _ratio(numerator: int, denominator: int | None) -> float | None:
    if denominator is None:
        return None
    return round(numerator / denominator, 6)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
