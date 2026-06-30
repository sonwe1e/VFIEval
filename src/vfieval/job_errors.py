from __future__ import annotations

from typing import Any, Mapping


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(*values: Any) -> int | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _needs_context(error: Mapping[str, Any], key: str) -> bool:
    return key not in error or error.get(key) is None or error.get(key) == ""


def job_error_context(job: Mapping[str, Any]) -> dict[str, Any]:
    payload = job.get("payload") or {}
    return {
        "job_id": _coerce_int(job.get("job_id"), job.get("id")),
        "job_kind": _clean_text(job.get("role") or job.get("kind")) or "job",
        "run_id": _coerce_int(payload.get("run_id")),
        "device": _clean_text(job.get("device") or payload.get("device")),
        "worker_id": _clean_text(job.get("worker_id")),
        "shard_index": _coerce_int(job.get("shard_index"), payload.get("shard_index")),
        "shard_count": _coerce_int(payload.get("shard_count")),
    }


def describe_job_failure(job: Mapping[str, Any], detail: Any | None = None) -> str:
    context = job_error_context(job)
    kind = context["job_kind"]
    if kind == "inference":
        label = "Inference shard" if context["shard_index"] is not None else "Inference job"
    elif kind == "metric":
        label = "Metric job"
    else:
        label = f"{kind.capitalize()} job"
    if context["shard_index"] is not None:
        label = f"{label} #{context['shard_index']}"
    if context["device"]:
        label = f"{label} on {context['device']}"
    detail_text = _clean_text(detail)
    if detail_text:
        return f"{label} failed: {detail_text}"
    return f"{label} failed"


def enrich_job_error(job: Mapping[str, Any], error: Mapping[str, Any] | None = None) -> dict[str, Any]:
    normalized = dict(error or {})
    if not normalized.get("message"):
        normalized["message"] = describe_job_failure(job)
    context = job_error_context(job)
    for key, value in context.items():
        if value is not None and _needs_context(normalized, key):
            normalized[key] = value
    return normalized
