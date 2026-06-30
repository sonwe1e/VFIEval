from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.metrics import METRIC_NAMES, create_metric
from vfieval.metrics.base import MetricUnavailable

METRIC_CACHE_VERSION = "metric-cache-v2"


def run_metric_job(db: Database, workspace: WorkspaceConfig, job_id: int) -> dict[str, Any]:
    job = db.get_job(job_id)
    payload = job["payload"]
    inference_job_ids = [int(value) for value in payload.get("inference_job_ids", [])]
    if not inference_job_ids:
        inference_job_ids = [int(payload["inference_job_id"])]
    inference_job_id = inference_job_ids[0]
    metric_names = list(payload.get("metric_names", []))
    run_id = int(payload["run_id"]) if payload.get("run_id") is not None else None
    if not metric_names:
        raise ValueError("metric job requires metric_names")
    unsupported = [name for name in metric_names if name not in METRIC_NAMES]
    if unsupported:
        raise ValueError(f"unsupported metrics: {', '.join(unsupported)}")

    artifacts = []
    pred_videos = []
    for current_job_id in inference_job_ids:
        artifacts.extend(db.list_artifacts(job_id=current_job_id, kind="pred"))
        pred_videos.extend(db.list_artifacts(job_id=current_job_id, kind="pred_video"))
    if not artifacts and not pred_videos:
        raise ValueError(f"inference job {inference_job_id} has no pred artifacts")

    inference_job = db.get_job(inference_job_id)
    dataset_id = int(inference_job["payload"]["dataset_id"])
    samples = {int(row["id"]): row for row in db.list_samples(dataset_id)}
    use_video_vmaf = "vmaf" in metric_names and bool(pred_videos)
    frame_metric_names = [name for name in metric_names if name != "vmaf" or not use_video_vmaf]
    total = len(artifacts) * len(frame_metric_names) + (len(pred_videos) if use_video_vmaf else 0)
    db.update_job_progress(job_id, 0, total)
    if run_id is not None:
        db.mark_run_started(run_id, "metric_running")
        db.update_run_progress(run_id, 0, total, "metric_running")

    current = 0
    summary: dict[str, dict[str, Any]] = {
        name: {"completed": 0, "unavailable": 0, "failed": 0, "skipped": 0, "mean": None}
        for name in metric_names
    }
    values: dict[str, list[float]] = {name: [] for name in metric_names}

    for artifact in artifacts:
        sample_id = artifact.get("sample_id")
        sample = samples.get(int(sample_id)) if sample_id is not None else None
        reference_path = Path(sample["gt_path"]) if sample and sample.get("gt_path") else None
        distorted_path = Path(artifact["path"])

        for metric_name in frame_metric_names:
            details: dict[str, Any]
            value: float | None
            if reference_path is None:
                status = "skipped"
                value = None
                details = {"reason": "sample has no ground-truth reference"}
            else:
                status, value, details = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name=metric_name,
                    reference_path=reference_path,
                    distorted_path=distorted_path,
                    sample_id=sample_id,
                )

            db.add_metric_result(
                job_id=job_id,
                inference_job_id=int(artifact["job_id"]),
                sample_id=int(sample_id) if sample_id is not None else None,
                metric_name=metric_name,
                status=status,
                value=value,
                details=details,
            )
            summary[metric_name][status] = int(summary[metric_name].get(status, 0)) + 1
            if status == "completed" and value is not None:
                values[metric_name].append(float(value))
            current += 1
            db.update_job_progress(job_id, current)
            if run_id is not None:
                db.update_run_progress(run_id, current)

    if use_video_vmaf:
        gt_videos = []
        for current_job_id in inference_job_ids:
            gt_videos.extend(db.list_artifacts(job_id=current_job_id, kind="gt_video"))
        gt_by_name = {
            artifact.get("metadata", {}).get("video_name"): artifact
            for artifact in gt_videos
            if artifact.get("metadata", {}).get("video_name")
        }
        for artifact in pred_videos:
            video_name = artifact.get("metadata", {}).get("video_name")
            reference_artifact = gt_by_name.get(video_name)
            if reference_artifact is None:
                status = "skipped"
                value = None
                details = {"reason": "video has no ground-truth reference", "video_name": video_name}
            else:
                status, value, details = _evaluate_with_cache(
                    db=db,
                    workspace=workspace,
                    metric_name="vmaf",
                    reference_path=Path(reference_artifact["path"]),
                    distorted_path=Path(artifact["path"]),
                    sample_id=None,
                )
                details = {"video_name": video_name, **details}
            db.add_metric_result(
                job_id=job_id,
                inference_job_id=int(artifact["job_id"]),
                sample_id=None,
                metric_name="vmaf",
                status=status,
                value=value,
                details=details,
            )
            summary["vmaf"][status] = int(summary["vmaf"].get(status, 0)) + 1
            if status == "completed" and value is not None:
                values["vmaf"].append(float(value))
            current += 1
            db.update_job_progress(job_id, current)
            if run_id is not None:
                db.update_run_progress(run_id, current)

    for metric_name, metric_values in values.items():
        if metric_values:
            summary[metric_name]["mean"] = sum(metric_values) / len(metric_values)

    result = {"inference_job_id": inference_job_id, "summary": summary}
    if run_id is not None:
        db.complete_run_metrics(run_id, summary)
    return result


def _evaluate_with_cache(
    db: Database,
    workspace: WorkspaceConfig,
    metric_name: str,
    reference_path: Path,
    distorted_path: Path,
    sample_id: int | None,
) -> tuple[str, float | None, dict[str, Any]]:
    config = {"adapter_version": METRIC_CACHE_VERSION}
    cache_key = metric_cache_key(metric_name, reference_path, distorted_path, config)
    cached = db.get_metric_cache(cache_key)
    if cached:
        return cached["status"], cached["value"], {"cached": True, **cached["details"]}

    metric = create_metric(metric_name, workspace)
    work_dir = workspace.tmp_dir / "metrics" / metric_name / hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
    try:
        result = metric.evaluate(reference_path, distorted_path, work_dir)
        db.set_metric_cache(cache_key, metric_name, result.status, result.value, result.details)
        return result.status, result.value, result.details
    except MetricUnavailable as exc:
        details = {"reason": str(exc)}
        db.set_metric_cache(cache_key, metric_name, "unavailable", None, details)
        return "unavailable", None, details
    except Exception as exc:
        return "failed", None, {"reason": str(exc), "type": type(exc).__name__, "sample_id": sample_id}


def metric_cache_key(
    metric_name: str,
    reference_path: Path,
    distorted_path: Path,
    config: dict[str, Any],
) -> str:
    data = {
        "metric": metric_name,
        "adapter_version": METRIC_CACHE_VERSION,
        "reference": _file_identity(reference_path),
        "distorted": _file_identity(distorted_path),
        "config": config,
    }
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
