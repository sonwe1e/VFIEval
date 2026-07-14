from __future__ import annotations

import uuid
from typing import Any

from vfieval.db import Database
from vfieval.metrics.health import metric_requires_video_input


def create_metric_wave(
    db: Database,
    run_id: int,
    metric_names: list[str],
    *,
    source: str,
    retry: bool = False,
) -> dict[str, Any]:
    """Queue one frame-metric shard per inference device.

    Video-level metrics stay on the leader shard because they consume complete
    encoded videos rather than independent frame samples.
    """
    run = db.get_run(run_id)
    inference_jobs = db.list_run_jobs(run_id, "inference")
    if not inference_jobs:
        inference_job_ids = db.run_inference_job_ids(run_id)
        inference_jobs = [
            {"job_id": job_id, "device": run.get("device"), "payload": {}, "shard_index": index}
            for index, job_id in enumerate(inference_job_ids)
        ]
    if not inference_jobs:
        raise ValueError("Run has no inference job")

    active = current_metric_wave_jobs(db, run)
    if any(row["status"] in {"queued", "running"} for row in active):
        raise ValueError("Run already has an active metric evaluation")

    frame_names = [name for name in metric_names if not metric_requires_video_input(name)]
    video_names = [name for name in metric_names if metric_requires_video_input(name)]
    if not frame_names and not video_names:
        raise ValueError("metric wave requires metric_names")

    wave_id = uuid.uuid4().hex
    all_inference_ids = [int(row["job_id"]) for row in inference_jobs]
    request = dict((run.get("metadata") or {}).get("request") or {})
    batch_size = _positive_int(request.get("metric_batch_size_per_device"))
    job_ids: list[int] = []
    devices: list[str] = []
    for index, inference_job in enumerate(inference_jobs):
        names = list(frame_names)
        if index == 0:
            names.extend(video_names)
        if not names:
            continue
        inference_job_id = int(inference_job["job_id"])
        device = str(inference_job.get("device") or run.get("device") or "cpu")
        sample_ids = (inference_job.get("payload") or {}).get("sample_ids")
        if sample_ids is None:
            sample_ids = [
                int(row["sample_id"])
                for row in db.list_artifacts(job_id=inference_job_id, kind="pred")
                if row.get("sample_id") is not None
            ]
        payload: dict[str, Any] = {
            "run_id": run_id,
            "dataset_id": int(run["dataset_id"]),
            "inference_job_id": inference_job_id,
            "inference_job_ids": all_inference_ids,
            "metric_names": names,
            "metric_device": device,
            "metric_wave_id": wave_id,
            "metric_wave_index": index,
            "metric_wave_count": len(inference_jobs) if frame_names else 1,
            "sample_ids": [int(value) for value in sample_ids],
            "retry": bool(retry),
        }
        if batch_size is not None:
            payload["metric_batch_size_per_device"] = batch_size
        job_id = db.add_run_job(
            run_id,
            "metric",
            payload,
            progress_total=0,
            shard_index=index,
            device=device,
            metadata={"source": source, "metric_wave_id": wave_id, "leader": index == 0},
        )
        job_ids.append(job_id)
        devices.append(device)
        if not frame_names:
            break
    if not job_ids:
        raise ValueError("metric wave did not produce any jobs")
    db.set_run_metric_job(run_id, job_ids[0])
    return {
        "run_id": run_id,
        "metric_job_id": job_ids[0],
        "metric_job_ids": job_ids,
        "metric_wave_id": wave_id,
        "metric_names": metric_names,
        "devices": devices,
    }


def current_metric_wave_jobs(db: Database, run: dict[str, Any]) -> list[dict[str, Any]]:
    leader_id = run.get("metric_job_id")
    if leader_id is None:
        return []
    jobs = db.list_run_jobs(int(run["id"]), "metric")
    leader = next((row for row in jobs if int(row["job_id"]) == int(leader_id)), None)
    if leader is None:
        return []
    wave_id = (leader.get("payload") or {}).get("metric_wave_id")
    if not wave_id:
        return [leader]
    return [row for row in jobs if (row.get("payload") or {}).get("metric_wave_id") == wave_id]


def metric_wave_status(db: Database, run: dict[str, Any]) -> str | None:
    jobs = current_metric_wave_jobs(db, run)
    if not jobs:
        return None
    statuses = {str(row.get("status") or "") for row in jobs}
    if "failed" in statuses:
        return "failed"
    if "running" in statuses:
        return "running"
    if statuses == {"completed"}:
        return "completed"
    if "queued" in statuses:
        return "queued"
    if "canceled" in statuses:
        return "canceled"
    return next(iter(statuses), None)


def update_metric_wave_progress(db: Database, run_id: int, wave_id: str) -> None:
    run = db.get_run(run_id)
    jobs = [
        row
        for row in db.list_run_jobs(run_id, "metric")
        if str((row.get("payload") or {}).get("metric_wave_id") or "") == str(wave_id)
    ]
    current = sum(int(row.get("progress_current") or 0) for row in jobs)
    total = sum(int(row.get("progress_total") or 0) for row in jobs)
    leader_id = run.get("metric_job_id")
    if leader_id is not None:
        db.update_run_metric_wave_progress(run_id, int(leader_id), current, total)


def maybe_complete_metric_wave(db: Database, run_id: int, wave_id: str) -> bool:
    run = db.get_run(run_id)
    jobs = current_metric_wave_jobs(db, run)
    if not jobs or any(str(row.get("payload", {}).get("metric_wave_id") or "") != wave_id for row in jobs):
        return False
    expected_count = max(int((row.get("payload") or {}).get("metric_wave_count") or 1) for row in jobs)
    if len(jobs) != expected_count:
        return False
    update_metric_wave_progress(db, run_id, wave_id)
    if any(row["status"] != "completed" for row in jobs):
        return False

    wave_names = {
        str(name)
        for row in jobs
        for name in list((row.get("payload") or {}).get("metric_names") or [])
    }
    merged = dict(run.get("metric_summary") or {}) if any(bool((row.get("payload") or {}).get("retry")) for row in jobs) else {}
    aggregate: dict[str, dict[str, Any]] = {}
    performance: list[dict[str, Any]] = []
    for row in jobs:
        result = row.get("result") or {}
        if result.get("performance"):
            performance.append({"job_id": int(row["job_id"]), "device": row.get("device"), "metrics": result["performance"]})
        for name, item in (result.get("summary") or {}).items():
            target = aggregate.setdefault(
                str(name),
                {"completed": 0, "unavailable": 0, "failed": 0, "skipped": 0, "mean": None, "value_sum": 0.0},
            )
            for status in ("completed", "unavailable", "failed", "skipped"):
                target[status] += int(item.get(status) or 0)
            value_sum = item.get("value_sum")
            if value_sum is None and item.get("mean") is not None:
                value_sum = float(item["mean"]) * int(item.get("completed") or 0)
            target["value_sum"] += float(value_sum or 0.0)
    for name in wave_names:
        item = aggregate.setdefault(
            name,
            {"completed": 0, "unavailable": 0, "failed": 0, "skipped": 0, "mean": None, "value_sum": 0.0},
        )
        completed = int(item.get("completed") or 0)
        item["mean"] = float(item["value_sum"]) / completed if completed else None
        item.pop("value_sum", None)
        merged[name] = item
    leader_id = int(run["metric_job_id"])
    return db.complete_run_metric_wave(run_id, leader_id, merged, performance)


def _positive_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None
