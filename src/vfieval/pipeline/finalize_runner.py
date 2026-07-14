from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from vfieval.config import WorkspaceConfig
from vfieval.db import Database, _combine_output_health
from vfieval.media_assets import sync_run_assets
from vfieval.pipeline.inference import _write_video_artifacts


def run_finalize_job(db: Database, workspace: WorkspaceConfig, job_id: int) -> dict[str, Any]:
    finalize_started = time.perf_counter()
    job = db.get_job(job_id)
    payload = job.get("payload") or {}
    run_id = int(payload["run_id"])
    run = db.get_run(run_id)
    inference_jobs = db.list_run_jobs(run_id, "inference")
    if not inference_jobs or any(row["status"] != "completed" for row in inference_jobs):
        raise ValueError("finalize requires all inference shards to be completed")
    run_dir = workspace.runs_dir / str(run_id)
    merged: dict[str, dict[str, Any]] = {}
    manifests = []
    for inference_job in inference_jobs:
        path = run_dir / "logs" / "shards" / f"{int(inference_job['job_id'])}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        manifests.append(str(path))
        for key, group in (data.get("video_groups") or {}).items():
            target = merged.setdefault(
                str(key),
                {name: value for name, value in group.items() if name != "frames"},
            )
            target.setdefault("frames", [])
            for frame in group.get("frames") or []:
                converted = dict(frame)
                for name in ("pred_path", "gt_path", "diff_path"):
                    if converted.get(name):
                        converted[name] = Path(str(converted[name]))
                target["frames"].append(converted)
    db.mark_run_started(run_id, "finalizing")
    db.update_job_progress(job_id, 0, max(1, len(merged)))
    artifact_job_id = int(inference_jobs[0]["job_id"])
    if merged:
        _write_video_artifacts(db, artifact_job_id, run_dir, merged)
    db.update_job_progress(job_id, max(1, len(merged)), max(1, len(merged)))
    sync_run_assets(db, workspace, run_id)

    output_health = _combine_output_health((row.get("result") or {}).get("output_health") for row in inference_jobs)
    performances = [row.get("result", {}).get("performance") or {} for row in inference_jobs]
    total_samples = sum(int(row.get("result", {}).get("samples") or 0) for row in inference_jobs)
    max_wall = max((float(row.get("total_wall_seconds") or 0.0) for row in performances), default=0.0)
    max_steady = max((float(row.get("steady_state_seconds") or 0.0) for row in performances), default=0.0)
    finalize_seconds = time.perf_counter() - finalize_started
    end_to_end_wall = max(
        max_wall + finalize_seconds,
        time.time() - float(run.get("started_at") or run.get("created_at") or time.time()),
    )
    result: dict[str, Any] = {
        "samples": total_samples,
        "output_dir": str(run_dir),
        "shards": [
            {
                "job_id": int(row["job_id"]),
                "device": row.get("device"),
                "samples": int(row.get("result", {}).get("samples") or 0),
                "performance": row.get("result", {}).get("performance") or {},
            }
            for row in inference_jobs
        ],
        "performance": {
            "artifact_profile": (run.get("metadata") or {}).get("artifact_profile") or "evaluation",
            "total_wall_seconds": end_to_end_wall,
            "finalize_seconds": finalize_seconds,
            "end_to_end_fps": (total_samples / end_to_end_wall) if end_to_end_wall > 0 else 0.0,
            "steady_state_fps": (total_samples / max_steady) if max_steady > 0 else 0.0,
            "device_count": len(inference_jobs),
        },
        "finalize": {"manifests": manifests, "video_count": len(merged)},
    }
    if output_health is not None:
        result["output_health"] = output_health
    model_loads = [
        {"job_id": int(row["job_id"]), "device": row.get("device"), "report": row.get("result", {}).get("model_load")}
        for row in inference_jobs
        if row.get("result", {}).get("model_load")
    ]
    if model_loads:
        result["model_load"] = model_loads[0]["report"]
        result["model_load_shards"] = model_loads
    try:
        from vfieval.performance import execution_profile_identity, record_execution_profile

        request = dict((run.get("metadata") or {}).get("request") or {})
        request.update(
            {
                "height": int(run.get("height") or 0),
                "width": int(run.get("width") or 0),
                "artifact_profile": result["performance"]["artifact_profile"],
                "device_model": next(
                    (str(row.get("device_name")) for row in performances if row.get("device_name")),
                    "",
                ),
            }
        )
        first_payload = (inference_jobs[0].get("payload") or {}) if inference_jobs else {}
        record_execution_profile(
            db,
            execution_profile_identity(workspace, request),
            {
                "batch_size": int(first_payload.get("batch_size") or 1),
                "prefetch_workers": first_payload.get("prefetch_workers"),
                "save_workers": first_payload.get("save_workers"),
                "max_save_inflight": first_payload.get("max_save_inflight"),
            },
            result["performance"],
        )
    except Exception:
        pass
    artifact_summary = db.summarize_run_artifacts(run_id)
    metrics = list(run.get("metrics") or [])
    if metrics:
        inference_job_ids = [int(row["job_id"]) for row in inference_jobs]
        run_devices = list((run.get("metadata") or {}).get("devices") or [])
        metric_device = str(
            run_devices[0]
            if run_devices
            else inference_jobs[0].get("device") or run.get("device") or "cpu"
        )
        metric_payload = {
            "run_id": run_id,
            "dataset_id": int(run["dataset_id"]),
            "inference_job_ids": inference_job_ids,
            "inference_job_id": inference_job_ids[0],
            "metric_names": metrics,
            "metric_device": metric_device,
        }
        metric_job_id = db.add_run_job(
            run_id,
            "metric",
            metric_payload,
            progress_total=0,
            shard_index=0,
            device=metric_device,
            metadata={"source": "finalize"},
        )
        db.complete_run_inference(run_id, result, artifact_summary, "metric_queued")
        db.set_run_metric_job(run_id, metric_job_id)
    else:
        db.complete_run_inference(run_id, result, artifact_summary, "completed")
    return result
