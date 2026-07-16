from __future__ import annotations

import time
from typing import Any

from vfieval.config import WorkspaceConfig
from vfieval.db import Database, _combine_output_health
from vfieval.media_assets import sync_run_assets
from vfieval.pipeline.artifact_integrity import (
    ArtifactIntegrityError,
    merge_integrity_reports,
    require_finalize_inputs,
    validate_finalize_video_artifact_integrity,
    write_integrity_report,
)
from vfieval.pipeline.inference import RunCanceled, _write_video_artifacts


def _require_finalizing(db: Database, run_id: int, job_id: int, phase: str) -> None:
    status = str(db.get_run(run_id).get("status") or "")
    if status in {"cancel_requested", "canceled"}:
        error = {"message": f"Run canceled during {phase}", "type": "RunCanceled"}
        raise RunCanceled(error["message"])
    if status == "failed":
        raise RunCanceled(f"Run failed during {phase}")
    if status != "finalizing":
        raise RuntimeError(f"run {run_id} left finalizing during {phase}: {status}")


def _require_finalize_job_progress(
    db: Database,
    run_id: int,
    job_id: int,
    accepted: bool,
    phase: str,
) -> None:
    if accepted:
        return
    _require_finalizing(db, run_id, job_id, phase)
    raise RuntimeError(f"finalize Job {job_id} rejected progress CAS during {phase}")


def run_finalize_job(db: Database, workspace: WorkspaceConfig, job_id: int) -> dict[str, Any]:
    finalize_started = time.perf_counter()
    job = db.get_job(job_id)
    payload = job.get("payload") or {}
    run_id = int(payload["run_id"])
    run = db.get_run(run_id)
    if str(run.get("status") or "") in {"cancel_requested", "canceled"}:
        _require_finalizing(db, run_id, job_id, "finalize startup")
    run_metadata = run.get("metadata") or {}
    request_metadata = run_metadata.get("request") or {}
    preview_height = int(request_metadata.get("visualize_height") or run_metadata.get("visualize_height") or 0) or None
    preview_width = int(request_metadata.get("visualize_width") or run_metadata.get("visualize_width") or 0) or None
    inference_jobs = db.list_run_jobs(run_id, "inference")
    if not inference_jobs or any(row["status"] != "completed" for row in inference_jobs):
        raise ValueError("finalize requires all inference shards to be completed")
    run_dir = workspace.runs_dir / str(run_id)
    integrity_path = run_dir / "logs" / "artifact_integrity.json"
    try:
        merged, input_integrity = require_finalize_inputs(db, run_id, inference_jobs, run_dir)
    except ArtifactIntegrityError as exc:
        write_integrity_report(integrity_path, exc.report)
        db.merge_run_result(run_id, {"artifact_integrity": exc.report})
        raise
    if not db.mark_run_started(run_id, "finalizing"):
        _require_finalizing(db, run_id, job_id, "finalize startup")
        raise RuntimeError(f"run {run_id} rejected finalize start from status {db.get_run(run_id)['status']}")
    _require_finalizing(db, run_id, job_id, "start")
    _require_finalize_job_progress(
        db,
        run_id,
        job_id,
        db.update_job_progress(job_id, 0, max(1, len(merged))),
        "startup",
    )
    artifact_job_id = int(inference_jobs[0]["job_id"])
    if merged:
        try:
            _write_video_artifacts(
                db,
                artifact_job_id,
                run_dir,
                merged,
                preview_height=preview_height,
                preview_width=preview_width,
            )
        except Exception as exc:
            encoding_integrity = merge_integrity_reports(
                "finalize",
                [input_integrity],
                run_id=run_id,
                phase="video_encoding",
            )
            encoding_integrity["errors"].append(
                {
                    "code": "video_encoding_failed",
                    "message": str(exc) or type(exc).__name__,
                    "error_type": type(exc).__name__,
                }
            )
            encoding_integrity["error_count"] = len(encoding_integrity["errors"])
            encoding_integrity["valid"] = False
            write_integrity_report(integrity_path, encoding_integrity)
            db.merge_run_result(run_id, {"artifact_integrity": encoding_integrity})
            raise ArtifactIntegrityError(encoding_integrity) from exc
        _require_finalizing(db, run_id, job_id, "video encoding")
    _require_finalize_job_progress(
        db,
        run_id,
        job_id,
        db.update_job_progress(job_id, max(1, len(merged)), max(1, len(merged))),
        "video encoding",
    )
    publish_pred_video = bool(
        run_metadata.get(
            "publish_compare_pred_video",
            request_metadata.get("publish_compare_pred_video", True),
        )
    )
    video_integrity = validate_finalize_video_artifact_integrity(
        db,
        artifact_job_id,
        [int(row["job_id"]) for row in inference_jobs],
        merged,
        publish_pred_video=publish_pred_video,
        expected_sample_ids=input_integrity.get("expected_sample_ids") or [],
    )
    integrity_report = merge_integrity_reports(
        "finalize",
        [input_integrity, video_integrity],
        run_id=run_id,
        phase="complete",
    )
    write_integrity_report(integrity_path, integrity_report)
    if not integrity_report["valid"]:
        db.merge_run_result(run_id, {"artifact_integrity": integrity_report})
        raise ArtifactIntegrityError(integrity_report)
    _require_finalizing(db, run_id, job_id, "artifact publication")
    try:
        sync_run_assets(db, workspace, run_id)
    except Exception as exc:
        db.invalidate_run_media_assets(run_id)
        publication_integrity = merge_integrity_reports(
            "finalize",
            [integrity_report],
            run_id=run_id,
            phase="media_publication",
        )
        publication_integrity["errors"].append(
            {
                "code": "media_publication_failed",
                "message": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
            }
        )
        publication_integrity["error_count"] = len(publication_integrity["errors"])
        publication_integrity["valid"] = False
        write_integrity_report(integrity_path, publication_integrity)
        db.merge_run_result(run_id, {"artifact_integrity": publication_integrity})
        raise ArtifactIntegrityError(publication_integrity) from exc
    if str(db.get_run(run_id).get("status") or "") != "finalizing":
        db.invalidate_run_media_assets(run_id)
        _require_finalizing(db, run_id, job_id, "artifact publication")

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
    artifact_profile = str(
        run_metadata.get("artifact_profile")
        or request_metadata.get("artifact_profile")
        or ((inference_jobs[0].get("payload") or {}).get("artifact_profile") if inference_jobs else None)
        or "evaluation"
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
            "artifact_profile": artifact_profile,
            "total_wall_seconds": end_to_end_wall,
            "finalize_seconds": finalize_seconds,
            "end_to_end_fps": (total_samples / end_to_end_wall) if end_to_end_wall > 0 else 0.0,
            "steady_state_fps": (total_samples / max_steady) if max_steady > 0 else 0.0,
            "device_count": len(inference_jobs),
        },
        "finalize": {
            "manifests": input_integrity.get("manifest_paths") or [],
            "video_count": len(merged),
            "integrity_report": str(integrity_path),
        },
        "artifact_integrity": integrity_report,
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
    _require_finalizing(db, run_id, job_id, "completion")
    if metrics:
        from vfieval.pipeline.metric_jobs import create_metric_wave

        try:
            create_metric_wave(
                db,
                run_id,
                metrics,
                source="finalize",
                result=result,
                artifact_summary=artifact_summary,
                source_job_id=job_id,
                source_job_result=result,
            )
        except Exception:
            if str(db.get_run(run_id).get("status") or "") not in {"metric_queued", "metric_running", "completed"}:
                db.invalidate_run_media_assets(run_id)
            if str(db.get_run(run_id).get("status") or "") in {"cancel_requested", "canceled"}:
                _require_finalizing(db, run_id, job_id, "metric publication")
            raise
    else:
        if not db.complete_run_inference(
            run_id,
            result,
            artifact_summary,
            "completed",
            source_job_id=job_id,
            source_job_result=result,
        ):
            if str(db.get_run(run_id).get("status") or "") != "completed":
                db.invalidate_run_media_assets(run_id)
            _require_finalizing(db, run_id, job_id, "finalize completion")
            raise RuntimeError(f"run {run_id} rejected finalize completion from status {db.get_run(run_id)['status']}")
    return result
