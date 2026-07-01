from __future__ import annotations

from typing import Any

from vfieval.config import WorkspaceConfig
from vfieval.datasets import scan_dataset
from vfieval.db import Database
from vfieval.pipeline.inference import RunCanceled


def run_decode_job(db: Database, workspace: WorkspaceConfig, job_id: int) -> dict[str, Any]:
    job = db.get_job(job_id)
    payload = job.get("payload") or {}
    run_id = int(payload["run_id"]) if payload.get("run_id") is not None else None
    dataset_id = int(payload["dataset_id"])
    total_frames = int(payload.get("total_frames") or job.get("progress_total") or 0)
    decode_backend = str(payload.get("decode_backend") or "auto")
    state: dict[str, Any] = {
        "phase": "decode",
        "status": "running",
        "backend": decode_backend,
        "current_video": None,
        "video_index": 0,
        "video_count": int(payload.get("video_count") or 0),
        "decoded_frames": 0,
        "total_frames": total_frames,
        "cache_hits": 0,
        "cache_misses": 0,
        "fallback_reason": None,
        "samples": 0,
    }

    def ensure_not_canceled() -> None:
        if run_id is None:
            return
        run = db.get_run(run_id)
        if run["status"] == "cancel_requested":
            error = {"message": "用户取消了解码任务", "type": "RunCanceled", "phase": "decode"}
            db.cancel_job(job_id, error)
            db.cancel_run(run_id, error)
            raise RunCanceled("用户取消了解码任务")

    def update_progress(event: dict[str, Any]) -> None:
        ensure_not_canceled()
        decoded_frames = int(event.get("decoded_frames") or state.get("decoded_frames") or 0)
        total = int(event.get("total_frames") or state.get("total_frames") or total_frames or 0)
        state.update(
            {
                "backend": event.get("backend") or state.get("backend") or decode_backend,
                "manifest_backend": event.get("manifest_backend") or state.get("manifest_backend"),
                "current_video": event.get("video_name") or state.get("current_video"),
                "video_index": int(event.get("video_index") or state.get("video_index") or 0),
                "video_count": int(event.get("video_count") or state.get("video_count") or 0),
                "decoded_frames": decoded_frames,
                "total_frames": total,
                "cache_hits": int(event.get("cache_hits") or state.get("cache_hits") or 0),
                "cache_misses": int(event.get("cache_misses") or state.get("cache_misses") or 0),
                "fallback_reason": event.get("fallback_reason") or state.get("fallback_reason"),
            }
        )
        db.update_job_progress(job_id, decoded_frames, total if total > 0 else None, result=dict(state))
        if run_id is not None:
            db.update_run_progress(run_id, decoded_frames, total if total > 0 else None, "decoding")

    ensure_not_canceled()
    db.update_job_progress(job_id, 0, total_frames if total_frames > 0 else None, result=dict(state))
    if run_id is not None:
        db.mark_run_started(run_id, "decoding")
        db.update_run_progress(run_id, 0, total_frames if total_frames > 0 else None, "decoding")

    samples = scan_dataset(
        db,
        workspace,
        dataset_id,
        progress_callback=update_progress,
        decode_backend=decode_backend,
    )
    ensure_not_canceled()
    if samples <= 0:
        raise ValueError("视频解码未生成可推理的样本")

    state.update({"status": "completed", "samples": int(samples)})
    db.update_job_progress(
        job_id,
        int(state.get("decoded_frames") or 0),
        int(state.get("total_frames") or 0) if int(state.get("total_frames") or 0) > 0 else None,
        result=dict(state),
    )
    return dict(state)
