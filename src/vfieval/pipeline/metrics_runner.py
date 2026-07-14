from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from PIL import Image

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.metrics import METRIC_NAMES, create_metric
from vfieval.metrics.base import MetricBatchOutOfMemory, MetricUnavailable
from vfieval.metrics.health import metric_cache_config, metric_requires_video_input
from vfieval.pipeline.inference import RunCanceled
from vfieval.media_assets import bind_metric_result, run_asset_pair, sync_run_assets

METRIC_CACHE_VERSION = "metric-cache-v3"


def run_metric_job(db: Database, workspace: WorkspaceConfig, job_id: int) -> dict[str, Any]:
    job = db.get_job(job_id)
    payload = job["payload"]
    inference_job_ids = [int(value) for value in payload.get("inference_job_ids", [])]
    if not inference_job_ids:
        inference_job_ids = [int(payload["inference_job_id"])]
    inference_job_id = inference_job_ids[0]
    metric_names = list(payload.get("metric_names", []))
    run_id = int(payload["run_id"]) if payload.get("run_id") is not None else None
    metric_device = str(payload.get("metric_device") or "cpu")
    if not metric_names:
        raise ValueError("metric job requires metric_names")
    unsupported = [name for name in metric_names if name not in METRIC_NAMES]
    if unsupported:
        raise ValueError(f"unsupported metrics: {', '.join(unsupported)}")
    _raise_if_canceled(db, run_id, job_id)
    if run_id is not None:
        sync_run_assets(db, workspace, run_id)

    artifacts = []
    pred_videos = []
    for current_job_id in inference_job_ids:
        artifacts.extend(db.list_artifacts(job_id=current_job_id, kind="pred"))
        pred_videos.extend(db.list_artifacts(job_id=current_job_id, kind="pred_video"))
    all_artifacts = list(artifacts)
    assigned_sample_ids = payload.get("sample_ids")
    if assigned_sample_ids is not None:
        allowed_sample_ids = {int(value) for value in assigned_sample_ids}
        artifacts = [row for row in artifacts if row.get("sample_id") is not None and int(row["sample_id"]) in allowed_sample_ids]
    if not artifacts and not pred_videos:
        raise ValueError(f"inference job {inference_job_id} has no pred artifacts")

    inference_job = db.get_job(inference_job_id)
    dataset_id = int(inference_job["payload"]["dataset_id"])
    dataset = db.get_dataset(dataset_id)
    samples = {int(row["id"]): row for row in db.list_samples(dataset_id)}
    video_metric_names = [name for name in metric_names if metric_requires_video_input(name)]
    frame_metric_names = [name for name in metric_names if name not in video_metric_names]
    metric_cache_configs = {name: metric_cache_config(workspace, name) for name in metric_names}
    video_metric_inputs = (
        _collect_video_metric_inputs(
            db=db,
            workspace=workspace,
            run_id=run_id,
            dataset=dataset,
            samples=samples,
            inference_job_ids=inference_job_ids,
            pred_artifacts=all_artifacts,
            pred_videos=pred_videos,
        )
        if video_metric_names
        else []
    )
    video_metric_units = len(video_metric_inputs) if video_metric_inputs else 1
    total = len(artifacts) * len(frame_metric_names) + (video_metric_units * len(video_metric_names))
    db.update_job_progress(job_id, 0, total)
    if run_id is not None:
        _raise_if_canceled(db, run_id, job_id)
        db.mark_run_started(run_id, "metric_running")
        _publish_metric_progress(db, job_id, run_id, 0, total, payload)

    current = 0
    summary: dict[str, dict[str, Any]] = {
        name: {"completed": 0, "unavailable": 0, "failed": 0, "skipped": 0, "mean": None}
        for name in metric_names
    }
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    performance: dict[str, dict[str, Any]] = {}
    identity_cache: dict[Path, dict[str, Any]] = {}

    # Run one frame metric at a time. Feature adapters then keep exactly one
    # model resident on the assigned accelerator and process image pairs in
    # batches instead of reconstructing the backbone for every frame.
    for metric_name in frame_metric_names:
        _raise_if_canceled(db, run_id, job_id)
        current, metric_performance = _run_frame_metric_batches(
            db=db,
            workspace=workspace,
            job_id=job_id,
            run_id=run_id,
            metric_name=metric_name,
            metric_device=metric_device,
            artifacts=artifacts,
            samples=samples,
            dataset=dataset,
            cache_config=metric_cache_configs[metric_name],
            summary=summary[metric_name],
            values=values[metric_name],
            current=current,
            total=total,
            requested_batch_size=_positive_int(payload.get("metric_batch_size_per_device")),
            identity_cache=identity_cache,
            job_payload=payload,
        )
        performance[metric_name] = metric_performance

    if video_metric_names:
        if not video_metric_inputs:
            for metric_name in video_metric_names:
                status = "unavailable"
                value = None
                details = {"reason": "metric requires video artifacts but run has no pred_video outputs"}
                metric_result_id = db.add_metric_result(
                    job_id=job_id,
                    inference_job_id=inference_job_id,
                    sample_id=None,
                    metric_name=metric_name,
                    status=status,
                    value=value,
                    details=details,
                )
                if run_id is not None:
                    bind_metric_result(db, metric_result_id, None, None)
                summary[metric_name][status] = int(summary[metric_name].get(status, 0)) + 1
                current += 1
                _publish_metric_progress(db, job_id, run_id, current, total, payload)
        else:
            for metric_name in video_metric_names:
                for video_input in video_metric_inputs:
                    _raise_if_canceled(db, run_id, job_id)
                    video_name = video_input.get("video_name")
                    track_details = dict(video_input.get("track_details") or {})
                    input_details = dict(video_input.get("input_details") or {})
                    error = str(video_input.get("error") or "")
                    reference_path = video_input.get("reference_path")
                    distorted_path = video_input.get("distorted_path")
                    if error:
                        status = "failed"
                        value = None
                        details = {
                            "reason": error,
                            "video_name": video_name,
                            **track_details,
                            **input_details,
                        }
                    elif reference_path is None:
                        status = "skipped"
                        value = None
                        details = {
                            "reason": "video has no ground-truth reference",
                            "video_name": video_name,
                            **track_details,
                            **input_details,
                        }
                    elif distorted_path is None:
                        status = "failed"
                        value = None
                        details = {
                            "reason": "video metric input has no distorted video",
                            "video_name": video_name,
                            **track_details,
                            **input_details,
                        }
                    else:
                        with _lease_metric_video_inputs(
                            db,
                            workspace,
                            [Path(reference_path), Path(distorted_path)],
                        ):
                            status, value, details = _evaluate_with_cache(
                                db=db,
                                workspace=workspace,
                                metric_name=metric_name,
                                reference_path=Path(reference_path),
                                distorted_path=Path(distorted_path),
                                sample_id=None,
                                cache_config=metric_cache_configs[metric_name],
                                metric_device=metric_device,
                                alignment_context=video_input.get("alignment_context") or _metric_alignment_context(dataset),
                            )
                        details = {
                            "video_name": video_name,
                            **track_details,
                            **input_details,
                            **details,
                        }
                    metric_result_id = db.add_metric_result(
                        job_id=job_id,
                        inference_job_id=int(video_input["inference_job_id"]),
                        sample_id=None,
                        metric_name=metric_name,
                        status=status,
                        value=value,
                        details=details,
                    )
                    if run_id is not None:
                        track_label = str(details.get("compare_track_label") or "")
                        reference_asset_id, distorted_asset_id = run_asset_pair(
                            db, run_id, str(video_name or ""), track_label
                        )
                        bind_metric_result(
                            db,
                            metric_result_id,
                            reference_asset_id,
                            distorted_asset_id,
                            video_name=str(video_name or ""),
                            track_label=track_label,
                        )
                    summary[metric_name][status] = int(summary[metric_name].get(status, 0)) + 1
                    if status == "completed" and value is not None:
                        values[metric_name].append(float(value))
                    current += 1
                    _publish_metric_progress(db, job_id, run_id, current, total, payload)

    for metric_name, metric_values in values.items():
        if metric_values:
            summary[metric_name]["mean"] = sum(metric_values) / len(metric_values)
            summary[metric_name]["value_sum"] = sum(metric_values)

    result = {
        "inference_job_id": inference_job_id,
        "summary": summary,
        "performance": performance,
        "wave_id": payload.get("metric_wave_id"),
    }
    if run_id is not None and not payload.get("metric_wave_id"):
        db.complete_run_metrics(
            run_id,
            {
                name: {key: value for key, value in item.items() if key != "value_sum"}
                for name, item in summary.items()
            },
        )
    return result


def _run_frame_metric_batches(
    *,
    db: Database,
    workspace: WorkspaceConfig,
    job_id: int,
    run_id: int | None,
    metric_name: str,
    metric_device: str,
    artifacts: list[dict[str, Any]],
    samples: dict[int, dict[str, Any]],
    dataset: dict[str, Any],
    cache_config: dict[str, Any],
    summary: dict[str, Any],
    values: list[float],
    current: int,
    total: int,
    requested_batch_size: int | None,
    identity_cache: dict[Path, dict[str, Any]],
    job_payload: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    outcomes: dict[int, tuple[str, float | None, dict[str, Any]]] = {}
    pending: list[dict[str, Any]] = []
    cache_hits = 0
    for index, artifact in enumerate(artifacts):
        sample_id = artifact.get("sample_id")
        sample = samples.get(int(sample_id)) if sample_id is not None else None
        reference_path = Path(sample["gt_path"]) if sample and sample.get("gt_path") else None
        if reference_path is None:
            outcomes[index] = ("skipped", None, {"reason": "sample has no ground-truth reference"})
            continue
        distorted_path = Path(artifact["path"])
        config: dict[str, Any] = {
            "cache_version": METRIC_CACHE_VERSION,
            "metric": cache_config,
            "metric_device": metric_device,
        }
        alignment = _metric_alignment_context(dataset, sample)
        if alignment:
            config["alignment"] = alignment
        cache_key = metric_cache_key(
            metric_name,
            reference_path,
            distorted_path,
            config,
            identity_cache=identity_cache,
        )
        cached = db.get_metric_cache(cache_key)
        if cached:
            cache_hits += 1
            outcomes[index] = (cached["status"], cached["value"], {"cached": True, **cached["details"]})
            continue
        pending.append(
            {
                "index": index,
                "sample_id": sample_id,
                "reference": reference_path,
                "distorted": distorted_path,
                "cache_key": cache_key,
                "work_dir": workspace.tmp_dir / "metrics" / metric_name / hashlib.sha1(cache_key.encode("utf-8")).hexdigest(),
            }
        )

    metric = create_metric(metric_name, workspace, device=metric_device) if pending else None
    default_batch = 8 if metric_name == "lpips_vit_patch" else 32
    batch_size = requested_batch_size or default_batch
    effective_batch = batch_size
    offset = 0
    unavailable_reason: str | None = None
    if outcomes:
        _publish_metric_progress(db, job_id, run_id, current + len(outcomes), total, job_payload)
    while offset < len(pending):
        _raise_if_canceled(db, run_id, job_id)
        size = min(batch_size, len(pending) - offset)
        rows = pending[offset : offset + size]
        try:
            if hasattr(metric, "evaluate_batch"):
                results = metric.evaluate_batch(
                    [(row["reference"], row["distorted"], row["work_dir"]) for row in rows]
                )
            else:
                results = [metric.evaluate(row["reference"], row["distorted"], row["work_dir"]) for row in rows]
            if len(results) != len(rows):
                raise RuntimeError("metric batch returned an unexpected result count")
            for row, result in zip(rows, results):
                db.set_metric_cache(row["cache_key"], metric_name, result.status, result.value, result.details)
                outcomes[row["index"]] = (result.status, result.value, dict(result.details))
            effective_batch = min(effective_batch, size)
            offset += size
            _publish_metric_progress(db, job_id, run_id, current + len(outcomes), total, job_payload)
        except MetricBatchOutOfMemory as exc:
            if size > 1:
                batch_size = max(1, size // 2)
                effective_batch = min(effective_batch, batch_size)
                continue
            unavailable_reason = str(exc)
            row = rows[0]
            details = {"reason": unavailable_reason, "device": metric_device, "batch_size": 1}
            db.set_metric_cache(row["cache_key"], metric_name, "unavailable", None, details)
            outcomes[row["index"]] = ("unavailable", None, details)
            offset += 1
            _publish_metric_progress(db, job_id, run_id, current + len(outcomes), total, job_payload)
        except MetricUnavailable as exc:
            unavailable_reason = str(exc)
            for row in pending[offset:]:
                details = {"reason": unavailable_reason, "device": metric_device}
                db.set_metric_cache(row["cache_key"], metric_name, "unavailable", None, details)
                outcomes[row["index"]] = ("unavailable", None, details)
            offset = len(pending)
            _publish_metric_progress(db, job_id, run_id, current + len(outcomes), total, job_payload)
        except Exception as exc:
            for row in rows:
                outcomes[row["index"]] = (
                    "failed",
                    None,
                    {"reason": str(exc), "type": type(exc).__name__, "sample_id": row["sample_id"]},
                )
            offset += size
            _publish_metric_progress(db, job_id, run_id, current + len(outcomes), total, job_payload)

    for index, artifact in enumerate(artifacts):
        sample_id = artifact.get("sample_id")
        sample = samples.get(int(sample_id)) if sample_id is not None else None
        status, value, raw_details = outcomes[index]
        details = {**_compare_track_details(sample, artifact), **raw_details}
        metric_result_id = db.add_metric_result(
            job_id=job_id,
            inference_job_id=int(artifact["job_id"]),
            sample_id=int(sample_id) if sample_id is not None else None,
            metric_name=metric_name,
            status=status,
            value=value,
            details=details,
        )
        if run_id is not None:
            sample_metadata = (sample or {}).get("metadata") or {}
            video_name = str(sample_metadata.get("video_name") or sample_metadata.get("video_file") or "")
            track_label = str(details.get("compare_track_label") or "")
            reference_asset_id, distorted_asset_id = run_asset_pair(db, run_id, video_name, track_label)
            bind_metric_result(
                db,
                metric_result_id,
                reference_asset_id,
                distorted_asset_id,
                video_name=video_name,
                track_label=track_label,
            )
        summary[status] = int(summary.get(status, 0)) + 1
        if status == "completed" and value is not None:
            values.append(float(value))
        current += 1

    elapsed = time.perf_counter() - started
    adapter_performance = metric.performance() if metric is not None and hasattr(metric, "performance") else {}
    return current, {
        "device": metric_device,
        "pairs": len(artifacts),
        "cache_hits": cache_hits,
        "requested_batch_size": requested_batch_size,
        "initial_batch_size": requested_batch_size or default_batch,
        "effective_batch_size": effective_batch if pending else 0,
        "elapsed_seconds": elapsed,
        "pairs_per_second": (len(artifacts) / elapsed) if elapsed > 0 else 0.0,
        "unavailable_reason": unavailable_reason,
        **adapter_performance,
    }


def _publish_metric_progress(
    db: Database,
    job_id: int,
    run_id: int | None,
    current: int,
    total: int,
    payload: dict[str, Any],
) -> None:
    db.update_job_progress(job_id, current, total)
    if run_id is None:
        return
    if payload.get("metric_wave_id"):
        from vfieval.pipeline.metric_jobs import update_metric_wave_progress

        update_metric_wave_progress(db, run_id, str(payload["metric_wave_id"]))
    else:
        db.update_run_progress(run_id, current, total, "metric_running")


def _positive_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _raise_if_canceled(db: Database, run_id: int | None, job_id: int) -> None:
    if run_id is None:
        return
    run = db.get_run(run_id)
    if run["status"] in {"cancel_requested", "canceled"}:
        error = {"message": "用户取消了 Run", "type": "RunCanceled"}
        db.cancel_job(job_id, error)
        db.cancel_run(run_id, error)
        raise RunCanceled("用户取消了 Run")


def _compare_track_details(sample: dict[str, Any] | None, artifact: dict[str, Any] | None) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for source in ((sample or {}).get("metadata") or {}, (artifact or {}).get("metadata") or {}):
        for key in (
            "compare_track_label",
            "compare_track_key",
            "compare_track_index",
            "compare_track_run_id",
            "compare_track_artifact_id",
        ):
            if key in source and source[key] is not None:
                details[key] = source[key]
    return details


def _metric_alignment_context(
    dataset: dict[str, Any],
    sample: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    """Return the durable Alignment Plan identity for metric cache isolation.

    Compare metrics operate on explicitly materialized aligned frames.  Their
    file identities usually differ too, but the Alignment Plan fingerprint is
    the actual transform contract and must be part of the cache key.  Normal
    inference datasets have no plan and intentionally keep their existing key
    shape.
    """
    dataset_metadata = dataset.get("metadata") or {}
    plan = dataset_metadata.get("alignment_plan") if isinstance(dataset_metadata, dict) else None
    plan_fingerprint = str(plan.get("fingerprint") or "") if isinstance(plan, dict) else ""
    sample_metadata = (sample or {}).get("metadata") or {}
    sample_fingerprint = (
        str(sample_metadata.get("alignment_fingerprint") or "")
        if isinstance(sample_metadata, dict)
        else ""
    )
    if not plan_fingerprint and not sample_fingerprint:
        return None
    context: dict[str, str] = {}
    if plan_fingerprint:
        context["plan_fingerprint"] = plan_fingerprint
    if sample_fingerprint:
        context["sample_fingerprint"] = sample_fingerprint
    return context


def _collect_video_metric_inputs(
    *,
    db: Database,
    workspace: WorkspaceConfig,
    run_id: int | None,
    dataset: dict[str, Any],
    samples: dict[int, dict[str, Any]],
    inference_job_ids: list[int],
    pred_artifacts: list[dict[str, Any]],
    pred_videos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve metric-only video pairs without changing artifact publication.

    Ordinary inference and legacy Compare Runs expose ``pred_video`` artifacts,
    which remain the preferred inputs.  An Item Compare deliberately does not:
    publishing its aligned Pred would make a derived comparison output appear
    reusable.  In that case the already-aligned per-sample PNGs are encoded
    into private, rebuildable ``compare_cache`` files instead.  These cache
    files are not artifacts and never enter the media catalog.
    """
    if pred_videos:
        return _published_video_metric_inputs(db, dataset, inference_job_ids, pred_videos)
    if not _is_item_compare_without_pred_video(db, run_id, dataset, samples):
        return []
    return _item_compare_video_metric_inputs(
        db=db,
        workspace=workspace,
        run_id=run_id,
        dataset=dataset,
        samples=samples,
        inference_job_ids=inference_job_ids,
        pred_artifacts=pred_artifacts,
    )


def _published_video_metric_inputs(
    db: Database,
    dataset: dict[str, Any],
    inference_job_ids: list[int],
    pred_videos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gt_videos: list[dict[str, Any]] = []
    for current_job_id in inference_job_ids:
        gt_videos.extend(db.list_artifacts(job_id=current_job_id, kind="gt_video"))
    gt_by_name = {
        artifact.get("metadata", {}).get("video_name"): artifact
        for artifact in gt_videos
        if artifact.get("metadata", {}).get("video_name")
    }
    alignment_context = _metric_alignment_context(dataset)
    inputs: list[dict[str, Any]] = []
    for artifact in pred_videos:
        video_name = artifact.get("metadata", {}).get("video_name")
        reference_artifact = gt_by_name.get(video_name)
        inputs.append(
            {
                "inference_job_id": int(artifact["job_id"]),
                "video_name": video_name,
                "reference_path": Path(reference_artifact["path"]) if reference_artifact is not None else None,
                "distorted_path": Path(artifact["path"]),
                "track_details": _compare_track_details(None, artifact),
                "alignment_context": alignment_context,
                "input_details": {"video_input": "published_pred_video"},
            }
        )
    return inputs


def _is_item_compare_without_pred_video(
    db: Database,
    run_id: int | None,
    dataset: dict[str, Any],
    samples: dict[int, dict[str, Any]],
) -> bool:
    """Recognize the 0711 Item Compare contract without trusting filenames."""
    if run_id is not None:
        try:
            run_metadata = db.get_run(int(run_id)).get("metadata") or {}
        except KeyError:
            run_metadata = {}
        request = run_metadata.get("request") or {}
        if not isinstance(request, dict):
            request = {}
        if (
            str(run_metadata.get("run_type") or "") == "video_compare"
            and (
                run_metadata.get("media_item_id") not in {None, "", 0}
                or request.get("media_item_id") not in {None, "", 0}
            )
            and not bool(
                run_metadata.get(
                    "publish_compare_pred_video",
                    request.get("publish_compare_pred_video", False),
                )
            )
        ):
            return True

    # The metadata check above is the normal server path.  This conservative
    # fallback keeps an interrupted/recovered Item Compare evaluable when the
    # run metadata predates the flag but its dataset still carries the explicit
    # Alignment Plan and Compare sample rows.  It cannot match a normal model
    # inference dataset because those have neither condition.
    dataset_metadata = dataset.get("metadata") or {}
    return bool(
        isinstance(dataset_metadata, dict)
        and isinstance(dataset_metadata.get("alignment_plan"), dict)
        and any(
            str((sample.get("metadata") or {}).get("source_type") or "") == "compare"
            for sample in samples.values()
        )
    )


def _item_compare_video_metric_inputs(
    *,
    db: Database,
    workspace: WorkspaceConfig,
    run_id: int | None,
    dataset: dict[str, Any],
    samples: dict[int, dict[str, Any]],
    inference_job_ids: list[int],
    pred_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one internal normalized video pair per Item Compare track.

    The sample rows already point to frames materialized by the shared
    Alignment Plan.  We deliberately use the Run's ``gt``/``pred`` image
    artifacts where available so metric inputs exactly match the visible Diff
    inputs, then validate every frame against the plan target before encoding.
    A damaged or partial track becomes a per-track metric failure rather than
    a silent truncation.
    """
    dataset_metadata = dataset.get("metadata") or {}
    plan = dataset_metadata.get("alignment_plan") if isinstance(dataset_metadata, dict) else None
    fingerprint = str(plan.get("fingerprint") or "") if isinstance(plan, dict) else ""
    target = plan.get("target") if isinstance(plan, dict) else None
    try:
        target_size = (int((target or {}).get("width") or 0), int((target or {}).get("height") or 0))
    except (TypeError, ValueError):
        target_size = (0, 0)

    pred_by_sample = {
        int(artifact["sample_id"]): artifact
        for artifact in pred_artifacts
        if artifact.get("sample_id") is not None
    }
    gt_by_sample = {
        int(artifact["sample_id"]): artifact
        for artifact in db.list_artifacts_by_samples(
            samples.keys(),
            job_ids=inference_job_ids,
            kind="gt",
        )
        if artifact.get("sample_id") is not None
    }
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for sample in samples.values():
        metadata = sample.get("metadata") or {}
        if str(metadata.get("source_type") or "") != "compare":
            continue
        video_name = str(metadata.get("video_name") or metadata.get("compare_group") or "compare")
        try:
            track_index = int(metadata.get("compare_track_index") or 0)
        except (TypeError, ValueError):
            track_index = 0
        track_key = str(metadata.get("compare_track_key") or f"pred_{chr(ord('a') + max(0, track_index))}")
        groups.setdefault((video_name, track_index, track_key), []).append(sample)

    expected_count = 0
    try:
        expected_count = int(((plan or {}).get("temporal") or {}).get("frame_count") or 0)
    except (AttributeError, TypeError, ValueError):
        expected_count = 0
    plan_error = ""
    if not fingerprint:
        plan_error = "Item Compare dataset has no Alignment Plan fingerprint"
    elif target_size[0] <= 0 or target_size[1] <= 0:
        plan_error = "Item Compare Alignment Plan has no valid target dimensions"

    inputs: list[dict[str, Any]] = []
    for (video_name, track_index, track_key), track_samples in sorted(groups.items()):
        ordered = sorted(track_samples, key=_compare_sample_order)
        first_sample = ordered[0]
        first_artifact = pred_by_sample.get(int(first_sample["id"]))
        track_metadata = first_sample.get("metadata") or {}
        track_label = str(track_metadata.get("compare_track_label") or f"Pred {track_index + 1}")
        track_details = _compare_track_details(first_sample, first_artifact)
        track_details.setdefault("compare_track_label", track_label)
        track_details.setdefault("compare_track_key", track_key)
        track_details.setdefault("compare_track_index", track_index)
        inference_job_id = int(
            (first_artifact or {}).get("job_id")
            or (inference_job_ids[0] if inference_job_ids else 0)
        )
        record: dict[str, Any] = {
            "inference_job_id": inference_job_id,
            "video_name": video_name,
            "track_details": track_details,
            "alignment_context": _metric_alignment_context(dataset, first_sample),
            "input_details": {
                "video_input": "aligned_compare_cache",
                "alignment_fingerprint": fingerprint or None,
                "alignment_slot": track_key,
            },
        }
        try:
            if plan_error:
                raise ValueError(plan_error)
            if inference_job_id <= 0:
                raise ValueError("Item Compare track has no inference job")
            frame_indices = [_compare_sample_frame_index(sample) for sample in ordered]
            if len(set(frame_indices)) != len(frame_indices):
                raise ValueError("Item Compare track contains duplicate frame indices")
            if expected_count > 0 and len(ordered) != expected_count:
                raise ValueError(
                    "Item Compare track frame count does not match Alignment Plan: "
                    f"expected {expected_count}, got {len(ordered)}"
                )
            fps = _compare_track_fps(ordered, plan)
            gt_paths: list[Path] = []
            pred_paths: list[Path] = []
            for sample in ordered:
                sample_id = int(sample["id"])
                pred_artifact = pred_by_sample.get(sample_id)
                if pred_artifact is None:
                    raise ValueError(f"Item Compare track is missing pred artifact for sample {sample_id}")
                pred_paths.append(Path(str(pred_artifact["path"])).resolve())
                gt_artifact = gt_by_sample.get(sample_id)
                gt_path = Path(str(gt_artifact["path"])).resolve() if gt_artifact is not None else Path(
                    str(sample.get("gt_path") or "")
                ).resolve()
                gt_paths.append(gt_path)
            _validate_aligned_video_frames("GT", gt_paths, target_size)
            _validate_aligned_video_frames(track_label, pred_paths, target_size)
            record["reference_path"] = _materialize_compare_metric_video(
                db=db,
                workspace=workspace,
                run_id=run_id,
                plan_fingerprint=fingerprint,
                video_name=video_name,
                track_key=track_key,
                role="gt",
                frame_paths=gt_paths,
                fps=fps,
            )
            record["distorted_path"] = _materialize_compare_metric_video(
                db=db,
                workspace=workspace,
                run_id=run_id,
                plan_fingerprint=fingerprint,
                video_name=video_name,
                track_key=track_key,
                role="pred",
                frame_paths=pred_paths,
                fps=fps,
            )
        except Exception as exc:
            record["error"] = f"aligned Compare video input unavailable: {exc}"
        inputs.append(record)
    return inputs


def _compare_sample_order(sample: dict[str, Any]) -> tuple[int, int, int]:
    metadata = sample.get("metadata") or {}
    return (
        _compare_sample_frame_index(sample),
        _safe_int(metadata.get("sample_index"), 0),
        int(sample["id"]),
    )


def _compare_sample_frame_index(sample: dict[str, Any]) -> int:
    metadata = sample.get("metadata") or {}
    return _safe_int(metadata.get("frame_index", metadata.get("sample_index", 0)), 0)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _compare_track_fps(samples: list[dict[str, Any]], plan: dict[str, Any] | None) -> float:
    values: list[float] = []
    for sample in samples:
        raw = (sample.get("metadata") or {}).get("fps")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(value)
    if values and any(abs(value - values[0]) > 1e-6 for value in values[1:]):
        raise ValueError("Item Compare track has inconsistent frame fps")
    if values:
        return values[0]
    try:
        planned = float(((plan or {}).get("temporal") or {}).get("fps"))
    except (AttributeError, TypeError, ValueError):
        planned = 0.0
    return planned if planned > 0 else 24.0


def _validate_aligned_video_frames(
    label: str,
    frame_paths: list[Path],
    target_size: tuple[int, int],
) -> None:
    if not frame_paths:
        raise ValueError(f"{label} has no aligned frames")
    for index, path in enumerate(frame_paths):
        if not path.is_file():
            raise FileNotFoundError(f"{label} frame {index} is unavailable: {path}")
        with Image.open(path) as image:
            if image.size != target_size:
                raise ValueError(
                    f"{label} frame {index} does not match Alignment Plan target "
                    f"{target_size[0]}x{target_size[1]}: got {image.size[0]}x{image.size[1]}"
                )


def _materialize_compare_metric_video(
    *,
    db: Database,
    workspace: WorkspaceConfig,
    run_id: int | None,
    plan_fingerprint: str,
    video_name: str,
    track_key: str,
    role: str,
    frame_paths: list[Path],
    fps: float,
) -> Path:
    """Encode a private normalized metric input under managed compare_cache.

    This intentionally has no ``db.add_artifact`` call.  The output is a
    transient/rebuildable cache entry used only while evaluating a video metric,
    never a reusable ``pred_video`` or catalog source.
    """
    # A valid Item Compare has one canonical GT sequence shared by every
    # strictly aligned prediction. Keep it as one cache object rather than
    # encoding an identical GT video once per track. Frame signatures remain in
    # the key, so a malformed legacy/recovered dataset with different GT paths
    # still cannot collide.
    cache_track_key = "gt" if str(role) == "gt" else str(track_key)
    signatures = []
    for path in frame_paths:
        stat = path.stat()
        signatures.append(
            {
                "path": path.resolve().as_posix(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "version": "item-compare-video-metric-v1",
                "run_id": int(run_id) if run_id is not None else None,
                "alignment_fingerprint": str(plan_fingerprint),
                "video_name": str(video_name),
                "track_key": cache_track_key,
                "role": str(role),
                "fps": float(fps),
                "frames": signatures,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    cache_root = (workspace.root / "compare_cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    output = (cache_root / f"{cache_key}.mp4").resolve()
    if output.parent != cache_root:
        raise ValueError("invalid Compare metric cache path")

    from vfieval.pipeline.inference import _write_mp4
    from vfieval.run_cleanup import CACHE_GRACE_SECONDS, cache_lease

    with cache_lease(db, workspace, "compare_cache", cache_key, output):
        if not output.is_file() or output.stat().st_size <= 0:
            temporary = output.with_name(f"{output.stem}.{uuid.uuid4().hex}.tmp.mp4")
            try:
                _write_mp4(frame_paths, temporary, float(fps))
                if not temporary.is_file() or temporary.stat().st_size <= 0:
                    raise RuntimeError("failed to encode normalized Compare metric video")
                os.replace(temporary, output)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
    # cache_lease refreshes its entry when leaving the context. Publish the
    # producer metadata afterwards so the generic runtime lease marker cannot
    # overwrite the Alignment Plan provenance.
    stat = output.stat()
    now = time.time()
    db.upsert_cache_entry(
        "compare_cache",
        cache_key,
        output,
        state="ready",
        size_bytes=int(stat.st_size),
        metadata={
            "purpose": "item_compare_video_metric_input",
            "run_id": int(run_id) if run_id is not None else None,
            "alignment_fingerprint": str(plan_fingerprint),
            "video_name": str(video_name),
            "track_key": cache_track_key,
            "role": str(role),
        },
        last_used_at=now,
        gc_after=now + CACHE_GRACE_SECONDS,
    )
    return output


@contextmanager
def _lease_metric_video_inputs(
    db: Database,
    workspace: WorkspaceConfig,
    paths: list[Path],
):
    """Keep internal compare-cache videos alive for the whole metric call.

    ``_materialize_compare_metric_video`` holds a lease while encoding, but a
    video metric can be substantially longer than the normal GC grace period.
    Retaining a short-lived lease here prevents storage GC from deleting a
    rebuildable input while VMAF/CGVQM is reading it.  Published videos live in
    Run storage and are deliberately not treated as cache entries.
    """
    from vfieval.run_cleanup import CACHE_GRACE_SECONDS, cache_lease

    cache_root = (workspace.root / "compare_cache").resolve()
    stack = ExitStack()
    restored_metadata: list[tuple[str, Path, dict[str, Any]]] = []
    seen: set[Path] = set()
    try:
        for raw_path in paths:
            path = Path(raw_path).resolve()
            if path in seen or path.parent != cache_root or path.suffix.lower() != ".mp4":
                continue
            seen.add(path)
            existing = db.get_cache_entry("compare_cache", path.stem)
            if existing is None:
                existing = next(
                    (
                        row
                        for row in db.list_cache_entries()
                        if row.get("cache_type") == "compare_cache"
                        and Path(str(row.get("storage_path") or "")).resolve() == path
                    ),
                    None,
                )
            if existing is not None:
                restored_metadata.append((str(existing["cache_key"]), path, dict(existing.get("metadata") or {})))
            stack.enter_context(cache_lease(db, workspace, "compare_cache", path.stem, path))
        yield
    finally:
        stack.close()
        # cache_lease intentionally stamps a generic runtime marker while it
        # acquires the entry. Restore the producer metadata after the metric so
        # storage diagnostics retain the Alignment Plan and private-purpose
        # provenance added by the materializer.
        now = time.time()
        for cache_key, path, metadata in restored_metadata:
            current = db.get_cache_entry("compare_cache", cache_key)
            if current is None or current.get("state") == "deleting" or not path.is_file():
                continue
            db.upsert_cache_entry(
                "compare_cache",
                cache_key,
                path,
                state=str(current.get("state") or "ready"),
                size_bytes=int(path.stat().st_size),
                metadata=metadata,
                last_used_at=now,
                gc_after=now + CACHE_GRACE_SECONDS,
            )


def _evaluate_with_cache(
    db: Database,
    workspace: WorkspaceConfig,
    metric_name: str,
    reference_path: Path,
    distorted_path: Path,
    sample_id: int | None,
    cache_config: dict[str, Any],
    metric_device: str = "cpu",
    alignment_context: dict[str, str] | None = None,
) -> tuple[str, float | None, dict[str, Any]]:
    config = {
        "cache_version": METRIC_CACHE_VERSION,
        "metric": cache_config,
        "metric_device": metric_device,
    }
    if alignment_context:
        config["alignment"] = dict(alignment_context)
    cache_key = metric_cache_key(metric_name, reference_path, distorted_path, config)
    cached = db.get_metric_cache(cache_key)
    if cached:
        return cached["status"], cached["value"], {"cached": True, **cached["details"]}

    metric = create_metric(metric_name, workspace, device=metric_device)
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
    identity_cache: dict[Path, dict[str, Any]] | None = None,
) -> str:
    data = {
        "metric": metric_name,
        "adapter_version": METRIC_CACHE_VERSION,
        "reference": _memoized_file_identity(reference_path, identity_cache),
        "distorted": _memoized_file_identity(distorted_path, identity_cache),
        "config": config,
    }
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _memoized_file_identity(
    path: Path,
    cache: dict[Path, dict[str, Any]] | None,
) -> dict[str, Any]:
    resolved = path.resolve()
    if cache is None:
        return _file_identity(resolved)
    if resolved not in cache:
        cache[resolved] = _file_identity(resolved)
    return cache[resolved]


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
