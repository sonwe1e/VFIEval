from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from vfieval.db import Database


REPORT_SCHEMA = "artifact-integrity-v1"
SHARD_MANIFEST_SCHEMA = "artifact-shard-v1"
CANONICAL_ARTIFACT_CONTRACT = "canonical-v1"
VIDEO_KINDS = {"pred_video", "gt_video", "diff_video"}
CANONICAL_IMAGE_KINDS = {"pred", "gt", "difference", "warp0", "warp1", "blend"}
CORE_SAMPLE_KINDS = {
    "pred",
    "gt",
    "difference",
    "flowt_0",
    "flowt_1",
    "mask0",
    "mask1",
    "warp0",
    "warp1",
    "blend",
}
VIDEO_FPS_TOLERANCE = 1e-6
VIDEO_TIMESTAMP_TOLERANCE = 1e-3


def strict_video_pair_issue(
    pred_artifact: Mapping[str, Any],
    gt_artifact: Mapping[str, Any],
    *,
    observations: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return a concrete integrity issue when a Pred/GT video pair is not exact.

    Spatial resizing is deliberately not part of this check: metrics may only
    consume already-aligned videos.  Exact probes are authoritative for the
    container streams; artifact metadata supplies timestamps when both sides
    publish them and also catches stale or inconsistent declarations.
    """

    observed_by_id = observations or {}
    inspected: dict[str, Mapping[str, Any]] = {}
    for role, artifact in (("Pred", pred_artifact), ("GT", gt_artifact)):
        artifact_id = int(artifact["id"])
        observed = observed_by_id.get(artifact_id)
        if observed is None:
            path = Path(str(artifact.get("path") or ""))
            if not path.is_file() or path.stat().st_size <= 0:
                return _video_pair_issue(
                    pred_artifact,
                    gt_artifact,
                    "video_pair_file_unavailable",
                    f"{role} video file is missing or empty",
                    role=role.lower(),
                )
            try:
                from vfieval.file_inputs import inspect_video

                observed = inspect_video(path, exact=True)
            except Exception as exc:
                return _video_pair_issue(
                    pred_artifact,
                    gt_artifact,
                    "video_pair_probe_failed",
                    f"could not inspect {role} video: {exc}",
                    role=role.lower(),
                    error_type=type(exc).__name__,
                )
        if not observed.get("decodable"):
            return _video_pair_issue(
                pred_artifact,
                gt_artifact,
                "video_pair_probe_failed",
                f"{role} video is not decodable: {observed.get('error') or 'unknown error'}",
                role=role.lower(),
            )
        inspected[role.lower()] = observed

    mismatches: dict[str, dict[str, Any]] = {}
    pred_observed = inspected["pred"]
    gt_observed = inspected["gt"]
    for field in ("frame_count", "width", "height"):
        pred_value = int(pred_observed.get(field) or 0)
        gt_value = int(gt_observed.get(field) or 0)
        if pred_value <= 0 or gt_value <= 0 or pred_value != gt_value:
            mismatches[f"observed_{field}"] = {"pred": pred_value, "gt": gt_value}
    pred_fps = float(pred_observed.get("fps") or 0.0)
    gt_fps = float(gt_observed.get("fps") or 0.0)
    if (
        pred_fps <= 0
        or gt_fps <= 0
        or abs(pred_fps - gt_fps) > VIDEO_FPS_TOLERANCE
    ):
        mismatches["observed_fps"] = {"pred": pred_fps, "gt": gt_fps}

    pred_metadata = dict(pred_artifact.get("metadata") or {})
    gt_metadata = dict(gt_artifact.get("metadata") or {})
    for field in ("frames", "width", "height"):
        pred_value = _optional_positive_number(pred_metadata.get(field), integer=True)
        gt_value = _optional_positive_number(gt_metadata.get(field), integer=True)
        if pred_value is not None and gt_value is not None and pred_value != gt_value:
            mismatches[f"declared_{field}"] = {"pred": pred_value, "gt": gt_value}
    declared_pred_fps = _optional_positive_number(pred_metadata.get("fps"), integer=False)
    declared_gt_fps = _optional_positive_number(gt_metadata.get("fps"), integer=False)
    if (
        declared_pred_fps is not None
        and declared_gt_fps is not None
        and abs(float(declared_pred_fps) - float(declared_gt_fps)) > VIDEO_FPS_TOLERANCE
    ):
        mismatches["declared_fps"] = {"pred": declared_pred_fps, "gt": declared_gt_fps}

    pred_timestamps = _artifact_video_timestamps(pred_metadata)
    gt_timestamps = _artifact_video_timestamps(gt_metadata)
    if pred_timestamps is not None and gt_timestamps is not None:
        if len(pred_timestamps) != len(gt_timestamps):
            mismatches["timestamps_length"] = {
                "pred": len(pred_timestamps),
                "gt": len(gt_timestamps),
            }
        else:
            first_mismatch = next(
                (
                    index
                    for index, (pred_value, gt_value) in enumerate(zip(pred_timestamps, gt_timestamps))
                    if abs(pred_value - gt_value) > VIDEO_TIMESTAMP_TOLERANCE
                ),
                None,
            )
            if first_mismatch is not None:
                mismatches["timestamps"] = {
                    "index": int(first_mismatch),
                    "pred": pred_timestamps[first_mismatch],
                    "gt": gt_timestamps[first_mismatch],
                }

    if not mismatches:
        return None
    return _video_pair_issue(
        pred_artifact,
        gt_artifact,
        "video_pair_mismatch",
        "Pred and GT videos do not have strict temporal and spatial identity",
        mismatches=mismatches,
    )


def _video_pair_issue(
    pred_artifact: Mapping[str, Any],
    gt_artifact: Mapping[str, Any],
    code: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    pred_metadata = dict(pred_artifact.get("metadata") or {})
    return {
        "code": str(code),
        "message": str(message),
        "video_name": str(pred_metadata.get("video_name") or ""),
        "pred_artifact_id": int(pred_artifact["id"]),
        "gt_artifact_id": int(gt_artifact["id"]),
        **details,
    }


def _optional_positive_number(value: Any, *, integer: bool) -> int | float | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value) if integer else float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _artifact_video_timestamps(metadata: Mapping[str, Any]) -> list[float] | None:
    for key in ("timestamps", "source_timestamps"):
        raw = metadata.get(key)
        if not isinstance(raw, (list, tuple)) or not raw:
            continue
        try:
            return [float(value) for value in raw]
        except (TypeError, ValueError):
            return None
    return None


def _canonical_contract_required(db: Database, job: Mapping[str, Any]) -> bool:
    payload = dict(job.get("payload") or {})
    if str(payload.get("artifact_contract") or "") == CANONICAL_ARTIFACT_CONTRACT:
        return True
    run_id = payload.get("run_id")
    if run_id is None:
        return False
    try:
        run = db.get_run(int(run_id))
    except (KeyError, ValueError):
        return False
    metadata = dict(run.get("metadata") or {})
    request = dict(metadata.get("request") or {})
    return str(metadata.get("artifact_contract") or request.get("artifact_contract") or "") == CANONICAL_ARTIFACT_CONTRACT


def _canonical_dimensions(db: Database, job: Mapping[str, Any]) -> tuple[int, int] | None:
    payload = dict(job.get("payload") or {})
    run_id = payload.get("run_id")
    if run_id is None:
        return None
    try:
        run = db.get_run(int(run_id))
    except (KeyError, ValueError):
        return None
    height = int(run.get("height") or 0)
    width = int(run.get("width") or 0)
    return (height, width) if height > 0 and width > 0 else None


class ArtifactIntegrityError(RuntimeError):
    """Raised when required Run artifacts are incomplete or ambiguous."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        errors = list(self.report.get("errors") or [])
        summary = "; ".join(str(item.get("message") or item.get("code") or "artifact error") for item in errors[:3])
        if len(errors) > 3:
            summary += f"; and {len(errors) - 3} more"
        super().__init__(f"artifact integrity validation failed: {summary or 'unknown error'}")


def _report(scope: str, **details: Any) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA,
        "scope": scope,
        **details,
        "errors": [],
        "warnings": [],
        "valid": True,
    }


def _issue(report: dict[str, Any], severity: str, code: str, message: str, **details: Any) -> None:
    report[severity].append({"code": code, "message": message, **details})


def _finish(report: dict[str, Any]) -> dict[str, Any]:
    report["error_count"] = len(report.get("errors") or [])
    report["warning_count"] = len(report.get("warnings") or [])
    report["valid"] = not bool(report["error_count"])
    return report


def merge_integrity_reports(scope: str, reports: Iterable[Mapping[str, Any]], **details: Any) -> dict[str, Any]:
    merged = _report(scope, **details)
    children = [dict(report) for report in reports]
    merged["checks"] = children
    for child in children:
        merged["errors"].extend(child.get("errors") or [])
        merged["warnings"].extend(child.get("warnings") or [])
    return _finish(merged)


def write_integrity_report(path: Path, report: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def expected_sample_ids_for_job(db: Database, job: Mapping[str, Any]) -> list[int]:
    payload = dict(job.get("payload") or {})
    explicit = payload.get("sample_ids")
    if explicit is not None:
        ids = [int(sample_id) for sample_id in explicit]
    elif payload.get("dataset_id") is not None:
        ids = [int(sample["id"]) for sample in db.list_samples(int(payload["dataset_id"]))]
    else:
        ids = []
    if str(payload.get("artifact_profile") or "evaluation") == "benchmark":
        ids = ids[: max(1, int(payload.get("benchmark_samples") or 200))]
    return ids


def _run_type(db: Database, payload: Mapping[str, Any]) -> str:
    run_type = str(payload.get("run_type") or "")
    if run_type:
        return run_type
    if payload.get("run_id") is None:
        return "model_inference"
    try:
        run = db.get_run(int(payload["run_id"]))
    except (KeyError, ValueError):
        return "model_inference"
    return str((run.get("metadata") or {}).get("run_type") or "model_inference")


def required_sample_kinds(
    db: Database,
    job: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> set[str]:
    payload = dict(job.get("payload") or {})
    profile = str(payload.get("artifact_profile") or "evaluation")
    if profile == "benchmark":
        return set()
    if _run_type(db, payload) == "video_compare":
        return {"pred", "gt", "difference"}

    required = {"pred"}
    if sample.get("gt_path"):
        required.update({"gt", "difference"})
    if profile == "diagnostic":
        required.update({"flowt_0", "flowt_1", "mask0", "mask1", "warp0", "warp1", "blend"})
    elif bool(payload.get("save_warp_blend")):
        required.update({"warp0", "warp1", "blend"})
    return required


def validate_job_artifact_integrity(
    db: Database,
    job_id: int,
    *,
    expected_sample_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    job = db.get_job(int(job_id))
    payload = dict(job.get("payload") or {})
    expected = [int(sample_id) for sample_id in (
        expected_sample_ids if expected_sample_ids is not None else expected_sample_ids_for_job(db, job)
    )]
    report = _report(
        "job_artifacts",
        job_id=int(job_id),
        artifact_profile=str(payload.get("artifact_profile") or "evaluation"),
        expected_sample_ids=sorted(expected),
    )
    if len(expected) != len(set(expected)):
        duplicates = sorted(sample_id for sample_id, count in Counter(expected).items() if count > 1)
        _issue(
            report,
            "errors",
            "duplicate_expected_sample",
            f"job {job_id} contains duplicate expected sample ids",
            sample_ids=duplicates,
        )
    expected_set = set(expected)
    samples: dict[int, dict[str, Any]] = {}
    for sample_id in sorted(expected_set):
        try:
            samples[sample_id] = db.get_sample(sample_id)
        except KeyError:
            _issue(
                report,
                "errors",
                "missing_sample",
                f"expected sample {sample_id} does not exist",
                sample_id=sample_id,
            )

    requirements = {
        sample_id: required_sample_kinds(db, job, sample)
        for sample_id, sample in samples.items()
    }
    report["required_kinds_by_sample"] = {
        str(sample_id): sorted(kinds) for sample_id, kinds in sorted(requirements.items())
    }
    artifacts = db.list_artifacts(job_id=int(job_id))
    canonical_required = _canonical_contract_required(db, job) or any(
        str((artifact.get("metadata") or {}).get("artifact_contract") or "") == CANONICAL_ARTIFACT_CONTRACT
        for artifact in artifacts
        if str(artifact.get("kind") or "") in CORE_SAMPLE_KINDS
    )
    canonical_dimensions = _canonical_dimensions(db, job)
    report["artifact_contract"] = CANONICAL_ARTIFACT_CONTRACT if canonical_required else "legacy"
    if canonical_dimensions is not None:
        report["canonical_height"], report["canonical_width"] = canonical_dimensions
    by_sample: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    sample_errors: set[int] = set()
    for artifact in artifacts:
        kind = str(artifact.get("kind") or "")
        sample_id_value = artifact.get("sample_id")
        if kind == "sample_error":
            metadata = artifact.get("metadata") or {}
            sample_id = int(sample_id_value) if sample_id_value is not None else None
            if sample_id is not None:
                sample_errors.add(sample_id)
            _issue(
                report,
                "errors",
                "sample_error",
                str(metadata.get("message") or f"sample {sample_id} save failed"),
                sample_id=sample_id,
                error_type=str(metadata.get("error_type") or ""),
                artifact_id=int(artifact["id"]),
            )
            continue
        if sample_id_value is None:
            if kind in CORE_SAMPLE_KINDS:
                _issue(
                    report,
                    "errors",
                    "unbound_core_artifact",
                    f"core artifact {kind} is not bound to a sample",
                    kind=kind,
                    artifact_id=int(artifact["id"]),
                )
            continue
        sample_id = int(sample_id_value)
        by_sample[sample_id][kind].append(artifact)

    required_union = set().union(*requirements.values()) if requirements else set()
    for sample_id, kinds in by_sample.items():
        if sample_id not in expected_set and any(kind in required_union for kind in kinds):
            _issue(
                report,
                "errors",
                "unexpected_sample_artifact",
                f"job {job_id} published core artifacts for unexpected sample {sample_id}",
                sample_id=sample_id,
            )

    core_path_owners: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_id, kinds in requirements.items():
        for kind in kinds:
            for artifact in by_sample.get(sample_id, {}).get(kind, []):
                path_value = str(artifact.get("path") or "")
                if path_value:
                    core_path_owners[str(Path(path_value).resolve())].append(
                        {
                            "sample_id": int(sample_id),
                            "kind": str(kind),
                            "artifact_id": int(artifact["id"]),
                        }
                    )
    shared_path_samples: set[int] = set()
    for path_value, owners in core_path_owners.items():
        if len({(owner["sample_id"], owner["kind"]) for owner in owners}) <= 1:
            continue
        shared_path_samples.update(int(owner["sample_id"]) for owner in owners)
        _issue(
            report,
            "errors",
            "shared_core_artifact_path",
            "multiple sample/kind identities reference the same core artifact path",
            path=path_value,
            owners=owners,
        )

    core_counts: Counter[str] = Counter()
    successful: list[int] = []
    optional_warning_samples: set[int] = set()
    profile = str(payload.get("artifact_profile") or "evaluation")
    if profile == "benchmark" and artifacts:
        _issue(
            report,
            "errors",
            "benchmark_published_artifacts",
            f"benchmark job {job_id} must not publish artifacts",
            artifact_count=len(artifacts),
        )

    for sample_id, kinds in sorted(requirements.items()):
        sample_ok = sample_id not in sample_errors and sample_id not in shared_path_samples
        for kind in sorted(kinds):
            rows = by_sample.get(sample_id, {}).get(kind, [])
            core_counts[kind] += len(rows)
            if len(rows) != 1:
                sample_ok = False
                code = "missing_core_artifact" if not rows else "duplicate_core_artifact"
                _issue(
                    report,
                    "errors",
                    code,
                    f"sample {sample_id} requires exactly one {kind} artifact, found {len(rows)}",
                    sample_id=sample_id,
                    kind=kind,
                    artifact_ids=[int(row["id"]) for row in rows],
                )
                continue
            artifact = rows[0]
            path = Path(str(artifact.get("path") or ""))
            file_valid = path.is_file() and path.stat().st_size > 0
            if not file_valid:
                sample_ok = False
                _issue(
                    report,
                    "errors",
                    "invalid_core_artifact_file",
                    f"sample {sample_id} {kind} artifact file is missing or empty",
                    sample_id=sample_id,
                    kind=kind,
                    artifact_id=int(artifact["id"]),
                )
            artifact_metadata = artifact.get("metadata") or {}
            if artifact_metadata.get("optional_warnings") and sample_id not in optional_warning_samples:
                optional_warning_samples.add(sample_id)
                for warning in list(artifact_metadata.get("optional_warnings") or []):
                    _issue(
                        report,
                        "warnings",
                        "optional_artifact_failed",
                        str((warning or {}).get("message") or "optional artifact generation failed"),
                        sample_id=sample_id,
                        kind=str((warning or {}).get("kind") or "extra"),
                        error_type=str((warning or {}).get("type") or ""),
                    )
            if canonical_required and kind in CANONICAL_IMAGE_KINDS:
                if str(artifact_metadata.get("artifact_contract") or "") != CANONICAL_ARTIFACT_CONTRACT:
                    sample_ok = False
                    _issue(
                        report,
                        "errors",
                        "missing_canonical_contract",
                        f"sample {sample_id} {kind} is not marked {CANONICAL_ARTIFACT_CONTRACT}",
                        sample_id=sample_id,
                        kind=kind,
                        artifact_id=int(artifact["id"]),
                    )
                if canonical_dimensions is not None:
                    expected_height, expected_width = canonical_dimensions
                    declared_height = int(artifact_metadata.get("canonical_height") or 0)
                    declared_width = int(artifact_metadata.get("canonical_width") or 0)
                    if (declared_height, declared_width) != canonical_dimensions:
                        sample_ok = False
                        _issue(
                            report,
                            "errors",
                            "canonical_metadata_size_mismatch",
                            f"sample {sample_id} {kind} declares {declared_height}x{declared_width}; expected {expected_height}x{expected_width}",
                            sample_id=sample_id,
                            kind=kind,
                            expected_height=expected_height,
                            expected_width=expected_width,
                            actual_height=declared_height,
                            actual_width=declared_width,
                            artifact_id=int(artifact["id"]),
                        )
                    if file_valid:
                        try:
                            with Image.open(path) as image:
                                actual_width, actual_height = image.size
                        except Exception as exc:
                            sample_ok = False
                            _issue(
                                report,
                                "errors",
                                "invalid_canonical_image",
                                f"sample {sample_id} {kind} cannot be decoded: {exc}",
                                sample_id=sample_id,
                                kind=kind,
                                artifact_id=int(artifact["id"]),
                            )
                        else:
                            if (actual_height, actual_width) != canonical_dimensions:
                                sample_ok = False
                                _issue(
                                    report,
                                    "errors",
                                    "canonical_file_size_mismatch",
                                    f"sample {sample_id} {kind} is {actual_height}x{actual_width}; expected {expected_height}x{expected_width}",
                                    sample_id=sample_id,
                                    kind=kind,
                                    expected_height=expected_height,
                                    expected_width=expected_width,
                                    actual_height=actual_height,
                                    actual_width=actual_width,
                                    artifact_id=int(artifact["id"]),
                                )
            if artifact_metadata.get("preview_warning"):
                warning = artifact_metadata["preview_warning"]
                _issue(
                    report,
                    "warnings",
                    "optional_preview_failed",
                    str((warning or {}).get("message") or "optional preview generation failed"),
                    sample_id=sample_id,
                    kind=kind,
                    artifact_id=int(artifact["id"]),
                    error_type=str((warning or {}).get("type") or ""),
                )
            preview_value = artifact_metadata.get("preview_path")
            if preview_value:
                preview_path = Path(str(preview_value))
                if not preview_path.is_file() or preview_path.stat().st_size <= 0:
                    _issue(
                        report,
                        "warnings",
                        "invalid_optional_preview",
                        f"sample {sample_id} {kind} preview is missing or empty",
                        sample_id=sample_id,
                        kind=kind,
                        artifact_id=int(artifact["id"]),
                    )
        if sample_ok:
            successful.append(sample_id)

    report["successful_sample_ids"] = successful
    report["core_artifact_counts"] = dict(sorted(core_counts.items()))
    return _finish(report)


def require_job_artifact_integrity(
    db: Database,
    job_id: int,
    *,
    expected_sample_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    report = validate_job_artifact_integrity(db, job_id, expected_sample_ids=expected_sample_ids)
    if not report["valid"]:
        raise ArtifactIntegrityError(report)
    return report


def _video_requirements(
    video_groups: Mapping[str, Mapping[str, Any]],
    *,
    publish_pred_video: bool,
    required_gt_video_names: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requirements: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for group_key, group in video_groups.items():
        frames = list(group.get("frames") or [])
        video_name = str(group.get("video_name") or group_key)
        if not frames:
            errors.append({
                "code": "empty_video_group",
                "message": f"video group {video_name} has no frames",
                "video_name": video_name,
            })
            continue
        requires_gt = video_name in required_gt_video_names
        is_multitrack = any(frame.get("track_key") or frame.get("track_label") for frame in frames)
        if is_multitrack:
            tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for frame in frames:
                track_key = str(frame.get("track_key") or frame.get("track_label") or "pred")
                tracks[track_key].append(frame)
            for track_key, track_frames in tracks.items():
                if publish_pred_video:
                    requirements.append({"kind": "pred_video", "video_name": video_name, "track_key": track_key, "frames": len(track_frames)})
                diff_present = [bool(frame.get("diff_path")) for frame in track_frames]
                if requires_gt:
                    requirements.append({"kind": "diff_video", "video_name": video_name, "track_key": track_key, "frames": len(track_frames)})
                    if not all(diff_present):
                        errors.append({
                            "code": "incomplete_video_group_frames",
                            "message": f"video group {video_name} track {track_key} has incomplete diff frames",
                            "video_name": video_name,
                            "track_key": track_key,
                            "kind": "diff_video",
                        })
                elif any(diff_present):
                    errors.append({
                        "code": "unexpected_video_group_frames",
                        "message": f"video group {video_name} track {track_key} has diff frames but its source samples have no GT",
                        "video_name": video_name,
                        "track_key": track_key,
                        "kind": "diff_video",
                    })
            gt_present = [bool(frame.get("gt_path")) for frame in frames]
            if requires_gt:
                requirements.append({
                    "kind": "gt_video",
                    "video_name": video_name,
                    "track_key": None,
                    "frames": len({int(frame.get("order") or 0) for frame in frames}),
                })
                if not all(gt_present):
                    errors.append({
                        "code": "incomplete_video_group_frames",
                        "message": f"video group {video_name} has incomplete GT frames",
                        "video_name": video_name,
                        "kind": "gt_video",
                    })
            elif any(gt_present):
                errors.append({
                    "code": "unexpected_video_group_frames",
                    "message": f"video group {video_name} has GT frames but its source samples have no GT",
                    "video_name": video_name,
                    "kind": "gt_video",
                })
            continue

        if publish_pred_video:
            requirements.append({"kind": "pred_video", "video_name": video_name, "track_key": None, "frames": len(frames)})
        for frame_kind, artifact_kind in (("gt_path", "gt_video"), ("diff_path", "diff_video")):
            present = [bool(frame.get(frame_kind)) for frame in frames]
            if requires_gt:
                requirements.append({"kind": artifact_kind, "video_name": video_name, "track_key": None, "frames": len(frames)})
                if not all(present):
                    errors.append({
                        "code": "incomplete_video_group_frames",
                        "message": f"video group {video_name} is missing one or more required {frame_kind} frames",
                        "video_name": video_name,
                        "kind": artifact_kind,
                    })
            elif any(present):
                errors.append({
                    "code": "unexpected_video_group_frames",
                    "message": f"video group {video_name} has {frame_kind} frames but its source samples have no GT",
                    "video_name": video_name,
                    "kind": artifact_kind,
                })
    return requirements, errors


def validate_video_artifact_integrity(
    db: Database,
    job_id: int,
    video_groups: Mapping[str, Mapping[str, Any]],
    *,
    publish_pred_video: bool = True,
    expected_sample_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    report = _report("video_artifacts", job_id=int(job_id))
    job = db.get_job(int(job_id))
    expected_ids = [int(sample_id) for sample_id in (
        expected_sample_ids if expected_sample_ids is not None else expected_sample_ids_for_job(db, job)
    )]
    expected_video_ids = []
    required_gt_video_names: set[str] = set()
    for sample_id in expected_ids:
        try:
            sample = db.get_sample(sample_id)
            if _is_video_sample(sample):
                expected_video_ids.append(sample_id)
                if str(sample.get("gt_path") or "").strip():
                    required_gt_video_names.add(_sample_video_name(sample))
        except KeyError:
            continue
    grouped_sample_ids: list[int] = []
    for group_key, group in video_groups.items():
        for frame in group.get("frames") or []:
            sample_id_value = frame.get("sample_id")
            if sample_id_value is None:
                _issue(
                    report,
                    "errors",
                    "video_frame_missing_sample_id",
                    f"video group {group_key} contains a frame without sample_id",
                    video_group=str(group_key),
                )
            else:
                grouped_sample_ids.append(int(sample_id_value))
            for field in ("pred_path", "gt_path", "diff_path"):
                if field != "pred_path" and not frame.get(field):
                    continue
                path = Path(str(frame.get(field) or ""))
                if not path.is_file() or path.stat().st_size <= 0:
                    _issue(
                        report,
                        "errors",
                        "invalid_video_frame_file",
                        f"video group {group_key} {field} is missing or empty",
                        video_group=str(group_key),
                        sample_id=int(sample_id_value) if sample_id_value is not None else None,
                        field=field,
                    )
    if Counter(grouped_sample_ids) != Counter(expected_video_ids):
        grouped_counts = Counter(grouped_sample_ids)
        expected_counts = Counter(expected_video_ids)
        _issue(
            report,
            "errors",
            "video_sample_coverage_mismatch",
            "video groups do not cover the job's video samples exactly once",
            missing_sample_ids=sorted((expected_counts - grouped_counts).elements()),
            unexpected_or_duplicate_sample_ids=sorted((grouped_counts - expected_counts).elements()),
        )
    report["expected_video_sample_ids"] = sorted(expected_video_ids)
    report["required_gt_video_names"] = sorted(required_gt_video_names)
    requirements, group_errors = _video_requirements(
        video_groups,
        publish_pred_video=publish_pred_video,
        required_gt_video_names=required_gt_video_names,
    )
    report["requirements"] = requirements
    report["errors"].extend(group_errors)

    expected: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for requirement in requirements:
        key = (str(requirement["kind"]), str(requirement["video_name"]), requirement.get("track_key"))
        if key in expected:
            _issue(
                report,
                "errors",
                "duplicate_video_requirement",
                f"duplicate video requirement for {key[0]} {key[1]} {key[2] or ''}".strip(),
                kind=key[0],
                video_name=key[1],
                track_key=key[2],
            )
        expected[key] = requirement

    actual: dict[tuple[str, str, str | None], list[dict[str, Any]]] = defaultdict(list)
    canonical_dimensions = _canonical_dimensions(db, job)
    job_canonical_required = _canonical_contract_required(db, job)
    video_path_owners: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in db.list_artifacts(job_id=int(job_id)):
        kind = str(artifact.get("kind") or "")
        if kind not in VIDEO_KINDS:
            continue
        metadata = artifact.get("metadata") or {}
        track_key_value = metadata.get("compare_track_key")
        track_key = str(track_key_value) if track_key_value not in (None, "") else None
        key = (kind, str(metadata.get("video_name") or ""), track_key)
        actual[key].append(artifact)
        path_value = str(artifact.get("path") or "")
        if path_value and (
            job_canonical_required
            or str(metadata.get("artifact_contract") or "") == CANONICAL_ARTIFACT_CONTRACT
        ):
            video_path_owners[str(Path(path_value).resolve())].append(
                {
                    "kind": kind,
                    "video_name": key[1],
                    "track_key": track_key,
                    "artifact_id": int(artifact["id"]),
                }
            )

    for path_value, owners in video_path_owners.items():
        identities = {
            (owner["kind"], owner["video_name"], owner["track_key"])
            for owner in owners
        }
        if len(identities) <= 1:
            continue
        _issue(
            report,
            "errors",
            "shared_canonical_video_path",
            "distinct canonical video identities reference the same encoded file",
            path=path_value,
            owners=owners,
        )

    for key, requirement in expected.items():
        rows = actual.get(key, [])
        if len(rows) != 1:
            code = "missing_video_artifact" if not rows else "duplicate_video_artifact"
            _issue(
                report,
                "errors",
                code,
                f"expected exactly one {key[0]} for {key[1]} {key[2] or ''}, found {len(rows)}".strip(),
                kind=key[0],
                video_name=key[1],
                track_key=key[2],
                artifact_ids=[int(row["id"]) for row in rows],
            )
            continue
        artifact = rows[0]
        path = Path(str(artifact.get("path") or ""))
        file_valid = path.is_file() and path.stat().st_size > 0
        if not file_valid:
            _issue(
                report,
                "errors",
                "invalid_video_artifact_file",
                f"{key[0]} for {key[1]} is missing or empty",
                kind=key[0],
                video_name=key[1],
                track_key=key[2],
                artifact_id=int(artifact["id"]),
            )
        artifact_metadata = artifact.get("metadata") or {}
        canonical_required = job_canonical_required or str(
            artifact_metadata.get("artifact_contract") or ""
        ) == CANONICAL_ARTIFACT_CONTRACT
        if canonical_required and str(artifact_metadata.get("artifact_contract") or "") != CANONICAL_ARTIFACT_CONTRACT:
            _issue(
                report,
                "errors",
                "missing_canonical_video_contract",
                f"{key[0]} for {key[1]} is not marked {CANONICAL_ARTIFACT_CONTRACT}",
                kind=key[0],
                video_name=key[1],
                track_key=key[2],
                artifact_id=int(artifact["id"]),
            )
        declared_width = int(artifact_metadata.get("width") or 0)
        declared_height = int(artifact_metadata.get("height") or 0)
        if canonical_required:
            if declared_width <= 0 or declared_height <= 0:
                _issue(
                    report,
                    "errors",
                    "missing_canonical_video_dimensions",
                    f"{key[0]} for {key[1]} does not declare canonical dimensions",
                    kind=key[0],
                    video_name=key[1],
                    track_key=key[2],
                    artifact_id=int(artifact["id"]),
                )
            if canonical_dimensions is not None and (
                declared_height,
                declared_width,
            ) != canonical_dimensions:
                _issue(
                    report,
                    "errors",
                    "canonical_video_metadata_size_mismatch",
                    f"{key[0]} for {key[1]} declares {declared_height}x{declared_width}; "
                    f"expected {canonical_dimensions[0]}x{canonical_dimensions[1]}",
                    kind=key[0],
                    video_name=key[1],
                    track_key=key[2],
                    expected_height=canonical_dimensions[0],
                    expected_width=canonical_dimensions[1],
                    actual_height=declared_height,
                    actual_width=declared_width,
                    artifact_id=int(artifact["id"]),
                )
        if artifact_metadata.get("preview_warning"):
            warning = artifact_metadata["preview_warning"]
            _issue(
                report,
                "warnings",
                "optional_video_preview_failed",
                str((warning or {}).get("message") or "optional video preview generation failed"),
                kind=key[0],
                video_name=key[1],
                track_key=key[2],
                artifact_id=int(artifact["id"]),
                error_type=str((warning or {}).get("type") or ""),
            )
        declared_frames = int((artifact.get("metadata") or {}).get("frames") or 0)
        if declared_frames != int(requirement["frames"]):
            _issue(
                report,
                "errors",
                "video_frame_count_mismatch",
                f"{key[0]} for {key[1]} declares {declared_frames} frames; expected {requirement['frames']}",
                kind=key[0],
                video_name=key[1],
                track_key=key[2],
                expected_frames=int(requirement["frames"]),
                actual_frames=declared_frames,
                artifact_id=int(artifact["id"]),
            )
        if canonical_required and file_valid:
            try:
                from vfieval.file_inputs import inspect_video

                observed = inspect_video(path, exact=True)
            except Exception as exc:
                _issue(
                    report,
                    "errors",
                    "video_probe_failed",
                    f"{key[0]} for {key[1]} could not be inspected: {exc}",
                    kind=key[0],
                    video_name=key[1],
                    track_key=key[2],
                    artifact_id=int(artifact["id"]),
                )
            else:
                observed_frames = int(observed.get("frame_count") or 0)
                report.setdefault("video_probes", []).append(
                    {
                        "artifact_id": int(artifact["id"]),
                        "kind": key[0],
                        "video_name": key[1],
                        "track_key": key[2],
                        "frame_count": observed_frames,
                        "width": int(observed.get("width") or 0),
                        "height": int(observed.get("height") or 0),
                        "decodable": bool(observed.get("decodable")),
                    }
                )
                if not observed.get("decodable"):
                    _issue(
                        report,
                        "errors",
                        "invalid_encoded_video",
                        f"{key[0]} for {key[1]} is not decodable: {observed.get('error') or 'unknown error'}",
                        kind=key[0],
                        video_name=key[1],
                        track_key=key[2],
                        artifact_id=int(artifact["id"]),
                    )
                if observed_frames != int(requirement["frames"]):
                    _issue(
                        report,
                        "errors",
                        "encoded_video_frame_count_mismatch",
                        f"{key[0]} for {key[1]} contains {observed_frames} frames; expected {requirement['frames']}",
                        kind=key[0],
                        video_name=key[1],
                        track_key=key[2],
                        expected_frames=int(requirement["frames"]),
                        actual_frames=observed_frames,
                        artifact_id=int(artifact["id"]),
                    )
                expected_height, expected_width = (
                    canonical_dimensions
                    if canonical_dimensions is not None
                    else (declared_height, declared_width)
                )
                observed_width = int(observed.get("width") or 0)
                observed_height = int(observed.get("height") or 0)
                if expected_height > 0 and expected_width > 0 and (
                    observed_height,
                    observed_width,
                ) != (expected_height, expected_width):
                    _issue(
                        report,
                        "errors",
                        "encoded_video_size_mismatch",
                        f"{key[0]} for {key[1]} is {observed_height}x{observed_width}; "
                        f"expected {expected_height}x{expected_width}",
                        kind=key[0],
                        video_name=key[1],
                        track_key=key[2],
                        expected_height=expected_height,
                        expected_width=expected_width,
                        actual_height=observed_height,
                        actual_width=observed_width,
                        artifact_id=int(artifact["id"]),
                    )
    for key, rows in actual.items():
        if key not in expected:
            _issue(
                report,
                "errors",
                "unexpected_video_artifact",
                f"unexpected {key[0]} artifact for {key[1]} {key[2] or ''}".strip(),
                kind=key[0],
                video_name=key[1],
                track_key=key[2],
                artifact_ids=[int(row["id"]) for row in rows],
            )
    report["video_artifact_counts"] = dict(sorted(Counter(key[0] for key, rows in actual.items() for _ in rows).items()))
    return _finish(report)


def validate_finalize_video_artifact_integrity(
    db: Database,
    publication_job_id: int,
    inference_job_ids: Iterable[int],
    video_groups: Mapping[str, Mapping[str, Any]],
    *,
    publish_pred_video: bool = True,
    expected_sample_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    report = validate_video_artifact_integrity(
        db,
        int(publication_job_id),
        video_groups,
        publish_pred_video=publish_pred_video,
        expected_sample_ids=expected_sample_ids,
    )
    for job_id in {int(value) for value in inference_job_ids} - {int(publication_job_id)}:
        rows = [
            artifact
            for artifact in db.list_artifacts(job_id=job_id)
            if str(artifact.get("kind") or "") in VIDEO_KINDS
        ]
        if rows:
            _issue(
                report,
                "errors",
                "unexpected_shard_video_artifact",
                f"inference shard job {job_id} contains video artifacts outside the finalize publication job",
                job_id=job_id,
                artifact_ids=[int(row["id"]) for row in rows],
                kinds=sorted(str(row.get("kind") or "") for row in rows),
            )
    return _finish(report)


def _is_video_sample(sample: Mapping[str, Any]) -> bool:
    metadata = sample.get("metadata") or {}
    return str(metadata.get("source_type") or "") == "video" or bool(metadata.get("compare_group"))


def _sample_video_name(sample: Mapping[str, Any]) -> str:
    """Mirror inference video grouping when resolving a sample's public name."""

    metadata = dict(sample.get("metadata") or {})
    explicit = metadata.get("video_name") or metadata.get("compare_group")
    if explicit:
        return str(explicit)
    video_key = str(metadata.get("video_path") or "video")
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_key.strip())
    return sanitized or "video"


def _validate_manifest_frame_identity(
    report: dict[str, Any],
    *,
    job_id: int,
    group_key: str,
    group: Mapping[str, Any],
    frame: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> None:
    """Validate manifest semantics independently from the manifest itself."""

    metadata = dict(sample.get("metadata") or {})
    is_compare = str(metadata.get("source_type") or "") == "compare" or bool(metadata.get("compare_group"))
    if is_compare:
        expected_group_key = str(metadata.get("compare_group") or metadata.get("video_name") or "compare")
        expected_video_name = str(metadata.get("video_name") or expected_group_key)
    else:
        expected_group_key = str(metadata.get("video_path") or metadata.get("video_name") or "video")
        expected_video_name = _sample_video_name(sample)

    actual_video_name = str(group.get("video_name") or group_key)
    if str(group_key) != expected_group_key or actual_video_name != expected_video_name:
        _issue(
            report,
            "errors",
            "manifest_frame_video_identity_mismatch",
            f"sample {sample['id']} is assigned to the wrong manifest video identity",
            job_id=int(job_id),
            sample_id=int(sample["id"]),
            expected_group=expected_group_key,
            actual_group=str(group_key),
            expected_video_name=expected_video_name,
            actual_video_name=actual_video_name,
        )

    order_value = metadata.get("frame_index")
    if order_value in {None, ""}:
        order_value = metadata.get("sample_index")
    if order_value not in {None, ""}:
        try:
            expected_order = int(order_value)
            actual_order = int(frame.get("order"))
        except (TypeError, ValueError):
            expected_order = int(order_value)
            actual_order = None
        if actual_order != expected_order:
            _issue(
                report,
                "errors",
                "manifest_frame_order_mismatch",
                f"sample {sample['id']} manifest order does not match its source frame identity",
                job_id=int(job_id),
                sample_id=int(sample["id"]),
                expected_order=expected_order,
                actual_order=actual_order,
            )

    expected_fps = _optional_positive_number(metadata.get("fps"), integer=False)
    actual_fps = _optional_positive_number(group.get("fps"), integer=False)
    if expected_fps is not None and (
        actual_fps is None
        or abs(float(expected_fps) - float(actual_fps)) > VIDEO_FPS_TOLERANCE
    ):
        _issue(
            report,
            "errors",
            "manifest_frame_source_mapping_mismatch",
            f"sample {sample['id']} manifest FPS does not match its source metadata",
            job_id=int(job_id),
            sample_id=int(sample["id"]),
            field="fps",
            expected=expected_fps,
            actual=actual_fps,
        )

    mapping_fields: list[tuple[str, Any, Any]] = []
    if is_compare:
        for frame_field, metadata_field in (
            ("track_label", "compare_track_label"),
            ("track_key", "compare_track_key"),
            ("track_run_id", "compare_track_run_id"),
            ("track_artifact_id", "compare_track_artifact_id"),
        ):
            mapping_fields.append((frame_field, metadata.get(metadata_field), frame.get(frame_field)))
    else:
        timestamps = metadata.get("timestamps")
        source_timestamp = timestamps.get("gt") if isinstance(timestamps, Mapping) else None
        mapping_fields.extend(
            [
                ("source_video_path", metadata.get("video_path"), group.get("source_video_path")),
                ("source_video_group", metadata.get("video_group"), group.get("source_video_group")),
                ("source_video_file", metadata.get("video_file"), group.get("source_video_file")),
                ("source_frame_index", metadata.get("gt_index"), frame.get("source_frame_index")),
                ("source_timestamp", source_timestamp, frame.get("source_timestamp")),
            ]
        )
    for field, expected_value, actual_value in mapping_fields:
        if field == "source_timestamp" and expected_value is not None and actual_value is not None:
            try:
                matches = abs(float(expected_value) - float(actual_value)) <= VIDEO_TIMESTAMP_TOLERANCE
            except (TypeError, ValueError):
                matches = False
        else:
            matches = expected_value == actual_value
        if matches:
            continue
        _issue(
            report,
            "errors",
            "manifest_frame_source_mapping_mismatch",
            f"sample {sample['id']} manifest {field} does not match its source metadata",
            job_id=int(job_id),
            sample_id=int(sample["id"]),
            field=field,
            expected=expected_value,
            actual=actual_value,
        )


def validate_finalize_inputs(
    db: Database,
    run_id: int,
    inference_jobs: Iterable[Mapping[str, Any]],
    run_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    jobs = [dict(job) for job in inference_jobs]
    report = _report("finalize_inputs", run_id=int(run_id))
    merged: dict[str, dict[str, Any]] = {}
    all_expected: list[int] = []
    all_successful: list[int] = []
    all_video_samples: list[int] = []
    all_manifest_video_samples: list[int] = []
    manifest_paths: list[str] = []
    run_dir_resolved = run_dir.resolve()

    for job in jobs:
        job_id = int(job["job_id"])
        expected = expected_sample_ids_for_job(db, job)
        all_expected.extend(expected)
        job_report = validate_job_artifact_integrity(db, job_id, expected_sample_ids=expected)
        all_successful.extend(int(sample_id) for sample_id in job_report.get("successful_sample_ids") or [])
        report.setdefault("job_checks", []).append(job_report)
        report["errors"].extend(job_report.get("errors") or [])
        report["warnings"].extend(job_report.get("warnings") or [])
        canonical_frame_paths = {
            (int(artifact["sample_id"]), str(artifact["kind"])): Path(str(artifact["path"])).resolve()
            for artifact in db.list_artifacts(job_id=job_id)
            if artifact.get("sample_id") is not None
            and str(artifact.get("kind") or "") in {"pred", "gt", "difference"}
        }

        video_sample_ids = []
        for sample_id in expected:
            try:
                if _is_video_sample(db.get_sample(sample_id)):
                    video_sample_ids.append(sample_id)
            except KeyError:
                continue
        all_video_samples.extend(video_sample_ids)
        manifest_path = run_dir / "logs" / "shards" / f"{job_id}.json"
        if not video_sample_ids and not manifest_path.is_file():
            continue
        if not manifest_path.is_file():
            _issue(
                report,
                "errors",
                "missing_shard_manifest",
                f"video shard {job_id} is missing its finalize manifest",
                job_id=job_id,
            )
            continue
        manifest_paths.append(str(manifest_path))
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _issue(
                report,
                "errors",
                "invalid_shard_manifest",
                f"cannot read shard manifest for job {job_id}: {exc}",
                job_id=job_id,
            )
            continue
        if data.get("version") != SHARD_MANIFEST_SCHEMA:
            _issue(
                report,
                "errors",
                "invalid_shard_manifest_version",
                f"shard manifest for job {job_id} must use {SHARD_MANIFEST_SCHEMA}",
                job_id=job_id,
                actual=data.get("version"),
            )
        if int(data.get("run_id") or -1) != int(run_id) or int(data.get("job_id") or -1) != job_id:
            _issue(
                report,
                "errors",
                "shard_manifest_identity_mismatch",
                f"shard manifest identity does not match run {run_id}, job {job_id}",
                job_id=job_id,
            )
        manifest_expected = [int(sample_id) for sample_id in data.get("expected_sample_ids") or []]
        manifest_successful = [int(sample_id) for sample_id in data.get("successful_sample_ids") or []]
        if sorted(manifest_expected) != sorted(expected):
            _issue(
                report,
                "errors",
                "shard_manifest_expected_samples_mismatch",
                f"shard manifest expected samples do not match job {job_id}",
                job_id=job_id,
                expected_sample_ids=sorted(expected),
                manifest_sample_ids=sorted(manifest_expected),
            )
        if sorted(manifest_successful) != sorted(expected):
            _issue(
                report,
                "errors",
                "shard_manifest_incomplete_samples",
                f"shard manifest for job {job_id} is missing successful samples",
                job_id=job_id,
                expected_sample_ids=sorted(expected),
                successful_sample_ids=sorted(manifest_successful),
            )
        if dict(data.get("core_artifact_counts") or {}) != dict(job_report.get("core_artifact_counts") or {}):
            _issue(
                report,
                "errors",
                "shard_manifest_artifact_count_mismatch",
                f"shard manifest artifact counts do not match SQLite for job {job_id}",
                job_id=job_id,
                expected_counts=job_report.get("core_artifact_counts") or {},
                manifest_counts=data.get("core_artifact_counts") or {},
            )

        for group_key, group_value in (data.get("video_groups") or {}).items():
            group = dict(group_value or {})
            if not list(group.get("frames") or []):
                _issue(
                    report,
                    "errors",
                    "empty_manifest_video_group",
                    f"video group {group_key} in shard manifest {job_id} has no frames",
                    job_id=job_id,
                    video_group=str(group_key),
                )
            target = merged.setdefault(
                str(group_key),
                {name: value for name, value in group.items() if name != "frames"},
            )
            for name, value in group.items():
                if name == "frames" or value is None or target.get(name) is None:
                    continue
                if target.get(name) != value:
                    _issue(
                        report,
                        "errors",
                        "inconsistent_shard_video_metadata",
                        f"video group {group_key} has inconsistent {name} across shards",
                        job_id=job_id,
                        video_group=str(group_key),
                        field=name,
                    )
            target.setdefault("frames", [])
            for frame_value in group.get("frames") or []:
                frame = dict(frame_value or {})
                sample_id_value = frame.get("sample_id")
                if sample_id_value is None:
                    _issue(
                        report,
                        "errors",
                        "manifest_frame_missing_sample_id",
                        f"video group {group_key} contains a frame without sample_id",
                        job_id=job_id,
                        video_group=str(group_key),
                    )
                else:
                    sample_id = int(sample_id_value)
                    all_manifest_video_samples.append(sample_id)
                    if sample_id not in video_sample_ids:
                        _issue(
                            report,
                            "errors",
                            "manifest_frame_wrong_shard",
                            f"sample {sample_id} is not assigned to video shard job {job_id}",
                            job_id=job_id,
                            sample_id=sample_id,
                        )
                    else:
                        try:
                            manifest_sample = db.get_sample(sample_id)
                        except KeyError:
                            manifest_sample = None
                        if manifest_sample is not None:
                            _validate_manifest_frame_identity(
                                report,
                                job_id=job_id,
                                group_key=str(group_key),
                                group=group,
                                frame=frame,
                                sample=manifest_sample,
                            )
                for name in ("pred_path", "gt_path", "diff_path"):
                    if not frame.get(name):
                        continue
                    path = Path(str(frame[name])).resolve()
                    try:
                        trusted = path.is_relative_to(run_dir_resolved)
                    except AttributeError:
                        trusted = run_dir_resolved == path or run_dir_resolved in path.parents
                    if not trusted or not path.is_file() or path.stat().st_size <= 0:
                        _issue(
                            report,
                            "errors",
                            "invalid_manifest_frame_file",
                            f"manifest frame {name} for job {job_id} is outside the Run or missing/empty",
                            job_id=job_id,
                            sample_id=int(sample_id_value) if sample_id_value is not None else None,
                            field=name,
                        )
                    artifact_kind = {
                        "pred_path": "pred",
                        "gt_path": "gt",
                        "diff_path": "difference",
                    }[name]
                    expected_path = (
                        canonical_frame_paths.get((int(sample_id_value), artifact_kind))
                        if sample_id_value is not None
                        else None
                    )
                    if expected_path is None or path != expected_path:
                        _issue(
                            report,
                            "errors",
                            "manifest_frame_artifact_mismatch",
                            f"manifest frame {name} for job {job_id} does not match its canonical SQLite artifact",
                            job_id=job_id,
                            sample_id=int(sample_id_value) if sample_id_value is not None else None,
                            field=name,
                            expected_path=str(expected_path) if expected_path is not None else None,
                            actual_path=str(path),
                        )
                    frame[name] = path
                target["frames"].append(frame)

    expected_counts = Counter(all_expected)
    duplicate_expected = sorted(sample_id for sample_id, count in expected_counts.items() if count > 1)
    if duplicate_expected:
        _issue(
            report,
            "errors",
            "overlapping_shard_samples",
            "inference shards contain overlapping sample assignments",
            sample_ids=duplicate_expected,
        )
    profiles = {str((job.get("payload") or {}).get("artifact_profile") or "evaluation") for job in jobs}
    dataset_ids = {int((job.get("payload") or {}).get("dataset_id")) for job in jobs if (job.get("payload") or {}).get("dataset_id") is not None}
    if profiles != {"benchmark"} and len(dataset_ids) == 1:
        dataset_id = next(iter(dataset_ids))
        dataset_sample_ids = {int(sample["id"]) for sample in db.list_samples(dataset_id)}
        expected_set = set(all_expected)
        if expected_set != dataset_sample_ids:
            _issue(
                report,
                "errors",
                "incomplete_shard_sample_partition",
                "inference shard sample assignments do not cover the Run dataset exactly",
                missing_sample_ids=sorted(dataset_sample_ids - expected_set),
                unexpected_sample_ids=sorted(expected_set - dataset_sample_ids),
            )
    if Counter(all_manifest_video_samples) != Counter(all_video_samples):
        manifest_counts = Counter(all_manifest_video_samples)
        video_counts = Counter(all_video_samples)
        _issue(
            report,
            "errors",
            "manifest_video_sample_coverage_mismatch",
            "shard manifests do not cover video samples exactly once",
            missing_sample_ids=sorted((video_counts - manifest_counts).elements()),
            unexpected_or_duplicate_sample_ids=sorted((manifest_counts - video_counts).elements()),
        )
    frame_keys: Counter[tuple[str, int, str]] = Counter()
    for group_key, group in merged.items():
        for frame in group.get("frames") or []:
            frame_keys[(str(group_key), int(frame.get("order") or 0), str(frame.get("track_key") or ""))] += 1
    duplicate_frames = [
        {"video_group": key[0], "order": key[1], "track_key": key[2]}
        for key, count in frame_keys.items()
        if count > 1
    ]
    if duplicate_frames:
        _issue(
            report,
            "errors",
            "duplicate_manifest_video_frame",
            "merged shard manifests contain duplicate video frame positions",
            frames=duplicate_frames,
        )

    report["manifest_paths"] = manifest_paths
    report["expected_sample_ids"] = sorted(all_expected)
    report["successful_sample_ids"] = sorted(all_successful)
    report["video_sample_ids"] = sorted(all_video_samples)
    report["merged_video_count"] = len(merged)
    return merged, _finish(report)


def require_finalize_inputs(
    db: Database,
    run_id: int,
    inference_jobs: Iterable[Mapping[str, Any]],
    run_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    merged, report = validate_finalize_inputs(db, run_id, inference_jobs, run_dir)
    if not report["valid"]:
        raise ArtifactIntegrityError(report)
    return merged, report


def validate_metric_retry_integrity(db: Database, run_id: int) -> dict[str, Any]:
    """Validate frozen inference/finalize outputs before a terminal Run retry."""
    run = db.get_run(int(run_id))
    inference_jobs = db.list_run_jobs(int(run_id), "inference")
    report = _report("metric_retry", run_id=int(run_id))
    report["content_revision"] = int(run.get("content_revision") or 0)
    if not inference_jobs:
        _issue(report, "errors", "missing_inference_job", "Run has no inference jobs")
        return _finish(report)
    incomplete = [
        {"job_id": int(job["job_id"]), "status": str(job.get("status") or "")}
        for job in inference_jobs
        if str(job.get("status") or "") != "completed"
    ]
    if incomplete:
        _issue(
            report,
            "errors",
            "inference_not_completed",
            "all inference jobs must be completed before a metric retry",
            jobs=incomplete,
        )
    for job in inference_jobs:
        child = validate_job_artifact_integrity(db, int(job["job_id"]))
        report.setdefault("checks", []).append(child)
        report["errors"].extend(child.get("errors") or [])
        report["warnings"].extend(child.get("warnings") or [])

    finalize_jobs = db.list_run_jobs(int(run_id), "finalize")
    if finalize_jobs:
        incomplete_finalize = [
            {"job_id": int(job["job_id"]), "status": str(job.get("status") or "")}
            for job in finalize_jobs
            if str(job.get("status") or "") != "completed"
        ]
        if incomplete_finalize:
            _issue(
                report,
                "errors",
                "finalize_not_completed",
                "the finalize job must be completed before a metric retry",
                jobs=incomplete_finalize,
            )

    video_samples: list[dict[str, Any]] = []
    for job in inference_jobs:
        for sample_id in expected_sample_ids_for_job(db, job):
            try:
                sample = db.get_sample(int(sample_id))
            except KeyError:
                continue
            if _is_video_sample(sample):
                video_samples.append(sample)
    if video_samples:
        video_names = {_sample_video_name(sample) for sample in video_samples}
        video_has_gt = {
            video_name: any(
                bool(sample.get("gt_path"))
                for sample in video_samples
                if _sample_video_name(sample) == video_name
            )
            for video_name in video_names
        }
        run_metadata = dict(run.get("metadata") or {})
        request = dict(run_metadata.get("request") or {})
        publish_pred = bool(
            run_metadata.get(
                "publish_compare_pred_video",
                request.get("publish_compare_pred_video", True),
            )
        )
        artifacts = [
            artifact
            for job in inference_jobs
            for artifact in db.list_artifacts(job_id=int(job["job_id"]))
            if str(artifact.get("kind") or "") in VIDEO_KINDS
        ]
        by_identity: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        path_owners: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for artifact in artifacts:
            metadata = dict(artifact.get("metadata") or {})
            identity = (
                str(artifact.get("kind") or ""),
                str(metadata.get("video_name") or ""),
                str(metadata.get("compare_track_key") or ""),
            )
            by_identity[identity].append(artifact)
            path_value = str(artifact.get("path") or "")
            if path_value:
                path_owners[str(Path(path_value).resolve())].append(identity)
        for identity, rows in by_identity.items():
            if len(rows) > 1:
                _issue(
                    report,
                    "errors",
                    "duplicate_video_artifact",
                    f"metric retry found duplicate video artifact identity {identity}",
                    identity=list(identity),
                    artifact_ids=[int(row["id"]) for row in rows],
                )
        for path_value, identities in path_owners.items():
            if len(set(identities)) > 1:
                _issue(
                    report,
                    "errors",
                    "shared_canonical_video_path",
                    "metric retry found distinct video identities sharing one encoded file",
                    path=path_value,
                    identities=[list(identity) for identity in identities],
                )
        canonical_required = str(
            run_metadata.get("artifact_contract") or request.get("artifact_contract") or ""
        ) == CANONICAL_ARTIFACT_CONTRACT or any(
            str((artifact.get("metadata") or {}).get("artifact_contract") or "")
            == CANONICAL_ARTIFACT_CONTRACT
            for artifact in artifacts
        )
        canonical_dimensions = (int(run.get("height") or 0), int(run.get("width") or 0))
        video_observations: dict[int, Mapping[str, Any]] = {}
        for video_name in sorted(video_names):
            if not video_has_gt[video_name]:
                unexpected_reference_artifacts = [
                    artifact
                    for artifact in artifacts
                    if str(artifact.get("kind") or "") in {"gt_video", "diff_video"}
                    and str((artifact.get("metadata") or {}).get("video_name") or "") == video_name
                ]
                for artifact in unexpected_reference_artifacts:
                    kind = str(artifact.get("kind") or "")
                    _issue(
                        report,
                        "errors",
                        "unexpected_video_artifact",
                        f"metric retry found {kind} for no-GT video {video_name}",
                        artifact_id=int(artifact["id"]),
                        kind=kind,
                        video_name=video_name,
                    )
            required_kinds = {"gt_video", "diff_video"} if video_has_gt[video_name] else set()
            if publish_pred:
                required_kinds.add("pred_video")
            for kind in sorted(required_kinds):
                rows = [
                    artifact
                    for artifact in artifacts
                    if str(artifact.get("kind") or "") == kind
                    and str((artifact.get("metadata") or {}).get("video_name") or "") == video_name
                ]
                if not rows:
                    _issue(
                        report,
                        "errors",
                        "missing_video_artifact",
                        f"metric retry requires {kind} for video {video_name}",
                        kind=kind,
                        video_name=video_name,
                    )
                    continue
                for artifact in rows:
                    metadata = dict(artifact.get("metadata") or {})
                    path = Path(str(artifact.get("path") or ""))
                    if not path.is_file() or path.stat().st_size <= 0:
                        _issue(
                            report,
                            "errors",
                            "invalid_video_artifact_file",
                            f"metric retry {kind} for {video_name} is missing or empty",
                            artifact_id=int(artifact["id"]),
                            kind=kind,
                            video_name=video_name,
                        )
                        continue
                    if canonical_required and str(metadata.get("artifact_contract") or "") != CANONICAL_ARTIFACT_CONTRACT:
                        _issue(
                            report,
                            "errors",
                            "missing_canonical_video_contract",
                            f"metric retry {kind} for {video_name} is not canonical-v1",
                            artifact_id=int(artifact["id"]),
                            kind=kind,
                            video_name=video_name,
                        )
                    try:
                        from vfieval.file_inputs import inspect_video

                        observed = inspect_video(path, exact=True)
                    except Exception as exc:
                        _issue(
                            report,
                            "errors",
                            "video_probe_failed",
                            f"metric retry could not inspect {kind} for {video_name}: {exc}",
                            artifact_id=int(artifact["id"]),
                        )
                        continue
                    video_observations[int(artifact["id"])] = observed
                    expected_frames = int(metadata.get("frames") or 0)
                    actual_frames = int(observed.get("frame_count") or 0)
                    if not observed.get("decodable") or expected_frames <= 0 or actual_frames != expected_frames:
                        _issue(
                            report,
                            "errors",
                            "invalid_encoded_video",
                            f"metric retry {kind} for {video_name} is not a complete encoded video",
                            artifact_id=int(artifact["id"]),
                            expected_frames=expected_frames,
                            actual_frames=actual_frames,
                            reason=observed.get("error"),
                        )
                    if canonical_required:
                        declared_dimensions = (
                            int(metadata.get("height") or 0),
                            int(metadata.get("width") or 0),
                        )
                        observed_dimensions = (
                            int(observed.get("height") or 0),
                            int(observed.get("width") or 0),
                        )
                        if (
                            declared_dimensions != canonical_dimensions
                            or observed_dimensions != canonical_dimensions
                        ):
                            _issue(
                                report,
                                "errors",
                                "canonical_video_size_mismatch",
                                f"metric retry {kind} for {video_name} does not match the Run's canonical dimensions",
                                artifact_id=int(artifact["id"]),
                                expected_height=canonical_dimensions[0],
                                expected_width=canonical_dimensions[1],
                                declared_height=declared_dimensions[0],
                                declared_width=declared_dimensions[1],
                                actual_height=observed_dimensions[0],
                                actual_width=observed_dimensions[1],
                            )
            if video_has_gt[video_name] and publish_pred:
                pred_rows = [
                    artifact
                    for artifact in artifacts
                    if str(artifact.get("kind") or "") == "pred_video"
                    and str((artifact.get("metadata") or {}).get("video_name") or "") == video_name
                ]
                for pred_artifact in pred_rows:
                    paired_gt = [
                        artifact
                        for artifact in artifacts
                        if str(artifact.get("kind") or "") == "gt_video"
                        and int(artifact.get("job_id") or 0) == int(pred_artifact.get("job_id") or 0)
                        and str((artifact.get("metadata") or {}).get("video_name") or "") == video_name
                    ]
                    if len(paired_gt) != 1:
                        _issue(
                            report,
                            "errors",
                            "video_pair_identity_mismatch",
                            f"metric retry requires one GT video from the same inference Job as each Pred for {video_name}",
                            video_name=video_name,
                            pred_artifact_id=int(pred_artifact["id"]),
                            pred_job_id=int(pred_artifact.get("job_id") or 0),
                            gt_artifact_ids=[int(row["id"]) for row in paired_gt],
                        )
                        continue
                    pair_issue = strict_video_pair_issue(
                        pred_artifact,
                        paired_gt[0],
                        observations=video_observations,
                    )
                    if pair_issue is not None:
                        report["errors"].append(pair_issue)
    return _finish(report)


def require_metric_retry_integrity(db: Database, run_id: int) -> dict[str, Any]:
    report = validate_metric_retry_integrity(db, int(run_id))
    if not report["valid"]:
        raise ArtifactIntegrityError(report)
    return report
