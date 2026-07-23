from __future__ import annotations

import errno
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.release import package_release_metadata


PROCESS_STARTED_AT = time.time()
JOB_LEASE_STALE_SECONDS = 90.0
QUEUE_CONSUMER_GRACE_SECONDS = 90.0
QUEUE_STALE_SECONDS = 300.0
DIAGNOSTICS_LOG_BUDGET_BYTES = 8 * 1024 * 1024
DIAGNOSTICS_LOG_FILE_BYTES = 1024 * 1024


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def release_info() -> dict[str, Any]:
    package = package_release_metadata()
    return {
        **package,
        "started_at": datetime.fromtimestamp(PROCESS_STARTED_AT, timezone.utc).isoformat(),
        "uptime_seconds": max(0.0, time.time() - PROCESS_STARTED_AT),
        "python": platform.python_version(),
    }


def _schema_summary(db: Database) -> dict[str, Any]:
    try:
        latest = db.get(
            """
            SELECT version, applied_at
            FROM schema_migrations
            WHERE version NOT LIKE 'maintenance:%'
            ORDER BY applied_at DESC
            LIMIT 1
            """
        )
        tables = db.get("SELECT COUNT(*) AS count FROM sqlite_master WHERE type = 'table'")
        return {
            "latest_migration": latest.get("version") if latest else None,
            "table_count": int((tables or {}).get("count") or 0),
        }
    except (sqlite3.Error, OSError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _lease_summary(db: Database) -> dict[str, Any]:
    method = getattr(db, "job_lease_summary", None)
    if callable(method):
        try:
            now = time.time()
            return method(now - JOB_LEASE_STALE_SECONDS, now=now)
        except (sqlite3.Error, OSError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    try:
        rows = db.query(
            "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status ORDER BY status"
        )
        return {"by_status": {str(row["status"]): int(row["count"]) for row in rows}}
    except sqlite3.Error as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _storage_summary(workspace: WorkspaceConfig) -> dict[str, Any]:
    root_exists = workspace.root.exists()
    probe = workspace.root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return {
            "status": "error",
            "workspace_exists": root_exists,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    storage = {
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "workspace_exists": root_exists,
    }
    storage["status"] = (
        "error"
        if not root_exists or storage["free_bytes"] < 1024**3
        else "warning"
        if storage["free_bytes"] < 5 * 1024**3
        or storage["free_bytes"] * 100 < storage["total_bytes"] * 5
        else "ok"
    )
    if not root_exists:
        storage["reason"] = "workspace does not exist"
    return storage


def _queue_summary(db: Database, *, now: float | None = None) -> dict[str, Any]:
    observed_at = time.time() if now is None else float(now)
    try:
        rows = db.query(
            """
            SELECT kind, COUNT(*) AS count, MIN(created_at) AS oldest_created_at
            FROM jobs
            WHERE status = 'queued'
            GROUP BY kind
            ORDER BY kind
            """
        )
        workers = db.query(
            """
            SELECT role, COUNT(*) AS count
            FROM workers
            WHERE last_seen_at >= ?
            GROUP BY role
            ORDER BY role
            """,
            (observed_at - JOB_LEASE_STALE_SECONDS,),
        )
    except (sqlite3.Error, OSError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    consumers = {str(row["role"]): int(row["count"] or 0) for row in workers}
    by_kind: dict[str, Any] = {}
    total = 0
    stale = 0
    without_consumer = 0
    oldest_created_at: float | None = None
    for row in rows:
        kind = str(row["kind"])
        count = int(row["count"] or 0)
        created_at = (
            float(row["oldest_created_at"])
            if row.get("oldest_created_at") is not None
            else None
        )
        age = (
            max(0.0, observed_at - created_at)
            if created_at is not None
            else None
        )
        consumer_count = int(consumers.get(kind, 0)) + int(consumers.get("all", 0))
        is_stale = age is not None and age >= QUEUE_STALE_SECONDS
        has_no_consumer = (
            consumer_count == 0
            and age is not None
            and age >= QUEUE_CONSUMER_GRACE_SECONDS
        )
        by_kind[kind] = {
            "count": count,
            "oldest_created_at": created_at,
            "oldest_age_seconds": age,
            "fresh_consumers": consumer_count,
            "stale": is_stale,
            "no_consumer": has_no_consumer,
        }
        total += count
        if is_stale:
            stale += count
        if has_no_consumer:
            without_consumer += count
        if created_at is not None:
            oldest_created_at = (
                created_at
                if oldest_created_at is None
                else min(oldest_created_at, created_at)
            )
    return {
        "queued": total,
        "stale": stale,
        "without_consumer": without_consumer,
        "oldest_created_at": oldest_created_at,
        "oldest_age_seconds": (
            max(0.0, observed_at - oldest_created_at)
            if oldest_created_at is not None
            else None
        ),
        "consumer_freshness_seconds": JOB_LEASE_STALE_SECONDS,
        "consumer_grace_seconds": QUEUE_CONSUMER_GRACE_SECONDS,
        "stale_after_seconds": QUEUE_STALE_SECONDS,
        "fresh_consumers_by_role": consumers,
        "by_kind": by_kind,
    }


def _readiness_reasons(
    *,
    schema: dict[str, Any],
    storage: dict[str, Any],
    leases: dict[str, Any],
    queues: dict[str, Any],
    maintenance: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if schema.get("error"):
        reasons.append("schema_error")
    if storage.get("status") == "error":
        reasons.append("storage_error")
    if leases.get("error"):
        reasons.append("lease_state_error")
    elif int(leases.get("stale") or 0) > 0:
        reasons.append("stale_job_leases")
    if queues.get("error"):
        reasons.append("queue_state_error")
    else:
        if int(queues.get("stale") or 0) > 0:
            reasons.append("queued_jobs_stale")
        if int(queues.get("without_consumer") or 0) > 0:
            reasons.append("queued_jobs_without_consumer")

    catalog = maintenance.get("catalog")
    if isinstance(catalog, dict) and catalog.get("state") == "failed":
        reasons.append("catalog_sync_failed")
    cache_coordination = maintenance.get("cache_coordination")
    if isinstance(cache_coordination, dict) and cache_coordination.get("state") == "failed":
        reasons.append("cache_coordination_failed")
    recovery = maintenance.get("job_recovery")
    if isinstance(recovery, dict):
        if recovery.get("last_error"):
            reasons.append("job_recovery_failed")
        if "running" in recovery and not recovery.get("running"):
            reasons.append("job_recovery_not_running")
    for name in ("run_cleanup", "campaign_cleanup"):
        cleanup = maintenance.get(name)
        by_status = cleanup.get("by_status") if isinstance(cleanup, dict) else {}
        if int((by_status or {}).get("failed") or 0) > 0:
            reasons.append(f"{name}_failed")
    return reasons


def health_snapshot(
    db: Database,
    workspace: WorkspaceConfig,
    *,
    catalog: dict[str, Any] | None = None,
    maintenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema = _schema_summary(db)
    storage = _storage_summary(workspace)
    leases = _lease_summary(db)
    queues = _queue_summary(db)
    maintenance_snapshot = dict(maintenance or {})
    reasons = _readiness_reasons(
        schema=schema,
        storage=storage,
        leases=leases,
        queues=queues,
        maintenance=maintenance_snapshot,
    )
    ready = not reasons
    return {
        "ok": ready,
        "live": True,
        "ready": ready,
        "reasons": reasons,
        "release": release_info(),
        "schema": schema,
        "storage": storage,
        "leases": leases,
        "queues": queues,
        "catalog": dict(catalog or {}),
        "maintenance": maintenance_snapshot,
    }


def _command_probe(command: list[str], *, timeout: float = 8.0) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if not executable:
        return {"status": "unavailable", "available": False, "reason": f"{command[0]} not found"}
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "error",
            "available": True,
            "path": executable,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    output = (completed.stdout or completed.stderr or "").strip()
    return {
        "status": "ok" if completed.returncode == 0 else "error",
        "available": True,
        "path": executable,
        "returncode": int(completed.returncode),
        "summary": output.splitlines()[0][:500] if output else "",
    }


def _workspace_write_probe(workspace: WorkspaceConfig) -> dict[str, Any]:
    try:
        with tempfile.NamedTemporaryFile(prefix="doctor-", dir=workspace.tmp_dir, delete=True) as handle:
            handle.write(b"vfieval")
            handle.flush()
        return {"status": "ok", "writable": True}
    except OSError as exc:
        return {"status": "error", "writable": False, "reason": f"{type(exc).__name__}: {exc}"}


def _database_probe(db: Database) -> dict[str, Any]:
    try:
        quick = db.get("PRAGMA quick_check") or {}
        journal = db.get("PRAGMA journal_mode") or {}
        quick_value = next(iter(quick.values()), "unknown")
        journal_value = next(iter(journal.values()), "unknown")
        return {
            "status": "ok" if str(quick_value).lower() == "ok" else "error",
            "quick_check": quick_value,
            "journal_mode": journal_value,
        }
    except (sqlite3.Error, OSError) as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def _port_probe(host: str, port: int) -> dict[str, Any]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, int(port)))
        return {
            "status": "ok",
            "bindable": True,
            "host": host,
            "port": int(port),
        }
    except OSError as exc:
        occupied = exc.errno == errno.EADDRINUSE or getattr(
            exc,
            "winerror",
            None,
        ) == 10048
        return {
            "status": "unavailable" if occupied else "error",
            "bindable": False,
            "host": host,
            "port": int(port),
            "reason": (
                f"{host}:{port} is already in use"
                if occupied
                else f"{type(exc).__name__}: {exc}"
            ),
        }
    finally:
        sock.close()


def _device_probe(target_devices: list[str] | None) -> dict[str, Any]:
    from vfieval.worker import detect_capabilities

    requested = list(
        dict.fromkeys(
            str(value).strip().lower()
            for value in (target_devices or [])
            if str(value).strip()
        )
    )
    try:
        capabilities = detect_capabilities()
    except Exception as exc:
        return {
            "status": "error",
            "requested": requested,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    available_ids = {
        str(row.get("id") or "").lower()
        for family in ("cuda", "npu")
        for row in (capabilities.get(family) or [])
        if isinstance(row, dict) and row.get("id")
    }
    available_families = {
        device_id.split(":", 1)[0]
        for device_id in available_ids
    }
    if capabilities.get("cpu"):
        available_ids.add("cpu")
        available_families.add("cpu")

    invalid = [
        target
        for target in requested
        if not re.fullmatch(r"(?:auto|cpu|cuda(?::\d+)?|npu(?::\d+)?)", target)
    ]
    missing = [
        target
        for target in requested
        if target not in invalid
        and target != "auto"
        and target not in available_ids
        and target not in available_families
    ]
    result = dict(capabilities)
    result.update(
        {
            "status": "error" if invalid else "unavailable" if missing else "ok",
            "requested": requested,
            "missing_targets": missing,
            "invalid_targets": invalid,
        }
    )
    if invalid:
        result["reason"] = f"invalid target devices: {', '.join(invalid)}"
    elif missing:
        result["reason"] = f"target devices unavailable: {', '.join(missing)}"
    return result


def run_doctor(
    db: Database,
    workspace: WorkspaceConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    target_devices: list[str] | None = None,
) -> dict[str, Any]:
    from vfieval.metrics.health import metrics_health

    from vfieval.ffmpeg_exe import resolve_ffmpeg

    _ffmpeg_exe = resolve_ffmpeg() or "ffmpeg"
    ffmpeg = _command_probe([_ffmpeg_exe, "-hide_banner", "-version"])
    ffprobe = _command_probe(["ffprobe", "-hide_banner", "-version"])
    encoders = _command_probe([_ffmpeg_exe, "-hide_banner", "-encoders"])
    if encoders.get("available"):
        executable = str(encoders.get("path") or "ffmpeg")
        try:
            completed = subprocess.run(
                [executable, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            encoders["libx264"] = "libx264" in (completed.stdout or "")
            if completed.returncode == 0 and not encoders["libx264"]:
                encoders["status"] = "unavailable"
                encoders["reason"] = "FFmpeg is available but libx264 is missing"
        except (OSError, subprocess.SubprocessError) as exc:
            encoders["status"] = "error"
            encoders["reason"] = f"{type(exc).__name__}: {exc}"
    try:
        metric_health = metrics_health(workspace)
        metric_rows = metric_health.get("metrics") or {}
        unavailable_metrics = sorted(
            str(name)
            for name, row in metric_rows.items()
            if not isinstance(row, dict) or row.get("status") != "available"
        )
        metric_health["status"] = "unavailable" if unavailable_metrics else "ok"
        metric_health["unavailable_metrics"] = unavailable_metrics
    except Exception as exc:
        metric_health = {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    try:
        storage = health_snapshot(db, workspace)["storage"]
    except Exception as exc:
        storage = {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    checks: dict[str, Any] = {
        "database": _database_probe(db),
        "workspace_permissions": _workspace_write_probe(workspace),
        "storage": storage,
        "port": _port_probe(host, port),
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "ffmpeg_encoders": encoders,
        "devices": _device_probe(target_devices),
        "metrics": metric_health,
        "dependencies": {
            name: _package_version(name)
            for name in ("torch", "torch-npu", "numpy", "Pillow", "opencv-python")
        },
    }
    hard_failures = [name for name, check in checks.items() if isinstance(check, dict) and check.get("status") == "error"]
    unavailable = [
        name
        for name, check in checks.items()
        if isinstance(check, dict) and check.get("status") == "unavailable"
    ]
    warnings = [name for name, check in checks.items() if isinstance(check, dict) and check.get("status") == "warning"]
    return {
        "ok": not hard_failures and not unavailable,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release": release_info(),
        "workspace": "<workspace>",
        "checks": checks,
        "summary": {
            "errors": hard_failures,
            "unavailable": unavailable,
            "warnings": warnings,
        },
    }


_SENSITIVE_KEYS = (
    "token",
    "secret",
    "password",
    "authorization",
    "cookie",
    "api_key",
    "access_key",
    "private_key",
)


def _sanitize_text(value: str, workspace: WorkspaceConfig) -> str:
    result = value
    for root in _diagnostic_redaction_roots(workspace):
        for variant in {root, root.replace("\\", "/")}:
            result = re.sub(re.escape(variant), "<workspace>", result, flags=re.I)
    result = re.sub(r"(/evaluate/)[A-Za-z0-9_-]+", r"\1<redacted>", result)
    result = re.sub(r"(/api/blind/)[A-Za-z0-9_-]+", r"\1<redacted>", result)
    result = re.sub(r"(/tasks/)[A-Za-z0-9_-]+", r"\1<redacted>", result)
    result = re.sub(r"(/reviews/)[A-Za-z0-9_-]+", r"\1<redacted>", result)
    result = re.sub(
        r"([?&](?:token|key|secret|signature|password)=)[^&\s\"']+",
        r"\1<redacted>",
        result,
        flags=re.I,
    )
    result = re.sub(r"(\bBearer\s+)[A-Za-z0-9._~+/=-]+", r"\1<redacted>", result, flags=re.I)
    result = re.sub(
        r'((?:"|\b)(?:token|secret|password|authorization|cookie|api_key|access_key|private_key)"?\s*[:=]\s*)'
        r'("(?:\\.|[^"\\])*"|[^\s,;}]+)',
        r'\1"<redacted>"',
        result,
        flags=re.I,
    )
    return result


def _diagnostic_redaction_roots(workspace: WorkspaceConfig) -> tuple[str, ...]:
    project_root = Path(os.getenv("VFIEVAL_PROJECT_ROOT") or workspace.root.parent).resolve()
    configured = [
        workspace.root.resolve(),
        workspace.root.parent.resolve(),
        project_root,
        Path(os.getenv("VFIEVAL_MODELS_DIR") or project_root / "models").resolve(),
        Path(os.getenv("VFIEVAL_VIDEOS_DIR") or project_root / "videos").resolve(),
        Path(os.getenv("VFIEVAL_CHECKPOINTS_DIR") or project_root / "checkpoints").resolve(),
        Path(os.getenv("VFIEVAL_METRIC_ASSETS_DIR") or project_root / "set" / "metrics").resolve(),
    ]
    return tuple(
        sorted(
            {str(path) for path in configured if str(path)},
            key=len,
            reverse=True,
        )
    )


def _sanitize(value: Any, workspace: WorkspaceConfig) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value, workspace)
    if isinstance(value, dict):
        return {
            str(key): "<redacted>"
            if any(part in str(key).lower() for part in _SENSITIVE_KEYS)
            else _sanitize(child, workspace)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(child, workspace) for child in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _tail_text(path: Path, max_bytes: int = 2 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        size = path.stat().st_size
        if size > max_bytes:
            handle.seek(-max_bytes, os.SEEK_END)
            handle.readline()
        return handle.read().decode("utf-8", errors="replace")


def _target_log_lines(text: str, kind: str, target_id: int) -> str:
    identity_key = "run_id" if kind == "run" else "campaign_id"
    selected: list[str] = []
    for line in text.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        value = payload.get(identity_key)
        if value is None and isinstance(payload.get("context"), dict):
            value = payload["context"].get(identity_key)
        try:
            matches = int(value) == int(target_id)
        except (TypeError, ValueError):
            matches = False
        if matches:
            selected.append(line)
    return "\n".join(selected) + ("\n" if selected else "")


def _trusted_log_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    allowed_names = {
        "model_load.log",
        "output_health.log",
        "artifact_integrity.json",
        "alignment.json",
        "manifest.json",
    }
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if len(relative.parts) > 3:
            continue
        suffix = path.suffix.lower()
        if (
            path.name in allowed_names
            or suffix in {".log", ".jsonl"}
            or (suffix == ".json" and relative.parts[0] in {"artifact_integrity", "logs", "shards"})
        ):
            paths.append(path)
    return sorted(paths, key=lambda candidate: candidate.as_posix())


def _collect_diagnostic_logs(
    workspace: WorkspaceConfig,
    *,
    kind: str,
    target_id: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    collected: dict[str, str] = {}
    total_bytes = 0
    omitted: list[str] = []

    def add(archive_name: str, text: str) -> None:
        nonlocal total_bytes
        sanitized = str(_sanitize(text, workspace))
        encoded = sanitized.encode("utf-8")
        remaining = DIAGNOSTICS_LOG_BUDGET_BYTES - total_bytes
        if remaining <= 0:
            omitted.append(archive_name)
            return
        if len(encoded) > remaining:
            encoded = encoded[-remaining:]
            sanitized = encoded.decode("utf-8", errors="replace")
            omitted.append(f"{archive_name}:truncated")
        collected[archive_name] = sanitized
        total_bytes += len(encoded)

    local_root = (
        workspace.runs_dir / str(target_id) / "logs"
        if kind == "run"
        else workspace.evaluations_dir / str(target_id)
    ).resolve()
    trusted_parent = (
        workspace.runs_dir.resolve()
        if kind == "run"
        else workspace.evaluations_dir.resolve()
    )
    if _is_path_within(local_root, trusted_parent):
        for path in _trusted_log_files(local_root):
            relative = path.relative_to(local_root).as_posix()
            text = _tail_text(path, DIAGNOSTICS_LOG_FILE_BYTES)
            if text:
                add(f"logs/{kind}/{relative}", text)

    global_log_dir = workspace.root / "logs"
    if global_log_dir.is_dir():
        candidates = sorted(
            [
                path
                for pattern in ("*.jsonl", "*.jsonl.1")
                for path in global_log_dir.glob(pattern)
                if path.is_file() and not path.is_symlink()
            ],
            key=lambda candidate: candidate.name,
        )
        for path in candidates:
            selected = _target_log_lines(
                _tail_text(path, DIAGNOSTICS_LOG_FILE_BYTES),
                kind,
                target_id,
            )
            if selected:
                stem = path.name.removesuffix(".jsonl").replace(".jsonl.", ".")
                add(f"logs/{stem}-tail.jsonl", selected)

    return collected, {
        "included_files": len(collected),
        "included_bytes": total_bytes,
        "budget_bytes": DIAGNOSTICS_LOG_BUDGET_BYTES,
        "omitted": omitted,
    }


def _is_path_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def create_diagnostics_bundle(
    db: Database,
    workspace: WorkspaceConfig,
    *,
    run_id: int | None = None,
    campaign_id: int | None = None,
    output: str | Path | None = None,
) -> Path:
    if (run_id is None) == (campaign_id is None):
        raise ValueError("select exactly one of run_id or campaign_id")
    kind = "run" if run_id is not None else "campaign"
    target_id = int(run_id if run_id is not None else campaign_id)
    if target_id <= 0:
        raise ValueError(f"{kind}_id must be positive")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(output).resolve() if output else (
        workspace.tmp_dir
        / "diagnostics"
        / f"vfieval-{kind}-{target_id}-{timestamp}-diagnostics.zip"
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"diagnostics output already exists: {output_path}")

    if run_id is not None:
        selected = {
            "run": db.get_run(target_id),
            "jobs": db.list_run_jobs(target_id),
            "artifact_summary": db.get("SELECT COUNT(*) AS count FROM artifacts WHERE job_id IN (SELECT job_id FROM run_jobs WHERE run_id = ?)", (target_id,)),
        }
    else:
        campaign = db.get("SELECT * FROM evaluation_campaigns_v2 WHERE id = ?", (target_id,))
        if campaign is None:
            raise KeyError(f"campaign {target_id} not found")
        selected = {
            "campaign": campaign,
            "preparation": db.get(
                "SELECT * FROM evaluation_preparations_v2 WHERE campaign_id = ? ORDER BY id DESC LIMIT 1",
                (target_id,),
            ),
            "counts": db.get(
                "SELECT COUNT(*) AS tasks FROM evaluation_tasks_v2 WHERE campaign_id = ?",
                (target_id,),
            ),
        }

    manifest = {
        "format": "vfieval-diagnostics-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection": {"kind": kind, "id": target_id},
        "privacy": "paths and credential-like fields are redacted",
    }
    log_tails, log_summary = _collect_diagnostic_logs(
        workspace,
        kind=kind,
        target_id=target_id,
    )
    manifest["logs"] = log_summary
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        archive.writestr("doctor.json", json.dumps(_sanitize(run_doctor(db, workspace), workspace), indent=2, ensure_ascii=False, default=str))
        archive.writestr("selection.json", json.dumps(_sanitize(selected, workspace), indent=2, ensure_ascii=False, default=str))
        for archive_name, log_text in log_tails.items():
            archive.writestr(archive_name, log_text)
    return output_path
