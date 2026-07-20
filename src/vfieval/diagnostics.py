from __future__ import annotations

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


PROCESS_STARTED_AT = time.time()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def release_info() -> dict[str, Any]:
    return {
        "version": _package_version("vfieval") or "0.1.0",
        "build_id": os.getenv("VFIEVAL_BUILD_ID", "development"),
        "started_at": datetime.fromtimestamp(PROCESS_STARTED_AT, timezone.utc).isoformat(),
        "uptime_seconds": max(0.0, time.time() - PROCESS_STARTED_AT),
        "python": platform.python_version(),
    }


def _schema_summary(db: Database) -> dict[str, Any]:
    try:
        latest = db.get("SELECT version, applied_at FROM schema_migrations ORDER BY applied_at DESC LIMIT 1")
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
        now = time.time()
        return method(now - 90.0, now=now)
    try:
        rows = db.query(
            "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status ORDER BY status"
        )
        return {"by_status": {str(row["status"]): int(row["count"]) for row in rows}}
    except sqlite3.Error as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def health_snapshot(
    db: Database,
    workspace: WorkspaceConfig,
    *,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usage = shutil.disk_usage(workspace.root)
    storage = {
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
    }
    storage["status"] = (
        "error"
        if storage["free_bytes"] < 1024**3
        else "warning"
        if storage["free_bytes"] < 5 * 1024**3 or storage["free_bytes"] * 100 < storage["total_bytes"] * 5
        else "ok"
    )
    return {
        "ok": True,
        "release": release_info(),
        "schema": _schema_summary(db),
        "storage": storage,
        "leases": _lease_summary(db),
        "catalog": dict(catalog or {}),
    }


def _command_probe(command: list[str], *, timeout: float = 8.0) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if not executable:
        return {"status": "warning", "available": False, "reason": f"{command[0]} not found"}
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


def _port_probe() -> dict[str, Any]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return {"status": "ok", "bindable": True, "sample_port": int(sock.getsockname()[1])}
    except OSError as exc:
        return {"status": "error", "bindable": False, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        sock.close()


def run_doctor(db: Database, workspace: WorkspaceConfig) -> dict[str, Any]:
    from vfieval.metrics.health import metrics_health
    from vfieval.worker import detect_capabilities

    ffmpeg = _command_probe(["ffmpeg", "-hide_banner", "-version"])
    ffprobe = _command_probe(["ffprobe", "-hide_banner", "-version"])
    encoders = _command_probe(["ffmpeg", "-hide_banner", "-encoders"])
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
                encoders["status"] = "warning"
                encoders["reason"] = "FFmpeg is available but libx264 is missing"
        except (OSError, subprocess.SubprocessError):
            pass
    checks: dict[str, Any] = {
        "database": _database_probe(db),
        "workspace_permissions": _workspace_write_probe(workspace),
        "storage": health_snapshot(db, workspace)["storage"],
        "port": _port_probe(),
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "ffmpeg_encoders": encoders,
        "devices": detect_capabilities(),
        "metrics": metrics_health(workspace),
        "dependencies": {
            name: _package_version(name)
            for name in ("torch", "torch-npu", "numpy", "Pillow", "opencv-python")
        },
    }
    hard_failures = [name for name, check in checks.items() if isinstance(check, dict) and check.get("status") == "error"]
    warnings = [name for name, check in checks.items() if isinstance(check, dict) and check.get("status") == "warning"]
    return {
        "ok": not hard_failures,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release": release_info(),
        "workspace": "<workspace>",
        "checks": checks,
        "summary": {"errors": hard_failures, "warnings": warnings},
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
    roots = (str(workspace.root.resolve()), str(workspace.root.parent.resolve()))
    result = value
    for root in roots:
        result = result.replace(root, "<workspace>").replace(root.replace("\\", "/"), "<workspace>")
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
    log_tails: dict[str, str] = {}
    log_dir = workspace.root / "logs"
    if log_dir.is_dir():
        for log_path in sorted(log_dir.glob("*.jsonl")):
            log_text = _tail_text(log_path)
            if log_text:
                log_tails[log_path.name] = str(_sanitize(log_text, workspace))
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        archive.writestr("doctor.json", json.dumps(_sanitize(run_doctor(db, workspace), workspace), indent=2, ensure_ascii=False, default=str))
        archive.writestr("selection.json", json.dumps(_sanitize(selected, workspace), indent=2, ensure_ascii=False, default=str))
        for filename, log_text in log_tails.items():
            stem = Path(filename).stem
            archive.writestr(f"logs/{stem}-tail.jsonl", log_text)
    return output_path
