from __future__ import annotations

from typing import Any, Iterable

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.diagnostics import health_snapshot
from vfieval.metrics import METRIC_NAMES


def runtime_health_payload(
    db: Database,
    workspace: WorkspaceConfig,
    *,
    catalog_sync: Any,
    cleanup_service: Any,
    job_recovery: Any | None = None,
    job_supervisor: Any | None = None,
    metric_names: Iterable[str] = METRIC_NAMES,
) -> dict[str, Any]:
    """Assemble the stable liveness/readiness response without doing work."""

    catalog_status = catalog_sync.status()
    cache_coordination = cleanup_service.cache_coordination_status()
    if not isinstance(cache_coordination, dict):
        cache_coordination = {
            "state": "unknown",
            "ready": False,
            "version": None,
        }
    job_recovery_status = (
        job_recovery.status()
        if job_recovery is not None
        else {"running": False, "leases": db.job_lease_summary()}
    )
    if job_supervisor is None:
        job_supervisor_status = {
            "configured": False,
            "required": False,
            "running": False,
            "thread_slots": {},
            "process_slots": {},
            "last_scan_at": None,
            "last_error": None,
        }
    else:
        job_supervisor_status = {
            **job_supervisor.status(),
            "configured": True,
            "required": True,
        }
    maintenance = {
        "catalog": catalog_status,
        "cache_coordination": cache_coordination,
        "job_recovery": job_recovery_status,
        "job_supervisor": job_supervisor_status,
        **db.cleanup_backlog_counts(),
    }
    health = health_snapshot(
        db,
        workspace,
        catalog=catalog_status,
        maintenance=maintenance,
    )
    reasons = list(health["reasons"])
    if job_supervisor_status["required"]:
        if job_supervisor_status.get("last_error"):
            reasons.append("job_supervisor_failed")
        if not job_supervisor_status.get("running"):
            reasons.append("job_supervisor_not_running")
    ready = not reasons
    return {
        "ok": ready,
        "live": health["live"],
        "ready": ready,
        "reasons": reasons,
        "metrics": list(metric_names),
        "release": health["release"],
        "schema": health["schema"],
        "storage": health["storage"],
        "leases": health["leases"],
        "queues": health["queues"],
        "maintenance": maintenance,
    }
