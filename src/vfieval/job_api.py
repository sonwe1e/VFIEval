from __future__ import annotations

from http import HTTPStatus
from typing import Any

from vfieval.db import Database
from vfieval.orchestration import wake_job_supervisor


class JobApiError(Exception):
    """A controlled internal-Job protocol rejection."""

    def __init__(
        self,
        status: HTTPStatus,
        message: str,
        code: str = "Error",
    ):
        super().__init__(message)
        self.status = status
        self.code = code


def _missing_field(name: str) -> JobApiError:
    return JobApiError(
        HTTPStatus.BAD_REQUEST,
        f"missing field '{name}'",
    )


def _worker_id(body: dict[str, Any]) -> str:
    if "worker_id" not in body:
        raise _missing_field("worker_id")
    worker_id = str(body.get("worker_id") or "").strip()
    if not worker_id:
        raise JobApiError(
            HTTPStatus.BAD_REQUEST,
            "worker_id must be a non-empty string",
            "JobRequestInvalid",
        )
    return worker_id


def register_worker_request(
    db: Database,
    body: dict[str, Any],
) -> dict[str, Any]:
    worker_id = _worker_id(body)
    role = body.get("role", "remote")
    db.register_worker(worker_id, role, body.get("capabilities") or {})
    return {"worker_id": worker_id, "worker": db.get_worker(worker_id)}


def create_job_request(
    db: Database,
    body: dict[str, Any],
) -> tuple[dict[str, Any], HTTPStatus]:
    if "kind" not in body:
        raise _missing_field("kind")
    kind = body["kind"]
    if kind not in {"decode", "inference", "metric"}:
        raise JobApiError(
            HTTPStatus.BAD_REQUEST,
            "kind must be decode, inference, or metric",
        )
    job_id = db.create_job(kind, body.get("payload") or {})
    wake_job_supervisor(db)
    return {"job_id": job_id, "kind": kind}, HTTPStatus.CREATED


def claim_job_request(
    db: Database,
    body: dict[str, Any],
) -> dict[str, Any]:
    worker_id = _worker_id(body)
    role = body.get("role", "remote")
    raw_kinds = body.get("kinds") or []
    if not isinstance(raw_kinds, list):
        raise JobApiError(
            HTTPStatus.BAD_REQUEST,
            "kinds must be a list",
            "JobRequestInvalid",
        )
    kinds = list(raw_kinds)
    db.register_worker(worker_id, role, body.get("capabilities") or {})
    return {"job": db.claim_next_job(worker_id, kinds)}


def _attempt(body: dict[str, Any]) -> int:
    try:
        return int(body.get("attempt"))
    except (TypeError, ValueError):
        return 0


def _fence(
    db: Database,
    job_id: int,
    body: dict[str, Any],
    *,
    heartbeat: bool,
) -> tuple[str, int, str]:
    worker_id = str(body.get("worker_id") or "").strip()
    lease_token = str(body.get("lease_token") or "").strip()
    attempt = _attempt(body)
    if worker_id and attempt > 0 and lease_token:
        return worker_id, attempt, lease_token
    if worker_id:
        current_job = db.get_job(job_id)
        if (
            str(current_job.get("status") or "") != "running"
            or str(current_job.get("worker_id") or "") != worker_id
        ):
            raise JobApiError(
                HTTPStatus.CONFLICT,
                "job lease is no longer owned by this worker; stop the stale worker",
                "JobLeaseLost",
            )
    message = (
        "worker_id, attempt, and lease_token are required for an "
        "owner-fenced Job heartbeat"
        if heartbeat
        else "worker_id, attempt, and lease_token are required for Job callbacks"
    )
    raise JobApiError(
        HTTPStatus.BAD_REQUEST,
        message,
        "JobFenceRequired",
    )


def job_callback_request(
    db: Database,
    job_id: int,
    action: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    if action not in {"complete", "fail", "progress"}:
        raise JobApiError(
            HTTPStatus.BAD_REQUEST,
            "unsupported Job callback action",
            "JobRequestInvalid",
        )
    worker_id, attempt, lease_token = _fence(
        db,
        job_id,
        body,
        heartbeat=False,
    )
    if action == "complete":
        accepted = db.complete_job(
            job_id,
            body.get("result") or {},
            worker_id=worker_id,
            attempt=attempt,
            lease_token=lease_token,
        )
    elif action == "fail":
        accepted = db.fail_job(
            job_id,
            body.get("error") or {},
            worker_id=worker_id,
            attempt=attempt,
            lease_token=lease_token,
        )
    else:
        try:
            current = int(body.get("current", 0))
        except (TypeError, ValueError) as exc:
            raise JobApiError(
                HTTPStatus.BAD_REQUEST,
                str(exc),
            ) from exc
        accepted = db.update_job_progress(
            job_id,
            current,
            body.get("total"),
            worker_id=worker_id,
            attempt=attempt,
            lease_token=lease_token,
        )
    if not accepted:
        raise JobApiError(
            HTTPStatus.CONFLICT,
            f"job {job_id} state rejected {action}",
            "JobStateConflict",
        )
    return {"job_id": job_id, "status": action}


def heartbeat_job_request(
    db: Database,
    job_id: int,
    body: dict[str, Any],
) -> dict[str, Any]:
    worker_id, attempt, lease_token = _fence(
        db,
        job_id,
        body,
        heartbeat=True,
    )
    if not db.heartbeat_job(job_id, worker_id, attempt, lease_token):
        raise JobApiError(
            HTTPStatus.CONFLICT,
            "job lease is no longer owned by this worker; stop the stale worker",
            "JobLeaseLost",
        )
    db.touch_worker(worker_id, body.get("capabilities") or None)
    return {"job_id": job_id, "status": "heartbeat"}
