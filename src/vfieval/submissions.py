from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

from vfieval.db import Database


_SUBMISSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_FINGERPRINT_IGNORED_FIELDS = {
    "submission_id",
    "preflight_token",
    "risk_ack_fingerprint",
    "_preflight_result",
    "_preflight_physical_inputs",
    "_workload_risk_ack_scope_fingerprint",
    "_confirm_current_workload",
}
T = TypeVar("T")


class SubmissionConflict(ValueError):
    def __init__(self, message: str, *, code: str):
        super().__init__(message)
        self.code = str(code)


def submission_fingerprint(body: dict[str, Any]) -> str:
    stable = {
        str(key): value
        for key, value in body.items()
        if str(key) not in _FINGERPRINT_IGNORED_FIELDS
        and not str(key).startswith("_")
    }
    encoded = json.dumps(
        stable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def idempotent_create(
    db: Database,
    *,
    scope: str,
    body: dict[str, Any],
    resource_type: str,
    create: Callable[[], T],
    resource_id: Callable[[T], int],
    replay: Callable[[int], T],
    mark_replay: Callable[[T, bool, str], T],
    recover: Callable[[str, str], T | None] | None = None,
) -> T:
    submission_id = str(body.get("submission_id") or "").strip()
    if not submission_id:
        return create()
    if _SUBMISSION_ID_RE.fullmatch(submission_id) is None:
        raise ValueError(
            "submission_id must be 8-128 characters using letters, digits, '.', '_', ':', or '-'"
        )

    fingerprint = submission_fingerprint(body)
    reservation = db.begin_submission(scope, submission_id, fingerprint)
    state = str(reservation.get("state") or "")
    if state == "conflict":
        raise SubmissionConflict(
            "submission_id was already used with a different request payload",
            code="submission_payload_conflict",
        )
    if state == "pending":
        reservation = _wait_for_submission(db, scope, submission_id, fingerprint)
        state = str(reservation.get("state") or "")
    if state == "completed":
        existing_id = reservation.get("resource_id")
        if existing_id is None:
            raise SubmissionConflict(
                "completed submission has no resource binding",
                code="submission_resource_missing",
            )
        try:
            existing = replay(int(existing_id))
        except KeyError as exc:
            raise SubmissionConflict(
                "the resource created by this submission is no longer available",
                code="submission_resource_missing",
            ) from exc
        return mark_replay(existing, True, submission_id)
    if state != "claimed":
        raise SubmissionConflict(
            "submission is already in progress; retry with the same submission_id",
            code="submission_in_progress",
        )

    claim_token = str(reservation["claim_token"])
    if recover is not None:
        recovered = recover(submission_id, fingerprint)
        if recovered is not None:
            recovered_id = int(resource_id(recovered))
            if not db.complete_submission(
                scope,
                submission_id,
                claim_token,
                resource_type=resource_type,
                resource_id=recovered_id,
            ):
                raise SubmissionConflict(
                    "submission ownership was lost while recovering its existing resource",
                    code="submission_claim_lost",
                )
            return mark_replay(recovered, True, submission_id)

    try:
        created = create()
        created_id = int(resource_id(created))
    except BaseException:
        db.abandon_submission(scope, submission_id, claim_token)
        raise
    if not db.complete_submission(
        scope,
        submission_id,
        claim_token,
        resource_type=resource_type,
        resource_id=created_id,
    ):
        # The resource already exists at this point. Keep the reservation so a
        # later stale-lease recovery can reconcile it instead of blindly
        # creating a duplicate.
        raise SubmissionConflict(
            "submission ownership was lost before the resource could be published",
            code="submission_claim_lost",
        )
    return mark_replay(created, False, submission_id)


def _wait_for_submission(
    db: Database,
    scope: str,
    submission_id: str,
    fingerprint: str,
    *,
    timeout: float = 3.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, float(timeout))
    while time.monotonic() < deadline:
        row = db.get_submission(scope, submission_id)
        if row is None:
            break
        if str(row.get("request_fingerprint") or "") != fingerprint:
            return {**row, "state": "conflict"}
        if str(row.get("status") or "") == "completed":
            return {**row, "state": "completed"}
        time.sleep(0.05)
    # A lease may have expired while this request was waiting. Re-enter the
    # atomic reservation path once so a crashed creator can be reclaimed.
    return db.begin_submission(scope, submission_id, fingerprint)
