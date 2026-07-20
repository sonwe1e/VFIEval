from __future__ import annotations

import threading
from typing import Any, Callable

from vfieval.db import Database, utc_ts


DEFAULT_JOB_HEARTBEAT_INTERVAL_SECONDS = 10.0
# 180 s = 18× the heartbeat interval.  On an 8-card Ascend A3 the metric and
# postprocess threads compete for the Python GIL, which can starve the
# heartbeat thread for 20–40 s under peak load.  90 s was too close to that
# ceiling; 180 s provides a comfortable margin without delaying detection of
# genuinely dead workers (JobRecoveryService sweeps every 15 s, so a truly
# silent worker is still fenced within ~3 minutes).
DEFAULT_JOB_LEASE_TIMEOUT_SECONDS = 180.0
DEFAULT_JOB_RECOVERY_INTERVAL_SECONDS = 15.0


class JobLeaseHeartbeat:
    """Renew one claimed Job lease for the whole worker execution boundary."""

    def __init__(
        self,
        db: Database,
        job_id: int,
        worker_id: str,
        *,
        interval_seconds: float = DEFAULT_JOB_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self.db = db
        self.job_id = int(job_id)
        self.worker_id = str(worker_id)
        self.interval_seconds = max(0.05, float(interval_seconds))
        self._stop_event = threading.Event()
        self._lost_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None

    @property
    def lease_lost(self) -> bool:
        return self._lost_event.is_set()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return not self.lease_lost
        self._stop_event.clear()
        self._lost_event.clear()
        if not self.db.heartbeat_job(self.job_id, self.worker_id):
            self._lost_event.set()
            return False
        self._thread = threading.Thread(
            target=self._run,
            name=f"vfieval-job-heartbeat-{self.job_id}",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))
        if thread is None or not thread.is_alive():
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                accepted = self.db.heartbeat_job(self.job_id, self.worker_id)
            except Exception as exc:  # SQLite contention is retried on the next pulse.
                self._last_error = f"{type(exc).__name__}: {exc}"
                continue
            self._last_error = None
            if not accepted:
                self._lost_event.set()
                return

    def __enter__(self) -> JobLeaseHeartbeat:
        if not self.start():
            raise RuntimeError(f"Job {self.job_id} rejected its worker heartbeat lease")
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.stop()


def recover_stale_jobs(
    db: Database,
    *,
    lease_timeout_seconds: float = DEFAULT_JOB_LEASE_TIMEOUT_SECONDS,
    now: float | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Recover one bounded batch of workers that stopped renewing leases."""
    timeout = max(1.0, float(lease_timeout_seconds))
    observed_at = utc_ts() if now is None else float(now)
    return db.recover_stale_jobs(
        observed_at - timeout,
        recovered_at=observed_at,
        lease_timeout_seconds=timeout,
        limit=limit,
    )


class JobRecoveryService:
    """Small background coordinator for generic decode/Run Job recovery.

    Campaign package preparation has its own claim-token lease and is
    intentionally outside this service.
    """

    def __init__(
        self,
        db: Database,
        *,
        lease_timeout_seconds: float = DEFAULT_JOB_LEASE_TIMEOUT_SECONDS,
        scan_interval_seconds: float = DEFAULT_JOB_RECOVERY_INTERVAL_SECONDS,
        batch_limit: int = 100,
        on_recovered: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> None:
        self.db = db
        self.lease_timeout_seconds = max(1.0, float(lease_timeout_seconds))
        self.scan_interval_seconds = max(0.1, float(scan_interval_seconds))
        self.batch_limit = max(1, int(batch_limit))
        self.on_recovered = on_recovered
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status_lock = threading.Lock()
        self._last_sweep_at: float | None = None
        self._last_error: str | None = None
        self._recovered_total = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.run_forever,
            name="vfieval-job-recovery",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))
        if thread is None or not thread.is_alive():
            self._thread = None

    def run_once(self, *, now: float | None = None) -> list[dict[str, Any]]:
        observed_at = utc_ts() if now is None else float(now)
        recovered = recover_stale_jobs(
            self.db,
            lease_timeout_seconds=self.lease_timeout_seconds,
            now=observed_at,
            limit=self.batch_limit,
        )
        with self._status_lock:
            self._last_sweep_at = observed_at
            self._last_error = None
            self._recovered_total += len(recovered)
        if recovered and self.on_recovered is not None:
            self.on_recovered(recovered)
        return recovered

    def run_forever(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                with self._status_lock:
                    self._last_sweep_at = utc_ts()
                    self._last_error = f"{type(exc).__name__}: {exc}"
            if self._stop_event.wait(self.scan_interval_seconds):
                return

    def status(self, *, now: float | None = None) -> dict[str, Any]:
        observed_at = utc_ts() if now is None else float(now)
        with self._status_lock:
            status = {
                "running": bool(self._thread is not None and self._thread.is_alive()),
                "last_sweep_at": self._last_sweep_at,
                "last_error": self._last_error,
                "recovered_total": self._recovered_total,
                "lease_timeout_seconds": self.lease_timeout_seconds,
                "scan_interval_seconds": self.scan_interval_seconds,
            }
        status["leases"] = self.db.job_lease_summary(
            observed_at - self.lease_timeout_seconds,
            now=observed_at,
        )
        return status

    def __enter__(self) -> JobRecoveryService:
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.stop()
