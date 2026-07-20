from __future__ import annotations

import threading
import time
from typing import Any, Callable

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.media_assets import sync_catalog


CatalogSync = Callable[[Database, WorkspaceConfig, bool], dict[str, Any] | None]


def _default_sync(
    db: Database,
    workspace: WorkspaceConfig,
    include_runs: bool,
) -> dict[str, Any]:
    catalog = sync_catalog(db, workspace, include_runs=bool(include_runs))
    return {
        "catalog": catalog,
        # ``sync_catalog`` already reconciles canonical GT Items. Repeating it
        # here both doubled the work and attempted to cast its report dict to
        # int, causing every default coordinator run to end in ``failed``.
        "canonical_items": int(catalog.get("media_items") or 0),
    }


class CatalogSyncCoordinator:
    """Coalesce expensive folder/catalog reconciliation into one background job."""

    def __init__(
        self,
        db: Database,
        workspace: WorkspaceConfig,
        *,
        sync_callback: CatalogSync | None = None,
    ) -> None:
        self.db = db
        self.workspace = workspace
        self._sync_callback = sync_callback or _default_sync
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._revision = 0
        self._state: dict[str, Any] = {
            "state": "idle",
            "phase": "idle",
            "catalog_revision": 0,
            "requested_at": None,
            "started_at": None,
            "completed_at": None,
            "duration_seconds": None,
            "include_runs": False,
            "error": None,
            "report": {},
        }

    def request_sync(self, *, include_runs: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if include_runs:
                    self._state["include_runs"] = True
                return self._snapshot_locked(joined=True)
            now = time.time()
            self._state.update(
                {
                    "state": "requested",
                    "phase": "catalog_reconciliation",
                    "requested_at": now,
                    "started_at": None,
                    "completed_at": None,
                    "duration_seconds": None,
                    "include_runs": bool(include_runs),
                    "error": None,
                    "report": {},
                }
            )
            thread = threading.Thread(
                target=self._run,
                name="vfieval-catalog-sync",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return self._snapshot_locked(joined=False)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked(joined=False)

    def wait(self, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        return self.status()

    def _run(self) -> None:
        with self._lock:
            self._state["state"] = "running"
            self._state["started_at"] = time.time()
            include_runs = bool(self._state.get("include_runs"))
        try:
            while True:
                report = self._sync_callback(self.db, self.workspace, include_runs) or {}
                with self._lock:
                    if bool(self._state.get("include_runs")) and not include_runs:
                        include_runs = True
                        continue
                    completed_at = time.time()
                    started_at = float(self._state.get("started_at") or completed_at)
                    self._revision += 1
                    self._state.update(
                        {
                            "state": "completed",
                            "phase": "completed",
                            "completed_at": completed_at,
                            "duration_seconds": max(0.0, completed_at - started_at),
                            "catalog_revision": self._revision,
                            "error": None,
                            "report": report,
                        }
                    )
                    # Clear ownership in the same critical section as the final
                    # scope check. A late include_runs join now starts a new
                    # reconciliation instead of being lost while this thread
                    # is between "completed" and returning.
                    self._thread = None
                    return
        except Exception as exc:
            with self._lock:
                completed_at = time.time()
                started_at = float(self._state.get("started_at") or completed_at)
                self._state.update(
                    {
                        "state": "failed",
                        "phase": "failed",
                        "completed_at": completed_at,
                        "duration_seconds": max(0.0, completed_at - started_at),
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "report": {},
                    }
                )
                self._thread = None
            return

    def _snapshot_locked(self, *, joined: bool) -> dict[str, Any]:
        return {
            **self._state,
            "catalog_revision": int(self._revision),
            "joined_existing": bool(joined),
        }
