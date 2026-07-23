from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from vfieval.api_validation import ApiValidationError, pagination_params
from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.job_api import (
    JobApiError,
    claim_job_request,
    create_job_request,
    heartbeat_job_request,
    job_callback_request,
)
from vfieval.runtime_api import runtime_health_payload
from vfieval.server import _make_handler


class _StatusProvider:
    def __init__(self, payload: dict):
        self.payload = dict(payload)

    def status(self) -> dict:
        return dict(self.payload)


class _CleanupStatus:
    def cache_coordination_status(self) -> dict:
        return {
            "state": "ready",
            "ready": True,
            "version": "test",
        }


class ApiModuleTests(unittest.TestCase):
    def _workspace_db(self, root: Path) -> tuple[WorkspaceConfig, Database]:
        workspace = WorkspaceConfig.from_root(root / ".vfieval")
        workspace.ensure()
        db = Database(workspace.db_path)
        db.init()
        return workspace, db

    def _health(
        self,
        root: Path,
        *,
        job_supervisor: _StatusProvider | None,
    ) -> dict:
        workspace, db = self._workspace_db(root)
        clean_health = {
            "ok": True,
            "live": True,
            "ready": True,
            "reasons": [],
            "release": {"build_id": "test"},
            "schema": {},
            "storage": {},
            "leases": {},
            "queues": {},
        }
        with patch(
            "vfieval.runtime_api.health_snapshot",
            return_value=clean_health,
        ):
            return runtime_health_payload(
                db,
                workspace,
                catalog_sync=_StatusProvider({"state": "completed"}),
                cleanup_service=_CleanupStatus(),
                job_recovery=_StatusProvider({"running": True, "last_error": None}),
                job_supervisor=job_supervisor,
                metric_names=("metric-a",),
            )

    def test_pagination_validation_preserves_clamped_contract(self) -> None:
        self.assertEqual(
            pagination_params(
                {"page": ["0"], "page_size": ["999"]},
                default_page_size=50,
            ),
            (1, 200),
        )
        with self.assertRaisesRegex(ApiValidationError, "page must be an integer"):
            pagination_params(
                {"page": ["not-a-number"]},
                default_page_size=50,
            )

    def test_runtime_health_does_not_require_an_unconfigured_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            health = self._health(Path(tmp), job_supervisor=None)

        self.assertTrue(health["ready"])
        self.assertNotIn("job_supervisor_not_running", health["reasons"])
        self.assertFalse(health["maintenance"]["job_supervisor"]["configured"])
        self.assertFalse(health["maintenance"]["job_supervisor"]["required"])

    def test_runtime_health_rejects_a_stopped_required_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            health = self._health(
                Path(tmp),
                job_supervisor=_StatusProvider(
                    {
                        "running": False,
                        "thread_slots": {},
                        "process_slots": {},
                        "last_scan_at": 123.0,
                        "last_error": None,
                    }
                ),
            )

        self.assertFalse(health["ready"])
        self.assertFalse(health["ok"])
        self.assertIn("job_supervisor_not_running", health["reasons"])

    def test_runtime_health_rejects_a_supervisor_last_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            health = self._health(
                Path(tmp),
                job_supervisor=_StatusProvider(
                    {
                        "running": True,
                        "thread_slots": {},
                        "process_slots": {},
                        "last_scan_at": 123.0,
                        "last_error": "RuntimeError: scan failed",
                    }
                ),
            )

        self.assertFalse(health["ready"])
        self.assertIn("job_supervisor_failed", health["reasons"])
        self.assertNotIn("job_supervisor_not_running", health["reasons"])

    def test_job_api_requires_and_accepts_the_current_claim_fence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db = self._workspace_db(Path(tmp))
            with patch("vfieval.job_api.wake_job_supervisor") as wake:
                created, status = create_job_request(
                    db,
                    {"kind": "decode", "payload": {"source": "test"}},
                )
            self.assertEqual(status, HTTPStatus.CREATED)
            wake.assert_called_once_with(db)

            claimed = claim_job_request(
                db,
                {
                    "worker_id": "worker-a",
                    "role": "decode",
                    "kinds": ["decode"],
                },
            )["job"]
            self.assertEqual(int(claimed["id"]), int(created["job_id"]))
            fence = {
                "worker_id": "worker-a",
                "attempt": int(claimed["claim_attempt"]),
                "lease_token": str(claimed["lease_token"]),
            }

            with self.assertRaises(JobApiError) as missing:
                job_callback_request(db, int(claimed["id"]), "progress", {})
            self.assertEqual(missing.exception.status, HTTPStatus.BAD_REQUEST)
            self.assertEqual(missing.exception.code, "JobFenceRequired")

            progress = job_callback_request(
                db,
                int(claimed["id"]),
                "progress",
                {**fence, "current": 2, "total": 3},
            )
            self.assertEqual(progress["status"], "progress")
            heartbeat = heartbeat_job_request(db, int(claimed["id"]), fence)
            self.assertEqual(heartbeat["status"], "heartbeat")
            completed = job_callback_request(
                db,
                int(claimed["id"]),
                "complete",
                {**fence, "result": {"ok": True}},
            )
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(db.get_job(int(claimed["id"]))["status"], "completed")

    def test_job_routes_keep_http_shapes_and_fencing_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = self._workspace_db(Path(tmp))
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                _make_handler(db, workspace),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"

            def get(path: str) -> tuple[int, dict]:
                with urllib.request.urlopen(f"{base_url}{path}", timeout=10) as response:
                    return int(response.status), json.loads(response.read().decode("utf-8"))

            def post(path: str, body: dict) -> tuple[int, dict]:
                request = urllib.request.Request(
                    f"{base_url}{path}",
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=10) as response:
                        return int(response.status), json.loads(response.read().decode("utf-8"))
                except urllib.error.HTTPError as exc:
                    return int(exc.code), json.loads(exc.read().decode("utf-8"))

            try:
                health_status, health = get("/api/health")
                self.assertEqual(health_status, 200)
                self.assertEqual(
                    set(health),
                    {
                        "ok",
                        "live",
                        "ready",
                        "reasons",
                        "metrics",
                        "release",
                        "schema",
                        "storage",
                        "leases",
                        "queues",
                        "maintenance",
                    },
                )

                register_status, registered = post(
                    "/api/workers/register",
                    {"worker_id": "http-worker", "role": "decode"},
                )
                self.assertEqual(register_status, 200)
                self.assertEqual(registered["worker_id"], "http-worker")

                create_status, created = post(
                    "/api/jobs",
                    {"kind": "decode", "payload": {"source": "http-test"}},
                )
                self.assertEqual(create_status, 201)
                self.assertEqual(created["kind"], "decode")
                job_id = int(created["job_id"])

                claim_status, claim = post(
                    "/api/jobs/claim",
                    {
                        "worker_id": "http-worker",
                        "role": "decode",
                        "kinds": ["decode"],
                    },
                )
                self.assertEqual(claim_status, 200)
                claimed = claim["job"]
                self.assertEqual(int(claimed["id"]), job_id)
                fence = {
                    "worker_id": "http-worker",
                    "attempt": int(claimed["claim_attempt"]),
                    "lease_token": str(claimed["lease_token"]),
                }

                missing_status, missing = post(
                    f"/api/jobs/{job_id}/heartbeat",
                    {},
                )
                self.assertEqual(missing_status, 400)
                self.assertEqual(missing["error"]["type"], "JobFenceRequired")

                progress_status, progress = post(
                    f"/api/jobs/{job_id}/progress",
                    {**fence, "current": 1, "total": 1},
                )
                self.assertEqual(progress_status, 200)
                self.assertEqual(progress["status"], "progress")

                complete_status, completed = post(
                    f"/api/jobs/{job_id}/complete",
                    {**fence, "result": {"ok": True}},
                )
                self.assertEqual(complete_status, 200)
                self.assertEqual(completed["status"], "complete")

                stale_status, stale = post(
                    f"/api/jobs/{job_id}/heartbeat",
                    fence,
                )
                self.assertEqual(stale_status, 409)
                self.assertEqual(stale["error"]["type"], "JobLeaseLost")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
