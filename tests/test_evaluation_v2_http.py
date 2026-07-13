from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vfieval.media_assets import (
    create_collection,
    sync_folder_assets,
    sync_run_assets,
    upsert_asset,
)
from vfieval.evaluations import create_campaign

from v13_test_utils import (
    add_completed_pred_run,
    make_workspace,
    start_server,
    stop_server,
    write_mp4,
)


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
) -> tuple[int, dict[str, str], bytes]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status), dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), dict(exc.headers.items()), exc.read()


def _json_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
) -> tuple[int, dict]:
    status, _headers, body = _request(base_url, path, method=method, payload=payload)
    return status, json.loads(body.decode("utf-8"))


class EvaluationCampaignV2HttpTests(unittest.TestCase):
    def _campaign_fixture(self, workspace, db, *, prefix: str = "http"):
        gt_path = write_mp4(
            workspace.runs_dir / f"{prefix}-shared" / "gt.mp4",
            [(0, 0, 0), (20, 0, 0), (40, 0, 0)],
        )
        pred_a = write_mp4(
            workspace.runs_dir / f"{prefix}-method-a" / "pred.mp4",
            [(0, 2, 0), (20, 2, 0), (40, 2, 0)],
        )
        pred_b = write_mp4(
            workspace.runs_dir / f"{prefix}-method-b" / "pred.mp4",
            [(0, 0, 2), (20, 0, 2), (40, 0, 2)],
        )
        run_a = add_completed_pred_run(
            db,
            workspace,
            f"{prefix}-method-a",
            pred_a,
            video_name="clip",
            gt_video_path=gt_path,
        )
        run_b = add_completed_pred_run(
            db,
            workspace,
            f"{prefix}-method-b",
            pred_b,
            video_name="clip",
            gt_video_path=gt_path,
        )
        sync_run_assets(db, workspace, run_a)
        sync_run_assets(db, workspace, run_b)
        body = {
            "name": f"{prefix}-internal",
            "public_title": "Opaque interpolation study",
            "target_votes": 1,
            "seed": 73,
            "methods": [
                {"run_id": run_a, "label": "Secret Method Alpha"},
                {"run_id": run_b, "label": "Secret Method Beta"},
            ],
        }
        return run_a, run_b, body

    def _publish_http(self, base_url: str, body: dict) -> dict:
        status, preview = _json_request(
            base_url,
            "/api/evaluation-campaigns/v2/preview",
            method="POST",
            payload=body,
        )
        self.assertEqual(status, 200, preview)
        self.assertEqual(preview["schema_version"], 2)
        self.assertEqual(preview["ready_video_names"], ["clip"])

        status, created = _json_request(
            base_url,
            "/api/evaluation-campaigns/v2",
            method="POST",
            payload=body,
        )
        self.assertEqual(status, 201, created)
        campaign_id = int(created["campaign"]["id"])

        status, draft = _json_request(
            base_url, f"/api/evaluation-campaigns/v2/{campaign_id}"
        )
        self.assertEqual(status, 200, draft)
        self.assertEqual(draft["campaign"]["status"], "draft")
        self.assertEqual(draft["campaign"]["campaign_key"], f"v2:{campaign_id}")
        self.assertEqual(draft["coverage"], {"items": 1, "tasks": 0, "votes": 0})

        status, queued = _json_request(
            base_url,
            f"/api/evaluation-campaigns/v2/{campaign_id}/publish",
            method="POST",
            payload={},
        )
        self.assertEqual(status, 202, queued)
        self.assertIn(queued["preparation"]["state"], {"queued", "running", "completed"})

        deadline = time.time() + 30
        while time.time() < deadline:
            status, detail = _json_request(
                base_url, f"/api/evaluation-campaigns/v2/{campaign_id}"
            )
            self.assertEqual(status, 200, detail)
            if detail["campaign"]["status"] in {"published", "failed"}:
                break
            time.sleep(0.05)
        else:
            self.fail("Campaign V2 preparation did not reach a terminal state")
        self.assertEqual(detail["campaign"]["status"], "published", detail)
        self.assertEqual(detail["preparation"]["state"], "completed")
        self.assertEqual(detail["coverage"], {"items": 1, "tasks": 1, "votes": 0})
        self.assertIsNotNone(detail["analysis"])
        self.assertEqual(detail["share_url"], detail["campaign"]["share_url"])
        self.assertRegex(str(detail["share_url"]), r"^/evaluate/[A-Za-z0-9_-]+$")
        self.assertNotIn("://", str(detail["share_url"]))
        return detail

    def test_canonical_media_endpoints_exclude_invalid_run_outputs_and_internal_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            gt_path = write_mp4(
                workspace.runs_dir / "filter-shared" / "gt.mp4",
                [(0, 0, 0), (1, 1, 1), (2, 2, 2)],
            )
            runs: list[int] = []
            for name, color in (("live", 10), ("deleted", 20), ("cleaned", 30)):
                pred = write_mp4(
                    workspace.runs_dir / f"filter-{name}" / "pred.mp4",
                    [(color, 0, 0)] * 3,
                )
                run_id = add_completed_pred_run(
                    db,
                    workspace,
                    f"filter-{name}",
                    pred,
                    video_name="clip",
                    gt_video_path=gt_path,
                )
                sync_run_assets(db, workspace, run_id)
                runs.append(run_id)
            live_run, deleted_run, cleaned_run = runs
            # These model both a historical soft-deleted record and the new
            # cleanup-only state. Starting the HTTP handler must not revive either.
            db.soft_delete_run(deleted_run)
            db.mark_run_artifacts_cleaned(cleaned_run)

            folder_path = write_mp4(
                workspace.root.parent / "videos" / "canonical" / "folder.mp4",
                [(4, 4, 4)] * 3,
            )
            sync_folder_assets(db, workspace)
            folder_row = db.get(
                "SELECT id FROM media_assets WHERE source_kind = 'folder' AND storage_path = ?",
                (str(folder_path.resolve()),),
            )
            self.assertIsNotNone(folder_row)

            collection = create_collection(db, "Canonical HTTP sources")
            source_kinds: dict[str, int] = {"folder": int(folder_row["id"])}
            managed_roots = {
                "upload": workspace.media_dir,
                "run_artifact": workspace.runs_dir,
                "evaluation_package": workspace.evaluations_dir,
            }
            for source_kind, managed_root in managed_roots.items():
                path = managed_root / "catalog-fixtures" / f"{source_kind}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source_kind.encode("ascii"))
                source_kinds[source_kind] = int(
                    upsert_asset(
                        db,
                        collection_id=int(collection["id"]),
                        source_key=f"http-source:{source_kind}",
                        source_kind=source_kind,
                        media_kind="video",
                        role="gt",
                        display_name=f"{source_kind}-gt",
                        original_name=path.name,
                        storage_path=path,
                        size_bytes=path.stat().st_size,
                    )["id"]
                )

            server, thread, base_url = start_server(db, workspace)
            try:
                status, outputs = _json_request(base_url, "/api/media/run-outputs")
                self.assertEqual(status, 200, outputs)
                # Upgrade-era Run outputs without an explicit canonical Item
                # binding are audit-only and must not re-enter new selectors.
                self.assertEqual(
                    {int(row["run_id"]) for row in outputs["runs"]},
                    set(),
                )
                self.assertNotIn(f'"run_id": {live_run}', json.dumps(outputs, sort_keys=True))
                serialized = json.dumps(outputs, sort_keys=True)
                self.assertNotIn(f'"run_id": {deleted_run}', serialized)
                self.assertNotIn(f'"run_id": {cleaned_run}', serialized)

                status, sources = _json_request(
                    base_url, "/api/media/sources?role=gt&sync=0&page_size=100"
                )
                self.assertEqual(status, 200, sources)
                returned = {int(asset["id"]): asset["source_kind"] for asset in sources["assets"]}
                self.assertEqual(
                    {source_kinds["folder"], source_kinds["upload"]},
                    set(returned),
                )
                self.assertEqual(set(returned.values()), {"folder", "upload"})
            finally:
                stop_server(server, thread)

    def test_campaign_v2_http_lifecycle_opaque_page_and_v1_route_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _run_a, _run_b, body = self._campaign_fixture(workspace, db, prefix="routes")
            legacy_campaign = create_campaign(
                db,
                {"name": "legacy-same-id", "target_votes": 1},
            )
            self.assertEqual(int(legacy_campaign["id"]), 1)
            server, thread, base_url = start_server(db, workspace)
            try:
                status, legacy_write = _json_request(
                    base_url,
                    "/api/evaluation-campaigns",
                    method="POST",
                    payload={"name": "legacy-same-id", "target_votes": 1},
                )
                self.assertEqual(status, 410, legacy_write)
                self.assertIn("read-only", legacy_write["error"]["message"])

                # V1 remains inspectable/exportable, but every workflow that
                # could create a participant session, candidate, task, or vote
                # is deliberately retired.
                for path, payload in (
                    ("/api/evaluators/session", {"evaluator_id": "legacy-browser"}),
                    ("/api/evaluation-campaigns/1/candidates", {}),
                    ("/api/evaluation-campaigns/1/publish", {}),
                    ("/api/evaluation-campaigns/1/close", {}),
                    ("/api/evaluation-tasks/adhoc", {}),
                    ("/api/evaluation-tasks/1/votes", {}),
                ):
                    legacy_status, legacy_response = _json_request(
                        base_url, path, method="POST", payload=payload
                    )
                    self.assertEqual(legacy_status, 410, (path, legacy_response))
                    self.assertIn("read-only", legacy_response["error"]["message"])

                detail = self._publish_http(base_url, body)
                campaign_id = int(detail["campaign"]["id"])
                self.assertEqual(campaign_id, 1)

                status, v1_detail = _json_request(base_url, "/api/evaluation-campaigns/1")
                self.assertEqual(status, 200, v1_detail)
                self.assertEqual(v1_detail["name"], "legacy-same-id")
                self.assertEqual(v1_detail["schema_version"], 1)
                self.assertEqual(v1_detail["campaign_key"], "v1:1")

                status, v2_detail = _json_request(base_url, "/api/evaluation-campaigns/v2/1")
                self.assertEqual(status, 200, v2_detail)
                self.assertEqual(v2_detail["campaign"]["name"], "routes-internal")
                self.assertEqual(v2_detail["campaign"]["schema_version"], 2)

                status, combined = _json_request(base_url, "/api/evaluation-campaigns")
                self.assertEqual(status, 200, combined)
                self.assertEqual(
                    {row["campaign_key"] for row in combined["campaigns"]},
                    {"v1:1", "v2:1"},
                )

                status, legacy_next = _json_request(
                    base_url,
                    "/api/evaluation-campaigns/1/next?evaluator_id=legacy-browser",
                )
                self.assertEqual(status, 410, legacy_next)
                self.assertIn("opaque Campaign V2", legacy_next["error"]["message"])

                status, legacy_media = _json_request(
                    base_url, "/api/evaluation-tasks/1/media/left"
                )
                self.assertEqual(status, 410, legacy_media)
                self.assertIn("opaque Campaign V2", legacy_media["error"]["message"])

                token = detail["campaign"]["public_token"]
                status, headers, page = _request(base_url, f"/evaluate/{token}")
                self.assertEqual(status, 200)
                self.assertIn("text/html", headers.get("Content-Type", ""))
                html = page.decode("utf-8")
                self.assertIn('/blind.js', html)
                self.assertNotIn('/app.js', html)
                self.assertNotIn('media-view', html)
                self.assertNotIn('evaluation-studio', html)

                script_status, script_headers, script = _request(base_url, "/blind.js")
                self.assertEqual(script_status, 200)
                self.assertIn("javascript", script_headers.get("Content-Type", ""))
                self.assertEqual(script_headers.get("Cache-Control"), "no-store")
                self.assertIn(b"initializeBlindPage", script)
            finally:
                stop_server(server, thread)

    def test_blind_http_payload_and_media_require_opaque_token_and_active_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            run_a, run_b, body = self._campaign_fixture(workspace, db, prefix="blind-http")
            server, thread, base_url = start_server(db, workspace)
            try:
                detail = self._publish_http(base_url, body)
                token = str(detail["campaign"]["public_token"])

                status, public = _json_request(base_url, f"/api/blind/{token}")
                self.assertEqual(status, 200, public)
                self.assertIsNone(public["task"])
                self.assertNotIn("results", public)

                evaluator_id = "browser-http-uuid"
                status, session = _json_request(
                    base_url,
                    f"/api/blind/{token}/session",
                    method="POST",
                    payload={"evaluator_id": evaluator_id, "display_name": "HTTP Alice"},
                )
                self.assertEqual(status, 201, session)
                task = session["task"]
                self.assertIsNotNone(task)

                # Participant JSON may contain only the opaque campaign/task URL
                # tokens and display-oriented data, never organizer identities.
                serialized = json.dumps(session, ensure_ascii=False, sort_keys=True)
                for forbidden in (
                    '"run_id"',
                    '"asset_id"',
                    '"task_id"',
                    '"assignment_id"',
                    "model_name",
                    "checkpoint",
                    "Secret Method Alpha",
                    "Secret Method Beta",
                    "browser-http-uuid",
                    str(workspace.root),
                ):
                    self.assertNotIn(forbidden, serialized)

                media_path = urllib.parse.urlsplit(task["left_url"]).path
                status, _headers, error_body = _request(base_url, media_path)
                self.assertEqual(status, 400, error_body)
                status, _headers, error_body = _request(
                    base_url,
                    f"{media_path}?assignment=not-the-issued-opaque-assignment",
                )
                self.assertEqual(status, 404, error_body)

                status, media_headers, media_body = _request(base_url, task["left_url"])
                self.assertEqual(status, 200)
                self.assertIn("video/", media_headers.get("Content-Type", ""))
                self.assertTrue(media_body)
                self.assertEqual(media_headers.get("Cache-Control"), "no-store")

                task_token = str(task["token"])
                status, heartbeat = _json_request(
                    base_url,
                    f"/api/blind/{token}/tasks/{task_token}/heartbeat",
                    method="POST",
                    payload={"evaluator_id": evaluator_id},
                )
                self.assertEqual(status, 200, heartbeat)
                self.assertTrue(heartbeat["ok"])

                with db.connection() as conn:
                    conn.execute(
                        "UPDATE evaluation_assignments_v2 SET lease_expires_at = ? WHERE evaluator_id = ?",
                        (time.time() - 1, evaluator_id),
                    )
                status, _headers, expired_body = _request(base_url, task["left_url"])
                self.assertEqual(status, 409, expired_body)
                status, expired_heartbeat = _json_request(
                    base_url,
                    f"/api/blind/{token}/tasks/{task_token}/heartbeat",
                    method="POST",
                    payload={"evaluator_id": evaluator_id},
                )
                self.assertEqual(status, 409, expired_heartbeat)
            finally:
                stop_server(server, thread)


if __name__ == "__main__":
    unittest.main()
