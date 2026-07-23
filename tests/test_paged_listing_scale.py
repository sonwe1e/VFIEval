from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vfieval.config import WorkspaceConfig
from vfieval.db import Database
from vfieval.evaluations_v2 import (
    ensure_v2_schema,
    list_campaign_summaries_page,
)

from v13_test_utils import get_json, make_workspace, start_server, stop_server


def _insert_campaigns(db: Database, count: int = 500) -> None:
    ensure_v2_schema(db)
    half = count // 2
    v2_statuses = ("draft", "preparing", "published", "failed", "closed", "archived")
    legacy_statuses = ("draft", "published", "closed")
    with db.connection() as conn:
        conn.executemany(
            """
            INSERT INTO evaluation_campaigns_v2(
                public_token, name, public_title, status, target_votes, seed,
                config_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, 3, ?, '{}', ?, ?)
            """,
            [
                (
                    f"token-{index:04d}",
                    f"v2-campaign-{index:04d}",
                    f"V2 public {index:04d}",
                    v2_statuses[index % len(v2_statuses)],
                    index,
                    float(index * 2 + 1),
                    float(index * 2 + 1),
                )
                for index in range(half)
            ],
        )
        conn.executemany(
            """
            INSERT INTO evaluation_campaigns(
                name, campaign_type, status, target_votes, seed,
                metadata_json, created_at, updated_at
            )
            VALUES (?, 'campaign', ?, 3, ?, ?, ?, ?)
            """,
            [
                (
                    f"legacy-campaign-{index:04d}",
                    legacy_statuses[index % len(legacy_statuses)],
                    index,
                    json.dumps(
                        {
                            "public_title": f"Legacy public {index:04d}",
                            **(
                                {"archived_at": float(index + 1)}
                                if index % 10 == 0
                                else {}
                            ),
                        }
                    ),
                    float(index * 2),
                    float(index * 2),
                )
                for index in range(count - half)
            ],
        )


class CampaignListPaginationTests(unittest.TestCase):
    def test_500_campaigns_use_two_lightweight_queries_and_page_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _insert_campaigns(db)
            started = time.monotonic()
            with (
                patch(
                    "vfieval.evaluations_v2.get_campaign_v2",
                    side_effect=AssertionError("list endpoint loaded Campaign detail"),
                ),
                patch.object(db, "query", wraps=db.query) as query_spy,
            ):
                payload = list_campaign_summaries_page(
                    db,
                    page=7,
                    page_size=30,
                )
            elapsed = time.monotonic() - started

            self.assertEqual(payload["total"], 500)
            self.assertEqual(payload["page"], 7)
            self.assertEqual(payload["page_count"], 17)
            self.assertEqual(len(payload["campaigns"]), 30)
            self.assertEqual(query_spy.call_count, 2)
            self.assertLess(elapsed, 10.0)
            for campaign in payload["campaigns"]:
                self.assertNotIn("items", campaign)
                self.assertNotIn("methods", campaign)
                self.assertNotIn("analysis", campaign)
                self.assertNotIn("bindings", campaign)

    def test_500_campaigns_multi_status_filter_is_parameterized_and_constant_query_count(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db = make_workspace(tmp)
            _insert_campaigns(db)
            with patch.object(db, "query", wraps=db.query) as query_spy:
                payload = list_campaign_summaries_page(
                    db,
                    page=2,
                    page_size=30,
                    status="published,closed,archived",
                )

            self.assertEqual(query_spy.call_count, 2)
            self.assertEqual(len(payload["campaigns"]), 30)
            self.assertEqual(
                payload["filters"]["status"],
                "published,closed,archived",
            )
            self.assertTrue(
                all(
                    row["status"] in {"published", "closed", "archived"}
                    for row in payload["campaigns"]
                )
            )
            for call in query_spy.call_args_list:
                sql = str(call.args[0])
                params = tuple(call.args[1])
                self.assertIn(" IN (?, ?, ?)", sql)
                self.assertNotIn("'published'", sql)
                self.assertTrue(
                    {"published", "closed", "archived"}.issubset(set(params))
                )

    def test_campaign_status_filter_rejects_unknown_or_empty_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _workspace, db = make_workspace(tmp)
            _insert_campaigns(db, count=6)
            for status in ("published,unknown", "published,,closed"):
                with self.subTest(status=status), self.assertRaises(ValueError):
                    list_campaign_summaries_page(db, status=status)

    def test_http_campaign_list_keeps_campaigns_and_returns_page_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, db = make_workspace(tmp)
            _insert_campaigns(db)
            server, thread, base_url = start_server(db, workspace)
            try:
                payload = get_json(
                    base_url,
                    "/api/evaluation-campaigns"
                    "?page=2&page_size=10&q=v2-campaign&status=published",
                )
            finally:
                stop_server(server, thread)

            self.assertEqual(payload["page"], 2)
            self.assertEqual(payload["page_size"], 10)
            self.assertGreater(payload["total"], 10)
            self.assertEqual(
                payload["page_count"],
                (payload["total"] + payload["page_size"] - 1)
                // payload["page_size"],
            )
            self.assertEqual(len(payload["campaigns"]), 10)
            self.assertTrue(
                all(row["status"] == "published" for row in payload["campaigns"])
            )
            self.assertTrue(
                all(row["campaign_key"].startswith("v2:") for row in payload["campaigns"])
            )


class RunListPaginationScaleTests(unittest.TestCase):
    def test_500_runs_batch_load_current_page_purge_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceConfig.from_root(Path(tmp) / ".vfieval")
            workspace.ensure()
            db = Database(workspace.db_path)
            db.init()
            model_id = db.register_model("model", "dummy", None, 8, 8, {})
            dataset_id = db.create_dataset("dataset", tmp, True)
            now = time.time()
            with db.connection() as conn:
                conn.executemany(
                    """
                    INSERT INTO runs(
                        name, model_id, dataset_id, height, width, batch_size,
                        device, precision, metrics_json, status, metadata_json,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, 8, 8, 1, 'cpu', 'fp32', '[]',
                            'completed', '{}', ?, ?)
                    """,
                    [
                        (
                            f"run-{index:04d}",
                            model_id,
                            dataset_id,
                            now + index,
                            now + index,
                        )
                        for index in range(500)
                    ],
                )
                recent_ids = [
                    int(row["id"])
                    for row in conn.execute(
                        "SELECT id FROM runs ORDER BY id DESC LIMIT 4"
                    ).fetchall()
                ]
                conn.executemany(
                    """
                    INSERT INTO run_purge_requests(
                        run_id, request_type, status, report_json, error_json,
                        requested_at, updated_at
                    )
                    VALUES (?, 'cleanup_artifacts', 'requested', '{}', '{}', ?, ?)
                    """,
                    [(run_id, now, now) for run_id in recent_ids],
                )

            started = time.monotonic()
            with (
                patch.object(db, "get", wraps=db.get) as get_spy,
                patch.object(db, "query", wraps=db.query) as query_spy,
            ):
                payload = db.list_runs_page(page=1, page_size=50)
            elapsed = time.monotonic() - started

            self.assertEqual(payload["total"], 500)
            self.assertEqual(payload["page_count"], 10)
            self.assertEqual(len(payload["runs"]), 50)
            self.assertEqual(get_spy.call_count, 2)
            # Database.get delegates to query, so this is two scalar reads plus
            # the page and one batched purge-request read, independent of page size.
            self.assertEqual(query_spy.call_count, 4)
            self.assertLess(elapsed, 10.0)
            purge_by_id = {
                int(row["id"]): row["purge_request"] for row in payload["runs"]
            }
            self.assertTrue(
                all(purge_by_id[run_id]["status"] == "requested" for run_id in recent_ids)
            )


class CampaignListFrontendContractTests(unittest.TestCase):
    def test_studio_campaign_list_has_server_filters_and_pager(self) -> None:
        studio_js = (ROOT / "src" / "vfieval" / "web" / "studio.js").read_text(
            encoding="utf-8"
        )
        index_html = (ROOT / "src" / "vfieval" / "web" / "index.html").read_text(
            encoding="utf-8"
        )

        self.assertIn("campaignPageSize", studio_js)
        self.assertIn('params.set("q", studioState.campaignQuery)', studio_js)
        self.assertIn('params.set("status", studioState.campaignStatus)', studio_js)
        self.assertIn("payload.page_count", studio_js)
        self.assertIn("data-studio-campaign-page", studio_js)
        self.assertIn("options.refreshSelected", studio_js)
        self.assertIn('id="studio-campaign-query"', index_html)
        self.assertIn('id="studio-campaign-status"', index_html)
        self.assertIn('id="studio-campaign-pager"', index_html)

    def test_frozen_packages_keep_independent_filter_and_pagination_state(self) -> None:
        studio_js = (ROOT / "src" / "vfieval" / "web" / "studio.js").read_text(
            encoding="utf-8"
        )
        render_start = studio_js.index("function renderPackages()")
        render_end = studio_js.index("\n  function ", render_start + 1)
        render_packages = studio_js[render_start:render_end]
        load_start = studio_js.index("async function loadPackages(")
        load_end = studio_js.index("\n  async function ", load_start + 1)
        load_packages = studio_js[load_start:load_end]
        campaign_start = studio_js.index("async function loadCampaigns(")
        campaign_end = studio_js.index("\n  async function ", campaign_start + 1)
        load_campaigns = studio_js[campaign_start:campaign_end]

        self.assertIn("packageCampaigns: []", studio_js)
        self.assertIn("packagePage: 1", studio_js)
        self.assertIn("packagePageCount: 1", studio_js)
        self.assertIn("studioState.packageCampaigns", render_packages)
        self.assertNotIn("studioState.campaigns.filter", render_packages)
        self.assertIn('status: "published,closed,archived"', load_packages)
        self.assertIn("studioState.packageRequestGeneration", load_packages)
        self.assertIn("data-studio-package-page", studio_js)
        self.assertNotIn("renderPackages()", load_campaigns)


if __name__ == "__main__":
    unittest.main()
