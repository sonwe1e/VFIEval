from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "vfieval" / "web"


class EvaluationStudioUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index_html = (WEB / "index.html").read_text(encoding="utf-8")
        cls.studio_js = (WEB / "studio.js").read_text(encoding="utf-8")
        cls.blind_html = (WEB / "blind.html").read_text(encoding="utf-8")
        cls.blind_js = (WEB / "blind.js").read_text(encoding="utf-8")
        cls.server_py = (ROOT / "src" / "vfieval" / "server.py").read_text(encoding="utf-8")

    def test_literal_dom_dependencies_exist(self) -> None:
        index_ids = set(re.findall(r'id="([^"]+)"', self.index_html))
        studio_ids = set(re.findall(r'el\("([^"]+)"\)', self.studio_js))
        self.assertEqual(studio_ids - index_ids, set())

        blind_ids = set(re.findall(r'id="([^"]+)"', self.blind_html))
        blind_script_ids = set(re.findall(r'byId\("([^"]+)"\)', self.blind_js))
        self.assertEqual(blind_script_ids - blind_ids, set())

    def test_blind_page_is_independent_and_uses_only_opaque_task_urls(self) -> None:
        self.assertIn('<script src="/blind.js"></script>', self.blind_html)
        self.assertNotIn('/app.js', self.blind_html)
        self.assertNotIn('class="nav-item', self.blind_html)
        self.assertIn('/api/blind/', self.blind_js)
        self.assertIn('task.token', self.blind_js)
        for forbidden in ("run_id", "checkpoint", "asset_id"):
            self.assertNotIn(forbidden, self.blind_js)

    def test_studio_keys_campaigns_by_schema_and_posts_item_first_methods(self) -> None:
        self.assertIn("selectedCampaignKey", self.studio_js)
        self.assertIn("function campaignKey(campaign)", self.studio_js)
        self.assertIn("campaignKey(row) === requestedKey", self.studio_js)
        self.assertIn("media_item_ids: ids", self.studio_js)
        self.assertIn("method_a: methodA", self.studio_js)
        self.assertIn("method_b: methodB", self.studio_js)
        self.assertIn("spatial_policy: spatialPolicy()", self.studio_js)

    def test_v2_objective_metrics_are_rendered_separately_from_human_rankings(self) -> None:
        self.assertIn("function objectiveMetrics(analysis, legacy)", self.studio_js)
        self.assertIn("analysis?.objective?.metrics", self.studio_js)
        self.assertIn('data-analysis-section="human"', self.studio_js)
        self.assertIn('data-analysis-section="objective"', self.studio_js)
        self.assertIn("row.method_label", self.studio_js)
        self.assertIn("row.direction", self.studio_js)
        self.assertIn("row.status_counts", self.studio_js)
        self.assertIn("不生成合成总分", self.studio_js)
        self.assertNotIn("combined_score", self.studio_js)

    def test_first_visit_waits_for_public_intro_before_session_creation(self) -> None:
        self.assertIn("if (!blindState.evaluatorName)", self.blind_js)
        self.assertIn("/session`,", self.blind_js)
        self.assertIn("display_name: displayName", self.blind_js)

    def test_blind_playback_and_lease_state_are_reset_between_tasks(self) -> None:
        self.assertIn('video.playbackRate = Number(byId("master-rate")', self.blind_js)
        self.assertIn('video.loop = Boolean(byId("master-loop")', self.blind_js)
        self.assertIn("function syncFromClockVideo(event)", self.blind_js)
        self.assertIn('byId("media-grid").replaceChildren()', self.blind_js)
        self.assertIn("clearInterval(blindState.leaseTimer)", self.blind_js)

    def test_v1_and_v2_actions_and_server_routes_stay_separate(self) -> None:
        self.assertIn("data-studio-legacy-archive", self.studio_js)
        self.assertIn("/api/evaluation-campaigns/${Number(campaignId)}/archive", self.studio_js)
        self.assertIn("/api/evaluation-campaigns/v2/${Number(campaignId)}/${action}", self.studio_js)
        self.assertIn('re.fullmatch(r"/evaluate/[A-Za-z0-9_-]+", path)', self.server_py)
        self.assertIn('path == "/api/media/run-outputs"', self.server_py)
        self.assertIn('r"/api/evaluation-campaigns/v2/', self.server_py)
        self.assertIn('r"/api/blind/', self.server_py)

    def test_studio_renders_and_copies_an_absolute_participant_link(self) -> None:
        self.assertIn("function participantShareUrl(shareUrl, campaign)", self.studio_js)
        self.assertIn("new URL(rawUrl, location.origin).href", self.studio_js)
        self.assertIn("const shareUrl = participantShareUrl(payload.share_url || campaign.share_url, campaign);", self.studio_js)
        self.assertIn('value="${safe(shareUrl)}"', self.studio_js)
        self.assertIn('data-copy-share="${safe(shareUrl)}"', self.studio_js)
        self.assertIn("navigator.clipboard.writeText(copy.dataset.copyShare)", self.studio_js)
        self.assertIn("function isLoopbackOrigin()", self.studio_js)
        self.assertIn('host === "0.0.0.0"', self.studio_js)
        self.assertIn("shareUrl && isLoopbackOrigin()", self.studio_js)
        self.assertIn("studio-share-warning", self.studio_js)
        self.assertIn("--host 0.0.0.0", self.studio_js)

    def test_external_methods_are_advanced_and_require_explicit_item_bindings(self) -> None:
        for hook in (
            'id="studio-item-group"',
            'id="studio-items"',
            'id="studio-method-a-source"',
            'id="studio-method-b-source"',
        ):
            self.assertIn(hook, self.index_html)
        self.assertIn('/api/media/item-groups?role=gt', self.studio_js)
        self.assertIn('/api/media/items?group_id=', self.studio_js)
        self.assertIn('/api/media/methods?', self.studio_js)
        self.assertIn('return { kind: "external", method_key:', self.studio_js)
        self.assertIn("External Pred 必须先显式绑定", self.index_html)
        self.assertNotIn("reference_asset_id", self.studio_js)
        self.assertNotIn("video_name: videoName", self.studio_js)
        self.assertNotIn("storage_path", self.studio_js)

    def test_coverage_matrix_surfaces_spatial_alignment_without_weakening_time(self) -> None:
        self.assertIn("时间映射严格验证", self.studio_js)
        self.assertIn("alignment_plan", self.studio_js)
        self.assertIn("target_width", self.studio_js)
        self.assertIn("resize_kind", self.studio_js)
        self.assertIn("smallest_pred", self.studio_js)
        self.assertIn("lanczos", self.studio_js)

    def test_compare_prefill_and_external_policy_follow_the_item_contract(self) -> None:
        app_js = (WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn("prefillFromCompare({", app_js)
        self.assertIn("async function prefillFromCompare(selection)", self.studio_js)
        self.assertIn("selection.predictions", self.studio_js)
        self.assertIn('option value="external"', self.index_html)
        self.assertIn('value === "external"', self.studio_js)
        self.assertIn("studio-allow-external-aspect-stretch", self.index_html)
        self.assertIn("allow_external_aspect_stretch", self.studio_js)
        self.assertIn("row.direction || row.resize_kind || row.operation", self.studio_js)


if __name__ == "__main__":
    unittest.main()
