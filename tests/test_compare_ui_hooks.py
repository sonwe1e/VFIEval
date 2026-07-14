from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CompareUiHookTests(unittest.TestCase):
    def test_compare_layers_and_master_video_controls_are_wired(self) -> None:
        app_js = (ROOT / "src" / "vfieval" / "web" / "app.js").read_text(encoding="utf-8")
        styles = (ROOT / "src" / "vfieval" / "web" / "styles.css").read_text(encoding="utf-8")

        self.assertIn("compare_layers", app_js)
        self.assertIn("data-compare-grid-columns", app_js)
        self.assertIn("data-master-video-play", app_js)
        self.assertIn("data-layer-frame", app_js)
        self.assertIn("highlightTimelineFrame", app_js)
        self.assertIn("syncActiveVideos", app_js)
        self.assertIn("compare-layer-grid", styles)
        self.assertIn("timeline-hover", styles)
        self.assertIn("--compare-grid-columns", styles)
        self.assertIn("grid-auto-flow: column", styles)

    def test_metric_health_video_tiles_and_chart_are_compact_but_readable(self) -> None:
        app_js = (ROOT / "src" / "vfieval" / "web" / "app.js").read_text(encoding="utf-8")
        styles = (ROOT / "src" / "vfieval" / "web" / "styles.css").read_text(encoding="utf-8")

        self.assertIn("<details class=\"metric-health-details\">", app_js)
        self.assertIn("renderMetricHealthSummary", app_js)
        self.assertIn("video-artifact-strip", app_js)
        self.assertIn("class=\"metric-chart-svg\"", app_js)
        self.assertIn("class=\"chart-grid\"", app_js)
        self.assertIn("class=\"current-marker\"", app_js)
        self.assertIn(".video-artifact-strip", styles)
        self.assertIn("aspect-ratio: 16 / 9;", styles)
        self.assertIn("min-height: 96px;", styles)
        self.assertNotIn("height: 32px;", styles)
        self.assertIn(".metric-line", styles)
        self.assertIn(".chart-scale", styles)

    def test_usability_controls_are_wired_without_legacy_compare_path_copy(self) -> None:
        app_js = (ROOT / "src" / "vfieval" / "web" / "app.js").read_text(encoding="utf-8")

        self.assertIn('api("/api/video-groups?summary=1")', app_js)
        refresh_catalog_body = app_js.split("async function refreshCatalog()", 1)[1].split("document.querySelectorAll", 1)[0]
        self.assertNotIn("loadVideoGroupPage", refresh_catalog_body)
        self.assertNotIn("loadCompareSources", refresh_catalog_body)
        self.assertIn("data-load-video-page", app_js)
        self.assertIn("data-video-page", app_js)
        self.assertIn("data-video-query", app_js)
        self.assertIn("data-video-sort", app_js)
        self.assertIn("preflightAbortController", app_js)
        self.assertIn("preflightPayloadKey", app_js)
        self.assertIn('api("/api/media/item-groups?role=gt")', app_js)
        self.assertIn("/api/media/items?group_id=", app_js)
        self.assertIn("/predictions", app_js)
        self.assertIn("pred_member_ids", app_js)
        self.assertIn('mode: "smallest_pred"', app_js)
        self.assertIn('filter: "lanczos"', app_js)
        self.assertIn("state.selectedComparePredMembers.size >= 2", app_js)
        self.assertIn("/compare-inputs`,", app_js)
        self.assertIn("/compare-inputs/${encodeURIComponent(slot)}/media", app_js)
        self.assertIn("variant=aligned", app_js)
        self.assertIn("renderAlignmentPlan", app_js)
        self.assertNotIn("填写 GT / Pred 路径", app_js)
        self.assertNotIn("reference_path ||", app_js)
        self.assertNotIn("distorted_path ||", app_js)

    def test_gt_first_picker_never_loads_global_pred_catalog(self) -> None:
        app_js = (ROOT / "src" / "vfieval" / "web" / "app.js").read_text(encoding="utf-8")
        loader = app_js.split("async function loadCompareSources", 1)[1].split(
            "function renderGroupVideoTable", 1
        )[0]
        self.assertIn("/api/media/item-groups?role=gt", loader)
        self.assertIn("/api/media/items?group_id=", loader)
        self.assertIn("/predictions", loader)
        self.assertNotIn("/api/media/assets?role=pred", loader)
        self.assertNotIn("/api/compare-sources/pred", loader)

    def test_alignment_report_uses_the_real_plan_and_external_confirmation(self) -> None:
        web = ROOT / "src" / "vfieval" / "web"
        app_js = (web / "app.js").read_text(encoding="utf-8")
        index_html = (web / "index.html").read_text(encoding="utf-8")
        styles = (web / "styles.css").read_text(encoding="utf-8")

        self.assertIn("function alignmentTemporal(plan)", app_js)
        self.assertIn("function alignmentDirection(row)", app_js)
        self.assertIn("temporal.frame_count", app_js)
        self.assertIn("temporal.timestamps_verified", app_js)
        self.assertIn("row?.direction || row?.resize_kind", app_js)
        self.assertIn("allow_external_aspect_stretch", app_js)
        self.assertIn('name="allow_external_aspect_stretch"', index_html)
        self.assertIn(".alignment-temporal-details", styles)
        self.assertIn(".compare-aspect-confirm", styles)

    def test_item_picker_exposes_member_temporal_mapping_and_slots(self) -> None:
        app_js = (ROOT / "src" / "vfieval" / "web" / "app.js").read_text(encoding="utf-8")

        self.assertIn("function compareTemporalSummary(row)", app_js)
        self.assertIn("mapping.source_frame_indices", app_js)
        self.assertIn("function compareSlotLabel(index)", app_js)
        self.assertIn('compat-badge compat-ok', app_js)
        self.assertIn("selectedMemberIds.indexOf(memberId)", app_js)

    def test_compare_input_tiles_can_switch_between_original_and_aligned_media(self) -> None:
        web = ROOT / "src" / "vfieval" / "web"
        app_js = (web / "app.js").read_text(encoding="utf-8")
        styles = (web / "styles.css").read_text(encoding="utf-8")

        self.assertIn("function compareInputSlotLabel(slot)", app_js)
        self.assertIn("data-compare-input-variant=\"aligned\"", app_js)
        self.assertIn("data-compare-input-variant=\"original\"", app_js)
        self.assertIn("data-compare-input-media", app_js)
        self.assertIn("media.load()", app_js)
        self.assertIn("data-compare-input-slot", app_js)
        self.assertIn(".compare-input-variants", styles)

    def test_external_pred_binding_is_an_explicit_item_scoped_advanced_action(self) -> None:
        web = ROOT / "src" / "vfieval" / "web"
        app_js = (web / "app.js").read_text(encoding="utf-8")
        index_html = (web / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="external-pred-binding-form"', index_html)
        self.assertIn('id="external-pred-item-group"', index_html)
        self.assertIn('id="external-pred-item"', index_html)
        self.assertIn('id="external-pred-asset"', index_html)
        self.assertIn('name="aspect_stretch_confirmed"', index_html)
        self.assertIn("function bindExternalPrediction(event)", app_js)
        self.assertIn("/api/media/items/${itemId}/external-predictions", app_js)
        self.assertIn("function optionalJsonObject(value, label)", app_js)
        self.assertIn("asset.source_kind === \"upload\"", app_js)
        self.assertNotIn('name="path"', index_html)

    def test_compare_submission_is_single_flight_and_immediately_visible(self) -> None:
        web = ROOT / "src" / "vfieval" / "web"
        app_js = (web / "app.js").read_text(encoding="utf-8")
        index_html = (web / "index.html").read_text(encoding="utf-8")

        self.assertIn('id="compare-submit-status"', index_html)
        self.assertIn("if (state.compareSubmitting)", app_js)
        self.assertIn('state.compareSubmitPhase = "preflight"', app_js)
        self.assertIn('state.compareSubmitPhase = "creating"', app_js)
        self.assertIn('state.compareSubmitPhase = "opening"', app_js)
        self.assertIn('form.setAttribute("aria-busy"', app_js)
        self.assertIn("请勿重复点击", app_js)

    def test_video_player_keeps_a_diagnostic_link_on_media_error(self) -> None:
        app_js = (ROOT / "src" / "vfieval" / "web" / "app.js").read_text(encoding="utf-8")

        self.assertIn("function handleVideoPlaybackError(video)", app_js)
        self.assertIn("打开原始视频", app_js)
        self.assertIn("视频产物可能损坏或编码不兼容", app_js)
        self.assertNotIn("浏览器无法播放此视频格式", app_js)


if __name__ == "__main__":
    unittest.main()
