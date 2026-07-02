from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CompareUiHookTests(unittest.TestCase):
    def test_compare_layers_and_master_video_controls_are_wired(self) -> None:
        app_js = (ROOT / "src" / "vfieval" / "web" / "app.js").read_text(encoding="utf-8")
        styles = (ROOT / "src" / "vfieval" / "web" / "styles.css").read_text(encoding="utf-8")

        self.assertIn("compare_layers", app_js)
        self.assertIn("extra_layers", app_js)
        self.assertIn("data-compare-layer-kind", app_js)
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
        self.assertIn("compareTrackLabels", app_js)
        self.assertIn("data-compare-track-label", app_js)
        self.assertIn("selectedCompareLayerKinds: new Set()", app_js)
        self.assertNotIn("填写 GT / Pred 路径", app_js)
        self.assertNotIn("reference_path ||", app_js)
        self.assertNotIn("distorted_path ||", app_js)


if __name__ == "__main__":
    unittest.main()
