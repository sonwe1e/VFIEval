from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RunResultFreshnessUiTests(unittest.TestCase):
    def setUp(self) -> None:
        web = ROOT / "src" / "vfieval" / "web"
        self.app_js = "\n".join(
            (
                (web / "app.js").read_text(encoding="utf-8"),
                (web / "run-detail.js").read_text(encoding="utf-8"),
            )
        )
        self.index_html = (web / "index.html").read_text(encoding="utf-8")
        self.styles = (web / "styles.css").read_text(encoding="utf-8")

    def test_revision_change_invalidates_only_result_payloads(self) -> None:
        self.assertIn("content_revision", self.app_js)
        self.assertIn("function invalidateRunResultCache(runId, options = {})", self.app_js)
        body = self.app_js.split("function invalidateRunResultCache", 1)[1].split(
            "function runContentRevisionChanged", 1
        )[0]
        self.assertIn("clearRunScopedEntries(state.runVideoTimelines, id)", body)
        self.assertIn("clearRunScopedEntries(state.sampleDetails, id)", body)
        self.assertNotIn("selectedVideoByRun", body)
        self.assertNotIn("selectedSampleByVideo", body)
        self.assertNotIn("timelineWindowStartByVideo", body)
        self.assertNotIn("selectedMetricByRun", body)

    def test_refreshes_are_single_flight_and_stale_payloads_are_rejected(self) -> None:
        self.assertIn("runsRefreshPromise", self.app_js)
        self.assertIn("runsRefreshQueued", self.app_js)
        self.assertIn("runSelectRequestGeneration", self.app_js)
        self.assertIn("timelineRequestGeneration", self.app_js)
        self.assertIn("currentRunResultGeneration(runId)", self.app_js)
        self.assertIn("selectionGeneration !== state.runSelectionGeneration", self.app_js)
        self.assertIn("abortSampleRequestsForRun", self.app_js)
        self.assertIn("abortTimelineRequest", self.app_js)

    def test_manual_refresh_and_pending_artifact_states_are_visible(self) -> None:
        self.assertIn('id="refresh" class="secondary" type="button">刷新当前结果</button>', self.index_html)
        self.assertIn("data-refresh-run-results", self.app_js)
        self.assertIn("refreshRunResults", self.app_js)
        self.assertIn("产物生成中，保存完成后会自动加载。", self.app_js)
        self.assertIn("这个 Run 完成刷新后仍没有可查看的产物。", self.app_js)
        self.assertIn("artifact-pending", self.styles)
        self.assertIn("aria-busy=\"true\"", self.app_js)

    def test_long_metric_timeline_uses_overview_and_bounded_detail_window(self) -> None:
        self.assertIn("const TIMELINE_WINDOW_SIZE = 160", self.app_js)
        self.assertIn("function renderMetricOverview", self.app_js)
        self.assertIn("overview-envelope", self.app_js)
        self.assertIn("overview-viewport", self.app_js)
        self.assertIn("function metricLineSegments", self.app_js)
        self.assertIn("function setGlobalSampleIndex", self.app_js)
        self.assertIn("data-overview-video", self.app_js)
        self.assertIn("chart-tooltip", self.styles)
        detail_body = self.app_js.split("function renderMetricChart", 1)[1].split(
            "function metricLineSegments", 1
        )[0]
        self.assertNotIn("renderMetricPoints", detail_body)

    def test_run_list_and_dual_lpips_timeline_keep_updates_scoped(self) -> None:
        self.assertIn("function renderRunProgress(run)", self.app_js)
        self.assertIn("设备 ·", self.app_js)
        self.assertIn("启动中…", self.app_js)
        self.assertIn('const LPIPS_PAIR = ["lpips_vit_patch", "lpips_convnext"]', self.app_js)
        self.assertIn("function renderDualLpipsCharts", self.app_js)
        self.assertIn("function _tryUpdateChartMarkers", self.app_js)
        self.assertIn("_tryUpdateChartMarkers(chart, video, selectedIndex, metricName)", self.app_js)
        self.assertIn("dual-chart-row", self.styles)

    def test_metric_batch_override_is_optional_advanced_setting(self) -> None:
        self.assertIn('name="metric_batch_size_per_device"', self.index_html)
        self.assertIn("metric_batch_size_per_device: data.metric_batch_size_per_device", self.app_js)

    def test_run_detail_can_retry_only_failed_or_unavailable_metrics(self) -> None:
        self.assertIn("function retriableMetricCount(runId)", self.app_js)
        self.assertIn("row?.failed", self.app_js)
        self.assertIn("row?.unavailable", self.app_js)
        self.assertIn("data-retry-run-metrics", self.app_js)
        self.assertIn("function retryRunMetrics(runId)", self.app_js)
        self.assertIn("/metrics/retry", self.app_js)


if __name__ == "__main__":
    unittest.main()
