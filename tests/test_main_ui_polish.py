from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "vfieval" / "web"


class MainUiPolishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_entry = (WEB / "app.js").read_text(encoding="utf-8")
        cls.compare = (WEB / "compare.js").read_text(encoding="utf-8")
        cls.run_detail = (WEB / "run-detail.js").read_text(encoding="utf-8")
        cls.media = (WEB / "media.js").read_text(encoding="utf-8")
        cls.app = "\n".join(
            (cls.app_entry, cls.compare, cls.run_detail, cls.media)
        )
        cls.shared = (WEB / "shared.js").read_text(encoding="utf-8")
        cls.html = (WEB / "index.html").read_text(encoding="utf-8")
        cls.styles = (WEB / "styles.css").read_text(encoding="utf-8")

    def section(self, start: str, end: str) -> str:
        return self.app.split(start, 1)[1].split(end, 1)[0]

    def test_selected_video_group_index_is_loaded_automatically(self) -> None:
        refresh = self.section(
            "async function refreshCatalogData",
            "async function refreshCatalog()",
        )
        selection = self.section(
            "function renderGroupVideoTable",
            "function renderVideoSelection",
        )

        self.assertIn("await loadSelectedVideoGroupIndexes()", refresh)
        self.assertIn('api("/api/video-selections"', self.app)
        self.assertIn("video_selection_token: state.videoSelectionToken", self.app)
        self.assertIn("正在加载轻量视频索引并默认全选", selection)
        self.assertIn('class="group-video-block selection-diagnostics"', selection)
        self.assertIn("逐视频诊断", self.app)
        self.assertNotIn("all_video_names", self.app)

    def test_polling_uses_incremental_live_updates_and_visible_backoff(self) -> None:
        refresh = self.section(
            "async function refreshRunsOnce",
            "function refreshRunsOnly",
        )
        poll = self.section("function startRunsPoll", "refreshCatalog()")

        self.assertIn("mergeSelectedRunSummary(nextRun)", refresh)
        self.assertIn("patchSelectedRunLiveDom()", refresh)
        self.assertIn("pollSelectedRunById", refresh)
        self.assertIn("consecutiveErrors", poll)
        self.assertIn("正在退避重试", poll)
        self.assertNotIn(".catch(() => {})", poll)
        self.assertIn('id="runs-poll-status"', self.html)

    def test_background_detail_refresh_preserves_playback_focus_and_drafts(self) -> None:
        capture = self.section(
            "function captureRunDetailUiState",
            "function restoreRunDetailUiState",
        )
        restore = self.section(
            "function restoreRunDetailUiState",
            "async function pollSelectedRunById",
        )
        select = self.section("async function selectRun", "function renderInferencePhase")

        self.assertIn('root.querySelectorAll("video")', capture)
        self.assertIn("[data-feedback-form]", capture)
        self.assertIn("video.currentTime = saved.currentTime", restore)
        self.assertIn("active?.focus({ preventScroll: true })", restore)
        self.assertIn("captureRunDetailUiState()", select)
        self.assertIn("restoreRunDetailUiState(preservedUi)", select)

    def test_media_library_filters_are_server_side_and_paged(self) -> None:
        path = self.section("function mediaAssetsPath", "function mediaRoleLabel")
        render = self.section("function renderMediaFilters", "function renderMediaLibrary")
        more = self.section(
            "async function loadMoreMediaSources",
            "async function createMediaCollection",
        )

        self.assertIn("/api/media/assets?", path)
        for key in ("q", "role", "source_kind", "collection_id"):
            self.assertIn(key, self.app)
        self.assertIn('data-media-filter="q"', render)
        self.assertIn('data-media-filter="source_kind"', render)
        self.assertIn("mediaAssetsPath(nextPage)", more)

    def test_media_binding_empty_states_use_workflow_terms_in_chinese(self) -> None:
        for text in (
            "暂无可用 GT Collection",
            "暂无可用 GT Item",
            "请先上传外部 Pred",
        ):
            self.assertIn(text, self.media)
        for stale_text in (
            "No canonical GT Collection",
            "No canonical GT Item",
            "Upload an External Pred first",
            "Upload a ready External Pred",
            "External Pred is now explicitly bound",
        ):
            self.assertNotIn(stale_text, self.media)
        self.assertIn("外部 GT", self.html)
        self.assertIn("外部 Pred", self.html)
        self.assertNotIn("external GT", self.html)
        self.assertNotIn("external Pred", self.html)
        self.assertNotIn("External Pred", self.html)

    def test_studio_uses_gt_pred_one_pred_two_workflow_terms(self) -> None:
        for text in ("GT 文件夹", "Pred 1 来源", "Pred 1", "Pred 2 来源", "Pred 2"):
            self.assertIn(text, self.html)
        for stale_text in ("方法 A 来源", "方法 B 来源"):
            self.assertNotIn(stale_text, self.html)

    def test_inference_draft_and_url_deep_links_are_versioned(self) -> None:
        self.assertIn('const INFERENCE_DRAFT_VERSION = 2', self.app)
        self.assertIn("function restoreInferenceDraft", self.app)
        self.assertIn("video_selection_token", self.app)
        self.assertNotIn("all_video_names", self.app)
        self.assertIn("function syncBrowserRoute", self.app)
        self.assertIn("window.history.pushState", self.app)
        self.assertIn('window.addEventListener("popstate"', self.app)
        for key in ("view", "run", "video", "frame"):
            self.assertIn(f'url.searchParams.set("{key}"', self.app)

    def test_file_hashing_runs_in_a_cancelable_web_worker(self) -> None:
        worker = self.section("function createSha256Worker", "async function sha256File")
        upload = self.section(
            "async function uploadExternalMedia",
            "function optionalJsonObject",
        )

        self.assertIn("new Worker(objectUrl)", worker)
        self.assertIn('self.postMessage({ type: "progress"', worker)
        self.assertIn("new AbortController()", upload)
        self.assertIn('task.phase = "uploading"', upload)
        self.assertIn('task.phase = "finalizing"', upload)
        self.assertIn("task.controller.abort()", self.app)

    def test_metric_charts_have_keyboard_readout_and_data_table(self) -> None:
        chart = self.section("function renderMetricChart", "function metricLineSegments")
        keydown = self.app.split('document.addEventListener("keydown"', 1)[1].split(
            'document.addEventListener("mouseover"', 1
        )[0]

        self.assertIn('role="slider"', chart)
        self.assertIn("aria-valuetext", chart)
        self.assertIn("data-chart-readout", chart)
        self.assertIn("renderMetricDataTable", chart)
        for key in ("ArrowLeft", "ArrowRight", "Home", "End"):
            self.assertIn(key, keydown)
        self.assertIn(".chart-plot:focus-visible", self.styles)

    def test_empty_feedback_does_not_render_a_zero_histogram(self) -> None:
        histogram = self.section(
            "function renderRatingHistogram",
            "function formatRatingKey",
        )
        stats = self.section("function renderStats", "function formatNumber")

        self.assertIn("if (!total)", histogram)
        self.assertIn("还没有主观反馈", stats)
        self.assertIn("hasFeedback", stats)

    def test_create_requests_carry_reusable_submission_ids(self) -> None:
        inference = self.section("async function startRun", "function statusBadge")
        compare = self.section(
            "async function startCompareRun",
            "async function createAdhocEvaluation",
        )

        self.assertIn('typeof root.crypto.randomUUID === "function"', self.shared)
        self.assertIn('Shared.createSubmissionId("run")', inference)
        self.assertIn('Shared.createSubmissionId("compare")', compare)
        self.assertIn("payload.submission_id = state.runSubmissionId", inference)
        self.assertIn("payload.submission_id = state.compareSubmissionId", compare)
        self.assertIn('state.runSubmissionId = ""', self.app)
        self.assertIn('state.compareSubmissionId = ""', self.app)


if __name__ == "__main__":
    unittest.main()
