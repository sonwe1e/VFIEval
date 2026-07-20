from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ExperimentExperienceUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_js = (ROOT / "src" / "vfieval" / "web" / "app.js").read_text(
            encoding="utf-8"
        )
        cls.styles = (ROOT / "src" / "vfieval" / "web" / "styles.css").read_text(
            encoding="utf-8"
        )
        cls.index_html = (ROOT / "src" / "vfieval" / "web" / "index.html").read_text(
            encoding="utf-8"
        )

    def test_high_risk_workload_is_visible_and_requires_matching_confirmation(self) -> None:
        render = self.app_js.split("function renderWorkloadEstimate", 1)[1].split(
            "function highRiskWorkloadConfirmation", 1
        )[0]
        confirm = self.app_js.split("function highRiskWorkloadConfirmation", 1)[1].split(
            "function renderPreflight", 1
        )[0]
        start = self.app_js.split("async function startRun", 1)[1].split(
            "function statusBadge", 1
        )[0]

        self.assertIn("input_tensor_bytes_lower_bound", render)
        self.assertIn("prefetch_host_bytes_lower_bound", render)
        self.assertIn("artifact_budget_bytes", render)
        self.assertIn('workload?.risk_level !== "high"', confirm)
        self.assertIn("window.confirm", confirm)
        self.assertIn("preflight_token", start)
        self.assertIn("risk_ack_fingerprint", start)
        self.assertLess(start.index("highRiskWorkloadConfirmation"), start.index('api("/api/runs"'))
        self.assertLess(start.index("risk_ack_fingerprint"), start.index('api("/api/runs"'))

    def test_automatic_preflight_is_quick_and_start_forces_deep(self) -> None:
        preflight = self.app_js.split("async function runPreflight", 1)[1].split(
            "function renderPreflightError", 1
        )[0]
        start = self.app_js.split("async function startRun", 1)[1].split(
            "function statusBadge", 1
        )[0]

        self.assertIn('options.level === "deep" ? "deep" : "quick"', preflight)
        self.assertIn("preflight_level: level", preflight)
        self.assertIn('if (level !== "deep") delete result.preflight_token', preflight)
        self.assertIn('runPreflight({ force: true, level: "deep" })', start)
        self.assertIn('state.preflightLevel !== "deep"', start)
        self.assertIn('state.preflightLevel === "deep" && state.preflight.preflight_token', start)

    def test_safe_defaults_and_profile_recommendation_require_explicit_application(self) -> None:
        self.assertIn('<option value="auto" selected>auto</option>', self.index_html)
        self.assertIn('<input name="batch_size" type="number" min="1" value="1">', self.index_html)
        self.assertIn(
            '<input name="batch_size_per_device" type="number" min="1" value="1">',
            self.index_html,
        )

        render = self.app_js.split("function renderExecutionProfileRecommendation", 1)[1].split(
            "function applyExecutionProfileRecommendation", 1
        )[0]
        apply = self.app_js.split("function applyExecutionProfileRecommendation", 1)[1].split(
            "function renderDecodePanel", 1
        )[0]
        self.assertIn("data-apply-execution-profile", render)
        self.assertIn("建议不会自动覆盖当前设置", render)
        self.assertIn("form.elements.batch_size.value", apply)
        self.assertIn("form.elements.batch_size_per_device.value", apply)
        self.assertIn("schedulePreflight(0)", apply)

    def test_preflight_exposes_execution_contract_and_resource_summary(self) -> None:
        summary = self.app_js.split("function renderPreflightExecutionSummary", 1)[1].split(
            "function highRiskWorkloadConfirmation", 1
        )[0]
        for label in (
            "执行设备",
            "设备卡数",
            "每设备 Batch",
            "解码后端",
            "评测契约",
            "单设备显存",
            "主机可用内存",
            "磁盘可用空间",
        ):
            self.assertIn(label, summary)
        self.assertIn("effective_devices", summary)
        self.assertIn("evaluation_contract", summary)
        self.assertIn("storage_capacity", summary)
        self.assertIn("remaining_after_request", summary)

    def test_deployment_health_banner_surfaces_relay_and_worker_failures(self) -> None:
        self.assertIn('id="deployment-health"', self.index_html)
        renderer = self.app_js.split("function renderDeploymentHealth", 1)[1].split(
            "async function refreshDeploymentHealth", 1
        )[0]
        refresh = self.app_js.split("async function refreshDeploymentHealth", 1)[1].split(
            "function escapeHtml", 1
        )[0]
        self.assertIn("maintenance?.job_recovery", renderer)
        self.assertIn("leases.stale", renderer)
        self.assertIn("storage.free_bytes", renderer)
        self.assertIn("Windows 中继", renderer)
        self.assertIn('api("/api/health")', refresh)
        self.assertIn(".deployment-health.bad", self.styles)

    def test_file_input_run_offers_clone_with_current_inputs(self) -> None:
        clone = self.app_js.split("async function cloneRunWithCurrentInputs", 1)[1].split(
            "async function retryRunMetrics", 1
        )[0]
        detail = self.app_js.split("function renderRunDetail", 1)[1].split(
            "async function loadSampleDetail", 1
        )[0]

        self.assertIn("data-clone-run", detail)
        self.assertIn("isFileInputRun(run)", detail)
        self.assertIn("`/api/runs/${runId}/clone`", clone)
        self.assertIn("error.payload", clone)
        self.assertIn('Number(error.status) !== 409', clone)
        self.assertIn('"WorkloadRiskConfirmationRequired"', clone)
        self.assertIn("highRiskWorkloadConfirmation(workload)", clone)
        self.assertIn("risk_ack_fingerprint", clone)
        self.assertIn("attempt < 3", clone)
        self.assertLess(
            clone.index("highRiskWorkloadConfirmation(workload)"),
            clone.index("risk_ack_fingerprint"),
        )

    def test_api_errors_preserve_structured_payload_for_recovery(self) -> None:
        api_source = self.app_js.split("async function api", 1)[1].split(
            "function escapeHtml", 1
        )[0]

        self.assertIn("error.status = response.status", api_source)
        self.assertIn("error.payload = data", api_source)

    def test_impossible_accelerator_default_falls_back_to_detected_hardware(self) -> None:
        resolver = self.app_js.split("function resolveInitialExecutionDefaults", 1)[1].split(
            "function renderGroupPicker", 1
        )[0]

        self.assertIn('mode === "multi_npu" && npu.length', resolver)
        self.assertIn('form.elements.execution_mode.value = "multi_cuda"', resolver)
        self.assertIn('form.elements.execution_mode.value = "single"', resolver)
        self.assertIn('return cuda[0]?.id || "cpu"', resolver)

    def test_run_history_uses_server_pagination_and_server_filters(self) -> None:
        request = self.app_js.split("function runListPath", 1)[1].split(
            "function applyRunListPayload", 1
        )[0]
        render = self.app_js.split("function renderRuns", 1)[1].split(
            "async function loadRunVideosPage", 1
        )[0]

        self.assertIn("page_size", request)
        self.assertIn("state.runFilters", request)
        self.assertIn("/api/runs?", request)
        self.assertIn('data-run-filter="q"', render)
        self.assertIn('data-run-filter="status"', render)
        self.assertIn('data-run-filter="run_type"', render)
        self.assertIn('data-run-filter="model"', render)
        self.assertIn("data-runs-page", render)
        self.assertIn("state.runsPage.total", render)
        toolbar_style = self.styles.split(".runs-toolbar {", 1)[1].split("}", 1)[0]
        self.assertIn("flex-wrap: wrap", toolbar_style)

    def test_compare_item_selection_survives_search_and_pagination(self) -> None:
        selected = self.app_js.split("function selectedCompareGt", 1)[1].split(
            "function selectedComparePredRows", 1
        )[0]
        ensure = self.app_js.split("function ensureCompareSelection", 1)[1].split(
            "function compareSourcePager", 1
        )[0]
        page_handler = self.app_js.split(
            'const comparePage = event.target.closest("[data-compare-page]")', 1
        )[1].split('const videoPage = event.target.closest("[data-video-page]")', 1)[0]

        self.assertIn("selectedCompareItemSnapshot", selected)
        self.assertNotIn("state.selectedCompareItemId = null", ensure)
        self.assertNotIn("state.selectedCompareItemId = null", page_handler)

    def test_catalog_refresh_is_explicit_coalesced_and_slice_resilient(self) -> None:
        sync = self.app_js.split("async function waitForCatalogSync", 1)[1].split(
            "async function refreshCatalogData", 1
        )[0]
        refresh = self.app_js.split("async function refreshCatalogData", 1)[1].split(
            "async function refreshCatalog()", 1
        )[0]
        media = self.app_js.split("async function loadMediaLibrary", 1)[1].split(
            "async function createMediaCollection", 1
        )[0]

        self.assertIn('api("/api/media/sync/status")', sync)
        self.assertIn('api("/api/media/sync"', sync)
        self.assertIn("requestCatalogSync(includeRuns = false)", sync)
        self.assertIn("include_runs: Boolean(includeRuns)", sync)
        self.assertIn("requestCatalogSync(Boolean(options.includeRuns))", sync)
        self.assertIn("Promise.allSettled", refresh)
        self.assertIn('"/api/metrics/health?refresh=1"', refresh)
        self.assertIn("Promise.allSettled", media)

    def test_checkpoint_catalog_is_selected_model_scoped_and_request_deduplicated(self) -> None:
        loader = self.app_js.split("async function loadCheckpointsForModel", 1)[1].split(
            "function renderSingleDeviceOptions", 1
        )[0]
        refresh = self.app_js.split("async function refreshCatalogData", 1)[1].split(
            "async function refreshCatalog()", 1
        )[0]

        self.assertIn("checkpointsByModel", loader)
        self.assertIn("checkpointRequestsByModel", loader)
        self.assertIn("?model_file=${encodeURIComponent(normalizedModel)}", loader)
        self.assertIn("state.checkpointRequestsByModel[normalizedModel]", loader)
        self.assertNotIn('api("/api/checkpoints")', refresh)
        self.assertIn("loadCheckpointsForModel", refresh)

    def test_catalog_sync_and_refresh_callers_join_shared_promises(self) -> None:
        sync = self.app_js.split("function shareCatalogSync", 1)[1].split(
            "async function runCatalogRefresh", 1
        )[0]
        startup = self.app_js.split("async function refreshCatalog()", 1)[1].split(
            'document.addEventListener("click"', 1
        )[0]

        self.assertIn("state.catalogSyncPromise", sync)
        self.assertIn("state.catalogRefreshPromise", sync)
        self.assertIn('api("/api/media/sync/status")', sync)
        self.assertIn("joinRunningCatalogSync", startup)
        self.assertLess(sync.index('api("/api/media/sync/status")'), sync.index('api("/api/media/sync"'))

    def test_run_poll_uses_global_active_count_outside_the_visible_page(self) -> None:
        poll = self.app_js.split("function startRunsPoll", 1)[1].split(
            "refreshCatalog()", 1
        )[0]
        refresh = self.app_js.split("async function refreshRunsOnce", 1)[1].split(
            "function refreshRunsOnly", 1
        )[0]

        self.assertIn("state.runsPage.active_total", poll)
        self.assertIn("hasActiveRunWork() ? 2000 : 10000", poll)
        self.assertIn("selectedNeedsRefresh", refresh)
        self.assertIn("await selectRun(state.selectedRun.id", refresh)

    def test_media_sources_are_visible_beyond_the_first_server_page(self) -> None:
        render = self.app_js.split("function renderMediaLibrary", 1)[1].split(
            "async function loadMediaLibrary", 1
        )[0]
        load_more = self.app_js.split("async function loadMoreMediaSources", 1)[1].split(
            "async function createMediaCollection", 1
        )[0]

        self.assertIn("mediaAssetsPage", render)
        self.assertIn("data-media-load-more", render)
        self.assertIn("已显示 ${state.mediaAssets.length}/${mediaTotal}", render)
        self.assertIn("/api/media/sources?page=${nextPage}", load_more)
        self.assertIn("new Map(state.mediaAssets.map", load_more)

    def test_external_prediction_item_binding_reads_one_server_page(self) -> None:
        loader = self.app_js.split("async function loadExternalPredictionBindingItems", 1)[1].split(
            "function renderMediaLibrary", 1
        )[0]
        render = self.app_js.split("function renderExternalPredictionBinding", 1)[1].split(
            "async function loadExternalPredictionBindingItems", 1
        )[0]

        self.assertIn("requestedPage", loader)
        self.assertIn("externalPredItemsPage", loader)
        self.assertNotIn("Promise.all", loader)
        self.assertIn("data-external-pred-item-page", render)
        self.assertIn("selectedExternalPredItem", render)

    def test_run_purge_preview_binds_impact_summary_to_every_mutation(self) -> None:
        request = self.app_js.split("async function requestRunPurgePreview", 1)[1].split(
            "function confirmRunPurgePreview", 1
        )[0]
        message = self.app_js.split("function runPurgePreviewMessage", 1)[1].split(
            "async function requestRunPurgePreview", 1
        )[0]
        batch = self.app_js.split("async function batchDeleteRuns", 1)[1].split(
            "async function cleanupRunArtifacts", 1
        )[0]
        cleanup = self.app_js.split("async function cleanupRunArtifacts", 1)[1].split(
            "async function submitRunFeedback", 1
        )[0]

        self.assertIn('api("/api/run-purge/preview"', request)
        self.assertIn("request_type: requestType", request)
        self.assertIn("run_ids: normalizedIds", request)
        self.assertIn("estimated_reclaimable_bytes", message)
        self.assertIn("shared_cache_bytes", message)
        self.assertIn("exclusive_cache_bytes", message)
        self.assertIn("runPurgeDependencyText(summary.dependencies)", message)
        self.assertIn('withRunPurgePreview("delete_run", ids', batch)
        self.assertIn("preview_token: previewToken", batch)
        self.assertIn('withRunPurgePreview("cleanup_artifacts", [runId]', cleanup)
        self.assertIn("preview_token: previewToken", cleanup)

    def test_subjective_feedback_is_not_labeled_as_objective_statistics(self) -> None:
        self.assertGreaterEqual(self.index_html.count("主观反馈统计"), 2)
        self.assertIn("Campaign 客观指标和 LPIPS 曲线", self.index_html)


if __name__ == "__main__":
    unittest.main()
