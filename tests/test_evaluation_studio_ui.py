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
        cls.studio_css = (WEB / "studio.css").read_text(encoding="utf-8")
        cls.blind_html = (WEB / "blind.html").read_text(encoding="utf-8")
        cls.blind_js = (WEB / "blind.js").read_text(encoding="utf-8")
        cls.blind_css = (WEB / "blind.css").read_text(encoding="utf-8")
        cls.server_py = (ROOT / "src" / "vfieval" / "server.py").read_text(encoding="utf-8")

    def _function_source(self, name: str) -> str:
        match = re.search(
            rf"^(?:async\s+)?function\s+{re.escape(name)}\s*\(",
            self.blind_js,
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(match, f"blind.js is missing function {name}")
        assert match is not None
        following = re.search(
            r"^(?:async\s+)?function\s+[A-Za-z_$][\w$]*\s*\(",
            self.blind_js[match.end():],
            flags=re.MULTILINE,
        )
        end = match.end() + following.start() if following is not None else len(self.blind_js)
        return self.blind_js[match.start():end]

    def test_literal_dom_dependencies_exist(self) -> None:
        index_ids = set(re.findall(r'id="([^"]+)"', self.index_html))
        studio_ids = set(re.findall(r'el\("([^"]+)"\)', self.studio_js))
        self.assertEqual(studio_ids - index_ids, set())

        blind_ids = set(re.findall(r'id="([^"]+)"', self.blind_html))
        blind_script_ids = set(re.findall(r'byId\("([^"]+)"\)', self.blind_js))
        self.assertEqual(blind_script_ids - blind_ids, set())

    def test_blind_page_is_independent_and_uses_only_opaque_task_urls(self) -> None:
        self.assertIn('<script src="/blind.js"', self.blind_html)
        self.assertNotIn('/app.js', self.blind_html)
        self.assertNotIn('class="nav-item', self.blind_html)
        self.assertIn('/api/blind/', self.blind_js)
        self.assertIn('task.token', self.blind_js)
        for forbidden in (
            "run_id",
            "checkpoint",
            "asset_id",
            "method_id",
            "binding_id",
            "task_id",
            "model_name",
            "source_run",
        ):
            self.assertNotIn(forbidden, self.blind_js)
        for anonymous_label in ("参考 GT", "候选 A", "候选 B"):
            self.assertIn(anonymous_label, self.blind_js)

    def test_blind_large_format_hooks_default_to_wipe_with_full_view_available(self) -> None:
        for hook in ("view-wipe", "view-full", "wipe-divider", "wipe-instructions", "sync-status"):
            self.assertIn(f'id="{hook}"', self.blind_html)
        self.assertRegex(
            self.blind_html,
            r'id="view-wipe"[^>]*aria-pressed="true"',
        )
        self.assertRegex(
            self.blind_html,
            r'id="view-full"[^>]*aria-pressed="false"',
        )
        self.assertRegex(
            self.blind_html,
            r'id="media-grid"[^>]*class="[^"]*view-wipe',
        )
        divider = re.search(
            r'<input\s+[^>]*id="wipe-divider"[^>]*>',
            self.blind_html,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(divider)
        assert divider is not None
        self.assertIn('value="50"', divider.group(0))
        self.assertIn('aria-valuetext=', divider.group(0))
        self.assertIn('aria-describedby="wipe-instructions"', divider.group(0))

    def test_blind_view_switch_reuses_exactly_three_media_nodes(self) -> None:
        render_task = self._function_source("renderTask")
        for side in ("reference", "left", "right"):
            self.assertIn(f'createMediaNode(task, "{side}"', render_task)
        self.assertEqual(render_task.count("createMediaNode(task,"), 3)
        self.assertIn("mediaCard(", render_task)
        self.assertIn("candidate-stage", render_task)
        self.assertLess(
            render_task.index('mediaCard(left, "候选 A"'),
            render_task.index('mediaCard(right, "候选 B"'),
        )

        set_view = self._function_source("setViewMode")
        self.assertIn("view-wipe", set_view)
        self.assertIn("view-full", set_view)
        self.assertNotIn("createMediaNode", set_view)
        self.assertNotIn("replaceContent", set_view)
        self.assertNotIn(".src =", set_view)
        self.assertNotIn("frameIndex", set_view)

    def test_blind_wipe_updates_clip_position_and_accessible_value(self) -> None:
        update = self._function_source("updateWipePosition")
        self.assertIn("--wipe-position", update)
        self.assertIn("aria-valuetext", update)
        self.assertIn("blindState.wipeDivider", update)
        initialize = self._function_source("initializeBlindPage")
        self.assertIn('blindState.wipeDivider = byId("wipe-divider")', initialize)
        for token in (
            "--wipe-position",
            "--media-aspect-ratio",
            ".candidate-stage",
            ".candidate-a",
            ".candidate-b",
            "clip-path: inset(",
            "touch-action: pan-y",
            ".media-grid.view-full",
        ):
            self.assertIn(token, self.blind_css)

    def test_blind_video_sync_uses_frame_callback_with_timeupdate_fallback(self) -> None:
        for function_name in (
            "startFrameSynchronization",
            "stopSynchronization",
            "syncFromClockVideo",
            "handleMediaWaiting",
        ):
            self._function_source(function_name)
        start_sync = self._function_source("startFrameSynchronization")
        self.assertIn("requestVideoFrameCallback", start_sync)
        self.assertIn('typeof', start_sync)
        self.assertIn('"function"', start_sync)
        self.assertIn('addEventListener("timeupdate"', self.blind_js)
        self.assertIn('addEventListener("waiting", handleMediaWaiting)', self.blind_js)
        self.assertIn('addEventListener("ended"', self.blind_js)
        self.assertNotIn("0.08", self.blind_js)

        waiting = self._function_source("handleMediaWaiting")
        self.assertIn("pauseSynchronizedPlayback(", waiting)
        pause_all = self._function_source("pauseSynchronizedPlayback")
        self.assertIn("activeVideos()", pause_all)
        self.assertIn("pause()", pause_all)
        self.assertIn("setSyncStatus(", pause_all)
        self.assertIn('byId("sync-status")', self._function_source("setSyncStatus"))
        for control in ("master-play", "master-seek", "master-rate", "master-loop"):
            self.assertIn(f'byId("{control}")', self.blind_js)

    def test_blind_task_replacement_stops_old_synchronization_and_media(self) -> None:
        stop = self._function_source("stopSynchronization")
        self.assertIn("cancelFrameSynchronization()", stop)
        self.assertIn("pause()", stop)
        cancel = self._function_source("cancelFrameSynchronization")
        self.assertIn("cancelVideoFrameCallback", cancel)
        render_task = self._function_source("renderTask")
        self.assertIn("stopSynchronization()", render_task)
        self.assertLess(render_task.index("stopSynchronization()"), render_task.index("replaceContent("))
        play = self._function_source("playSynchronizedVideos")
        self.assertIn("const mediaGeneration = blindState.mediaGeneration", play)
        self.assertIn("videos.every((video) => isCurrentMedia(video))", play)
        reference_play = self._function_source("handleReferencePlay")
        self.assertIn("mediaGeneration !== blindState.mediaGeneration", reference_play)
        self.assertIn("isCurrentMedia(peer)", reference_play)

    def test_blind_frame_sequences_share_one_index_across_all_three_images(self) -> None:
        factory = self._function_source("createMediaNode")
        self.assertIn('document.createElement("img")', factory)
        self.assertIn("dataset.frameBase", factory)
        self.assertIn("blindState.frameIndex", factory)
        update = self._function_source("updateFrame")
        self.assertIn('querySelectorAll("[data-frame-base]")', update)
        self.assertIn("withFrame(image.dataset.frameBase, blindState.frameIndex)", update)
        set_view = self._function_source("setViewMode")
        self.assertNotIn("frameIndex", set_view)
        self.assertNotIn(".src =", set_view)

    def test_blind_clip_path_fallback_forces_full_view(self) -> None:
        supports_wipe = self._function_source("supportsWipeView")
        self.assertIn('CSS.supports("aspect-ratio", "16 / 9")', supports_wipe)
        self.assertIn('CSS.supports("clip-path", clipValue)', supports_wipe)
        self.assertIn("calc(100% - var(--wipe-position, 50%))", supports_wipe)
        initialize = self._function_source("initializeBlindPage")
        self.assertIn("blindState.wipeSupported = supportsWipeView()", initialize)
        self.assertIn('setViewMode(blindState.wipeSupported ? "wipe" : "full")', initialize)
        set_view = self._function_source("setViewMode")
        self.assertIn("wipeButton.disabled = !blindState.wipeSupported", set_view)
        self.assertIn("@supports not ((clip-path: inset(0 50% 0 0))", self.blind_css)
        self.assertIn("-webkit-clip-path: inset(0 50% 0 0)", self.blind_css)

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

    def test_blind_page_generates_an_evaluator_id_without_secure_random_uuid(self) -> None:
        self.assertIn("function newEvaluatorId()", self.blind_js)
        self.assertIn('typeof window.crypto.randomUUID === "function"', self.blind_js)
        self.assertIn("browser-${Date.now()}-${Math.random().toString(16).slice(2)}", self.blind_js)
        self.assertIn('readLocalValue("vfieval-evaluator-id") || newEvaluatorId()', self.blind_js)
        self.assertNotIn('|| crypto.randomUUID()', self.blind_js)

    def test_blind_page_survives_restricted_storage_and_surfaces_startup_errors(self) -> None:
        self.assertIn("function readLocalValue(key)", self.blind_js)
        self.assertIn("function writeLocalValue(key, value)", self.blind_js)
        self.assertIn("function removeLocalValue(key)", self.blind_js)
        self.assertIn("function initializeBlindPage()", self.blind_js)
        self.assertIn("initializeBlindPage();", self.blind_js)
        self.assertRegex(
            self.blind_js,
            r"try \{\s+initializeBlindPage\(\);\s+window\.__vfievalBlindReady = true;\s+\} catch \(error\) \{\s+showError\(error\);",
        )

    def test_blind_page_avoids_newer_syntax_and_reports_script_boot_failures(self) -> None:
        self.assertNotIn("?.", self.blind_js)
        self.assertNotIn("??", self.blind_js)
        self.assertNotIn("Promise.allSettled", self.blind_js)
        self.assertIn("function replaceContent(element, ...nodes)", self.blind_js)
        self.assertIn("window.__vfievalBlindReady = true;", self.blind_js)
        self.assertIn("盲评页面脚本未能加载", self.blind_html)
        self.assertIn("无法加载 /blind.js", self.blind_html)
        self.assertIn("window.__vfievalBlindBootError", self.blind_html)
        self.assertIn("连接盲评服务超时", self.blind_js)
        self.assertIn("Promise.race", self.blind_js)

    def test_blind_session_submit_has_visible_pending_and_failure_state(self) -> None:
        start = self.blind_js.index("async function saveSession(event)")
        end = self.blind_js.index("\nasync function submitVote", start)
        handler = self.blind_js[start:end]
        self.assertIn('submitButton.textContent = "正在进入…";', handler)
        self.assertIn("submitButton.disabled = true;", handler)
        self.assertIn("finally {", handler)
        self.assertIn("submitButton.disabled = false;", handler)
        self.assertIn('writeLocalValue("vfieval-evaluator-name", displayName);', handler)

    def test_blind_vote_captures_its_form_before_await(self) -> None:
        start = self.blind_js.index("async function submitVote(event)")
        end = self.blind_js.index("\nfunction activeVideos()", start)
        handler = self.blind_js[start:end]
        self.assertIn("const voteForm = event.currentTarget;", handler)
        self.assertIn('const buttons = voteForm.querySelectorAll("button");', handler)
        self.assertIn("const form = new FormData(voteForm);", handler)
        self.assertIn("voteForm.reset();", handler)
        self.assertLess(handler.index("const voteForm = event.currentTarget;"), handler.index("await blindApi"))
        self.assertNotIn("event.currentTarget", handler[handler.index("await blindApi"):])
        self.assertNotIn("event.currentTarget.reset()", handler)

    def test_blind_playback_and_lease_state_are_reset_between_tasks(self) -> None:
        self.assertIn('const rateControl = byId("master-rate");', self.blind_js)
        self.assertIn("video.playbackRate = Number((rateControl && rateControl.value) || 1);", self.blind_js)
        self.assertIn('const loopControl = byId("master-loop");', self.blind_js)
        self.assertIn("video.loop = Boolean(loopControl && loopControl.checked);", self.blind_js)
        self.assertIn("function syncFromClockVideo(event)", self.blind_js)
        self.assertIn('replaceContent(byId("media-grid"))', self.blind_js)
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

    def test_campaign_preparation_progress_keeps_legacy_summary_and_shows_fine_details(self) -> None:
        start = self.studio_js.index("function preparationProgressMarkup(progress, campaign)")
        end = self.studio_js.index("\n  function renderCampaignDetail", start)
        renderer = self.studio_js[start:end]
        self.assertIn("const legacyMarkup", renderer)
        self.assertIn(
            '<div class="studio-progress"><progress max="${total}" value="${current}"></progress><span>${current}/${total} · ${safe(phase)}</span></div>',
            renderer,
        )
        self.assertIn("if (!hasFineProgress) return legacyMarkup;", renderer)
        for field in (
            "overall_fraction",
            "stage",
            "item_index",
            "item_name",
            "frame_current",
            "frame_total",
            "pipeline",
            "timings",
        ):
            self.assertIn(field, renderer)
        self.assertIn("...report, ...progress", renderer)
        self.assertIn("Item ${Number(details.item_index)}/${total}", renderer)
        self.assertIn("帧 ${frameCurrent}${frameTotal}", renderer)
        self.assertIn("总进度 ${(fraction * 100).toFixed(1)}%", renderer)
        self.assertIn("studio-progress-detailed", renderer)
        self.assertIn(".studio-progress-detail", self.studio_css)

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
