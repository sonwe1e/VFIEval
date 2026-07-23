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
        cls.shared_js = (WEB / "shared.js").read_text(encoding="utf-8")
        cls.app_entry_js = (WEB / "app.js").read_text(encoding="utf-8")
        cls.compare_js = (WEB / "compare.js").read_text(encoding="utf-8")
        cls.run_detail_js = (WEB / "run-detail.js").read_text(encoding="utf-8")
        cls.media_js = (WEB / "media.js").read_text(encoding="utf-8")
        cls.app_js = "\n".join(
            (
                cls.app_entry_js,
                cls.compare_js,
                cls.run_detail_js,
                cls.media_js,
            )
        )
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

    def _studio_function_source(self, name: str) -> str:
        match = re.search(
            rf"^  (?:async\s+)?function\s+{re.escape(name)}\s*\(",
            self.studio_js,
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(match, f"studio.js is missing function {name}")
        assert match is not None
        following = re.search(
            r"^  (?:async\s+)?function\s+[A-Za-z_$][\w$]*\s*\(",
            self.studio_js[match.end():],
            flags=re.MULTILINE,
        )
        end = match.end() + following.start() if following is not None else len(self.studio_js)
        return self.studio_js[match.start():end]

    def test_literal_dom_dependencies_exist(self) -> None:
        index_ids = set(re.findall(r'id="([^"]+)"', self.index_html))
        studio_ids = set(re.findall(r'el\("([^"]+)"\)', self.studio_js))
        self.assertEqual(studio_ids - index_ids, set())

        blind_ids = set(re.findall(r'id="([^"]+)"', self.blind_html))
        blind_script_ids = set(re.findall(r'byId\("([^"]+)"\)', self.blind_js))
        self.assertEqual(blind_script_ids - blind_ids, set())

    def test_blind_page_is_independent_and_uses_only_opaque_task_urls(self) -> None:
        self.assertIn('<script src="/shared.js"', self.blind_html)
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
        create_media = self._function_source("createMediaNode")
        self.assertIn('addEventListener("pause", handleMediaPause)', create_media)
        self.assertIn('addEventListener("ended", handleMediaEnded)', create_media)
        for event_name in ("timeupdate", "play", "seeking", "seeked", "ratechange"):
            self.assertIn(f'addEventListener("{event_name}"', create_media)
        self.assertNotIn('if (side === "reference")', create_media)
        self.assertNotIn("0.08", self.blind_js)

        waiting = self._function_source("handleMediaWaiting")
        self.assertIn("blindState.syncPlayIntent", waiting)
        self.assertIn("blindState.syncAttempt += 1", waiting)
        self.assertIn("pauseVideosInternally()", waiting)
        self.assertIn("markRecoveryContext(initialBuffering)", waiting)
        self.assertIn("bufferingDiagnostic(media, target)", waiting)
        self.assertNotIn("pauseSynchronizedPlayback(", waiting)
        ready = self._function_source("handleMediaCanPlay")
        self.assertIn("maybeResumeBufferedPlayback()", ready)
        self.assertIn(
            "playSynchronizedVideos(true)",
            self._function_source("maybeResumeBufferedPlayback"),
        )
        pause_all = self._function_source("pauseSynchronizedPlayback")
        self.assertIn("pauseVideosInternally()", pause_all)
        internal_pause = self._function_source("pauseVideosInternally")
        self.assertIn("activeVideos()", internal_pause)
        self.assertIn("pause()", internal_pause)
        self.assertNotIn('dataset.mediaSide === "reference"', internal_pause)
        self.assertIn("setSyncStatus(", pause_all)
        seeked = self._function_source("handleReferenceSeeked")
        self.assertIn("blindState.syncWaiting", seeked)
        self.assertIn('"stalled"', seeked)
        self.assertIn('byId("sync-status")', self._function_source("setSyncStatus"))
        for control in ("master-play", "master-seek", "master-rate", "master-loop"):
            self.assertIn(f'byId("{control}")', self.blind_js)

    def test_blind_task_replacement_stops_old_synchronization_and_media(self) -> None:
        stop = self._function_source("stopSynchronization")
        self.assertIn("cancelFrameSynchronization()", stop)
        self.assertIn("pauseVideosInternally(allVideos())", stop)
        self.assertIn("abortPreload()", stop)
        self.assertIn("stopStreamMonitoring()", stop)
        self.assertIn("releaseBlobUrls()", stop)
        cancel = self._function_source("cancelFrameSynchronization")
        self.assertIn("cancelVideoFrameCallback", cancel)
        render_task = self._function_source("renderTask")
        self.assertIn("stopSynchronization()", render_task)
        self.assertLess(render_task.index("stopSynchronization()"), render_task.index("replaceContent("))
        play = self._function_source("playSynchronizedVideos")
        self.assertIn("const mediaGeneration = blindState.mediaGeneration", play)
        self.assertIn("const scopeEpoch = blindState.scopeEpoch", play)
        self.assertIn("videos.every((video) => isCurrentMedia(video) && isActiveMedia(video))", play)
        reference_play = self._function_source("handleReferencePlay")
        self.assertIn("isActiveClock(clock)", reference_play)
        self.assertIn("playSynchronizedVideos(false)", reference_play)

    def test_blind_playback_scope_is_dynamic_and_resets_with_each_view_and_task(self) -> None:
        for scope in ("all", "reference", "candidates", "left", "right"):
            self.assertIn(f'data-playback-scope="{scope}"', self.blind_html)
        self.assertIn("两个视图同步", self.blind_html)
        self.assertIn("仅候选对比", self.blind_html)
        dock_start = self.blind_html.index('<div class="media-control-dock">')
        dock_end = self.blind_html.index('<div id="media-grid"', dock_start)
        dock_markup = self.blind_html[dock_start:dock_end]
        self.assertIn('id="playback-scope"', dock_markup)
        self.assertIn('id="playback-controls"', dock_markup)
        self.assertIn('id="frame-control"', dock_markup)
        self.assertIn(".media-control-dock { position: sticky", self.blind_css)

        scopes = self._function_source("playbackScopesForView")
        self.assertIn('["all", "reference", "left", "right"]', scopes)
        self.assertIn('["all", "reference", "candidates"]', scopes)
        active = self._function_source("activeMediaSides")
        for side_scope in ("reference", "candidates", "left", "right"):
            self.assertIn(f'blindState.playbackScope === "{side_scope}"', active)
        clock = self._function_source("playbackClock")
        self.assertIn('dataset.mediaSide === "reference"', clock)
        self.assertIn("videos[0]", clock)

        set_scope = self._function_source("setPlaybackScope")
        self.assertIn("pauseAndAlignForPlaybackChange()", set_scope)
        self.assertIn("advancePlaybackScopeEpoch()", set_scope)
        self.assertIn("blindState.syncClock = playbackClock()", set_scope)
        self.assertNotIn("playSynchronizedVideos", set_scope)
        self.assertNotIn(".play()", set_scope)
        set_view = self._function_source("setViewMode")
        self.assertIn(
            'setPlaybackScope(changed ? "all" : blindState.playbackScope, true)',
            set_view,
        )
        self.assertNotIn("playSynchronizedVideos", set_view)
        self.assertNotIn(".play()", set_view)
        self.assertNotIn("scopeByView", self.blind_js)
        render = self._function_source("renderTask")
        self.assertIn('blindState.viewMode = blindState.wipeSupported ? "wipe" : "full"', render)
        self.assertIn('blindState.playbackScope = "all"', render)

    def test_blind_short_video_preload_is_atomic_abortable_and_reports_anonymous_progress(self) -> None:
        self.assertIn("FULL_BLOB_PRELOAD_TOTAL_MAX_BYTES = 256 * 1024 * 1024", self.blind_js)
        self.assertIn("FULL_BLOB_PRELOAD_MAX_DURATION_SECONDS = 30", self.blind_js)
        self.assertNotIn("SHORT_VIDEO_BLOB_MAX_FRAMES", self.blind_js)
        prepare = self._function_source("prepareTaskVideoSources")
        self.assertIn("task && task.duration_seconds", prepare)
        self.assertIn("Promise.all(sourceUrls.map", prepare)
        self.assertIn("contentLengths.reduce", prepare)
        self.assertIn("FULL_BLOB_PRELOAD_TOTAL_MAX_BYTES", prepare)
        self.assertIn("preloadContextIsCurrent(mediaGeneration, scopeEpoch, videos)", prepare)
        self.assertLess(prepare.index("const blobs = await Promise.all"), prepare.index("createdUrls.push"))

        reader = self._function_source("readPreloadResponseBlob")
        self.assertIn("response.body.getReader()", reader)
        self.assertIn("row.received", reader)
        self.assertIn("signal && signal.aborted", reader)
        self.assertIn("preloadContextIsCurrent(mediaGeneration, scopeEpoch, videos)", reader)
        self.assertIn("preloadOperationWithNoProgressTimeout", reader)
        timeout = self._function_source("preloadOperationWithNoProgressTimeout")
        self.assertIn("STREAM_NO_PROGRESS_RELOAD_MS", timeout)
        self.assertIn('error.name = "TimeoutError"', timeout)
        progress = self._function_source("updateAtomicPreloadProgress")
        for label in ('row.label', 'row.received', 'row.total', '" · "'):
            self.assertIn(label, progress)
        self.assertIn('["GT", "A", "B"]', prepare)

        abort = self._function_source("abortPreload")
        self.assertIn("controller.abort()", abort)
        self.assertIn("preloadController", self.blind_js)
        advance = self._function_source("advancePlaybackScopeEpoch")
        self.assertIn("blindState.scopeEpoch += 1", advance)
        self.assertIn("abortPreload()", advance)
        self.assertIn("startTaskVideoPreparation(currentTask())", advance)

    def test_blind_stream_watermarks_reload_and_fatal_vote_gate_are_explicit(self) -> None:
        for contract in (
            "STREAM_INITIAL_WATER_SECONDS = 10",
            "STREAM_LOW_WATER_SECONDS = 1.5",
            "STREAM_RESUME_WATER_SECONDS = 5",
            "STREAM_NO_PROGRESS_RELOAD_MS = 60_000",
            "STREAM_MONITOR_INTERVAL_MS = 250",
        ):
            self.assertIn(contract, self.blind_js)
        play = self._function_source("playSynchronizedVideos")
        self.assertIn("videos.every(mediaReadyForInitialPlayback)", play)
        self.assertIn("markRecoveryContext(true)", play)
        self.assertIn(
            "blindState.initialBufferScopeEpoch !== scopeEpoch",
            play,
        )
        self.assertIn("requiresInitialBuffer", play)
        self.assertIn("blindState.initialBufferScopeEpoch = scopeEpoch", play)
        self.assertLess(play.index("videos.every(mediaReadyForInitialPlayback)"), play.index("video.play()"))
        waiting = self._function_source("handleMediaWaiting")
        self.assertIn("const initialBuffering = blindState.syncInitialBuffering", waiting)
        self.assertIn("markRecoveryContext(initialBuffering)", waiting)
        self.assertIn("bufferingDiagnostic(media, target)", waiting)
        self.assertIn("isActiveMedia(media)", waiting)
        canplay = self._function_source("handleMediaCanPlay")
        self.assertIn("isActiveMedia(media)", canplay)
        self.assertIn("recoveryContextIsCurrent()", canplay)
        self.assertIn("finishMediaReloadIfReady()", canplay)
        self.assertLess(
            canplay.index("finishMediaReloadIfReady()"),
            canplay.index("!isActiveMedia(media)"),
        )

        stalled = self._function_source("stopStalledAutoResume")
        self.assertIn("streamAutoResumeBlocked = true", stalled)
        self.assertIn('setHidden("media-reload", false)', stalled)
        self.assertNotIn(".src =", stalled)
        reload_media = self._function_source("reloadTaskMedia")
        self.assertIn("video.src = sourceUrl", reload_media)
        self.assertIn("mediaReloadGeneration", reload_media)
        finish_reload = self._function_source("finishMediaReloadIfReady")
        self.assertIn("mediaReloadScopeEpoch", finish_reload)
        self.assertIn("mediaReloadAttempt", finish_reload)
        self.assertIn("clearTaskMediaFatal()", finish_reload)

        media_error = self._function_source("handleMediaErrorState")
        self.assertIn("isCurrentMedia(media)", media_error)
        self.assertNotIn("isActiveMedia(media)", media_error)
        self.assertIn("markTaskMediaFatal", media_error)
        fatal = self._function_source("markTaskMediaFatal")
        self.assertIn("setVoteMediaBlocked(true)", fatal)
        submit = self._function_source("submitVote")
        self.assertIn("blindState.mediaFatal", submit)
        self.assertIn("blindState.mediaReloadPending", submit)
        self.assertIn("blindState.frameSequencePending", submit)

    def test_blind_video_vote_waits_for_metadata_first_frame_and_sync_readiness(self) -> None:
        self.assertIn('mediaReadiness: "idle"', self.blind_js)
        render = self._function_source("renderTask")
        self.assertIn('blindState.mediaReadiness = task ? "media_pending" : "idle"', render)
        self.assertIn("beginTaskVideoReadiness()", render)
        self.assertNotIn("三路视频已就绪", render)

        create_media = self._function_source("createMediaNode")
        self.assertIn('addEventListener("loadedmetadata", handleMediaMetadata)', create_media)
        self.assertIn('addEventListener("loadeddata", handleMediaFirstFrame)', create_media)
        self.assertLess(
            create_media.index('addEventListener("loadeddata"'),
            create_media.index("configureVideoSource(video, url)"),
        )

        ready = self._function_source("finishTaskVideoReadinessIfReady")
        for contract in (
            'blindState.mediaReadiness !== "media_pending"',
            'video.dataset.metadataReady === "true"',
            'video.dataset.firstFrameReady === "true"',
            "videoDurationCompatibility(videos)",
            "Math.abs(Number(video.currentTime || 0) - anchor) <= threshold",
            'setMediaReadiness("media_ready")',
        ):
            self.assertIn(contract, ready)
        self.assertLess(
            ready.index('video.dataset.metadataReady === "true"'),
            ready.index('setMediaReadiness("media_ready")'),
        )
        self.assertLess(
            ready.index('video.dataset.firstFrameReady === "true"'),
            ready.index('setMediaReadiness("media_ready")'),
        )

        submit = self._function_source("submitVote")
        self.assertIn('blindState.mediaReadiness !== "media_ready"', submit)
        self.assertIn("setVoteMediaBlocked", submit)
        self.assertIn('setMediaReadiness("media_pending")', self._function_source("reloadTaskMedia"))

    def test_frontend_request_errors_keep_copyable_request_and_support_ids(self) -> None:
        self.assertIn('id="request-diagnostic"', self.index_html)
        self.assertIn('id="request-diagnostic"', self.blind_html)

        studio_request = self._studio_function_source("request")
        self.assertIn("Shared.request(path", studio_request)
        self.assertIn("fetchOptions", studio_request)
        self.assertIn("reportRequestFailure(error)", studio_request)
        self.assertIn("window.showRequestDiagnostic", self.studio_js)

        blind_request = self._function_source("blindApi")
        self.assertIn("Shared.request(path", blind_request)
        self.assertIn("timeoutMs: 15_000", blind_request)
        self.assertIn("requireJsonSuccess: true", blind_request)
        self.assertIn("showRequestDiagnostic(error)", blind_request)
        diagnostic = self._function_source("showRequestDiagnostic")
        self.assertIn("request_id:", diagnostic)
        self.assertIn("support_id:", diagnostic)
        self.assertIn("Shared.copyText", diagnostic)
        self.assertIn("recovery_suggestion", diagnostic)
        for field in ("code", "message", "request_id", "support_id", "details"):
            self.assertIn(field, self.shared_js)

    def test_campaign_creation_is_single_flight_and_disables_the_form(self) -> None:
        self.assertRegex(
            self.index_html,
            r'id="studio-create-campaign"[^>]*disabled',
        )
        self.assertIn('id="studio-submit-status"', self.index_html)
        create = self._studio_function_source("createCampaign")
        self.assertIn("campaignCreationFlight.isLocked()", create)
        self.assertIn("campaignCreationFlight.tryLock()", create)
        self.assertIn("campaignCreationFlight.release()", create)
        self.assertIn("studioState.campaignSubmitting = true", create)
        self.assertIn('studioState.campaignSubmitPhase = "creating"', create)
        self.assertIn('studioState.campaignSubmitPhase = "publishing"', create)
        self.assertIn("studioState.campaignSubmitting = false", create)
        render = self._studio_function_source("renderCampaignSubmissionState")
        self.assertIn('form.setAttribute("aria-busy"', render)
        self.assertIn("submit.disabled = studioState.campaignSubmitting", render)
        self.assertIn("请勿重复点击", render)

    def test_blind_integer_frame_timeline_previews_then_commits_frame_midpoints(self) -> None:
        for hook in ("master-prev", "master-next", "master-frame-label", "frame-prev", "frame-next"):
            self.assertIn(f'id="{hook}"', self.blind_html)
        self.assertRegex(self.blind_html, r'id="master-seek"[^>]*step="1"')
        initialize = self._function_source("initializeBlindPage")
        self.assertIn('addEventListener("input", (event) => previewVideoFrame', initialize)
        self.assertIn('addEventListener("change", (event) => commitVideoFrameScrub', initialize)
        self.assertIn('addEventListener("pointerup", finishInterruptedVideoScrub)', initialize)
        self.assertIn('addEventListener("pointercancel", finishInterruptedVideoScrub)', initialize)
        self.assertIn('addEventListener("lostpointercapture", finishInterruptedVideoScrub)', initialize)
        self.assertIn('addEventListener("blur", finishInterruptedVideoScrub)', initialize)
        self.assertIn('addEventListener("input", (event) => previewSequenceFrame', initialize)
        self.assertIn('addEventListener("change", (event) => updateFrame', initialize)
        self.assertNotIn(".src =", self._function_source("previewSequenceFrame"))
        self.assertNotIn("currentTime", self._function_source("previewVideoFrame"))

        seek_frame = self._function_source("seekVideoToFrame")
        self.assertIn("(index + 0.5) / fps", seek_frame)
        self.assertIn("(index + 0.5) / taskFrameCount()", seek_frame)
        self.assertIn("pendingFrameIndex", seek_frame)
        reliability = self._function_source("frameSteppingIsReliable")
        self.assertIn("taskFramesPerSecond()", reliability)
        self.assertIn("reliableMediaDuration(video)", reliability)
        step = self._function_source("stepVideoFrame")
        self.assertIn("frameSteppingIsReliable(playbackClock())", step)
        start_sync = self._function_source("startFrameSynchronization")
        self.assertIn("metadata.mediaTime", start_sync)
        self.assertIn("updateMasterSeek(mediaTime", start_sync)
        paused_correction = self._function_source("schedulePausedFrameCorrection")
        self.assertIn("requestVideoFrameCallback", paused_correction)
        self.assertIn("metadata.mediaTime", paused_correction)
        self.assertIn("mediaGeneration !== blindState.mediaGeneration", paused_correction)
        self.assertIn("scopeEpoch !== blindState.scopeEpoch", paused_correction)
        self.assertIn("attempt !== blindState.syncAttempt", paused_correction)
        seeked = self._function_source("handleReferenceSeeked")
        self.assertIn("schedulePausedFrameCorrection(clock)", seeked)
        update_seek = self._function_source("updateMasterSeek")
        self.assertIn("blindState.videoScrubbing", update_seek)
        preview = self._function_source("previewVideoFrame")
        self.assertIn("beginVideoFrameScrub()", preview)
        self.assertIn("blindState.videoScrubDirty = true", preview)
        commit = self._function_source("commitVideoFrameScrub")
        self.assertIn("blindState.videoScrubbing = false", commit)
        self.assertIn("blindState.videoScrubDirty", commit)
        self.assertIn("seekVideos(value)", commit)

        for token in (
            "position: sticky",
            'grid-template-areas: "play timeline timeline" "rate loop status"',
            ".frame-stepper",
        ):
            self.assertIn(token, self.blind_css)

    def test_blind_vote_ratings_reviews_and_sticky_controls_are_present(self) -> None:
        for token in (
            'name="choice"',
            'id="left-rating"',
            'id="right-rating"',
            'step="0.25"',
            'id="review-list"',
        ):
            self.assertIn(token, self.blind_html)
        self.assertIn("position: sticky", self.blind_css)
        self.assertIn("grid-template-areas", self.blind_css)
        submit = self._function_source("submitVote")
        self.assertIn("left_rating", submit)
        self.assertIn("right_rating", submit)
        self.assertIn("reasons: []", submit)
        self.assertIn("function loadReviews", self.blind_js)
        self.assertIn("function openReview", self.blind_js)

    def test_blind_unpublished_lifecycle_never_renders_completion_or_reviews(self) -> None:
        availability = self._function_source("campaignParticipantAvailable")
        lifecycle = self._function_source("unavailableCampaignMessage")
        render = self._function_source("renderPayload")
        load = self._function_source("loadBlindPayload")
        visibility = self._function_source("initializeBlindPage")
        self.assertIn('["published", "closed", "archived"]', availability)
        for status in ('status === "preparing"', 'status === "failed"', 'status === "draft"'):
            self.assertIn(status, lifecycle)
        self.assertIn("盲测正在发布准备中", lifecycle)
        self.assertIn("页面会自动刷新", lifecycle)
        self.assertIn("请联系组织者确认发布状态", lifecycle)
        self.assertIn("if (unavailable) {", render)
        self.assertIn('setHidden("complete-panel", true)', render)
        self.assertIn("if (unavailable.retry) scheduleTaskRetry()", render)
        self.assertLess(render.index("if (unavailable) {"), render.index("if (complete) {"))
        self.assertIn("const payload = await blindApi", load)
        self.assertIn("renderPayload(payload);", load)
        self.assertIn("if (campaignParticipantAvailable(payload.campaign))", load)
        self.assertLess(
            load.index("renderPayload(payload);"),
            load.index('byId("progress").textContent = "等待加入"'),
        )
        self.assertIn('blindState.payload.campaign.status || ""', visibility)
        self.assertIn(') === "preparing"', visibility)

    def test_blind_frame_sequences_share_one_index_across_all_three_images(self) -> None:
        factory = self._function_source("createMediaNode")
        self.assertIn('document.createElement("img")', factory)
        self.assertIn("dataset.frameBase", factory)
        self.assertIn("handleFrameSequenceLoad", factory)
        self.assertIn("handleFrameSequenceError", factory)
        self.assertNotIn(".src =", factory.split('document.createElement("video")')[0])
        render = self._function_source("renderTask")
        self.assertIn("startFrameSequenceRequest(allFrameSequenceImages(), blindState.frameIndex)", render)
        update = self._function_source("updateFrame")
        self.assertIn("replaceFrameSequenceImages(blindState.frameIndex)", update)
        source = self._function_source("setFrameSequenceSource")
        self.assertIn("withFrame(image.dataset.frameBase, frame)", source)
        self.assertIn("withReloadNonce", source)
        set_view = self._function_source("setViewMode")
        self.assertNotIn("frameIndex", set_view)
        self.assertNotIn(".src =", set_view)

    def test_blind_frame_sequence_errors_block_votes_and_retry_all_three_atomically(self) -> None:
        error = self._function_source("handleFrameSequenceError")
        self.assertIn("isCurrentMedia(image)", error)
        self.assertIn("mediaLoadError", error)
        load_error = self._function_source("mediaLoadError")
        self.assertIn("markTaskMediaFatal(message)", load_error)
        start = self._function_source("startFrameSequenceRequest")
        self.assertIn("frameSequencePending = true", start)
        self.assertIn('setMediaReadiness("media_pending")', start)
        self.assertIn("armFrameSequenceWatchdog()", start)
        watchdog = self._function_source("armFrameSequenceWatchdog")
        self.assertIn("STREAM_NO_PROGRESS_RELOAD_MS", watchdog)
        self.assertIn("markTaskMediaFatal", watchdog)
        finish_request = self._function_source("finishFrameSequenceRequestIfReady")
        self.assertIn("images.length !== 3", finish_request)
        self.assertIn("frameSequenceRequestToken", finish_request)
        self.assertIn('setMediaReadiness("media_ready")', finish_request)

        reload_media = self._function_source("reloadTaskMedia")
        self.assertIn("allFrameSequenceImages()", reload_media)
        self.assertIn("reloadFrameSequenceMedia()", reload_media)
        reload_frames = self._function_source("reloadFrameSequenceMedia")
        self.assertIn("images.length !== 3", reload_frames)
        self.assertIn("replaceFrameSequenceImages(blindState.frameIndex, nonce)", reload_frames)
        self.assertIn("mediaReloadGeneration", reload_frames)
        self.assertIn("mediaReloadScopeEpoch", reload_frames)
        self.assertIn("mediaReloadAttempt", reload_frames)
        self.assertIn("STREAM_NO_PROGRESS_RELOAD_MS", reload_frames)
        finish = self._function_source("finishFrameSequenceReloadIfReady")
        self.assertIn("images.length !== 3", finish)
        self.assertIn('image.dataset.reloadReady === "true"', finish)
        self.assertIn("clearTaskMediaFatal()", finish)
        replacement = self._function_source("replacementFrameSequenceImage")
        self.assertIn("image.replaceWith(replacement)", replacement)

    def test_blind_bfcache_pause_and_restore_preserve_media_lifecycle(self) -> None:
        initialize = self._function_source("initializeBlindPage")
        self.assertIn('addEventListener("pagehide", handleBlindPageHide)', initialize)
        self.assertIn('addEventListener("pageshow", handleBlindPageShow)', initialize)
        page_hide = self._function_source("handleBlindPageHide")
        self.assertIn("event.persisted", page_hide)
        self.assertIn("pauseForPageCache()", page_hide)
        self.assertIn("stopSynchronization()", page_hide)
        cached_pause = self._function_source("pauseForPageCache")
        self.assertNotIn("syncAttempt += 1", cached_pause)
        self.assertNotIn("releaseBlobUrls()", cached_pause)
        self.assertIn("abortPreload()", cached_pause)
        self.assertIn("clearMediaReloadTimer()", cached_pause)
        page_show = self._function_source("handleBlindPageShow")
        self.assertIn("startStreamMonitoring()", page_show)
        self.assertIn("streamLastProgressAt", page_show)
        self.assertIn("armFrameSequenceWatchdog()", page_show)
        self.assertIn("finishMediaReloadIfReady()", page_show)
        self.assertIn("finishFrameSequenceRequestIfReady()", page_show)

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
        self.assertIn("function campaignItemSelectionPayload()", self.studio_js)
        self.assertIn("selection_token: studioState.itemSelectionToken", self.studio_js)
        self.assertIn("...campaignItemSelectionPayload()", self.studio_js)
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

    def test_historical_evaluation_contract_warning_is_visible(self) -> None:
        detail = self._studio_function_source("renderCampaignDetail")
        self.assertIn("contract_warnings", detail)
        self.assertIn("evaluation-contract-warning", detail)
        self.assertIn("midpoint-triplet-v2", detail)

    def test_v2_lpips_curve_is_lazy_fingerprint_scoped_and_race_guarded(self) -> None:
        self.assertIn("/objective-curve?item_id=", self.studio_js)
        self.assertIn("data-objective-curve-item", self.studio_js)
        self.assertIn("data-objective-curve-metric", self.studio_js)
        self.assertIn('class="objective-curve-line series-', self.studio_js)
        self.assertIn("objectiveCurveFingerprint", self.studio_js)
        self.assertIn("objectiveCurveInFlight?.key === key", self.studio_js)
        self.assertIn("studioState.objectiveCurveInFlight !== requestToken", self.studio_js)
        self.assertIn("objectiveCurveErrors[key]", self.studio_js)
        self.assertIn("data-objective-curve-retry", self.studio_js)
        self.assertIn(".objective-curve-plot", self.studio_css)

        load_curve = self._studio_function_source("loadObjectiveCurve")
        before_request = load_curve[:load_curve.index("const curve = await request(")]
        self.assertNotIn("renderCampaignDetail", before_request)
        self.assertIn(
            "if (studioState.objectiveCurveInFlight?.key === key) return",
            load_curve,
        )
        self.assertIn(
            "studioState.objectiveCurveInFlight !== requestToken",
            load_curve,
        )
        self.assertIn(
            "generation !== studioState.objectiveCurveRequestGeneration) return",
            load_curve,
        )
        supersede = self._studio_function_source("supersedeObjectiveCurveRequest")
        self.assertIn("studioState.objectiveCurveInFlight = null", supersede)
        self.assertIn("studioState.objectiveCurveLoadingKey = \"\"", supersede)

        render_chart = self._studio_function_source("renderObjectiveCurveChart")
        reason_index = render_chart.index("Object.keys(series.reason_counts || {})")
        no_completed_index = render_chart.index("if (!completed.length)")
        self.assertLess(reason_index, no_completed_index)

    def test_v2_lpips_curve_has_keyboard_readout_and_equivalent_table(self) -> None:
        render_chart = self._studio_function_source("renderObjectiveCurveChart")
        render_table = self._studio_function_source(
            "renderObjectiveCurveDataTable"
        )

        self.assertIn("data-objective-curve-point", render_chart)
        self.assertIn("data-objective-curve-order", render_chart)
        self.assertIn("data-objective-curve-label", render_chart)
        self.assertIn('aria-label="${safe(label)}"', render_chart)
        self.assertIn("data-objective-curve-readout", render_chart)
        self.assertIn('role="status"', render_chart)
        self.assertIn("renderObjectiveCurveDataTable(curve, pointRows)", render_chart)
        self.assertIn("objective-curve-data-table", render_table)
        self.assertIn("<thead>", render_table)
        self.assertIn("<tbody>${rows}</tbody>", render_table)
        self.assertIn('scope="row"', render_table)
        self.assertIn('document.addEventListener("focusin"', self.studio_js)
        self.assertIn("point.dataset.objectiveCurveLabel", self.studio_js)
        self.assertIn('document.addEventListener("keydown"', self.studio_js)
        self.assertIn("ArrowLeft: current - 1", self.studio_js)
        self.assertIn("ArrowRight: current + 1", self.studio_js)
        self.assertIn("].focus();", self.studio_js)
        self.assertIn(".objective-curve-readout", self.studio_css)
        self.assertIn(".objective-curve-data-table", self.studio_css)

    def test_v2_campaign_permanent_delete_and_dependency_entry_are_exposed(self) -> None:
        self.assertIn("data-studio-delete", self.studio_js)
        self.assertIn('method: "DELETE"', self.studio_js)
        self.assertIn("confirm_destructive", self.studio_js)
        self.assertIn("openCampaign,", self.studio_js)
        self.assertIn("data-open-campaign-dependency", self.app_js)
        self.assertIn("VFIEvalStudio.openCampaign", self.app_js)
        self.assertIn("sibling shard failed the run", self.app_js)
        self.assertIn("Run already failed", self.app_js)
        self.assertIn("级联取消", self.app_js)

    def test_first_visit_waits_for_public_intro_before_session_creation(self) -> None:
        self.assertIn("if (!blindState.evaluatorName)", self.blind_js)
        self.assertIn("/session`,", self.blind_js)
        self.assertIn("display_name: displayName", self.blind_js)

    def test_blind_page_generates_an_evaluator_id_without_secure_random_uuid(self) -> None:
        self.assertIn("function newEvaluatorId()", self.blind_js)
        self.assertIn('Shared.createSubmissionId("browser")', self.blind_js)
        self.assertIn('Shared.storageGet("vfieval-evaluator-id", "") || newEvaluatorId()', self.blind_js)
        self.assertIn('typeof root.crypto.randomUUID === "function"', self.shared_js)
        self.assertIn("Math.random()", self.shared_js)

    def test_blind_page_survives_restricted_storage_and_surfaces_startup_errors(self) -> None:
        self.assertIn("function storageGet(key, fallbackValue)", self.shared_js)
        self.assertIn("function storageSet(key, value)", self.shared_js)
        self.assertIn("function storageRemove(key)", self.shared_js)
        self.assertIn("catch (_error)", self.shared_js)
        self.assertIn("function initializeBlindPage()", self.blind_js)
        self.assertIn("initializeBlindPage();", self.blind_js)
        self.assertRegex(
            self.blind_js,
            r"try \{\s+initializeBlindPage\(\);\s+window\.__vfievalBlindReady = true;\s+\} catch \(error\) \{\s+showError\(error\);",
        )

    def test_blind_page_avoids_newer_syntax_and_reports_script_boot_failures(self) -> None:
        self.assertNotIn("?.", self.blind_js)
        self.assertNotIn("??", self.blind_js)
        self.assertNotIn("?.", self.shared_js)
        self.assertNotIn("??", self.shared_js)
        self.assertNotIn("Promise.allSettled", self.blind_js)
        self.assertIn("function replaceContent(element, ...nodes)", self.blind_js)
        self.assertIn("window.__vfievalBlindReady = true;", self.blind_js)
        self.assertIn("盲评页面脚本未能加载", self.blind_html)
        self.assertIn("无法加载 /shared.js", self.blind_html)
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
        self.assertIn('Shared.storageSet("vfieval-evaluator-name", displayName);', handler)

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

    def test_studio_campaign_preparation_poll_is_serialized_and_cancellable(self) -> None:
        start_poll = self._studio_function_source("startPreparationPoll")
        stop_poll = self._studio_function_source("stopPreparationPoll")

        self.assertIn("setTimeout(pollOnce, delay)", start_poll)
        self.assertIn("PREPARATION_POLL_BASE_MS", start_poll)
        self.assertIn("PREPARATION_POLL_MAX_MS", start_poll)
        self.assertNotIn("setInterval", start_poll)
        self.assertIn("const pollOnce = async () =>", start_poll)
        self.assertIn("await request(`/api/evaluation-campaigns/v2/", start_poll)
        self.assertIn("scheduleNext();", start_poll)
        self.assertIn("preparationPollGeneration", start_poll)
        self.assertIn("stillCurrent()", start_poll)
        self.assertIn("clearTimeout(studioState.preparationPoll)", stop_poll)
        self.assertIn("studioState.preparationPollGeneration += 1", stop_poll)
        self.assertIn('window.addEventListener("pagehide", () => {', self.studio_js)
        self.assertIn("stopPreparationPoll();", self.studio_js)
        self.assertIn('window.addEventListener("pageshow", (event) => {', self.studio_js)
        self.assertIn("if (!event.persisted || !studioState.selectedCampaignKey) return", self.studio_js)
        self.assertIn("openCampaign(studioState.selectedCampaignKey, false)", self.studio_js)

    def test_studio_publish_reconciles_only_lost_responses_and_adopts_created_campaign(self) -> None:
        request_source = self._studio_function_source("request")
        publish_request = self._studio_function_source("requestCampaignPublish")
        committed = self._studio_function_source("campaignPublishCommitted")
        create = self._studio_function_source("createCampaign")

        self.assertIn("Shared.request(path", request_source)
        self.assertIn("if (publishError.status != null) throw publishError", publish_request)
        self.assertIn("await readCampaignTruth(campaignId)", publish_request)
        self.assertIn("campaignPublishCommitted(campaign)", publish_request)
        for status in ("preparing", "published", "completed", "failed"):
            self.assertIn(f'"{status}"', committed)
        publish = self._studio_function_source("publishCampaign")
        poll_start = publish.index("startPreparationPoll(key, campaignId)")
        best_effort_refresh = publish.index(
            "loadCampaigns({ preserveMissingKey: key })"
        )
        self.assertLess(poll_start, best_effort_refresh)
        self.assertIn("loadPackages({ page: 1 })", publish)
        self.assertIn("The publish result is authoritative", publish)
        selection = 'studioState.selectedCampaignKey = `v2:${campaignId}`'
        self.assertIn(selection, create)
        self.assertLess(create.index(selection), create.index("await publishCampaign(campaignId)"))

    def test_studio_delete_clears_then_reconciles_campaign_state(self) -> None:
        read_truth = self._studio_function_source("readCampaignTruth")
        delete_campaign = self._studio_function_source("deleteCampaign")

        self.assertIn("Number(error.status) === 404", read_truth)
        delete_request = delete_campaign.index('method: "DELETE"')
        self.assertLess(delete_campaign.index("stopPreparationPoll();"), delete_request)
        self.assertLess(delete_campaign.index("studioState.selectedCampaignKey = null;"), delete_request)
        self.assertIn("truth = await readCampaignTruth(campaignId)", delete_campaign)
        self.assertIn("if (truth.exists)", delete_campaign)
        self.assertIn("studioState.selectedCampaignKey = deletingKey", delete_campaign)
        self.assertIn("startPreparationPoll(deletingKey, campaignId)", delete_campaign)
        self.assertIn("response_recovered: true", delete_campaign)
        local_removal = delete_campaign.index(
            "studioState.campaigns = studioState.campaigns.filter((row) => campaignKey(row) !== deletingKey)",
        )
        confirmed_refresh = delete_campaign.index(
            "await Promise.all([loadCampaigns(), loadPackages()])",
            local_removal,
        )
        self.assertLess(local_removal, confirmed_refresh)
        self.assertIn("DELETE response or reconciliation GET already confirmed deletion", delete_campaign)
        self.assertLess(confirmed_refresh, delete_campaign.index("notify(result.cleanup_pending"))

    def test_studio_surfaces_persistent_campaign_cleanup_and_retry(self) -> None:
        load_requests = self._studio_function_source("loadCleanupRequests")
        retry_request = self._studio_function_source("retryCleanupRequest")
        remember_request = self._studio_function_source("rememberCleanupRequest")
        delete_campaign = self._studio_function_source("deleteCampaign")

        self.assertIn('request("/api/evaluation-cleanup-requests")', load_requests)
        self.assertIn("payload.requests || []", load_requests)
        self.assertIn("cleanup_request_id", remember_request)
        self.assertIn("cleanup_status", remember_request)
        self.assertIn("rememberCleanupRequest(result, campaignId)", delete_campaign)
        self.assertIn("await loadCleanupRequests()", delete_campaign)
        self.assertIn("data-evaluation-cleanup-panel", self.studio_js)
        self.assertIn("data-evaluation-cleanup-retry", self.studio_js)
        self.assertIn("/api/evaluation-cleanup-requests/${Number(requestId)}/retry", retry_request)
        self.assertIn('method: "POST"', retry_request)
        self.assertIn("loadCleanupRequests()", self._studio_function_source("load"))

    def test_studio_renders_and_copies_an_absolute_participant_link(self) -> None:
        self.assertIn("function participantShareUrl(shareUrl, campaign)", self.studio_js)
        availability = self._studio_function_source("campaignParticipantAvailable")
        self.assertIn('["published", "closed", "archived"]', availability)
        self.assertIn("new URL(rawUrl, location.origin).href", self.studio_js)
        self.assertIn("const shareUrl = campaignParticipantAvailable(campaign)", self.studio_js)
        self.assertIn("? participantShareUrl(payload.share_url || campaign.share_url, campaign)", self.studio_js)
        self.assertIn('value="${safe(shareUrl)}"', self.studio_js)
        self.assertIn('data-copy-share="${safe(shareUrl)}"', self.studio_js)
        self.assertIn("Shared.copyText(copy.dataset.copyShare)", self.studio_js)
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
        self.assertIn("外部 Pred 必须先显式绑定", self.index_html)
        self.assertNotIn("reference_asset_id", self.studio_js)
        self.assertNotIn("video_name: videoName", self.studio_js)
        self.assertNotIn("storage_path", self.studio_js)

    def test_campaign_item_picker_uses_one_server_page_and_preserves_selection(self) -> None:
        loader = self.studio_js.split("async function loadItems", 1)[1].split(
            "function methodOptions", 1
        )[0]
        render = self.studio_js.split("function renderItems", 1)[1].split(
            "async function loadItemGroups", 1
        )[0]
        pager = self.studio_js.split("function itemPagerMarkup", 1)[1].split(
            "function renderItems", 1
        )[0]

        self.assertIn("requestedPage", loader)
        self.assertIn("studioState.itemPage", loader)
        self.assertIn("payload.items || []", loader)
        self.assertNotIn("Promise.all", loader)
        self.assertNotIn("selectedItemIds.delete", loader)
        self.assertIn("data-studio-item-page", pager)
        self.assertIn("翻页和搜索不会取消已选视频", render)

    def test_select_all_filtered_uses_a_persisted_server_selection_token(self) -> None:
        select_all = self._studio_function_source("selectAllFilteredItems")
        load_methods = self._studio_function_source("loadMethodsForSelection")
        draft = self._studio_function_source("campaignDraftPayload")

        self.assertIn('request("/api/media/item-selections"', select_all)
        self.assertIn("payload.selection_token", select_all)
        self.assertNotIn("for (let page =", select_all)
        self.assertIn("selection_token=", load_methods)
        self.assertIn("selection_token:", draft)
        self.assertIn("selection_expires_at:", draft)

    def test_coverage_matrix_surfaces_spatial_alignment_without_weakening_time(self) -> None:
        self.assertIn("时间映射严格验证", self.studio_js)
        self.assertIn("alignment_plan", self.studio_js)
        self.assertIn("target_width", self.studio_js)
        self.assertIn("resize_kind", self.studio_js)
        self.assertIn("smallest_pred", self.studio_js)
        self.assertIn("lanczos", self.studio_js)

    def test_compare_prefill_and_external_policy_follow_the_item_contract(self) -> None:
        app_js = self.compare_js
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
