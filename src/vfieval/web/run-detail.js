"use strict";

// Domain functions intentionally share app.js's classic-script global
// environment so state, request primitives, and caches remain singletons.

function currentVideoTimeline() {
  if (!state.selectedRun) return null;
  const videoName = state.selectedVideoByRun[state.selectedRun.id];
  return videoName ? state.runVideoTimelines[`${state.selectedRun.id}:${videoName}`] : null;
}

// Multi-track compare stores one sample per (track, frame): the same frame
// index appears once per pred track, all sharing GT. Group siblings so the
// frame viewer can show GT + every pred track side by side in one row.
function compareFrameSiblings(video, sample) {
  if (!video || !sample) return sample ? [sample] : [];
  const frame = Number(sample.frame_index);
  const siblings = (video.samples || []).filter((row) => Number(row.frame_index) === frame);
  if (!siblings.length) return [sample];
  return siblings.sort((left, right) =>
    Number(left.track_index ?? 0) - Number(right.track_index ?? 0)
    || String(left.track_label || "").localeCompare(String(right.track_label || "")),
  );
}

// Index of the first sample belonging to the next/previous distinct frame,
// so compare stepping advances by frame rather than by track.
function compareStepIndex(video, currentIndex, direction) {
  const samples = video.samples || [];
  const current = samples[currentIndex];
  if (!current) return currentIndex;
  const currentFrame = Number(current.frame_index);
  if (direction > 0) {
    for (let index = currentIndex + 1; index < samples.length; index += 1) {
      if (Number(samples[index].frame_index) !== currentFrame) return index;
    }
    return currentIndex;
  }
  let target = currentIndex;
  for (let index = currentIndex - 1; index >= 0; index -= 1) {
    if (Number(samples[index].frame_index) !== currentFrame) {
      const prevFrame = Number(samples[index].frame_index);
      // Walk back to the first sample of that previous frame.
      target = index;
      for (let back = index - 1; back >= 0 && Number(samples[back].frame_index) === prevFrame; back -= 1) {
        target = back;
      }
      return target;
    }
  }
  return currentIndex;
}

function statusBadge(status) {
  return `<span class="status ${escapeHtml(status)}">${escapeHtml(STATUS_LABELS[status] || status)}</span>`;
}

function runPurgeState(run) {
  return String(run?.purge_request?.status || "");
}

function runPurgeDeletesRecord(run) {
  return String(run?.purge_request?.request_type || "") === "delete_run";
}

function runStatusDisplay(run) {
  const purge = runPurgeState(run);
  if (["requested", "canceling", "purging"].includes(purge)) {
    return `${statusBadge(run.status)} <span class="status deleting">${runPurgeDeletesRecord(run) ? "正在删除" : "正在清理产物"}</span>`;
  }
  if (purge === "failed") {
    return `${statusBadge(run.status)} <span class="status failed">清理失败</span>`;
  }
  return statusBadge(run.status);
}

function runContentRevision(run) {
  if (run?.content_revision === undefined || run?.content_revision === null) return null;
  const revision = Number(run.content_revision);
  return Number.isFinite(revision) ? revision : null;
}

function currentRunResultGeneration(runId) {
  return Number(state.runResultGenerations[Number(runId)] || 0);
}

function abortSampleRequestsForRun(runId) {
  const prefix = `${Number(runId)}:`;
  for (const [key, controller] of Object.entries(state.sampleAbortControllers)) {
    if (!key.startsWith(prefix)) continue;
    controller.abort();
    delete state.sampleAbortControllers[key];
    delete state.sampleDetailLoading[key];
  }
}

function abortTimelineRequest(runId = null) {
  if (!state.timelineAbortController) return;
  if (runId !== null && Number(state.timelineAbortRunId) !== Number(runId)) return;
  state.timelineAbortController.abort();
  state.timelineAbortController = null;
  state.timelineAbortRunId = null;
  state.timelineRequestGeneration += 1;
}

function abortRunDetailRequests(runId) {
  abortSampleRequestsForRun(runId);
  abortTimelineRequest(runId);
}

function clearRunScopedEntries(target, runId) {
  const prefix = `${Number(runId)}:`;
  for (const key of Object.keys(target)) {
    if (key.startsWith(prefix)) delete target[key];
  }
}

function invalidateRunResultCache(runId, options = {}) {
  const id = Number(runId);
  state.runResultGenerations[id] = currentRunResultGeneration(id) + 1;
  abortRunDetailRequests(id);
  clearRunScopedEntries(state.runVideoTimelines, id);
  clearRunScopedEntries(state.sampleDetails, id);
  clearRunScopedEntries(state.sampleDetailLoading, id);
  delete state.compareInputsByRun[id];
  if (Number(state.selectedRun?.id) === id) {
    state.runVideosPage = null;
    if (options.render !== false) renderRunDetail();
  }
}

function runContentRevisionChanged(nextRun) {
  const nextRevision = runContentRevision(nextRun);
  const observedRevision = state.runContentRevisions[Number(nextRun?.id)];
  return nextRevision !== null
    && observedRevision !== undefined
    && Number(observedRevision) !== nextRevision;
}

function shouldRefreshSelectedRun(nextRun) {
  if (!state.selectedRun || !nextRun) return false;
  if (!TERMINAL_STATUSES.has(nextRun.status)) return true;
  return runContentRevisionChanged(nextRun)
    || Number(nextRun.updated_at || 0) !== Number(state.selectedRun.updated_at || 0)
    || String(nextRun.status || "") !== String(state.selectedRun.status || "")
    || Number(nextRun.progress_current || 0) !== Number(state.selectedRun.progress_current || 0)
    || Number(nextRun.progress_total || 0) !== Number(state.selectedRun.progress_total || 0);
}

function mergeSelectedRunSummary(nextRun) {
  if (!state.selectedRun || Number(state.selectedRun.id) !== Number(nextRun?.id)) return;
  state.selectedRun = {
    ...state.selectedRun,
    ...nextRun,
    metadata: {
      ...(state.selectedRun.metadata || {}),
      ...(nextRun.metadata || {}),
    },
  };
}

function patchSelectedRunLiveDom() {
  const run = state.selectedRun;
  if (!run) return;
  const root = $("run-detail");
  const status = root?.querySelector("[data-run-live-status]");
  const progress = root?.querySelector("[data-run-live-progress]");
  const inference = root?.querySelector("[data-run-live-inference]");
  const metric = root?.querySelector("[data-run-live-metric]");
  const cancel = root?.querySelector("[data-run-live-cancel]");
  if (status) status.innerHTML = runStatusDisplay(run);
  if (progress) progress.textContent = `${run.progress_current || 0}/${run.progress_total || 0}`;
  if (inference) inference.textContent = renderInferencePhase(run);
  if (metric) metric.textContent = renderMetricPhase(run);
  if (cancel) cancel.disabled = TERMINAL_STATUSES.has(run.status);
}

function captureRunDetailUiState() {
  const root = $("run-detail");
  if (!root) return null;
  const interactive = Array.from(root.querySelectorAll("input, textarea, select, button, a"));
  const activeIndex = root.contains(document.activeElement) ? interactive.indexOf(document.activeElement) : -1;
  const forms = Array.from(root.querySelectorAll("[data-feedback-form], [data-feedback-edit-form]")).map((form) => ({
    key: form.dataset.feedbackForm
      ? `feedback:${form.dataset.feedbackForm}`
      : `edit:${form.dataset.feedbackEditForm}`,
    fields: Array.from(form.elements).filter((element) => element.name).map((element) => ({
      name: element.name,
      value: element.value,
      checked: Boolean(element.checked),
      selectionStart: typeof element.selectionStart === "number" ? element.selectionStart : null,
      selectionEnd: typeof element.selectionEnd === "number" ? element.selectionEnd : null,
    })),
  }));
  const videos = Array.from(root.querySelectorAll("video")).map((video, index) => ({
    index,
    src: video.getAttribute("src") || "",
    currentTime: Number(video.currentTime || 0),
    paused: video.paused,
    muted: video.muted,
    volume: video.volume,
    playbackRate: video.playbackRate,
  }));
  return {
    activeIndex,
    forms,
    videos,
    windowX: window.scrollX,
    windowY: window.scrollY,
    rootScrollTop: root.scrollTop,
  };
}

function restoreRunDetailUiState(snapshot) {
  if (!snapshot) return;
  const root = $("run-detail");
  if (!root) return;
  for (const saved of snapshot.forms || []) {
    const selector = saved.key.startsWith("feedback:")
      ? `[data-feedback-form="${CSS.escape(saved.key.slice(9))}"]`
      : `[data-feedback-edit-form="${CSS.escape(saved.key.slice(5))}"]`;
    const form = root.querySelector(selector);
    if (!form) continue;
    for (const field of saved.fields || []) {
      const element = Array.from(form.elements).find((candidate) => candidate.name === field.name);
      if (!element) continue;
      if (["checkbox", "radio"].includes(element.type)) element.checked = field.checked;
      else element.value = field.value;
    }
  }
  const nextVideos = Array.from(root.querySelectorAll("video"));
  for (const saved of snapshot.videos || []) {
    const video = nextVideos.find((candidate, index) =>
      (candidate.getAttribute("src") || "") === saved.src || index === saved.index);
    if (!video) continue;
    video.muted = saved.muted;
    video.volume = saved.volume;
    video.playbackRate = saved.playbackRate;
    const restorePlayback = () => {
      if (Number.isFinite(saved.currentTime)) {
        try { video.currentTime = saved.currentTime; } catch (_error) {}
      }
      if (!saved.paused) video.play().catch(() => {});
    };
    if (video.readyState >= 1) restorePlayback();
    else video.addEventListener("loadedmetadata", restorePlayback, { once: true });
  }
  requestAnimationFrame(() => {
    root.scrollTop = snapshot.rootScrollTop;
    window.scrollTo(snapshot.windowX, snapshot.windowY);
    if (snapshot.activeIndex >= 0) {
      const interactive = Array.from(root.querySelectorAll("input, textarea, select, button, a"));
      const active = interactive[snapshot.activeIndex];
      active?.focus({ preventScroll: true });
      const savedForms = (snapshot.forms || []).flatMap((form) => form.fields || []);
      const savedField = active?.name ? savedForms.find((field) => field.name === active.name) : null;
      if (savedField && savedField.selectionStart !== null && typeof active.setSelectionRange === "function") {
        active.setSelectionRange(savedField.selectionStart, savedField.selectionEnd);
      }
    }
  });
}

async function pollSelectedRunById(runId, options = {}) {
  const id = Number(runId);
  const previous = state.selectedRun;
  if (!previous || Number(previous.id) !== id) return;
  const nextRun = await api(`/api/runs/${id}`);
  if (!state.selectedRun || Number(state.selectedRun.id) !== id) return;
  const becameTerminal = !TERMINAL_STATUSES.has(previous.status) && TERMINAL_STATUSES.has(nextRun.status);
  const cleanedChanged = String(nextRun.artifact_cleaned_at || "") !== String(previous.artifact_cleaned_at || "");
  const invalidateResults = Boolean(options.forceSelected)
    || runContentRevisionChanged(nextRun)
    || becameTerminal
    || cleanedChanged;
  if (invalidateResults) {
    await selectRun(id, { quiet: true, invalidateResults: true });
    return;
  }
  mergeSelectedRunSummary(nextRun);
  patchSelectedRunLiveDom();
}

function runListPath(options = {}) {
  const page = Math.max(1, Number(options.page || state.runsPage.page || 1));
  const pageSize = Math.max(1, Math.min(200, Number(state.runsPage.page_size || 30)));
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
  for (const [key, value] of Object.entries(state.runFilters || {})) {
    const normalized = String(value || "").trim();
    if (normalized) params.set(key, normalized);
  }
  return `/api/runs?${params.toString()}`;
}

function applyRunListPayload(payload) {
  if (Array.isArray(payload)) {
    state.runs = payload;
    state.runsPage = {
      ...state.runsPage,
      page: 1,
      page_count: 1,
      total: payload.length,
      active_total: payload.filter((run) =>
        !TERMINAL_STATUSES.has(run.status)
        || ["requested", "canceling", "purging"].includes(runPurgeState(run))).length,
    };
    return;
  }
  state.runs = Array.isArray(payload?.runs) ? payload.runs : [];
  state.runsPage = {
    ...state.runsPage,
    page: Math.max(1, Number(payload?.page || 1)),
    page_size: Math.max(1, Number(payload?.page_size || state.runsPage.page_size || 30)),
    page_count: Math.max(1, Number(payload?.page_count || 1)),
    total: Math.max(0, Number(payload?.total || 0)),
    active_total: Math.max(0, Number(payload?.active_total || 0)),
  };
}

async function requestRunsPage(options = {}) {
  return api(runListPath(options));
}

async function refreshRunsOnce(options = {}) {
  const requestGeneration = ++state.runsRefreshGeneration;
  let payload = await requestRunsPage(options);
  if (requestGeneration !== state.runsRefreshGeneration) return;
  const rows = Array.isArray(payload) ? payload : (payload?.runs || []);
  const page = Number(Array.isArray(payload) ? 1 : payload?.page || options.page || state.runsPage.page || 1);
  if (!rows.length && page > 1) {
    payload = await requestRunsPage({ page: page - 1 });
    if (requestGeneration !== state.runsRefreshGeneration) return;
  }
  applyRunListPayload(payload);
  renderRuns();
  if (!isRunsViewActive()) {
    return;
  }
  if (state.selectedRun) {
    const nextRun = state.runs.find((item) => Number(item.id) === Number(state.selectedRun.id));
    if (!nextRun) {
      // Paging and filters only change the history slice. Keep an already
      // selected detail visible, and keep polling it by ID while it is active.
      const selectedNeedsRefresh = !!options.forceSelected
        || !TERMINAL_STATUSES.has(state.selectedRun.status)
        || ["requested", "canceling", "purging"].includes(runPurgeState(state.selectedRun));
      if (selectedNeedsRefresh) {
        try {
          await pollSelectedRunById(state.selectedRun.id, {
            forceSelected: !!options.forceSelected,
          });
        } catch (error) {
          if (Number(error.status || 0) !== 404) throw error;
          abortRunDetailRequests(state.selectedRun.id);
          state.selectedRun = null;
          state.runVideosPage = null;
          renderEmptyRunDetail();
        }
      }
    } else {
      const becameTerminal = !TERMINAL_STATUSES.has(state.selectedRun.status)
        && TERMINAL_STATUSES.has(nextRun.status);
      const cleanedChanged = String(nextRun.artifact_cleaned_at || "")
        !== String(state.selectedRun.artifact_cleaned_at || "");
      const invalidateResults = !!options.forceSelected
        || runContentRevisionChanged(nextRun)
        || becameTerminal
        || cleanedChanged;
      if (invalidateResults) {
        await selectRun(state.selectedRun.id, { quiet: true, invalidateResults });
      } else if (shouldRefreshSelectedRun(nextRun)) {
        mergeSelectedRunSummary(nextRun);
        patchSelectedRunLiveDom();
      }
    }
  }
  if (!state.selectedRun && state.runs.length) {
    await selectRun(state.runs[0].id, { quiet: true });
    return;
  }
  if (!state.selectedRun) {
    renderEmptyRunDetail();
  }
}

function refreshRunsOnly(options = {}) {
  if (options.forceSelected && state.selectedRun) {
    invalidateRunResultCache(state.selectedRun.id);
  }
  state.runsRefreshQueued = true;
  state.runsRefreshPendingForce = state.runsRefreshPendingForce || !!options.forceSelected;
  if (options.page !== undefined && options.page !== null) {
    state.runsRefreshPendingPage = Math.max(1, Number(options.page || 1));
  }
  if (state.runsRefreshPromise) return state.runsRefreshPromise;

  const refreshLoop = async () => {
    while (state.runsRefreshQueued) {
      const forceSelected = state.runsRefreshPendingForce;
      const page = state.runsRefreshPendingPage;
      state.runsRefreshQueued = false;
      state.runsRefreshPendingForce = false;
      state.runsRefreshPendingPage = null;
      await refreshRunsOnce({ forceSelected, page });
    }
  };
  const promise = refreshLoop().finally(() => {
    if (state.runsRefreshPromise === promise) state.runsRefreshPromise = null;
  });
  state.runsRefreshPromise = promise;
  return promise;
}

function scheduleRunFilterRefresh(delay = 300) {
  clearTimeout(state.runFilterTimer);
  state.runFilterTimer = setTimeout(() => {
    state.runFilterTimer = null;
    refreshRunsOnly({ page: 1 }).catch((error) => toast(error.message));
  }, delay);
}

function runSourceLabel(run) {
  if ((run.metadata?.run_type || "model_inference") === "video_compare") {
    const requestReference = run.metadata?.request?.reference || {};
    const reference = requestReference.group && requestReference.video
      ? `${requestReference.group}/${requestReference.video}`
      : pathBasename(run.metadata?.reference_path ?? "-");
    const tracks = (run.metadata?.distorted_tracks || [])
      .map((track) => track.track_label || track.label || track.video_name || pathBasename(track.distorted_path ?? ""))
      .filter(Boolean);
    return `${reference} -> ${tracks.join(", ") || "-"}`;
  }
  return `${run.metadata?.model_file || run.model_name || "-"} / ${run.metadata?.video_group || run.dataset_name || "-"}`;
}

function runTypeLabel(value) {
  return {
    model_inference: "模型推理",
    video_compare: "视频对比",
  }[String(value || "")] || String(value || "-");
}

function pathBasename(value) {
  const text = String(value || "");
  return text.split(/[\\/]/).filter(Boolean).pop() || text || "-";
}

function renderRuns() {
  const activeFilter = document.activeElement?.dataset?.runFilter || "";
  const activeSelection = activeFilter && typeof document.activeElement.selectionStart === "number"
    ? [document.activeElement.selectionStart, document.activeElement.selectionEnd]
    : null;
  // Drop selections for runs that are no longer in the list (e.g. refreshed away).
  const liveIds = new Set(state.runs.map((run) => Number(run.id)));
  for (const id of Array.from(state.selectedRunIds)) {
    if (!liveIds.has(Number(id))) state.selectedRunIds.delete(id);
  }
  const allSelected = state.runs.length > 0 && state.runs.every((run) => state.selectedRunIds.has(Number(run.id)));
  const selectedCount = state.selectedRunIds.size;
  const statusOptions = [
    ["", "全部状态"],
    ["decoding", "解码中"],
    ["queued", "排队中"],
    ["running", "推理中"],
    ["completed", "已完成"],
    ["failed", "失败"],
    ["canceled", "已取消"],
  ];
  const typeOptions = [["", "全部类型"], ["model_inference", "模型推理"], ["video_compare", "视频对比"]];
  const filterOptionHtml = (rows, current) => rows.map(([value, label]) =>
    `<option value="${escapeHtml(value)}" ${String(current || "") === value ? "selected" : ""}>${escapeHtml(label)}</option>`).join("");
  const toolbar = `
    <div class="runs-toolbar">
      <label class="runs-select-all">
        <input type="checkbox" data-runs-select-all ${allSelected ? "checked" : ""}>
        <span>全选本页</span>
      </label>
      <button class="secondary danger" data-runs-batch-delete type="button" ${selectedCount ? "" : "disabled"}>批量删除${selectedCount ? ` (${selectedCount})` : ""}</button>
      <label>
        <span class="muted">搜索</span>
        <input data-run-filter="q" value="${escapeHtml(state.runFilters.q)}" placeholder="名称、Run ID 或来源">
      </label>
      <label>
        <span class="muted">状态</span>
        <select data-run-filter="status">${filterOptionHtml(statusOptions, state.runFilters.status)}</select>
      </label>
      <label>
        <span class="muted">类型</span>
        <select data-run-filter="run_type">${filterOptionHtml(typeOptions, state.runFilters.run_type)}</select>
      </label>
      <label>
        <span class="muted">模型</span>
        <input data-run-filter="model" value="${escapeHtml(state.runFilters.model)}" placeholder="模型文件">
      </label>
      <button class="secondary" data-runs-filter-reset type="button">清除筛选</button>
    </div>
  `;
  const rows = table(state.runs, [
    {
      label: "",
      render: (run) => `<input type="checkbox" data-run-select="${run.id}" ${state.selectedRunIds.has(Number(run.id)) ? "checked" : ""}>`,
    },
    { label: "Run", render: (run) => `#${escapeHtml(run.id)}` },
    { label: "名称", render: (run) => escapeHtml(run.name || "-") },
    { label: "状态", render: (run) => runStatusDisplay(run) },
    { label: "类型", render: (run) => escapeHtml(runTypeLabel(run.metadata?.run_type || "model_inference")) },
    { label: "来源", render: (run) => escapeHtml(runSourceLabel(run)) },
    { label: "进度", render: (run) => renderRunProgress(run) },
    { label: "操作", render: (run) => `<button class="view-detail-btn" data-run-id="${run.id}" type="button">查看详情 →</button>` },
  ], { rowAttrs: (run) => `data-run-id="${run.id}" class="clickable-row"` });
  const pager = `
    <div class="pager runs-pager">
      <button class="secondary" data-runs-page="${state.runsPage.page - 1}" ${state.runsPage.page <= 1 ? "disabled" : ""} type="button">上一页</button>
      <span class="muted">第 ${escapeHtml(state.runsPage.page)} / ${escapeHtml(state.runsPage.page_count)} 页，共 ${escapeHtml(state.runsPage.total)} 条</span>
      <button class="secondary" data-runs-page="${state.runsPage.page + 1}" ${state.runsPage.page >= state.runsPage.page_count ? "disabled" : ""} type="button">下一页</button>
    </div>
  `;
  $("runs-table").innerHTML = toolbar + rows + pager;
  if (activeFilter) {
    const nextActive = $("runs-table").querySelector(`[data-run-filter="${activeFilter}"]`);
    nextActive?.focus();
    if (activeSelection && typeof nextActive?.setSelectionRange === "function") {
      nextActive.setSelectionRange(activeSelection[0], activeSelection[1]);
    }
  }
}

async function loadRunVideosPage(runId, page = 1) {
  const selectionGeneration = state.runSelectionGeneration;
  const resultGeneration = currentRunResultGeneration(runId);
  const payload = await api(`/api/runs/${runId}/videos?page=${page}&page_size=20`);
  if (selectionGeneration !== state.runSelectionGeneration
      || resultGeneration !== currentRunResultGeneration(runId)
      || Number(state.selectedRun?.id) !== Number(runId)) return null;
  state.runVideosPage = payload;
  state.runVideoPageByRun[runId] = Number(state.runVideosPage?.page || page || 1);
  const videos = state.runVideosPage?.videos || [];
  if (!videos.length) {
    return payload;
  }
  const selectedVideoName = state.selectedVideoByRun[runId];
  const selectedIsVisible = videos.some((item) => item.video_name === selectedVideoName);
  const nextVideoName = selectedIsVisible ? selectedVideoName : videos[0].video_name;
  state.selectedVideoByRun[runId] = nextVideoName;
  if (nextVideoName && !state.runVideoTimelines[`${runId}:${nextVideoName}`]) {
    await loadRunVideoTimeline(runId, nextVideoName);
  }
  return payload;
}

async function selectRun(runId, options = {}) {
  const id = Number(runId);
  const selectionChanged = Number(state.selectedRun?.id) !== id;
  const preservedUi = !selectionChanged && options.quiet && options.preserveUi !== false
    ? captureRunDetailUiState()
    : null;
  if (selectionChanged) {
    state.runSelectionGeneration += 1;
    if (state.selectedRun) abortRunDetailRequests(state.selectedRun.id);
  }
  const selectionGeneration = state.runSelectionGeneration;
  const selectRequestGeneration = ++state.runSelectRequestGeneration;
  if (state.runSelectAbortController) state.runSelectAbortController.abort();
  const controller = new AbortController();
  state.runSelectAbortController = controller;
  try {
    const [run, metricSummary] = await Promise.all([
      api(`/api/runs/${id}`, { signal: controller.signal }),
      api(`/api/runs/${id}/metric-summary`, { signal: controller.signal }),
    ]);
    if (selectRequestGeneration !== state.runSelectRequestGeneration
        || selectionGeneration !== state.runSelectionGeneration) return null;

    const previousRunId = state.selectedRun?.id;
    if (previousRunId !== undefined && Number(previousRunId) !== id) {
      abortRunDetailRequests(previousRunId);
      state.runVideosPage = null;
    }
    state.selectedRun = run;
    state.metricSummary = metricSummary;

    const revision = runContentRevision(run);
    const observedRevision = state.runContentRevisions[id];
    const revisionChanged = revision !== null
      && observedRevision !== undefined
      && Number(observedRevision) !== revision;
    if (options.invalidateResults || revisionChanged) {
      invalidateRunResultCache(id, { render: false });
    }
    if (revision !== null) state.runContentRevisions[id] = revision;

    // On an explicit selection, show the preserved video/frame selection as a
    // skeleton. During background refresh, keep the existing DOM alive until
    // fresh scoped data is ready so playback, focus and feedback drafts survive.
    if (!preservedUi) renderRunDetail();
    if (isCompareRun(run)) {
      state.compareInputsByRun[id] = await api(`/api/runs/${id}/compare-inputs`, { signal: controller.signal })
        .catch((error) => ({ error: error.message }));
      if (selectRequestGeneration !== state.runSelectRequestGeneration
          || selectionGeneration !== state.runSelectionGeneration) return null;
    } else {
      delete state.compareInputsByRun[id];
    }
    await loadRunVideosPage(id, state.runVideoPageByRun[id] || 1);
    if (selectRequestGeneration !== state.runSelectRequestGeneration
        || selectionGeneration !== state.runSelectionGeneration) return null;
    renderRunDetail();
    restoreRunDetailUiState(preservedUi);
    if (!options.quiet) switchView("runs");
    syncBrowserRoute({ view: "runs", replace: Boolean(options.quiet) });
    return run;
  } catch (error) {
    if (error.name === "AbortError") return null;
    throw error;
  } finally {
    if (state.runSelectAbortController === controller) {
      state.runSelectAbortController = null;
    }
  }
}

function renderInferencePhase(run) {
  const runType = run.metadata?.run_type || "model_inference";
  if (run.status === "decoding") return "正在解码视频帧";
  if (run.status === "queued") return runType === "video_compare" ? "等待生成对比产物" : "等待推理";
  if (run.status === "running") return runType === "video_compare" ? "生成对比产物中" : "推理中";
  if (run.status === "failed" && run.error?.phase === "decode") return "解码失败";
  if (run.status === "failed") return "失败";
  if (run.status === "canceled" || run.status === "cancel_requested") return "已取消";
  return run.inference_job_id ? "已完成" : "未开始";
}

function renderMetricPhase(run) {
  const metrics = run.metrics || [];
  if (!metrics.length) return "未选择指标";
  if (run.status === "decoding") return "等待解码完成";
  if (run.status === "metric_queued") return "评测排队";
  if (run.status === "metric_running") return "评测中";
  if (run.metric_job_id) return "已完成";
  return "等待前一阶段完成";
}

function renderRunError(run) {
  if (!run.error || !Object.keys(run.error).length) return "";
  const cloneAction = isFileInputRun(run)
    ? `<button class="secondary" data-clone-run="${escapeHtml(run.id)}" type="button">Clone with current inputs</button>`
    : "";
  const parts = [];
  if (run.error.device) parts.push(`Device ${run.error.device}`);
  if (run.error.shard_index !== undefined && run.error.shard_index !== null) {
    const suffix = run.error.shard_count ? ` of ${run.error.shard_count}` : "";
    parts.push(`Shard #${run.error.shard_index}${suffix}`);
  }
  if (run.error.worker_id) parts.push(`Worker ${run.error.worker_id}`);
  if (run.error.job_id) parts.push(`Job ${run.error.job_id}`);
  if (parts.length) {
    const meta = `<p class="muted">${escapeHtml(parts.join(" | "))}</p>`;
    return `<div class="message error run-error-banner"><p><strong>${escapeHtml(run.error.type || "Error")}</strong>: ${escapeHtml(run.error.message || JSON.stringify(run.error))}</p>${meta}<button class="secondary" data-retry-run="${escapeHtml(run.id)}" type="button">Retry this Run</button>${cloneAction}</div>`;
  }
  return `<div class="message error run-error-banner"><p><strong>${escapeHtml(run.error.type || "Error")}</strong>: ${escapeHtml(run.error.message || JSON.stringify(run.error))}</p><button class="secondary" data-retry-run="${escapeHtml(run.id)}" type="button">Retry this Run</button>${cloneAction}</div>`;
}

function runExecutionTarget(run) {
  const executionMode = run.metadata?.execution_mode || run.device || "-";
  const devices = run.metadata?.devices || [];
  if (!devices.length) return escapeHtml(executionMode);
  return `${escapeHtml(executionMode)} / ${escapeHtml(devices.join(", "))}`;
}

function renderJobProgress(job) {
  const current = Number(job.progress_current || 0);
  const total = Number(job.progress_total || 0);
  if (total > 0) return `${escapeHtml(current)}/${escapeHtml(total)}`;
  if (current > 0) return escapeHtml(current);
  if (job.status === "running") return "启动中…";
  return "-";
}

function renderRunProgress(run) {
  const current = Number(run.progress_current || 0);
  const total = Number(run.progress_total || 0);
  const devices = run.metadata?.devices || [];
  if (run.status === "running" && devices.length > 1) {
    const progressText = total > 0 ? `${current}/${total}` : "启动中…";
    return `<span title="${escapeHtml(devices.join(", "))}">${escapeHtml(String(devices.length))}设备 · ${escapeHtml(String(progressText))}</span>`;
  }
  if (run.status === "running" && total === 0) return "启动中…";
  return `${escapeHtml(String(current))}/${escapeHtml(String(total))}`;
}

function renderJobLabel(job) {
  if (job.role === "decode" || job.kind === "decode") {
    return "decode";
  }
  if (job.role === "inference") {
    return `inference #${escapeHtml(job.shard_index ?? 0)}`;
  }
  return escapeHtml(job.role || job.kind || "job");
}

function renderJobDetail(job) {
  const result = job.result || {};
  if (job.role === "decode" || job.kind === "decode") {
    const bits = [];
    if (result.phase) bits.push(`phase=${result.phase}`);
    if (result.backend) bits.push(`backend=${result.backend}`);
    if (result.manifest_backend) bits.push(`cached=${result.manifest_backend}`);
    if (result.current_video) bits.push(`video=${result.current_video}`);
    if (result.cache_hits || result.cache_misses) bits.push(`cache ${result.cache_hits || 0}/${result.cache_misses || 0}`);
    if (result.cache_miss_videos?.length) bits.push(`miss=${result.cache_miss_videos.join(", ")}`);
    if (result.fallback_reason) bits.push(`fallback=${result.fallback_reason}`);
    return escapeHtml(bits.join(" | ") || "-");
  }
  return "-";
}

function renderJobError(job) {
  const message = job.error?.message || job.error?.type || "";
  if (!message) return "-";
  if (job.status === "canceled" && [
    "sibling shard failed the run",
    "Run already failed",
  ].includes(message)) {
    return `<span class="muted" title="${escapeHtml(message)}">级联取消</span>`;
  }
  return `<span title="${escapeHtml(message)}">${escapeHtml(message)}</span>`;
}

function renderRunJobs(run) {
  const jobs = run.jobs || [];
  if (!jobs.length) return "";
  return `
    <section class="jobs-panel">
      <h3>执行作业</h3>
      <div class="table compact-table">${table(jobs, [
        { label: "作业", render: (job) => renderJobLabel(job) },
        { label: "设备", render: (job) => escapeHtml(job.device || job.payload?.device || "-") },
        { label: "状态", render: (job) => statusBadge(job.status) },
        { label: "进度", render: (job) => renderJobProgress(job) },
        { label: "详情", render: (job) => renderJobDetail(job) },
        { label: "Worker", render: (job) => escapeHtml(job.worker_id || "-") },
        { label: "错误原因", render: (job) => renderJobError(job) },
      ])}</div>
    </section>
  `;
}

function renderCleanedArtifactsNotice(run) {
  if (!run?.artifact_cleaned_at) return "";
  return `<div class="message warn"><p><strong>产物已清理</strong>: 这个 Run 的磁盘产物已经删除；时间线和指标摘要仍可查看，如需重新查看 Pred / GT / Diff，请重试重新生成。</p></div>`;
}

function renderModelLoadReport(run) {
  const report = run?.result?.model_load;
  if (!report || typeof report !== "object") return "";
  const matched = Number(report.matched ?? 0);
  const total = Number(report.total_in_checkpoint ?? 0);
  const missing = Array.isArray(report.missing_keys) ? report.missing_keys : [];
  const unexpected = Array.isArray(report.unexpected_keys) ? report.unexpected_keys : [];
  const hasProblem = missing.length > 0 || unexpected.length > 0;
  const cls = hasProblem ? "message warn" : "message";
  const summary = `已匹配权重: ${matched} / ${total} (missing ${missing.length}, unexpected ${unexpected.length})`;
  const detailBits = [];
  if (missing.length) {
    detailBits.push(`<details><summary>missing keys (${missing.length})</summary><pre>${escapeHtml(missing.slice(0, 100).join("\n"))}</pre></details>`);
  }
  if (unexpected.length) {
    detailBits.push(`<details><summary>unexpected keys (${unexpected.length})</summary><pre>${escapeHtml(unexpected.slice(0, 100).join("\n"))}</pre></details>`);
  }
  return `<div class="${cls}"><p><strong>Checkpoint</strong>: ${escapeHtml(summary)}</p>${detailBits.join("")}</div>`;
}

function formatHealthNumber(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  if (Math.abs(num) > 0 && Math.abs(num) < 0.01) return num.toExponential(2);
  return num.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function renderOutputHealthReport(run) {
  const report = run?.result?.output_health;
  if (!report || typeof report !== "object") return "";
  const stats = report.stats || {};
  const warnings = Array.isArray(report.warnings) ? report.warnings : [];
  const hasProblem = warnings.length > 0 || report.has_nan || (report.flow_flat && report.mask_flat);
  const cls = hasProblem ? "message warn" : "message";
  const summary = [
    `samples ${report.samples ?? "-"}`,
    `flow abs_max ${formatHealthNumber(stats.flowt_0?.abs_max)} / ${formatHealthNumber(stats.flowt_1?.abs_max)}`,
    `mask std ${formatHealthNumber(stats.mask0?.std)} / ${formatHealthNumber(stats.mask1?.std)}`,
  ].join(", ");
  const detail = warnings.length
    ? `<ul>${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul>`
    : "";
  return `<div class="${cls}"><p><strong>Output health</strong>: ${escapeHtml(summary)}</p>${detail}</div>`;
}

function renderRunPurgeNotice(run) {
  const request = run?.purge_request;
  if (!request) return "";
  const state = String(request.status || "");
  if (["requested", "canceling", "purging"].includes(state)) {
    const phase = request.report?.phase || state;
    const deleting = String(request.request_type || "") === "delete_run";
    return `<div class="message warn"><p><strong>${deleting ? "正在删除" : "正在清理产物"}</strong>: ${escapeHtml(phase)}。VFIEval 会等待 worker 完全停止并安全清理 Run 产物和缓存引用${deleting ? "，完成后再隐藏记录" : ""}。</p></div>`;
  }
  if (state === "failed") {
    const campaignId = Number(request.error?.campaign_id || 0);
    const campaignAction = campaignId > 0 && request.error?.action === "open_campaign"
      ? `<button type="button" class="secondary" data-open-campaign-dependency="${campaignId}">前往盲测记录</button>`
      : "";
    return `<div class="message error"><p><strong>删除清理失败</strong>: ${escapeHtml(request.error?.message || "unknown error")}。处理依赖后，再次点击“删除记录”即可重试。</p>${campaignAction}</div>`;
  }
  return "";
}

class Sha256Hasher {
  constructor() {
    this.h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
    this.buffer = new Uint8Array(64);
    this.bufferLength = 0;
    this.bytesHashed = 0;
    this.finished = false;
  }

  update(data) {
    if (this.finished) throw new Error("SHA-256 hasher is already finalized");
    const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
    this.bytesHashed += bytes.length;
    let position = 0;
    while (position < bytes.length) {
      const take = Math.min(bytes.length - position, 64 - this.bufferLength);
      this.buffer.set(bytes.subarray(position, position + take), this.bufferLength);
      this.bufferLength += take;
      position += take;
      if (this.bufferLength === 64) {
        this.process(this.buffer);
        this.bufferLength = 0;
      }
    }
    return this;
  }

  process(chunk) {
    const k = Sha256Hasher.K;
    const w = new Uint32Array(64);
    for (let i = 0; i < 16; i += 1) {
      const offset = i * 4;
      w[i] = ((chunk[offset] << 24) | (chunk[offset + 1] << 16) | (chunk[offset + 2] << 8) | chunk[offset + 3]) >>> 0;
    }
    for (let i = 16; i < 64; i += 1) {
      const x = w[i - 15];
      const y = w[i - 2];
      const s0 = (((x >>> 7) | (x << 25)) ^ ((x >>> 18) | (x << 14)) ^ (x >>> 3)) >>> 0;
      const s1 = (((y >>> 17) | (y << 15)) ^ ((y >>> 19) | (y << 13)) ^ (y >>> 10)) >>> 0;
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = this.h;
    for (let i = 0; i < 64; i += 1) {
      const s1 = (((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7))) >>> 0;
      const choice = ((e & f) ^ (~e & g)) >>> 0;
      const t1 = (h + s1 + choice + k[i] + w[i]) >>> 0;
      const s0 = (((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10))) >>> 0;
      const majority = ((a & b) ^ (a & c) ^ (b & c)) >>> 0;
      const t2 = (s0 + majority) >>> 0;
      h = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    [a, b, c, d, e, f, g, h].forEach((value, index) => { this.h[index] = (this.h[index] + value) >>> 0; });
  }

  hex() {
    if (!this.finished) {
      const bits = this.bytesHashed * 8;
      this.buffer[this.bufferLength] = 0x80;
      this.bufferLength += 1;
      if (this.bufferLength > 56) {
        this.buffer.fill(0, this.bufferLength, 64);
        this.process(this.buffer);
        this.bufferLength = 0;
      }
      this.buffer.fill(0, this.bufferLength, 56);
      const high = Math.floor(bits / 0x100000000);
      const low = bits >>> 0;
      for (let i = 0; i < 4; i += 1) this.buffer[56 + i] = (high >>> (24 - i * 8)) & 0xff;
      for (let i = 0; i < 4; i += 1) this.buffer[60 + i] = (low >>> (24 - i * 8)) & 0xff;
      this.process(this.buffer);
      this.finished = true;
    }
    return Array.from(this.h).map((value) => value.toString(16).padStart(8, "0")).join("");
  }
}

Sha256Hasher.K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function renderPerformanceReport(run) {
  const report = run?.result?.performance;
  if (!report || typeof report !== "object") return "";
  const memory = report.device_memory || {};
  const deviceSeconds = report.device_seconds || {};
  return `
    <section class="performance-report">
      <h3>Performance</h3>
      <div class="summary-grid">
        <div><span>Profile</span><strong>${escapeHtml(report.artifact_profile || "-")}</strong></div>
        <div><span>End-to-end FPS</span><strong>${formatNumber(report.end_to_end_fps)}</strong></div>
        <div><span>Steady FPS</span><strong>${formatNumber(report.steady_state_fps)}</strong></div>
        <div><span>Total wall</span><strong>${formatNumber(report.total_wall_seconds)}s</strong></div>
        <div><span>Prefetch wait</span><strong>${formatNumber(report.prefetch_wait_seconds)}s</strong></div>
        <div><span>Save backpressure</span><strong>${formatNumber(report.save_backpressure_seconds)}s</strong></div>
        <div><span>Device model</span><strong>${formatNumber(deviceSeconds.transfer_and_model)}s</strong></div>
        <div><span>Device post</span><strong>${formatNumber(deviceSeconds.postprocess)}s</strong></div>
        <div><span>Peak allocated</span><strong>${formatBytes(memory.max_memory_allocated || 0)}</strong></div>
        <div><span>Save queue peak</span><strong>${escapeHtml(report.save_max_inflight ?? "-")}</strong></div>
      </div>
    </section>
  `;
}

function renderExecutionProfileRecommendation(preflight) {
  const profile = preflight?.execution_profile;
  if (!profile) return "";
  const settings = profile.settings || {};
  const performance = profile.performance || {};
  return `<div class="message"><p><strong>Benchmark 建议</strong>：batch ${escapeHtml(settings.batch_size_per_device ?? settings.batch_size ?? "-")}，prefetch ${escapeHtml(settings.prefetch_workers ?? "-")}，save ${escapeHtml(settings.save_workers ?? "-")}；历史稳态 ${formatNumber(performance.steady_state_fps)} FPS。建议不会自动覆盖当前设置。</p><button class="secondary" data-apply-execution-profile type="button">应用这组建议</button></div>`;
}

function applyExecutionProfileRecommendation() {
  const settings = state.preflight?.execution_profile?.settings;
  if (!settings || typeof settings !== "object") {
    toast("当前没有可应用的 Benchmark 建议");
    return;
  }
  const form = $("infer-form");
  const applied = [];
  const batchSize = Number(settings.batch_size_per_device ?? settings.batch_size);
  if (Number.isInteger(batchSize) && batchSize > 0) {
    form.elements.batch_size.value = String(batchSize);
    form.elements.batch_size_per_device.value = String(batchSize);
    applied.push(`batch ${batchSize}`);
  }
  for (const name of ["prefetch_workers", "save_workers", "max_save_inflight"]) {
    const value = Number(settings[name]);
    if (!Number.isInteger(value) || value <= 0 || !form.elements[name]) continue;
    form.elements[name].value = String(value);
    applied.push(`${name} ${value}`);
  }
  if (!applied.length) {
    toast("这条历史建议没有可应用的参数");
    return;
  }
  schedulePreflight(0);
  toast(`已应用建议：${applied.join("，")}`);
}

function renderDecodePanel(run) {
  const job = (run.jobs || []).find((item) => item.role === "decode" || item.kind === "decode");
  if (!job && run.status !== "decoding") return "";
  const result = job?.result || {};
  const current = Number(job?.progress_current ?? run.progress_current ?? result.decoded_frames ?? 0);
  const total = Number(job?.progress_total ?? run.progress_total ?? result.total_frames ?? 0);
  const percent = total > 0 ? Math.max(0, Math.min(100, Math.round((current / total) * 100))) : 0;
  const backend = result.backend || "auto";
  const currentVideo = result.current_video || "-";
  const cacheText = `${result.cache_hits || 0} hit / ${result.cache_misses || 0} miss`;
  const phase = result.phase || (backend === "cache" ? "indexing_cached_frames" : "decoding");
  const cacheMissVideos = result.cache_miss_videos || [];
  const decodeTitle = phase === "checking_cache" ? "Checking decode cache" : phase === "indexing_cached_frames" ? "Reusing decoded cache" : "Decode progress";
  const decodeHint = phase === "checking_cache"
    ? "Checking whether decoded frames can be reused before inference jobs are queued."
    : phase === "indexing_cached_frames"
      ? "Reusing decoded frames and rebuilding this Run's sample index."
      : "Frames are decoded before inference jobs are queued.";
  return `
    <section class="decode-panel">
      <div class="panel-head">
        <div>
          <h3>${escapeHtml(decodeTitle)}</h3>
          <p class="muted">${escapeHtml(decodeHint)}</p>
        </div>
        ${statusBadge(job?.status || run.status)}
      </div>
      <div class="progress-bar" aria-label="decode progress"><span style="width: ${percent}%"></span></div>
      <div class="summary-grid">
        <div><span>Frames</span><strong>${escapeHtml(current)}/${escapeHtml(total || "-")}</strong></div>
        <div><span>Current video</span><strong>${escapeHtml(currentVideo)}</strong></div>
        <div><span>Backend</span><strong>${escapeHtml(backend)}</strong></div>
        <div><span>Cache</span><strong>${escapeHtml(cacheText)}</strong></div>
      </div>
      ${cacheMissVideos.length ? `<div class="message warn"><p><strong>Cache miss</strong>: ${escapeHtml(cacheMissVideos.join(", "))} will be decoded again because its cache is missing or stale.</p></div>` : ""}
      ${result.fallback_reason ? `<div class="message warn"><p><strong>Decode fallback</strong>: ${escapeHtml(result.fallback_reason)}</p></div>` : ""}
    </section>
  `;
}

async function loadRunVideoTimeline(runId, videoName, options = {}) {
  const metric = options.metric || state.selectedMetricByRun[runId] || "";
  const windowKey = `${runId}:${videoName}`;
  const windowStart = Number(
    options.windowStart ?? state.timelineWindowStartByVideo[windowKey] ?? 0,
  );
  abortTimelineRequest();
  const controller = new AbortController();
  const requestGeneration = ++state.timelineRequestGeneration;
  const resultGeneration = currentRunResultGeneration(runId);
  const selectionGeneration = state.runSelectionGeneration;
  state.timelineAbortController = controller;
  state.timelineAbortRunId = Number(runId);
  try {
    const payload = await api(
      `/api/runs/${runId}/videos/${encodeURIComponent(videoName)}/timeline?bucket_count=160&window_start=${windowStart}&window_size=${TIMELINE_WINDOW_SIZE}${metric ? `&metric=${encodeURIComponent(metric)}` : ""}`,
      { signal: controller.signal },
    );
    if (requestGeneration !== state.timelineRequestGeneration
        || selectionGeneration !== state.runSelectionGeneration
        || resultGeneration !== currentRunResultGeneration(runId)
        || Number(state.selectedRun?.id) !== Number(runId)) return null;
    state.runVideoTimelines[windowKey] = payload;
    state.timelineWindowStartByVideo[windowKey] = Number(payload.window_start || 0);
    return payload;
  } catch (error) {
    if (error.name === "AbortError") return null;
    throw error;
  } finally {
    if (state.timelineAbortController === controller) {
      state.timelineAbortController = null;
      state.timelineAbortRunId = null;
    }
  }
}

function renderRunVideosPager() {
  const page = state.runVideosPage;
  if (!page || Number(page.total_pages || 1) <= 1) return "";
  return `
    <div class="pager run-video-pager">
      <button class="secondary" data-run-videos-page="${Number(page.page || 1) - 1}" ${Number(page.page || 1) <= 1 ? "disabled" : ""} type="button">上一页</button>
      <span class="muted">第 ${escapeHtml(page.page || 1)} / ${escapeHtml(page.total_pages || 1)} 页，当前 ${escapeHtml((page.videos || []).length)} / ${escapeHtml(page.filtered_count || 0)} 个视频</span>
      <button class="secondary" data-run-videos-page="${Number(page.page || 1) + 1}" ${Number(page.page || 1) >= Number(page.total_pages || 1) ? "disabled" : ""} type="button">下一页</button>
    </div>
  `;
}

function renderRunResultPlaceholder(run) {
  if (!state.runVideosPage) {
    return `
      <div class="artifact-pending" aria-live="polite" aria-busy="true">
        <div class="timeline-skeleton"><span></span><span></span><span></span></div>
        <p class="muted">正在刷新这个 Run 的结果…</p>
      </div>
    `;
  }
  if (!TERMINAL_STATUSES.has(run.status) && !run.artifact_cleaned_at) {
    return `
      <div class="artifact-pending" aria-live="polite" aria-busy="true">
        <div class="timeline-skeleton"><span></span><span></span><span></span></div>
        <p class="muted">产物生成中，完成后会自动显示，无需刷新整个页面。</p>
      </div>
    `;
  }
  const reason = run.artifact_cleaned_at
    ? "这个 Run 的产物已清理。"
    : "这个 Run 完成刷新后仍没有可查看的产物。";
  return `<div class="message warn"><p><strong>没有可加载的产物</strong>: ${escapeHtml(reason)}</p></div>`;
}

function compareInputRows(payload) {
  if (Array.isArray(payload?.inputs)) return payload.inputs;
  if (Array.isArray(payload?.slots)) return payload.slots;
  const rows = [];
  if (payload?.gt || payload?.reference) rows.push({ slot: "gt", label: "GT", ...(payload.gt || payload.reference) });
  for (const [index, row] of (payload?.predictions || payload?.preds || []).entries()) {
    rows.push({ slot: row.slot || `pred_${index + 1}`, label: row.label || row.method_label || `Pred ${String.fromCharCode(65 + index)}`, ...row });
  }
  return rows;
}

function compareInputSlotLabel(slot) {
  const normalized = String(slot || "").toLowerCase();
  if (normalized === "gt" || normalized === "reference") return "GT";
  const match = normalized.match(/^pred_?([a-z])$/);
  return match ? `Pred ${match[1].toUpperCase()}` : String(slot || "input");
}

function renderCompareInputMedia(runId, row) {
  const slot = String(row.slot || row.kind || "");
  if (!slot) return "";
  const label = row.label || row.method_label || row.display_name || row.item_display_name || slot;
  const slotLabel = compareInputSlotLabel(slot);
  const base = `/api/runs/${Number(runId)}/compare-inputs/${encodeURIComponent(slot)}/media`;
  const aligned = `${base}?variant=aligned`;
  const original = `${base}?variant=original`;
  const media = String(row.media_kind || row.kind || "video") === "frame_sequence"
    ? `<img data-compare-input-media data-compare-input-base="${escapeHtml(base)}" loading="lazy" src="${aligned}" alt="${escapeHtml(`${slotLabel} ${label} aligned`)}">`
    : `<video data-compare-input-media data-compare-input-base="${escapeHtml(base)}" controls playsinline preload="metadata" src="${aligned}"></video>`;
  return `<article class="compare-input-tile" data-compare-input-slot="${escapeHtml(slot)}"><div class="panel-head"><div><h4>${escapeHtml(`${slotLabel} · ${label}`)}</h4><p class="muted">${escapeHtml([row.width && row.height ? `${row.width}×${row.height}` : "", row.frame_count ? `${row.frame_count} 帧` : ""].filter(Boolean).join(" · "))}</p></div><span class="compat-badge">${escapeHtml(`${slotLabel} · ${row.snapshot_active ? "snapshot" : "original member"}`)}</span></div><div class="segmented compare-input-variants" role="group" aria-label="${escapeHtml(`${slotLabel} input variant`)}"><button class="secondary active" data-compare-input-variant="aligned" type="button" aria-pressed="true">Aligned</button><button class="secondary" data-compare-input-variant="original" type="button" aria-pressed="false">Original</button></div>${media}<a href="${original}" target="_blank" rel="noreferrer">在新标签页打开原始输入</a></article>`;
}

function renderCompareInputs(run) {
  if (!isCompareRun(run)) return "";
  const payload = state.compareInputsByRun[Number(run.id)];
  if (!payload) return `<section class="compare-input-panel" aria-busy="true"><div class="timeline-skeleton"><span></span><span></span><span></span></div><p class="muted">正在加载 Compare 输入与 Alignment Plan…</p></section>`;
  if (payload.error) return `<div class="message warn"><p>Compare 输入暂时不可用：${escapeHtml(payload.error)}</p></div>`;
  const rows = compareInputRows(payload);
  const plan = payload.alignment_plan || payload.alignment || run.result?.alignment_plan || run.metadata?.alignment_plan || {};
  return `<section class="compare-input-panel"><div class="panel-head"><div><h3>Compare 输入</h3><p class="muted">这些 Pred 引用原模型输出；Compare 不发布新的可复用 Pred。来源删除前会先切换到不可复用快照。</p></div></div>${renderAlignmentPlan(plan)}<div class="compare-input-grid">${rows.map((row) => renderCompareInputMedia(run.id, row)).join("")}</div></section>`;
}

function retriableMetricCount(runId) {
  if (Number(state.metricSummary?.run_id) !== Number(runId)) return 0;
  return Object.values(state.metricSummary?.metrics || {}).reduce(
    (total, row) => total + Number(row?.failed || 0) + Number(row?.unavailable || 0),
    0,
  );
}

function renderArtifactStorageDiagnostic(diagnostic) {
  if (!diagnostic || typeof diagnostic !== "object") return "";
  const predicted = Number(diagnostic.predicted_artifact_budget_bytes);
  const actual = Number(diagnostic.actual_artifact_bytes);
  const hasPredicted = diagnostic.predicted_artifact_budget_bytes != null
    && Number.isFinite(predicted) && predicted >= 0;
  const hasActual = diagnostic.actual_artifact_bytes != null
    && Number.isFinite(actual) && actual >= 0;
  if (!hasPredicted && !hasActual) return "";
  const ratio = Number(diagnostic.budget_utilization_ratio);
  const ratioText = Number.isFinite(ratio) ? ` · ${(ratio * 100).toFixed(1)}%` : "";
  const measurement = diagnostic.measurement === "partial"
    ? "（部分产物尚无字节记录）"
    : diagnostic.measurement === "in_progress"
      ? "（生成中）"
      : diagnostic.measurement === "cleaned"
        ? "（产物已清理）"
        : "";
  return `
    <div>
      <span>产物空间（实际 / 预算）</span>
      <strong>${escapeHtml(hasActual ? formatBytes(actual) : "-")} / ${escapeHtml(hasPredicted ? formatBytes(predicted) : "-")}${escapeHtml(ratioText)}</strong>
      ${measurement ? `<small class="muted">${escapeHtml(measurement)}</small>` : ""}
    </div>
  `;
}

function renderRunDetail() {
  const run = state.selectedRun;
  if (!run) {
    renderEmptyRunDetail();
    return;
  }
  const videos = state.runVideosPage?.videos || [];
  const selectedVideoName = state.selectedVideoByRun[run.id] || videos[0]?.video_name;
  const video = selectedVideoName ? state.runVideoTimelines[`${run.id}:${selectedVideoName}`] : null;
  $("run-detail").innerHTML = `
    <div class="panel-head">
      <div>
        <h2>#${run.id} ${escapeHtml(run.name)}</h2>
        <p class="muted">${escapeHtml(runSourceLabel(run))}</p>
      </div>
      <div class="actions">
        <span data-run-live-status>${runStatusDisplay(run)}</span>
        ${retriableMetricCount(run.id) > 0 ? `<button class="secondary" data-retry-run-metrics="${run.id}" ${["metric_queued", "metric_running"].includes(run.status) ? "disabled" : ""} type="button">重试失败/不可用指标 (${retriableMetricCount(run.id)})</button>` : ""}
        <button class="secondary" data-refresh-run-results="${run.id}" type="button">刷新结果</button>
        <button class="secondary" data-rename-run="${run.id}" type="button">重命名</button>
        <button class="secondary" data-run-live-cancel data-cancel-run="${run.id}" ${TERMINAL_STATUSES.has(run.status) ? "disabled" : ""} type="button">取消</button>
        <button class="secondary" data-retry-run="${run.id}" type="button">重试</button>
        ${isFileInputRun(run) ? `<button class="secondary" data-clone-run="${run.id}" type="button">按当前输入克隆</button>` : ""}
        <button class="secondary danger" data-delete-run="${run.id}" ${["requested", "canceling", "purging"].includes(runPurgeState(run)) ? "disabled" : ""} type="button">${runPurgeState(run) === "failed" && runPurgeDeletesRecord(run) ? "重试删除" : "删除记录"}</button>
        <button class="secondary" data-cleanup-run="${run.id}" ${TERMINAL_STATUSES.has(run.status) && !run.artifact_cleaned_at && !["requested", "canceling", "purging"].includes(runPurgeState(run)) ? "" : "disabled"} type="button">清理产物</button>
      </div>
    </div>
    <details class="run-meta" ${state.runMetaCollapsed ? "" : "open"}>
      <summary>记录详情</summary>
      <div class="summary-grid">
        <div><span>Run 类型</span><strong>${escapeHtml(runTypeLabel(run.metadata?.run_type || "model_inference"))}</strong></div>
        <div><span>进度</span><strong data-run-live-progress>${escapeHtml(run.progress_current || 0)}/${escapeHtml(run.progress_total || 0)}</strong></div>
        <div><span>推理阶段</span><strong data-run-live-inference>${escapeHtml(renderInferencePhase(run))}</strong></div>
        <div><span>评测阶段</span><strong data-run-live-metric>${escapeHtml(renderMetricPhase(run))}</strong></div>
        <div><span>输出目录</span><strong>${escapeHtml(run.metadata?.output_dir || run.result?.output_dir || "-")}</strong></div>
        <div><span>产物数</span><strong>${escapeHtml(run.artifact_summary?.total || 0)}</strong></div>
        ${renderArtifactStorageDiagnostic(run.artifact_storage)}
      </div>
      ${renderRunError(run)}
      ${renderRunPurgeNotice(run)}
      ${renderCleanedArtifactsNotice(run)}
      ${renderModelLoadReport(run)}
      ${renderOutputHealthReport(run)}
      ${renderPerformanceReport(run)}
      ${renderDecodePanel(run)}
      <div class="message"><p><strong>Execution</strong>: ${runExecutionTarget(run)}</p></div>
      ${renderPortableMetricHealthTable(run.metadata?.metric_health || {})}
      ${renderRunJobs(run)}
    </details>
    ${renderCompareInputs(run)}
    <div class="run-workspace">
      <aside class="video-tabs">
        <h3>视频</h3>
        ${videos.length ? videos.map((item) => `
          <button class="video-tab ${item.video_name === selectedVideoName ? "active" : ""}" data-run-video="${escapeHtml(item.video_name)}" type="button">
            <strong>${escapeHtml(item.video_file || item.video_name)}</strong>
            <span>${escapeHtml(item.sample_count || 0)} samples</span>
          </button>
        `).join("") : `<p class="muted">${TERMINAL_STATUSES.has(run.status) ? "没有可查看的视频。" : "产物生成中…"}</p>`}
        ${renderRunVideosPager()}
      </aside>
      <section class="sample-viewer">
        ${video ? renderVideoTimeline(video) : renderRunResultPlaceholder(run)}
      </section>
    </div>
    ${renderRunFeedback(run, selectedVideoName, video)}
  `;
}

function ratingStars(rating) {
  const score = Number(rating);
  if (!Number.isFinite(score) || score <= 0) return "<span class=\"muted\">未打分</span>";
  const clamped = Math.max(0, Math.min(5, score));
  const full = Math.floor(clamped);
  const hasHalf = clamped - full >= 0.25 && clamped - full < 0.875;
  const rounded = clamped - full >= 0.875;
  const fullCount = full + (rounded ? 1 : 0);
  const half = hasHalf ? 1 : 0;
  const empty = Math.max(0, 5 - fullCount - half);
  return `<span class="rating-stars" title="${escapeHtml(formatRating(score))}/5">${"★".repeat(fullCount)}${half ? "⯨" : ""}${"☆".repeat(empty)} <span class="rating-value">${escapeHtml(formatRating(score))}</span></span>`;
}

// Ratings use a 0.25 step; drop trailing zeros so 4.00 shows as "4" and 4.25 stays "4.25".
function formatRating(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return String(Math.round(num * 100) / 100);
}

// 1.00–5.00 in 0.25 increments, highest first for the dropdown.
const RATING_OPTIONS = Array.from({ length: 17 }, (_, i) => (5 - i * 0.25));

// Rating <select> options: blank ("不打分") plus 5.00 → 1.00 in 0.25 steps.
// `selected` is a numeric value to pre-select when editing an existing entry.
function ratingOptions(selected) {
  const chosen = selected === null || selected === undefined || selected === ""
    ? ""
    : formatRating(selected);
  const blank = `<option value="" ${chosen === "" ? "selected" : ""}>不打分</option>`;
  const opts = RATING_OPTIONS.map((score) => {
    const value = formatRating(score);
    return `<option value="${value}" ${chosen === value ? "selected" : ""}>${value} 分</option>`;
  }).join("");
  return blank + opts;
}

// Compare runs expose pred tracks (each its own model/checkpoint); a single
// inference run has none. The picker lets the reviewer say which pred they are
// scoring so the row records the right weight for the stats tab.
function feedbackTrackOptions(video) {
  const tracks = (video && video.video_artifact_tracks) || [];
  const labels = [];
  const seen = new Set();
  for (const track of tracks) {
    const label = String(track.track_label || "").trim();
    if (label && !seen.has(label)) {
      seen.add(label);
      labels.push(label);
    }
  }
  return labels;
}

function renderRunFeedback(run, selectedVideoName, video) {
  const feedback = run.feedback || [];
  const rated = feedback.filter((item) => item.rating !== null && item.rating !== undefined);
  const mean = rated.length ? (rated.reduce((sum, item) => sum + Number(item.rating), 0) / rated.length) : null;
  const videoName = selectedVideoName || "";
  const trackLabels = feedbackTrackOptions(video);
  const trackField = trackLabels.length ? `
        <label>
          <span>对比轨道</span>
          <select name="track_label">
            <option value="">整体 / 未指定</option>
            ${trackLabels.map((label) => `<option value="${escapeHtml(label)}">${escapeHtml(label)}</option>`).join("")}
          </select>
        </label>` : "";
  return `
    <section class="feedback-panel">
      <div class="panel-head">
        <div>
          <h3>评分与问题</h3>
          <p class="muted">评分绑定当前视频${trackLabels.length ? "与所选对比轨道" : ""}，可在“统计”页按视频、模型、权重汇总。评分以 0.25 为分度。</p>
        </div>
        <div class="metric-summary">
          <span>反馈 ${escapeHtml(feedback.length)}</span>
          <span>平均分 ${mean === null ? "-" : formatRating(mean)}</span>
        </div>
      </div>
      <form class="feedback-form" data-feedback-form="${escapeHtml(run.id)}">
        <input type="hidden" name="video" value="${escapeHtml(videoName)}">
        <p class="feedback-context muted">评分对象：<strong>${escapeHtml(videoName || "（先选择一个视频）")}</strong></p>
        <label>
          <span>用户名</span>
          <input name="username" value="${escapeHtml(state.feedbackUsername || "")}" placeholder="你的名字" maxlength="80">
        </label>
        <label>
          <span>评分</span>
          <select name="rating">${ratingOptions("")}</select>
        </label>
        ${trackField}
        <label class="wide-field">
          <span>问题 / 备注</span>
          <textarea name="issue" rows="2" placeholder="描述发现的问题或想记录的内容" maxlength="2000"></textarea>
        </label>
        <button type="submit" ${videoName ? "" : "disabled"}>提交反馈</button>
      </form>
      <div class="feedback-list">
        ${feedback.length ? feedback.map((item) => renderFeedbackItem(run, item)).join("") : "<p class=\"muted\">还没有反馈。</p>"}
      </div>
    </section>
  `;
}

// One feedback row. When its id is in `state.editingFeedback`, render an inline
// edit form instead of the static view so a mis-scored review can be corrected.
function renderFeedbackItem(run, item) {
  const context = [
    item.video ? `视频 ${item.video}` : "",
    item.track_label ? `轨道 ${item.track_label}` : "",
    item.model_name ? `模型 ${item.model_name}` : "",
    item.checkpoint ? `权重 ${item.checkpoint}` : "",
  ].filter(Boolean).map((text) => `<span class="feedback-tag">${escapeHtml(text)}</span>`).join("");
  const edited = item.updated_at && item.created_at && Number(item.updated_at) - Number(item.created_at) > 1
    ? "<span class=\"muted\">（已编辑）</span>"
    : "";
  if (Number(state.editingFeedback) === Number(item.id)) {
    return `
      <article class="feedback-item editing">
        <form class="feedback-edit-form" data-feedback-edit-form="${escapeHtml(item.id)}" data-feedback-run="${escapeHtml(run.id)}">
          <label>
            <span>用户名</span>
            <input name="username" value="${escapeHtml(item.username || "")}" maxlength="80">
          </label>
          <label>
            <span>评分</span>
            <select name="rating">${ratingOptions(item.rating)}</select>
          </label>
          <label class="wide-field">
            <span>问题 / 备注</span>
            <textarea name="issue" rows="2" maxlength="2000">${escapeHtml(item.issue || "")}</textarea>
          </label>
          <div class="feedback-edit-actions">
            <button type="submit">保存</button>
            <button class="secondary" data-feedback-cancel-edit type="button">取消</button>
          </div>
        </form>
      </article>
    `;
  }
  return `
    <article class="feedback-item">
      <div class="feedback-item-head">
        <strong>${escapeHtml(item.username || "匿名")}</strong>
        ${ratingStars(item.rating)}
        ${edited}
        <button class="secondary feedback-edit" data-feedback-edit="${escapeHtml(item.id)}" data-feedback-run="${escapeHtml(run.id)}" type="button">编辑</button>
        <button class="secondary danger feedback-delete" data-feedback-delete="${escapeHtml(item.id)}" data-feedback-run="${escapeHtml(run.id)}" type="button">删除</button>
      </div>
      ${context ? `<div class="feedback-tags">${context}</div>` : ""}
      ${item.issue ? `<p class="feedback-issue">${escapeHtml(item.issue)}</p>` : "<p class=\"muted\">无问题描述。</p>"}
    </article>
  `;
}

function metricNamesForVideo(video) {
  const names = new Set();
  for (const sample of video.samples || []) {
    Object.keys(sample.metrics || {}).forEach((name) => names.add(name));
  }
  return Array.from(names).sort();
}

function selectedMetric(video) {
  const names = metricNamesForVideo(video);
  if (!names.length || !state.selectedRun) return null;
  const current = state.selectedMetricByRun[state.selectedRun.id];
  if (current && names.includes(current)) return current;
  const fallback = names.find((name) => (video.metric_summary?.[name]?.completed || 0) > 0) || names[0];
  state.selectedMetricByRun[state.selectedRun.id] = fallback;
  return fallback;
}

function metricReason(metric) {
  return metric?.details?.reason || metric?.details?.type || "";
}

function renderVideoMetricSummary(video) {
  const entries = Object.entries(video.video_metrics || {}).sort(([left], [right]) => left.localeCompare(right));
  if (!entries.length) return "";
  return `
    <div class="metric-summary">
      <span class="metric-health">video-only</span>
      ${entries.map(([name, metric]) => {
        const reason = metricReason(metric);
        const value = metric.value === null || metric.value === undefined ? "" : ` ${formatNumber(metric.value)}`;
        const detail = reason ? `${metric.status}${value} - ${reason}` : `${metric.status}${value}`;
        return `<span title="${escapeHtml(`${name}: ${detail}`)}"><strong>${escapeHtml(name)}</strong> ${escapeHtml(detail)}</span>`;
      }).join("")}
    </div>
  `;
}

function renderMetricSummaryPills(video, metricName) {
  const summary = video.metric_summary?.[metricName] || state.metricSummary?.metrics?.[metricName];
  if (!summary) return renderVideoMetricSummary(video);
  const reason = (summary.reasons || [])[0];
  return `
    <div class="metric-summary">
      <span>pending ${escapeHtml(summary.pending || 0)}</span>
      <span>running ${escapeHtml(summary.running || 0)}</span>
      <span>completed ${escapeHtml(summary.completed || 0)}</span>
      <span>unavailable ${escapeHtml(summary.unavailable || 0)}</span>
      <span>failed ${escapeHtml(summary.failed || 0)}</span>
      <span>skipped ${escapeHtml(summary.skipped || 0)}</span>
      <span>missing ${escapeHtml(summary.missing || 0)}</span>
      <span>mean ${formatNumber(summary.mean)}</span>
      ${reason ? `<span title="${escapeHtml(reason)}">原因 ${escapeHtml(reason)}</span>` : ""}
    </div>
  `;
}

function renderMetricToolbar(video, metricName) {
  const names = metricNamesForVideo(video);
  if (!names.length) {
    return `<div class="metric-toolbar"><span class="muted">这个视频没有逐帧指标。</span>${renderVideoMetricSummary(video)}</div>`;
  }
  return `
    <div class="metric-toolbar">
      <label>
        <span>指标曲线</span>
        <select data-metric-select="${escapeHtml(video.video_name)}">
          ${names.map((name) => `<option value="${escapeHtml(name)}" ${name === metricName ? "selected" : ""}>${escapeHtml(name)}</option>`).join("")}
        </select>
      </label>
      ${renderMetricSummaryPills(video, metricName)}${renderVideoMetricSummary(video)}
    </div>
  `;
}

function countMetricStatuses(samples, metricName) {
  const counts = {};
  for (const sample of samples) {
    const status = sample.metrics?.[metricName]?.status || "missing";
    counts[status] = (counts[status] || 0) + 1;
  }
  return counts;
}

function metricPointReadout(sample, metricName) {
  const metric = sample?.metrics?.[metricName] || {};
  const value = metric.value === null || metric.value === undefined
    ? (metricReason(metric) || metric.status || "无数据")
    : formatNumber(metric.value);
  return `帧 ${sample?.frame_index ?? "-"} · ${metricName} · ${value}`;
}

function renderMetricDataTable(video, samples, metricName, selectedIndex) {
  const rows = samples.map((sample, index) => ({ sample, index }));
  return `
    <details class="metric-data-table">
      <summary>查看逐帧数据表（${samples.length} 行）</summary>
      <div class="table compact-table">${table(rows, [
        {
          label: "帧",
          render: (row) => `<button class="link-button" data-chart-video="${escapeHtml(video.video_name)}" data-chart-sample="${row.index}" type="button" ${row.index === selectedIndex ? 'aria-current="true"' : ""}>${escapeHtml(row.sample.frame_index)}</button>`,
        },
        {
          label: "状态",
          render: (row) => escapeHtml(row.sample.metrics?.[metricName]?.status || "missing"),
        },
        {
          label: metricName,
          render: (row) => {
            const metric = row.sample.metrics?.[metricName];
            return metric?.value === null || metric?.value === undefined
              ? escapeHtml(metricReason(metric) || "-")
              : formatNumber(metric.value);
          },
        },
      ])}</div>
    </details>
  `;
}

function renderMetricChart(video, selectedIndex, metricName) {
  const samples = video.samples || [];
  if (!samples.length) return "";
  if (!metricName) {
    return `<div class="chart empty-chart">${renderVideoMetricSummary(video)}</div>`;
  }
  const values = samples.map((sample) => {
    const metric = sample.metrics?.[metricName];
    return metric?.status === "completed" && metric.value !== null ? Number(metric.value) : null;
  });
  const valid = values.filter((value) => value !== null && Number.isFinite(value));
  if (!valid.length) {
    const statuses = countMetricStatuses(samples, metricName);
    return `
      <div class="chart empty-chart">
        <div class="chart-head">
          <strong>${escapeHtml(metricName)}</strong>
          <span class="muted">没有 completed 点，可通过下面的状态条定位样本</span>
        </div>
        ${renderStatusStrip(samples, metricName, selectedIndex)}
        <p class="muted">pending ${statuses.pending || 0} / running ${statuses.running || 0} / completed ${statuses.completed || 0} / unavailable ${statuses.unavailable || 0} / failed ${statuses.failed || 0} / skipped ${statuses.skipped || 0} / missing ${statuses.missing || 0}</p>
      </div>
    `;
  }
  const min = Math.min(...valid);
  const max = Math.max(...valid);
  const segments = metricLineSegments(values, min, max);
  const markerX = samples.length <= 1 ? 50 : 4 + (selectedIndex / (samples.length - 1)) * 92;
  const selectedValue = values[selectedIndex];
  const selectedY = selectedValue === null || !Number.isFinite(selectedValue)
    ? 46
    : 42 - (max === min ? 0.5 : (selectedValue - min) / (max - min)) * 30;
  const selectedSample = samples[selectedIndex] || samples[0];
  // The SVG stretches its 0..100 / 0..56 coordinate space to the container with
  // preserveAspectRatio="none", so it can only carry shapes that tolerate
  // non-uniform scaling (grid lines, the polyline, the vertical marker). The
  // sample points are rendered as an absolutely-positioned HTML overlay instead:
  // CSS-sized dots stay perfectly round no matter the container aspect ratio.
  return `
    <div class="chart" data-chart-video="${escapeHtml(video.video_name)}" data-chart-metric="${escapeHtml(metricName)}">
      <div class="chart-head">
        <strong>${escapeHtml(metricName)}</strong>
        <span class="muted">点击曲线定位；方向键移动，Home/End 跳至首尾</span>
      </div>
      <div class="chart-plot" tabindex="0" role="slider" aria-label="${escapeHtml(`${metricName} 逐帧曲线`)}"
        aria-valuemin="0" aria-valuemax="${Math.max(0, samples.length - 1)}" aria-valuenow="${selectedIndex}"
        aria-valuetext="${escapeHtml(metricPointReadout(selectedSample, metricName))}">
        <svg class="metric-chart-svg" viewBox="0 0 100 56" preserveAspectRatio="none" role="img">
          <g class="chart-grid">
            <line x1="4" x2="96" y1="12" y2="12"></line>
            <line x1="4" x2="96" y1="27" y2="27"></line>
            <line x1="4" x2="96" y1="42" y2="42"></line>
          </g>
          ${segments.map((points) => `<polyline class="metric-line" points="${points}" fill="none"></polyline>`).join("")}
          <line class="current-marker" x1="${markerX.toFixed(2)}" x2="${markerX.toFixed(2)}" y1="8" y2="46"></line>
        </svg>
        <div class="chart-points">
          <span class="selected-metric-point" style="left: ${markerX.toFixed(2)}%; top: ${((selectedY / 56) * 100).toFixed(2)}%"></span>
          ${renderWorstMetricPoints(video, samples, metricName, min, max)}
        </div>
        <div class="chart-tooltip" hidden></div>
      </div>
      <div class="chart-scale">
        <span>min ${formatNumber(min)}</span>
        <span>max ${formatNumber(max)}</span>
      </div>
      <p class="chart-readout" data-chart-readout aria-live="polite">${escapeHtml(metricPointReadout(selectedSample, metricName))}</p>
      ${renderMetricDataTable(video, samples, metricName, selectedIndex)}
    </div>
  `;
}

function metricLineSegments(values, min, max) {
  const segments = [];
  let current = [];
  values.forEach((value, index) => {
    if (value === null || !Number.isFinite(value)) {
      if (current.length) segments.push(current.join(" "));
      current = [];
      return;
    }
    const x = values.length <= 1 ? 50 : 4 + (index / (values.length - 1)) * 92;
    const normalized = max === min ? 0.5 : (value - min) / (max - min);
    current.push(`${x.toFixed(2)},${(42 - normalized * 30).toFixed(2)}`);
  });
  if (current.length) segments.push(current.join(" "));
  return segments;
}

function renderDualLpipsCharts(video, selectedIndex) {
  return `<div class="dual-chart-row">${LPIPS_PAIR.map((name) => renderMetricChart(video, selectedIndex, name)).join("")}</div>`;
}

function renderWorstMetricPoints(video, samples, metricName, min, max) {
  const worstFrames = new Set((video.worst_samples?.[metricName] || []).map((row) => Number(row.frame_index)));
  return samples.map((sample, index) => {
    if (!worstFrames.has(Number(sample.frame_index))) return "";
    const metric = sample.metrics?.[metricName];
    if (metric?.status !== "completed" || metric.value === null) return "";
    const x = samples.length <= 1 ? 50 : 4 + (index / (samples.length - 1)) * 92;
    const normalized = max === min ? 0.5 : (Number(metric.value) - min) / (max - min);
    const top = ((42 - normalized * 30) / 56) * 100;
    return `<button class="metric-point worst" data-chart-video="${escapeHtml(video.video_name)}" data-chart-sample="${index}" style="left:${x.toFixed(2)}%;top:${top.toFixed(2)}%" title="worst frame ${escapeHtml(sample.frame_index)}: ${formatNumber(metric.value)}" type="button"></button>`;
  }).join("");
}

function renderMetricOverview(video, metricName, selectedIndex) {
  const buckets = video.overview || [];
  const total = Number(video.sample_count || 0);
  if (!metricName || total <= TIMELINE_WINDOW_SIZE || !buckets.length) return "";
  const valid = buckets.filter((row) => row.mean !== null && Number.isFinite(Number(row.mean)));
  if (!valid.length) return "";
  const min = Math.min(...valid.map((row) => Number(row.min)));
  const max = Math.max(...valid.map((row) => Number(row.max)));
  const y = (value) => 24 - (max === min ? 0.5 : (Number(value) - min) / (max - min)) * 16;
  const x = (index) => buckets.length <= 1 ? 50 : 2 + (index / (buckets.length - 1)) * 96;
  const upper = valid.map((row) => `${x(row.bucket_index).toFixed(2)},${y(row.max).toFixed(2)}`);
  const lower = [...valid].reverse().map((row) => `${x(row.bucket_index).toFixed(2)},${y(row.min).toFixed(2)}`);
  const mean = valid.map((row) => `${x(row.bucket_index).toFixed(2)},${y(row.mean).toFixed(2)}`).join(" ");
  const windowStart = Number(video.window_start || 0);
  const windowEnd = Math.min(total, windowStart + Number(video.window_size || TIMELINE_WINDOW_SIZE));
  const viewX = total <= 1 ? 0 : (windowStart / (total - 1)) * 100;
  const viewWidth = total <= 1 ? 100 : Math.min(100 - viewX, Math.max(1, ((windowEnd - windowStart) / total) * 100));
  const globalIndex = windowStart + selectedIndex;
  const markerX = total <= 1 ? 50 : (globalIndex / (total - 1)) * 100;
  return `
    <div class="chart overview-chart" data-overview-video="${escapeHtml(video.video_name)}" title="点击总览加载对应细节窗口">
      <div class="chart-head"><strong>全片总览</strong><span class="muted">${total} 帧 · 阴影为 min–max，曲线为均值</span></div>
      <div class="overview-plot">
        <svg viewBox="0 0 100 28" preserveAspectRatio="none" aria-label="${escapeHtml(metricName)} overview">
          <polygon class="overview-envelope" points="${[...upper, ...lower].join(" ")}"></polygon>
          <polyline class="overview-mean" points="${mean}" fill="none"></polyline>
          <rect class="overview-viewport" x="${viewX.toFixed(2)}" y="2" width="${viewWidth.toFixed(2)}" height="24"></rect>
          <line class="current-marker" x1="${markerX.toFixed(2)}" x2="${markerX.toFixed(2)}" y1="2" y2="26"></line>
        </svg>
      </div>
    </div>`;
}

function renderStatusStrip(samples, metricName, selectedIndex) {
  return `
    <div class="status-strip">
      ${samples.map((sample, index) => {
        const metric = sample.metrics?.[metricName];
        const status = metric?.status || "missing";
        return `<button class="status-dot ${escapeHtml(status)} ${index === selectedIndex ? "active" : ""}" data-sample-jump="${index}" data-frame-index="${escapeHtml(sample.frame_index)}" type="button" title="${escapeHtml(status)} ${escapeHtml(metricReason(metric))}"></button>`;
      }).join("")}
    </div>
  `;
}

function renderWorstSamples(video, metricName) {
  const rows = metricName ? (video.worst_samples?.[metricName] || []) : [];
  if (!rows.length) return "";
  return `
    <section class="worst-samples">
      <div class="chart-head">
        <strong>最差样本</strong>
        <span class="muted">按 ${escapeHtml(metricName)} 排序</span>
      </div>
      <div class="worst-list">
        ${rows.map((row) => `
          <button class="worst-item" data-sample-video="${escapeHtml(video.video_name)}" data-sample-frame="${escapeHtml(row.frame_index)}" type="button">
            <span>frame ${escapeHtml(row.frame_index)}</span>
            <strong>${formatNumber(row.value)}</strong>
            <small>${row.timestamp === null || row.timestamp === undefined ? "-" : `${formatNumber(row.timestamp)}s`}</small>
          </button>
        `).join("")}
      </div>
    </section>
  `;
}

function renderVideoPlayer(label, artifact) {
  if (!artifact) return "";
  const item = typeof artifact === "object" ? artifact : { id: artifact };
  if (!item.id && !item.preview_url && !item.original_url) return "";
  const previewUrl = item.preview_url || item.url || `/api/files/${item.id}?variant=preview`;
  const originalUrl = item.original_url || `/api/files/${item.id}`;
  return `
    <div class="video-artifact">
      <span>${escapeHtml(label)}</span>
      <video controls playsinline preload="metadata" src="${escapeHtml(previewUrl)}" data-original-url="${escapeHtml(originalUrl)}" onerror="handleVideoPlaybackError(this)"></video>
      <a class="muted" href="${escapeHtml(originalUrl)}" target="_blank" rel="noreferrer">打开原始视频</a>
      <div class="video-playback-error message error" hidden></div>
    </div>
  `;
}

function handleVideoPlaybackError(video) {
  const host = video.closest(".video-artifact");
  const status = host?.querySelector(".video-playback-error");
  if (!status) return;
  const code = Number(video.error?.code || 0);
  const reason = code === 3
    ? "浏览器解码失败，视频产物可能损坏或编码不兼容。"
    : code === 2
      ? "视频加载中断，请检查服务连接后重试。"
      : "视频无法加载；文件可能尚未就绪、已被清理，或编码不受当前浏览器支持。";
  status.hidden = false;
  status.innerHTML = `${escapeHtml(reason)} <a href="${escapeHtml(video.dataset.originalUrl || video.currentSrc || video.src)}" target="_blank" rel="noreferrer">打开原始视频</a>`;
}

function renderVideoArtifacts(video) {
  const tracks = video.video_artifact_tracks || [];
  let items = "";
  if (tracks.length) {
    items = tracks
      .map((item) => renderVideoPlayer(`${item.track_label || "shared"} / ${item.kind}`, item))
      .join("");
  } else {
    items = `
      ${renderVideoPlayer("pred", video.video_artifacts?.pred_video)}
      ${renderVideoPlayer("gt", video.video_artifacts?.gt_video)}
      ${renderVideoPlayer("diff", video.video_artifacts?.diff_video)}
    `;
  }
  return items.trim() ? `<div class="video-artifact-strip">${items}</div>` : "";
}

function renderVideoMasterControls(video) {
  const count = (video.video_artifact_tracks || []).length
    || ["pred_video", "gt_video", "diff_video"].filter((kind) => video.video_artifacts?.[kind]).length;
  if (count <= 1) return "";
  return `
    <div class="video-master-controls">
      <button class="secondary" data-master-video-play="${escapeHtml(video.video_name)}" type="button">全部播放</button>
      <button class="secondary" data-master-video-pause="${escapeHtml(video.video_name)}" type="button">全部暂停</button>
      <button class="secondary" data-master-video-sync="${escapeHtml(video.video_name)}" type="button">同步时间</button>
    </div>
  `;
}

function renderTimelineWindowNav(video) {
  const total = Number(video.sample_count || 0);
  const windowStart = Number(video.window_start || 0);
  const windowSize = Number(video.window_size || TIMELINE_WINDOW_SIZE);
  const shown = (video.samples || []).length;
  // Only surface window navigation when the video has more samples than fit in
  // a single window; otherwise the whole timeline is already on screen.
  if (total <= shown && windowStart === 0) return "";
  const windowEnd = windowStart + shown;
  const hasPrev = windowStart > 0;
  const hasNext = windowEnd < total;
  const prevStart = Math.max(0, windowStart - windowSize);
  const nextStart = windowStart + windowSize;
  return `
    <div class="sample-controls window-nav">
      <button class="secondary" data-window-start="${prevStart}" ${hasPrev ? "" : "disabled"} type="button">← 上一段</button>
      <span class="muted">帧 ${windowStart + 1}–${windowEnd} / 共 ${total}</span>
      <button class="secondary" data-window-start="${nextStart}" ${hasNext ? "" : "disabled"} type="button">下一段 →</button>
    </div>
  `;
}

function _buildFrameChartHtml(video, selectedIndex, metricName) {
  const names = metricNamesForVideo(video);
  const isDual = LPIPS_PAIR.every((n) => names.includes(n)) && LPIPS_PAIR_SET.has(metricName);
  if (isDual) {
    return `${renderMetricOverview(video, LPIPS_PAIR[0], selectedIndex)}${renderDualLpipsCharts(video, selectedIndex)}${renderWorstSamples(video, metricName)}`;
  }
  return `${renderMetricOverview(video, metricName, selectedIndex)}${renderMetricChart(video, selectedIndex, metricName)}${renderWorstSamples(video, metricName)}`;
}

function renderFrameRegion(video, selectedIndex, metricName) {
  const samples = video.samples || [];
  const sample = samples[selectedIndex] || null;
  const windowStart = Number(video.window_start || 0);
  const globalIndex = windowStart + selectedIndex;
  const total = Number(video.sample_count || samples.length);
  // The slider element sits between the two updatable containers and is never
  // rewritten on frame change, so dragging it stays smooth. Only #frame-chart,
  // #frame-preview and the counter are refreshed in place; the <video> players
  // live outside #frame-region entirely and never reload.
  return `
    ${renderMetricToolbar(video, metricName)}
    ${renderTimelineWindowNav(video)}
    <div id="frame-chart">
      ${_buildFrameChartHtml(video, selectedIndex, metricName)}
    </div>
    <div class="sample-controls">
      <button class="secondary" data-sample-step="-1" type="button">上一帧</button>
      <input data-sample-range="${escapeHtml(video.video_name)}" type="range" min="0" max="${Math.max(0, total - 1)}" value="${globalIndex}">
      <button class="secondary" data-sample-step="1" type="button">下一帧</button>
      <span class="muted" id="frame-counter">${globalIndex + 1}/${total || 0}</span>
    </div>
    <div id="frame-preview">
      ${sample ? renderSamplePreview(sample, video) : "<p class=\"muted\">没有样本。</p>"}
    </div>
  `;
}

function renderVideoTimeline(video) {
  const samples = video.samples || [];
  const key = `${state.selectedRun.id}:${video.video_name}`;
  const selectedIndex = Math.min(Number(state.selectedSampleByVideo[key] || 0), Math.max(0, samples.length - 1));
  state.selectedSampleByVideo[key] = selectedIndex;
  const metricName = selectedMetric(video);
  // The video players live outside #frame-region so that stepping through
  // frames only re-renders the frame-dependent chart/preview and never
  // recreates the <video> elements (which would reload and stutter playback).
  return `
    <div class="panel-head compact-head">
      <div>
        <h3>${escapeHtml(video.video_file || video.video_name)}</h3>
        <p class="muted">${samples.length} 个样本，FPS ${formatNumber(video.fps)}</p>
      </div>
      <div class="actions">
        ${renderVideoMasterControls(video)}
      </div>
    </div>
    ${renderVideoArtifacts(video)}
    <div id="frame-region" data-frame-region="${escapeHtml(video.video_name)}">
      ${renderFrameRegion(video, selectedIndex, metricName)}
    </div>
  `;
}

function _tryUpdateChartMarkers(chart, video, selectedIndex, metricName) {
  // Fast path: update only SVG position markers without rebuilding innerHTML.
  // Returns false when the chart DOM doesn't match the current state (different
  // video, metric, or window) and a full rebuild is required instead.
  const overviewEl = chart.querySelector(".overview-chart");
  if (overviewEl && overviewEl.dataset.overviewVideo !== video.video_name) return false;
  const chartEls = Array.from(chart.querySelectorAll(".chart[data-chart-video]"));
  if (!chartEls.length) return false;
  for (const el of chartEls) {
    if (el.dataset.chartVideo !== video.video_name) return false;
  }
  const samples = video.samples || [];
  const total = Number(video.sample_count || samples.length);
  const windowStart = Number(video.window_start || 0);
  const globalIndex = windowStart + selectedIndex;
  // Update overview current-marker
  if (overviewEl) {
    const markerX = total <= 1 ? 50 : (globalIndex / (total - 1)) * 100;
    const m = overviewEl.querySelector(".current-marker");
    if (m) { m.setAttribute("x1", markerX.toFixed(2)); m.setAttribute("x2", markerX.toFixed(2)); }
  }
  // Update per-chart markers and selected-point
  for (const chartEl of chartEls) {
    const chartMetric = chartEl.dataset.chartMetric || metricName;
    if (!chartMetric) continue;
    const values = samples.map((s) => {
      const mv = s.metrics?.[chartMetric];
      return mv?.status === "completed" && mv.value !== null ? Number(mv.value) : null;
    });
    const valid = values.filter((v) => v !== null && Number.isFinite(v));
    if (!valid.length) continue;
    const min = Math.min(...valid);
    const max = Math.max(...valid);
    const markerX = samples.length <= 1 ? 50 : 4 + (selectedIndex / (samples.length - 1)) * 92;
    const sv = values[selectedIndex];
    const selectedY = sv === null || !Number.isFinite(sv)
      ? 46
      : 42 - (max === min ? 0.5 : (sv - min) / (max - min)) * 30;
    const mainSvg = chartEl.querySelector(".metric-chart-svg");
    if (mainSvg) {
      const m = mainSvg.querySelector(".current-marker");
      if (m) { m.setAttribute("x1", markerX.toFixed(2)); m.setAttribute("x2", markerX.toFixed(2)); }
    }
    const point = chartEl.querySelector(".selected-metric-point");
    if (point) {
      point.style.left = `${markerX.toFixed(2)}%`;
      point.style.top = `${((selectedY / 56) * 100).toFixed(2)}%`;
    }
    const sample = samples[selectedIndex] || samples[0];
    const control = chartEl.querySelector(".chart-plot");
    control?.setAttribute("aria-valuenow", String(selectedIndex));
    control?.setAttribute("aria-valuetext", metricPointReadout(sample, chartMetric));
    const readout = chartEl.querySelector("[data-chart-readout]");
    if (readout) readout.textContent = metricPointReadout(sample, chartMetric);
    chartEl.querySelectorAll("[data-chart-sample]").forEach((button) => {
      if (Number(button.dataset.chartSample) === selectedIndex) button.setAttribute("aria-current", "true");
      else button.removeAttribute("aria-current");
    });
  }
  return true;
}

function updateFrameRegion() {
  const region = document.getElementById("frame-region");
  if (!region || !state.selectedRun) return false;
  const videoName = state.selectedVideoByRun[state.selectedRun.id];
  const video = videoName ? state.runVideoTimelines[`${state.selectedRun.id}:${videoName}`] : null;
  if (!video || region.dataset.frameRegion !== videoName) return false;
  const samples = video.samples || [];
  const key = `${state.selectedRun.id}:${videoName}`;
  const selectedIndex = Math.min(Number(state.selectedSampleByVideo[key] || 0), Math.max(0, samples.length - 1));
  state.selectedSampleByVideo[key] = selectedIndex;
  const metricName = selectedMetric(video);
  const chart = region.querySelector("#frame-chart");
  const preview = region.querySelector("#frame-preview");
  const counter = region.querySelector("#frame-counter");
  const slider = region.querySelector("[data-sample-range]");
  if (!chart || !preview) return false;
  const windowStart = Number(video.window_start || 0);
  const total = Number(video.sample_count || samples.length);
  // Fast path: only update SVG markers when chart structure is already correct.
  if (_tryUpdateChartMarkers(chart, video, selectedIndex, metricName)) {
    preview.innerHTML = samples[selectedIndex] ? renderSamplePreview(samples[selectedIndex], video) : "<p class=\"muted\">没有样本。</p>";
    if (counter) counter.textContent = `${windowStart + selectedIndex + 1}/${total || 0}`;
    if (slider && Number(slider.value) !== windowStart + selectedIndex) slider.value = String(windowStart + selectedIndex);
    return true;
  }
  // Full rebuild when structure has changed (new video, metric, or window).
  chart.innerHTML = _buildFrameChartHtml(video, selectedIndex, metricName);
  preview.innerHTML = samples[selectedIndex] ? renderSamplePreview(samples[selectedIndex], video) : "<p class=\"muted\">没有样本。</p>";
  if (counter) counter.textContent = `${windowStart + selectedIndex + 1}/${total || 0}`;
  // Only sync the slider's value when the change did not originate from the
  // slider itself; overwriting it mid-drag would fight the pointer.
  if (slider && Number(slider.value) !== windowStart + selectedIndex) slider.value = String(windowStart + selectedIndex);
  return true;
}

function sampleDetail(sampleId) {
  if (!state.selectedRun) return null;
  return state.sampleDetails[`${state.selectedRun.id}:${sampleId}`] || null;
}

async function loadSampleDetail(sampleId) {
  if (!state.selectedRun) return;
  const runId = Number(state.selectedRun.id);
  const key = `${runId}:${sampleId}`;
  if (state.sampleDetails[key] || state.sampleDetailLoading[key]) return;
  const resultGeneration = currentRunResultGeneration(runId);
  const selectionGeneration = state.runSelectionGeneration;
  state.sampleDetailLoading[key] = true;
  // Each sample gets its own abort controller so sibling loads in a compare
  // frame group (GT + predA + predB share a frame) do not cancel each other.
  if (state.sampleAbortControllers[key]) state.sampleAbortControllers[key].abort();
  const controller = new AbortController();
  state.sampleAbortControllers[key] = controller;
  try {
    const payload = await api(`/api/runs/${runId}/samples/${sampleId}`, { signal: controller.signal });
    if (state.sampleAbortControllers[key] !== controller
        || selectionGeneration !== state.runSelectionGeneration
        || resultGeneration !== currentRunResultGeneration(runId)
        || Number(state.selectedRun?.id) !== runId) return;
    state.sampleDetails[key] = payload;
  } catch (error) {
    if (error.name === "AbortError") return;
    if (selectionGeneration === state.runSelectionGeneration
        && resultGeneration === currentRunResultGeneration(runId)
        && Number(state.selectedRun?.id) === runId) {
      state.sampleDetails[key] = { sample_id: sampleId, artifacts: {}, extra_artifacts: [], load_error: error.message };
    }
  } finally {
    if (state.sampleAbortControllers[key] === controller) {
      delete state.sampleAbortControllers[key];
      delete state.sampleDetailLoading[key];
    }
    // Prefer an in-place frame update so late-arriving sample detail does not
    // recreate the video players; fall back to a full render if unavailable.
    if (selectionGeneration === state.runSelectionGeneration
        && resultGeneration === currentRunResultGeneration(runId)
        && Number(state.selectedRun?.id) === runId
        && !updateFrameRegion()) renderRunDetail();
  }
}

function renderSampleMetrics(sample) {
  const entries = Object.entries(sample.metrics || {});
  if (!entries.length) return "";
  return `<span>${entries.map(([name, metric]) => `${escapeHtml(name)}=${metric.value === null || metric.value === undefined ? `${escapeHtml(metric.status)} ${escapeHtml(metricReason(metric))}` : formatNumber(metric.value)}`).join(" / ")}</span>`;
}

function renderPreviewSlot(sample, kind, label) {
  const artifact = sample.artifacts?.[kind];
  if (!artifact) {
    return `<div class="preview-slot"><span>${escapeHtml(label)}</span><p class="muted">暂无</p></div>`;
  }
  const url = artifact.preview_url || artifact.original_url;
  const href = artifact.original_url || url;
  return `
    <a class="preview-slot" href="${escapeHtml(href)}" target="_blank" rel="noreferrer">
      <span>${escapeHtml(label)}</span>
      <img src="${escapeHtml(url)}" alt="${escapeHtml(label)}" loading="lazy">
    </a>
  `;
}

function renderExtraArtifacts(sample) {
  const extras = sample.extra_artifacts || [];
  if (!extras.length) return "";
  const expanded = !!state.expandedExtraArtifactsBySample[sample.sample_id];
  return `
    <div class="extra-artifacts">
      <button class="secondary" data-extra-toggle="${escapeHtml(sample.sample_id)}" type="button">${expanded ? "隐藏附加可视化" : `附加可视化 (${extras.length})`}</button>
      ${expanded ? `
        <div class="preview-grid extra-grid">
          ${extras.map((item) => `
            <a class="preview-slot" href="${escapeHtml(item.original_url || `/api/files/${item.id}`)}" target="_blank" rel="noreferrer">
              <span>${escapeHtml(item.kind)}</span>
              <img src="${escapeHtml(item.preview_url || `/api/files/${item.id}`)}" alt="${escapeHtml(item.kind)}" loading="lazy">
            </a>
          `).join("")}
        </div>
      ` : "<p class=\"muted\">展开后按需加载 extra_* 预览。</p>"}
    </div>
  `;
}

function renderCompareLayers(sample) {
  const layers = sample.compare_layers || [];
  if (!layers.length) return "";
  const columns = state.compareGridColumns || 3;
  return `
    <section class="compare-layer-panel">
      <div class="chart-head">
        <strong>Compare layers</strong>
        <div class="segmented">
          ${[2, 3, 4].map((count) => `
            <button class="secondary ${Number(columns) === count ? "active" : ""}" data-compare-grid-columns="${count}" type="button">${count}</button>
          `).join("")}
        </div>
      </div>
      <div class="compare-layer-grid" style="--compare-grid-columns: ${escapeHtml(columns)}">
        ${layers.map((layer) => {
          const artifact = layer.artifact || {};
          const url = artifact.preview_url || artifact.original_url;
          const href = artifact.original_url || url;
          return `
            <a class="compare-layer-tile" data-layer-video="${escapeHtml(sample.metadata?.video_name || "")}" data-layer-frame="${escapeHtml(sample.frame_index ?? "")}" href="${escapeHtml(href || "#")}" target="_blank" rel="noreferrer">
              <span class="chip-row">
                <small>${escapeHtml(layer.track_label || `run-${layer.track_run_id || "-"}`)}</small>
                <strong>${escapeHtml(layer.kind || "-")}</strong>
              </span>
              ${url ? `<img src="${escapeHtml(url)}" alt="${escapeHtml(`${layer.track_label || ""} ${layer.kind || ""}`)}" loading="lazy">` : "<p class=\"muted\">no preview</p>"}
            </a>
          `;
        }).join("")}
      </div>
    </section>
  `;
}

function sampleLayerOptions(run, sample) {
  const groups = previewGroupsForRun(run);
  const options = [];
  for (const group of Object.values(groups)) {
    for (const [kind, label] of group.items) {
      options.push([kind, label]);
    }
  }
  // When a sample's artifacts are loaded, hide layers that were not saved
  // (e.g. warp/blend when save_warp_blend was off) so the slot pickers only
  // offer kinds that actually resolve to an image.
  const artifacts = sample?.artifacts;
  if (artifacts && Object.keys(artifacts).length) {
    const present = options.filter(([kind]) => artifacts[kind]);
    if (present.length) return present;
  }
  return options;
}

function slotSelection(sampleId, options) {
  const stored = state.slotSelectionBySample[sampleId];
  if (stored) return stored;
  const kinds = options.map(([kind]) => kind);
  const left = kinds.includes("gt") ? "gt" : kinds[0] || "pred";
  const right = kinds.includes("pred") ? "pred" : (kinds[1] || kinds[0] || "pred");
  return { left, right };
}

function renderBigSlot(sample, options, slot, selectedKind) {
  const optionHtml = options
    .map(([kind, label]) => `<option value="${escapeHtml(kind)}" ${kind === selectedKind ? "selected" : ""}>${escapeHtml(label)}</option>`)
    .join("");
  const artifact = sample.artifacts?.[selectedKind];
  const label = (options.find(([kind]) => kind === selectedKind) || [selectedKind, selectedKind])[1];
  let body;
  if (!artifact) {
    body = "<p class=\"muted\">暂无</p>";
  } else {
    const url = artifact.preview_url || artifact.original_url;
    const href = artifact.original_url || url;
    body = `<a href="${escapeHtml(href)}" target="_blank" rel="noreferrer"><img src="${escapeHtml(url)}" alt="${escapeHtml(label)}" loading="lazy"></a><a class="muted" href="${escapeHtml(href)}" target="_blank" rel="noreferrer">打开原图</a>`;
  }
  return `
    <div class="big-slot">
      <div class="big-slot-head">
        <select data-slot="${escapeHtml(slot)}" data-slot-sample="${escapeHtml(sample.sample_id)}">${optionHtml}</select>
      </div>
      <div class="big-slot-body">${body}</div>
    </div>
  `;
}

function comparePreviewPayload(sample) {
  // Merge loaded detail (artifacts, files) over the timeline sample so a slot
  // can render as soon as its own detail arrives, independent of siblings.
  const detail = sampleDetail(sample.sample_id);
  if (!detail && sample.has_artifacts !== false) {
    loadSampleDetail(sample.sample_id);
  }
  if (!detail) return { ...sample, _loading: sample.has_artifacts !== false };
  return {
    ...detail,
    ...sample,
    artifacts: detail.artifacts || sample.artifacts || {},
    extra_artifacts: detail.extra_artifacts || sample.extra_artifacts || [],
    compare_layers: detail.compare_layers || sample.compare_layers || [],
    sample_files: detail.sample_files || sample.sample_files || {},
    load_error: detail.load_error,
    _loading: false,
  };
}

function renderCompareTrackSlot(payload) {
  const label = payload.track_label || `run-${payload.track_index ?? "-"}`;
  let body;
  if (payload._loading) {
    body = "<div class=\"sample-loading skeleton-card\" aria-busy=\"true\"><span></span><span></span></div>";
  } else if (payload.load_error) {
    body = `<p class="muted">加载失败: ${escapeHtml(payload.load_error)}</p>`;
  } else {
    const artifact = payload.artifacts?.pred;
    if (!artifact) {
      body = "<p class=\"muted\">暂无</p>";
    } else {
      const url = artifact.preview_url || artifact.original_url;
      const href = artifact.original_url || url;
      body = `<a href="${escapeHtml(href)}" target="_blank" rel="noreferrer"><img src="${escapeHtml(url)}" alt="${escapeHtml(label)}" loading="lazy"></a><a class="muted" href="${escapeHtml(href)}" target="_blank" rel="noreferrer">打开原图</a>`;
    }
  }
  return `
    <div class="big-slot">
      <div class="big-slot-head"><strong class="compare-track-title">${escapeHtml(label)}</strong>${renderSampleMetrics(payload)}</div>
      <div class="big-slot-body">${body}</div>
    </div>
  `;
}

function renderCompareGtSlot(sample) {
  const artifact = sample.artifacts?.gt;
  let body;
  if (sample._loading) {
    body = "<div class=\"sample-loading skeleton-card\" aria-busy=\"true\"><span></span><span></span></div>";
  } else if (artifact) {
    const url = artifact.preview_url || artifact.original_url;
    const href = artifact.original_url || url;
    body = `<a href="${escapeHtml(href)}" target="_blank" rel="noreferrer"><img src="${escapeHtml(url)}" alt="GT" loading="lazy"></a><a class="muted" href="${escapeHtml(href)}" target="_blank" rel="noreferrer">打开原图</a>`;
  } else {
    const href = sample.sample_files?.gt || `/api/sample-files/${sample.sample_id}/gt`;
    body = `<a class="muted" href="${escapeHtml(href)}" target="_blank" rel="noreferrer">打开原图</a>`;
  }
  return `
    <div class="big-slot compare-gt-slot">
      <div class="big-slot-head"><strong class="compare-track-title">GT</strong></div>
      <div class="big-slot-body">${body}</div>
    </div>
  `;
}

function renderCompareFrameGroup(video, sample) {
  const tracks = compareFrameSiblings(video, sample);
  const gtPayload = comparePreviewPayload(tracks[0]);
  const trackPayloads = tracks.map((item) => comparePreviewPayload(item));
  const columns = tracks.length + 1;
  return `
    <div class="sample-meta">
      <strong>frame ${escapeHtml(sample.frame_index)}</strong>
      <span>${sample.timestamp === null || sample.timestamp === undefined ? "-" : `${formatNumber(sample.timestamp)}s`}</span>
      <span class="muted">${escapeHtml(tracks.length)} 条轨道 + GT</span>
    </div>
    <div class="big-slots compare-row" style="--compare-slot-columns: ${escapeHtml(columns)}">
      ${renderCompareGtSlot(gtPayload)}
      ${trackPayloads.map((payload) => renderCompareTrackSlot(payload)).join("")}
    </div>
    ${trackPayloads.flatMap((payload) => (payload.compare_layers?.length ? [renderCompareLayers(payload)] : [])).join("")}
  `;
}

function renderSamplePreview(sample, video) {
  const artifactsPending = sample.has_artifacts === false
    && !sample.error
    && !state.selectedRun?.artifact_cleaned_at
    && !TERMINAL_STATUSES.has(state.selectedRun?.status);
  if (artifactsPending) {
    return `
      <div class="sample-meta">
        <strong>${escapeHtml(sample.sample_name)}</strong>
        <span>frame ${escapeHtml(sample.frame_index)}</span>
        <span>${sample.timestamp === null || sample.timestamp === undefined ? "-" : `${formatNumber(sample.timestamp)}s`}</span>
        ${renderSampleMetrics(sample)}
      </div>
      <div class="artifact-pending sample-loading" aria-live="polite" aria-busy="true">
        <div class="skeleton-card"><span></span><span></span></div>
        <p class="muted">产物生成中，保存完成后会自动加载。</p>
      </div>
    `;
  }
  if (video && isCompareRun(state.selectedRun)) {
    return renderCompareFrameGroup(video, sample);
  }
  const detail = sampleDetail(sample.sample_id);
  if (!detail && sample.has_artifacts !== false) {
    loadSampleDetail(sample.sample_id);
  }
  if (sample.has_artifacts === false) {
    let reason;
    if (sample.error) {
      reason = `样本处理失败: ${escapeHtml(sample.error.error_type || "Error")}: ${escapeHtml(sample.error.message || "unknown")}`;
    } else if (state.selectedRun?.artifact_cleaned_at) {
      reason = "这个 Run 的产物已清理；如需重新查看预览，请重试重新生成。";
    } else {
      reason = "这个样本当前没有可用产物。";
    }
    return `
      <div class="sample-meta">
        <strong>${escapeHtml(sample.sample_name)}</strong>
        <span>frame ${escapeHtml(sample.frame_index)}</span>
        <span>${sample.timestamp === null || sample.timestamp === undefined ? "-" : `${formatNumber(sample.timestamp)}s`}</span>
        ${renderSampleMetrics(sample)}
      </div>
      <div class="message warn"><p><strong>没有可加载的产物</strong>: ${reason}</p></div>
    `;
  }
  const payload = detail
    ? {
        ...detail,
        ...sample,
        artifacts: detail.artifacts || sample.artifacts || {},
        extra_artifacts: detail.extra_artifacts || sample.extra_artifacts || [],
        compare_layers: detail.compare_layers || sample.compare_layers || [],
        sample_files: detail.sample_files || sample.sample_files || {},
        load_error: detail.load_error,
      }
    : sample;
  const options = sampleLayerOptions(state.selectedRun, payload);
  const selection = slotSelection(payload.sample_id, options);
  const loadState = detail
    ? (payload.load_error
        ? `<div class="message error"><p><strong>样本产物加载失败</strong>: ${escapeHtml(payload.load_error)}</p></div>`
        : "")
    : "<div class=\"sample-loading skeleton-card\" aria-busy=\"true\"><span></span><span></span></div>";
  return `
    <div class="sample-meta">
      <strong>${escapeHtml(payload.sample_name)}</strong>
      <span>frame ${escapeHtml(payload.frame_index)}</span>
      <span>${payload.timestamp === null || payload.timestamp === undefined ? "-" : `${formatNumber(payload.timestamp)}s`}</span>
      ${renderSampleMetrics(payload)}
    </div>
    ${loadState}
    <div class="big-slots-head">
      <div class="segmented">
        <button class="secondary ${state.compareSlotLayout === "side" ? "active" : ""}" data-slot-layout="side" type="button">左右</button>
        <button class="secondary ${state.compareSlotLayout === "stack" ? "active" : ""}" data-slot-layout="stack" type="button">上下</button>
      </div>
    </div>
    <div class="big-slots ${state.compareSlotLayout === "stack" ? "stacked" : ""}">
      ${renderBigSlot(payload, options, "left", selection.left)}
      ${renderBigSlot(payload, options, "right", selection.right)}
    </div>
    ${renderCompareLayers(payload)}
    ${renderExtraArtifacts(payload)}
  `;
}

function setSampleIndex(videoName, index) {
  if (!state.selectedRun) return;
  const runId = Number(state.selectedRun.id);
  const video = state.runVideoTimelines[`${runId}:${videoName}`];
  if (!video) return;
  const max = Math.max(0, (video.samples || []).length - 1);
  const key = `${runId}:${videoName}`;
  const nextIndex = Math.max(0, Math.min(max, index));
  if (Number(state.selectedSampleByVideo[key] || 0) !== nextIndex) {
    abortSampleRequestsForRun(runId);
  }
  state.selectedSampleByVideo[key] = nextIndex;
  // Update only the frame-dependent region so the video players are not
  // recreated (which would reload and stutter). Fall back to a full render if
  // the region is not on the page (e.g. video not yet rendered).
  if (!updateFrameRegion()) renderRunDetail();
  syncBrowserRoute({ view: "runs", replace: true });
}

async function setGlobalSampleIndex(videoName, globalIndex) {
  if (!state.selectedRun) return;
  const runId = Number(state.selectedRun.id);
  const key = `${runId}:${videoName}`;
  let video = state.runVideoTimelines[key];
  if (!video) video = await loadRunVideoTimeline(runId, videoName);
  if (!video || Number(state.selectedRun?.id) !== runId) return;
  const total = Number(video.sample_count || 0);
  const target = Math.max(0, Math.min(Math.max(0, total - 1), Number(globalIndex)));
  let windowStart = Number(video.window_start || 0);
  let localIndex = target - windowStart;
  if (localIndex < 0 || localIndex >= (video.samples || []).length) {
    const centeredStart = Math.max(0, Math.min(Math.max(0, total - TIMELINE_WINDOW_SIZE), target - Math.floor(TIMELINE_WINDOW_SIZE / 2)));
    abortSampleRequestsForRun(runId);
    video = await loadRunVideoTimeline(runId, videoName, { windowStart: centeredStart });
    if (!video || Number(state.selectedRun?.id) !== runId) return;
    windowStart = Number(video.window_start || 0);
    localIndex = target - windowStart;
  }
  state.selectedVideoByRun[runId] = video.video_name;
  state.selectedSampleByVideo[`${runId}:${video.video_name}`] = Math.max(0, Math.min((video.samples || []).length - 1, localIndex));
  if (!updateFrameRegion()) renderRunDetail();
  syncBrowserRoute({ view: "runs", replace: true });
}

function activeVideoElements() {
  return Array.from(document.querySelectorAll(".sample-viewer .video-artifact video"));
}

function syncActiveVideos(action) {
  const videos = activeVideoElements();
  if (!videos.length) return;
  const leader = videos.find((video) => !Number.isNaN(video.currentTime)) || videos[0];
  const currentTime = leader.currentTime || 0;
  for (const video of videos) {
    if (Math.abs((video.currentTime || 0) - currentTime) > 0.05) {
      video.currentTime = currentTime;
    }
    if (action === "play") video.play().catch(() => {});
    if (action === "pause") video.pause();
  }
}

function highlightTimelineFrame(frameIndex) {
  document.querySelectorAll(".timeline-hover").forEach((item) => item.classList.remove("timeline-hover"));
  if (frameIndex === null || frameIndex === undefined || frameIndex === "") return;
  document.querySelectorAll(`[data-frame-index="${CSS.escape(String(frameIndex))}"]`).forEach((item) => {
    item.classList.add("timeline-hover");
  });
}

async function setSampleByFrame(videoName, frameIndex) {
  if (!state.selectedRun) return;
  const runId = Number(state.selectedRun.id);
  abortSampleRequestsForRun(runId);
  let video = state.runVideoTimelines[`${runId}:${videoName}`];
  if (!video) {
    video = await loadRunVideoTimeline(runId, videoName);
  }
  if (!video || Number(state.selectedRun?.id) !== runId) return;
  let index = (video.samples || []).findIndex((sample) => Number(sample.frame_index) === Number(frameIndex));
  if (index < 0) {
    const windowStart = Math.max(0, Number(frameIndex) - 150);
    video = await loadRunVideoTimeline(runId, videoName, { windowStart });
    if (!video || Number(state.selectedRun?.id) !== runId) return;
    index = (video.samples || []).findIndex((sample) => Number(sample.frame_index) === Number(frameIndex));
  }
  state.selectedVideoByRun[runId] = video.video_name;
  state.selectedSampleByVideo[`${runId}:${video.video_name}`] = Math.max(0, index);
  renderRunDetail();
  syncBrowserRoute({ view: "runs", replace: true });
}

async function cancelRun(runId) {
  await api(`/api/runs/${runId}/cancel`, { method: "POST", body: "{}" });
  toast("已请求取消");
  await refreshRunsOnly();
}

async function refreshRunResults(runId = null) {
  if (runId !== null && Number(state.selectedRun?.id) !== Number(runId)) {
    await selectRun(runId, { quiet: true });
  }
  await refreshRunsOnly({ forceSelected: !!state.selectedRun });
  toast(state.selectedRun ? `Run #${state.selectedRun.id} 结果已刷新` : "运行列表已刷新");
}

async function retryRun(runId) {
  const created = await api(`/api/runs/${runId}/retry`, { method: "POST", body: "{}" });
  toast(`重试 Run #${created.run_id} 已开始`);
  await refreshRunsOnly();
  await selectRun(created.run_id);
}

async function cloneRunWithCurrentInputs(runId) {
  let created;
  let requestBody = {};
  for (let attempt = 0; attempt < 3 && !created; attempt += 1) {
    try {
      created = await api(`/api/runs/${runId}/clone`, {
        method: "POST",
        body: JSON.stringify(requestBody),
      });
    } catch (error) {
      const payload = error.payload || {};
      const workload = payload.workload;
      if (
        Number(error.status) !== 409
        || String(payload.type || payload.error?.type || "") !== "WorkloadRiskConfirmationRequired"
        || workload?.risk_level !== "high"
      ) {
        throw error;
      }
      if (!highRiskWorkloadConfirmation(workload)) {
        toast("已取消 Clone，当前文件与配置保持不变");
        return;
      }
      requestBody = {
        risk_ack_fingerprint: String(workload.risk_fingerprint),
      };
    }
  }
  if (!created) {
    throw new Error("Clone 工作量在确认期间连续变化，请检查当前输入后重试");
  }
  toast(`已按当前输入启动 Clone Run #${created.run_id}`);
  await refreshRunsOnly();
  await selectRun(created.run_id);
}

async function retryRunMetrics(runId) {
  await api(`/api/runs/${runId}/metrics/retry`, { method: "POST", body: "{}" });
  toast(`Run #${runId} 的失败/不可用指标已重新排队`);
  await refreshRunsOnly({ forceSelected: Number(state.selectedRun?.id) === Number(runId) });
}

function runPurgeBytes(value) {
  const bytes = Number(value || 0);
  return bytes > 0 ? formatBytes(bytes) : "0 B";
}

function runPurgeDependencyText(dependencies) {
  const rows = dependencies && typeof dependencies === "object" ? dependencies : {};
  const sections = [
    ["Campaign", rows.campaign_ids],
    ["Compare Run", rows.compare_run_ids],
    ["活动 Job", rows.active_job_ids],
  ];
  const visible = sections.flatMap(([label, values]) => {
    const ids = Array.isArray(values) ? values.map(Number).filter((value) => value > 0) : [];
    if (!ids.length) return [];
    const shown = ids.slice(0, 10).map((value) => `#${value}`).join(", ");
    return [`${label}: ${shown}${ids.length > 10 ? ` 等 ${ids.length} 项` : ""}`];
  });
  return visible.length ? visible.join("；") : "无已知依赖";
}

function runPurgeReasonLabel(reason) {
  const labels = {
    ready: "可执行",
    run_not_terminal: "Run 尚未结束",
    active_worker: "仍有活动 Worker/Job",
    artifacts_already_cleaned: "产物已清理",
    run_already_deleted: "Run 已删除",
  };
  return labels[String(reason || "")] || String(reason || "未知原因");
}

function runPurgePreviewMessage(preview) {
  const requestType = String(preview?.request_type || "");
  const operation = requestType === "cleanup_artifacts" ? "清理可视产物" : "永久删除 Run";
  const summary = preview?.summary || {};
  const runs = Array.isArray(preview?.runs) ? preview.runs : [];
  const runLines = runs.slice(0, 12).map((run) => {
    const bytes = run.bytes || {};
    const status = run.allowed ? "可执行" : `阻止：${runPurgeReasonLabel(run.reason)}`;
    const name = String(run.name || "").replaceAll(/\s+/g, " ").slice(0, 80);
    return `• #${Number(run.run_id)}${name ? ` “${name}”` : ""} · ${run.status || "-"} · ${status} · 目录 ${runPurgeBytes(bytes.exclusive_run_bytes)}`;
  });
  if (runs.length > 12) runLines.push(`• 另有 ${runs.length - 12} 条 Run（已计入汇总）`);
  const immediate = summary.estimated_reclaimable_bytes ?? summary.exclusive_run_bytes;
  return [
    `${operation}影响预览`,
    "",
    ...runLines,
    "",
    `预计可回收（Run 目录）：${runPurgeBytes(immediate)}`,
    `引用缓存总量：${runPurgeBytes(summary.referenced_cache_bytes)}`,
    `共享缓存（保留）：${runPurgeBytes(summary.shared_cache_bytes)}`,
    `仅由本次选择持有的缓存：${runPurgeBytes(summary.exclusive_cache_bytes)}`,
    `缓存宽限期后潜在可回收：${runPurgeBytes(summary.potential_cache_bytes_after_grace)}`,
    `依赖：${runPurgeDependencyText(summary.dependencies)}`,
  ].join("\n");
}

async function requestRunPurgePreview(requestType, runIds) {
  const normalizedIds = Array.from(new Set((runIds || []).map(Number).filter((value) => value > 0))).sort((a, b) => a - b);
  if (!normalizedIds.length) throw new Error("请选择至少一条 Run 记录");
  const preview = await api("/api/run-purge/preview", {
    method: "POST",
    body: JSON.stringify({ request_type: requestType, run_ids: normalizedIds }),
  });
  if (!preview?.preview_token) throw new Error("删除影响预览未返回确认令牌，请重试");
  return preview;
}

function confirmRunPurgePreview(preview) {
  const message = runPurgePreviewMessage(preview);
  const blocked = (preview.runs || []).filter((run) => !run.allowed);
  if (blocked.length) {
    window.alert(`${message}\n\n当前有 ${blocked.length} 条 Run 不允许执行，请先处理上面的阻止原因。`);
    return false;
  }
  const consequence = preview.request_type === "cleanup_artifacts"
    ? "Run 记录会保留，但原图、视频和 Diff 将不再可查看。"
    : "任务记录与可视产物会进入持久清理队列。此操作不可撤销。";
  return window.confirm(`${message}\n\n${consequence}\n\n确认继续？`);
}

async function withRunPurgePreview(requestType, runIds, mutation) {
  if (state.runPurgeSubmitting) {
    toast("正在生成或提交删除影响预览，请勿重复点击");
    return null;
  }
  state.runPurgeSubmitting = true;
  try {
    toast("正在生成删除影响预览…");
    const preview = await requestRunPurgePreview(requestType, runIds);
    if (!confirmRunPurgePreview(preview)) return null;
    return await mutation(String(preview.preview_token));
  } catch (error) {
    toast(error.message || String(error));
    return null;
  } finally {
    state.runPurgeSubmitting = false;
  }
}

async function deleteRun(runId) {
  const result = await withRunPurgePreview("delete_run", [runId], (previewToken) =>
    api(`/api/runs/${runId}?preview_token=${encodeURIComponent(previewToken)}`, { method: "DELETE" }));
  if (!result) return;
  toast(result.deleted ? `Run #${runId} 已清理并删除` : `Run #${runId} 已进入删除队列`);
  await refreshRunsOnly();
}

async function renameRun(runId) {
  const run = state.runs.find((item) => Number(item.id) === Number(runId)) || state.selectedRun;
  const current = run?.name || "";
  const next = window.prompt("重命名运行记录", current);
  if (next === null) return;
  const trimmed = next.trim();
  if (!trimmed || trimmed === current) return;
  const updated = await api(`/api/runs/${runId}/rename`, {
    method: "POST",
    body: JSON.stringify({ name: trimmed }),
  });
  toast(`已重命名为 ${updated.run?.name || trimmed}`);
  if (Number(state.selectedRun?.id) === Number(runId)) {
    state.selectedRun = updated.run || state.selectedRun;
  }
  await refreshRunsOnly();
  if (Number(state.selectedRun?.id) === Number(runId)) renderRunDetail();
}

async function batchDeleteRuns() {
  const ids = Array.from(state.selectedRunIds);
  if (!ids.length) return;
  const result = await withRunPurgePreview("delete_run", ids, (previewToken) =>
    api(`/api/runs/batch-delete`, {
      method: "POST",
      body: JSON.stringify({ run_ids: ids, preview_token: previewToken }),
    }));
  if (!result) return;
  const accepted = new Set((result.accepted || result.deleted || []).map(Number));
  const deleted = new Set((result.deleted || []).map(Number));
  if (state.selectedRun && deleted.has(Number(state.selectedRun.id))) {
    state.selectedRun = null;
  }
  state.selectedRunIds.clear();
  toast(`已受理 ${accepted.size} 条删除${result.failures?.length ? `，${result.failures.length} 条失败` : ""}`);
  await refreshRunsOnly();
  if (!state.selectedRun) renderEmptyRunDetail();
}

async function cleanupRunArtifacts(runId) {
  const result = await withRunPurgePreview("cleanup_artifacts", [runId], (previewToken) =>
    api(`/api/runs/${runId}/cleanup-artifacts`, {
      method: "POST",
      body: JSON.stringify({ preview_token: previewToken }),
    }));
  if (!result) return;
  toast(result.artifact_cleaned ? `Run #${runId} 产物已清理` : `Run #${runId} 已进入清理队列`);
  await refreshRunsOnly({ forceSelected: Number(state.selectedRun?.id) === Number(runId) });
}

async function submitRunFeedback(runId, form) {
  const data = formData(form);
  const username = String(data.username || "").trim();
  const issue = String(data.issue || "").trim();
  const rating = data.rating ? Number(data.rating) : null;
  const video = String(data.video || "").trim();
  const trackLabel = String(data.track_label || "").trim();
  if (!issue && rating === null) {
    toast("请至少填写评分或问题");
    return;
  }
  if (!video) {
    toast("请先选择要评分的视频");
    return;
  }
  // Remember the name so the reviewer doesn't retype it on every run.
  state.feedbackUsername = username;
  await api(`/api/runs/${runId}/feedback`, {
    method: "POST",
    body: JSON.stringify({ username, rating, issue, video, track_label: trackLabel }),
  });
  toast("反馈已提交");
  if (Number(state.selectedRun?.id) === Number(runId)) {
    await selectRun(runId, { quiet: true, preserveUi: false });
  }
}

async function submitFeedbackEdit(runId, feedbackId, form) {
  const data = formData(form);
  const username = String(data.username || "").trim();
  const issue = String(data.issue || "").trim();
  const rating = data.rating ? Number(data.rating) : null;
  if (!issue && rating === null) {
    toast("请至少填写评分或问题");
    return;
  }
  await api(`/api/runs/${runId}/feedback/${feedbackId}`, {
    method: "POST",
    // Send rating explicitly (possibly null) so clearing it is honored.
    body: JSON.stringify({ username, rating, issue }),
  });
  state.editingFeedback = null;
  toast("反馈已更新");
  if (Number(state.selectedRun?.id) === Number(runId)) {
    await selectRun(runId, { quiet: true, preserveUi: false });
  }
}

async function deleteRunFeedback(runId, feedbackId) {
  await api(`/api/runs/${runId}/feedback/${feedbackId}`, { method: "DELETE" });
  if (Number(state.editingFeedback) === Number(feedbackId)) state.editingFeedback = null;
  toast("反馈已删除");
  if (Number(state.selectedRun?.id) === Number(runId)) {
    await selectRun(runId, { quiet: true, preserveUi: false });
  }
}

async function loadStats() {
  const f = state.statsFilters || {};
  const params = new URLSearchParams();
  if (f.dataset) params.set("dataset", f.dataset);
  if (f.model) params.set("model", f.model);
  if (f.checkpoint) params.set("checkpoint", f.checkpoint);
  if (f.video) params.set("video", f.video);
  const qs = params.toString();
  state.feedbackStats = await api(`/api/feedback${qs ? `?${qs}` : ""}`);
  renderStats();
}

// Render one 0.25-step rating histogram from a distribution map keyed by
// "1.00".."5.00". Shared by the overall chart and per-group (video/checkpoint)
// charts so they read the same.
function renderRatingHistogram(distribution) {
  const keys = Array.from({ length: 17 }, (_, i) => formatRatingKey(5 - i * 0.25));
  const total = keys.reduce((sum, key) => sum + Number(distribution[key] || 0), 0);
  if (!total) return '<p class="muted">当前范围内还没有评分。</p>';
  const maxCount = Math.max(1, ...keys.map((key) => Number(distribution[key] || 0)));
  return `
    <div class="rating-bars">
      ${keys.map((key) => {
        const count = Number(distribution[key] || 0);
        const width = Math.round((count / maxCount) * 100);
        return `
          <div class="rating-bar-row">
            <span class="rating-bar-label">${escapeHtml(key)}</span>
            <span class="rating-bar-track"><span class="rating-bar-fill" style="width: ${width}%"></span></span>
            <span class="rating-bar-count">${escapeHtml(count)}</span>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

// The backend distribution is keyed on 0.25-step strings; keep the frontend key
// format identical so lookups line up.
function formatRatingKey(value) {
  return Number(value).toFixed(2);
}

function statsFilterControls(options, filters) {
  const select = (name, label, values, current) => `
    <label>
      <span>${escapeHtml(label)}</span>
      <select data-stats-filter="${name}">
        <option value="">全部</option>
        ${values.map((value) => `<option value="${escapeHtml(value)}" ${String(current) === String(value) ? "selected" : ""}>${escapeHtml(value)}</option>`).join("")}
      </select>
    </label>
  `;
  return `
    <section class="stats-filters">
      ${select("dataset", "数据集", options.datasets || [], filters.dataset)}
      ${select("model", "模型", options.models || [], filters.model)}
      ${select("checkpoint", "权重", options.checkpoints || [], filters.checkpoint)}
      ${select("video", "视频", options.videos || [], filters.video)}
      <button class="secondary" data-stats-filter-reset type="button">清除筛选</button>
    </section>
  `;
}

// A collapsible per-group section: summary table plus one rating histogram per
// group row, so "某个视频的评分分布" / "某个权重的评分分布" are both first-class.
function renderGroupedDistributions(title, rows, labelFor) {
  if (!rows.length) return "";
  return `
    <section class="stats-block">
      <h3>${escapeHtml(title)}</h3>
      <div class="stats-group-grid">
        ${rows.map((row) => `
          <article class="stats-group-card">
            <header>
              <strong>${escapeHtml(labelFor(row))}</strong>
              <span class="muted">${escapeHtml(row.count || 0)} 条 · 均分 ${row.average_rating === null || row.average_rating === undefined ? "-" : formatRating(row.average_rating)}</span>
            </header>
            ${renderRatingHistogram(row.rating_distribution || {})}
          </article>
        `).join("")}
      </div>
    </section>
  `;
}

function renderStats() {
  const stats = state.feedbackStats;
  const host = $("stats-content");
  if (!host) return;
  if (!stats) {
    host.innerHTML = "<p class=\"muted\">正在加载统计数据...</p>";
    return;
  }
  const distribution = stats.rating_distribution || {};
  const byRun = stats.by_run || [];
  const byUser = stats.by_user || [];
  const byVideo = stats.by_video || [];
  const byCheckpoint = stats.by_checkpoint || [];
  const recent = stats.recent || [];
  const options = stats.filter_options || {};
  const filters = stats.filters || {};
  const hasFeedback = Number(stats.total || 0) > 0;
  host.innerHTML = `
    ${statsFilterControls(options, state.statsFilters)}
    <div class="summary-grid">
      <div><span>反馈总数</span><strong>${escapeHtml(stats.total || 0)}</strong></div>
      <div><span>打分数</span><strong>${escapeHtml(stats.rating_count || 0)}</strong></div>
      <div><span>平均分</span><strong>${stats.average_rating === null || stats.average_rating === undefined ? "-" : formatRating(stats.average_rating)}</strong></div>
      <div><span>问题数</span><strong>${escapeHtml(stats.issue_count || 0)}</strong></div>
    </div>
    ${Object.keys(filters).length ? `<p class="muted">已筛选：${Object.entries(filters).map(([k, v]) => `${escapeHtml(k)}=${escapeHtml(v)}`).join("， ")}</p>` : ""}
    ${hasFeedback ? `<section class="stats-block">
      <h3>总体评分分布（0.25 分度）</h3>
      ${renderRatingHistogram(distribution)}
    </section>` : '<div class="empty-state"><strong>还没有主观反馈</strong><p class="muted">在 Run Detail 中提交评分或问题后，这里会显示汇总。</p></div>'}
    ${hasFeedback ? renderGroupedDistributions("按视频的评分分布", byVideo, (row) => row.video || "（未指定）") : ""}
    ${hasFeedback ? renderGroupedDistributions("按模型 / 权重的评分分布", byCheckpoint, (row) => `${row.model_name || "?"} / ${row.checkpoint || "-"}`) : ""}
    ${hasFeedback ? `
    <section class="stats-block">
      <h3>按用户</h3>
      <div class="table compact-table">${table(byUser, [
        { label: "用户名", render: (row) => escapeHtml(row.username || "匿名") },
        { label: "反馈数", render: (row) => escapeHtml(row.count || 0) },
        { label: "打分数", render: (row) => escapeHtml(row.rating_count || 0) },
        { label: "平均分", render: (row) => row.average_rating === null || row.average_rating === undefined ? "-" : formatRating(row.average_rating) },
        { label: "问题数", render: (row) => escapeHtml(row.issues || 0) },
      ])}</div>
    </section>
    <section class="stats-block">
      <h3>按运行记录</h3>
      <div class="table compact-table">${table(byRun, [
        { label: "Run", render: (row) => `#${escapeHtml(row.run_id)}` },
        { label: "名称", render: (row) => escapeHtml(row.run_name || "-") },
        { label: "反馈数", render: (row) => escapeHtml(row.count || 0) },
        { label: "平均分", render: (row) => row.average_rating === null || row.average_rating === undefined ? "-" : formatRating(row.average_rating) },
        { label: "问题数", render: (row) => escapeHtml(row.issues || 0) },
        { label: "操作", render: (row) => `<button class="view-detail-btn" data-stats-run="${escapeHtml(row.run_id)}" type="button">查看 →</button>` },
      ])}</div>
    </section>
    <section class="stats-block">
      <h3>最近反馈</h3>
      <div class="feedback-list">
        ${recent.length ? recent.map((item) => `
          <article class="feedback-item">
            <div class="feedback-item-head">
              <strong>${escapeHtml(item.username || "匿名")}</strong>
              ${ratingStars(item.rating)}
              <span class="muted">#${escapeHtml(item.run_id)} ${escapeHtml(item.run_name || "")}</span>
            </div>
            ${[item.video ? `视频 ${item.video}` : "", item.model_name ? `模型 ${item.model_name}` : "", item.checkpoint ? `权重 ${item.checkpoint}` : ""].filter(Boolean).length ? `<div class="feedback-tags">${[item.video ? `视频 ${item.video}` : "", item.model_name ? `模型 ${item.model_name}` : "", item.checkpoint ? `权重 ${item.checkpoint}` : ""].filter(Boolean).map((text) => `<span class="feedback-tag">${escapeHtml(text)}</span>`).join("")}</div>` : ""}
            ${item.issue ? `<p class="feedback-issue">${escapeHtml(item.issue)}</p>` : "<p class=\"muted\">无问题描述。</p>"}
          </article>
        `).join("") : "<p class=\"muted\">还没有反馈。</p>"}
      </div>
    </section>
    ` : ""}
  `;
}
