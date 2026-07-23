const Shared = window.VFIEvalShared;
if (!Shared) throw new Error("VFIEval shared frontend primitives failed to load");

const STATUS_LABELS = {
  decoding: "解码中",
  queued: "排队中",
  running: "推理中",
  completed: "已完成",
  failed: "失败",
  cancel_requested: "取消中",
  canceled: "已取消",
  metric_queued: "评测排队",
  metric_running: "评测中",
  finalize_queued: "视频合成排队",
  finalizing: "视频合成中",
};

const TERMINAL_STATUSES = new Set(["completed", "failed", "canceled"]);
const METRICS = ["lpips_vit_patch", "lpips_convnext", "vmaf", "cgvqm"];
const LPIPS_PAIR = ["lpips_vit_patch", "lpips_convnext"];
const LPIPS_PAIR_SET = new Set(LPIPS_PAIR);
const TIMELINE_WINDOW_SIZE = 160;
const INFERENCE_DRAFT_KEY = "vfieval:inference-draft";
const INFERENCE_DRAFT_VERSION = 2;
const ROUTE_VIEWS = new Set(["create", "compare", "media", "evaluations", "runs", "stats"]);
const PREVIEW_GROUPS_MODEL = {
  images: { label: "图像", items: [["gt", "GT"], ["pred", "Pred"], ["difference", "Diff"]] },
  flow: { label: "Flow", items: [["flowt_0", "Flow t->0"], ["flowt_1", "Flow t->1"]] },
  mask: { label: "Mask", items: [["mask0", "Mask0"], ["mask1", "Mask1"]] },
  warp: { label: "Warp", items: [["warp0", "Warp0"], ["warp1", "Warp1"], ["blend", "Blend"]] },
};
const PREVIEW_GROUPS_COMPARE = {
  images: { label: "图像", items: [["gt", "GT"], ["pred", "Pred"]] },
};

const state = {
  modelFiles: [],
  videoGroups: [],
  checkpoints: [],
  checkpointsByModel: {},
  checkpointRequestsByModel: {},
  devices: null,
  executionDefaultsResolved: false,
  runs: [],
  runsPage: {
    page: 1,
    page_size: 30,
    page_count: 1,
    total: 0,
    active_total: 0,
  },
  runFilters: { q: "", status: "", run_type: "", model: "" },
  runFilterTimer: null,
  metricHealth: null,
  catalogSync: null,
  catalogSyncPromise: null,
  catalogRefreshPromise: null,
  catalogRefreshIncludeMedia: false,
  preflight: null,
  preflightTimer: null,
  selectedRun: null,
  metricSummary: null,
  runVideosPage: null,
  runVideoTimelines: {},
  compareInputsByRun: {},
  timelineWindowStartByVideo: {},
  sampleDetails: {},
  sampleDetailLoading: {},
  selectedGroups: new Set(),
  videoPages: {},
  videoPageLoading: {},
  videoPageQuery: {},
  videoPageSort: {},
  videoSelectionToken: "",
  videoSelectionGroupsKey: "",
  videoSelectionTotal: 0,
  videoSelectionGroupCounts: {},
  videoSelectionValidated: false,
  videoSelectionLoading: false,
  videoSelectionRequestGeneration: 0,
  expandedVideoGroups: new Set(),
  runVideoPageByRun: {},
  selectedVideoByRun: {},
  selectedSampleByVideo: {},
  selectedMetricByRun: {},
  runMetaCollapsed: false,
  selectedArtifactGroupBySample: {},
  expandedExtraArtifactsBySample: {},
  selectedCudaDevices: new Set(),
  selectedNpuDevices: new Set(),
  compareItemGroups: [],
  compareItems: [],
  compareItemsMeta: null,
  comparePredictions: [],
  comparePredByMember: {},
  compareSourcesLoaded: false,
  compareItemQuery: "",
  compareItemPage: 1,
  selectedCompareGroupId: "",
  selectedCompareItemId: null,
  selectedCompareItemSnapshot: null,
  selectedComparePredMembers: new Set(),
  compareSourceRequestGeneration: 0,
  preflightAbortController: null,
  preflightPayloadKey: "",
  preflightLevel: "",
  runSubmitting: false,
  runSubmitPhase: "",
  runSubmitError: "",
  runSubmissionId: "",
  comparePreflight: null,
  comparePreflightTimer: null,
  comparePreflightPayloadKey: "",
  comparePreflightAbortController: null,
  compareSubmitting: false,
  compareSubmitPhase: "",
  compareSubmitError: "",
  compareSubmissionId: "",
  sampleAbortControllers: {},
  timelineAbortController: null,
  timelineAbortRunId: null,
  runSelectAbortController: null,
  runSelectRequestGeneration: 0,
  runSelectionGeneration: 0,
  timelineRequestGeneration: 0,
  runResultGenerations: {},
  runContentRevisions: {},
  runPoll: {
    lastSuccessAt: 0,
    lastErrorAt: 0,
    consecutiveErrors: 0,
    error: "",
  },
  runsRefreshPromise: null,
  runsRefreshQueued: false,
  runsRefreshPendingForce: false,
  runsRefreshPendingPage: null,
  runsRefreshGeneration: 0,
  compareGridColumns: 3,
  slotSelectionBySample: {},
  compareSlotLayout: "side",
  selectedRunIds: new Set(),
  runPurgeSubmitting: false,
  feedbackUsername: "",
  feedbackStats: null,
  editingFeedback: null,
  statsFilters: { dataset: "", model: "", checkpoint: "", video: "" },
  mediaCollections: [],
  mediaAssets: [],
  mediaAssetsPage: { page: 1, page_size: 200, page_count: 1, total: 0 },
  mediaFilters: { q: "", role: "", source_kind: "", collection_id: "" },
  mediaFilterTimer: null,
  mediaLibraryRequestGeneration: 0,
  externalPredItemGroups: [],
  externalPredAssets: [],
  externalPredItems: [],
  selectedExternalPredGroupId: "",
  selectedExternalPredItem: null,
  selectedExternalPredAsset: null,
  externalPredItemsPage: { page: 1, page_size: 100, page_count: 1, total: 0 },
  externalPredItemRequestGeneration: 0,
  activeUpload: null,
  uploadTask: null,
  uploadPaused: false,
  runtimeHealth: null,
  inferenceDraftRestored: false,
  inferenceDraftTimer: null,
  applyingRoute: false,
};

const $ = (id) => document.getElementById(id);
const runCreationFlight = Shared.createSingleFlight();
const compareCreationFlight = Shared.createSingleFlight();

function showRequestDiagnostic(error, context = "请求") {
  if (!error || error.name === "AbortError") return;
  const host = $("request-diagnostic");
  if (!host) return;
  const requestId = String(error.request_id || error.requestId || "");
  const supportId = String(error.support_id || error.supportId || "");
  const message = String(error.message || error || "请求失败");
  const recovery = String(error.recovery_suggestion || Shared.recoverySuggestion(error));
  const diagnosticText = [
    `${context}失败：${message}`,
    `恢复建议：${recovery}`,
    requestId ? `request_id: ${requestId}` : "",
    supportId ? `support_id: ${supportId}` : "",
  ].filter(Boolean).join("\n");
  host.innerHTML = `
    <div>
      <strong>${escapeHtml(context)}失败</strong>
      <p>${escapeHtml(message)}</p>
      <p class="request-diagnostic-recovery">建议：${escapeHtml(recovery)}</p>
      <p class="request-diagnostic-ids">${requestId ? `request_id: ${escapeHtml(requestId)}` : "request_id: 未返回"}${supportId ? `<br>support_id: ${escapeHtml(supportId)}` : ""}</p>
    </div>
    <div class="request-diagnostic-copy">
      <button class="secondary" type="button" data-copy-request-diagnostic>复制诊断信息</button>
      <button class="secondary" type="button" data-dismiss-request-diagnostic>关闭</button>
    </div>
  `;
  host.classList.remove("hidden");
  host.querySelector("[data-copy-request-diagnostic]")?.addEventListener("click", () => {
    Shared.copyText(diagnosticText)
      .then(() => toast("诊断信息已复制"))
      .catch(() => toast("无法自动复制，请手动选择诊断信息"));
  });
  host.querySelector("[data-dismiss-request-diagnostic]")?.addEventListener("click", () => {
    host.classList.add("hidden");
  });
  error.diagnosticShown = true;
}

window.showRequestDiagnostic = showRequestDiagnostic;

async function api(path, options = {}) {
  return Shared.request(path, {
    fetchOptions: options,
    networkMessage: "无法连接 VFIEval 服务",
    messageFormatter: (data, response) => (
      response.status === 507 ? _formatStorageCapacityError(data) : ""
    ),
    onDiagnostic: (error) => {
      if (String(error.payload?.error?.type || "") !== "WorkloadRiskConfirmationRequired") {
        showRequestDiagnostic(error, "VFIEval 请求");
      }
    },
  });
}

function _formatStorageCapacityError(payload) {
  const cap = payload?.error?.capacity;
  if (!cap) return payload?.error?.message || "磁盘空间不足，无法创建任务";
  const free = Number(cap.free_bytes || 0);
  const needed = Number(cap.required_free_bytes || 0);
  const reserved = Number(cap.reserved_bytes || 0);
  const shortfall = Math.max(0, needed - free);
  const parts = [`磁盘空间不足，无法创建任务。可用 ${formatBytes(free)}`];
  if (reserved > 0) parts.push(`已预留 ${formatBytes(reserved)}`);
  parts.push(`还需 ${formatBytes(shortfall)}`);
  return parts.join("，") + "。请清理旧运行产物后重试。";
}

function renderDeploymentHealth(health, error = null) {
  const host = $("deployment-health");
  if (!host) return;
  if (error || !health) {
    host.className = "deployment-health bad";
    host.textContent = "服务连接异常：无法读取健康状态，请检查计算服务器或 Windows 中继。";
    return;
  }
  const release = health.release || {};
  const storage = health.storage || {};
  const leases = health.leases || {};
  const recovery = health.maintenance?.job_recovery || {};
  const catalog = health.maintenance?.catalog || {};
  const warnings = [];
  if (storage.status === "error" || storage.status === "warning") warnings.push(`磁盘剩余 ${formatBytes(storage.free_bytes || 0)}`);
  if (Number(leases.stale || 0) > 0) warnings.push(`${Number(leases.stale)} 个任务心跳已过期`);
  if (recovery.last_error) warnings.push("任务恢复服务异常");
  if (["failed", "error"].includes(String(catalog.state || ""))) warnings.push("媒体目录同步失败");
  host.className = `deployment-health ${warnings.length ? "warn" : "ok"}`;
  host.textContent = warnings.length
    ? `服务需要关注 · ${warnings.join(" · ")} · Build ${release.build_id || release.version || "unknown"}`
    : `服务正常 · ${Number(leases.running || 0)} 个后台任务 · 磁盘可用 ${formatBytes(storage.free_bytes || 0)} · Build ${release.build_id || release.version || "development"}`;
}

async function refreshDeploymentHealth() {
  try {
    state.runtimeHealth = await api("/api/health");
    renderDeploymentHealth(state.runtimeHealth);
  } catch (error) {
    renderDeploymentHealth(null, error);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formData(form) {
  return Object.fromEntries(new FormData(form).entries());
}

function stableStringify(value) {
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(",")}]`;
  }
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function readInferenceDraft() {
  const parsed = Shared.storageJsonGet(INFERENCE_DRAFT_KEY, null);
  return parsed?.version === INFERENCE_DRAFT_VERSION ? parsed : null;
}

function saveInferenceDraft() {
  const form = $("infer-form");
  if (!form || !state.inferenceDraftRestored) return;
  const fields = {};
  for (const element of Array.from(form.elements)) {
    if (!element.name || ["file", "submit", "button"].includes(element.type)) continue;
    if (element.name === "metrics") continue;
    fields[element.name] = element.type === "checkbox" ? Boolean(element.checked) : element.value;
  }
  const draft = {
    version: INFERENCE_DRAFT_VERSION,
    saved_at: Date.now(),
    fields,
    metrics: selectedMetrics(),
    groups: selectedGroupNames(),
    video_selection: {
      token: state.videoSelectionToken,
      groups_key: state.videoSelectionGroupsKey,
      total: state.videoSelectionTotal,
      group_counts: state.videoSelectionGroupCounts,
    },
    cuda_devices: Array.from(state.selectedCudaDevices),
    npu_devices: Array.from(state.selectedNpuDevices),
  };
  Shared.storageJsonSet(INFERENCE_DRAFT_KEY, draft);
}

function scheduleInferenceDraftSave(delay = 250) {
  clearTimeout(state.inferenceDraftTimer);
  state.inferenceDraftTimer = setTimeout(() => {
    state.inferenceDraftTimer = null;
    saveInferenceDraft();
  }, delay);
}

async function restoreInferenceDraft() {
  if (state.inferenceDraftRestored) return;
  const draft = readInferenceDraft();
  const form = $("infer-form");
  if (!draft || !form) {
    state.inferenceDraftRestored = true;
    return;
  }
  const fields = draft.fields || {};
  const availableGroups = new Set(state.videoGroups.map((row) => row.name));
  const restoredGroups = (draft.groups || []).filter((name) => availableGroups.has(name));
  if (restoredGroups.length) state.selectedGroups = new Set(restoredGroups);
  const restoredSelection = draft.video_selection || {};
  const restoredGroupsKey = stableStringify(restoredGroups);
  if (
    restoredSelection.token
    && restoredSelection.groups_key === restoredGroupsKey
  ) {
    state.videoSelectionToken = String(restoredSelection.token);
    state.videoSelectionGroupsKey = restoredGroupsKey;
    state.videoSelectionTotal = Number(restoredSelection.total || 0);
    state.videoSelectionGroupCounts = { ...(restoredSelection.group_counts || {}) };
    state.videoSelectionValidated = false;
  }
  if (Array.isArray(draft.cuda_devices)) state.selectedCudaDevices = new Set(draft.cuda_devices.map(String));
  if (Array.isArray(draft.npu_devices)) state.selectedNpuDevices = new Set(draft.npu_devices.map(String));
  for (const [name, value] of Object.entries(fields)) {
    const element = form.elements[name];
    if (!element || name === "checkpoint") continue;
    if (element.type === "checkbox") element.checked = Boolean(value);
    else if (element.tagName !== "SELECT" || Array.from(element.options).some((option) => option.value === String(value))) {
      element.value = String(value ?? "");
    }
  }
  await loadCheckpointsForModel(form.elements.model_file.value);
  if (fields.checkpoint && Array.from(form.elements.checkpoint.options).some((option) => option.value === String(fields.checkpoint))) {
    form.elements.checkpoint.value = String(fields.checkpoint);
  }
  const selectedMetricNames = new Set((draft.metrics || []).map(String));
  document.querySelectorAll("#metrics-options input[name='metrics']").forEach((input) => {
    input.checked = selectedMetricNames.has(input.value);
  });
  state.inferenceDraftRestored = true;
  renderGroupPicker();
  renderSingleDeviceOptions(fields.device || form.elements.device.value || "auto");
  renderDeviceOptions();
  renderCustomSizeVisibility();
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2600);
}

function currentViewName() {
  return document.querySelector(".nav-item.active")?.dataset.view || "create";
}

function selectedRouteFrame() {
  if (!state.selectedRun) return null;
  const videoName = state.selectedVideoByRun[state.selectedRun.id];
  const video = videoName ? state.runVideoTimelines[`${state.selectedRun.id}:${videoName}`] : null;
  if (!video) return null;
  const selectedIndex = Number(state.selectedSampleByVideo[`${state.selectedRun.id}:${videoName}`] || 0);
  return video.samples?.[selectedIndex]?.frame_index ?? null;
}

function syncBrowserRoute(options = {}) {
  if (state.applyingRoute) return;
  const view = options.view || currentViewName();
  const url = new URL(window.location.href);
  url.searchParams.set("view", ROUTE_VIEWS.has(view) ? view : "create");
  if (view === "runs" && state.selectedRun) {
    url.searchParams.set("run", String(state.selectedRun.id));
    const videoName = state.selectedVideoByRun[state.selectedRun.id];
    const frame = selectedRouteFrame();
    if (videoName) url.searchParams.set("video", videoName);
    else url.searchParams.delete("video");
    if (frame !== null && frame !== undefined) url.searchParams.set("frame", String(frame));
    else url.searchParams.delete("frame");
  } else {
    url.searchParams.delete("run");
    url.searchParams.delete("video");
    url.searchParams.delete("frame");
  }
  const next = `${url.pathname}${url.search}${url.hash}`;
  if (next === `${window.location.pathname}${window.location.search}${window.location.hash}`) return;
  if (options.replace) window.history.replaceState({}, "", next);
  else window.history.pushState({}, "", next);
}

function switchView(view, options = {}) {
  const normalized = ROUTE_VIEWS.has(view) ? view : "create";
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === normalized));
  document.querySelectorAll(".view").forEach((item) => item.classList.toggle("active", item.id === `view-${normalized}`));
  if (options.updateRoute !== false) syncBrowserRoute({ view: normalized, replace: Boolean(options.replace) });
}

function isRunsViewActive() {
  return $("view-runs")?.classList.contains("active");
}

function renderEmptyRunDetail() {
  $("run-detail").innerHTML = "<p class=\"muted\">选择一个 Run 查看详情。</p>";
}

function currentRunType() {
  return $("infer-form").elements.run_type?.value || "model_inference";
}

function selectedGroupNames() {
  // Preserve catalog order so the merged run and default name are deterministic.
  return state.videoGroups
    .map((item) => item.name)
    .filter((name) => state.selectedGroups.has(name));
}

function primaryGroupName() {
  return selectedGroupNames()[0] || "";
}

function isMultiGroup() {
  return selectedGroupNames().length > 1;
}

function currentGroup() {
  const name = primaryGroupName();
  return state.videoGroups.find((item) => item.name === name) || null;
}

function groupByName(name) {
  return state.videoGroups.find((item) => item.name === name) || null;
}

function isCompareRun(run) {
  return (run?.metadata?.run_type || "model_inference") === "video_compare";
}

function isFileInputRun(run) {
  return !isCompareRun(run) && Boolean(
    run?.metadata?.request?.model_file || run?.metadata?.model_file,
  );
}

function previewGroupsForRun(run) {
  return isCompareRun(run)
    ? PREVIEW_GROUPS_COMPARE
    : PREVIEW_GROUPS_MODEL;
}

function metricHealthBadge(name) {
  const row = state.metricHealth?.metrics?.[name];
  if (!row) return "";
  const title = row.reason || row.status;
  return `<small class="metric-health ${escapeHtml(row.status)}" title="${escapeHtml(title)}">${escapeHtml(row.status)}</small>`;
}

function decodeBackendStatus(name) {
  const row = state.devices?.decode_backends?.[name];
  if (row === true) return { available: true, label: "available", detail: "" };
  if (row === false || !row) return { available: false, label: "missing", detail: `${name} is not available` };
  return {
    available: Boolean(row.available),
    label: row.available ? "available" : "missing",
    detail: row.error || row.version || row.path || "",
  };
}

function renderDecodeBackendNotice() {
  const ffmpeg = decodeBackendStatus("ffmpeg");
  const opencv = decodeBackendStatus("opencv");
  const preferred = ffmpeg.available ? "ffmpeg" : (opencv.available ? "opencv fallback" : "none");
  const severity = ffmpeg.available || opencv.available ? "message" : "message warn";
  return `
    <div class="${severity}">
      <p><strong>Decode backend</strong>: ${escapeHtml(preferred)}. FFmpeg ${escapeHtml(ffmpeg.label)}${ffmpeg.detail ? ` (${escapeHtml(ffmpeg.detail)})` : ""}; OpenCV ${escapeHtml(opencv.label)}${opencv.detail ? ` (${escapeHtml(opencv.detail)})` : ""}.</p>
    </div>
  `;
}

function renderMetricOptions() {
  const selected = new Set(selectedMetrics());
  $("metrics-options").innerHTML = METRICS.map((name) => `
    <label class="check-item">
      <input type="checkbox" name="metrics" value="${escapeHtml(name)}" ${selected.has(name) ? "checked" : ""}>
      <span>${escapeHtml(name)} ${metricHealthBadge(name)}</span>
    </label>
  `).join("") + "<p class=\"muted metric-hint\">不可用指标会明确显示原因，不会被替换成其它分数。</p>";
}

function renderModeSections() {
  document.querySelectorAll("[data-mode-section]").forEach((item) => {
    item.hidden = item.dataset.modeSection !== "model_inference";
  });
  $("video-selection").hidden = false;
}

function renderOptions() {
  const form = $("infer-form");
  const previousModel = form.elements.model_file.value;
  const previousCheckpoint = form.elements.checkpoint?.value || "none";
  const previousDevice = form.elements.device?.value || "auto";

  form.elements.model_file.innerHTML = state.modelFiles
    .map((item) => `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)}</option>`)
    .join("");

  form.elements.model_file.value = state.modelFiles.some((item) => item.name === previousModel)
    ? previousModel
    : (state.modelFiles.some((item) => item.name === "test_average.py") ? "test_average.py" : state.modelFiles[0]?.name || "");

  if (!form.elements.run_type.value) {
    form.elements.run_type.value = "model_inference";
  }

  // Drop any previously-selected groups that no longer exist, then default to
  // one group so the form is usable on first load.
  const available = new Set(state.videoGroups.map((item) => item.name));
  for (const name of Array.from(state.selectedGroups)) {
    if (!available.has(name)) state.selectedGroups.delete(name);
  }
  if (!state.selectedGroups.size && state.videoGroups.length) {
    const initial = state.videoGroups.some((item) => item.name === "test_style")
      ? "test_style"
      : state.videoGroups[0].name;
    state.selectedGroups.add(initial);
  }

  state.checkpoints = state.checkpointsByModel[form.elements.model_file.value] || [];
  renderCheckpointOptions(previousCheckpoint);
  const fallbackDevice = resolveInitialExecutionDefaults();
  renderSingleDeviceOptions(fallbackDevice || previousDevice);
  renderDeviceOptions();
  renderGroupPicker();
  renderCustomSizeVisibility();
  renderModeSections();
}

function renderGroupPicker() {
  const host = $("video-group-picker");
  if (!host) return;
  if (!state.videoGroups.length) {
    host.innerHTML = "<p class=\"muted\">videos/ 下没有可用的视频集。</p>";
    return;
  }
  host.innerHTML = state.videoGroups.map((item) => `
    <label class="check-item">
      <input type="checkbox" data-group-toggle="${escapeHtml(item.name)}" ${state.selectedGroups.has(item.name) ? "checked" : ""}>
      <span>${escapeHtml(item.name)} (${escapeHtml(item.video_count)})</span>
    </label>
  `).join("");
}

function renderCheckpointOptions(previousValue = null) {
  const form = $("infer-form");
  const modelFile = form.elements.model_file.value || "";
  const modelStem = modelFile.replace(/\.py$/i, "");
  const rows = (state.checkpoints || []).filter((item) => !modelFile || item.model === modelStem);
  form.elements.checkpoint.innerHTML = [
    "<option value=\"none\">不加载权重</option>",
    "<option value=\"auto\">自动选择最新权重</option>",
    ...rows.map((item) => `<option value="${escapeHtml(item.relative_path)}">${escapeHtml(item.relative_path)}</option>`),
  ].join("");
  const desired = previousValue ?? form.elements.checkpoint.value;
  form.elements.checkpoint.value = rows.some((item) => item.relative_path === desired) || ["none", "auto"].includes(desired)
    ? desired
    : "none";
}

async function loadCheckpointsForModel(modelFile, options = {}) {
  const normalizedModel = String(modelFile || "").trim();
  const form = $("infer-form");
  if (!normalizedModel) {
    state.checkpoints = [];
    renderCheckpointOptions("none");
    return [];
  }
  const cached = state.checkpointsByModel[normalizedModel];
  if (!options.force && Array.isArray(cached)) {
    if (form.elements.model_file.value === normalizedModel) {
      const previous = form.elements.checkpoint?.value || "none";
      state.checkpoints = cached;
      renderCheckpointOptions(previous);
    }
    return cached;
  }
  if (state.checkpointRequestsByModel[normalizedModel]) {
    return state.checkpointRequestsByModel[normalizedModel];
  }

  if (!Array.isArray(cached) && form.elements.model_file.value === normalizedModel) {
    const previous = form.elements.checkpoint?.value || "none";
    state.checkpoints = [];
    renderCheckpointOptions(previous);
  }
  const request = api(`/api/checkpoints?model_file=${encodeURIComponent(normalizedModel)}`)
    .then((rows) => {
      const normalizedRows = Array.isArray(rows) ? rows : [];
      state.checkpointsByModel[normalizedModel] = normalizedRows;
      if (form.elements.model_file.value === normalizedModel) {
        const previous = form.elements.checkpoint?.value || "none";
        state.checkpoints = normalizedRows;
        renderCheckpointOptions(previous);
      }
      return normalizedRows;
    })
    .finally(() => {
      delete state.checkpointRequestsByModel[normalizedModel];
    });
  state.checkpointRequestsByModel[normalizedModel] = request;
  return request;
}

function renderSingleDeviceOptions(previousValue = null) {
  const form = $("infer-form");
  const select = form.elements.device;
  if (!select) return;
  const cuda = state.devices?.cuda || [];
  const npu = state.devices?.npu || [];
  const options = [
    { value: "auto", label: "auto" },
    { value: "cpu", label: "cpu" },
    ...cuda.map((item) => ({ value: item.id, label: `${item.id} ${item.name || ""}`.trim() })),
    ...npu.map((item) => ({ value: item.id, label: `${item.id} ${item.name || ""}`.trim() })),
  ];
  select.innerHTML = options
    .map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(item.label)}</option>`)
    .join("");
  const desired = previousValue ?? select.value;
  const normalizedDesired = desired === "cuda"
    ? (cuda[0]?.id || "auto")
    : desired === "npu"
      ? (npu[0]?.id || "auto")
      : desired;
  select.value = options.some((item) => item.value === normalizedDesired) ? normalizedDesired : "auto";
  select.disabled = (form.elements.execution_mode.value || "single") !== "single";
}

function renderDeviceOptions() {
  const container = $("device-options");
  if (!container) return;
  const cuda = state.devices?.cuda || [];
  const npu = state.devices?.npu || [];
  if (!state.selectedCudaDevices.size && cuda.length) {
    state.selectedCudaDevices = new Set(cuda.map((item) => item.id));
  }
  if (!state.selectedNpuDevices.size && npu.length) {
    state.selectedNpuDevices = new Set(npu.map((item) => item.id));
  }
  const executionMode = $("infer-form").elements.execution_mode.value || "single";
  const npuError = state.devices?.errors?.npu;
  container.innerHTML = `
    <div class="device-panel ${executionMode === "multi_cuda" ? "visible" : ""}">
      <span>CUDA 多卡</span>
      ${cuda.length ? cuda.map((item) => `
        <label class="check-item">
          <input type="checkbox" data-cuda-device="${escapeHtml(item.id)}" ${state.selectedCudaDevices.has(item.id) ? "checked" : ""}>
          <span>${escapeHtml(item.id)} ${escapeHtml(item.name || "")}</span>
        </label>
      `).join("") : "<p class=\"muted\">未检测到可用 CUDA 设备。</p>"}
    </div>
    <div class="device-panel ${executionMode === "multi_npu" ? "visible" : ""}">
      <span>NPU 多卡</span>
      ${npu.length ? npu.map((item) => `
        <label class="check-item">
          <input type="checkbox" data-npu-device="${escapeHtml(item.id)}" ${state.selectedNpuDevices.has(item.id) ? "checked" : ""}>
          <span>${escapeHtml(item.id)} ${escapeHtml(item.name || "")}</span>
        </label>
      `).join("") : `<p class="muted">${escapeHtml(npuError || "未检测到可用 NPU 设备。")}</p>`}
      </div>
    </div>
  `;
}

function renderCustomSizeVisibility() {
  const mode = $("infer-form").elements.resolution_mode.value;
  document.querySelectorAll(".custom-size").forEach((item) => item.classList.toggle("visible", mode === "custom"));
}

function currentVideoSelectionGroupsKey() {
  return stableStringify(selectedGroupNames());
}

function resetVideoSelectionSnapshot() {
  state.videoSelectionToken = "";
  state.videoSelectionGroupsKey = "";
  state.videoSelectionTotal = 0;
  state.videoSelectionGroupCounts = {};
  state.videoSelectionValidated = false;
  state.videoSelectionRequestGeneration += 1;
}

function applyVideoSelectionSnapshot(payload) {
  state.videoSelectionToken = String(payload.video_selection_token || "");
  state.videoSelectionGroupsKey = stableStringify(payload.video_groups || []);
  state.videoSelectionTotal = Number(payload.total || 0);
  state.videoSelectionGroupCounts = { ...(payload.group_counts || {}) };
  state.videoSelectionValidated = true;
}

async function ensureVideoSelectionSnapshot(options = {}) {
  const groups = selectedGroupNames();
  const groupsKey = stableStringify(groups);
  if (!groups.length) {
    resetVideoSelectionSnapshot();
    return null;
  }
  if (
    !options.force
    && state.videoSelectionToken
    && state.videoSelectionGroupsKey === groupsKey
    && state.videoSelectionValidated
  ) {
    return {
      video_selection_token: state.videoSelectionToken,
      video_groups: groups,
      total: state.videoSelectionTotal,
      group_counts: state.videoSelectionGroupCounts,
    };
  }
  const requestGeneration = ++state.videoSelectionRequestGeneration;
  state.videoSelectionLoading = true;
  renderVideoSelection();
  try {
    let payload = null;
    if (
      !options.force
      && state.videoSelectionToken
      && state.videoSelectionGroupsKey === groupsKey
    ) {
      try {
        payload = await api(`/api/video-selections/${encodeURIComponent(state.videoSelectionToken)}?page=1&page_size=1`);
        if (stableStringify(payload.video_groups || []) !== groupsKey) payload = null;
      } catch (_error) {
        payload = null;
      }
    }
    if (!payload) {
      payload = await api("/api/video-selections", {
        method: "POST",
        body: JSON.stringify({ video_groups: groups }),
      });
    }
    if (requestGeneration !== state.videoSelectionRequestGeneration) return null;
    applyVideoSelectionSnapshot(payload);
    return payload;
  } finally {
    if (requestGeneration === state.videoSelectionRequestGeneration) {
      state.videoSelectionLoading = false;
      renderVideoSelection();
    }
  }
}

async function mutateVideoSelectionSnapshot(groupName, operation, options = {}) {
  await ensureVideoSelectionSnapshot();
  if (!state.videoSelectionToken) throw new Error("视频选择尚未就绪");
  const requestGeneration = ++state.videoSelectionRequestGeneration;
  state.videoSelectionLoading = true;
  renderVideoSelection();
  try {
    const payload = await api("/api/video-selections", {
      method: "POST",
      body: JSON.stringify({
        base_selection_token: state.videoSelectionToken,
        operation,
        video_group: groupName,
        q: options.q || "",
        video_names: options.videoNames,
      }),
    });
    if (requestGeneration !== state.videoSelectionRequestGeneration) return;
    applyVideoSelectionSnapshot(payload);
    await loadVideoGroupPage(
      groupName,
      Number(state.videoPages[groupName]?.page || 1),
      { render: false, skipSnapshot: true },
    );
    schedulePreflight(0);
    scheduleInferenceDraftSave();
  } finally {
    if (requestGeneration === state.videoSelectionRequestGeneration) {
      state.videoSelectionLoading = false;
      renderVideoSelection();
    }
  }
}

function selectedVideoCountForGroup(groupName) {
  return Number(state.videoSelectionGroupCounts[groupName] || 0);
}

function resolveInitialExecutionDefaults() {
  if (state.executionDefaultsResolved || !state.devices) return null;
  state.executionDefaultsResolved = true;
  const form = $("infer-form");
  const mode = form.elements.execution_mode.value || "single";
  const cuda = state.devices.cuda || [];
  const npu = state.devices.npu || [];
  if (mode === "multi_npu" && npu.length) return null;
  if (mode === "multi_cuda" && cuda.length) return null;
  if (npu.length) {
    form.elements.execution_mode.value = "multi_npu";
    return npu[0].id;
  }
  if (cuda.length > 1) {
    form.elements.execution_mode.value = "multi_cuda";
    return cuda[0].id;
  }
  form.elements.execution_mode.value = "single";
  return cuda[0]?.id || "cpu";
}

async function loadVideoGroupPage(groupName, page = 1, options = {}) {
  if (!groupName) return;
  if (!options.skipSnapshot) await ensureVideoSelectionSnapshot();
  const query = state.videoPageQuery[groupName] || "";
  const sort = state.videoPageSort[groupName] || "name";
  state.videoPageLoading[groupName] = true;
  if (options.render !== false) renderVideoSelection();
  try {
    const selection = state.videoSelectionToken
      ? `&video_selection_token=${encodeURIComponent(state.videoSelectionToken)}`
      : "";
    const payload = await api(`/api/video-groups/${encodeURIComponent(groupName)}/videos?page=${page}&page_size=50&q=${encodeURIComponent(query)}&sort=${encodeURIComponent(sort)}${selection}`);
    state.videoPages[groupName] = payload;
    return payload;
  } finally {
    state.videoPageLoading[groupName] = false;
    if (options.render !== false) renderVideoSelection();
  }
}

async function loadSelectedVideoGroupIndexes() {
  await ensureVideoSelectionSnapshot();
  const pending = selectedGroupNames().filter((name) => !state.videoPages[name]);
  if (!pending.length) return;
  pending.forEach((name) => { state.videoPageLoading[name] = true; });
  renderVideoSelection();
  await Promise.all(pending.map((name) => loadVideoGroupPage(name, 1, { render: false })));
  renderVideoSelection();
}

function videoSelectionPager(page, groupName) {
  if (!page || Number(page.total_pages || 1) <= 1) return "";
  const g = escapeHtml(groupName);
  return `
    <div class="pager video-page-pager">
      <button class="secondary" data-video-page="${Number(page.page || 1) - 1}" data-video-group="${g}" ${Number(page.page || 1) <= 1 ? "disabled" : ""} type="button">上一页</button>
      <span class="muted">第 ${escapeHtml(page.page || 1)} / ${escapeHtml(page.total_pages || 1)} 页，筛选 ${escapeHtml(page.filtered_count || 0)} / ${escapeHtml(page.video_count || 0)} 个视频</span>
      <button class="secondary" data-video-page="${Number(page.page || 1) + 1}" data-video-group="${g}" ${Number(page.page || 1) >= Number(page.total_pages || 1) ? "disabled" : ""} type="button">下一页</button>
    </div>
  `;
}

function renderGroupVideoTable(groupName) {
  const group = groupByName(groupName);
  if (!group) return "";
  const page = state.videoPages[groupName];
  if (!page) {
    return `
      <section class="group-video-block" data-group-block="${escapeHtml(groupName)}">
        <div class="panel-head">
          <div>
            <h3>${escapeHtml(groupName)}</h3>
            <p class="muted">${state.videoPageLoading[groupName] ? "正在加载轻量视频索引并默认全选…" : `视频索引暂不可用（${escapeHtml(group.video_count)} 个视频）。`}</p>
          </div>
          ${state.videoPageLoading[groupName] ? '<span class="inline-spinner" role="status">加载中…</span>' : `<button class="secondary" data-load-video-page="${escapeHtml(groupName)}" type="button">重试加载</button>`}
        </div>
      </section>
    `;
  }
  const selectedCount = selectedVideoCountForGroup(groupName);
  const query = state.videoPageQuery[groupName] || "";
  const sort = state.videoPageSort[groupName] || "name";
  const selectionBusy = state.videoSelectionLoading ? "disabled" : "";
  return `
    <details class="group-video-block selection-diagnostics" data-group-block="${escapeHtml(groupName)}" ${query || state.expandedVideoGroups.has(groupName) ? "open" : ""}>
      <summary>
        <span><strong>${escapeHtml(groupName)}</strong> · 已选 ${selectedCount}/${page.video_count}</span>
        <span class="muted">展开筛选或调整单个视频</span>
      </summary>
      <div class="panel-head diagnostics-head">
        <div>
          <p class="muted">默认全选；搜索、翻页和排序均在服务端执行。</p>
        </div>
        <div class="actions">
          <button class="secondary" data-video-select="all-filtered" data-group="${escapeHtml(groupName)}" type="button" ${selectionBusy}>全选筛选</button>
          <button class="secondary" data-video-select="none-filtered" data-group="${escapeHtml(groupName)}" type="button" ${selectionBusy}>清空筛选</button>
          <button class="secondary" data-video-select="invert-filtered" data-group="${escapeHtml(groupName)}" type="button" ${selectionBusy}>反选筛选</button>
        </div>
      </div>
      <div class="video-tools">
        <label>
          <span>搜索视频</span>
          <input data-video-query data-group="${escapeHtml(groupName)}" value="${escapeHtml(query)}" placeholder="文件名">
        </label>
        <label>
          <span>排序</span>
          <select data-video-sort data-group="${escapeHtml(groupName)}">
            ${[
              ["name", "名称"],
              ["-name", "名称倒序"],
              ["frame_count", "帧数"],
              ["-frame_count", "帧数倒序"],
              ["duration", "时长"],
              ["-duration", "时长倒序"],
              ["resolution", "分辨率"],
              ["-resolution", "分辨率倒序"],
              ["triplets", "Triplets"],
              ["-triplets", "Triplets 倒序"],
            ].map(([value, label]) => `<option value="${escapeHtml(value)}" ${sort === value ? "selected" : ""}>${escapeHtml(label)}</option>`).join("")}
          </select>
        </label>
      </div>
      <div class="table compact-table">${table(page.videos || [], [
        { label: "", render: (row) => `<input type="checkbox" data-video-name="${escapeHtml(row.name)}" data-group="${escapeHtml(groupName)}" ${row.selected ? "checked" : ""} ${selectionBusy}>` },
        { label: "视频", render: (row) => escapeHtml(row.name) },
        { label: "帧数", render: (row) => escapeHtml(row.frame_count) },
        { label: "Triplets", render: (row) => escapeHtml(row.valid_triplets ?? 0) },
        { label: "FPS", render: (row) => formatNumber(row.fps) },
        { label: "分辨率", render: (row) => `${escapeHtml(row.width)}x${escapeHtml(row.height)}` },
        { label: "缓存", render: (row) => escapeHtml(row.cache_status || "-") },
      ])}</div>
      ${videoSelectionPager(page, groupName)}
    </details>
  `;
}

function renderVideoSelection() {
  const groups = selectedGroupNames();
  if (!groups.length) {
    $("video-selection").innerHTML = "<p class=\"muted\">先选择一个或多个视频集。</p>";
    return;
  }
  const totalSelected = state.videoSelectionTotal;
  const multi = groups.length > 1;
  $("video-selection").innerHTML = `
    <div class="panel-head">
      <div>
        <h2>视频选择</h2>
        <p class="muted">${state.videoSelectionLoading
          ? "正在建立服务端视频选择快照…"
          : multi
          ? `已合并 ${groups.length} 个视频集，视频以“视频集/文件名”限定，共 ${totalSelected} 个已选。`
          : `已自动载入当前视频集并默认全选，共 ${totalSelected} 个已选。需要排除视频时再展开下方明细。`}</p>
      </div>
    </div>
    ${groups.map((name) => renderGroupVideoTable(name)).join("")}
  `;
}

function selectedMetrics() {
  return Array.from(document.querySelectorAll("input[name='metrics']:checked")).map((item) => item.value);
}

function payloadFromForm() {
  const data = formData($("infer-form"));
  const groups = selectedGroupNames();
  const multi = groups.length > 1;
  return {
    run_type: "model_inference",
    model_file: data.model_file,
    checkpoint: data.checkpoint || "none",
    // Single-group runs keep the legacy `video_group` field (byte-identical
    // caches/reference keys); multi-group runs send `video_groups`.
    video_group: multi ? undefined : (groups[0] || ""),
    video_groups: multi ? groups : undefined,
    video_selection_token: state.videoSelectionToken || undefined,
    resolution_mode: data.resolution_mode || "original",
    height: data.height ? Number(data.height) : null,
    width: data.width ? Number(data.width) : null,
    device: data.device || "auto",
    execution_mode: data.execution_mode || "single",
    devices: data.execution_mode === "multi_npu"
      ? Array.from(state.selectedNpuDevices)
      : Array.from(state.selectedCudaDevices),
    precision: data.precision || "auto",
    visualize_height: data.visualize_height ? Number(data.visualize_height) : null,
    visualize_width: data.visualize_width ? Number(data.visualize_width) : null,
    batch_size: Number(data.batch_size || 1),
    batch_size_per_device: Number(data.batch_size_per_device || data.batch_size || 1),
    metric_batch_size_per_device: data.metric_batch_size_per_device ? Number(data.metric_batch_size_per_device) : null,
    artifact_profile: data.artifact_profile || "evaluation",
    prefetch_workers: data.prefetch_workers ? Number(data.prefetch_workers) : null,
    save_workers: data.save_workers ? Number(data.save_workers) : null,
    max_save_inflight: data.max_save_inflight ? Number(data.max_save_inflight) : null,
    frame_step: Number(data.frame_step || 1),
    max_frames: data.max_frames ? Number(data.max_frames) : null,
    metrics: selectedMetrics(),
  };
}

function schedulePreflight(delay = 600) {
  if (!state.runSubmitting && state.runSubmitError) {
    state.runSubmitError = "";
    state.runSubmissionId = "";
    renderRunSubmissionState();
  }
  if (state.runSubmitting) return;
  clearTimeout(state.preflightTimer);
  state.preflightTimer = setTimeout(() => runPreflight().catch(renderPreflightError), delay);
}

async function runPreflight(options = {}) {
  const payload = payloadFromForm();
  const level = options.level === "deep" ? "deep" : "quick";
  if (!payload) {
    state.preflight = null;
    state.preflightPayloadKey = "";
    state.preflightLevel = "";
    renderPreflight();
    return;
  }
  const groups = selectedGroupNames();
  if (!payload.model_file || !groups.length) {
    state.preflight = null;
    state.preflightPayloadKey = "";
    state.preflightLevel = "";
    renderPreflight();
    return;
  }
  if (
    !state.videoSelectionToken
    || state.videoSelectionGroupsKey !== stableStringify(groups)
    || state.videoSelectionTotal < 1
  ) {
    state.preflight = null;
    state.preflightPayloadKey = "";
    state.preflightLevel = "";
    renderPreflight();
    return;
  }
  // Every selected group must have its video list loaded before preflight so the
  // selection is real, not just the summary count.
  if (groups.some((name) => !state.videoPages[name])) {
    state.preflight = null;
    state.preflightPayloadKey = "";
    state.preflightLevel = "";
    renderPreflight();
    return;
  }
  const payloadKey = stableStringify(payload);
  if (
    !options.force
    && state.preflight
    && state.preflightPayloadKey === payloadKey
    && (level === "quick" || state.preflightLevel === "deep")
  ) {
    renderPreflight();
    return;
  }
  if (state.preflightAbortController) {
    state.preflightAbortController.abort();
  }
  const controller = new AbortController();
  state.preflightAbortController = controller;
  try {
    const requestPayload = { ...payload, preflight_level: level };
    const result = await api("/api/preflight", {
      method: "POST",
      body: JSON.stringify(requestPayload),
      signal: controller.signal,
    });
    if (state.preflightAbortController !== controller) return;
    // Quick preflight is advisory. Never let an older server response make it
    // authoritative for Run creation by carrying a deep-preflight token.
    if (level !== "deep") delete result.preflight_token;
    state.preflight = result;
    state.preflightPayloadKey = payloadKey;
    state.preflightLevel = level;
    renderPreflight();
  } catch (error) {
    if (error.name === "AbortError") return;
    throw error;
  } finally {
    if (state.preflightAbortController === controller) {
      state.preflightAbortController = null;
    }
  }
}

function renderPreflightError(error) {
  state.preflight = { ok: false, errors: [{ title: "预检查请求失败", message: error.message }], warnings: [] };
  renderPreflight();
}

function renderMetricSetup(row) {
  const summary = row.setup_summary || "-";
  const requirements = Array.isArray(row.setup_requirements) ? row.setup_requirements : [];
  const diagnostics = [
    row.implementation_mode ? `<li><strong>mode</strong>: ${escapeHtml(row.implementation_mode)}</li>` : "",
    row.manifest_path ? `<li><strong>manifest</strong>: ${escapeHtml(row.manifest_path)}</li>` : "",
    row.resolved_executable ? `<li><strong>executable</strong>: ${escapeHtml(row.resolved_executable)}</li>` : "",
    Array.isArray(row.driver_command) && row.driver_command.length
      ? `<li><strong>driver</strong>: ${escapeHtml(row.driver_command.join(" "))}</li>`
      : "",
  ].filter(Boolean).join("");
  const list = requirements.length
    ? `<ul>${requirements.map((item) => `<li><strong>${escapeHtml(item.kind || "requirement")}</strong>: ${escapeHtml(item.target || "-")} <span>${escapeHtml(item.description || "")}</span></li>`).join("")}</ul>`
    : "";
  const detailList = diagnostics ? `<ul>${diagnostics}</ul>` : "";
  return `<div class="metric-setup"><p>${escapeHtml(summary)}</p>${detailList}${list}</div>`;
}

function renderMetricSetupList(rows) {
  if (!rows.length) return "";
  return `
    <div class="metric-setup-list">
      ${rows.map((row) => `
        <article class="metric-setup-card">
          <h4>${escapeHtml(row.name)}</h4>
          ${renderMetricSetup(row)}
        </article>
      `).join("")}
    </div>
  `;
}

function renderMetricHealthSummary(rows) {
  const problemRows = rows.filter((row) => !row.available);
  if (!problemRows.length) return "<p class=\"muted\">All configured metrics are available.</p>";
  return `
    <div class="metric-health-summary">
      ${problemRows.map((row) => `
        <span title="${escapeHtml(row.reason || row.status)}">
          <strong>${escapeHtml(row.name)}</strong> ${escapeHtml(row.status)}
        </span>
      `).join("")}
    </div>
  `;
}

function renderMetricHealthTable(rowsByName) {
  const rows = Object.entries(rowsByName || {}).map(([name, row]) => ({ name, ...row }));
  if (!rows.length) return "";
  return `
    <section class="metric-health-table">
      <h3>指标环境</h3>
      <div class="table compact-table">${table(rows, [
        { label: "指标", render: (row) => escapeHtml(row.name) },
        { label: "状态", render: (row) => escapeHtml(row.status) },
        { label: "粒度", render: (row) => escapeHtml(row.granularity || "-") },
        { label: "时间线", render: (row) => row.supports_timeline ? "yes" : "no" },
        { label: "Input", render: (row) => escapeHtml(row.input_mode || "-") },
        { label: "Evaluator", render: (row) => escapeHtml(row.evaluator || "-") },
        { label: "期望路径", render: (row) => escapeHtml(row.weights_path || (row.expected_paths || [])[0] || "-") },
        { label: "原因", render: (row) => escapeHtml(row.reason || "-") },
      ])}</div>
      ${renderMetricSetupList(rows)}
    </section>
  `;
}

function renderPortableMetricHealthTable(rowsByName) {
  const rows = Object.entries(rowsByName || {}).map(([name, row]) => ({ name, ...row }));
  if (!rows.length) return "";
  return `
    <section class="metric-health-table">
      <h3>Metric Health</h3>
      <div class="table compact-table">${table(rows, [
        { label: "Metric", render: (row) => escapeHtml(row.name) },
        { label: "Status", render: (row) => escapeHtml(row.status) },
        { label: "Mode", render: (row) => escapeHtml(row.implementation_mode || "-") },
        { label: "Granularity", render: (row) => escapeHtml(row.granularity || "-") },
        { label: "Timeline", render: (row) => row.supports_timeline ? "yes" : "no" },
        { label: "Input", render: (row) => escapeHtml(row.input_mode || "-") },
        { label: "Evaluator", render: (row) => escapeHtml(row.evaluator || "-") },
        { label: "Path", render: (row) => escapeHtml(row.manifest_path || row.weights_path || (row.expected_paths || [])[0] || "-") },
        { label: "Exec", render: (row) => escapeHtml(row.resolved_executable || row.executable || "-") },
        { label: "Reason", render: (row) => escapeHtml(row.reason || "-") },
      ])}</div>
      ${renderMetricSetupList(rows)}
    </section>
  `;
}

function renderMetricEnvironmentPanel() {
  const container = $("metric-environment");
  if (!container) return;
  if (!state.metricHealth) {
    container.innerHTML = `
      <div class="panel-head">
        <h2>指标环境</h2>
      </div>
      <p class="muted">正在检查本地指标依赖、权重和 evaluator...</p>
    `;
    return;
  }
  const rows = Object.values(state.metricHealth.metrics || {});
  const available = rows.filter((row) => row.available).length;
  const unavailable = rows.length - available;
  const rowsByName = state.metricHealth?.metrics || {};
  container.innerHTML = `
    <div class="panel-head">
      <div>
        <h2>指标环境</h2>
        <p class="muted">评测资产目录 <code>${escapeHtml(state.metricHealth.asset_root || "set/metrics")}</code>。缺失资产不会自动下载；放入本地目录后点击“刷新文件列表”即可重新检测。</p>
      </div>
      <div class="metric-summary">
        <span>available ${escapeHtml(available)}</span>
        <span>unavailable ${escapeHtml(unavailable)}</span>
      </div>
    </div>
    ${renderMetricHealthSummary(Object.entries(rowsByName).map(([name, row]) => ({ name, ...row })))}
    <details class="metric-health-details">
      <summary>Metric Health</summary>
      ${renderPortableMetricHealthTable(rowsByName)}
    </details>
  `;
}

function renderMessages(kind, items) {
  if (!items?.length) return "";
  const cls = kind === "errors" ? "message error" : "message warn";
  return `<div class="${cls}">${items.map((item) => `<p><strong>${escapeHtml(item.title || "提示")}</strong>: ${escapeHtml(item.message || item)}</p>`).join("")}</div>`;
}

function workloadRiskReasonLabel(reason) {
  const row = reason && typeof reason === "object" ? reason : { code: String(reason || "unknown") };
  const labels = {
    input_pair_device_memory_ge_5_percent: "单设备输入张量下界已达到显存的 5%",
    prefetch_host_memory_ge_25_percent: "预取输入下界已达到可用主机内存的 25%",
    unknown_device_memory_batch_pixels_gt_16000000: "无法读取设备显存，且单设备批次像素数超过 1600 万",
  };
  const label = labels[row.code] || row.code || "未知风险";
  if (row.actual_bytes && row.available_bytes) {
    return `${label}（${formatBytes(row.actual_bytes)} / ${formatBytes(row.available_bytes)}）`;
  }
  if (row.batch_pixels_per_device) {
    return `${label}（${Number(row.batch_pixels_per_device).toLocaleString()} 像素）`;
  }
  return label;
}

function renderWorkloadEstimate(workload) {
  if (!workload || typeof workload !== "object") return "";
  const effective = workload.effective || {};
  const risks = Array.isArray(workload.risk_reasons) ? workload.risk_reasons : [];
  const highRisk = workload.risk_level === "high";
  return `
    <section class="workload-estimate ${highRisk ? "message warn" : ""}">
      <div class="panel-head">
        <h3>工作量与资源下界</h3>
        <span class="${highRisk ? "bad-text" : "ok-text"}">${highRisk ? "高风险，启动时需确认" : "常规"}</span>
      </div>
      <div class="summary-grid">
        <div><span>单设备批次</span><strong>${escapeHtml(effective.batch_size_per_device ?? "-")}</strong></div>
        <div><span>有效分辨率</span><strong>${escapeHtml(effective.width ?? "-")}×${escapeHtml(effective.height ?? "-")}</strong></div>
        <div><span>输入张量下界</span><strong>${escapeHtml(formatBytes(workload.input_tensor_bytes_lower_bound))}</strong></div>
        <div><span>主机预取下界</span><strong>${escapeHtml(formatBytes(workload.prefetch_host_bytes_lower_bound))}</strong></div>
        <div><span>产物空间预算</span><strong>${escapeHtml(formatBytes(workload.artifact_budget_bytes))}</strong></div>
        <div><span>样本数</span><strong>${escapeHtml(effective.sample_count ?? "-")}</strong></div>
      </div>
      ${risks.length ? `<ul>${risks.map((reason) => `<li>${escapeHtml(workloadRiskReasonLabel(reason))}</li>`).join("")}</ul>` : ""}
      <p class="muted">内存数字是可解释的下界，不包含模型参数、激活、输出及分配器工作区；产物数字是规划预算。</p>
    </section>
  `;
}

function renderPreflightExecutionSummary(result, isQuick) {
  const device = result?.device || {};
  const workload = result?.workload || {};
  const effective = workload.effective || {};
  const videos = result?.video_group?.videos || [];
  const devices = Array.isArray(device.effective_devices) && device.effective_devices.length
    ? device.effective_devices.map((value) => String(value))
    : (device.effective_device && !String(device.effective_device).startsWith("multi_")
      ? [String(device.effective_device)]
      : []);
  const deviceCount = Number(device.device_count || devices.length || 0);
  const decoderBackends = Array.from(new Set(videos
    .map((row) => row.decode_backend || row.manifest_backend || row.cache_backend || row.decoder_backend)
    .filter(Boolean)
    .map((value) => String(value))));
  const decoder = result?.decode?.actual_backend
    || result?.decode?.backend
    || result?.cache?.backend
    || decoderBackends.join(" / ")
    || result?.video_group?.decode_backend
    || (isQuick ? "深度预检时确认" : "解码任务启动时确认");
  const evaluationContract = result?.evaluation_contract
    || result?.contracts?.evaluation
    || result?.video_group?.evaluation_contract
    || "-";
  const storageCapacity = workload?.storage_capacity || result?.storage_capacity || {};
  const diskFree = storageCapacity.free_bytes
    ?? result?.storage?.free_bytes
    ?? result?.disk?.free_bytes
    ?? workload?.storage?.free_bytes
    ?? workload?.disk_free_bytes;
  const diskParts = diskFree == null ? [] : [formatBytes(diskFree)];
  if (storageCapacity.reserved_bytes != null) {
    diskParts.push(`已预留 ${formatBytes(storageCapacity.reserved_bytes)}`);
  }
  const remainingAfterRequest = storageCapacity.remaining_after_request_bytes
    ?? storageCapacity.remaining_after_request;
  if (remainingAfterRequest != null) {
    diskParts.push(`本次后 ${formatBytes(remainingAfterRequest)}`);
  }
  const diskClass = storageCapacity.sufficient === false ? " class=\"bad-text\"" : "";
  return `
    <div><span>执行设备</span><strong>${escapeHtml(devices.join(", ") || device.effective_device || "-")}</strong></div>
    <div><span>设备卡数</span><strong>${escapeHtml(deviceCount || (isQuick ? "待确认" : "-"))}</strong></div>
    <div><span>每设备 Batch</span><strong>${escapeHtml(effective.batch_size_per_device ?? "-")}</strong></div>
    <div><span>解码后端</span><strong>${escapeHtml(decoder)}</strong></div>
    <div><span>评测契约</span><strong>${escapeHtml(evaluationContract)}</strong></div>
    <div><span>单设备显存</span><strong>${escapeHtml(formatBytes(effective.device_memory_bytes))}</strong></div>
    <div><span>主机可用内存</span><strong>${escapeHtml(formatBytes(effective.host_available_memory_bytes))}</strong></div>
    <div><span>磁盘可用空间</span><strong${diskClass}>${escapeHtml(diskParts.join("；") || "接口未提供")}</strong></div>
  `;
}

function highRiskWorkloadConfirmation(workload) {
  if (workload?.risk_level !== "high") return true;
  const fingerprint = String(workload.risk_fingerprint || "").trim();
  if (!fingerprint) {
    throw new Error("高风险预检查没有返回风险指纹，请刷新后重试");
  }
  const reasons = (Array.isArray(workload.risk_reasons) ? workload.risk_reasons : [])
    .map((reason) => `• ${workloadRiskReasonLabel(reason)}`)
    .join("\n");
  const effective = workload.effective || {};
  const summary = [
    `设备：${effective.device || "-"} / ${effective.precision || "-"}`,
    `单设备批次：${effective.batch_size_per_device ?? "-"}，分辨率：${effective.width ?? "-"}×${effective.height ?? "-"}`,
    `输入张量下界：${formatBytes(workload.input_tensor_bytes_lower_bound)}`,
    `主机预取下界：${formatBytes(workload.prefetch_host_bytes_lower_bound)}`,
    `产物空间预算：${formatBytes(workload.artifact_budget_bytes)}`,
  ].join("\n");
  return window.confirm(`当前配置被判定为高风险：\n\n${reasons || "• 资源风险阈值已触发"}\n\n${summary}\n\n确认仍按当前吞吐配置启动？`);
}

function renderPreflight() {
  const result = state.preflight;
  const start = $("start-run");
  if (!state.modelFiles.length || !state.videoGroups.length) {
    start.disabled = true;
    $("preflight").innerHTML = `
      <h2>准备输入</h2>
      <p class="muted">需要在项目根目录放置 <code>models/*.py</code> 和 <code>videos/*/</code>。</p>
    `;
    renderRunSubmissionState();
    return;
  }
  if (!result) {
    start.disabled = true;
    const groups = selectedGroupNames();
    if (groups.length && groups.every((name) => !state.videoPages[name])) {
      $("preflight").innerHTML = "<p class=\"muted\">正在载入视频索引，完成后会自动快速预检查。</p>";
      renderRunSubmissionState();
      return;
    }
    $("preflight").innerHTML = "<p class=\"muted\">选择模型和视频后会自动预检查。</p>";
    renderRunSubmissionState();
    return;
  }
  start.disabled = state.runSubmitting || !result.ok;

  const videos = result.video_group?.videos || [];
  const isQuick = String(result.preflight_level || state.preflightLevel) === "quick";
  const interfaceStatus = isQuick
    ? "启动前深度检查"
    : (result.model?.interface_ok ? "通过" : "失败");
  const cacheStatus = isQuick ? "启动前深度检查" : (result.cache?.status || "-");
  $("preflight").innerHTML = `
    <div class="panel-head">
      <h2>${isQuick ? "快速预检查" : "运行前深度预检查"}</h2>
      ${result.ok ? "<span class=\"ok-text\">通过</span>" : "<span class=\"bad-text\">未通过</span>"}
    </div>
    <div class="summary-grid">
      <div><span>模型</span><strong>${escapeHtml(result.model?.name || "-")}</strong></div>
      <div><span>接口检查</span><strong>${interfaceStatus}</strong></div>
      <div><span>已选视频</span><strong>${escapeHtml(result.video_group?.video_count ?? 0)}</strong></div>
      <div><span>真实总帧数</span><strong>${escapeHtml(result.video_group?.frame_count ?? 0)}</strong></div>
      <div><span>总时长</span><strong>${formatDuration(result.video_group?.duration_seconds)}</strong></div>
      <div><span>Triplets</span><strong>${escapeHtml(result.video_group?.triplets ?? 0)}</strong></div>
      <div><span>缓存状态</span><strong>${escapeHtml(cacheStatus)}</strong></div>
      <div><span>设备/精度</span><strong>${escapeHtml(`${result.device?.effective_device || "-"} / ${result.device?.effective_precision || "-"}`)}</strong></div>
      <div><span>支持精度</span><strong>${escapeHtml((result.device?.supported_precisions || []).join(" / ") || "-")}</strong></div>
      ${renderPreflightExecutionSummary(result, isQuick)}
    </div>
    ${renderMessages("errors", result.errors || [])}
    ${renderMessages("warnings", result.warnings || [])}
    ${renderWorkloadEstimate(result.workload)}
    ${renderExecutionProfileRecommendation(result)}
    ${renderPortableMetricHealthTable(result.metrics?.health || {})}
    ${renderDecodeBackendNotice()}
    <details class="preflight-diagnostics">
      <summary>逐视频诊断（${escapeHtml(videos.length)} 个）</summary>
      <p class="muted">帧数来源、解码缓存和分辨率仅用于诊断；默认摘要已经覆盖是否可以启动。</p>
      <div class="table">${table(videos, [
        { label: "视频", render: (row) => escapeHtml(row.name) },
        { label: "真实帧数", render: (row) => `${escapeHtml(row.frame_count)} <span class="muted">${escapeHtml(row.frame_count_source || "")}</span>` },
        { label: "时长", render: (row) => formatDuration(row.duration_seconds) },
        { label: "Triplets", render: (row) => escapeHtml(row.triplets ?? 0) },
        { label: "分辨率", render: (row) => `${escapeHtml(row.width)}x${escapeHtml(row.height)}` },
        { label: "FPS", render: (row) => formatNumber(row.fps) },
        { label: "缓存", render: (row) => escapeHtml(row.cache_status === "not_checked" ? "待深度检查" : (row.cache_status || "-")) },
      ])}</div>
    </details>
  `;
  renderRunSubmissionState();
}

function table(rows, columns, options = {}) {
  if (!rows?.length) return "<p class=\"muted\">暂无数据。</p>";
  const rowAttrs = typeof options.rowAttrs === "function" ? options.rowAttrs : null;
  return `
    <table>
      <thead><tr>${columns.map((column) => `<th>${column.label}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((row) => `<tr ${rowAttrs ? rowAttrs(row) : ""}>${columns.map((column) => `<td>${column.render(row)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table>
  `;
}

function renderRunSubmissionState() {
  const form = $("infer-form");
  const start = $("start-run");
  const status = $("run-submit-status");
  if (!form || !start || !status) return;
  const labels = {
    selection: "正在确认视频选择…",
    preflight: "正在进行运行前深度检查…",
    creating: "正在创建推理 Run…",
    opening: "Run 已创建，正在打开…",
  };
  form.setAttribute("aria-busy", state.runSubmitting ? "true" : "false");
  start.disabled = state.runSubmitting || !state.preflight?.ok;
  start.textContent = state.runSubmitting
    ? (labels[state.runSubmitPhase] || "正在创建…")
    : "开始任务";
  if (state.runSubmitting) {
    status.hidden = false;
    status.className = "run-submit-status message";
    status.textContent = `${labels[state.runSubmitPhase] || "正在处理…"} 请勿重复点击。`;
  } else if (state.runSubmitError) {
    status.hidden = false;
    status.className = "run-submit-status message error";
    status.textContent = `推理任务创建失败：${state.runSubmitError}`;
  } else {
    status.hidden = true;
    status.className = "run-submit-status";
    status.textContent = "";
  }
}

async function startRun(event) {
  event.preventDefault();
  if (runCreationFlight.isLocked()) {
    toast("推理任务正在创建，请勿重复点击");
    return;
  }
  const groups = selectedGroupNames();
  if (!groups.length) {
    toast("请先选择至少一个视频集");
    return;
  }
  if (!runCreationFlight.tryLock()) {
    toast("推理任务正在创建，请勿重复点击");
    return;
  }
  state.runSubmitting = true;
  state.runSubmitPhase = "selection";
  state.runSubmitError = "";
  clearTimeout(state.preflightTimer);
  renderRunSubmissionState();
  try {
    await ensureVideoSelectionSnapshot();
    if (state.videoSelectionTotal < 1) {
      toast("请先选择至少一个视频");
      return;
    }
    state.runSubmitPhase = "preflight";
    renderRunSubmissionState();
    await runPreflight({ force: true, level: "deep" });
    let payload = payloadFromForm();
    if (state.preflightPayloadKey !== stableStringify(payload)) {
      await runPreflight({ force: true, level: "deep" });
      payload = payloadFromForm();
    }
    if (!state.preflight?.ok || state.preflightLevel !== "deep") {
      toast("预检查未通过");
      return;
    }
    if (state.preflightPayloadKey !== stableStringify(payload)) {
      throw new Error("运行配置在预检查期间发生变化，请重新点击开始任务");
    }
    const workload = state.preflight.workload;
    if (!highRiskWorkloadConfirmation(workload)) {
      toast("已取消启动，当前配置保持不变");
      return;
    }
    if (state.preflightLevel === "deep" && state.preflight.preflight_token) {
      payload.preflight_token = String(state.preflight.preflight_token);
    }
    if (workload?.risk_level === "high") {
      payload.risk_ack_fingerprint = String(workload.risk_fingerprint);
    }
    state.runSubmitPhase = "creating";
    renderRunSubmissionState();
    state.runSubmissionId = state.runSubmissionId || Shared.createSubmissionId("run");
    payload.submission_id = state.runSubmissionId;
    const created = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.runSubmissionId = "";
    state.runSubmitPhase = "opening";
    renderRunSubmissionState();
    toast(`Run #${created.run_id} 已开始`);
    switchView("runs");
    state.runsPage.page = 1;
    state.runFilters = { q: "", status: "", run_type: "", model: "" };
    await refreshRunsOnly({ page: 1 });
    await selectRun(created.run_id);
  } catch (error) {
    state.runSubmitError = error.message || String(error);
    throw error;
  } finally {
    state.runSubmitting = false;
    state.runSubmitPhase = "";
    runCreationFlight.release();
    renderRunSubmissionState();
  }
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(2);
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return "-";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / (1024 ** index)).toFixed(index ? 2 : 0)} ${units[index]}`;
}

function formatDuration(value) {
  const seconds = Number(value || 0);
  if (!Number.isFinite(seconds) || seconds <= 0) return "-";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

async function refreshCatalogData(options = {}) {
  const runsRequestGeneration = ++state.runsRefreshGeneration;
  const requests = [
    ["模型", "modelFiles", api("/api/model-files")],
    ["视频集", "videoGroups", api("/api/video-groups?summary=1")],
    ["运行记录", "runs", requestRunsPage()],
    ["指标环境", "metricHealth", api(options.refreshMetricHealth ? "/api/metrics/health?refresh=1" : "/api/metrics/health")],
    ["设备", "devices", api("/api/devices")],
  ];
  const results = await Promise.allSettled(requests.map((row) => row[2]));
  const failures = [];
  let runsUpdated = false;
  results.forEach((result, index) => {
    const [label, key] = requests[index];
    if (result.status === "rejected") {
      failures.push(`${label}: ${result.reason?.message || result.reason}`);
      return;
    }
    if (key === "runs") {
      if (runsRequestGeneration === state.runsRefreshGeneration) {
        applyRunListPayload(result.value);
        runsUpdated = true;
      }
    } else {
      state[key] = result.value;
    }
  });
  renderMetricOptions();
  renderOptions();
  try {
    if (!state.inferenceDraftRestored) await restoreInferenceDraft();
    await loadCheckpointsForModel($("infer-form").elements.model_file.value, {
      force: Boolean(options.refreshCheckpoints),
    });
  } catch (error) {
    failures.push(`Checkpoint: ${error.message || error}`);
  }
  try {
    await loadSelectedVideoGroupIndexes();
  } catch (error) {
    failures.push(`视频索引: ${error.message || error}`);
  }
  renderMetricEnvironmentPanel();
  renderVideoSelection();
  renderRuns();
  schedulePreflight(0);
  if (runsUpdated && isRunsViewActive()) {
    if (!state.selectedRun && state.runs.length) {
      await selectRun(state.runs[0].id, { quiet: true });
    } else if (state.selectedRun) {
      const exists = state.runs.some((item) => Number(item.id) === Number(state.selectedRun.id));
      if (exists) await selectRun(state.selectedRun.id, { quiet: true });
      else {
        state.selectedRun = null;
        renderEmptyRunDetail();
      }
    } else {
      renderEmptyRunDetail();
    }
  }
  return { failures };
}

async function openView(view, options = {}) {
  const normalized = ROUTE_VIEWS.has(view) ? view : "create";
  switchView(normalized, {
    updateRoute: options.updateRoute !== false,
    replace: Boolean(options.replace),
  });
  if (normalized === "compare") {
    renderCompareMetricOptions();
    if (!state.compareSourcesLoaded) await loadCompareSources({ gtPage: 1, predPage: 1 });
    else renderCompareSelection();
    scheduleComparePreflight(0);
    return;
  }
  if (normalized === "stats") {
    await loadStats();
    return;
  }
  if (normalized === "media") {
    await loadMediaLibrary();
    await window.VFIEvalStudio?.load?.();
    return;
  }
  if (normalized === "evaluations") {
    if (!window.VFIEvalStudio) throw new Error("Evaluation Studio 尚未加载");
    await window.VFIEvalStudio.load();
    return;
  }
  if (normalized !== "runs") return;
  const requestedRunId = Math.max(0, Number(options.runId || 0));
  if (requestedRunId) {
    await selectRun(requestedRunId, { quiet: true });
  } else if (!state.runs.length) {
    renderEmptyRunDetail();
    return;
  } else if (!state.selectedRun) {
    await selectRun(state.runs[0].id, { quiet: true });
  } else {
    const exists = state.runs.some((row) => Number(row.id) === Number(state.selectedRun.id));
    await selectRun(exists ? state.selectedRun.id : state.runs[0].id, { quiet: true });
  }
  if (!state.selectedRun) return;
  const videoName = String(options.video || "").trim();
  if (videoName) {
    state.selectedVideoByRun[state.selectedRun.id] = videoName;
    await loadRunVideoTimeline(state.selectedRun.id, videoName);
    renderRunDetail();
  }
  if (options.frame !== undefined && options.frame !== null && options.frame !== "") {
    await setSampleByFrame(
      state.selectedVideoByRun[state.selectedRun.id] || videoName,
      Number(options.frame),
    );
  }
}

async function applyRouteFromLocation() {
  const params = new URLSearchParams(window.location.search);
  const view = ROUTE_VIEWS.has(params.get("view")) ? params.get("view") : "create";
  state.applyingRoute = true;
  try {
    await openView(view, {
      updateRoute: false,
      runId: params.get("run"),
      video: params.get("video"),
      frame: params.get("frame"),
    });
  } finally {
    state.applyingRoute = false;
  }
}

async function refreshCatalog() {
  const failures = [];
  try {
    await joinRunningCatalogSync();
  } catch (error) {
    failures.push(`目录同步: ${error.message || error}`);
  }
  const result = await refreshCatalogData();
  return { failures: [...failures, ...(result.failures || [])] };
}

document.addEventListener("click", (event) => {
  const executionProfile = event.target.closest?.("[data-apply-execution-profile]");
  if (executionProfile) {
    applyExecutionProfileRecommendation();
    return;
  }
  const campaignDependency = event.target.closest?.("[data-open-campaign-dependency]");
  if (campaignDependency) {
    const campaignId = Number(campaignDependency.dataset.openCampaignDependency || 0);
    (async () => {
      if (!campaignId) return;
      switchView("evaluations");
      if (!window.VFIEvalStudio) throw new Error("Evaluation Studio 尚未加载");
      await window.VFIEvalStudio.load();
      await window.VFIEvalStudio.openCampaign(`v2:${campaignId}`);
    })().catch((error) => toast(error.message));
    return;
  }
  const button = event.target.closest?.("[data-compare-input-variant]");
  if (!button) return;
  const tile = button.closest("[data-compare-input-slot]");
  const media = tile?.querySelector("[data-compare-input-media]");
  const variant = button.dataset.compareInputVariant;
  const base = media?.dataset.compareInputBase;
  if (!media || !base || !["original", "aligned"].includes(variant)) return;
  const next = `${base}?variant=${variant}`;
  if (media.src !== new URL(next, window.location.origin).href) {
    media.src = next;
    if (media.tagName === "VIDEO") media.load();
  }
  tile.querySelectorAll("[data-compare-input-variant]").forEach((item) => {
    const active = item === button;
    item.classList.toggle("active", active);
    item.setAttribute("aria-pressed", String(active));
  });
});

document.querySelectorAll(".nav-item").forEach((item) => item.addEventListener("click", () => {
  openView(item.dataset.view).catch((error) => toast(error.message));
}));
window.addEventListener("popstate", () => {
  applyRouteFromLocation().catch((error) => toast(error.message));
});
$("infer-form").addEventListener("submit", (event) => startRun(event).catch((error) => toast(error.message)));
$("compare-form").addEventListener("submit", (event) => startCompareRun(event).catch((error) => toast(error.message)));
$("create-adhoc-evaluation").addEventListener("click", () => createAdhocEvaluation().catch((error) => toast(error.message)));
$("collection-form").addEventListener("submit", (event) => createMediaCollection(event).catch((error) => toast(error.message)));
$("upload-form").addEventListener("submit", (event) => uploadExternalMedia(event).catch((error) => toast(error.message)));
$("external-pred-binding-form").addEventListener("submit", (event) => bindExternalPrediction(event).catch((error) => toast(error.message)));
$("external-pred-item-group").addEventListener("change", (event) => selectExternalPredictionBindingGroup(event).catch((error) => toast(error.message)));
$("external-pred-item").addEventListener("change", (event) => {
  state.selectedExternalPredItem = state.externalPredItems.find((item) =>
    String(compareItemId(item)) === String(event.target.value || "")) || state.selectedExternalPredItem;
});
$("external-pred-asset").addEventListener("change", (event) => {
  state.selectedExternalPredAsset = externalPredictionAssets().find((asset) =>
    String(asset.id) === String(event.target.value || "")) || state.selectedExternalPredAsset;
});
$("refresh").addEventListener("click", () => refreshRunResults().catch((error) => toast(error.message)));
$("refresh-files").addEventListener("click", (event) => runCatalogRefresh(event.currentTarget).catch((error) => toast(error.message)));
$("refresh-compare-sources").addEventListener("click", () => loadCompareSources({ gtPage: 1, predPage: 1 }).then(() => scheduleComparePreflight(0)).then(() => toast("对比来源已刷新")).catch((error) => toast(error.message)));
$("refresh-stats").addEventListener("click", () => loadStats().then(() => toast("统计数据已刷新")).catch((error) => toast(error.message)));
$("infer-form").addEventListener("input", () => {
  schedulePreflight();
  scheduleInferenceDraftSave();
});
$("compare-form").addEventListener("input", () => scheduleComparePreflight());
$("refresh-media").addEventListener("click", (event) => runCatalogRefresh(event.currentTarget, {
  includeMedia: true,
  includeRuns: true,
}).catch((error) => toast(error.message)));
$("pause-upload").addEventListener("click", () => {
  const task = state.uploadTask;
  if (!task) {
    toast("当前没有进行中的文件校验或上传");
    return;
  }
  state.uploadPaused = true;
  task.controller.abort();
  task.worker?.terminate();
  toast(task.phase === "hashing" ? "正在取消文件校验" : "正在暂停上传，已完成分片可续传");
});
$("infer-form").addEventListener("change", async (event) => {
  renderCustomSizeVisibility();
  if (event.target.name === "model_file") {
    try {
      await loadCheckpointsForModel(event.target.value);
    } catch (error) {
      toast(`Checkpoint: ${error.message || error}`);
    }
  }
  if (event.target.name === "execution_mode") {
    renderSingleDeviceOptions($("infer-form").elements.device.value || "auto");
    renderDeviceOptions();
  }
  schedulePreflight(0);
  scheduleInferenceDraftSave();
});

document.addEventListener("input", (event) => {
  const mediaFilter = event.target.closest?.("[data-media-filter]");
  if (mediaFilter) {
    state.mediaFilters[mediaFilter.dataset.mediaFilter] = mediaFilter.value || "";
    scheduleMediaFilterRefresh(mediaFilter.dataset.mediaFilter === "q" ? 300 : 0);
    return;
  }
  const filter = event.target.closest?.("[data-run-filter]");
  if (!filter) return;
  state.runFilters[filter.dataset.runFilter] = filter.value || "";
  state.runsPage.page = 1;
  scheduleRunFilterRefresh(["q", "model"].includes(filter.dataset.runFilter) ? 300 : 0);
});

document.addEventListener("change", (event) => {
  const mediaFilter = event.target.closest("[data-media-filter]");
  if (mediaFilter) {
    state.mediaFilters[mediaFilter.dataset.mediaFilter] = mediaFilter.value || "";
    scheduleMediaFilterRefresh(0);
    return;
  }
  const runFilter = event.target.closest("[data-run-filter]");
  if (runFilter) {
    state.runFilters[runFilter.dataset.runFilter] = runFilter.value || "";
    state.runsPage.page = 1;
    scheduleRunFilterRefresh(0);
    return;
  }
  const statsFilter = event.target.closest("[data-stats-filter]");
  if (statsFilter) {
    state.statsFilters[statsFilter.dataset.statsFilter] = statsFilter.value || "";
    loadStats().catch((error) => toast(error.message));
    return;
  }
  const groupToggle = event.target.closest("[data-group-toggle]");
  if (groupToggle) {
    const name = groupToggle.dataset.groupToggle;
    if (groupToggle.checked) {
      state.selectedGroups.add(name);
    } else {
      state.selectedGroups.delete(name);
    }
    resetVideoSelectionSnapshot();
    for (const selectedGroup of selectedGroupNames()) {
      delete state.videoPages[selectedGroup];
    }
    renderVideoSelection();
    loadSelectedVideoGroupIndexes()
      .then(() => schedulePreflight(0))
      .catch((error) => toast(error.message));
    scheduleInferenceDraftSave();
    return;
  }
  const compareGroup = event.target.closest("[data-compare-group]");
  if (compareGroup) {
    state.selectedCompareGroupId = compareGroup.value || "";
    state.selectedCompareItemId = null;
    state.selectedCompareItemSnapshot = null;
    state.selectedComparePredMembers.clear();
    state.compareItemQuery = "";
    state.compareItemPage = 1;
    renderCompareSubmissionState();
    loadCompareSources({ page: 1 }).then(() => scheduleComparePreflight(0)).catch((error) => toast(error.message));
    return;
  }
  const compareItem = event.target.closest("[data-compare-item]");
  if (compareItem) {
    state.selectedCompareItemId = Number(compareItem.dataset.compareItem);
    state.selectedCompareItemSnapshot = (state.compareItems || []).find(
      (row) => compareItemId(row) === state.selectedCompareItemId,
    ) || null;
    state.selectedComparePredMembers.clear();
    renderCompareSubmissionState();
    loadCompareSources().then(() => scheduleComparePreflight(0)).catch((error) => toast(error.message));
    return;
  }
  const comparePred = event.target.closest("[data-compare-pred]");
  if (comparePred) {
    const memberId = Number(comparePred.dataset.comparePred);
    if (comparePred.checked) {
      if (state.selectedComparePredMembers.size >= 2) {
        comparePred.checked = false;
        toast("Compare 最多选择两份 Pred");
        return;
      }
      state.selectedComparePredMembers.add(memberId);
    } else state.selectedComparePredMembers.delete(memberId);
    renderCompareSelection();
    scheduleComparePreflight(0);
    return;
  }
  const compareQuery = event.target.closest("[data-compare-query]");
  if (compareQuery) {
    state.compareItemQuery = compareQuery.value || "";
    state.compareItemPage = 1;
    loadCompareSources().then(() => scheduleComparePreflight(0)).catch((error) => toast(error.message));
    return;
  }
  const compareMetric = event.target.closest("#compare-metrics-options input[name='compare_metrics']");
  if (compareMetric) {
    scheduleComparePreflight(0);
    return;
  }
  const videoQuery = event.target.closest("[data-video-query]");
  if (videoQuery) {
    const name = videoQuery.dataset.group;
    if (!name) return;
    state.videoPageQuery[name] = videoQuery.value || "";
    loadVideoGroupPage(name, 1).then(() => schedulePreflight(0)).catch((error) => toast(error.message));
    return;
  }
  const videoSort = event.target.closest("[data-video-sort]");
  if (videoSort) {
    const name = videoSort.dataset.group;
    if (!name) return;
    state.videoPageSort[name] = videoSort.value || "name";
    loadVideoGroupPage(name, 1).then(() => schedulePreflight(0)).catch((error) => toast(error.message));
    return;
  }
  const videoCheckbox = event.target.closest("[data-video-name]");
  if (videoCheckbox) {
    const name = videoCheckbox.dataset.group;
    if (!name) return;
    mutateVideoSelectionSnapshot(
      name,
      videoCheckbox.checked ? "add" : "remove",
      { videoNames: [videoCheckbox.dataset.videoName] },
    ).catch((error) => toast(error.message));
    return;
  }
  const cudaDevice = event.target.closest("[data-cuda-device]");
  if (cudaDevice) {
    if (cudaDevice.checked) state.selectedCudaDevices.add(cudaDevice.dataset.cudaDevice);
    else state.selectedCudaDevices.delete(cudaDevice.dataset.cudaDevice);
    renderDeviceOptions();
    schedulePreflight(0);
    return;
  }
  const npuDevice = event.target.closest("[data-npu-device]");
  if (npuDevice) {
    if (npuDevice.checked) state.selectedNpuDevices.add(npuDevice.dataset.npuDevice);
    else state.selectedNpuDevices.delete(npuDevice.dataset.npuDevice);
    renderDeviceOptions();
    schedulePreflight(0);
    return;
  }
  const metricSelect = event.target.closest("[data-metric-select]");
  if (metricSelect && state.selectedRun) {
    state.selectedMetricByRun[state.selectedRun.id] = metricSelect.value;
    const videoName = metricSelect.dataset.metricSelect;
    loadRunVideoTimeline(state.selectedRun.id, videoName, { metric: metricSelect.value })
      .then(() => { if (!updateFrameRegion()) renderRunDetail(); })
      .catch((error) => toast(error.message));
    return;
  }
  const slotSelect = event.target.closest("[data-slot]");
  if (slotSelect) {
    const sampleId = slotSelect.dataset.slotSample;
    const options = sampleLayerOptions(state.selectedRun);
    const current = slotSelection(sampleId, options);
    state.slotSelectionBySample[sampleId] = { ...current, [slotSelect.dataset.slot]: slotSelect.value };
    if (!updateFrameRegion()) renderRunDetail();
    return;
  }
  const range = event.target.closest("[data-sample-range]");
  if (range) {
    setGlobalSampleIndex(range.dataset.sampleRange, Number(range.value)).catch((error) => toast(error.message));
  }
});

document.addEventListener("submit", (event) => {
  const feedbackForm = event.target.closest("[data-feedback-form]");
  if (feedbackForm) {
    event.preventDefault();
    submitRunFeedback(Number(feedbackForm.dataset.feedbackForm), feedbackForm).catch((error) => toast(error.message));
    return;
  }
  const feedbackEditForm = event.target.closest("[data-feedback-edit-form]");
  if (feedbackEditForm) {
    event.preventDefault();
    submitFeedbackEdit(
      Number(feedbackEditForm.dataset.feedbackRun),
      Number(feedbackEditForm.dataset.feedbackEditForm),
      feedbackEditForm,
    ).catch((error) => toast(error.message));
  }
});

document.addEventListener("click", async (event) => {
  const externalPredItemPage = event.target.closest("[data-external-pred-item-page]");
  if (externalPredItemPage) {
    const selected = state.externalPredItems.find((item) =>
      String(compareItemId(item)) === String($("external-pred-item")?.value || ""));
    if (selected) state.selectedExternalPredItem = selected;
    await loadExternalPredictionBindingItems({ page: Number(externalPredItemPage.dataset.externalPredItemPage || 1) });
    return;
  }
  const mediaLoadMore = event.target.closest("[data-media-load-more]");
  if (mediaLoadMore) {
    await loadMoreMediaSources();
    return;
  }
  if (event.target.closest("[data-media-filter-reset]")) {
    state.mediaFilters = { q: "", role: "", source_kind: "", collection_id: "" };
    await reloadMediaAssets(1);
    return;
  }
  const mediaDelete = event.target.closest("[data-media-delete]");
  if (mediaDelete) {
    await deleteMediaAsset(Number(mediaDelete.dataset.mediaDelete));
    return;
  }
  const feedbackDelete = event.target.closest("[data-feedback-delete]");
  if (feedbackDelete) {
    await deleteRunFeedback(Number(feedbackDelete.dataset.feedbackRun), Number(feedbackDelete.dataset.feedbackDelete));
    return;
  }
  const feedbackEdit = event.target.closest("[data-feedback-edit]");
  if (feedbackEdit) {
    state.editingFeedback = Number(feedbackEdit.dataset.feedbackEdit);
    renderRunDetail();
    return;
  }
  const feedbackCancelEdit = event.target.closest("[data-feedback-cancel-edit]");
  if (feedbackCancelEdit) {
    state.editingFeedback = null;
    renderRunDetail();
    return;
  }
  const statsFilterReset = event.target.closest("[data-stats-filter-reset]");
  if (statsFilterReset) {
    state.statsFilters = { dataset: "", model: "", checkpoint: "", video: "" };
    await loadStats().catch((error) => toast(error.message));
    return;
  }
  const statsRun = event.target.closest("[data-stats-run]");
  if (statsRun) {
    await openView("runs", { runId: Number(statsRun.dataset.statsRun) });
    return;
  }
  const loadVideoPageBtn = event.target.closest("[data-load-video-page]");
  if (loadVideoPageBtn) {
    const groupName = loadVideoPageBtn.dataset.loadVideoPage || primaryGroupName();
    if (!groupName) return;
    await loadVideoGroupPage(groupName, 1);
    schedulePreflight(0);
    return;
  }
  if (event.target.closest("[data-refresh-compare-sources]")) {
    await loadCompareSources({ gtPage: 1, predPage: 1 });
    scheduleComparePreflight(0);
    return;
  }
  const comparePage = event.target.closest("[data-compare-page]");
  if (comparePage) {
    const page = Number(comparePage.dataset.page || 1);
    await loadCompareSources({ page });
    scheduleComparePreflight(0);
    return;
  }
  const videoPage = event.target.closest("[data-video-page]");
  if (videoPage) {
    const groupName = videoPage.dataset.videoGroup;
    if (!groupName) return;
    await loadVideoGroupPage(groupName, Number(videoPage.dataset.videoPage || 1));
    schedulePreflight(0);
    return;
  }
  const videoSelect = event.target.closest("[data-video-select]");
  if (videoSelect) {
    const groupName = videoSelect.dataset.group;
    if (!groupName) return;
    const operations = {
      "all-filtered": "add_filtered",
      "none-filtered": "remove_filtered",
      "invert-filtered": "toggle_filtered",
    };
    mutateVideoSelectionSnapshot(
      groupName,
      operations[videoSelect.dataset.videoSelect],
      { q: state.videoPageQuery[groupName] || "" },
    ).catch((error) => toast(error.message));
    return;
  }
  const runSelect = event.target.closest("[data-run-select]");
  if (runSelect) {
    // Toggle without opening the detail view; the checkbox lives inside a
    // clickable row so stopPropagation is not enough — handle it first.
    const id = Number(runSelect.dataset.runSelect);
    if (runSelect.checked) state.selectedRunIds.add(id);
    else state.selectedRunIds.delete(id);
    renderRuns();
    return;
  }
  const selectAll = event.target.closest("[data-runs-select-all]");
  if (selectAll) {
    if (selectAll.checked) state.runs.forEach((run) => state.selectedRunIds.add(Number(run.id)));
    else state.selectedRunIds.clear();
    renderRuns();
    return;
  }
  if (event.target.closest("[data-runs-batch-delete]")) {
    await batchDeleteRuns();
    return;
  }
  const runsPage = event.target.closest("[data-runs-page]");
  if (runsPage) {
    state.runsPage.page = Math.max(1, Number(runsPage.dataset.runsPage || 1));
    state.selectedRunIds.clear();
    await refreshRunsOnly({ page: state.runsPage.page });
    return;
  }
  if (event.target.closest("[data-runs-filter-reset]")) {
    state.runFilters = { q: "", status: "", run_type: "", model: "" };
    state.runsPage.page = 1;
    state.selectedRunIds.clear();
    await refreshRunsOnly({ page: 1 });
    return;
  }
  const runButton = event.target.closest("[data-run-id]");
  if (runButton) {
    await selectRun(Number(runButton.dataset.runId));
    return;
  }
  const refreshResultsButton = event.target.closest("[data-refresh-run-results]");
  if (refreshResultsButton) {
    await refreshRunResults(Number(refreshResultsButton.dataset.refreshRunResults));
    return;
  }
  const cancelButton = event.target.closest("[data-cancel-run]");
  if (cancelButton) {
    await cancelRun(Number(cancelButton.dataset.cancelRun));
    return;
  }
  const retryMetricsButton = event.target.closest("[data-retry-run-metrics]");
  if (retryMetricsButton) {
    await retryRunMetrics(Number(retryMetricsButton.dataset.retryRunMetrics));
    return;
  }
  const retryButton = event.target.closest("[data-retry-run]");
  if (retryButton) {
    await retryRun(Number(retryButton.dataset.retryRun));
    return;
  }
  const cloneButton = event.target.closest("[data-clone-run]");
  if (cloneButton) {
    await cloneRunWithCurrentInputs(Number(cloneButton.dataset.cloneRun));
    return;
  }
  const deleteButton = event.target.closest("[data-delete-run]");
  if (deleteButton) {
    await deleteRun(Number(deleteButton.dataset.deleteRun));
    return;
  }
  const cleanupButton = event.target.closest("[data-cleanup-run]");
  if (cleanupButton) {
    await cleanupRunArtifacts(Number(cleanupButton.dataset.cleanupRun));
    return;
  }
  const renameButton = event.target.closest("[data-rename-run]");
  if (renameButton) {
    await renameRun(Number(renameButton.dataset.renameRun));
    return;
  }
  const runVideosPage = event.target.closest("[data-run-videos-page]");
  if (runVideosPage && state.selectedRun) {
    abortSampleRequestsForRun(state.selectedRun.id);
    await loadRunVideosPage(state.selectedRun.id, Number(runVideosPage.dataset.runVideosPage));
    renderRunDetail();
    return;
  }
  const gridColumns = event.target.closest("[data-compare-grid-columns]");
  if (gridColumns) {
    state.compareGridColumns = Number(gridColumns.dataset.compareGridColumns || 3);
    if (!updateFrameRegion()) renderRunDetail();
    return;
  }
  const slotLayout = event.target.closest("[data-slot-layout]");
  if (slotLayout) {
    state.compareSlotLayout = slotLayout.dataset.slotLayout === "stack" ? "stack" : "side";
    if (!updateFrameRegion()) renderRunDetail();
    return;
  }
  if (event.target.closest("[data-master-video-play]")) {
    syncActiveVideos("play");
    return;
  }
  if (event.target.closest("[data-master-video-pause]")) {
    syncActiveVideos("pause");
    return;
  }
  if (event.target.closest("[data-master-video-sync]")) {
    syncActiveVideos("sync");
    return;
  }
  const videoTab = event.target.closest("[data-run-video]");
  if (videoTab && state.selectedRun) {
    abortSampleRequestsForRun(state.selectedRun.id);
    state.selectedVideoByRun[state.selectedRun.id] = videoTab.dataset.runVideo;
    if (!state.runVideoTimelines[`${state.selectedRun.id}:${videoTab.dataset.runVideo}`]) {
      await loadRunVideoTimeline(state.selectedRun.id, videoTab.dataset.runVideo);
    }
    renderRunDetail();
    syncBrowserRoute({ view: "runs", replace: true });
    return;
  }
  const windowNav = event.target.closest("[data-window-start]");
  if (windowNav && state.selectedRun) {
    const videoName = state.selectedVideoByRun[state.selectedRun.id];
    const nextStart = Math.max(0, Number(windowNav.dataset.windowStart || 0));
    abortSampleRequestsForRun(state.selectedRun.id);
    await loadRunVideoTimeline(state.selectedRun.id, videoName, { windowStart: nextStart });
    // Reset the in-window selection to the first frame of the new window.
    state.selectedSampleByVideo[`${state.selectedRun.id}:${videoName}`] = 0;
    if (!updateFrameRegion()) renderRunDetail();
    syncBrowserRoute({ view: "runs", replace: true });
    return;
  }
  const stepButton = event.target.closest("[data-sample-step]");
  if (stepButton && state.selectedRun) {
    const videoName = state.selectedVideoByRun[state.selectedRun.id];
    const key = `${state.selectedRun.id}:${videoName}`;
    const currentIndex = Number(state.selectedSampleByVideo[key] || 0);
    const direction = Number(stepButton.dataset.sampleStep);
    // Compare runs store one sample per (track, frame), so a naive ±1 step
    // would move between tracks of the same frame. Step by distinct frame.
    if (isCompareRun(state.selectedRun)) {
      const video = state.runVideoTimelines[key];
      if (video) {
        const nextIndex = compareStepIndex(video, currentIndex, direction);
        if (nextIndex !== currentIndex) {
          setSampleIndex(videoName, nextIndex);
          return;
        }
      }
    }
    const video = state.runVideoTimelines[key];
    const globalIndex = Number(video?.window_start || 0) + currentIndex + direction;
    await setGlobalSampleIndex(videoName, globalIndex);
    return;
  }
  const statusDot = event.target.closest("[data-sample-jump]");
  if (statusDot && state.selectedRun) {
    const videoName = state.selectedVideoByRun[state.selectedRun.id];
    setSampleIndex(videoName, Number(statusDot.dataset.sampleJump));
    return;
  }
  const worstItem = event.target.closest("[data-sample-frame]");
  if (worstItem) {
    await setSampleByFrame(worstItem.dataset.sampleVideo, Number(worstItem.dataset.sampleFrame));
    return;
  }
  const chartPoint = event.target.closest("[data-chart-sample]");
  if (chartPoint && state.selectedRun) {
    setSampleIndex(chartPoint.dataset.chartVideo, Number(chartPoint.dataset.chartSample));
    return;
  }
  const overview = event.target.closest("[data-overview-video]");
  if (overview && state.selectedRun) {
    const video = state.runVideoTimelines[`${state.selectedRun.id}:${overview.dataset.overviewVideo}`];
    if (!video) return;
    const rect = overview.querySelector(".overview-plot")?.getBoundingClientRect() || overview.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width)));
    await setGlobalSampleIndex(overview.dataset.overviewVideo, Math.round(ratio * Math.max(0, Number(video.sample_count || 1) - 1)));
    return;
  }
  const artifactGroup = event.target.closest("[data-artifact-group]");
  if (artifactGroup) {
    state.selectedArtifactGroupBySample[artifactGroup.dataset.artifactSample] = artifactGroup.dataset.artifactGroup;
    if (!updateFrameRegion()) renderRunDetail();
    return;
  }
  const extraToggle = event.target.closest("[data-extra-toggle]");
  if (extraToggle) {
    const sampleId = extraToggle.dataset.extraToggle;
    state.expandedExtraArtifactsBySample[sampleId] = !state.expandedExtraArtifactsBySample[sampleId];
    if (!updateFrameRegion()) renderRunDetail();
    return;
  }
  const chart = event.target.closest("[data-chart-video]");
  if (chart && event.target.closest(".chart-plot") && state.selectedRun) {
    const video = state.runVideoTimelines[`${state.selectedRun.id}:${chart.dataset.chartVideo}`];
    if (!video?.samples?.length) return;
    const plot = chart.querySelector(".chart-plot") || chart;
    const rect = plot.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    if (chart.dataset.chartMetric) state.selectedMetricByRun[state.selectedRun.id] = chart.dataset.chartMetric;
    setSampleIndex(chart.dataset.chartVideo, Math.round(ratio * (video.samples.length - 1)));
  }
});

document.addEventListener("keydown", (event) => {
  const chart = event.target.closest?.(".chart[data-chart-video]");
  if (!chart || !state.selectedRun) return;
  const video = state.runVideoTimelines[`${state.selectedRun.id}:${chart.dataset.chartVideo}`];
  const samples = video?.samples || [];
  if (!samples.length) return;
  const key = `${state.selectedRun.id}:${chart.dataset.chartVideo}`;
  const current = Number(state.selectedSampleByVideo[key] || 0);
  const target = {
    ArrowLeft: current - 1,
    ArrowDown: current - 1,
    ArrowRight: current + 1,
    ArrowUp: current + 1,
    Home: 0,
    End: samples.length - 1,
  }[event.key];
  if (target === undefined) return;
  event.preventDefault();
  if (chart.dataset.chartMetric) state.selectedMetricByRun[state.selectedRun.id] = chart.dataset.chartMetric;
  setSampleIndex(chart.dataset.chartVideo, target);
});

document.addEventListener("mouseover", (event) => {
  const layerTile = event.target.closest("[data-layer-frame]");
  if (layerTile) {
    highlightTimelineFrame(layerTile.dataset.layerFrame);
  }
});

document.addEventListener("mousemove", (event) => {
  const chart = event.target.closest("[data-chart-video]");
  if (!chart || !state.selectedRun) return;
  const video = state.runVideoTimelines[`${state.selectedRun.id}:${chart.dataset.chartVideo}`];
  const samples = video?.samples || [];
  if (!samples.length) return;
  const plot = chart.querySelector(".chart-plot");
  const tooltip = chart.querySelector(".chart-tooltip");
  if (!plot || !tooltip) return;
  const rect = plot.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width)));
  const index = Math.round(ratio * (samples.length - 1));
  const sample = samples[index];
  const metricName = chart.dataset.chartMetric || selectedMetric(video);
  const metric = sample?.metrics?.[metricName];
  tooltip.hidden = false;
  tooltip.style.left = `${ratio * 100}%`;
  tooltip.textContent = `frame ${sample?.frame_index ?? "-"} · ${metric?.status || "missing"} · ${metric?.value === null || metric?.value === undefined ? metricReason(metric) : formatNumber(metric.value)}`;
});

document.addEventListener("mouseout", (event) => {
  if (event.target.closest("[data-layer-frame]")) {
    highlightTimelineFrame(null);
  }
  const chart = event.target.closest("[data-chart-video]");
  if (chart && !chart.contains(event.relatedTarget)) {
    const tooltip = chart.querySelector(".chart-tooltip");
    if (tooltip) tooltip.hidden = true;
  }
});

// `toggle` does not bubble, so capture it to persist the run-meta collapse
// state across the 2s poll re-render of a running run.
document.addEventListener("toggle", (event) => {
  const details = event.target;
  if (details instanceof HTMLDetailsElement && details.classList.contains("run-meta")) {
    state.runMetaCollapsed = !details.open;
  } else if (details instanceof HTMLDetailsElement && details.classList.contains("selection-diagnostics")) {
    const groupName = details.dataset.groupBlock;
    if (!groupName) return;
    if (details.open) state.expandedVideoGroups.add(groupName);
    else state.expandedVideoGroups.delete(groupName);
  }
}, true);

function startRunsPoll() {
  let timer = null;
  const hasActiveRunWork = () => Number(state.runsPage.active_total || 0) > 0
    || state.runs.some((run) =>
      !TERMINAL_STATUSES.has(run.status)
      || ["requested", "canceling", "purging"].includes(runPurgeState(run)));
  const renderPollStatus = () => {
    const host = $("runs-poll-status");
    if (!host) return;
    if (state.runPoll.error) {
      const lastGood = state.runPoll.lastSuccessAt
        ? new Date(state.runPoll.lastSuccessAt).toLocaleTimeString()
        : "尚未成功";
      host.classList.add("stale");
      host.textContent = `自动刷新暂时中断 · 上次成功 ${lastGood} · 正在退避重试`;
      host.title = state.runPoll.error;
      return;
    }
    host.classList.remove("stale");
    host.removeAttribute("title");
    host.textContent = state.runPoll.lastSuccessAt
      ? `自动刷新正常 · ${new Date(state.runPoll.lastSuccessAt).toLocaleTimeString()}`
      : "正在连接自动刷新…";
  };
  const pollDelay = () => {
    const base = hasActiveRunWork() ? 2000 : 10000;
    return Math.min(60000, base * (2 ** Math.min(4, state.runPoll.consecutiveErrors)));
  };
  const pollOnce = async () => {
    try {
      await refreshRunsOnly();
      state.runPoll.lastSuccessAt = Date.now();
      state.runPoll.consecutiveErrors = 0;
      state.runPoll.error = "";
    } catch (error) {
      state.runPoll.lastErrorAt = Date.now();
      state.runPoll.consecutiveErrors += 1;
      state.runPoll.error = error.message || String(error);
    }
    renderPollStatus();
  };
  const schedule = (delay = pollDelay()) => {
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(async () => {
      timer = null;
      if (!document.hidden) await pollOnce();
      schedule(pollDelay());
    }, delay);
  };
  renderPollStatus();
  schedule();
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (timer !== null) clearTimeout(timer);
      timer = null;
      return;
    }
    pollOnce().finally(() => schedule(pollDelay()));
  });
}

function bootstrapApp() {
  refreshCatalog()
    .then(async (result) => {
      if (result.failures?.length) toast(`启动时有 ${result.failures.length} 个数据面板未能加载`);
      await applyRouteFromLocation();
    })
    .catch((error) => toast(error.message));
  refreshDeploymentHealth();
  setInterval(() => {
    if (!document.hidden) refreshDeploymentHealth();
  }, 30000);
  startRunsPoll();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrapApp, { once: true });
} else {
  bootstrapApp();
}
