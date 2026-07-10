const STATUS_LABELS = {
  decoding: "Decoding",
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
const TIMELINE_WINDOW_SIZE = 1000;
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
  devices: null,
  runs: [],
  metricHealth: null,
  preflight: null,
  preflightTimer: null,
  selectedRun: null,
  metricSummary: null,
  runVideosPage: null,
  runVideoTimelines: {},
  timelineWindowStartByVideo: {},
  sampleDetails: {},
  sampleDetailLoading: {},
  selectedVideosByGroup: {},
  selectedGroups: new Set(),
  videoPages: {},
  videoPageQuery: {},
  videoPageSort: {},
  runVideoPageByRun: {},
  selectedVideoByRun: {},
  selectedSampleByVideo: {},
  selectedMetricByRun: {},
  runMetaCollapsed: false,
  selectedArtifactGroupBySample: {},
  expandedExtraArtifactsBySample: {},
  selectedCudaDevices: new Set(),
  selectedNpuDevices: new Set(),
  compareSources: { gt: [], pred: [] },
  compareSourcesMeta: { gt: null, pred: null },
  comparePredByArtifact: {},
  compareSourcesLoaded: false,
  compareGtQuery: "",
  comparePredQuery: "",
  compareGtPage: 1,
  comparePredPage: 1,
  selectedCompareGtKey: "",
  selectedComparePredArtifacts: new Set(),
  selectedCompareLayerKinds: new Set(),
  compareTrackLabels: {},
  preflightAbortController: null,
  preflightPayloadKey: "",
  comparePreflight: null,
  comparePreflightTimer: null,
  comparePreflightPayloadKey: "",
  comparePreflightAbortController: null,
  sampleAbortControllers: {},
  timelineAbortController: null,
  compareGridColumns: 3,
  slotSelectionBySample: {},
  compareSlotLayout: "side",
  selectedRunIds: new Set(),
  feedbackUsername: "",
  feedbackStats: null,
  editingFeedback: null,
  statsFilters: { dataset: "", model: "", checkpoint: "", video: "" },
  mediaCollections: [],
  mediaAssets: [],
  activeUpload: null,
  uploadPaused: false,
  evaluatorId: localStorage.getItem("vfieval-evaluator-id") || "",
  evaluatorName: localStorage.getItem("vfieval-evaluator-name") || "",
  evaluationCampaigns: [],
  selectedCampaignId: null,
  campaignCandidates: [],
  currentEvaluationTask: null,
  campaignAnalysis: null,
  evaluationTaskStartedAt: 0,
  evaluationFrameIndex: 0,
  campaignAnalysisFilters: { video: "", model: "", checkpoint: "", collection_id: "", evaluator_id: "" },
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error?.message || response.statusText);
  }
  return data;
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

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2600);
}

function switchView(view) {
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  document.querySelectorAll(".view").forEach((item) => item.classList.toggle("active", item.id === `view-${view}`));
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

function previewGroupsForRun(run) {
  return isCompareRun(run)
    ? PREVIEW_GROUPS_COMPARE
    : PREVIEW_GROUPS_MODEL;
}

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

  renderCheckpointOptions(previousCheckpoint);
  renderSingleDeviceOptions(previousDevice);
  renderDeviceOptions();
  renderGroupPicker();
  for (const name of selectedGroupNames()) ensureVideoSelection(name);
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

function ensureVideoSelection(groupName) {
  const group = state.videoGroups.find((item) => item.name === groupName);
  if (!group) return new Set();
  const page = state.videoPages[groupName];
  const names = page?.all_video_names || (group.videos || []).map((video) => video.name);
  if (!names.length && !state.selectedVideosByGroup[groupName]) {
    return new Set();
  }
  if (!state.selectedVideosByGroup[groupName]) {
    state.selectedVideosByGroup[groupName] = new Set(names);
  }
  return state.selectedVideosByGroup[groupName];
}

function groupVideoNames(groupName) {
  const group = groupByName(groupName);
  if (!group) return [];
  const page = state.videoPages[groupName];
  return page?.all_video_names || (group.videos || []).map((video) => video.name);
}

function selectedVideoNamesForGroup(groupName) {
  const names = groupVideoNames(groupName);
  const selected = ensureVideoSelection(groupName);
  return names.filter((name) => selected.has(name));
}

function selectedVideoNames() {
  // Single-group runs keep bare file names (byte-identical caches/reference keys);
  // multi-group runs qualify every selection as "group/file".
  const groups = selectedGroupNames();
  if (!groups.length) return [];
  if (groups.length === 1) {
    return selectedVideoNamesForGroup(groups[0]);
  }
  const qualified = [];
  for (const name of groups) {
    for (const video of selectedVideoNamesForGroup(name)) {
      qualified.push(`${name}/${video}`);
    }
  }
  return qualified;
}

function selectedSourceAssets() {
  const result = [];
  for (const groupName of selectedGroupNames()) {
    const assetIds = state.videoPages[groupName]?.asset_ids || {};
    for (const videoName of selectedVideoNamesForGroup(groupName)) {
      const assetId = Number(assetIds[videoName] || 0);
      if (assetId > 0) result.push({ asset_id: assetId });
    }
  }
  return result;
}

async function loadVideoGroupPage(groupName, page = 1) {
  if (!groupName) return;
  const query = state.videoPageQuery[groupName] || "";
  const sort = state.videoPageSort[groupName] || "name";
  const payload = await api(`/api/video-groups/${encodeURIComponent(groupName)}/videos?page=${page}&page_size=50&q=${encodeURIComponent(query)}&sort=${encodeURIComponent(sort)}`);
  state.videoPages[groupName] = payload;
  ensureVideoSelection(groupName);
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

function compareGtKey(row) {
  if (row.asset_id) return `asset:${row.asset_id}`;
  // GT is always a source clip now (one card per videos/ clip). The pred-aligned
  // GT is reconstructed backend-side from the pred's recorded source frame
  // indices, so there is no per-run GT card to disambiguate.
  return `${row.group || ""}::${row.video || ""}`;
}

function selectedCompareGt() {
  return (state.compareSources.gt || []).find((row) => compareGtKey(row) === state.selectedCompareGtKey) || null;
}

function selectedComparePredRows() {
  return Array.from(state.selectedComparePredArtifacts)
    .map((artifactId) => state.comparePredByArtifact[Number(artifactId)])
    .filter(Boolean)
    .map((row) => ({ ...row, track_label: compareTrackLabel(row) }));
}

function compareTrackLabel(row) {
  const artifactId = Number(row.artifact_id);
  return state.compareTrackLabels[artifactId] || row.compare_track_label || row.run_name || `run-${row.run_id}`;
}

// External assets must match exactly. VFIEval-generated Pred assets may carry
// source_frame_indices; only that platform-owned mapping can construct an
// aligned GT at the inference resolution.
function compareCompatibility(gt, predRow) {
  if (!gt) return null;
  const reasons = [];
  const gtFrames = Number(gt.frame_count || 0);
  const predFrames = Number(predRow.frame_count || 0);
  const mapped = Array.isArray(predRow.source_frame_indices)
    && predRow.source_frame_indices.length === predFrames;
  // GT is now the source clip (N frames); a pred is the interpolated subset
  // (N-step frames), so pred < gt is the expected relationship — the backend
  // reconstructs the aligned GT from the pred's recorded source frame indices.
  // Only a pred with MORE frames than the source clip is genuinely anomalous.
  if (gtFrames && predFrames && gtFrames !== predFrames && !mapped) {
    reasons.push(`strict 帧数不一致：${predFrames} vs ${gtFrames}`);
  }
  const gtW = Number(gt.width || 0);
  const gtH = Number(gt.height || 0);
  const predW = Number(predRow.width || 0);
  const predH = Number(predRow.height || 0);
  if (gtW && gtH && predW && predH && (gtW !== predW || gtH !== predH) && !mapped) {
    reasons.push(`strict 分辨率不一致：${predW}x${predH} vs ${gtW}x${gtH}`);
  }
  return { ok: reasons.length === 0, reasons };
}

function ensureCompareSelection() {
  const gtRows = state.compareSources.gt || [];
  if ((!state.selectedCompareGtKey || !gtRows.some((row) => compareGtKey(row) === state.selectedCompareGtKey)) && gtRows.length) {
    state.selectedCompareGtKey = compareGtKey(gtRows[0]);
  }
  for (const row of state.compareSources.pred || []) {
    const artifactId = Number(row.artifact_id);
    state.comparePredByArtifact[artifactId] = row;
    if (!state.compareTrackLabels[artifactId]) {
      state.compareTrackLabels[artifactId] = compareTrackLabel(row);
    }
  }
}

function comparePredFilterVideo() {
  const gt = selectedCompareGt();
  return gt?.video_name || (gt?.video ? String(gt.video).replace(/\.[^.]+$/, "") : "");
}

function compareSourcePager(meta, type) {
  if (!meta || Number(meta.total_pages || 1) <= 1) return "";
  return `
    <div class="pager compact-pager">
      <button class="secondary" data-compare-page="${type}" data-page="${Number(meta.page || 1) - 1}" ${Number(meta.page || 1) <= 1 ? "disabled" : ""} type="button">上一页</button>
      <span class="muted">第 ${escapeHtml(meta.page || 1)} / ${escapeHtml(meta.total_pages || 1)} 页，${escapeHtml(meta.filtered_count || 0)} 条</span>
      <button class="secondary" data-compare-page="${type}" data-page="${Number(meta.page || 1) + 1}" ${Number(meta.page || 1) >= Number(meta.total_pages || 1) ? "disabled" : ""} type="button">下一页</button>
    </div>
  `;
}

async function loadCompareSources(options = {}) {
  if (options.gtPage !== undefined) state.compareGtPage = Math.max(1, Number(options.gtPage || 1));
  if (options.predPage !== undefined) state.comparePredPage = Math.max(1, Number(options.predPage || 1));
  const gtPayload = await api(`/api/media/assets?role=gt&page=${state.compareGtPage}&page_size=50&q=${encodeURIComponent(state.compareGtQuery)}`);
  state.compareSources.gt = (gtPayload.assets || []).map((row) => ({
    ...row,
    kind: "media_asset",
    asset_id: row.id,
    group: row.collection_name,
    video: row.original_name || row.display_name,
    video_name: row.provenance?.video_name || row.provenance?.video || row.display_name,
  }));
  state.compareSourcesMeta.gt = { ...gtPayload, total_pages: gtPayload.page_count, filtered_count: gtPayload.total };
  ensureCompareSelection();
  const predPayload = await api(`/api/media/assets?role=pred&page=${state.comparePredPage}&page_size=50&q=${encodeURIComponent(state.comparePredQuery)}`);
  state.compareSources.pred = (predPayload.assets || []).map((row) => ({
    ...row,
    kind: "media_asset",
    artifact_id: row.id,
    asset_id: row.id,
    run_id: row.provenance?.run_id,
    run_name: row.provenance?.run_name || row.display_name,
    video: row.provenance?.video_name || row.display_name,
    compare_track_label: row.provenance?.track_label || "",
    source_frame_indices: row.metadata?.source_frame_indices || null,
  }));
  state.compareSourcesMeta.pred = { ...predPayload, total_pages: predPayload.page_count, filtered_count: predPayload.total };
  state.compareSourcesLoaded = true;
  ensureCompareSelection();
  renderCompareSelection();
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
            <p class="muted">只加载了分组摘要（${escapeHtml(group.video_count)} 个视频）。打开列表后再筛选、翻页。</p>
          </div>
          <button class="secondary" data-load-video-page="${escapeHtml(groupName)}" type="button">加载视频列表</button>
        </div>
      </section>
    `;
  }
  const selected = ensureVideoSelection(groupName);
  const query = state.videoPageQuery[groupName] || "";
  const sort = state.videoPageSort[groupName] || "name";
  return `
    <section class="group-video-block" data-group-block="${escapeHtml(groupName)}">
      <div class="panel-head">
        <div>
          <h3>${escapeHtml(groupName)}</h3>
          <p class="muted">默认全选；筛选、翻页和排序都在服务端执行。</p>
        </div>
        <div class="actions">
          <span class="muted">${selected.size}/${page.video_count} 已选</span>
          <button class="secondary" data-video-select="all-filtered" data-group="${escapeHtml(groupName)}" type="button">全选筛选</button>
          <button class="secondary" data-video-select="none-filtered" data-group="${escapeHtml(groupName)}" type="button">清空筛选</button>
          <button class="secondary" data-video-select="invert-filtered" data-group="${escapeHtml(groupName)}" type="button">反选筛选</button>
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
        { label: "", render: (row) => `<input type="checkbox" data-video-name="${escapeHtml(row.name)}" data-group="${escapeHtml(groupName)}" ${selected.has(row.name) ? "checked" : ""}>` },
        { label: "视频", render: (row) => escapeHtml(row.name) },
        { label: "帧数", render: (row) => escapeHtml(row.frame_count) },
        { label: "Triplets", render: (row) => escapeHtml(row.valid_triplets ?? 0) },
        { label: "FPS", render: (row) => formatNumber(row.fps) },
        { label: "分辨率", render: (row) => `${escapeHtml(row.width)}x${escapeHtml(row.height)}` },
        { label: "缓存", render: (row) => escapeHtml(row.cache_status || "-") },
      ])}</div>
      ${videoSelectionPager(page, groupName)}
    </section>
  `;
}

function renderVideoSelection() {
  const groups = selectedGroupNames();
  if (!groups.length) {
    $("video-selection").innerHTML = "<p class=\"muted\">先选择一个或多个视频集。</p>";
    return;
  }
  const totalSelected = groups.reduce((sum, name) => sum + selectedVideoNamesForGroup(name).length, 0);
  const multi = groups.length > 1;
  $("video-selection").innerHTML = `
    <div class="panel-head">
      <div>
        <h2>视频选择</h2>
        <p class="muted">${multi
          ? `已合并 ${groups.length} 个视频集，视频以“视频集/文件名”限定，共 ${totalSelected} 个已选。`
          : `默认全选；筛选、翻页和排序都在服务端执行，共 ${totalSelected} 个已选。`}</p>
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
  const sourceAssets = selectedSourceAssets();
  const selectedVideos = selectedVideoNames();
  return {
    run_type: "model_inference",
    model_file: data.model_file,
    checkpoint: data.checkpoint || "none",
    // Single-group runs keep the legacy `video_group` field (byte-identical
    // caches/reference keys); multi-group runs send `video_groups`.
    video_group: multi ? undefined : (groups[0] || ""),
    video_groups: multi ? groups : undefined,
    source_assets: sourceAssets.length === selectedVideos.length ? sourceAssets : undefined,
    selected_videos: selectedVideos,
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
    artifact_profile: data.artifact_profile || "evaluation",
    prefetch_workers: data.prefetch_workers ? Number(data.prefetch_workers) : null,
    save_workers: data.save_workers ? Number(data.save_workers) : null,
    max_save_inflight: data.max_save_inflight ? Number(data.max_save_inflight) : null,
    frame_step: Number(data.frame_step || 1),
    max_frames: data.max_frames ? Number(data.max_frames) : null,
    metrics: selectedMetrics(),
  };
}

function comparePayloadFromForm() {
  const data = formData($("compare-form"));
  const gt = selectedCompareGt();
  const predRows = selectedComparePredRows();
  if (!gt || predRows.length < 1) {
    return null;
  }
  // With a GT selected, it is the reference and every pred is a distorted
  // track. Without a GT (preds-only comparison), promote the first pred to the
  // reference role — the backend accepts a run_artifact as reference.
  let reference;
  let distortedRows;
  if (gt) {
    // GT is always a source clip now; the backend reconstructs each pred's
    // aligned GT from the clip using the pred's recorded source_frame_indices.
    reference = { kind: "media_asset", asset_id: Number(gt.asset_id || gt.id) };
    distortedRows = predRows;
  } else {
    const head = predRows[0];
    reference = {
      kind: "media_asset",
      asset_id: Number(head.asset_id || head.artifact_id),
      label: head.track_label || compareTrackLabel(head),
    };
    distortedRows = predRows.slice(1);
  }
  return {
    run_type: "video_compare",
    reference,
    distorted: distortedRows.map((row) => ({
      kind: "media_asset",
      asset_id: Number(row.asset_id || row.artifact_id),
      label: row.track_label || compareTrackLabel(row),
    })),
    extra_layers: state.selectedCompareLayerKinds.size
      ? distortedRows.filter((row) => row.run_id).map((row) => ({
          source: "run_artifact",
          run_id: Number(row.run_id),
          kinds: Array.from(state.selectedCompareLayerKinds),
        }))
      : [],
    align_mode: data.align_mode || "strict",
    metrics: compareSelectedMetrics(),
  };
}

function compareSelectedMetrics() {
  return Array.from(document.querySelectorAll("#compare-metrics-options input[name='compare_metrics']:checked")).map((item) => item.value);
}

function schedulePreflight(delay = 600) {
  clearTimeout(state.preflightTimer);
  state.preflightTimer = setTimeout(() => runPreflight().catch(renderPreflightError), delay);
}

async function runPreflight(options = {}) {
  const payload = payloadFromForm();
  if (!payload) {
    state.preflight = null;
    state.preflightPayloadKey = "";
    renderPreflight();
    return;
  }
  const groups = selectedGroupNames();
  if (!payload.model_file || !groups.length) {
    state.preflight = null;
    state.preflightPayloadKey = "";
    renderPreflight();
    return;
  }
  // Every selected group must have its video list loaded before preflight so the
  // selection is real, not just the summary count.
  if (groups.some((name) => !state.videoPages[name])) {
    state.preflight = null;
    state.preflightPayloadKey = "";
    renderPreflight();
    return;
  }
  const payloadKey = stableStringify(payload);
  if (!options.force && state.preflight && state.preflightPayloadKey === payloadKey) {
    renderPreflight();
    return;
  }
  if (state.preflightAbortController) {
    state.preflightAbortController.abort();
  }
  const controller = new AbortController();
  state.preflightAbortController = controller;
  try {
    const result = await api("/api/preflight", {
      method: "POST",
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (state.preflightAbortController !== controller) return;
    state.preflight = result;
    state.preflightPayloadKey = payloadKey;
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

function renderPreflight() {
  const result = state.preflight;
  const start = $("start-run");
  if (!state.modelFiles.length || !state.videoGroups.length) {
    start.disabled = true;
    $("preflight").innerHTML = `
      <h2>准备输入</h2>
      <p class="muted">需要在项目根目录放置 <code>models/*.py</code> 和 <code>videos/*/</code>。</p>
    `;
    return;
  }
  if (!result) {
    start.disabled = true;
    const groups = selectedGroupNames();
    if (groups.length && groups.every((name) => !state.videoPages[name])) {
      $("preflight").innerHTML = "<p class=\"muted\">加载视频列表并确认选择后会自动预检查。</p>";
      return;
    }
    $("preflight").innerHTML = "<p class=\"muted\">选择模型和视频后会自动预检查。</p>";
    return;
  }
  start.disabled = !result.ok;

  const videos = result.video_group?.videos || [];
  $("preflight").innerHTML = `
    <div class="panel-head">
      <h2>运行前预检查</h2>
      ${result.ok ? "<span class=\"ok-text\">通过</span>" : "<span class=\"bad-text\">未通过</span>"}
    </div>
    <div class="summary-grid">
      <div><span>模型</span><strong>${escapeHtml(result.model?.name || "-")}</strong></div>
      <div><span>接口检查</span><strong>${result.model?.interface_ok ? "通过" : "失败"}</strong></div>
      <div><span>已选视频</span><strong>${escapeHtml(result.video_group?.video_count ?? 0)}</strong></div>
      <div><span>真实总帧数</span><strong>${escapeHtml(result.video_group?.frame_count ?? 0)}</strong></div>
      <div><span>总时长</span><strong>${formatDuration(result.video_group?.duration_seconds)}</strong></div>
      <div><span>Triplets</span><strong>${escapeHtml(result.video_group?.triplets ?? 0)}</strong></div>
      <div><span>缓存状态</span><strong>${escapeHtml(result.cache?.status || "-")}</strong></div>
      <div><span>设备/精度</span><strong>${escapeHtml(`${result.device?.effective_device || "-"} / ${result.device?.effective_precision || "-"}`)}</strong></div>
      <div><span>支持精度</span><strong>${escapeHtml((result.device?.supported_precisions || []).join(" / ") || "-")}</strong></div>
    </div>
    ${renderMessages("errors", result.errors || [])}
    ${renderMessages("warnings", result.warnings || [])}
    ${renderExecutionProfileRecommendation(result)}
    ${renderPortableMetricHealthTable(result.metrics?.health || {})}
    ${renderDecodeBackendNotice()}
    <h3>本次推理视频</h3>
    <div class="table">${table(videos, [
      { label: "视频", render: (row) => escapeHtml(row.name) },
      { label: "真实帧数", render: (row) => `${escapeHtml(row.frame_count)} <span class="muted">${escapeHtml(row.frame_count_source || "")}</span>` },
      { label: "时长", render: (row) => formatDuration(row.duration_seconds) },
      { label: "Triplets", render: (row) => escapeHtml(row.triplets ?? 0) },
      { label: "分辨率", render: (row) => `${escapeHtml(row.width)}x${escapeHtml(row.height)}` },
      { label: "FPS", render: (row) => formatNumber(row.fps) },
      { label: "缓存", render: (row) => escapeHtml(row.cache_status || "-") },
    ])}</div>
  `;
}

function renderCompareGtCards(gtRows) {
  if (!gtRows.length) return "<p class=\"muted\">videos/ 下暂无可用 GT。</p>";
  // GT is always a source clip now. The backend reconstructs the pred-aligned
  // GT subset from this clip using each pred's recorded source_frame_indices,
  // so one card per clip is enough — no per-run GT duplicates.
  return gtRows.map((row) => {
    const key = compareGtKey(row);
    const active = state.selectedCompareGtKey === key;
    const title = `${escapeHtml(row.group)}/${escapeHtml(row.video)}`;
    return `
      <label class="source-card${active ? " selected" : ""}">
        <input type="radio" name="compare_gt_pick" data-compare-gt="${escapeHtml(key)}" ${active ? "checked" : ""}>
        <span class="source-card-body">
          <span class="source-card-title">${title} <span class="compat-badge">${escapeHtml(row.source_kind || "asset")}</span></span>
          <span class="source-card-meta">
            <span>${escapeHtml(row.frame_count || 0)} 帧</span>
            <span>${escapeHtml(row.width || "-")}×${escapeHtml(row.height || "-")}</span>
            <span>${formatNumber(row.fps)} fps</span>
          </span>
        </span>
      </label>
    `;
  }).join("");
}

function renderComparePredCards(predRows, selectedPreds) {
  if (!predRows.length) return "<p class=\"muted\">暂无已完成 Run 的 pred 产物。</p>";
  const gt = selectedCompareGt();
  return predRows.map((row) => {
    const artifactId = Number(row.artifact_id);
    const active = selectedPreds.has(artifactId);
    const compat = compareCompatibility(gt, row);
    const badge = compat
      ? (compat.ok
          ? "<span class=\"compat-badge compat-ok\">对齐</span>"
          : `<span class="compat-badge compat-warn" title="${escapeHtml(compat.reasons.join("；"))}">strict 不匹配</span>`)
      : "";
    const warning = compat && !compat.ok
      ? `<span class="source-card-warn">${escapeHtml(compat.reasons.join("；"))}</span>`
      : "";
    // Keep the row selectable so preflight can return a precise per-track
    // strict-alignment error; the server never normalizes external inputs.
    return `
      <div class="source-card${active ? " selected" : ""}${compat && !compat.ok ? " incompatible" : ""}">
        <label class="source-card-pick">
          <input type="checkbox" data-compare-pred="${escapeHtml(row.artifact_id)}" ${active ? "checked" : ""}>
          <span class="source-card-body">
            <span class="source-card-title">${row.run_id ? `#${escapeHtml(row.run_id)} ` : ""}${escapeHtml(row.run_name || row.display_name || "External Pred")} ${badge}</span>
            <span class="source-card-meta">
              <span>${escapeHtml(row.video || "-")}</span>
              <span>${escapeHtml(row.frame_count || 0)} 帧</span>
              <span>${escapeHtml(row.width || "-")}×${escapeHtml(row.height || "-")}</span>
            </span>
            ${warning}
          </span>
        </label>
        <label class="source-card-track">
          <span>Track</span>
          <input data-compare-track-label="${escapeHtml(row.artifact_id)}" value="${escapeHtml(compareTrackLabel(row))}">
        </label>
      </div>
    `;
  }).join("");
}

function renderCompareSelection() {
  ensureCompareSelection();
  if (!state.compareSourcesLoaded) {
    $("compare-selection").innerHTML = `
      <div class="panel-head">
        <div>
          <h2>对比来源</h2>
          <p class="muted">选择一个 GT 和至少一个 Pred，或至少两个 Pred（可留空 GT）。</p>
        </div>
        <button class="secondary" data-refresh-compare-sources type="button">加载来源</button>
      </div>
      <div class="timeline-skeleton" aria-busy="true"><span></span><span></span><span></span></div>
    `;
    return;
  }
  const gtRows = state.compareSources.gt || [];
  const predRows = state.compareSources.pred || [];
  const selectedPreds = state.selectedComparePredArtifacts;
  const selectedLayerCount = state.selectedCompareLayerKinds.size;
  const estimatedLayers = selectedPreds.size * selectedLayerCount;
  const trackCount = selectedPreds.size + (state.selectedCompareGtKey ? 1 : 0);
  $("compare-selection").innerHTML = `
    <div class="panel-head">
      <div>
        <h2>对比来源</h2>
        <p class="muted">GT 来自 videos/，Pred 来自已完成 Run 的 pred_video 产物。GT 可留空做纯预测对比。</p>
      </div>
      <div class="metric-summary">
        <span>GT ${escapeHtml(state.compareSourcesMeta.gt?.filtered_count ?? gtRows.length)}</span>
        <span>Pred ${escapeHtml(state.compareSourcesMeta.pred?.filtered_count ?? predRows.length)}</span>
        <span>已选 ${escapeHtml(trackCount)} 条轨道</span>
      </div>
    </div>
    <div class="compare-source-grid">
      <section class="compare-source-col">
        <div class="compare-col-head">
          <h3>GT（可选）</h3>
          <button class="secondary" data-compare-gt-clear type="button">清除 GT</button>
        </div>
        <div class="source-tools">
          <label>
            <span>搜索</span>
            <input data-compare-query="gt" value="${escapeHtml(state.compareGtQuery)}" placeholder="group 或文件名">
          </label>
        </div>
        <div class="source-card-list">${renderCompareGtCards(gtRows)}</div>
        ${compareSourcePager(state.compareSourcesMeta.gt, "gt")}
      </section>
      <section class="compare-source-col">
        <div class="compare-col-head">
          <h3>Pred tracks</h3>
        </div>
        <div class="source-tools">
          <label>
            <span>搜索</span>
            <input data-compare-query="pred" value="${escapeHtml(state.comparePredQuery)}" placeholder="Run、视频或 track">
          </label>
        </div>
        <div class="source-card-list">${renderComparePredCards(predRows, selectedPreds)}</div>
        ${compareSourcePager(state.compareSourcesMeta.pred, "pred")}
      </section>
    </div>
    <details class="compare-layer-picker">
      <summary>Extra layers (${escapeHtml(estimatedLayers)} previews)</summary>
      <div class="checkbox-row">
        ${["flowt_0", "flowt_1", "mask0", "mask1", "warp0", "warp1", "blend"].map((kind) => `
          <label class="check-item">
            <input type="checkbox" data-compare-layer-kind="${escapeHtml(kind)}" ${state.selectedCompareLayerKinds.has(kind) ? "checked" : ""}>
            <span>${escapeHtml(kind)}</span>
          </label>
        `).join("")}
      </div>
    </details>
  `;
}

function renderCompareMetricOptions() {
  const selected = new Set(compareSelectedMetrics());
  $("compare-metrics-options").innerHTML = METRICS.map((name) => `
    <label class="check-item">
      <input type="checkbox" name="compare_metrics" value="${escapeHtml(name)}" ${selected.has(name) ? "checked" : ""}>
      <span>${escapeHtml(name)} ${metricHealthBadge(name)}</span>
    </label>
  `).join("") + "<p class=\"muted metric-hint\">不可用指标会明确显示原因，不会被替换成其它分数。</p>";
}

function scheduleComparePreflight(delay = 600) {
  clearTimeout(state.comparePreflightTimer);
  state.comparePreflightTimer = setTimeout(() => runComparePreflight().catch(renderComparePreflightError), delay);
}

async function runComparePreflight(options = {}) {
  const payload = comparePayloadFromForm();
  if (!payload) {
    state.comparePreflight = null;
    state.comparePreflightPayloadKey = "";
    renderComparePreflight();
    return;
  }
  const payloadKey = stableStringify(payload);
  if (!options.force && state.comparePreflight && state.comparePreflightPayloadKey === payloadKey) {
    renderComparePreflight();
    return;
  }
  if (state.comparePreflightAbortController) {
    state.comparePreflightAbortController.abort();
  }
  const controller = new AbortController();
  state.comparePreflightAbortController = controller;
  try {
    const result = await api("/api/preflight", {
      method: "POST",
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (state.comparePreflightAbortController !== controller) return;
    state.comparePreflight = result;
    state.comparePreflightPayloadKey = payloadKey;
    renderComparePreflight();
  } catch (error) {
    if (error.name === "AbortError") return;
    throw error;
  } finally {
    if (state.comparePreflightAbortController === controller) {
      state.comparePreflightAbortController = null;
    }
  }
}

function renderComparePreflightError(error) {
  state.comparePreflight = { ok: false, errors: [{ title: "预检查请求失败", message: error.message }], warnings: [] };
  renderComparePreflight();
}

function renderComparePreflight() {
  const result = state.comparePreflight;
  const start = $("start-compare");
  if (!result) {
    start.disabled = true;
    $("compare-preflight").innerHTML = state.compareSourcesLoaded
      ? "<p class=\"muted\">选好 GT + Pred，或至少两个 Pred 后会自动预检查。</p>"
      : "<p class=\"muted\">先加载对比来源。</p>";
    return;
  }
  start.disabled = !result.ok;
  const tracks = result.distorted_tracks || (result.distorted ? [result.distorted] : []);
  const trackLabels = tracks.map((track) => track.track_label || track.label || track.name).filter(Boolean).join(", ");
  $("compare-preflight").innerHTML = `
    <div class="panel-head">
      <h2>运行前预检查</h2>
      ${result.ok ? "<span class=\"ok-text\">通过</span>" : "<span class=\"bad-text\">未通过</span>"}
    </div>
    <div class="summary-grid">
      <div><span>模式</span><strong>video_compare</strong></div>
      <div><span>参考</span><strong>${escapeHtml(result.reference?.name || "-")}</strong></div>
      <div><span>对比轨道</span><strong>${escapeHtml(`${tracks.length || 0} track(s)`)}</strong></div>
      <div><span>Tracks</span><strong>${escapeHtml(trackLabels || "-")}</strong></div>
      <div><span>帧数</span><strong>${escapeHtml(result.alignment?.frame_count ?? "-")}</strong></div>
      <div><span>分辨率</span><strong>${escapeHtml(`${result.alignment?.width || "-"}x${result.alignment?.height || "-"}`)}</strong></div>
      <div><span>FPS</span><strong>${formatNumber(result.alignment?.fps)}</strong></div>
    </div>
    ${renderMessages("errors", result.errors || [])}
    ${renderMessages("warnings", result.warnings || [])}
    ${renderPortableMetricHealthTable(result.metrics?.health || {})}
  `;
}

async function startCompareRun(event) {
  event.preventDefault();
  const payload = comparePayloadFromForm();
  if (!payload) {
    toast("请至少选择两条对比轨道（GT+Pred 或两个 Pred）");
    return;
  }
  await runComparePreflight({ force: true });
  if (!state.comparePreflight?.ok) {
    toast("预检查未通过");
    return;
  }
  const created = await api("/api/runs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  toast(`Run #${created.run_id} 已开始`);
  switchView("runs");
  await refreshRunsOnly();
  await selectRun(created.run_id);
}

async function createAdhocEvaluation() {
  const gt = selectedCompareGt();
  const preds = selectedComparePredRows();
  if (!gt || preds.length !== 2) {
    throw new Error("自由盲评需要选择一份 GT 和恰好两份 Pred");
  }
  const created = await api("/api/evaluation-tasks/adhoc", {
    method: "POST",
    body: JSON.stringify({
      reference_asset_id: Number(gt.asset_id || gt.id),
      candidate_asset_ids: preds.map((row) => Number(row.asset_id || row.id)),
      video_name: gt.video_name || gt.video || gt.display_name || "adhoc",
      name: `Adhoc · ${gt.display_name || gt.video || "Compare"}`,
      target_votes: 1,
    }),
  });
  state.selectedCampaignId = Number(created.campaign.id);
  switchView("evaluations");
  await loadEvaluations();
  toast("自由盲评任务已生成");
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

async function startRun(event) {
  event.preventDefault();
  const groups = selectedGroupNames();
  if (!groups.length) {
    toast("请先选择至少一个视频集");
    return;
  }
  if (!selectedVideoNames().length) {
    toast("请先选择至少一个视频");
    return;
  }
  const payload = payloadFromForm();
  await runPreflight({ force: true });
  if (!state.preflight?.ok) {
    toast("预检查未通过");
    return;
  }
  const created = await api("/api/runs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  toast(`Run #${created.run_id} 已开始`);
  switchView("runs");
  await refreshRunsOnly();
  await selectRun(created.run_id);
}

function statusBadge(status) {
  return `<span class="status ${escapeHtml(status)}">${escapeHtml(STATUS_LABELS[status] || status)}</span>`;
}

function shouldRefreshSelectedRun(nextRun) {
  if (!state.selectedRun || !nextRun) return false;
  if (!TERMINAL_STATUSES.has(nextRun.status)) return true;
  return Number(nextRun.updated_at || 0) !== Number(state.selectedRun.updated_at || 0)
    || String(nextRun.status || "") !== String(state.selectedRun.status || "")
    || Number(nextRun.progress_current || 0) !== Number(state.selectedRun.progress_current || 0)
    || Number(nextRun.progress_total || 0) !== Number(state.selectedRun.progress_total || 0);
}

async function refreshRunsOnly() {
  state.runs = await api("/api/runs");
  renderRuns();
  if (!isRunsViewActive()) {
    return;
  }
  if (state.selectedRun) {
    const nextRun = state.runs.find((item) => Number(item.id) === Number(state.selectedRun.id));
    if (!nextRun) {
      state.selectedRun = null;
      renderEmptyRunDetail();
    } else if (shouldRefreshSelectedRun(nextRun)) {
      await selectRun(state.selectedRun.id, { quiet: true });
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

function pathBasename(value) {
  const text = String(value || "");
  return text.split(/[\\/]/).filter(Boolean).pop() || text || "-";
}

function renderRuns() {
  // Drop selections for runs that are no longer in the list (e.g. refreshed away).
  const liveIds = new Set(state.runs.map((run) => Number(run.id)));
  for (const id of Array.from(state.selectedRunIds)) {
    if (!liveIds.has(Number(id))) state.selectedRunIds.delete(id);
  }
  const allSelected = state.runs.length > 0 && state.runs.every((run) => state.selectedRunIds.has(Number(run.id)));
  const selectedCount = state.selectedRunIds.size;
  const toolbar = `
    <div class="runs-toolbar">
      <label class="runs-select-all">
        <input type="checkbox" data-runs-select-all ${allSelected ? "checked" : ""}>
        <span>全选</span>
      </label>
      <button class="secondary danger" data-runs-batch-delete type="button" ${selectedCount ? "" : "disabled"}>批量删除${selectedCount ? ` (${selectedCount})` : ""}</button>
    </div>
  `;
  const rows = table(state.runs, [
    {
      label: "",
      render: (run) => `<input type="checkbox" data-run-select="${run.id}" ${state.selectedRunIds.has(Number(run.id)) ? "checked" : ""}>`,
    },
    { label: "Run", render: (run) => `#${escapeHtml(run.id)}` },
    { label: "名称", render: (run) => escapeHtml(run.name || "-") },
    { label: "状态", render: (run) => statusBadge(run.status) },
    { label: "类型", render: (run) => escapeHtml(run.metadata?.run_type || "model_inference") },
    { label: "来源", render: (run) => escapeHtml(runSourceLabel(run)) },
    { label: "进度", render: (run) => `${escapeHtml(run.progress_current || 0)}/${escapeHtml(run.progress_total || 0)}` },
    { label: "操作", render: (run) => `<button class="view-detail-btn" data-run-id="${run.id}" type="button">查看详情 →</button>` },
  ], { rowAttrs: (run) => `data-run-id="${run.id}" class="clickable-row"` });
  $("runs-table").innerHTML = toolbar + rows;
}

async function loadRunVideosPage(runId, page = 1) {
  state.runVideosPage = await api(`/api/runs/${runId}/videos?page=${page}&page_size=20`);
  state.runVideoPageByRun[runId] = Number(state.runVideosPage?.page || page || 1);
  const videos = state.runVideosPage?.videos || [];
  if (!videos.length) {
    delete state.selectedVideoByRun[runId];
    return;
  }
  const selectedVideoName = state.selectedVideoByRun[runId];
  const selectedIsVisible = videos.some((item) => item.video_name === selectedVideoName);
  const nextVideoName = selectedIsVisible ? selectedVideoName : videos[0].video_name;
  state.selectedVideoByRun[runId] = nextVideoName;
  if (nextVideoName && !state.runVideoTimelines[`${runId}:${nextVideoName}`]) {
    await loadRunVideoTimeline(runId, nextVideoName);
  }
}

async function selectRun(runId, options = {}) {
  const previousRunId = state.selectedRun?.id;
  state.selectedRun = await api(`/api/runs/${runId}`);
  if (previousRunId !== state.selectedRun.id) {
    state.sampleDetails = {};
    state.sampleDetailLoading = {};
    state.runVideoTimelines = {};
  }
  state.metricSummary = await api(`/api/runs/${runId}/metric-summary`);
  await loadRunVideosPage(runId, state.runVideoPageByRun[runId] || 1);
  renderRunDetail();
  if (!options.quiet) switchView("runs");
}

function renderInferencePhase(run) {
  const runType = run.metadata?.run_type || "model_inference";
  if (run.status === "decoding") return "Decoding video frames";
  if (run.status === "queued") return runType === "video_compare" ? "等待生成对比产物" : "等待推理";
  if (run.status === "running") return runType === "video_compare" ? "生成对比产物中" : "推理中";
  if (run.status === "failed" && run.error?.phase === "decode") return "Decode failed";
  if (run.status === "failed") return "失败";
  if (run.status === "canceled" || run.status === "cancel_requested") return "已取消";
  return run.inference_job_id ? "已完成" : "未开始";
}

function renderMetricPhase(run) {
  const metrics = run.metrics || [];
  if (!metrics.length) return "未选择指标";
  if (run.status === "decoding") return "Waiting for decode";
  if (run.status === "metric_queued") return "评测排队";
  if (run.status === "metric_running") return "评测中";
  if (run.metric_job_id) return "已完成";
  return "等待前一阶段完成";
}

function renderRunError(run) {
  if (!run.error || !Object.keys(run.error).length) return "";
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
    return `<div class="message error run-error-banner"><p><strong>${escapeHtml(run.error.type || "Error")}</strong>: ${escapeHtml(run.error.message || JSON.stringify(run.error))}</p>${meta}<button class="secondary" data-retry-run="${escapeHtml(run.id)}" type="button">Retry this Run</button></div>`;
  }
  return `<div class="message error run-error-banner"><p><strong>${escapeHtml(run.error.type || "Error")}</strong>: ${escapeHtml(run.error.message || JSON.stringify(run.error))}</p><button class="secondary" data-retry-run="${escapeHtml(run.id)}" type="button">Retry this Run</button></div>`;
  return `<div class="message error"><p><strong>${escapeHtml(run.error.type || "错误")}</strong>: ${escapeHtml(run.error.message || JSON.stringify(run.error))}</p></div>`;
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
  return "-";
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

async function sha256File(file, onProgress = () => {}) {
  const hasher = new Sha256Hasher();
  const chunkSize = 8 * 1024 * 1024;
  for (let offset = 0; offset < file.size; offset += chunkSize) {
    const bytes = new Uint8Array(await file.slice(offset, Math.min(file.size, offset + chunkSize)).arrayBuffer());
    hasher.update(bytes);
    onProgress(Math.min(file.size, offset + bytes.length), file.size);
  }
  return hasher.hex();
}

async function sha256Blob(blob) {
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return Array.from(new Uint8Array(digest)).map((value) => value.toString(16).padStart(2, "0")).join("");
}

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
  return `<div class="message"><p><strong>Benchmark 建议</strong>：batch ${escapeHtml(settings.batch_size ?? "-")}，prefetch ${escapeHtml(settings.prefetch_workers ?? "-")}，save ${escapeHtml(settings.save_workers ?? "-")}；历史稳态 ${formatNumber(performance.steady_state_fps)} FPS。用户设置仍会优先。</p></div>`;
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
  if (state.timelineAbortController) state.timelineAbortController.abort();
  const controller = new AbortController();
  state.timelineAbortController = controller;
  try {
    const payload = await api(
      `/api/runs/${runId}/videos/${encodeURIComponent(videoName)}/timeline?bucket_count=160&window_start=${windowStart}&window_size=${TIMELINE_WINDOW_SIZE}${metric ? `&metric=${encodeURIComponent(metric)}` : ""}`,
      { signal: controller.signal },
    );
    state.runVideoTimelines[windowKey] = payload;
    state.timelineWindowStartByVideo[windowKey] = Number(payload.window_start || 0);
    return payload;
  } catch (error) {
    if (error.name === "AbortError") return state.runVideoTimelines[`${runId}:${videoName}`] || null;
    throw error;
  } finally {
    if (state.timelineAbortController === controller) state.timelineAbortController = null;
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
        ${statusBadge(run.status)}
        <button class="secondary" data-rename-run="${run.id}" type="button">重命名</button>
        <button class="secondary" data-cancel-run="${run.id}" ${TERMINAL_STATUSES.has(run.status) ? "disabled" : ""} type="button">取消</button>
        <button class="secondary" data-retry-run="${run.id}" type="button">重试</button>
        <button class="secondary danger" data-delete-run="${run.id}" type="button">删除记录</button>
        <button class="secondary" data-cleanup-run="${run.id}" ${TERMINAL_STATUSES.has(run.status) && !run.artifact_cleaned_at ? "" : "disabled"} type="button">清理产物</button>
      </div>
    </div>
    <details class="run-meta" ${state.runMetaCollapsed ? "" : "open"}>
      <summary>记录详情</summary>
      <div class="summary-grid">
        <div><span>Run 类型</span><strong>${escapeHtml(run.metadata?.run_type || "model_inference")}</strong></div>
        <div><span>进度</span><strong>${escapeHtml(run.progress_current || 0)}/${escapeHtml(run.progress_total || 0)}</strong></div>
        <div><span>推理阶段</span><strong>${escapeHtml(renderInferencePhase(run))}</strong></div>
        <div><span>评测阶段</span><strong>${escapeHtml(renderMetricPhase(run))}</strong></div>
        <div><span>输出目录</span><strong>${escapeHtml(run.metadata?.output_dir || run.result?.output_dir || "-")}</strong></div>
        <div><span>产物数</span><strong>${escapeHtml(run.artifact_summary?.total || 0)}</strong></div>
      </div>
      ${renderRunError(run)}
      ${renderCleanedArtifactsNotice(run)}
      ${renderModelLoadReport(run)}
      ${renderOutputHealthReport(run)}
      ${renderPerformanceReport(run)}
      ${renderDecodePanel(run)}
      <div class="message"><p><strong>Execution</strong>: ${runExecutionTarget(run)}</p></div>
      ${renderPortableMetricHealthTable(run.metadata?.metric_health || {})}
      ${renderRunJobs(run)}
    </details>
    <div class="run-workspace">
      <aside class="video-tabs">
        <h3>视频</h3>
        ${videos.length ? videos.map((item) => `
          <button class="video-tab ${item.video_name === selectedVideoName ? "active" : ""}" data-run-video="${escapeHtml(item.video_name)}" type="button">
            <strong>${escapeHtml(item.video_file || item.video_name)}</strong>
            <span>${escapeHtml(item.sample_count || 0)} samples</span>
          </button>
        `).join("") : "<p class=\"muted\">运行完成后这里会显示可查看的视频。</p>"}
        ${renderRunVideosPager()}
      </aside>
      <section class="sample-viewer">
        ${video ? renderVideoTimeline(video) : (selectedVideoName ? "<div class=\"timeline-skeleton\" aria-busy=\"true\"><span></span><span></span><span></span></div>" : "<p class=\"muted\">暂无结果。</p>")}
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
  const points = values
    .map((value, index) => {
      if (value === null) return null;
      const x = samples.length <= 1 ? 50 : 4 + (index / (samples.length - 1)) * 92;
      const normalized = max === min ? 0.5 : (value - min) / (max - min);
      const y = 42 - normalized * 30;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .filter(Boolean)
    .join(" ");
  const markerX = samples.length <= 1 ? 50 : 4 + (selectedIndex / (samples.length - 1)) * 92;
  const selectedValue = values[selectedIndex];
  const selectedY = selectedValue === null || !Number.isFinite(selectedValue)
    ? 46
    : 42 - (max === min ? 0.5 : (selectedValue - min) / (max - min)) * 30;
  // The SVG stretches its 0..100 / 0..56 coordinate space to the container with
  // preserveAspectRatio="none", so it can only carry shapes that tolerate
  // non-uniform scaling (grid lines, the polyline, the vertical marker). The
  // sample points are rendered as an absolutely-positioned HTML overlay instead:
  // CSS-sized dots stay perfectly round no matter the container aspect ratio.
  return `
    <div class="chart" data-chart-video="${escapeHtml(video.video_name)}">
      <div class="chart-head">
        <strong>${escapeHtml(metricName)}</strong>
        <span class="muted">点击曲线定位当前帧</span>
      </div>
      <div class="chart-plot">
        <svg class="metric-chart-svg" viewBox="0 0 100 56" preserveAspectRatio="none" role="img">
          <g class="chart-grid">
            <line x1="4" x2="96" y1="12" y2="12"></line>
            <line x1="4" x2="96" y1="27" y2="27"></line>
            <line x1="4" x2="96" y1="42" y2="42"></line>
          </g>
          <polyline class="metric-line" points="${points}" fill="none"></polyline>
          <line class="current-marker" x1="${markerX.toFixed(2)}" x2="${markerX.toFixed(2)}" y1="8" y2="46"></line>
        </svg>
        <div class="chart-points">
          <span class="selected-metric-point" style="left: ${markerX.toFixed(2)}%; top: ${((selectedY / 56) * 100).toFixed(2)}%"></span>
          ${renderMetricPoints(video.video_name, samples, metricName, min, max, selectedIndex)}
        </div>
      </div>
      <div class="chart-scale">
        <span>min ${formatNumber(min)}</span>
        <span>max ${formatNumber(max)}</span>
      </div>
    </div>
  `;
}

function renderMetricPoints(videoName, samples, metricName, min, max, selectedIndex) {
  return samples.map((sample, index) => {
    const metric = sample.metrics?.[metricName];
    const x = samples.length <= 1 ? 50 : 4 + (index / (samples.length - 1)) * 92;
    const status = metric?.status || "missing";
    const value = metric?.value;
    const normalized = status === "completed" && value !== null && max !== min ? (Number(value) - min) / (max - min) : 0.5;
    const y = status === "completed" ? 42 - normalized * 30 : 46;
    const top = (y / 56) * 100;
    const selectedClass = index === selectedIndex ? " selected" : "";
    return `<button class="metric-point ${escapeHtml(status)}${selectedClass}" data-chart-video="${escapeHtml(videoName)}" data-chart-sample="${index}" data-frame-index="${escapeHtml(sample.frame_index)}" style="left: ${x.toFixed(2)}%; top: ${top.toFixed(2)}%" type="button" title="${escapeHtml(status)} ${escapeHtml(value ?? metricReason(metric))}"></button>`;
  }).join("");
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

function renderVideoPlayer(label, artifactId) {
  if (!artifactId) return "";
  const url = `/api/files/${artifactId}`;
  return `
    <div class="video-artifact">
      <span>${escapeHtml(label)}</span>
      <video controls playsinline preload="metadata" src="${escapeHtml(url)}" onerror="this.outerHTML='<p class=\\'muted\\'>浏览器无法播放此视频格式。</p>'"></video>
    </div>
  `;
}

function renderVideoArtifacts(video) {
  const tracks = video.video_artifact_tracks || [];
  let items = "";
  if (tracks.length) {
    items = tracks
      .map((item) => renderVideoPlayer(`${item.track_label || "shared"} / ${item.kind}`, item.id))
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
      <button class="secondary" data-master-video-play="${escapeHtml(video.video_name)}" type="button">Play all</button>
      <button class="secondary" data-master-video-pause="${escapeHtml(video.video_name)}" type="button">Pause all</button>
      <button class="secondary" data-master-video-sync="${escapeHtml(video.video_name)}" type="button">Sync time</button>
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
      ${renderMetricChart(video, selectedIndex, metricName)}
      ${renderWorstSamples(video, metricName)}
    </div>
    <div class="sample-controls">
      <button class="secondary" data-sample-step="-1" type="button">上一帧</button>
      <input data-sample-range="${escapeHtml(video.video_name)}" type="range" min="0" max="${Math.max(0, samples.length - 1)}" value="${selectedIndex}">
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
  chart.innerHTML = `${renderMetricChart(video, selectedIndex, metricName)}${renderWorstSamples(video, metricName)}`;
  preview.innerHTML = samples[selectedIndex] ? renderSamplePreview(samples[selectedIndex], video) : "<p class=\"muted\">没有样本。</p>";
  const windowStart = Number(video.window_start || 0);
  const total = Number(video.sample_count || samples.length);
  if (counter) counter.textContent = `${windowStart + selectedIndex + 1}/${total || 0}`;
  // Only sync the slider's value when the change did not originate from the
  // slider itself; overwriting it mid-drag would fight the pointer.
  if (slider && Number(slider.value) !== selectedIndex) slider.value = String(selectedIndex);
  return true;
}

function sampleDetail(sampleId) {
  if (!state.selectedRun) return null;
  return state.sampleDetails[`${state.selectedRun.id}:${sampleId}`] || null;
}

async function loadSampleDetail(sampleId) {
  if (!state.selectedRun) return;
  const key = `${state.selectedRun.id}:${sampleId}`;
  if (state.sampleDetails[key] || state.sampleDetailLoading[key]) return;
  state.sampleDetailLoading[key] = true;
  // Each sample gets its own abort controller so sibling loads in a compare
  // frame group (GT + predA + predB share a frame) do not cancel each other.
  if (state.sampleAbortControllers[key]) state.sampleAbortControllers[key].abort();
  const controller = new AbortController();
  state.sampleAbortControllers[key] = controller;
  try {
    state.sampleDetails[key] = await api(`/api/runs/${state.selectedRun.id}/samples/${sampleId}`, { signal: controller.signal });
  } catch (error) {
    if (error.name === "AbortError") return;
    state.sampleDetails[key] = { sample_id: sampleId, artifacts: {}, extra_artifacts: [], load_error: error.message };
  } finally {
    if (state.sampleAbortControllers[key] === controller) delete state.sampleAbortControllers[key];
    delete state.sampleDetailLoading[key];
    // Prefer an in-place frame update so late-arriving sample detail does not
    // recreate the video players; fall back to a full render if unavailable.
    if (!updateFrameRegion()) renderRunDetail();
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
    const url = artifact.original_url || artifact.preview_url;
    const href = artifact.original_url || url;
    body = `<a href="${escapeHtml(href)}" target="_blank" rel="noreferrer"><img src="${escapeHtml(url)}" alt="${escapeHtml(label)}" loading="lazy"></a>`;
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
      const url = artifact.original_url || artifact.preview_url;
      const href = artifact.original_url || url;
      body = `<a href="${escapeHtml(href)}" target="_blank" rel="noreferrer"><img src="${escapeHtml(url)}" alt="${escapeHtml(label)}" loading="lazy"></a>`;
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
  // GT is shared across tracks; use the img0/gt sample file so it renders even
  // before any track's detail has loaded (compare gt == reference frame).
  const url = sample.sample_files?.gt || `/api/sample-files/${sample.sample_id}/gt`;
  return `
    <div class="big-slot compare-gt-slot">
      <div class="big-slot-head"><strong class="compare-track-title">GT</strong></div>
      <div class="big-slot-body"><a href="${escapeHtml(url)}" target="_blank" rel="noreferrer"><img src="${escapeHtml(url)}" alt="GT" loading="lazy"></a></div>
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
  const video = state.runVideoTimelines[`${state.selectedRun.id}:${videoName}`];
  if (!video) return;
  const max = Math.max(0, (video.samples || []).length - 1);
  state.selectedSampleByVideo[`${state.selectedRun.id}:${videoName}`] = Math.max(0, Math.min(max, index));
  // Update only the frame-dependent region so the video players are not
  // recreated (which would reload and stutter). Fall back to a full render if
  // the region is not on the page (e.g. video not yet rendered).
  if (!updateFrameRegion()) renderRunDetail();
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
  let video = state.runVideoTimelines[`${state.selectedRun.id}:${videoName}`];
  if (!video) {
    video = await loadRunVideoTimeline(state.selectedRun.id, videoName);
  }
  let index = (video.samples || []).findIndex((sample) => Number(sample.frame_index) === Number(frameIndex));
  if (index < 0) {
    const windowStart = Math.max(0, Number(frameIndex) - 150);
    video = await loadRunVideoTimeline(state.selectedRun.id, videoName, { windowStart });
    index = (video.samples || []).findIndex((sample) => Number(sample.frame_index) === Number(frameIndex));
  }
  state.selectedVideoByRun[state.selectedRun.id] = video.video_name;
  state.selectedSampleByVideo[`${state.selectedRun.id}:${video.video_name}`] = Math.max(0, index);
  renderRunDetail();
}

async function cancelRun(runId) {
  await api(`/api/runs/${runId}/cancel`, { method: "POST", body: "{}" });
  toast("已请求取消");
  await refreshRunsOnly();
}

async function retryRun(runId) {
  const created = await api(`/api/runs/${runId}/retry`, { method: "POST", body: "{}" });
  toast(`重试 Run #${created.run_id} 已开始`);
  await refreshRunsOnly();
  await selectRun(created.run_id);
}

async function deleteRun(runId) {
  const wasSelected = Number(state.selectedRun?.id) === Number(runId);
  await api(`/api/runs/${runId}`, { method: "DELETE" });
  toast(`Run #${runId} 已从列表隐藏`);
  if (wasSelected) {
    state.selectedRun = null;
  }
  await refreshRunsOnly();
  if (!state.selectedRun) {
    renderEmptyRunDetail();
  }
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
  if (!window.confirm(`确认删除选中的 ${ids.length} 条运行记录？`)) return;
  const result = await api(`/api/runs/batch-delete`, {
    method: "POST",
    body: JSON.stringify({ run_ids: ids }),
  });
  const deleted = new Set((result.deleted || []).map(Number));
  if (state.selectedRun && deleted.has(Number(state.selectedRun.id))) {
    state.selectedRun = null;
  }
  state.selectedRunIds.clear();
  toast(`已删除 ${result.deleted?.length || 0} 条记录`);
  await refreshRunsOnly();
  if (!state.selectedRun) renderEmptyRunDetail();
}

async function cleanupRunArtifacts(runId) {
  await api(`/api/runs/${runId}/cleanup-artifacts`, { method: "POST", body: "{}" });
  toast(`Run #${runId} 产物已清理`);
  await refreshRunsOnly();
  if (state.selectedRun?.id === runId) {
    await selectRun(runId, { quiet: true });
  }
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
    await selectRun(runId, { quiet: true });
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
    await selectRun(runId, { quiet: true });
  }
}

async function deleteRunFeedback(runId, feedbackId) {
  await api(`/api/runs/${runId}/feedback/${feedbackId}`, { method: "DELETE" });
  if (Number(state.editingFeedback) === Number(feedbackId)) state.editingFeedback = null;
  toast("反馈已删除");
  if (Number(state.selectedRun?.id) === Number(runId)) {
    await selectRun(runId, { quiet: true });
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
  host.innerHTML = `
    ${statsFilterControls(options, state.statsFilters)}
    <div class="summary-grid">
      <div><span>反馈总数</span><strong>${escapeHtml(stats.total || 0)}</strong></div>
      <div><span>打分数</span><strong>${escapeHtml(stats.rating_count || 0)}</strong></div>
      <div><span>平均分</span><strong>${stats.average_rating === null || stats.average_rating === undefined ? "-" : formatRating(stats.average_rating)}</strong></div>
      <div><span>问题数</span><strong>${escapeHtml(stats.issue_count || 0)}</strong></div>
    </div>
    ${Object.keys(filters).length ? `<p class="muted">已筛选：${Object.entries(filters).map(([k, v]) => `${escapeHtml(k)}=${escapeHtml(v)}`).join("， ")}</p>` : ""}
    <section class="stats-block">
      <h3>总体评分分布（0.25 分度）</h3>
      ${renderRatingHistogram(distribution)}
    </section>
    ${renderGroupedDistributions("按视频的评分分布", byVideo, (row) => row.video || "（未指定）")}
    ${renderGroupedDistributions("按模型 / 权重的评分分布", byCheckpoint, (row) => `${row.model_name || "?"} / ${row.checkpoint || "-"}`)}
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
  `;
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

function mediaAssetContent(asset) {
  if (asset.state !== "ready") return `<span class="muted">${escapeHtml(asset.state || "unavailable")}</span>`;
  if (asset.media_kind === "video") {
    return `<video controls playsinline preload="metadata" src="/api/media/assets/${Number(asset.id)}/content"></video>`;
  }
  return `<a href="/api/media/assets/${Number(asset.id)}/content" target="_blank" rel="noreferrer">查看首帧</a>`;
}

function renderMediaLibrary() {
  const collectionSelect = $("upload-form")?.elements.collection_id;
  if (collectionSelect) {
    const selected = collectionSelect.value;
    collectionSelect.innerHTML = state.mediaCollections
      .map((collection) => `<option value="${Number(collection.id)}">${escapeHtml(collection.name)} (${Number(collection.asset_count || 0)})</option>`)
      .join("");
    if (state.mediaCollections.some((row) => String(row.id) === String(selected))) collectionSelect.value = selected;
  }
  const host = $("media-content");
  if (!host) return;
  host.innerHTML = `<div class="table compact-table">${table(state.mediaAssets, [
    { label: "预览", render: (asset) => mediaAssetContent(asset) },
    { label: "别名", render: (asset) => `<strong>${escapeHtml(asset.display_name)}</strong><br><span class="muted">${escapeHtml(asset.original_name || "-")}</span>` },
    { label: "Collection", render: (asset) => escapeHtml(asset.collection_name || "-") },
    { label: "来源 / 角色", render: (asset) => `${escapeHtml(asset.source_kind)} / ${escapeHtml(asset.role)}` },
    { label: "媒体", render: (asset) => `${escapeHtml(asset.media_kind)}<br><span class="muted">${Number(asset.width || 0)}×${Number(asset.height || 0)} · ${Number(asset.frame_count || 0)} 帧 · ${formatNumber(asset.fps)} fps</span>` },
    { label: "大小", render: (asset) => formatBytes(asset.size_bytes) },
    { label: "状态", render: (asset) => statusBadge(asset.state || "ready") },
    { label: "操作", render: (asset) => asset.source_kind === "upload" ? `<button class="danger secondary" data-media-delete="${Number(asset.id)}" type="button">软删除</button>` : "-" },
  ])}</div>`;
}

async function loadMediaLibrary() {
  const [collectionsPayload, assetsPayload] = await Promise.all([
    api("/api/media/collections"),
    api("/api/media/assets?page_size=200&state="),
  ]);
  state.mediaCollections = collectionsPayload.collections || [];
  state.mediaAssets = assetsPayload.assets || [];
  renderMediaLibrary();
}

async function createMediaCollection(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const values = formData(form);
  await api("/api/media/collections", {
    method: "POST",
    body: JSON.stringify({ name: values.name }),
  });
  form.reset();
  await loadMediaLibrary();
  toast("Collection 已创建");
}

function uploadResumeKey(file, values) {
  return `vfieval-upload:${values.collection_id}:${values.role}:${values.media_kind}:${values.display_name}:${file.name}:${file.size}:${file.lastModified}`;
}

async function rawJsonRequest(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error?.message || data.message || response.statusText);
  return data;
}

async function uploadExternalMedia(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const values = formData(form);
  const file = form.elements.file.files?.[0];
  if (!file) throw new Error("请选择上传文件");
  if (!values.collection_id) throw new Error("请先创建 Collection");
  const progress = $("upload-progress");
  const setProgress = (text) => { if (progress) progress.textContent = text; };
  const resumeKey = uploadResumeKey(file, values);
  state.uploadPaused = false;
  setProgress("正在计算完整文件 SHA-256…");
  const fileSha256 = await sha256File(file, (done, total) => {
    setProgress(`正在校验文件：${Math.round((done / Math.max(1, total)) * 100)}%`);
  });

  let session = null;
  const savedUploadId = localStorage.getItem(resumeKey);
  if (savedUploadId) {
    try {
      session = await api(`/api/uploads/${savedUploadId}`);
      if (session.status !== "uploading" || Number(session.total_size) !== file.size || session.sha256 !== fileSha256) session = null;
    } catch (_error) {
      session = null;
    }
  }
  if (!session) {
    const created = await api("/api/uploads", {
      method: "POST",
      body: JSON.stringify({
        collection_id: Number(values.collection_id),
        role: values.role,
        media_kind: values.media_kind,
        display_name: values.display_name,
        original_name: file.name,
        total_size: file.size,
        sha256: fileSha256,
        fps: values.fps ? Number(values.fps) : null,
      }),
    });
    session = created.upload;
    localStorage.setItem(resumeKey, session.id);
  }
  state.activeUpload = session.id;
  const chunkSize = Number(session.chunk_size || 8 * 1024 * 1024);
  const uploaded = new Set((session.parts || []).map((part) => Number(part.part_index)));
  const partCount = Math.ceil(file.size / chunkSize);
  for (let index = 0; index < partCount; index += 1) {
    if (state.uploadPaused) {
      state.activeUpload = null;
      setProgress("上传已暂停；重新选择同一文件并提交即可从现有分片续传。");
      return;
    }
    if (uploaded.has(index)) continue;
    const start = index * chunkSize;
    const end = Math.min(file.size, start + chunkSize);
    const part = file.slice(start, end);
    setProgress(`上传分片 ${index + 1}/${partCount}…`);
    const partSha256 = await sha256Blob(part);
    await rawJsonRequest(`/api/uploads/${session.id}/parts/${index}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/octet-stream",
        "Content-Range": `bytes ${start}-${end - 1}/${file.size}`,
        "X-Chunk-SHA256": partSha256,
      },
      body: part,
    });
  }
  setProgress("正在服务端校验并建立媒体资产…");
  await api(`/api/uploads/${session.id}/complete`, { method: "POST", body: "{}" });
  localStorage.removeItem(resumeKey);
  state.activeUpload = null;
  form.reset();
  setProgress("上传完成");
  await loadMediaLibrary();
  toast("媒体资产已上传");
}

async function deleteMediaAsset(assetId) {
  if (!window.confirm("软删除该媒体资产？历史 Campaign、投票和统计不会被删除。")) return;
  await api(`/api/media/assets/${Number(assetId)}`, { method: "DELETE" });
  await loadMediaLibrary();
  await loadCompareSources({ gtPage: 1, predPage: 1 });
  toast("媒体资产已软删除");
}

function ensureEvaluatorId() {
  if (state.evaluatorId) return state.evaluatorId;
  state.evaluatorId = crypto.randomUUID
    ? crypto.randomUUID()
    : `browser-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  localStorage.setItem("vfieval-evaluator-id", state.evaluatorId);
  return state.evaluatorId;
}

function renderCampaignAnalysis() {
  const host = $("campaign-analysis");
  if (!host) return;
  const analysis = state.campaignAnalysis;
  if (!analysis) {
    host.innerHTML = "<p class=\"muted\">发布 Campaign 后显示覆盖率、胜率矩阵、Bradley–Terry 排名和客观指标统计。</p>";
    return;
  }
  const coverage = analysis.coverage || {};
  const ranking = analysis.human?.ranking || [];
  const headToHead = analysis.human?.head_to_head || [];
  const metrics = analysis.objective?.metrics || [];
  const correlations = analysis.cross_analysis?.metrics || [];
  const evaluatorVotes = analysis.evaluator_votes || [];
  const reasonRows = Object.entries(analysis.quality_reasons || {}).map(([reason, count]) => ({ reason, count }));
  const campaignId = Number(analysis.campaign?.id || state.selectedCampaignId);
  const analysisParams = new URLSearchParams();
  for (const [key, value] of Object.entries(state.campaignAnalysisFilters || {})) {
    if (value) analysisParams.set(key, value);
  }
  analysisParams.set("format", "csv");
  host.innerHTML = `
    <div class="panel-head"><div><h3>Campaign 分析</h3><p class="muted">人工与客观指标分别展示，不生成混合总分。</p></div><div class="actions"><a class="secondary button-link" href="/api/evaluation-campaigns/${campaignId}/analysis?${escapeHtml(analysisParams.toString())}">排名 CSV</a><a class="secondary button-link" href="/api/evaluation-campaigns/${campaignId}/export?format=csv">完整 CSV</a><a class="secondary button-link" href="/api/evaluation-campaigns/${campaignId}/export">完整 JSON</a></div></div>
    <section class="stats-filters">
      ${[["video", "视频"], ["model", "模型"], ["checkpoint", "权重"], ["evaluator_id", "评测员 UUID"]].map(([name, label]) => `<label><span>${label}</span><input data-campaign-filter="${name}" value="${escapeHtml(state.campaignAnalysisFilters[name] || "")}"></label>`).join("")}
      <label><span>Collection</span><select data-campaign-filter="collection_id"><option value="">全部</option>${state.mediaCollections.map((collection) => `<option value="${Number(collection.id)}" ${String(state.campaignAnalysisFilters.collection_id) === String(collection.id) ? "selected" : ""}>${escapeHtml(collection.name)}</option>`).join("")}</select></label>
      <button data-campaign-analysis-apply type="button">应用筛选</button>
      <button data-campaign-analysis-reset class="secondary" type="button">清除</button>
    </section>
    <div class="summary-grid">
      <div><span>覆盖任务</span><strong>${Number(coverage.completed_tasks || 0)}/${Number(coverage.tasks || 0)}</strong></div>
      <div><span>每对目标票数</span><strong>${Number(coverage.target_votes_per_task || 0)}</strong></div>
      <div><span>结论状态</span><strong>${coverage.provisional ? "provisional" : "complete"}</strong></div>
      <div><span>评测一致率</span><strong>${analysis.agreement_rate == null ? "-" : formatNumber(Number(analysis.agreement_rate) * 100) + "%"}</strong></div>
    </div>
    <section class="stats-block"><h3>Bradley–Terry 排名（平局半胜）</h3><div class="table compact-table">${table(ranking, [
      { label: "排名", render: (row) => String(ranking.indexOf(row) + 1) },
      { label: "候选", render: (row) => escapeHtml(row.label || `asset-${row.asset_id}`) },
      { label: "模型 / 权重", render: (row) => `${escapeHtml(row.model_name || "-")}<br><span class="muted">${escapeHtml(row.checkpoint || "-")}</span>` },
      { label: "得分", render: (row) => formatNumber(row.score) },
      { label: "95% CI", render: (row) => `${formatNumber(row.ci95?.[0])} – ${formatNumber(row.ci95?.[1])}` },
    ])}</div></section>
    <section class="stats-block"><h3>Head-to-head</h3><div class="table compact-table">${table(headToHead, [
      { label: "候选对", render: (row) => `${Number(row.asset_a_id)} vs ${Number(row.asset_b_id)}` },
      { label: "票数", render: (row) => Number(row.votes || 0) },
      { label: "A 半胜", render: (row) => formatNumber(row.wins_first) },
      { label: "B 半胜", render: (row) => formatNumber(row.wins_second) },
      { label: "A / B 胜率", render: (row) => `${formatNumber(Number(row.win_rate_a || 0) * 100)}% / ${formatNumber(Number(row.win_rate_b || 0) * 100)}%` },
      { label: "平局", render: (row) => Number(row.ties || 0) },
    ])}</div></section>
    <section class="stats-block"><h3>覆盖与质量原因</h3><div class="table compact-table">${table(reasonRows, [
      { label: "原因", render: (row) => escapeHtml(row.reason) },
      { label: "次数", render: (row) => Number(row.count || 0) },
    ])}</div><div class="table compact-table">${table(evaluatorVotes, [
      { label: "评测员", render: (row) => escapeHtml(row.evaluator_name || row.evaluator_id) },
      { label: "票数", render: (row) => Number(row.votes || 0) },
    ])}</div></section>
    <section class="stats-block"><h3>客观指标</h3><div class="table compact-table">${table(metrics, [
      { label: "指标", render: (row) => escapeHtml(row.metric_name) },
      { label: "方向", render: (row) => escapeHtml(row.direction || "-") },
      { label: "候选资产", render: (row) => Number(row.asset_id) },
      { label: "有效样本", render: (row) => Number(row.count || 0) },
      { label: "均值 / 中位数", render: (row) => `${formatNumber(row.mean)} / ${formatNumber(row.median)}` },
      { label: "P10 / P90", render: (row) => `${formatNumber(row.p10)} / ${formatNumber(row.p90)}` },
      { label: "状态", render: (row) => escapeHtml(JSON.stringify(row.status_counts || {})) },
    ])}</div></section>
    ${correlations.length ? `<section class="stats-block"><h3>人工排名与指标排名 Spearman</h3><div class="table compact-table">${table(correlations, [
      { label: "指标", render: (row) => escapeHtml(row.metric_name) },
      { label: "候选数", render: (row) => Number(row.candidate_count || 0) },
      { label: "Spearman", render: (row) => formatNumber(row.spearman) },
      { label: "冲突候选", render: (row) => escapeHtml((row.conflict_asset_ids || []).join(", ") || "-") },
    ])}</div></section>` : ""}
  `;
}

function renderEvaluationTask() {
  const host = $("evaluation-task");
  if (!host) return;
  const task = state.currentEvaluationTask;
  if (!state.evaluatorId || !state.evaluatorName) {
    host.innerHTML = "<p class=\"muted\">先填写评测员显示名，再参加已发布的 Campaign。</p>";
    return;
  }
  if (!task) {
    host.innerHTML = "<p class=\"muted\">当前 Campaign 暂无待评任务，或尚未发布。</p>";
    return;
  }
  const taskMedia = (side) => task[`${side}_media_kind`] === "frame_sequence"
    ? `<img data-evaluation-frame-side="${escapeHtml(side)}" src="${escapeHtml(`${task[`${side}_url`]}&frame=${state.evaluationFrameIndex}`)}" alt="${escapeHtml(side)}">`
    : `<video controls playsinline preload="metadata" src="${escapeHtml(task[`${side}_url`])}"></video>`;
  const hasFrameSequence = ["reference", "left", "right"].some((side) => task[`${side}_media_kind`] === "frame_sequence");
  host.innerHTML = `
    <div class="panel-head"><div><h3>匿名任务 · ${escapeHtml(task.video_name)}</h3><p class="muted">参考 GT 仅用于观察；选择更好的匿名候选，或选择平局。</p></div></div>
    <div class="evaluation-grid">
      <article><h4>参考 GT</h4>${taskMedia("reference")}</article>
      <article><h4>候选 A</h4>${taskMedia("left")}</article>
      <article><h4>候选 B</h4>${taskMedia("right")}</article>
    </div>
    ${hasFrameSequence ? `<label class="wide-field"><span>单帧位置 ${state.evaluationFrameIndex + 1}/${Number(task.frame_count || 1)}</span><input data-evaluation-frame type="range" min="0" max="${Math.max(0, Number(task.frame_count || 1) - 1)}" value="${state.evaluationFrameIndex}"></label>` : ""}
    <div class="checkbox-row evaluation-reasons">
      ${(task.quality_reasons || []).map((reason) => `<label><input type="checkbox" data-evaluation-reason value="${escapeHtml(reason)}">${escapeHtml(reason)}</label>`).join("")}
    </div>
    <div class="field-grid">
      <label><span>置信度</span><select id="evaluation-confidence"><option value="">未填写</option><option value="low">低</option><option value="medium">中</option><option value="high">高</option></select></label>
      <label><span>备注</span><textarea id="evaluation-note" maxlength="4000"></textarea></label>
    </div>
    <div class="evaluation-actions">
      <button data-evaluation-vote="left" type="button">候选 A 更好</button>
      <button data-evaluation-vote="tie" class="secondary" type="button">平局</button>
      <button data-evaluation-vote="right" type="button">候选 B 更好</button>
    </div>
  `;
  state.evaluationTaskStartedAt = Date.now();
}

function fillCampaignControls() {
  const candidateForm = $("candidate-form");
  if (!candidateForm) return;
  const campaignSelect = candidateForm.elements.campaign_id;
  campaignSelect.innerHTML = state.evaluationCampaigns
    .filter((campaign) => campaign.status === "draft")
    .map((campaign) => `<option value="${Number(campaign.id)}">${escapeHtml(campaign.name)}</option>`)
    .join("");
  if (state.selectedCampaignId && state.evaluationCampaigns.some((row) => Number(row.id) === Number(state.selectedCampaignId) && row.status === "draft")) {
    campaignSelect.value = String(state.selectedCampaignId);
  }
  const gt = state.mediaAssets.filter((asset) => asset.state === "ready" && ["source", "gt"].includes(asset.role));
  const pred = state.mediaAssets.filter((asset) => asset.state === "ready" && asset.role === "pred");
  candidateForm.elements.reference_asset_id.innerHTML = gt.map((asset) => `<option value="${Number(asset.id)}">${escapeHtml(asset.collection_name || "-")} · ${escapeHtml(asset.display_name)}</option>`).join("");
  candidateForm.elements.asset_id.innerHTML = pred.map((asset) => `<option value="${Number(asset.id)}">${escapeHtml(asset.collection_name || "-")} · ${escapeHtml(asset.display_name)}</option>`).join("");
}

function renderCampaigns() {
  const host = $("campaign-list");
  if (!host) return;
  host.innerHTML = `<div class="table compact-table">${table(state.evaluationCampaigns, [
    { label: "Campaign", render: (row) => `<strong>${escapeHtml(row.name)}</strong><br><span class="muted">#${Number(row.id)}</span>` },
    { label: "状态", render: (row) => statusBadge(row.status) },
    { label: "候选 / 任务 / 投票", render: (row) => `${Number(row.candidates || 0)} / ${Number(row.tasks || 0)} / ${Number(row.votes || 0)}` },
    { label: "目标票数", render: (row) => Number(row.target_votes || 0) },
    { label: "操作", render: (row) => `<button class="secondary" data-campaign-select="${Number(row.id)}" type="button">打开</button>${row.status === "draft" ? ` <button data-campaign-publish="${Number(row.id)}" type="button">发布</button>` : ""}${row.status === "published" ? ` <button class="secondary" data-campaign-close="${Number(row.id)}" type="button">关闭</button>` : ""}` },
  ])}</div>`;
  if (state.selectedCampaignId) {
    host.insertAdjacentHTML("beforeend", `<section class="stats-block"><h3>已选 Campaign 候选</h3><div class="table compact-table">${table(state.campaignCandidates, [
      { label: "视频", render: (row) => escapeHtml(row.video_name) },
      { label: "GT", render: (row) => Number(row.reference_asset_id) },
      { label: "候选", render: (row) => escapeHtml(row.label_snapshot) },
      { label: "模型 / 权重", render: (row) => `${escapeHtml(row.model_snapshot || "-")} / ${escapeHtml(row.checkpoint_snapshot || "-")}` },
    ])}</div></section>`);
  }
  fillCampaignControls();
}

async function loadEvaluationCampaign(campaignId) {
  state.selectedCampaignId = Number(campaignId);
  const selected = state.evaluationCampaigns.find((row) => Number(row.id) === Number(campaignId));
  const candidatesPayload = await api(`/api/evaluation-campaigns/${Number(campaignId)}/candidates`);
  state.campaignCandidates = candidatesPayload.candidates || [];
  state.currentEvaluationTask = null;
  state.evaluationFrameIndex = 0;
  state.campaignAnalysis = null;
  if (["published", "closed"].includes(selected?.status)) {
    await loadCampaignAnalysis(campaignId);
    if (selected?.status === "published" && state.evaluatorId && state.evaluatorName) {
      const next = await api(`/api/evaluation-campaigns/${Number(campaignId)}/next?evaluator_id=${encodeURIComponent(state.evaluatorId)}`);
      state.currentEvaluationTask = next.task || null;
    }
  }
  renderCampaigns();
  renderEvaluationTask();
  renderCampaignAnalysis();
}

async function loadCampaignAnalysis(campaignId = state.selectedCampaignId) {
  if (!campaignId) return;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(state.campaignAnalysisFilters || {})) {
    if (value) params.set(key, value);
  }
  state.campaignAnalysis = await api(`/api/evaluation-campaigns/${Number(campaignId)}/analysis${params.toString() ? `?${params}` : ""}`);
}

async function loadEvaluations() {
  if (!state.mediaAssets.length) await loadMediaLibrary();
  const campaignsPayload = await api("/api/evaluation-campaigns");
  state.evaluationCampaigns = campaignsPayload.campaigns || [];
  const evaluatorForm = $("evaluator-form");
  if (evaluatorForm) evaluatorForm.elements.display_name.value = state.evaluatorName;
  const validSelection = state.evaluationCampaigns.some((row) => Number(row.id) === Number(state.selectedCampaignId));
  if (!validSelection) state.selectedCampaignId = state.evaluationCampaigns[0]?.id || null;
  if (state.selectedCampaignId) {
    await loadEvaluationCampaign(state.selectedCampaignId);
  } else {
    state.campaignCandidates = [];
    state.currentEvaluationTask = null;
    state.campaignAnalysis = null;
    renderCampaigns();
    renderEvaluationTask();
    renderCampaignAnalysis();
  }
}

async function saveEvaluator(event) {
  event.preventDefault();
  const displayName = String(formData(event.currentTarget).display_name || "").trim();
  const evaluatorId = ensureEvaluatorId();
  const result = await api("/api/evaluators/session", {
    method: "POST",
    body: JSON.stringify({ evaluator_id: evaluatorId, display_name: displayName }),
  });
  state.evaluatorName = result.evaluator.display_name;
  localStorage.setItem("vfieval-evaluator-name", state.evaluatorName);
  await loadEvaluations();
  toast("评测员信息已保存");
}

async function createEvaluationCampaign(event) {
  event.preventDefault();
  const values = formData(event.currentTarget);
  const result = await api("/api/evaluation-campaigns", {
    method: "POST",
    body: JSON.stringify({ name: values.name, target_votes: Number(values.target_votes || 3) }),
  });
  event.currentTarget.reset();
  state.selectedCampaignId = Number(result.campaign.id);
  await loadEvaluations();
  toast("Campaign 已创建");
}

async function addEvaluationCandidate(event) {
  event.preventDefault();
  const values = formData(event.currentTarget);
  await api(`/api/evaluation-campaigns/${Number(values.campaign_id)}/candidates`, {
    method: "POST",
    body: JSON.stringify({
      reference_asset_id: Number(values.reference_asset_id),
      asset_id: Number(values.asset_id),
      video_name: values.video_name,
      label: values.label || undefined,
    }),
  });
  state.selectedCampaignId = Number(values.campaign_id);
  await loadEvaluations();
  toast("候选已添加");
}

async function publishEvaluationCampaign(campaignId) {
  await api(`/api/evaluation-campaigns/${Number(campaignId)}/publish`, { method: "POST", body: "{}" });
  state.selectedCampaignId = Number(campaignId);
  await loadEvaluations();
  toast("Campaign 已发布");
}

async function closeEvaluationCampaign(campaignId) {
  await api(`/api/evaluation-campaigns/${Number(campaignId)}/close`, { method: "POST", body: "{}" });
  state.selectedCampaignId = Number(campaignId);
  await loadEvaluations();
  toast("Campaign 已关闭");
}

async function submitEvaluationVote(choice) {
  const task = state.currentEvaluationTask;
  if (!task) return;
  const reasons = Array.from(document.querySelectorAll("[data-evaluation-reason]:checked")).map((item) => item.value);
  await api(`/api/evaluation-tasks/${Number(task.id)}/votes`, {
    method: "POST",
    body: JSON.stringify({
      evaluator_id: state.evaluatorId,
      choice,
      reasons,
      confidence: $("evaluation-confidence")?.value || "",
      note: $("evaluation-note")?.value || "",
      duration_ms: Math.max(0, Date.now() - state.evaluationTaskStartedAt),
    }),
  });
  await loadEvaluations();
  toast("投票已保存");
}

async function refreshCatalog() {
  const [modelFiles, videoGroups, runs, metricHealth, checkpoints, devices] = await Promise.all([
    api("/api/model-files"),
    api("/api/video-groups?summary=1"),
    api("/api/runs"),
    api("/api/metrics/health"),
    api("/api/checkpoints"),
    api("/api/devices"),
  ]);
  Object.assign(state, { modelFiles, videoGroups, runs, metricHealth, checkpoints, devices });
  renderMetricOptions();
  renderOptions();
  renderMetricEnvironmentPanel();
  renderVideoSelection();
  renderRuns();
  schedulePreflight(0);
  if (isRunsViewActive()) {
    if (!state.selectedRun && runs.length) {
      await selectRun(runs[0].id, { quiet: true });
    } else if (state.selectedRun) {
      const exists = runs.some((item) => Number(item.id) === Number(state.selectedRun.id));
      if (exists) await selectRun(state.selectedRun.id, { quiet: true });
      else renderEmptyRunDetail();
    } else {
      renderEmptyRunDetail();
    }
  }
}

document.querySelectorAll(".nav-item").forEach((item) => item.addEventListener("click", () => {
  (async () => {
    const view = item.dataset.view;
    switchView(view);
    if (view === "compare") {
      renderCompareMetricOptions();
      if (!state.compareSourcesLoaded) {
        await loadCompareSources({ gtPage: 1, predPage: 1 });
      } else {
        renderCompareSelection();
      }
      scheduleComparePreflight(0);
      return;
    }
    if (view === "stats") {
      await loadStats();
      return;
    }
    if (view === "media") {
      await loadMediaLibrary();
      return;
    }
    if (view === "evaluations") {
      await loadEvaluations();
      return;
    }
    if (view !== "runs") {
      return;
    }
    if (!state.runs.length) {
      renderEmptyRunDetail();
      return;
    }
    if (!state.selectedRun) {
      await selectRun(state.runs[0].id, { quiet: true });
      return;
    }
    const exists = state.runs.some((row) => Number(row.id) === Number(state.selectedRun.id));
    await selectRun(exists ? state.selectedRun.id : state.runs[0].id, { quiet: true });
  })().catch((error) => toast(error.message));
}));
$("infer-form").addEventListener("submit", (event) => startRun(event).catch((error) => toast(error.message)));
$("compare-form").addEventListener("submit", (event) => startCompareRun(event).catch((error) => toast(error.message)));
$("create-adhoc-evaluation").addEventListener("click", () => createAdhocEvaluation().catch((error) => toast(error.message)));
$("collection-form").addEventListener("submit", (event) => createMediaCollection(event).catch((error) => toast(error.message)));
$("upload-form").addEventListener("submit", (event) => uploadExternalMedia(event).catch((error) => toast(error.message)));
$("evaluator-form").addEventListener("submit", (event) => saveEvaluator(event).catch((error) => toast(error.message)));
$("campaign-form").addEventListener("submit", (event) => createEvaluationCampaign(event).catch((error) => toast(error.message)));
$("candidate-form").addEventListener("submit", (event) => addEvaluationCandidate(event).catch((error) => toast(error.message)));
$("refresh").addEventListener("click", () => refreshRunsOnly().catch((error) => toast(error.message)));
$("refresh-files").addEventListener("click", () => refreshCatalog().then(() => toast("文件列表已刷新")).catch((error) => toast(error.message)));
$("refresh-compare-sources").addEventListener("click", () => loadCompareSources({ gtPage: 1, predPage: 1 }).then(() => scheduleComparePreflight(0)).then(() => toast("对比来源已刷新")).catch((error) => toast(error.message)));
$("refresh-stats").addEventListener("click", () => loadStats().then(() => toast("统计数据已刷新")).catch((error) => toast(error.message)));
$("infer-form").addEventListener("input", () => schedulePreflight());
$("compare-form").addEventListener("input", () => scheduleComparePreflight());
$("refresh-media").addEventListener("click", () => loadMediaLibrary().then(() => toast("媒体资产已刷新")).catch((error) => toast(error.message)));
$("pause-upload").addEventListener("click", () => {
  state.uploadPaused = true;
  toast(state.activeUpload ? "将在当前分片完成后暂停" : "当前没有进行中的上传");
});
$("refresh-evaluations").addEventListener("click", () => loadEvaluations().then(() => toast("盲评数据已刷新")).catch((error) => toast(error.message)));

$("infer-form").addEventListener("change", async (event) => {
  renderCustomSizeVisibility();
  if (event.target.name === "model_file") {
    renderCheckpointOptions();
  }
  if (event.target.name === "execution_mode") {
    renderSingleDeviceOptions($("infer-form").elements.device.value || "auto");
    renderDeviceOptions();
  }
  schedulePreflight(0);
});

document.addEventListener("change", (event) => {
  const evaluationFrame = event.target.closest("[data-evaluation-frame]");
  if (evaluationFrame && state.currentEvaluationTask) {
    state.evaluationFrameIndex = Number(evaluationFrame.value || 0);
    document.querySelectorAll("[data-evaluation-frame-side]").forEach((image) => {
      const url = new URL(image.src, window.location.origin);
      url.searchParams.set("frame", String(state.evaluationFrameIndex));
      image.src = `${url.pathname}${url.search}`;
    });
    const label = evaluationFrame.closest("label")?.querySelector("span");
    if (label) label.textContent = `单帧位置 ${state.evaluationFrameIndex + 1}/${Number(state.currentEvaluationTask.frame_count || 1)}`;
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
      ensureVideoSelection(name);
      loadVideoGroupPage(name, 1).then(() => schedulePreflight(0)).catch((error) => toast(error.message));
    } else {
      state.selectedGroups.delete(name);
    }
    renderVideoSelection();
    schedulePreflight(0);
    return;
  }
  const compareGt = event.target.closest("[data-compare-gt]");
  if (compareGt) {
    state.selectedCompareGtKey = compareGt.dataset.compareGt;
    state.selectedComparePredArtifacts = new Set();
    state.comparePredPage = 1;
    loadCompareSources({ predPage: 1 }).then(() => scheduleComparePreflight(0)).catch((error) => toast(error.message));
    return;
  }
  const comparePred = event.target.closest("[data-compare-pred]");
  if (comparePred) {
    const artifactId = Number(comparePred.dataset.comparePred);
    if (comparePred.checked) state.selectedComparePredArtifacts.add(artifactId);
    else state.selectedComparePredArtifacts.delete(artifactId);
    renderCompareSelection();
    scheduleComparePreflight(0);
    return;
  }
  const compareLayerKind = event.target.closest("[data-compare-layer-kind]");
  if (compareLayerKind) {
    const kind = compareLayerKind.dataset.compareLayerKind;
    if (compareLayerKind.checked) state.selectedCompareLayerKinds.add(kind);
    else state.selectedCompareLayerKinds.delete(kind);
    renderCompareSelection();
    scheduleComparePreflight(0);
    return;
  }
  const compareQuery = event.target.closest("[data-compare-query]");
  if (compareQuery) {
    if (compareQuery.dataset.compareQuery === "gt") {
      state.compareGtQuery = compareQuery.value || "";
      state.compareGtPage = 1;
      state.comparePredPage = 1;
    } else {
      state.comparePredQuery = compareQuery.value || "";
      state.comparePredPage = 1;
    }
    loadCompareSources().then(() => scheduleComparePreflight(0)).catch((error) => toast(error.message));
    return;
  }
  const compareTrackLabel = event.target.closest("[data-compare-track-label]");
  if (compareTrackLabel) {
    state.compareTrackLabels[Number(compareTrackLabel.dataset.compareTrackLabel)] = compareTrackLabel.value || "";
    scheduleComparePreflight(0);
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
    const selected = ensureVideoSelection(name);
    if (videoCheckbox.checked) selected.add(videoCheckbox.dataset.videoName);
    else selected.delete(videoCheckbox.dataset.videoName);
    renderVideoSelection();
    schedulePreflight(0);
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
    setSampleIndex(range.dataset.sampleRange, Number(range.value));
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
  if (event.target.closest("[data-campaign-analysis-apply]")) {
    document.querySelectorAll("[data-campaign-filter]").forEach((input) => {
      state.campaignAnalysisFilters[input.dataset.campaignFilter] = input.value || "";
    });
    await loadCampaignAnalysis();
    renderCampaignAnalysis();
    return;
  }
  if (event.target.closest("[data-campaign-analysis-reset]")) {
    state.campaignAnalysisFilters = { video: "", model: "", checkpoint: "", collection_id: "", evaluator_id: "" };
    await loadCampaignAnalysis();
    renderCampaignAnalysis();
    return;
  }
  const campaignSelect = event.target.closest("[data-campaign-select]");
  if (campaignSelect) {
    await loadEvaluationCampaign(Number(campaignSelect.dataset.campaignSelect));
    return;
  }
  const campaignPublish = event.target.closest("[data-campaign-publish]");
  if (campaignPublish) {
    await publishEvaluationCampaign(Number(campaignPublish.dataset.campaignPublish));
    return;
  }
  const campaignClose = event.target.closest("[data-campaign-close]");
  if (campaignClose) {
    await closeEvaluationCampaign(Number(campaignClose.dataset.campaignClose));
    return;
  }
  const evaluationVote = event.target.closest("[data-evaluation-vote]");
  if (evaluationVote) {
    await submitEvaluationVote(evaluationVote.dataset.evaluationVote);
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
    switchView("runs");
    await selectRun(Number(statsRun.dataset.statsRun));
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
  if (event.target.closest("[data-refresh-compare-sources]")) {
    await loadCompareSources({ gtPage: 1, predPage: 1 });
    scheduleComparePreflight(0);
    return;
  }
  if (event.target.closest("[data-compare-gt-clear]")) {
    state.selectedCompareGtKey = "";
    renderCompareSelection();
    scheduleComparePreflight(0);
    return;
  }
  const comparePage = event.target.closest("[data-compare-page]");
  if (comparePage) {
    const page = Number(comparePage.dataset.page || 1);
    if (comparePage.dataset.comparePage === "gt") {
      await loadCompareSources({ gtPage: page, predPage: 1 });
    } else {
      await loadCompareSources({ predPage: page });
    }
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
    const page = state.videoPages[groupName];
    const names = page?.filtered_video_names || page?.all_video_names || [];
    const selected = ensureVideoSelection(groupName);
    if (videoSelect.dataset.videoSelect === "all-filtered") {
      state.selectedVideosByGroup[groupName] = new Set([...selected, ...names]);
    } else if (videoSelect.dataset.videoSelect === "none-filtered") {
      state.selectedVideosByGroup[groupName] = new Set(Array.from(selected).filter((name) => !names.includes(name)));
    } else {
      const next = new Set(selected);
      for (const name of names) {
        if (next.has(name)) next.delete(name);
        else next.add(name);
      }
      state.selectedVideosByGroup[groupName] = next;
    }
    renderVideoSelection();
    schedulePreflight(0);
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
  const runButton = event.target.closest("[data-run-id]");
  if (runButton) {
    await selectRun(Number(runButton.dataset.runId));
    return;
  }
  const cancelButton = event.target.closest("[data-cancel-run]");
  if (cancelButton) {
    await cancelRun(Number(cancelButton.dataset.cancelRun));
    return;
  }
  const retryButton = event.target.closest("[data-retry-run]");
  if (retryButton) {
    await retryRun(Number(retryButton.dataset.retryRun));
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
    state.selectedVideoByRun[state.selectedRun.id] = videoTab.dataset.runVideo;
    if (!state.runVideoTimelines[`${state.selectedRun.id}:${videoTab.dataset.runVideo}`]) {
      await loadRunVideoTimeline(state.selectedRun.id, videoTab.dataset.runVideo);
    }
    renderRunDetail();
    return;
  }
  const windowNav = event.target.closest("[data-window-start]");
  if (windowNav && state.selectedRun) {
    const videoName = state.selectedVideoByRun[state.selectedRun.id];
    const nextStart = Math.max(0, Number(windowNav.dataset.windowStart || 0));
    await loadRunVideoTimeline(state.selectedRun.id, videoName, { windowStart: nextStart });
    // Reset the in-window selection to the first frame of the new window.
    state.selectedSampleByVideo[`${state.selectedRun.id}:${videoName}`] = 0;
    if (!updateFrameRegion()) renderRunDetail();
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
        setSampleIndex(videoName, compareStepIndex(video, currentIndex, direction));
        return;
      }
    }
    setSampleIndex(videoName, currentIndex + direction);
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
  if (chart && state.selectedRun) {
    const video = state.runVideoTimelines[`${state.selectedRun.id}:${chart.dataset.chartVideo}`];
    if (!video?.samples?.length) return;
    const rect = chart.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    setSampleIndex(chart.dataset.chartVideo, Math.round(ratio * (video.samples.length - 1)));
  }
});

document.addEventListener("mouseover", (event) => {
  const layerTile = event.target.closest("[data-layer-frame]");
  if (layerTile) {
    highlightTimelineFrame(layerTile.dataset.layerFrame);
  }
});

document.addEventListener("mouseout", (event) => {
  if (event.target.closest("[data-layer-frame]")) {
    highlightTimelineFrame(null);
  }
});

// `toggle` does not bubble, so capture it to persist the run-meta collapse
// state across the 2s poll re-render of a running run.
document.addEventListener("toggle", (event) => {
  const details = event.target;
  if (details instanceof HTMLDetailsElement && details.classList.contains("run-meta")) {
    state.runMetaCollapsed = !details.open;
  }
}, true);

function startRunsPoll() {
  setInterval(() => {
    if (document.hidden) return;
    refreshRunsOnly().catch(() => {});
  }, 2000);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refreshRunsOnly().catch(() => {});
  });
}

refreshCatalog().catch((error) => toast(error.message));
startRunsPoll();
