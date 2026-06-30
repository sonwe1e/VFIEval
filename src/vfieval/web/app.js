const STATUS_LABELS = {
  queued: "排队中",
  running: "推理中",
  completed: "已完成",
  failed: "失败",
  cancel_requested: "取消中",
  canceled: "已取消",
  metric_queued: "评测排队",
  metric_running: "评测中",
};

const TERMINAL_STATUSES = new Set(["completed", "failed", "canceled"]);
const METRICS = ["lpips_vit_patch", "lpips_convnext", "vmaf", "cgvqm"];
const PREVIEW_GROUPS_MODEL = {
  images: { label: "图像", items: [["gt", "GT"], ["pred", "Pred"], ["difference", "Diff"]] },
  flow: { label: "Flow", items: [["flowt_0", "Flow t->0"], ["flowt_1", "Flow t->1"]] },
  mask: { label: "Mask", items: [["mask0", "Mask0"], ["mask1", "Mask1"]] },
  warp: { label: "Warp", items: [["warp0", "Warp0"], ["warp1", "Warp1"], ["blend", "Blend"]] },
};
const PREVIEW_GROUPS_COMPARE = {
  images: PREVIEW_GROUPS_MODEL.images,
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
  sampleDetails: {},
  sampleDetailLoading: {},
  selectedVideosByGroup: {},
  videoPages: {},
  runVideoPageByRun: {},
  selectedVideoByRun: {},
  selectedSampleByVideo: {},
  selectedMetricByRun: {},
  selectedArtifactGroupBySample: {},
  expandedExtraArtifactsBySample: {},
  selectedCudaDevices: new Set(),
  selectedNpuDevices: new Set(),
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

function currentGroup() {
  if (currentRunType() !== "model_inference") return null;
  const name = $("infer-form").elements.video_group.value;
  return state.videoGroups.find((item) => item.name === name) || null;
}

function previewGroupsForRun(run) {
  return (run?.metadata?.run_type || "model_inference") === "video_compare"
    ? PREVIEW_GROUPS_COMPARE
    : PREVIEW_GROUPS_MODEL;
}

function metricHealthBadge(name) {
  const row = state.metricHealth?.metrics?.[name];
  if (!row) return "";
  const title = row.reason || row.status;
  return `<small class="metric-health ${escapeHtml(row.status)}" title="${escapeHtml(title)}">${escapeHtml(row.status)}</small>`;
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
  const runType = currentRunType();
  document.querySelectorAll("[data-mode-section]").forEach((item) => {
    item.hidden = item.dataset.modeSection !== runType;
  });
  $("video-selection").hidden = runType !== "model_inference";
}

function renderOptions() {
  const form = $("infer-form");
  const previousModel = form.elements.model_file.value;
  const previousGroup = form.elements.video_group.value;
  const previousCheckpoint = form.elements.checkpoint?.value || "none";
  const previousDevice = form.elements.device?.value || "auto";

  form.elements.model_file.innerHTML = state.modelFiles
    .map((item) => `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)}</option>`)
    .join("");
  form.elements.video_group.innerHTML = state.videoGroups
    .map((item) => `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)} (${item.video_count})</option>`)
    .join("");

  form.elements.model_file.value = state.modelFiles.some((item) => item.name === previousModel)
    ? previousModel
    : (state.modelFiles.some((item) => item.name === "test_average.py") ? "test_average.py" : state.modelFiles[0]?.name || "");
  form.elements.video_group.value = state.videoGroups.some((item) => item.name === previousGroup)
    ? previousGroup
    : (state.videoGroups.some((item) => item.name === "test_style") ? "test_style" : state.videoGroups[0]?.name || "");

  if (!form.elements.run_type.value) {
    form.elements.run_type.value = "model_inference";
  }

  renderCheckpointOptions(previousCheckpoint);
  renderSingleDeviceOptions(previousDevice);
  renderDeviceOptions();
  ensureVideoSelection(form.elements.video_group.value);
  renderCustomSizeVisibility();
  renderModeSections();
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
  if (!state.selectedVideosByGroup[groupName]) {
    const page = state.videoPages[groupName];
    const names = page?.all_video_names || (group.videos || []).map((video) => video.name);
    state.selectedVideosByGroup[groupName] = new Set(names);
  }
  return state.selectedVideosByGroup[groupName];
}

function selectedVideoNames() {
  const group = currentGroup();
  if (!group) return [];
  const page = state.videoPages[group.name];
  const names = page?.all_video_names || (group.videos || []).map((video) => video.name);
  const selected = ensureVideoSelection(group.name);
  return names.filter((name) => selected.has(name));
}

async function loadVideoGroupPage(groupName, page = 1) {
  if (!groupName) return;
  const payload = await api(`/api/video-groups/${encodeURIComponent(groupName)}/videos?page=${page}&page_size=50`);
  state.videoPages[groupName] = payload;
  ensureVideoSelection(groupName);
  renderVideoSelection();
}

function renderVideoSelection() {
  if (currentRunType() !== "model_inference") {
    $("video-selection").innerHTML = `
      <h2>对比输入</h2>
      <p class="muted">双视频对比模式直接使用上面的 GT / Pred 路径，不需要视频集列表。</p>
    `;
    return;
  }
  const group = currentGroup();
  const page = group ? state.videoPages[group.name] : null;
  if (!group) {
    $("video-selection").innerHTML = "<p class=\"muted\">先选择一个视频集。</p>";
    return;
  }
  if (!page) {
    $("video-selection").innerHTML = "<p class=\"muted\">正在加载视频列表...</p>";
    return;
  }
  const selected = ensureVideoSelection(group.name);
  $("video-selection").innerHTML = `
    <div class="panel-head">
      <div>
        <h2>视频选择</h2>
        <p class="muted">默认全选；你可以只选择本次需要推理的视频。</p>
      </div>
      <div class="actions">
        <span class="muted">${selected.size}/${page.video_count} 已选</span>
        <button class="secondary" data-video-select="all" type="button">全选</button>
        <button class="secondary" data-video-select="none" type="button">清空</button>
        <button class="secondary" data-video-select="invert" type="button">反选</button>
      </div>
    </div>
    <div class="table compact-table">${table(page.videos || [], [
      { label: "", render: (row) => `<input type="checkbox" data-video-name="${escapeHtml(row.name)}" ${selected.has(row.name) ? "checked" : ""}>` },
      { label: "视频", render: (row) => escapeHtml(row.name) },
      { label: "帧数", render: (row) => escapeHtml(row.frame_count) },
      { label: "Triplets", render: (row) => escapeHtml(row.valid_triplets ?? 0) },
      { label: "FPS", render: (row) => formatNumber(row.fps) },
      { label: "分辨率", render: (row) => `${escapeHtml(row.width)}x${escapeHtml(row.height)}` },
      { label: "缓存", render: (row) => escapeHtml(row.cache_status || "-") },
    ])}</div>
  `;
}

function selectedMetrics() {
  return Array.from(document.querySelectorAll("input[name='metrics']:checked")).map((item) => item.value);
}

function payloadFromForm() {
  const data = formData($("infer-form"));
  if ((data.run_type || "model_inference") === "video_compare") {
    return {
      run_type: "video_compare",
      reference: data.reference || "",
      distorted: data.distorted || "",
      align_mode: data.align_mode || "strict",
      metrics: selectedMetrics(),
    };
  }
  return {
    run_type: "model_inference",
    model_file: data.model_file,
    checkpoint: data.checkpoint || "none",
    video_group: data.video_group,
    selected_videos: selectedVideoNames(),
    resolution_mode: data.resolution_mode || "original",
    height: data.height ? Number(data.height) : null,
    width: data.width ? Number(data.width) : null,
    device: data.device || "auto",
    execution_mode: data.execution_mode || "single",
    devices: data.execution_mode === "multi_npu"
      ? Array.from(state.selectedNpuDevices)
      : Array.from(state.selectedCudaDevices),
    precision: data.precision || "auto",
    batch_size: Number(data.batch_size || 1),
    batch_size_per_device: Number(data.batch_size_per_device || data.batch_size || 1),
    frame_step: Number(data.frame_step || 1),
    max_frames: data.max_frames ? Number(data.max_frames) : null,
    metrics: selectedMetrics(),
  };
}

function schedulePreflight(delay = 250) {
  clearTimeout(state.preflightTimer);
  state.preflightTimer = setTimeout(() => runPreflight().catch(renderPreflightError), delay);
}

async function runPreflight() {
  const payload = payloadFromForm();
  if (payload.run_type === "video_compare") {
    if (!payload.reference || !payload.distorted) {
      state.preflight = null;
      renderPreflight();
      return;
    }
  } else if (!payload.model_file || !payload.video_group) {
    state.preflight = null;
    renderPreflight();
    return;
  }
  state.preflight = await api("/api/preflight", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  renderPreflight();
}

function renderPreflightError(error) {
  state.preflight = { ok: false, errors: [{ title: "预检查请求失败", message: error.message }], warnings: [] };
  renderPreflight();
}

function renderMetricSetup(row) {
  const summary = row.setup_summary || "-";
  const requirements = Array.isArray(row.setup_requirements) ? row.setup_requirements : [];
  const list = requirements.length
    ? `<ul>${requirements.map((item) => `<li><strong>${escapeHtml(item.kind || "requirement")}</strong>: ${escapeHtml(item.target || "-")} <span>${escapeHtml(item.description || "")}</span></li>`).join("")}</ul>`
    : "";
  return `<div class="metric-setup"><p>${escapeHtml(summary)}</p>${list}</div>`;
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
    ${renderMetricHealthTable(state.metricHealth?.metrics || {})}
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
  if (currentRunType() === "model_inference" && (!state.modelFiles.length || !state.videoGroups.length)) {
    start.disabled = true;
    $("preflight").innerHTML = `
      <h2>准备输入</h2>
      <p class="muted">需要在项目根目录放置 <code>models/*.py</code> 和 <code>videos/*/</code>。</p>
    `;
    return;
  }
  if (!result) {
    start.disabled = true;
    $("preflight").innerHTML = currentRunType() === "video_compare"
      ? "<p class=\"muted\">填写 GT / Pred 路径后会自动预检查。</p>"
      : "<p class=\"muted\">选择模型和视频后会自动预检查。</p>";
    return;
  }
  start.disabled = !result.ok;

  if (result.run_type === "video_compare") {
    $("preflight").innerHTML = `
      <div class="panel-head">
        <h2>运行前预检查</h2>
        ${result.ok ? "<span class=\"ok-text\">通过</span>" : "<span class=\"bad-text\">未通过</span>"}
      </div>
      <div class="summary-grid">
        <div><span>模式</span><strong>video_compare</strong></div>
        <div><span>GT</span><strong>${escapeHtml(result.reference?.name || "-")}</strong></div>
        <div><span>Pred</span><strong>${escapeHtml(result.distorted?.name || "-")}</strong></div>
        <div><span>帧数</span><strong>${escapeHtml(result.alignment?.frame_count ?? "-")}</strong></div>
        <div><span>分辨率</span><strong>${escapeHtml(`${result.alignment?.width || "-"}x${result.alignment?.height || "-"}`)}</strong></div>
        <div><span>FPS</span><strong>${formatNumber(result.alignment?.fps)}</strong></div>
      </div>
      ${renderMessages("errors", result.errors || [])}
      ${renderMessages("warnings", result.warnings || [])}
      ${renderMetricHealthTable(result.metrics?.health || {})}
    `;
    return;
  }

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
    ${renderMetricHealthTable(result.metrics?.health || {})}
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

function table(rows, columns) {
  if (!rows?.length) return "<p class=\"muted\">暂无数据。</p>";
  return `
    <table>
      <thead><tr>${columns.map((column) => `<th>${column.label}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((row) => `<tr>${columns.map((column) => `<td>${column.render(row)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table>
  `;
}

async function startRun(event) {
  event.preventDefault();
  await runPreflight();
  if (!state.preflight?.ok) {
    toast("预检查未通过");
    return;
  }
  const created = await api("/api/runs", {
    method: "POST",
    body: JSON.stringify(payloadFromForm()),
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
    return `${run.metadata?.reference_path || "-"} -> ${run.metadata?.distorted_path || "-"}`;
  }
  return `${run.metadata?.model_file || run.model_name || "-"} / ${run.metadata?.video_group || run.dataset_name || "-"}`;
}

function renderRuns() {
  $("runs-table").innerHTML = table(state.runs, [
    { label: "Run", render: (run) => `<button class="link-button" data-run-id="${run.id}" type="button">#${run.id}</button>` },
    { label: "状态", render: (run) => statusBadge(run.status) },
    { label: "类型", render: (run) => escapeHtml(run.metadata?.run_type || "model_inference") },
    { label: "来源", render: (run) => escapeHtml(runSourceLabel(run)) },
    { label: "进度", render: (run) => `${escapeHtml(run.progress_current || 0)}/${escapeHtml(run.progress_total || 0)}` },
    { label: "输出", render: (run) => escapeHtml(run.metadata?.output_dir || run.result?.output_dir || "-") },
  ]);
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
  if (run.status === "queued") return runType === "video_compare" ? "等待生成对比产物" : "等待推理";
  if (run.status === "running") return runType === "video_compare" ? "生成对比产物中" : "推理中";
  if (run.status === "failed") return "失败";
  if (run.status === "canceled" || run.status === "cancel_requested") return "已取消";
  return run.inference_job_id ? "已完成" : "未开始";
}

function renderMetricPhase(run) {
  const metrics = run.metrics || [];
  if (!metrics.length) return "未选择指标";
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
    return `<div class="message error"><p><strong>${escapeHtml(run.error.type || "Error")}</strong>: ${escapeHtml(run.error.message || JSON.stringify(run.error))}</p>${meta}</div>`;
  }
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
  if (job.role === "inference") {
    return `inference #${escapeHtml(job.shard_index ?? 0)}`;
  }
  return escapeHtml(job.role || job.kind || "job");
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

async function loadRunVideoTimeline(runId, videoName, options = {}) {
  const metric = options.metric || state.selectedMetricByRun[runId] || "";
  const windowStart = Number(options.windowStart ?? 0);
  const payload = await api(
    `/api/runs/${runId}/videos/${encodeURIComponent(videoName)}/timeline?bucket_count=160&window_start=${windowStart}&window_size=300${metric ? `&metric=${encodeURIComponent(metric)}` : ""}`,
  );
  state.runVideoTimelines[`${runId}:${videoName}`] = payload;
  return payload;
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
        <button class="secondary" data-cancel-run="${run.id}" ${TERMINAL_STATUSES.has(run.status) ? "disabled" : ""} type="button">取消</button>
        <button class="secondary" data-retry-run="${run.id}" type="button">重试</button>
        <button class="secondary danger" data-delete-run="${run.id}" type="button">删除记录</button>
        <button class="secondary" data-cleanup-run="${run.id}" ${TERMINAL_STATUSES.has(run.status) && !run.artifact_cleaned_at ? "" : "disabled"} type="button">清理产物</button>
      </div>
    </div>
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
    <div class="message"><p><strong>Execution</strong>: ${runExecutionTarget(run)}</p></div>
    ${renderMetricHealthTable(run.metadata?.metric_health || {})}
    ${renderRunJobs(run)}
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
        ${video ? renderVideoTimeline(video) : (selectedVideoName ? "<p class=\"muted\">正在加载时间轴...</p>" : "<p class=\"muted\">暂无结果。</p>")}
      </section>
    </div>
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
      const x = samples.length <= 1 ? 0 : (index / (samples.length - 1)) * 100;
      const normalized = max === min ? 0.5 : (value - min) / (max - min);
      const y = 34 - normalized * 28;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .filter(Boolean)
    .join(" ");
  const markerX = samples.length <= 1 ? 0 : (selectedIndex / (samples.length - 1)) * 100;
  return `
    <div class="chart" data-chart-video="${escapeHtml(video.video_name)}">
      <div class="chart-head">
        <strong>${escapeHtml(metricName)}</strong>
        <span class="muted">点击曲线定位当前帧</span>
      </div>
      <svg viewBox="0 0 100 40" preserveAspectRatio="none" role="img">
        <polyline points="${points}" fill="none" stroke="currentColor" stroke-width="1.6"></polyline>
        <line x1="${markerX.toFixed(2)}" x2="${markerX.toFixed(2)}" y1="3" y2="37"></line>
        ${renderMetricPoints(video.video_name, samples, metricName, min, max)}
      </svg>
    </div>
  `;
}

function renderMetricPoints(videoName, samples, metricName, min, max) {
  return samples.map((sample, index) => {
    const metric = sample.metrics?.[metricName];
    const x = samples.length <= 1 ? 0 : (index / (samples.length - 1)) * 100;
    const status = metric?.status || "missing";
    const value = metric?.value;
    const normalized = status === "completed" && value !== null && max !== min ? (Number(value) - min) / (max - min) : 0.5;
    const y = status === "completed" ? 34 - normalized * 28 : 36;
    return `<circle class="metric-point ${escapeHtml(status)}" data-chart-video="${escapeHtml(videoName)}" data-chart-sample="${index}" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="1.4"><title>${escapeHtml(status)} ${escapeHtml(value ?? metricReason(metric))}</title></circle>`;
  }).join("");
}

function renderStatusStrip(samples, metricName, selectedIndex) {
  return `
    <div class="status-strip">
      ${samples.map((sample, index) => {
        const metric = sample.metrics?.[metricName];
        const status = metric?.status || "missing";
        return `<button class="status-dot ${escapeHtml(status)} ${index === selectedIndex ? "active" : ""}" data-sample-jump="${index}" type="button" title="${escapeHtml(status)} ${escapeHtml(metricReason(metric))}"></button>`;
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
  return `<a class="small-video-link" href="/api/files/${artifactId}" target="_blank" rel="noreferrer">${escapeHtml(label)} 视频</a>`;
}

function renderVideoTimeline(video) {
  const samples = video.samples || [];
  const key = `${state.selectedRun.id}:${video.video_name}`;
  const selectedIndex = Math.min(Number(state.selectedSampleByVideo[key] || 0), Math.max(0, samples.length - 1));
  state.selectedSampleByVideo[key] = selectedIndex;
  const sample = samples[selectedIndex] || null;
  const metricName = selectedMetric(video);
  return `
    <div class="panel-head compact-head">
      <div>
        <h3>${escapeHtml(video.video_file || video.video_name)}</h3>
        <p class="muted">${samples.length} 个样本，FPS ${formatNumber(video.fps)}</p>
      </div>
      <div class="actions">
        ${renderVideoPlayer("pred", video.video_artifacts?.pred_video)}
        ${renderVideoPlayer("gt", video.video_artifacts?.gt_video)}
        ${renderVideoPlayer("diff", video.video_artifacts?.diff_video)}
      </div>
    </div>
    ${renderMetricToolbar(video, metricName)}
    ${renderMetricChart(video, selectedIndex, metricName)}
    ${renderWorstSamples(video, metricName)}
    <div class="sample-controls">
      <button class="secondary" data-sample-step="-1" type="button">上一帧</button>
      <input data-sample-range="${escapeHtml(video.video_name)}" type="range" min="0" max="${Math.max(0, samples.length - 1)}" value="${selectedIndex}">
      <button class="secondary" data-sample-step="1" type="button">下一帧</button>
      <span class="muted">${selectedIndex + 1}/${samples.length || 0}</span>
    </div>
    ${sample ? renderSamplePreview(sample) : "<p class=\"muted\">没有样本。</p>"}
  `;
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
  try {
    state.sampleDetails[key] = await api(`/api/runs/${state.selectedRun.id}/samples/${sampleId}`);
  } catch (error) {
    state.sampleDetails[key] = { sample_id: sampleId, artifacts: {}, extra_artifacts: [], load_error: error.message };
  } finally {
    delete state.sampleDetailLoading[key];
    renderRunDetail();
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

function renderSamplePreview(sample) {
  const detail = sampleDetail(sample.sample_id);
  if (!detail && sample.has_artifacts !== false) {
    loadSampleDetail(sample.sample_id);
  }
  if (sample.has_artifacts === false) {
    const reason = state.selectedRun?.artifact_cleaned_at
      ? "这个 Run 的产物已清理；如需重新查看预览，请重试重新生成。"
      : "这个样本当前没有可用产物。";
    return `
      <div class="sample-meta">
        <strong>${escapeHtml(sample.sample_name)}</strong>
        <span>frame ${escapeHtml(sample.frame_index)}</span>
        <span>${sample.timestamp === null || sample.timestamp === undefined ? "-" : `${formatNumber(sample.timestamp)}s`}</span>
        ${renderSampleMetrics(sample)}
      </div>
      <div class="message warn"><p><strong>没有可加载的产物</strong>: ${escapeHtml(reason)}</p></div>
    `;
  }
  const payload = detail
    ? {
        ...detail,
        ...sample,
        artifacts: detail.artifacts || sample.artifacts || {},
        extra_artifacts: detail.extra_artifacts || sample.extra_artifacts || [],
        sample_files: detail.sample_files || sample.sample_files || {},
        load_error: detail.load_error,
      }
    : sample;
  const groups = previewGroupsForRun(state.selectedRun);
  const groupKey = state.selectedArtifactGroupBySample[payload.sample_id] || "images";
  const group = groups[groupKey] || groups.images;
  const loadState = detail
    ? (payload.load_error
        ? `<div class="message error"><p><strong>样本产物加载失败</strong>: ${escapeHtml(payload.load_error)}</p></div>`
        : "")
    : "<p class=\"muted sample-loading\">正在按需加载这一帧的产物...</p>";
  return `
    <div class="sample-meta">
      <strong>${escapeHtml(payload.sample_name)}</strong>
      <span>frame ${escapeHtml(payload.frame_index)}</span>
      <span>${payload.timestamp === null || payload.timestamp === undefined ? "-" : `${formatNumber(payload.timestamp)}s`}</span>
      ${renderSampleMetrics(payload)}
    </div>
    ${loadState}
    <div class="artifact-tabs">
      ${Object.entries(groups).map(([key, value]) => `
        <button class="secondary ${key === groupKey ? "active" : ""}" data-artifact-group="${escapeHtml(key)}" data-artifact-sample="${escapeHtml(payload.sample_id)}" type="button">${escapeHtml(value.label)}</button>
      `).join("")}
    </div>
    <div class="preview-grid">
      ${group.items.map(([kind, label]) => renderPreviewSlot(payload, kind, label)).join("")}
    </div>
    ${renderExtraArtifacts(payload)}
  `;
}

function setSampleIndex(videoName, index) {
  if (!state.selectedRun) return;
  const video = state.runVideoTimelines[`${state.selectedRun.id}:${videoName}`];
  if (!video) return;
  const max = Math.max(0, (video.samples || []).length - 1);
  state.selectedSampleByVideo[`${state.selectedRun.id}:${videoName}`] = Math.max(0, Math.min(max, index));
  renderRunDetail();
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

async function cleanupRunArtifacts(runId) {
  await api(`/api/runs/${runId}/cleanup-artifacts`, { method: "POST", body: "{}" });
  toast(`Run #${runId} 产物已清理`);
  await refreshRunsOnly();
  if (state.selectedRun?.id === runId) {
    await selectRun(runId, { quiet: true });
  }
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(2);
}

function formatDuration(value) {
  const seconds = Number(value || 0);
  if (!Number.isFinite(seconds) || seconds <= 0) return "-";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

async function refreshCatalog() {
  const [modelFiles, videoGroups, runs, metricHealth, checkpoints, devices] = await Promise.all([
    api("/api/model-files"),
    api("/api/video-groups"),
    api("/api/runs"),
    api("/api/metrics/health"),
    api("/api/checkpoints"),
    api("/api/devices"),
  ]);
  Object.assign(state, { modelFiles, videoGroups, runs, metricHealth, checkpoints, devices });
  renderMetricOptions();
  renderOptions();
  renderMetricEnvironmentPanel();
  if (currentRunType() === "model_inference") {
    await loadVideoGroupPage($("infer-form").elements.video_group.value);
  } else {
    renderVideoSelection();
  }
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
    switchView(item.dataset.view);
    if (item.dataset.view !== "runs") {
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
$("refresh").addEventListener("click", () => refreshRunsOnly().catch((error) => toast(error.message)));
$("refresh-files").addEventListener("click", () => refreshCatalog().then(() => toast("文件列表已刷新")).catch((error) => toast(error.message)));
$("infer-form").addEventListener("input", () => schedulePreflight());

$("infer-form").addEventListener("change", async (event) => {
  renderModeSections();
  renderCustomSizeVisibility();
  if (event.target.name === "model_file") {
    renderCheckpointOptions();
  }
  if (event.target.name === "execution_mode") {
    renderSingleDeviceOptions($("infer-form").elements.device.value || "auto");
    renderDeviceOptions();
  }
  if (event.target.name === "run_type") {
    if (currentRunType() === "model_inference") {
      await loadVideoGroupPage($("infer-form").elements.video_group.value);
    } else {
      renderVideoSelection();
    }
  }
  if (event.target.name === "video_group" && currentRunType() === "model_inference") {
    ensureVideoSelection(event.target.value);
    await loadVideoGroupPage(event.target.value);
  }
  schedulePreflight(0);
});

document.addEventListener("change", (event) => {
  const videoCheckbox = event.target.closest("[data-video-name]");
  if (videoCheckbox) {
    const group = currentGroup();
    if (!group) return;
    const selected = ensureVideoSelection(group.name);
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
      .then(() => renderRunDetail())
      .catch((error) => toast(error.message));
    return;
  }
  const range = event.target.closest("[data-sample-range]");
  if (range) {
    setSampleIndex(range.dataset.sampleRange, Number(range.value));
  }
});

document.addEventListener("click", async (event) => {
  const videoSelect = event.target.closest("[data-video-select]");
  if (videoSelect) {
    const group = currentGroup();
    if (!group) return;
    const page = state.videoPages[group.name];
    const names = page?.all_video_names || [];
    const selected = ensureVideoSelection(group.name);
    if (videoSelect.dataset.videoSelect === "all") {
      state.selectedVideosByGroup[group.name] = new Set(names);
    } else if (videoSelect.dataset.videoSelect === "none") {
      state.selectedVideosByGroup[group.name] = new Set();
    } else {
      state.selectedVideosByGroup[group.name] = new Set(names.filter((name) => !selected.has(name)));
    }
    renderVideoSelection();
    schedulePreflight(0);
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
  const runVideosPage = event.target.closest("[data-run-videos-page]");
  if (runVideosPage && state.selectedRun) {
    await loadRunVideosPage(state.selectedRun.id, Number(runVideosPage.dataset.runVideosPage));
    renderRunDetail();
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
  const stepButton = event.target.closest("[data-sample-step]");
  if (stepButton && state.selectedRun) {
    const videoName = state.selectedVideoByRun[state.selectedRun.id];
    const key = `${state.selectedRun.id}:${videoName}`;
    setSampleIndex(videoName, Number(state.selectedSampleByVideo[key] || 0) + Number(stepButton.dataset.sampleStep));
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
    renderRunDetail();
    return;
  }
  const extraToggle = event.target.closest("[data-extra-toggle]");
  if (extraToggle) {
    const sampleId = extraToggle.dataset.extraToggle;
    state.expandedExtraArtifactsBySample[sampleId] = !state.expandedExtraArtifactsBySample[sampleId];
    renderRunDetail();
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

refreshCatalog().catch((error) => toast(error.message));
setInterval(() => refreshRunsOnly().catch(() => {}), 2000);
