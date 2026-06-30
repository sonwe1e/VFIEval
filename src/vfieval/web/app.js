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

const TERMINAL = new Set(["completed", "failed", "canceled"]);
const METRICS = ["lpips_vit_patch", "lpips_convnext", "vmaf", "cgvqm"];
const CORE_PREVIEW = [
  ["gt", "GT"],
  ["pred", "Pred"],
  ["difference", "Diff"],
  ["flowt_0", "Flow t->0"],
  ["flowt_1", "Flow t->1"],
  ["mask0", "Mask0"],
  ["mask1", "Mask1"],
  ["warp0", "Warp0"],
  ["warp1", "Warp1"],
  ["blend", "Blend"],
];

const state = {
  modelFiles: [],
  videoGroups: [],
  checkpoints: [],
  devices: null,
  runs: [],
  metricHealth: null,
  selectedRun: null,
  runTimeline: null,
  runVideosPage: null,
  runVideoTimelines: {},
  metricSummary: null,
  preflight: null,
  preflightTimer: null,
  selectedVideosByGroup: {},
  selectedVideoByRun: {},
  selectedSampleByVideo: {},
  selectedMetricByRun: {},
  videoSearchByGroup: {},
  videoSearchTimer: null,
  videoPageByGroup: {},
  videoSortByGroup: {},
  videoPages: {},
  sampleDetails: {},
  sampleDetailLoading: {},
  selectedCudaDevices: new Set(),
  lastCatalogRefresh: null,
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

function statusBadge(status) {
  return `<span class="status ${escapeHtml(status)}">${escapeHtml(STATUS_LABELS[status] || status)}</span>`;
}

function switchView(view) {
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  document.querySelectorAll(".view").forEach((item) => item.classList.toggle("active", item.id === `view-${view}`));
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
  Object.assign(state, { modelFiles, videoGroups, runs, metricHealth, checkpoints, devices, lastCatalogRefresh: new Date() });
  renderMetricOptions();
  renderOptions();
  renderCheckpointOptions();
  renderDeviceOptions();
  await loadVideoGroupPage($("infer-form").elements.video_group.value);
  renderRuns();
  schedulePreflight(0);
  if (!state.selectedRun && runs.length) {
    await selectRun(runs[0].id, { quiet: true });
  } else if (state.selectedRun) {
    await selectRun(state.selectedRun.id, { quiet: true });
  }
}

async function refreshRunsOnly() {
  state.runs = await api("/api/runs");
  renderRuns();
  if (state.selectedRun) await selectRun(state.selectedRun.id, { quiet: true });
}

function currentGroup() {
  const name = $("infer-form").elements.video_group.value;
  return state.videoGroups.find((item) => item.name === name) || null;
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
  const selected = ensureVideoSelection(group.name);
  const page = state.videoPages[group.name];
  const names = page?.all_video_names || (group.videos || []).map((video) => video.name);
  return names.filter((name) => selected.has(name));
}

function renderMetricOptions() {
  $("metrics-options").innerHTML = METRICS.map(
    (name) => `
      <label class="check-item">
        <input type="checkbox" name="metrics" value="${escapeHtml(name)}">
        <span>${escapeHtml(name)} ${metricHealthBadge(name)}</span>
      </label>
    `,
  ).join("") + "<p class=\"muted metric-hint\">指标会在推理完成后进入评测阶段；缺少外部 evaluator 时会显示 unavailable，不会替换成其它分数。</p>";
}

function metricHealthBadge(name) {
  const health = state.metricHealth?.metrics?.[name];
  if (!health) return "";
  return `<small class="metric-health ${escapeHtml(health.status)}" title="${escapeHtml(health.reason || "")}">${escapeHtml(health.status)}</small>`;
}

function renderOptions() {
  const form = $("infer-form");
  const previousModel = form.elements.model_file.value;
  const previousGroup = form.elements.video_group.value;
  const previousCheckpoint = form.elements.checkpoint?.value || "none";
  form.elements.model_file.innerHTML = state.modelFiles
    .map((item) => `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)}</option>`)
    .join("");
  form.elements.video_group.innerHTML = state.videoGroups
    .map((item) => `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)} (${item.video_count} 个视频)</option>`)
    .join("");
  form.elements.model_file.value = state.modelFiles.some((item) => item.name === previousModel)
    ? previousModel
    : (state.modelFiles.some((item) => item.name === "test_average.py") ? "test_average.py" : state.modelFiles[0]?.name || "");
  form.elements.video_group.value = state.videoGroups.some((item) => item.name === previousGroup)
    ? previousGroup
    : (state.videoGroups.some((item) => item.name === "test_style") ? "test_style" : state.videoGroups[0]?.name || "");
  renderCheckpointOptions(previousCheckpoint);
  renderDeviceOptions();
  ensureVideoSelection(form.elements.video_group.value);
  renderCustomSizeVisibility();
}

function renderCheckpointOptions(previousValue = null) {
  const form = $("infer-form");
  if (!form.elements.checkpoint) return;
  const modelFile = form.elements.model_file.value;
  const modelStem = modelFile.replace(/\.py$/i, "");
  const rows = (state.checkpoints || []).filter((item) => !modelFile || item.model === modelStem);
  const options = [
    `<option value="none">不加载权重</option>`,
    `<option value="auto">自动选择最新权重</option>`,
    ...rows.map((item) => `<option value="${escapeHtml(item.relative_path)}">${escapeHtml(item.relative_path)}</option>`),
  ];
  form.elements.checkpoint.innerHTML = options.join("");
  const desired = previousValue ?? form.elements.checkpoint.value;
  form.elements.checkpoint.value = rows.some((item) => item.relative_path === desired) || ["none", "auto"].includes(desired)
    ? desired
    : "none";
}

function renderDeviceOptions() {
  const container = $("device-options");
  if (!container) return;
  const cuda = state.devices?.cuda || [];
  if (!state.selectedCudaDevices.size) {
    state.selectedCudaDevices = new Set(cuda.map((item) => item.id));
  }
  const executionMode = $("infer-form").elements.execution_mode?.value || "single";
  container.innerHTML = `
    <div class="device-panel ${executionMode === "multi_cuda" ? "visible" : ""}">
      <span>CUDA 多卡</span>
      ${cuda.length ? cuda.map((item) => `
        <label class="check-item">
          <input type="checkbox" data-cuda-device="${escapeHtml(item.id)}" ${state.selectedCudaDevices.has(item.id) ? "checked" : ""}>
          <span>${escapeHtml(item.id)} ${escapeHtml(item.name || "")}</span>
        </label>
      `).join("") : "<p class=\"muted\">当前没有检测到可用 CUDA 设备。</p>"}
    </div>
  `;
}

function renderCustomSizeVisibility() {
  const mode = $("infer-form").elements.resolution_mode.value;
  document.querySelectorAll(".custom-size").forEach((item) => item.classList.toggle("visible", mode === "custom"));
}

function selectedMetrics() {
  return Array.from(document.querySelectorAll("input[name='metrics']:checked")).map((item) => item.value);
}

function selectedCudaDevices() {
  return Array.from(state.selectedCudaDevices || []);
}

function payloadFromForm() {
  const data = formData($("infer-form"));
  return {
    model_file: data.model_file,
    checkpoint: data.checkpoint || "none",
    video_group: data.video_group,
    selected_videos: selectedVideoNames(),
    resolution_mode: data.resolution_mode || "original",
    height: data.height ? Number(data.height) : null,
    width: data.width ? Number(data.width) : null,
    device: data.device || "auto",
    execution_mode: data.execution_mode || "single",
    devices: selectedCudaDevices(),
    precision: data.precision || "auto",
    batch_size: Number(data.batch_size || 1),
    batch_size_per_device: Number(data.batch_size_per_device || data.batch_size || 1),
    frame_step: Number(data.frame_step || 1),
    max_frames: data.max_frames ? Number(data.max_frames) : null,
    metrics: selectedMetrics(),
  };
}

function renderVideoSelection() {
  const group = currentGroup();
  const container = $("video-selection");
  if (!group) {
    container.innerHTML = "<h2>视频选择</h2><p class=\"muted\">先选择一个视频集。</p>";
    return;
  }
  const selected = ensureVideoSelection(group.name);
  const page = state.videoPages[group.name];
  if (!page) {
    container.innerHTML = "<h2>视频选择</h2><p class=\"muted\">正在读取视频列表...</p>";
    return;
  }
  const search = state.videoSearchByGroup[group.name] || "";
  const sort = state.videoSortByGroup[group.name] || "name";
  const selectedCount = (page.all_video_names || []).filter((name) => selected.has(name)).length;
  container.innerHTML = `
    <div class="panel-head">
      <div>
        <h2>视频选择</h2>
        <p class="muted">默认全选，可以搜索、翻页，并只勾选本次要推理的视频。</p>
      </div>
      <div class="actions">
        <span class="muted">${selectedCount}/${page.video_count} 已选</span>
        <button class="secondary" data-video-select="all" type="button">全选</button>
        <button class="secondary" data-video-select="invert" type="button">反选</button>
        <button class="secondary" data-video-select="none" type="button">清空</button>
      </div>
    </div>
    <div class="video-tools">
      <label class="video-filter">
        <span>搜索视频</span>
        <input data-video-search="${escapeHtml(group.name)}" type="search" value="${escapeHtml(search)}" placeholder="按文件名过滤，不会改变已选状态">
      </label>
      <label>
        <span>排序</span>
        <select data-video-sort="${escapeHtml(group.name)}">
          ${["name", "-duration", "-frame_count", "-resolution", "-triplets"].map((item) => `<option value="${item}" ${item === sort ? "selected" : ""}>${escapeHtml(item)}</option>`).join("")}
        </select>
      </label>
    </div>
    <div class="table compact-table">${table(page.videos || [], [
      { label: "", render: (row) => `<input type="checkbox" data-video-name="${escapeHtml(row.name)}" ${selected.has(row.name) ? "checked" : ""}>` },
      { label: "缩略图", render: (row) => row.thumbnail_url ? `<img class="video-thumb" src="${escapeHtml(row.thumbnail_url)}" alt="">` : "<span class=\"muted\">-</span>" },
      { label: "视频", render: (row) => escapeHtml(row.name) },
      { label: "真实帧数", render: (row) => `${escapeHtml(row.frame_count)}${row.frame_count_source === "exact" ? "" : " *"}` },
      { label: "时长", render: (row) => formatDuration(row.duration_seconds) },
      { label: "Triplets", render: (row) => escapeHtml(row.valid_triplets ?? 0) },
      { label: "FPS", render: (row) => formatNumber(row.fps) },
      { label: "分辨率", render: (row) => `${escapeHtml(row.width)}x${escapeHtml(row.height)}` },
      { label: "缓存", render: (row) => escapeHtml(row.cache_status || "-") },
      { label: "状态", render: (row) => row.decodable ? "可解码" : escapeHtml(row.error || "异常") },
    ])}</div>
    <div class="pager">
      <button class="secondary" data-video-page="${page.page - 1}" ${page.page <= 1 ? "disabled" : ""} type="button">上一页</button>
      <span class="muted">第 ${escapeHtml(page.page)} / ${escapeHtml(page.total_pages)} 页，当前筛选 ${escapeHtml(page.filtered_count)} 个</span>
      <button class="secondary" data-video-page="${page.page + 1}" ${page.page >= page.total_pages ? "disabled" : ""} type="button">下一页</button>
    </div>
  `;
}

async function loadVideoGroupPage(groupName, page = null) {
  if (!groupName) return;
  const nextPage = page ?? state.videoPageByGroup[groupName] ?? 1;
  const q = encodeURIComponent(state.videoSearchByGroup[groupName] || "");
  const sort = encodeURIComponent(state.videoSortByGroup[groupName] || "name");
  const payload = await api(`/api/video-groups/${encodeURIComponent(groupName)}/videos?page=${nextPage}&page_size=50&q=${q}&sort=${sort}`);
  state.videoPages[groupName] = payload;
  state.videoPageByGroup[groupName] = payload.page;
  ensureVideoSelection(groupName);
  renderVideoSelection();
}

function schedulePreflight(delay = 250) {
  clearTimeout(state.preflightTimer);
  state.preflightTimer = setTimeout(() => runPreflight().catch((error) => renderPreflightError(error)), delay);
}

async function runPreflight() {
  const payload = payloadFromForm();
  if (!payload.model_file || !payload.video_group) {
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
    </div>
    ${renderMessages("errors", result.errors || [])}
    ${renderMessages("warnings", result.warnings || [])}
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

function renderMessages(kind, items) {
  if (!items.length) return "";
  const cls = kind === "errors" ? "message error" : "message warn";
  return `<div class="${cls}">${items.map((item) => `<p><strong>${escapeHtml(item.title || "提示")}</strong>：${escapeHtml(item.message || item)}</p>`).join("")}</div>`;
}

function table(rows, columns) {
  if (!rows.length) return "<p class=\"muted\">暂无数据。</p>";
  return `
    <table>
      <thead><tr>${columns.map((col) => `<th>${col.label}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((row) => `<tr>${columns.map((col) => `<td>${col.render(row)}</td>`).join("")}</tr>`).join("")}</tbody>
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

function renderRuns() {
  $("runs-table").innerHTML = table(state.runs, [
    { label: "Run", render: (run) => `<button class="link-button" data-run-id="${run.id}" type="button">#${run.id}</button>` },
    { label: "状态", render: (run) => statusBadge(run.status) },
    { label: "模型", render: (run) => escapeHtml(run.metadata?.model_file || run.model_name || "-") },
    { label: "视频集", render: (run) => escapeHtml(run.metadata?.video_group || run.dataset_name || "-") },
    { label: "视频", render: (run) => escapeHtml((run.metadata?.selected_videos || []).length || "-") },
    { label: "进度", render: (run) => `${run.progress_current}/${run.progress_total || "?"}` },
    { label: "输出", render: (run) => escapeHtml(run.metadata?.output_dir || run.result?.output_dir || "-") },
  ]);
}

async function selectRun(runId, options = {}) {
  const previousRunId = state.selectedRun?.id;
  state.selectedRun = await api(`/api/runs/${runId}`);
  if (previousRunId !== state.selectedRun.id) {
    state.sampleDetails = {};
    state.sampleDetailLoading = {};
    state.runVideoTimelines = {};
  }
  [state.runVideosPage, state.metricSummary] = await Promise.all([
    api(`/api/runs/${runId}/videos?page_size=50`),
    api(`/api/runs/${runId}/metric-summary`),
  ]);
  const firstVideo = state.selectedVideoByRun[runId] || state.runVideosPage?.videos?.[0]?.video_name;
  if (firstVideo) {
    state.selectedVideoByRun[runId] = firstVideo;
    await loadRunVideoTimeline(runId, firstVideo);
  }
  renderRunDetail();
  if (!options.quiet) switchView("runs");
}

function renderRunDetail() {
  const run = state.selectedRun;
  if (!run) {
    $("run-detail").innerHTML = "<p class=\"muted\">选择一个 Run 查看结果。</p>";
    return;
  }
  const videos = state.runVideosPage?.videos || [];
  const selectedVideoName = state.selectedVideoByRun[run.id] || videos[0]?.video_name;
  if (selectedVideoName) state.selectedVideoByRun[run.id] = selectedVideoName;
  const videoSummary = videos.find((item) => item.video_name === selectedVideoName) || videos[0] || null;
  const video = videoSummary ? state.runVideoTimelines[`${run.id}:${videoSummary.video_name}`] : null;
  $("run-detail").innerHTML = `
    <div class="panel-head">
      <div>
        <h2>#${run.id} ${escapeHtml(run.name)}</h2>
        <p class="muted">${escapeHtml(run.metadata?.model_file || run.model_name)} / ${escapeHtml(run.metadata?.video_group || run.dataset_name)}</p>
      </div>
      <div class="actions">
        ${statusBadge(run.status)}
        <button class="secondary" data-cancel-run="${run.id}" ${TERMINAL.has(run.status) ? "disabled" : ""} type="button">取消</button>
        <button class="secondary" data-retry-run="${run.id}" type="button">重试</button>
      </div>
    </div>
    <div class="summary-grid">
      <div><span>进度</span><strong>${run.progress_current}/${run.progress_total || "?"}</strong></div>
      <div><span>推理阶段</span><strong>${renderInferencePhase(run)}</strong></div>
      <div><span>评测阶段</span><strong>${renderMetricPhase(run)}</strong></div>
      <div><span>输出目录</span><strong>${escapeHtml(run.metadata?.output_dir || run.result?.output_dir || "-")}</strong></div>
      <div><span>推理 FPS</span><strong>${formatNumber(run.result?.model_fps)}</strong></div>
      <div><span>产物</span><strong>${escapeHtml(run.artifact_summary?.total || 0)}</strong></div>
    </div>
    ${renderShardJobs(run)}
    ${renderRunError(run)}
    <div class="run-workspace">
      <aside class="video-tabs">
        <h3>视频</h3>
        ${videos.map((item) => `
          <button class="video-tab ${item.video_name === selectedVideoName ? "active" : ""}" data-run-video="${escapeHtml(item.video_name)}" type="button">
            <strong>${escapeHtml(item.video_file || item.video_name)}</strong>
            <span>${escapeHtml(item.sample_count || 0)} samples</span>
          </button>
        `).join("") || "<p class=\"muted\">推理完成后这里会显示视频。</p>"}
      </aside>
      <section class="sample-viewer">
        ${video ? renderVideoTimeline(video) : (videoSummary ? "<p class=\"muted\">正在加载这个视频的时间轴...</p>" : "<p class=\"muted\">暂无可查看结果。</p>")}
      </section>
    </div>
  `;
}

async function loadRunVideoTimeline(runId, videoName, options = {}) {
  const key = `${runId}:${videoName}`;
  const metric = options.metric || state.selectedMetricByRun[runId] || "";
  const windowStart = Number(options.windowStart ?? 0);
  const payload = await api(`/api/runs/${runId}/videos/${encodeURIComponent(videoName)}/timeline?bucket_count=160&window_start=${windowStart}&window_size=300${metric ? `&metric=${encodeURIComponent(metric)}` : ""}`);
  state.runVideoTimelines[key] = payload;
  return payload;
}

function renderInferencePhase(run) {
  if (run.status === "queued") return "等待推理";
  if (run.status === "running") return "推理中";
  if (run.status === "failed") return "失败";
  if (run.status === "canceled" || run.status === "cancel_requested") return "已取消";
  return run.inference_job_id ? "已完成" : "未开始";
}

function renderShardJobs(run) {
  const jobs = (run.jobs || []).filter((job) => job.role === "inference");
  if (jobs.length <= 1) return "";
  return `
    <section class="shard-panel">
      <h3>多卡分片</h3>
      <div class="table compact-table">${table(jobs, [
        { label: "Shard", render: (job) => `#${escapeHtml(job.shard_index)}` },
        { label: "设备", render: (job) => escapeHtml(job.device || job.payload?.device || "-") },
        { label: "状态", render: (job) => statusBadge(job.status) },
        { label: "进度", render: (job) => `${escapeHtml(job.progress_current || 0)}/${escapeHtml(job.progress_total || 0)}` },
        { label: "样本", render: (job) => escapeHtml((job.payload?.sample_ids || []).length || "-") },
        { label: "错误", render: (job) => escapeHtml(job.error?.message || "-") },
      ])}</div>
    </section>
  `;
}

function renderMetricPhase(run) {
  const metrics = run.metrics || [];
  if (!metrics.length) return "未选择指标";
  if (run.status === "metric_queued") return "评测排队";
  if (run.status === "metric_running") return "评测中";
  if (run.metric_job_id) return "已完成";
  return "等待推理完成";
}

function renderRunError(run) {
  if (!run.error || !Object.keys(run.error).length) return "";
  return `<div class="message error"><p><strong>${escapeHtml(run.error.type || "错误")}</strong>：${escapeHtml(run.error.message || JSON.stringify(run.error))}</p></div>`;
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

function renderVideoPlayer(label, artifactId) {
  if (!artifactId) return "";
  const url = `/api/files/${artifactId}`;
  return `<a class="small-video-link" href="${url}" target="_blank" rel="noreferrer">${escapeHtml(label)} 视频</a>`;
}

function renderMetricToolbar(video, metricName) {
  const names = metricNamesForVideo(video);
  if (!names.length) {
    return `<div class="metric-toolbar"><span class="muted">这个 Run 没有逐帧指标。</span>${renderVideoMetricSummary(video)}</div>`;
  }
  return `
    <div class="metric-toolbar">
      <label>
        <span>指标曲线</span>
        <select data-metric-select="${escapeHtml(video.video_name)}">
          ${names.map((name) => `<option value="${escapeHtml(name)}" ${name === metricName ? "selected" : ""}>${escapeHtml(name)}</option>`).join("")}
        </select>
      </label>
      ${renderMetricSummaryPills(video, metricName)}
    </div>
  `;
}

function renderMetricSummaryPills(video, metricName) {
  const summary = video.metric_summary?.[metricName] || state.metricSummary?.metrics?.[metricName];
  if (!summary) return renderVideoMetricSummary(video);
  const reason = (summary.reasons || [])[0];
  return `
    <div class="metric-summary">
      <span>completed ${escapeHtml(summary.completed || 0)}</span>
      <span>unavailable ${escapeHtml(summary.unavailable || 0)}</span>
      <span>failed ${escapeHtml(summary.failed || 0)}</span>
      <span>skipped ${escapeHtml(summary.skipped || 0)}</span>
      <span>mean ${formatNumber(summary.mean)}</span>
      ${reason ? `<span title="${escapeHtml(reason)}">原因 ${escapeHtml(reason)}</span>` : ""}
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
  const firstCompleted = names.find((name) => (video.metric_summary?.[name]?.completed || 0) > 0);
  const selected = firstCompleted || names[0];
  state.selectedMetricByRun[state.selectedRun.id] = selected;
  return selected;
}

function renderMetricChart(video, selectedIndex, metricName) {
  const samples = video.samples || [];
  if (!samples.length) return "";
  if (!metricName) {
    return `
      <div class="chart empty-chart">
        <p class="muted">未选择逐帧指标。可以用滑杆按帧序查看样本。</p>
        ${renderVideoMetricSummary(video)}
      </div>
    `;
  }
  const values = samples.map((sample) => {
    const metric = sample.metrics?.[metricName];
    return metric?.status === "completed" && metric.value !== null ? Number(metric.value) : null;
  });
  const valid = values.filter((value) => value !== null && Number.isFinite(value));
  if (!valid.length) {
    const statuses = countMetricStatuses(samples, metricName);
    return `
      <div class="chart empty-chart" data-chart-video="${escapeHtml(video.video_name)}">
        <div class="chart-head">
          <strong>${escapeHtml(metricName)}</strong>
          <span class="muted">没有 completed 点，可点击状态点定位样本</span>
        </div>
        ${renderStatusStrip(samples, metricName, selectedIndex)}
        <p class="muted">completed ${statuses.completed || 0} / unavailable ${statuses.unavailable || 0} / failed ${statuses.failed || 0} / skipped ${statuses.skipped || 0}</p>
      </div>
    `;
  }
  const min = Math.min(...valid);
  const max = Math.max(...valid);
  const points = values.map((value, index) => {
    if (value === null) return null;
    const x = samples.length <= 1 ? 0 : (index / (samples.length - 1)) * 100;
    const normalized = value === null || max === min ? 0.5 : (value - min) / (max - min);
    const y = 34 - normalized * 28;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).filter(Boolean).join(" ");
  const markerX = samples.length <= 1 ? 0 : (selectedIndex / (samples.length - 1)) * 100;
  return `
    <div class="chart" data-chart-video="${escapeHtml(video.video_name)}">
      <div class="chart-head">
        <strong>${escapeHtml(metricName)}</strong>
        <span class="muted">点击曲线定位帧</span>
      </div>
      <svg viewBox="0 0 100 40" preserveAspectRatio="none" role="img">
        <polyline points="${points}" fill="none" stroke="currentColor" stroke-width="1.6"></polyline>
        <line x1="${markerX.toFixed(2)}" x2="${markerX.toFixed(2)}" y1="3" y2="37"></line>
        ${renderMetricPoints(samples, metricName, min, max)}
      </svg>
    </div>
  `;
}

function renderMetricPoints(samples, metricName, min, max) {
  return samples.map((sample, index) => {
    const metric = sample.metrics?.[metricName];
    const x = samples.length <= 1 ? 0 : (index / (samples.length - 1)) * 100;
    const status = metric?.status || "missing";
    const value = metric?.value;
    const normalized = status === "completed" && value !== null && max !== min ? (Number(value) - min) / (max - min) : 0.5;
    const y = status === "completed" ? 34 - normalized * 28 : 36;
    return `<circle class="metric-point ${escapeHtml(status)}" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="1.4"><title>${escapeHtml(status)} ${value ?? metricReason(metric)}</title></circle>`;
  }).join("");
}

function renderStatusStrip(samples, metricName, selectedIndex) {
  return `
    <div class="status-strip">
      ${samples.map((sample, index) => {
        const metric = sample.metrics?.[metricName];
        const status = metric?.status || "missing";
        return `<button class="status-dot ${escapeHtml(status)} ${index === selectedIndex ? "active" : ""}" data-sample-jump="${index}" title="${escapeHtml(status)} ${escapeHtml(metricReason(metric))}" type="button"></button>`;
      }).join("")}
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

function metricReason(metric) {
  if (!metric) return "";
  return metric.details?.reason || metric.details?.type || "";
}

function renderVideoMetricSummary(video) {
  const entries = Object.entries(video.video_metrics || {});
  if (!entries.length) return "";
  return `<div class="metric-summary">${entries.map(([name, metric]) => `<span>${escapeHtml(name)}: ${escapeHtml(metric.status)} ${metric.value === null || metric.value === undefined ? "" : formatNumber(metric.value)}</span>`).join("")}</div>`;
}

function renderWorstSamples(video, metricName) {
  const rows = metricName ? (video.worst_samples?.[metricName] || []) : [];
  if (!metricName || !rows.length) return "";
  return `
    <section class="worst-samples">
      <div class="chart-head">
        <strong>最差样本</strong>
        <span class="muted">按 ${escapeHtml(metricName)} 排序</span>
      </div>
      <div class="worst-list">
        ${rows.map((row) => `
          <button class="worst-item" data-sample-id="${escapeHtml(row.sample_id)}" data-sample-video="${escapeHtml(video.video_name)}" data-sample-frame="${escapeHtml(row.frame_index)}" type="button">
            <span>frame ${escapeHtml(row.frame_index)}</span>
            <strong>${formatNumber(row.value)}</strong>
            <small>${row.timestamp === null || row.timestamp === undefined ? "-" : `${formatNumber(row.timestamp)}s`}</small>
          </button>
        `).join("")}
      </div>
    </section>
  `;
}

function renderSamplePreview(sample) {
  const detail = sampleDetail(sample.sample_id);
  if (!detail) {
    loadSampleDetail(sample.sample_id);
  }
  const payload = detail ? { ...sample, ...detail } : sample;
  return `
    <div class="sample-meta">
      <strong>${escapeHtml(payload.sample_name)}</strong>
      <span>frame ${escapeHtml(payload.frame_index)}</span>
      <span>${payload.timestamp === null || payload.timestamp === undefined ? "-" : `${formatNumber(payload.timestamp)}s`}</span>
      ${renderSampleMetrics(payload)}
    </div>
    ${detail ? "" : "<p class=\"muted sample-loading\">正在按需加载这一帧的产物...</p>"}
    <div class="preview-grid">
      ${CORE_PREVIEW.map(([kind, label]) => renderPreviewSlot(payload, kind, label)).join("")}
    </div>
    ${renderExtraArtifacts(payload)}
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
    renderRunDetail();
  } catch (error) {
    state.sampleDetails[key] = { sample_id: sampleId, artifacts: {}, extra_artifacts: [], sample_files: {}, load_error: error.message };
    renderRunDetail();
  } finally {
    delete state.sampleDetailLoading[key];
  }
}

function renderSampleMetrics(sample) {
  const entries = Object.entries(sample.metrics || {});
  if (!entries.length) return "";
  return `<span>${entries.map(([name, metric]) => `${escapeHtml(name)}=${metric.value === null || metric.value === undefined ? `${escapeHtml(metric.status)} ${escapeHtml(metricReason(metric))}` : formatNumber(metric.value)}`).join(" / ")}</span>`;
}

function renderPreviewSlot(sample, kind, label) {
  let url = null;
  if (kind === "gt") {
    url = sample.sample_files?.gt || null;
  } else if (sample.artifacts?.[kind]) {
    url = `/api/files/${sample.artifacts[kind]}`;
  }
  if (!url) {
    return `<div class="preview-slot"><span>${escapeHtml(label)}</span><p class="muted">暂无</p></div>`;
  }
  return `
    <a class="preview-slot" href="${url}" target="_blank" rel="noreferrer">
      <span>${escapeHtml(label)}</span>
      <img src="${url}" alt="${escapeHtml(label)}" loading="lazy">
    </a>
  `;
}

function renderExtraArtifacts(sample) {
  const extras = sample.extra_artifacts || [];
  if (!extras.length) return "";
  return `
    <details class="extra-artifacts">
      <summary>附加可视化</summary>
      <div class="preview-grid extra-grid">
        ${extras.map((item) => `
          <a class="preview-slot" href="/api/files/${item.id}" target="_blank" rel="noreferrer">
            <span>${escapeHtml(item.kind)}</span>
            <img src="/api/files/${item.id}" alt="${escapeHtml(item.kind)}" loading="lazy">
          </a>
        `).join("")}
      </div>
    </details>
  `;
}

function setSampleIndex(videoName, index) {
  if (!state.selectedRun) return;
  const video = currentTimelineVideo(videoName);
  if (!video) return;
  const max = Math.max(0, (video.samples || []).length - 1);
  state.selectedSampleByVideo[`${state.selectedRun.id}:${videoName}`] = Math.max(0, Math.min(max, index));
  renderRunDetail();
}

function setSampleById(sampleId) {
  if (!state.selectedRun) return;
  for (const video of Object.values(state.runVideoTimelines || {})) {
    const index = (video.samples || []).findIndex((sample) => Number(sample.sample_id) === Number(sampleId));
    if (index >= 0) {
      state.selectedVideoByRun[state.selectedRun.id] = video.video_name;
      state.selectedSampleByVideo[`${state.selectedRun.id}:${video.video_name}`] = index;
      renderRunDetail();
      return;
    }
  }
}

async function setSampleByFrame(videoName, frameIndex) {
  if (!state.selectedRun || !videoName || Number.isNaN(frameIndex)) return;
  const windowStart = Math.max(0, frameIndex - 150);
  const video = await loadRunVideoTimeline(state.selectedRun.id, videoName, { windowStart });
  const index = (video.samples || []).findIndex((sample) => Number(sample.frame_index) === Number(frameIndex));
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

document.querySelectorAll(".nav-item").forEach((item) => item.addEventListener("click", () => switchView(item.dataset.view)));
$("infer-form").addEventListener("submit", (event) => startRun(event).catch((error) => toast(error.message)));
$("refresh").addEventListener("click", () => refreshRunsOnly().catch((error) => toast(error.message)));
$("refresh-files").addEventListener("click", () => refreshCatalog().then(() => toast("文件列表已刷新")).catch((error) => toast(error.message)));
$("infer-form").addEventListener("change", (event) => {
  renderCustomSizeVisibility();
  if (event.target.name === "model_file") {
    renderCheckpointOptions();
  }
  if (event.target.name === "execution_mode") {
    renderDeviceOptions();
  }
  if (event.target.name === "video_group") {
    ensureVideoSelection(event.target.value);
    loadVideoGroupPage(event.target.value, 1).catch((error) => toast(error.message));
  }
  schedulePreflight();
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
  const range = event.target.closest("[data-sample-range]");
  if (range) {
    setSampleIndex(range.dataset.sampleRange, Number(range.value));
    return;
  }
  const metricSelect = event.target.closest("[data-metric-select]");
  if (metricSelect && state.selectedRun) {
    state.selectedMetricByRun[state.selectedRun.id] = metricSelect.value;
    const videoName = metricSelect.dataset.metricSelect;
    loadRunVideoTimeline(state.selectedRun.id, videoName, { metric: metricSelect.value }).then(renderRunDetail).catch((error) => toast(error.message));
  }
  const videoSort = event.target.closest("[data-video-sort]");
  if (videoSort) {
    state.videoSortByGroup[videoSort.dataset.videoSort] = videoSort.value;
    loadVideoGroupPage(videoSort.dataset.videoSort, 1).then(() => schedulePreflight(0)).catch((error) => toast(error.message));
  }
  const cudaDevice = event.target.closest("[data-cuda-device]");
  if (cudaDevice) {
    if (cudaDevice.checked) state.selectedCudaDevices.add(cudaDevice.dataset.cudaDevice);
    else state.selectedCudaDevices.delete(cudaDevice.dataset.cudaDevice);
    renderDeviceOptions();
    schedulePreflight(0);
  }
});

document.addEventListener("input", (event) => {
  const search = event.target.closest("[data-video-search]");
  if (search) {
    state.videoSearchByGroup[search.dataset.videoSearch] = search.value;
    const groupName = search.dataset.videoSearch;
    clearTimeout(state.videoSearchTimer);
    state.videoSearchTimer = setTimeout(() => {
      loadVideoGroupPage(groupName, 1).then(() => schedulePreflight(0)).catch((error) => toast(error.message));
    }, 200);
  }
});

document.addEventListener("click", async (event) => {
  const videoSelect = event.target.closest("[data-video-select]");
  if (videoSelect) {
    const group = currentGroup();
    if (!group) return;
    const selected = ensureVideoSelection(group.name);
    const page = state.videoPages[group.name];
    const names = page?.all_video_names || (group.videos || []).map((video) => video.name);
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

  const videoPage = event.target.closest("[data-video-page]");
  if (videoPage) {
    const group = currentGroup();
    if (!group) return;
    await loadVideoGroupPage(group.name, Number(videoPage.dataset.videoPage));
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
  const worstItem = event.target.closest("[data-sample-id]");
  if (worstItem) {
    await setSampleByFrame(worstItem.dataset.sampleVideo, Number(worstItem.dataset.sampleFrame));
    return;
  }
  const chart = event.target.closest("[data-chart-video]");
  if (chart) {
    const video = currentTimelineVideo(chart.dataset.chartVideo);
    if (!video || !video.samples?.length) return;
    const rect = chart.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    setSampleIndex(chart.dataset.chartVideo, Math.round(ratio * (video.samples.length - 1)));
  }
});

function currentTimelineVideo(videoName) {
  if (!state.selectedRun) return null;
  return state.runVideoTimelines[`${state.selectedRun.id}:${videoName}`] || null;
}

refreshCatalog().catch((error) => toast(error.message));
setInterval(() => refreshRunsOnly().catch(() => {}), 2000);
