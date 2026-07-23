"use strict";

// Domain functions intentionally share app.js's classic-script global
// environment so state, request primitives, and caches remain singletons.

function compareItemId(row) {
  return Number(row?.item_id || row?.id || 0);
}

function compareMemberId(row) {
  return Number(row?.member_id || row?.id || 0);
}

function selectedCompareGt() {
  return (state.compareItems || []).find((row) => compareItemId(row) === Number(state.selectedCompareItemId))
    || (compareItemId(state.selectedCompareItemSnapshot) === Number(state.selectedCompareItemId)
      ? state.selectedCompareItemSnapshot
      : null);
}

function selectedComparePredRows() {
  return Array.from(state.selectedComparePredMembers)
    .map((memberId) => state.comparePredByMember[Number(memberId)])
    .filter(Boolean)
    .map((row) => ({ ...row, track_label: compareTrackLabel(row) }));
}

function compareTrackLabel(row) {
  return row.method_label
    || row.track_label
    || row.display_name
    || row.run_name
    || `Run #${row.run_id || row.producer_run_id || "-"}`;
}

function compareTemporalSummary(row) {
  const mapping = row?.temporal_mapping && typeof row.temporal_mapping === "object"
    ? row.temporal_mapping
    : {};
  const indices = Array.isArray(row?.source_frame_indices)
    ? row.source_frame_indices
    : mapping.source_frame_indices;
  if (Array.isArray(indices)) {
    if (!indices.length) return "indexed temporal mapping (0 frames)";
    const first = Number(indices[0]);
    const last = Number(indices[indices.length - 1]);
    return `indexed temporal mapping: ${indices.length} frames (${first}–${last})`;
  }
  const timestamps = Array.isArray(mapping.timestamps)
    ? mapping.timestamps
    : (Array.isArray(mapping.source_timestamps) ? mapping.source_timestamps : null);
  if (timestamps?.length) return `exact temporal alignment · ${timestamps.length} recorded timestamps`;
  return "exact temporal alignment";
}

function compareSlotLabel(index) {
  return index < 0 ? "available Pred" : `Pred ${String.fromCharCode(65 + index)}`;
}

function compareSpatialTarget(predRows = selectedComparePredRows()) {
  const sized = predRows.filter((row) => Number(row.width) > 0 && Number(row.height) > 0);
  if (!sized.length) return null;
  return [...sized].sort((left, right) => {
    const area = Number(left.width) * Number(left.height) - Number(right.width) * Number(right.height);
    if (area) return area;
    const edge = Math.max(Number(left.width), Number(left.height)) - Math.max(Number(right.width), Number(right.height));
    if (edge) return edge;
    return Number(left.width) - Number(right.width) || Number(left.height) - Number(right.height);
  })[0];
}

function ensureCompareSelection() {
  const selectedOnPage = (state.compareItems || []).find(
    (row) => compareItemId(row) === Number(state.selectedCompareItemId),
  );
  if (selectedOnPage) state.selectedCompareItemSnapshot = selectedOnPage;
  const available = new Set();
  for (const row of state.comparePredictions || []) {
    const memberId = compareMemberId(row);
    if (!memberId) continue;
    available.add(memberId);
    state.comparePredByMember[memberId] = row;
  }
  for (const memberId of Array.from(state.selectedComparePredMembers)) {
    if (!available.has(Number(memberId))) state.selectedComparePredMembers.delete(memberId);
  }
}

function compareSourcePager(meta) {
  if (!meta || Number(meta.total_pages || 1) <= 1) return "";
  return `
    <div class="pager compact-pager">
      <button class="secondary" data-compare-page data-page="${Number(meta.page || 1) - 1}" ${Number(meta.page || 1) <= 1 ? "disabled" : ""} type="button">上一页</button>
      <span class="muted">第 ${escapeHtml(meta.page || 1)} / ${escapeHtml(meta.total_pages || 1)} 页，${escapeHtml(meta.filtered_count || 0)} 条</span>
      <button class="secondary" data-compare-page data-page="${Number(meta.page || 1) + 1}" ${Number(meta.page || 1) >= Number(meta.total_pages || 1) ? "disabled" : ""} type="button">下一页</button>
    </div>
  `;
}

async function loadCompareSources(options = {}) {
  const generation = ++state.compareSourceRequestGeneration;
  if (options.page !== undefined || options.gtPage !== undefined) {
    state.compareItemPage = Math.max(1, Number(options.page || options.gtPage || 1));
  }
  if (options.reset) {
    state.selectedCompareGroupId = "";
    state.selectedCompareItemId = null;
    state.selectedCompareItemSnapshot = null;
    state.selectedComparePredMembers.clear();
  }
  const groupPayload = await api("/api/media/item-groups?role=gt");
  if (generation !== state.compareSourceRequestGeneration) return;
  state.compareItemGroups = groupPayload.groups || groupPayload.item_groups || [];
  const groupIds = new Set(state.compareItemGroups.map((row) => String(row.group_id || row.collection_id || row.id)));
  if (!groupIds.has(String(state.selectedCompareGroupId))) {
    state.selectedCompareGroupId = String(state.compareItemGroups[0]?.group_id || state.compareItemGroups[0]?.collection_id || state.compareItemGroups[0]?.id || "");
    state.selectedCompareItemId = null;
    state.selectedCompareItemSnapshot = null;
    state.selectedComparePredMembers.clear();
  }
  if (!state.selectedCompareGroupId) {
    state.compareItems = [];
    state.comparePredictions = [];
    state.compareItemsMeta = null;
    state.compareSourcesLoaded = true;
    renderCompareSelection();
    return;
  }
  const itemsPayload = await api(`/api/media/items?group_id=${encodeURIComponent(state.selectedCompareGroupId)}&q=${encodeURIComponent(state.compareItemQuery)}&page=${state.compareItemPage}&page_size=50`);
  if (generation !== state.compareSourceRequestGeneration) return;
  state.compareItems = itemsPayload.items || [];
  state.compareItemsMeta = {
    ...itemsPayload,
    total_pages: itemsPayload.total_pages || itemsPayload.page_count,
    filtered_count: itemsPayload.filtered_count ?? itemsPayload.total ?? state.compareItems.length,
  };
  ensureCompareSelection();
  if (state.selectedCompareItemId) {
    const predPayload = await api(`/api/media/items/${Number(state.selectedCompareItemId)}/predictions`);
    if (generation !== state.compareSourceRequestGeneration) return;
    state.comparePredictions = predPayload.predictions || predPayload.members || [];
  } else {
    state.comparePredictions = [];
  }
  state.compareSourcesLoaded = true;
  ensureCompareSelection();
  renderCompareSelection();
}

function comparePayloadFromForm() {
  const gt = selectedCompareGt();
  const predRows = selectedComparePredRows();
  if (!gt || predRows.length < 1 || predRows.length > 2) {
    return null;
  }
  return {
    run_type: "video_compare",
    media_item_id: compareItemId(gt),
    pred_member_ids: predRows.map(compareMemberId),
    spatial_policy: {
      mode: "smallest_pred",
      filter: "lanczos",
      allow_known_aspect_stretch: true,
      allow_external_aspect_stretch: Boolean(
        $("compare-form").elements.allow_external_aspect_stretch?.checked,
      ),
    },
    metrics: compareSelectedMetrics(),
  };
}

function compareSelectedMetrics() {
  return Array.from(document.querySelectorAll("#compare-metrics-options input[name='compare_metrics']:checked")).map((item) => item.value);
}

function compareItemGt(row) {
  return row?.canonical_gt || row?.gt || row?.canonical_asset || row || {};
}

function compareItemTitle(row) {
  return row?.display_name || row?.item_key || row?.video_name || `Media Item #${compareItemId(row)}`;
}

function renderCompareGtCards(items) {
  if (!items.length) return "<p class=\"muted\">这个文件夹中没有匹配的 GT Media Item。</p>";
  return items.map((row) => {
    const itemId = compareItemId(row);
    const active = Number(state.selectedCompareItemId) === itemId;
    const gt = compareItemGt(row);
    return `
      <label class="source-card${active ? " selected" : ""}">
        <input type="radio" name="compare_gt_pick" data-compare-item="${itemId}" ${active ? "checked" : ""}>
        <span class="source-card-body">
          <span class="source-card-title">${escapeHtml(compareItemTitle(row))} <span class="compat-badge">canonical GT</span></span>
          <span class="source-card-meta">
            <span>${escapeHtml(gt.frame_count || row.frame_count || 0)} 帧</span>
            <span>${escapeHtml(gt.width || row.width || "-")}×${escapeHtml(gt.height || row.height || "-")}</span>
            <span>${formatNumber(gt.fps ?? row.fps)} fps</span>
          </span>
        </span>
      </label>
    `;
  }).join("");
}

function renderComparePredCards(predRows, selectedPreds) {
  if (!state.selectedCompareItemId) return "<p class=\"muted\">先选择一个 GT 视频。</p>";
  if (!predRows.length) return "<p class=\"muted\">这个 GT 还没有可复用的模型 Pred 或已绑定的外部 Pred。</p>";
  const selectedMemberIds = Array.from(selectedPreds).map(Number);
  return predRows.map((row) => {
    const memberId = compareMemberId(row);
    const active = selectedPreds.has(memberId);
    const atLimit = selectedPreds.size >= 2 && !active;
    const external = String(row.member_role || row.producer_kind || "").includes("external");
    const temporal = row.temporal_mapping_summary || compareTemporalSummary(row);
    const slot = compareSlotLabel(selectedMemberIds.indexOf(memberId));
    return `
      <div class="source-card${active ? " selected" : ""}">
        <label class="source-card-pick">
          <input type="checkbox" data-compare-pred="${memberId}" ${active ? "checked" : ""} ${atLimit ? "disabled" : ""}>
          <span class="source-card-body">
            <span class="source-card-title">${row.run_id || row.producer_run_id ? `#${escapeHtml(row.run_id || row.producer_run_id)} ` : ""}${escapeHtml(compareTrackLabel(row))} <span class="compat-badge">${external ? "External" : "model Pred"}</span>${active ? ` <span class="compat-badge compat-ok">${escapeHtml(slot)}</span>` : ""}</span>
            <span class="source-card-meta">
              <span>${escapeHtml(row.frame_count || 0)} 帧</span>
              <span>${escapeHtml(row.width || "-")}×${escapeHtml(row.height || "-")}</span>
              <span>${formatNumber(row.fps)} fps</span>
            </span>
            <span class="source-card-warn">${escapeHtml(temporal)}</span>
          </span>
        </label>
      </div>
    `;
  }).join("");
}

function compareResizeKind(width, height, targetWidth, targetHeight) {
  if (!width || !height || !targetWidth || !targetHeight) return "unknown";
  if (width === targetWidth && height === targetHeight) return "none";
  const x = targetWidth / width;
  const y = targetHeight / height;
  if (x >= 1 && y >= 1) return "upscale";
  if (x <= 1 && y <= 1) return "downscale";
  return "mixed";
}

function renderCompareAlignmentSummary(item, preds) {
  if (!item || !preds.length) return "<p class=\"muted\">选择 Pred 后显示空间规范化计划。时间身份、帧映射、FPS 与时间戳仍由服务端严格验证。</p>";
  const target = compareSpatialTarget(preds);
  if (!target) return "<p class=\"muted\">所选 Pred 缺少尺寸元数据；预检查将给出具体原因。</p>";
  const targetWidth = Number(target.width);
  const targetHeight = Number(target.height);
  const gt = compareItemGt(item);
  const rows = [
    { label: "GT", width: Number(gt.width || item.width), height: Number(gt.height || item.height) },
    ...preds.map((row, index) => ({ label: `Pred ${String.fromCharCode(65 + index)} · ${compareTrackLabel(row)}`, width: Number(row.width), height: Number(row.height) })),
  ];
  return `
    <div class="alignment-plan-summary">
      <div class="panel-head"><div><h3>预期空间规范化</h3><p class="muted">目标 ${targetWidth}×${targetHeight} · LANCZOS；原始文件不会被覆盖。</p></div><span class="compat-badge compat-ok">时间严格</span></div>
      <div class="alignment-plan-grid">${rows.map((row) => {
        const scaleX = row.width ? targetWidth / row.width : 0;
        const scaleY = row.height ? targetHeight / row.height : 0;
        const aspectChanged = scaleX && scaleY && Math.abs(scaleX - scaleY) > 1e-6;
        return `<div><strong>${escapeHtml(row.label)}</strong><span>${row.width || "-"}×${row.height || "-"} → ${targetWidth}×${targetHeight}</span><small>${escapeHtml(compareResizeKind(row.width, row.height, targetWidth, targetHeight))}${scaleX ? ` · ${scaleX.toFixed(4)}×${scaleY.toFixed(4)}` : ""}${aspectChanged ? " · 宽高比变化（已记录）" : ""}</small></div>`;
      }).join("")}</div>
    </div>`;
}

function renderCompareSelection() {
  ensureCompareSelection();
  if (!state.compareSourcesLoaded) {
    $("compare-selection").innerHTML = `
      <div class="panel-head">
        <div>
          <h2>对比来源</h2>
          <p class="muted">先选择 GT 文件夹，再选择其中一个 Media Item。</p>
        </div>
        <button class="secondary" data-refresh-compare-sources type="button">加载来源</button>
      </div>
      <div class="timeline-skeleton" aria-busy="true"><span></span><span></span><span></span></div>
    `;
    return;
  }
  const gtRows = state.compareItems || [];
  const predRows = state.comparePredictions || [];
  const selectedPreds = state.selectedComparePredMembers;
  const selectedRows = selectedComparePredRows();
  const groups = state.compareItemGroups || [];
  $("compare-selection").innerHTML = `
    <div class="panel-head">
      <div>
        <h2>对比来源</h2>
        <p class="muted">GT 使用精确 canonical Media Item 身份；Pred 只显示绑定到该 Item 的有效模型输出或外部 Pred。Compare Run 的输出不会再次出现。</p>
      </div>
      <div class="metric-summary">
        <span>${escapeHtml(groups.length)} 个文件夹</span>
        <span>${escapeHtml(state.compareItemsMeta?.filtered_count ?? gtRows.length)} 个 GT</span>
        <span>已选 ${escapeHtml(selectedPreds.size)} / 2 Pred</span>
      </div>
    </div>
    <div class="compare-source-grid">
      <section class="compare-source-col">
        <div class="compare-col-head">
          <h3>1. GT 文件夹与视频</h3>
        </div>
        <div class="source-tools">
          <label>
            <span>文件夹</span>
            <select data-compare-group>
              ${groups.map((row) => {
                const id = String(row.group_id || row.collection_id || row.id);
                return `<option value="${escapeHtml(id)}" ${id === String(state.selectedCompareGroupId) ? "selected" : ""}>${escapeHtml(row.name || row.display_name || row.collection_name || id)} (${Number(row.item_count || row.count || 0)})</option>`;
              }).join("")}
            </select>
          </label>
          <label>
            <span>筛选当前文件夹</span>
            <input data-compare-query="item" value="${escapeHtml(state.compareItemQuery)}" placeholder="视频文件名">
          </label>
        </div>
        <div class="source-card-list">${renderCompareGtCards(gtRows)}</div>
        ${compareSourcePager(state.compareItemsMeta)}
      </section>
      <section class="compare-source-col">
        <div class="compare-col-head">
          <h3>2. 对应 Pred（1–2 份）</h3>
        </div>
        <div class="source-card-list">${renderComparePredCards(predRows, selectedPreds)}</div>
      </section>
    </div>
    ${renderCompareAlignmentSummary(selectedCompareGt(), selectedRows)}
  `;
  renderCompareSubmissionState();
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
  if (!state.compareSubmitting && state.compareSubmitError) {
    state.compareSubmitError = "";
    state.compareSubmissionId = "";
    renderCompareSubmissionState();
  }
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

function alignmentTemporal(plan) {
  const temporal = plan?.temporal;
  return temporal && typeof temporal === "object" && !Array.isArray(temporal) ? temporal : {};
}

function alignmentTransformRows(plan) {
  const source = Array.isArray(plan?.transforms)
    ? plan.transforms
    : Object.entries(plan?.transforms || plan?.sources || {}).map(([label, row]) => ({ label, ...row }));
  return source.filter((row) => row && typeof row === "object");
}

function alignmentDirection(row) {
  return String(row?.direction || row?.resize_kind || row?.operation || "none");
}

function alignmentTemporalSummary(plan) {
  const temporal = alignmentTemporal(plan);
  const mode = temporal.mode || plan?.temporal_summary || plan?.temporal_status || "strict";
  const details = [];
  if (temporal.frame_count != null) details.push(`${Number(temporal.frame_count)} 帧`);
  if (temporal.mapping_count != null) {
    const range = temporal.mapping_first != null && temporal.mapping_last != null
      ? ` (${temporal.mapping_first}–${temporal.mapping_last})`
      : "";
    details.push(`映射 ${Number(temporal.mapping_count)}${range}`);
  }
  if (temporal.fps != null) details.push(`${Number(temporal.fps).toFixed(3)} fps`);
  if (typeof temporal.timestamps_verified === "boolean") {
    details.push(temporal.timestamps_verified ? "时间戳已核验" : "无可核验时间戳");
  }
  return { mode: String(mode), details };
}

function renderAlignmentPlan(plan) {
  if (!plan || !Object.keys(plan).length) return "";
  const target = plan.target || plan.target_size || {};
  const targetWidth = Number(plan.target_width || target.width || plan.width || 0);
  const targetHeight = Number(plan.target_height || target.height || plan.height || 0);
  const transforms = alignmentTransformRows(plan);
  const temporal = alignmentTemporalSummary(plan);
  return `
    <section class="alignment-report">
      <div class="panel-head"><div><h3>Alignment Plan</h3><p class="muted">时间映射严格校验；空间变换实际用于 Diff 与指标。</p></div>${plan.fingerprint ? `<code>${escapeHtml(String(plan.fingerprint).slice(0, 16))}</code>` : ""}</div>
      <div class="summary-grid">
        <div><span>目标尺寸</span><strong>${targetWidth || "-"}×${targetHeight || "-"}</strong></div>
        <div><span>空间策略</span><strong>${escapeHtml(plan.mode || plan.spatial_mode || "smallest_pred")}</strong></div>
        <div><span>插值</span><strong>${escapeHtml(plan.filter || plan.interpolation || "lanczos")}</strong></div>
        <div><span>时间</span><strong>${escapeHtml(temporal.mode)}</strong></div>
      </div>
      ${temporal.details.length ? `<p class="alignment-temporal-details">${escapeHtml(temporal.details.join(" · "))}</p>` : ""}
      ${transforms.length ? `<div class="alignment-plan-grid">${transforms.map((row) => {
        const original = row.original || row.source || {};
        const width = Number(row.original_width || original.width || row.width || 0);
        const height = Number(row.original_height || original.height || row.height || 0);
        const scaleX = row.scale_x == null ? (width && targetWidth ? targetWidth / width : null) : Number(row.scale_x);
        const scaleY = row.scale_y == null ? (height && targetHeight ? targetHeight / height : null) : Number(row.scale_y);
        return `<div><strong>${escapeHtml(row.label || row.slot || row.kind || "source")}</strong><span>${width || "-"}×${height || "-"} → ${targetWidth || "-"}×${targetHeight || "-"}</span><small>${escapeHtml(alignmentDirection(row))}${scaleX != null && scaleY != null ? ` · ${scaleX.toFixed(4)}×${scaleY.toFixed(4)}` : ""}${row.aspect_changed ? " · 宽高比变化" : ""}</small></div>`;
      }).join("")}</div>` : ""}
    </section>`;
}

function renderComparePreflight() {
  const result = state.comparePreflight;
  const start = $("start-compare");
  if (!result) {
    start.disabled = true;
    $("compare-preflight").innerHTML = state.compareSourcesLoaded
      ? "<p class=\"muted\">选好一个 GT Media Item 和一至两份对应 Pred 后会自动预检查。</p>"
      : "<p class=\"muted\">先加载对比来源。</p>";
    renderCompareSubmissionState();
    return;
  }
  start.disabled = !result.ok;
  const predictions = result.predictions || result.pred_members || result.distorted_tracks || [];
  const trackLabels = predictions.map((track) => track.method_label || track.track_label || track.label || track.name).filter(Boolean).join(", ");
  const plan = result.alignment_plan || result.alignment || {};
  const target = plan.target || plan.target_size || {};
  const temporal = alignmentTemporal(plan);
  $("compare-preflight").innerHTML = `
    <div class="panel-head">
      <h2>运行前预检查</h2>
      ${result.ok ? "<span class=\"ok-text\">通过</span>" : "<span class=\"bad-text\">未通过</span>"}
    </div>
    <div class="summary-grid">
      <div><span>模式</span><strong>video_compare</strong></div>
      <div><span>GT Item</span><strong>${escapeHtml(result.item?.display_name || selectedCompareGt()?.display_name || "-")}</strong></div>
      <div><span>Pred</span><strong>${escapeHtml(`${predictions.length || selectedComparePredRows().length} 个`)}</strong></div>
      <div><span>方法</span><strong>${escapeHtml(trackLabels || selectedComparePredRows().map(compareTrackLabel).join(", ") || "-")}</strong></div>
      <div><span>帧数</span><strong>${escapeHtml(temporal.frame_count ?? plan.frame_count ?? result.frame_count ?? "-")}</strong></div>
      <div><span>目标分辨率</span><strong>${escapeHtml(`${plan.target_width || target.width || plan.width || "-"}x${plan.target_height || target.height || plan.height || "-"}`)}</strong></div>
      <div><span>FPS</span><strong>${formatNumber(temporal.fps ?? plan.fps ?? result.fps)}</strong></div>
    </div>
    ${renderMessages("errors", result.errors || [])}
    ${renderMessages("warnings", result.warnings || [])}
    ${renderAlignmentPlan(plan)}
    ${renderPortableMetricHealthTable(result.metrics?.health || {})}
  `;
  renderCompareSubmissionState();
}

function renderCompareSubmissionState() {
  const form = $("compare-form");
  const start = $("start-compare");
  const handoff = $("create-adhoc-evaluation");
  const status = $("compare-submit-status");
  if (!form || !start || !handoff || !status) return;
  const labels = {
    preflight: "正在重新预检查…",
    creating: "正在创建对比 Run…",
    opening: "Run 已创建，正在打开…",
  };
  form.setAttribute("aria-busy", state.compareSubmitting ? "true" : "false");
  start.disabled = state.compareSubmitting || !state.comparePreflight?.ok;
  const handoffReady = Boolean(selectedCompareGt()) && selectedComparePredRows().length === 2;
  handoff.disabled = state.compareSubmitting || !handoffReady;
  handoff.title = handoffReady
    ? "将当前 GT Media Item 与两份 Pred 带入 Evaluation Studio"
    : "请选择一个 GT Media Item 和恰好两份 Pred";
  start.textContent = state.compareSubmitting ? (labels[state.compareSubmitPhase] || "正在创建…") : "开始对比";
  if (state.compareSubmitting) {
    status.hidden = false;
    status.className = "compare-submit-status message";
    status.textContent = `${labels[state.compareSubmitPhase] || "正在处理…"} 请勿重复点击。`;
  } else if (state.compareSubmitError) {
    status.hidden = false;
    status.className = "compare-submit-status message error";
    status.textContent = `对比任务创建失败：${state.compareSubmitError}`;
  } else {
    status.hidden = true;
    status.className = "compare-submit-status";
    status.textContent = "";
  }
}

async function startCompareRun(event) {
  event.preventDefault();
  if (compareCreationFlight.isLocked()) {
    toast("对比任务正在创建，请勿重复点击");
    return;
  }
  const payload = comparePayloadFromForm();
  if (!payload) {
    toast("请选择一个 GT Media Item 和一至两份对应 Pred");
    return;
  }
  if (!compareCreationFlight.tryLock()) {
    toast("对比任务正在创建，请勿重复点击");
    return;
  }
  state.compareSubmitting = true;
  state.compareSubmitPhase = "preflight";
  state.compareSubmitError = "";
  clearTimeout(state.comparePreflightTimer);
  renderCompareSubmissionState();
  try {
    await runComparePreflight({ force: true });
    if (!state.comparePreflight?.ok) {
      toast("预检查未通过");
      return;
    }
    state.compareSubmitPhase = "creating";
    renderCompareSubmissionState();
    state.compareSubmissionId = state.compareSubmissionId || Shared.createSubmissionId("compare");
    payload.submission_id = state.compareSubmissionId;
    const created = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.compareSubmissionId = "";
    state.compareSubmitPhase = "opening";
    renderCompareSubmissionState();
    toast(`Run #${created.run_id} 已开始`);
    switchView("runs");
    await refreshRunsOnly();
    await selectRun(created.run_id);
  } catch (error) {
    state.compareSubmitError = error.message || String(error);
    throw error;
  } finally {
    state.compareSubmitting = false;
    state.compareSubmitPhase = "";
    compareCreationFlight.release();
    renderCompareSubmissionState();
  }
}

async function createAdhocEvaluation() {
  const gt = selectedCompareGt();
  const preds = selectedComparePredRows();
  if (!gt || preds.length !== 2) {
    throw new Error("Campaign V2 需要一个 GT Media Item 和恰好两份对应 Pred");
  }
  switchView("evaluations");
  if (!window.VFIEvalStudio) throw new Error("Evaluation Studio 尚未加载");
  await window.VFIEvalStudio.prefillFromCompare({
    groupId: state.selectedCompareGroupId,
    item: gt,
    predictions: preds,
  });
  toast("已将 GT Media Item 与两份方法带入 Evaluation Studio");
}
