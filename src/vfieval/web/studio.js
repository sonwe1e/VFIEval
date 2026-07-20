(function () {
  const studioState = {
    campaigns: [],
    runOutputs: [],
    itemGroups: [],
    items: [],
    methods: [],
    selectedGroupId: "",
    selectedItemIds: new Set(),
    itemQuery: "",
    itemPage: 1,
    itemPageSize: 100,
    itemPageCount: 1,
    itemTotal: 0,
    preview: null,
    selectedCampaignKey: null,
    preparationPoll: null,
    preparationPollKey: null,
    preparationPollGeneration: 0,
    campaignRequestGeneration: 0,
    objectiveCurveRequestGeneration: 0,
    campaignDetail: null,
    objectiveCurveSelection: null,
    objectiveCurveCache: {},
    objectiveCurveLoadingKey: "",
    objectiveCurveInFlight: null,
    objectiveCurveErrors: {},
    itemRequestGeneration: 0,
    methodRequestGeneration: 0,
    previewGeneration: 0,
    itemQueryTimer: null,
    storagePreview: null,
    cleanupRequests: [],
  };

  const el = (id) => document.getElementById(id);

  async function request(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(data.error?.message || response.statusText);
      error.status = response.status;
      error.payload = data;
      throw error;
    }
    return data;
  }

  function safe(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function notify(message) {
    if (typeof window.toast === "function") window.toast(message);
  }

  function itemId(item) {
    return Number(item?.item_id || item?.id || 0);
  }

  function groupId(group) {
    return String(group?.group_id || group?.collection_id || group?.id || "");
  }

  function methodKind(method) {
    return String(method?.kind || method?.source_kind || "run") === "external" ? "external" : "run";
  }

  function methodKey(method) {
    if (!method) return "";
    return methodKind(method) === "run"
      ? `run:${Number(method.run_id || method.producer_run_id || 0)}`
      : `external:${String(method.method_key || method.id || "")}`;
  }

  function methodDescriptor(method) {
    if (method?.descriptor && typeof method.descriptor === "object") return method.descriptor;
    if (methodKind(method) === "run") return { kind: "run", run_id: Number(method.run_id) };
    return { kind: "external", method_key: String(method.method_key || method.id) };
  }

  function methodSourceMode(slot) {
    return el(`studio-method-${slot}-source`)?.value === "external" ? "external" : "run";
  }

  function selectedMethodRow(slot) {
    const value = el(`studio-method-${slot}`)?.value || "";
    return studioState.methods.find((row) => methodKey(row) === value) || null;
  }

  function selectedMethod(slot) {
    const row = selectedMethodRow(slot);
    return row ? methodDescriptor(row) : null;
  }

  function selectedItemIds() {
    return Array.from(studioState.selectedItemIds).map(Number).sort((left, right) => left - right);
  }

  function spatialPolicy() {
    return {
      mode: "smallest_pred",
      filter: "lanczos",
      allow_known_aspect_stretch: true,
      allow_external_aspect_stretch: Boolean(el("studio-allow-external-aspect-stretch")?.checked),
    };
  }

  function invalidateCoveragePreview() {
    studioState.previewGeneration += 1;
    studioState.preview = null;
    renderCoverage();
  }

  function renderItemGroupOptions() {
    const select = el("studio-item-group");
    if (!select) return;
    select.innerHTML = studioState.itemGroups.length
      ? studioState.itemGroups.map((group) => {
          const id = groupId(group);
          return `<option value="${safe(id)}" ${id === String(studioState.selectedGroupId) ? "selected" : ""}>${safe(group.name || group.display_name || group.slug || id)} (${Number(group.item_count || group.count || 0)})</option>`;
        }).join("")
      : "<option value=\"\">没有可用 GT 文件夹</option>";
  }

  function itemMetadata(item) {
    return [
      item.frame_count ? `${Number(item.frame_count)} 帧` : "",
      item.width && item.height ? `${Number(item.width)}×${Number(item.height)}` : "",
      item.fps != null ? `${Number(item.fps).toFixed(3)} fps` : "",
      item.media_kind || "",
    ].filter(Boolean).join(" · ");
  }

  function itemPagerMarkup() {
    const page = Math.max(1, Number(studioState.itemPage || 1));
    const pageCount = Math.max(1, Number(studioState.itemPageCount || 1));
    return `<div class="pager">
      <span class="muted">共 ${Number(studioState.itemTotal || 0)} 个匹配 Item · 第 ${page}/${pageCount} 页</span>
      <button class="secondary" data-studio-item-page="${page - 1}" type="button" ${page <= 1 ? "disabled" : ""}>上一页</button>
      <button class="secondary" data-studio-item-page="${page + 1}" type="button" ${page >= pageCount ? "disabled" : ""}>下一页</button>
    </div>`;
  }

  function renderItems() {
    const host = el("studio-items");
    if (!host) return;
    if (!studioState.selectedGroupId) {
      host.innerHTML = "<p class=\"muted\">没有可用的 GT 文件夹。</p>";
      return;
    }
    if (!studioState.items.length) {
      host.innerHTML = `<p class="muted">当前文件夹中没有匹配的 GT Media Item。</p>${itemPagerMarkup()}`;
      return;
    }
    host.innerHTML = `
      <div class="coverage-summary"><strong>已选 ${studioState.selectedItemIds.size} 个视频</strong><span class="muted">每个 Item 都绑定精确 canonical GT；翻页和搜索不会取消已选视频。</span></div>
      <div class="studio-item-grid">${studioState.items.map((item) => {
        const id = itemId(item);
        return `<label class="studio-item-card ${studioState.selectedItemIds.has(id) ? "selected" : ""}">
          <input type="checkbox" data-studio-item="${id}" ${studioState.selectedItemIds.has(id) ? "checked" : ""}>
          <span><strong>${safe(item.display_name || item.item_key || `Media Item #${id}`)}</strong><small>${safe(itemMetadata(item))}</small><small>GT asset #${Number(item.canonical_gt_asset_id || 0)}</small></span>
        </label>`;
      }).join("")}</div>
      ${itemPagerMarkup()}`;
  }

  async function loadItemGroups() {
    const payload = await request("/api/media/item-groups?role=gt");
    studioState.itemGroups = payload.groups || payload.item_groups || [];
    const ids = new Set(studioState.itemGroups.map(groupId));
    if (!ids.has(String(studioState.selectedGroupId))) {
      studioState.selectedGroupId = groupId(studioState.itemGroups[0]);
      studioState.selectedItemIds.clear();
      studioState.itemPage = 1;
    }
    renderItemGroupOptions();
    await loadItems();
  }

  async function loadItems(options = {}) {
    const generation = ++studioState.itemRequestGeneration;
    if (!studioState.selectedGroupId) {
      studioState.items = [];
      studioState.itemPage = 1;
      studioState.itemPageCount = 1;
      studioState.itemTotal = 0;
      renderItems();
      await loadMethodsForSelection();
      return;
    }
    const requestedPage = Math.max(1, Number(options.page || studioState.itemPage || 1));
    const path = `/api/media/items?group_id=${encodeURIComponent(studioState.selectedGroupId)}&q=${encodeURIComponent(studioState.itemQuery)}&page=${requestedPage}&page_size=${studioState.itemPageSize}`;
    const payload = await request(path);
    if (generation !== studioState.itemRequestGeneration) return;
    const pageCount = Math.max(1, Number(payload.page_count || payload.total_pages || 1));
    if (requestedPage > pageCount) return loadItems({ page: pageCount });
    studioState.items = payload.items || [];
    studioState.itemPage = Math.max(1, Number(payload.page || requestedPage));
    studioState.itemPageCount = pageCount;
    studioState.itemTotal = Math.max(0, Number(payload.total || 0));
    renderItems();
  }

  function methodOptions(slot, selected = "") {
    const mode = methodSourceMode(slot);
    const methods = studioState.methods.filter((row) => methodKind(row) === mode);
    const placeholder = mode === "run" ? "选择模型 Run" : "选择已绑定的 External 方法";
    return `<option value="">${placeholder}</option>${methods.map((method) => {
      const key = methodKey(method);
      const coverage = `${Number(method.covered_count || (method.covered_item_ids || []).length || 0)}/${Number(method.total_items || studioState.selectedItemIds.size || 0)}`;
      const label = method.label || method.run_name || method.method_key || key;
      return `<option value="${safe(key)}" ${key === selected ? "selected" : ""}>${safe(label)} · 覆盖 ${coverage}${method.complete ? " · 完整" : " · 有缺失"}</option>`;
    }).join("")}`;
  }

  function fillMethodSelects() {
    for (const slot of ["a", "b"]) {
      const select = el(`studio-method-${slot}`);
      if (!select) continue;
      const previous = select.value;
      select.innerHTML = methodOptions(slot, previous);
      if (![...select.options].some((option) => option.value === previous)) select.value = "";
    }
    const details = el("studio-upload-advanced");
    if (details && ["a", "b"].some((slot) => methodSourceMode(slot) === "external")) details.open = true;
  }

  async function loadMethodsForSelection() {
    const generation = ++studioState.methodRequestGeneration;
    const ids = selectedItemIds();
    if (!ids.length) {
      studioState.methods = [];
      fillMethodSelects();
      return;
    }
    const query = ids.map((id) => `item_id=${encodeURIComponent(id)}`).join("&");
    const payload = await request(`/api/media/methods?${query}`);
    if (generation !== studioState.methodRequestGeneration) return;
    studioState.methods = payload.methods || [];
    fillMethodSelects();
  }

  function matrixRows(preview) {
    return preview?.items || preview?.matrix || [];
  }

  function rowReady(row) {
    return row.ready === true
      || row.eligible === true
      || ["ready", "compatible", "validated", "aligned", "ok"].includes(String(row.status || row.alignment_status || ""));
  }

  function outputSummary(payload, fallback) {
    if (!payload || payload.missing) return "缺失";
    return [
      payload.label || fallback,
      payload.frame_count ? `${Number(payload.frame_count)} 帧` : "",
      payload.width && payload.height ? `${Number(payload.width)}×${Number(payload.height)}` : "",
      payload.fps != null ? `${Number(payload.fps).toFixed(3)} fps` : "",
    ].filter(Boolean).join(" · ");
  }

  function alignmentPlanSummary(plan) {
    if (!plan) return "-";
    const target = plan.target || plan.target_size || {};
    const width = Number(plan.target_width || target.width || plan.width || 0);
    const height = Number(plan.target_height || target.height || plan.height || 0);
    const fingerprint = plan.fingerprint ? ` · ${String(plan.fingerprint).slice(0, 12)}` : "";
    const temporal = plan.temporal && typeof plan.temporal === "object" ? plan.temporal : {};
    const transforms = Array.isArray(plan.transforms)
      ? plan.transforms
      : Object.entries(plan.transforms || plan.sources || {}).map(([label, row]) => ({ label, ...row }));
    const changes = transforms
      .filter((row) => String(row.direction || row.resize_kind || row.operation || "none") !== "none")
      .map((row) => `${row.label || row.slot}: ${row.direction || row.resize_kind || row.operation}`)
      .join("；");
    const temporalLabel = temporal.frame_count != null
      ? ` · ${Number(temporal.frame_count)} 帧${temporal.mode ? ` ${temporal.mode}` : ""}`
      : "";
    return `${width || "-"}×${height || "-"} · ${plan.filter || plan.interpolation || "lanczos"}${temporalLabel}${changes ? ` · ${changes}` : ""}${fingerprint}`;
  }

  function renderCoverage() {
    const host = el("studio-coverage");
    if (!host) return;
    const rows = matrixRows(studioState.preview);
    if (!rows.length) {
      host.innerHTML = "<p class=\"muted\">选择视频和两份方法后检查覆盖与对齐。</p>";
      return;
    }
    const readyCount = rows.filter(rowReady).length;
    host.innerHTML = `
      <div class="coverage-summary">
        <strong>${readyCount}/${studioState.selectedItemIds.size} 个 Media Item 可发布</strong>
        <span class="muted">时间映射严格验证；空间尺寸会按每个 Item 的较小 Pred 规范化并写入报告。</span>
      </div>
      <div class="coverage-table"><table>
        <thead><tr><th>GT Item</th><th>方法 A</th><th>方法 B</th><th>空间规范化</th><th>状态</th></tr></thead>
        <tbody>${rows.map((row) => {
          const ready = rowReady(row);
          const reason = row.reason || row.error || row.alignment_reason || (Array.isArray(row.reasons) ? row.reasons.join("；") : "") || (ready ? "时间严格对齐" : "缺失或不兼容");
          const item = row.item || row.reference || {};
          const a = row.method_a || row.methods?.a || row.binding_a;
          const b = row.method_b || row.methods?.b || row.binding_b;
          return `<tr class="${ready ? "ready" : "blocked"}"><td><strong>${safe(item.display_name || row.display_name || row.item_key || `Media Item #${itemId(row)}`)}</strong><br><small>${safe(outputSummary(item, "GT"))}</small></td><td>${safe(outputSummary(a, "Pred A"))}</td><td>${safe(outputSummary(b, "Pred B"))}</td><td>${safe(alignmentPlanSummary(row.alignment_plan || row.alignment))}</td><td><span class="studio-status ${ready ? "ok" : "warn"}">${safe(reason)}</span></td></tr>`;
        }).join("")}</tbody>
      </table></div>`;
  }

  async function previewCoverage() {
    const ids = selectedItemIds();
    const methodA = selectedMethod("a");
    const methodB = selectedMethod("b");
    if (!ids.length) throw new Error("请至少选择一个 GT Media Item");
    if (!methodA || !methodB) throw new Error("请选择两份 Pred 方法");
    if (JSON.stringify(methodA) === JSON.stringify(methodB)) throw new Error("两份 Pred 方法必须不同");
    const generation = ++studioState.previewGeneration;
    const signature = JSON.stringify([ids, methodA, methodB]);
    const preview = await request("/api/evaluation-campaigns/v2/preview", {
      method: "POST",
      body: JSON.stringify({
        media_item_ids: ids,
        method_a: methodA,
        method_b: methodB,
        spatial_policy: spatialPolicy(),
      }),
    });
    if (generation !== studioState.previewGeneration
        || signature !== JSON.stringify([selectedItemIds(), selectedMethod("a"), selectedMethod("b")])) return;
    studioState.preview = preview;
    renderCoverage();
  }

  function campaignStatus(campaign) {
    if (campaign?.archived) return "archived";
    return campaign?.preparation_status || campaign?.status || "draft";
  }

  function campaignPreparationActive(campaign) {
    const lifecycleStatus = String(campaign?.status || "");
    const preparationStatus = String(campaign?.preparation_status || campaign?.preparation?.state || "");
    return lifecycleStatus === "preparing"
      || ["requested", "queued", "preparing", "running"].includes(preparationStatus);
  }

  function campaignPublishCommitted(campaign) {
    const lifecycleStatus = String(campaign?.status || "");
    const preparationStatus = String(campaign?.preparation_status || campaign?.preparation?.state || "");
    return ["preparing", "published", "failed"].includes(lifecycleStatus)
      || ["requested", "queued", "preparing", "running", "completed", "succeeded", "failed"]
        .includes(preparationStatus);
  }

  function campaignKey(campaign) {
    return String(campaign.campaign_key || `${Number(campaign.schema_version || 1) >= 2 ? "v2" : "v1"}:${Number(campaign.id)}`);
  }

  function cleanupRequestStatus(requestRow) {
    const status = String(requestRow?.status || "requested");
    if (status === "running") return "正在清理";
    if (status === "failed") return "清理失败";
    return "等待后台清理";
  }

  function cleanupRequestsMarkup() {
    const requests = studioState.cleanupRequests || [];
    if (!requests.length) return "";
    const failed = requests.filter((row) => String(row.status || "") === "failed").length;
    return `
      <section class="stats-block" data-evaluation-cleanup-panel>
        <div class="panel-head"><div><h3>盲测删除清理</h3><p class="muted">记录已删除；残留文件由后台继续清理，失败项可以在这里重试。</p></div><span class="studio-status">${failed ? `${failed} 项失败` : `${requests.length} 项处理中`}</span></div>
        <div class="coverage-table"><table><thead><tr><th>Campaign</th><th>状态</th><th>尝试次数</th><th></th></tr></thead><tbody>${requests.map((row) => `
          <tr>
            <td>#${Number(row.campaign_id || 0)}</td>
            <td>${safe(cleanupRequestStatus(row))}</td>
            <td>${Number(row.attempt_count || 0)}</td>
            <td>${String(row.status || "") === "failed" ? `<button class="secondary" type="button" data-evaluation-cleanup-retry="${Number(row.id)}">重试清理</button>` : "后台会自动重试"}</td>
          </tr>`).join("")}</tbody></table></div>
      </section>`;
  }

  function rememberCleanupRequest(result, campaignId) {
    const requestId = Number(result?.cleanup_request_id || 0);
    if (!requestId) return;
    studioState.cleanupRequests = studioState.cleanupRequests.filter((row) => Number(row.id) !== requestId);
    if (String(result.cleanup_status || "") !== "completed") {
      studioState.cleanupRequests.unshift({
        id: requestId,
        campaign_id: Number(campaignId),
        status: result.cleanup_status || "requested",
        attempt_count: 0,
      });
    }
  }

  function renderCampaignList() {
    const host = el("studio-campaign-list");
    if (!host) return;
    const cleanupMarkup = cleanupRequestsMarkup();
    if (!studioState.campaigns.length) {
      host.innerHTML = `${cleanupMarkup}<p class="muted">还没有 Campaign。</p>`;
      return;
    }
    host.innerHTML = cleanupMarkup + studioState.campaigns.map((campaign) => `
      <button class="studio-campaign-row ${campaignKey(campaign) === studioState.selectedCampaignKey ? "active" : ""}" data-studio-campaign="${safe(campaignKey(campaign))}" type="button">
        <span><strong>${safe(campaign.name)}</strong><small>${safe(campaign.public_title || campaign.metadata?.public_title || "匿名视频质量评测")}</small></span>
        <span class="studio-status">${safe(campaignStatus(campaign))}</span>
      </button>`).join("");
  }

  function renderPackages() {
    const host = el("media-packages-content");
    if (!host) return;
    const packages = studioState.campaigns.filter((campaign) =>
      Number(campaign.schema_version || 1) >= 2
      && ["published", "closed", "archived"].includes(String(campaign.status || "")),
    );
    host.innerHTML = packages.length ? packages.map((campaign) => `
      <div class="derived-video">
        <span><strong>${safe(campaign.name)}</strong><small class="muted">${safe(campaign.public_title || campaign.metadata?.public_title || "匿名视频质量评测")}</small></span>
        <div><span class="studio-status">${safe(campaign.status)}</span><span class="studio-status">${Number(campaign.item_count || 0)} 视频</span></div>
      </div>`).join("") : "<p class=\"muted\">还没有已发布的冻结评测包。</p>";
  }

  function analysisRanking(analysis) {
    return analysis?.human?.ranking || analysis?.ranking || [];
  }

  function objectiveMetrics(analysis, legacy) {
    if (legacy || !Array.isArray(analysis?.objective?.metrics)) return [];
    return analysis.objective.metrics;
  }

  function objectiveMetricNumber(value) {
    const number = Number(value);
    return value == null || !Number.isFinite(number) ? "-" : number.toFixed(4);
  }

  function objectiveMetricStatuses(statusCounts) {
    if (!statusCounts || typeof statusCounts !== "object" || Array.isArray(statusCounts)) return "-";
    const entries = Object.entries(statusCounts).sort(([left], [right]) => left.localeCompare(right));
    return entries.length ? entries.map(([status, count]) => `${safe(status)}: ${Number(count || 0)}`).join(" · ") : "-";
  }

  function objectiveCurveChoices(analysis, campaign) {
    const allowed = new Set(["lpips_vit_patch", "lpips_convnext"]);
    const itemsById = new Map((campaign?.items || []).map((item) => [Number(item.id), item]));
    const grouped = new Map();
    for (const row of analysis?.objective?.items || []) {
      const metricName = String(row.metric_name || "");
      const itemIdValue = Number(row.item_id || 0);
      if (!allowed.has(metricName) || !itemIdValue || !itemsById.has(itemIdValue)) continue;
      const key = `${itemIdValue}:${metricName}`;
      const current = grouped.get(key) || {
        itemId: itemIdValue,
        videoName: String(itemsById.get(itemIdValue)?.video_name || row.video_name || ""),
        metricName,
        methods: new Set(),
        completedMethods: new Set(),
      };
      current.methods.add(Number(row.method_id || 0));
      if (Number(row.frame_coverage?.completed || 0) > 0) {
        current.completedMethods.add(Number(row.method_id || 0));
      }
      grouped.set(key, current);
    }
    return Array.from(grouped.values())
      .map((row) => ({
        ...row,
        methodCount: row.methods.size,
        completedMethodCount: row.completedMethods.size,
      }))
      .sort((left, right) => left.videoName.localeCompare(right.videoName)
        || left.metricName.localeCompare(right.metricName));
  }

  function objectiveCurveFingerprint(analysis) {
    return String(analysis?.objective?.metric_fingerprint || "");
  }

  function objectiveCurveKey(campaignId, selection, fingerprint = "") {
    return `${Number(campaignId)}:${Number(selection?.itemId || 0)}:${String(selection?.metricName || "")}:${String(fingerprint)}`;
  }

  function supersedeObjectiveCurveRequest() {
    studioState.objectiveCurveRequestGeneration += 1;
    studioState.objectiveCurveInFlight = null;
    studioState.objectiveCurveLoadingKey = "";
  }

  function objectiveCurveSegments(points, minOrdinal, maxOrdinal, minValue, maxValue) {
    const segments = [];
    let current = [];
    for (const point of points || []) {
      const ordinal = Number(point.ordinal);
      const value = point.status === "completed" && point.value != null ? Number(point.value) : Number.NaN;
      if (!Number.isFinite(value)) {
        if (current.length) segments.push(current.join(" "));
        current = [];
      } else {
        const x = maxOrdinal === minOrdinal ? 50 : 4 + ((ordinal - minOrdinal) / (maxOrdinal - minOrdinal)) * 92;
        const y = maxValue === minValue ? 26 : 44 - ((value - minValue) / (maxValue - minValue)) * 36;
        current.push(`${x.toFixed(3)},${y.toFixed(3)}`);
      }
    }
    if (current.length) segments.push(current.join(" "));
    return segments;
  }

  function renderObjectiveCurveChart(curve) {
    if (!curve?.series?.length) return "<p class=\"muted\">该 Item 没有可用的 LPIPS 曲线数据。</p>";
    const reasons = Array.from(new Set(curve.series.flatMap((series) => [
      ...Object.keys(series.reason_counts || {}),
      ...(series.points || []).map((point) => point.reason || ""),
    ]).filter(Boolean)));
    const completed = curve.series.flatMap((series) => (series.points || []).filter((point) =>
      point.status === "completed" && point.value != null && Number.isFinite(Number(point.value))));
    if (!completed.length) {
      return `<div class="message warn"><p>没有 completed 的 LPIPS 点。${curve.series.map((series) => `${safe(series.method_label)}：${objectiveMetricStatuses(series.status_counts)}`).join("；")}${reasons.length ? `。原因：${safe(reasons.slice(0, 5).join("；"))}` : ""}</p></div>`;
    }
    const values = completed.map((point) => Number(point.value));
    const maxOrdinal = Math.max(0, Number(curve.frame_count || 1) - 1);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const lines = curve.series.map((series, index) => objectiveCurveSegments(
      series.points, 0, maxOrdinal, minValue, maxValue,
    ).map((points) => `<polyline class="objective-curve-line series-${index === 0 ? "a" : "b"}" points="${points}" fill="none"></polyline>`).join("")).join("");
    const dots = curve.series.map((series, index) => (series.points || []).filter((point) =>
      point.status === "completed" && point.value != null && Number.isFinite(Number(point.value))).map((point) => {
      const ordinal = Number(point.ordinal);
      const frame = Number(point.frame_index);
      const value = Number(point.value);
      const x = maxOrdinal === 0 ? 50 : 4 + (ordinal / maxOrdinal) * 92;
      const y = maxValue === minValue ? 26 : 44 - ((value - minValue) / (maxValue - minValue)) * 36;
      return `<circle tabindex="0" class="objective-curve-dot series-${index === 0 ? "a" : "b"}" cx="${x.toFixed(3)}" cy="${y.toFixed(3)}" r="0.75"><title>${safe(series.method_label)} · frame ${frame} · ${value.toFixed(6)}</title></circle>`;
    }).join("")).join("");
    const unavailable = curve.series.flatMap((series) => (series.points || []).filter((point) => point.status !== "completed"));
    const unavailableDots = curve.series.map((series, index) => (series.points || []).filter((point) => point.status !== "completed").map((point) => {
      const ordinal = Number(point.ordinal);
      const x = maxOrdinal === 0 ? 50 : 4 + (ordinal / maxOrdinal) * 92;
      const y = index === 0 ? 47 : 49;
      return `<circle tabindex="0" class="objective-curve-missing series-${index === 0 ? "a" : "b"}" cx="${x.toFixed(3)}" cy="${y}" r="0.8"><title>${safe(series.method_label)} · frame ${Number(point.frame_index)} · ${safe(point.status)}${point.reason ? ` · ${safe(point.reason)}` : ""}</title></circle>`;
    }).join("")).join("");
    return `
      <div class="objective-curve-legend">${curve.series.map((series, index) => `<span><i class="series-${index === 0 ? "a" : "b"}"></i>${safe(series.method_label)} · ${objectiveMetricStatuses(series.status_counts)}</span>`).join("")}</div>
      <div class="objective-curve-plot"><svg viewBox="0 0 100 52" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${safe(curve.metric_name)} 双曲线对比"><g class="objective-curve-grid"><line x1="4" x2="96" y1="8" y2="8"></line><line x1="4" x2="96" y1="26" y2="26"></line><line x1="4" x2="96" y1="44" y2="44"></line></g>${lines}${dots}${unavailableDots}</svg></div>
      <div class="objective-curve-scale"><span>frame ${Number(curve.series[0]?.points?.[0]?.frame_index ?? 0)}</span><span>LPIPS ${minValue.toFixed(6)} – ${maxValue.toFixed(6)}</span><span>frame ${Number((curve.series[0]?.points || [])[(curve.series[0]?.points || []).length - 1]?.frame_index ?? 0)}</span></div>
      <p class="muted">双方 completed 重合帧 ${Number(curve.completed_overlap || 0)}。数值越低越好；断线表示缺失或不可用。${reasons.length ? ` 原因：${safe(reasons.slice(0, 3).join("；"))}` : ""}</p>`;
  }

  function renderObjectiveCurvePanel(analysis, campaign) {
    const choices = objectiveCurveChoices(analysis, campaign);
    if (!choices.length) return "";
    const current = studioState.objectiveCurveSelection;
    const validCurrent = choices.some((row) => row.itemId === Number(current?.itemId)
      && row.metricName === String(current?.metricName || ""));
    if (!validCurrent) {
      const preferred = choices.find((row) => row.completedMethodCount >= 2) || choices[0];
      studioState.objectiveCurveSelection = { itemId: preferred.itemId, metricName: preferred.metricName };
    }
    const selection = studioState.objectiveCurveSelection;
    const itemChoices = Array.from(new Map(choices.map((row) => [row.itemId, row])).values());
    const metricChoices = choices.filter((row) => row.itemId === Number(selection.itemId));
    const fingerprint = objectiveCurveFingerprint(analysis);
    const key = objectiveCurveKey(campaign.id, selection, fingerprint);
    const curve = studioState.objectiveCurveCache[key];
    const curveError = studioState.objectiveCurveErrors[key];
    const body = studioState.objectiveCurveLoadingKey === key
      ? "<p class=\"muted\">正在读取逐帧 LPIPS 数据…</p>"
      : curveError
        ? `<div class="message error"><p>${safe(curveError)}</p><button type="button" class="secondary" data-objective-curve-retry>重试读取</button></div>`
      : curve
        ? renderObjectiveCurveChart(curve)
        : "<p class=\"muted\">等待读取逐帧 LPIPS 数据。</p>";
    return `<section class="stats-block studio-objective-curve" data-analysis-section="objective-curve"><div class="studio-analysis-head"><div><h3>LPIPS 双曲线</h3><p class="muted">按 Campaign 精确绑定对比两个预测方法，不触发重新计算。</p></div><div class="objective-curve-controls"><label><span>Item</span><select data-objective-curve-item>${itemChoices.map((row) => `<option value="${row.itemId}" ${row.itemId === Number(selection.itemId) ? "selected" : ""}>${safe(row.videoName)}</option>`).join("")}</select></label><label><span>指标</span><select data-objective-curve-metric>${metricChoices.map((row) => `<option value="${safe(row.metricName)}" ${row.metricName === selection.metricName ? "selected" : ""}>${safe(row.metricName)}</option>`).join("")}</select></label></div></div><div class="objective-curve-body">${body}</div></section>`;
  }

  function participantShareUrl(shareUrl, campaign) {
    const rawUrl = String(
      shareUrl || (campaign?.share_token ? `/evaluate/${campaign.share_token}` : ""),
    ).trim();
    if (!rawUrl) return "";
    try {
      return new URL(rawUrl, location.origin).href;
    } catch (_error) {
      return rawUrl;
    }
  }

  function campaignParticipantAvailable(campaign) {
    return ["published", "closed", "archived"].includes(String(campaign?.status || ""));
  }

  function isLoopbackOrigin() {
    const host = String(location.hostname || "").toLowerCase();
    return host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0" || host === "::1" || host === "[::1]";
  }

  function preparationProgressMarkup(progress, campaign) {
    const report = progress.report && typeof progress.report === "object" ? progress.report : {};
    const details = { ...report, ...progress };
    const total = Number(progress.total || details.total || 0);
    if (!total) return "";
    const current = Number(progress.current || details.current || 0);
    const phase = progress.phase || details.phase || campaignStatus(campaign);
    const legacyMarkup = `<div class="studio-progress"><progress max="${total}" value="${current}"></progress><span>${current}/${total} · ${safe(phase)}</span></div>`;
    const timingEntries = details.timings && typeof details.timings === "object" && !Array.isArray(details.timings)
      ? Object.entries(details.timings).filter(([_name, value]) => Number.isFinite(Number(value)))
      : [];
    const fineValues = [
      details.overall_fraction,
      details.stage,
      details.item_index,
      details.frame_current,
      details.frame_total,
      details.pipeline,
    ];
    const hasFineProgress = fineValues.some((value) => value != null && value !== "") || timingEntries.length > 0;
    if (!hasFineProgress) return legacyMarkup;

    const hasOverallFraction = details.overall_fraction != null && details.overall_fraction !== "";
    const fractionValue = hasOverallFraction ? Number(details.overall_fraction) : Number.NaN;
    const fraction = Number.isFinite(fractionValue)
      ? Math.min(1, Math.max(0, fractionValue))
      : Math.min(1, Math.max(0, current / total));
    const meta = [];
    if (details.item_index != null && details.item_index !== "") {
      const itemName = details.item_name == null || details.item_name === "" ? "" : ` · ${safe(details.item_name)}`;
      meta.push(`<span>Item ${Number(details.item_index)}/${total}${itemName}</span>`);
    }
    if (details.stage != null && details.stage !== "") meta.push(`<span>阶段 ${safe(details.stage)}</span>`);
    if (details.frame_current != null || details.frame_total != null) {
      const frameCurrent = details.frame_current == null || details.frame_current === "" ? "–" : Number(details.frame_current);
      const frameTotal = details.frame_total == null || details.frame_total === "" ? "" : `/${Number(details.frame_total)}`;
      meta.push(`<span>帧 ${frameCurrent}${frameTotal}</span>`);
    }
    if (details.pipeline != null && details.pipeline !== "") meta.push(`<span>管线 ${safe(details.pipeline)}</span>`);
    if (hasOverallFraction) {
      meta.push(`<span>总进度 ${(fraction * 100).toFixed(1)}%</span>`);
    }
    if (timingEntries.length) {
      meta.push(`<span>耗时 ${timingEntries.map(([name, value]) => `${safe(name)} ${Number(value).toFixed(2)}s`).join(" · ")}</span>`);
    }
    return `<div class="studio-progress studio-progress-detailed"><progress max="1" value="${fraction}"></progress><div class="studio-progress-detail"><span>${current}/${total} · ${safe(phase)}</span><small>${meta.join("")}</small></div></div>`;
  }

  function renderCampaignDetail(payload) {
    const host = el("studio-campaign-detail");
    if (!host) return;
    const campaign = payload?.campaign || payload;
    if (!campaign?.id) {
      studioState.campaignDetail = null;
      studioState.objectiveCurveSelection = null;
      host.innerHTML = "<p class=\"muted\">选择 Campaign 查看准备进度、分享链接和结果。</p>";
      return;
    }
    studioState.campaignDetail = payload;
    const progress = payload.preparation || campaign.preparation || {};
    const coverage = payload.coverage || campaign.coverage || {};
    const analysis = payload.analysis || null;
    const ranking = analysisRanking(analysis);
    const shareUrl = campaignParticipantAvailable(campaign)
      ? participantShareUrl(payload.share_url || campaign.share_url, campaign)
      : "";
    const legacy = Number(campaign.schema_version || 1) < 2;
    const objective = objectiveMetrics(analysis, legacy);
    const contractWarnings = Array.isArray(campaign.contract_warnings)
      ? campaign.contract_warnings
      : (Array.isArray(payload?.contract_warnings) ? payload.contract_warnings : []);
    host.innerHTML = `
      <div class="studio-detail-head"><div><p class="eyebrow">${legacy ? "历史 Campaign" : "Campaign V2"}</p><h3>${safe(campaign.name)}</h3><p class="muted">${safe(campaign.public_title || campaign.metadata?.public_title || "匿名视频质量评测")}</p></div><span class="studio-status">${safe(campaignStatus(campaign))}</span></div>
      ${legacy ? "<div class=\"message warn\"><p>旧 schema v1 Campaign 只读保留；可继续导出，不会按标签猜测迁移。</p></div>" : ""}
      ${contractWarnings.length ? `<div class="message warn evaluation-contract-warning"><p>该历史 Campaign 的源 Run 不满足当前 midpoint-triplet-v2 评测契约；现有结果保持只读，请按提示重新运行源实验后再创建新 Campaign。</p><p>${safe(contractWarnings.join("；"))}</p></div>` : ""}
      ${preparationProgressMarkup(progress, campaign)}
      ${progress.error?.message || campaign.preparation_error?.message ? `<div class="message error"><p>${safe(progress.error?.message || campaign.preparation_error?.message)}</p></div>` : ""}
      <div class="summary-grid"><div><span>视频</span><strong>${Number(coverage.items || campaign.item_count || 0)}</strong></div><div><span>任务</span><strong>${Number(coverage.tasks || campaign.task_count || 0)}</strong></div><div><span>投票</span><strong>${Number(coverage.votes || campaign.vote_count || 0)}</strong></div><div><span>目标票数/视频</span><strong>${Number(campaign.target_votes || 0)}</strong></div></div>
      ${shareUrl ? `<label class="studio-share"><span>参与链接</span><div><input readonly value="${safe(shareUrl)}"><button data-copy-share="${safe(shareUrl)}" type="button">复制</button></div></label>` : ""}
      ${shareUrl && isLoopbackOrigin() ? `<div class="message warn studio-share-warning"><p>当前链接使用本机回环地址，只能在这台电脑上打开。若要让受控内网参与者访问，请以 <code>--host 0.0.0.0</code> 启动服务，再从 <code>http://&lt;服务器内网 IP&gt;:8765</code> 打开 Studio 并重新复制链接；不要将服务暴露到公网。</p></div>` : ""}
      <div class="studio-actions">${!legacy && ["draft", "failed"].includes(campaign.status) ? `<button data-studio-publish="${Number(campaign.id)}" type="button">${campaign.status === "failed" ? "重试发布" : "发布并冻结"}</button>` : ""}${!legacy && campaign.status === "published" ? `<button data-studio-close="${Number(campaign.id)}" type="button">关闭</button>` : ""}${!legacy && ["closed", "failed"].includes(campaign.status) ? `<button class="secondary" data-studio-archive="${Number(campaign.id)}" type="button">归档</button>` : ""}${!legacy ? `<button class="danger secondary" data-studio-delete="${Number(campaign.id)}" type="button">永久删除</button>` : ""}${legacy && !campaign.archived ? `<button class="secondary" data-studio-legacy-archive="${Number(campaign.id)}" type="button">归档</button>` : ""}${legacy && campaign.status === "draft" && Number(campaign.vote_count || campaign.votes || 0) === 0 ? `<button class="danger secondary" data-studio-legacy-discard="${Number(campaign.id)}" type="button">丢弃空 Draft</button>` : ""}${legacy ? `<a class="secondary button-link" href="/api/evaluation-campaigns/${Number(campaign.id)}/export">导出</a>` : `<a class="secondary button-link" href="/api/evaluation-campaigns/v2/${Number(campaign.id)}/export">导出</a>`}</div>
      ${ranking.length ? `<section class="stats-block studio-human-results" data-analysis-section="human"><h3>人工排名</h3><div class="coverage-table"><table><thead><tr><th>排名</th><th>方法</th><th>得分</th><th>95% CI</th></tr></thead><tbody>${ranking.map((row, index) => `<tr><td>${index + 1}</td><td>${safe(row.label || row.name)}</td><td>${Number(row.score || 0).toFixed(4)}</td><td>${safe((row.ci95 || []).join(" – ") || "-")}</td></tr>`).join("")}</tbody></table></div></section>` : ""}
      ${!legacy && objective.length ? `<section class="stats-block studio-objective-results" data-analysis-section="objective"><div class="studio-analysis-head"><h3>客观指标</h3><p class="muted">客观指标与人工排名分别展示，不生成合成总分。</p></div><div class="coverage-table"><table><thead><tr><th>指标</th><th>方法</th><th>方向</th><th>状态</th><th>有效样本</th><th>均值 / 中位数</th><th>P10 / P90</th></tr></thead><tbody>${objective.map((row) => `<tr><td>${safe(row.metric_name || "-")}</td><td>${safe(row.method_label || "-")}</td><td>${safe(row.direction || "-")}</td><td>${objectiveMetricStatuses(row.status_counts)}</td><td>${Number(row.count || 0)}</td><td>${objectiveMetricNumber(row.mean)} / ${objectiveMetricNumber(row.median)}</td><td>${objectiveMetricNumber(row.p10)} / ${objectiveMetricNumber(row.p90)}</td></tr>`).join("")}</tbody></table></div></section>` : ""}
      ${!legacy ? renderObjectiveCurvePanel(analysis, campaign) : ""}`;
    const selection = studioState.objectiveCurveSelection;
    if (!legacy && selection) loadObjectiveCurve(campaign.id, selection).catch((error) => notify(error.message));
  }

  async function loadObjectiveCurve(campaignId, selection) {
    const detail = studioState.campaignDetail;
    const fingerprint = objectiveCurveFingerprint(detail?.analysis);
    const key = objectiveCurveKey(campaignId, selection, fingerprint);
    if (studioState.objectiveCurveCache[key] || studioState.objectiveCurveErrors[key]) return;
    if (studioState.objectiveCurveInFlight?.key === key) return;
    const generation = ++studioState.objectiveCurveRequestGeneration;
    const requestToken = { key, generation };
    studioState.objectiveCurveInFlight = requestToken;
    studioState.objectiveCurveLoadingKey = key;
    try {
      const curve = await request(`/api/evaluation-campaigns/v2/${Number(campaignId)}/objective-curve?item_id=${Number(selection.itemId)}&metric_name=${encodeURIComponent(selection.metricName)}`);
      if (studioState.objectiveCurveInFlight !== requestToken
          || generation !== studioState.objectiveCurveRequestGeneration) return;
      studioState.objectiveCurveCache[key] = curve;
    } catch (error) {
      if (studioState.objectiveCurveInFlight !== requestToken
          || generation !== studioState.objectiveCurveRequestGeneration) return;
      studioState.objectiveCurveErrors[key] = error.message || String(error);
      throw error;
    } finally {
      if (studioState.objectiveCurveInFlight === requestToken) {
        studioState.objectiveCurveInFlight = null;
        studioState.objectiveCurveLoadingKey = "";
      }
      if (generation === studioState.objectiveCurveRequestGeneration) {
        if (studioState.campaignDetail) renderCampaignDetail(studioState.campaignDetail);
      }
    }
  }

  async function loadCampaigns(options = {}) {
    const payload = await request("/api/evaluation-campaigns");
    studioState.campaigns = payload.campaigns || [];
    const preserveMissingKey = String(options.preserveMissingKey || "");
    if (studioState.selectedCampaignKey
        && studioState.selectedCampaignKey !== preserveMissingKey
        && !studioState.campaigns.some((campaign) => campaignKey(campaign) === studioState.selectedCampaignKey)) {
      stopPreparationPoll();
      supersedeObjectiveCurveRequest();
      studioState.selectedCampaignKey = null;
      renderCampaignDetail(null);
    }
    renderCampaignList();
    renderPackages();
    if (studioState.selectedCampaignKey) await openCampaign(studioState.selectedCampaignKey, false);
  }

  async function loadCleanupRequests() {
    const payload = await request("/api/evaluation-cleanup-requests");
    studioState.cleanupRequests = payload.requests || [];
    renderCampaignList();
  }

  async function retryCleanupRequest(requestId) {
    await request(`/api/evaluation-cleanup-requests/${Number(requestId)}/retry`, {
      method: "POST",
      body: "{}",
    });
    await loadCleanupRequests();
    notify("已重新提交盲测文件清理，后台会继续处理");
  }

  function normalizeRunOutputs(payload) {
    return {
      ...payload,
      runs: (payload.runs || []).map((run) => {
        if ((run.videos || []).some((video) => Array.isArray(video.tracks))) return run;
        const grouped = new Map();
        for (const output of run.videos || run.outputs || []) {
          const name = String(output.video_name || output.video || "");
          const video = grouped.get(name) || { video_name: name, tracks: [] };
          video.tracks.push({ ...output });
          grouped.set(name, video);
        }
        return { ...run, videos: Array.from(grouped.values()) };
      }),
    };
  }

  function renderDerivedRuns(payload) {
    const host = el("media-derived-content");
    if (!host) return;
    const runs = payload.runs || [];
    host.innerHTML = runs.length ? runs.map((run) => `
      <details class="derived-run">
        <summary><span><strong>#${Number(run.run_id || run.id)} ${safe(run.run_name || run.name)}</strong><small>${safe([run.model_name, run.checkpoint].filter(Boolean).join(" / ") || "-")}</small></span><span>${Number((run.videos || []).length)} 视频</span></summary>
        <div class="derived-videos">${(run.videos || []).map((video) => `<div class="derived-video"><strong>${safe(video.video_name || video.name)}</strong><div>${(video.tracks || []).map((track) => `<span class="studio-status">${safe(track.track_label || track.display_name || "Pred")} · ${Number(track.frame_count || 0)} 帧 · ${Number(track.width || 0)}×${Number(track.height || 0)}</span>`).join("")}</div></div>`).join("")}</div>
      </details>`).join("") : "<p class=\"muted\">没有有效且已绑定 Media Item 的 Run 输出。</p>";
  }

  async function loadRunOutputs() {
    const payload = normalizeRunOutputs(await request("/api/media/run-outputs"));
    studioState.runOutputs = payload.runs || [];
    renderDerivedRuns(payload);
  }

  async function openCampaign(key, rerenderList = true) {
    const requestedKey = String(key).includes(":") ? String(key) : `v2:${Number(key)}`;
    const requestGeneration = ++studioState.campaignRequestGeneration;
    if (studioState.selectedCampaignKey !== requestedKey) {
      supersedeObjectiveCurveRequest();
      studioState.objectiveCurveSelection = null;
      studioState.campaignDetail = null;
    }
    if (studioState.preparationPollKey && studioState.preparationPollKey !== requestedKey) stopPreparationPoll();
    studioState.selectedCampaignKey = requestedKey;
    const campaign = studioState.campaigns.find((row) => campaignKey(row) === requestedKey);
    const campaignId = Number(campaign?.id || requestedKey.split(":").pop());
    const isV2 = requestedKey.startsWith("v2:") || Number(campaign?.schema_version || 1) >= 2;
    const payload = await request(isV2 ? `/api/evaluation-campaigns/v2/${campaignId}` : `/api/evaluation-campaigns/${campaignId}`);
    if (requestGeneration !== studioState.campaignRequestGeneration || studioState.selectedCampaignKey !== requestedKey) return;
    renderCampaignDetail(payload);
    if (rerenderList) renderCampaignList();
    if (campaignPreparationActive(payload.campaign || payload)) startPreparationPoll(requestedKey, campaignId);
    else if (studioState.preparationPollKey === requestedKey) stopPreparationPoll();
  }

  async function readCampaignTruth(campaignId) {
    try {
      return {
        exists: true,
        payload: await request(`/api/evaluation-campaigns/v2/${Number(campaignId)}`),
      };
    } catch (error) {
      if (Number(error.status) === 404) return { exists: false, payload: null };
      throw error;
    }
  }

  async function requestCampaignPublish(campaignId) {
    try {
      return await request(`/api/evaluation-campaigns/v2/${Number(campaignId)}/publish`, {
        method: "POST",
        body: "{}",
      });
    } catch (publishError) {
      if (publishError.status != null) throw publishError;
      let truth;
      try {
        truth = await readCampaignTruth(campaignId);
      } catch (_reconciliationError) {
        throw publishError;
      }
      const campaign = truth.payload?.campaign || truth.payload;
      if (truth.exists && campaignPublishCommitted(campaign)) return truth.payload;
      if (truth.exists) publishError.campaignDetail = truth.payload;
      throw publishError;
    }
  }

  async function publishCampaign(campaignId) {
    const key = `v2:${Number(campaignId)}`;
    let payload;
    try {
      payload = await requestCampaignPublish(campaignId);
    } catch (error) {
      if (studioState.selectedCampaignKey === key && error.campaignDetail) {
        renderCampaignDetail(error.campaignDetail);
      }
      try {
        await loadCampaigns();
      } catch (_refreshError) {
        // Keep the publish error as the actionable failure.
      }
      throw error;
    }
    if (studioState.selectedCampaignKey === key) renderCampaignDetail(payload);
    if (studioState.selectedCampaignKey === key
        && campaignPreparationActive(payload.campaign || payload)) {
      startPreparationPoll(key, campaignId);
    }
    try {
      await loadCampaigns({ preserveMissingKey: key });
    } catch (_refreshError) {
      // The publish result is authoritative; polling can refresh the selected detail.
    }
    return payload;
  }

  async function createCampaign(event) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const ids = selectedItemIds();
    const methodA = selectedMethod("a");
    const methodB = selectedMethod("b");
    if (!studioState.preview || !ids.length || !methodA || !methodB) throw new Error("请先检查视频覆盖与对齐");
    const rows = matrixRows(studioState.preview);
    if (rows.length !== ids.length || rows.some((row) => !rowReady(row))) throw new Error("所有已选 Media Item 都必须由两种方法完整覆盖并通过对齐");
    const created = await request("/api/evaluation-campaigns/v2", {
      method: "POST",
      body: JSON.stringify({
        name: String(form.get("name") || "").trim(),
        public_title: String(form.get("public_title") || "").trim(),
        target_votes: Number(form.get("target_votes") || 3),
        media_item_ids: ids,
        method_a: methodA,
        method_b: methodB,
        spatial_policy: spatialPolicy(),
        result_policy: "after_personal_completion",
      }),
    });
    const campaignId = Number(created.campaign?.id || created.id);
    stopPreparationPoll();
    studioState.campaignRequestGeneration += 1;
    supersedeObjectiveCurveRequest();
    studioState.selectedCampaignKey = `v2:${campaignId}`;
    studioState.campaignDetail = created.campaign ? created : { campaign: created };
    studioState.preview = null;
    renderCoverage();
    renderCampaignDetail(studioState.campaignDetail);
    const published = await publishCampaign(campaignId);
    notify(campaignStatus(published.campaign || published) === "failed"
      ? "Campaign 准备失败，请查看保留的准备错误后重试"
      : "Campaign 已进入规范化、校验与冻结队列");
  }

  function stopPreparationPoll() {
    if (studioState.preparationPoll !== null) clearTimeout(studioState.preparationPoll);
    studioState.preparationPoll = null;
    studioState.preparationPollKey = null;
    studioState.preparationPollGeneration += 1;
  }

  function startPreparationPoll(key, campaignId) {
    const requestedKey = String(key);
    if (studioState.preparationPollKey === requestedKey) return;
    stopPreparationPoll();
    studioState.preparationPollKey = requestedKey;
    const generation = studioState.preparationPollGeneration;
    const stillCurrent = () => generation === studioState.preparationPollGeneration
      && studioState.preparationPollKey === requestedKey
      && studioState.selectedCampaignKey === requestedKey;
    const scheduleNext = () => {
      if (!stillCurrent()) return;
      studioState.preparationPoll = setTimeout(pollOnce, 2000);
    };
    const pollOnce = async () => {
      if (!stillCurrent()) return;
      studioState.preparationPoll = null;
      try {
        const payload = await request(`/api/evaluation-campaigns/v2/${Number(campaignId)}`);
        if (!stillCurrent()) return;
        renderCampaignDetail(payload);
        if (!campaignPreparationActive(payload.campaign || payload)) {
          stopPreparationPoll();
          await loadCampaigns();
          return;
        }
      } catch (_error) {
        // Keep the last useful state; the serialized next poll can recover.
      }
      scheduleNext();
    };
    scheduleNext();
  }

  async function campaignAction(action, campaignId) {
    if (action === "publish") {
      await publishCampaign(campaignId);
      return;
    }
    await request(`/api/evaluation-campaigns/v2/${Number(campaignId)}/${action}`, { method: "POST", body: "{}" });
    await loadCampaigns();
    await openCampaign(`v2:${Number(campaignId)}`);
  }

  async function deleteCampaign(campaignId) {
    const campaign = studioState.campaigns.find((row) =>
      Number(row.schema_version || 1) >= 2 && Number(row.id) === Number(campaignId));
    if (!campaign) throw new Error("Campaign V2 不存在或已经删除");
    if (!window.confirm(`永久删除盲测记录“${campaign.name || campaign.id}”？此操作不可撤销。`)) return;
    const destructive = ["preparing", "published", "closed", "archived"].includes(String(campaign.status || ""))
      || Number(campaign.vote_count || campaign.votes || 0) > 0;
    if (destructive && !window.confirm("该记录正在准备、已经发布或包含投票。继续将永久删除任务、投票、分析缓存和冻结媒体。确认继续？")) return;
    const deletingKey = `v2:${Number(campaignId)}`;
    stopPreparationPoll();
    studioState.campaignRequestGeneration += 1;
    const deleteGeneration = studioState.campaignRequestGeneration;
    supersedeObjectiveCurveRequest();
    studioState.selectedCampaignKey = null;
    studioState.campaignDetail = null;
    studioState.objectiveCurveSelection = null;
    renderCampaignDetail(null);
    renderCampaignList();
    let result;
    try {
      result = await request(`/api/evaluation-campaigns/v2/${Number(campaignId)}`, {
        method: "DELETE",
        body: JSON.stringify({ confirm: true, confirm_destructive: destructive }),
      });
    } catch (deleteError) {
      let truth;
      try {
        truth = await readCampaignTruth(campaignId);
      } catch (_reconciliationError) {
        throw deleteError;
      }
      if (truth.exists) {
        if (studioState.campaignRequestGeneration === deleteGeneration
            && studioState.selectedCampaignKey === null) {
          studioState.selectedCampaignKey = deletingKey;
          renderCampaignDetail(truth.payload);
          renderCampaignList();
        }
        let refreshed = false;
        try {
          await loadCampaigns();
          refreshed = true;
        } catch (_refreshError) {
          // Preserve the reconciled detail and report the original delete failure.
        }
        if (!refreshed && studioState.selectedCampaignKey === deletingKey
            && campaignPreparationActive(truth.payload?.campaign || truth.payload)) {
          startPreparationPoll(deletingKey, campaignId);
        }
        throw deleteError;
      }
      result = { cleanup_pending: false, reclaimed_bytes: 0, response_recovered: true };
    }
    if (studioState.selectedCampaignKey === deletingKey) {
      stopPreparationPoll();
      studioState.campaignRequestGeneration += 1;
      supersedeObjectiveCurveRequest();
      studioState.selectedCampaignKey = null;
      studioState.campaignDetail = null;
      studioState.objectiveCurveSelection = null;
      renderCampaignDetail(null);
    }
    studioState.campaigns = studioState.campaigns.filter((row) => campaignKey(row) !== deletingKey);
    rememberCleanupRequest(result, campaignId);
    renderCampaignList();
    renderPackages();
    try {
      await loadCampaigns();
    } catch (_refreshError) {
      // The DELETE response or reconciliation GET already confirmed deletion.
    }
    try {
      await loadCleanupRequests();
    } catch (_cleanupRefreshError) {
      // Keep the cleanup request returned by DELETE visible until the next refresh.
    }
    notify(result.cleanup_pending
      ? "盲测记录已删除；隔离文件仍被占用，后台会继续清理，可在上方查看或重试"
      : result.response_recovered
        ? "盲测记录已永久删除（已通过服务端状态核对）"
        : `盲测记录已永久删除，释放 ${formatBytes(result.reclaimed_bytes || 0)}`);
  }

  async function legacyCampaignArchive(campaignId) {
    await request(`/api/evaluation-campaigns/${Number(campaignId)}/archive`, { method: "POST", body: "{}" });
    await loadCampaigns();
    await openCampaign(`v1:${Number(campaignId)}`);
  }

  async function legacyCampaignDiscard(campaignId) {
    if (!window.confirm("确认丢弃这个无投票的历史 Draft？此操作不可撤销。")) return;
    await request(`/api/evaluation-campaigns/${Number(campaignId)}/discard`, { method: "POST", body: "{}" });
    studioState.selectedCampaignKey = null;
    renderCampaignDetail(null);
    await loadCampaigns();
  }

  async function load() {
    await Promise.all([loadCampaigns(), loadCleanupRequests(), loadRunOutputs(), loadItemGroups()]);
    renderItemGroupOptions();
    renderItems();
    fillMethodSelects();
    renderCoverage();
  }

  async function prefillFromCompare(selection) {
    await load();
    const id = itemId(selection?.item);
    if (!id) throw new Error("Compare 没有有效的 Media Item");
    if (selection.groupId && String(selection.groupId) !== String(studioState.selectedGroupId)) {
      studioState.selectedGroupId = String(selection.groupId);
      studioState.selectedItemIds.clear();
      studioState.itemQuery = "";
      studioState.itemPage = 1;
      const query = el("studio-item-query");
      if (query) query.value = "";
      renderItemGroupOptions();
      await loadItems({ page: 1 });
    }
    if (!studioState.items.some((item) => itemId(item) === id)) {
      studioState.itemQuery = String(selection.item?.display_name || selection.item?.item_key || "");
      studioState.itemPage = 1;
      const query = el("studio-item-query");
      if (query) query.value = studioState.itemQuery;
      await loadItems({ page: 1 });
    }
    if (!studioState.items.some((item) => itemId(item) === id)) throw new Error("所选 GT Item 不在当前文件夹或已不可用");
    studioState.selectedItemIds = new Set([id]);
    renderItems();
    await loadMethodsForSelection();
    const predictions = selection.predictions || [];
    if (predictions.length !== 2) throw new Error("快速盲评需要恰好两份 Pred");
    for (const [index, prediction] of predictions.entries()) {
      const slot = index ? "b" : "a";
      const runId = Number(prediction.producer_run_id || prediction.run_id || 0);
      const key = runId ? `run:${runId}` : `external:${String(prediction.method_key || "")}`;
      const method = studioState.methods.find((row) => methodKey(row) === key);
      if (!method) throw new Error("所选 Pred 的方法没有完整覆盖该 Item");
      el(`studio-method-${slot}-source`).value = methodKind(method);
      fillMethodSelects();
      el(`studio-method-${slot}`).value = key;
    }
    const name = el("studio-wizard-form")?.elements?.name;
    if (name) name.value = `Blind · ${selection.item.display_name || selection.item.item_key || "Compare"}`;
    invalidateCoveragePreview();
    await previewCoverage();
  }

  function formatBytes(value) {
    let size = Number(value || 0);
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let unit = 0;
    while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
    return `${size.toFixed(unit ? 2 : 0)} ${units[unit]}`;
  }

  function renderStoragePreview(preview) {
    const host = el("storage-gc-content");
    if (!host) return;
    const summary = preview.summary || {};
    const runs = preview.runs || [];
    const caches = preview.caches || preview.entries || [];
    const blocked = preview.blocked || [];
    const eligibleRuns = runs.filter((row) => row.eligible !== false);
    const eligibleCaches = caches.filter((row) => row.eligible);
    host.innerHTML = `
      <div class="summary-grid"><div><span>历史 Run 残留</span><strong>${eligibleRuns.length}</strong></div><div><span>Run 空间</span><strong>${formatBytes(summary.run_bytes || eligibleRuns.reduce((sum, row) => sum + Number(row.size_bytes || row.reclaimable_bytes || 0), 0))}</strong></div><div><span>孤儿缓存</span><strong>${eligibleCaches.length}</strong></div><div><span>缓存空间</span><strong>${formatBytes(summary.eligible_bytes || summary.cache_bytes || eligibleCaches.reduce((sum, row) => sum + Number(row.size_bytes || 0), 0))}</strong></div></div>
      ${eligibleRuns.length ? `<section class="stats-block"><h3>可清理的历史 Run</h3><div class="coverage-table"><table><thead><tr><th>Run</th><th>状态</th><th>空间</th></tr></thead><tbody>${eligibleRuns.map((row) => `<tr><td>#${Number(row.run_id || row.id)} ${safe(row.name || "")}</td><td>${safe(row.reason || "已隐藏")}</td><td>${formatBytes(row.size_bytes || row.reclaimable_bytes || 0)}</td></tr>`).join("")}</tbody></table></div></section>` : ""}
      ${eligibleCaches.length ? `<section class="stats-block"><h3>无引用缓存</h3><div class="coverage-table"><table><thead><tr><th>类型</th><th>Key</th><th>空间</th></tr></thead><tbody>${eligibleCaches.map((row) => `<tr><td>${safe(row.cache_type)}</td><td><code>${safe(String(row.cache_key || "").slice(0, 18))}</code></td><td>${formatBytes(row.size_bytes)}</td></tr>`).join("")}</tbody></table></div></section>` : ""}
      ${blocked.length ? `<div class="message warn"><p>${blocked.length} 项因活动引用、lease 或 Campaign 保护被跳过。</p></div>` : ""}
      ${!eligibleRuns.length && !eligibleCaches.length ? "<p class=\"muted\">当前没有达到安全清理条件的条目。</p>" : ""}`;
    const button = el("storage-gc-run");
    if (button) button.disabled = !eligibleRuns.length && !eligibleCaches.length;
  }

  async function previewStorageGc() {
    studioState.storagePreview = await request("/api/storage/gc/preview");
    renderStoragePreview(studioState.storagePreview);
  }

  async function executeStorageGc() {
    const preview = studioState.storagePreview;
    if (!preview?.preview_token) throw new Error("请先生成有效的存储清理预览");
    if (!window.confirm("确认清理预览中所有符合条件的历史 Run 目录与孤儿缓存？")) return;
    const result = await request("/api/storage/gc", {
      method: "POST",
      body: JSON.stringify({ confirm: true, preview_token: preview.preview_token }),
    });
    notify(`存储清理完成，释放 ${formatBytes(result.reclaimed_bytes || 0)}`);
    await Promise.all([previewStorageGc(), loadRunOutputs(), loadCampaigns()]);
  }

  document.addEventListener("change", (event) => {
    if (event.target.matches?.("[data-objective-curve-item]")) {
      const detail = studioState.campaignDetail;
      const campaign = detail?.campaign || detail;
      const choices = objectiveCurveChoices(detail?.analysis, campaign)
        .filter((row) => row.itemId === Number(event.target.value));
      const preferred = choices.find((row) => row.completedMethodCount >= 2) || choices[0];
      if (preferred) {
        supersedeObjectiveCurveRequest();
        studioState.objectiveCurveSelection = {
          itemId: preferred.itemId,
          metricName: preferred.metricName,
        };
        renderCampaignDetail(detail);
      }
      return;
    }
    if (event.target.matches?.("[data-objective-curve-metric]")) {
      const current = studioState.objectiveCurveSelection;
      if (current) {
        supersedeObjectiveCurveRequest();
        studioState.objectiveCurveSelection = {
          itemId: Number(current.itemId),
          metricName: String(event.target.value || ""),
        };
        renderCampaignDetail(studioState.campaignDetail);
      }
      return;
    }
    if (event.target.matches?.("#studio-item-group")) {
      studioState.selectedGroupId = event.target.value || "";
      studioState.selectedItemIds.clear();
      studioState.itemQuery = "";
      studioState.itemPage = 1;
      const query = el("studio-item-query");
      if (query) query.value = "";
      invalidateCoveragePreview();
      loadItems({ page: 1 }).then(loadMethodsForSelection).catch((error) => notify(error.message));
      return;
    }
    const item = event.target.closest?.("[data-studio-item]");
    if (item) {
      const id = Number(item.dataset.studioItem);
      if (item.checked) studioState.selectedItemIds.add(id);
      else studioState.selectedItemIds.delete(id);
      renderItems();
      invalidateCoveragePreview();
      loadMethodsForSelection().catch((error) => notify(error.message));
      return;
    }
    if (event.target.matches?.("#studio-method-a-source, #studio-method-b-source")) {
      fillMethodSelects();
      invalidateCoveragePreview();
      return;
    }
    if (event.target.matches?.("#studio-allow-external-aspect-stretch")) {
      invalidateCoveragePreview();
      return;
    }
    if (event.target.matches?.("#studio-method-a, #studio-method-b")) invalidateCoveragePreview();
  });

  el("studio-item-query")?.addEventListener("input", (event) => {
    studioState.itemQuery = event.target.value || "";
    studioState.itemPage = 1;
    clearTimeout(studioState.itemQueryTimer);
    studioState.itemQueryTimer = setTimeout(() => loadItems({ page: 1 }).catch((error) => notify(error.message)), 250);
  });

  document.addEventListener("click", (event) => {
    const itemPage = event.target.closest?.("[data-studio-item-page]");
    if (itemPage) {
      loadItems({ page: Number(itemPage.dataset.studioItemPage || 1) }).catch((error) => notify(error.message));
      return;
    }
    const curveRetry = event.target.closest?.("[data-objective-curve-retry]");
    if (curveRetry) {
      const detail = studioState.campaignDetail;
      const campaign = detail?.campaign || detail;
      const selection = studioState.objectiveCurveSelection;
      if (campaign?.id && selection) {
        const key = objectiveCurveKey(
          campaign.id,
          selection,
          objectiveCurveFingerprint(detail?.analysis),
        );
        delete studioState.objectiveCurveErrors[key];
        renderCampaignDetail(detail);
      }
      return;
    }
    const campaign = event.target.closest?.("[data-studio-campaign]");
    if (campaign) return openCampaign(campaign.dataset.studioCampaign).catch((error) => notify(error.message));
    const copy = event.target.closest?.("[data-copy-share]");
    if (copy) return navigator.clipboard.writeText(copy.dataset.copyShare).then(() => notify("参与链接已复制"));
    const publish = event.target.closest?.("[data-studio-publish]");
    if (publish) return campaignAction("publish", publish.dataset.studioPublish).catch((error) => notify(error.message));
    const close = event.target.closest?.("[data-studio-close]");
    if (close) return campaignAction("close", close.dataset.studioClose).catch((error) => notify(error.message));
    const archive = event.target.closest?.("[data-studio-archive]");
    if (archive) return campaignAction("archive", archive.dataset.studioArchive).catch((error) => notify(error.message));
    const campaignDelete = event.target.closest?.("[data-studio-delete]");
    if (campaignDelete) return deleteCampaign(campaignDelete.dataset.studioDelete).catch((error) => notify(error.message));
    const cleanupRetry = event.target.closest?.("[data-evaluation-cleanup-retry]");
    if (cleanupRetry) return retryCleanupRequest(cleanupRetry.dataset.evaluationCleanupRetry).catch((error) => notify(error.message));
    const legacyArchive = event.target.closest?.("[data-studio-legacy-archive]");
    if (legacyArchive) return legacyCampaignArchive(legacyArchive.dataset.studioLegacyArchive).catch((error) => notify(error.message));
    const legacyDiscard = event.target.closest?.("[data-studio-legacy-discard]");
    if (legacyDiscard) return legacyCampaignDiscard(legacyDiscard.dataset.studioLegacyDiscard).catch((error) => notify(error.message));
  });

  window.VFIEvalStudio = {
    load,
    refresh: load,
    openCampaign,
    previewCoverage,
    createCampaign,
    prefillFromCompare,
  };

  el("studio-refresh")?.addEventListener("click", () => load().catch((error) => notify(error.message)));
  el("studio-preview")?.addEventListener("click", () => previewCoverage().catch((error) => notify(error.message)));
  el("studio-wizard-form")?.addEventListener("submit", (event) => createCampaign(event).catch((error) => notify(error.message)));
  el("storage-gc-preview")?.addEventListener("click", () => previewStorageGc().catch((error) => notify(error.message)));
  el("storage-gc-run")?.addEventListener("click", () => executeStorageGc().catch((error) => notify(error.message)));
  window.addEventListener("pagehide", stopPreparationPoll);
  window.addEventListener("pageshow", (event) => {
    if (!event.persisted || !studioState.selectedCampaignKey) return;
    openCampaign(studioState.selectedCampaignKey, false).catch((error) => notify(error.message));
  });
}());
