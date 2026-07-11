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
    preview: null,
    selectedCampaignKey: null,
    preparationPoll: null,
    preparationPollKey: null,
    campaignRequestGeneration: 0,
    itemRequestGeneration: 0,
    methodRequestGeneration: 0,
    previewGeneration: 0,
    itemQueryTimer: null,
    storagePreview: null,
  };

  const el = (id) => document.getElementById(id);

  async function request(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || response.statusText);
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

  function renderItems() {
    const host = el("studio-items");
    if (!host) return;
    if (!studioState.selectedGroupId) {
      host.innerHTML = "<p class=\"muted\">没有可用的 GT 文件夹。</p>";
      return;
    }
    if (!studioState.items.length) {
      host.innerHTML = "<p class=\"muted\">当前文件夹中没有匹配的 GT Media Item。</p>";
      return;
    }
    host.innerHTML = `
      <div class="coverage-summary"><strong>已选 ${studioState.selectedItemIds.size} 个视频</strong><span class="muted">每个 Item 都绑定精确 canonical GT；搜索不会取消已选视频。</span></div>
      <div class="studio-item-grid">${studioState.items.map((item) => {
        const id = itemId(item);
        return `<label class="studio-item-card ${studioState.selectedItemIds.has(id) ? "selected" : ""}">
          <input type="checkbox" data-studio-item="${id}" ${studioState.selectedItemIds.has(id) ? "checked" : ""}>
          <span><strong>${safe(item.display_name || item.item_key || `Media Item #${id}`)}</strong><small>${safe(itemMetadata(item))}</small><small>GT asset #${Number(item.canonical_gt_asset_id || 0)}</small></span>
        </label>`;
      }).join("")}</div>`;
  }

  async function loadItemGroups() {
    const payload = await request("/api/media/item-groups?role=gt");
    studioState.itemGroups = payload.groups || payload.item_groups || [];
    const ids = new Set(studioState.itemGroups.map(groupId));
    if (!ids.has(String(studioState.selectedGroupId))) {
      studioState.selectedGroupId = groupId(studioState.itemGroups[0]);
      studioState.selectedItemIds.clear();
    }
    renderItemGroupOptions();
    await loadItems();
  }

  async function loadItems() {
    const generation = ++studioState.itemRequestGeneration;
    if (!studioState.selectedGroupId) {
      studioState.items = [];
      renderItems();
      await loadMethodsForSelection();
      return;
    }
    const path = `/api/media/items?group_id=${encodeURIComponent(studioState.selectedGroupId)}&q=${encodeURIComponent(studioState.itemQuery)}&page=1&page_size=200`;
    const first = await request(path);
    const pageCount = Math.max(1, Number(first.page_count || first.total_pages || 1));
    const pages = pageCount > 1
      ? await Promise.all(Array.from({ length: pageCount - 1 }, (_row, index) => request(`/api/media/items?group_id=${encodeURIComponent(studioState.selectedGroupId)}&q=${encodeURIComponent(studioState.itemQuery)}&page=${index + 2}&page_size=200`)))
      : [];
    if (generation !== studioState.itemRequestGeneration) return;
    const byId = new Map();
    for (const item of [first, ...pages].flatMap((page) => page.items || [])) byId.set(itemId(item), item);
    studioState.items = Array.from(byId.values());
    if (!studioState.itemQuery) {
      for (const id of Array.from(studioState.selectedItemIds)) {
        if (!byId.has(Number(id))) studioState.selectedItemIds.delete(Number(id));
      }
    }
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
    if (campaign.archived) return "archived";
    return campaign.preparation_status || campaign.status || "draft";
  }

  function campaignKey(campaign) {
    return String(campaign.campaign_key || `${Number(campaign.schema_version || 1) >= 2 ? "v2" : "v1"}:${Number(campaign.id)}`);
  }

  function renderCampaignList() {
    const host = el("studio-campaign-list");
    if (!host) return;
    if (!studioState.campaigns.length) {
      host.innerHTML = "<p class=\"muted\">还没有 Campaign。</p>";
      return;
    }
    host.innerHTML = studioState.campaigns.map((campaign) => `
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

  function isLoopbackOrigin() {
    const host = String(location.hostname || "").toLowerCase();
    return host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0" || host === "::1" || host === "[::1]";
  }

  function renderCampaignDetail(payload) {
    const host = el("studio-campaign-detail");
    if (!host) return;
    const campaign = payload?.campaign || payload;
    if (!campaign?.id) {
      host.innerHTML = "<p class=\"muted\">选择 Campaign 查看准备进度、分享链接和结果。</p>";
      return;
    }
    const progress = payload.preparation || campaign.preparation || {};
    const coverage = payload.coverage || campaign.coverage || {};
    const analysis = payload.analysis || null;
    const ranking = analysisRanking(analysis);
    const shareUrl = participantShareUrl(payload.share_url || campaign.share_url, campaign);
    const legacy = Number(campaign.schema_version || 1) < 2;
    const objective = objectiveMetrics(analysis, legacy);
    host.innerHTML = `
      <div class="studio-detail-head"><div><p class="eyebrow">${legacy ? "历史 Campaign" : "Campaign V2"}</p><h3>${safe(campaign.name)}</h3><p class="muted">${safe(campaign.public_title || campaign.metadata?.public_title || "匿名视频质量评测")}</p></div><span class="studio-status">${safe(campaignStatus(campaign))}</span></div>
      ${legacy ? "<div class=\"message warn\"><p>旧 schema v1 Campaign 只读保留；可继续导出，不会按标签猜测迁移。</p></div>" : ""}
      ${progress.total ? `<div class="studio-progress"><progress max="${Number(progress.total)}" value="${Number(progress.current || 0)}"></progress><span>${Number(progress.current || 0)}/${Number(progress.total)} · ${safe(progress.phase || campaignStatus(campaign))}</span></div>` : ""}
      ${progress.error?.message || campaign.preparation_error?.message ? `<div class="message error"><p>${safe(progress.error?.message || campaign.preparation_error?.message)}</p></div>` : ""}
      <div class="summary-grid"><div><span>视频</span><strong>${Number(coverage.items || campaign.item_count || 0)}</strong></div><div><span>任务</span><strong>${Number(coverage.tasks || campaign.task_count || 0)}</strong></div><div><span>投票</span><strong>${Number(coverage.votes || campaign.vote_count || 0)}</strong></div><div><span>目标票数/视频</span><strong>${Number(campaign.target_votes || 0)}</strong></div></div>
      ${shareUrl ? `<label class="studio-share"><span>参与链接</span><div><input readonly value="${safe(shareUrl)}"><button data-copy-share="${safe(shareUrl)}" type="button">复制</button></div></label>` : ""}
      ${shareUrl && isLoopbackOrigin() ? `<div class="message warn studio-share-warning"><p>当前链接使用本机回环地址，只能在这台电脑上打开。若要让受控内网参与者访问，请以 <code>--host 0.0.0.0</code> 启动服务，再从 <code>http://&lt;服务器内网 IP&gt;:8765</code> 打开 Studio 并重新复制链接；不要将服务暴露到公网。</p></div>` : ""}
      <div class="studio-actions">${!legacy && ["draft", "failed"].includes(campaign.status) ? `<button data-studio-publish="${Number(campaign.id)}" type="button">${campaign.status === "failed" ? "重试发布" : "发布并冻结"}</button>` : ""}${!legacy && campaign.status === "published" ? `<button data-studio-close="${Number(campaign.id)}" type="button">关闭</button>` : ""}${!legacy && ["closed", "failed"].includes(campaign.status) ? `<button class="secondary" data-studio-archive="${Number(campaign.id)}" type="button">归档</button>` : ""}${legacy && !campaign.archived ? `<button class="secondary" data-studio-legacy-archive="${Number(campaign.id)}" type="button">归档</button>` : ""}${legacy && campaign.status === "draft" && Number(campaign.vote_count || campaign.votes || 0) === 0 ? `<button class="danger secondary" data-studio-legacy-discard="${Number(campaign.id)}" type="button">丢弃空 Draft</button>` : ""}${legacy ? `<a class="secondary button-link" href="/api/evaluation-campaigns/${Number(campaign.id)}/export">导出</a>` : `<a class="secondary button-link" href="/api/evaluation-campaigns/v2/${Number(campaign.id)}/export">导出</a>`}</div>
      ${ranking.length ? `<section class="stats-block studio-human-results" data-analysis-section="human"><h3>人工排名</h3><div class="coverage-table"><table><thead><tr><th>排名</th><th>方法</th><th>得分</th><th>95% CI</th></tr></thead><tbody>${ranking.map((row, index) => `<tr><td>${index + 1}</td><td>${safe(row.label || row.name)}</td><td>${Number(row.score || 0).toFixed(4)}</td><td>${safe((row.ci95 || []).join(" – ") || "-")}</td></tr>`).join("")}</tbody></table></div></section>` : ""}
      ${!legacy && objective.length ? `<section class="stats-block studio-objective-results" data-analysis-section="objective"><div class="studio-analysis-head"><h3>客观指标</h3><p class="muted">客观指标与人工排名分别展示，不生成合成总分。</p></div><div class="coverage-table"><table><thead><tr><th>指标</th><th>方法</th><th>方向</th><th>状态</th><th>有效样本</th><th>均值 / 中位数</th><th>P10 / P90</th></tr></thead><tbody>${objective.map((row) => `<tr><td>${safe(row.metric_name || "-")}</td><td>${safe(row.method_label || "-")}</td><td>${safe(row.direction || "-")}</td><td>${objectiveMetricStatuses(row.status_counts)}</td><td>${Number(row.count || 0)}</td><td>${objectiveMetricNumber(row.mean)} / ${objectiveMetricNumber(row.median)}</td><td>${objectiveMetricNumber(row.p10)} / ${objectiveMetricNumber(row.p90)}</td></tr>`).join("")}</tbody></table></div></section>` : ""}`;
  }

  async function loadCampaigns() {
    const payload = await request("/api/evaluation-campaigns");
    studioState.campaigns = payload.campaigns || [];
    if (studioState.selectedCampaignKey
        && !studioState.campaigns.some((campaign) => campaignKey(campaign) === studioState.selectedCampaignKey)) {
      studioState.selectedCampaignKey = null;
      renderCampaignDetail(null);
    }
    renderCampaignList();
    renderPackages();
    if (studioState.selectedCampaignKey) await openCampaign(studioState.selectedCampaignKey, false);
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
    if (studioState.preparationPoll && studioState.preparationPollKey !== requestedKey) {
      clearInterval(studioState.preparationPoll);
      studioState.preparationPoll = null;
      studioState.preparationPollKey = null;
    }
    studioState.selectedCampaignKey = requestedKey;
    const campaign = studioState.campaigns.find((row) => campaignKey(row) === requestedKey);
    const campaignId = Number(campaign?.id || requestedKey.split(":").pop());
    const isV2 = requestedKey.startsWith("v2:") || Number(campaign?.schema_version || 1) >= 2;
    const payload = await request(isV2 ? `/api/evaluation-campaigns/v2/${campaignId}` : `/api/evaluation-campaigns/${campaignId}`);
    if (requestGeneration !== studioState.campaignRequestGeneration || studioState.selectedCampaignKey !== requestedKey) return;
    renderCampaignDetail(payload);
    if (rerenderList) renderCampaignList();
    if (["requested", "queued", "preparing", "running"].includes(campaignStatus(payload.campaign || payload))) startPreparationPoll(requestedKey, campaignId);
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
    await request(`/api/evaluation-campaigns/v2/${campaignId}/publish`, { method: "POST", body: "{}" });
    studioState.selectedCampaignKey = `v2:${campaignId}`;
    studioState.preview = null;
    renderCoverage();
    await loadCampaigns();
    startPreparationPoll(`v2:${campaignId}`, campaignId);
    notify("Campaign 已进入规范化、校验与冻结队列");
  }

  function startPreparationPoll(key, campaignId) {
    if (studioState.preparationPoll) clearInterval(studioState.preparationPoll);
    studioState.preparationPollKey = key;
    studioState.preparationPoll = setInterval(async () => {
      if (studioState.selectedCampaignKey !== key) {
        clearInterval(studioState.preparationPoll);
        studioState.preparationPoll = null;
        studioState.preparationPollKey = null;
        return;
      }
      try {
        const payload = await request(`/api/evaluation-campaigns/v2/${Number(campaignId)}`);
        if (studioState.selectedCampaignKey !== key) return;
        renderCampaignDetail(payload);
        if (!["requested", "queued", "preparing", "running"].includes(campaignStatus(payload.campaign || payload))) {
          clearInterval(studioState.preparationPoll);
          studioState.preparationPoll = null;
          studioState.preparationPollKey = null;
          await loadCampaigns();
        }
      } catch (_error) {
        // Keep the last useful state; the next poll can recover.
      }
    }, 2000);
  }

  async function campaignAction(action, campaignId) {
    await request(`/api/evaluation-campaigns/v2/${Number(campaignId)}/${action}`, { method: "POST", body: "{}" });
    await loadCampaigns();
    await openCampaign(`v2:${Number(campaignId)}`);
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
    await Promise.all([loadCampaigns(), loadRunOutputs(), loadItemGroups()]);
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
      const query = el("studio-item-query");
      if (query) query.value = "";
      renderItemGroupOptions();
      await loadItems();
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
    if (event.target.matches?.("#studio-item-group")) {
      studioState.selectedGroupId = event.target.value || "";
      studioState.selectedItemIds.clear();
      studioState.itemQuery = "";
      const query = el("studio-item-query");
      if (query) query.value = "";
      invalidateCoveragePreview();
      loadItems().then(loadMethodsForSelection).catch((error) => notify(error.message));
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
    clearTimeout(studioState.itemQueryTimer);
    studioState.itemQueryTimer = setTimeout(() => loadItems().catch((error) => notify(error.message)), 250);
  });

  document.addEventListener("click", (event) => {
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
    const legacyArchive = event.target.closest?.("[data-studio-legacy-archive]");
    if (legacyArchive) return legacyCampaignArchive(legacyArchive.dataset.studioLegacyArchive).catch((error) => notify(error.message));
    const legacyDiscard = event.target.closest?.("[data-studio-legacy-discard]");
    if (legacyDiscard) return legacyCampaignDiscard(legacyDiscard.dataset.studioLegacyDiscard).catch((error) => notify(error.message));
  });

  window.VFIEvalStudio = { load, refresh: load, previewCoverage, createCampaign, prefillFromCompare };

  el("studio-refresh")?.addEventListener("click", () => load().catch((error) => notify(error.message)));
  el("studio-preview")?.addEventListener("click", () => previewCoverage().catch((error) => notify(error.message)));
  el("studio-wizard-form")?.addEventListener("submit", (event) => createCampaign(event).catch((error) => notify(error.message)));
  el("storage-gc-preview")?.addEventListener("click", () => previewStorageGc().catch((error) => notify(error.message)));
  el("storage-gc-run")?.addEventListener("click", () => executeStorageGc().catch((error) => notify(error.message)));
}());
