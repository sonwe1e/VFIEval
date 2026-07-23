(function () {
  const Shared = window.VFIEvalShared;
  if (!Shared) throw new Error("VFIEval shared frontend primitives failed to load");

  const CAMPAIGN_DRAFT_KEY = "vfieval:campaign-draft:v1";
  const CAMPAIGN_DRAFT_VERSION = 1;
  const MAX_DRAFT_RECONCILE_PAGES = 50;
  const BULK_ITEM_PAGE_SIZE = 200;
  const METHOD_ITEM_CHUNK_SIZE = 80;
  const PREPARATION_POLL_BASE_MS = 2000;
  const PREPARATION_POLL_MAX_MS = 30000;

  const studioState = {
    campaigns: [],
    campaignQuery: "",
    campaignStatus: "",
    campaignPage: 1,
    campaignPageSize: 30,
    campaignPageCount: 1,
    campaignTotal: 0,
    campaignListRequestGeneration: 0,
    campaignQueryTimer: null,
    packageCampaigns: [],
    packagePage: 1,
    packagePageSize: 30,
    packagePageCount: 1,
    packageTotal: 0,
    packageRequestGeneration: 0,
    runOutputs: [],
    itemGroups: [],
    items: [],
    methods: [],
    selectedGroupId: "",
    selectedItemIds: new Set(),
    itemSelectionToken: "",
    itemSelectionTokenTotal: 0,
    itemSelectionTokenExpiresAt: 0,
    itemQuery: "",
    itemPage: 1,
    itemPageSize: 100,
    itemPageCount: 1,
    itemTotal: 0,
    itemCache: new Map(),
    itemOnlySelected: false,
    selectedItemPage: 1,
    itemBulkController: null,
    itemBulkProgress: null,
    draftHydrated: false,
    draftRestoring: false,
    draftSaveTimer: null,
    draftNotice: "",
    draftCommitted: false,
    preview: null,
    selectedCampaignKey: null,
    preparationPoll: null,
    preparationPollKey: null,
    preparationPollGeneration: 0,
    preparationLastSuccessAt: 0,
    preparationPollFailures: 0,
    preparationPollError: "",
    preparationNextDelayMs: PREPARATION_POLL_BASE_MS,
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
    campaignSubmitting: false,
    campaignSubmitPhase: "",
    campaignSubmitError: "",
    campaignSubmissionId: "",
    campaignSubmissionFingerprint: "",
  };

  const el = (id) => document.getElementById(id);
  const campaignCreationFlight = Shared.createSingleFlight();

  function reportRequestFailure(error, context = "Evaluation Studio 请求") {
    if (typeof window.showRequestDiagnostic === "function") {
      window.showRequestDiagnostic(error, context);
    }
  }

  async function request(path, options = {}) {
    const { suppressDiagnostic = false, ...fetchOptions } = options;
    return Shared.request(path, {
      fetchOptions,
      suppressDiagnostic,
      networkMessage: "无法连接 VFIEval 服务",
      onDiagnostic: (error) => reportRequestFailure(error),
    });
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

  function usesItemSelectionToken() {
    return Boolean(studioState.itemSelectionToken);
  }

  function selectedItemCount() {
    return usesItemSelectionToken()
      ? Number(studioState.itemSelectionTokenTotal || 0)
      : studioState.selectedItemIds.size;
  }

  function itemIsSelected(id) {
    return usesItemSelectionToken() || studioState.selectedItemIds.has(Number(id));
  }

  function campaignItemSelectionPayload() {
    if (usesItemSelectionToken()) {
      return { selection_token: studioState.itemSelectionToken };
    }
    return { media_item_ids: selectedItemIds() };
  }

  function campaignItemSelectionSignature() {
    return JSON.stringify(campaignItemSelectionPayload());
  }

  function resetCampaignSubmissionIdentity() {
    studioState.campaignSubmissionId = "";
    studioState.campaignSubmissionFingerprint = "";
  }

  function campaignSubmissionIdFor(payload) {
    const fingerprint = JSON.stringify(payload);
    if (!studioState.campaignSubmissionId
        || studioState.campaignSubmissionFingerprint !== fingerprint) {
      studioState.campaignSubmissionId = Shared.createSubmissionId("studio");
      studioState.campaignSubmissionFingerprint = fingerprint;
    }
    return studioState.campaignSubmissionId;
  }

  function campaignDraftPayload() {
    const form = el("studio-wizard-form");
    return {
      version: CAMPAIGN_DRAFT_VERSION,
      saved_at: new Date().toISOString(),
      group_id: String(studioState.selectedGroupId || ""),
      item_ids: selectedItemIds(),
      selection_token: String(studioState.itemSelectionToken || ""),
      selection_total: Number(studioState.itemSelectionTokenTotal || 0),
      selection_expires_at: Number(studioState.itemSelectionTokenExpiresAt || 0),
      item_query: String(studioState.itemQuery || ""),
      only_selected: Boolean(studioState.itemOnlySelected),
      name: String(form?.elements?.name?.value || ""),
      public_title: String(form?.elements?.public_title?.value || ""),
      target_votes: Number(form?.elements?.target_votes?.value || 3),
      method_a_source: methodSourceMode("a"),
      method_a_key: String(el("studio-method-a")?.value || ""),
      method_b_source: methodSourceMode("b"),
      method_b_key: String(el("studio-method-b")?.value || ""),
      allow_external_aspect_stretch: Boolean(el("studio-allow-external-aspect-stretch")?.checked),
    };
  }

  function readCampaignDraft() {
    const parsed = Shared.storageJsonGet(CAMPAIGN_DRAFT_KEY, null);
    if (!parsed) return null;
    if (Number(parsed.version) !== CAMPAIGN_DRAFT_VERSION) {
      Shared.storageRemove(CAMPAIGN_DRAFT_KEY);
      return null;
    }
    return parsed;
  }

  function saveCampaignDraft() {
    if (studioState.draftRestoring || studioState.draftCommitted) return;
    if (!Shared.storageJsonSet(CAMPAIGN_DRAFT_KEY, campaignDraftPayload())) {
      studioState.draftNotice = "浏览器无法保存 Campaign 草稿；本次填写只在当前页面保留。";
      updateItemSelectionUi();
    }
  }

  function queueCampaignDraftSave() {
    if (studioState.draftRestoring || studioState.draftCommitted) return;
    clearTimeout(studioState.draftSaveTimer);
    studioState.draftSaveTimer = setTimeout(saveCampaignDraft, 120);
  }

  function markCampaignDraftDirty() {
    studioState.draftCommitted = false;
    queueCampaignDraftSave();
  }

  function clearCampaignDraft() {
    clearTimeout(studioState.draftSaveTimer);
    studioState.draftSaveTimer = null;
    studioState.draftNotice = "";
    studioState.draftCommitted = true;
    Shared.storageRemove(CAMPAIGN_DRAFT_KEY);
  }

  function primeCampaignDraft(draft) {
    if (!draft) return;
    studioState.draftRestoring = true;
    studioState.selectedGroupId = String(draft.group_id || "");
    studioState.selectedItemIds = new Set(
      (Array.isArray(draft.item_ids) ? draft.item_ids : [])
        .map(Number)
        .filter((id) => Number.isInteger(id) && id > 0),
    );
    studioState.itemSelectionToken = String(draft.selection_token || "");
    studioState.itemSelectionTokenTotal = Math.max(0, Number(draft.selection_total || 0));
    studioState.itemSelectionTokenExpiresAt = Math.max(
      0,
      Number(draft.selection_expires_at || 0),
    );
    studioState.itemQuery = String(draft.item_query || "");
    studioState.itemOnlySelected = Boolean(draft.only_selected)
      && !studioState.itemSelectionToken;
    studioState.selectedItemPage = 1;
    const query = el("studio-item-query");
    if (query) query.value = studioState.itemQuery;
  }

  function applyCampaignDraftControls(draft) {
    if (!draft) return [];
    const reconciled = [];
    const form = el("studio-wizard-form");
    if (form?.elements?.name) form.elements.name.value = String(draft.name || "");
    if (form?.elements?.public_title) {
      form.elements.public_title.value = String(draft.public_title || "匿名视频质量评测");
    }
    if (form?.elements?.target_votes) {
      const requestedVotes = Number(draft.target_votes);
      form.elements.target_votes.value = String(
        Number.isFinite(requestedVotes) ? Math.min(1000, Math.max(1, requestedVotes)) : 3,
      );
    }
    for (const slot of ["a", "b"]) {
      const source = el(`studio-method-${slot}-source`);
      const select = el(`studio-method-${slot}`);
      const sourceValue = draft[`method_${slot}_source`] === "external" ? "external" : "run";
      const requestedKey = String(draft[`method_${slot}_key`] || "");
      if (source) source.value = sourceValue;
      fillMethodSelects();
      if (select && requestedKey && [...select.options].some((option) => option.value === requestedKey)) {
        select.value = requestedKey;
      } else if (requestedKey) {
        reconciled.push(`方法 ${slot.toUpperCase()} 已不可用`);
      }
    }
    const allowStretch = el("studio-allow-external-aspect-stretch");
    if (allowStretch) allowStretch.checked = Boolean(draft.allow_external_aspect_stretch);
    return reconciled;
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
    if (!studioState.campaignSubmitting) studioState.campaignSubmitError = "";
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

  function cacheItems(items) {
    for (const item of items || []) {
      const id = itemId(item);
      if (id) studioState.itemCache.set(id, item);
    }
  }

  function selectedCachedItems() {
    return selectedItemIds()
      .map((id) => studioState.itemCache.get(id))
      .filter(Boolean)
      .sort((left, right) => String(left.display_name || left.item_key || "").localeCompare(
        String(right.display_name || right.item_key || ""),
      ));
  }

  function itemView() {
    if (!studioState.itemOnlySelected) {
      return {
        items: studioState.items,
        page: Math.max(1, Number(studioState.itemPage || 1)),
        pageCount: Math.max(1, Number(studioState.itemPageCount || 1)),
        total: Math.max(0, Number(studioState.itemTotal || 0)),
      };
    }
    if (usesItemSelectionToken()) {
      return {
        items: studioState.items,
        page: Math.max(1, Number(studioState.selectedItemPage || 1)),
        pageCount: Math.max(1, Number(studioState.itemPageCount || 1)),
        total: selectedItemCount(),
      };
    }
    const allItems = selectedCachedItems();
    const pageCount = Math.max(1, Math.ceil(allItems.length / studioState.itemPageSize));
    const page = Math.min(pageCount, Math.max(1, Number(studioState.selectedItemPage || 1)));
    studioState.selectedItemPage = page;
    const start = (page - 1) * studioState.itemPageSize;
    return {
      items: allItems.slice(start, start + studioState.itemPageSize),
      page,
      pageCount,
      total: selectedItemCount(),
    };
  }

  function itemPagerMarkup(view = itemView()) {
    return `<div class="pager" data-studio-item-pager>
      <span class="muted" data-studio-pager-summary>${studioState.itemOnlySelected ? "仅看已选" : "当前筛选"} · 共 ${Number(view.total || 0)} 个 Item · 第 ${view.page}/${view.pageCount} 页</span>
      <button class="secondary" data-studio-item-page="${view.page - 1}" type="button" ${view.page <= 1 ? "disabled" : ""}>上一页</button>
      <button class="secondary" data-studio-item-page="${view.page + 1}" type="button" ${view.page >= view.pageCount ? "disabled" : ""}>下一页</button>
    </div>`;
  }

  function itemCardsMarkup(items) {
    if (!items.length) {
      return `<p class="muted">${studioState.itemOnlySelected ? "尚未选择任何视频。" : "当前文件夹中没有匹配的 GT Media Item。"}</p>`;
    }
    return items.map((item) => {
      const id = itemId(item);
      return `<label class="studio-item-card ${itemIsSelected(id) ? "selected" : ""}">
        <input type="checkbox" data-studio-item="${id}" ${itemIsSelected(id) ? "checked" : ""}>
        <span><strong>${safe(item.display_name || item.item_key || `Media Item #${id}`)}</strong><small>${safe(itemMetadata(item))}</small><small>GT asset #${Number(item.canonical_gt_asset_id || 0)}</small></span>
      </label>`;
    }).join("");
  }

  function createItemCardElement(item) {
    const id = itemId(item);
    const label = document.createElement("label");
    label.className = `studio-item-card${itemIsSelected(id) ? " selected" : ""}`;
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.studioItem = String(id);
    input.checked = itemIsSelected(id);
    const body = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = String(item.display_name || item.item_key || `Media Item #${id}`);
    const metadata = document.createElement("small");
    metadata.textContent = itemMetadata(item);
    const asset = document.createElement("small");
    asset.textContent = `GT asset #${Number(item.canonical_gt_asset_id || 0)}`;
    body.append(title, metadata, asset);
    label.append(input, body);
    return label;
  }

  function bulkSelectionStatus() {
    const progress = studioState.itemBulkProgress;
    if (!progress) return "";
    if (progress.error) return progress.error;
    if (progress.message) return progress.message;
    return `正在读取筛选结果：${Number(progress.page || 0)}/${Number(progress.pageCount || 0)} 页 · 已发现 ${Number(progress.found || 0)} 个 Item`;
  }

  function renderItems() {
    const host = el("studio-items");
    if (!host) return;
    if (!studioState.selectedGroupId) {
      host.innerHTML = "<p class=\"muted\">没有可用的 GT 文件夹。</p>";
      return;
    }
    const view = itemView();
    host.innerHTML = `
      <div class="studio-item-toolbar">
        <div><strong data-studio-selected-count>已选 ${selectedItemCount()} 个视频</strong><span class="muted">每个 Item 都绑定精确 canonical GT；翻页和搜索不会取消已选视频。</span></div>
        <div class="studio-item-actions">
          <button class="secondary" data-studio-select-page type="button" ${studioState.itemOnlySelected || !studioState.items.length ? "disabled" : ""}>选择当前页</button>
          <button class="secondary" data-studio-select-filtered type="button" ${studioState.itemOnlySelected || !studioState.itemTotal || studioState.itemBulkController || usesItemSelectionToken() ? "disabled" : ""}>${usesItemSelectionToken() ? "已全选当前筛选" : "选择全部筛选结果"}</button>
          <button class="secondary" data-studio-clear-selection type="button" ${selectedItemCount() ? "" : "disabled"}>清空选择</button>
          <button class="secondary" data-studio-only-selected type="button" aria-pressed="${studioState.itemOnlySelected ? "true" : "false"}">${studioState.itemOnlySelected ? "显示筛选结果" : "仅看已选"}</button>
        </div>
        <div class="studio-bulk-progress" data-studio-bulk-progress ${studioState.itemBulkProgress ? "" : "hidden"}>
          <span role="status" aria-live="polite">${safe(bulkSelectionStatus())}</span>
          <button class="secondary" data-studio-cancel-bulk type="button" ${studioState.itemBulkController ? "" : "hidden"}>取消</button>
        </div>
        <p class="studio-draft-notice muted" data-studio-draft-notice ${studioState.draftNotice ? "" : "hidden"}>${safe(studioState.draftNotice)}</p>
      </div>
      <div class="studio-item-grid" data-studio-item-grid>${itemCardsMarkup(view.items)}</div>
      ${itemPagerMarkup(view)}`;
  }

  function updateItemSelectionUi() {
    const host = el("studio-items");
    if (!host) return;
    const count = host.querySelector("[data-studio-selected-count]");
    if (count) count.textContent = `已选 ${selectedItemCount()} 个视频`;
    for (const input of host.querySelectorAll("[data-studio-item]")) {
      const selected = itemIsSelected(Number(input.dataset.studioItem));
      input.checked = selected;
      input.closest(".studio-item-card")?.classList.toggle("selected", selected);
      if (studioState.itemOnlySelected && !selected) input.closest(".studio-item-card")?.remove();
    }
    if (studioState.itemOnlySelected) {
      const grid = host.querySelector("[data-studio-item-grid]");
      if (grid && !grid.querySelector(".studio-item-card")) {
        const visible = itemView().items;
        if (visible.length) {
          grid.replaceChildren(...visible.map(createItemCardElement));
        } else {
          const empty = document.createElement("p");
          empty.className = "muted";
          empty.textContent = "尚未选择任何视频。";
          grid.replaceChildren(empty);
        }
      }
      const view = itemView();
      const pager = host.querySelector("[data-studio-item-pager]");
      const summary = pager?.querySelector("[data-studio-pager-summary]");
      if (summary) summary.textContent = `仅看已选 · 共 ${view.total} 个 Item · 第 ${view.page}/${view.pageCount} 页`;
      const buttons = pager?.querySelectorAll("[data-studio-item-page]") || [];
      if (buttons[0]) {
        buttons[0].dataset.studioItemPage = String(view.page - 1);
        buttons[0].disabled = view.page <= 1;
      }
      if (buttons[1]) {
        buttons[1].dataset.studioItemPage = String(view.page + 1);
        buttons[1].disabled = view.page >= view.pageCount;
      }
    }
    const clear = host.querySelector("[data-studio-clear-selection]");
    if (clear) clear.disabled = !selectedItemCount();
    const selectPage = host.querySelector("[data-studio-select-page]");
    if (selectPage) selectPage.disabled = studioState.itemOnlySelected || !studioState.items.length;
    const selectFiltered = host.querySelector("[data-studio-select-filtered]");
    if (selectFiltered) {
      selectFiltered.disabled = studioState.itemOnlySelected
        || !studioState.itemTotal
        || Boolean(studioState.itemBulkController)
        || usesItemSelectionToken();
      selectFiltered.textContent = usesItemSelectionToken()
        ? "已全选当前筛选"
        : "选择全部筛选结果";
    }
    const progress = host.querySelector("[data-studio-bulk-progress]");
    if (progress) {
      progress.hidden = !studioState.itemBulkProgress;
      const status = progress.querySelector("[role=status]");
      if (status) status.textContent = bulkSelectionStatus();
      const cancel = progress.querySelector("[data-studio-cancel-bulk]");
      if (cancel) cancel.hidden = !studioState.itemBulkController;
    }
    const notice = host.querySelector("[data-studio-draft-notice]");
    if (notice) {
      notice.hidden = !studioState.draftNotice;
      notice.textContent = studioState.draftNotice;
    }
    renderCampaignSubmissionState();
  }

  function cancelBulkItemSelection() {
    if (!studioState.itemBulkController) return;
    studioState.itemBulkController.abort();
    studioState.itemBulkController = null;
    studioState.itemBulkProgress = {
      ...(studioState.itemBulkProgress || {}),
      error: "已取消选择全部筛选结果；原有选择保持不变。",
    };
    updateItemSelectionUi();
  }

  function clearItemSelectionToken({ preserveVisible = false, notice = "" } = {}) {
    if (!usesItemSelectionToken()) return;
    studioState.selectedItemIds = preserveVisible
      ? new Set(studioState.items.map(itemId).filter((id) => id > 0))
      : new Set();
    studioState.itemSelectionToken = "";
    studioState.itemSelectionTokenTotal = 0;
    studioState.itemSelectionTokenExpiresAt = 0;
    studioState.selectedItemPage = 1;
    if (notice) studioState.draftNotice = notice;
  }

  async function fetchItemPage(page, options = {}) {
    const pageSize = Math.min(BULK_ITEM_PAGE_SIZE, Math.max(1, Number(options.pageSize || studioState.itemPageSize)));
    const path = `/api/media/items?group_id=${encodeURIComponent(studioState.selectedGroupId)}&q=${encodeURIComponent(options.query ?? studioState.itemQuery)}&page=${Math.max(1, Number(page || 1))}&page_size=${pageSize}`;
    return request(path, options.signal ? { signal: options.signal } : {});
  }

  async function fetchSelectionTokenPage(page, options = {}) {
    if (!usesItemSelectionToken()) throw new Error("筛选选择令牌已失效");
    const pageSize = Math.min(
      BULK_ITEM_PAGE_SIZE,
      Math.max(1, Number(options.pageSize || studioState.itemPageSize)),
    );
    const path = `/api/media/item-selections/${encodeURIComponent(studioState.itemSelectionToken)}?page=${Math.max(1, Number(page || 1))}&page_size=${pageSize}`;
    return request(path, options.signal ? { signal: options.signal } : {});
  }

  async function loadItemGroups() {
    const payload = await request("/api/media/item-groups?role=gt");
    studioState.itemGroups = payload.groups || payload.item_groups || [];
    const ids = new Set(studioState.itemGroups.map(groupId));
    if (!ids.has(String(studioState.selectedGroupId))) {
      if (studioState.selectedGroupId) studioState.draftNotice = "草稿中的 GT 文件夹已不可用，已切换到当前首个可用文件夹。";
      studioState.selectedGroupId = groupId(studioState.itemGroups[0]);
      studioState.selectedItemIds.clear();
      clearItemSelectionToken();
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
    const tokenPage = usesItemSelectionToken() && studioState.itemOnlySelected;
    const requestedPage = Math.max(
      1,
      Number(
        options.page
        || (tokenPage ? studioState.selectedItemPage : studioState.itemPage)
        || 1,
      ),
    );
    const payload = tokenPage
      ? await fetchSelectionTokenPage(requestedPage)
      : await fetchItemPage(requestedPage);
    if (generation !== studioState.itemRequestGeneration) return;
    const pageCount = Math.max(1, Number(payload.page_count || payload.total_pages || 1));
    if (requestedPage > pageCount) return loadItems({ page: pageCount });
    studioState.items = payload.items || [];
    if (tokenPage) studioState.selectedItemPage = Math.max(1, Number(payload.page || requestedPage));
    else studioState.itemPage = Math.max(1, Number(payload.page || requestedPage));
    studioState.itemPageCount = pageCount;
    if (tokenPage) {
      studioState.itemSelectionTokenTotal = Math.max(0, Number(payload.total || 0));
      studioState.itemSelectionTokenExpiresAt = Number(payload.expires_at || 0);
    } else {
      studioState.itemTotal = Math.max(0, Number(payload.total || 0));
    }
    cacheItems(studioState.items);
    renderItems();
  }

  function currentItemFilterSignature() {
    return JSON.stringify([String(studioState.selectedGroupId), String(studioState.itemQuery)]);
  }

  async function applyItemSelectionChange() {
    resetCampaignSubmissionIdentity();
    updateItemSelectionUi();
    invalidateCoveragePreview();
    markCampaignDraftDirty();
    await loadMethodsForSelection();
  }

  async function selectCurrentItemPage() {
    for (const item of studioState.items) {
      const id = itemId(item);
      if (id) studioState.selectedItemIds.add(id);
    }
    await applyItemSelectionChange();
  }

  async function clearSelectedItems() {
    studioState.selectedItemIds.clear();
    clearItemSelectionToken();
    studioState.selectedItemPage = 1;
    await applyItemSelectionChange();
  }

  async function selectAllFilteredItems() {
    if (studioState.itemBulkController) return;
    const controller = new AbortController();
    const signature = currentItemFilterSignature();
    studioState.itemBulkController = controller;
    studioState.itemBulkProgress = {
      message: `正在保存 ${studioState.itemTotal} 个筛选结果的选择快照…`,
    };
    updateItemSelectionUi();
    try {
      const payload = await request("/api/media/item-selections", {
        method: "POST",
        signal: controller.signal,
        body: JSON.stringify({
          group_id: Number(studioState.selectedGroupId),
          q: studioState.itemQuery,
        }),
      });
      if (signature !== currentItemFilterSignature()) return;
      if (studioState.itemBulkController === controller) studioState.itemBulkController = null;
      studioState.selectedItemIds.clear();
      studioState.itemSelectionToken = String(payload.selection_token || "");
      studioState.itemSelectionTokenTotal = Math.max(0, Number(payload.total || 0));
      studioState.itemSelectionTokenExpiresAt = Number(payload.expires_at || 0);
      studioState.itemBulkProgress = {
        message: `已选择全部 ${studioState.itemSelectionTokenTotal} 个筛选结果。`,
      };
      await applyItemSelectionChange();
    } catch (error) {
      if (error?.name !== "AbortError") {
        studioState.itemBulkProgress = {
          ...(studioState.itemBulkProgress || {}),
          error: `选择筛选结果失败：${error.message || String(error)}`,
        };
        reportRequestFailure(error, "选择全部筛选结果");
      }
    } finally {
      if (studioState.itemBulkController === controller) studioState.itemBulkController = null;
      updateItemSelectionUi();
    }
  }

  async function reconcileDraftItems(draft) {
    if (studioState.itemSelectionToken) {
      try {
        const payload = await fetchSelectionTokenPage(1, { pageSize: studioState.itemPageSize });
        if (String(payload.group_id || "") !== String(studioState.selectedGroupId)) {
          throw new Error("草稿选择快照不属于当前 GT 文件夹");
        }
        if (String(payload.q || "") !== String(studioState.itemQuery || "")) {
          throw new Error("草稿选择快照的筛选条件已变化");
        }
        studioState.itemSelectionTokenTotal = Math.max(0, Number(payload.total || 0));
        studioState.itemSelectionTokenExpiresAt = Number(payload.expires_at || 0);
        cacheItems(payload.items || []);
        if (draft?.only_selected) {
          studioState.itemOnlySelected = true;
          studioState.items = payload.items || [];
          studioState.selectedItemPage = Math.max(1, Number(payload.page || 1));
          studioState.itemPageCount = Math.max(1, Number(payload.page_count || 1));
        }
        return [];
      } catch (_error) {
        clearItemSelectionToken();
        return ["草稿中的筛选选择已过期或失效，已安全清空"];
      }
    }
    const requestedIds = new Set(
      (Array.isArray(draft?.item_ids) ? draft.item_ids : [])
        .map(Number)
        .filter((id) => Number.isInteger(id) && id > 0),
    );
    if (!requestedIds.size || !studioState.selectedGroupId) {
      studioState.selectedItemIds.clear();
      return [];
    }
    const foundIds = new Set();
    let pageCount = 1;
    let truncated = false;
    for (let page = 1; page <= pageCount; page += 1) {
      if (page > MAX_DRAFT_RECONCILE_PAGES) {
        truncated = true;
        break;
      }
      const payload = await fetchItemPage(page, { pageSize: BULK_ITEM_PAGE_SIZE, query: "" });
      pageCount = Math.max(1, Number(payload.page_count || payload.total_pages || 1));
      const items = payload.items || [];
      cacheItems(items);
      for (const item of items) {
        const id = itemId(item);
        if (requestedIds.has(id)) foundIds.add(id);
      }
      if (foundIds.size === requestedIds.size) break;
    }
    studioState.selectedItemIds = foundIds;
    const removed = [...requestedIds].filter((id) => !foundIds.has(id));
    const reconciled = [];
    if (removed.length) reconciled.push(`${removed.length} 个已删除或失效的 Item 已从草稿移除`);
    if (truncated) {
      reconciled.push(`GT 文件夹超过 ${MAX_DRAFT_RECONCILE_PAGES * BULK_ITEM_PAGE_SIZE} 个 Item，未验证到的草稿选择已安全移除`);
    }
    return reconciled;
  }

  function methodOptions(slot, selected = "") {
    const mode = methodSourceMode(slot);
    const methods = studioState.methods.filter((row) => methodKind(row) === mode);
    const placeholder = mode === "run" ? "选择模型 Run" : "选择已绑定的 External 方法";
    return `<option value="">${placeholder}</option>${methods.map((method) => {
      const key = methodKey(method);
      const coverage = `${Number(method.covered_count || (method.covered_item_ids || []).length || 0)}/${Number(method.total_items || selectedItemCount() || 0)}`;
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
    if (!selectedItemCount()) {
      studioState.methods = [];
      fillMethodSelects();
      return;
    }
    if (usesItemSelectionToken()) {
      const payload = await request(
        `/api/media/methods?selection_token=${encodeURIComponent(studioState.itemSelectionToken)}`,
      );
      if (generation !== studioState.methodRequestGeneration) return;
      studioState.methods = (payload.methods || []).sort((left, right) => (
        Number(right.covered_count || 0) - Number(left.covered_count || 0)
        || String(left.label || "").localeCompare(String(right.label || ""))
      ));
      fillMethodSelects();
      return;
    }
    const grouped = new Map();
    for (let start = 0; start < ids.length; start += METHOD_ITEM_CHUNK_SIZE) {
      const chunk = ids.slice(start, start + METHOD_ITEM_CHUNK_SIZE);
      const query = chunk.map((id) => `item_id=${encodeURIComponent(id)}`).join("&");
      const payload = await request(`/api/media/methods?${query}`);
      if (generation !== studioState.methodRequestGeneration) return;
      for (const method of payload.methods || []) {
        const key = methodKey(method);
        if (!key) continue;
        const aggregate = grouped.get(key) || {
          ...method,
          bindings: [],
          covered_item_ids: [],
        };
        const covered = new Set((aggregate.covered_item_ids || []).map(Number));
        for (const id of method.covered_item_ids || []) covered.add(Number(id));
        const bindings = new Map(
          (aggregate.bindings || []).map((binding) => [Number(binding.item_id), binding]),
        );
        for (const binding of method.bindings || []) {
          bindings.set(Number(binding.item_id), binding);
        }
        aggregate.covered_item_ids = [...covered].sort((left, right) => left - right);
        aggregate.bindings = [...bindings.values()];
        grouped.set(key, aggregate);
      }
    }
    if (generation !== studioState.methodRequestGeneration) return;
    studioState.methods = [...grouped.values()].map((method) => {
      const covered = new Set((method.covered_item_ids || []).map(Number));
      const missing = ids.filter((id) => !covered.has(id));
      return {
        ...method,
        covered_count: covered.size,
        total_items: ids.length,
        missing_item_ids: missing,
        complete: !missing.length,
      };
    }).sort((left, right) => (
      Number(right.covered_count || 0) - Number(left.covered_count || 0)
      || String(left.label || "").localeCompare(String(right.label || ""))
    ));
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
      renderCampaignSubmissionState();
      return;
    }
    const readyCount = rows.filter(rowReady).length;
    host.innerHTML = `
      <div class="coverage-summary">
        <strong>${readyCount}/${selectedItemCount()} 个 Media Item 可发布</strong>
        <span class="muted">时间映射严格验证；空间尺寸会按每个 Item 的较小 Pred 规范化并写入报告。</span>
      </div>
      <div class="coverage-table"><table>
        <thead><tr><th>GT Item</th><th>Pred 1</th><th>Pred 2</th><th>空间规范化</th><th>状态</th></tr></thead>
        <tbody>${rows.map((row) => {
          const ready = rowReady(row);
          const reason = row.reason || row.error || row.alignment_reason || (Array.isArray(row.reasons) ? row.reasons.join("；") : "") || (ready ? "时间严格对齐" : "缺失或不兼容");
          const item = row.item || row.reference || {};
          const a = row.method_a || row.methods?.a || row.binding_a;
          const b = row.method_b || row.methods?.b || row.binding_b;
          return `<tr class="${ready ? "ready" : "blocked"}"><td><strong>${safe(item.display_name || row.display_name || row.item_key || `Media Item #${itemId(row)}`)}</strong><br><small>${safe(outputSummary(item, "GT"))}</small></td><td>${safe(outputSummary(a, "Pred 1"))}</td><td>${safe(outputSummary(b, "Pred 2"))}</td><td>${safe(alignmentPlanSummary(row.alignment_plan || row.alignment))}</td><td><span class="studio-status ${ready ? "ok" : "warn"}">${safe(reason)}</span></td></tr>`;
        }).join("")}</tbody>
      </table></div>`;
    renderCampaignSubmissionState();
  }

  function campaignReadyForCreation() {
    const rows = matrixRows(studioState.preview);
    return Boolean(
      studioState.preview
      && selectedItemCount()
      && selectedMethod("a")
      && selectedMethod("b")
      && rows.length === selectedItemCount()
      && rows.every((row) => rowReady(row)),
    );
  }

  function renderCampaignSubmissionState() {
    const form = el("studio-wizard-form");
    const submit = el("studio-create-campaign");
    const status = el("studio-submit-status");
    if (!form || !submit || !status) return;
    const labels = {
      creating: "正在创建 Campaign…",
      publishing: "Campaign 已创建，正在发布冻结评测包…",
    };
    form.setAttribute("aria-busy", studioState.campaignSubmitting ? "true" : "false");
    submit.disabled = studioState.campaignSubmitting || !campaignReadyForCreation();
    submit.textContent = studioState.campaignSubmitting
      ? (labels[studioState.campaignSubmitPhase] || "正在处理…")
      : "创建并发布冻结评测包";
    if (studioState.campaignSubmitting) {
      status.hidden = false;
      status.className = "run-submit-status message";
      status.textContent = `${labels[studioState.campaignSubmitPhase] || "正在处理…"} 请勿重复点击。`;
    } else if (studioState.campaignSubmitError) {
      status.hidden = false;
      status.className = "run-submit-status message error";
      status.textContent = `Campaign 创建失败：${studioState.campaignSubmitError}`;
    } else {
      status.hidden = true;
      status.className = "run-submit-status";
      status.textContent = "";
    }
  }

  async function previewCoverage() {
    const methodA = selectedMethod("a");
    const methodB = selectedMethod("b");
    if (!selectedItemCount()) throw new Error("请至少选择一个 GT Media Item");
    if (!methodA || !methodB) throw new Error("请选择两份 Pred 方法");
    if (JSON.stringify(methodA) === JSON.stringify(methodB)) throw new Error("两份 Pred 方法必须不同");
    const generation = ++studioState.previewGeneration;
    const signature = JSON.stringify([campaignItemSelectionSignature(), methodA, methodB]);
    const preview = await request("/api/evaluation-campaigns/v2/preview", {
      method: "POST",
      body: JSON.stringify({
        ...campaignItemSelectionPayload(),
        method_a: methodA,
        method_b: methodB,
        spatial_policy: spatialPolicy(),
      }),
    });
    if (generation !== studioState.previewGeneration
        || signature !== JSON.stringify([
          campaignItemSelectionSignature(),
          selectedMethod("a"),
          selectedMethod("b"),
        ])) return;
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
    const pager = el("studio-campaign-pager");
    if (!host) return;
    const cleanupMarkup = cleanupRequestsMarkup();
    if (!studioState.campaigns.length) {
      host.innerHTML = `${cleanupMarkup}<p class="muted">${studioState.campaignTotal ? "当前页没有 Campaign。" : "没有符合条件的 Campaign。"}</p>`;
      if (pager) {
        pager.innerHTML = `<span>共 ${Number(studioState.campaignTotal || 0)} 项</span>`;
      }
      return;
    }
    host.innerHTML = cleanupMarkup + studioState.campaigns.map((campaign) => `
      <button class="studio-campaign-row ${campaignKey(campaign) === studioState.selectedCampaignKey ? "active" : ""}" data-studio-campaign="${safe(campaignKey(campaign))}" type="button">
        <span><strong>${safe(campaign.name)}</strong><small>${safe(campaign.public_title || campaign.metadata?.public_title || "匿名视频质量评测")}</small><small>${Number(campaign.item_count || campaign.candidates || 0)} 视频 · ${Number(campaign.vote_count || campaign.votes || 0)} 票</small></span>
        <span class="studio-status">${safe(campaignStatus(campaign))}</span>
      </button>`).join("");
    if (pager) {
      pager.innerHTML = `
        <button class="secondary" type="button" data-studio-campaign-page="${studioState.campaignPage - 1}" ${studioState.campaignPage <= 1 ? "disabled" : ""}>上一页</button>
        <span>第 ${Number(studioState.campaignPage)} / ${Number(studioState.campaignPageCount)} 页 · 共 ${Number(studioState.campaignTotal)} 项</span>
        <button class="secondary" type="button" data-studio-campaign-page="${studioState.campaignPage + 1}" ${studioState.campaignPage >= studioState.campaignPageCount ? "disabled" : ""}>下一页</button>`;
    }
  }

  function renderPackages() {
    const host = el("media-packages-content");
    if (!host) return;
    const packages = studioState.packageCampaigns.filter((campaign) =>
      Number(campaign.schema_version || 1) >= 2
      && ["published", "closed", "archived"].includes(String(campaign.status || "")),
    );
    const cards = packages.length ? packages.map((campaign) => `
      <div class="derived-video">
        <span><strong>${safe(campaign.name)}</strong><small class="muted">${safe(campaign.public_title || campaign.metadata?.public_title || "匿名视频质量评测")}</small></span>
        <div><span class="studio-status">${safe(campaign.status)}</span><span class="studio-status">${Number(campaign.item_count || 0)} 视频</span></div>
      </div>`).join("") : studioState.packageTotal
      ? "<p class=\"muted\">本页没有可显示的 V2 冻结评测包，可继续翻页。</p>"
      : "<p class=\"muted\">还没有已发布的冻结评测包。</p>";
    const pager = studioState.packagePageCount > 1 ? `
      <div class="pager compact-pager" aria-label="冻结评测包分页">
        <button class="secondary" type="button" data-studio-package-page="${studioState.packagePage - 1}" ${studioState.packagePage <= 1 ? "disabled" : ""}>上一页</button>
        <span>第 ${Number(studioState.packagePage)} / ${Number(studioState.packagePageCount)} 页</span>
        <button class="secondary" type="button" data-studio-package-page="${studioState.packagePage + 1}" ${studioState.packagePage >= studioState.packagePageCount ? "disabled" : ""}>下一页</button>
      </div>` : "";
    host.innerHTML = cards + pager;
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

  function objectiveCurvePointRows(curve) {
    return curve.series.flatMap((series, seriesIndex) =>
      (series.points || []).map((point) => ({ point, series, seriesIndex })))
      .sort((left, right) =>
        Number(left.point.ordinal) - Number(right.point.ordinal)
        || left.seriesIndex - right.seriesIndex)
      .map((row, pointOrder) => ({ ...row, pointOrder }));
  }

  function objectiveCurvePointLabel(metricName, series, point) {
    const status = String(point.status || "missing");
    const value = Number(point.value);
    const measurement = status === "completed"
      && point.value != null
      && Number.isFinite(value)
      ? `${metricName} ${value.toFixed(6)}`
      : status;
    return [
      String(series.method_label || "方法"),
      `frame ${Number(point.frame_index)}`,
      measurement,
      point.reason ? String(point.reason) : "",
    ].filter(Boolean).join(" · ");
  }

  function renderObjectiveCurveDataTable(curve, pointRows) {
    const rows = pointRows.map(({ point, series }) => {
      const status = String(point.status || "missing");
      const value = Number(point.value);
      const valueText = status === "completed"
        && point.value != null
        && Number.isFinite(value)
        ? value.toFixed(6)
        : "—";
      return `<tr>
        <th scope="row">${safe(series.method_label || "方法")}</th>
        <td>${Number(point.frame_index)}</td>
        <td>${safe(status)}</td>
        <td>${valueText}</td>
        <td>${safe(point.reason || "—")}</td>
      </tr>`;
    }).join("");
    return `<details class="metric-data-table objective-curve-data-table">
      <summary>等价数据表（${Number(pointRows.length)} 个点）</summary>
      <div class="table compact-table">
        <table>
          <caption>${safe(curve.metric_name)} 逐帧数据</caption>
          <thead><tr><th scope="col">方法</th><th scope="col">帧</th><th scope="col">状态</th><th scope="col">数值</th><th scope="col">原因</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </details>`;
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
    const pointRows = objectiveCurvePointRows(curve);
    const lines = curve.series.map((series, index) => objectiveCurveSegments(
      series.points, 0, maxOrdinal, minValue, maxValue,
    ).map((points) => `<polyline class="objective-curve-line series-${index === 0 ? "a" : "b"}" points="${points}" fill="none"></polyline>`).join("")).join("");
    const dots = pointRows.filter(({ point }) =>
      point.status === "completed" && point.value != null && Number.isFinite(Number(point.value))).map(({
      point, pointOrder, series, seriesIndex,
    }) => {
      const ordinal = Number(point.ordinal);
      const value = Number(point.value);
      const x = maxOrdinal === 0 ? 50 : 4 + (ordinal / maxOrdinal) * 92;
      const y = maxValue === minValue ? 26 : 44 - ((value - minValue) / (maxValue - minValue)) * 36;
      const label = objectiveCurvePointLabel(curve.metric_name, series, point);
      return `<circle tabindex="0" data-objective-curve-point data-objective-curve-order="${pointOrder}" data-objective-curve-label="${safe(label)}" aria-label="${safe(label)}" class="objective-curve-dot series-${seriesIndex === 0 ? "a" : "b"}" cx="${x.toFixed(3)}" cy="${y.toFixed(3)}" r="0.75"><title>${safe(label)}</title></circle>`;
    }).join("");
    const unavailableDots = pointRows.filter(({ point }) => point.status !== "completed").map(({
      point, pointOrder, series, seriesIndex,
    }) => {
      const ordinal = Number(point.ordinal);
      const x = maxOrdinal === 0 ? 50 : 4 + (ordinal / maxOrdinal) * 92;
      const y = seriesIndex === 0 ? 47 : 49;
      const label = objectiveCurvePointLabel(curve.metric_name, series, point);
      return `<circle tabindex="0" data-objective-curve-point data-objective-curve-order="${pointOrder}" data-objective-curve-label="${safe(label)}" aria-label="${safe(label)}" class="objective-curve-missing series-${seriesIndex === 0 ? "a" : "b"}" cx="${x.toFixed(3)}" cy="${y}" r="0.8"><title>${safe(label)}</title></circle>`;
    }).join("");
    const initialPoint = pointRows.find(({ point }) =>
      point.status !== "completed"
      || (point.value != null && Number.isFinite(Number(point.value))));
    const initialReadout = initialPoint
      ? objectiveCurvePointLabel(
          curve.metric_name,
          initialPoint.series,
          initialPoint.point,
        )
      : "聚焦曲线数据点后显示读数";
    return `
      <div class="objective-curve-legend">${curve.series.map((series, index) => `<span><i class="series-${index === 0 ? "a" : "b"}"></i>${safe(series.method_label)} · ${objectiveMetricStatuses(series.status_counts)}</span>`).join("")}</div>
      <div class="objective-curve-plot"><svg viewBox="0 0 100 52" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${safe(curve.metric_name)} 双曲线对比"><g class="objective-curve-grid"><line x1="4" x2="96" y1="8" y2="8"></line><line x1="4" x2="96" y1="26" y2="26"></line><line x1="4" x2="96" y1="44" y2="44"></line></g>${lines}${dots}${unavailableDots}</svg></div>
      <div class="objective-curve-scale"><span>frame ${Number(curve.series[0]?.points?.[0]?.frame_index ?? 0)}</span><span>LPIPS ${minValue.toFixed(6)} – ${maxValue.toFixed(6)}</span><span>frame ${Number((curve.series[0]?.points || [])[(curve.series[0]?.points || []).length - 1]?.frame_index ?? 0)}</span></div>
      <p class="objective-curve-readout" data-objective-curve-readout role="status" aria-live="polite">当前读数：${safe(initialReadout)}</p>
      <p class="muted">双方 completed 重合帧 ${Number(curve.completed_overlap || 0)}。数值越低越好；断线表示缺失或不可用。${reasons.length ? ` 原因：${safe(reasons.slice(0, 3).join("；"))}` : ""}</p>
      ${renderObjectiveCurveDataTable(curve, pointRows)}`;
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

  function preparationPollStatusMarkup(campaign) {
    if (!campaignPreparationActive(campaign) && !studioState.preparationPollError) return "";
    const refreshedAt = studioState.preparationLastSuccessAt
      ? new Date(studioState.preparationLastSuccessAt).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        })
      : "尚未成功";
    const error = studioState.preparationPollError;
    const retrySeconds = Math.max(1, Math.round(studioState.preparationNextDelayMs / 1000));
    return `<div class="studio-refresh-state ${error ? "stale" : "fresh"}" role="status" aria-live="polite">
      <span>上次成功刷新：${safe(refreshedAt)}</span>
      ${error ? `<span>状态可能已过期：${safe(error)}；${retrySeconds} 秒后重试（第 ${Number(studioState.preparationPollFailures)} 次）。</span>` : "<span>准备状态自动刷新中。</span>"}
    </div>`;
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
      ${!legacy ? preparationPollStatusMarkup(campaign) : ""}
      ${progress.error?.message || campaign.preparation_error?.message ? `<div class="message error"><p>${safe(progress.error?.message || campaign.preparation_error?.message)}</p></div>` : ""}
      <div class="summary-grid"><div><span>视频</span><strong>${Number(coverage.items || campaign.item_count || 0)}</strong></div><div><span>任务</span><strong>${Number(coverage.tasks || campaign.task_count || 0)}</strong></div><div><span>投票</span><strong>${Number(coverage.votes || campaign.vote_count || 0)}</strong></div><div><span>目标票数/视频</span><strong>${Number(campaign.target_votes || 0)}</strong></div></div>
      ${shareUrl ? `<label class="studio-share"><span>参与链接</span><div><input readonly value="${safe(shareUrl)}"><button data-copy-share="${safe(shareUrl)}" type="button">复制</button></div></label>` : ""}
      ${shareUrl && isLoopbackOrigin() ? `<div class="message warn studio-share-warning"><p>当前链接使用本机回环地址，只能在这台电脑上打开。若要让受控内网参与者访问，请以 <code>--host 0.0.0.0</code> 启动服务，再从 <code>http://&lt;服务器内网 IP&gt;:8765</code> 打开 Studio 并重新复制链接；不要将服务暴露到公网。</p></div>` : ""}
      <div class="studio-actions">${!legacy && ["draft", "failed"].includes(campaign.status) ? `<button data-studio-publish="${Number(campaign.id)}" type="button">${campaign.status === "failed" ? "重试发布" : "发布并冻结"}</button>` : ""}${!legacy && campaign.status === "published" ? `<button data-studio-close="${Number(campaign.id)}" type="button">关闭</button>` : ""}${!legacy && ["closed", "failed"].includes(campaign.status) ? `<button class="secondary" data-studio-archive="${Number(campaign.id)}" type="button">归档</button>` : ""}${!legacy ? `<button class="danger secondary" data-studio-delete="${Number(campaign.id)}" type="button">永久删除</button>` : ""}${legacy && !campaign.archived ? `<button class="secondary" data-studio-legacy-archive="${Number(campaign.id)}" type="button">归档</button>` : ""}${legacy ? `<a class="secondary button-link" href="/api/evaluation-campaigns/${Number(campaign.id)}/export">导出</a>` : `<a class="secondary button-link" href="/api/evaluation-campaigns/v2/${Number(campaign.id)}/export">导出</a>`}</div>
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
    const requestedPage = Math.max(
      1,
      Number(options.page || (options.resetPage ? 1 : studioState.campaignPage) || 1),
    );
    const params = new URLSearchParams({
      page: String(requestedPage),
      page_size: String(studioState.campaignPageSize),
    });
    if (studioState.campaignQuery) params.set("q", studioState.campaignQuery);
    if (studioState.campaignStatus) params.set("status", studioState.campaignStatus);
    const generation = ++studioState.campaignListRequestGeneration;
    const payload = await request(`/api/evaluation-campaigns?${params.toString()}`);
    if (generation !== studioState.campaignListRequestGeneration) return;
    studioState.campaigns = payload.campaigns || [];
    studioState.campaignPage = Number(payload.page || requestedPage);
    studioState.campaignPageCount = Number(payload.page_count || 1);
    studioState.campaignTotal = Number(payload.total || studioState.campaigns.length);
    const preserveMissingKey = String(options.preserveMissingKey || "");
    renderCampaignList();
    if (studioState.selectedCampaignKey && options.refreshSelected) {
      try {
        await openCampaign(studioState.selectedCampaignKey, false);
      } catch (error) {
        if (Number(error.status) !== 404
            || studioState.selectedCampaignKey === preserveMissingKey) throw error;
        stopPreparationPoll();
        supersedeObjectiveCurveRequest();
        studioState.selectedCampaignKey = null;
        renderCampaignDetail(null);
        renderCampaignList();
      }
    }
  }

  async function loadPackages(options = {}) {
    const requestedPage = Math.max(
      1,
      Number(options.page || studioState.packagePage || 1),
    );
    const params = new URLSearchParams({
      page: String(requestedPage),
      page_size: String(studioState.packagePageSize),
      status: "published,closed,archived",
    });
    const generation = ++studioState.packageRequestGeneration;
    const payload = await request(`/api/evaluation-campaigns?${params.toString()}`);
    if (generation !== studioState.packageRequestGeneration) return;
    studioState.packageCampaigns = payload.campaigns || [];
    studioState.packagePage = Number(payload.page || requestedPage);
    studioState.packagePageCount = Number(payload.page_count || 1);
    studioState.packageTotal = Number(
      payload.total || studioState.packageCampaigns.length,
    );
    renderPackages();
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
    studioState.preparationLastSuccessAt = Date.now();
    studioState.preparationPollFailures = 0;
    studioState.preparationPollError = "";
    studioState.preparationNextDelayMs = PREPARATION_POLL_BASE_MS;
    renderCampaignDetail(payload);
    if (rerenderList) renderCampaignList();
    if (campaignPreparationActive(payload.campaign || payload)) startPreparationPoll(requestedKey, campaignId);
    else if (studioState.preparationPollKey === requestedKey) stopPreparationPoll();
  }

  async function readCampaignTruth(campaignId) {
    try {
      return {
        exists: true,
        payload: await request(`/api/evaluation-campaigns/v2/${Number(campaignId)}`, {
          suppressDiagnostic: true,
        }),
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
      await Promise.all([
        loadCampaigns({ preserveMissingKey: key }),
        loadPackages({ page: 1 }),
      ]);
    } catch (_refreshError) {
      // The publish result is authoritative; polling can refresh the selected detail.
    }
    return payload;
  }

  async function createCampaign(event) {
    event.preventDefault();
    if (studioState.campaignSubmitting || campaignCreationFlight.isLocked()) {
      notify("Campaign 正在创建，请勿重复点击");
      return;
    }
    const form = new FormData(event.currentTarget);
    const methodA = selectedMethod("a");
    const methodB = selectedMethod("b");
    if (!studioState.preview || !selectedItemCount() || !methodA || !methodB) throw new Error("请先检查视频覆盖与对齐");
    const rows = matrixRows(studioState.preview);
    if (rows.length !== selectedItemCount() || rows.some((row) => !rowReady(row))) throw new Error("所有已选 Media Item 都必须由两种方法完整覆盖并通过对齐");
    if (!campaignCreationFlight.tryLock()) {
      notify("Campaign 正在创建，请勿重复点击");
      return;
    }
    studioState.campaignSubmitting = true;
    studioState.campaignSubmitPhase = "creating";
    studioState.campaignSubmitError = "";
    renderCampaignSubmissionState();
    try {
      const creationPayload = {
        name: String(form.get("name") || "").trim(),
        public_title: String(form.get("public_title") || "").trim(),
        target_votes: Number(form.get("target_votes") || 3),
        ...campaignItemSelectionPayload(),
        method_a: methodA,
        method_b: methodB,
        spatial_policy: spatialPolicy(),
        result_policy: "after_personal_completion",
      };
      const submissionId = campaignSubmissionIdFor(creationPayload);
      const created = await request("/api/evaluation-campaigns/v2", {
        method: "POST",
        body: JSON.stringify({ ...creationPayload, submission_id: submissionId }),
      });
      const campaignId = Number(created.campaign?.id || created.id);
      resetCampaignSubmissionIdentity();
      clearCampaignDraft();
      stopPreparationPoll();
      studioState.campaignRequestGeneration += 1;
      supersedeObjectiveCurveRequest();
      studioState.selectedCampaignKey = `v2:${campaignId}`;
      studioState.campaignDetail = created.campaign ? created : { campaign: created };
      studioState.preview = null;
      studioState.campaignSubmitPhase = "publishing";
      renderCoverage();
      renderCampaignDetail(studioState.campaignDetail);
      const published = await publishCampaign(campaignId);
      notify(campaignStatus(published.campaign || published) === "failed"
        ? "Campaign 准备失败，请查看保留的准备错误后重试"
        : "Campaign 已进入规范化、校验与冻结队列");
    } catch (error) {
      studioState.campaignSubmitError = error.message || String(error);
      throw error;
    } finally {
      studioState.campaignSubmitting = false;
      studioState.campaignSubmitPhase = "";
      campaignCreationFlight.release();
      renderCampaignSubmissionState();
    }
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
    studioState.preparationPollFailures = 0;
    studioState.preparationPollError = "";
    studioState.preparationNextDelayMs = PREPARATION_POLL_BASE_MS;
    if (!studioState.preparationLastSuccessAt) studioState.preparationLastSuccessAt = Date.now();
    const generation = studioState.preparationPollGeneration;
    const stillCurrent = () => generation === studioState.preparationPollGeneration
      && studioState.preparationPollKey === requestedKey
      && studioState.selectedCampaignKey === requestedKey;
    const scheduleNext = (delay = studioState.preparationNextDelayMs) => {
      if (!stillCurrent()) return;
      studioState.preparationPoll = setTimeout(pollOnce, delay);
    };
    const pollOnce = async () => {
      if (!stillCurrent()) return;
      studioState.preparationPoll = null;
      try {
        const payload = await request(`/api/evaluation-campaigns/v2/${Number(campaignId)}`, {
          suppressDiagnostic: true,
        });
        if (!stillCurrent()) return;
        studioState.preparationLastSuccessAt = Date.now();
        studioState.preparationPollFailures = 0;
        studioState.preparationPollError = "";
        studioState.preparationNextDelayMs = PREPARATION_POLL_BASE_MS;
        renderCampaignDetail(payload);
        if (!campaignPreparationActive(payload.campaign || payload)) {
          stopPreparationPoll();
          await Promise.all([loadCampaigns(), loadPackages({ page: 1 })]);
          return;
        }
      } catch (error) {
        if (!stillCurrent()) return;
        studioState.preparationPollFailures += 1;
        studioState.preparationPollError = error.message || String(error);
        studioState.preparationNextDelayMs = Math.min(
          PREPARATION_POLL_MAX_MS,
          PREPARATION_POLL_BASE_MS * (2 ** Math.max(0, studioState.preparationPollFailures - 1)),
        );
        if (studioState.campaignDetail) renderCampaignDetail(studioState.campaignDetail);
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
    await Promise.all([loadCampaigns(), loadPackages({ page: 1 })]);
    await openCampaign(`v2:${Number(campaignId)}`);
  }

  async function deleteCampaign(campaignId) {
    const detailCampaign = studioState.campaignDetail?.campaign
      || studioState.campaignDetail;
    const campaign = studioState.campaigns.find((row) =>
      Number(row.schema_version || 1) >= 2 && Number(row.id) === Number(campaignId))
      || (
        Number(detailCampaign?.schema_version || 1) >= 2
        && Number(detailCampaign?.id) === Number(campaignId)
        ? detailCampaign
        : null
      );
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
    studioState.packageCampaigns = studioState.packageCampaigns.filter(
      (row) => campaignKey(row) !== deletingKey,
    );
    renderPackages();
    try {
      await Promise.all([loadCampaigns(), loadPackages()]);
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

  async function load() {
    const draft = studioState.draftHydrated ? null : readCampaignDraft();
    studioState.draftHydrated = true;
    primeCampaignDraft(draft);
    try {
      await Promise.all([
        loadCampaigns({ refreshSelected: true }),
        loadPackages(),
        loadCleanupRequests(),
        loadRunOutputs(),
        loadItemGroups(),
      ]);
      const reconciled = draft && studioState.draftNotice
        ? [studioState.draftNotice.replace(/[。.]$/, "")]
        : [];
      if (draft) reconciled.push(...await reconcileDraftItems(draft));
      renderItemGroupOptions();
      renderItems();
      await loadMethodsForSelection();
      if (draft) {
        reconciled.push(...applyCampaignDraftControls(draft));
        studioState.draftNotice = reconciled.length
          ? `已恢复 Campaign 草稿并完成校正：${reconciled.join("；")}。`
          : "已恢复上次未发布的 Campaign 草稿。";
      }
      renderItems();
      fillMethodSelects();
      renderCoverage();
    } finally {
      studioState.draftRestoring = false;
    }
    if (draft) queueCampaignDraftSave();
  }

  async function prefillFromCompare(selection) {
    await load();
    const id = itemId(selection?.item);
    if (!id) throw new Error("Compare 没有有效的 Media Item");
    if (selection.groupId && String(selection.groupId) !== String(studioState.selectedGroupId)) {
      studioState.selectedGroupId = String(selection.groupId);
      studioState.selectedItemIds.clear();
      clearItemSelectionToken();
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
    clearItemSelectionToken();
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
    resetCampaignSubmissionIdentity();
    markCampaignDraftDirty();
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
    if (event.target.matches?.("#studio-campaign-status")) {
      studioState.campaignStatus = String(event.target.value || "");
      studioState.campaignPage = 1;
      loadCampaigns({ page: 1 }).catch((error) => notify(error.message));
      return;
    }
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
      cancelBulkItemSelection();
      resetCampaignSubmissionIdentity();
      studioState.itemBulkProgress = null;
      clearItemSelectionToken();
      studioState.selectedGroupId = event.target.value || "";
      studioState.selectedItemIds.clear();
      studioState.itemQuery = "";
      studioState.itemOnlySelected = false;
      studioState.selectedItemPage = 1;
      studioState.itemPage = 1;
      const query = el("studio-item-query");
      if (query) query.value = "";
      invalidateCoveragePreview();
      markCampaignDraftDirty();
      loadItems({ page: 1 }).then(loadMethodsForSelection).catch((error) => notify(error.message));
      return;
    }
    const item = event.target.closest?.("[data-studio-item]");
    if (item) {
      const id = Number(item.dataset.studioItem);
      if (usesItemSelectionToken()) {
        clearItemSelectionToken({
          preserveVisible: true,
          notice: "已取消“全选筛选结果”，并保留当前页选择；请重新检查覆盖。",
        });
      }
      if (item.checked) studioState.selectedItemIds.add(id);
      else studioState.selectedItemIds.delete(id);
      applyItemSelectionChange().catch((error) => notify(error.message));
      return;
    }
    if (event.target.matches?.("#studio-method-a-source, #studio-method-b-source")) {
      resetCampaignSubmissionIdentity();
      fillMethodSelects();
      invalidateCoveragePreview();
      markCampaignDraftDirty();
      return;
    }
    if (event.target.matches?.("#studio-allow-external-aspect-stretch")) {
      resetCampaignSubmissionIdentity();
      invalidateCoveragePreview();
      markCampaignDraftDirty();
      return;
    }
    if (event.target.matches?.("#studio-method-a, #studio-method-b")) {
      resetCampaignSubmissionIdentity();
      invalidateCoveragePreview();
      markCampaignDraftDirty();
    }
  });

  el("studio-item-query")?.addEventListener("input", (event) => {
    cancelBulkItemSelection();
    studioState.itemBulkProgress = null;
    const invalidatedToken = usesItemSelectionToken();
    clearItemSelectionToken({
      notice: invalidatedToken ? "筛选条件已变化，原“全选筛选结果”已失效。" : "",
    });
    studioState.itemQuery = event.target.value || "";
    studioState.itemOnlySelected = false;
    studioState.selectedItemPage = 1;
    studioState.itemPage = 1;
    if (invalidatedToken) {
      resetCampaignSubmissionIdentity();
      invalidateCoveragePreview();
      loadMethodsForSelection().catch((error) => notify(error.message));
    }
    markCampaignDraftDirty();
    clearTimeout(studioState.itemQueryTimer);
    studioState.itemQueryTimer = setTimeout(() => loadItems({ page: 1 }).catch((error) => notify(error.message)), 250);
  });

  document.addEventListener("focusin", (event) => {
    const point = event.target.closest?.("[data-objective-curve-point]");
    if (!point) return;
    const readout = point.closest(".studio-objective-curve")
      ?.querySelector("[data-objective-curve-readout]");
    if (readout) {
      readout.textContent = `当前读数：${point.dataset.objectiveCurveLabel || "无可用读数"}`;
    }
  });

  document.addEventListener("keydown", (event) => {
    const point = event.target.closest?.("[data-objective-curve-point]");
    if (!point) return;
    const plot = point.closest(".objective-curve-plot");
    const points = Array.from(
      plot?.querySelectorAll("[data-objective-curve-point]") || [],
    ).sort((left, right) =>
      Number(left.dataset.objectiveCurveOrder)
      - Number(right.dataset.objectiveCurveOrder));
    const current = points.indexOf(point);
    const target = {
      ArrowLeft: current - 1,
      ArrowDown: current - 1,
      ArrowRight: current + 1,
      ArrowUp: current + 1,
      Home: 0,
      End: points.length - 1,
    }[event.key];
    if (target == null || !points.length) return;
    event.preventDefault();
    points[Math.max(0, Math.min(points.length - 1, target))].focus();
  });

  document.addEventListener("click", (event) => {
    const packagePage = event.target.closest?.("[data-studio-package-page]");
    if (packagePage) {
      loadPackages({
        page: Math.max(1, Number(packagePage.dataset.studioPackagePage || 1)),
      }).catch((error) => notify(error.message));
      return;
    }
    const campaignPage = event.target.closest?.("[data-studio-campaign-page]");
    if (campaignPage) {
      loadCampaigns({
        page: Math.max(1, Number(campaignPage.dataset.studioCampaignPage || 1)),
      }).catch((error) => notify(error.message));
      return;
    }
    const itemPage = event.target.closest?.("[data-studio-item-page]");
    if (itemPage) {
      if (studioState.itemOnlySelected) {
        studioState.selectedItemPage = Math.max(1, Number(itemPage.dataset.studioItemPage || 1));
        if (usesItemSelectionToken()) {
          loadItems({ page: studioState.selectedItemPage }).catch((error) => notify(error.message));
        } else {
          renderItems();
        }
        return;
      }
      loadItems({ page: Number(itemPage.dataset.studioItemPage || 1) }).catch((error) => notify(error.message));
      return;
    }
    if (event.target.closest?.("[data-studio-select-page]")) {
      selectCurrentItemPage().catch((error) => notify(error.message));
      return;
    }
    if (event.target.closest?.("[data-studio-select-filtered]")) {
      selectAllFilteredItems().catch((error) => notify(error.message));
      return;
    }
    if (event.target.closest?.("[data-studio-clear-selection]")) {
      clearSelectedItems().catch((error) => notify(error.message));
      return;
    }
    if (event.target.closest?.("[data-studio-cancel-bulk]")) {
      cancelBulkItemSelection();
      return;
    }
    if (event.target.closest?.("[data-studio-only-selected]")) {
      studioState.itemOnlySelected = !studioState.itemOnlySelected;
      studioState.selectedItemPage = 1;
      if (usesItemSelectionToken()) {
        loadItems({ page: 1 }).catch((error) => notify(error.message));
      } else {
        renderItems();
      }
      markCampaignDraftDirty();
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
    if (copy) {
      return Shared.copyText(copy.dataset.copyShare)
        .then(() => notify("参与链接已复制"))
        .catch(() => notify("无法自动复制，请手动选择参与链接"));
    }
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
  el("studio-campaign-query")?.addEventListener("input", (event) => {
    studioState.campaignQuery = String(event.target.value || "").trim();
    studioState.campaignPage = 1;
    clearTimeout(studioState.campaignQueryTimer);
    studioState.campaignQueryTimer = setTimeout(
      () => loadCampaigns({ page: 1 }).catch((error) => notify(error.message)),
      250,
    );
  });
  el("studio-preview")?.addEventListener("click", () => previewCoverage().catch((error) => notify(error.message)));
  el("studio-wizard-form")?.addEventListener("submit", (event) => createCampaign(event).catch((error) => notify(error.message)));
  el("studio-wizard-form")?.addEventListener("input", (event) => {
    if (event.target.matches?.('[name="name"], [name="public_title"], [name="target_votes"]')) {
      resetCampaignSubmissionIdentity();
    }
    markCampaignDraftDirty();
  });
  el("storage-gc-preview")?.addEventListener("click", () => previewStorageGc().catch((error) => notify(error.message)));
  el("storage-gc-run")?.addEventListener("click", () => executeStorageGc().catch((error) => notify(error.message)));
  window.addEventListener("pagehide", () => {
    saveCampaignDraft();
    cancelBulkItemSelection();
    stopPreparationPoll();
  });
  window.addEventListener("pageshow", (event) => {
    if (!event.persisted || !studioState.selectedCampaignKey) return;
    openCampaign(studioState.selectedCampaignKey, false).catch((error) => notify(error.message));
  });
}());
