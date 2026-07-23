"use strict";

// Domain functions intentionally share app.js's classic-script global
// environment so state, request primitives, and caches remain singletons.

function createSha256Worker() {
  const source = `
    ${Sha256Hasher.toString()}
    Sha256Hasher.K = new Uint32Array(${JSON.stringify(Array.from(Sha256Hasher.K))});
    self.onmessage = async (event) => {
      const file = event.data.file;
      const chunkSize = Number(event.data.chunkSize || 8388608);
      const hasher = new Sha256Hasher();
      try {
        for (let offset = 0; offset < file.size; offset += chunkSize) {
          const end = Math.min(file.size, offset + chunkSize);
          hasher.update(new Uint8Array(await file.slice(offset, end).arrayBuffer()));
          self.postMessage({ type: "progress", done: end, total: file.size });
        }
        self.postMessage({ type: "complete", hex: hasher.hex() });
      } catch (error) {
        self.postMessage({ type: "error", message: error && error.message ? error.message : String(error) });
      }
    };
  `;
  const objectUrl = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
  const worker = new Worker(objectUrl);
  worker.objectUrl = objectUrl;
  return worker;
}

async function sha256File(file, onProgress = () => {}, options = {}) {
  const worker = createSha256Worker();
  options.onWorker?.(worker);
  return new Promise((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      worker.terminate();
      URL.revokeObjectURL(worker.objectUrl);
      options.signal?.removeEventListener("abort", abort);
    };
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      cleanup();
      callback(value);
    };
    const abort = () => finish(reject, new DOMException("文件校验已取消", "AbortError"));
    worker.onmessage = (event) => {
      if (event.data?.type === "progress") {
        onProgress(Number(event.data.done || 0), Number(event.data.total || file.size));
      } else if (event.data?.type === "complete") {
        finish(resolve, String(event.data.hex || ""));
      } else if (event.data?.type === "error") {
        finish(reject, new Error(event.data.message || "SHA-256 计算失败"));
      }
    };
    worker.onerror = (event) => finish(reject, new Error(event.message || "SHA-256 Worker 运行失败"));
    if (options.signal?.aborted) {
      abort();
      return;
    }
    options.signal?.addEventListener("abort", abort, { once: true });
    worker.postMessage({ file, chunkSize: 8 * 1024 * 1024 });
  });
}

async function sha256Blob(blob) {
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return Array.from(new Uint8Array(digest)).map((value) => value.toString(16).padStart(2, "0")).join("");
}

function mediaAssetContent(asset) {
  if (asset.state !== "ready") return `<span class="muted">${escapeHtml(asset.state || "unavailable")}</span>`;
  if (asset.media_kind === "video") {
    return `<video controls playsinline preload="metadata" src="/api/media/assets/${Number(asset.id)}/content"></video>`;
  }
  return `<a href="/api/media/assets/${Number(asset.id)}/content" target="_blank" rel="noreferrer">查看首帧</a>`;
}

function externalPredictionAssets() {
  return (state.externalPredAssets || []).filter((asset) =>
    asset.source_kind === "upload"
    && asset.role === "pred"
    && asset.state === "ready",
  );
}

function renderExternalPredictionBinding() {
  const form = $("external-pred-binding-form");
  if (!form) return;
  const groups = state.externalPredItemGroups || [];
  const groupSelect = form.elements.group_id;
  const itemSelect = form.elements.item_id;
  const assetSelect = form.elements.asset_id;
  const submit = $("bind-external-pred");
  const status = $("external-pred-binding-status");
  const currentItem = (state.externalPredItems || []).find((item) =>
    String(compareItemId(item)) === String(itemSelect.value || ""));
  if (currentItem) state.selectedExternalPredItem = currentItem;
  const currentAsset = externalPredictionAssets().find((asset) =>
    String(asset.id) === String(assetSelect.value || ""));
  if (currentAsset) state.selectedExternalPredAsset = currentAsset;
  const previousItem = String(
    itemSelect.value
    || compareItemId(state.selectedExternalPredItem)
    || "",
  );
  const previousAsset = String(assetSelect.value || state.selectedExternalPredAsset?.id || "");
  groupSelect.innerHTML = groups.length
    ? groups.map((group) => {
        const id = String(group.group_id || group.collection_id || group.id || "");
        return `<option value="${escapeHtml(id)}" ${id === String(state.selectedExternalPredGroupId) ? "selected" : ""}>${escapeHtml(group.name || group.display_name || group.collection_name || id)} (${Number(group.item_count || group.count || 0)})</option>`;
      }).join("")
    : '<option value="">暂无可用 GT Collection</option>';
  const items = state.externalPredItems || [];
  const selectedSnapshot = state.selectedExternalPredItem;
  const itemOptions = selectedSnapshot
    && !items.some((item) => compareItemId(item) === compareItemId(selectedSnapshot))
    ? [selectedSnapshot, ...items]
    : items;
  itemSelect.innerHTML = itemOptions.length
    ? itemOptions.map((item) => {
        const id = compareItemId(item);
        const metadata = [
          item.width && item.height ? `${item.width}×${item.height}` : "",
          item.frame_count ? `${item.frame_count} frames` : "",
          item.fps != null ? `${formatNumber(item.fps)} fps` : "",
        ].filter(Boolean).join(" · ");
        return `<option value="${id}">${escapeHtml(compareItemTitle(item))}${metadata ? ` · ${escapeHtml(metadata)}` : ""}</option>`;
      }).join("")
    : '<option value="">暂无可用 GT Item</option>';
  if (itemOptions.some((item) => String(compareItemId(item)) === previousItem)) itemSelect.value = previousItem;
  const itemPager = $("external-pred-item-pager");
  if (itemPager) {
    const meta = state.externalPredItemsPage || {};
    const page = Math.max(1, Number(meta.page || 1));
    const pageCount = Math.max(1, Number(meta.page_count || 1));
    itemPager.innerHTML = `
      <span class="muted">GT Items ${Number(meta.total || items.length)} · 第 ${page}/${pageCount} 页</span>
      <button class="secondary" data-external-pred-item-page="${page - 1}" type="button" ${page <= 1 ? "disabled" : ""}>上一页</button>
      <button class="secondary" data-external-pred-item-page="${page + 1}" type="button" ${page >= pageCount ? "disabled" : ""}>下一页</button>
    `;
  }
  const assets = externalPredictionAssets();
  const selectedAssetSnapshot = state.selectedExternalPredAsset;
  const assetOptions = selectedAssetSnapshot
    && !assets.some((asset) => Number(asset.id) === Number(selectedAssetSnapshot.id))
    ? [selectedAssetSnapshot, ...assets]
    : assets;
  assetSelect.innerHTML = assetOptions.length
    ? assetOptions.map((asset) => {
        const metadata = [
          asset.width && asset.height ? `${asset.width}×${asset.height}` : "",
          asset.frame_count ? `${asset.frame_count} frames` : "",
          asset.fps != null ? `${formatNumber(asset.fps)} fps` : "",
          asset.collection_name || "",
        ].filter(Boolean).join(" · ");
        return `<option value="${Number(asset.id)}">${escapeHtml(asset.display_name || asset.original_name || `Uploaded Pred #${asset.id}`)}${metadata ? ` · ${escapeHtml(metadata)}` : ""}</option>`;
      }).join("")
    : '<option value="">请先上传外部 Pred</option>';
  if (assetOptions.some((asset) => String(asset.id) === previousAsset)) assetSelect.value = previousAsset;
  const ready = Boolean(groups.length && itemOptions.length && assetOptions.length);
  if (submit) submit.disabled = !ready;
  if (status) {
    status.textContent = ready
      ? "This is an explicit GT Item binding. It does not make unbound or Compare-derived media reusable."
      : (!groups.length
        ? "Create or sync a Collection with a canonical GT Item first."
        : (!itemOptions.length
          ? "The selected Collection has no ready canonical GT Item."
          : "请先上传并选择一份已就绪的外部 Pred。"));
  }
}

async function loadExternalPredictionBindingItems(options = {}) {
  const generation = ++state.externalPredItemRequestGeneration;
  const groupId = Number(state.selectedExternalPredGroupId || 0);
  if (!groupId) {
    state.externalPredItems = [];
    state.externalPredItemsPage = { page: 1, page_size: 100, page_count: 1, total: 0 };
    renderExternalPredictionBinding();
    return;
  }
  const requestedPage = Math.max(1, Number(options.page || state.externalPredItemsPage.page || 1));
  const pageSize = Math.max(1, Number(state.externalPredItemsPage.page_size || 100));
  const payload = await api(`/api/media/items?group_id=${encodeURIComponent(groupId)}&page=${requestedPage}&page_size=${pageSize}`);
  if (generation !== state.externalPredItemRequestGeneration) return;
  const pageCount = Math.max(1, Number(payload.page_count || payload.total_pages || 1));
  if (requestedPage > pageCount) return loadExternalPredictionBindingItems({ page: pageCount });
  state.externalPredItems = payload.items || [];
  state.externalPredItemsPage = {
    page: Math.max(1, Number(payload.page || requestedPage)),
    page_size: Math.max(1, Number(payload.page_size || pageSize)),
    page_count: pageCount,
    total: Math.max(0, Number(payload.total || 0)),
  };
  renderExternalPredictionBinding();
}

function mediaAssetsPath(page = 1) {
  const pageSize = Math.max(1, Math.min(200, Number(state.mediaAssetsPage?.page_size || 50)));
  const params = new URLSearchParams({
    page: String(Math.max(1, Number(page || 1))),
    page_size: String(pageSize),
    state: "ready",
  });
  for (const [key, value] of Object.entries(state.mediaFilters || {})) {
    const normalized = String(value || "").trim();
    if (normalized) params.set(key === "collection_id" ? "collection_id" : key, normalized);
  }
  return `/api/media/assets?${params.toString()}`;
}

function mediaRoleLabel(role) {
  return { gt: "GT", pred: "Pred", reference: "GT", distorted: "Pred" }[String(role || "")] || String(role || "-");
}

function mediaSourceKindLabel(kind) {
  return {
    folder: "项目文件夹",
    upload: "外部上传",
    run_artifact: "Run 产物",
    evaluation_package: "冻结评测包",
  }[String(kind || "")] || String(kind || "-");
}

function mediaKindLabel(kind) {
  return { video: "视频", frame_sequence: "帧序列" }[String(kind || "")] || String(kind || "-");
}

function renderMediaFilters() {
  const filters = state.mediaFilters || {};
  const collectionOptions = state.mediaCollections.map((collection) =>
    `<option value="${Number(collection.id)}" ${String(filters.collection_id) === String(collection.id) ? "selected" : ""}>${escapeHtml(collection.name)}</option>`,
  ).join("");
  return `
    <div class="media-filter-bar" role="search">
      <label>
        <span>搜索</span>
        <input data-media-filter="q" value="${escapeHtml(filters.q)}" placeholder="别名或文件名">
      </label>
      <label>
        <span>角色</span>
        <select data-media-filter="role">
          <option value="">全部角色</option>
          <option value="gt" ${filters.role === "gt" ? "selected" : ""}>GT</option>
          <option value="pred" ${filters.role === "pred" ? "selected" : ""}>Pred</option>
        </select>
      </label>
      <label>
        <span>来源</span>
        <select data-media-filter="source_kind">
          <option value="">全部来源</option>
          ${[
            ["folder", "项目文件夹"],
            ["upload", "外部上传"],
            ["run_artifact", "Run 产物"],
            ["evaluation_package", "冻结评测包"],
          ].map(([value, label]) => `<option value="${value}" ${filters.source_kind === value ? "selected" : ""}>${label}</option>`).join("")}
        </select>
      </label>
      <label>
        <span>Collection</span>
        <select data-media-filter="collection_id">
          <option value="">全部 Collection</option>
          ${collectionOptions}
        </select>
      </label>
      <button class="secondary" data-media-filter-reset type="button">清除筛选</button>
    </div>
  `;
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
  const mediaMeta = state.mediaAssetsPage || {};
  const mediaTotal = Math.max(Number(mediaMeta.total || 0), state.mediaAssets.length);
  const hasMoreMedia = Number(mediaMeta.page || 1) < Number(mediaMeta.page_count || 1);
  host.innerHTML = `${renderMediaFilters()}
    ${state.mediaAssets.length ? `<div class="table compact-table">${table(state.mediaAssets, [
    { label: "预览", render: (asset) => mediaAssetContent(asset) },
    { label: "别名", render: (asset) => `<strong>${escapeHtml(asset.display_name)}</strong><br><span class="muted">${escapeHtml(asset.original_name || "-")}</span>` },
    { label: "Collection", render: (asset) => escapeHtml(asset.collection_name || "-") },
    { label: "来源 / 角色", render: (asset) => `${escapeHtml(mediaSourceKindLabel(asset.source_kind))} / ${escapeHtml(mediaRoleLabel(asset.role))}` },
    { label: "媒体", render: (asset) => `${escapeHtml(mediaKindLabel(asset.media_kind))}<br><span class="muted">${Number(asset.width || 0)}×${Number(asset.height || 0)} · ${Number(asset.frame_count || 0)} 帧 · ${formatNumber(asset.fps)} fps</span>` },
    { label: "大小", render: (asset) => formatBytes(asset.size_bytes) },
    { label: "状态", render: (asset) => statusBadge(asset.state || "ready") },
    { label: "操作", render: (asset) => asset.source_kind === "upload" ? `<button class="danger secondary" data-media-delete="${Number(asset.id)}" type="button">软删除</button>` : "-" },
  ])}</div>` : '<div class="empty-state"><strong>没有匹配的媒体</strong><p class="muted">调整筛选条件，或同步项目文件夹与外部上传。</p></div>'}
    <div class="pager">
      <span class="muted">已显示 ${state.mediaAssets.length}/${mediaTotal} 个媒体源</span>
      ${hasMoreMedia ? '<button class="secondary" data-media-load-more type="button">加载更多</button>' : ""}
    </div>`;
  renderExternalPredictionBinding();
}

async function loadMediaLibrary() {
  const generation = ++state.mediaLibraryRequestGeneration;
  const requests = [
    ["Collections", api("/api/media/collections")],
    ["媒体资产", api(mediaAssetsPath(1))],
    ["GT Item Groups", api("/api/media/item-groups?role=gt")],
    ["上传 Pred", api("/api/media/assets?role=pred&source_kind=upload&state=ready&page=1&page_size=200")],
  ];
  const results = await Promise.allSettled(requests.map((row) => row[1]));
  if (generation !== state.mediaLibraryRequestGeneration) return { failures: [] };
  const failures = [];
  const collectionsResult = results[0];
  const assetsResult = results[1];
  const groupsResult = results[2];
  const externalPredAssetsResult = results[3];
  if (collectionsResult.status === "fulfilled") {
    state.mediaCollections = collectionsResult.value.collections || [];
  } else failures.push(`${requests[0][0]}: ${collectionsResult.reason?.message || collectionsResult.reason}`);
  if (assetsResult.status === "fulfilled") {
    state.mediaAssets = assetsResult.value.assets || [];
    state.mediaAssetsPage = {
      page: Math.max(1, Number(assetsResult.value.page || 1)),
      page_size: Math.max(1, Number(assetsResult.value.page_size || 200)),
      page_count: Math.max(1, Number(assetsResult.value.page_count || 1)),
      total: Math.max(0, Number(assetsResult.value.total || 0)),
    };
  } else failures.push(`${requests[1][0]}: ${assetsResult.reason?.message || assetsResult.reason}`);
  if (groupsResult.status === "fulfilled") {
    state.externalPredItemGroups = groupsResult.value.groups || groupsResult.value.item_groups || [];
    const groupIds = new Set(state.externalPredItemGroups.map((group) =>
      String(group.group_id || group.collection_id || group.id || ""),
    ));
    if (!groupIds.has(String(state.selectedExternalPredGroupId))) {
      state.selectedExternalPredGroupId = String(
        state.externalPredItemGroups[0]?.group_id
        || state.externalPredItemGroups[0]?.collection_id
        || state.externalPredItemGroups[0]?.id
        || "",
      );
      state.selectedExternalPredItem = null;
      state.externalPredItemsPage = { page: 1, page_size: 100, page_count: 1, total: 0 };
    }
  } else failures.push(`${requests[2][0]}: ${groupsResult.reason?.message || groupsResult.reason}`);
  if (externalPredAssetsResult.status === "fulfilled") {
    state.externalPredAssets = externalPredAssetsResult.value.assets || [];
  } else failures.push(`${requests[3][0]}: ${externalPredAssetsResult.reason?.message || externalPredAssetsResult.reason}`);
  try {
    await loadExternalPredictionBindingItems();
  } catch (error) {
    failures.push(`外部 Pred Item：${error.message}`);
  }
  renderMediaLibrary();
  return { failures };
}

async function reloadMediaAssets(page = 1) {
  const generation = ++state.mediaLibraryRequestGeneration;
  const payload = await api(mediaAssetsPath(page));
  if (generation !== state.mediaLibraryRequestGeneration) return;
  state.mediaAssets = payload.assets || [];
  state.mediaAssetsPage = {
    page: Math.max(1, Number(payload.page || page)),
    page_size: Math.max(1, Number(payload.page_size || state.mediaAssetsPage.page_size || 50)),
    page_count: Math.max(1, Number(payload.page_count || 1)),
    total: Math.max(0, Number(payload.total || 0)),
  };
  renderMediaLibrary();
}

function scheduleMediaFilterRefresh(delay = 250) {
  clearTimeout(state.mediaFilterTimer);
  state.mediaFilterTimer = setTimeout(() => {
    state.mediaFilterTimer = null;
    reloadMediaAssets(1).catch((error) => toast(error.message));
  }, delay);
}

async function loadMoreMediaSources() {
  const current = state.mediaAssetsPage || {};
  const nextPage = Math.max(1, Number(current.page || 1) + 1);
  if (nextPage > Math.max(1, Number(current.page_count || 1))) return;
  const generation = ++state.mediaLibraryRequestGeneration;
  const pageSize = Math.max(1, Number(current.page_size || 200));
  state.mediaAssetsPage.page_size = pageSize;
  const payload = await api(mediaAssetsPath(nextPage));
  if (generation !== state.mediaLibraryRequestGeneration) return;
  const byId = new Map(state.mediaAssets.map((asset) => [Number(asset.id), asset]));
  for (const asset of payload.assets || []) byId.set(Number(asset.id), asset);
  state.mediaAssets = Array.from(byId.values());
  state.mediaAssetsPage = {
    page: Math.max(1, Number(payload.page || nextPage)),
    page_size: Math.max(1, Number(payload.page_size || pageSize)),
    page_count: Math.max(1, Number(payload.page_count || 1)),
    total: Math.max(0, Number(payload.total || state.mediaAssets.length)),
  };
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
  return Shared.request(path, {
    fetchOptions: options,
    defaultJsonHeader: false,
    networkMessage: "上传请求无法连接服务",
    onDiagnostic: (error) => showRequestDiagnostic(error, "媒体上传"),
  });
}

function updateUploadTaskUi(text = "") {
  const task = state.uploadTask;
  const form = $("upload-form");
  const progress = $("upload-progress");
  const submit = form?.querySelector('button[type="submit"]');
  const pause = $("pause-upload");
  if (progress && text) progress.textContent = text;
  if (progress) progress.dataset.phase = task?.phase || "idle";
  if (form) form.setAttribute("aria-busy", task ? "true" : "false");
  if (submit) submit.disabled = Boolean(task);
  if (pause) {
    pause.disabled = !task;
    pause.textContent = task?.phase === "hashing" ? "取消文件校验" : "暂停上传";
  }
}

async function uploadExternalMedia(event) {
  event.preventDefault();
  if (state.uploadTask) {
    toast("当前文件仍在校验或上传，请先等待或暂停");
    return;
  }
  const form = event.currentTarget;
  const values = formData(form);
  const file = form.elements.file.files?.[0];
  if (!file) throw new Error("请选择上传文件");
  if (!values.collection_id) throw new Error("请先创建 Collection");
  const resumeKey = uploadResumeKey(file, values);
  const task = {
    phase: "hashing",
    controller: new AbortController(),
    worker: null,
    sessionId: null,
  };
  state.uploadTask = task;
  state.activeUpload = "hashing";
  state.uploadPaused = false;
  updateUploadTaskUi("文件校验 · 正在后台计算 SHA-256（0%）");
  try {
    const fileSha256 = await sha256File(file, (done, total) => {
      updateUploadTaskUi(`文件校验 · ${Math.round((done / Math.max(1, total)) * 100)}%`);
    }, {
      signal: task.controller.signal,
      onWorker: (worker) => { task.worker = worker; },
    });
    task.worker = null;
    task.phase = "resuming";
    updateUploadTaskUi("续传检查 · 正在查找已有上传分片…");

    let session = null;
    const savedUploadId = Shared.storageGet(resumeKey, "");
    if (savedUploadId) {
      try {
        session = await api(`/api/uploads/${savedUploadId}`, { signal: task.controller.signal });
        if (session.status !== "uploading" || Number(session.total_size) !== file.size || session.sha256 !== fileSha256) session = null;
      } catch (error) {
        if (error.name === "AbortError") throw error;
        session = null;
      }
    }
    if (!session) {
      const created = await api("/api/uploads", {
        method: "POST",
        signal: task.controller.signal,
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
      Shared.storageSet(resumeKey, session.id);
    }
    task.sessionId = session.id;
    task.phase = "uploading";
    state.activeUpload = session.id;
    const chunkSize = Number(session.chunk_size || 8 * 1024 * 1024);
    const uploaded = new Set((session.parts || []).map((part) => Number(part.part_index)));
    const partCount = Math.ceil(file.size / chunkSize);
    for (let index = 0; index < partCount; index += 1) {
      if (state.uploadPaused) throw new DOMException("上传已暂停", "AbortError");
      if (uploaded.has(index)) continue;
      const start = index * chunkSize;
      const end = Math.min(file.size, start + chunkSize);
      const part = file.slice(start, end);
      updateUploadTaskUi(`分片上传 · ${index + 1}/${partCount}`);
      const partSha256 = await sha256Blob(part);
      await rawJsonRequest(`/api/uploads/${session.id}/parts/${index}`, {
        method: "PUT",
        signal: task.controller.signal,
        headers: {
          "Content-Type": "application/octet-stream",
          "Content-Range": `bytes ${start}-${end - 1}/${file.size}`,
          "X-Chunk-SHA256": partSha256,
        },
        body: part,
      });
    }
    task.phase = "finalizing";
    updateUploadTaskUi("服务端校验 · 正在建立媒体资产…");
    await api(`/api/uploads/${session.id}/complete`, {
      method: "POST",
      signal: task.controller.signal,
      body: "{}",
    });
    Shared.storageRemove(resumeKey);
    form.reset();
    await loadMediaLibrary();
    updateUploadTaskUi("上传完成");
    toast("媒体资产已上传");
  } catch (error) {
    if (error.name === "AbortError") {
      updateUploadTaskUi(task.phase === "hashing"
        ? "文件校验已取消；没有创建上传会话。"
        : "上传已暂停；重新选择同一文件并提交即可从已有分片续传。");
      return;
    }
    throw error;
  } finally {
    task.worker?.terminate();
    state.activeUpload = null;
    state.uploadTask = null;
    state.uploadPaused = false;
    updateUploadTaskUi();
  }
}

function optionalJsonObject(value, label) {
  const text = String(value || "").trim();
  if (!text) return null;
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (_error) {
    throw new Error(`${label} must be valid JSON`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object`);
  }
  return parsed;
}

async function bindExternalPrediction(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const itemId = Number(form.elements.item_id.value || 0);
  const assetId = Number(form.elements.asset_id.value || 0);
  const methodKey = String(form.elements.method_key.value || "").trim();
  if (!itemId || !assetId || !methodKey) {
    throw new Error("Select one canonical GT Item, one uploaded Pred, and a method key");
  }
  const temporalMapping = optionalJsonObject(form.elements.temporal_mapping.value, "Temporal mapping");
  const spatialOrigin = optionalJsonObject(form.elements.spatial_origin.value, "Spatial origin");
  const payload = {
    asset_id: assetId,
    method_key: methodKey,
    aspect_stretch_confirmed: Boolean(form.elements.aspect_stretch_confirmed.checked),
  };
  if (temporalMapping) payload.temporal_mapping = temporalMapping;
  if (spatialOrigin) payload.spatial_origin = spatialOrigin;
  await api(`/api/media/items/${itemId}/external-predictions`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  form.elements.temporal_mapping.value = "";
  form.elements.spatial_origin.value = "";
  form.elements.aspect_stretch_confirmed.checked = false;
  await loadMediaLibrary();
  if (state.compareSourcesLoaded) {
    await loadCompareSources();
    scheduleComparePreflight(0);
  }
  toast("外部 Pred 已显式绑定到所选 GT Media Item");
}

async function selectExternalPredictionBindingGroup(event) {
  state.selectedExternalPredGroupId = event.target.value || "";
  state.selectedExternalPredItem = null;
  state.externalPredItems = [];
  state.externalPredItemsPage = { page: 1, page_size: 100, page_count: 1, total: 0 };
  renderExternalPredictionBinding();
  await loadExternalPredictionBindingItems({ page: 1 });
}

async function deleteMediaAsset(assetId) {
  if (!window.confirm("软删除该媒体资产？历史 Campaign、投票和统计不会被删除。")) return;
  await api(`/api/media/assets/${Number(assetId)}`, { method: "DELETE" });
  if (Number(state.selectedExternalPredAsset?.id || 0) === Number(assetId)) {
    state.selectedExternalPredAsset = null;
  }
  await loadMediaLibrary();
  await loadCompareSources({ gtPage: 1, predPage: 1 });
  toast("媒体资产已软删除");
}

function catalogSyncPayload(payload) {
  return payload?.sync && typeof payload.sync === "object" ? payload.sync : payload;
}

async function waitForCatalogSync(initial) {
  let status = catalogSyncPayload(initial) || {};
  const deadline = Date.now() + (5 * 60 * 1000);
  while (["requested", "running"].includes(String(status.state || status.status || ""))) {
    if (Date.now() >= deadline) throw new Error("目录同步仍在后台运行，请稍后再次刷新");
    await new Promise((resolve) => setTimeout(resolve, 400));
    status = catalogSyncPayload(await api("/api/media/sync/status")) || {};
    state.catalogSync = status;
  }
  state.catalogSync = status;
  if (String(status.state || status.status || "") === "failed") {
    throw new Error(status.error?.message || status.error || "目录同步失败");
  }
  return status;
}

function shareCatalogSync(operation) {
  if (state.catalogSyncPromise) return state.catalogSyncPromise;
  const shared = Promise.resolve().then(operation);
  state.catalogSyncPromise = shared;
  shared.finally(() => {
    if (state.catalogSyncPromise === shared) state.catalogSyncPromise = null;
  }).catch(() => {});
  return shared;
}

function joinRunningCatalogSync() {
  return shareCatalogSync(async () => {
    const current = catalogSyncPayload(await api("/api/media/sync/status")) || {};
    state.catalogSync = current;
    if (String(current.state || current.status || "") === "failed") {
      throw new Error(current.error?.message || current.error || "目录同步失败");
    }
    return ["requested", "running"].includes(String(current.state || current.status || ""))
      ? waitForCatalogSync(current)
      : current;
  });
}

function requestCatalogSync(includeRuns = false) {
  return shareCatalogSync(async () => {
    const current = catalogSyncPayload(await api("/api/media/sync/status")) || {};
    state.catalogSync = current;
    if (["requested", "running"].includes(String(current.state || current.status || ""))) {
      return waitForCatalogSync(current);
    }
    const started = await api("/api/media/sync", {
      method: "POST",
      body: JSON.stringify({ include_runs: Boolean(includeRuns) }),
    });
    state.catalogSync = catalogSyncPayload(started);
    return waitForCatalogSync(started);
  });
}

async function syncCatalogAndRefresh(options = {}) {
  if (options.includeMedia) state.catalogRefreshIncludeMedia = true;
  if (state.catalogRefreshPromise) return state.catalogRefreshPromise;
  const shared = (async () => {
    await requestCatalogSync(Boolean(options.includeRuns));
    const catalog = await refreshCatalogData({ refreshMetricHealth: true, refreshCheckpoints: true });
    const failures = [...(catalog.failures || [])];
    if (state.catalogRefreshIncludeMedia) {
      const mediaResults = await Promise.allSettled([
        loadMediaLibrary(),
        window.VFIEvalStudio?.load?.(),
      ]);
      if (mediaResults[0].status === "fulfilled") {
        failures.push(...(mediaResults[0].value?.failures || []));
      } else failures.push(`媒体资产: ${mediaResults[0].reason?.message || mediaResults[0].reason}`);
      if (mediaResults[1].status === "rejected") {
        failures.push(`Evaluation Studio: ${mediaResults[1].reason?.message || mediaResults[1].reason}`);
      }
    }
    return { failures };
  })();
  state.catalogRefreshPromise = shared;
  try {
    return await shared;
  } finally {
    if (state.catalogRefreshPromise === shared) state.catalogRefreshPromise = null;
    state.catalogRefreshIncludeMedia = false;
  }
}

async function runCatalogRefresh(button, options = {}) {
  const originalLabel = button?.textContent || "刷新";
  if (button) {
    button.disabled = true;
    button.textContent = "正在同步…";
  }
  try {
    const result = await syncCatalogAndRefresh(options);
    if (result.failures.length) {
      toast(`目录已同步，但 ${result.failures.length} 个面板刷新失败`);
    } else {
      toast(options.includeMedia ? "媒体目录已同步" : "文件目录已同步");
    }
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = originalLabel;
    }
  }
}
