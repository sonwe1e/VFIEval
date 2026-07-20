function newEvaluatorId() {
  try {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
  } catch (_error) {
    // Some embedded/mobile browsers expose crypto but deny access to it.
  }
  return `browser-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function readLocalValue(key) {
  try {
    return window.localStorage ? window.localStorage.getItem(key) || "" : "";
  } catch (_error) {
    return "";
  }
}

function writeLocalValue(key, value) {
  try {
    if (window.localStorage) window.localStorage.setItem(key, value);
  } catch (_error) {
    // A usable in-memory session is better than aborting page initialization.
  }
}

function removeLocalValue(key) {
  try {
    if (window.localStorage) window.localStorage.removeItem(key);
  } catch (_error) {
    // Ignore storage restrictions; the in-memory session is already cleared.
  }
}

const blindState = {
  token: decodeURIComponent(location.pathname.split("/").filter(Boolean).pop() || ""),
  evaluatorId: readLocalValue("vfieval-evaluator-id") || newEvaluatorId(),
  evaluatorName: readLocalValue("vfieval-evaluator-name"),
  payload: null,
  reviews: [],
  reviewMode: false,
  reviewReadOnly: false,
  reviewTask: null,
  taskStartedAt: 0,
  frameIndex: 0,
  videoTargetFrameIndex: 0,
  videoScrubbing: false,
  videoScrubDirty: false,
  mediaGeneration: 0,
  scopeEpoch: 0,
  viewMode: "wipe",
  playbackScope: "all",
  wipeSupported: true,
  wipeDivider: null,
  syncClock: null,
  syncPlayIntent: false,
  syncWaiting: false,
  syncInitialBuffering: false,
  syncAttempt: 0,
  syncRecoveryGeneration: -1,
  syncRecoveryScopeEpoch: -1,
  syncRecoveryAttempt: -1,
  initialBufferScopeEpoch: -1,
  syncFrameCallbackId: null,
  syncFrameCallbackGeneration: 0,
  syncUsesFrameCallback: false,
  streamMonitorTimer: null,
  streamAutoResumeBlocked: false,
  preloadController: null,
  preloadPromise: null,
  preloadProgressText: "",
  blobUrls: [],
  mediaFatal: false,
  mediaReloadPending: false,
  mediaReloadGeneration: -1,
  mediaReloadScopeEpoch: -1,
  mediaReloadAttempt: -1,
  mediaReloadTimer: null,
  frameSequencePending: false,
  frameSequenceRequestToken: 0,
  leaseTimer: null,
  retryTimer: null,
};

const FULL_BLOB_PRELOAD_TOTAL_MAX_BYTES = 256 * 1024 * 1024;
const FULL_BLOB_PRELOAD_MAX_DURATION_SECONDS = 30;
const STREAM_INITIAL_WATER_SECONDS = 10;
const STREAM_LOW_WATER_SECONDS = 1.5;
const STREAM_RESUME_WATER_SECONDS = 5;
const STREAM_NO_PROGRESS_RELOAD_MS = 60_000;
const STREAM_MONITOR_INTERVAL_MS = 250;

writeLocalValue("vfieval-evaluator-id", blindState.evaluatorId);
const byId = (id) => document.getElementById(id);

async function blindApi(path, options = {}) {
  if (typeof window.fetch !== "function") {
    throw new Error("当前浏览器不支持盲评所需的网络功能，请升级浏览器后重试。");
  }
  let timeoutId = null;
  const timeout = new Promise((_resolve, reject) => {
    timeoutId = window.setTimeout(
      () => reject(new Error("连接盲评服务超时，请检查端口映射是否同时转发了 /api/blind/ 请求。")),
      15_000,
    );
  });
  let response;
  try {
    response = await Promise.race([
      window.fetch(path, {
        ...options,
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      }),
      timeout,
    ]);
  } finally {
    if (timeoutId) window.clearTimeout(timeoutId);
  }
  const contentType = response.headers.get("content-type") || "";
  let data = {};
  if (contentType.includes("application/json")) {
    data = await response.json();
  } else {
    const message = (await response.text()).trim();
    if (response.ok) throw new Error("服务器返回了无法识别的响应，请刷新后重试。");
    data = { error: { message } };
  }
  if (!response.ok) throw new Error((data.error && data.error.message) || response.statusText || "请求失败");
  return data;
}

function showToast(message) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2200);
}

function showError(error) {
  const panel = byId("message-panel");
  panel.textContent = error.message || String(error);
  panel.classList.remove("hidden");
}

function setHidden(id, hidden) {
  const element = byId(id);
  if (element) element.classList.toggle("hidden", Boolean(hidden));
}

function replaceContent(element, ...nodes) {
  if (typeof element.replaceChildren === "function") {
    element.replaceChildren(...nodes);
    return;
  }
  while (element.firstChild) element.removeChild(element.firstChild);
  nodes.forEach((node) => element.appendChild(node));
}

function withFrame(url, frame) {
  const separator = String(url).includes("?") ? "&" : "?";
  return `${url}${separator}frame=${Number(frame)}`;
}

function withReloadNonce(url, nonce) {
  if (!nonce) return String(url);
  const separator = String(url).includes("?") ? "&" : "?";
  return `${url}${separator}reload=${encodeURIComponent(String(nonce))}`;
}

function currentTask() {
  return blindState.reviewTask
    || (blindState.payload && (blindState.payload.task || blindState.payload.next_task))
    || null;
}

function setSyncStatus(message, state = "ready") {
  const status = byId("sync-status");
  if (!status) return;
  status.textContent = message;
  status.dataset.state = state;
}

function supportsWipeView() {
  if (!window.CSS || typeof window.CSS.supports !== "function") return false;
  const supportsAspectRatio = window.CSS.supports("aspect-ratio", "16 / 9");
  const clipValue = "inset(0 calc(100% - var(--wipe-position, 50%)) 0 0)";
  const supportsClip = window.CSS.supports("clip-path", clipValue)
    || window.CSS.supports("-webkit-clip-path", clipValue);
  return supportsAspectRatio && supportsClip;
}

function updateWipePosition(value) {
  const position = Math.max(0, Math.min(100, Number(value) || 0));
  const divider = blindState.wipeDivider;
  if (divider) {
    divider.value = String(position);
    divider.setAttribute(
      "aria-valuetext",
      `候选 A 显示 ${Math.round(position)}%，候选 B 显示 ${Math.round(100 - position)}%`,
    );
  }
  const grid = byId("media-grid");
  if (grid) grid.style.setProperty("--wipe-position", `${position}%`);
}

function playbackScopesForView(mode) {
  return mode === "full"
    ? ["all", "reference", "left", "right"]
    : ["all", "reference", "candidates"];
}

function playbackScopeLabel(scope) {
  if (scope === "reference") return "仅 GT";
  if (scope === "candidates") return "仅候选对比";
  if (scope === "left") return "仅 A";
  if (scope === "right") return "仅 B";
  return blindState.viewMode === "full" ? "三路同步" : "两个视图同步";
}

function updatePlaybackScopeControls() {
  const fullView = blindState.viewMode === "full";
  document.querySelectorAll("[data-playback-scope]").forEach((button) => {
    const scope = String(button.dataset.playbackScope || "");
    if (scope === "all") button.textContent = fullView ? "三路同步" : "两个视图同步";
    if (scope === "candidates") button.textContent = "仅候选对比";
    button.setAttribute("aria-label", playbackScopeLabel(scope));
    button.setAttribute("aria-pressed", String(scope === blindState.playbackScope));
    if (button.classList.contains("scope-wipe-only")) button.classList.toggle("hidden", fullView);
    if (button.classList.contains("scope-full-only")) button.classList.toggle("hidden", !fullView);
  });
}

function advancePlaybackScopeEpoch() {
  const reloadWasPending = blindState.mediaReloadPending;
  const reloadRequired = blindState.mediaFatal
    || reloadWasPending
    || blindState.streamAutoResumeBlocked;
  if (reloadWasPending) blindState.mediaFatal = true;
  if (reloadWasPending) clearMediaReloadTimer();
  blindState.scopeEpoch += 1;
  abortPreload();
  blindState.syncInitialBuffering = false;
  blindState.syncRecoveryGeneration = -1;
  blindState.syncRecoveryScopeEpoch = -1;
  blindState.syncRecoveryAttempt = -1;
  blindState.streamAutoResumeBlocked = reloadRequired;
  blindState.mediaReloadPending = false;
  const videos = allVideos();
  const usesBlobs = videos.length && videos.every((video) => video.dataset.preloadDecision === "blob");
  const usesFinalStreaming = videos.length && videos.every((video) => video.dataset.preloadDecision === "streaming");
  if (videos.length && !usesBlobs && !usesFinalStreaming) {
    videos.forEach((video) => { video.dataset.preloadState = "probing"; });
    startTaskVideoPreparation(currentTask());
  }
  setHidden("media-reload", !reloadRequired);
  const reloadButton = byId("media-reload");
  if (reloadRequired && reloadButton) {
    reloadButton.disabled = false;
    reloadButton.textContent = "重新加载";
  }
}

function setPlaybackScope(scope, suppressPause = false) {
  const allowed = playbackScopesForView(blindState.viewMode);
  const nextScope = allowed.includes(scope) ? scope : "all";
  const changed = nextScope !== blindState.playbackScope;
  if (changed && !suppressPause) {
    pauseAndAlignForPlaybackChange();
    advancePlaybackScopeEpoch();
  }
  blindState.playbackScope = nextScope;
  blindState.syncClock = playbackClock();
  if (blindState.syncClock) {
    updateMasterSeek(
      Number(blindState.syncClock.currentTime || 0),
      Number(blindState.syncClock.duration),
    );
  }
  activeVideos().forEach((video) => {
    video.dataset.streamLastProgressAt = String(Date.now());
  });
  updatePlaybackScopeControls();
  const playButton = byId("master-play");
  if (playButton && !blindState.syncPlayIntent) playButton.textContent = "播放所选";
  if (changed && !suppressPause) {
    setSyncStatus(`播放范围已切换为${playbackScopeLabel(nextScope)}；三路画面已对齐暂停。`, "ready");
  }
}

function setViewMode(mode) {
  const requested = mode === "full" ? "full" : "wipe";
  const nextMode = requested === "wipe" && !blindState.wipeSupported ? "full" : requested;
  const changed = nextMode !== blindState.viewMode;
  if (changed) {
    pauseAndAlignForPlaybackChange();
    advancePlaybackScopeEpoch();
  }
  blindState.viewMode = nextMode;
  const grid = byId("media-grid");
  if (grid) {
    grid.classList.toggle("view-wipe", nextMode === "wipe");
    grid.classList.toggle("view-full", nextMode === "full");
  }
  const wipeButton = byId("view-wipe");
  const fullButton = byId("view-full");
  if (wipeButton) {
    wipeButton.disabled = !blindState.wipeSupported;
    wipeButton.setAttribute("aria-pressed", String(nextMode === "wipe"));
  }
  if (fullButton) fullButton.setAttribute("aria-pressed", String(nextMode === "full"));
  if (blindState.wipeDivider) {
    blindState.wipeDivider.disabled = nextMode !== "wipe";
    blindState.wipeDivider.setAttribute("aria-hidden", String(nextMode !== "wipe"));
  }
  setPlaybackScope(changed ? "all" : blindState.playbackScope, true);
  if (changed) {
    const label = nextMode === "wipe" ? "分割线" : "完整";
    setSyncStatus(`${label}视图已切换；三路画面已对齐暂停，不会自动播放。`, "ready");
  }
}

function isCurrentMedia(media) {
  const grid = byId("media-grid");
  return Boolean(
    media
    && grid
    && Number(media.dataset.mediaGeneration) === blindState.mediaGeneration
    && grid.contains(media)
  );
}

function updateMediaAspectRatio(media) {
  if (!isCurrentMedia(media)) return;
  const width = Number(media.videoWidth || media.naturalWidth || 0);
  const height = Number(media.videoHeight || media.naturalHeight || 0);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return;
  const grid = byId("media-grid");
  if (!grid) return;
  const side = String(media.dataset.mediaSide || "");
  if (grid.dataset.aspectSource === "reference" && side !== "reference") return;
  grid.style.setProperty("--media-aspect-ratio", `${width} / ${height}`);
  grid.dataset.aspectSource = side || "media";
}

function mediaLoadError(label) {
  const message = `${label} 无法加载；投票已禁用，请重新加载后再继续。`;
  markTaskMediaFatal(message);
  showError(new Error(message));
}

function allFrameSequenceImages() {
  const grid = byId("media-grid");
  if (!grid) return [];
  return Array.from(grid.querySelectorAll("img[data-frame-base]")).filter(isCurrentMedia);
}

function handleFrameSequenceLoad(event) {
  const image = event.currentTarget;
  if (!isCurrentMedia(image)) return;
  updateMediaAspectRatio(image);
  if (
    !blindState.frameSequencePending
    || Number(image.dataset.frameRequestToken) !== blindState.frameSequenceRequestToken
  ) return;
  image.dataset.reloadReady = "true";
  finishFrameSequenceRequestIfReady();
}

function handleFrameSequenceError(event) {
  const image = event.currentTarget;
  if (
    !isCurrentMedia(image)
    || Number(image.dataset.frameRequestToken) !== blindState.frameSequenceRequestToken
  ) return;
  mediaLoadError(image.dataset.mediaLabel || "帧序列媒体");
}

function setFrameSequenceSource(image, frame, reloadNonce = null) {
  if (!image) return;
  if (reloadNonce !== null) {
    if (reloadNonce) image.dataset.reloadNonce = String(reloadNonce);
    else delete image.dataset.reloadNonce;
  }
  const nonce = String(image.dataset.reloadNonce || "");
  image.dataset.frameRequestGeneration = String(blindState.mediaGeneration);
  image.dataset.frameRequestScopeEpoch = String(blindState.scopeEpoch);
  image.dataset.frameRequestAttempt = String(blindState.syncAttempt);
  image.dataset.frameRequestToken = String(blindState.frameSequenceRequestToken);
  image.dataset.reloadReady = "false";
  image.src = withReloadNonce(withFrame(image.dataset.frameBase, frame), nonce);
}

function armFrameSequenceWatchdog() {
  if (!blindState.frameSequencePending) return;
  const mediaGeneration = blindState.mediaGeneration;
  const requestToken = blindState.frameSequenceRequestToken;
  clearMediaReloadTimer();
  blindState.mediaReloadTimer = window.setTimeout(() => {
    if (
      blindState.frameSequencePending
      && mediaGeneration === blindState.mediaGeneration
      && requestToken === blindState.frameSequenceRequestToken
    ) {
      markTaskMediaFatal("帧序列加载 60 秒仍未完成；请检查媒体服务后重新加载。");
    }
  }, STREAM_NO_PROGRESS_RELOAD_MS);
}

function startFrameSequenceRequest(images, frame, reloadNonce = null) {
  if (images.length !== 3) {
    blindState.frameSequencePending = false;
    markTaskMediaFatal("帧序列媒体不完整，无法安全显示；请联系组织者重新发布。");
    return false;
  }
  blindState.frameSequenceRequestToken += 1;
  blindState.frameSequencePending = true;
  setVoteMediaBlocked(true);
  images.forEach((image) => setFrameSequenceSource(image, frame, reloadNonce));
  armFrameSequenceWatchdog();
  setSyncStatus(
    `正在加载第 ${clampFrameIndex(frame) + 1} 帧的匿名 GT/A/B；三路全部就绪前投票保持禁用。`,
    "stalled",
  );
  return true;
}

function finishFrameSequenceRequestIfReady() {
  if (!blindState.frameSequencePending) return;
  const images = allFrameSequenceImages();
  if (
    images.length !== 3
    || !images.every((image) => (
      image.dataset.reloadReady === "true"
      && Number(image.dataset.frameRequestToken) === blindState.frameSequenceRequestToken
    ))
  ) return;
  blindState.frameSequencePending = false;
  clearMediaReloadTimer();
  if (blindState.mediaReloadPending) {
    finishFrameSequenceReloadIfReady();
    return;
  }
  if (!blindState.mediaFatal) {
    setVoteMediaBlocked(false);
    setSyncStatus(
      `三路帧序列已定位到第 ${blindState.frameIndex + 1} 帧。`,
      "ready",
    );
  }
}

function replacementFrameSequenceImage(image) {
  const replacement = document.createElement("img");
  replacement.alt = image.alt;
  replacement.dataset.mediaSide = String(image.dataset.mediaSide || "");
  replacement.dataset.mediaLabel = String(image.dataset.mediaLabel || image.alt || "帧序列媒体");
  replacement.dataset.mediaGeneration = String(blindState.mediaGeneration);
  replacement.dataset.frameBase = String(image.dataset.frameBase || "");
  if (image.dataset.reloadNonce) {
    replacement.dataset.reloadNonce = String(image.dataset.reloadNonce);
  }
  replacement.addEventListener("load", handleFrameSequenceLoad);
  replacement.addEventListener("error", handleFrameSequenceError);
  image.replaceWith(replacement);
  return replacement;
}

function replaceFrameSequenceImages(frame, reloadNonce = null) {
  const replacements = allFrameSequenceImages().map(replacementFrameSequenceImage);
  startFrameSequenceRequest(replacements, frame, reloadNonce);
  return replacements;
}

function responseTotalBytes(response) {
  if (!response || !response.headers) return 0;
  const contentRange = String(response.headers.get("content-range") || "");
  const rangeMatch = contentRange.match(/\/(\d+)$/);
  if (rangeMatch) return Number(rangeMatch[1] || 0);
  return Number(response.headers.get("content-length") || 0);
}

function releaseBlobUrls() {
  if (!window.URL || typeof window.URL.revokeObjectURL !== "function") {
    blindState.blobUrls = [];
    return;
  }
  blindState.blobUrls.forEach((url) => {
    try {
      window.URL.revokeObjectURL(url);
    } catch (_error) {
      // A detached task may already have released its object URL.
    }
  });
  blindState.blobUrls = [];
}

function preloadContextIsCurrent(mediaGeneration, scopeEpoch, videos) {
  return mediaGeneration === blindState.mediaGeneration
    && scopeEpoch === blindState.scopeEpoch
    && videos.every((video) => isCurrentMedia(video));
}

function abortPreload() {
  const controller = blindState.preloadController;
  blindState.preloadController = null;
  blindState.preloadPromise = null;
  blindState.preloadProgressText = "";
  if (controller) {
    try {
      controller.abort();
    } catch (_error) {
      // Generation and scope epoch checks also discard stale fetch completions.
    }
  }
}

function activateStreamingSources(videos, mediaGeneration, scopeEpoch) {
  if (!preloadContextIsCurrent(mediaGeneration, scopeEpoch, videos)) return;
  videos.forEach((video) => {
    const wasBlob = video.dataset.preloadState === "blob";
    video.dataset.preloadDecision = "streaming";
    video.dataset.preloadState = "streaming";
    video.preload = "auto";
    if (wasBlob) video.src = String(video.dataset.sourceUrl || "");
    if (!blindState.syncPlayIntent && video.paused) video.load();
  });
}

function preloadAbortError() {
  const error = new Error("preload aborted");
  error.name = "AbortError";
  return error;
}

async function preloadOperationWithNoProgressTimeout(operation, label) {
  let timeoutId = null;
  const timeout = new Promise((_resolve, reject) => {
    timeoutId = window.setTimeout(() => {
      const error = new Error(`${label} 60 秒没有进展`);
      error.name = "TimeoutError";
      reject(error);
    }, STREAM_NO_PROGRESS_RELOAD_MS);
  });
  try {
    return await Promise.race([operation, timeout]);
  } finally {
    if (timeoutId !== null) window.clearTimeout(timeoutId);
  }
}

function updateAtomicPreloadProgress(progress, mediaGeneration, scopeEpoch, videos) {
  if (!preloadContextIsCurrent(mediaGeneration, scopeEpoch, videos)) return;
  const text = progress.map((row) => {
    if (row.done) return `${row.label} 100%`;
    if (row.indeterminate) return `${row.label} 下载中`;
    const percent = Math.max(0, Math.min(99, Math.floor((row.received / row.total) * 100)));
    return `${row.label} ${percent}%`;
  }).join(" · ");
  if (text === blindState.preloadProgressText) return;
  blindState.preloadProgressText = text;
  setSyncStatus(`短视频完整缓存：${text}`, "ready");
}

async function readPreloadResponseBlob(
  response,
  progress,
  progressIndex,
  mediaGeneration,
  scopeEpoch,
  videos,
  signal,
) {
  const row = progress[progressIndex];
  const contextIsCurrent = () => (
    preloadContextIsCurrent(mediaGeneration, scopeEpoch, videos)
    && !(signal && signal.aborted)
  );
  if (!contextIsCurrent()) throw preloadAbortError();
  if (response.body && typeof response.body.getReader === "function") {
    const reader = response.body.getReader();
    const chunks = [];
    while (true) {
      if (!contextIsCurrent()) {
        try {
          await reader.cancel();
        } catch (_error) {
          // AbortController and epoch checks already make the read unusable.
        }
        throw preloadAbortError();
      }
      const result = await preloadOperationWithNoProgressTimeout(
        reader.read(),
        `${row.label} 完整下载`,
      );
      if (result.done) break;
      const chunk = result.value;
      row.received += Number((chunk && chunk.byteLength) || 0);
      if (row.received > row.total) throw new Error("full preload exceeded declared length");
      chunks.push(chunk);
      updateAtomicPreloadProgress(progress, mediaGeneration, scopeEpoch, videos);
    }
    if (row.received !== row.total) throw new Error("full preload length mismatch");
    row.done = true;
    updateAtomicPreloadProgress(progress, mediaGeneration, scopeEpoch, videos);
    return new Blob(chunks, { type: response.headers.get("content-type") || "video/mp4" });
  }
  row.indeterminate = true;
  updateAtomicPreloadProgress(progress, mediaGeneration, scopeEpoch, videos);
  const blob = await preloadOperationWithNoProgressTimeout(
    response.blob(),
    `${row.label} 完整下载`,
  );
  if (!contextIsCurrent()) throw preloadAbortError();
  row.received = blob.size;
  row.done = true;
  if (row.received !== row.total) throw new Error("full preload length mismatch");
  updateAtomicPreloadProgress(progress, mediaGeneration, scopeEpoch, videos);
  return blob;
}

async function prepareTaskVideoSources(task, videos, mediaGeneration, scopeEpoch) {
  const durationSeconds = Number(task && task.duration_seconds);
  if (
    !Number.isFinite(durationSeconds)
    || durationSeconds <= 0
    || durationSeconds > FULL_BLOB_PRELOAD_MAX_DURATION_SECONDS
    || typeof window.fetch !== "function"
    || !window.URL
    || typeof window.URL.createObjectURL !== "function"
  ) {
    activateStreamingSources(videos, mediaGeneration, scopeEpoch);
    return;
  }
  const controller = typeof AbortController === "function" ? new AbortController() : null;
  blindState.preloadController = controller;
  blindState.preloadProgressText = "";
  videos.forEach((video) => { video.dataset.preloadDecision = "pending"; });
  const requestOptions = controller ? { signal: controller.signal } : {};
  const probeOptions = controller
    ? { headers: { Range: "bytes=0-0" }, signal: controller.signal }
    : { headers: { Range: "bytes=0-0" } };
  const sourceUrls = videos.map((video) => String(video.dataset.sourceUrl || ""));
  const createdUrls = [];
  try {
    const probes = await Promise.all(sourceUrls.map((url) => (
      preloadOperationWithNoProgressTimeout(
        window.fetch(url, probeOptions),
        "媒体长度探测",
      )
    )));
    const contentLengths = probes.map((response) => responseTotalBytes(response));
    const totalBytes = contentLengths.reduce((total, value) => total + Number(value || 0), 0);
    probes.forEach((response) => {
      if (response.body && typeof response.body.cancel === "function") {
        response.body.cancel().catch(() => {});
      }
    });
    if (
      !probes.every((response) => response.ok)
      || !contentLengths.every((value) => Number.isFinite(value) && value > 0)
      || totalBytes > FULL_BLOB_PRELOAD_TOTAL_MAX_BYTES
    ) {
      activateStreamingSources(videos, mediaGeneration, scopeEpoch);
      if (preloadContextIsCurrent(mediaGeneration, scopeEpoch, videos)) {
        setSyncStatus("三路文件大小缺失或合计超过 256 MiB，统一使用流式缓冲。", "ready");
      }
      return;
    }
    if (!preloadContextIsCurrent(mediaGeneration, scopeEpoch, videos)) return;
    videos.forEach((video) => { video.dataset.preloadState = "loading-blob-group"; });
    const responses = await Promise.all(sourceUrls.map((url) => (
      preloadOperationWithNoProgressTimeout(
        window.fetch(url, requestOptions),
        "短视频完整下载连接",
      )
    )));
    if (!responses.every((response) => response.ok)) throw new Error("full preload response failed");
    const progress = ["GT", "A", "B"].map((label, index) => ({
      label,
      received: 0,
      total: contentLengths[index],
      done: false,
      indeterminate: false,
    }));
    updateAtomicPreloadProgress(progress, mediaGeneration, scopeEpoch, videos);
    const blobs = await Promise.all(responses.map((response, index) => readPreloadResponseBlob(
      response,
      progress,
      index,
      mediaGeneration,
      scopeEpoch,
      videos,
      controller && controller.signal,
    )));
    if (
      !blobs.every((blob, index) => blob.size === contentLengths[index])
      || !preloadContextIsCurrent(mediaGeneration, scopeEpoch, videos)
    ) throw new Error("full preload size or context changed");
    blobs.forEach((blob) => { createdUrls.push(window.URL.createObjectURL(blob)); });
    if (!preloadContextIsCurrent(mediaGeneration, scopeEpoch, videos)) throw new Error("full preload context changed");
    releaseBlobUrls();
    blindState.blobUrls = createdUrls.slice();
    videos.forEach((video, index) => {
      video.dataset.preloadDecision = "blob";
      video.dataset.preloadState = "blob";
      video.src = createdUrls[index];
      video.preload = "auto";
      video.load();
    });
    setSyncStatus("短视频完整缓存就绪：GT 100% · A 100% · B 100%。", "ready");
  } catch (error) {
    if (controller && !controller.signal.aborted) controller.abort();
    createdUrls.forEach((url) => {
      try {
        window.URL.revokeObjectURL(url);
      } catch (_revokeError) {
        // The next task cleanup remains idempotent.
      }
    });
    if (createdUrls.length) blindState.blobUrls = [];
    const aborted = error && error.name === "AbortError";
    if (!aborted && preloadContextIsCurrent(mediaGeneration, scopeEpoch, videos)) {
      activateStreamingSources(videos, mediaGeneration, scopeEpoch);
      setSyncStatus("三路完整缓存未能原子完成，已全部回退到流式缓冲。", "ready");
    }
  } finally {
    if (blindState.preloadController === controller) blindState.preloadController = null;
  }
}

function configureVideoSource(video, sourceUrl) {
  video.dataset.sourceUrl = sourceUrl;
  video.dataset.preloadState = "probing";
  video.dataset.preloadDecision = "pending";
  video.dataset.streamLastProgressAt = String(Date.now());
  video.dataset.streamProgressMarker = "";
  video.preload = "metadata";
  video.src = sourceUrl;
}

function startTaskVideoPreparation(task) {
  const videos = allVideos();
  if (
    !task
    || videos.length !== 3
    || blindState.mediaFatal
    || videos.every((video) => video.dataset.preloadState === "blob")
  ) return;
  const mediaGeneration = blindState.mediaGeneration;
  const scopeEpoch = blindState.scopeEpoch;
  blindState.preloadPromise = prepareTaskVideoSources(
    task,
    videos,
    mediaGeneration,
    scopeEpoch,
  );
}

function recordMediaProgress(video) {
  if (!isCurrentMedia(video)) return;
  const marker = `${Number(video.readyState || 0)}:${highestBufferedEnd(video).toFixed(3)}`;
  if (marker === video.dataset.streamProgressMarker) return;
  video.dataset.streamProgressMarker = marker;
  video.dataset.streamLastProgressAt = String(Date.now());
}

function handleMediaProgress(event) {
  const media = event.currentTarget;
  if (!isCurrentMedia(media)) return;
  recordMediaProgress(media);
  finishMediaReloadIfReady();
  if (isActiveMedia(media)) maybeResumeBufferedPlayback();
}

function handleMediaMetadata(event) {
  const media = event.currentTarget;
  if (!isCurrentMedia(media)) return;
  updateMediaAspectRatio(media);
  recordMediaProgress(media);
  const pendingFrame = Number(media.dataset.pendingFrameIndex);
  if (Number.isFinite(pendingFrame)) {
    delete media.dataset.pendingFrameIndex;
    seekVideoToFrame(media, pendingFrame);
  } else {
    const pendingTime = Number(media.dataset.pendingAlignmentTime);
    if (Number.isFinite(pendingTime)) {
      delete media.dataset.pendingAlignmentTime;
      try {
        media.currentTime = Math.min(Math.max(0, pendingTime), Number(media.duration || pendingTime));
      } catch (_error) {
        media.dataset.pendingAlignmentTime = String(pendingTime);
      }
    }
  }
  if (isActiveClock(media)) {
    updateMasterSeek(Number(media.currentTime || 0), Number(media.duration));
  }
  finishMediaReloadIfReady();
}

function createMediaNode(task, side, label) {
  const kind = task[`${side}_media_kind`] || "video";
  const url = task[`${side}_url`];
  if (kind === "frame_sequence") {
    const image = document.createElement("img");
    image.alt = label;
    image.dataset.mediaSide = side;
    image.dataset.mediaLabel = label;
    image.dataset.mediaGeneration = String(blindState.mediaGeneration);
    image.dataset.frameBase = url;
    image.addEventListener("load", handleFrameSequenceLoad);
    image.addEventListener("error", handleFrameSequenceError);
    return image;
  }

  const video = document.createElement("video");
  video.controls = false;
  video.playsInline = true;
  video.dataset.mediaSide = side;
  video.dataset.mediaLabel = label;
  video.dataset.mediaGeneration = String(blindState.mediaGeneration);
  const rateControl = byId("master-rate");
  const loopControl = byId("master-loop");
  video.playbackRate = Number((rateControl && rateControl.value) || 1);
  video.loop = Boolean(loopControl && loopControl.checked);
  configureVideoSource(video, url);
  video.addEventListener("loadedmetadata", handleMediaMetadata);
  video.addEventListener("resize", () => updateMediaAspectRatio(video));
  video.addEventListener("progress", handleMediaProgress);
  video.addEventListener("durationchange", handleMediaProgress);
  video.addEventListener("waiting", handleMediaWaiting);
  video.addEventListener("canplay", handleMediaCanPlay);
  video.addEventListener("playing", handleMediaPlaying);
  video.addEventListener("pause", handleMediaPause);
  video.addEventListener("ended", handleMediaEnded);
  video.addEventListener("error", handleMediaErrorState);
  video.addEventListener("timeupdate", syncFromClockVideo);
  video.addEventListener("play", handleReferencePlay);
  video.addEventListener("seeking", handleReferenceSeeking);
  video.addEventListener("seeked", handleReferenceSeeked);
  video.addEventListener("ratechange", handleReferenceRateChange);
  return video;
}

function mediaCard(media, label, className = "") {
  const card = document.createElement("article");
  card.className = `media-card ${className}`.trim();
  const heading = document.createElement("h3");
  heading.textContent = label;
  card.append(heading, media);
  return card;
}

function populateVoteForm(task) {
  const form = byId("vote-form");
  if (!form) return;
  form.reset();
  const vote = (task && task.vote) || {};
  form.querySelectorAll('input[name="choice"]').forEach((input) => {
    input.checked = input.value === String(vote.choice || "");
  });
  const leftRating = form.elements.left_rating;
  const rightRating = form.elements.right_rating;
  if (leftRating) leftRating.value = vote.left_rating == null ? "" : String(vote.left_rating);
  if (rightRating) rightRating.value = vote.right_rating == null ? "" : String(vote.right_rating);
  if (form.elements.confidence) form.elements.confidence.value = String(vote.confidence || "");
  if (form.elements.note) form.elements.note.value = String(vote.note || "");
  const readOnly = Boolean(task && task.read_only);
  form.querySelectorAll("input, select, textarea, button").forEach((control) => {
    control.disabled = readOnly;
  });
  const save = byId("save-vote");
  if (save) {
    save.classList.toggle("hidden", readOnly);
    save.textContent = task && task.review ? "保存修改" : "保存并继续";
  }
}

function renderTask(task) {
  stopSynchronization();
  blindState.scopeEpoch += 1;
  blindState.mediaGeneration += 1;
  blindState.viewMode = blindState.wipeSupported ? "wipe" : "full";
  blindState.playbackScope = "all";
  blindState.mediaFatal = false;
  blindState.mediaReloadPending = false;
  blindState.frameSequencePending = false;
  blindState.frameSequenceRequestToken += 1;
  blindState.streamAutoResumeBlocked = false;
  setHidden("media-reload", true);
  setHidden("task-panel", !task);
  if (!task) {
    blindState.reviewTask = null;
    blindState.reviewMode = false;
    blindState.reviewReadOnly = false;
    setHidden("review-toolbar", true);
    if (blindState.leaseTimer) clearInterval(blindState.leaseTimer);
    blindState.leaseTimer = null;
    replaceContent(byId("media-grid"));
    byId("master-play").textContent = "播放所选";
    setSyncStatus("当前没有待评媒体。", "ready");
    return;
  }
  blindState.reviewMode = Boolean(task.review);
  blindState.reviewReadOnly = Boolean(task.read_only);
  if (task.review && blindState.leaseTimer) {
    clearInterval(blindState.leaseTimer);
    blindState.leaseTimer = null;
  }
  setHidden("review-toolbar", !task.review);
  byId("review-mode-label").textContent = task.read_only
    ? "Campaign 已关闭，只读回看"
    : "正在修改已提交结果";
  populateVoteForm(task);
  byId("task-video").textContent = task.video_name || "视频";
  const grid = byId("media-grid");
  grid.style.setProperty("--media-aspect-ratio", "16 / 9");
  grid.dataset.aspectSource = "";
  blindState.frameIndex = 0;
  const reference = createMediaNode(task, "reference", "参考 GT");
  const left = createMediaNode(task, "left", "候选 A");
  const right = createMediaNode(task, "right", "候选 B");
  const referenceCard = mediaCard(reference, "参考 GT", "reference-card");
  const candidateStage = document.createElement("section");
  candidateStage.className = "candidate-stage";
  candidateStage.setAttribute("aria-label", "候选 A 与候选 B 对比");
  candidateStage.append(
    mediaCard(left, "候选 A", "candidate-layer candidate-a"),
    mediaCard(right, "候选 B", "candidate-layer candidate-b"),
  );
  if (blindState.wipeDivider) candidateStage.appendChild(blindState.wipeDivider);
  replaceContent(
    grid,
    referenceCard,
    candidateStage,
  );
  updateMediaAspectRatio(reference);
  updateMediaAspectRatio(left);
  updateMediaAspectRatio(right);
  updateWipePosition(50);
  setViewMode(blindState.viewMode);
  const frameCount = Math.max(1, Number(task.frame_count || 1));
  const mediaKinds = new Set(["reference", "left", "right"].map((side) => task[`${side}_media_kind`] || "video"));
  const hasFrames = mediaKinds.has("frame_sequence");
  const syncStatus = byId("sync-status");
  const playbackControls = byId("playback-controls");
  const mediaToolbar = document.querySelector(".blind-media-toolbar");
  if (syncStatus && (hasFrames ? mediaToolbar : playbackControls)) {
    (hasFrames ? mediaToolbar : playbackControls).appendChild(syncStatus);
  }
  if (mediaKinds.size > 1) {
    showError(new Error("这个任务混用了视频与帧序列，无法提供可靠同步播放，请联系组织者重新发布。"));
    byId("vote-form").classList.add("hidden");
  } else {
    byId("vote-form").classList.remove("hidden");
  }
  setHidden("frame-control", !hasFrames);
  setHidden("playback-controls", hasFrames);
  setHidden("playback-scope", hasFrames);
  byId("frame-range").max = String(frameCount - 1);
  byId("frame-range").value = "0";
  byId("frame-label").textContent = `帧 1/${frameCount}`;
  byId("master-seek").max = String(frameCount - 1);
  byId("master-seek").step = "1";
  byId("master-seek").value = "0";
  updateVideoFrameControls(0);
  updateSequenceFrameControls(0);
  byId("master-play").textContent = "播放所选";
  if (hasFrames) {
    if (mediaKinds.size === 1) {
      startFrameSequenceRequest(allFrameSequenceImages(), blindState.frameIndex);
    }
  } else {
    blindState.syncClock = playbackClock() || reference;
    startStreamMonitoring();
    startTaskVideoPreparation(task);
    setSyncStatus(`三路视频已就绪；${playbackScopeLabel(blindState.playbackScope)}将参与同步播放。`, "ready");
  }
  blindState.taskStartedAt = Date.now();
  if (!task.review) startLeaseHeartbeat(task);
}

function renderResults(results) {
  const host = byId("live-results");
  const ranking = (results && results.human && results.human.ranking)
    || (results && results.ranking)
    || [];
  if (!ranking.length) {
    host.textContent = "尚无足够投票生成结果。";
    return;
  }
  const grid = document.createElement("div");
  grid.className = "result-grid";
  ranking.forEach((row, index) => {
    const card = document.createElement("div");
    card.className = "result-card";
    const name = document.createElement("span");
    name.textContent = `#${index + 1} ${row.label || row.name || "候选"}`;
    const score = document.createElement("strong");
    const scoreValue = row.score != null ? row.score : (row.win_rate != null ? row.win_rate : 0);
    score.textContent = Number(scoreValue).toFixed(4);
    const detail = document.createElement("small");
    detail.textContent = row.ci95 ? `95% CI ${row.ci95[0]}–${row.ci95[1]}` : `${Number(row.votes || 0)} 票`;
    card.append(name, score, detail);
    grid.appendChild(card);
  });
  replaceContent(host, grid);
}

function ratingLabel(value) {
  return value == null || value === "" ? "未评分" : `${Number(value).toFixed(2)} 分`;
}

function renderReviews(payload) {
  const list = (payload && payload.reviews) || [];
  blindState.reviews = list;
  const host = byId("review-list");
  if (!host) return;
  if (!list.length) {
    host.textContent = "尚无可回看的已评视频。";
    return;
  }
  const fragment = document.createDocumentFragment();
  list.forEach((row, index) => {
    const vote = row.vote || {};
    const item = document.createElement("div");
    item.className = "review-row";
    const summary = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = row.video_name || `已评视频 ${index + 1}`;
    const detail = document.createElement("p");
    const choice = vote.choice === "left" ? "A 更好" : vote.choice === "right" ? "B 更好" : "平局";
    detail.className = "muted";
    detail.textContent = `${choice} · A ${ratingLabel(vote.left_rating)} · B ${ratingLabel(vote.right_rating)}`;
    summary.append(title, detail);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary";
    button.dataset.reviewTask = row.task_token;
    button.textContent = payload.editable ? "回看并修改" : "只读回看";
    item.append(summary, button);
    fragment.appendChild(item);
  });
  replaceContent(host, fragment);
}

async function loadReviews() {
  if (!blindState.evaluatorName) return;
  const payload = await blindApi(
    `/api/blind/${encodeURIComponent(blindState.token)}/reviews?evaluator_id=${encodeURIComponent(blindState.evaluatorId)}`,
  );
  renderReviews(payload);
}

async function openReview(taskToken) {
  const payload = await blindApi(
    `/api/blind/${encodeURIComponent(blindState.token)}/reviews/${encodeURIComponent(taskToken)}?evaluator_id=${encodeURIComponent(blindState.evaluatorId)}`,
  );
  blindState.reviewTask = payload.task;
  blindState.reviewMode = true;
  blindState.reviewReadOnly = !payload.editable;
  setHidden("complete-panel", true);
  renderTask(payload.task);
}

async function leaveReview() {
  blindState.reviewTask = null;
  blindState.reviewMode = false;
  blindState.reviewReadOnly = false;
  await loadBlindPayload();
}

function campaignParticipantAvailable(campaign) {
  return ["published", "closed", "archived"].includes(
    String(campaign && campaign.status || ""),
  );
}

function unavailableCampaignMessage(campaign) {
  const status = String(campaign && campaign.status || "");
  if (status === "preparing") {
    return {
      progress: "盲测发布准备中",
      message: "盲测正在发布准备中，请稍候；页面会自动刷新。",
      retry: true,
    };
  }
  if (status === "failed") {
    return {
      progress: "盲测暂不可用",
      message: "盲测暂时无法参与，可能是发布未完成；请联系组织者确认发布状态。",
      retry: false,
    };
  }
  if (status === "draft") {
    return {
      progress: "盲测尚未发布",
      message: "盲测尚未发布，请稍后再试或联系组织者。",
      retry: false,
    };
  }
  return null;
}

function renderPayload(payload) {
  blindState.payload = payload;
  const campaign = payload.campaign || {};
  const progress = payload.progress || {};
  byId("campaign-title").textContent = campaign.public_title || campaign.title || "匿名视频质量评测";
  const unavailable = unavailableCampaignMessage(campaign);
  if (unavailable) {
    byId("progress").textContent = unavailable.progress;
    setHidden("session-panel", true);
    renderTask(null);
    setHidden("complete-panel", true);
    const message = byId("message-panel");
    message.textContent = unavailable.message;
    message.classList.remove("hidden");
    if (unavailable.retry) scheduleTaskRetry();
    else {
      if (blindState.retryTimer) clearTimeout(blindState.retryTimer);
      blindState.retryTimer = null;
    }
    return;
  }
  byId("progress").textContent = `${Number(progress.completed || 0)}/${Number(progress.total || 0)} 已完成`;
  setHidden("session-panel", Boolean(blindState.evaluatorName));
  renderTask(payload.task || payload.next_task || null);
  const complete = Boolean(progress.complete);
  setHidden("complete-panel", !complete);
  if (complete) {
    renderResults(payload.results);
    loadReviews().catch(showError);
  }
  const noTask = !(payload.task || payload.next_task);
  if (!noTask || complete) {
    if (blindState.retryTimer) clearTimeout(blindState.retryTimer);
    blindState.retryTimer = null;
  }
  if (noTask && !complete && blindState.evaluatorName) {
    const message = byId("message-panel");
    const closed = ["closed", "archived"].includes(String(campaign.status || ""));
    message.textContent = closed
      ? "Campaign 已关闭，当前没有可领取的任务。"
      : "其他评测员正在处理剩余名额；系统会在 lease 释放后自动重试。";
    message.classList.remove("hidden");
    if (!closed) scheduleTaskRetry();
  } else if (!payload.error) {
    byId("message-panel").classList.add("hidden");
  }
}

function scheduleTaskRetry() {
  if (blindState.retryTimer) clearTimeout(blindState.retryTimer);
  blindState.retryTimer = setTimeout(() => {
    blindState.retryTimer = null;
    if (!document.hidden) loadBlindPayload().catch(showError);
  }, 5000);
}

async function loadBlindPayload() {
  if (!blindState.token) throw new Error("无效的盲评链接");
  if (!blindState.evaluatorName) {
    const payload = await blindApi(`/api/blind/${encodeURIComponent(blindState.token)}`);
    renderPayload(payload);
    if (campaignParticipantAvailable(payload.campaign)) {
      byId("progress").textContent = "等待加入";
    }
    return;
  }
  try {
    renderPayload(await blindApi(`/api/blind/${encodeURIComponent(blindState.token)}?evaluator_id=${encodeURIComponent(blindState.evaluatorId)}`));
  } catch (error) {
    if (!String(error.message || "").toLowerCase().includes("session")) throw error;
    blindState.evaluatorName = "";
    removeLocalValue("vfieval-evaluator-name");
    setHidden("session-panel", false);
    byId("progress").textContent = "会话已失效，请重新加入";
  }
}

async function saveSession(event) {
  event.preventDefault();
  const sessionForm = event.currentTarget;
  const submitButton = sessionForm.querySelector('button[type="submit"]');
  const form = new FormData(sessionForm);
  const displayName = String(form.get("display_name") || "").trim();
  if (!displayName) return;
  const oldLabel = (submitButton && submitButton.textContent) || "进入盲评";
  if (submitButton) {
    submitButton.disabled = true;
    submitButton.textContent = "正在进入…";
  }
  setHidden("message-panel", true);
  try {
    const payload = await blindApi(`/api/blind/${encodeURIComponent(blindState.token)}/session`, {
      method: "POST",
      body: JSON.stringify({ evaluator_id: blindState.evaluatorId, display_name: displayName }),
    });
    blindState.evaluatorName = displayName;
    writeLocalValue("vfieval-evaluator-name", displayName);
    renderPayload(payload);
  } finally {
    if (submitButton) {
      submitButton.disabled = false;
      submitButton.textContent = oldLabel;
    }
  }
}

async function submitVote(event) {
  event.preventDefault();
  const voteForm = event.currentTarget;
  const task = currentTask();
  if (!task || blindState.reviewReadOnly) return;
  if (
    blindState.mediaFatal
    || blindState.mediaReloadPending
    || blindState.frameSequencePending
  ) {
    showError(new Error("媒体需要重新加载并恢复成功后才能提交评测。"));
    return;
  }
  const form = new FormData(voteForm);
  const choice = String(form.get("choice") || "");
  if (!choice) {
    showError(new Error("请选择候选 A、平局或候选 B。"));
    return;
  }
  const buttons = voteForm.querySelectorAll("button");
  buttons.forEach((button) => { button.disabled = true; });
  const wasReview = blindState.reviewMode;
  const leftRaw = String(form.get("left_rating") || "").trim();
  const rightRaw = String(form.get("right_rating") || "").trim();
  try {
    const payload = await blindApi(
      `/api/blind/${encodeURIComponent(blindState.token)}/tasks/${encodeURIComponent(task.token)}/vote`,
      {
        method: "POST",
        body: JSON.stringify({
          evaluator_id: blindState.evaluatorId,
          choice,
          reasons: [],
          left_rating: leftRaw ? Number(leftRaw) : null,
          right_rating: rightRaw ? Number(rightRaw) : null,
          confidence: String(form.get("confidence") || ""),
          note: String(form.get("note") || ""),
          duration_ms: Math.max(0, Date.now() - blindState.taskStartedAt),
        }),
      },
    );
    voteForm.reset();
    if (wasReview) {
      blindState.reviewTask = null;
      blindState.reviewMode = false;
      blindState.reviewReadOnly = false;
      await loadBlindPayload();
      showToast("修改已保存");
      return;
    }
    renderPayload({
      ...blindState.payload,
      ...payload,
      campaign: payload.campaign || (blindState.payload && blindState.payload.campaign) || {},
      task: payload.next_task || null,
    });
    showToast("投票已保存");
  } finally {
    buttons.forEach((button) => { button.disabled = false; });
  }
}

function allVideos() {
  return Array.from(document.querySelectorAll("#media-grid video"));
}

function activeMediaSides() {
  if (blindState.playbackScope === "reference") return ["reference"];
  if (blindState.playbackScope === "candidates") return ["left", "right"];
  if (blindState.playbackScope === "left") return ["left"];
  if (blindState.playbackScope === "right") return ["right"];
  return ["reference", "left", "right"];
}

function isActiveMedia(media) {
  return isCurrentMedia(media) && activeMediaSides().includes(String(media.dataset.mediaSide || ""));
}

function activeVideos() {
  return allVideos().filter((video) => isActiveMedia(video));
}

function referenceVideo() {
  return allVideos().find((video) => video.dataset.mediaSide === "reference") || null;
}

function playbackClock() {
  const videos = activeVideos();
  return videos.find((video) => video.dataset.mediaSide === "reference") || videos[0] || null;
}

function isActiveClock(media) {
  return isActiveMedia(media) && media === (blindState.syncClock || playbackClock());
}

function pauseAndAlignForPlaybackChange() {
  const oldClock = blindState.syncClock && isCurrentMedia(blindState.syncClock)
    ? blindState.syncClock
    : (playbackClock() || referenceVideo());
  const anchor = Number((oldClock && oldClock.currentTime) || 0);
  blindState.syncPlayIntent = false;
  blindState.syncWaiting = false;
  blindState.syncInitialBuffering = false;
  blindState.videoScrubbing = false;
  blindState.videoScrubDirty = false;
  blindState.syncAttempt += 1;
  clearRecoveryContext();
  cancelFrameSynchronization();
  pauseVideosInternally(allVideos());
  alignAllVideos(anchor);
  const playButton = byId("master-play");
  if (playButton) playButton.textContent = "播放所选";
}

function markRecoveryContext(initialBuffering) {
  blindState.syncInitialBuffering = Boolean(initialBuffering);
  blindState.syncRecoveryGeneration = blindState.mediaGeneration;
  blindState.syncRecoveryScopeEpoch = blindState.scopeEpoch;
  blindState.syncRecoveryAttempt = blindState.syncAttempt;
}

function clearRecoveryContext() {
  blindState.syncInitialBuffering = false;
  blindState.syncRecoveryGeneration = -1;
  blindState.syncRecoveryScopeEpoch = -1;
  blindState.syncRecoveryAttempt = -1;
}

function recoveryContextIsCurrent() {
  return !blindState.streamAutoResumeBlocked
    && blindState.syncRecoveryGeneration === blindState.mediaGeneration
    && blindState.syncRecoveryScopeEpoch === blindState.scopeEpoch
    && blindState.syncRecoveryAttempt === blindState.syncAttempt;
}

function synchronizationDriftThreshold(clock) {
  const task = currentTask() || {};
  const frameCount = Number(task.frame_count || 0);
  const duration = Number(clock && clock.duration);
  if (frameCount > 1 && Number.isFinite(duration) && duration > 0) {
    return Math.max(0.004, duration / frameCount);
  }
  return 1 / 24;
}

function taskFrameCount() {
  const task = currentTask() || {};
  return Math.max(1, Number(task.frame_count || 1));
}

function taskFramesPerSecond() {
  const task = currentTask() || {};
  const fps = Number(task.fps);
  return Number.isFinite(fps) && fps > 0 ? fps : 0;
}

function reliableMediaDuration(video) {
  const mediaDuration = Number(video && video.duration);
  if (Number.isFinite(mediaDuration) && mediaDuration > 0) return mediaDuration;
  const task = currentTask() || {};
  const taskDuration = Number(task.duration_seconds);
  return Number.isFinite(taskDuration) && taskDuration > 0 ? taskDuration : 0;
}

function frameSteppingIsReliable(video) {
  return taskFramesPerSecond() > 0 || reliableMediaDuration(video) > 0;
}

function clampFrameIndex(value) {
  return Math.max(0, Math.min(taskFrameCount() - 1, Math.round(Number(value) || 0)));
}

function frameIndexFromTime(currentTime, duration) {
  const total = taskFrameCount();
  const fps = taskFramesPerSecond();
  if (fps > 0) {
    return Math.max(0, Math.min(total - 1, Math.floor(Number(currentTime || 0) * fps + 0.000001)));
  }
  if (!Number.isFinite(duration) || duration <= 0) return 0;
  return Math.max(0, Math.min(total - 1, Math.floor((Number(currentTime || 0) / duration) * total + 0.000001)));
}

function updateVideoFrameControls(frameIndex, preview = false) {
  const index = clampFrameIndex(frameIndex);
  blindState.videoTargetFrameIndex = index;
  const total = taskFrameCount();
  const seek = byId("master-seek");
  if (seek) {
    seek.max = String(total - 1);
    seek.step = "1";
    seek.value = String(index);
  }
  const label = byId("master-frame-label");
  if (label) label.textContent = `${preview ? "目标帧" : "帧"} ${index + 1}/${total}`;
  const previous = byId("master-prev");
  const next = byId("master-next");
  const reliable = frameSteppingIsReliable(playbackClock());
  if (previous) previous.disabled = !reliable || index <= 0;
  if (next) next.disabled = !reliable || index >= total - 1;
}

function updateSequenceFrameControls(frameIndex) {
  const index = clampFrameIndex(frameIndex);
  const total = taskFrameCount();
  const previous = byId("frame-prev");
  const next = byId("frame-next");
  if (previous) previous.disabled = index <= 0;
  if (next) next.disabled = index >= total - 1;
}

function seekVideoToFrame(video, frameIndex) {
  if (!isCurrentMedia(video)) return;
  const index = clampFrameIndex(frameIndex);
  const fps = taskFramesPerSecond();
  const duration = reliableMediaDuration(video);
  if (fps <= 0 && duration <= 0) {
    delete video.dataset.pendingAlignmentTime;
    video.dataset.pendingFrameIndex = String(index);
    return;
  }
  const rawTarget = fps > 0
    ? (index + 0.5) / fps
    : duration * ((index + 0.5) / taskFrameCount());
  const target = duration > 0
    ? Math.min(Math.max(0, rawTarget), Math.max(0, duration - 0.001))
    : Math.max(0, rawTarget);
  try {
    video.currentTime = target;
  } catch (_error) {
    video.dataset.pendingFrameIndex = String(index);
  }
}

function updateMasterSeek(currentTime, duration) {
  if (!Number.isFinite(duration) || duration <= 0) return;
  if (blindState.videoScrubbing) return;
  updateVideoFrameControls(frameIndexFromTime(currentTime, duration));
}

function synchronizeFollowers(anchor, force = false) {
  const clock = blindState.syncClock || playbackClock();
  if (!clock || !Number.isFinite(Number(anchor))) return;
  const threshold = synchronizationDriftThreshold(clock);
  activeVideos().forEach((peer) => {
    if (peer === clock) return;
    const peerTime = Number(peer.currentTime || 0);
    if (!force && Math.abs(peerTime - Number(anchor)) <= threshold) return;
    if (force && Math.abs(peerTime - Number(anchor)) <= 0.001) return;
    const peerDuration = Number(peer.duration);
    const target = Number.isFinite(peerDuration) && peerDuration > 0
      ? Math.min(Math.max(0, Number(anchor)), peerDuration)
      : Math.max(0, Number(anchor));
    try {
      peer.currentTime = target;
    } catch (_error) {
      // Metadata may still be loading. The next clock callback will retry.
    }
  });
}

function alignAllVideos(anchor) {
  if (!Number.isFinite(Number(anchor))) return;
  allVideos().forEach((video) => {
    const duration = Number(video.duration);
    const target = Number.isFinite(duration) && duration > 0
      ? Math.min(Math.max(0, Number(anchor)), duration)
      : Math.max(0, Number(anchor));
    if (!Number.isFinite(duration) || duration <= 0) {
      delete video.dataset.pendingFrameIndex;
      video.dataset.pendingAlignmentTime = String(target);
    }
    try {
      video.currentTime = target;
    } catch (_error) {
      // Metadata handlers finish alignment once the media duration is known.
    }
  });
}

function cancelFrameSynchronization() {
  blindState.syncFrameCallbackGeneration += 1;
  const clock = blindState.syncClock;
  if (
    blindState.syncFrameCallbackId !== null
    && clock
    && typeof clock.cancelVideoFrameCallback === "function"
  ) {
    try {
      clock.cancelVideoFrameCallback(blindState.syncFrameCallbackId);
    } catch (_error) {
      // Generation checks still prevent a stale callback from touching the next task.
    }
  }
  blindState.syncFrameCallbackId = null;
  blindState.syncUsesFrameCallback = false;
}

function startFrameSynchronization() {
  cancelFrameSynchronization();
  const clock = blindState.syncClock || playbackClock();
  if (
    !clock
    || !blindState.syncPlayIntent
    || typeof clock.requestVideoFrameCallback !== "function"
  ) return;
  blindState.syncClock = clock;
  const generation = blindState.syncFrameCallbackGeneration;
  const onFrame = (_now, metadata) => {
    blindState.syncFrameCallbackId = null;
    if (
      generation !== blindState.syncFrameCallbackGeneration
      || !blindState.syncPlayIntent
      || blindState.syncClock !== clock
    ) return;
    const mediaTime = metadata && Number.isFinite(Number(metadata.mediaTime))
      ? Number(metadata.mediaTime)
      : Number(clock.currentTime || 0);
    synchronizeFollowers(mediaTime);
    updateMasterSeek(mediaTime, Number(clock.duration));
    try {
      blindState.syncFrameCallbackId = clock.requestVideoFrameCallback(onFrame);
      blindState.syncUsesFrameCallback = true;
    } catch (_error) {
      blindState.syncFrameCallbackId = null;
      blindState.syncUsesFrameCallback = false;
    }
  };
  try {
    blindState.syncFrameCallbackId = clock.requestVideoFrameCallback(onFrame);
    blindState.syncUsesFrameCallback = true;
  } catch (_error) {
    blindState.syncFrameCallbackId = null;
    blindState.syncUsesFrameCallback = false;
  }
}

function schedulePausedFrameCorrection(clock) {
  if (
    !isActiveClock(clock)
    || blindState.syncPlayIntent
    || typeof clock.requestVideoFrameCallback !== "function"
  ) return;
  blindState.syncClock = clock;
  cancelFrameSynchronization();
  const callbackGeneration = blindState.syncFrameCallbackGeneration;
  const mediaGeneration = blindState.mediaGeneration;
  const scopeEpoch = blindState.scopeEpoch;
  const attempt = blindState.syncAttempt;
  let callbackId = null;
  const onFrame = (_now, metadata) => {
    if (blindState.syncFrameCallbackId === callbackId) {
      blindState.syncFrameCallbackId = null;
    }
    if (
      callbackGeneration !== blindState.syncFrameCallbackGeneration
      || mediaGeneration !== blindState.mediaGeneration
      || scopeEpoch !== blindState.scopeEpoch
      || attempt !== blindState.syncAttempt
      || blindState.syncPlayIntent
      || !isActiveClock(clock)
    ) return;
    const mediaTime = metadata && Number.isFinite(Number(metadata.mediaTime))
      ? Number(metadata.mediaTime)
      : Number(clock.currentTime || 0);
    updateMasterSeek(mediaTime, Number(clock.duration));
  };
  try {
    callbackId = clock.requestVideoFrameCallback(onFrame);
    blindState.syncFrameCallbackId = callbackId;
  } catch (_error) {
    blindState.syncFrameCallbackId = null;
  }
}

function applyPlaybackSettings() {
  const rateControl = byId("master-rate");
  const loopControl = byId("master-loop");
  const rate = Number((rateControl && rateControl.value) || 1);
  const loop = Boolean(loopControl && loopControl.checked);
  allVideos().forEach((video) => {
    video.playbackRate = rate;
    video.loop = loop;
  });
}

function syncFromClockVideo(event) {
  const video = event.currentTarget;
  if (!isActiveClock(video)) return;
  if (!Number.isFinite(video.duration) || video.duration <= 0) return;
  updateMasterSeek(Number(video.currentTime || 0), Number(video.duration));
  if (blindState.syncPlayIntent && !blindState.syncUsesFrameCallback) {
    synchronizeFollowers(Number(video.currentTime || 0));
  }
}

function handleReferenceSeeking(event) {
  const clock = event.currentTarget;
  if (!isActiveClock(clock)) return;
  synchronizeFollowers(Number(clock.currentTime || 0), true);
  updateMasterSeek(Number(clock.currentTime || 0), Number(clock.duration));
  if (!blindState.syncWaiting) setSyncStatus("正在将三路视频定位到同一帧…", "ready");
}

function handleReferenceSeeked(event) {
  const clock = event.currentTarget;
  if (!isActiveClock(clock)) return;
  synchronizeFollowers(Number(clock.currentTime || 0), true);
  updateMasterSeek(Number(clock.currentTime || 0), Number(clock.duration));
  if (blindState.syncWaiting) {
    setSyncStatus("媒体仍在缓冲，三路视频已保持对齐暂停。", "stalled");
    return;
  }
  if (!blindState.syncPlayIntent) schedulePausedFrameCorrection(clock);
  setSyncStatus(
    blindState.syncPlayIntent
      ? `${playbackScopeLabel(blindState.playbackScope)}同步播放中。`
      : "三路视频已定位到同一帧。",
    blindState.syncPlayIntent ? "playing" : "ready",
  );
}

function handleReferenceRateChange(event) {
  const clock = event.currentTarget;
  if (!isActiveClock(clock)) return;
  allVideos().forEach((peer) => {
    if (peer !== clock) peer.playbackRate = clock.playbackRate;
  });
}

function pauseVideosInternally(videos = activeVideos()) {
  videos.forEach((video) => {
    try {
      if (!video.paused) {
        video.dataset.syncInternalPausePending = String(
          Number(video.dataset.syncInternalPausePending || 0) + 1,
        );
      }
      video.pause();
    } catch (_error) {
      // Detached media is abandoned by the generation guard.
    }
  });
}

function stopSynchronization() {
  blindState.syncPlayIntent = false;
  blindState.syncWaiting = false;
  blindState.videoScrubbing = false;
  blindState.videoScrubDirty = false;
  clearRecoveryContext();
  blindState.syncAttempt += 1;
  cancelFrameSynchronization();
  abortPreload();
  clearMediaReloadTimer();
  pauseVideosInternally(allVideos());
  stopStreamMonitoring();
  releaseBlobUrls();
  blindState.syncClock = null;
}

function pauseSynchronizedPlayback(message = "已暂停，参与播放的媒体保持在同一帧。", state = "ready") {
  blindState.syncPlayIntent = false;
  blindState.syncWaiting = false;
  clearRecoveryContext();
  blindState.syncAttempt += 1;
  cancelFrameSynchronization();
  pauseVideosInternally();
  const playButton = byId("master-play");
  if (playButton) playButton.textContent = "播放所选";
  setSyncStatus(message, state);
}

function handlePlayFailure(error) {
  blindState.syncWaiting = false;
  blindState.syncPlayIntent = false;
  clearRecoveryContext();
  blindState.syncAttempt += 1;
  cancelFrameSynchronization();
  pauseVideosInternally();
  const message = error && error.message ? error.message : "未知播放错误";
  if (activeVideos().some((video) => Boolean(video.error))) {
    markTaskMediaFatal("检测到真实媒体解码或网络错误；投票已禁用，请重新加载。");
    showError(new Error(`媒体播放失败：${message}`));
    return;
  }
  setSyncStatus("所选媒体未能启动，已保持对齐暂停。", "error");
  showError(new Error(`无法同步播放所选媒体：${message}`));
}

function isAbortPlayError(error) {
  return Boolean(error && (error.name === "AbortError" || error.code === 20));
}

function highestBufferedEnd(video) {
  let end = 0;
  try {
    for (let index = 0; index < video.buffered.length; index += 1) {
      end = Math.max(end, Number(video.buffered.end(index) || 0));
    }
  } catch (_error) {
    return 0;
  }
  return end;
}

function bufferedAhead(video) {
  const currentTime = Number(video.currentTime || 0);
  let ahead = 0;
  try {
    for (let index = 0; index < video.buffered.length; index += 1) {
      const start = Number(video.buffered.start(index) || 0);
      const end = Number(video.buffered.end(index) || 0);
      if (currentTime + 0.05 >= start && currentTime <= end + 0.05) {
        ahead = Math.max(ahead, end - currentTime);
      }
    }
  } catch (_error) {
    return 0;
  }
  return Math.max(0, ahead);
}

function requiredBufferedSeconds(video, targetSeconds) {
  const duration = Number(video.duration);
  const currentTime = Number(video.currentTime || 0);
  if (!Number.isFinite(duration) || duration <= 0) return targetSeconds;
  return Math.max(0, Math.min(targetSeconds, duration - currentTime));
}

function anonymousMediaLabel(video) {
  const side = String(video && video.dataset.mediaSide || "");
  if (side === "reference") return "参考 GT";
  if (side === "left") return "候选 A";
  if (side === "right") return "候选 B";
  return "媒体";
}

function bufferingDiagnostic(video, targetSeconds) {
  const ahead = bufferedAhead(video);
  const required = requiredBufferedSeconds(video, targetSeconds);
  const reason = ahead + 0.05 < required ? "取流不足" : "浏览器解码等待";
  return `${anonymousMediaLabel(video)}${reason}（已连续缓冲 ${ahead.toFixed(1)} 秒）`;
}

function mediaBelowLowWatermark(video) {
  if (video.dataset.preloadState === "blob") return false;
  if (Number(video.readyState || 0) < 3) return true;
  const required = requiredBufferedSeconds(video, STREAM_LOW_WATER_SECONDS);
  return required > 0.05 && bufferedAhead(video) + 0.05 < required;
}

function mediaReadyForInitialPlayback(video) {
  if (!isActiveMedia(video) || video.error || Number(video.readyState || 0) < 3) return false;
  if (video.dataset.preloadState === "blob") return true;
  const required = requiredBufferedSeconds(video, STREAM_INITIAL_WATER_SECONDS);
  return required <= 0.05 || bufferedAhead(video) + 0.05 >= required;
}

function mediaReadyForPlayback(video) {
  if (!isActiveMedia(video)) return false;
  if (video.error || Number(video.readyState || 0) < 3) return false;
  if (video.dataset.preloadState === "blob") return true;
  const target = blindState.syncInitialBuffering
    ? STREAM_INITIAL_WATER_SECONDS
    : STREAM_RESUME_WATER_SECONDS;
  const required = requiredBufferedSeconds(video, target);
  return required <= 0.05 || bufferedAhead(video) + 0.05 >= required;
}

function stopStreamMonitoring() {
  if (blindState.streamMonitorTimer) clearInterval(blindState.streamMonitorTimer);
  blindState.streamMonitorTimer = null;
}

function stopStalledAutoResume(videos) {
  if (!videos.length || !blindState.syncPlayIntent) return;
  blindState.streamAutoResumeBlocked = true;
  blindState.syncPlayIntent = false;
  blindState.syncWaiting = false;
  blindState.syncAttempt += 1;
  clearRecoveryContext();
  cancelFrameSynchronization();
  pauseVideosInternally();
  byId("master-play").textContent = "等待重新加载";
  setHidden("media-reload", false);
  setSyncStatus("流式媒体 60 秒没有缓冲进展，已停止自动恢复；请点击“重新加载”。", "stalled");
}

function setVoteMediaBlocked(blocked) {
  const form = byId("vote-form");
  if (!form) return;
  form.dataset.mediaFatal = String(Boolean(blocked));
  const readOnly = Boolean(blindState.reviewReadOnly);
  form.querySelectorAll("input, select, textarea, button").forEach((control) => {
    control.disabled = Boolean(blocked) || readOnly;
  });
}

function clearMediaReloadTimer() {
  if (blindState.mediaReloadTimer !== null) {
    window.clearTimeout(blindState.mediaReloadTimer);
    blindState.mediaReloadTimer = null;
  }
}

function markTaskMediaFatal(message) {
  clearMediaReloadTimer();
  abortPreload();
  blindState.mediaFatal = true;
  blindState.mediaReloadPending = false;
  blindState.frameSequencePending = false;
  blindState.streamAutoResumeBlocked = true;
  blindState.syncPlayIntent = false;
  blindState.syncWaiting = false;
  blindState.syncAttempt += 1;
  clearRecoveryContext();
  cancelFrameSynchronization();
  pauseVideosInternally();
  setVoteMediaBlocked(true);
  const playButton = byId("master-play");
  if (playButton) playButton.textContent = "等待重新加载";
  const messagePanel = byId("message-panel");
  if (messagePanel) messagePanel.dataset.mediaFatal = "true";
  const reloadButton = byId("media-reload");
  if (reloadButton) {
    reloadButton.disabled = false;
    reloadButton.textContent = "重新加载";
  }
  setHidden("media-reload", false);
  setSyncStatus(message, "error");
}

function clearTaskMediaFatal() {
  clearMediaReloadTimer();
  blindState.mediaFatal = false;
  blindState.streamAutoResumeBlocked = false;
  setVoteMediaBlocked(blindState.frameSequencePending);
  const messagePanel = byId("message-panel");
  if (messagePanel && messagePanel.dataset.mediaFatal === "true") {
    delete messagePanel.dataset.mediaFatal;
    messagePanel.classList.add("hidden");
  }
  const reloadButton = byId("media-reload");
  if (reloadButton) {
    reloadButton.disabled = false;
    reloadButton.textContent = "重新加载";
  }
  setHidden("media-reload", true);
}

function finishMediaReloadIfReady() {
  if (
    !blindState.mediaReloadPending
    || blindState.mediaReloadGeneration !== blindState.mediaGeneration
    || blindState.mediaReloadScopeEpoch !== blindState.scopeEpoch
    || blindState.mediaReloadAttempt !== blindState.syncAttempt
  ) return;
  const videos = allVideos();
  if (
    videos.length !== 3
    || !videos.every((video) => isCurrentMedia(video) && !video.error && Number(video.readyState || 0) >= 3)
  ) return;
  blindState.mediaReloadPending = false;
  clearTaskMediaFatal();
  blindState.syncClock = playbackClock();
  byId("master-play").textContent = "播放所选";
  setSyncStatus("三路媒体重新加载成功并已对齐暂停；请点击播放。", "ready");
}

function finishFrameSequenceReloadIfReady() {
  if (
    !blindState.mediaReloadPending
    || blindState.mediaReloadGeneration !== blindState.mediaGeneration
    || blindState.mediaReloadScopeEpoch !== blindState.scopeEpoch
    || blindState.mediaReloadAttempt !== blindState.syncAttempt
  ) return;
  const images = allFrameSequenceImages();
  if (
    images.length !== 3
    || !images.every((image) => (
      image.dataset.reloadReady === "true"
      && Number(image.dataset.frameRequestGeneration) === blindState.mediaGeneration
      && Number(image.dataset.frameRequestScopeEpoch) === blindState.scopeEpoch
      && Number(image.dataset.frameRequestAttempt) === blindState.syncAttempt
    ))
  ) return;
  blindState.mediaReloadPending = false;
  clearTaskMediaFatal();
  setSyncStatus(
    `三路帧序列已重新加载并保持在第 ${blindState.frameIndex + 1} 帧。`,
    "ready",
  );
}

function reloadFrameSequenceMedia() {
  const images = allFrameSequenceImages();
  if (images.length !== 3) {
    markTaskMediaFatal("帧序列媒体不完整，无法安全重载；请联系组织者重新发布。");
    return;
  }
  abortPreload();
  blindState.scopeEpoch += 1;
  blindState.syncPlayIntent = false;
  blindState.syncWaiting = false;
  blindState.streamAutoResumeBlocked = false;
  blindState.syncAttempt += 1;
  clearRecoveryContext();
  cancelFrameSynchronization();
  blindState.mediaReloadPending = true;
  blindState.mediaReloadGeneration = blindState.mediaGeneration;
  blindState.mediaReloadScopeEpoch = blindState.scopeEpoch;
  blindState.mediaReloadAttempt = blindState.syncAttempt;
  setVoteMediaBlocked(true);
  const reloadButton = byId("media-reload");
  if (reloadButton) {
    reloadButton.disabled = true;
    reloadButton.textContent = "重新加载中…";
  }
  setHidden("media-reload", false);
  const nonce = `${Date.now()}-${blindState.syncAttempt}`;
  replaceFrameSequenceImages(blindState.frameIndex, nonce);
  const mediaGeneration = blindState.mediaGeneration;
  const scopeEpoch = blindState.scopeEpoch;
  const attempt = blindState.syncAttempt;
  clearMediaReloadTimer();
  blindState.mediaReloadTimer = window.setTimeout(() => {
    if (
      blindState.mediaReloadPending
      && mediaGeneration === blindState.mediaGeneration
      && scopeEpoch === blindState.scopeEpoch
      && attempt === blindState.syncAttempt
    ) {
      markTaskMediaFatal("帧序列重新加载 60 秒仍未完成；请检查媒体服务后重试。");
    }
  }, STREAM_NO_PROGRESS_RELOAD_MS);
  setSyncStatus(
    `正在重新加载第 ${blindState.frameIndex + 1} 帧的匿名 GT/A/B；三路全部成功前投票保持禁用。`,
    "stalled",
  );
}

function reloadTaskMedia() {
  const videos = allVideos();
  const frameImages = allFrameSequenceImages();
  if (frameImages.length) {
    if (videos.length) {
      markTaskMediaFatal("任务混用了视频与帧序列，无法安全重载；请联系组织者重新发布。");
      return;
    }
    reloadFrameSequenceMedia();
    return;
  }
  if (!videos.length) return;
  const clock = blindState.syncClock || playbackClock() || referenceVideo();
  const frameIndex = clock
    ? frameIndexFromTime(Number(clock.currentTime || 0), Number(clock.duration))
    : clampFrameIndex(byId("master-seek") && byId("master-seek").value);
  abortPreload();
  blindState.scopeEpoch += 1;
  blindState.syncPlayIntent = false;
  blindState.syncWaiting = false;
  blindState.streamAutoResumeBlocked = false;
  blindState.syncAttempt += 1;
  clearRecoveryContext();
  cancelFrameSynchronization();
  pauseVideosInternally(allVideos());
  releaseBlobUrls();
  blindState.mediaReloadPending = true;
  blindState.mediaReloadGeneration = blindState.mediaGeneration;
  blindState.mediaReloadScopeEpoch = blindState.scopeEpoch;
  blindState.mediaReloadAttempt = blindState.syncAttempt;
  setVoteMediaBlocked(true);
  const reloadButton = byId("media-reload");
  if (reloadButton) {
    reloadButton.disabled = true;
    reloadButton.textContent = "重新加载中…";
  }
  videos.forEach((video) => {
    const sourceUrl = String(video.dataset.sourceUrl || "");
    video.dataset.preloadDecision = "streaming";
    video.dataset.preloadState = "streaming";
    video.dataset.syncBuffering = "true";
    video.dataset.streamLastProgressAt = String(Date.now());
    video.dataset.streamProgressMarker = "";
    video.dataset.pendingFrameIndex = String(frameIndex);
    video.preload = "auto";
    video.src = sourceUrl;
    video.load();
  });
  setHidden("media-reload", false);
  setSyncStatus("正在重新加载三路媒体；成功后会保持对齐暂停，不会自动播放。", "stalled");
}

function monitorStreamingWatermarks() {
  if (blindState.mediaReloadPending) {
    const reloadingVideos = allVideos();
    reloadingVideos.forEach((video) => recordMediaProgress(video));
    finishMediaReloadIfReady();
    if (blindState.mediaReloadPending) {
      const now = Date.now();
      const reloadStalled = reloadingVideos.some((video) => (
        now - Number(video.dataset.streamLastProgressAt || now) >= STREAM_NO_PROGRESS_RELOAD_MS
      ));
      if (reloadStalled) {
        blindState.mediaReloadPending = false;
        blindState.mediaFatal = true;
        blindState.streamAutoResumeBlocked = true;
        const reloadButton = byId("media-reload");
        if (reloadButton) {
          reloadButton.disabled = false;
          reloadButton.textContent = "重新加载";
        }
        setHidden("media-reload", false);
        setSyncStatus("重新加载 60 秒仍无进展，请检查媒体服务后再次点击“重新加载”。", "error");
      }
    }
  }
  const videos = activeVideos();
  videos.forEach((video) => recordMediaProgress(video));
  if (!blindState.syncPlayIntent || !videos.length) return;
  if (!blindState.syncWaiting) {
    const low = videos.find((video) => mediaBelowLowWatermark(video));
    if (low) handleMediaWaiting({ currentTarget: low });
  }
  if (!blindState.syncWaiting) return;
  maybeResumeBufferedPlayback();
  if (!blindState.syncWaiting) return;
  const now = Date.now();
  const stalled = videos.filter((video) => (
    video.dataset.preloadState !== "blob"
    && !mediaReadyForPlayback(video)
    && now - Number(video.dataset.streamLastProgressAt || now) >= STREAM_NO_PROGRESS_RELOAD_MS
  ));
  stopStalledAutoResume(stalled);
}

function startStreamMonitoring() {
  stopStreamMonitoring();
  allVideos().forEach((video) => recordMediaProgress(video));
  blindState.streamMonitorTimer = setInterval(
    monitorStreamingWatermarks,
    STREAM_MONITOR_INTERVAL_MS,
  );
}

async function playSynchronizedVideos(automatic = false) {
  const mediaGeneration = blindState.mediaGeneration;
  const scopeEpoch = blindState.scopeEpoch;
  const videos = activeVideos();
  if (!videos.length) return;
  if (blindState.mediaFatal || blindState.streamAutoResumeBlocked) {
    setHidden("media-reload", false);
    setSyncStatus("媒体需要重新加载后才能继续播放。", "stalled");
    return;
  }
  const attempt = ++blindState.syncAttempt;
  const playbackIsCurrent = () => (
    mediaGeneration === blindState.mediaGeneration
    && scopeEpoch === blindState.scopeEpoch
    && attempt === blindState.syncAttempt
    && videos.every((video) => isCurrentMedia(video) && isActiveMedia(video))
  );
  if (!automatic && blindState.preloadPromise) {
    setSyncStatus("正在确定三路短视频的完整缓存策略…", "ready");
    try {
      await blindState.preloadPromise;
    } catch (_error) {
      // Source preparation owns its fallback; context checks below reject stale work.
    }
    if (!playbackIsCurrent()) return;
  }
  const clock = playbackClock() || videos[0];
  blindState.syncClock = clock;
  const requiresInitialBuffer = blindState.initialBufferScopeEpoch !== scopeEpoch;
  if (!automatic && requiresInitialBuffer && !videos.every(mediaReadyForInitialPlayback)) {
    blindState.syncPlayIntent = true;
    blindState.syncWaiting = true;
    videos.forEach((video) => {
      video.dataset.syncBuffering = "true";
      video.dataset.streamLastProgressAt = String(Date.now());
      video.preload = "auto";
    });
    markRecoveryContext(true);
    pauseVideosInternally();
    byId("master-play").textContent = "首次缓冲中";
    const blocked = videos.find((video) => !mediaReadyForInitialPlayback(video)) || videos[0];
    setSyncStatus(`${bufferingDiagnostic(blocked, STREAM_INITIAL_WATER_SECONDS)}；首次播放尚未调用 play。`, "stalled");
    return;
  }
  blindState.initialBufferScopeEpoch = scopeEpoch;
  blindState.syncPlayIntent = true;
  blindState.syncWaiting = false;
  clearRecoveryContext();
  videos.forEach((video) => { video.dataset.syncBuffering = "false"; });
  applyPlaybackSettings();
  synchronizeFollowers(Number(clock.currentTime || 0), true);
  setSyncStatus(
    automatic
      ? "缓冲完成，正在恢复所选媒体的同步播放…"
      : `正在对齐并启动${playbackScopeLabel(blindState.playbackScope)}…`,
    "ready",
  );
  try {
    await Promise.all(videos.map((video) => {
      const result = video.play();
      return result && typeof result.then === "function" ? result : Promise.resolve();
    }));
  } catch (error) {
    if (
      !playbackIsCurrent()
      || blindState.syncWaiting
    ) return;
    if (isAbortPlayError(error)) {
      pauseSynchronizedPlayback(
        "播放启动被浏览器中断，三路视频已对齐暂停；请点击继续。",
        "stalled",
      );
      byId("master-play").textContent = "点击继续播放";
      return;
    }
    if (automatic && error && error.name === "NotAllowedError") {
      blindState.syncPlayIntent = false;
      blindState.syncAttempt += 1;
      pauseVideosInternally();
      synchronizeFollowers(Number(clock.currentTime || 0), true);
      byId("master-play").textContent = "点击继续播放";
      setSyncStatus("三路视频已对齐；浏览器阻止自动恢复，请点击继续。", "stalled");
      return;
    }
    handlePlayFailure(error);
    return;
  }
  if (!playbackIsCurrent() || !blindState.syncPlayIntent || blindState.syncWaiting) return;
  startFrameSynchronization();
  byId("master-play").textContent = "暂停所选";
  setSyncStatus(`${playbackScopeLabel(blindState.playbackScope)}同步播放中。`, "playing");
}

async function toggleMasterPlayback() {
  const videos = activeVideos();
  if (!videos.length) return;
  if (blindState.syncPlayIntent || videos.some((video) => !video.paused)) {
    pauseSynchronizedPlayback();
    return;
  }
  await playSynchronizedVideos(false);
}

function handleReferencePlay(event) {
  const clock = event.currentTarget;
  if (!isActiveClock(clock)) return;
  blindState.syncClock = clock;
  if (!blindState.syncPlayIntent) {
    playSynchronizedVideos(false).catch(handlePlayFailure);
    return;
  }
  if (blindState.syncWaiting) return;
  startFrameSynchronization();
}

function handleMediaPause(event) {
  const clock = event.currentTarget;
  if (!isCurrentMedia(clock)) return;
  const internalPausePending = Number(clock.dataset.syncInternalPausePending || 0);
  if (internalPausePending > 0) {
    if (internalPausePending === 1) delete clock.dataset.syncInternalPausePending;
    else clock.dataset.syncInternalPausePending = String(internalPausePending - 1);
    return;
  }
  if (!isActiveMedia(clock)) return;
  if (!blindState.syncPlayIntent || blindState.syncWaiting || clock.ended) return;
  pauseSynchronizedPlayback();
}

function handleMediaEnded(event) {
  const clock = event.currentTarget;
  if (!isActiveMedia(clock)) return;
  if (clock.loop && blindState.syncPlayIntent) return;
  const anchor = Number(clock.currentTime || clock.duration || 0);
  blindState.syncWaiting = false;
  blindState.syncPlayIntent = false;
  clearRecoveryContext();
  blindState.syncAttempt += 1;
  cancelFrameSynchronization();
  pauseVideosInternally();
  alignAllVideos(anchor);
  byId("master-play").textContent = "播放所选";
  setSyncStatus("播放已结束，三路画面已停在对齐位置。", "ready");
}

function handleMediaWaiting(event) {
  const media = event.currentTarget;
  if (
    !isActiveMedia(media)
    || !blindState.syncPlayIntent
    || blindState.streamAutoResumeBlocked
  ) return;
  const enteredBuffering = !blindState.syncWaiting;
  const initialBuffering = blindState.syncInitialBuffering;
  media.dataset.syncBuffering = "true";
  blindState.syncWaiting = true;
  blindState.syncAttempt += 1;
  markRecoveryContext(initialBuffering);
  cancelFrameSynchronization();
  pauseVideosInternally();
  if (enteredBuffering) {
    const now = Date.now();
    activeVideos().forEach((video) => {
      video.dataset.streamLastProgressAt = String(now);
    });
  }
  byId("master-play").textContent = "缓冲中";
  const target = initialBuffering ? STREAM_INITIAL_WATER_SECONDS : STREAM_RESUME_WATER_SECONDS;
  setSyncStatus(`${bufferingDiagnostic(media, target)}；所选视频已对齐暂停。`, "stalled");
}

function maybeResumeBufferedPlayback() {
  if (
    !blindState.syncPlayIntent
    || !blindState.syncWaiting
    || !recoveryContextIsCurrent()
  ) return;
  const videos = activeVideos();
  if (!videos.length || !videos.every(mediaReadyForPlayback)) return;
  blindState.syncWaiting = false;
  const clock = blindState.syncClock || playbackClock();
  if (clock) synchronizeFollowers(Number(clock.currentTime || 0), true);
  clearRecoveryContext();
  playSynchronizedVideos(true).catch(handlePlayFailure);
}

function handleMediaCanPlay(event) {
  const media = event.currentTarget;
  if (!isCurrentMedia(media)) return;
  recordMediaProgress(media);
  finishMediaReloadIfReady();
  if (!isActiveMedia(media)) return;
  if (blindState.syncWaiting && !recoveryContextIsCurrent()) return;
  media.dataset.syncBuffering = "false";
  maybeResumeBufferedPlayback();
}

function handleMediaPlaying(event) {
  const media = event.currentTarget;
  if (!isActiveMedia(media)) return;
  media.dataset.syncBuffering = "false";
  if (
    blindState.syncPlayIntent
    && !blindState.syncWaiting
    && activeVideos().every((video) => !video.paused)
  ) {
    setSyncStatus(`${playbackScopeLabel(blindState.playbackScope)}同步播放中。`, "playing");
  }
}

function handleMediaErrorState(event) {
  const media = event.currentTarget;
  if (!isCurrentMedia(media)) return;
  const detail = media.error && media.error.message
    ? media.error.message
    : `MediaError ${Number((media.error && media.error.code) || 0)}`;
  markTaskMediaFatal("检测到真实媒体解码或网络错误；投票已禁用，请重新加载。");
  showError(new Error(`${media.dataset.mediaLabel || "媒体"} 播放失败：${detail}`));
}

function seekVideos(value) {
  const index = clampFrameIndex(value);
  const reliable = frameSteppingIsReliable(playbackClock());
  blindState.videoScrubbing = false;
  blindState.videoScrubDirty = false;
  blindState.syncPlayIntent = false;
  blindState.syncWaiting = false;
  clearRecoveryContext();
  blindState.syncAttempt += 1;
  cancelFrameSynchronization();
  pauseVideosInternally(allVideos());
  allVideos().forEach((video) => seekVideoToFrame(video, index));
  blindState.syncClock = playbackClock();
  updateVideoFrameControls(index);
  const playButton = byId("master-play");
  if (playButton) playButton.textContent = "播放所选";
  setSyncStatus(
    reliable
      ? `三路视频已对齐到第 ${index + 1} 帧的中点并暂停。`
      : "已记录目标位置；等待可靠 FPS 或媒体时长后定位，逐帧按钮暂不可用。",
    "ready",
  );
}

function previewVideoFrame(value) {
  if (!blindState.videoScrubbing) beginVideoFrameScrub();
  blindState.videoScrubDirty = true;
  updateVideoFrameControls(clampFrameIndex(value), true);
}

function beginVideoFrameScrub() {
  if (!blindState.videoScrubbing) blindState.videoScrubDirty = false;
  blindState.videoScrubbing = true;
}

function commitVideoFrameScrub(value) {
  if (!blindState.videoScrubbing) return;
  const shouldSeek = blindState.videoScrubDirty;
  blindState.videoScrubbing = false;
  blindState.videoScrubDirty = false;
  if (shouldSeek) {
    seekVideos(value);
    return;
  }
  const clock = playbackClock();
  if (clock) updateMasterSeek(Number(clock.currentTime || 0), Number(clock.duration));
}

function finishInterruptedVideoScrub() {
  if (!blindState.videoScrubbing) return;
  const seek = byId("master-seek");
  commitVideoFrameScrub(seek ? seek.value : blindState.videoTargetFrameIndex);
}

function stepVideoFrame(delta) {
  if (!frameSteppingIsReliable(playbackClock())) return;
  const seek = byId("master-seek");
  seekVideos(clampFrameIndex(Number((seek && seek.value) || 0) + Number(delta || 0)));
}

function previewSequenceFrame(value) {
  const index = clampFrameIndex(value);
  byId("frame-label").textContent = `目标帧 ${index + 1}/${taskFrameCount()}`;
  updateSequenceFrameControls(index);
}

function updateFrame(value) {
  blindState.frameIndex = clampFrameIndex(value);
  const total = taskFrameCount();
  byId("frame-range").value = String(blindState.frameIndex);
  byId("frame-label").textContent = `帧 ${blindState.frameIndex + 1}/${total}`;
  updateSequenceFrameControls(blindState.frameIndex);
  replaceFrameSequenceImages(blindState.frameIndex);
}

function stepFrameSequence(delta) {
  const range = byId("frame-range");
  updateFrame(clampFrameIndex(Number((range && range.value) || blindState.frameIndex) + Number(delta || 0)));
}

function startLeaseHeartbeat(task) {
  if (blindState.leaseTimer) clearInterval(blindState.leaseTimer);
  if (!task || !task.token) return;
  blindState.leaseTimer = setInterval(() => {
    if (document.hidden) return;
    blindApi(`/api/blind/${encodeURIComponent(blindState.token)}/tasks/${encodeURIComponent(task.token)}/heartbeat`, {
      method: "POST",
      body: JSON.stringify({ evaluator_id: blindState.evaluatorId }),
    }).catch(() => {});
  }, 60_000);
}

function pauseForPageCache() {
  blindState.syncPlayIntent = false;
  blindState.syncWaiting = false;
  blindState.videoScrubbing = false;
  blindState.videoScrubDirty = false;
  clearRecoveryContext();
  cancelFrameSynchronization();
  abortPreload();
  clearMediaReloadTimer();
  pauseVideosInternally(allVideos());
  stopStreamMonitoring();
  const playButton = byId("master-play");
  if (playButton) playButton.textContent = "播放所选";
}

function handleBlindPageHide(event) {
  if (event && event.persisted) {
    pauseForPageCache();
    return;
  }
  stopSynchronization();
}

function handleBlindPageShow(event) {
  if (!event || !event.persisted) return;
  const videos = allVideos();
  const task = currentTask();
  if (videos.length) {
    const now = String(Date.now());
    videos.forEach((video) => { video.dataset.streamLastProgressAt = now; });
    if (videos.some((video) => video.dataset.preloadDecision === "pending")) {
      startTaskVideoPreparation(task);
    }
    startStreamMonitoring();
  }
  if (blindState.frameSequencePending) armFrameSequenceWatchdog();
  finishMediaReloadIfReady();
  finishFrameSequenceRequestIfReady();
  finishFrameSequenceReloadIfReady();
  if (task && !task.review) startLeaseHeartbeat(task);
  if (blindState.mediaReloadPending || blindState.frameSequencePending) return;
  if (blindState.mediaFatal || blindState.streamAutoResumeBlocked) {
    setHidden("media-reload", false);
    setSyncStatus("页面已恢复；媒体仍需重新加载后才能继续。", "stalled");
    return;
  }
  if (videos.length) {
    blindState.syncClock = playbackClock();
    setSyncStatus("页面已恢复；所选视频保持对齐暂停，请点击播放。", "ready");
  }
}

function initializeBlindPage() {
  blindState.wipeDivider = byId("wipe-divider");
  const syncStatus = byId("sync-status");
  if (syncStatus) byId("playback-controls").appendChild(syncStatus);
  blindState.wipeSupported = supportsWipeView();
  updateWipePosition(50);
  setViewMode(blindState.wipeSupported ? "wipe" : "full");
  if (!blindState.wipeSupported) {
    const wipeButton = byId("view-wipe");
    if (wipeButton) wipeButton.title = "当前浏览器不支持重叠裁剪，请使用完整视图";
    setSyncStatus("当前浏览器不支持重叠裁剪，已切换到完整视图。", "ready");
  }
  byId("session-form").addEventListener("submit", (event) => saveSession(event).catch(showError));
  byId("vote-form").addEventListener("submit", (event) => submitVote(event).catch(showError));
  byId("master-play").addEventListener("click", () => toggleMasterPlayback().catch(showError));
  byId("master-seek").addEventListener("pointerdown", beginVideoFrameScrub);
  byId("master-seek").addEventListener("input", (event) => previewVideoFrame(event.currentTarget.value));
  byId("master-seek").addEventListener("change", (event) => commitVideoFrameScrub(event.currentTarget.value));
  byId("master-seek").addEventListener("pointerup", finishInterruptedVideoScrub);
  byId("master-seek").addEventListener("pointercancel", finishInterruptedVideoScrub);
  byId("master-seek").addEventListener("lostpointercapture", finishInterruptedVideoScrub);
  byId("master-seek").addEventListener("blur", finishInterruptedVideoScrub);
  byId("master-prev").addEventListener("click", () => stepVideoFrame(-1));
  byId("master-next").addEventListener("click", () => stepVideoFrame(1));
  byId("master-rate").addEventListener("change", () => applyPlaybackSettings());
  byId("master-loop").addEventListener("change", () => applyPlaybackSettings());
  byId("media-reload").addEventListener("click", reloadTaskMedia);
  byId("frame-range").addEventListener("input", (event) => previewSequenceFrame(event.currentTarget.value));
  byId("frame-range").addEventListener("change", (event) => updateFrame(event.currentTarget.value));
  byId("frame-prev").addEventListener("click", () => stepFrameSequence(-1));
  byId("frame-next").addEventListener("click", () => stepFrameSequence(1));
  byId("view-wipe").addEventListener("click", () => setViewMode("wipe"));
  byId("view-full").addEventListener("click", () => setViewMode("full"));
  byId("playback-scope").addEventListener("click", (event) => {
    const button = event.target.closest("[data-playback-scope]");
    if (button) setPlaybackScope(String(button.dataset.playbackScope || "all"));
  });
  byId("review-back").addEventListener("click", () => leaveReview().catch(showError));
  byId("review-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-review-task]");
    if (button) openReview(button.dataset.reviewTask).catch(showError);
  });
  if (blindState.wipeDivider) {
    blindState.wipeDivider.addEventListener("input", (event) => updateWipePosition(event.currentTarget.value));
  }
  window.addEventListener("pagehide", handleBlindPageHide);
  window.addEventListener("pageshow", handleBlindPageShow);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    if (blindState.payload && blindState.payload.task) startLeaseHeartbeat(blindState.payload.task);
    else if (
      String(
        blindState.payload && blindState.payload.campaign
          ? blindState.payload.campaign.status || ""
          : "",
      ) === "preparing"
      || (blindState.evaluatorName
        && !(blindState.payload && blindState.payload.progress && blindState.payload.progress.complete))
    ) loadBlindPayload().catch(showError);
  });
  loadBlindPayload().catch(showError);
}

try {
  initializeBlindPage();
  window.__vfievalBlindReady = true;
} catch (error) {
  showError(error);
}
