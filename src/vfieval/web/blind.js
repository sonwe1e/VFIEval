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
  taskStartedAt: 0,
  frameIndex: 0,
  mediaGeneration: 0,
  viewMode: "wipe",
  wipeSupported: true,
  wipeDivider: null,
  syncClock: null,
  syncExpectedPlaying: false,
  syncWaiting: false,
  syncFrameCallbackId: null,
  syncFrameCallbackGeneration: 0,
  syncUsesFrameCallback: false,
  leaseTimer: null,
  retryTimer: null,
};

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

function currentTask() {
  return (blindState.payload && (blindState.payload.task || blindState.payload.next_task)) || null;
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

function setViewMode(mode) {
  const requested = mode === "full" ? "full" : "wipe";
  const nextMode = requested === "wipe" && !blindState.wipeSupported ? "full" : requested;
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
  showError(new Error(`${label} 无法加载，请刷新后重试；若仍失败，请联系组织者重新发布。`));
}

function createMediaNode(task, side, label) {
  const kind = task[`${side}_media_kind`] || "video";
  const url = task[`${side}_url`];
  if (kind === "frame_sequence") {
    const image = document.createElement("img");
    image.alt = label;
    image.dataset.mediaSide = side;
    image.dataset.mediaGeneration = String(blindState.mediaGeneration);
    image.dataset.frameBase = url;
    image.src = withFrame(url, blindState.frameIndex);
    image.addEventListener("load", () => updateMediaAspectRatio(image));
    image.addEventListener("error", () => {
      if (isCurrentMedia(image)) mediaLoadError(label);
    });
    return image;
  }

  const video = document.createElement("video");
  video.controls = false;
  video.playsInline = true;
  video.preload = "metadata";
  video.dataset.mediaSide = side;
  video.dataset.mediaGeneration = String(blindState.mediaGeneration);
  const rateControl = byId("master-rate");
  const loopControl = byId("master-loop");
  video.playbackRate = Number((rateControl && rateControl.value) || 1);
  video.loop = Boolean(loopControl && loopControl.checked);
  video.src = url;
  video.addEventListener("loadedmetadata", () => updateMediaAspectRatio(video));
  video.addEventListener("resize", () => updateMediaAspectRatio(video));
  video.addEventListener("waiting", handleMediaWaiting);
  video.addEventListener("error", () => {
    if (isCurrentMedia(video)) mediaLoadError(label);
  });
  if (side === "reference") {
    video.addEventListener("timeupdate", syncFromClockVideo);
    video.addEventListener("play", handleReferencePlay);
    video.addEventListener("pause", handleReferencePause);
    video.addEventListener("seeking", handleReferenceSeeking);
    video.addEventListener("seeked", handleReferenceSeeked);
    video.addEventListener("ratechange", handleReferenceRateChange);
    video.addEventListener("ended", handleReferenceEnded);
  }
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

function renderTask(task) {
  stopSynchronization();
  blindState.mediaGeneration += 1;
  setHidden("task-panel", !task);
  if (!task) {
    if (blindState.leaseTimer) clearInterval(blindState.leaseTimer);
    blindState.leaseTimer = null;
    replaceContent(byId("media-grid"));
    byId("master-play").textContent = "全部播放";
    setSyncStatus("当前没有待评媒体。", "ready");
    return;
  }
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
  const reasons = byId("quality-reasons");
  replaceContent(reasons, ...(task.quality_reasons || []).map((reason) => {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = "reason";
    input.value = reason;
    label.append(input, document.createTextNode(reason));
    return label;
  }));
  const frameCount = Math.max(1, Number(task.frame_count || 1));
  const mediaKinds = new Set(["reference", "left", "right"].map((side) => task[`${side}_media_kind`] || "video"));
  const hasFrames = mediaKinds.has("frame_sequence");
  if (mediaKinds.size > 1) {
    showError(new Error("这个任务混用了视频与帧序列，无法提供可靠同步播放，请联系组织者重新发布。"));
    byId("vote-form").classList.add("hidden");
  } else {
    byId("vote-form").classList.remove("hidden");
  }
  setHidden("frame-control", !hasFrames);
  setHidden("playback-controls", hasFrames);
  byId("frame-range").max = String(frameCount - 1);
  byId("frame-range").value = "0";
  byId("frame-label").textContent = `帧 1/${frameCount}`;
  byId("master-seek").value = "0";
  byId("master-play").textContent = "全部播放";
  if (hasFrames) {
    setSyncStatus("三路帧序列已定位到同一帧。", "ready");
  } else {
    blindState.syncClock = reference;
    setSyncStatus("三路视频已就绪，播放时将以参考 GT 为时钟。", "ready");
  }
  blindState.taskStartedAt = Date.now();
  startLeaseHeartbeat(task);
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

function renderPayload(payload) {
  blindState.payload = payload;
  const campaign = payload.campaign || {};
  const progress = payload.progress || {};
  byId("campaign-title").textContent = campaign.public_title || campaign.title || "匿名视频质量评测";
  byId("progress").textContent = `${Number(progress.completed || 0)}/${Number(progress.total || 0)} 已完成`;
  setHidden("session-panel", Boolean(blindState.evaluatorName));
  renderTask(payload.task || payload.next_task || null);
  const complete = Boolean(progress.complete);
  setHidden("complete-panel", !complete);
  if (complete) renderResults(payload.results);
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
    renderPayload(await blindApi(`/api/blind/${encodeURIComponent(blindState.token)}`));
    byId("progress").textContent = "等待加入";
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
  const submitter = event.submitter || document.activeElement;
  const task = blindState.payload && (blindState.payload.task || blindState.payload.next_task);
  if (!task || !submitter || !submitter.value) return;
  const buttons = voteForm.querySelectorAll("button");
  buttons.forEach((button) => { button.disabled = true; });
  const form = new FormData(voteForm);
  try {
    const payload = await blindApi(
      `/api/blind/${encodeURIComponent(blindState.token)}/tasks/${encodeURIComponent(task.token)}/vote`,
      {
        method: "POST",
        body: JSON.stringify({
          evaluator_id: blindState.evaluatorId,
          choice: submitter.value,
          reasons: form.getAll("reason"),
          confidence: String(form.get("confidence") || ""),
          note: String(form.get("note") || ""),
          duration_ms: Math.max(0, Date.now() - blindState.taskStartedAt),
        }),
      },
    );
    voteForm.reset();
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

function activeVideos() {
  return Array.from(document.querySelectorAll("#media-grid video"));
}

function referenceVideo() {
  return activeVideos().find((video) => video.dataset.mediaSide === "reference") || null;
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

function updateMasterSeek(currentTime, duration) {
  if (!Number.isFinite(duration) || duration <= 0) return;
  const seek = byId("master-seek");
  if (seek) seek.value = String(Math.round((Number(currentTime || 0) / duration) * 1000));
}

function synchronizeFollowers(anchor, force = false) {
  const clock = blindState.syncClock || referenceVideo();
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

function stopSynchronization() {
  blindState.syncExpectedPlaying = false;
  blindState.syncWaiting = false;
  cancelFrameSynchronization();
  activeVideos().forEach((video) => {
    try {
      video.pause();
    } catch (_error) {
      // A detached or failed media element is safe to abandon during task replacement.
    }
  });
  blindState.syncClock = null;
}

function startFrameSynchronization() {
  cancelFrameSynchronization();
  const clock = blindState.syncClock || referenceVideo();
  if (
    !clock
    || !blindState.syncExpectedPlaying
    || typeof clock.requestVideoFrameCallback !== "function"
  ) return;
  blindState.syncClock = clock;
  const generation = blindState.syncFrameCallbackGeneration;
  const onFrame = (_now, metadata) => {
    blindState.syncFrameCallbackId = null;
    if (
      generation !== blindState.syncFrameCallbackGeneration
      || !blindState.syncExpectedPlaying
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

function applyPlaybackSettings() {
  const rateControl = byId("master-rate");
  const loopControl = byId("master-loop");
  const rate = Number((rateControl && rateControl.value) || 1);
  const loop = Boolean(loopControl && loopControl.checked);
  activeVideos().forEach((video) => {
    video.playbackRate = rate;
    video.loop = loop;
  });
}

function pauseSynchronizedPlayback(message = "已暂停，三路媒体保持在同一时间位置。", state = "ready") {
  blindState.syncExpectedPlaying = false;
  cancelFrameSynchronization();
  activeVideos().forEach((video) => video.pause());
  const playButton = byId("master-play");
  if (playButton) playButton.textContent = "全部播放";
  setSyncStatus(message, state);
}

function handlePlayFailure(_error) {
  blindState.syncWaiting = false;
  pauseSynchronizedPlayback("浏览器未能同时启动三路媒体，已全部暂停。", "error");
  showError(new Error("无法同时播放三路媒体，请重试。"));
}

async function playSynchronizedVideos() {
  const mediaGeneration = blindState.mediaGeneration;
  const videos = activeVideos();
  if (!videos.length) return;
  const playbackIsCurrent = () => (
    mediaGeneration === blindState.mediaGeneration
    && videos.every((video) => isCurrentMedia(video))
  );
  const clock = videos.find((video) => video.dataset.mediaSide === "reference") || videos[0];
  blindState.syncClock = clock;
  blindState.syncWaiting = false;
  applyPlaybackSettings();
  synchronizeFollowers(Number(clock.currentTime || 0), true);
  blindState.syncExpectedPlaying = true;
  setSyncStatus("正在对齐并启动三路视频…", "ready");
  try {
    await Promise.all(videos.map((video) => {
      const attempt = video.play();
      return attempt && typeof attempt.then === "function" ? attempt : Promise.resolve();
    }));
  } catch (_error) {
    videos.forEach((video) => video.pause());
    if (!playbackIsCurrent()) return;
    blindState.syncExpectedPlaying = false;
    cancelFrameSynchronization();
    setSyncStatus("浏览器未能同时启动三路媒体，已全部暂停。", "error");
    throw new Error("无法同时播放三路媒体，请确认浏览器允许播放后重试。");
  }
  if (!playbackIsCurrent()) {
    videos.forEach((video) => video.pause());
    return;
  }
  if (!blindState.syncExpectedPlaying) return;
  startFrameSynchronization();
  byId("master-play").textContent = "全部暂停";
  setSyncStatus("同步播放中；参考 GT 是主时钟。", "playing");
}

async function toggleMasterPlayback() {
  const videos = activeVideos();
  if (!videos.length) return;
  if (blindState.syncExpectedPlaying || videos.some((video) => !video.paused)) {
    blindState.syncWaiting = false;
    pauseSynchronizedPlayback();
    return;
  }
  await playSynchronizedVideos();
}

function syncFromClockVideo(event) {
  const video = event.currentTarget;
  if (!isCurrentMedia(video)) return;
  if (!Number.isFinite(video.duration) || video.duration <= 0) return;
  updateMasterSeek(Number(video.currentTime || 0), Number(video.duration));
  if (blindState.syncExpectedPlaying && !blindState.syncUsesFrameCallback) {
    synchronizeFollowers(Number(video.currentTime || 0));
  }
}

function handleReferencePlay(event) {
  const clock = event.currentTarget;
  if (!isCurrentMedia(clock)) return;
  const mediaGeneration = blindState.mediaGeneration;
  if (blindState.syncClock && blindState.syncClock !== clock) return;
  blindState.syncClock = clock;
  if (!blindState.syncExpectedPlaying) {
    blindState.syncExpectedPlaying = true;
    blindState.syncWaiting = false;
    applyPlaybackSettings();
    synchronizeFollowers(Number(clock.currentTime || 0), true);
    activeVideos().forEach((peer) => {
      if (peer === clock || !peer.paused) return;
      try {
        const attempt = peer.play();
        if (attempt && typeof attempt.catch === "function") {
          attempt.catch((error) => {
            if (mediaGeneration !== blindState.mediaGeneration || !isCurrentMedia(peer)) {
              peer.pause();
              return;
            }
            handlePlayFailure(error);
          });
        }
      } catch (error) {
        if (mediaGeneration === blindState.mediaGeneration && isCurrentMedia(peer)) {
          handlePlayFailure(error);
        }
      }
    });
  }
  startFrameSynchronization();
  byId("master-play").textContent = "全部暂停";
  setSyncStatus("同步播放中；参考 GT 是主时钟。", "playing");
}

function handleReferencePause(event) {
  const clock = event.currentTarget;
  if (!isCurrentMedia(clock)) return;
  if (!blindState.syncExpectedPlaying || blindState.syncWaiting || clock.ended) return;
  pauseSynchronizedPlayback();
}

function handleReferenceSeeking(event) {
  const clock = event.currentTarget;
  if (!isCurrentMedia(clock)) return;
  synchronizeFollowers(Number(clock.currentTime || 0), true);
  updateMasterSeek(Number(clock.currentTime || 0), Number(clock.duration));
  if (!blindState.syncWaiting) setSyncStatus("正在将三路视频定位到同一时间…", "ready");
}

function handleReferenceSeeked(event) {
  const clock = event.currentTarget;
  if (!isCurrentMedia(clock)) return;
  synchronizeFollowers(Number(clock.currentTime || 0), true);
  updateMasterSeek(Number(clock.currentTime || 0), Number(clock.duration));
  setSyncStatus(
    blindState.syncExpectedPlaying ? "同步播放中；参考 GT 是主时钟。" : "三路视频已定位到同一时间。",
    blindState.syncExpectedPlaying ? "playing" : "ready",
  );
}

function handleReferenceRateChange(event) {
  const clock = event.currentTarget;
  if (!isCurrentMedia(clock)) return;
  activeVideos().forEach((peer) => {
    if (peer !== clock) peer.playbackRate = clock.playbackRate;
  });
}

function handleReferenceEnded(event) {
  const clock = event.currentTarget;
  if (!isCurrentMedia(clock)) return;
  if (clock.loop && blindState.syncExpectedPlaying) return;
  blindState.syncWaiting = false;
  blindState.syncExpectedPlaying = false;
  cancelFrameSynchronization();
  activeVideos().forEach((peer) => {
    if (peer !== clock) {
      peer.pause();
      try {
        if (Number.isFinite(peer.duration)) peer.currentTime = peer.duration;
      } catch (_error) {
        // A failed follower can remain at its last decoded frame.
      }
    }
  });
  byId("master-play").textContent = "全部播放";
  setSyncStatus("播放已结束，三路视频均已停止。", "ready");
}

function handleMediaWaiting(event) {
  if (!isCurrentMedia(event.currentTarget)) return;
  if (!blindState.syncExpectedPlaying) return;
  blindState.syncWaiting = true;
  pauseSynchronizedPlayback(
    "检测到媒体缓冲，已暂停全部视频以保持对齐；缓冲完成后请重新播放。",
    "stalled",
  );
}

function seekVideos(value) {
  const clock = blindState.syncClock || referenceVideo();
  if (!clock || !Number.isFinite(clock.duration) || clock.duration <= 0) return;
  const target = clock.duration * (Number(value) / 1000);
  try {
    clock.currentTime = target;
  } catch (_error) {
    return;
  }
  synchronizeFollowers(target, true);
  updateMasterSeek(target, Number(clock.duration));
}

function updateFrame(value) {
  blindState.frameIndex = Number(value);
  const task = (blindState.payload && (blindState.payload.task || blindState.payload.next_task)) || {};
  const total = Math.max(1, Number(task.frame_count || 1));
  byId("frame-label").textContent = `帧 ${blindState.frameIndex + 1}/${total}`;
  document.querySelectorAll("[data-frame-base]").forEach((image) => { image.src = withFrame(image.dataset.frameBase, blindState.frameIndex); });
  setSyncStatus(`三路帧序列已定位到第 ${blindState.frameIndex + 1} 帧。`, "ready");
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

function initializeBlindPage() {
  blindState.wipeDivider = byId("wipe-divider");
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
  byId("master-seek").addEventListener("input", (event) => seekVideos(event.currentTarget.value));
  byId("master-rate").addEventListener("change", () => applyPlaybackSettings());
  byId("master-loop").addEventListener("change", () => applyPlaybackSettings());
  byId("frame-range").addEventListener("input", (event) => updateFrame(event.currentTarget.value));
  byId("view-wipe").addEventListener("click", () => setViewMode("wipe"));
  byId("view-full").addEventListener("click", () => setViewMode("full"));
  if (blindState.wipeDivider) {
    blindState.wipeDivider.addEventListener("input", (event) => updateWipePosition(event.currentTarget.value));
  }
  window.addEventListener("pagehide", stopSynchronization);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    if (blindState.payload && blindState.payload.task) startLeaseHeartbeat(blindState.payload.task);
    else if (
      blindState.evaluatorName
      && !(blindState.payload && blindState.payload.progress && blindState.payload.progress.complete)
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
