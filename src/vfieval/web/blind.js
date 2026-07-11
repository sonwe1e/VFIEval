const blindState = {
  token: decodeURIComponent(location.pathname.split("/").filter(Boolean).pop() || ""),
  evaluatorId: localStorage.getItem("vfieval-evaluator-id") || crypto.randomUUID(),
  evaluatorName: localStorage.getItem("vfieval-evaluator-name") || "",
  payload: null,
  taskStartedAt: 0,
  frameIndex: 0,
  leaseTimer: null,
  retryTimer: null,
};

localStorage.setItem("vfieval-evaluator-id", blindState.evaluatorId);
const byId = (id) => document.getElementById(id);

async function blindApi(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || response.statusText);
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
  byId(id)?.classList.toggle("hidden", Boolean(hidden));
}

function withFrame(url, frame) {
  const separator = String(url).includes("?") ? "&" : "?";
  return `${url}${separator}frame=${Number(frame)}`;
}

function mediaElement(task, side, label) {
  const kind = task[`${side}_media_kind`] || "video";
  const url = task[`${side}_url`];
  const card = document.createElement("article");
  card.className = "media-card";
  const heading = document.createElement("h3");
  heading.textContent = label;
  card.appendChild(heading);
  if (kind === "frame_sequence") {
    const image = document.createElement("img");
    image.alt = label;
    image.dataset.frameBase = url;
    image.src = withFrame(url, blindState.frameIndex);
    card.appendChild(image);
  } else {
    const video = document.createElement("video");
    video.controls = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.dataset.mediaSide = side;
    video.playbackRate = Number(byId("master-rate")?.value || 1);
    video.loop = Boolean(byId("master-loop")?.checked);
    video.src = url;
    if (side === "reference") video.addEventListener("timeupdate", syncFromClockVideo);
    card.appendChild(video);
  }
  return card;
}

function renderTask(task) {
  setHidden("task-panel", !task);
  if (!task) {
    if (blindState.leaseTimer) clearInterval(blindState.leaseTimer);
    blindState.leaseTimer = null;
    activeVideos().forEach((video) => video.pause());
    byId("media-grid").replaceChildren();
    byId("master-play").textContent = "全部播放";
    return;
  }
  byId("task-video").textContent = task.video_name || "视频";
  const grid = byId("media-grid");
  grid.replaceChildren(
    mediaElement(task, "reference", "参考 GT"),
    mediaElement(task, "left", "候选 A"),
    mediaElement(task, "right", "候选 B"),
  );
  const reasons = byId("quality-reasons");
  reasons.replaceChildren(...(task.quality_reasons || []).map((reason) => {
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
  blindState.frameIndex = 0;
  byId("frame-range").max = String(frameCount - 1);
  byId("frame-range").value = "0";
  byId("frame-label").textContent = `帧 1/${frameCount}`;
  blindState.taskStartedAt = Date.now();
  startLeaseHeartbeat(task);
}

function renderResults(results) {
  const host = byId("live-results");
  const ranking = results?.human?.ranking || results?.ranking || [];
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
    score.textContent = Number(row.score ?? row.win_rate ?? 0).toFixed(4);
    const detail = document.createElement("small");
    detail.textContent = row.ci95 ? `95% CI ${row.ci95[0]}–${row.ci95[1]}` : `${Number(row.votes || 0)} 票`;
    card.append(name, score, detail);
    grid.appendChild(card);
  });
  host.replaceChildren(grid);
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
    localStorage.removeItem("vfieval-evaluator-name");
    setHidden("session-panel", false);
    byId("progress").textContent = "会话已失效，请重新加入";
  }
}

async function saveSession(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const displayName = String(form.get("display_name") || "").trim();
  const payload = await blindApi(`/api/blind/${encodeURIComponent(blindState.token)}/session`, {
    method: "POST",
    body: JSON.stringify({ evaluator_id: blindState.evaluatorId, display_name: displayName }),
  });
  blindState.evaluatorName = displayName;
  localStorage.setItem("vfieval-evaluator-name", displayName);
  renderPayload(payload);
}

async function submitVote(event) {
  event.preventDefault();
  const submitter = event.submitter;
  const task = blindState.payload?.task || blindState.payload?.next_task;
  if (!task || !submitter?.value) return;
  const buttons = event.currentTarget.querySelectorAll("button");
  buttons.forEach((button) => { button.disabled = true; });
  const form = new FormData(event.currentTarget);
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
    event.currentTarget.reset();
    renderPayload({
      ...blindState.payload,
      ...payload,
      campaign: payload.campaign || blindState.payload?.campaign || {},
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

async function toggleMasterPlayback() {
  const videos = activeVideos();
  if (!videos.length) return;
  const shouldPlay = videos.every((video) => video.paused);
  if (shouldPlay) {
    const clock = videos.find((video) => video.dataset.mediaSide === "reference") || videos[0];
    const anchor = Number(clock.currentTime || 0);
    videos.forEach((video) => { video.currentTime = anchor; });
    await Promise.allSettled(videos.map((video) => video.play()));
  } else {
    videos.forEach((video) => video.pause());
  }
  byId("master-play").textContent = shouldPlay ? "全部暂停" : "全部播放";
}

function syncFromClockVideo(event) {
  const video = event.currentTarget;
  if (!Number.isFinite(video.duration) || video.duration <= 0) return;
  byId("master-seek").value = String(Math.round((video.currentTime / video.duration) * 1000));
  activeVideos().forEach((peer) => {
    if (peer === video || !Number.isFinite(peer.duration)) return;
    if (Math.abs(Number(peer.currentTime || 0) - Number(video.currentTime || 0)) > 0.08) {
      peer.currentTime = Math.min(Number(video.currentTime || 0), peer.duration || Number(video.currentTime || 0));
    }
  });
}

function seekVideos(value) {
  activeVideos().forEach((video) => {
    if (Number.isFinite(video.duration) && video.duration > 0) video.currentTime = video.duration * (Number(value) / 1000);
  });
}

function updateFrame(value) {
  blindState.frameIndex = Number(value);
  const task = blindState.payload?.task || blindState.payload?.next_task || {};
  const total = Math.max(1, Number(task.frame_count || 1));
  byId("frame-label").textContent = `帧 ${blindState.frameIndex + 1}/${total}`;
  document.querySelectorAll("[data-frame-base]").forEach((image) => { image.src = withFrame(image.dataset.frameBase, blindState.frameIndex); });
}

function startLeaseHeartbeat(task) {
  if (blindState.leaseTimer) clearInterval(blindState.leaseTimer);
  if (!task?.token) return;
  blindState.leaseTimer = setInterval(() => {
    if (document.hidden) return;
    blindApi(`/api/blind/${encodeURIComponent(blindState.token)}/tasks/${encodeURIComponent(task.token)}/heartbeat`, {
      method: "POST",
      body: JSON.stringify({ evaluator_id: blindState.evaluatorId }),
    }).catch(() => {});
  }, 60_000);
}

byId("session-form").addEventListener("submit", (event) => saveSession(event).catch(showError));
byId("vote-form").addEventListener("submit", (event) => submitVote(event).catch(showError));
byId("master-play").addEventListener("click", () => toggleMasterPlayback().catch(showError));
byId("master-seek").addEventListener("input", (event) => seekVideos(event.currentTarget.value));
byId("master-rate").addEventListener("change", (event) => activeVideos().forEach((video) => { video.playbackRate = Number(event.currentTarget.value); }));
byId("master-loop").addEventListener("change", (event) => activeVideos().forEach((video) => { video.loop = event.currentTarget.checked; }));
byId("frame-range").addEventListener("input", (event) => updateFrame(event.currentTarget.value));
document.addEventListener("visibilitychange", () => {
  if (document.hidden) return;
  if (blindState.payload?.task) startLeaseHeartbeat(blindState.payload.task);
  else if (blindState.evaluatorName && !blindState.payload?.progress?.complete) loadBlindPayload().catch(showError);
});

loadBlindPayload().catch(showError);
