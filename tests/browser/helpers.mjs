import { expect } from "@playwright/test";

export const baseRun = {
  id: 1,
  name: "Browser fixture run",
  status: "running",
  content_revision: 7,
  progress_current: 1,
  progress_total: 3,
  device: "cpu",
  metadata: {
    run_type: "model_inference",
    model_file: "test_average.py",
    video_group: "fixtures",
    output_dir: ".vfieval/runs/1",
  },
  metrics: ["lpips_vit_patch"],
  jobs: [],
  feedback: [],
  artifact_summary: { total: 3 },
  result: {},
};

export const baseTimeline = {
  video_name: "clip.mp4",
  video_file: "clip.mp4",
  fps: 24,
  sample_count: 3,
  window_start: 0,
  window_size: 160,
  metric_summary: {
    lpips_vit_patch: {
      pending: 0,
      running: 0,
      completed: 3,
      unavailable: 0,
      failed: 0,
      skipped: 0,
      missing: 0,
      mean: 0.2,
    },
  },
  samples: [
    {
      id: 101,
      frame_index: 0,
      timestamp: 0,
      metrics: { lpips_vit_patch: { status: "completed", value: 0.1, details: {} } },
      artifacts: {},
    },
    {
      id: 102,
      frame_index: 1,
      timestamp: 1 / 24,
      metrics: { lpips_vit_patch: { status: "completed", value: 0.2, details: {} } },
      artifacts: {},
    },
    {
      id: 103,
      frame_index: 2,
      timestamp: 2 / 24,
      metrics: { lpips_vit_patch: { status: "completed", value: 0.3, details: {} } },
      artifacts: {},
    },
  ],
  video_artifacts: {
    pred_video: {
      id: 501,
      preview_url: "/fixtures/pending-run-video.mp4",
      original_url: "/fixtures/pending-run-video.mp4",
    },
  },
  video_artifact_tracks: [],
  worst_samples: {},
  overview: [],
  video_metrics: {},
};

export function json(route, payload, status = 200, headers = {}) {
  return route.fulfill({
    status,
    contentType: "application/json",
    headers,
    body: JSON.stringify(payload),
  });
}

function pathname(route) {
  return new URL(route.request().url()).pathname;
}

function runList(run, progress) {
  const row = run ? { ...run, progress_current: progress } : null;
  return {
    runs: row ? [row] : [],
    page: 1,
    page_size: 30,
    page_count: 1,
    total: row ? 1 : 0,
    active_total: row && !["completed", "failed", "canceled"].includes(row.status) ? 1 : 0,
  };
}

export async function installMainApi(page, options = {}) {
  const run = options.run || null;
  const timeline = options.timeline || baseTimeline;
  const counters = {
    preflight: 0,
    creates: 0,
    runLists: 0,
  };

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();
    if (options.intercept) {
      const handled = await options.intercept({ route, request, url, path, method, counters });
      if (handled) return;
    }

    if (path === "/api/health") {
      await json(route, {
        ok: true,
        live: true,
        ready: true,
        release: { build_id: "browser-test" },
        storage: { status: "ok", free_bytes: 1024 ** 3 },
        leases: { running: run?.status === "running" ? 1 : 0, stale: 0 },
        maintenance: {},
        queues: {},
      });
      return;
    }
    if (path === "/api/media/sync/status") {
      await json(route, { state: "idle" });
      return;
    }
    if (path === "/api/media/sync" && method === "POST") {
      await json(route, { state: "completed" });
      return;
    }
    if (path === "/api/model-files") {
      await json(route, [{ name: "test_average.py" }]);
      return;
    }
    if (path === "/api/checkpoints") {
      await json(route, []);
      return;
    }
    if (path === "/api/video-groups") {
      await json(route, [{ name: "fixtures", video_count: 1 }]);
      return;
    }
    if (path === "/api/video-selections" && method === "POST") {
      await json(route, {
        video_selection_token: "browser-video-selection-token",
        schema_version: "video-selection-v1",
        video_groups: ["fixtures"],
        total: 1,
        group_counts: { fixtures: 1 },
        created_at: 1,
        expires_at: 9999999999,
      });
      return;
    }
    if (path === "/api/video-groups/fixtures/videos") {
      await json(route, {
        page: 1,
        total_pages: 1,
        filtered_count: 1,
        video_count: 1,
        videos: [{
          name: "clip.mp4",
          frame_count: 3,
          valid_triplets: 1,
          fps: 24,
          width: 64,
          height: 64,
          cache_status: "ready",
          selected: true,
        }],
        selection: {
          video_selection_token: "browser-video-selection-token",
          total: 1,
          group_count: 1,
        },
      });
      return;
    }
    if (path === "/api/metrics/health") {
      await json(route, { asset_root: "set/metrics", metrics: {} });
      return;
    }
    if (path === "/api/devices") {
      await json(route, {
        cuda: [],
        npu: [],
        errors: {},
        decode_backends: {
          ffmpeg: { available: true, version: "fixture" },
          opencv: { available: true, version: "fixture" },
        },
      });
      return;
    }
    if (path === "/api/preflight" && method === "POST") {
      counters.preflight += 1;
      const body = request.postDataJSON();
      await json(route, {
        ok: true,
        errors: [],
        warnings: [],
        video_group: { video_count: 1, frame_count: 3 },
        videos: [],
        workload: { risk_level: "low" },
        ...(body.preflight_level === "deep" ? { preflight_token: "browser-deep-token" } : {}),
      });
      return;
    }
    if (path === "/api/runs" && method === "POST") {
      counters.creates += 1;
      if (options.createDelayMs) {
        await new Promise((resolve) => setTimeout(resolve, options.createDelayMs));
      }
      await json(route, { run_id: 1 });
      return;
    }
    if (path === "/api/runs" && method === "GET") {
      counters.runLists += 1;
      await json(route, runList(run, Number(run?.progress_current || 0) + counters.runLists - 1));
      return;
    }
    if (path === "/api/runs/1") {
      await json(route, run || { ...baseRun, status: "completed" });
      return;
    }
    if (path === "/api/runs/1/metric-summary") {
      await json(route, {
        run_id: 1,
        metrics: timeline.metric_summary,
      });
      return;
    }
    if (path === "/api/runs/1/videos") {
      await json(route, {
        page: 1,
        page_size: 20,
        total_pages: 1,
        filtered_count: 1,
        videos: [{
          video_name: timeline.video_name,
          video_file: timeline.video_file,
          sample_count: timeline.sample_count,
        }],
      });
      return;
    }
    if (path === "/api/runs/1/videos/clip.mp4/timeline") {
      await json(route, timeline);
      return;
    }
    if (path.startsWith("/api/samples/")) {
      await json(route, { sample: {}, artifacts: {}, metrics: {} });
      return;
    }
    if (path === "/api/media/collections") {
      await json(route, []);
      return;
    }
    if (path === "/api/media/assets") {
      await json(route, { assets: [], page: 1, page_count: 1, total: 0 });
      return;
    }
    if (path === "/api/media/item-groups") {
      await json(route, []);
      return;
    }
    if (path === "/api/evaluation-campaigns") {
      await json(route, []);
      return;
    }
    await json(route, {});
  });

  return counters;
}

export function blindTask(overrides = {}) {
  return {
    token: "task-token",
    video_name: "clip.mp4",
    frame_count: 1,
    fps: 24,
    reference_media_kind: "video",
    left_media_kind: "video",
    right_media_kind: "video",
    reference_url: "/fixtures/pending-reference.mp4",
    left_url: "/fixtures/pending-left.mp4",
    right_url: "/fixtures/pending-right.mp4",
    ...overrides,
  };
}

export async function installBlindApi(page, task) {
  const counters = { votes: 0 };
  await page.addInitScript(() => {
    localStorage.setItem("vfieval-evaluator-id", "browser-evaluator");
    localStorage.setItem("vfieval-evaluator-name", "Browser Tester");
  });
  await page.route("**/api/blind/**", async (route) => {
    const request = route.request();
    const path = pathname(route);
    if (path.endsWith("/heartbeat")) {
      await json(route, { ok: true });
      return;
    }
    if (path.endsWith("/vote")) {
      counters.votes += 1;
      await json(route, { progress: { completed: 1, total: 1, complete: true } });
      return;
    }
    if (path.endsWith("/reviews")) {
      await json(route, { reviews: [], editable: true });
      return;
    }
    if (request.method() === "POST" && path.endsWith("/session")) {
      await json(route, {});
      return;
    }
    await json(route, {
      campaign: { status: "published", public_title: "浏览器盲评" },
      progress: { completed: 0, total: 1, complete: false },
      task,
    });
  });
  return counters;
}

export async function waitForMainReady(page) {
  await expect(page.locator("#deployment-health")).toContainText("服务正常");
  await expect(page.locator("#video-selection")).toContainText("已自动载入");
}
