import { test, expect } from "@playwright/test";
import {
  baseRun,
  baseTimeline,
  installMainApi,
  json,
  waitForMainReady,
} from "./helpers.mjs";

test.describe("main application reliability", () => {
  test("non-JSON errors retain copyable request and support identifiers", async ({ page, context }) => {
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await installMainApi(page, {
      intercept: async ({ route, path }) => {
        if (path !== "/api/health") return false;
        await route.fulfill({
          status: 503,
          contentType: "text/plain; charset=utf-8",
          headers: {
            "X-Request-ID": "request-browser-123",
            "X-Support-ID": "support-browser-456",
          },
          body: "upstream maintenance",
        });
        return true;
      },
    });

    await page.goto("/");
    const diagnostic = page.locator("#request-diagnostic");
    await expect(diagnostic).toBeVisible();
    await expect(diagnostic).toContainText("upstream maintenance");
    await expect(diagnostic).toContainText("建议：");
    await expect(diagnostic).toContainText("request_id: request-browser-123");
    await expect(diagnostic).toContainText("support_id: support-browser-456");

    await diagnostic.getByRole("button", { name: "复制诊断信息" }).click();
    await expect.poll(() => page.evaluate(() => navigator.clipboard.readText()))
      .toContain("support_id: support-browser-456");
  });

  test("shared error normalization preserves structured recovery fields", async ({ page }) => {
    await installMainApi(page);
    await page.goto("/");
    const normalized = await page.evaluate(() => {
      const response = new Response("", {
        status: 409,
        headers: {
          "X-Request-ID": "header-request",
          "X-Support-ID": "header-support",
        },
      });
      const error = window.VFIEvalShared.createError({
        response,
        payload: {
          error: {
            code: "submission_conflict",
            message: "提交标识对应的内容不同",
            request_id: "payload-request",
            support_id: "payload-support",
            details: { field: "submission_id" },
          },
        },
      });
      return {
        code: error.code,
        message: error.message,
        request_id: error.request_id,
        support_id: error.support_id,
        details: error.details,
        recovery_suggestion: error.recovery_suggestion,
      };
    });
    expect(normalized).toEqual({
      code: "submission_conflict",
      message: "提交标识对应的内容不同",
      request_id: "payload-request",
      support_id: "payload-support",
      details: { field: "submission_id" },
      recovery_suggestion: "当前状态或提交内容已变化，请刷新最新状态后再试；不要连续重复提交。",
    });
  });

  test("offline failures remain visible instead of being swallowed", async ({ page }) => {
    let abortSync = false;
    await installMainApi(page, {
      intercept: async ({ route, path }) => {
        if (abortSync && path === "/api/media/sync/status") {
          await route.abort("internetdisconnected");
          return true;
        }
        return false;
      },
    });
    await page.goto("/");
    await waitForMainReady(page);

    abortSync = true;
    await page.getByRole("button", { name: "刷新文件列表" }).click();
    const diagnostic = page.locator("#request-diagnostic");
    await expect(diagnostic).toBeVisible();
    await expect(diagnostic).toContainText("VFIEval 请求失败");
    await expect(diagnostic).toContainText("request_id: 未返回");
  });

  test("a rapid double click sends one create request", async ({ page }) => {
    const counters = await installMainApi(page, { createDelayMs: 600 });
    await page.goto("/");
    await waitForMainReady(page);
    const start = page.locator("#start-run");
    await expect(start).toBeEnabled();

    await start.dblclick({ delay: 25 });
    await expect(page.locator("#infer-form")).toHaveAttribute("aria-busy", "true");
    await expect.poll(() => counters.creates).toBe(1);
    await page.waitForTimeout(800);
    expect(counters.creates).toBe(1);
  });

  test("polling preserves the player node, focus, and feedback draft", async ({ page }) => {
    const counters = await installMainApi(page, { run: baseRun, timeline: baseTimeline });
    await page.goto("/?view=runs&run=1&video=clip.mp4");

    const player = page.locator(".video-artifact video").first();
    const issue = page.locator('[data-feedback-form="1"] textarea[name="issue"]');
    await expect(player).toBeAttached();
    await issue.fill("轮询期间保留这段草稿");
    await issue.focus();
    await player.evaluate((video) => {
      video.dataset.browserIdentity = "same-player";
      Object.defineProperty(video, "currentTime", {
        configurable: true,
        writable: true,
        value: 12.5,
      });
    });
    const listCountBefore = counters.runLists;
    // Exercise the exact refresh used by the visibility-aware polling loop
    // without waiting for its initial idle backoff interval.
    await page.evaluate(() => window.refreshRunsOnly());
    expect(counters.runLists).toBeGreaterThan(listCountBefore);
    await expect(issue).toBeFocused();
    await expect(issue).toHaveValue("轮询期间保留这段草稿");
    await expect(player).toHaveAttribute("data-browser-identity", "same-player");
    expect(await player.evaluate((video) => video.currentTime)).toBe(12.5);
  });

  test("metric charts expose keyboard readout and an equivalent data table", async ({ page }) => {
    await installMainApi(page, { run: baseRun, timeline: baseTimeline });
    await page.goto("/?view=runs&run=1&video=clip.mp4");

    const chart = page.locator('.chart[data-chart-metric="lpips_vit_patch"] .chart-plot');
    await expect(chart).toBeVisible();
    await chart.focus();
    await page.keyboard.press("End");
    await expect(chart).toHaveAttribute("aria-valuenow", "2");
    await expect(chart).toHaveAttribute("aria-valuetext", /帧 2 .* 0\.3/);
    await expect(page.locator("[data-chart-readout]").first()).toContainText("帧 2");

    const table = page.locator(".metric-data-table").first();
    await table.locator("summary").click();
    await expect(table.getByRole("table")).toBeVisible();
    await expect(table.locator("tbody tr")).toHaveCount(3);
    await expect(table.locator("tbody")).toContainText("0.3");
  });

  test("Studio objective curves support point navigation, fixed readout, and data table", async ({ page }) => {
    const pageErrors = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    const campaign = {
      id: 1,
      schema_version: 2,
      campaign_key: "v2:1",
      name: "Browser objective campaign",
      public_title: "Objective curve fixture",
      public_token: "objective-fixture",
      status: "published",
      target_votes: 3,
      item_count: 1,
      task_count: 1,
      vote_count: 1,
      items: [{ id: 11, video_name: "clip.mp4" }],
    };
    const analysis = {
      objective: {
        metric_fingerprint: "browser-objective-v1",
        metrics: [
          {
            metric_name: "lpips_vit_patch",
            method_label: "Pred 1",
            direction: "lower_is_better",
            status_counts: { completed: 2 },
            count: 2,
            mean: 0.15,
            median: 0.15,
            p10: 0.1,
            p90: 0.2,
          },
          {
            metric_name: "lpips_vit_patch",
            method_label: "Pred 2",
            direction: "lower_is_better",
            status_counts: { completed: 2 },
            count: 2,
            mean: 0.25,
            median: 0.25,
            p10: 0.2,
            p90: 0.3,
          },
        ],
        items: [
          {
            item_id: 11,
            video_name: "clip.mp4",
            metric_name: "lpips_vit_patch",
            method_id: 1,
            frame_coverage: { completed: 2 },
          },
          {
            item_id: 11,
            video_name: "clip.mp4",
            metric_name: "lpips_vit_patch",
            method_id: 2,
            frame_coverage: { completed: 2 },
          },
        ],
      },
    };
    const curve = {
      metric_name: "lpips_vit_patch",
      frame_count: 2,
      completed_overlap: 2,
      series: [
        {
          method_label: "Pred 1",
          status_counts: { completed: 2 },
          reason_counts: {},
          points: [
            { ordinal: 0, frame_index: 0, status: "completed", value: 0.1 },
            { ordinal: 1, frame_index: 1, status: "completed", value: 0.2 },
          ],
        },
        {
          method_label: "Pred 2",
          status_counts: { completed: 2 },
          reason_counts: {},
          points: [
            { ordinal: 0, frame_index: 0, status: "completed", value: 0.2 },
            { ordinal: 1, frame_index: 1, status: "completed", value: 0.3 },
          ],
        },
      ],
    };
    await installMainApi(page, {
      intercept: async ({ route, path, method }) => {
        if (path === "/api/evaluation-campaigns" && method === "GET") {
          await json(route, {
            campaigns: [campaign],
            page: 1,
            page_size: 30,
            page_count: 1,
            total: 1,
          });
          return true;
        }
        if (path === "/api/evaluation-campaigns/v2/1/objective-curve") {
          await json(route, curve);
          return true;
        }
        if (path === "/api/evaluation-campaigns/v2/1") {
          await json(route, { campaign, analysis, coverage: { items: 1, tasks: 1, votes: 1 } });
          return true;
        }
        return false;
      },
    });

    await page.goto("/?view=evaluations");
    await waitForMainReady(page);
    expect(pageErrors).toEqual([]);
    await page.locator('[data-studio-campaign="v2:1"]').click();

    const points = page.locator("[data-objective-curve-point]");
    const readout = page.locator("[data-objective-curve-readout]");
    await expect(points).toHaveCount(4);
    await points.first().focus();
    await expect(readout).toContainText("Pred 1 · frame 0 · lpips_vit_patch 0.100000");
    await page.keyboard.press("ArrowRight");
    await expect(points.nth(1)).toBeFocused();
    await expect(readout).toContainText("Pred 2 · frame 0 · lpips_vit_patch 0.200000");

    const table = page.locator(".objective-curve-data-table");
    await table.locator("summary").click();
    await expect(table.getByRole("table")).toBeVisible();
    await expect(table.locator("tbody tr")).toHaveCount(4);
    await expect(table.locator("tbody")).toContainText("0.300000");
  });
});
