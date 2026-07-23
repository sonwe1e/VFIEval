import { test, expect } from "@playwright/test";
import { blindTask, installBlindApi } from "./helpers.mjs";

test.describe("blind evaluation reliability", () => {
  test("voting remains locked while GT, A, and B media are pending", async ({ page }) => {
    const counters = await installBlindApi(page, blindTask());
    await page.goto("/evaluate/browser-token", { waitUntil: "domcontentloaded" });
    await expect(page.locator("#task-panel")).toBeVisible();
    await expect(page.locator("#sync-status")).toContainText(/准备|等待|就绪/);
    await expect(page.locator("#save-vote")).toBeDisabled();
    await expect(page.locator('input[name="choice"]')).toHaveCount(3);
    for (const choice of await page.locator('input[name="choice"]').all()) {
      await expect(choice).toBeDisabled();
    }
    expect(counters.votes).toBe(0);
  });

  test("the complete blind task does not overflow a narrow viewport", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    const pixelTask = blindTask({
      reference_media_kind: "frame_sequence",
      left_media_kind: "frame_sequence",
      right_media_kind: "frame_sequence",
      reference_url: "/fixtures/pixel.png",
      left_url: "/fixtures/pixel.png",
      right_url: "/fixtures/pixel.png",
    });
    await installBlindApi(page, pixelTask);
    await page.goto("/evaluate/browser-token");
    await expect(page.locator("#task-panel")).toBeVisible();
    await expect(page.locator("#media-grid img")).toHaveCount(3);
    await expect(page.locator("#save-vote")).toBeEnabled();

    const layout = await page.evaluate(() => ({
      viewport: window.innerWidth,
      document: document.documentElement.scrollWidth,
      body: document.body.scrollWidth,
      shellRight: document.querySelector(".blind-shell")?.getBoundingClientRect().right || 0,
    }));
    expect(layout.document).toBeLessThanOrEqual(layout.viewport);
    expect(layout.body).toBeLessThanOrEqual(layout.viewport);
    expect(layout.shellRight).toBeLessThanOrEqual(layout.viewport);
  });

  test("staggered three-stream loading unlocks voting only after every first frame", async ({ page }) => {
    const pixel = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
      "base64",
    );
    const delays = { reference: 80, left: 260, right: 900 };
    let fulfilled = 0;
    await page.route("**/fixtures/slow-*.png*", async (route) => {
      const match = new URL(route.request().url()).pathname.match(/slow-(reference|left|right)\.png$/);
      const side = match?.[1] || "right";
      await new Promise((resolve) => setTimeout(resolve, delays[side]));
      fulfilled += 1;
      await route.fulfill({ status: 200, contentType: "image/png", body: pixel });
    });
    const slowTask = blindTask({
      reference_media_kind: "frame_sequence",
      left_media_kind: "frame_sequence",
      right_media_kind: "frame_sequence",
      reference_url: "/fixtures/slow-reference.png",
      left_url: "/fixtures/slow-left.png",
      right_url: "/fixtures/slow-right.png",
    });
    await installBlindApi(page, slowTask);
    await page.goto("/evaluate/browser-token", { waitUntil: "domcontentloaded" });
    await expect(page.locator("#task-panel")).toBeVisible();
    await expect(page.locator("#save-vote")).toBeDisabled();

    await expect.poll(() => fulfilled).toBe(2);
    await expect(page.locator("#save-vote")).toBeDisabled();
    for (const choice of await page.locator('input[name="choice"]').all()) {
      await expect(choice).toBeDisabled();
    }

    await expect.poll(() => fulfilled).toBe(3);
    await expect(page.locator("#save-vote")).toBeEnabled();
    await expect(page.locator('input[name="choice"]').first()).toBeEnabled();
    await expect(page.locator("#sync-status")).toContainText(/三路帧序列已定位|就绪/);
  });
});
