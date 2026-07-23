import os from "node:os";
import path from "node:path";
import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.VFIEVAL_BROWSER_TEST_PORT || 4173);

export default defineConfig({
  testDir: "./tests/browser",
  testMatch: "**/*.spec.mjs",
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  expect: { timeout: 5_000 },
  outputDir: path.join(os.tmpdir(), "vfieval-playwright-artifacts"),
  reporter: process.env.CI ? [["github"], ["line"]] : "line",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    ...devices["Desktop Chrome"],
    headless: true,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "node tests/browser/fixture-server.mjs",
    url: `http://127.0.0.1:${port}/__fixture_health`,
    reuseExistingServer: !process.env.CI,
    timeout: 15_000,
    env: {
      VFIEVAL_BROWSER_TEST_PORT: String(port),
    },
  },
});
