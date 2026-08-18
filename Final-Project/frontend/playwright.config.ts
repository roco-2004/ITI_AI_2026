import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:5173",
    channel: "msedge",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    ...devices["Desktop Edge"],
    viewport: { width: 1440, height: 1000 },
  },
});
