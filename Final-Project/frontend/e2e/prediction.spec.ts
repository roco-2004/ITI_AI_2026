import { expect, test } from "@playwright/test";
import path from "node:path";

const screenshotDir = path.resolve(import.meta.dirname, "../../docs/screenshots");

test("submits a property and displays a finite INR estimate", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });

  await page.goto("/");
  await expect(page).toHaveTitle(/India House Price Predictor/);
  await expect(page.getByRole("heading", { name: /Understand a home/ })).toBeVisible();
  await expect(page.getByLabel("Location")).toBeEnabled();
  await page.screenshot({ path: path.join(screenshotDir, "home.png"), fullPage: true });

  await page.getByRole("button", { name: /Estimate property price/ }).click();
  await expect(page.getByText("Choose a location.")).toBeVisible();

  await page.getByLabel("Location").selectOption("agra");
  await page.getByLabel(/Carpet area/).fill("1200");
  await page.screenshot({ path: path.join(screenshotDir, "prediction-form.png"), fullPage: true });

  await page.getByRole("button", { name: /Estimate property price/ }).click();
  await expect(page).toHaveURL(/\/result$/);
  await expect(page.getByRole("heading", { name: /Lakh/ })).toBeVisible();
  await expect(page.getByText(/not a professional appraisal/)).toBeVisible();
  await expect(page.getByText("agra", { exact: true })).toBeVisible();
  await page.screenshot({ path: path.join(screenshotDir, "prediction-result.png"), fullPage: true });
  expect(consoleErrors).toEqual([]);

  await page.reload();
  await expect(page.getByRole("heading", { name: /Lakh/ })).toBeVisible();
  await expect(page.getByRole("link", { name: "Estimate another property" })).toBeVisible();
});

test("renders the application not-found route", async ({ page }) => {
  await page.goto("/not-a-real-route");
  await expect(page.getByRole("heading", { name: "This address has no listing." })).toBeVisible();
  await expect(page.getByRole("link", { name: "Back to estimator" })).toBeVisible();
});

test("renders FastAPI Swagger documentation", async ({ page }) => {
  await page.goto("http://localhost:8000/docs");
  await expect(page.locator(".swagger-ui")).toBeVisible();
  await expect(page.getByText("/api/predict", { exact: true })).toBeVisible();
  await page.screenshot({ path: path.join(screenshotDir, "swagger.png"), fullPage: true });
});

test("keeps the home page within a mobile viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Understand a home/ })).toBeVisible();
  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
});
