/**
 * End-to-end regression: per-conversation model persistence + indicator tooltip.
 *
 * Flow:
 *   1. Sign in with E2E_EMAIL / E2E_PASSWORD (skipped if not provided).
 *   2. On conversation A: change model from its current value → model X.
 *   3. Create conversation B (New Chat) and change its model → model Y (≠ X).
 *   4. Reload the page.
 *   5. For each conversation, click its sidebar row and verify:
 *        - The model picker shows the model we chose before refresh.
 *        - The indicator's data-previous-model-id matches what was selected
 *          immediately before the change.
 *        - Hovering the indicator opens a tooltip that displays the previous
 *          model label, the current model label, and a parseable timestamp.
 *
 * Requires a real account because guest mode has a single ephemeral thread.
 */
import { test, expect, type Page } from "@playwright/test";

const EMAIL = process.env.E2E_EMAIL;
const PASSWORD = process.env.E2E_PASSWORD;

test.skip(!EMAIL || !PASSWORD, "Set E2E_EMAIL and E2E_PASSWORD to run.");

async function signIn(page: Page) {
  await page.goto("/");
  // AuthScreen defaults to signup; switch to signin if needed.
  const signInTab = page.getByRole("button", { name: /^sign in$/i });
  if (await signInTab.isVisible().catch(() => false)) {
    await signInTab.click().catch(() => {});
  }
  await page.locator('input[autocomplete="email"]').fill(EMAIL!);
  await page.locator('input[autocomplete="current-password"], input[autocomplete="new-password"]').first().fill(PASSWORD!);
  await page.getByRole("button", { name: /sign in|continue/i }).click();
  await page.waitForSelector('[data-testid="conversation-row"]', { timeout: 30_000 });
}

async function currentModelId(page: Page) {
  return await page.locator('[data-testid="model-picker-trigger"]').getAttribute("data-model-id");
}

async function pickDifferentModel(page: Page, exclude: string[]) {
  await page.locator('[data-testid="model-picker-trigger"]').click();
  const options = page.locator('[data-testid="model-option"]');
  await options.first().waitFor();
  const count = await options.count();
  for (let i = 0; i < count; i++) {
    const id = await options.nth(i).getAttribute("data-model-id");
    if (id && !exclude.includes(id)) {
      await options.nth(i).click();
      return id;
    }
  }
  throw new Error("Could not find a model different from " + exclude.join(","));
}

async function clickConversation(page: Page, id: string) {
  await page.locator(`[data-testid="conversation-row"][data-conversation-id="${id}"]`).click();
  // Wait until the indicator reflects this thread (model id may differ).
  await page.locator('[data-testid="model-changed-indicator"]').waitFor();
}

test("model selection persists per conversation and tooltip shows previous/new/time", async ({ page }) => {
  await signIn(page);

  // --- Conversation A ---
  const rowA = page.locator('[data-testid="conversation-row"]').first();
  const convAId = await rowA.getAttribute("data-conversation-id");
  expect(convAId).toBeTruthy();
  await rowA.click();

  const beforeA = (await currentModelId(page))!;
  const chosenA = await pickDifferentModel(page, [beforeA]);
  await expect(page.locator('[data-testid="model-picker-trigger"]')).toHaveAttribute(
    "data-model-id",
    chosenA,
  );

  // --- Conversation B ---
  await page.locator('[data-testid="new-chat-button"]').click();
  // The new conversation becomes the first row.
  await page.waitForFunction(
    (prev) => {
      const first = document.querySelector('[data-testid="conversation-row"]');
      return first && first.getAttribute("data-conversation-id") !== prev;
    },
    convAId,
  );
  const convBId = await page
    .locator('[data-testid="conversation-row"]')
    .first()
    .getAttribute("data-conversation-id");
  expect(convBId).toBeTruthy();
  expect(convBId).not.toBe(convAId);

  const beforeB = (await currentModelId(page))!;
  const chosenB = await pickDifferentModel(page, [beforeB, chosenA]);

  // --- Refresh and verify both threads survive ---
  await page.reload();
  await page.waitForSelector('[data-testid="conversation-row"]');

  // Conversation A
  await clickConversation(page, convAId!);
  await expect(page.locator('[data-testid="model-picker-trigger"]')).toHaveAttribute(
    "data-model-id",
    chosenA,
  );
  const indicatorA = page.locator('[data-testid="model-changed-indicator"]');
  await expect(indicatorA).toHaveAttribute("data-model-id", chosenA);
  await expect(indicatorA).toHaveAttribute("data-previous-model-id", beforeA);

  await indicatorA.hover();
  const tooltipA = page.locator('[data-testid="model-changed-tooltip"]');
  await expect(tooltipA).toBeVisible();
  await expect(tooltipA.locator('[data-testid="model-changed-current"]')).toContainText("Current:");
  await expect(tooltipA.locator('[data-testid="model-changed-previous"]')).toContainText("Previous:");
  const atTextA = (await tooltipA.locator('[data-testid="model-changed-at"]').innerText()).replace(/^At:\s*/i, "");
  expect(Number.isNaN(Date.parse(atTextA))).toBe(false);

  // Conversation B
  await clickConversation(page, convBId!);
  await expect(page.locator('[data-testid="model-picker-trigger"]')).toHaveAttribute(
    "data-model-id",
    chosenB,
  );
  const indicatorB = page.locator('[data-testid="model-changed-indicator"]');
  await expect(indicatorB).toHaveAttribute("data-model-id", chosenB);
  await expect(indicatorB).toHaveAttribute("data-previous-model-id", beforeB);

  await indicatorB.hover();
  const tooltipB = page.locator('[data-testid="model-changed-tooltip"]');
  await expect(tooltipB).toBeVisible();
  await expect(tooltipB.locator('[data-testid="model-changed-current"]')).toContainText("Current:");
  await expect(tooltipB.locator('[data-testid="model-changed-previous"]')).toContainText("Previous:");
  const atTextB = (await tooltipB.locator('[data-testid="model-changed-at"]').innerText()).replace(/^At:\s*/i, "");
  expect(Number.isNaN(Date.parse(atTextB))).toBe(false);

  // Sanity: the two threads picked different models.
  expect(chosenA).not.toBe(chosenB);
});
