/**
 * E2E regression: switching between Groq-hosted models repeatedly must keep
 * the dropdown working and never break chat-response layering. Guest mode is
 * fine — no auth required.
 *
 * What we verify:
 *   1. The model picker dropdown opens, lists Groq models (badge "GROQ"),
 *      and closes after selecting.
 *   2. Switching across several Groq models in a row updates the trigger's
 *      data-model-id each time.
 *   3. After each switch, the chat composer textarea is reachable and the
 *      dropdown is not visually covering it (i.e. dropdown actually closed).
 *   4. Sending a prompt while a Groq model is selected renders an assistant
 *      message, and the dropdown can still be reopened on top of the chat.
 */
import { test, expect, type Page } from "@playwright/test";

async function mockChat(page: Page) {
  await page.route("**/api/chat", async (route) => {
    const body =
      'data: {"type":"text-delta","delta":"hi from groq"}\n\n' +
      "data: [DONE]\n\n";
    await route.fulfill({
      status: 200,
      headers: {
        "content-type": "text/event-stream",
        "x-guest-remaining": "49",
      },
      body,
    });
  });
}

async function openPicker(page: Page) {
  await page.locator('[data-testid="model-picker-trigger"]').click();
  await page.locator('[data-testid="model-option"]').first().waitFor();
}

async function groqOptionIds(page: Page): Promise<string[]> {
  const opts = page.locator('[data-testid="model-option"]');
  const count = await opts.count();
  const ids: string[] = [];
  for (let i = 0; i < count; i++) {
    const opt = opts.nth(i);
    const hasGroqBadge = await opt.locator("text=GROQ").count();
    const id = await opt.getAttribute("data-model-id");
    if (hasGroqBadge > 0 && id) ids.push(id);
  }
  return ids;
}

test("switching between Groq models keeps dropdown + chat layering intact", async ({ page }) => {
  await mockChat(page);
  await page.goto("/");

  // If auth screen shows, continue as guest.
  const guest = page.getByRole("button", { name: /continue as guest|guest/i });
  if (await guest.isVisible().catch(() => false)) {
    await guest.click();
  }

  await page.locator('[data-testid="model-picker-trigger"]').waitFor({ timeout: 15_000 });

  // Disable Auto mode so we can manually pick.
  const autoBtn = page.getByRole("button", { name: /^auto$/i });
  if (await autoBtn.getAttribute("class").then((c) => c?.includes("bg-primary"))) {
    await autoBtn.click();
  }

  await openPicker(page);
  const groqIds = await groqOptionIds(page);
  expect(groqIds.length).toBeGreaterThanOrEqual(3);

  // Switch through at least 4 Groq models in a row.
  const sequence = groqIds.slice(0, Math.min(5, groqIds.length));
  for (const id of sequence) {
    // Reopen if closed
    const open = await page.locator('[data-testid="model-option"]').first().isVisible().catch(() => false);
    if (!open) await openPicker(page);

    await page.locator(`[data-testid="model-option"][data-model-id="${id}"]`).first().click();

    // Dropdown must close
    await expect(page.locator('[data-testid="model-option"]').first()).toBeHidden();

    // Trigger reflects the new model
    await expect(page.locator('[data-testid="model-picker-trigger"]')).toHaveAttribute(
      "data-model-id",
      id,
    );

    // Composer textarea is reachable (not covered)
    const textarea = page.locator("textarea").first();
    await expect(textarea).toBeVisible();
    await textarea.click(); // would throw if intercepted by an overlay
  }

  // Send a prompt under the last selected Groq model.
  const textarea = page.locator("textarea").first();
  await textarea.fill("Hello Groq");
  await textarea.press("Enter");

  // Wait for the streamed assistant text to render.
  await expect(page.getByText("hi from groq")).toBeVisible({ timeout: 10_000 });

  // Dropdown can still open on top of the chat after a response.
  await openPicker(page);
  await expect(page.locator('[data-testid="model-option"]').first()).toBeVisible();

  // And the chat message remains visible underneath (layering ok).
  await expect(page.getByText("hi from groq")).toBeVisible();
});
