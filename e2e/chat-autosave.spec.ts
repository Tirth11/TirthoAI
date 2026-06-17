/**
 * Regression: guest chat history must autosave to localStorage and be
 * restored — with stable scroll position — after a full page refresh.
 *
 * Runs in guest mode (no auth required) and mocks /api/chat so the test is
 * deterministic and offline-safe.
 */
import { test, expect, type Page } from "@playwright/test";

async function enterGuest(page: Page) {
  await page.goto("/");
  const guestBtn = page.getByRole("button", { name: /try free without an account/i });
  if (await guestBtn.isVisible().catch(() => false)) {
    await guestBtn.click();
  }
  await page.locator('[data-testid="chat-scroll-region"]').waitFor({ timeout: 15_000 });
}

async function sendPrompt(page: Page, text: string) {
  const textarea = page.locator("textarea").first();
  await textarea.fill(text);
  await textarea.press("Enter");
  await expect(page.getByText(text, { exact: false }).first()).toBeVisible({ timeout: 5_000 });
}

test.describe("Chat autosave + restore", () => {
  test.beforeEach(async ({ page }) => {
    await page.route("**/api/chat", async (route) => {
      await route.fulfill({
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "x-guest-remaining": "49",
        },
        body: "data: [DONE]\n\n",
      });
    });
  });

  test("messages persist across a full page refresh", async ({ page }) => {
    await enterGuest(page);

    const prompts = [
      "autosave check one",
      "autosave check two",
      "autosave check three",
    ];
    for (const p of prompts) await sendPrompt(page, p);

    // Confirm storage was written before reloading.
    const stored = await page.evaluate(
      () => window.localStorage.getItem("tirthoai.guest-messages.v1"),
    );
    expect(stored, "guest messages should be autosaved").toBeTruthy();
    for (const p of prompts) expect(stored).toContain(p);

    await page.reload();
    await page.locator('[data-testid="chat-scroll-region"]').waitFor({ timeout: 15_000 });

    // All prior messages should still be on screen after the refresh.
    for (const p of prompts) {
      await expect(page.getByText(p)).toBeVisible({ timeout: 5_000 });
    }

    // The chat container itself must not collapse after restore.
    const scroll = page.locator('[data-testid="chat-scroll-region"]');
    const h = await scroll.evaluate((el) => (el as HTMLElement).clientHeight);
    expect(h).toBeGreaterThan(200);
  });

  test("scroll position is stable (no jump) right after restore", async ({ page }) => {
    await enterGuest(page);
    for (let i = 0; i < 4; i++) {
      await sendPrompt(page, `restore-scroll seed ${i} ${"lorem ipsum ".repeat(40)}`);
    }

    await page.reload();
    const scroll = page.locator('[data-testid="chat-scroll-region"]');
    await scroll.waitFor({ timeout: 15_000 });

    // Capture scroll position twice with a small gap. After restore, the
    // container should settle into a stable position — no late auto-scroll
    // surprises that would yank the user away.
    const first = await scroll.evaluate((el) => (el as HTMLElement).scrollTop);
    await page.waitForTimeout(400);
    const second = await scroll.evaluate((el) => (el as HTMLElement).scrollTop);

    expect(Math.abs(second - first)).toBeLessThan(40);
  });
});
