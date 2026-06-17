/**
 * Regression: sending multiple prompts must never collapse the chat container
 * or yank the scroll position back to the bottom when the user has scrolled up.
 *
 * Runs in guest mode (no auth required) and mocks /api/chat so the test is
 * deterministic and offline-safe. The AI SDK renders the user message
 * optimistically as soon as `sendMessage` is called, which is enough to
 * exercise the layout invariants we care about.
 */
import { test, expect, type Page } from "@playwright/test";

const MIN_SCROLL_HEIGHT_PX = 200;

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
  // The user bubble shows up immediately even if the network response fails.
  await expect(page.getByText(text, { exact: false }).first()).toBeVisible({ timeout: 5_000 });
}

test.describe("Chat layout regression", () => {
  test.beforeEach(async ({ page }) => {
    // Mock the model endpoint so the test never depends on backend availability.
    await page.route("**/api/chat", async (route) => {
      await route.fulfill({
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "x-guest-remaining": "49",
        },
        // Minimal stream: no assistant tokens. The optimistic user bubble
        // is what we assert against; the assistant stub just keeps useChat happy.
        body: 'data: [DONE]\n\n',
      });
    });
  });

  test("3 messages do not collapse the chat container", async ({ page }) => {
    await enterGuest(page);

    const scroll = page.locator('[data-testid="chat-scroll-region"]');
    await expect(scroll).toBeVisible();

    const startHeight = await scroll.evaluate((el) => (el as HTMLElement).clientHeight);
    expect(startHeight).toBeGreaterThan(MIN_SCROLL_HEIGHT_PX);

    for (const prompt of [
      "regression test message one",
      "regression test message two",
      "regression test message three",
    ]) {
      await sendPrompt(page, prompt);
      // After each send, the scroll region must still have real height.
      const h = await scroll.evaluate((el) => (el as HTMLElement).clientHeight);
      expect(h).toBeGreaterThan(MIN_SCROLL_HEIGHT_PX);
    }

    // All three user bubbles should still be in the DOM.
    await expect(page.getByText("regression test message one")).toBeVisible();
    await expect(page.getByText("regression test message two")).toBeVisible();
    await expect(page.getByText("regression test message three")).toBeVisible();
  });

  test("scroll position is preserved when user has scrolled up", async ({ page }) => {
    await enterGuest(page);
    const scroll = page.locator('[data-testid="chat-scroll-region"]');

    // Seed a few messages so there's something to scroll through.
    for (let i = 0; i < 3; i++) {
      await sendPrompt(page, `seed message ${i} ${"lorem ipsum ".repeat(40)}`);
    }

    // Scroll the user up to the top of the chat region.
    await scroll.evaluate((el) => {
      (el as HTMLElement).scrollTop = 0;
    });
    const before = await scroll.evaluate((el) => (el as HTMLElement).scrollTop);

    // Send another prompt; we expect NO jump-to-bottom because the user is
    // explicitly scrolled away.
    await sendPrompt(page, "another prompt while scrolled up");
    await page.waitForTimeout(300);

    const after = await scroll.evaluate((el) => (el as HTMLElement).scrollTop);
    // Tolerate a few pixels of layout shift, but the scroll must NOT have
    // been yanked to the bottom of the now-taller content.
    expect(after).toBeLessThan(before + 80);
  });
});
