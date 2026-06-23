import { chromium } from "@playwright/test";
import fs from "node:fs";

const OUT = "C:/Users/tirth/calm_shots";
fs.mkdirSync(OUT, { recursive: true });
const log = [];
const browser = await chromium.launch();

async function shoot(name, width, height, actions) {
  const ctx = await browser.newContext({ viewport: { width, height } });
  const page = await ctx.newPage();
  page.on("console", (m) => {
    if (m.type() === "error") log.push(`[${name}] CONSOLE ${m.text()}`);
  });
  page.on("pageerror", (e) => log.push(`[${name}] PAGEERROR ${e.message}`));
  try {
    await page.goto("http://localhost:8080/", { waitUntil: "domcontentloaded", timeout: 60000 });
    await page.waitForTimeout(3000);
    if (actions) await actions(page);
    const bodyText = (await page.locator("body").innerText().catch(() => "")).slice(0, 400);
    log.push(`[${name}] VISIBLE: ${bodyText.replace(/\n+/g, " | ")}`);
    await page.screenshot({
      path: `${OUT}/${name}.png`,
      fullPage: false,
      animations: "disabled",
      caret: "hide",
      timeout: 15000,
    });
    log.push(`[${name}] screenshot OK`);
  } catch (e) {
    log.push(`[${name}] SHOOT_ERROR ${e.message.split("\n")[0]}`);
  } finally {
    await ctx.close();
  }
}

const enterGuest = async (page) => {
  const btn = page.getByText(/try free without an account/i);
  if (await btn.count()) {
    await btn.first().click();
    await page.waitForTimeout(3000);
  } else {
    log.push("[guest] guest button NOT found");
  }
};

await shoot("auth-desktop", 1280, 800);
await shoot("auth-mobile", 390, 844);
await shoot("guest-desktop", 1280, 800, enterGuest);
await shoot("guest-mobile", 390, 844, enterGuest);

await browser.close();
fs.writeFileSync(`${OUT}/log.txt`, log.join("\n") || "no log");
console.log(log.join("\n"));
console.log("DONE");
