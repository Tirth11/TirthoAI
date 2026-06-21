import { chromium } from "@playwright/test";
import fs from "node:fs";

const OUT = "C:/Users/tirth/calm_shots";
fs.mkdirSync(OUT, { recursive: true });

const msgs = [
  { id: "m1", role: "user", parts: [{ type: "text", text: "Can you explain how React hooks work, with a quick example?" }] },
  {
    id: "m2",
    role: "assistant",
    parts: [
      {
        type: "text",
        text: "## React Hooks\n\nHooks let function components use state and lifecycle features without classes.\n\n- **useState** — local state\n- **useEffect** — run side effects\n- **useMemo** — cache expensive values\n\n```jsx\nfunction Counter() {\n  const [n, setN] = useState(0);\n  return <button onClick={() => setN(n + 1)}>{n}</button>;\n}\n```\n\nThat's the core idea — composable, reusable stateful logic.",
      },
    ],
  },
  { id: "m3", role: "user", parts: [{ type: "text", text: "Nice. Now write a haiku about Mumbai rain." }] },
  {
    id: "m4",
    role: "assistant",
    parts: [{ type: "text", text: "Monsoon over docks —\ngrey sky stitched to silver sea,\nthe city exhales." }],
  },
  { id: "m5", role: "user", parts: [{ type: "text", text: "Give me 5 ideas for a weekend side project in TypeScript." }] },
  {
    id: "m6",
    role: "assistant",
    parts: [
      {
        type: "text",
        text: "Here are five:\n\n1. **Markdown note app** with local-first sync\n2. **CLI weather dashboard** using a public API\n3. **Habit tracker** PWA with charts\n4. **URL shortener** on edge functions\n5. **Pomodoro timer** browser extension\n\nWant me to scaffold any of these?",
      },
    ],
  },
];

const browser = await chromium.launch();
async function shoot(name, w, h) {
  const ctx = await browser.newContext({ viewport: { width: w, height: h } });
  await ctx.addInitScript(
    (args) => {
      localStorage.setItem(args.mode, "1");
      localStorage.setItem(args.key, args.val);
    },
    { mode: "tirthoai_guest_mode", key: "tirthoai.guest-messages.v1", val: JSON.stringify(msgs) },
  );
  const page = await ctx.newPage();
  await page.goto("http://localhost:8080/", { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForTimeout(3500);
  await page.screenshot({ path: `${OUT}/${name}.png`, animations: "disabled", caret: "hide", timeout: 15000 });
  await ctx.close();
}
await shoot("chat-desktop", 1280, 800);
await shoot("chat-mobile", 390, 844);
await browser.close();
console.log("DONE");
