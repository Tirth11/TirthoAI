// Markdown rendering client: parses in a Web Worker, sanitizes on main thread.
// Per-bubble request coalescing — only the latest text for a given id resolves.
import DOMPurify from "dompurify";

type Pending = {
  reqId: number;
  resolve: (html: string) => void;
};

let worker: Worker | null = null;
let nextReqId = 1;
const pendingByBubble = new Map<string, Pending>();

function getWorker(): Worker | null {
  if (typeof window === "undefined" || typeof Worker === "undefined") return null;
  if (worker) return worker;
  try {
    worker = new Worker(new URL("../workers/markdown.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (e: MessageEvent<{ id: string; html: string }>) => {
      const { id, html } = e.data ?? { id: "", html: "" };
      const p = pendingByBubble.get(id);
      if (!p) return;
      pendingByBubble.delete(id);
      // Sanitize on main thread (needs DOM).
      const safe = DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
      p.resolve(safe);
    };
    worker.onerror = () => {
      // On worker crash, fail-open: resolve all pending with empty so caller falls back.
      for (const [id, p] of pendingByBubble) {
        p.resolve("");
        pendingByBubble.delete(id);
      }
    };
    return worker;
  } catch {
    return null;
  }
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

export function renderMarkdown(bubbleId: string, text: string): Promise<string> {
  const w = getWorker();
  if (!w) {
    // SSR or worker unavailable — return escaped, preformatted text wrapped safely.
    return Promise.resolve(`<pre>${escapeHtml(text)}</pre>`);
  }
  // Supersede any in-flight request for the same bubble.
  const prev = pendingByBubble.get(bubbleId);
  if (prev) {
    prev.resolve(""); // will be ignored by caller (stale)
  }
  return new Promise<string>((resolve) => {
    const reqId = nextReqId++;
    pendingByBubble.set(bubbleId, { reqId, resolve });
    w.postMessage({ id: bubbleId, text });
  });
}

// Test-only helpers
export const __test = {
  reset() {
    pendingByBubble.clear();
    worker?.terminate?.();
    worker = null;
  },
  pendingSize: () => pendingByBubble.size,
};
