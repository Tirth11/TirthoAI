// Markdown rendering client: parses in a Web Worker, sanitizes on main thread.
// Per-bubble request coalescing — only the latest text for a given id resolves.
// Supports cooperative cancellation so the worker doesn't keep parsing stale
// bubbles after the user navigates, refreshes, or sends a new prompt.
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
    worker.onmessage = (e: MessageEvent<{ id: string; reqId?: number; html: string }>) => {
      const { id, reqId, html } = e.data ?? { id: "", html: "" };
      const p = pendingByBubble.get(id);
      if (!p) return;
      // Drop stale responses (a newer request for this bubble was issued).
      if (typeof reqId === "number" && reqId !== p.reqId) return;
      pendingByBubble.delete(id);
      const safe = DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
      p.resolve(safe);
    };
    worker.onerror = () => {
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
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function renderMarkdown(bubbleId: string, text: string): Promise<string> {
  const w = getWorker();
  if (!w) {
    return Promise.resolve(`<pre>${escapeHtml(text)}</pre>`);
  }
  const prev = pendingByBubble.get(bubbleId);
  if (prev) prev.resolve(""); // supersede
  return new Promise<string>((resolve) => {
    const reqId = nextReqId++;
    pendingByBubble.set(bubbleId, { reqId, resolve });
    w.postMessage({ id: bubbleId, reqId, text });
  });
}

/** Cancel an in-flight request for a single bubble (e.g. message unmounted). */
export function cancelMarkdown(bubbleId: string): void {
  const p = pendingByBubble.get(bubbleId);
  if (!p) return;
  pendingByBubble.delete(bubbleId);
  p.resolve("");
}

/**
 * Hard-cancel: drop every pending request and terminate the worker so any
 * in-flight `marked.parse` on a huge string stops immediately. The worker is
 * lazily recreated on the next `renderMarkdown` call.
 */
export function cancelAllMarkdown(): void {
  for (const [, p] of pendingByBubble) p.resolve("");
  pendingByBubble.clear();
  if (worker) {
    try {
      worker.terminate();
    } catch {
      /* ignore */
    }
    worker = null;
  }
}

export const __test = {
  reset() {
    cancelAllMarkdown();
  },
  pendingSize: () => pendingByBubble.size,
};
