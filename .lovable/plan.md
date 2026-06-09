# Keep the UI responsive on huge model responses

Two changes, both contained to the chat surface. No backend or schema changes.

## 1. Markdown in a Web Worker

Move the heavy work (markdown → HTML + sanitization) off the main thread so a 50KB assistant reply can't block input, scrolling, or the stop button.

- New file `src/workers/markdown.worker.ts`
  - Uses `marked` (fast, streaming-friendly) for parse and `dompurify` for sanitization.
  - Message protocol: `{ id, text }` in → `{ id, html }` out. `id` lets us drop stale results when newer text arrives for the same bubble.
- New file `src/lib/markdown-client.ts`
  - Lazily spins up a single shared worker via `new Worker(new URL("../workers/markdown.worker.ts", import.meta.url), { type: "module" })`.
  - Exposes `renderMarkdown(bubbleId, text): Promise<string>` that resolves with sanitized HTML, ignoring superseded requests per `bubbleId`.
  - SSR-safe: no-op fallback when `window`/`Worker` is undefined; in that case returns the raw text and lets the existing `ReactMarkdown` path render on the client after hydration.
- Replace `<ReactMarkdown>` in `AssistantMarkdown` (in `src/components/ChatWindow.tsx`) with a small component that:
  - Calls `renderMarkdown(message.id, deferredText)` inside an effect.
  - Stores the last successful HTML in state and renders it with `dangerouslySetInnerHTML` (safe — DOMPurify ran in the worker).
  - Keeps showing the previous HTML while the next chunk is parsing, so the bubble never flashes empty.
- Remove the `react-markdown` import from `ChatWindow.tsx` once the worker path is wired. Keep the package installed in case other components use it; remove only if grep confirms no other call sites.

## 2. Streaming backpressure + chunked rendering

Right now every streamed token triggers a React render of the active bubble and a worker re-parse. On long replies that's thousands of renders. We coalesce.

- In `ChatWindow.tsx`, derive a `renderMessages` view of `messages` that the list maps over, instead of mapping `messages` directly.
- While `status === "streaming"`, throttle updates to the last (assistant) message using `requestAnimationFrame` + a minimum interval (e.g. 80ms):
  - Keep a ref with the latest streamed text from `useChat`.
  - A rAF loop flushes that text into `renderMessages` state at most once per frame and not more often than the interval.
  - When `status` flips back to `ready`/`error`/aborted, flush immediately so the final text is never truncated.
- The worker call inside `AssistantMarkdown` then naturally fires at the throttled cadence, not per token.
- Sidebar/scroll-to-bottom logic uses the same throttled signal so autoscroll doesn't fight the rAF loop.
- Hard ceiling: if a single message exceeds ~200KB of text, switch that bubble to a plain `<pre className="whitespace-pre-wrap">` fallback (no markdown parsing) and show a small "Rendering as plain text — response too large for rich formatting" note. Prevents pathological inputs from ever hanging the worker queue.

## Verification

- Unit: extend `src/lib/__tests__/model-indicator.test.tsx` style with a new `markdown-client.test.ts` that mocks `Worker` and asserts stale-request cancellation per `bubbleId`.
- Manual: paste a long markdown doc (e.g. a 30KB README) into a prompt, confirm input stays responsive, stop button cancels within a frame, and the final rendered HTML matches what `react-markdown` produced before.
- Performance check via `browser--performance_profile` on a long-streamed reply: main-thread long tasks during streaming should drop substantially vs. the current build.

## Technical details

- Packages to add: `marked`, `dompurify`, `@types/dompurify` (dev). All bundle-safe; the worker is built by Vite as a separate chunk.
- Worker uses `marked.parse(text, { async: false, gfm: true, breaks: true })` then `DOMPurify.sanitize(html, { USE_PROFILES: { html: true } })`.
- The worker runs in the browser only — Cloudflare Worker SSR is unaffected. The worker file must not import anything Node-only.
- `dangerouslySetInnerHTML` is acceptable here because every string passed to it has been through DOMPurify on the same render path; no other component should adopt this pattern without the same guarantee.
- `useDeferredValue` stays in place as a second line of defense for the rare frame where the worker result lands during a high-priority update.
