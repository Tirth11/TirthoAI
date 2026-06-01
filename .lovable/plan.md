## Goal

1. The preview never shows a blank white screen — even during the first Vite dependency-optimization reload, the user sees a branded loading state.
2. The chat UI feels more friendly: clearer empty state, friendlier copy, smoother visual polish, and small usability fixes (better keyboard hints, mobile-friendly sidebar, nicer toasts/confirm).

## 1. Pre-hydration splash (the "never blank" fix)

The blank period happens before React hydrates — neither the route loader nor the React loading spinners can run yet. The only thing that's live is the static HTML the server sends.

Approach: render the splash directly inside `<body>` in `src/routes/__root.tsx`'s `RootShell` as an inline element with id `boot-splash`, sitting next to `{children}`. A tiny inline `<style>` and an inline `<script>` (also in the shell) handle:

- Visible immediately, even before any JS is parsed.
- Smoothly fades out and is removed from the DOM the first time React commits real content (detected via a `MutationObserver` on the body, or a `requestIdleCallback` fallback that hides it once the root has rendered children).
- Honors the user's theme: reads `localStorage["tirthoai-theme"]` and/or `prefers-color-scheme` so the splash background matches the app's background (no light/dark flash).

What the splash shows:
- Centered TirthoAI logo mark (a CSS-only gradient square with a sparkle SVG inlined — no external assets, no font loading).
- "TirthoAI" wordmark and a soft "Loading your workspace…" subtitle.
- A subtle animated gradient bar / pulsing dot so it feels alive instead of frozen.
- All styled with inline CSS variables that match the app palette so it visually continues into the real UI.

Also tighten the in-app loading states so the handoff is seamless:
- The auth-loading and conversation-loading screens already use `Loader2`; upgrade them to the same branded "logo + label + shimmer" treatment so the transition from splash → app feels continuous.
- A new `<BrandedLoader label="…"/>` component in `src/components/BrandedLoader.tsx` is reused by both the auth gate and the chat hydration state.

## 2. UI friendliness pass

Small, focused improvements — no behavior changes to chat/auth logic.

### Auth screen (`src/components/AuthScreen.tsx`)
- Add a one-line tagline under "TirthoAI" ("Your multi-model AI workspace").
- Show inline validation messages instead of relying only on toasts for the most common cases (empty fields, password too short).
- Add a "Show password" eye toggle.
- Add a "Forgot password?" link that triggers `supabase.auth.resetPasswordForEmail`.
- Slightly larger touch targets and clearer focus rings.

### Sidebar (`src/components/Sidebar.tsx`)
- Collapsible on mobile: a hamburger button in the chat header opens/closes the sidebar as a slide-in drawer below the `md` breakpoint; on desktop it stays as the fixed left column.
- Replace the native `confirm("Delete this conversation?")` with a small inline confirm (two-step: trash icon turns red and asks "Sure?") — feels much nicer than a browser dialog.
- Group conversations by recency (Today / Yesterday / Previous 7 days / Older).
- Search box at the top of the list (client-side filter on title).
- Active conversation gets a small gradient accent bar on the left edge.

### Chat window (`src/components/ChatWindow.tsx`)
- Friendlier empty state: greet by first name if available ("Hey there 👋"), and keep the suggestion grid but make each card show the matching model category icon/color it would trigger.
- Add keyboard hint chip in the composer ("⌘/Ctrl+Enter to send · Shift+Enter for newline") and actually support Cmd/Ctrl+Enter.
- Copy-to-clipboard button on each assistant message bubble (and on every code block via a small react-markdown component override).
- Smooth auto-scroll only when the user is already near the bottom (don't yank them up if they scrolled to read earlier messages).
- Stop-generation button visible while `status === "streaming"` (wires `useChat`'s `stop`).
- Better attachment chips: image attachments show a small thumbnail preview.
- Replace the model-category pill text with a tooltip explaining what that category is good for.

### Global polish
- Configure `Toaster` once at the root level with branded colors, instead of duplicating it in every screen.
- Add a tiny `<title>` updater that prepends the active conversation title (`"Acme plan — TirthoAI"`).
- Add an `aria-live="polite"` region for streaming status for screen readers.

## Technical notes

- All new code stays client-side and TypeScript-strict; no new dependencies (we reuse `lucide-react`, `sonner`, `react-markdown`).
- Splash markup/script lives inline in `__root.tsx` so it survives any hydration error and works on the very first byte the browser receives.
- Splash is removed (not just hidden) after fade-out so it never intercepts clicks.
- Theme detection in the splash mirrors the logic in `src/hooks/use-theme.ts` (same `localStorage` key) to prevent the dark→light flash users see today.
- No changes to auth, DB schema, server functions, or the AI gateway.

## Files

New
- `src/components/BrandedLoader.tsx`

Edited
- `src/routes/__root.tsx` — inline splash markup, style, and remove-on-mount script; mount a single `Toaster`.
- `src/routes/index.tsx` — use `BrandedLoader`; pass user's display name to chat.
- `src/components/AuthScreen.tsx` — inline validation, show-password, forgot-password, tagline.
- `src/components/Sidebar.tsx` — mobile drawer, grouped+searchable list, inline delete confirm.
- `src/components/ChatWindow.tsx` — friendlier empty state, copy buttons, smart auto-scroll, stop button, image thumbnails, keyboard hints, a11y live region.
- `src/styles.css` — a couple of small utility classes for the splash fade and the gradient accent bar.
