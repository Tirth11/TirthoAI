import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Outlet,
  Link,
  createRootRouteWithContext,
  useRouter,
  HeadContent,
  Scripts,
} from "@tanstack/react-router";
import { useEffect, type ReactNode } from "react";
import { Toaster } from "sonner";

import appCss from "../styles.css?url";
import { reportLovableError } from "../lib/lovable-error-reporting";
import { BuildStatusOverlay } from "../components/BuildStatusOverlay";
import { SafeAreaDiagnostic } from "../components/SafeAreaDiagnostic";

function NotFoundComponent() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-7xl font-bold text-foreground">404</h1>
        <h2 className="mt-4 text-xl font-semibold text-foreground">Page not found</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <div className="mt-6">
          <Link
            to="/"
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Go home
          </Link>
        </div>
      </div>
    </div>
  );
}

function ErrorComponent({ error, reset }: { error: Error; reset: () => void }) {
  console.error(error);
  const router = useRouter();
  useEffect(() => {
    reportLovableError(error, { boundary: "tanstack_root_error_component" });
  }, [error]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-xl font-semibold tracking-tight text-foreground">
          This page didn't load
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Something went wrong on our end. You can try refreshing or head back home.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <button
            onClick={() => {
              router.invalidate();
              reset();
            }}
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Try again
          </button>
          <a
            href="/"
            className="inline-flex items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent"
          >
            Go home
          </a>
        </div>
      </div>
    </div>
  );
}

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1, viewport-fit=cover" },
      { title: "TirthoAI — Multi-Model AI Workspace" },
      {
        name: "description",
        content:
          "TirthoAI is a multi-model AI chat platform — reasoning, coding, vision, and creative models in one place, with persistent chat history.",
      },
      { name: "author", content: "TirthoAI" },
      { property: "og:title", content: "TirthoAI — Multi-Model AI Workspace" },
      {
        property: "og:description",
        content: "Chat with the best AI models. Your history is saved automatically.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
    links: [
      {
        rel: "stylesheet",
        href: appCss,
      },
    ],
  }),
  shellComponent: RootShell,
  component: RootComponent,
  notFoundComponent: NotFoundComponent,
  errorComponent: ErrorComponent,
});

// Inline script: synchronously set the dark/light class on <html> before any
// CSS or markup paints, so the splash matches the user's saved theme.
const THEME_BOOT_SCRIPT = `(function(){try{var k='tirthoai.theme';var s=localStorage.getItem(k);var prefersDark=window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches;var t=s||(prefersDark?'dark':'dark');if(t==='dark')document.documentElement.classList.add('dark');}catch(e){document.documentElement.classList.add('dark');}})();`;

// Inline styles for the boot splash. Self-contained so they work even when
// the main stylesheet hasn't loaded yet.
const SPLASH_STYLES = `
#boot-splash{position:fixed;inset:0;z-index:2147483647;display:flex;align-items:center;justify-content:center;background:#fafbfd;color:#1a1a2e;font-family:'Inter',system-ui,-apple-system,sans-serif;transition:opacity 380ms ease, visibility 380ms ease;}
.dark #boot-splash{background:#0f1020;color:#f3f3f8;}
#boot-splash.is-hidden{opacity:0;visibility:hidden;pointer-events:none;}
#boot-splash .bs-inner{display:flex;flex-direction:column;align-items:center;gap:18px;}
#boot-splash .bs-logo{width:64px;height:64px;border-radius:18px;background:linear-gradient(135deg,#7c3aed,#c026d3);box-shadow:0 18px 50px -12px rgba(124,58,237,.55);display:flex;align-items:center;justify-content:center;color:#fff;animation:bsPulse 1.6s ease-in-out infinite;}
#boot-splash .bs-title{font-size:18px;font-weight:700;letter-spacing:-.01em;text-align:center;}
#boot-splash .bs-sub{margin-top:4px;font-size:12px;opacity:.65;text-align:center;}
#boot-splash .bs-bar{position:relative;width:160px;height:4px;border-radius:999px;background:rgba(120,120,140,.18);overflow:hidden;}
#boot-splash .bs-bar::after{content:'';position:absolute;inset:0;width:40%;border-radius:999px;background:linear-gradient(90deg,#7c3aed,#c026d3);animation:bsSlide 1.4s ease-in-out infinite;}
@keyframes bsPulse{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(.94);opacity:.82}}
@keyframes bsSlide{0%{transform:translateX(-100%)}100%{transform:translateX(350%)}}
`;

// Script that hides the splash once React has committed real content.
// Uses a MutationObserver to detect the first non-splash element added to
// <body>, with a 400ms minimum so the splash isn't a flicker, and a 15s
// safety timeout so it cannot get stuck.
const SPLASH_HIDE_SCRIPT = `(function(){var SHOWN_AT=Date.now();var MIN_SHOW=400;function hide(){var el=document.getElementById('boot-splash');if(!el||el.classList.contains('is-hidden'))return;var wait=Math.max(0,MIN_SHOW-(Date.now()-SHOWN_AT));setTimeout(function(){el.classList.add('is-hidden');setTimeout(function(){el&&el.parentNode&&el.parentNode.removeChild(el)},500);},wait);}window.__tirthoHideSplash=hide;var obs=new MutationObserver(function(){var body=document.body;if(!body)return;for(var i=0;i<body.children.length;i++){var c=body.children[i];if(c.id==='boot-splash')continue;if(c.tagName==='SCRIPT'||c.tagName==='STYLE')continue;if(c.children&&c.children.length>0){hide();obs.disconnect();return;}}});if(document.body){obs.observe(document.body,{childList:true,subtree:true});}else{document.addEventListener('DOMContentLoaded',function(){obs.observe(document.body,{childList:true,subtree:true});});}setTimeout(hide,15000);})();`;

function RootShell({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* eslint-disable-next-line react/no-danger */}
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT_SCRIPT }} />
        <style dangerouslySetInnerHTML={{ __html: SPLASH_STYLES }} />
        <HeadContent />
      </head>
      <body suppressHydrationWarning>
        <div id="boot-splash" aria-hidden="false" suppressHydrationWarning>
          <div className="bs-inner">
            <div className="bs-logo">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 3l1.6 4.4L18 9l-4.4 1.6L12 15l-1.6-4.4L6 9l4.4-1.6z" />
                <path d="M19 14l.8 2.2L22 17l-2.2.8L19 20l-.8-2.2L16 17l2.2-.8z" />
              </svg>
            </div>
            <div>
              <div className="bs-title">TirthoAI</div>
              <div className="bs-sub">Loading your workspace…</div>
            </div>
            <div className="bs-bar" />
          </div>
        </div>
        {children}
        <script dangerouslySetInnerHTML={{ __html: SPLASH_HIDE_SCRIPT }} />
        <Scripts />
      </body>
    </html>
  );
}

function RootComponent() {
  const { queryClient } = Route.useRouteContext();

  // Belt-and-braces: once React mounts, explicitly call the splash hider so
  // we never depend solely on the MutationObserver heuristic.
  useEffect(() => {
    const w = window as unknown as { __tirthoHideSplash?: () => void };
    w.__tirthoHideSplash?.();
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <Outlet />
      <Toaster
        position="top-center"
        richColors
        theme="system"
        toastOptions={{
          style: {
            borderRadius: "12px",
          },
        }}
      />
      <BuildStatusOverlay />
    </QueryClientProvider>
  );
}
