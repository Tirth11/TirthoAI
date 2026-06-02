import { useEffect, useState } from "react";

type BuildIssue = {
  kind: "error" | "promise" | "vite";
  message: string;
  stack?: string;
  file?: string;
  at: number;
};

/**
 * In-app build/runtime status panel.
 *
 * Catches three classes of failures that otherwise leave the user staring at
 * a blank white screen:
 *  - `window.error` (uncaught runtime / bundler-injected errors)
 *  - `unhandledrejection`
 *  - Vite HMR `vite:error` events (transpile / transform failures)
 *
 * Shown as a non-blocking bottom-right panel in dev. Dismissible. The panel
 * is intentionally NOT rendered in production builds.
 */
export function BuildStatusOverlay() {
  const [issue, setIssue] = useState<BuildIssue | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    if (!import.meta.env.DEV) return;

    const onError = (e: ErrorEvent) => {
      setDismissed(false);
      setIssue({
        kind: "error",
        message: e.message || String(e.error),
        stack: e.error?.stack,
        file: e.filename,
        at: Date.now(),
      });
    };
    const onRejection = (e: PromiseRejectionEvent) => {
      const reason = e.reason;
      setDismissed(false);
      setIssue({
        kind: "promise",
        message: reason?.message || String(reason),
        stack: reason?.stack,
        at: Date.now(),
      });
    };
    const onViteError = (e: Event) => {
      const detail = (e as CustomEvent).detail as
        | { err?: { message?: string; stack?: string; loc?: { file?: string } } }
        | undefined;
      setDismissed(false);
      setIssue({
        kind: "vite",
        message: detail?.err?.message ?? "Vite build error",
        stack: detail?.err?.stack,
        file: detail?.err?.loc?.file,
        at: Date.now(),
      });
    };

    window.addEventListener("error", onError);
    window.addEventListener("unhandledrejection", onRejection);
    window.addEventListener("vite:error", onViteError as EventListener);

    // Subscribe to Vite HMR error payloads as well.
    const hot = (import.meta as unknown as { hot?: { on: (e: string, cb: (p: unknown) => void) => void } }).hot;
    hot?.on("vite:error", (payload: unknown) => {
      const err = (payload as { err?: { message?: string; stack?: string; loc?: { file?: string } } })?.err;
      setDismissed(false);
      setIssue({
        kind: "vite",
        message: err?.message ?? "Vite build error",
        stack: err?.stack,
        file: err?.loc?.file,
        at: Date.now(),
      });
    });

    return () => {
      window.removeEventListener("error", onError);
      window.removeEventListener("unhandledrejection", onRejection);
      window.removeEventListener("vite:error", onViteError as EventListener);
    };
  }, []);

  if (!import.meta.env.DEV || !issue || dismissed) return null;

  const kindLabel =
    issue.kind === "vite" ? "Build error" : issue.kind === "promise" ? "Unhandled rejection" : "Runtime error";

  return (
    <div
      role="alert"
      style={{
        position: "fixed",
        right: 16,
        bottom: 16,
        zIndex: 2147483646,
        maxWidth: 480,
        maxHeight: "50vh",
        overflow: "auto",
        background: "#1a0b0b",
        color: "#fee2e2",
        border: "1px solid #dc2626",
        borderRadius: 12,
        padding: "12px 14px",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
        fontSize: 12,
        lineHeight: 1.5,
        boxShadow: "0 20px 50px -10px rgba(220,38,38,.45)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
        <strong style={{ color: "#fecaca", fontFamily: "system-ui, sans-serif", fontSize: 13 }}>
          ⚠ {kindLabel}
        </strong>
        <div style={{ display: "flex", gap: 6 }}>
          <button
            onClick={() => location.reload()}
            style={{
              background: "#dc2626",
              color: "white",
              border: 0,
              borderRadius: 6,
              padding: "3px 8px",
              cursor: "pointer",
              fontSize: 11,
            }}
          >
            Reload
          </button>
          <button
            onClick={() => setDismissed(true)}
            style={{
              background: "transparent",
              color: "#fecaca",
              border: "1px solid #7f1d1d",
              borderRadius: 6,
              padding: "3px 8px",
              cursor: "pointer",
              fontSize: 11,
            }}
          >
            Dismiss
          </button>
        </div>
      </div>
      {issue.file && (
        <div style={{ marginTop: 6, color: "#fca5a5", fontSize: 11 }}>{issue.file}</div>
      )}
      <pre style={{ margin: "8px 0 0", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
        {issue.message}
      </pre>
      {issue.stack && (
        <details style={{ marginTop: 6 }}>
          <summary style={{ cursor: "pointer", color: "#fca5a5" }}>Stack</summary>
          <pre style={{ margin: "6px 0 0", whiteSpace: "pre-wrap", wordBreak: "break-word", opacity: 0.85 }}>
            {issue.stack}
          </pre>
        </details>
      )}
    </div>
  );
}
