import { useEffect, useState } from "react";

/**
 * Floating diagnostic overlay that reports current safe-area-inset values,
 * visualViewport keyboard inset, and flags layout elements whose own padding
 * is smaller than the OS-reported inset (i.e. real overlap risk).
 *
 * Visibility: appears when `?safearea=1` is in the URL, or after pressing
 * Ctrl/Cmd + Shift + S. Pure client-only.
 */
type Insets = { top: number; right: number; bottom: number; left: number };

function readInsets(): Insets {
  if (typeof window === "undefined") return { top: 0, right: 0, bottom: 0, left: 0 };
  const probe = document.createElement("div");
  probe.style.cssText =
    "position:fixed;top:0;left:0;width:0;height:0;pointer-events:none;" +
    "padding-top:env(safe-area-inset-top);" +
    "padding-right:env(safe-area-inset-right);" +
    "padding-bottom:env(safe-area-inset-bottom);" +
    "padding-left:env(safe-area-inset-left);";
  document.body.appendChild(probe);
  const cs = getComputedStyle(probe);
  const insets = {
    top: parseFloat(cs.paddingTop) || 0,
    right: parseFloat(cs.paddingRight) || 0,
    bottom: parseFloat(cs.paddingBottom) || 0,
    left: parseFloat(cs.paddingLeft) || 0,
  };
  probe.remove();
  return insets;
}

function bottomPaddingOf(selector: string): number | null {
  if (typeof document === "undefined") return null;
  const el = document.querySelector(selector) as HTMLElement | null;
  if (!el) return null;
  return parseFloat(getComputedStyle(el).paddingBottom) || 0;
}

function topPaddingOf(selector: string): number | null {
  if (typeof document === "undefined") return null;
  const el = document.querySelector(selector) as HTMLElement | null;
  if (!el) return null;
  return parseFloat(getComputedStyle(el).paddingTop) || 0;
}

export function SafeAreaDiagnostic() {
  const [visible, setVisible] = useState(false);
  const [insets, setInsets] = useState<Insets>({ top: 0, right: 0, bottom: 0, left: 0 });
  const [kbInset, setKbInset] = useState(0);
  const [headerTopPad, setHeaderTopPad] = useState<number | null>(null);
  const [inputBottomPad, setInputBottomPad] = useState<number | null>(null);

  // Show via ?safearea=1 or Ctrl/Cmd+Shift+S
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const params = new URLSearchParams(window.location.search);
      if (params.get("safearea") === "1") setVisible(true);
    } catch {
      /* noop */
    }
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "s") {
        e.preventDefault();
        setVisible((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Sample insets + keyboard inset on a tick while visible.
  useEffect(() => {
    if (!visible) return;
    let raf = 0;
    const sample = () => {
      setInsets(readInsets());
      const vv = window.visualViewport;
      if (vv) {
        const ki = Math.max(0, window.innerHeight - vv.height - vv.offsetTop);
        setKbInset(ki);
      }
      setHeaderTopPad(topPaddingOf("header"));
      // The composer is the bordered container above the input form.
      setInputBottomPad(bottomPaddingOf(".shrink-0.border-t"));
      raf = window.requestAnimationFrame(sample);
    };
    sample();
    const vv = window.visualViewport;
    vv?.addEventListener("resize", sample);
    vv?.addEventListener("scroll", sample);
    return () => {
      window.cancelAnimationFrame(raf);
      vv?.removeEventListener("resize", sample);
      vv?.removeEventListener("scroll", sample);
    };
  }, [visible]);

  if (!visible) return null;

  const topOverlapRisk = headerTopPad !== null && headerTopPad < insets.top;
  const bottomOverlapRisk = inputBottomPad !== null && inputBottomPad < insets.bottom;
  const keyboardOverlap = kbInset > 0 && (inputBottomPad ?? 0) < kbInset;

  return (
    <div
      role="dialog"
      aria-label="Safe-area diagnostic"
      className="fixed left-2 right-2 bottom-2 z-[9999] mx-auto max-w-sm rounded-xl border border-border bg-card/95 p-3 text-[11px] font-mono shadow-2xl backdrop-blur sm:left-auto sm:right-3 sm:bottom-3"
      style={{ pointerEvents: "auto" }}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-xs font-semibold text-foreground">Safe-area inspector</span>
        <button
          type="button"
          onClick={() => setVisible(false)}
          className="rounded px-1.5 py-0.5 text-muted-foreground hover:bg-accent"
          aria-label="Close diagnostic"
        >
          ×
        </button>
      </div>

      <Row label="inset-top" value={`${insets.top.toFixed(1)}px`} />
      <Row label="inset-right" value={`${insets.right.toFixed(1)}px`} />
      <Row label="inset-bottom" value={`${insets.bottom.toFixed(1)}px`} />
      <Row label="inset-left" value={`${insets.left.toFixed(1)}px`} />
      <Row label="keyboard inset" value={`${kbInset.toFixed(1)}px`} />
      <hr className="my-2 border-border" />
      <Row
        label="header pad-top"
        value={headerTopPad === null ? "n/a" : `${headerTopPad.toFixed(1)}px`}
      />
      <Row
        label="composer pad-bot"
        value={inputBottomPad === null ? "n/a" : `${inputBottomPad.toFixed(1)}px`}
      />

      <div className="mt-2 space-y-1">
        <Flag ok={!topOverlapRisk} okMsg="Header clears top inset" badMsg="Header < top inset (notch overlap)" />
        <Flag
          ok={!bottomOverlapRisk}
          okMsg="Composer clears bottom inset"
          badMsg="Composer < bottom inset (home indicator overlap)"
        />
        <Flag
          ok={!keyboardOverlap}
          okMsg="Composer clears keyboard"
          badMsg="Keyboard covers composer"
        />
      </div>

      {insets.top + insets.bottom + insets.left + insets.right === 0 && (
        <p className="mt-2 text-[10px] leading-tight text-muted-foreground">
          All insets are 0 — this device/browser reports no safe-area. Real iOS with notch returns
          non-zero values once <code>viewport-fit=cover</code> is set.
        </p>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-2 py-0.5">
      <span className="text-muted-foreground">{label}</span>
      <span className="tabular-nums text-foreground">{value}</span>
    </div>
  );
}

function Flag({ ok, okMsg, badMsg }: { ok: boolean; okMsg: string; badMsg: string }) {
  return (
    <div
      className={
        "flex items-start gap-1.5 rounded-md px-2 py-1 text-[10.5px] leading-tight " +
        (ok
          ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
          : "bg-destructive/15 text-destructive")
      }
    >
      <span aria-hidden>{ok ? "✓" : "⚠"}</span>
      <span>{ok ? okMsg : badMsg}</span>
    </div>
  );
}
