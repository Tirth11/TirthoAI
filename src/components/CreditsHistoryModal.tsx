import { useEffect, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { X, Gift, Zap, RotateCcw, Loader2 } from "lucide-react";
import { FREE_CREDITS } from "@/hooks/use-credits";
import { cn } from "@/lib/utils";

interface CreditEvent {
  id: string;
  event_type: "grant" | "debit" | "refund" | "bonus";
  amount: number;
  balance_after: number;
  reason: string | null;
  created_at: string;
}

interface Props {
  open: boolean;
  onClose: () => void;
  userId: string;
  currentCredits: number | null;
}

function formatWhen(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const diff = (now.getTime() - d.getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86_400) return `${Math.floor(diff / 3600)}h ago`;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function CreditsHistoryModal({ open, onClose, userId, currentCredits }: Props) {
  const [events, setEvents] = useState<CreditEvent[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    setLoading(true);
    supabase
      .from("credit_events")
      .select("id,event_type,amount,balance_after,reason,created_at")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(200)
      .then(({ data, error }) => {
        if (!alive) return;
        if (error) {
          console.error(error);
          setEvents([]);
        } else {
          setEvents((data ?? []) as CreditEvent[]);
        }
        setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [open, userId]);

  // ESC to close
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const used = currentCredits !== null ? FREE_CREDITS - currentCredits : null;
  const pct = currentCredits !== null ? Math.max(0, (currentCredits / FREE_CREDITS) * 100) : 0;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Credits history"
    >
      <div
        className="w-full max-w-md overflow-hidden rounded-2xl border border-border bg-card shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3 border-b border-border px-5 py-4">
          <div>
            <h2 className="text-base font-bold">Credits & History</h2>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Every grant and debit on your account
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Balance card */}
        <div className="px-5 pt-4">
          <div
            className="rounded-xl p-4 text-white shadow-md"
            style={{ background: "var(--gradient-primary)" }}
          >
            <div className="flex items-baseline justify-between">
              <span className="text-xs font-medium uppercase tracking-wider opacity-90">
                Free balance
              </span>
              <span className="text-[11px] opacity-90">of {FREE_CREDITS}</span>
            </div>
            <div className="mt-1 text-3xl font-bold">{currentCredits ?? "—"}</div>
            <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-white/25">
              <div
                className="h-full rounded-full bg-white transition-all"
                style={{ width: `${pct}%` }}
              />
            </div>
            {used !== null && (
              <div className="mt-2 text-[11px] opacity-90">{used} messages used so far</div>
            )}
          </div>
        </div>

        {/* History list */}
        <div className="max-h-[55vh] overflow-y-auto px-5 py-4">
          {loading && (
            <div className="flex items-center justify-center py-10 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
            </div>
          )}
          {!loading && events && events.length === 0 && (
            <div className="py-10 text-center text-xs text-muted-foreground">
              No credit activity yet.
            </div>
          )}
          {!loading && events && events.length > 0 && (
            <ol className="space-y-2">
              {events.map((ev) => {
                const isGrant = ev.event_type === "grant" || ev.event_type === "bonus";
                const isRefund = ev.event_type === "refund";
                const Icon = isGrant ? Gift : isRefund ? RotateCcw : Zap;
                const positive = ev.amount > 0;
                return (
                  <li
                    key={ev.id}
                    className="flex items-start gap-3 rounded-lg border border-border bg-background/40 px-3 py-2.5"
                  >
                    <div
                      className={cn(
                        "mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full",
                        isGrant
                          ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                          : isRefund
                            ? "bg-blue-500/15 text-blue-600 dark:text-blue-400"
                            : "bg-muted text-muted-foreground",
                      )}
                    >
                      <Icon className="h-3.5 w-3.5" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-baseline justify-between gap-2">
                        <span className="truncate text-xs font-medium text-foreground">
                          {ev.reason ?? labelFor(ev.event_type)}
                        </span>
                        <span
                          className={cn(
                            "shrink-0 text-xs font-semibold tabular-nums",
                            positive
                              ? "text-emerald-600 dark:text-emerald-400"
                              : "text-foreground",
                          )}
                        >
                          {positive ? "+" : ""}
                          {ev.amount}
                        </span>
                      </div>
                      <div className="mt-0.5 flex items-baseline justify-between gap-2 text-[10px] text-muted-foreground">
                        <span>{formatWhen(ev.created_at)}</span>
                        <span>balance {ev.balance_after}</span>
                      </div>
                    </div>
                  </li>
                );
              })}
            </ol>
          )}
        </div>
      </div>
    </div>
  );
}

function labelFor(t: CreditEvent["event_type"]): string {
  switch (t) {
    case "grant":
      return "Credits granted";
    case "bonus":
      return "Bonus credits";
    case "refund":
      return "Credit refund";
    case "debit":
      return "AI message";
  }
}
