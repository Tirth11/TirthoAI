import { useEffect, useMemo, useState } from "react";
import { Loader2, X, Crown, Zap, AlertTriangle, Copy, Check } from "lucide-react";
import { toast } from "sonner";
import { MODELS, getModelById, DEFAULT_MODEL } from "@/lib/models";
import { supabase } from "@/integrations/supabase/client";
import { cn } from "@/lib/utils";
import { useModelHealth } from "@/hooks/use-model-health";
import { isModelDown } from "@/lib/model-fallback";
import type { UIMessage } from "ai";
import type { CompareResult } from "@/routes/api/chat.compare";

interface Props {
  open: boolean;
  onClose: () => void;
  prompt: string;
  history: UIMessage[];
  /** Persist last-used selection for this user across sessions. */
  storageKey?: string;
}

const DEFAULT_PICK = [
  DEFAULT_MODEL,
  "openai/gpt-5-mini",
  "llama-3.3-70b-versatile",
];

const MAX_PICK = 4;

export function CompareDialog({ open, onClose, prompt, history, storageKey = "tirthoai.compare.models.v1" }: Props) {
  const [picked, setPicked] = useState<string[]>(() => {
    if (typeof window === "undefined") return DEFAULT_PICK;
    try {
      const raw = localStorage.getItem(storageKey);
      if (raw) {
        const arr = JSON.parse(raw);
        if (Array.isArray(arr) && arr.every((x) => typeof x === "string") && arr.length > 0) {
          return arr.slice(0, MAX_PICK);
        }
      }
    } catch { /* ignore */ }
    return DEFAULT_PICK;
  });
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<CompareResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { health } = useModelHealth();

  useEffect(() => {
    if (!open) return;
    setResults(null);
    setError(null);
  }, [open, prompt]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try { localStorage.setItem(storageKey, JSON.stringify(picked)); } catch { /* ignore */ }
  }, [picked, storageKey]);

  // Drop picks that have since become unhealthy so we never fan out to a known-down model.
  useEffect(() => {
    setPicked((prev) => {
      const next = prev.filter((id) => !isModelDown(id, health));
      return next.length === prev.length ? prev : next;
    });
  }, [health]);

  const grouped = useMemo(() => {
    const by: Record<string, typeof MODELS> = {};
    for (const m of MODELS) {
      if (isModelDown(m.id, health)) continue; // hide unhealthy models from selection
      const k = (m.provider ?? "lovable") as string;
      (by[k] ||= []).push(m);
    }
    return by;
  }, [health]);

  const togglePick = (id: string) => {
    setPicked((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= MAX_PICK) {
        toast.error(`Pick up to ${MAX_PICK} models`);
        return prev;
      }
      return [...prev, id];
    });
  };

  const run = async () => {
    if (!prompt.trim()) {
      toast.error("Type a prompt first, then click Compare.");
      return;
    }
    if (picked.length < 2) {
      toast.error("Pick at least 2 models to compare.");
      return;
    }
    setRunning(true);
    setError(null);
    setResults(null);
    try {
      const { data } = await supabase.auth.getSession();
      const token = data.session?.access_token;
      if (!token) throw new Error("Please sign in to compare models.");

      // Build full message list: prior history + the new user prompt
      const userMsg: UIMessage = {
        id: `compare-${Date.now()}`,
        role: "user",
        parts: [{ type: "text", text: prompt }],
      };
      const messages = [...history, userMsg];

      const res = await fetch("/api/chat/compare", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ messages, modelIds: picked }),
      });
      if (!res.ok) {
        const txt = await res.text();
        let msg = txt;
        try { msg = JSON.parse(txt).message ?? txt; } catch { /* ignore */ }
        throw new Error(msg || `Compare failed (${res.status})`);
      }
      const json = (await res.json()) as { results: CompareResult[] };
      setResults(json.results);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Compare failed";
      setError(msg);
      toast.error(msg);
    } finally {
      setRunning(false);
    }
  };

  if (!open) return null;

  const winner = results
    ? results
        .filter((r) => r.ok)
        .reduce<CompareResult | null>(
          (best, r) => (!best || (r.usage?.totalTokens ?? 0) > (best.usage?.totalTokens ?? 0) ? r : best),
          null,
        )
    : null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-2 backdrop-blur-sm sm:p-6"
      onClick={onClose}
    >
      <div
        className="flex max-h-[92vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-border px-5 py-3">
          <div>
            <h2 className="text-base font-semibold">Compare models</h2>
            <p className="text-xs text-muted-foreground">
              Pick 2–{MAX_PICK} models. Costs 1 credit per model.
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

        <div className="flex-1 overflow-y-auto p-5">
          {!results && (
            <>
              <div className="mb-4 rounded-lg border border-border bg-background/50 p-3">
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  Prompt
                </div>
                <p className="line-clamp-3 whitespace-pre-wrap break-words text-sm">
                  {prompt || <span className="italic text-muted-foreground">Type a prompt in the chat box, then re-open Compare.</span>}
                </p>
              </div>
              <div className="space-y-4">
                {Object.entries(grouped).map(([prov, list]) => (
                  <div key={prov}>
                    <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      {prov}
                    </div>
                    <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
                      {list.map((m) => {
                        const on = picked.includes(m.id);
                        return (
                          <button
                            key={`${prov}-${m.id}-${m.label}`}
                            type="button"
                            onClick={() => togglePick(m.id)}
                            className={cn(
                              "flex items-center gap-2 rounded-lg border px-2.5 py-2 text-left text-xs transition",
                              on
                                ? "border-primary bg-primary/10 text-foreground"
                                : "border-border bg-background hover:border-primary/40",
                            )}
                          >
                            <span
                              className={cn(
                                "flex h-4 w-4 shrink-0 items-center justify-center rounded border",
                                on ? "border-primary bg-primary text-primary-foreground" : "border-border",
                              )}
                            >
                              {on && <Check className="h-3 w-3" />}
                            </span>
                            <span className="truncate">
                              {m.badge} {m.label}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}

          {results && (
            <ResultsGrid results={results} winnerModelId={winner?.modelId ?? null} prompt={prompt} />
          )}

          {error && (
            <div className="mt-4 flex items-center gap-2 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertTriangle className="h-3.5 w-3.5" /> {error}
            </div>
          )}
        </div>

        <div className="flex shrink-0 items-center justify-between gap-3 border-t border-border bg-background/60 px-5 py-3">
          <div className="text-xs text-muted-foreground">
            {results ? (
              <>Compared {results.length} models · winner uses the most tokens (longest answer).</>
            ) : (
              <>{picked.length} selected · −{picked.length} credits</>
            )}
          </div>
          <div className="flex gap-2">
            {results && (
              <button
                type="button"
                onClick={() => setResults(null)}
                className="rounded-lg border border-border px-3 py-1.5 text-xs font-semibold hover:bg-accent"
              >
                Change models
              </button>
            )}
            <button
              type="button"
              onClick={run}
              disabled={running || picked.length < 2 || !prompt.trim()}
              className="inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold text-primary-foreground shadow-sm transition disabled:cursor-not-allowed disabled:opacity-50"
              style={{ background: "var(--gradient-primary)" }}
            >
              {running ? (
                <>
                  <Loader2 className="h-3 w-3 animate-spin" /> Running…
                </>
              ) : results ? (
                <>Re-run compare</>
              ) : (
                <>Run compare</>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ResultsGrid({
  results,
  winnerModelId,
  prompt,
}: {
  results: CompareResult[];
  winnerModelId: string | null;
  prompt: string;
}) {
  return (
    <div>
      <div className="mb-3 rounded-lg border border-border bg-background/50 p-3">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Prompt
        </div>
        <p className="line-clamp-2 whitespace-pre-wrap break-words text-sm">{prompt}</p>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        {results.map((r) => (
          <ResultCard key={r.modelId} r={r} isWinner={r.ok && r.modelId === winnerModelId} />
        ))}
      </div>
    </div>
  );
}

function ResultCard({ r, isWinner }: { r: CompareResult; isWinner: boolean }) {
  const [copied, setCopied] = useState(false);
  const cfg = getModelById(r.modelId);
  const total = r.usage?.totalTokens ?? 0;

  const copy = async () => {
    if (!r.text) return;
    try {
      await navigator.clipboard.writeText(r.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error("Copy failed");
    }
  };

  return (
    <div
      className={cn(
        "relative flex h-[420px] flex-col overflow-hidden rounded-xl border bg-background shadow-sm",
        isWinner ? "border-amber-500 ring-2 ring-amber-500/40" : "border-border",
      )}
    >
      {isWinner && (
        <div className="absolute -top-px right-3 inline-flex items-center gap-1 rounded-b-md bg-amber-500 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-amber-950 shadow">
          <Crown className="h-3 w-3" /> Most tokens
        </div>
      )}
      <div className="shrink-0 border-b border-border px-3 py-2">
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <div className="truncate text-xs font-semibold">
              {cfg?.badge ?? ""} {r.label}
            </div>
            <div className="truncate text-[10px] text-muted-foreground">
              {cfg?.provider ?? "lovable"} · {cfg?.id ?? r.modelId}
            </div>
          </div>
          {r.ok && (
            <span className="inline-flex items-center gap-1 rounded-full border border-primary/30 bg-primary/5 px-1.5 py-0.5 text-[10px] font-semibold tabular-nums text-primary">
              <Zap className="h-2.5 w-2.5" /> {total} tok
            </span>
          )}
        </div>
        {r.ok && r.usage && (
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] tabular-nums text-muted-foreground">
            <span>in: {r.usage.promptTokens}</span>
            <span>out: {r.usage.completionTokens}</span>
            <span>{r.latencyMs} ms</span>
          </div>
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-2 text-xs">
        {r.ok ? (
          <pre className="whitespace-pre-wrap break-words font-sans text-xs leading-relaxed text-foreground">
            {r.text}
          </pre>
        ) : (
          <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-2 text-destructive">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <div>
              <div className="text-[10px] font-semibold uppercase">Failed</div>
              <div>{r.error}</div>
            </div>
          </div>
        )}
      </div>
      {r.ok && r.text && (
        <div className="shrink-0 border-t border-border px-3 py-1.5 text-right">
          <button
            type="button"
            onClick={copy}
            className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            {copied ? <><Check className="h-3 w-3" /> Copied</> : <><Copy className="h-3 w-3" /> Copy</>}
          </button>
        </div>
      )}
    </div>
  );
}
