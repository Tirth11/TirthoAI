import { createFileRoute } from "@tanstack/react-router";
import { MODELS } from "@/lib/models";

/** Result for a single model probe. */
export type ModelHealth = {
  ok: boolean;
  provider: string;
  status?: number;
  error?: string;
  latencyMs?: number;
};

export type HealthReport = {
  checkedAt: string;
  models: Record<string, ModelHealth>;
};

// In-memory cache (per worker instance). 5 minute TTL.
let cache: { at: number; report: HealthReport } | null = null;
const TTL_MS = 5 * 60 * 1000;

async function probeNvidia(modelId: string, key: string): Promise<ModelHealth> {
  const t0 = Date.now();
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 12_000);
    const res = await fetch("https://integrate.api.nvidia.com/v1/chat/completions", {
      method: "POST",
      signal: ctrl.signal,
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: modelId,
        messages: [{ role: "user", content: "hi" }],
        max_tokens: 1,
        stream: false,
      }),
    });
    clearTimeout(timer);
    return { ok: res.ok, provider: "nvidia", status: res.status, latencyMs: Date.now() - t0 };
  } catch (e) {
    return {
      ok: false,
      provider: "nvidia",
      error: e instanceof Error ? e.message.slice(0, 120) : "network_error",
      latencyMs: Date.now() - t0,
    };
  }
}

async function probeLovable(key: string): Promise<ModelHealth> {
  const t0 = Date.now();
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 10_000);
    const res = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      signal: ctrl.signal,
      headers: { "Lovable-API-Key": key, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "google/gemini-3-flash-preview",
        messages: [{ role: "user", content: "hi" }],
        max_tokens: 1,
        stream: false,
      }),
    });
    clearTimeout(timer);
    return { ok: res.ok, provider: "lovable", status: res.status, latencyMs: Date.now() - t0 };
  } catch (e) {
    return {
      ok: false,
      provider: "lovable",
      error: e instanceof Error ? e.message.slice(0, 120) : "network_error",
      latencyMs: Date.now() - t0,
    };
  }
}

async function runReport(): Promise<HealthReport> {
  const lovableKey = process.env.LOVABLE_API_KEY;
  const nvidiaKey = process.env.NVIDIA_API_KEY;

  const lovableModels = MODELS.filter((m) => (m.provider ?? "lovable") === "lovable");
  const nvidiaModels = MODELS.filter((m) => m.provider === "nvidia");

  // One probe per provider for lovable (all lovable models share gateway health).
  const lovableHealthPromise = lovableKey
    ? probeLovable(lovableKey)
    : Promise.resolve<ModelHealth>({ ok: false, provider: "lovable", error: "no_key" });

  // Probe each NVIDIA model individually since availability varies.
  const nvProbes = nvidiaKey
    ? nvidiaModels.map((m) => probeNvidia(m.id, nvidiaKey).then((h) => [m.id, h] as const))
    : nvidiaModels.map((m) =>
        Promise.resolve([m.id, { ok: false, provider: "nvidia", error: "no_key" }] as const),
      );

  const [lovableHealth, ...nvResults] = await Promise.all([lovableHealthPromise, ...nvProbes]);

  const models: Record<string, ModelHealth> = {};
  for (const m of lovableModels) models[m.id] = lovableHealth;
  for (const [id, h] of nvResults) models[id] = h;

  return { checkedAt: new Date().toISOString(), models };
}

export const Route = createFileRoute("/api/health/models")({
  server: {
    handlers: {
      GET: async ({ request }) => {
        const url = new URL(request.url);
        const force = url.searchParams.get("refresh") === "1";
        const now = Date.now();
        if (!force && cache && now - cache.at < TTL_MS) {
          return Response.json(cache.report, {
            headers: { "Cache-Control": "public, max-age=60" },
          });
        }
        const report = await runReport();
        cache = { at: now, report };
        return Response.json(report, { headers: { "Cache-Control": "public, max-age=60" } });
      },
    },
  },
});
