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

async function probeGroq(modelId: string, key: string): Promise<ModelHealth> {
  const t0 = Date.now();
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 12_000);
    const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
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
    return { ok: res.ok, provider: "groq", status: res.status, latencyMs: Date.now() - t0 };
  } catch (e) {
    return {
      ok: false,
      provider: "groq",
      error: e instanceof Error ? e.message.slice(0, 120) : "network_error",
      latencyMs: Date.now() - t0,
    };
  }
}

async function probePollinations(modelId: string): Promise<ModelHealth> {
  const t0 = Date.now();
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 12_000);
    const res = await fetch("https://text.pollinations.ai/openai", {
      method: "POST",
      signal: ctrl.signal,
      headers: { "Content-Type": "application/json", "X-Referrer": "tirthoai" },
      body: JSON.stringify({ model: modelId, messages: [{ role: "user", content: "hi" }], max_tokens: 1, stream: false }),
    });
    clearTimeout(timer);
    return { ok: res.ok, provider: "pollinations", status: res.status, latencyMs: Date.now() - t0 };
  } catch (e) {
    return {
      ok: false,
      provider: "pollinations",
      error: e instanceof Error ? e.message.slice(0, 120) : "network_error",
      latencyMs: Date.now() - t0,
    };
  }
}

async function probeOpenRouter(modelId: string, key: string): Promise<ModelHealth> {
  const t0 = Date.now();
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 12_000);
    const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      signal: ctrl.signal,
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", "HTTP-Referer": "https://tirthoai.app", "X-Title": "TirthoAI" },
      body: JSON.stringify({ model: modelId, messages: [{ role: "user", content: "hi" }], max_tokens: 1, stream: false }),
    });
    clearTimeout(timer);
    return { ok: res.ok, provider: "openrouter", status: res.status, latencyMs: Date.now() - t0 };
  } catch (e) {
    return {
      ok: false,
      provider: "openrouter",
      error: e instanceof Error ? e.message.slice(0, 120) : "network_error",
      latencyMs: Date.now() - t0,
    };
  }
}

async function runReport(): Promise<HealthReport> {
  const nvidiaKey = process.env.NVIDIA_API_KEY;
  const groqKey = process.env.GROQ_API_KEY;
  const openrouterKey = process.env.OPENROUTER_API_KEY;

  const nvidiaModels = MODELS.filter((m) => m.provider === "nvidia");
  const groqModels = MODELS.filter((m) => m.provider === "groq");
  const pollModels = MODELS.filter((m) => m.provider === "pollinations");
  const orModels = MODELS.filter((m) => m.provider === "openrouter");
  const pollProbes = pollModels.map((m) => probePollinations(m.id).then((h) => [m.id, h] as const));
  const orProbes = openrouterKey
    ? orModels.map((m) => probeOpenRouter(m.id, openrouterKey).then((h) => [m.id, h] as const))
    : orModels.map((m) => Promise.resolve([m.id, { ok: false, provider: "openrouter", error: "no_key" }] as const));

  const nvProbes = nvidiaKey
    ? nvidiaModels.map((m) => probeNvidia(m.id, nvidiaKey).then((h) => [m.id, h] as const))
    : nvidiaModels.map((m) =>
        Promise.resolve([m.id, { ok: false, provider: "nvidia", error: "no_key" }] as const),
      );

  const groqProbes = groqKey
    ? groqModels.map((m) => probeGroq(m.id, groqKey).then((h) => [m.id, h] as const))
    : groqModels.map((m) =>
        Promise.resolve([m.id, { ok: false, provider: "groq", error: "no_key" }] as const),
      );

  const settled = await Promise.all([...nvProbes, ...groqProbes, ...pollProbes, ...orProbes]);
  const providerResults = settled as ReadonlyArray<readonly [string, ModelHealth]>;

  const models: Record<string, ModelHealth> = {};
  for (const [id, h] of providerResults) models[id] = h;

  // Server-side structured log so health failures are visible without the UI.
  const down = Object.entries(models)
    .filter(([, h]) => !h.ok)
    .map(([id, h]) => ({ id, provider: h.provider, status: h.status ?? null, error: h.error ?? null }));
  if (down.length > 0) {
    // eslint-disable-next-line no-console
    console.warn("[health] models_down", { count: down.length, down });
  }

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
