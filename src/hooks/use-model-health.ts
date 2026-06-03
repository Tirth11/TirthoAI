import { useEffect, useState } from "react";

export type ModelHealth = {
  ok: boolean;
  provider: string;
  status?: number;
  error?: string;
  latencyMs?: number;
};

export type HealthMap = Record<string, ModelHealth>;

let inflight: Promise<HealthMap> | null = null;
let cached: { at: number; map: HealthMap } | null = null;
const TTL_MS = 5 * 60 * 1000;

async function fetchHealth(force = false): Promise<HealthMap> {
  if (!force && cached && Date.now() - cached.at < TTL_MS) return cached.map;
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const res = await fetch(`/api/health/models${force ? "?refresh=1" : ""}`);
      if (!res.ok) return {};
      const json = (await res.json()) as { models?: HealthMap };
      const map = json.models ?? {};
      cached = { at: Date.now(), map };
      return map;
    } catch {
      return {};
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}

export function useModelHealth() {
  const [health, setHealth] = useState<HealthMap>(() => cached?.map ?? {});
  const [loading, setLoading] = useState(!cached);

  useEffect(() => {
    let alive = true;
    fetchHealth().then((m) => {
      if (!alive) return;
      setHealth(m);
      setLoading(false);
    });
    return () => {
      alive = false;
    };
  }, []);

  const refresh = async () => {
    setLoading(true);
    const m = await fetchHealth(true);
    setHealth(m);
    setLoading(false);
  };

  return { health, loading, refresh };
}
