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
// Background sweep interval — refreshes the model-down cache without page reloads.
const SWEEP_MS = 3 * 60 * 1000;

type Subscriber = (m: HealthMap) => void;
const subscribers = new Set<Subscriber>();
let sweepTimer: ReturnType<typeof setInterval> | null = null;
let visibilityBound = false;

function notify(map: HealthMap) {
  subscribers.forEach((cb) => {
    try {
      cb(map);
    } catch {
      // ignore subscriber errors
    }
  });
}

async function fetchHealth(force = false): Promise<HealthMap> {
  if (!force && cached && Date.now() - cached.at < TTL_MS) return cached.map;
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const res = await fetch(`/api/health/models${force ? "?refresh=1" : ""}`);
      if (!res.ok) return cached?.map ?? {};
      const json = (await res.json()) as { models?: HealthMap };
      const map = json.models ?? {};
      cached = { at: Date.now(), map };
      notify(map);
      return map;
    } catch {
      return cached?.map ?? {};
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}

function ensureBackgroundSweep() {
  if (typeof window === "undefined") return;
  if (!sweepTimer) {
    sweepTimer = setInterval(() => {
      // Skip sweep when tab is hidden — resume on visibilitychange.
      if (typeof document !== "undefined" && document.visibilityState === "hidden") return;
      void fetchHealth(true);
    }, SWEEP_MS);
  }
  if (!visibilityBound && typeof document !== "undefined") {
    visibilityBound = true;
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible" && cached && Date.now() - cached.at > SWEEP_MS) {
        void fetchHealth(true);
      }
    });
  }
}

export function useModelHealth() {
  const [health, setHealth] = useState<HealthMap>(() => cached?.map ?? {});
  const [loading, setLoading] = useState(!cached);

  useEffect(() => {
    let alive = true;
    const sub: Subscriber = (m) => {
      if (alive) setHealth(m);
    };
    subscribers.add(sub);
    ensureBackgroundSweep();

    fetchHealth().then((m) => {
      if (!alive) return;
      setHealth(m);
      setLoading(false);
    });

    return () => {
      alive = false;
      subscribers.delete(sub);
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
