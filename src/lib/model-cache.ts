// Per-thread selected-model cache. Survives reloads and is read synchronously
// at mount time so the picker shows the right model before the DB fetch lands.
//
// Storage shape is versioned in two ways:
//   1. KEY contains an app-schema version — bump it whenever the on-disk
//      shape itself changes (fields added/renamed). Old keys are removed.
//   2. The stored payload embeds MODELS_SCHEMA_SIGNATURE. If the registry of
//      models or their provider/category mapping changes between sessions,
//      we drop the cache so we don't keep pointing threads at a removed
//      model or a model that has been rerouted to a different provider.

import { MODELS_SCHEMA_SIGNATURE, getModelById, MODELS } from "@/lib/models";

const APP_SCHEMA_VERSION = "v2";
const KEY = `tirthoai.thread-model.${APP_SCHEMA_VERSION}`;
const LEGACY_KEYS = ["tirthoai.thread-model.v1"];

export interface ThreadModelEntry {
  modelId: string;
  updatedAt: string;
  /** Model id that was active immediately before the current one. */
  previousModelId?: string;
}

interface Stored {
  sig: string;
  threads: Record<string, ThreadModelEntry>;
}

const EMPTY: Stored = { sig: MODELS_SCHEMA_SIGNATURE, threads: {} };

function pruneLegacy() {
  if (typeof window === "undefined") return;
  for (const k of LEGACY_KEYS) {
    try {
      localStorage.removeItem(k);
    } catch {
      /* ignore */
    }
  }
}

function isKnownModel(id: string | undefined): boolean {
  return !!id && !!getModelById(id);
}

function read(): Stored {
  if (typeof window === "undefined") return { ...EMPTY, threads: {} };
  pruneLegacy();
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...EMPTY, threads: {} };
    const parsed = JSON.parse(raw) as Partial<Stored> | null;
    if (!parsed || parsed.sig !== MODELS_SCHEMA_SIGNATURE || typeof parsed.threads !== "object") {
      // Schema / provider mapping changed -> drop everything.
      const fresh = { ...EMPTY, threads: {} };
      write(fresh);
      return fresh;
    }
    // Defensive: drop any entry pointing at a model that no longer exists.
    const threads = parsed.threads as Record<string, ThreadModelEntry>;
    let mutated = false;
    for (const tid of Object.keys(threads)) {
      const e = threads[tid];
      if (!isKnownModel(e?.modelId)) {
        delete threads[tid];
        mutated = true;
      } else if (e.previousModelId && !isKnownModel(e.previousModelId)) {
        delete e.previousModelId;
        mutated = true;
      }
    }
    const next: Stored = { sig: MODELS_SCHEMA_SIGNATURE, threads };
    if (mutated) write(next);
    return next;
  } catch {
    return { ...EMPTY, threads: {} };
  }
}

function write(c: Stored) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(KEY, JSON.stringify(c));
  } catch {
    /* ignore quota */
  }
}

const LAST_MODEL_KEY = "tirthoai.last-model.v1";

export const ModelCache = {
  get(threadId: string): ThreadModelEntry | undefined {
    return read().threads[threadId];
  },
  set(threadId: string, modelId: string) {
    const c = read();
    const prev = c.threads[threadId];
    c.threads[threadId] = {
      modelId,
      updatedAt: new Date().toISOString(),
      previousModelId:
        prev && prev.modelId && prev.modelId !== modelId ? prev.modelId : prev?.previousModelId,
    };
    write(c);
    // Track globally as the "last picked" model so a brand-new conversation
    // inherits the agent the user just chose.
    if (typeof window !== "undefined" && isKnownModel(modelId)) {
      try {
        localStorage.setItem(LAST_MODEL_KEY, modelId);
      } catch {
        /* ignore quota */
      }
    }
  },
  remove(threadId: string) {
    const c = read();
    delete c.threads[threadId];
    write(c);
  },
  /** Last model the user picked anywhere in the app. Survives reloads. */
  getLast(): string | undefined {
    if (typeof window === "undefined") return undefined;
    try {
      const id = localStorage.getItem(LAST_MODEL_KEY) ?? undefined;
      return isKnownModel(id ?? undefined) ? (id ?? undefined) : undefined;
    } catch {
      return undefined;
    }
  },
  setLast(modelId: string) {
    if (typeof window === "undefined" || !isKnownModel(modelId)) return;
    try {
      localStorage.setItem(LAST_MODEL_KEY, modelId);
    } catch {
      /* ignore quota */
    }
  },
  /** Exposed for diagnostics / tests. */
  _signature: MODELS_SCHEMA_SIGNATURE,
  _modelCount: MODELS.length,
};

