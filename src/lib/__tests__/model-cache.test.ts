import { describe, it, expect, beforeEach, vi } from "vitest";

// We must be able to mutate the MODELS registry between tests to simulate
// a provider mapping change. vi.doMock + dynamic import lets us reload
// model-cache.ts after each registry change so MODELS_SCHEMA_SIGNATURE
// is recomputed.

const KEY_V2_PREFIX = "tirthoai.thread-model.v2";
const KEY_V1 = "tirthoai.thread-model.v1";

async function loadCacheWithModels(models: Array<{ id: string; provider?: string; category: string }>) {
  vi.resetModules();
  vi.doMock("@/lib/models", () => {
    const MODELS = models.map((m) => ({
      label: m.id,
      id: m.id,
      provider: m.provider,
      category: m.category,
      badge: "x",
      description: "x",
    }));
    const parts = MODELS.map((m) => `${m.id}|${m.provider ?? "groq"}|${m.category}`).sort();
    let h = 5381;
    const s = parts.join(";");
    for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
    const MODELS_SCHEMA_SIGNATURE = `v2.${(h >>> 0).toString(36)}`;
    return {
      MODELS,
      MODELS_SCHEMA_SIGNATURE,
      getModelById: (id: string) => MODELS.find((m) => m.id === id),
    };
  });
  return await import("@/lib/model-cache");
}

describe("ModelCache", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("persists and reads a thread's selected model across reloads", async () => {
    const { ModelCache } = await loadCacheWithModels([
      { id: "a/one", category: "general" },
      { id: "a/two", category: "reasoning" },
    ]);
    ModelCache.set("thread-1", "a/two");

    // Simulate a fresh page load: clear module cache, re-import with same registry.
    const reloaded = await loadCacheWithModels([
      { id: "a/one", category: "general" },
      { id: "a/two", category: "reasoning" },
    ]);
    expect(reloaded.ModelCache.get("thread-1")?.modelId).toBe("a/two");
  });

  it("tracks previousModelId when the selection changes", async () => {
    const { ModelCache } = await loadCacheWithModels([
      { id: "a/one", category: "general" },
      { id: "a/two", category: "reasoning" },
    ]);
    ModelCache.set("t", "a/one");
    ModelCache.set("t", "a/two");
    const entry = ModelCache.get("t");
    expect(entry?.modelId).toBe("a/two");
    expect(entry?.previousModelId).toBe("a/one");
  });

  it("invalidates the cache when MODELS provider mapping changes", async () => {
    const first = await loadCacheWithModels([
      { id: "a/one", provider: "groq", category: "general" },
    ]);
    first.ModelCache.set("t", "a/one");
    expect(first.ModelCache.get("t")?.modelId).toBe("a/one");

    // Same id, different provider → signature changes → cache must reset.
    const second = await loadCacheWithModels([
      { id: "a/one", provider: "nvidia", category: "general" },
    ]);
    expect(second.ModelCache.get("t")).toBeUndefined();
  });

  it("drops entries pointing at a removed model", async () => {
    const first = await loadCacheWithModels([
      { id: "a/one", category: "general" },
      { id: "a/removed", category: "general" },
    ]);
    first.ModelCache.set("t", "a/removed");

    // Manually rewrite the stored payload so the signature still matches
    // the new registry below, but the modelId no longer exists.
    const second = await loadCacheWithModels([{ id: "a/one", category: "general" }]);
    // Simulate stale payload with new signature but obsolete model
    localStorage.setItem(
      `${KEY_V2_PREFIX}`,
      JSON.stringify({
        sig: second.ModelCache._signature,
        threads: {
          t: { modelId: "a/removed", updatedAt: new Date().toISOString() },
        },
      }),
    );
    expect(second.ModelCache.get("t")).toBeUndefined();
  });

  it("prunes legacy v1 localStorage keys on read", async () => {
    localStorage.setItem(KEY_V1, JSON.stringify({ old: true }));
    const { ModelCache } = await loadCacheWithModels([{ id: "a/one", category: "general" }]);
    ModelCache.get("anything");
    expect(localStorage.getItem(KEY_V1)).toBeNull();
  });
});
