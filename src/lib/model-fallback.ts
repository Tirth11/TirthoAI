import { MODELS, DEFAULT_MODEL, getModelById, type ModelConfig } from "@/lib/models";
import type { HealthMap } from "@/hooks/use-model-health";

/**
 * Pick a healthy fallback when the active model has failed health checks.
 *
 * Preference order:
 *   1. Same provider + same category, healthy.
 *   2. Same category, healthy.
 *   3. Same provider, healthy.
 *   4. Any healthy model.
 *   5. The default model (last resort, even if unknown health).
 */
export function pickHealthyFallback(
  currentId: string,
  health: HealthMap,
): { id: string; reason: "same-provider-category" | "same-category" | "same-provider" | "any" | "default" } {
  const current = getModelById(currentId);
  const isHealthy = (m: ModelConfig) => {
    const h = health[m.id];
    // unknown health is treated as ok so we don't churn before first probe.
    return !h || h.ok;
  };

  if (current) {
    const provider = current.provider ?? "groq";
    const sameProviderCategory = MODELS.find(
      (m) => m.id !== currentId && (m.provider ?? "groq") === provider && m.category === current.category && isHealthy(m),
    );
    if (sameProviderCategory) return { id: sameProviderCategory.id, reason: "same-provider-category" };

    const sameCategory = MODELS.find(
      (m) => m.id !== currentId && m.category === current.category && isHealthy(m),
    );
    if (sameCategory) return { id: sameCategory.id, reason: "same-category" };

    const sameProvider = MODELS.find(
      (m) => m.id !== currentId && (m.provider ?? "groq") === provider && isHealthy(m),
    );
    if (sameProvider) return { id: sameProvider.id, reason: "same-provider" };
  }

  const anyHealthy = MODELS.find((m) => m.id !== currentId && isHealthy(m));
  if (anyHealthy) return { id: anyHealthy.id, reason: "any" };

  return { id: DEFAULT_MODEL, reason: "default" };
}

export function isModelDown(modelId: string, health: HealthMap): boolean {
  const h = health[modelId];
  return !!h && !h.ok;
}
