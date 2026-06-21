export type ModelCategory = "reasoning" | "coding" | "creative" | "vision" | "general";

export type ModelProvider = "nvidia" | "anthropic" | "perplexity" | "groq" | "pollinations" | "openrouter";

/** A concrete provider route: which gateway + the model id on that gateway. */
export interface ModelRoute {
  provider: ModelProvider;
  id: string;
}

export interface ModelConfig {
  label: string; // clean, user-facing name (no provider shown in the UI)
  id: string; // stable id = the PRIMARY route's model id
  category: ModelCategory;
  badge: string;
  description: string;
  supportsVision?: boolean;
  /** Primary provider/gateway for this model. */
  provider?: ModelProvider;
  /**
   * Ordered backup routes tried (server-side) when the primary provider has no
   * key or fails. Never shown in the UI — they just keep the model working
   * end-to-end. Pollinations (keyless) is appended as a universal last resort.
   */
  fallbacks?: ModelRoute[];
}

// Deduped catalog. Each entry is ONE logical model with a clean name; the same
// underlying model offered by multiple providers is collapsed into a single
// entry whose `fallbacks` list the other providers (used only for resilience).
export const MODELS: ModelConfig[] = [
  // ── General ──
  {
    label: "Llama 3.3 70B", id: "llama-3.3-70b-versatile", provider: "groq", category: "general",
    badge: "⚡", description: "Fast, capable everyday assistant (default)",
    fallbacks: [
      { provider: "nvidia", id: "meta/llama-3.3-70b-instruct" },
      { provider: "openrouter", id: "meta-llama/llama-3.3-70b-instruct" },
      { provider: "openrouter", id: "meta-llama/llama-3.3-70b-instruct:free" },
    ],
  },
  {
    label: "Llama 3.1 8B", id: "llama-3.1-8b-instant", provider: "groq", category: "general",
    badge: "🚀", description: "Tiny + lightning-fast for quick tasks",
    fallbacks: [{ provider: "openrouter", id: "meta-llama/llama-3.2-3b-instruct:free" }],
  },
  {
    label: "GPT-OSS 20B", id: "openai/gpt-oss-20b", provider: "groq", category: "general",
    badge: "🔓", description: "Open-weight GPT-OSS, balanced",
    fallbacks: [{ provider: "openrouter", id: "openai/gpt-oss-20b:free" }],
  },
  {
    label: "Mistral Medium 3.5", id: "mistralai/mistral-medium-3.5-128b", provider: "nvidia", category: "general",
    badge: "🌬️", description: "Mistral Medium 3.5",
  },
  {
    label: "Mistral Small 4", id: "mistralai/mistral-small-4-119b-2603", provider: "nvidia", category: "general",
    badge: "💨", description: "Mistral Small 4 — efficient",
  },
  {
    label: "Free Assistant", id: "openai", provider: "pollinations", category: "general",
    badge: "🆓", description: "Completely free — no API key required",
  },

  // ── Reasoning ──
  {
    label: "GPT-OSS 120B", id: "openai/gpt-oss-120b", provider: "groq", category: "reasoning",
    badge: "🧠", description: "Large open-weight reasoning model",
    fallbacks: [{ provider: "openrouter", id: "openai/gpt-oss-120b:free" }],
  },
  {
    label: "GLM 5.1", id: "z-ai/glm-5.1", provider: "nvidia", category: "reasoning",
    badge: "🐉", description: "Zhipu GLM 5.1 — strong reasoning",
  },
  {
    label: "Qwen3 Next 80B", id: "qwen/qwen3-next-80b-a3b-instruct", provider: "nvidia", category: "reasoning",
    badge: "🧩", description: "Qwen3 Next — advanced reasoning",
    fallbacks: [{ provider: "openrouter", id: "qwen/qwen3-next-80b-a3b-instruct:free" }],
  },
  {
    label: "Llama 4 Scout", id: "meta-llama/llama-4-scout-17b-16e-instruct", provider: "groq", category: "reasoning",
    badge: "🦅", description: "Llama 4 Scout MoE",
  },

  // ── Coding ──
  {
    label: "Qwen 3 32B", id: "qwen/qwen3-32b", provider: "groq", category: "coding",
    badge: "💻", description: "Fast coding model",
    fallbacks: [{ provider: "openrouter", id: "qwen/qwen3-coder:free" }],
  },
  {
    label: "Qwen3 Coder 480B", id: "qwen/qwen3-coder-480b-a35b-instruct", provider: "nvidia", category: "coding",
    badge: "🛠️", description: "Large Qwen3 Coder MoE",
    fallbacks: [{ provider: "openrouter", id: "qwen/qwen3-coder:free" }],
  },

  // ── Creative ──
  {
    label: "Kimi K2", id: "moonshotai/kimi-k2-instruct", provider: "groq", category: "creative",
    badge: "🌙", description: "Moonshot Kimi K2 — creative writing",
    fallbacks: [{ provider: "nvidia", id: "moonshotai/kimi-k2.6" }],
  },
  {
    label: "Gemma 4 31B", id: "google/gemma-4-31b-it:free", provider: "openrouter", category: "creative",
    badge: "✍️", description: "Google Gemma 4 — balanced creative output",
  },

  // ── Vision ──
  {
    label: "Llama 4 Maverick", id: "meta-llama/llama-4-maverick-17b-128e-instruct", provider: "groq", category: "vision",
    badge: "🖼️", description: "Multimodal — understands images", supportsVision: true,
  },
];

export const CATEGORY_META: Record<ModelCategory, { label: string; icon: string; color: string }> = {
  reasoning: { label: "Reasoning", icon: "🧠", color: "from-violet-500 to-fuchsia-500" },
  coding:    { label: "Coding",    icon: "💻", color: "from-indigo-500 to-blue-500" },
  creative:  { label: "Creative",  icon: "✍️", color: "from-pink-500 to-rose-500" },
  vision:    { label: "Vision",    icon: "🖼️", color: "from-emerald-500 to-teal-500" },
  general:   { label: "General",   icon: "🚀", color: "from-amber-500 to-orange-500" },
};

export const DEFAULT_MODEL = "llama-3.3-70b-versatile";

/**
 * Signature of the model registry. Bumps when ids/providers/categories change so
 * client caches auto-invalidate stale entries.
 */
export const MODELS_SCHEMA_SIGNATURE: string = (() => {
  const parts = MODELS
    .map((m) => `${m.id}|${m.provider ?? "groq"}|${m.category}|${(m.fallbacks ?? []).map((f) => f.provider + ":" + f.id).join(",")}`)
    .sort();
  let h = 5381;
  const s = parts.join(";");
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return `v3.${(h >>> 0).toString(36)}`;
})();

export function getModelById(id: string): ModelConfig | undefined {
  return MODELS.find((m) => m.id === id);
}

export function getModelByLabel(label: string): ModelConfig | undefined {
  return MODELS.find((m) => m.label === label);
}

/**
 * Ordered list of provider routes to try for a built-in model: the primary
 * route first, then its fallbacks. Used server-side for automatic failover.
 */
export function routesFor(cfg: ModelConfig): ModelRoute[] {
  return [{ provider: cfg.provider ?? "groq", id: cfg.id }, ...(cfg.fallbacks ?? [])];
}

const DEFAULTS_BY_CATEGORY: Record<ModelCategory, string> = {
  reasoning: "openai/gpt-oss-120b",
  coding: "qwen/qwen3-32b",
  creative: "moonshotai/kimi-k2-instruct",
  vision: "meta-llama/llama-4-maverick-17b-128e-instruct",
  general: DEFAULT_MODEL,
};

export function autoSelectModel(text: string, hasImage: boolean): string {
  if (hasImage) return DEFAULTS_BY_CATEGORY.vision;
  const t = (text || "").toLowerCase();
  const wordCount = t.split(/\s+/).filter(Boolean).length;
  const greet = ["hi", "hello", "hey", "yo", "sup"];
  if (wordCount <= 4 && greet.some((g) => t === g || t.startsWith(g + " "))) return DEFAULT_MODEL;

  const kws: Record<ModelCategory, string[]> = {
    coding: ["code", "python", "javascript", "typescript", "html", "css", "react", "debug", "function", "class", "bug", "stack trace", "error", "compile", "regex", "api", "sql"],
    vision: ["image", "photo", "picture", "screenshot", "describe this", "what's in"],
    reasoning: ["think", "reason", "solve", "prove", "logic", "math", "calculate", "complex", "analyze", "step by step", "why does"],
    creative: ["story", "poem", "haiku", "write", "creative", "blog", "essay", "draft", "tagline", "slogan"],
    general: [],
  };

  const scores: Record<ModelCategory, number> = { reasoning: 0, coding: 0, creative: 0, vision: 0, general: 0 };
  for (const cat of Object.keys(kws) as ModelCategory[]) {
    for (const kw of kws[cat]) if (t.includes(kw)) scores[cat] += 1;
  }
  const best = (Object.keys(scores) as ModelCategory[]).reduce((a, b) => (scores[b] > scores[a] ? b : a), "general");
  if (scores[best] < 1) return DEFAULT_MODEL;
  return DEFAULTS_BY_CATEGORY[best];
}
