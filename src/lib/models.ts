export type ModelCategory = "reasoning" | "coding" | "creative" | "vision" | "general";

export type ModelProvider = "lovable" | "nvidia" | "anthropic" | "perplexity";

export interface ModelConfig {
  label: string;
  id: string; // gateway model id
  category: ModelCategory;
  badge: string;
  description: string;
  supportsVision?: boolean;
  /** Which gateway to route to. Defaults to "lovable". */
  provider?: ModelProvider;
}

export const MODELS: ModelConfig[] = [
  // Reasoning
  { label: "GPT-5.4 Reasoner", id: "openai/gpt-5.4", category: "reasoning", badge: "🧠", description: "Advanced multi-step reasoning & analysis" },
  { label: "Gemini 3.1 Pro", id: "google/gemini-3.1-pro-preview", category: "reasoning", badge: "🏛️", description: "Strong next-gen Gemini reasoning" },
  { label: "GPT-5.2 Thinker", id: "openai/gpt-5.2", category: "reasoning", badge: "🔬", description: "Complex problem solving" },

  // Coding
  { label: "GPT-5.4 Coder", id: "openai/gpt-5.4", category: "coding", badge: "💻", description: "Best-in-class code generation" },
  { label: "Gemini 3.5 Flash", id: "google/gemini-3.5-flash", category: "coding", badge: "⚡", description: "Fast, capable code helper" },

  // Creative
  { label: "GPT-5 Mini", id: "openai/gpt-5-mini", category: "creative", badge: "✍️", description: "Creative writing & storytelling" },
  { label: "Gemini 2.5 Flash", id: "google/gemini-2.5-flash", category: "creative", badge: "💎", description: "Balanced creative output" },

  // Vision
  { label: "Gemini 2.5 Pro Vision", id: "google/gemini-2.5-pro", category: "vision", badge: "🖼️", description: "Powerful multimodal image understanding", supportsVision: true },
  { label: "GPT-5 Vision", id: "openai/gpt-5", category: "vision", badge: "👁️", description: "Image analysis & description", supportsVision: true },

  // General
  { label: "Gemini 3 Flash", id: "google/gemini-3-flash-preview", category: "general", badge: "🚀", description: "Fast everyday assistant (default)" },
  { label: "Gemini 3.1 Flash Lite", id: "google/gemini-3.1-flash-lite-preview", category: "general", badge: "🪶", description: "Cost-efficient, high volume" },
  { label: "GPT-5 Nano", id: "openai/gpt-5-nano", category: "general", badge: "⚡", description: "Quick simple tasks" },

  // NVIDIA-hosted (verified live on NVIDIA Build)
  { label: "GLM 5.1", id: "z-ai/glm-5.1", provider: "nvidia", category: "reasoning", badge: "🐉", description: "Zhipu GLM 5.1 via NVIDIA" },
  { label: "Kimi K2.6", id: "moonshotai/kimi-k2.6", provider: "nvidia", category: "creative", badge: "🌙", description: "Moonshot Kimi K2.6 via NVIDIA" },
  { label: "Llama 3.3 70B", id: "meta/llama-3.3-70b-instruct", provider: "nvidia", category: "general", badge: "🦙", description: "Meta Llama 3.3 70B via NVIDIA" },
  
  { label: "Qwen3 Coder 480B", id: "qwen/qwen3-coder-480b-a35b-instruct", provider: "nvidia", category: "coding", badge: "🧩", description: "Qwen3 Coder MoE via NVIDIA" },
  { label: "Qwen3 Next 80B", id: "qwen/qwen3-next-80b-a3b-instruct", provider: "nvidia", category: "reasoning", badge: "🧠", description: "Qwen3 Next via NVIDIA" },
  { label: "GPT-OSS 20B", id: "openai/gpt-oss-20b", provider: "nvidia", category: "general", badge: "🔓", description: "OpenAI GPT-OSS 20B via NVIDIA" },
  { label: "Mistral Medium 3.5", id: "mistralai/mistral-medium-3.5-128b", provider: "nvidia", category: "general", badge: "🌬️", description: "Mistral Medium 3.5 via NVIDIA" },
  { label: "Mistral Small 4", id: "mistralai/mistral-small-4-119b-2603", provider: "nvidia", category: "general", badge: "💨", description: "Mistral Small 4 via NVIDIA" },
];


export const CATEGORY_META: Record<ModelCategory, { label: string; icon: string; color: string }> = {
  reasoning: { label: "Reasoning", icon: "🧠", color: "from-violet-500 to-fuchsia-500" },
  coding:    { label: "Coding",    icon: "💻", color: "from-indigo-500 to-blue-500" },
  creative:  { label: "Creative",  icon: "✍️", color: "from-pink-500 to-rose-500" },
  vision:    { label: "Vision",    icon: "🖼️", color: "from-emerald-500 to-teal-500" },
  general:   { label: "General",   icon: "🚀", color: "from-amber-500 to-orange-500" },
};

export const DEFAULT_MODEL = "google/gemini-3-flash-preview";

/**
 * Signature of the model registry + provider mapping. Bumps automatically
 * whenever a model id, its provider, or its category changes. Used by client
 * caches (e.g. ModelCache) to auto-invalidate stale entries that may point at
 * a removed model or a model that has been rerouted to a different provider.
 */
export const MODELS_SCHEMA_SIGNATURE: string = (() => {
  const parts = MODELS
    .map((m) => `${m.id}|${m.provider ?? "lovable"}|${m.category}`)
    .sort();
  // tiny fast hash (djb2)
  let h = 5381;
  const s = parts.join(";");
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return `v2.${(h >>> 0).toString(36)}`;
})();

export function getModelById(id: string): ModelConfig | undefined {
  return MODELS.find((m) => m.id === id);
}

export function getModelByLabel(label: string): ModelConfig | undefined {
  return MODELS.find((m) => m.label === label);
}

const DEFAULTS_BY_CATEGORY: Record<ModelCategory, string> = {
  reasoning: "openai/gpt-5.4",
  coding: "openai/gpt-5.4",
  creative: "openai/gpt-5-mini",
  vision: "google/gemini-2.5-pro",
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
