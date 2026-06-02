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

  // NVIDIA-hosted (free tier via NVIDIA Build)
  { label: "GLM 5.1", id: "z-ai/glm-5.1", provider: "nvidia", category: "reasoning", badge: "🐉", description: "Zhipu GLM 5.1 via NVIDIA" },
  { label: "GLM 4.6", id: "z-ai/glm-4.6", provider: "nvidia", category: "general", badge: "🐲", description: "Zhipu GLM 4.6 via NVIDIA" },
  { label: "DeepSeek V4 Flash", id: "deepseek-ai/deepseek-v4-flash", provider: "nvidia", category: "reasoning", badge: "🌊", description: "DeepSeek thinking model via NVIDIA" },
  { label: "DeepSeek R1", id: "deepseek-ai/deepseek-r1", provider: "nvidia", category: "reasoning", badge: "🔮", description: "DeepSeek R1 reasoning via NVIDIA" },
  { label: "Kimi K2", id: "moonshotai/kimi-k2-instruct", provider: "nvidia", category: "creative", badge: "🌙", description: "Moonshot Kimi K2 via NVIDIA" },
  { label: "Llama 3.3 70B", id: "meta/llama-3.3-70b-instruct", provider: "nvidia", category: "general", badge: "🦙", description: "Meta Llama 3.3 70B via NVIDIA" },
  { label: "Qwen 2.5 Coder 32B", id: "qwen/qwen2.5-coder-32b-instruct", provider: "nvidia", category: "coding", badge: "🧩", description: "Qwen Coder via NVIDIA" },
  { label: "Llama 4 Maverick", id: "meta/llama-4-maverick-17b-128e-instruct", provider: "nvidia", category: "general", badge: "🦅", description: "Meta Llama 4 Maverick via NVIDIA" },
  { label: "Llama 4 Scout", id: "meta/llama-4-scout-17b-16e-instruct", provider: "nvidia", category: "reasoning", badge: "🛰️", description: "Meta Llama 4 Scout via NVIDIA" },
  { label: "Mistral Small 3.1", id: "mistralai/mistral-small-24b-instruct", provider: "nvidia", category: "general", badge: "🌬️", description: "Mistral Small via NVIDIA" },
  { label: "Nemotron 70B", id: "nvidia/llama-3.1-nemotron-70b-instruct", provider: "nvidia", category: "reasoning", badge: "⚙️", description: "NVIDIA Nemotron 70B" },
  { label: "Qwen 3 235B", id: "qwen/qwen3-235b-a22b", provider: "nvidia", category: "reasoning", badge: "🧠", description: "Qwen 3 MoE via NVIDIA" },

  // Anthropic Claude (requires ANTHROPIC_API_KEY)
  { label: "Claude Sonnet 4.5", id: "claude-sonnet-4-5", provider: "anthropic", category: "reasoning", badge: "🎼", description: "Anthropic Claude Sonnet 4.5" },
  { label: "Claude Opus 4.1", id: "claude-opus-4-1", provider: "anthropic", category: "reasoning", badge: "🪕", description: "Anthropic Claude Opus 4.1 — top quality" },
  { label: "Claude Haiku 4.5", id: "claude-haiku-4-5", provider: "anthropic", category: "general", badge: "🍃", description: "Fast, cheap Claude Haiku" },

  // Perplexity Sonar (requires PERPLEXITY_API_KEY) — web-grounded answers
  { label: "Sonar", id: "sonar", provider: "perplexity", category: "general", badge: "🔎", description: "Perplexity web-grounded search" },
  { label: "Sonar Pro", id: "sonar-pro", provider: "perplexity", category: "reasoning", badge: "🛰️", description: "Perplexity multi-step research with citations" },
  { label: "Sonar Reasoning", id: "sonar-reasoning", provider: "perplexity", category: "reasoning", badge: "🧭", description: "Chain-of-thought with live search" },
];


export const CATEGORY_META: Record<ModelCategory, { label: string; icon: string; color: string }> = {
  reasoning: { label: "Reasoning", icon: "🧠", color: "from-violet-500 to-fuchsia-500" },
  coding:    { label: "Coding",    icon: "💻", color: "from-indigo-500 to-blue-500" },
  creative:  { label: "Creative",  icon: "✍️", color: "from-pink-500 to-rose-500" },
  vision:    { label: "Vision",    icon: "🖼️", color: "from-emerald-500 to-teal-500" },
  general:   { label: "General",   icon: "🚀", color: "from-amber-500 to-orange-500" },
};

export const DEFAULT_MODEL = "google/gemini-3-flash-preview";

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
