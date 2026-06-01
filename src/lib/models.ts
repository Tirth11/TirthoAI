export type ModelCategory = "reasoning" | "coding" | "creative" | "vision" | "general";

export type ModelProvider = "lovable" | "nvidia";

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
