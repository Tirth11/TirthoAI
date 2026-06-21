// Shared (client-safe) types and presets for user-added LLM models.

export type UserModelProvider =
  | "openai"
  | "anthropic"
  | "groq"
  | "openrouter"
  | "together"
  | "mistral"
  | "deepseek"
  | "custom";

export type UserModelCategory = "reasoning" | "coding" | "creative" | "vision" | "general";

export interface UserModelPreset {
  id: UserModelProvider;
  label: string;
  baseUrl: string;
  sampleModelId: string;
  docsUrl: string;
  badge: string;
}

export const PROVIDER_PRESETS: UserModelPreset[] = [
  {
    id: "openai",
    label: "OpenAI",
    baseUrl: "https://api.openai.com/v1",
    sampleModelId: "gpt-4o-mini",
    docsUrl: "https://platform.openai.com/api-keys",
    badge: "🟢",
  },
  {
    id: "anthropic",
    label: "Anthropic (Claude)",
    baseUrl: "https://api.anthropic.com/v1",
    sampleModelId: "claude-3-5-sonnet-latest",
    docsUrl: "https://console.anthropic.com/settings/keys",
    badge: "🟠",
  },
  {
    id: "groq",
    label: "Groq",
    baseUrl: "https://api.groq.com/openai/v1",
    sampleModelId: "llama-3.3-70b-versatile",
    docsUrl: "https://console.groq.com/keys",
    badge: "⚡",
  },
  {
    id: "openrouter",
    label: "OpenRouter",
    baseUrl: "https://openrouter.ai/api/v1",
    sampleModelId: "anthropic/claude-3.5-sonnet",
    docsUrl: "https://openrouter.ai/keys",
    badge: "🛣️",
  },
  {
    id: "together",
    label: "Together AI",
    baseUrl: "https://api.together.xyz/v1",
    sampleModelId: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    docsUrl: "https://api.together.xyz/settings/api-keys",
    badge: "🤝",
  },
  {
    id: "mistral",
    label: "Mistral",
    baseUrl: "https://api.mistral.ai/v1",
    sampleModelId: "mistral-large-latest",
    docsUrl: "https://console.mistral.ai/api-keys/",
    badge: "🌬️",
  },
  {
    id: "deepseek",
    label: "DeepSeek",
    baseUrl: "https://api.deepseek.com/v1",
    sampleModelId: "deepseek-chat",
    docsUrl: "https://platform.deepseek.com/api_keys",
    badge: "🌊",
  },
  {
    id: "custom",
    label: "Custom (OpenAI-compatible)",
    baseUrl: "",
    sampleModelId: "",
    docsUrl: "",
    badge: "🧩",
  },
];

export function presetFor(provider: string): UserModelPreset | undefined {
  return PROVIDER_PRESETS.find((p) => p.id === provider);
}

export const USER_MODEL_PREFIX = "user:";

export function isUserModelId(id: string): boolean {
  return id.startsWith(USER_MODEL_PREFIX);
}

export function userModelRowId(id: string): string {
  return id.slice(USER_MODEL_PREFIX.length);
}

export interface UserModelDTO {
  id: string;
  label: string;
  provider: UserModelProvider;
  base_url: string;
  model_id: string;
  category: UserModelCategory;
  enabled: boolean;
  has_key: boolean;
  key_hint: string | null;
  created_at: string;
}
