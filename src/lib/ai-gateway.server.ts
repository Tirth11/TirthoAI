import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type { ModelProvider } from "./models";

export function createNvidiaProvider(nvidiaApiKey: string) {
  return createOpenAICompatible({
    name: "nvidia",
    baseURL: "https://integrate.api.nvidia.com/v1",
    headers: {
      Authorization: `Bearer ${nvidiaApiKey}`,
    },
  });
}

export function createAnthropicProvider(anthropicApiKey: string) {
  // Anthropic exposes an OpenAI-compatible endpoint at /v1
  return createOpenAICompatible({
    name: "anthropic",
    baseURL: "https://api.anthropic.com/v1",
    headers: {
      "x-api-key": anthropicApiKey,
      "anthropic-version": "2023-06-01",
    },
  });
}

export function createPerplexityProvider(perplexityApiKey: string) {
  return createOpenAICompatible({
    name: "perplexity",
    baseURL: "https://api.perplexity.ai",
    headers: {
      Authorization: `Bearer ${perplexityApiKey}`,
    },
  });
}

export function createGroqProvider(groqApiKey: string) {
  return createOpenAICompatible({
    name: "groq",
    baseURL: "https://api.groq.com/openai/v1",
    headers: {
      Authorization: `Bearer ${groqApiKey}`,
    },
  });
}

// Pollinations.ai — FREE, no API key (anonymous tier). OpenAI-compatible.
// Sourced from the "no-cost-ai" catalog: lets users chat with zero setup/keys.
export function createPollinationsProvider() {
  return createOpenAICompatible({
    name: "pollinations",
    baseURL: "https://text.pollinations.ai/openai",
    // Identify the app per Pollinations guidance; no key required.
    headers: { "X-Referrer": "tirthoai" },
  });
}

// OpenRouter — one key, hundreds of models (including many ":free" ones).
export function createOpenRouterProvider(openRouterApiKey: string) {
  return createOpenAICompatible({
    name: "openrouter",
    baseURL: "https://openrouter.ai/api/v1",
    headers: {
      Authorization: `Bearer ${openRouterApiKey}`,
      // OpenRouter attribution headers (recommended, optional).
      "HTTP-Referer": "https://tirthoai.app",
      "X-Title": "TirthoAI",
    },
  });
}

// ─── Shared provider routing (used by /api/chat and /api/chat/compare) ───
// Whether a provider can be used right now (key present, or keyless).
export function providerUsable(p: ModelProvider): boolean {
  switch (p) {
    case "groq": return !!process.env.GROQ_API_KEY;
    case "nvidia": return !!process.env.NVIDIA_API_KEY;
    case "openrouter": return !!process.env.OPENROUTER_API_KEY;
    case "anthropic": return !!process.env.ANTHROPIC_API_KEY;
    case "perplexity": return !!process.env.PERPLEXITY_API_KEY;
    case "pollinations": return true; // free, no key required
    default: return false;
  }
}

// Build a language model for a provider + model id (assumes providerUsable).
export function makeProviderModel(p: ModelProvider, id: string) {
  switch (p) {
    case "groq": return createGroqProvider(process.env.GROQ_API_KEY as string)(id);
    case "nvidia": return createNvidiaProvider(process.env.NVIDIA_API_KEY as string)(id);
    case "openrouter": return createOpenRouterProvider(process.env.OPENROUTER_API_KEY as string)(id);
    case "anthropic": return createAnthropicProvider(process.env.ANTHROPIC_API_KEY as string)(id);
    case "perplexity": return createPerplexityProvider(process.env.PERPLEXITY_API_KEY as string)(id);
    case "pollinations": return createPollinationsProvider()(id);
    default: return createPollinationsProvider()(id);
  }
}
