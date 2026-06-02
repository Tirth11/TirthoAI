import { createOpenAICompatible } from "@ai-sdk/openai-compatible";

export function createLovableAiGatewayProvider(lovableApiKey: string) {
  return createOpenAICompatible({
    name: "lovable",
    baseURL: "https://ai.gateway.lovable.dev/v1",
    headers: {
      "Lovable-API-Key": lovableApiKey,
      "X-Lovable-AIG-SDK": "vercel-ai-sdk",
    },
  });
}

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
