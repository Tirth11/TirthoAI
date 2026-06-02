import { createLovableAiGatewayProvider, createNvidiaProvider } from "@/lib/ai-gateway.server";
import { DEFAULT_MODEL, getModelById } from "@/lib/models";
import { supabaseAdmin } from "@/integrations/supabase/client.server";
import { createFileRoute } from "@tanstack/react-router";
import { convertToModelMessages, streamText, type UIMessage } from "ai";

type ChatRequestBody = { messages?: unknown; modelId?: unknown };

export const Route = createFileRoute("/api/chat")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        // --- Auth: require bearer token ---
        const authHeader = request.headers.get("authorization") ?? "";
        const token = authHeader.toLowerCase().startsWith("bearer ")
          ? authHeader.slice(7).trim()
          : "";
        if (!token) {
          return new Response("Unauthorized", { status: 401 });
        }
        const { data: userData, error: userErr } = await supabaseAdmin.auth.getUser(token);
        if (userErr || !userData.user) {
          return new Response("Unauthorized", { status: 401 });
        }
        const userId = userData.user.id;

        // --- Credits: atomic decrement, block at 0 ---
        const { data: remaining, error: creditErr } = await supabaseAdmin.rpc(
          "consume_credit",
          { _user_id: userId },
        );
        if (creditErr) {
          return new Response("Could not check credits", { status: 500 });
        }
        if (typeof remaining === "number" && remaining < 0) {
          return new Response(
            JSON.stringify({
              error: "out_of_credits",
              message:
                "You've used all 500 free credits. Free tier is exhausted — thanks for trying TirthoAI!",
            }),
            { status: 402, headers: { "Content-Type": "application/json" } },
          );
        }

        const body = (await request.json()) as ChatRequestBody;
        const { messages, modelId } = body;
        if (!Array.isArray(messages)) {
          return new Response("Messages are required", { status: 400 });
        }

        const key = process.env.LOVABLE_API_KEY;
        if (!key) {
          return new Response("Missing LOVABLE_API_KEY", { status: 500 });
        }

        const requestedId = typeof modelId === "string" ? modelId : DEFAULT_MODEL;
        const chosenConfig = getModelById(requestedId);
        const chosen = chosenConfig?.id ?? DEFAULT_MODEL;
        const provider = chosenConfig?.provider ?? "lovable";

        let model;
        if (provider === "nvidia") {
          const nvKey = process.env.NVIDIA_API_KEY;
          if (!nvKey) {
            return new Response(
              JSON.stringify({ error: "missing_nvidia_key", message: "NVIDIA provider is not configured on the server." }),
              { status: 500, headers: { "Content-Type": "application/json" } },
            );
          }
          model = createNvidiaProvider(nvKey)(chosen);
        } else {
          model = createLovableAiGatewayProvider(key)(chosen);
        }

        // Classify provider errors into safe, user-facing messages.
        // Never include API keys, tokens, or full provider response bodies.
        const classifyProviderError = (err: unknown): { status: number; code: string; message: string } => {
          const raw = err instanceof Error ? err.message : String(err ?? "");
          const safe = raw
            .replace(/Bearer\s+[A-Za-z0-9._-]+/gi, "Bearer ***")
            .replace(/nvapi-[A-Za-z0-9_-]+/g, "nvapi-***")
            .replace(/sk-[A-Za-z0-9_-]+/g, "sk-***");

          const statusMatch = /\b(401|403|404|429|5\d{2})\b/.exec(safe);
          const status: number | undefined =
            (err as { statusCode?: number })?.statusCode ??
            (err as { status?: number })?.status ??
            (statusMatch ? Number(statusMatch[1]) : undefined);

          if (provider === "nvidia") {
            if (status === 401) {
              return {
                status: 401,
                code: "nvidia_unauthorized",
                message:
                  "NVIDIA rejected the API key (401). NVIDIA_API_KEY is missing, expired, or invalid. Rotate the server secret and retry.",
              };
            }
            if (status === 403) {
              return {
                status: 403,
                code: "nvidia_forbidden",
                message:
                  "NVIDIA refused the request (403). The key is valid but lacks access to this model, or the account is out of quota.",
              };
            }
            if (status === 404) {
              return {
                status: 404,
                code: "nvidia_model_not_found",
                message: `NVIDIA returned 404 for model "${chosen}". The model id may be wrong or unavailable to this account.`,
              };
            }
            if (status === 429) {
              return { status: 429, code: "nvidia_rate_limited", message: "NVIDIA rate limit hit (429). Retry shortly." };
            }
            if (status && status >= 500) {
              return {
                status: 502,
                code: "nvidia_upstream_error",
                message: `NVIDIA upstream error (${status}). Provider is having issues — retry shortly.`,
              };
            }
            return { status: 500, code: "nvidia_request_failed", message: `NVIDIA request failed: ${safe.slice(0, 300)}` };
          }

          if (status === 401 || status === 403) {
            return { status, code: "lovable_unauthorized", message: "AI gateway rejected the request." };
          }
          if (status === 429) {
            return { status: 429, code: "rate_limited", message: "Rate limit hit. Retry shortly." };
          }
          return { status: 500, code: "ai_request_failed", message: safe.slice(0, 300) || "AI request failed" };
        };

        try {
          const result = streamText({
            model,
            system:
              "You are TirthoAI, a friendly, capable multi-model AI assistant. " +
              "Respond clearly and use markdown (headings, lists, fenced code blocks) when it improves clarity. " +
              "If the user shares an image or file, refer to it naturally.",
            messages: await convertToModelMessages(messages as UIMessage[]),
            onError: ({ error }) => {
              const c = classifyProviderError(error);
              console.error(`[chat] provider=${provider} model=${chosen} code=${c.code} status=${c.status} :: ${c.message}`);
            },
          });

          return result.toUIMessageStreamResponse({
            originalMessages: messages as UIMessage[],
            headers: { "x-credits-remaining": String(remaining ?? "") },
            onError: (error) => {
              const c = classifyProviderError(error);
              return JSON.stringify({ error: c.code, message: c.message });
            },
          });
        } catch (err) {
          const c = classifyProviderError(err);
          console.error(`[chat] provider=${provider} model=${chosen} code=${c.code} status=${c.status} :: ${c.message}`);
          return new Response(
            JSON.stringify({ error: c.code, message: c.message }),
            { status: c.status, headers: { "Content-Type": "application/json" } },
          );
        }
      },
    },
  },
});
