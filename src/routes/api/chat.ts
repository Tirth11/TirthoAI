import { createLovableAiGatewayProvider, createNvidiaProvider, createAnthropicProvider, createPerplexityProvider } from "@/lib/ai-gateway.server";
import { DEFAULT_MODEL, getModelById } from "@/lib/models";
import { isUserModelId, userModelRowId } from "@/lib/user-models-shared";
import { supabaseAdmin } from "@/integrations/supabase/client.server";
import { createFileRoute } from "@tanstack/react-router";
import { convertToModelMessages, streamText, type UIMessage } from "ai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";

type ChatRequestBody = { messages?: unknown; modelId?: unknown };

export const Route = createFileRoute("/api/chat")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        // --- Auth: bearer OR guest header ---
        const authHeader = request.headers.get("authorization") ?? "";
        const token = authHeader.toLowerCase().startsWith("bearer ")
          ? authHeader.slice(7).trim()
          : "";
        const guestId = (request.headers.get("x-guest-id") ?? "").trim();

        let userId: string | null = null;
        let isGuest = false;
        let remaining: number | null = null;

        if (token) {
          const { data: userData, error: userErr } = await supabaseAdmin.auth.getUser(token);
          if (userErr || !userData.user) {
            return new Response("Unauthorized", { status: 401 });
          }
          userId = userData.user.id;
          const { data: r, error: creditErr } = await supabaseAdmin.rpc(
            "consume_credit",
            { _user_id: userId },
          );
          if (creditErr) return new Response("Could not check credits", { status: 500 });
          if (typeof r === "number" && r < 0) {
            return new Response(
              JSON.stringify({
                error: "out_of_credits",
                message: "You've used all 500 free credits.",
              }),
              { status: 402, headers: { "Content-Type": "application/json" } },
            );
          }
          remaining = typeof r === "number" ? r : null;
        } else if (guestId && /^[A-Za-z0-9_-]{8,128}$/.test(guestId)) {
          isGuest = true;
          const { data: r, error: gErr } = await supabaseAdmin.rpc(
            "consume_guest_credit",
            { _guest_id: guestId, _limit: 50 },
          );
          if (gErr) return new Response("Could not check guest credits", { status: 500 });
          if (typeof r === "number" && r < 0) {
            return new Response(
              JSON.stringify({
                error: "out_of_guest_credits",
                message: "You've used your 50 free guest messages. Sign up to keep going.",
              }),
              { status: 402, headers: { "Content-Type": "application/json" } },
            );
          }
          remaining = typeof r === "number" ? r : null;
        } else {
          return new Response("Unauthorized", { status: 401 });
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

        let model;
        let chosen = DEFAULT_MODEL;
        let provider: "lovable" | "nvidia" | "anthropic" | "perplexity" | "user" = "lovable";

        if (isUserModelId(requestedId)) {
          if (isGuest || !userId) {
            return new Response(
              JSON.stringify({ error: "guest_no_user_models", message: "Sign up to use custom models." }),
              { status: 403, headers: { "Content-Type": "application/json" } },
            );
          }
          // User-added (BYO) model — look up row, decrypt key, call OpenAI-compatible endpoint.
          const rowId = userModelRowId(requestedId);
          const { data: row, error: rowErr } = await supabaseAdmin
            .from("user_models")
            .select("base_url,model_id,api_key_ciphertext,enabled,user_id")
            .eq("id", rowId)
            .maybeSingle();
          if (rowErr || !row || row.user_id !== userId) {
            return new Response(
              JSON.stringify({ error: "user_model_not_found", message: "That custom model isn't available." }),
              { status: 404, headers: { "Content-Type": "application/json" } },
            );
          }
          if (!row.enabled) {
            return new Response(
              JSON.stringify({ error: "user_model_disabled", message: "This custom model is disabled. Re-enable it in Settings." }),
              { status: 400, headers: { "Content-Type": "application/json" } },
            );
          }
          if (!row.api_key_ciphertext) {
            return new Response(
              JSON.stringify({ error: "user_model_no_key", message: "This custom model has no API key. Edit it in Settings." }),
              { status: 400, headers: { "Content-Type": "application/json" } },
            );
          }
          // Defense-in-depth SSRF guard: refuse non-https or private/internal hosts
          // even if a legacy row contains one.
          const isSafeBaseUrl = (raw: string): boolean => {
            try {
              const u = new URL(raw);
              if (u.protocol !== "https:") return false;
              const h = u.hostname.toLowerCase();
              const bare = h.startsWith("[") && h.endsWith("]") ? h.slice(1, -1) : h;
              if (["localhost", "metadata.google.internal", "metadata", "instance-data"].includes(bare)) return false;
              const v4 = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(bare);
              if (v4) {
                const a = Number(v4[1]); const b = Number(v4[2]);
                if (a === 10 || a === 127 || a === 0) return false;
                if (a === 169 && b === 254) return false;
                if (a === 172 && b >= 16 && b <= 31) return false;
                if (a === 192 && b === 168) return false;
                if (a === 100 && b >= 64 && b <= 127) return false;
                if (a >= 224) return false;
              } else if (bare.includes(":")) {
                if (bare === "::1" || bare === "::") return false;
                if (/^f[cd][0-9a-f]{2}:/i.test(bare)) return false;
                if (/^fe80:/i.test(bare)) return false;
              }
              return true;
            } catch { return false; }
          };
          if (!isSafeBaseUrl(row.base_url)) {
            return new Response(
              JSON.stringify({ error: "user_model_unsafe_base_url", message: "This custom model's base URL is not allowed. Update it in Settings to an https public endpoint." }),
              { status: 400, headers: { "Content-Type": "application/json" } },
            );
          }
          const { decryptSecret } = await import("@/lib/secret-crypto.server");

          let plainKey: string;
          try {
            plainKey = await decryptSecret(row.api_key_ciphertext);
          } catch {
            return new Response(
              JSON.stringify({ error: "user_model_key_decrypt_failed", message: "Stored key can't be decrypted. Re-enter it in Settings." }),
              { status: 500, headers: { "Content-Type": "application/json" } },
            );
          }
          chosen = row.model_id;
          provider = "user";
          model = createOpenAICompatible({
            name: "user-byo",
            baseURL: row.base_url,
            headers: { Authorization: `Bearer ${plainKey}` },
          })(chosen);
        } else {
          const chosenConfig = getModelById(requestedId);
          chosen = chosenConfig?.id ?? DEFAULT_MODEL;
          const builtinProvider = chosenConfig?.provider ?? "lovable";
          if (builtinProvider === "nvidia") {
            const nvKey = process.env.NVIDIA_API_KEY;
            if (!nvKey) {
              return new Response(
                JSON.stringify({ error: "missing_nvidia_key", message: "NVIDIA provider is not configured on the server." }),
                { status: 500, headers: { "Content-Type": "application/json" } },
              );
            }
            provider = "nvidia";
            model = createNvidiaProvider(nvKey)(chosen);
          } else {
            provider = "lovable";
            model = createLovableAiGatewayProvider(key)(chosen);
          }
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
            headers: {
              "x-credits-remaining": String(remaining ?? ""),
              ...(isGuest ? { "x-guest-remaining": String(remaining ?? "") } : {}),
            },
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
