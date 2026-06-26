import { providerUsable, makeProviderModel } from "@/lib/ai-gateway.server";
import { DEFAULT_MODEL, getModelById, routesFor, VISION_ROUTES, dedupeRoutes, type ModelProvider, type ModelRoute } from "@/lib/models";
import { isUserModelId, userModelRowId } from "@/lib/user-models-shared";
import { supabaseAdmin } from "@/integrations/supabase/client.server";
import { createFileRoute } from "@tanstack/react-router";
import { convertToModelMessages, streamText, type UIMessage } from "ai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createClient } from "@supabase/supabase-js";

type ChatRequestBody = {
  messages?: unknown;
  modelId?: unknown;
  /** Per-chat persona / system prompt (optional). */
  system?: unknown;
  /** Sampling temperature 0–2 (optional). */
  temperature?: unknown;
  /** Max output tokens 1–8192 (optional). */
  maxTokens?: unknown;
};

// Request-scoped Supabase client using the public (anon) key. With a user's
// bearer token it acts as that authenticated user; without one it's the anon
// role. The credit RPCs (consume_credit / consume_guest_credit) are granted to
// authenticated / anon, so this avoids needing the service-role key for chat.
function getRequestClient(token?: string) {
  const url = process.env.SUPABASE_URL || process.env.VITE_SUPABASE_URL;
  const anonKey = process.env.SUPABASE_PUBLISHABLE_KEY || process.env.VITE_SUPABASE_PUBLISHABLE_KEY;
  if (!url || !anonKey) throw new Error("Missing Supabase URL / publishable key");
  return createClient(url, anonKey, {
    global: token ? { headers: { Authorization: `Bearer ${token}` } } : {},
    auth: { persistSession: false, autoRefreshToken: false },
  });
}


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
          const sb = getRequestClient(token);
          const { data: userData, error: userErr } = await sb.auth.getUser(token);
          if (userErr || !userData.user) {
            return new Response("Unauthorized", { status: 401 });
          }
          userId = userData.user.id;
          const { data: r, error: creditErr } = await sb.rpc(
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
          const sb = getRequestClient();
          const { data: r, error: gErr } = await sb.rpc(
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

        const requestedId = typeof modelId === "string" ? modelId : DEFAULT_MODEL;

        // Optional per-chat generation controls — validated + clamped so a
        // crafted client can't push absurd values into the provider.
        const persona = typeof body.system === "string" ? body.system.trim().slice(0, 4000) : "";
        const temperature =
          typeof body.temperature === "number" && Number.isFinite(body.temperature)
            ? Math.min(2, Math.max(0, body.temperature))
            : undefined;
        const maxOutputTokens =
          typeof body.maxTokens === "number" && Number.isFinite(body.maxTokens)
            ? Math.min(8192, Math.max(1, Math.floor(body.maxTokens)))
            : undefined;
        const baseSystem =
          "You are TirthoAI, a friendly, capable multi-model AI assistant. " +
          "Respond clearly and use markdown (headings, lists, fenced code blocks) when it improves clarity. " +
          "If the user shares an image or file, refer to it naturally.";
        // A custom persona replaces the personality but we still nudge markdown
        // formatting so rendering stays consistent.
        const systemPrompt = persona
          ? `${persona}\n\nFormat responses with markdown (headings, lists, fenced code blocks) when it improves clarity.`
          : baseSystem;

        // Does the conversation carry any image content? Images can come from
        // the current turn OR from earlier in the history — and either way the
        // request can ONLY be served by a vision-capable model. A text follow-up
        // in a chat that already has an image still ships that image part to the
        // model, so we must check the whole message list, not just the last turn.
        const hasImageContent =
          messages.some(
            (m) =>
              Array.isArray((m as { parts?: unknown }).parts) &&
              (m as { parts: Array<{ type?: string; mediaType?: string }> }).parts.some(
                (p) =>
                  p?.type === "file" &&
                  typeof p?.mediaType === "string" &&
                  p.mediaType.startsWith("image/"),
              ),
          );

        let model;
        let chosen = DEFAULT_MODEL;
        let provider: "nvidia" | "anthropic" | "perplexity" | "groq" | "pollinations" | "openrouter" | "gemini" | "together" | "user" = "groq";

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
          // Ordered routes: the model's primary provider + its backups.
          const baseRoutes: ModelRoute[] = chosenConfig
            ? routesFor(chosenConfig)
            : [{ provider: "groq" as ModelProvider, id: requestedId }];
          // When the request carries image content, force a vision-capable
          // route. Image parts sent to a text-only model are rejected by the
          // provider (NVIDIA: "multimodal processing is not enabled"; some
          // OpenAI-compatible endpoints: "messages[N].content must be a
          // string"). So we drop the requested route entirely unless it is
          // itself a vision model, then fall through to the shared vision list.
          const explicit: ModelRoute[] = hasImageContent
            ? dedupeRoutes([
                ...(chosenConfig?.supportsVision ? baseRoutes : []),
                ...VISION_ROUTES,
              ])
            : baseRoutes;
          // Pick the first route whose provider is configured (primary preferred,
          // then backups). A keyless fallback (Pollinations) guarantees the
          // request always has a working route even if no keyed provider is set.
          // (Pollinations is text-only here, so it's skipped when images are
          // present — a vision route from VISION_ROUTES is used instead.)
          const route =
            explicit.find((r) => providerUsable(r.provider)) ??
            (hasImageContent
              ? (VISION_ROUTES.find((r) => providerUsable(r.provider)) ?? VISION_ROUTES[0])
              : ({ provider: "pollinations", id: "openai" } as ModelRoute));
          provider = route.provider;
          chosen = route.id;
          model = makeProviderModel(route.provider, route.id);
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

          if (provider === "gemini") {
            if (status === 401 || status === 403 || /API_KEY_INVALID|API key not valid/i.test(safe)) {
              return {
                status: 401,
                code: "gemini_unauthorized",
                message:
                  "Google rejected the Gemini API key. GEMINI_API_KEY is missing or invalid — generate a key at https://aistudio.google.com/apikey and set it on the server.",
              };
            }
            if (status === 404) {
              return { status: 404, code: "gemini_model_not_found", message: `Gemini returned 404 for model "${chosen}".` };
            }
            if (status === 429) {
              return { status: 429, code: "gemini_rate_limited", message: "Gemini rate limit / quota hit (429). Retry shortly." };
            }
          }

          if (status === 401 || status === 403) {
            return { status, code: "provider_unauthorized", message: "The AI provider rejected the request (check the API key)." };
          }
          if (status === 429) {
            return { status: 429, code: "rate_limited", message: "Rate limit hit. Retry shortly." };
          }
          return { status: 500, code: "ai_request_failed", message: safe.slice(0, 300) || "AI request failed" };
        };

        try {
          const result = streamText({
            model,
            system: systemPrompt,
            ...(temperature !== undefined ? { temperature } : {}),
            ...(maxOutputTokens !== undefined ? { maxOutputTokens } : {}),
            messages: await convertToModelMessages(
              (messages as UIMessage[]).map((m) => ({
                ...m,
                parts: m.parts?.filter((p) => p.type !== "reasoning"),
              }))
            ),
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
