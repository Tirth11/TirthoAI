import { providerUsable, makeProviderModel } from "@/lib/ai-gateway.server";
import { DEFAULT_MODEL, getModelById, routesFor, type ModelProvider } from "@/lib/models";
import { createClient } from "@supabase/supabase-js";
import { createFileRoute } from "@tanstack/react-router";
import { generateText, convertToModelMessages, type UIMessage } from "ai";

// Request-scoped Supabase client (anon key + user token) — avoids needing the
// service-role key. consume_credit is granted to the authenticated role.
function getRequestClient(token: string) {
  const url = process.env.SUPABASE_URL || process.env.VITE_SUPABASE_URL;
  const anonKey = process.env.SUPABASE_PUBLISHABLE_KEY || process.env.VITE_SUPABASE_PUBLISHABLE_KEY;
  if (!url || !anonKey) throw new Error("Missing Supabase URL / publishable key");
  return createClient(url, anonKey, {
    global: { headers: { Authorization: `Bearer ${token}` } },
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

type CompareBody = { messages?: unknown; modelIds?: unknown };

export type CompareResult = {
  modelId: string;
  label: string;
  ok: boolean;
  text?: string;
  error?: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  latencyMs: number;
};

const MAX_MODELS = 4;

export const Route = createFileRoute("/api/chat/compare")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const authHeader = request.headers.get("authorization") ?? "";
        const token = authHeader.toLowerCase().startsWith("bearer ")
          ? authHeader.slice(7).trim()
          : "";
        if (!token) return new Response("Unauthorized", { status: 401 });

        const sb = getRequestClient(token);
        const { data: userData, error: userErr } = await sb.auth.getUser(token);
        if (userErr || !userData.user) return new Response("Unauthorized", { status: 401 });
        const userId = userData.user.id;

        const body = (await request.json()) as CompareBody;
        const { messages, modelIds } = body;
        if (!Array.isArray(messages) || !Array.isArray(modelIds) || modelIds.length === 0) {
          return new Response("messages and modelIds required", { status: 400 });
        }
        const ids = (modelIds as unknown[])
          .filter((x): x is string => typeof x === "string")
          .slice(0, MAX_MODELS);
        if (ids.length === 0) return new Response("no valid models", { status: 400 });

        // Charge 1 credit per model fanned out
        let remaining: number | null = null;
        for (let i = 0; i < ids.length; i++) {
          const { data: r, error } = await sb.rpc("consume_credit", {
            _user_id: userId,
          });
          if (error) return new Response("credit check failed", { status: 500 });
          if (typeof r === "number" && r < 0) {
            return new Response(
              JSON.stringify({
                error: "out_of_credits",
                message: `You need ${ids.length} credits to compare ${ids.length} models.`,
              }),
              { status: 402, headers: { "Content-Type": "application/json" } },
            );
          }
          if (typeof r === "number") remaining = r;
        }

        const modelMessages = await convertToModelMessages(messages as UIMessage[]);

        const runOne = async (mid: string): Promise<CompareResult> => {
          const started = Date.now();
          const cfg = getModelById(mid);
          const label = cfg?.label ?? mid;
          // Try the model's routes (primary + backups) + a keyless last resort,
          // in order, until one succeeds — so a down provider doesn't fail the card.
          const routes = [
            ...(cfg ? routesFor(cfg) : [{ provider: "groq" as ModelProvider, id: mid }]),
            { provider: "pollinations" as ModelProvider, id: "openai" },
          ].filter((r) => providerUsable(r.provider));

          let lastErr = "No provider configured";
          for (const route of routes) {
            try {
              const model = makeProviderModel(route.provider, route.id);
              const result = await generateText({
                model,
                system:
                  "You are TirthoAI, a friendly multi-model AI assistant. Respond clearly using markdown when helpful.",
                messages: modelMessages,
              });
              const u = (result.usage ?? {}) as unknown as Record<string, number | undefined>;
              const pt = Number(u.inputTokens ?? u.promptTokens ?? 0) || 0;
              const ct = Number(u.outputTokens ?? u.completionTokens ?? 0) || 0;
              const tt = Number(u.totalTokens ?? pt + ct) || pt + ct;
              return {
                modelId: mid,
                label,
                ok: true,
                text: result.text,
                usage: { promptTokens: pt, completionTokens: ct, totalTokens: tt },
                latencyMs: Date.now() - started,
              };
            } catch (err) {
              lastErr = (err instanceof Error ? err.message : String(err))
                .replace(/Bearer\s+[A-Za-z0-9._-]+/gi, "Bearer ***")
                .replace(/sk-[A-Za-z0-9_-]+/g, "sk-***")
                .slice(0, 300);
              // fall through to the next route
            }
          }
          return { modelId: mid, label, ok: false, error: lastErr, latencyMs: Date.now() - started };
        };

        const results = await Promise.all(ids.map(runOne));

        return new Response(
          JSON.stringify({ results, creditsRemaining: remaining, defaultModel: DEFAULT_MODEL }),
          {
            status: 200,
            headers: {
              "Content-Type": "application/json",
              "x-credits-remaining": String(remaining ?? ""),
            },
          },
        );
      },
    },
  },
});
