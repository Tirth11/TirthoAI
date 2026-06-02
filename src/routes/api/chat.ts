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
          // Prefer a per-user override stored in user_api_keys, fall back to env.
          const { data: keyRow } = await supabaseAdmin
            .from("user_api_keys")
            .select("nvidia_api_key")
            .eq("user_id", userId)
            .maybeSingle();
          const nvKey = keyRow?.nvidia_api_key?.trim() || process.env.NVIDIA_API_KEY;
          if (!nvKey) {
            return new Response(
              JSON.stringify({ error: "missing_nvidia_key", message: "NVIDIA_API_KEY is not configured. Set one in Settings." }),
              { status: 500, headers: { "Content-Type": "application/json" } },
            );
          }
          model = createNvidiaProvider(nvKey)(chosen);
        } else {
          model = createLovableAiGatewayProvider(key)(chosen);
        }

        try {
          const result = streamText({
            model,
            system:
              "You are TirthoAI, a friendly, capable multi-model AI assistant. " +
              "Respond clearly and use markdown (headings, lists, fenced code blocks) when it improves clarity. " +
              "If the user shares an image or file, refer to it naturally.",
            messages: await convertToModelMessages(messages as UIMessage[]),
          });

          return result.toUIMessageStreamResponse({
            originalMessages: messages as UIMessage[],
            headers: { "x-credits-remaining": String(remaining ?? "") },
          });
        } catch (err) {
          const message = err instanceof Error ? err.message : "AI request failed";
          return new Response(message, { status: 500 });
        }
      },
    },
  },
});
