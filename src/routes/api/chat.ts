import { createLovableAiGatewayProvider } from "@/lib/ai-gateway.server";
import { DEFAULT_MODEL, getModelById } from "@/lib/models";
import { createFileRoute } from "@tanstack/react-router";
import { convertToModelMessages, streamText, type UIMessage } from "ai";

type ChatRequestBody = { messages?: unknown; modelId?: unknown };

export const Route = createFileRoute("/api/chat")({
  server: {
    handlers: {
      POST: async ({ request }) => {
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
        const chosen = getModelById(requestedId)?.id ?? DEFAULT_MODEL;

        const gateway = createLovableAiGatewayProvider(key);
        const model = gateway(chosen);

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
          });
        } catch (err) {
          const message = err instanceof Error ? err.message : "AI request failed";
          return new Response(message, { status: 500 });
        }
      },
    },
  },
});
