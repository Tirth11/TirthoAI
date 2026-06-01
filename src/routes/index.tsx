import { createFileRoute } from "@tanstack/react-router";
import { ChatWindow } from "@/components/ChatWindow";
import { Sparkles } from "lucide-react";
import { Toaster } from "sonner";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "TirthoAI — Your AI Chat Companion" },
      {
        name: "description",
        content: "TirthoAI is a fast, friendly AI assistant powered by Lovable AI.",
      },
      { property: "og:title", content: "TirthoAI — Your AI Chat Companion" },
      {
        property: "og:description",
        content: "Chat with TirthoAI, a fast, friendly AI assistant.",
      },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <div className="flex h-screen flex-col bg-background">
      <header className="border-b border-border bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center gap-2 px-4 py-3 sm:px-6">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary/60 shadow-md shadow-primary/20">
            <Sparkles className="h-4 w-4 text-primary-foreground" />
          </div>
          <h1 className="text-base font-semibold tracking-tight">TirthoAI</h1>
          <span className="ml-auto text-xs text-muted-foreground">Lovable AI</span>
        </div>
      </header>
      <main className="flex-1 overflow-hidden">
        <ChatWindow />
      </main>
      <Toaster position="top-center" richColors />
    </div>
  );
}
