import { createFileRoute, useNavigate, useParams } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Sparkles, Loader2, Eye, ArrowRight } from "lucide-react";
import { fetchSharedChat, type SharedChatRow } from "@/lib/shared-chat";
import { markdownToHtml } from "@/lib/chat-export";

export const Route = createFileRoute("/share/$id")({
  component: SharedChatPage,
  ssr: false,
  head: () => ({
    meta: [
      { title: "Shared chat — TirthoAI" },
      { name: "description", content: "A read-only conversation shared from TirthoAI." },
      { property: "og:title", content: "Shared chat — TirthoAI" },
      { property: "og:description", content: "View a conversation shared from TirthoAI." },
      // Shared links are unlisted — don't index them.
      { name: "robots", content: "noindex" },
    ],
  }),
});

const textOf = (m: SharedChatRow["messages"][number]) =>
  (m.parts ?? []).map((p) => (p.type === "text" ? p.text : "")).join("");

function SharedChatPage() {
  const navigate = useNavigate();
  const { id } = useParams({ from: "/share/$id" });
  const [loading, setLoading] = useState(true);
  const [chat, setChat] = useState<SharedChatRow | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    fetchSharedChat(id)
      .then((row) => {
        if (!alive) return;
        if (!row) setError("This shared chat doesn't exist or was removed.");
        else setChat(row);
      })
      .catch((e) => alive && setError(e instanceof Error ? e.message : "Could not load this shared chat."))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [id]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error || !chat) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-background px-4 text-center">
        <p className="text-sm text-foreground">{error || "Shared chat not found."}</p>
        <button
          onClick={() => navigate({ to: "/" })}
          className="rounded-lg px-4 py-2 text-sm font-semibold text-primary-foreground"
          style={{ background: "var(--gradient-primary)" }}
        >
          Go to TirthoAI
        </button>
      </div>
    );
  }

  const messages = (chat.messages ?? []).filter((m) => m.role !== "system");

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center gap-3 px-4 py-3">
          <div
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-white"
            style={{ background: "var(--gradient-primary)" }}
          >
            <Sparkles className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-sm font-bold tracking-tight text-foreground">{chat.title}</h1>
            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground">
              <Eye className="h-3 w-3" /> Read-only shared chat
            </span>
          </div>
          <button
            onClick={() => navigate({ to: "/" })}
            className="inline-flex shrink-0 items-center gap-1 rounded-full px-3 py-1.5 text-xs font-semibold text-primary-foreground"
            style={{ background: "var(--gradient-primary)" }}
          >
            Open TirthoAI <ArrowRight className="h-3 w-3" />
          </button>
        </div>
      </header>

      {/* Messages — read only, no composer */}
      <main className="mx-auto max-w-3xl px-4 py-6">
        <div className="space-y-4">
          {messages.map((m, i) => {
            const isUser = m.role === "user";
            return (
              <div key={i} className={isUser ? "flex justify-end" : "flex justify-start"}>
                <div
                  className={
                    isUser
                      ? "max-w-[85%] rounded-2xl rounded-br-sm bg-primary/10 px-4 py-3 text-sm text-foreground"
                      : "max-w-[95%] rounded-2xl rounded-bl-sm border border-border bg-card px-4 py-3 text-sm text-foreground"
                  }
                >
                  <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                    {isUser ? "You" : "Assistant"}
                  </div>
                  <div
                    className="prose prose-sm max-w-none dark:prose-invert prose-pre:overflow-auto"
                    dangerouslySetInnerHTML={{ __html: markdownToHtml(textOf(m)) }}
                  />
                </div>
              </div>
            );
          })}
          {messages.length === 0 && (
            <p className="py-12 text-center text-sm text-muted-foreground">This shared chat has no messages.</p>
          )}
        </div>

        {/* Footer CTA — read-only notice + invite to start their own */}
        <div className="mt-10 flex flex-col items-center gap-3 rounded-2xl border border-border bg-card p-6 text-center">
          <p className="text-sm font-medium text-foreground">This is a read-only snapshot.</p>
          <p className="text-xs text-muted-foreground">
            You can't reply to or continue this conversation. Start your own to chat with multiple AI models.
          </p>
          <button
            onClick={() => navigate({ to: "/" })}
            className="inline-flex items-center gap-1.5 rounded-lg px-4 py-2 text-sm font-semibold text-primary-foreground shadow-md transition hover:opacity-95"
            style={{ background: "var(--gradient-primary)", boxShadow: "var(--shadow-glow)" }}
          >
            <Sparkles className="h-4 w-4" /> Start your own chat
          </button>
        </div>
      </main>
    </div>
  );
}
