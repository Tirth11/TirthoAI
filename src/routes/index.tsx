import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState, useCallback } from "react";
import { Toaster } from "sonner";
import { Sidebar } from "@/components/Sidebar";
import { ChatWindow } from "@/components/ChatWindow";
import { ConvStore, type Conversation } from "@/lib/conversations";
import { DEFAULT_MODEL } from "@/lib/models";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "TirthoAI — Multi-Model AI Platform" },
      {
        name: "description",
        content:
          "TirthoAI is a free multi-model AI chat platform — reasoning, coding, vision, and creative models in one place, powered by Lovable AI.",
      },
      { property: "og:title", content: "TirthoAI — Multi-Model AI Platform" },
      {
        property: "og:description",
        content: "Chat with the best AI models, auto-picked for your prompt.",
      },
    ],
  }),
  component: Index,
  ssr: false,
});

function Index() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);

  const refresh = useCallback(() => {
    setConversations(ConvStore.list());
  }, []);

  useEffect(() => {
    const list = ConvStore.list();
    if (list.length === 0) {
      const created = ConvStore.create(DEFAULT_MODEL);
      setConversations([created]);
      setActiveId(created.id);
    } else {
      setConversations(list);
      setActiveId(list[0].id);
    }
    setHydrated(true);
  }, []);

  const handleNew = () => {
    const created = ConvStore.create(DEFAULT_MODEL);
    setConversations(ConvStore.list());
    setActiveId(created.id);
  };

  const handleDelete = (id: string) => {
    ConvStore.delete(id);
    const list = ConvStore.list();
    if (list.length === 0) {
      const created = ConvStore.create(DEFAULT_MODEL);
      setConversations([created]);
      setActiveId(created.id);
    } else {
      setConversations(list);
      if (activeId === id) setActiveId(list[0].id);
    }
  };

  const handleRename = (id: string, title: string) => {
    ConvStore.rename(id, title);
    refresh();
  };

  if (!hydrated) {
    return <div className="flex h-screen items-center justify-center bg-background" />;
  }

  const active = conversations.find((c) => c.id === activeId) ?? conversations[0];

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={handleNew}
        onDelete={handleDelete}
        onRename={handleRename}
      />
      <main className="flex-1 min-w-0">
        {active && <ChatWindow key={active.id} conversation={active} onUpdate={refresh} />}
      </main>
      <Toaster position="top-center" richColors theme="dark" />
    </div>
  );
}
