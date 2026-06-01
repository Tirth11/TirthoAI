import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState, useCallback } from "react";
import { Toaster } from "sonner";
import { Loader2 } from "lucide-react";
import { Sidebar } from "@/components/Sidebar";
import { ChatWindow } from "@/components/ChatWindow";
import { AuthScreen } from "@/components/AuthScreen";
import { ChatDB, type DBConversation } from "@/lib/chat-db";
import { DEFAULT_MODEL } from "@/lib/models";
import { useAuthSession } from "@/hooks/use-auth";
import { supabase } from "@/integrations/supabase/client";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "TirthoAI — Multi-Model AI Platform" },
      {
        name: "description",
        content:
          "TirthoAI is a multi-model AI chat platform — reasoning, coding, vision, and creative models in one place, with persistent chat history.",
      },
      { property: "og:title", content: "TirthoAI — Multi-Model AI Platform" },
      {
        property: "og:description",
        content: "Chat with the best AI models. Your history is saved automatically.",
      },
    ],
  }),
  component: Index,
  ssr: false,
});

function Index() {
  const { session, loading: authLoading } = useAuthSession();

  if (authLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
        <Toaster position="top-center" richColors theme="dark" />
      </div>
    );
  }

  if (!session) {
    return (
      <>
        <AuthScreen />
        <Toaster position="top-center" richColors theme="dark" />
      </>
    );
  }

  return <ChatLayout userEmail={session.user.email ?? "User"} />;
}

function ChatLayout({ userEmail }: { userEmail: string }) {
  const [conversations, setConversations] = useState<DBConversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const list = await ChatDB.listConversations();
      setConversations(list);
      return list;
    } catch (e) {
      console.error(e);
      return [];
    }
  }, []);

  useEffect(() => {
    (async () => {
      let list = await refresh();
      if (list.length === 0) {
        const created = await ChatDB.createConversation(DEFAULT_MODEL);
        list = [created];
        setConversations(list);
      }
      setActiveId(list[0].id);
      setReady(true);
    })();
  }, [refresh]);

  const handleNew = async () => {
    const created = await ChatDB.createConversation(DEFAULT_MODEL);
    await refresh();
    setActiveId(created.id);
  };

  const handleDelete = async (id: string) => {
    await ChatDB.deleteConversation(id);
    const list = await refresh();
    if (list.length === 0) {
      const created = await ChatDB.createConversation(DEFAULT_MODEL);
      setConversations([created]);
      setActiveId(created.id);
    } else if (activeId === id) {
      setActiveId(list[0].id);
    }
  };

  const handleRename = async (id: string, title: string) => {
    await ChatDB.updateConversation(id, { title });
    await refresh();
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
  };

  if (!ready) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
        <Toaster position="top-center" richColors theme="dark" />
      </div>
    );
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
        userEmail={userEmail}
        onSignOut={handleSignOut}
      />
      <main className="flex-1 min-w-0">
        {active && (
          <ChatWindow
            key={active.id}
            conversation={active}
            onConversationChange={refresh}
          />
        )}
      </main>
      <Toaster position="top-center" richColors theme="dark" />
    </div>
  );
}
