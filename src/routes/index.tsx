import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState, useCallback } from "react";
import { Sidebar } from "@/components/Sidebar";
import { ChatWindow } from "@/components/ChatWindow";
import { AuthScreen } from "@/components/AuthScreen";
import { BrandedLoader } from "@/components/BrandedLoader";
import { ChatDB, type DBConversation } from "@/lib/chat-db";
import { DEFAULT_MODEL } from "@/lib/models";
import { useAuthSession } from "@/hooks/use-auth";
import { supabase } from "@/integrations/supabase/client";

export const Route = createFileRoute("/")({
  component: Index,
  ssr: false,
});

function Index() {
  const { session, loading: authLoading } = useAuthSession();

  if (authLoading) {
    return <BrandedLoader label="Checking your session…" />;
  }

  if (!session) {
    return <AuthScreen />;
  }

  return <ChatLayout userEmail={session.user.email ?? "User"} userId={session.user.id} />;
}

function ChatLayout({ userEmail, userId }: { userEmail: string; userId: string }) {
  const [conversations, setConversations] = useState<DBConversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

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
    setSidebarOpen(false);
  };

  const handleSelect = (id: string) => {
    setActiveId(id);
    setSidebarOpen(false);
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
    return <BrandedLoader label="Loading your conversations…" />;
  }

  const active = conversations.find((c) => c.id === activeId) ?? conversations[0];

  // Update document title with active conversation
  if (typeof document !== "undefined" && active) {
    document.title = `${active.title} — TirthoAI`;
  }

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      {/* Sidebar — fixed drawer on mobile, static column on desktop */}
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={handleSelect}
        onNew={handleNew}
        onDelete={handleDelete}
        onRename={handleRename}
        userEmail={userEmail}
        userId={userId}
        onSignOut={handleSignOut}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <main className="flex-1 min-w-0">
        {active && (
          <ChatWindow
            key={active.id}
            conversation={active}
            onConversationChange={refresh}
            onOpenSidebar={() => setSidebarOpen(true)}
            userEmail={userEmail}
            userId={userId}
          />
        )}
      </main>
    </div>
  );
}
