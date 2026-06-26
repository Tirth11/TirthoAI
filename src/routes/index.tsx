import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState, useCallback, useSyncExternalStore, useRef } from "react";
import { toast } from "sonner";
import type { UIMessage } from "ai";
import { parseSharePayload } from "@/lib/chat-export";
import { Sidebar } from "@/components/Sidebar";
import { ChatWindow } from "@/components/ChatWindow";
import { AuthScreen } from "@/components/AuthScreen";
import { BrandedLoader } from "@/components/BrandedLoader";
import { ChatDB, type DBConversation } from "@/lib/chat-db";
import { DEFAULT_MODEL } from "@/lib/models";
import { ModelCache } from "@/lib/model-cache";
import { useAuthSession } from "@/hooks/use-auth";
import { supabase } from "@/integrations/supabase/client";
import { isGuestMode, enterGuestMode, exitGuestMode } from "@/lib/guest";


export const Route = createFileRoute("/")({
  component: Index,
  ssr: false,
  head: () => ({
    meta: [
      { property: "og:url", content: "https://tirthoai.app/" },
    ],
    links: [{ rel: "canonical", href: "https://tirthoai.app/" }],
  }),
});

function subscribeGuest(cb: () => void) {
  window.addEventListener("guest-mode-changed", cb);
  return () => window.removeEventListener("guest-mode-changed", cb);
}

function Index() {
  const { session, loading: authLoading } = useAuthSession();
  const guest = useSyncExternalStore(
    subscribeGuest,
    () => isGuestMode(),
    () => false,
  );
  const [authMode, setAuthMode] = useState<"signin" | "signup">("signup");
  const [forceAuth, setForceAuth] = useState(false);

  if (authLoading) {
    return <BrandedLoader label="Checking your session…" />;
  }

  if (session) {
    if (guest) exitGuestMode();
    // Key by user id so switching accounts fully remounts (loads that user's
    // own conversations from the DB; no state leak between users).
    return (
      <ChatLayout
        key={session.user.id}
        userEmail={session.user.email ?? "User"}
        userId={session.user.id}
      />
    );
  }

  if (guest && !forceAuth) {
    return (
      <GuestLayout
        onGoToAuth={(mode) => {
          setAuthMode(mode);
          setForceAuth(true);
        }}
      />
    );
  }

  return (
    <AuthScreen
      initialMode={authMode}
      onContinueAsGuest={() => {
        enterGuestMode();
        setForceAuth(false);
      }}
    />
  );
}

function GuestLayout({ onGoToAuth }: { onGoToAuth: (mode: "signin" | "signup") => void }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const conversation: DBConversation = {
    id: "guest",
    title: "Guest chat",
    category: "general",
    model_id: DEFAULT_MODEL,
    model_updated_at: new Date().toISOString(),
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
  // Guests can chat in this single free conversation; any action that needs
  // persistence (new chat, rename, delete, import, history) nudges them to sign up.
  const promptSignup = () => {
    setSidebarOpen(false);
    onGoToAuth("signup");
  };
  return (
    <div className="flex h-dvh min-h-0 overflow-hidden bg-background text-foreground">
      <Sidebar
        conversations={[conversation]}
        activeId="guest"
        onSelect={() => setSidebarOpen(false)}
        onNew={promptSignup}
        onImport={promptSignup}
        onDelete={promptSignup}
        onRename={promptSignup}
        userEmail="Guest"
        userId="guest"
        onSignOut={() => onGoToAuth("signin")}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <main className="h-full min-h-0 min-w-0 flex-1 overflow-hidden">
        <ChatWindow
          key="guest"
          conversation={conversation}
          onConversationChange={() => {}}
          onOpenSidebar={() => setSidebarOpen(true)}
          userEmail="Guest"
          userId="guest"
          guest
          onGuestSignUp={() => onGoToAuth("signup")}
          onGuestSignIn={() => onGoToAuth("signin")}
        />
      </main>
    </div>
  );
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
        const created = await ChatDB.createConversation(ModelCache.getLast() ?? DEFAULT_MODEL);
        list = [created];
        setConversations(list);
      }
      setActiveId(list[0].id);
      setReady(true);
    })();
  }, [refresh]);

  const handleNew = async () => {
    const created = await ChatDB.createConversation(ModelCache.getLast() ?? DEFAULT_MODEL);
    await refresh();
    setActiveId(created.id);
    setSidebarOpen(false);
  };

  // Import a chat that someone shared as a .tirthoai.json file.
  const importInputRef = useRef<HTMLInputElement>(null);
  const handleImportFile = async (file: File) => {
    try {
      const payload = parseSharePayload(await file.text());
      const created = await ChatDB.createConversation(
        payload.model_id || ModelCache.getLast() || DEFAULT_MODEL,
        payload.category || "general",
      );
      await ChatDB.updateConversation(created.id, { title: payload.title || "Imported chat" });
      for (const m of payload.messages) {
        const msg = {
          id: (globalThis.crypto?.randomUUID?.() ?? `imp_${Math.random().toString(36).slice(2)}`),
          role: m.role,
          parts: m.parts,
        } as UIMessage;
        await ChatDB.insertMessage(created.id, msg);
      }
      await refresh();
      setActiveId(created.id);
      setSidebarOpen(false);
      toast.success("Chat imported");
    } catch (e) {
      console.error(e);
      toast.error(e instanceof Error ? e.message : "Could not import that file");
    }
  };

  const handleSelect = (id: string) => {
    setActiveId(id);
    setSidebarOpen(false);
  };

  // Open a model's compare result as a new chat thread (seeded with the prompt +
  // that model's answer) so the user can continue one-on-one with that model.
  const handleOpenInChat = async (modelId: string, userText: string, assistantText: string) => {
    const conv = await ChatDB.createConversation(modelId);
    const title = userText.trim().replace(/\s+/g, " ").slice(0, 40) || "Compared chat";
    await ChatDB.updateConversation(conv.id, { title });
    await ChatDB.insertMessage(conv.id, {
      id: crypto.randomUUID(),
      role: "user",
      parts: [{ type: "text", text: userText }],
    } as UIMessage);
    await ChatDB.insertMessage(conv.id, {
      id: crypto.randomUUID(),
      role: "assistant",
      parts: [{ type: "text", text: assistantText }],
    } as UIMessage);
    await refresh();
    setActiveId(conv.id);
    setSidebarOpen(false);
  };

  const handleDelete = async (id: string) => {
    await ChatDB.deleteConversation(id);
    const list = await refresh();
    if (list.length === 0) {
      const created = await ChatDB.createConversation(ModelCache.getLast() ?? DEFAULT_MODEL);
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

  if (typeof document !== "undefined" && active) {
    document.title = `${active.title} — TirthoAI`;
  }

  return (
    <div className="flex h-dvh min-h-0 overflow-hidden bg-background text-foreground">
      <input
        ref={importInputRef}
        type="file"
        accept=".json,application/json"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleImportFile(f);
          e.target.value = "";
        }}
      />
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={handleSelect}
        onNew={handleNew}
        onImport={() => importInputRef.current?.click()}
        onDelete={handleDelete}
        onRename={handleRename}
        userEmail={userEmail}
        userId={userId}
        onSignOut={handleSignOut}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <main className="h-full min-h-0 min-w-0 flex-1 overflow-hidden">
        {active && (
          <ChatWindow
            key={active.id}
            conversation={active}
            onConversationChange={refresh}
            onOpenSidebar={() => setSidebarOpen(true)}
            userEmail={userEmail}
            userId={userId}
            onOpenInChat={handleOpenInChat}
          />
        )}
      </main>
    </div>
  );
}
