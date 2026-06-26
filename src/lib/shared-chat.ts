// Read-only chat sharing.
// createShareLink() snapshots the current conversation into the public `shared_chats`
// table and returns a link. fetchSharedChat() loads a snapshot for the public
// /share/:id view. The snapshot is immutable and the public route is view-only, so
// a shared link can be read but never prompted/continued (see the SQL migration).
import { supabase } from "@/integrations/supabase/client";
import type { UIMessage } from "ai";

export interface SharedChatRow {
  id: string;
  title: string;
  category: string | null;
  model_id: string | null;
  messages: Array<{ role: UIMessage["role"]; parts: UIMessage["parts"] }>;
  created_at: string;
}

// `shared_chats` isn't in the generated Supabase types yet; use a loosely-typed
// handle so the rest of the app stays type-clean without editing generated files.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const sharedChats = () => (supabase as any).from("shared_chats");

export async function createShareLink(args: {
  conversationId?: string;
  title: string;
  category?: string;
  modelId?: string;
  messages: UIMessage[];
}): Promise<string> {
  const { data: userRes } = await supabase.auth.getUser();
  const userId = userRes.user?.id;
  if (!userId) throw new Error("Sign in to share a chat.");
  if (!args.messages || args.messages.length === 0) throw new Error("Nothing to share yet.");

  const snapshot = args.messages.map((m) => ({ role: m.role, parts: m.parts }));
  const { data, error } = await sharedChats()
    .insert({
      user_id: userId,
      conversation_id: args.conversationId ?? null,
      title: args.title || "Shared chat",
      category: args.category ?? null,
      model_id: args.modelId ?? null,
      messages: snapshot,
    })
    .select("id")
    .single();
  if (error) {
    // Surface the real Postgres/Supabase reason (e.g. table not migrated yet).
    const hint = /relation .* does not exist|shared_chats/i.test(error.message || "")
      ? " — run the shared_chats migration on your Supabase project."
      : "";
    throw new Error(`${error.message || "Share failed"}${hint}`);
  }
  return `${window.location.origin}/share/${data.id}`;
}

export async function fetchSharedChat(id: string): Promise<SharedChatRow | null> {
  const { data, error } = await sharedChats()
    .select("id,title,category,model_id,messages,created_at")
    .eq("id", id)
    .maybeSingle();
  if (error) throw error;
  return (data as SharedChatRow) ?? null;
}
