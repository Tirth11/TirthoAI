import { supabase } from "@/integrations/supabase/client";
import type { UIMessage } from "ai";

export interface DBConversation {
  id: string;
  title: string;
  category: string;
  model_id: string;
  created_at: string;
  updated_at: string;
}

export interface DBMessage {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  parts: UIMessage["parts"];
  created_at: string;
}

export const ChatDB = {
  async listConversations(): Promise<DBConversation[]> {
    const { data, error } = await supabase
      .from("conversations")
      .select("*")
      .order("updated_at", { ascending: false });
    if (error) throw error;
    return (data ?? []) as DBConversation[];
  },

  async createConversation(modelId: string, category = "general"): Promise<DBConversation> {
    const { data: userRes } = await supabase.auth.getUser();
    const userId = userRes.user?.id;
    if (!userId) throw new Error("Not signed in");
    const { data, error } = await supabase
      .from("conversations")
      .insert({ user_id: userId, model_id: modelId, category, title: "New Chat" })
      .select()
      .single();
    if (error) throw error;
    return data as DBConversation;
  },

  async updateConversation(id: string, patch: Partial<Pick<DBConversation, "title" | "model_id" | "category">>) {
    const { error } = await supabase.from("conversations").update(patch).eq("id", id);
    if (error) throw error;
  },

  async deleteConversation(id: string) {
    const { error } = await supabase.from("conversations").delete().eq("id", id);
    if (error) throw error;
  },

  async listMessages(conversationId: string): Promise<UIMessage[]> {
    const { data, error } = await supabase
      .from("messages")
      .select("*")
      .eq("conversation_id", conversationId)
      .order("created_at", { ascending: true });
    if (error) throw error;
    return (data ?? []).map((m) => ({
      id: m.id,
      role: m.role as UIMessage["role"],
      parts: (m.parts as unknown as UIMessage["parts"]) ?? [],
    })) as UIMessage[];
  },

  async insertMessage(conversationId: string, message: UIMessage) {
    const { data: userRes } = await supabase.auth.getUser();
    const userId = userRes.user?.id;
    if (!userId) return;
    const { error } = await supabase.from("messages").insert({
      id: message.id,
      conversation_id: conversationId,
      user_id: userId,
      role: message.role,
      parts: message.parts as unknown as never,
    });
    if (error) console.error("insertMessage", error);
  },
};
