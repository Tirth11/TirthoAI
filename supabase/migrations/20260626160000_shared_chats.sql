-- Read-only shareable chats.
-- A "share" is an immutable SNAPSHOT of a conversation at share time. We do NOT
-- expose the private conversations/messages tables publicly — instead the owner
-- copies a snapshot into shared_chats, which is the only thing the public can read.
-- Recipients can VIEW a shared chat by its id (the link) but can never write to it
-- or prompt/continue it (no UPDATE policy, no access to the live tables).

CREATE TABLE public.shared_chats (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  conversation_id UUID,                 -- soft reference (no FK: a share outlives its source chat)
  title TEXT NOT NULL DEFAULT 'Shared chat',
  category TEXT,
  model_id TEXT,
  messages JSONB NOT NULL DEFAULT '[]'::jsonb,   -- snapshot: [{ role, parts }]
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX shared_chats_user_id_created_at_idx
  ON public.shared_chats(user_id, created_at DESC);

-- anon + authenticated may READ a shared chat (by its link/id). Only owners write.
GRANT SELECT ON public.shared_chats TO anon, authenticated;
GRANT INSERT, DELETE ON public.shared_chats TO authenticated;
GRANT ALL ON public.shared_chats TO service_role;

ALTER TABLE public.shared_chats ENABLE ROW LEVEL SECURITY;

-- Anyone with the link can view it — READ ONLY. This is the only public surface.
CREATE POLICY "Anyone can view shared chats"
  ON public.shared_chats FOR SELECT TO anon, authenticated
  USING (true);

-- Only the signed-in owner can create a share, and only for their own user_id.
CREATE POLICY "Users can create own shares"
  ON public.shared_chats FOR INSERT TO authenticated
  WITH CHECK (auth.uid() = user_id);

-- Only the owner can revoke (delete) their share. NOTE: there is intentionally NO
-- UPDATE policy — snapshots are immutable, so a shared link can never be edited or
-- turned into a live, promptable conversation.
CREATE POLICY "Users can delete own shares"
  ON public.shared_chats FOR DELETE TO authenticated
  USING (auth.uid() = user_id);
