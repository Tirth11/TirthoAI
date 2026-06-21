ALTER TABLE public.conversations
  ADD COLUMN IF NOT EXISTS model_updated_at timestamptz NOT NULL DEFAULT now();