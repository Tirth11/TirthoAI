
CREATE TABLE public.user_credits (
  user_id UUID PRIMARY KEY,
  credits INTEGER NOT NULL DEFAULT 500,
  total_used INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

GRANT SELECT ON public.user_credits TO authenticated;
GRANT ALL ON public.user_credits TO service_role;

ALTER TABLE public.user_credits ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own credits"
ON public.user_credits FOR SELECT
TO authenticated
USING (auth.uid() = user_id);

-- Auto-create credits row on new user signup
CREATE OR REPLACE FUNCTION public.handle_new_user_credits()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  INSERT INTO public.user_credits (user_id, credits)
  VALUES (NEW.id, 500)
  ON CONFLICT (user_id) DO NOTHING;
  RETURN NEW;
END;
$$;

CREATE TRIGGER on_auth_user_created_credits
AFTER INSERT ON auth.users
FOR EACH ROW
EXECUTE FUNCTION public.handle_new_user_credits();

-- Atomic decrement; returns remaining credits or -1 if insufficient
CREATE OR REPLACE FUNCTION public.consume_credit(_user_id UUID)
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  remaining INTEGER;
BEGIN
  INSERT INTO public.user_credits (user_id, credits)
  VALUES (_user_id, 500)
  ON CONFLICT (user_id) DO NOTHING;

  UPDATE public.user_credits
  SET credits = credits - 1,
      total_used = total_used + 1,
      updated_at = now()
  WHERE user_id = _user_id AND credits > 0
  RETURNING credits INTO remaining;

  IF remaining IS NULL THEN
    RETURN -1;
  END IF;
  RETURN remaining;
END;
$$;
