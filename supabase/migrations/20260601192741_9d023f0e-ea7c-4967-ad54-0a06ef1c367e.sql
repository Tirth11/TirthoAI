
-- Credit event log
CREATE TABLE public.credit_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL,
  event_type TEXT NOT NULL CHECK (event_type IN ('grant','debit','refund','bonus')),
  amount INTEGER NOT NULL,
  balance_after INTEGER NOT NULL,
  reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_credit_events_user_created ON public.credit_events (user_id, created_at DESC);

GRANT SELECT ON public.credit_events TO authenticated;
GRANT ALL ON public.credit_events TO service_role;

ALTER TABLE public.credit_events ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own credit events"
ON public.credit_events FOR SELECT
TO authenticated
USING (auth.uid() = user_id);

-- Log signup grant
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

  INSERT INTO public.credit_events (user_id, event_type, amount, balance_after, reason)
  VALUES (NEW.id, 'grant', 500, 500, 'Welcome bonus — 500 free credits');

  RETURN NEW;
END;
$$;

-- Log each debit
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

  INSERT INTO public.credit_events (user_id, event_type, amount, balance_after, reason)
  VALUES (_user_id, 'debit', -1, remaining, 'AI message');

  RETURN remaining;
END;
$$;

REVOKE EXECUTE ON FUNCTION public.handle_new_user_credits() FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.consume_credit(UUID) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.consume_credit(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION public.handle_new_user_credits() TO service_role;

-- Backfill grant event for existing users who don't have one
INSERT INTO public.credit_events (user_id, event_type, amount, balance_after, reason)
SELECT uc.user_id, 'grant', 500, uc.credits + uc.total_used, 'Welcome bonus — 500 free credits'
FROM public.user_credits uc
WHERE NOT EXISTS (
  SELECT 1 FROM public.credit_events ce
  WHERE ce.user_id = uc.user_id AND ce.event_type = 'grant'
);
