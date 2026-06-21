
CREATE TABLE public.guest_usage (
  guest_id TEXT PRIMARY KEY,
  credits_used INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

GRANT ALL ON public.guest_usage TO service_role;
ALTER TABLE public.guest_usage ENABLE ROW LEVEL SECURITY;

CREATE OR REPLACE FUNCTION public.consume_guest_credit(_guest_id TEXT, _limit INTEGER DEFAULT 50)
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  used INTEGER;
BEGIN
  IF _guest_id IS NULL OR length(_guest_id) < 8 OR length(_guest_id) > 128 THEN
    RETURN -1;
  END IF;

  INSERT INTO public.guest_usage (guest_id, credits_used)
  VALUES (_guest_id, 0)
  ON CONFLICT (guest_id) DO NOTHING;

  UPDATE public.guest_usage
  SET credits_used = credits_used + 1,
      updated_at = now()
  WHERE guest_id = _guest_id AND credits_used < _limit
  RETURNING credits_used INTO used;

  IF used IS NULL THEN
    RETURN -1;
  END IF;

  RETURN _limit - used;
END;
$$;
