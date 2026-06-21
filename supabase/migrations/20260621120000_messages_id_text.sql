-- Message ids are generated client-side by the AI SDK (useChat) as short,
-- non-UUID strings (e.g. "Qo3KA5vCeOXcAEhY"). The original UUID column rejected
-- them (22P02), so messages were never persisted. Store ids as text instead.
ALTER TABLE public.messages ALTER COLUMN id DROP DEFAULT;
ALTER TABLE public.messages ALTER COLUMN id TYPE text USING id::text;
ALTER TABLE public.messages ALTER COLUMN id SET DEFAULT gen_random_uuid()::text;
