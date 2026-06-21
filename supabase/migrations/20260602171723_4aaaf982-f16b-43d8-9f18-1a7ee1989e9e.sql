
-- Trigger / helper functions: should never be called via the API
REVOKE EXECUTE ON FUNCTION public.handle_new_user_credits() FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.handle_new_user_profile() FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.touch_conversation_updated_at() FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.set_updated_at() FROM PUBLIC, anon, authenticated;

-- consume_credit: authenticated users only
REVOKE EXECUTE ON FUNCTION public.consume_credit(uuid) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.consume_credit(uuid) TO authenticated;

-- consume_guest_credit: must remain callable by anon (guest chat) and authenticated
REVOKE EXECUTE ON FUNCTION public.consume_guest_credit(text, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.consume_guest_credit(text, integer) TO anon, authenticated;

-- guest_usage: RLS is enabled but had no policy. Block all direct API access;
-- only the SECURITY DEFINER function consume_guest_credit may read/write it.
DROP POLICY IF EXISTS "block_direct_access" ON public.guest_usage;
CREATE POLICY "block_direct_access" ON public.guest_usage
  FOR ALL
  TO anon, authenticated
  USING (false)
  WITH CHECK (false);
