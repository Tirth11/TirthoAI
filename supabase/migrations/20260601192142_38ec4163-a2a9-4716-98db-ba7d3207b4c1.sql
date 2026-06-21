
REVOKE EXECUTE ON FUNCTION public.handle_new_user_credits() FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.consume_credit(UUID) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.consume_credit(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION public.handle_new_user_credits() TO service_role;
