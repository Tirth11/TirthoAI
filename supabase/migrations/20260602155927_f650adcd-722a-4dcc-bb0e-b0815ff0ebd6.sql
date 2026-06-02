
REVOKE EXECUTE ON FUNCTION public.consume_guest_credit(TEXT, INTEGER) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.consume_guest_credit(TEXT, INTEGER) TO service_role;
