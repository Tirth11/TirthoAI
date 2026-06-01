import { supabase } from "@/integrations/supabase/client";
import { useEffect, useState } from "react";
import type { Session, User } from "@supabase/supabase-js";
import { enforceRememberMe } from "@/lib/remember-me";

export function useAuthSession() {
  const [session, setSession] = useState<Session | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    // Subscribe FIRST per Supabase guidance
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, s) => {
      if (!mounted) return;
      setSession(s);
      setUser(s?.user ?? null);
    });
    // Enforce remember-me, THEN fetch existing session
    enforceRememberMe().then(() => {
      supabase.auth.getSession().then(({ data }) => {
        if (!mounted) return;
        setSession(data.session);
        setUser(data.session?.user ?? null);
        setLoading(false);
      });
    });
    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, []);

  return { session, user, loading };
}
