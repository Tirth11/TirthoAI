import { useEffect, useState, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";

export const FREE_CREDITS = 500;

export function useCredits(userId: string | undefined) {
  const [credits, setCredits] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    if (!userId) {
      setCredits(null);
      setLoading(false);
      return;
    }
    const { data, error } = await supabase
      .from("user_credits")
      .select("credits")
      .eq("user_id", userId)
      .maybeSingle();
    if (!error) {
      setCredits(data?.credits ?? FREE_CREDITS);
    }
    setLoading(false);
  }, [userId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Realtime updates. Use a unique channel name per mount so React strict-mode
  // re-mounts don't reuse an already-subscribed channel (which throws when
  // .on() is called after .subscribe()).
  useEffect(() => {
    if (!userId) return;
    const channelName = `credits-${userId}-${Math.random().toString(36).slice(2, 10)}`;
    const channel = supabase.channel(channelName);
    channel
      .on(
        "postgres_changes",
        {
          event: "*",
          schema: "public",
          table: "user_credits",
          filter: `user_id=eq.${userId}`,
        },
        (payload) => {
          const newRow = payload.new as { credits?: number } | undefined;
          if (newRow && typeof newRow.credits === "number") {
            setCredits(newRow.credits);
          }
        },
      )
      .subscribe();
    return () => {
      supabase.removeChannel(channel);
    };
  }, [userId]);

  return { credits, loading, refresh, setCredits };
}
