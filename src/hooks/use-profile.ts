import { useCallback, useEffect, useState } from "react";
import { useServerFn } from "@tanstack/react-start";
import { getProfile, updateProfile, type ProfileDTO } from "@/lib/profile.functions";

type Listener = (p: ProfileDTO | null) => void;
let cached: ProfileDTO | null = null;
const listeners = new Set<Listener>();

function emit(p: ProfileDTO | null) {
  cached = p;
  listeners.forEach((l) => l(p));
}

export function useProfile() {
  const get = useServerFn(getProfile);
  const upd = useServerFn(updateProfile);

  const [profile, setProfile] = useState<ProfileDTO | null>(cached);
  const [loading, setLoading] = useState(cached === null);

  useEffect(() => {
    const l: Listener = (p) => setProfile(p);
    listeners.add(l);
    return () => {
      listeners.delete(l);
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const { profile } = await get();
      emit(profile);
    } catch (e) {
      console.error("getProfile failed", e);
    } finally {
      setLoading(false);
    }
  }, [get]);

  useEffect(() => {
    if (cached === null) void refresh();
    else setLoading(false);
  }, [refresh]);

  const save = useCallback(
    async (patch: { display_name?: string; avatar_url?: string | null }) => {
      const { profile } = await upd({ data: patch });
      emit(profile);
      return profile;
    },
    [upd],
  );

  return { profile, loading, refresh, save };
}
