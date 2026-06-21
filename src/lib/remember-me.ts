/**
 * Lightweight "Remember me" implementation that layers on top of Supabase's
 * default localStorage session persistence. If the user did NOT tick remember-me,
 * we sign them out as soon as they open a fresh browser window/tab (the
 * sessionStorage marker disappears between sessions, while localStorage survives).
 */
import { supabase } from "@/integrations/supabase/client";

const REMEMBER_KEY = "tirthoai_remember";
const ALIVE_KEY = "tirthoai_session_alive";

export function setRemember(remember: boolean) {
  try {
    localStorage.setItem(REMEMBER_KEY, remember ? "1" : "0");
    sessionStorage.setItem(ALIVE_KEY, "1");
  } catch {
    /* storage unavailable */
  }
}

export function getRemember(): boolean {
  try {
    return localStorage.getItem(REMEMBER_KEY) !== "0";
  } catch {
    return true;
  }
}

/**
 * Called once at app boot. If remember-me is OFF and this is a fresh browser
 * session (sessionStorage doesn't have the alive marker), sign out before any
 * UI renders.
 */
export async function enforceRememberMe(): Promise<void> {
  try {
    const remember = localStorage.getItem(REMEMBER_KEY) !== "0";
    const alive = sessionStorage.getItem(ALIVE_KEY) === "1";
    if (!remember && !alive) {
      await supabase.auth.signOut();
    }
    sessionStorage.setItem(ALIVE_KEY, "1");
  } catch {
    /* ignore */
  }
}
