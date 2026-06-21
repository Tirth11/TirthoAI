// Guest (anonymous) trial state. Stored in localStorage; the server is the
// source of truth for the remaining count via the x-guest-id header.

export const GUEST_FREE_CREDITS = 50;
const GUEST_ID_KEY = "tirthoai_guest_id";
const GUEST_REMAINING_KEY = "tirthoai_guest_remaining";

function randomId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID().replace(/-/g, "");
  }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export function getGuestId(): string {
  if (typeof window === "undefined") return "";
  let id = localStorage.getItem(GUEST_ID_KEY);
  if (!id) {
    id = randomId();
    localStorage.setItem(GUEST_ID_KEY, id);
  }
  return id;
}

export function getGuestRemaining(): number {
  if (typeof window === "undefined") return GUEST_FREE_CREDITS;
  const raw = localStorage.getItem(GUEST_REMAINING_KEY);
  if (raw === null) return GUEST_FREE_CREDITS;
  const n = Number(raw);
  return Number.isFinite(n) ? Math.max(0, Math.min(GUEST_FREE_CREDITS, n)) : GUEST_FREE_CREDITS;
}

export function setGuestRemaining(n: number) {
  if (typeof window === "undefined") return;
  localStorage.setItem(GUEST_REMAINING_KEY, String(Math.max(0, n)));
  window.dispatchEvent(new CustomEvent("guest-credits-changed"));
}

const GUEST_MODE_KEY = "tirthoai_guest_mode";
export function isGuestMode(): boolean {
  if (typeof window === "undefined") return false;
  return localStorage.getItem(GUEST_MODE_KEY) === "1";
}
export function enterGuestMode() {
  if (typeof window === "undefined") return;
  localStorage.setItem(GUEST_MODE_KEY, "1");
  window.dispatchEvent(new CustomEvent("guest-mode-changed"));
}
export function exitGuestMode() {
  if (typeof window === "undefined") return;
  localStorage.removeItem(GUEST_MODE_KEY);
  window.dispatchEvent(new CustomEvent("guest-mode-changed"));
}
