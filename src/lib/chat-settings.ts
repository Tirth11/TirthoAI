// Per-thread generation settings (persona / temperature / max tokens).
// Stored in localStorage so they survive reloads and work for guests too —
// mirrors ModelCache. These are sent in the /api/chat request body and the
// server validates + clamps them (see src/routes/api/chat.ts).

export interface ChatSettingsEntry {
  /** Per-chat persona / system prompt. Empty/undefined = default assistant. */
  system?: string;
  /** Sampling temperature, 0–2. Undefined = provider default. */
  temperature?: number;
  /** Max output tokens, 1–8192. Undefined = provider default. */
  maxTokens?: number;
}

const KEY = "tirthoai.chat-settings.v1";
type Stored = Record<string, ChatSettingsEntry>;

function read(): Stored {
  if (typeof window === "undefined") return {};
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Stored | null;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function write(s: Stored) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* ignore quota */
  }
}

/** Drop empty/out-of-range fields so we never persist or send junk. */
function normalize(e: ChatSettingsEntry): ChatSettingsEntry {
  const out: ChatSettingsEntry = {};
  if (typeof e.system === "string" && e.system.trim()) out.system = e.system.trim().slice(0, 4000);
  if (typeof e.temperature === "number" && Number.isFinite(e.temperature)) {
    out.temperature = Math.min(2, Math.max(0, e.temperature));
  }
  if (typeof e.maxTokens === "number" && Number.isFinite(e.maxTokens)) {
    out.maxTokens = Math.min(8192, Math.max(1, Math.floor(e.maxTokens)));
  }
  return out;
}

export const ChatSettings = {
  get(threadId: string): ChatSettingsEntry {
    return read()[threadId] ?? {};
  },
  set(threadId: string, entry: ChatSettingsEntry) {
    const s = read();
    const normalized = normalize(entry);
    if (Object.keys(normalized).length === 0) {
      delete s[threadId];
    } else {
      s[threadId] = normalized;
    }
    write(s);
    return normalized;
  },
  remove(threadId: string) {
    const s = read();
    delete s[threadId];
    write(s);
  },
};
