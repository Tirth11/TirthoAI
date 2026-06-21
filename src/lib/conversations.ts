import type { UIMessage } from "ai";

export interface Conversation {
  id: string;
  title: string;
  category: string;
  modelId: string;
  messages: UIMessage[];
  createdAt: number;
  updatedAt: number;
}

const KEY = "tirthoai.conversations.v1";

function read(): Conversation[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    return JSON.parse(raw) as Conversation[];
  } catch {
    return [];
  }
}

function write(list: Conversation[]) {
  if (typeof window === "undefined") return;
  localStorage.setItem(KEY, JSON.stringify(list));
}

export const ConvStore = {
  list(): Conversation[] {
    return read().sort((a, b) => b.updatedAt - a.updatedAt);
  },
  get(id: string): Conversation | undefined {
    return read().find((c) => c.id === id);
  },
  create(modelId: string, category = "general"): Conversation {
    const conv: Conversation = {
      id: crypto.randomUUID(),
      title: "New Chat",
      category,
      modelId,
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    const list = read();
    list.push(conv);
    write(list);
    return conv;
  },
  update(id: string, patch: Partial<Conversation>) {
    const list = read();
    const idx = list.findIndex((c) => c.id === id);
    if (idx === -1) return;
    list[idx] = { ...list[idx], ...patch, updatedAt: Date.now() };
    write(list);
  },
  rename(id: string, title: string) {
    this.update(id, { title });
  },
  delete(id: string) {
    write(read().filter((c) => c.id !== id));
  },
  saveMessages(id: string, messages: UIMessage[]) {
    this.update(id, { messages });
  },
};

export function deriveTitle(text: string): string {
  const t = text.trim().replace(/\s+/g, " ");
  if (!t) return "New Chat";
  return t.length > 40 ? t.slice(0, 40) + "…" : t;
}
