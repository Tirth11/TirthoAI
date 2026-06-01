import {
  Plus,
  MessageSquare,
  Trash2,
  Pencil,
  Moon,
  Sun,
  Sparkles,
  LogOut,
  Search,
  X,
  Check,
} from "lucide-react";
import { useMemo, useState } from "react";
import { type DBConversation } from "@/lib/chat-db";
import { cn } from "@/lib/utils";
import { useTheme } from "@/hooks/use-theme";

interface Props {
  conversations: DBConversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
  userEmail: string;
  onSignOut: () => void;
  /** Mobile drawer open state — ignored on desktop. */
  isOpen: boolean;
  onClose: () => void;
}

type BucketKey = "today" | "yesterday" | "week" | "older";
const BUCKET_LABELS: Record<BucketKey, string> = {
  today: "Today",
  yesterday: "Yesterday",
  week: "Previous 7 days",
  older: "Older",
};

function bucketOf(iso: string): BucketKey {
  const d = new Date(iso).getTime();
  const now = Date.now();
  const dayMs = 86_400_000;
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);
  if (d >= startOfToday.getTime()) return "today";
  if (d >= startOfToday.getTime() - dayMs) return "yesterday";
  if (d >= now - 7 * dayMs) return "week";
  return "older";
}

export function Sidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  onRename,
  userEmail,
  onSignOut,
  isOpen,
  onClose,
}: Props) {
  const { theme, toggle } = useTheme();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? conversations.filter((c) => c.title.toLowerCase().includes(q)) : conversations;
  }, [conversations, query]);

  const grouped = useMemo(() => {
    const out: Record<BucketKey, DBConversation[]> = {
      today: [],
      yesterday: [],
      week: [],
      older: [],
    };
    for (const c of filtered) out[bucketOf(c.updated_at)].push(c);
    return out;
  }, [filtered]);

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/40 backdrop-blur-sm md:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={cn(
          "z-40 flex h-full w-72 shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground transition-transform duration-200 ease-out",
          "fixed inset-y-0 left-0 md:static md:translate-x-0",
          isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        )}
      >
        {/* Brand */}
        <div className="flex items-center gap-2.5 px-4 pt-4 pb-3">
          <div
            className="flex h-9 w-9 items-center justify-center rounded-lg text-white shadow-lg"
            style={{ background: "var(--gradient-primary)" }}
          >
            <Sparkles className="h-4 w-4" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-bold leading-tight">TirthoAI</div>
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
              Multi-Model Workspace
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded p-1.5 text-muted-foreground hover:bg-sidebar-accent/60 md:hidden"
            aria-label="Close menu"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* New chat */}
        <div className="px-3 pb-2">
          <button
            onClick={onNew}
            className="flex w-full items-center justify-center gap-2 rounded-xl px-3 py-2.5 text-sm font-semibold text-primary-foreground shadow-md transition hover:opacity-95 active:scale-[0.98]"
            style={{ background: "var(--gradient-primary)", boxShadow: "var(--shadow-glow)" }}
          >
            <Plus className="h-4 w-4" />
            New Chat
          </button>
        </div>

        {/* Search */}
        <div className="px-3 pb-2">
          <div className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search conversations"
              className="w-full rounded-lg border border-sidebar-border bg-background/40 py-1.5 pl-8 pr-2 text-xs placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
              aria-label="Search conversations"
            />
          </div>
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto px-2 pb-2">
          {filtered.length === 0 && (
            <div className="px-3 py-8 text-center text-xs text-muted-foreground">
              {query ? "No matches" : "No conversations yet"}
            </div>
          )}

          {(Object.keys(BUCKET_LABELS) as BucketKey[]).map((bucket) => {
            const items = grouped[bucket];
            if (items.length === 0) return null;
            return (
              <div key={bucket} className="mb-3">
                <div className="px-2.5 pt-1.5 pb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  {BUCKET_LABELS[bucket]}
                </div>
                <ul className="space-y-0.5">
                  {items.map((c) => {
                    const active = c.id === activeId;
                    const isEditing = editingId === c.id;
                    const isConfirming = confirmDelete === c.id;
                    return (
                      <li key={c.id} className="relative">
                        {active && (
                          <span
                            className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-r-full"
                            style={{ background: "var(--gradient-primary)" }}
                            aria-hidden="true"
                          />
                        )}
                        <div
                          className={cn(
                            "group flex items-center gap-2 rounded-lg pl-3 pr-1.5 py-2 text-sm transition cursor-pointer",
                            active
                              ? "bg-sidebar-accent text-sidebar-accent-foreground"
                              : "hover:bg-sidebar-accent/60"
                          )}
                          onClick={() => !isEditing && !isConfirming && onSelect(c.id)}
                        >
                          <MessageSquare className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                          {isEditing ? (
                            <input
                              autoFocus
                              value={editValue}
                              onChange={(e) => setEditValue(e.target.value)}
                              onBlur={() => {
                                onRename(c.id, editValue.trim() || c.title);
                                setEditingId(null);
                              }}
                              onKeyDown={(e) => {
                                if (e.key === "Enter") {
                                  onRename(c.id, editValue.trim() || c.title);
                                  setEditingId(null);
                                }
                                if (e.key === "Escape") setEditingId(null);
                              }}
                              className="flex-1 rounded bg-background px-1.5 py-0.5 text-sm outline-none ring-1 ring-primary"
                              onClick={(e) => e.stopPropagation()}
                            />
                          ) : (
                            <span className="flex-1 truncate">{c.title}</span>
                          )}

                          {!isEditing && !isConfirming && (
                            <div className="flex items-center gap-0.5 opacity-0 transition group-hover:opacity-100 focus-within:opacity-100">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setEditValue(c.title);
                                  setEditingId(c.id);
                                }}
                                className="rounded p-1 hover:bg-background/60"
                                aria-label="Rename"
                                title="Rename"
                              >
                                <Pencil className="h-3 w-3" />
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setConfirmDelete(c.id);
                                }}
                                className="rounded p-1 text-destructive hover:bg-background/60"
                                aria-label="Delete"
                                title="Delete"
                              >
                                <Trash2 className="h-3 w-3" />
                              </button>
                            </div>
                          )}

                          {isConfirming && (
                            <div
                              className="flex items-center gap-1"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <span className="text-[10px] font-medium text-destructive">
                                Sure?
                              </span>
                              <button
                                onClick={() => {
                                  setConfirmDelete(null);
                                  onDelete(c.id);
                                }}
                                className="rounded p-1 text-destructive hover:bg-destructive/10"
                                aria-label="Confirm delete"
                                title="Confirm delete"
                              >
                                <Check className="h-3 w-3" />
                              </button>
                              <button
                                onClick={() => setConfirmDelete(null)}
                                className="rounded p-1 text-muted-foreground hover:bg-background/60"
                                aria-label="Cancel"
                                title="Cancel"
                              >
                                <X className="h-3 w-3" />
                              </button>
                            </div>
                          )}
                        </div>
                      </li>
                    );
                  })}
                </ul>
              </div>
            );
          })}
        </div>

        {/* Footer */}
        <div className="border-t border-sidebar-border px-3 py-3">
          <div className="mb-2 flex items-center gap-2 rounded-lg bg-background/40 px-2.5 py-2">
            <div
              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[11px] font-bold text-white"
              style={{ background: "var(--gradient-primary)" }}
            >
              {userEmail.slice(0, 1).toUpperCase()}
            </div>
            <span className="flex-1 truncate text-xs" title={userEmail}>
              {userEmail}
            </span>
            <button
              onClick={onSignOut}
              className="rounded p-1 text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground"
              aria-label="Sign out"
              title="Sign out"
            >
              <LogOut className="h-3.5 w-3.5" />
            </button>
          </div>
          <button
            onClick={toggle}
            className="flex w-full items-center justify-center gap-2 rounded-lg border border-sidebar-border bg-background/40 px-3 py-2 text-xs font-medium hover:bg-sidebar-accent/60"
          >
            {theme === "dark" ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
            {theme === "dark" ? "Light mode" : "Dark mode"}
          </button>
          <p className="mt-2 text-center text-[10px] text-muted-foreground">
            Powered by Lovable AI
          </p>
        </div>
      </aside>
    </>
  );
}
