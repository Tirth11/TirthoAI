import { Plus, MessageSquare, Trash2, Pencil, Moon, Sun, Sparkles, LogOut } from "lucide-react";
import { useState } from "react";
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
}: Props) {
  const { theme, toggle } = useTheme();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground">
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-4 pt-4 pb-3">
        <div
          className="flex h-9 w-9 items-center justify-center rounded-lg text-white shadow-lg"
          style={{ background: "var(--gradient-primary)" }}
        >
          <Sparkles className="h-4 w-4" />
        </div>
        <div className="flex-1">
          <div className="text-sm font-bold leading-tight">TirthoAI</div>
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Multi-Model Platform
          </div>
        </div>
      </div>

      {/* New chat */}
      <div className="px-3 pb-3">
        <button
          onClick={onNew}
          className="flex w-full items-center justify-center gap-2 rounded-xl px-3 py-2.5 text-sm font-semibold text-primary-foreground shadow-md transition hover:opacity-95"
          style={{ background: "var(--gradient-primary)", boxShadow: "var(--shadow-glow)" }}
        >
          <Plus className="h-4 w-4" />
          New Chat
        </button>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto px-2 pb-2">
        {conversations.length === 0 && (
          <div className="px-3 py-8 text-center text-xs text-muted-foreground">
            No conversations yet
          </div>
        )}
        <ul className="space-y-1">
          {conversations.map((c) => {
            const active = c.id === activeId;
            return (
              <li key={c.id}>
                <div
                  className={cn(
                    "group flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm transition cursor-pointer",
                    active
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "hover:bg-sidebar-accent/60"
                  )}
                  onClick={() => editingId !== c.id && onSelect(c.id)}
                >
                  <MessageSquare className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  {editingId === c.id ? (
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
                    />
                  ) : (
                    <span className="flex-1 truncate">{c.title}</span>
                  )}
                  {editingId !== c.id && (
                    <div className="flex items-center gap-0.5 opacity-0 transition group-hover:opacity-100">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditValue(c.title);
                          setEditingId(c.id);
                        }}
                        className="rounded p-1 hover:bg-background/60"
                        aria-label="Rename"
                      >
                        <Pencil className="h-3 w-3" />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (confirm("Delete this conversation?")) onDelete(c.id);
                        }}
                        className="rounded p-1 text-destructive hover:bg-background/60"
                        aria-label="Delete"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
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
          <span className="flex-1 truncate text-xs">{userEmail}</span>
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
  );
}
