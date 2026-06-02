import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  Send,
  Sparkles,
  Loader2,
  Paperclip,
  X,
  Image as ImageIcon,
  FileText,
  Menu,
  Copy,
  Check,
  Square,
  Zap,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { ModelPicker } from "@/components/ModelPicker";
import { autoSelectModel, getModelById, CATEGORY_META } from "@/lib/models";
import { ChatDB, type DBConversation } from "@/lib/chat-db";
import { supabase } from "@/integrations/supabase/client";
import { useCredits, FREE_CREDITS } from "@/hooks/use-credits";
import {
  GUEST_FREE_CREDITS,
  getGuestId,
  getGuestRemaining,
  setGuestRemaining,
} from "@/lib/guest";
import { SignupPrompt } from "@/components/SignupPrompt";

interface Props {
  conversation: DBConversation;
  onConversationChange: () => void | Promise<unknown>;
  onOpenSidebar: () => void;
  userEmail: string;
  userId: string;
  guest?: boolean;
  onGuestSignUp?: () => void;
  onGuestSignIn?: () => void;
}

const TEXT_EXTS = [".txt", ".md", ".csv", ".json", ".log", ".html", ".xml", ".yaml", ".yml"];

async function readTextFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result ?? ""));
    r.onerror = reject;
    r.readAsText(file);
  });
}

function deriveTitle(text: string): string {
  const t = text.trim().replace(/\s+/g, " ");
  if (!t) return "New Chat";
  return t.length > 40 ? t.slice(0, 40) + "…" : t;
}

function firstName(email: string): string {
  const local = email.split("@")[0] ?? "";
  if (!local) return "there";
  const cleaned = local.replace(/[._-].*/, "");
  return cleaned ? cleaned.charAt(0).toUpperCase() + cleaned.slice(1) : "there";
}

export function ChatWindow({
  conversation,
  onConversationChange,
  onOpenSidebar,
  userEmail,
  userId,
  guest = false,
  onGuestSignUp,
  onGuestSignIn,
}: Props) {
  const { credits: authedCredits, refresh: refreshCredits } = useCredits(guest ? undefined : userId);
  const [guestCredits, setGuestCreditsState] = useState<number>(() =>
    guest ? getGuestRemaining() : GUEST_FREE_CREDITS,
  );
  useEffect(() => {
    if (!guest) return;
    const sync = () => setGuestCreditsState(getGuestRemaining());
    window.addEventListener("guest-credits-changed", sync);
    return () => window.removeEventListener("guest-credits-changed", sync);
  }, [guest]);

  const credits = guest ? guestCredits : authedCredits;
  const totalCredits = guest ? GUEST_FREE_CREDITS : FREE_CREDITS;
  const outOfCredits = credits !== null && credits <= 0;
  const [showSignup, setShowSignup] = useState(false);
  const [modelId, setModelId] = useState(conversation.model_id);
  const [autoMode, setAutoMode] = useState(true);
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<File[]>([]);
  const [initialMessages, setInitialMessages] = useState<UIMessage[] | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const persistedIdsRef = useRef<Set<string>>(new Set());
  const stickToBottomRef = useRef(true);

  // Load messages for this conversation
  useEffect(() => {
    let alive = true;
    setInitialMessages(null);
    persistedIdsRef.current = new Set();
    if (guest) {
      setInitialMessages([]);
      return () => {
        alive = false;
      };
    }
    ChatDB.listMessages(conversation.id)
      .then((msgs) => {
        if (!alive) return;
        msgs.forEach((m) => persistedIdsRef.current.add(m.id));
        setInitialMessages(msgs);
      })
      .catch((e) => {
        console.error(e);
        if (alive) setInitialMessages([]);
      });
    return () => {
      alive = false;
    };
  }, [conversation.id, guest]);

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
        fetch: async (url, init) => {
          const headers = new Headers(init?.headers);
          if (guest) {
            headers.set("x-guest-id", getGuestId());
          } else {
            const { data } = await supabase.auth.getSession();
            const token = data.session?.access_token;
            if (token) headers.set("Authorization", `Bearer ${token}`);
          }
          const res = await fetch(url, { ...init, headers });
          if (guest) {
            const r = res.headers.get("x-guest-remaining");
            if (r !== null && r !== "") {
              const n = Number(r);
              if (Number.isFinite(n)) setGuestRemaining(n);
            }
          }
          if (res.status === 402) {
            const text = await res.clone().text();
            let code = "";
            let msg = "You're out of free credits.";
            try {
              const json = JSON.parse(text);
              code = json.error ?? "";
              msg = json.message ?? msg;
            } catch {
              /* ignore */
            }
            if (guest || code === "out_of_guest_credits") {
              setGuestRemaining(0);
              setShowSignup(true);
            }
            throw new Error(msg);
          }
          return res;
        },
      }),
    [guest],
  );

  const { messages, sendMessage, status, stop } = useChat({
    id: conversation.id,
    messages: initialMessages ?? [],
    transport,
    onError: (err) => toast.error(err.message || "Something went wrong"),
    onFinish: () => {
      if (!guest) refreshCredits();
    },
  });

  const isLoading = status === "submitted" || status === "streaming";

  // Persist new messages when streaming finishes
  useEffect(() => {
    if (status !== "ready" || guest) return;
    (async () => {
      for (const m of messages) {
        if (persistedIdsRef.current.has(m.id)) continue;
        const hasText = m.parts.some((p) => p.type === "text" && p.text);
        const hasFile = m.parts.some((p) => p.type === "file");
        if (!hasText && !hasFile) continue;
        await ChatDB.insertMessage(conversation.id, m);
        persistedIdsRef.current.add(m.id);
      }
      onConversationChange();
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, messages, conversation.id]);

  // Track whether the user is near the bottom (so we only auto-scroll then)
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      stickToBottomRef.current = distance < 120;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  // Smart auto-scroll: only follow if user hasn't scrolled away.
  useEffect(() => {
    if (!stickToBottomRef.current) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, status]);

  // Mobile keyboard handling: track visualViewport so the input stays above
  // the on-screen keyboard, and keep the latest message in view when it opens.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const vv = window.visualViewport;
    if (!vv) return;
    const update = () => {
      // bottom inset = how much of the layout viewport is hidden by the keyboard
      const inset = Math.max(0, window.innerHeight - vv.height - vv.offsetTop);
      document.documentElement.style.setProperty("--kb-inset", `${inset}px`);
      if (stickToBottomRef.current) {
        const el = scrollRef.current;
        if (el) el.scrollTop = el.scrollHeight;
      }
    };
    update();
    vv.addEventListener("resize", update);
    vv.addEventListener("scroll", update);
    return () => {
      vv.removeEventListener("resize", update);
      vv.removeEventListener("scroll", update);
      document.documentElement.style.setProperty("--kb-inset", "0px");
    };
  }, []);

  // When the textarea gains focus on mobile, force-scroll to the bottom so
  // the latest message remains visible above the keyboard.
  const handleInputFocus = () => {
    stickToBottomRef.current = true;
    requestAnimationFrame(() => {
      const el = scrollRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    });
  };

  useEffect(() => {
    setModelId(conversation.model_id);
    setInput("");
    setAttachments([]);
    stickToBottomRef.current = true;
  }, [conversation.id, conversation.model_id]);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, [input]);

  const handleAttach = (files: FileList | null) => {
    if (!files) return;
    const arr = Array.from(files).slice(0, 5);
    setAttachments((prev) => [...prev, ...arr].slice(0, 5));
  };

  const submit = async (textOverride?: string) => {
    const raw = (textOverride ?? input).trim();
    if (!raw && attachments.length === 0) return;
    if (isLoading) return;
    if (outOfCredits) {
      toast.error("You're out of free credits. Free tier is exhausted.");
      return;
    }

    const hasImage = attachments.some((f) => f.type.startsWith("image/"));

    let prefix = "";
    const imageFiles: File[] = [];
    for (const f of attachments) {
      if (f.type.startsWith("image/")) {
        imageFiles.push(f);
      } else if (TEXT_EXTS.some((ext) => f.name.toLowerCase().endsWith(ext))) {
        try {
          const content = await readTextFile(f);
          prefix += `\n\n--- File: ${f.name} ---\n${content.slice(0, 50_000)}\n`;
        } catch {
          /* ignore */
        }
      } else {
        prefix += `\n\n[Attached unsupported file: ${f.name}]`;
      }
    }

    const finalText = (prefix ? prefix + "\n\n" : "") + raw;

    let useModelId = modelId;
    if (autoMode) {
      useModelId = autoSelectModel(raw, hasImage);
    } else if (hasImage && !getModelById(modelId)?.supportsVision) {
      useModelId = "google/gemini-2.5-pro";
    }
    setModelId(useModelId);
    if (useModelId !== conversation.model_id && !guest) {
      ChatDB.updateConversation(conversation.id, { model_id: useModelId }).catch(console.error);
    }

    if (messages.length === 0 && !guest) {
      ChatDB.updateConversation(conversation.id, {
        title: deriveTitle(raw || "Image chat"),
      })
        .then(onConversationChange)
        .catch(console.error);
    }

    const dt = new DataTransfer();
    imageFiles.forEach((f) => dt.items.add(f));
    const fileList = dt.files.length > 0 ? dt.files : undefined;

    setInput("");
    setAttachments([]);
    stickToBottomRef.current = true;

    await sendMessage({ text: finalText, files: fileList }, { body: { modelId: useModelId } });
  };

  const activeModel = getModelById(modelId);
  const cat = activeModel?.category ?? "general";
  const catMeta = CATEGORY_META[cat];

  const suggestions: Array<{ text: string; icon: string; hint: string }> = [
    { text: "Explain transformers like I'm 12", icon: "🧠", hint: "Reasoning" },
    { text: "Write a React debounce hook with tests", icon: "💻", hint: "Coding" },
    { text: "A haiku about rainy Mumbai", icon: "✍️", hint: "Creative" },
    { text: "Ideas for a weekend side project", icon: "🚀", hint: "Brainstorm" },
  ];

  if (initialMessages === null) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const greetName = firstName(userEmail);

  return (
    <>
    <div
      className="flex h-full min-h-0 flex-col bg-background"
      style={{ height: "calc(100% - var(--kb-inset, 0px))" }}
    >
      <header
        className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-border bg-background/80 px-3 py-2.5 backdrop-blur sm:flex-nowrap sm:px-6 sm:py-3"
        style={{
          paddingTop: "max(0.625rem, env(safe-area-inset-top))",
          paddingLeft: "max(0.75rem, env(safe-area-inset-left))",
          paddingRight: "max(0.75rem, env(safe-area-inset-right))",
        }}
      >
        <div className="flex items-center gap-2 min-w-0">
          <button
            onClick={onOpenSidebar}
            className="rounded-lg p-1.5 text-muted-foreground transition hover:bg-accent hover:text-foreground md:hidden"
            aria-label="Open menu"
          >
            <Menu className="h-5 w-5" />
          </button>
          <span
            className={cn(
              "hidden sm:inline-flex items-center gap-1 rounded-full bg-gradient-to-r px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-white",
              catMeta.color
            )}
            title={`${catMeta.label} model active`}
          >
            {catMeta.icon} {catMeta.label}
          </span>
          <h1 className="truncate text-sm font-semibold text-foreground">
            {conversation.title}
          </h1>
        </div>
          <div className="flex min-w-0 flex-1 items-center justify-end gap-1.5 sm:flex-none sm:gap-2">
          {credits !== null && (
            <span
              className={cn(
                "hidden sm:inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[11px] font-semibold",
                outOfCredits
                  ? "border-destructive/40 bg-destructive/10 text-destructive"
                  : credits < 50
                    ? "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400"
                    : "border-primary/30 bg-primary/5 text-primary",
              )}
              title={`${credits} of ${totalCredits} free credits remaining`}
            >
              <Zap className="h-3 w-3" />
              {credits}/{totalCredits}
            </span>
          )}
          <ModelPicker
            modelId={modelId}
            onChange={(id) => {
              setModelId(id);
              setAutoMode(false);
              if (!guest) ChatDB.updateConversation(conversation.id, { model_id: id }).catch(console.error);
            }}
            autoMode={autoMode}
            onAutoToggle={setAutoMode}
          />
        </div>
      </header>

      {outOfCredits && (
        <div className="border-b border-destructive/30 bg-destructive/10 px-4 py-2.5 text-center text-xs font-medium text-destructive sm:px-6">
          {guest ? (
            <>
              You've used all {GUEST_FREE_CREDITS} free guest messages.{" "}
              <button
                type="button"
                onClick={() => setShowSignup(true)}
                className="underline font-semibold hover:opacity-80"
              >
                Sign up to keep going
              </button>
              .
            </>
          ) : (
            <>You've used all {FREE_CREDITS} free credits. The free tier is exhausted — thanks for trying TirthoAI!</>
          )}
        </div>
      )}

      <div
        ref={scrollRef}
        className="min-h-0 flex-1 overflow-y-auto px-3 py-5 sm:px-6 sm:py-6"
        style={{
          paddingLeft: "max(0.75rem, env(safe-area-inset-left))",
          paddingRight: "max(0.75rem, env(safe-area-inset-right))",
        }}
      >
        <div className="mx-auto max-w-3xl space-y-5 sm:space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center pt-8 pb-4 text-center">
              <div
                className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl text-white shadow-xl"
                style={{
                  background: "var(--gradient-primary)",
                  boxShadow: "var(--shadow-glow)",
                }}
              >
                <Sparkles className="h-8 w-8" />
              </div>
              <h2 className="text-2xl font-bold tracking-tight">
                Hey {greetName} 👋
              </h2>
              <p className="mt-1 max-w-md text-sm text-muted-foreground">
                Ask anything — I'll pick the best model for the job automatically. Your history is
                saved.
              </p>
              <div className="mt-8 grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
                {suggestions.map((s) => (
                  <button
                    key={s.text}
                    type="button"
                    onClick={() => submit(s.text)}
                    className="group flex items-start gap-3 rounded-xl border border-border bg-card px-4 py-3 text-left text-sm transition hover:border-primary/50 hover:shadow-md"
                  >
                    <span className="text-lg leading-none pt-0.5">{s.icon}</span>
                    <div className="flex-1">
                      <div className="text-foreground group-hover:text-primary">{s.text}</div>
                      <div className="mt-0.5 text-[10px] uppercase tracking-wider text-muted-foreground">
                        {s.hint}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}

          {status === "submitted" && (
            <div className="flex gap-3">
              <div
                className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-white"
                style={{ background: "var(--gradient-primary)" }}
              >
                <Sparkles className="h-4 w-4" />
              </div>
              <div className="rounded-2xl border border-border bg-card px-4 py-3">
                <div className="flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.3s]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.15s]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground" />
                </div>
              </div>
            </div>
          )}

          {/* Screen-reader announcement of live streaming */}
          <div className="sr-only" aria-live="polite">
            {status === "streaming" ? "Assistant is responding…" : ""}
          </div>
        </div>
      </div>

      <div
        className="shrink-0 border-t border-border bg-background/80 px-3 py-3 backdrop-blur sm:px-6 sm:py-4"
        style={{
          paddingBottom: "max(0.75rem, env(safe-area-inset-bottom))",
          paddingLeft: "max(0.75rem, env(safe-area-inset-left))",
          paddingRight: "max(0.75rem, env(safe-area-inset-right))",
        }}
      >
        <div className="mx-auto max-w-3xl">
          {attachments.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {attachments.map((f, i) => (
                <AttachmentChip
                  key={i}
                  file={f}
                  onRemove={() => setAttachments((p) => p.filter((_, j) => j !== i))}
                />
              ))}
            </div>
          )}

          <form
            onSubmit={(e) => {
              e.preventDefault();
              submit();
            }}
            className="flex items-end gap-2 rounded-2xl border border-border bg-card p-2 shadow-sm transition focus-within:border-primary/60 focus-within:ring-2 focus-within:ring-primary/20"
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*,.txt,.md,.csv,.json,.log,.html,.xml,.yaml,.yml"
              className="hidden"
              onChange={(e) => {
                handleAttach(e.target.files);
                if (fileInputRef.current) fileInputRef.current.value = "";
              }}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition hover:bg-accent hover:text-foreground"
              aria-label="Attach"
              title="Attach images or text files"
            >
              <Paperclip className="h-4 w-4" />
            </button>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onFocus={handleInputFocus}
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                  e.preventDefault();
                  submit();
                  return;
                }
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
              placeholder={
                outOfCredits
                  ? "You've used all your free credits"
                  : `Message TirthoAI… (Shift+Enter for newline)`
              }
              rows={1}
              className="flex-1 resize-none bg-transparent px-1 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none disabled:cursor-not-allowed"
              disabled={isLoading || outOfCredits}
            />
            {isLoading ? (
              <button
                type="button"
                onClick={() => stop()}
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-destructive text-destructive-foreground shadow-md transition hover:opacity-90"
                aria-label="Stop generating"
                title="Stop generating"
              >
                <Square className="h-3.5 w-3.5 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={(!input.trim() && attachments.length === 0) || outOfCredits}
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-primary-foreground shadow-md transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
                style={{ background: "var(--gradient-primary)" }}
                aria-label="Send"
              >
                <Send className="h-4 w-4" />
              </button>
            )}
          </form>
          <p className="mt-2 text-center text-[11px] text-muted-foreground">
            <kbd className="rounded border border-border bg-background px-1 py-0.5 font-mono text-[10px]">
              Enter
            </kbd>{" "}
            to send ·{" "}
            <kbd className="rounded border border-border bg-background px-1 py-0.5 font-mono text-[10px]">
              Shift+Enter
            </kbd>{" "}
            newline ·{" "}
            {autoMode ? "Auto-picks the best model" : `Using ${activeModel?.label ?? "model"}`}
          </p>
        </div>
      </div>
    </div>

      {guest && (
        <SignupPrompt
          open={showSignup}
          variant={outOfCredits ? "exhausted" : "soft"}
          onClose={() => setShowSignup(false)}
          onSignUp={() => onGuestSignUp?.()}
          onSignIn={() => onGuestSignIn?.()}
        />
      )}
    </>
  );
}

function AttachmentChip({ file, onRemove }: { file: File; onRemove: () => void }) {
  const [thumb, setThumb] = useState<string | null>(null);
  const isImage = file.type.startsWith("image/");

  useEffect(() => {
    if (!isImage) return;
    const url = URL.createObjectURL(file);
    setThumb(url);
    return () => URL.revokeObjectURL(url);
  }, [file, isImage]);

  return (
    <div className="flex items-center gap-2 rounded-lg border border-border bg-card px-2 py-1.5 text-xs">
      {isImage && thumb ? (
        <img
          src={thumb}
          alt={file.name}
          className="h-6 w-6 rounded object-cover"
        />
      ) : isImage ? (
        <ImageIcon className="h-3.5 w-3.5 text-emerald-500" />
      ) : (
        <FileText className="h-3.5 w-3.5 text-primary" />
      )}
      <span className="max-w-[180px] truncate">{file.name}</span>
      <button
        onClick={onRemove}
        className="rounded p-0.5 hover:bg-accent"
        aria-label={`Remove ${file.name}`}
      >
        <X className="h-3 w-3" />
      </button>
    </div>
  );
}

function MessageBubble({ message }: { message: UIMessage }) {
  const isUser = message.role === "user";
  const text = message.parts.map((p) => (p.type === "text" ? p.text : "")).join("");
  const fileParts = message.parts.filter((p) => p.type === "file") as Array<{
    type: "file";
    mediaType?: string;
    url?: string;
    filename?: string;
  }>;
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error("Copy failed");
    }
  };

  return (
    <div className={cn("group flex gap-3", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div
          className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-white shadow-md"
          style={{ background: "var(--gradient-primary)" }}
        >
          <Sparkles className="h-4 w-4" />
        </div>
      )}
      <div className="flex max-w-[85%] flex-col">
        <div
          className={cn(
            "rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm",
            isUser
              ? "text-primary-foreground"
              : "border border-border bg-card text-card-foreground"
          )}
          style={isUser ? { background: "var(--gradient-primary)" } : undefined}
        >
          {fileParts.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {fileParts.map((p, i) =>
                p.mediaType?.startsWith("image/") && p.url ? (
                  <img
                    key={i}
                    src={p.url}
                    alt={p.filename ?? "attachment"}
                    className="max-h-48 rounded-lg border border-border/30"
                  />
                ) : (
                  <div key={i} className="rounded-lg bg-background/20 px-2 py-1 text-xs">
                    📎 {p.filename ?? "file"}
                  </div>
                )
              )}
            </div>
          )}
          {isUser ? (
            <p className="whitespace-pre-wrap">{text}</p>
          ) : (
            <div className="prose prose-sm dark:prose-invert max-w-none prose-pre:bg-muted prose-pre:text-foreground prose-code:before:content-none prose-code:after:content-none prose-code:rounded prose-code:bg-muted prose-code:px-1 prose-code:py-0.5">
              <ReactMarkdown>{text || "…"}</ReactMarkdown>
            </div>
          )}
        </div>
        {!isUser && text && (
          <button
            onClick={copy}
            className="mt-1 inline-flex w-fit items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-muted-foreground opacity-0 transition hover:bg-accent hover:text-foreground group-hover:opacity-100"
            aria-label="Copy message"
          >
            {copied ? (
              <>
                <Check className="h-3 w-3" /> Copied
              </>
            ) : (
              <>
                <Copy className="h-3 w-3" /> Copy
              </>
            )}
          </button>
        )}
      </div>
    </div>
  );
}
