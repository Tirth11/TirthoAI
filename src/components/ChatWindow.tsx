import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { useEffect, useMemo, useRef, useState, memo, useDeferredValue, useCallback } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { renderMarkdown, cancelMarkdown, cancelAllMarkdown } from "@/lib/markdown-client";
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
import { ModelCache } from "@/lib/model-cache";
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from "@/components/ui/tooltip";
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

function formatRelativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  if (Number.isNaN(diff)) return "just now";
  const s = Math.floor(diff / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

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
  const cached = ModelCache.get(conversation.id);
  const [modelId, setModelId] = useState(cached?.modelId ?? conversation.model_id);
  const [modelUpdatedAt, setModelUpdatedAt] = useState<string>(
    cached?.updatedAt ?? conversation.model_updated_at ?? conversation.updated_at,
  );
  const [previousModelId, setPreviousModelId] = useState<string | undefined>(cached?.previousModelId);
  const [autoMode, setAutoMode] = useState(true);
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<File[]>([]);
  const [initialMessages, setInitialMessages] = useState<UIMessage[] | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const persistedIdsRef = useRef<Set<string>>(new Set());
  const pendingPromptModelRef = useRef<string | null>(null);
  const pendingCostRef = useRef<number | null>(null);
  const creditsRef = useRef<number | null>(null);
  const [promptMeta, setPromptMeta] = useState<Record<string, { modelId: string; cost: number }>>({});
  const stickToBottomRef = useRef(true);

  useEffect(() => {
    creditsRef.current = credits ?? null;
  }, [credits]);

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
          const creditsBefore = creditsRef.current;
          const res = await fetch(url, { ...init, headers });
          const remainHeader = guest
            ? res.headers.get("x-guest-remaining")
            : res.headers.get("x-credits-remaining");
          if (remainHeader !== null && remainHeader !== "") {
            const remain = Number(remainHeader);
            if (Number.isFinite(remain)) {
              if (guest) setGuestRemaining(remain);
              if (creditsBefore !== null) {
                const delta = creditsBefore - remain;
                pendingCostRef.current = delta > 0 ? delta : 0;
              }
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

  // Stable cancel helper: abort the in-flight stream AND drop all queued
  // markdown work so the worker stops parsing stale bubbles.
  const abortAll = useCallback(() => {
    try {
      stop();
    } catch {
      /* ignore */
    }
    cancelAllMarkdown();
  }, [stop]);

  // Abort when switching conversations or unmounting (route change).
  useEffect(() => {
    return () => {
      abortAll();
    };
  }, [conversation.id, abortAll]);

  // Abort on page refresh / tab close so the worker doesn't keep grinding.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const onUnload = () => abortAll();
    window.addEventListener("beforeunload", onUnload);
    window.addEventListener("pagehide", onUnload);
    return () => {
      window.removeEventListener("beforeunload", onUnload);
      window.removeEventListener("pagehide", onUnload);
    };
  }, [abortAll]);

  // Streaming backpressure: coalesce token-by-token updates into ≤1 paint per frame
  // (and not more often than ~80ms) so the render path can't saturate the main thread.
  const [renderMessages, setRenderMessages] = useState<UIMessage[]>(messages);
  const lastFlushAtRef = useRef(0);
  const rafIdRef = useRef<number | null>(null);
  const latestMessagesRef = useRef(messages);
  useEffect(() => {
    latestMessagesRef.current = messages;
    if (status !== "streaming") {
      // Idle / submitted / error / done — flush immediately so the final text shows.
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
      setRenderMessages(messages);
      lastFlushAtRef.current = performance.now();
      return;
    }
    if (rafIdRef.current !== null) return;
    const MIN_INTERVAL = 80;
    const tick = () => {
      rafIdRef.current = null;
      const now = performance.now();
      const wait = MIN_INTERVAL - (now - lastFlushAtRef.current);
      if (wait > 0) {
        rafIdRef.current = requestAnimationFrame(tick);
        return;
      }
      lastFlushAtRef.current = now;
      setRenderMessages(latestMessagesRef.current);
    };
    rafIdRef.current = requestAnimationFrame(tick);
  }, [messages, status]);
  useEffect(
    () => () => {
      if (rafIdRef.current !== null) cancelAnimationFrame(rafIdRef.current);
    },
    [],
  );

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
  }, [renderMessages, status]);

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
    const c = ModelCache.get(conversation.id);
    setModelId(c?.modelId ?? conversation.model_id);
    setModelUpdatedAt(c?.updatedAt ?? conversation.model_updated_at ?? conversation.updated_at);
    setPreviousModelId(c?.previousModelId);
    setInput("");
    setAttachments([]);
    stickToBottomRef.current = true;
  }, [conversation.id, conversation.model_id, conversation.model_updated_at, conversation.updated_at]);

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
    if (useModelId !== modelId) {
      setPreviousModelId(modelId);
    }
    if (useModelId !== conversation.model_id) {
      ModelCache.set(conversation.id, useModelId);
      setModelUpdatedAt(new Date().toISOString());
      if (!guest) {
        ChatDB.updateConversation(conversation.id, { model_id: useModelId })
          .then(onConversationChange)
          .catch(console.error);
      }
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

    pendingPromptModelRef.current = useModelId;
    await sendMessage({ text: finalText, files: fileList }, { body: { modelId: useModelId } });
  };

  // Tag each new user message with the model used + actual credit cost
  // (computed from the x-credits-remaining header, fallback to 1).
  useEffect(() => {
    const pending = pendingPromptModelRef.current;
    if (!pending) return;
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    if (lastUser && !promptMeta[lastUser.id]) {
      const cost = pendingCostRef.current ?? 1;
      setPromptMeta((prev) => ({ ...prev, [lastUser.id]: { modelId: pending, cost } }));
      pendingPromptModelRef.current = null;
      pendingCostRef.current = null;
    }
  }, [messages, promptMeta, status]);

  // Virtualized message list — ONLY for very long conversations.
  // For short lists, virtualization adds overhead and risks zero-height
  // measurement glitches that make the chat appear blank after a few messages.
  const VIRTUALIZE_THRESHOLD = 50;
  const shouldVirtualize = renderMessages.length > VIRTUALIZE_THRESHOLD;
  const rowVirtualizer = useVirtualizer({
    count: shouldVirtualize ? renderMessages.length : 0,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 180,
    overscan: 6,
    getItemKey: (i) => renderMessages[i]?.id ?? i,
  });

  // Smart auto-scroll for the virtualizer: jump to last item when sticking.
  useEffect(() => {
    if (!shouldVirtualize) return;
    if (!stickToBottomRef.current) return;
    if (renderMessages.length === 0) return;
    rowVirtualizer.scrollToIndex(renderMessages.length - 1, { align: "end" });
  }, [renderMessages.length, rowVirtualizer, shouldVirtualize]);

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
    <div className="flex h-full min-h-0 flex-1 flex-col bg-background" style={{ minHeight: 0 }}>
      <header
        className="relative z-20 grid shrink-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2 border-b border-border bg-background px-3 py-2.5 sm:flex sm:flex-nowrap sm:items-center sm:justify-between sm:px-6 sm:py-3"
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
          <div className="flex min-w-0 items-center justify-end gap-1.5 sm:gap-2">
          {guest && (
            <div className="hidden sm:flex items-center gap-1.5 shrink-0">
              <button
                type="button"
                onClick={() => onGuestSignIn?.()}
                className="rounded-full border border-border bg-background px-3 py-1.5 text-xs font-semibold text-foreground transition hover:bg-accent hover:text-accent-foreground"
              >
                Log in
              </button>
              <button
                type="button"
                onClick={() => onGuestSignUp?.()}
                className="rounded-full bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground shadow-sm transition hover:bg-primary/90"
              >
                Sign up
              </button>
            </div>
          )}
          {credits !== null && (
            <span
              className={cn(
                "hidden sm:inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[11px] font-semibold shrink-0",
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
          <div className="min-w-0 max-w-[60vw] sm:max-w-xs">
            <ModelPicker
              modelId={modelId}
              onChange={(id) => {
                if (id !== modelId) setPreviousModelId(modelId);
                setModelId(id);
                setAutoMode(false);
                ModelCache.set(conversation.id, id);
                setModelUpdatedAt(new Date().toISOString());
                if (!guest) {
                  ChatDB.updateConversation(conversation.id, { model_id: id })
                    .then(onConversationChange)
                    .catch(console.error);
                }
              }}
              autoMode={autoMode}
              onAutoToggle={setAutoMode}
              hideUserModels={guest}
            />
          </div>
        </div>

      </header>

      {/* Model-change subtitle row (desktop only) */}
      <div className="relative z-10 hidden shrink-0 justify-end border-b border-border/60 bg-background px-6 py-1 sm:flex">
        <TooltipProvider delayDuration={150}>
          <Tooltip>
            <TooltipTrigger asChild>
              <span
                data-testid="model-changed-indicator"
                data-model-id={modelId}
                data-previous-model-id={previousModelId ?? ""}
                data-model-updated-at={modelUpdatedAt}
                className="cursor-help truncate text-[10px] text-muted-foreground/70 tabular-nums"
              >
                changed {formatRelativeTime(modelUpdatedAt)} · {userEmail}
              </span>
            </TooltipTrigger>
            <TooltipContent side="bottom" align="end" className="max-w-xs text-xs">
              <div className="space-y-1" data-testid="model-changed-tooltip">
                <div className="font-semibold">Model change</div>
                <div data-testid="model-changed-previous">
                  <span className="text-muted-foreground">Previous: </span>
                  {previousModelId
                    ? (getModelById(previousModelId)?.label ?? previousModelId)
                    : "—"}
                </div>
                <div data-testid="model-changed-current">
                  <span className="text-muted-foreground">Current: </span>
                  {getModelById(modelId)?.label ?? modelId}
                </div>
                <div data-testid="model-changed-at">
                  <span className="text-muted-foreground">At: </span>
                  <span className="tabular-nums">
                    {new Date(modelUpdatedAt).toLocaleString(undefined, {
                      dateStyle: "medium",
                      timeStyle: "medium",
                    })}
                  </span>
                </div>
                <div data-testid="model-changed-by">
                  <span className="text-muted-foreground">By: </span>
                  {userEmail}
                </div>
              </div>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>


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
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 py-5 sm:px-6 sm:py-6"
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

          {renderMessages.length > 0 && !shouldVirtualize && (
            <div className="space-y-5 sm:space-y-6">
              {renderMessages.map((m) => (
                <MessageBubble key={m.id} message={m} meta={promptMeta[m.id]} />
              ))}
            </div>
          )}

          {renderMessages.length > 0 && shouldVirtualize && (
            <div
              style={{
                height: `${rowVirtualizer.getTotalSize()}px`,
                position: "relative",
                width: "100%",
              }}
            >
              {rowVirtualizer.getVirtualItems().map((virtualRow) => {
                const m = renderMessages[virtualRow.index];
                if (!m) return null;
                return (
                  <div
                    key={virtualRow.key}
                    data-index={virtualRow.index}
                    ref={rowVirtualizer.measureElement}
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      transform: `translateY(${virtualRow.start}px)`,
                      paddingBottom: "1.25rem",
                    }}
                  >
                    <MessageBubble message={m} meta={promptMeta[m.id]} />
                  </div>
                );
              })}
            </div>
          )}

          {(status === "submitted" || status === "streaming") && (
            <div className="flex justify-end" data-testid="prompt-progress-pill" data-status={status}>
              <div
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-semibold",
                  status === "streaming"
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                    : "border-primary/30 bg-primary/5 text-primary",
                )}
              >
                <Loader2 className="h-3 w-3 animate-spin" />
                {status === "submitted" ? "Sending prompt" : "Streaming response"}
                {pendingCostRef.current !== null && (
                  <> · −{pendingCostRef.current} credit{pendingCostRef.current === 1 ? "" : "s"}</>
                )}
                {" · "}
                {getModelById(pendingPromptModelRef.current ?? modelId)?.label ?? modelId}
              </div>
            </div>
          )}

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
          paddingBottom: "calc(max(0.75rem, env(safe-area-inset-bottom)) + var(--kb-inset, 0px))",
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
          alt={`Attached image preview: ${file.name.replace(/\.[^.]+$/, "")}`}
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

const MAX_RICH_MARKDOWN_BYTES = 200_000;

const AssistantMarkdown = memo(function AssistantMarkdown({
  bubbleId,
  text,
}: {
  bubbleId: string;
  text: string;
}) {
  const deferred = useDeferredValue(text);
  const [html, setHtml] = useState<string>("");
  const tooLarge = deferred.length > MAX_RICH_MARKDOWN_BYTES;

  useEffect(() => {
    if (tooLarge) return;
    let cancelled = false;
    renderMarkdown(bubbleId, deferred).then((result) => {
      if (cancelled) return;
      // renderMarkdown resolves stale requests with "" — keep prior HTML in that case.
      if (result === "" && deferred.length > 0) return;
      setHtml(result);
    });
    return () => {
      cancelled = true;
    };
  }, [bubbleId, deferred, tooLarge]);

  // When the bubble unmounts (virtualizer scroll-off, conversation switch),
  // drop any pending worker request for this bubble.
  useEffect(() => {
    return () => cancelMarkdown(bubbleId);
  }, [bubbleId]);

  if (tooLarge) {
    return (
      <div>
        <div className="mb-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-[10px] font-medium text-amber-700 dark:text-amber-300">
          Rendering as plain text — response too large for rich formatting.
        </div>
        <pre className="whitespace-pre-wrap break-words text-sm">{deferred}</pre>
      </div>
    );
  }

  if (!html) {
    return <p className="whitespace-pre-wrap break-words">{deferred || "…"}</p>;
  }
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
});

const MessageBubble = memo(function MessageBubble({
  message,
  meta,
}: {
  message: UIMessage;
  meta?: { modelId: string; cost: number };
}) {
  const isUser = message.role === "user";
  const text = useMemo(
    () => message.parts.map((p) => (p.type === "text" ? p.text : "")).join(""),
    [message.parts],
  );
  const fileParts = useMemo(
    () =>
      message.parts.filter((p) => p.type === "file") as Array<{
        type: "file";
        mediaType?: string;
        url?: string;
        filename?: string;
      }>,
    [message.parts],
  );
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
                    alt={`Chat image attachment${p.filename ? `: ${p.filename.replace(/\.[^.]+$/, "")}` : ""}`}
                    className="max-h-48 rounded-lg border border-border/30"
                    loading="lazy"
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
            <p className="whitespace-pre-wrap break-words">{text}</p>
          ) : (
            <div className="prose prose-sm dark:prose-invert max-w-none break-words prose-pre:overflow-x-auto prose-pre:bg-muted prose-pre:text-foreground prose-code:before:content-none prose-code:after:content-none prose-code:rounded prose-code:bg-muted prose-code:px-1 prose-code:py-0.5">
              <AssistantMarkdown bubbleId={message.id} text={text} />
            </div>
          )}
        </div>
        {isUser && meta && (
          <div className="mt-1 self-end inline-flex items-center gap-1 rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold text-primary">
            <Zap className="h-2.5 w-2.5" />
            −{meta.cost} credit · {getModelById(meta.modelId)?.label ?? meta.modelId}
          </div>
        )}
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
}, (prev, next) => prev.message === next.message && prev.meta === next.meta);

export {};
