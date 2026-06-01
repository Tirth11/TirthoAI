import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Send, Sparkles, Loader2, Paperclip, X, Image as ImageIcon, FileText } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { ModelPicker } from "@/components/ModelPicker";
import { autoSelectModel, getModelById, CATEGORY_META } from "@/lib/models";
import { ChatDB, type DBConversation } from "@/lib/chat-db";

interface Props {
  conversation: DBConversation;
  onConversationChange: () => void | Promise<unknown>;
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

export function ChatWindow({ conversation, onConversationChange }: Props) {
  const [modelId, setModelId] = useState(conversation.model_id);
  const [autoMode, setAutoMode] = useState(true);
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<File[]>([]);
  const [initialMessages, setInitialMessages] = useState<UIMessage[] | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const persistedIdsRef = useRef<Set<string>>(new Set());

  // Load messages for this conversation
  useEffect(() => {
    let alive = true;
    setInitialMessages(null);
    persistedIdsRef.current = new Set();
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
  }, [conversation.id]);

  const transport = useMemo(() => new DefaultChatTransport({ api: "/api/chat" }), []);

  const { messages, sendMessage, status } = useChat({
    id: conversation.id,
    messages: initialMessages ?? [],
    transport,
    onError: (err) => toast.error(err.message || "Something went wrong"),
  });

  const isLoading = status === "submitted" || status === "streaming";

  // Persist new messages when streaming finishes
  useEffect(() => {
    if (status !== "ready") return;
    (async () => {
      for (const m of messages) {
        if (persistedIdsRef.current.has(m.id)) continue;
        // Only persist messages with non-empty content
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

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, status]);

  useEffect(() => {
    setModelId(conversation.model_id);
    setInput("");
    setAttachments([]);
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
    if (useModelId !== conversation.model_id) {
      ChatDB.updateConversation(conversation.id, { model_id: useModelId }).catch(console.error);
    }

    if (messages.length === 0) {
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

    await sendMessage(
      { text: finalText, files: fileList },
      { body: { modelId: useModelId } }
    );
  };

  const activeModel = getModelById(modelId);
  const cat = activeModel?.category ?? "general";
  const catMeta = CATEGORY_META[cat];

  const suggestions = [
    { text: "Explain transformers like I'm 12", icon: "🧠" },
    { text: "Write a React debounce hook with tests", icon: "💻" },
    { text: "A haiku about rainy Mumbai", icon: "✍️" },
    { text: "Ideas for a weekend side project", icon: "🚀" },
  ];

  if (initialMessages === null) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col bg-background">
      <header className="flex items-center justify-between border-b border-border bg-background/80 px-4 py-3 backdrop-blur sm:px-6">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-full bg-gradient-to-r px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-white",
              catMeta.color
            )}
          >
            {catMeta.icon} {catMeta.label}
          </span>
          <h1 className="truncate text-sm font-semibold text-foreground">{conversation.title}</h1>
        </div>
        <ModelPicker
          modelId={modelId}
          onChange={(id) => {
            setModelId(id);
            setAutoMode(false);
            ChatDB.updateConversation(conversation.id, { model_id: id }).catch(console.error);
          }}
          autoMode={autoMode}
          onAutoToggle={setAutoMode}
        />
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        <div className="mx-auto max-w-3xl space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center pt-8 pb-4 text-center">
              <div
                className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl text-white shadow-xl"
                style={{ background: "var(--gradient-primary)", boxShadow: "var(--shadow-glow)" }}
              >
                <Sparkles className="h-8 w-8" />
              </div>
              <h2 className="text-2xl font-bold tracking-tight">Welcome to TirthoAI</h2>
              <p className="mt-1 max-w-md text-sm text-muted-foreground">
                Chat with the best AI models — your history is saved automatically.
              </p>
              <div className="mt-8 grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
                {suggestions.map((s) => (
                  <button
                    key={s.text}
                    type="button"
                    onClick={() => submit(s.text)}
                    className="group flex items-start gap-3 rounded-xl border border-border bg-card px-4 py-3 text-left text-sm transition hover:border-primary/50 hover:shadow-md"
                  >
                    <span className="text-lg">{s.icon}</span>
                    <span className="text-foreground group-hover:text-primary">{s.text}</span>
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
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-border bg-background/80 px-4 py-4 backdrop-blur sm:px-6">
        <div className="mx-auto max-w-3xl">
          {attachments.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {attachments.map((f, i) => (
                <div key={i} className="flex items-center gap-2 rounded-lg border border-border bg-card px-2.5 py-1.5 text-xs">
                  {f.type.startsWith("image/") ? (
                    <ImageIcon className="h-3.5 w-3.5 text-emerald-500" />
                  ) : (
                    <FileText className="h-3.5 w-3.5 text-primary" />
                  )}
                  <span className="max-w-[180px] truncate">{f.name}</span>
                  <button
                    onClick={() => setAttachments((p) => p.filter((_, j) => j !== i))}
                    className="rounded p-0.5 hover:bg-accent"
                    aria-label="Remove"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
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
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
              placeholder="Message TirthoAI… (Shift+Enter for newline)"
              rows={1}
              className="flex-1 resize-none bg-transparent px-1 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || (!input.trim() && attachments.length === 0)}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-primary-foreground shadow-md transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
              style={{ background: "var(--gradient-primary)" }}
              aria-label="Send"
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </button>
          </form>
          <p className="mt-2 text-center text-[11px] text-muted-foreground">
            Saved to your account • {autoMode ? "Auto-picks the best model" : `Using ${activeModel?.label}`}
          </p>
        </div>
      </div>
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

  return (
    <div className={cn("flex gap-3", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div
          className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-white shadow-md"
          style={{ background: "var(--gradient-primary)" }}
        >
          <Sparkles className="h-4 w-4" />
        </div>
      )}
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm",
          isUser ? "text-primary-foreground" : "border border-border bg-card text-card-foreground"
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
    </div>
  );
}
