import { useEffect, useRef, useState } from "react";
import type { UIMessage } from "ai";
import { Download, Share2, FileText, FileType, ChevronDown } from "lucide-react";
import { toast } from "sonner";
import { downloadText, downloadWord, downloadPdf, downloadShareFile } from "@/lib/chat-export";

// Conversation-level Export / Share menu shown in the chat header.
export function ChatExportMenu({
  title,
  messages,
  category,
  modelId,
}: {
  title: string;
  messages: UIMessage[];
  category?: string;
  modelId?: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const disabled = !messages || messages.length === 0;

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const run = (fn: () => void, label: string) => {
    try {
      fn();
    } catch (e) {
      console.error(e);
      toast.error(`Could not ${label}`);
      return;
    }
    setOpen(false);
  };

  const items = [
    { icon: FileType, label: "Download PDF", onClick: () => run(() => downloadPdf(title, messages), "export PDF") },
    { icon: FileText, label: "Download Word (.doc)", onClick: () => run(() => downloadWord(title, messages), "export Word") },
    { icon: FileText, label: "Download Text (.txt)", onClick: () => run(() => downloadText(title, messages), "export text") },
    {
      icon: Share2,
      label: "Share chat (file)",
      onClick: () =>
        run(() => {
          downloadShareFile(title, messages, { category, model_id: modelId });
          toast.success("Shared file downloaded — send it to anyone; they can Import it into TirthoAI.");
        }, "share chat"),
    },
  ];

  return (
    <div className="relative shrink-0" ref={ref}>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 rounded-full border border-border bg-background px-2.5 py-1.5 text-xs font-semibold text-foreground transition hover:bg-accent hover:text-accent-foreground disabled:opacity-40"
        title="Export or share this chat"
        aria-label="Export or share this chat"
      >
        <Download className="h-3.5 w-3.5" />
        <span className="hidden sm:inline">Export</span>
        <ChevronDown className="h-3 w-3" />
      </button>
      {open && (
        <div className="absolute right-0 z-50 mt-1 w-52 overflow-hidden rounded-lg border border-border bg-popover py-1 shadow-lg">
          {items.map((it) => (
            <button
              key={it.label}
              type="button"
              onClick={it.onClick}
              className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-popover-foreground transition hover:bg-accent hover:text-accent-foreground"
            >
              <it.icon className="h-3.5 w-3.5 text-muted-foreground" />
              {it.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
