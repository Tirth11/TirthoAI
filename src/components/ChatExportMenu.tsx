import { useEffect, useRef, useState } from "react";
import type { UIMessage } from "ai";
import { Download, FileText, FileType, ChevronDown } from "lucide-react";
import { toast } from "sonner";
import { downloadText, downloadWord, downloadPdf } from "@/lib/chat-export";

// Conversation-level Export menu: clean, human-readable downloads (PDF / Word / Text).
// (Sharing a live, read-only LINK is handled separately by the Share button — see
// ChatWindow's bottom action bar — so we no longer dump a raw .json file here.)
export function ChatExportMenu({
  title,
  messages,
}: {
  title: string;
  messages: UIMessage[];
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
  ];

  return (
    <div className="relative shrink-0" ref={ref}>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 rounded-full border border-border bg-background px-3 py-1.5 text-xs font-semibold text-foreground transition hover:bg-accent hover:text-accent-foreground disabled:opacity-40"
        title="Export this chat"
        aria-label="Export this chat"
      >
        <Download className="h-3.5 w-3.5" />
        <span>Export</span>
        <ChevronDown className="h-3 w-3" />
      </button>
      {open && (
        <div className="absolute bottom-full right-0 z-50 mb-1 w-52 overflow-hidden rounded-lg border border-border bg-popover py-1 shadow-lg">
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
