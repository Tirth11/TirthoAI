// Chat export / share / clipboard helpers.
// Used by the per-message Copy button and the conversation Export/Share menu so
// pasted/downloaded content is clean (no literal ** or ## markers) and formatted.
import { marked } from "marked";
import type { UIMessage } from "ai";

const textOf = (m: UIMessage) =>
  (m.parts ?? []).map((p) => (p.type === "text" ? p.text : "")).join("");

// Render markdown → sanitized-ish HTML fragment (marked output; we control input).
export function markdownToHtml(md: string): string {
  return marked.parse(md ?? "", { async: false }) as string;
}

// Strip markdown to readable plain text — removes #, *, `, >, link syntax, etc.
export function markdownToPlainText(md: string): string {
  let t = md ?? "";
  t = t.replace(/```[\s\S]*?```/g, (b) => b.replace(/```[a-zA-Z]*\n?/g, "").replace(/```/g, "")); // code fences
  t = t.replace(/`([^`]+)`/g, "$1"); // inline code
  t = t.replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1"); // images → alt
  t = t.replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1 ($2)"); // links → text (url)
  t = t.replace(/^\s{0,3}#{1,6}\s+/gm, ""); // headings
  t = t.replace(/^\s{0,3}>\s?/gm, ""); // blockquotes
  t = t.replace(/(\*\*|__)(.*?)\1/g, "$2"); // bold
  t = t.replace(/(\*|_)(.*?)\1/g, "$2"); // italic
  t = t.replace(/~~(.*?)~~/g, "$2"); // strikethrough
  t = t.replace(/^\s*[-*+]\s+/gm, "• "); // bullet lists
  t = t.replace(/^\s*\d+\.\s+/gm, (m) => m.replace(/\s+$/, " ")); // numbered lists keep number
  t = t.replace(/^([-*_]\s*){3,}$/gm, "────────"); // hr
  return t.trim();
}

// Copy a single message: rich HTML (for Word/Docs) + clean plain text fallback.
export async function copyMessage(md: string): Promise<boolean> {
  const html = `<meta charset="utf-8">${markdownToHtml(md)}`;
  const plain = markdownToPlainText(md);
  try {
    if (typeof ClipboardItem !== "undefined" && navigator.clipboard?.write) {
      await navigator.clipboard.write([
        new ClipboardItem({
          "text/html": new Blob([html], { type: "text/html" }),
          "text/plain": new Blob([plain], { type: "text/plain" }),
        }),
      ]);
      return true;
    }
  } catch {
    /* fall through to plain text */
  }
  await navigator.clipboard.writeText(plain);
  return true;
}

// Build a single markdown document for an entire conversation.
export function conversationToMarkdown(title: string, messages: UIMessage[]): string {
  const lines: string[] = [`# ${title || "Conversation"}`, ""];
  for (const m of messages) {
    const who = m.role === "user" ? "You" : m.role === "assistant" ? "Assistant" : "System";
    lines.push(`## ${who}`, "", textOf(m).trim(), "");
  }
  return lines.join("\n");
}

// Full HTML document (used for Word + PDF export).
export function conversationToHtmlDoc(title: string, messages: UIMessage[]): string {
  const body = messages
    .map((m) => {
      const who = m.role === "user" ? "You" : m.role === "assistant" ? "Assistant" : "System";
      return `<section class="msg ${m.role}"><h3>${who}</h3>${markdownToHtml(textOf(m))}</section>`;
    })
    .join("\n");
  return `<!doctype html><html><head><meta charset="utf-8"><title>${escapeHtml(title)}</title>
<style>
  body{font-family:Segoe UI,Arial,sans-serif;color:#1a1a1a;max-width:760px;margin:24px auto;padding:0 16px;line-height:1.55;}
  h1{font-size:22px;border-bottom:2px solid #eee;padding-bottom:8px;}
  .msg{margin:18px 0;padding:12px 16px;border-radius:10px;border:1px solid #eee;}
  .msg.user{background:#f5f3ff;} .msg.assistant{background:#fafafa;}
  .msg h3{margin:0 0 6px;font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#7c3aed;}
  pre{background:#f4f4f5;padding:10px;border-radius:8px;overflow:auto;} code{font-family:Consolas,monospace;}
  table{border-collapse:collapse;} td,th{border:1px solid #ddd;padding:4px 8px;}
</style></head><body><h1>${escapeHtml(title || "Conversation")}</h1>${body}</body></html>`;
}

function escapeHtml(s: string): string {
  return (s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c] as string));
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const safeName = (s: string) => (s || "conversation").replace(/[^\w\-]+/g, "_").slice(0, 60);

// .txt — clean plain text.
export function downloadText(title: string, messages: UIMessage[]) {
  const md = conversationToMarkdown(title, messages);
  triggerDownload(new Blob([markdownToPlainText(md)], { type: "text/plain;charset=utf-8" }), `${safeName(title)}.txt`);
}

// .doc — Word opens this HTML-based document with formatting (no extra deps).
export function downloadWord(title: string, messages: UIMessage[]) {
  const html = conversationToHtmlDoc(title, messages);
  triggerDownload(new Blob(["﻿", html], { type: "application/msword" }), `${safeName(title)}.doc`);
}

// PDF — open a print window with the formatted doc; the browser's "Save as PDF"
// produces a clean PDF. Avoids bundling a heavy PDF library.
export function downloadPdf(title: string, messages: UIMessage[]) {
  const html = conversationToHtmlDoc(title, messages);
  const w = window.open("", "_blank");
  if (!w) return;
  w.document.open();
  w.document.write(html + `<script>window.onload=function(){setTimeout(function(){window.print();},250);};<\/script>`);
  w.document.close();
}

// ── Share / import: portable JSON a recipient can import into their TirthoAI ──
export interface SharedConversation {
  app: "tirthoai";
  type: "conversation";
  version: 1;
  title: string;
  category?: string;
  model_id?: string;
  exported_at: string;
  messages: Array<{ role: UIMessage["role"]; parts: UIMessage["parts"] }>;
}

export function buildSharePayload(
  title: string,
  messages: UIMessage[],
  meta?: { category?: string; model_id?: string },
): SharedConversation {
  return {
    app: "tirthoai",
    type: "conversation",
    version: 1,
    title: title || "Shared chat",
    category: meta?.category,
    model_id: meta?.model_id,
    exported_at: new Date().toISOString(),
    messages: messages.map((m) => ({ role: m.role, parts: m.parts })),
  };
}

export function downloadShareFile(
  title: string,
  messages: UIMessage[],
  meta?: { category?: string; model_id?: string },
) {
  const payload = buildSharePayload(title, messages, meta);
  triggerDownload(
    new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
    `${safeName(title)}.tirthoai.json`,
  );
}

// Validate + normalize an imported share file.
export function parseSharePayload(raw: string): SharedConversation {
  const obj = JSON.parse(raw);
  if (!obj || obj.app !== "tirthoai" || obj.type !== "conversation" || !Array.isArray(obj.messages)) {
    throw new Error("This file is not a valid TirthoAI shared chat.");
  }
  return obj as SharedConversation;
}
