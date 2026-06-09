/// <reference lib="webworker" />
import { marked } from "marked";

type Req = { id: string; text: string };
type Res = { id: string; html: string; error?: string };

marked.setOptions({ gfm: true, breaks: true });

self.onmessage = (e: MessageEvent<Req>) => {
  const { id, text } = e.data ?? { id: "", text: "" };
  try {
    const html = marked.parse(text ?? "", { async: false }) as string;
    (self as unknown as Worker).postMessage({ id, html } satisfies Res);
  } catch (err) {
    (self as unknown as Worker).postMessage({
      id,
      html: "",
      error: err instanceof Error ? err.message : String(err),
    } satisfies Res);
  }
};
