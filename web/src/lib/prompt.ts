// Mirror of batlm/prompt.py + rag.format_context. Train/serve prompt skew is a silent killer.
import type { Chunk } from "./bm25";

export function formatContext(hits: Chunk[]): string {
  if (!hits.length) return "CASE FILES: (no matching files)";
  return "CASE FILES:\n" + hits.map((c, i) => `[${i + 1}] (${c.section}) ${c.text}`).join("\n");
}

export function render(persona: string, question: string, hits: Chunk[]): string {
  const msgs = [
    { role: "system", content: persona },
    { role: "user", content: `${formatContext(hits)}\n\nQUESTION: ${question}` },
  ];
  return msgs.map((m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join("") + "<|im_start|>assistant\n";
}
