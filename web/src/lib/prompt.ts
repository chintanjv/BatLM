// Mirror of batlm/prompt.py + rag.format_context. The prompt must match training character for character.
import type { Passage } from "./search";

export function formatContext(hits: Passage[]): string {
  if (!hits.length) return "CASE FILES: (no matching files)";
  return "CASE FILES:\n" + hits.map((c, i) => `[${i + 1}] (${c.title}: ${c.section}) ${c.text}`).join("\n");
}

export function render(persona: string, question: string, hits: Passage[]): string {
  const msgs = [
    { role: "system", content: persona },
    { role: "user", content: `${formatContext(hits)}\n\nQUESTION: ${question}` },
  ];
  return msgs.map((m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join("") + "<|im_start|>assistant\n";
}
