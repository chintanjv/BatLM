// Mirror of batlm/prompt.py + rag.format_context. The prompt must match training character for character.
import type { Passage, Target } from "./search";

/** One answered exchange of the conversation. `topic` is who it was about, so a follow-up can inherit it. */
export type Turn = { q: string; a: string; topic: Target[] };

const HISTORY = 2;        // earlier exchanges shown to the model
const ANSWER_WORDS = 60;  // each earlier answer is cut to this many words

export function formatContext(hits: Passage[]): string {
  if (!hits.length) return "CASE FILES: (no matching files)";
  return "CASE FILES:\n" + hits.map((c, i) => `[${i + 1}] (${c.title}: ${c.section}) ${c.text}`).join("\n");
}

export function render(persona: string, question: string, hits: Passage[], history: Turn[] = []): string {
  const msgs = [{ role: "system", content: persona }];
  for (const t of history.slice(-HISTORY)) {
    msgs.push({ role: "user", content: `QUESTION: ${t.q}` });
    msgs.push({ role: "assistant", content: t.a.trim().split(/\s+/).slice(0, ANSWER_WORDS).join(" ") });
  }
  msgs.push({ role: "user", content: `${formatContext(hits)}\n\nQUESTION: ${question}` });
  return msgs.map((m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join("") + "<|im_start|>assistant\n";
}
