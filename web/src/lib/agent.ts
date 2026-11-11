// Browser mirror of batlm/agent.py: search -> off-topic gate -> prompt.
import { render } from "./prompt";
import type { Hit, Index } from "./search";

/** Below this search score, with no Batman subject named in the question, the question is off-topic. */
export const MIN_SCORE = 10.0;

export type Plan =
  | { kind: "refuse"; answer: string }
  | { kind: "generate"; prompt: string; sources: Hit[] };

export type Knowledge = { index: Index; persona: string; refusal: string };

export function plan({ index, persona, refusal }: Knowledge, question: string): Plan {
  const sources = index.search(question, 3);
  if (!index.mentions(question).length && (sources[0]?.score ?? 0) < MIN_SCORE) return { kind: "refuse", answer: refusal };
  return { kind: "generate", prompt: render(persona, question, sources.map((h) => h.passage)), sources };
}
