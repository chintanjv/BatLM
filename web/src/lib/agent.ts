// Browser mirror of batlm/agent.py: search -> off-topic gate -> prompt, one conversation turn at a time.
import { render, type Turn } from "./prompt";
import type { Hit, Index, Target } from "./search";

/** Below this search score, with no Batman subject named or carried over from the previous turn, the question is off-topic. */
export const MIN_SCORE = 16.0;
/** Passages shown to the model (must match BATLM_TOP_K at training time). */
export const TOP_K = Number(import.meta.env.VITE_BATLM_TOP_K) || 3;

export type Plan =
  | { kind: "refuse"; answer: string }
  /** topic: who this turn is about, for the next turn's follow-up. named: does the question itself name a Batman subject? */
  | { kind: "generate"; prompt: string; sources: Hit[]; topic: Target[]; named: boolean };

export type Knowledge = { index: Index; persona: string; refusal: string; missing: string };

export function plan({ index, persona, refusal }: Knowledge, question: string, history: Turn[] = []): Plan {
  const { named, carried } = index.subjects(question, history[history.length - 1]);
  const sources = index.search(question, TOP_K, carried);
  if (!named.length && !carried.length && (sources[0]?.score ?? 0) < MIN_SCORE) return { kind: "refuse", answer: refusal };
  return {
    kind: "generate", prompt: render(persona, question, sources.map((h) => h.passage), history), sources,
    topic: named.length ? [...carried.slice(0, 1), ...named] : carried.slice(0, 2), named: named.length > 0,
  };
}

/** A question that names a Batman subject is never "outside the Batman universe": at worst the files lack the answer. */
export function settle(answer: string, named: boolean, missing: string): string {
  return named && answer.toLowerCase().includes("outside my case files") ? missing : answer;
}
