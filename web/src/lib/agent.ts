// ORACLE protocol — browser mirror of batlm/agent.py.
// PLAN (deterministic router) -> ACT (tools) -> GUARD (confidence gate) -> ANSWER (LLM) -> TRACE.
import { BM25, type Chunk, type Hit } from "./bm25";
import { render } from "./prompt";

export type Fact = { q: string[]; a: string; keys: string[] };
export type Step = { tool: string; input: string; ms: number; detail?: string; hits?: Hit[] };
export type Plan =
  | { kind: "quiz"; question: string; fact: Fact; trace: Step[] }
  | { kind: "grade"; correct: boolean; answer: string; trace: Step[] }
  | { kind: "refuse"; answer: string; trace: Step[] }
  | { kind: "generate"; prompt: string; evidence: Hit[]; trace: Step[] };

export const MIN_SCORE = 3.0;

export class Oracle {
  pendingQuiz: Fact | null = null;
  constructor(private index: BM25, private facts: Fact[], private persona: string, private refusal: string) {}

  private timeline(year: string): Hit[] {
    return this.index.chunks
      .filter((c) => c.text.includes(year))
      .slice(0, 3)
      .map((chunk) => ({ score: chunk.text.split(year).length - 1 + 5, chunk }));
  }

  run(msg: string): Plan {
    const m = msg.toLowerCase();
    const trace: Step[] = [];
    const time = <T,>(tool: string, input: string, fn: () => T, detail?: (r: T) => string): T => {
      const t0 = performance.now();
      const r = fn();
      trace.push({ tool, input, ms: +(performance.now() - t0).toFixed(1), detail: detail?.(r) });
      return r;
    };

    // PLAN
    let steps: [string, string][];
    const year = m.match(/\b(19[3-9]\d|20[0-2]\d)\b/);
    const cmp = m.match(/(?:compare|difference between)\s+(.+?)\s+(?:and|vs\.?|with)\s+(.+?)[?.]?$/);
    if (this.pendingQuiz) steps = [["grade", msg]];
    else if (/\b(quiz|test me|challenge me)\b/.test(m)) steps = [["quiz", ""]];
    else if (year && /\b(happen|in|during|year)\b/.test(m)) steps = [["timeline", year[1]], ["search_case_files", msg]];
    else if (cmp) steps = [["search_case_files", cmp[1]], ["search_case_files", cmp[2]]];
    else steps = [["search_case_files", msg]];
    trace.push({ tool: "planner", input: steps.map((s) => s[0]).join(" → ") + (steps[0][0] === "search_case_files" || steps[0][0] === "timeline" ? " → answer" : ""), ms: 0 });

    // ACT
    const evidence: Hit[] = [];
    for (const [tool, arg] of steps) {
      if (tool === "quiz") {
        const fact = this.facts[Math.floor(Math.random() * this.facts.length)];
        this.pendingQuiz = fact;
        time("quiz", "random case file", () => fact, (f) => f.q[0]);
        return { kind: "quiz", question: fact.q[0], fact, trace };
      }
      if (tool === "grade") {
        const fact = this.pendingQuiz!;
        this.pendingQuiz = null;
        const correct = time("grade", arg, () => fact.keys.every((k) => m.includes(k.toLowerCase())), (ok) => (ok ? "match" : "no match") + ` · keys: ${fact.keys.join(", ")}`);
        return { kind: "grade", correct, answer: fact.a, trace };
      }
      const hits = time(tool, arg, () => (tool === "timeline" ? this.timeline(arg) : this.index.search(arg, 3)), (h) => `${h.length} files`);
      trace[trace.length - 1].hits = hits;
      for (const h of hits) if (!evidence.some((e) => e.chunk.id === h.chunk.id)) evidence.push(h);
    }

    // GUARD
    const best = Math.max(0, ...evidence.map((e) => e.score));
    if (best < MIN_SCORE && !this.index.mentionsAnchor(msg)) {
      time("guardrail", `max score ${best.toFixed(1)} < ${MIN_SCORE}`, () => null, () => "refuse · LLM not called");
      return { kind: "refuse", answer: this.refusal, trace };
    }
    time("guardrail", best >= MIN_SCORE ? `max score ${best.toFixed(1)} ≥ ${MIN_SCORE}` : "domain anchor present", () => null, () => "pass");

    // ANSWER (prompt only — generation is streamed by the worker)
    const ctx: Chunk[] = [...evidence].sort((a, b) => b.score - a.score).slice(0, 3).map((h) => h.chunk);
    const prompt = render(this.persona, msg, ctx);
    trace.push({ tool: "batlm.generate", input: `${prompt.length} chars · ${ctx.length} files`, ms: 0 });
    return { kind: "generate", prompt, evidence: evidence.sort((a, b) => b.score - a.score).slice(0, 3), trace };
  }
}
