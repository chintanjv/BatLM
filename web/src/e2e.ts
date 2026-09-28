// Browser end-to-end test page (dev only: /e2e.html). Drives the *real* worker + Oracle, exactly as the app does,
// and publishes results on window.__e2e for scripts/e2e-browser.mjs to collect.
import { Oracle } from "./lib/agent";
import { BM25 } from "./lib/bm25";
import type { WorkerOut } from "./worker";

type Row = { q: string; kind: string; answer: string; tokens?: number; ms?: number; ttftMs?: number };
const w = window as unknown as { __e2e: { done: boolean; device?: string; dtype?: string; loadMs?: number; rows: Row[]; error?: string } };
w.__e2e = { done: false, rows: [] };
const log = (s: string) => { document.getElementById("log")!.textContent += "\n" + s; };

const QUESTIONS = new URLSearchParams(location.search).get("q")?.split("|") ?? [
  "Who is Batman?", "Who broke Batman's back?", "What is inside the Batcave?", "Who is Superman's father?", "How do I bake bread?",
];

(async () => {
  const [chunks, f] = await Promise.all([fetch("/data/chunks.json").then((r) => r.json()), fetch("/data/facts.json").then((r) => r.json())]);
  const oracle = new Oracle(new BM25(chunks), f.facts, f.persona, f.refusal);
  const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  let onMsg: (m: WorkerOut) => void = () => {};
  worker.onmessage = (e: MessageEvent<WorkerOut>) => onMsg(e.data);
  const t0 = performance.now();
  await new Promise<void>((res, rej) => {
    onMsg = (m) => {
      if (m.type === "ready") { w.__e2e.device = m.device; w.__e2e.dtype = m.dtype; res(); }
      if (m.type === "error") rej(new Error(m.message));
    };
    worker.postMessage({ type: "load", model: "batlm" });
  });
  w.__e2e.loadMs = performance.now() - t0;
  log(`loaded ${w.__e2e.device}/${w.__e2e.dtype} in ${(w.__e2e.loadMs / 1000).toFixed(1)}s`);
  for (const q of QUESTIONS) {
    const plan = oracle.run(q);
    if (plan.kind !== "generate") { w.__e2e.rows.push({ q, kind: plan.kind, answer: "answer" in plan ? plan.answer : "" }); continue; }
    let text = "";
    const stats = await new Promise<{ tokens: number; ms: number; ttftMs: number }>((res, rej) => {
      onMsg = (m) => {
        if (m.type === "token") text += m.text;
        if (m.type === "done") res(m);
        if (m.type === "error") rej(new Error(m.message));
      };
      worker.postMessage({ type: "generate", model: "batlm", id: q, prompt: plan.prompt, maxTokens: 80 });
    });
    w.__e2e.rows.push({ q, kind: "generate", answer: text.trim(), tokens: stats.tokens, ms: stats.ms, ttftMs: stats.ttftMs });
    log(`${q} -> ${text.trim()}`);
  }
  w.__e2e.done = true;
})().catch((e) => { w.__e2e.error = String(e?.message ?? e); w.__e2e.done = true; log("ERROR " + w.__e2e.error); });
