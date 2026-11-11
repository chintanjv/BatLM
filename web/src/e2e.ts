// Browser test page (dev only: /e2e.html). Drives the real worker and the real search, exactly as the app does,
// and publishes results on window.__e2e for scripts/e2e-browser.mjs to collect.
import { plan, type Knowledge } from "./lib/agent";
import { Index } from "./lib/search";
import type { WorkerOut } from "./worker";

type Row = { q: string; kind: string; answer: string; tokens?: number; ms?: number; capped?: boolean };
const w = window as unknown as {
  __questions?: string[];
  __e2e: { done: boolean; device?: string; dtype?: string; loadMs?: number; rows: Row[]; error?: string };
};
w.__e2e = { done: false, rows: [] };
const log = (s: string) => { document.getElementById("log")!.textContent += "\n" + s; };
const MAX_TOKENS = 160; // same cap as the app (hooks/useModel.ts)

(async () => {
  // The driver script may hand over its own question list; give it a moment to do so.
  for (let i = 0; i < 20 && !w.__questions; i++) await new Promise((r) => setTimeout(r, 100));
  const questions = w.__questions ?? ["who is the batman?", "What is the Joker's real identity?", "Who is Alfred?", "What are the top 5 Batman villains?", "How do I bake bread?"];
  const d = await fetch("/data/index.json").then((r) => r.json());
  const knowledge: Knowledge = { index: new Index(d.passages), persona: d.persona, refusal: d.refusal };
  let onMsg: (m: WorkerOut) => void = () => {};
  const spawn = () => {
    const wk = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    wk.onmessage = (e: MessageEvent<WorkerOut>) => onMsg(e.data);
    return wk;
  };
  let worker = spawn();
  const t0 = performance.now();
  await new Promise<void>((res, rej) => {
    onMsg = (m) => {
      if (m.type === "ready") { w.__e2e.device = m.device; w.__e2e.dtype = m.dtype; res(); }
      if (m.type === "gpu-failed") { // same recovery as the app: replace the poisoned worker with a CPU-only one
        log("gpu-failed -> CPU worker: " + m.message.slice(0, 120));
        worker.terminate();
        worker = spawn();
        worker.postMessage({ type: "load", forceWasm: true });
      }
      if (m.type === "error") rej(new Error(m.message));
    };
    worker.postMessage({ type: "load" });
  });
  w.__e2e.loadMs = performance.now() - t0;
  log(`loaded ${w.__e2e.device}/${w.__e2e.dtype} in ${(w.__e2e.loadMs / 1000).toFixed(1)}s`);
  for (const q of questions) {
    const p = plan(knowledge, q);
    if (p.kind === "refuse") { w.__e2e.rows.push({ q, kind: "refuse", answer: p.answer }); continue; }
    let text = "";
    const stats = await new Promise<{ tokens: number; ms: number }>((res, rej) => {
      onMsg = (m) => {
        if (m.type === "token") text += m.text;
        if (m.type === "done") res(m);
        if (m.type === "error") rej(new Error(m.message));
      };
      worker.postMessage({ type: "generate", id: q, prompt: p.prompt, maxTokens: MAX_TOKENS });
    });
    w.__e2e.rows.push({ q, kind: "generate", answer: text.trim(), tokens: stats.tokens, ms: stats.ms, capped: stats.tokens >= MAX_TOKENS });
    log(`${q} -> ${text.trim()}`);
  }
  w.__e2e.done = true;
})().catch((e) => { w.__e2e.error = String(e?.message ?? e); w.__e2e.done = true; log("ERROR " + w.__e2e.error); });
