// Browser test page (dev only: /e2e.html). Drives the real worker and the real search, exactly as the app does,
// and publishes results on window.__e2e for scripts/e2e-browser.mjs to collect.
// The input is a list of conversations; each turn is asked with the answers before it as history.
import { plan, settle, type Knowledge } from "./lib/agent";
import type { Turn } from "./lib/prompt";
import { Index } from "./lib/search";
import type { WorkerOut } from "./worker";

type Conv = { id: string; turns: string[] };
type Row = { conv: string; turn: number; q: string; kind: string; answer: string; tokens?: number; ms?: number; capped?: boolean };
const w = window as unknown as {
  __questions?: Conv[];
  __e2e: { done: boolean; device?: string; dtype?: string; loadMs?: number; rows: Row[]; error?: string };
};
w.__e2e = { done: false, rows: [] };
const log = (s: string) => { document.getElementById("log")!.textContent += "\n" + s; };
const MAX_TOKENS = 160; // same cap as the app (hooks/useModel.ts)

(async () => {
  // With ?wait=1 the driver script hands over its own conversations once the page is up.
  if (location.search.includes("wait")) for (let i = 0; i < 600 && !w.__questions; i++) await new Promise((r) => setTimeout(r, 100));
  const convs: Conv[] = w.__questions ?? [
    { id: "demo-1", turns: ["what is the joker's real name?", "was he an arkham patient?"] },
    { id: "demo-2", turns: ["Who is Alfred?"] },
    { id: "demo-3", turns: ["What are the top 5 Batman villains?", "Catwoman is a villain? Are you sure?"] },
    { id: "demo-4", turns: ["How do I bake bread?"] },
  ];
  const d = await fetch("/data/index.json").then((r) => r.json());
  const knowledge: Knowledge = { index: new Index(d.passages), persona: d.persona, refusal: d.refusal, missing: d.missing ?? "I don't have that in my case files." };
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
  for (const c of convs) {
    const history: Turn[] = [];
    for (let n = 0; n < c.turns.length; n++) {
      const q = c.turns[n];
      const p = plan(knowledge, q, history);
      if (p.kind === "refuse") { w.__e2e.rows.push({ conv: c.id, turn: n, q, kind: "refuse", answer: p.answer }); continue; }
      let text = "";
      const stats = await new Promise<{ tokens: number; ms: number }>((res, rej) => {
        onMsg = (m) => {
          if (m.type === "token") text += m.text;
          if (m.type === "done") res(m);
          if (m.type === "error") rej(new Error(m.message));
        };
        worker.postMessage({ type: "generate", id: `${c.id}:${n}`, prompt: p.prompt, maxTokens: MAX_TOKENS });
      });
      const answer = settle(text.trim(), p.named, knowledge.missing);
      history.push({ q, a: answer, topic: p.topic });
      w.__e2e.rows.push({ conv: c.id, turn: n, q, kind: "generate", answer, tokens: stats.tokens, ms: stats.ms, capped: stats.tokens >= MAX_TOKENS });
      log(`${q} -> ${answer}`);
    }
  }
  w.__e2e.done = true;
})().catch((e) => { w.__e2e.error = String(e?.message ?? e); w.__e2e.done = true; log("ERROR " + w.__e2e.error); });
