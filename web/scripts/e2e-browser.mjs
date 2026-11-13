// Real-browser end-to-end test via the Chrome DevTools Protocol (no Puppeteer needed; Node 22+ has WebSocket).
//   npm run dev            # in another terminal (serves /e2e.html and /models/)
//   node scripts/e2e-browser.mjs [--cpu] [--questions file.json] [--out rows.json] [base_url]
// Default: headless Chrome on this machine's GPU (WebGPU). --cpu disables the GPU to test the WASM fallback.
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const flag = (name) => { const i = process.argv.indexOf(name); return i < 0 ? null : process.argv[i + 1]; };
const cpu = process.argv.includes("--cpu");
const base = process.argv.find((a) => a.startsWith("http")) ?? "http://localhost:5173";
const questions = flag("--questions") ? JSON.parse(readFileSync(flag("--questions"), "utf8")) : null;
const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const port = 9333;
const args = ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "batlm-e2e-"))}`,
  "--no-first-run", ...(process.env.CI ? ["--no-sandbox"] : []), ...(cpu ? ["--disable-gpu"] : ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--use-angle=metal"]), "about:blank"];
const chrome = spawn(CHROME, args, { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

try {
  let target;
  for (let i = 0; i < 50 && !target; i++) {
    await sleep(200);
    target = await fetch(`http://127.0.0.1:${port}/json/new?${base}/e2e.html${questions ? "?wait=1" : ""}`, { method: "PUT" }).then((r) => r.json()).catch(() => null);
  }
  if (!target) throw new Error(`could not reach Chrome on port ${port} (set CHROME to its path)`);
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0;
  const pending = new Map();
  ws.onmessage = (e) => { const m = JSON.parse(e.data); pending.get(m.id)?.(m); pending.delete(m.id); };
  const evaluate = (expression) => new Promise((res) => {
    pending.set(++id, (m) => res(m.result?.result?.value));
    ws.send(JSON.stringify({ id, method: "Runtime.evaluate", params: { expression, returnByValue: true } }));
  });
  const t0 = Date.now();
  let state;
  for (;;) {
    await sleep(2000);
    if (questions) await evaluate(`window.__questions ??= ${JSON.stringify(questions)}`); // repeated: the page may still be loading
    state = await evaluate("window.__e2e && JSON.stringify(window.__e2e)").then((s) => (s ? JSON.parse(s) : null));
    if (state?.done) break;
    const limit = Number(process.env.E2E_MINUTES) || 40;
    if (Date.now() - t0 > limit * 60_000) throw new Error(`timed out after ${limit} min`);
    if (!state && Date.now() - t0 > 60_000) throw new Error("the test page never started (is /e2e.html served? use the dev server)");
    if (process.env.CI) { if (Math.round((Date.now() - t0) / 2000) % 60 === 0) console.log(`  waiting… ${Math.round((Date.now() - t0) / 60000)} min · answered ${state?.rows.length ?? 0}`); }
    else process.stdout.write(`\r  waiting… ${Math.round((Date.now() - t0) / 1000)}s · answered ${state?.rows.length ?? 0}`);
  }
  const backend = `${state.device}/${state.dtype}`;
  console.log(`\n\nbackend: ${backend} · load ${(state.loadMs / 1000).toFixed(1)}s`);
  if (state.error) { console.log("ERROR:", state.error); process.exitCode = 1; }
  const timed = state.rows.filter((r) => r.tokens);
  if (timed.length) console.log(`speed: ${(timed.reduce((s, r) => s + r.tokens, 0) / (timed.reduce((s, r) => s + r.ms, 0) / 1000)).toFixed(1)} tokens/s over ${timed.length} answers · hit the token limit: ${state.rows.filter((r) => r.capped).length}`);
  if (flag("--out")) writeFileSync(flag("--out"), JSON.stringify({ backend, rows: state.rows }, null, 1));
  else for (const r of state.rows) console.log(`\n? ${r.q}\n  ${r.answer}${r.tokens ? ` [${r.tokens} tokens]` : ` [${r.kind}]`}`);
  ws.close();
} finally {
  chrome.kill();
}
