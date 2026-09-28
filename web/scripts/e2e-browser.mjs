// Real-browser end-to-end test via the Chrome DevTools Protocol (no Puppeteer needed; Node 22+ has WebSocket).
//   npm run dev            # in another terminal (serves /e2e.html and /models/)
//   node scripts/e2e-browser.mjs [--webgpu] [base_url]
// Without --webgpu, Chrome runs headless with no GPU -> the WASM fallback path (dtype q4).
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const webgpu = process.argv.includes("--webgpu");
const base = process.argv.find((a) => a.startsWith("http")) ?? "http://localhost:5173";
const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const port = 9333;
const args = ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "batlm-e2e-"))}`,
  "--no-first-run", ...(webgpu ? ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-features=WebGPUService"] : ["--disable-gpu"]), "about:blank"];
const chrome = spawn(CHROME, args, { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

try {
  let target;
  for (let i = 0; i < 50 && !target; i++) {
    await sleep(200);
    target = await fetch(`http://127.0.0.1:${port}/json/new?${base}/e2e.html`, { method: "PUT" }).then((r) => r.json()).catch(() => null);
  }
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
    await sleep(3000);
    state = await evaluate("window.__e2e && JSON.stringify(window.__e2e)").then((s) => (s ? JSON.parse(s) : null));
    if (state?.done) break;
    if (Date.now() - t0 > 20 * 60_000) throw new Error("timed out after 20 min");
    process.stdout.write(`\r  waiting… ${Math.round((Date.now() - t0) / 1000)}s · answered ${state?.rows.length ?? 0}`);
  }
  console.log(`\n\nbackend: ${state.device}/${state.dtype} · load ${(state.loadMs / 1000).toFixed(1)}s`);
  if (state.error) { console.log("ERROR:", state.error); process.exitCode = 1; }
  for (const r of state.rows) {
    const perf = r.tokens ? ` [${r.tokens} tok, ${(r.tokens / (r.ms / 1000)).toFixed(1)} tok/s, ttft ${Math.round(r.ttftMs)}ms]` : ` [${r.kind}]`;
    console.log(`\n? ${r.q}\n  ${r.answer}${perf}`);
  }
  ws.close();
} finally {
  chrome.kill();
}
