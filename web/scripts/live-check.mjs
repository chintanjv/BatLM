// Drive the deployed site like a visitor: click the first suggestion and wait for the answer.
//   node scripts/live-check.mjs [https://batlm.vercel.app]
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const url = process.argv[2] ?? "https://batlm.vercel.app";
const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const chrome = spawn(CHROME, ["--headless=new", "--remote-debugging-port=9334", `--user-data-dir=${mkdtempSync(join(tmpdir(), "batlm-live-"))}`,
  "--no-first-run", "--enable-unsafe-webgpu", "--enable-features=WebGPU", "--use-angle=metal", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
try {
  let target;
  for (let i = 0; i < 50 && !target; i++) {
    await sleep(200);
    target = await fetch(`http://127.0.0.1:9334/json/new?${url}`, { method: "PUT" }).then((r) => r.json()).catch(() => null);
  }
  if (!target) throw new Error("could not reach Chrome on port 9334 (set CHROME to its path)");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0;
  const pending = new Map();
  ws.onmessage = (e) => { const m = JSON.parse(e.data); pending.get(m.id)?.(m); pending.delete(m.id); };
  const evaluate = (expression) => new Promise((res) => {
    pending.set(++id, (m) => res(m.result?.result?.value));
    ws.send(JSON.stringify({ id, method: "Runtime.evaluate", params: { expression, returnByValue: true } }));
  });
  await sleep(4000);
  const asked = await evaluate(`(() => { const b = document.querySelector('.chip'); b?.click(); return b?.textContent ?? null; })()`);
  if (!asked) throw new Error("no suggestion chip found: is this the BatLM site?");
  console.log("asked:", asked);
  const t0 = Date.now();
  let last = "";
  for (;;) {
    await sleep(3000);
    const s = JSON.parse(await evaluate(`JSON.stringify({ status: document.querySelector('.status')?.innerText, answer: document.querySelector('.msg.bot .ans')?.innerText ?? "",
      error: document.querySelector('.msg.bot .err')?.innerText ?? "", done: !!document.querySelector('.msg.bot .meta') })`));
    const line = `${s.status} | ${s.answer.slice(0, 100)}`;
    if (line !== last) { console.log(`${Math.round((Date.now() - t0) / 1000)}s | ${line}`); last = line; }
    if (s.error) { console.log("FAILED:", s.error); process.exitCode = 1; break; }
    if (s.done) { console.log("ANSWER:", s.answer); break; }
    if (Date.now() - t0 > 12 * 60_000) { console.log("FAILED: no answer after 12 minutes"); process.exitCode = 1; break; }
  }
  ws.close();
} finally {
  chrome.kill();
}
