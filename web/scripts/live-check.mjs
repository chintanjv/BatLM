// Drive the *deployed* app like a visitor: click a suggestion, wait for BatLM's answer.
//   node scripts/live-check.mjs https://batlm.vercel.app
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const url = process.argv[2] ?? "https://batlm.vercel.app";
const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const chrome = spawn(CHROME, ["--headless=new", "--remote-debugging-port=9334", `--user-data-dir=${mkdtempSync(join(tmpdir(), "batlm-live-"))}`,
  "--no-first-run", "--disable-gpu", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
try {
  let t;
  for (let i = 0; i < 50 && !t; i++) { await sleep(200); t = await fetch(`http://127.0.0.1:9334/json/new?${url}/?boot=0`, { method: "PUT" }).then((r) => r.json()).catch(() => null); }
  const ws = new WebSocket(t.webSocketDebuggerUrl); await new Promise((r) => (ws.onopen = r));
  let id = 0; const pend = new Map();
  ws.onmessage = (e) => { const m = JSON.parse(e.data); pend.get(m.id)?.(m); };
  const ev = (expression) => new Promise((res) => { pend.set(++id, (m) => res(m.result?.result?.value)); ws.send(JSON.stringify({ id, method: "Runtime.evaluate", params: { expression, returnByValue: true } })); });
  await sleep(6000);
  console.log("clicked:", await ev(`(() => { const b=[...document.querySelectorAll('.suggest button')].find(b=>b.textContent==='Who created Batman?'); b?.click(); return !!b; })()`));
  const t0 = Date.now(); let last = "";
  for (;;) {
    await sleep(4000);
    const s = JSON.parse(await ev(`JSON.stringify({card:[...document.querySelectorAll('.modelcard')][0]?.innerText, chip:document.querySelectorAll('.chip')[1]?.innerText, bot:[...document.querySelectorAll('.msg.bot .msg-body')].pop()?.innerText, toast:document.querySelector('.toast')?.innerText, streaming:!!document.querySelector('.caret'), stats:document.querySelector('.stats')?.innerText})`));
    const line = `${Math.round((Date.now() - t0) / 1000)}s | ${s.chip} | ${(s.card || "").split("\n").pop()} | ${s.bot?.slice(0, 90)}${s.toast ? " | TOAST: " + s.toast : ""}`;
    if (line.slice(line.indexOf("|")) !== last) { console.log(line); last = line.slice(line.indexOf("|")); }
    if (s.stats || /ERROR|FAULT/.test((s.card || "") + (s.bot || "")) || Date.now() - t0 > 12 * 60_000) { console.log("FINAL:", s.bot, "|", s.stats); break; }
  }
  ws.close();
} finally { chrome.kill(); }
