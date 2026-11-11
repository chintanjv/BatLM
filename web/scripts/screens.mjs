// Render the running app at phone, foldable, tablet and desktop sizes and save screenshots (dev check).
//   npm run dev   # in another terminal
//   node scripts/screens.mjs <output dir> [base_url]
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const out = process.argv[2] ?? "screens";
const base = process.argv[3] ?? "http://localhost:5173";
const SIZES = [["fold-cover", 280, 653], ["phone-small", 360, 740], ["phone", 390, 844], ["phone-landscape", 844, 390],
  ["fold-open", 717, 512], ["tablet", 820, 1180], ["desktop", 1440, 900]];
const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const chrome = spawn(CHROME, ["--headless=new", "--remote-debugging-port=9335", `--user-data-dir=${mkdtempSync(join(tmpdir(), "batlm-screens-"))}`,
  "--no-first-run", "--enable-unsafe-webgpu", "--enable-features=WebGPU", "--use-angle=metal", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync(out, { recursive: true });
try {
  let target;
  for (let i = 0; i < 50 && !target; i++) {
    await sleep(200);
    target = await fetch(`http://127.0.0.1:9335/json/new?${base}`, { method: "PUT" }).then((r) => r.json()).catch(() => null);
  }
  if (!target) throw new Error("could not reach Chrome on port 9335 (set CHROME to its path)");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0;
  const pending = new Map();
  ws.onmessage = (e) => { const m = JSON.parse(e.data); pending.get(m.id)?.(m); pending.delete(m.id); };
  const send = (method, params = {}) => new Promise((res) => { pending.set(++id, (m) => res(m.result)); ws.send(JSON.stringify({ id, method, params })); });
  const evaluate = (expression) => send("Runtime.evaluate", { expression, returnByValue: true }).then((r) => r?.result?.value);
  const shoot = async (state) => {
    for (const [name, width, height] of SIZES) {
      await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 2, mobile: width < 900 });
      await send("Emulation.setTouchEmulationEnabled", { enabled: width < 900 });
      await sleep(500);
      const wide = await evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth || [...document.querySelectorAll('.app *')].some((e) => e.getBoundingClientRect().right > innerWidth + 1)");
      const { data } = await send("Page.captureScreenshot", { format: "png" });
      writeFileSync(join(out, `${state}-${name}.png`), Buffer.from(data, "base64"));
      console.log(`${state}-${name} ${width}x${height}${wide ? "  OVERFLOWS SIDEWAYS" : ""}`);
    }
  };
  await sleep(3000);
  await shoot("empty");
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
  await evaluate("document.querySelectorAll('.chip')[3].click()");
  for (let i = 0; i < 90 && !(await evaluate("!!document.querySelector('.msg.bot .meta')")); i++) await sleep(1000);
  await evaluate("document.querySelector('.srcbtn')?.click()");
  await sleep(600);
  await shoot("answer");
  ws.close();
} finally {
  chrome.kill();
}
