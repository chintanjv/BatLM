// Inference runs off the main thread so animations stay at 60fps while the model thinks.
import { AutoModelForCausalLM, AutoTokenizer, InterruptableStoppingCriteria, TextStreamer, env } from "@huggingface/transformers";
import type { PreTrainedModel, PreTrainedTokenizer } from "@huggingface/transformers";
import { BPE } from "./lib/bpe";
import { NanoGPT, sample } from "./lib/nano";

export type ModelKey = "batlm" | "nano";
export type WorkerIn =
  | { type: "load"; model: ModelKey }
  | { type: "generate"; model: ModelKey; id: string; prompt: string; maxTokens: number; temperature?: number }
  | { type: "abort" };
export type WorkerOut =
  | { type: "progress"; model: ModelKey; file: string; loaded: number; total: number }
  | { type: "ready"; model: ModelKey; device: string; dtype: string; params: string }
  | { type: "token"; id: string; text: string }
  | { type: "done"; id: string; tokens: number; ms: number; ttftMs: number }
  | { type: "error"; model?: ModelKey; message: string };

// Local path in dev ("batlm-360m" -> /models/batlm-360m/), or a Hub repo id in production.
const BATLM_ID: string = import.meta.env.VITE_BATLM_MODEL ?? "batlm-360m";
const NANO_BASE: string = import.meta.env.VITE_NANO_BASE ?? "/models/nano";
const remote = BATLM_ID.includes("/");
env.allowLocalModels = !remote;
env.allowRemoteModels = remote;
env.localModelPath = "/models/";

const post = (m: WorkerOut) => self.postMessage(m);

let batlm: { tok: PreTrainedTokenizer; model: PreTrainedModel } | null = null;
let nano: { tok: BPE; model: NanoGPT } | null = null;
const stopper = new InterruptableStoppingCriteria();
let nanoAbort = false;

async function loadBatlm() {
  if (batlm) return;
  const gpu = (self.navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  const hasGpu = !!(gpu && (await gpu.requestAdapter().catch(() => null)));
  const device = hasGpu ? "webgpu" : "wasm";
  const dtype = hasGpu ? "q4f16" : "q4";
  const progress_callback = (p: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (p.status === "progress" && p.file) post({ type: "progress", model: "batlm", file: p.file, loaded: p.loaded ?? 0, total: p.total ?? 0 });
  };
  const tok = await AutoTokenizer.from_pretrained(BATLM_ID, { progress_callback });
  const model = await AutoModelForCausalLM.from_pretrained(BATLM_ID, { dtype, device, progress_callback });
  // Warm-up compiles WebGPU shaders now, so the first real answer isn't slow.
  await model.generate({ ...tok("hi"), max_new_tokens: 1 });
  batlm = { tok, model };
  post({ type: "ready", model: "batlm", device, dtype, params: "362M" });
}

async function fetchWithProgress(url: string, file: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  const reader = res.body.getReader();
  const parts: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    post({ type: "progress", model: "nano", file, loaded, total: total || loaded });
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out.buffer;
}

async function loadNano() {
  if (nano) return;
  const [manifest, tokJson, buf] = await Promise.all([
    fetch(`${NANO_BASE}/manifest.json`).then((r) => r.json()),
    fetch(`${NANO_BASE}/tokenizer.json`).then((r) => r.json()),
    fetchWithProgress(`${NANO_BASE}/weights.bin`, "weights.bin"),
  ]);
  const model = new NanoGPT(manifest, buf);
  nano = { tok: new BPE(tokJson), model };
  const n = Object.values(manifest.tensors as Record<string, { shape: number[] }>)
    .reduce((s, t) => s + t.shape.reduce((a, b) => a * b, 1), 0);
  post({ type: "ready", model: "nano", device: "ts-engine", dtype: "fp32", params: `${(n / 1e6).toFixed(2)}M` });
}

async function generateBatlm(id: string, prompt: string, maxTokens: number) {
  const { tok, model } = batlm!;
  const t0 = performance.now();
  let first = 0, count = 0;
  const streamer = new TextStreamer(tok, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text: string) => post({ type: "token", id, text }),
    token_callback_function: () => { if (!count++) first = performance.now(); },
  });
  stopper.reset();
  await model.generate({
    ...tok(prompt, { add_special_tokens: false }),
    max_new_tokens: maxTokens,
    do_sample: false,
    repetition_penalty: 1.1,
    eos_token_id: tok.encode("<|im_end|>", { add_special_tokens: false })[0],
    streamer,
    stopping_criteria: stopper,
  });
  post({ type: "done", id, tokens: count, ms: performance.now() - t0, ttftMs: first - t0 });
}

async function generateNano(id: string, prompt: string, maxTokens: number, temperature = 0.8) {
  const { tok, model } = nano!;
  const t0 = performance.now();
  let first = 0;
  const ids = tok.encode(prompt);
  let logits = model.prefill(ids);
  const decoder = new TextDecoder("utf-8");
  nanoAbort = false;
  let n = 0;
  for (; n < maxTokens && !nanoAbort; n++) {
    const next = sample(logits, temperature);
    if (next === tok.eot) break;
    if (!n) first = performance.now();
    const text = decoder.decode(tok.bytes(next), { stream: true });
    if (text) post({ type: "token", id, text });
    ids.push(next);
    logits = model.pos >= model.cfg.block_size ? model.prefill(ids) : model.step(next);
    if (n % 4 === 3) await new Promise((r) => setTimeout(r, 0)); // let "abort" messages in
  }
  post({ type: "done", id, tokens: n, ms: performance.now() - t0, ttftMs: first - t0 });
}

// Memoized so duplicate "load" messages never trigger a second 270 MB download.
const loading: Partial<Record<ModelKey, Promise<void>>> = {};

self.onmessage = async (e: MessageEvent<WorkerIn>) => {
  const m = e.data;
  try {
    if (m.type === "load") {
      loading[m.model] ??= (m.model === "batlm" ? loadBatlm() : loadNano()).catch((err) => { delete loading[m.model]; throw err; });
      await loading[m.model];
    }
    else if (m.type === "abort") { stopper.interrupt(); nanoAbort = true; }
    else if (m.type === "generate") {
      if (m.model === "batlm") await generateBatlm(m.id, m.prompt, m.maxTokens);
      else await generateNano(m.id, m.prompt, m.maxTokens, m.temperature);
    }
  } catch (err) {
    post({ type: "error", model: m.type === "load" ? m.model : undefined, message: String((err as Error)?.message ?? err) });
  }
};
