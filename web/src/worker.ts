// Inference runs off the main thread so the page stays responsive while the model works.
import { AutoModelForCausalLM, AutoTokenizer, InterruptableStoppingCriteria, TextStreamer, env } from "@huggingface/transformers";
import type { PreTrainedModel, PreTrainedTokenizer } from "@huggingface/transformers";
import ortMjs from "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs?url";
import ortWasm from "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm?url";

export type WorkerIn =
  | { type: "load"; forceWasm?: boolean }
  | { type: "generate"; id: string; prompt: string; maxTokens: number }
  | { type: "abort" };
export type WorkerOut =
  | { type: "progress"; file: string; loaded: number; total: number }
  | { type: "ready"; device: "webgpu" | "wasm"; dtype: string }
  | { type: "token"; id: string; text: string }
  | { type: "done"; id: string; tokens: number; ms: number }
  | { type: "gpu-failed"; message: string }
  | { type: "error"; loading: boolean; message: string };

// Dev server: local files (public/models/batlm-360m/). Production builds: the Hugging Face Hub repo.
// VITE_BATLM_MODEL overrides either, for testing a new fine-tune.
export const MODEL_ID: string = import.meta.env.VITE_BATLM_MODEL || (import.meta.env.DEV ? "batlm-360m" : "chintanjv/BatLM-360M");
const remote = MODEL_ID.includes("/");
env.allowLocalModels = !remote;
env.allowRemoteModels = remote;
env.localModelPath = "/models/";

// Serve the ONNX runtime from this site instead of a CDN, so the app has no third-party dependency at
// run time. (Older Safari without WebGPU needs a different build; it keeps the library's default.)
const wasm = env.backends.onnx.wasm;
const paths = wasm?.wasmPaths;
if (wasm && typeof paths === "object" && String(paths.mjs).includes(".asyncify."))
  wasm.wasmPaths = { mjs: new URL(ortMjs, self.location.href).href, wasm: new URL(ortWasm, self.location.href).href };

const post = (m: WorkerOut) => self.postMessage(m);
const stopper = new InterruptableStoppingCriteria();
let loaded: { tok: PreTrainedTokenizer; model: PreTrainedModel; eos: number } | null = null;

async function load(forceWasm = false) {
  // Fastest path the device supports: WebGPU with 16-bit floats (smallest download), WebGPU without, or CPU.
  type Adapter = { features: { has(f: string): boolean } };
  const gpu = (self.navigator as Navigator & { gpu?: { requestAdapter(): Promise<Adapter | null> } }).gpu;
  const adapter = gpu && !forceWasm ? await gpu.requestAdapter().catch(() => null) : null;
  const device = adapter ? "webgpu" : "wasm";
  const dtype = adapter?.features.has("shader-f16") ? "q4f16" : "q4";
  const progress_callback = (p: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (p.status === "progress" && p.file) post({ type: "progress", file: p.file, loaded: p.loaded ?? 0, total: p.total ?? 0 });
  };
  if (!remote) {
    // A missing local model makes static hosts return index.html (HTTP 200), which fails cryptically later.
    const r = await fetch(`/models/${MODEL_ID}/config.json`).catch(() => null);
    if (!r?.ok || !(r.headers.get("content-type") ?? "").includes("json"))
      throw new Error(`Model files not found at /models/${MODEL_ID}/. Run "make export", or set VITE_BATLM_MODEL to a Hub repo id.`);
  }
  const tok = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback });
  let model: PreTrainedModel;
  try {
    model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, { dtype, device, progress_callback });
    // Warm-up compiles the WebGPU shaders now, so the first answer is fast, and surfaces GPU faults early.
    await model.generate({ ...tok("hi"), max_new_tokens: 1 });
  } catch (err) {
    if (device !== "webgpu") throw err;
    // A lost WebGPU device poisons this worker's runtime: ask the page for a fresh CPU-only worker.
    post({ type: "gpu-failed", message: String((err as Error)?.message ?? err) });
    return new Promise<void>(() => {}); // this worker is about to be terminated
  }
  loaded = { tok, model, eos: tok.encode("<|im_end|>", { add_special_tokens: false })[0] };
  post({ type: "ready", device, dtype });
}

async function generate(id: string, prompt: string, maxTokens: number) {
  const { tok, model, eos } = loaded!;
  const t0 = performance.now();
  let tokens = 0;
  const streamer = new TextStreamer(tok, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text: string) => post({ type: "token", id, text }),
    token_callback_function: () => { tokens++; },
  });
  stopper.reset();
  await model.generate({
    ...tok(prompt, { add_special_tokens: false }),
    max_new_tokens: maxTokens,
    do_sample: false,
    repetition_penalty: 1.0, // answers repeat names and list markers from the files; a penalty garbles them
    eos_token_id: eos,
    streamer,
    stopping_criteria: stopper,
  });
  post({ type: "done", id, tokens, ms: performance.now() - t0 });
}

let loading: Promise<void> | null = null; // memoized: a second "load" never starts a second download
let queue: Promise<void> = Promise.resolve(); // one answer at a time, even if a new question arrives during a stop

self.onmessage = async (e: MessageEvent<WorkerIn>) => {
  const m = e.data;
  try {
    if (m.type === "load") await (loading ??= load(m.forceWasm).catch((err) => { loading = null; throw err; }));
    else if (m.type === "abort") stopper.interrupt();
    else {
      const job = queue.then(() => generate(m.id, m.prompt, m.maxTokens));
      queue = job.catch(() => {});
      await job;
    }
  } catch (err) {
    post({ type: "error", loading: m.type === "load", message: String((err as Error)?.message ?? err) });
  }
};
