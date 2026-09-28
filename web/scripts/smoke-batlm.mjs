// End-to-end smoke test of the *browser* pipeline, run in Node: our TS Oracle (BM25 + guardrail + prompt)
// feeding the exported ONNX model through transformers.js. Only the ONNX Runtime backend differs from the browser.
// Run: node scripts/smoke-batlm.mjs [q4|q4f16]
import { readFileSync } from "node:fs";
import { build } from "esbuild";
// Use the *browser* build (onnxruntime-web / WASM) — exactly what visitors without WebGPU run.
const { AutoModelForCausalLM, AutoTokenizer, env } = await import("../node_modules/@huggingface/transformers/dist/transformers.web.js");
console.log("onnx env keys", Object.keys(env.backends.onnx ?? {}));


const out = await build({ entryPoints: ["src/lib/agent.ts"], bundle: true, write: false, format: "esm", platform: "node" });
const { Oracle } = await import("data:text/javascript;base64," + Buffer.from(out.outputFiles[0].text).toString("base64"));
const out2 = await build({ entryPoints: ["src/lib/bm25.ts"], bundle: true, write: false, format: "esm", platform: "node" });
const { BM25 } = await import("data:text/javascript;base64," + Buffer.from(out2.outputFiles[0].text).toString("base64"));

const chunks = JSON.parse(readFileSync("public/data/chunks.json"));
const facts = JSON.parse(readFileSync("public/data/facts.json"));
const oracle = new Oracle(new BM25(chunks), facts.facts, facts.persona, facts.refusal);

env.allowRemoteModels = false;
env.localModelPath = "http://localhost:8765/models/"; // serve with: cd public && python3 -m http.server 8765
const dtype = process.argv[2] ?? "q4";
const tok = await AutoTokenizer.from_pretrained("batlm-360m");
const model = await AutoModelForCausalLM.from_pretrained("batlm-360m", { dtype, device: "wasm" });
const eos = tok.encode("<|im_end|>", { add_special_tokens: false })[0]; console.log("eos id", eos);

for (const q of ["Who is Batman?", "Who broke Batman's back?", "What is inside the Batcave?", "How do I bake bread?", "Who is Superman's father?"]) {
  const plan = oracle.run(q);
  if (plan.kind !== "generate") { console.log(`\n? ${q}\n  [${plan.kind}] ${plan.answer}`); continue; }
  const t0 = performance.now();
  const ids = await model.generate({ ...tok(plan.prompt, { add_special_tokens: false }), max_new_tokens: 64, do_sample: false, repetition_penalty: 1.1, eos_token_id: eos });
  const n = ids.dims[1];
  const text = tok.decode(ids.slice(null, [tok(plan.prompt, { add_special_tokens: false }).input_ids.dims[1], n]), { skip_special_tokens: true });
  console.log(`\n? ${q}\n  ${text.trim()}  (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
}
