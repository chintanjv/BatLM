// Proves the TypeScript nano engine reproduces PyTorch's logits (and the BPE tokenizer its token ids).
// Run: npm run parity   (after python -m batlm.export_web)
import { readFileSync } from "node:fs";
import { build } from "esbuild";

const out = await build({ entryPoints: ["src/lib/nano.ts", "src/lib/bpe.ts"], bundle: false, write: false, format: "esm", outdir: "x" });
const load = async (i) => import("data:text/javascript;base64," + Buffer.from(out.outputFiles[i].text).toString("base64"));
const { NanoGPT } = await load(0);
const { BPE } = await load(1);

const dir = "public/models/nano";
const manifest = JSON.parse(readFileSync(`${dir}/manifest.json`));
const bin = readFileSync(`${dir}/weights.bin`);
const ref = JSON.parse(readFileSync(`${dir}/reference.json`));
const tok = new BPE(JSON.parse(readFileSync(`${dir}/tokenizer.json`)));

const ids = tok.encode("Batman is the secret identity of");
if (JSON.stringify(ids) !== JSON.stringify(ref.ids)) throw new Error(`tokenizer mismatch: ${ids} vs ${ref.ids}`);
const model = new NanoGPT(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const logits = model.prefill(ids);
let maxDiff = 0;
ref.logits.forEach((v, i) => (maxDiff = Math.max(maxDiff, Math.abs(v - logits[i]))));
console.log(`tokenizer ids match (${ids.length} tokens) · max |logit diff| = ${maxDiff.toExponential(2)}`);
if (maxDiff > 1e-3) throw new Error("PARITY FAILED");
console.log("PARITY OK");
