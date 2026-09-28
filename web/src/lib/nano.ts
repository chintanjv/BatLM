// BatLM-nano inference engine — a GPT forward pass in plain TypeScript, with a KV cache.
// Mirrors batlm/model.py exactly; verified against PyTorch logits by web/scripts/nano-parity.mjs.

export type NanoConfig = { vocab_size: number; block_size: number; n_layer: number; n_head: number; n_embd: number };
type Manifest = { config: NanoConfig; tensors: Record<string, { offset: number; shape: number[] }> };

// Linear layer, PyTorch layout: W is [out, in], y = W x + b
function linear(x: Float32Array, W: Float32Array, b: Float32Array | null, nIn: number, nOut: number, out = new Float32Array(nOut)) {
  for (let o = 0; o < nOut; o++) {
    let s = b ? b[o] : 0;
    const row = o * nIn;
    for (let i = 0; i < nIn; i++) s += W[row + i] * x[i];
    out[o] = s;
  }
  return out;
}

function layerNorm(x: Float32Array, w: Float32Array, b: Float32Array) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2;
  const inv = 1 / Math.sqrt(v / n + 1e-5);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
  return out;
}

// erf via Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7) — PyTorch's nn.GELU uses the exact erf form.
function erf(x: number) {
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}
const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2));

export class NanoGPT {
  cfg: NanoConfig;
  private w: Record<string, Float32Array> = {};
  private kCache: Float32Array[];
  private vCache: Float32Array[];
  pos = 0;

  constructor(manifest: Manifest, buf: ArrayBuffer) {
    this.cfg = manifest.config;
    for (const [name, t] of Object.entries(manifest.tensors)) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      this.w[name] = new Float32Array(buf, t.offset, n);
    }
    const { n_layer, block_size, n_embd } = this.cfg;
    this.kCache = Array.from({ length: n_layer }, () => new Float32Array(block_size * n_embd));
    this.vCache = Array.from({ length: n_layer }, () => new Float32Array(block_size * n_embd));
  }

  reset() { this.pos = 0; }

  /** Feed one token at position this.pos; returns next-token logits. */
  step(token: number): Float32Array {
    const { n_embd: C, n_head, n_layer, vocab_size } = this.cfg;
    const hs = C / n_head, T = this.pos + 1, scale = 1 / Math.sqrt(hs);
    const W = this.w;
    let x = new Float32Array(C);
    for (let i = 0; i < C; i++) x[i] = W["wte.weight"][token * C + i] + W["wpe.weight"][this.pos * C + i];

    for (let l = 0; l < n_layer; l++) {
      const p = `blocks.${l}.`;
      // --- causal self-attention (only the new query; keys/values come from the cache) ---
      const h = layerNorm(x, W[p + "ln1.weight"], W[p + "ln1.bias"]);
      const qkv = linear(h, W[p + "attn.qkv.weight"], W[p + "attn.qkv.bias"], C, 3 * C);
      this.kCache[l].set(qkv.subarray(C, 2 * C), this.pos * C);
      this.vCache[l].set(qkv.subarray(2 * C, 3 * C), this.pos * C);
      const y = new Float32Array(C);
      const att = new Float32Array(T);
      for (let hd = 0; hd < n_head; hd++) {
        const o = hd * hs;
        let max = -Infinity;
        for (let t = 0; t < T; t++) {
          let s = 0;
          for (let i = 0; i < hs; i++) s += qkv[o + i] * this.kCache[l][t * C + o + i];
          att[t] = s * scale;
          if (att[t] > max) max = att[t];
        }
        let sum = 0;
        for (let t = 0; t < T; t++) { att[t] = Math.exp(att[t] - max); sum += att[t]; }
        for (let t = 0; t < T; t++) {
          const a = att[t] / sum;
          for (let i = 0; i < hs; i++) y[o + i] += a * this.vCache[l][t * C + o + i];
        }
      }
      const proj = linear(y, W[p + "attn.proj.weight"], W[p + "attn.proj.bias"], C, C);
      for (let i = 0; i < C; i++) x[i] += proj[i];
      // --- MLP ---
      const h2 = layerNorm(x, W[p + "ln2.weight"], W[p + "ln2.bias"]);
      const fc = linear(h2, W[p + "mlp.0.weight"], W[p + "mlp.0.bias"], C, 4 * C);
      for (let i = 0; i < fc.length; i++) fc[i] = gelu(fc[i]);
      const mo = linear(fc, W[p + "mlp.2.weight"], W[p + "mlp.2.bias"], 4 * C, C);
      for (let i = 0; i < C; i++) x[i] += mo[i];
    }
    x = layerNorm(x, W["ln_f.weight"], W["ln_f.bias"]);
    this.pos++;
    return linear(x, W["wte.weight"], null, C, vocab_size); // weight-tied head
  }

  /** Run a whole prompt; if it would overflow the context window, keep the most recent half. */
  prefill(tokens: number[]): Float32Array {
    this.reset();
    const keep = tokens.slice(-Math.floor(this.cfg.block_size / 2));
    let logits = new Float32Array(this.cfg.vocab_size);
    for (const t of keep) logits = this.step(t);
    return logits;
  }
}

export function sample(logits: Float32Array, temperature = 0.8, topK = 40): number {
  const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, topK);
  const max = logits[idx[0]];
  const p = idx.map((i) => Math.exp((logits[i] - max) / temperature));
  const sum = p.reduce((a, b) => a + b, 0);
  let r = Math.random() * sum;
  for (let i = 0; i < idx.length; i++) { r -= p[i]; if (r <= 0) return idx[i]; }
  return idx[0];
}
