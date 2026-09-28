"""Stage 9 — Package everything the browser needs.

    python -m batlm.export_web [--skip-batlm]

  web/public/data/      chunks.json (RAG index), facts.json (quiz + persona), report.json (evals)
  web/public/models/nano/        model.onnx + tokenizer.json + config.json   (~8 MB)
  web/public/models/batlm-360m/  transformers.js layout: config + tokenizer + onnx/model_{q4f16,q4}.onnx
                                 (dev only; in production the site loads it from the Hugging Face Hub)

Every export is checked for numerical parity against PyTorch before it is trusted.
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import torch

from . import sft_data

ROOT = pathlib.Path(__file__).resolve().parent.parent
PUB = ROOT / "web" / "public"
CK = ROOT / "checkpoints"


def export_data():
    d = PUB / "data"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "data" / "chunks.json", d / "chunks.json")
    json.dump({"persona": sft_data.PERSONA, "refusal": sft_data.REFUSAL,
               "facts": [{"q": f["q"], "a": f["a"], "keys": f["keys"]} for f in sft_data.F]},
              open(d / "facts.json", "w"))
    rep = ROOT / "evals" / "report.json"
    if rep.exists():
        shutil.copy(rep, d / "report.json")
    print("data ->", d)


def export_nano():
    """Raw float32 weights + a manifest, consumed by our own TypeScript engine (web/src/lib/nano.ts).
    Also writes reference logits so web/scripts/nano-parity.mjs can prove the JS engine == PyTorch."""
    from .model import GPT, GPTConfig
    from .tokenizer import BPETokenizer
    src, dst = CK / "nano", PUB / "models" / "nano"
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(src / "config.json"))
    model = GPT(GPTConfig(**cfg))
    model.load_state_dict(torch.load(src / "model.pt"))
    model.eval()
    tensors, offset = {}, 0
    with open(dst / "weights.bin", "wb") as f:
        for name, t in model.state_dict().items():
            if name == "head.weight":  # tied to wte.weight
                continue
            b = t.detach().float().contiguous().numpy().astype("<f4").tobytes()
            f.write(b)
            tensors[name] = {"offset": offset, "shape": list(t.shape)}
            offset += len(b)
    json.dump({"config": cfg, "tensors": tensors}, open(dst / "manifest.json", "w"))
    shutil.copy(src / "tokenizer.json", dst / "tokenizer.json")
    tok = BPETokenizer.load(src / "tokenizer.json")
    ids = tok.encode("Batman is the secret identity of")
    with torch.no_grad():
        logits = model(torch.tensor([ids]))[0, -1]
    json.dump({"ids": ids, "logits": [round(v, 5) for v in logits.tolist()]}, open(dst / "reference.json", "w"))
    print(f"nano -> {dst} ({offset / 1e6:.1f} MB weights)")


def export_batlm():
    """Build browser models with onnxruntime-genai's model builder (the same tool behind HF's official
    SmolLM2 ONNX files): 4-bit MatMulNBits weights + fused GroupQueryAttention.

      onnx/model_q4f16.onnx  fp16 activations/KV-cache  -> WebGPU
      onnx/model_q4.onnx     fp32 activations/KV-cache  -> WASM fallback

    The builder lives in a separate venv (.venv-export) and runs via tools/genai_build.py,
    which shims dtypes missing from torch 2.2.2 (the last Intel-Mac wheel).
    """
    import onnx

    src, dst = CK / "batlm-merged", PUB / "models" / "batlm-360m"
    py = ROOT / ".venv-export" / "bin" / "python"
    (dst / "onnx").mkdir(parents=True, exist_ok=True)
    for dtype, ep in (("q4f16", "webgpu"), ("q4", "cpu")):
        build = CK / f"genai-{dtype}"
        if not (build / "model.onnx").exists():
            # Quantization recipe chosen by tools/eval_onnx.py on v1 (25 Qs; fp32 = 84%):
            #   default RTN int4 (block 32) 56% · k_quant_mixed block 32 52% · k_quant_mixed block 16 72%  <- used
            # k_quant_mixed (from llama.cpp) keeps sensitive layers + lm_head at 8-bit; block 16 = finer scales.
            # WASM has no GatherBlockQuantized kernel (found by web/scripts/e2e-browser.mjs), so the CPU/WASM
            # build keeps a plain embedding table instead of sharing the quantized LM head (+170 MB, same accuracy).
            extra = ["int4_algo_config=k_quant_mixed", "int4_block_size=16"] + (["shared_embeddings=false"] if ep == "cpu" else [])
            subprocess.check_call([str(py), str(ROOT / "tools" / "genai_build.py"), "-i", str(src), "-o", str(build),
                                   "-p", "int4", "-e", ep, "-c", str(CK / "genai-cache"), "--extra_options", *extra])
        # Re-save as one self-contained file (< 2 GB protobuf limit) under the name transformers.js expects.
        model = onnx.load(str(build / "model.onnx"), load_external_data=True)
        onnx.save(model, str(dst / "onnx" / f"model_{dtype}.onnx"), save_as_external_data=False)
        print(f"  model_{dtype}.onnx: {(dst / 'onnx' / f'model_{dtype}.onnx').stat().st_size / 1e6:.0f} MB")
    for f in ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copy(src / f, dst / f)
    cfg = json.load(open(src / "config.json"))
    cfg["transformers.js_config"] = {"kv_cache_dtype": {"q4f16": "float16"}, "dtype": "q4f16"}
    json.dump(cfg, open(dst / "config.json", "w"), indent=1)
    print("batlm ->", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-batlm", action="store_true")
    args = ap.parse_args()
    export_data()
    export_nano()
    if not args.skip_batlm:
        export_batlm()


if __name__ == "__main__":
    main()
