"""Stage 8 — Package what the browser needs.

    python -m batlm.export_web [--skip-model]

  web/public/data/index.json       system prompt, refusal text and every search passage (cards + Wikipedia)
  web/public/models/batlm-360m/    transformers.js layout: config + tokenizer + onnx/model_{q4f16,q4}.onnx
                                   (local dev only; the live site loads the same files from the Hugging Face Hub)
"""
import argparse
import json
import pathlib
import shutil
import subprocess

from . import rag, sft_data

ROOT = pathlib.Path(__file__).resolve().parent.parent
PUB = ROOT / "web" / "public"
CK = ROOT / "checkpoints"


def export_data():
    d = PUB / "data"
    d.mkdir(parents=True, exist_ok=True)
    passages = rag.load().passages
    json.dump({"persona": sft_data.PERSONA, "refusal": sft_data.REFUSAL, "passages": passages},
              open(d / "index.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"data -> {d / 'index.json'} ({len(passages)} passages, {(d / 'index.json').stat().st_size / 1e6:.1f} MB)")


def export_model():
    """Build the browser models with onnxruntime-genai's model builder: 4-bit weights, fused attention.

      onnx/model_q4f16.onnx  16-bit activations  -> WebGPU
      onnx/model_q4.onnx     32-bit activations  -> CPU (WASM) fallback

    The builder lives in its own environment (.venv-export, see requirements-export.txt).
    Always rebuilds: a stale build would silently ship old weights.
    """
    import onnx

    src, dst = CK / "batlm-merged", PUB / "models" / "batlm-360m"
    py = ROOT / ".venv-export" / "bin" / "python"
    (dst / "onnx").mkdir(parents=True, exist_ok=True)
    for dtype, ep in (("q4f16", "webgpu"), ("q4", "cpu")):
        build = CK / f"genai-{dtype}"
        shutil.rmtree(build, ignore_errors=True)
        # k_quant_mixed keeps the most sensitive layers at 8-bit; block 16 gives finer scales. This recipe lost
        # the least accuracy of those tried. The CPU build keeps a plain embedding table because WASM has no
        # kernel for the shared quantized one.
        extra = ["int4_algo_config=k_quant_mixed", "int4_block_size=16"] + (["shared_embeddings=false"] if ep == "cpu" else [])
        subprocess.check_call([str(py), "-m", "onnxruntime_genai.models.builder", "-i", str(src), "-o", str(build),
                               "-p", "int4", "-e", ep, "-c", str(CK / "genai-cache"), "--extra_options", *extra])
        # Re-save as one self-contained file under the name transformers.js expects.
        model = onnx.load(str(build / "model.onnx"), load_external_data=True)
        onnx.save(model, str(dst / "onnx" / f"model_{dtype}.onnx"), save_as_external_data=False)
        print(f"  model_{dtype}.onnx: {(dst / 'onnx' / f'model_{dtype}.onnx').stat().st_size / 1e6:.0f} MB")
    for f in ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copy(src / f, dst / f)
    cfg = json.load(open(src / "config.json"))
    cfg["transformers.js_config"] = {"kv_cache_dtype": {"q4f16": "float16"}, "dtype": "q4f16"}
    json.dump(cfg, open(dst / "config.json", "w"), indent=1)
    shutil.copy(ROOT / "docs" / "MODEL_CARD.md", dst / "README.md")
    print("model ->", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    export_data()
    if not args.skip_model:
        export_model()


if __name__ == "__main__":
    main()
