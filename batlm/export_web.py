"""Stage 8 — Package what the browser needs.

    python -m batlm.export_web [--skip-model]

  web/public/data/index.json       system prompt, the two decline messages and every search passage (cards + Wikipedia)
  web/public/models/batlm-360m/    transformers.js layout: config + tokenizer + onnx/model_{q4f16,q4}.onnx
                                   (local dev only; the live site loads the same files from the Hugging Face Hub)
"""
import argparse
import inspect
import json
import os
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
    # `held` and `keys` exist for evals only: the browser must not ship or search them
    passages = [{k: v for k, v in p.items() if k not in ("held", "keys")} for p in rag.load().passages]
    json.dump({"persona": sft_data.PERSONA, "refusal": sft_data.REFUSAL, "missing": sft_data.MISSING, "passages": passages},
              open(d / "index.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"data -> {d / 'index.json'} ({len(passages)} passages, {(d / 'index.json').stat().st_size / 1e6:.1f} MB)")


def merge_adapter():
    """Rebuild the full fp32 weights from the LoRA adapter (what `cloud/kaggle.sh fetch` brings back)."""
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .prompt import BASE_MODEL
    # The cloud run may use a newer peft that writes settings this version does not know: keep the ones it does.
    saved = json.load(open(CK / "batlm-lora" / "adapter_config.json"))
    known = inspect.signature(LoraConfig.__init__).parameters
    config = LoraConfig(**{k: v for k, v in saved.items() if k in known and k != "peft_type"})
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32),
                                      str(CK / "batlm-lora"), config=config)
    model.merge_and_unload().save_pretrained(CK / "batlm-merged", safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(CK / "batlm-merged")
    print("merged adapter ->", CK / "batlm-merged")


def export_model(bits="attn+down", only=None):
    """Build the browser models with onnxruntime-genai's model builder: 4-bit weights, fused attention.

      onnx/model_q4f16.onnx  16-bit activations  -> WebGPU
      onnx/model_q4.onnx     32-bit activations  -> CPU (WASM) fallback

    The builder lives in its own environment (.venv-export, see requirements-export.txt).
    Always rebuilds: a stale build would silently ship old weights.
    """
    import onnx

    src, dst = CK / "batlm-merged", PUB / "models" / "batlm-360m"
    py = os.environ.get("BATLM_EXPORT_PYTHON") or ROOT / ".venv-export" / "bin" / "python"  # the cloud run uses one environment
    adapter, merged = CK / "batlm-lora" / "adapter_model.safetensors", src / "model.safetensors"
    if adapter.exists() and (not merged.exists() or merged.stat().st_mtime < adapter.stat().st_mtime):
        merge_adapter()  # the adapter is newer than the merged weights (it came from a cloud run)
    (dst / "onnx").mkdir(parents=True, exist_ok=True)
    external = {}  # builds too large for one file (over 2 GB) keep their weights in a second file
    for dtype, ep in (("q4f16", "webgpu"), ("q4", "cpu")):
        if only and dtype != only:
            continue
        build = CK / f"genai-{dtype}"
        shutil.rmtree(build, ignore_errors=True)
        # k_quant_mixed keeps the most sensitive layers at 8-bit; block 16 gives finer scales. This recipe lost
        # the least accuracy of those tried. The CPU build keeps a plain embedding table because WASM has no
        # kernel for the shared quantized one.
        extra = ["int4_algo_config=k_quant_mixed", "int4_block_size=16"] + (["shared_embeddings=false"] if ep == "cpu" else [])
        subprocess.check_call([str(py), "-m", "batlm.quant_build", "-i", str(src), "-o", str(build),
                               "-p", "int4", "-e", ep, "-c", str(CK / "genai-cache"), "--extra_options", *extra],
                              cwd=ROOT, env={**os.environ, "BATLM_8BIT": bits})
        # Re-save as one self-contained file under the name transformers.js expects.
        model = onnx.load(str(build / "model.onnx"), load_external_data=True)
        out = dst / "onnx" / f"model_{dtype}.onnx"
        try:
            onnx.save(model, str(out), save_as_external_data=False)
        except Exception as e:  # protobuf cannot write a file over 2 GB
            print(f"  model_{dtype}.onnx does not fit in one file ({type(e).__name__}); writing its weights beside it")
            out.unlink(missing_ok=True)
            model = onnx.load(str(build / "model.onnx"), load_external_data=True)
            onnx.save(model, str(out), save_as_external_data=True, all_tensors_to_one_file=True, location=f"model_{dtype}.onnx_data")
            external[f"model_{dtype}.onnx"] = True
        del model
        size = sum(f.stat().st_size for f in (dst / "onnx").glob(f"model_{dtype}.onnx*"))
        print(f"  model_{dtype}.onnx: {size / 1e6:.0f} MB")
    for f in ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        if (src / f).exists():  # newer transformers versions no longer write every one of these
            shutil.copy(src / f, dst / f)
    cfg = json.load(open(src / "config.json"))
    cfg["transformers.js_config"] = {"kv_cache_dtype": {"q4f16": "float16"}, "dtype": "q4f16", **({"use_external_data_format": external} if external else {})}
    json.dump(cfg, open(dst / "config.json", "w"), indent=1)
    shutil.copy(ROOT / "docs" / "MODEL_CARD.md", dst / "README.md")
    print("model ->", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--bits", default="attn+down", choices=["mixed", "attn", "attn+down", "all"], help="which layers stay 8-bit (see batlm/quant_build.py)")
    ap.add_argument("--only", choices=["q4f16", "q4"], help="build just one of the two browser models")
    args = ap.parse_args()
    export_data()
    if not args.skip_model:
        export_model(args.bits, args.only)


if __name__ == "__main__":
    main()
