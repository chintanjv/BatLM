"""Quantization eval: score an exported ONNX build (the exact weights the browser runs) on the eval set.

    .venv-export/bin/python tools/eval_onnx.py checkpoints/genai-q4 [--limit 30] [--name q4]

Runs with onnxruntime-genai (CPU), same prompts/decoding/scoring as batlm/evals.py, so the
numbers are directly comparable to the PyTorch fp32 rows — the gap is the cost of quantization.
"""
import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import onnxruntime_genai as og  # noqa: E402

from batlm import rag, sft_data  # noqa: E402
from batlm.evals import f1, is_refusal, key_hit, summarize  # noqa: E402
from batlm.prompt import render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--name", default="onnx")
    args = ap.parse_args()
    model = og.Model(args.model_dir)
    tok = og.Tokenizer(model)
    index, items = rag.load(), sft_data.eval_items()[: args.limit]
    rows = []
    for it in items:
        hits = [c for _, c in index.search(it["q"], k=3)]
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=2048, do_sample=False, repetition_penalty=1.1)
        gen = og.Generator(model, params)
        prompt_ids = tok.encode(render(it["q"], hits))
        gen.append_tokens(prompt_ids)
        t0, out = time.time(), []
        while not gen.is_done() and len(out) < 64:
            gen.generate_next_token()
            out.append(gen.get_next_tokens()[0])
        pred = tok.decode(out).replace("<|im_end|>", "").strip()
        oos = it["kind"] == "out_of_scope"
        rows.append({**it, "pred": pred, "latency_s": round(time.time() - t0, 2),
                     "correct": is_refusal(pred) if oos else (key_hit(pred, it["keys"]) and not is_refusal(pred)),
                     "f1": round(f1(pred, it["a"]), 3), "refused": is_refusal(pred)})
        print(f"[{args.name}] {'✓' if rows[-1]['correct'] else '✗'} {it['q'][:50]:50s} -> {pred[:90]}", flush=True)
    s = summarize(rows)
    print(json.dumps(s["overall"]))
    (ROOT / "evals").mkdir(exist_ok=True)
    json.dump(rows, open(ROOT / "evals" / f"rows_{args.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
