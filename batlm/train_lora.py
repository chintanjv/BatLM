"""Stage 5b — Instruction-tune SmolLM2-360M-Instruct into BatLM with LoRA (CPU).

    python -m batlm.train_lora [--epochs 3]

Each training example = system persona + retrieved CASE FILES + question -> grounded answer.
Loss is computed only on the answer tokens. If BM25 misses the gold passage for a training
question we swap it in ("oracle insertion"), so the model learns to *read* context rather than
memorize — and learns to refuse when the files are irrelevant.

Outputs checkpoints/batlm-lora/ (adapter) and checkpoints/batlm-merged/ (full weights for export).
"""
import argparse
import json
import math
import pathlib
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import rag, sft_data
from .rag import gold_chunk
from .prompt import BASE_MODEL, render

ROOT = pathlib.Path(__file__).resolve().parent.parent
CK = ROOT / "checkpoints"


def build_examples(index, seed=0, oos_extra=None):
    rng = random.Random(seed)
    keys_by_answer = {f["a"]: f["keys"] for f in sft_data.F}
    out = []
    for it in (sft_data.train_items() if oos_extra is None else sft_data.train_items(oos_extra)):
        hits = [c for _, c in index.search(it["q"], k=3)]
        if not it.get("oos"):
            keys = keys_by_answer[it["a"]]
            ctx = " ".join(c["text"] for c in hits).lower()
            if not all(k.lower() in ctx for k in keys):
                g = gold_chunk(index, keys)
                if g and g not in hits:
                    hits = hits[:2] + [g]
                    rng.shuffle(hits)
        out.append({"prompt": render(it["q"], hits), "answer": it["a"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--oos-extra", type=int, default=20, help="near-miss refusal examples to add (see sft_data)")
    args = ap.parse_args()
    torch.manual_seed(1939)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    examples = build_examples(rag.load(), oos_extra=args.oos_extra)
    data = []
    for ex in examples:
        p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        a = tok(ex["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        data.append((torch.tensor(p + a), torch.tensor([-100] * len(p) + a)))
    print(f"{len(data)} examples, mean {sum(len(x) for x, _ in data) / len(data):.0f} tokens")
    (CK / "sft_examples.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with open(CK / "sft_examples.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total = args.epochs * len(data) // args.accum
    step, log, t0 = 0, [], time.time()
    model.train()
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(data)
        running = 0.0
        for i, (x, y) in enumerate(data):
            loss = model(input_ids=x[None], labels=y[None]).loss / args.accum
            loss.backward()
            running += loss.item()
            if (i + 1) % args.accum == 0:
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / total))
                for g in opt.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                log.append({"step": step, "epoch": epoch, "loss": running, "lr": lr, "t": round(time.time() - t0, 1)})
                if step % 5 == 0:
                    print(f"epoch {epoch} step {step:4d}/{total} | loss {running:.3f} | lr {lr:.2e} | "
                          f"{time.time() - t0:.0f}s", flush=True)
                running = 0.0

    model.save_pretrained(CK / "batlm-lora")
    json.dump(log, open(CK / "batlm-lora" / "log.json", "w"), indent=1)
    merged = model.merge_and_unload()
    merged.save_pretrained(CK / "batlm-merged", safe_serialization=True)
    tok.save_pretrained(CK / "batlm-merged")
    print(f"done in {time.time() - t0:.0f}s -> {CK / 'batlm-merged'}")


if __name__ == "__main__":
    main()
