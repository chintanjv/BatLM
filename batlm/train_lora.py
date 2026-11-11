"""Stage 5 — Fine-tune SmolLM2-360M-Instruct into BatLM with LoRA.

    python -m batlm.train_lora [--epochs 2]

Each example is: system prompt + the CASE FILES search really returns for the question + the question
-> the answer. Loss is computed only on the answer tokens. Nothing is slipped into the context by hand:
an example whose answer is not in the retrieved files is dropped and reported, because training on
it would teach the model to answer from memory.

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
from .prompt import BASE_MODEL, render

ROOT = pathlib.Path(__file__).resolve().parent.parent
CK = ROOT / "checkpoints"


def build_examples(index):
    out, dropped = [], []
    for it in sft_data.train_items(index):
        hits = [c for _, c in index.search(it["q"], k=3)]
        if it["kind"] in ("card", "wiki") and not any(it["a"] in c["text"] for c in hits):
            dropped.append(it["q"])
            continue
        out.append({"prompt": render(it["q"], hits), "answer": it["a"], "kind": it["kind"]})
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="build and report the examples, then stop")
    args = ap.parse_args()
    torch.manual_seed(1939)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    examples, dropped = build_examples(rag.load())
    kinds = {k: sum(e["kind"] == k for e in examples) for k in ("card", "wiki", "missing", "off_topic")}
    print(f"{len(examples)} examples {kinds}; dropped {len(dropped)} whose answer search did not return")
    for q in dropped[:40]:
        print("  dropped:", q)
    CK.mkdir(exist_ok=True)
    with open(CK / "sft_examples.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    if args.dry_run:
        return

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    data = []
    for ex in examples:
        p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        a = tok(ex["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        data.append((torch.tensor(p + a), torch.tensor([-100] * len(p) + a)))
    print(f"mean {sum(len(x) for x, _ in data) / len(data):.0f} tokens per example, device {device}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total = args.epochs * len(data) // args.accum
    step, log, t0 = 0, [], time.time()
    model.train()
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(data)
        running = 0.0
        for i, (x, y) in enumerate(data):
            loss = model(input_ids=x[None].to(device), labels=y[None].to(device)).loss / args.accum
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
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step:4d}/{total} | loss {running:.3f} | lr {lr:.2e} | {time.time() - t0:.0f}s", flush=True)
                running = 0.0

    model.save_pretrained(CK / "batlm-lora")
    json.dump(log, open(CK / "batlm-lora" / "log.json", "w"), indent=1)
    merged = model.merge_and_unload().to("cpu")
    merged.save_pretrained(CK / "batlm-merged", safe_serialization=True)
    tok.save_pretrained(CK / "batlm-merged")
    print(f"done in {time.time() - t0:.0f}s -> {CK / 'batlm-merged'}")


if __name__ == "__main__":
    main()
