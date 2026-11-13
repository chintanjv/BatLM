"""Stage 5 — Fine-tune the base model into BatLM with LoRA.

    python -m batlm.train_lora [--epochs 2] [--data checkpoints/sft_teacher.jsonl]

The base model is BATLM_BASE (default SmolLM2-360M-Instruct); any ChatML model works, which is how the
candidates of different sizes are trained on the same data.

Each example is: system prompt + the conversation so far + the CASE FILES search really returns for the
question + the question -> the answer. Loss is computed only on the answer tokens. Nothing is slipped
into the context by hand: a lookup whose answer is not in the retrieved files is dropped and reported,
because training on it would teach the model to answer from memory.

Examples come from the templates in batlm.sft_data and, when present, the teacher-written conversations
from batlm.sft_gen (which already carry the prompt the product would build).

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
from .agent import plan
from .prompt import BASE_MODEL

ROOT = pathlib.Path(__file__).resolve().parent.parent
CK = ROOT / "checkpoints"


def build_examples(index, extra: pathlib.Path | None = None):
    out, dropped = [], []
    for it in sft_data.train_items(index):
        p = plan(index, it["q"])
        if p.prompt is None:  # the off-topic gate answers this one; the model never sees it
            continue
        if it["kind"] in ("card", "wiki", "fact") and not any(it["a"] in c["text"] for c in p.hits):
            dropped.append(it["q"])
            continue
        out.append({"prompt": p.prompt, "answer": it["a"], "kind": it["kind"]})
    if extra and extra.exists():
        for line in extra.read_text().splitlines():
            e = json.loads(line)
            out.append({"prompt": e["prompt"], "answer": e["answer"], "kind": "teacher-" + e["kind"]})
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8, help="examples per forward pass")
    ap.add_argument("--accum", type=int, default=2, help="forward passes per optimizer step")
    ap.add_argument("--max-tokens", type=int, default=1536, help="longer examples are left out")
    ap.add_argument("--data", default=str(CK / "sft_teacher.jsonl"), help="teacher-written examples (used if the file exists)")
    ap.add_argument("--dry-run", action="store_true", help="build and report the examples, then stop")
    args = ap.parse_args()
    torch.manual_seed(1939)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    examples, dropped = build_examples(rag.load(), pathlib.Path(args.data))
    kinds = {}
    for e in examples:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"{len(examples)} examples {kinds}; dropped {len(dropped)} whose answer search did not return")
    for q in dropped[:40]:
        print("  dropped:", q)
    CK.mkdir(exist_ok=True)
    with open(CK / "sft_examples.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    if args.dry_run:
        return

    print("base model:", BASE_MODEL)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    pad, data, long = tok.convert_tokens_to_ids("<|im_end|>"), [], 0
    for ex in examples:
        p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        a = tok(ex["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p) + len(a) > args.max_tokens:
            long += 1
            continue
        data.append((p + a, [-100] * len(p) + a))
    print(f"mean {sum(len(x) for x, _ in data) / len(data):.0f} tokens per example, {long} left out as too long, device {device}")

    def batches(epoch: int) -> list[list[int]]:
        """Examples of similar length share a batch (little padding); the batches themselves come in random order."""
        order = list(range(len(data)))
        random.Random(epoch).shuffle(order)
        pool = args.batch * 64
        out = []
        for i in range(0, len(order), pool):
            group = sorted(order[i:i + pool], key=lambda j: len(data[j][0]))
            out += [group[j:j + args.batch] for j in range(0, len(group), args.batch)]
        random.Random(epoch + 1000).shuffle(out)
        return out

    def collate(idx: list[int]):
        n = max(len(data[j][0]) for j in idx)
        x = torch.full((len(idx), n), pad, dtype=torch.long)
        y = torch.full((len(idx), n), -100, dtype=torch.long)
        m = torch.zeros((len(idx), n), dtype=torch.long)
        for r, j in enumerate(idx):
            ids, labels = data[j]
            x[r, :len(ids)] = torch.tensor(ids)
            y[r, :len(ids)] = torch.tensor(labels)
            m[r, :len(ids)] = 1
        return x.to(device), y.to(device), m.to(device)

    amp = device == "cuda"  # half-precision forward passes: several times faster on a cloud GPU
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total = max(args.epochs * (len(batches(0)) // args.accum), 1)
    step, log, t0 = 0, [], time.time()
    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for i, idx in enumerate(batches(epoch)):
            x, y, m = collate(idx)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = model(input_ids=x, attention_mask=m, labels=y).loss / args.accum
            scaler.scale(loss).backward()
            running += loss.item()
            if (i + 1) % args.accum == 0:
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * min(step / total, 1.0)))
                for g in opt.param_groups:
                    g["lr"] = lr
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                log.append({"step": step, "epoch": epoch, "loss": running, "lr": lr, "t": round(time.time() - t0, 1)})
                if step % 50 == 0:
                    print(f"epoch {epoch} step {step:5d}/{total} | loss {running:.3f} | lr {lr:.2e} | {time.time() - t0:.0f}s", flush=True)
                running = 0.0

    model.save_pretrained(CK / "batlm-lora")
    json.dump(log, open(CK / "batlm-lora" / "log.json", "w"), indent=1)
    merged = model.merge_and_unload().to("cpu")
    merged.save_pretrained(CK / "batlm-merged", safe_serialization=True)
    tok.save_pretrained(CK / "batlm-merged")
    print(f"done in {time.time() - t0:.0f}s -> {CK / 'batlm-merged'}")


if __name__ == "__main__":
    main()
