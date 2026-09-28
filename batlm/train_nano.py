"""Stage 4 — Pretrain BatLM-nano from scratch on the Batman corpus (CPU-friendly).

    python -m batlm.train_nano [--iters 3000]

Writes checkpoints/nano/{tokenizer.json, model.pt, config.json, log.json}.
The last 10% of the corpus is held out so you can *see* overfitting in log.json:
with ~40KB of text, train loss keeps falling long after val loss bottoms out.
"""
import argparse
import json
import math
import pathlib
import time
from dataclasses import asdict

import torch

from .model import GPT, GPTConfig
from .tokenizer import BPETokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "checkpoints" / "nano"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--vocab", type=int, default=768)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    torch.manual_seed(1939)  # Detective Comics #27
    OUT.mkdir(parents=True, exist_ok=True)

    text = "\n".join((ROOT / "data" / "corpus" / f).read_text() for f in ("wikipedia.txt", "fandom.txt"))
    t0 = time.time()
    tok = BPETokenizer.train(text, vocab_size=args.vocab, verbose=True)
    tok.save(OUT / "tokenizer.json")
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"tokenizer: {tok.vocab_size} tokens, corpus {len(text)} chars -> {len(ids)} tokens "
          f"({len(text) / len(ids):.2f} chars/token) in {time.time() - t0:.0f}s")

    n = int(0.9 * len(ids))
    train, val = ids[:n], ids[n:]
    cfg = GPTConfig(vocab_size=tok.vocab_size)
    model = GPT(cfg)
    print(f"model: {model.num_params() / 1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

    def batch(split):
        data = train if split == "train" else val
        ix = torch.randint(len(data) - cfg.block_size - 1, (args.batch,))
        return (torch.stack([data[i:i + cfg.block_size] for i in ix]),
                torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix]))

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {s: sum(model(*batch(s))[1].item() for _ in range(20)) / 20 for s in ("train", "val")}
        model.train()
        return out

    log, best, warmup = [], float("inf"), 100
    t0 = time.time()
    for it in range(args.iters + 1):
        lr = args.lr * min(1, (it + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * it / args.iters))
        for g in opt.param_groups:
            g["lr"] = lr
        if it % 100 == 0:
            l = evaluate()
            log.append({"iter": it, **l, "lr": lr, "t": round(time.time() - t0, 1)})
            flag = ""
            if l["val"] < best:
                best, flag = l["val"], " *"
                torch.save(model.state_dict(), OUT / "model.pt")
            print(f"iter {it:5d} | train {l['train']:.3f} | val {l['val']:.3f} (ppl {math.exp(l['val']):.1f}) "
                  f"| lr {lr:.2e} | {time.time() - t0:.0f}s{flag}", flush=True)
        x, y = batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    json.dump(asdict(cfg), open(OUT / "config.json", "w"), indent=1)
    json.dump(log, open(OUT / "log.json", "w"), indent=1)
    model.load_state_dict(torch.load(OUT / "model.pt"))
    model.eval()
    for prompt in ["Batman is", "The Joker", "Bruce Wayne"]:
        out = model.generate(torch.tensor([tok.encode(prompt)]), 60)
        print(f"\n>>> {tok.decode(out[0].tolist())}")


if __name__ == "__main__":
    main()
