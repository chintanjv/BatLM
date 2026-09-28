"""Stage 7 — Evals. Measures every layer separately so you know *where* quality is lost.

    python -m batlm.evals [--configs base+rag ft+rag ft+oracle] [--limit N]

  retrieval   recall@k: do the top-k chunks contain the answer's key facts?
  generation  key-fact accuracy, token F1 vs reference, refusal accuracy on out-of-scope Qs
  ablations   base+rag   stock SmolLM2 with our retrieval (did fine-tuning help?)
              ft+rag     BatLM as deployed
              ft+oracle  BatLM given the gold passage (upper bound: isolates retrieval errors)
  nano        held-out perplexity of the from-scratch model

Writes evals/report.json (rendered on the website's DIAGNOSTICS panel).
"""
import argparse
import json
import math
import pathlib
import re
import time
from collections import Counter, defaultdict

import torch

from . import rag, sft_data
from .prompt import BASE_MODEL, render
from .rag import gold_chunk

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evals"


def norm_tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def f1(pred, ref):
    p, r = norm_tokens(pred), norm_tokens(ref)
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


NUMBER_WORDS = {"eight": "8", "five": "5", "forty": "40", "twenty-seven": "27"}


def key_hit(pred, keys):
    """All keys must appear. A key may list alternatives with '|'. Number words count as digits."""
    pred = pred.lower().replace(",", "")
    for w, d in NUMBER_WORDS.items():
        pred = pred.replace(w, d)
    return all(any(alt.lower().replace(",", "") in pred for alt in k.split("|")) for k in keys)


def is_refusal(pred):
    return "outside my case files" in pred.lower()


def retrieval_eval(index, items, ks=(1, 3, 5)):
    items = [i for i in items if i["kind"] != "out_of_scope"]
    out = {}
    for k in ks:
        hits = [key_hit(" ".join(c["text"] for _, c in index.search(i["q"], k)), i["keys"]) for i in items]
        out[f"recall@{k}"] = round(sum(hits) / len(hits), 3)
    return out


def nano_eval():
    from .model import GPT, GPTConfig
    from .tokenizer import BPETokenizer
    d = ROOT / "checkpoints" / "nano"
    if not (d / "model.pt").exists():
        return None
    tok = BPETokenizer.load(d / "tokenizer.json")
    model = GPT(GPTConfig(**json.load(open(d / "config.json"))))
    model.load_state_dict(torch.load(d / "model.pt"))
    model.eval()
    text = "\n".join((ROOT / "data" / "corpus" / f).read_text() for f in ("wikipedia.txt", "fandom.txt"))
    ids = torch.tensor(tok.encode(text))
    val = ids[int(0.9 * len(ids)):]
    bs, losses = model.cfg.block_size, []
    with torch.no_grad():
        for i in range(0, len(val) - bs - 1, bs):
            _, l = model(val[None, i:i + bs], val[None, i + 1:i + bs + 1])
            losses.append(l.item())
    log = json.load(open(d / "log.json"))
    return {"params": model.num_params(), "val_loss": round(sum(losses) / len(losses), 3),
            "val_ppl": round(math.exp(sum(losses) / len(losses)), 1), "vocab": tok.vocab_size,
            "curve": [{"iter": r["iter"], "train": round(r["train"], 3), "val": round(r["val"], 3)} for r in log]}


def gen_eval(config, index, items, limit=None):
    """Configs: base+rag | ft+rag | ft+oracle | agent (= ft+rag behind the ORACLE guardrail, i.e. what ships)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .agent import MIN_SCORE
    path = BASE_MODEL if config.startswith("base") else str(ROOT / "checkpoints" / "batlm-merged")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
    rows = []
    for it in items[:limit]:
        scored = index.search(it["q"], k=3)
        hits = [c for _, c in scored]
        if config.endswith("oracle") and it["kind"] != "out_of_scope":
            g = gold_chunk(index, it["keys"])
            # Gold passage first, padded with retrieved ones: training always showed 3 passages, and a lone
            # passage is out-of-distribution (v2 refused 51/77 when given only the gold chunk).
            hits = ([g] + [c for c in hits if c is not g])[:3] if g else hits
        t0 = time.time()
        if config == "agent" and (scored[0][0] if scored else 0) < MIN_SCORE and not index.mentions_anchor(it["q"]):
            pred = sft_data.REFUSAL  # guardrail: the LLM is never called
        else:
            ids = tok(render(it["q"], hits), return_tensors="pt", add_special_tokens=False).input_ids
            with torch.no_grad():
                out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=64, do_sample=False,
                                     repetition_penalty=1.1, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
            pred = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        oos = it["kind"] == "out_of_scope"
        rows.append({**it, "pred": pred, "latency_s": round(time.time() - t0, 2),
                     "correct": is_refusal(pred) if oos else (key_hit(pred, it["keys"]) and not is_refusal(pred)),
                     "f1": round(f1(pred, it["a"]), 3), "refused": is_refusal(pred)})
        print(f"[{config}] {'✓' if rows[-1]['correct'] else '✗'} {it['q'][:50]:50s} -> {pred[:90]}", flush=True)
    return summarize(rows), rows


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    summary = {k: {"n": len(v), "accuracy": round(sum(r["correct"] for r in v) / len(v), 3),
                   "f1": round(sum(r["f1"] for r in v) / len(v), 3)} for k, v in by.items()}
    summary["overall"] = {"n": len(rows), "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3),
                          "f1": round(sum(r["f1"] for r in rows) / len(rows), 3),
                          "false_refusals": sum(r["refused"] for r in rows if r["kind"] != "out_of_scope"),
                          "mean_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 2)}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=["base+rag", "ft+rag", "agent", "ft+oracle"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--reuse", nargs="*", default=[], help="configs whose saved rows to re-score instead of re-running")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    index, items = rag.load(), sft_data.eval_items()
    report = {"generated": time.strftime("%Y-%m-%d %H:%M"), "n_items": len(items),
              "retrieval": retrieval_eval(index, items), "nano": nano_eval(), "configs": {}}
    print("retrieval:", report["retrieval"])
    print("nano:", {k: v for k, v in (report["nano"] or {}).items() if k != "curve"})
    for c in args.configs:
        path = OUT / f"rows_{c.replace('+', '_')}.json"
        if c in args.reuse and path.exists():
            rows = json.load(open(path))
            summary = summarize(rows)
        else:
            summary, rows = gen_eval(c, index, items, args.limit)
            json.dump(rows, open(path, "w"), indent=1)
        report["configs"][c] = summary
        print(c, json.dumps(summary["overall"]))
        # Written after every config so a crash never loses finished work.
        json.dump(report, open(OUT / "report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
