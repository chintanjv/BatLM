"""Stage 7 — Evals: how often is the answer right, and where is quality lost?

    python -m batlm.evals                         # search + generation with checkpoints/batlm-merged
    python -m batlm.evals --questions             # write evals/questions.json for the browser run
    python -m batlm.evals --browser rows.json     # score answers produced by web/scripts/e2e-browser.mjs

  search      is the passage holding the answer among the three shown to the model?
  generation  an answer is correct when it contains every required fact, contains no forbidden claim,
              is not a refusal, and ended by itself before the token limit. Off-topic and unanswerable
              questions are correct when the model declines.

Writes evals/report_<run>.json and evals/rows_<run>.json (every question, answer and verdict).
"""
import argparse
import json
import pathlib
import time
from collections import defaultdict

from . import rag, sft_data
from .agent import hf_generate_fn, plan
from .cards import key_hit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evals"


def declined(pred: str) -> bool:
    p = pred.lower()
    return "outside my case files" in p or "don't have that" in p


def judge(it: dict, pred: str, capped: bool) -> bool:
    if it["kind"] in ("off_topic", "missing"):
        return declined(pred)
    return not capped and not declined(pred) and key_hit(pred, it["keys"]) and not (it.get("bad") and key_hit(pred, it["bad"]))


def search_eval(index, items) -> dict:
    """Share of answerable questions whose answer text (or, for owner questions, key facts) is in the top 3."""
    hit = total = 0
    for it in items:
        if it["kind"] in ("off_topic", "missing"):
            continue
        ctx = " ".join(c["text"] for _, c in index.search(it["q"], 3))
        total += 1
        hit += (it["a"] in ctx) if it["a"] else key_hit(ctx, it["keys"])
    return {"n": total, "recall@3": round(hit / total, 3)}


def summarize(rows) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    out = {k: {"n": len(v), "accuracy": round(sum(r["correct"] for r in v) / len(v), 3)} for k, v in by.items()}
    answerable = [r for r in rows if r["kind"] not in ("off_topic", "missing")]
    out["answerable"] = {"n": len(answerable), "accuracy": round(sum(r["correct"] for r in answerable) / len(answerable), 3)}
    out["overall"] = {"n": len(rows), "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3),
                      "hit_token_limit": sum(r["capped"] for r in rows),
                      "wrongly_declined": sum(declined(r["pred"]) for r in answerable)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "checkpoints" / "batlm-merged"))
    ap.add_argument("--questions", action="store_true")
    ap.add_argument("--browser", help="rows file from web/scripts/e2e-browser.mjs")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    index = rag.load()
    items = sft_data.eval_items(index)[:args.limit]
    if args.questions:
        json.dump([it["q"] for it in items], open(OUT / "questions.json", "w"), indent=1)
        print(f"{len(items)} questions -> {OUT / 'questions.json'}")
        return

    report = {"generated": time.strftime("%Y-%m-%d %H:%M"), "passages": len(index.passages), "search": search_eval(index, items)}
    print("search:", report["search"])
    rows = []
    if args.browser:
        run = json.load(open(args.browser))
        got = {r["q"]: r for r in run["rows"]}
        report["model"] = "browser build (4-bit ONNX), " + run.get("backend", "")
        for it in items:
            r = got[it["q"]]
            rows.append({**it, "pred": r["answer"], "capped": bool(r.get("capped"))})
    else:
        gen = hf_generate_fn(args.model)
        report["model"] = "fp32 PyTorch, " + pathlib.Path(args.model).name
        for it in items:
            prompt, _ = plan(index, it["q"])
            pred, capped = gen(prompt) if prompt else (sft_data.REFUSAL, False)
            rows.append({**it, "pred": pred, "capped": capped})
    for r in rows:
        r["correct"] = judge(r, r["pred"], r["capped"])
        print(f"{'✓' if r['correct'] else '✗'} [{r['kind']:9s}] {r['q'][:52]:52s} -> {r['pred'][:100]}", flush=True)
    report["results"] = summarize(rows)
    name = "browser" if args.browser else "pytorch"
    json.dump(rows, open(OUT / f"rows_{name}.json", "w"), indent=1, ensure_ascii=False)
    json.dump(report, open(OUT / f"report_{name}.json", "w"), indent=1)
    print(json.dumps(report["results"], indent=1))


if __name__ == "__main__":
    main()
