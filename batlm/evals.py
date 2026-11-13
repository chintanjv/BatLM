"""Stage 7 — Evals: how often is the answer right, and where is quality lost?

    python -m batlm.evals                         # search + generation with checkpoints/batlm-merged
    python -m batlm.evals --search                # search only: no model needed
    python -m batlm.evals --questions             # write evals/questions.json for the browser run
    python -m batlm.evals --browser rows.json     # score answers produced by web/scripts/e2e-browser.mjs

The eval is a list of conversations. Most have one turn; the hand-written ones in evals/handwritten.json
include follow-ups, where each turn is asked with the model's own earlier answers as history, as on the site.

  search      is the passage holding the answer among those shown to the model?
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
from .agent import TOP_K, hf_generate_fn, plan, respond
from .cards import key_hit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evals"
DECLINE = ("off_topic", "near_topic", "missing")  # kinds where the right answer is to decline
KINDS = ("lookup", "yesno", "challenge", "followup", "compare", "list", "multipart", "why", "typo") + DECLINE


def handwritten() -> list[dict]:
    """evals/handwritten.json: conversations written by hand, never generated and never trained on.
    A group {"kind", "items": [[question, keys, forbidden?], ...]} is shorthand for one-turn conversations."""
    convs = []
    for n, c in enumerate(json.loads((OUT / "handwritten.json").read_text())):
        if "items" in c:
            for m, it in enumerate(c["items"]):
                convs.append({"id": f"hand-{c['kind']}-{m}", "turns": [
                    {"q": it[0], "keys": it[1] if len(it) > 1 else "", "bad": it[2] if len(it) > 2 else "", "kind": c["kind"]}]})
        else:
            convs.append({"id": c.get("id", f"hand-talk-{n}"), "turns": c["turns"]})
    for c in convs:
        c["turns"] = [{"a": "", "keys": "", "bad": "", **t} for t in c["turns"]]
    return convs


def conversations(index) -> list[dict]:
    return [{"id": f"auto-{i}", "turns": [{"bad": "", **it}]} for i, it in enumerate(sft_data.eval_items(index))] + handwritten()


def declined(pred: str) -> bool:
    p = pred.lower()
    return "outside my case files" in p or "don't have that" in p


def judge(it: dict, pred: str, capped: bool) -> bool:
    if it["kind"] in DECLINE:
        return declined(pred)
    return not capped and not declined(pred) and key_hit(pred, it["keys"]) and not (it.get("bad") and key_hit(pred, it["bad"]))


def search_eval(index, convs) -> dict:
    """Share of answerable questions whose answer text (or, where no answer text is given, key facts) is among
    the passages shown. History here uses the reference answer, since no model is involved."""
    by, misses, gated, leaked, off = defaultdict(lambda: [0, 0]), [], [], 0, 0
    for c in convs:
        history = []
        for t in c["turns"]:
            p = plan(index, t["q"], history)
            if t["kind"] in DECLINE:
                off += t["kind"] == "off_topic"
                leaked += t["kind"] == "off_topic" and p.prompt is not None
            else:
                ctx = " ".join(h["text"] for h in p.hits)
                ok = (t["a"] in ctx) if t["a"] else key_hit(ctx, t["keys"])
                by[t["kind"]][0] += ok
                by[t["kind"]][1] += 1
                if p.prompt is None:
                    gated.append(t["q"])
                elif not ok:
                    misses.append(t["q"])
            if p.prompt is not None:
                history.append({"q": t["q"], "a": t["a"] or t["keys"], "topic": p.topic})
    hit, total = sum(v[0] for v in by.values()), sum(v[1] for v in by.values())
    train_off = sum(plan(index, q).prompt is not None for q in sft_data.OFF_TOPIC)
    return {"n": total, f"recall@{TOP_K}": round(hit / total, 3),
            "by_kind": {k: {"n": v[1], "recall": round(v[0] / v[1], 3)} for k, v in sorted(by.items())},
            "answerable_refused_by_gate": gated, "off_topic_reaching_model": f"{leaked} of {off}",
            "training_off_topic_reaching_model": f"{train_off} of {len(sft_data.OFF_TOPIC)}", "misses": misses}


def summarize(rows) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    out = {k: {"n": len(v), "accuracy": round(sum(r["correct"] for r in v) / len(v), 3)} for k, v in sorted(by.items())}
    answerable = [r for r in rows if r["kind"] not in DECLINE]
    hand = [r for r in rows if r["conv"].startswith("hand-")]
    out["handwritten"] = {"n": len(hand), "accuracy": round(sum(r["correct"] for r in hand) / max(len(hand), 1), 3)}
    out["answerable"] = {"n": len(answerable), "accuracy": round(sum(r["correct"] for r in answerable) / len(answerable), 3)}
    out["overall"] = {"n": len(rows), "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3),
                      "hit_token_limit": sum(r["capped"] for r in rows),
                      "wrongly_declined": sum(declined(r["pred"]) for r in answerable),
                      "batman_question_called_off_topic": sum("outside my case files" in r["pred"].lower() for r in answerable)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "checkpoints" / "batlm-merged"))
    ap.add_argument("--search", action="store_true", help="score search only; no model needed")
    ap.add_argument("--questions", action="store_true")
    ap.add_argument("--browser", help="rows file from web/scripts/e2e-browser.mjs")
    ap.add_argument("--name", help="name of this run, used in the output file names")
    ap.add_argument("--only", choices=["auto", "hand"], help="run only the generated or only the hand-written conversations")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    index = rag.load()
    convs = [c for c in conversations(index) if not args.only or c["id"].startswith(args.only)][:args.limit]
    if args.questions:
        json.dump([{"id": c["id"], "turns": [t["q"] for t in c["turns"]]} for c in convs], open(OUT / "questions.json", "w"), indent=1)
        print(f"{len(convs)} conversations -> {OUT / 'questions.json'}")
        return

    report = {"generated": time.strftime("%Y-%m-%d %H:%M"), "passages": len(index.passages), "search": search_eval(index, convs)}
    print("search:", json.dumps(report["search"], indent=1))
    if args.search:
        json.dump(report, open(OUT / "report_search.json", "w"), indent=1)
        return
    rows = []
    if args.browser:
        run = json.load(open(args.browser))
        got = {(r["conv"], r["turn"]): r for r in run["rows"]}
        report["model"] = "browser build (4-bit ONNX), " + run.get("backend", "")
        for c in convs:
            for n, t in enumerate(c["turns"]):
                r = got[(c["id"], n)]
                rows.append({"conv": c["id"], "turn": n, **t, "pred": r["answer"], "capped": bool(r.get("capped"))})
    else:
        gen = hf_generate_fn(args.model)
        report["model"] = "fp32 PyTorch, " + pathlib.Path(args.model).name
        for c in convs:
            history = []
            for n, t in enumerate(c["turns"]):
                pred, capped, p = respond(index, gen, t["q"], history)
                rows.append({"conv": c["id"], "turn": n, **t, "pred": pred, "capped": capped,
                             "sources": [f"{h['title']}: {h['section']}" for h in p.hits]})
                r = rows[-1]
                r["correct"] = judge(r, pred, capped)
                print(f"{'✓' if r['correct'] else '✗'} [{r['kind']:10s}] {r['q'][:52]:52s} -> {pred[:100]!r}", flush=True)
    for r in rows:
        r["correct"] = judge(r, r["pred"], r["capped"])
    report["results"] = summarize(rows)
    name = args.name or ("browser" if args.browser else "pytorch")
    json.dump(rows, open(OUT / f"rows_{name}.json", "w"), indent=1, ensure_ascii=False)
    json.dump(report, open(OUT / f"report_{name}.json", "w"), indent=1)
    print(json.dumps(report["results"], indent=1))


if __name__ == "__main__":
    main()
