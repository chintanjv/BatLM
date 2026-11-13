"""Stage 7 — Evals: how often is the answer right, and where is quality lost?

    python -m batlm.evals                         # search + generation with checkpoints/batlm-merged
    python -m batlm.evals --search                # search only: no model needed
    python -m batlm.evals --onnx DIR              # generation with a quantized build (onnxruntime-genai folder)
    python -m batlm.evals --questions             # write evals/questions.json for the browser run
    python -m batlm.evals --browser rows.json     # score answers produced by web/scripts/e2e-browser.mjs
    python -m batlm.evals --judge rows.json       # have the teacher mark the answers that have no key words

The eval is a list of conversations, from three sources:
  auto-*  one-turn questions generated from the fact cards and pages (batlm.sft_data.eval_items)
  hand-*  evals/handwritten.json: written by hand, including follow-ups; never generated, never trained on
  gen-*   evals/generated.json: written by the teacher about subjects held out of training (batlm.sft_gen)
Each turn of a conversation is asked with the model's own earlier answers as history, as on the site.

  search      is the passage holding the answer among those shown to the model?
  generation  an answer is correct when it contains every required fact, contains no forbidden claim,
              is not a refusal, and ended by itself before the token limit. Off-topic and unanswerable
              questions are correct when the model declines. A teacher-written question has a reference
              answer instead of key words; the teacher marks those (--judge), and until then they are
              left out of the scores.

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

JUDGE = """{earlier}QUESTION: {q}
REFERENCE ANSWER (known to be correct): {ref}
STUDENT ANSWER: {pred}

Does the student answer give the same key fact or facts as the reference, without saying anything that contradicts it? Different wording and extra correct detail are fine. A refusal, or "I don't have that", is wrong. Reply with one word: CORRECT or WRONG."""


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
    return convs


def conversations(index) -> list[dict]:
    convs = [{"id": f"auto-{i}", "turns": [it]} for i, it in enumerate(sft_data.eval_items(index))] + handwritten()
    if (OUT / "generated.json").exists():
        convs += json.loads((OUT / "generated.json").read_text())
    for c in convs:
        c["turns"] = [{"a": "", "keys": "", "bad": "", "ref": "", **t} for t in c["turns"]]
    return convs


def declined(pred: str) -> bool:
    p = pred.lower()
    return "outside my case files" in p or "don't have that" in p


def judge(it: dict, pred: str, capped: bool):
    """True or False; None when only the teacher can tell (a reference answer and no key words)."""
    if it["kind"] in DECLINE:
        return declined(pred)
    if capped or declined(pred):
        return False
    if not it["keys"]:
        return it.get("verdict")
    return key_hit(pred, it["keys"]) and not (it.get("bad") and key_hit(pred, it["bad"]))


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
            elif t["a"] or t["keys"]:
                ctx = " ".join(h["text"] for h in p.hits)
                ok = (t["a"] in ctx) if t["a"] else key_hit(ctx, t["keys"])
                by[t["kind"]][0] += ok
                by[t["kind"]][1] += 1
                if p.prompt is None:
                    gated.append(t["q"])
                elif not ok:
                    misses.append(t["q"])
            if p.prompt is not None:
                history.append({"q": t["q"], "a": t["a"] or t["ref"] or t["keys"], "topic": p.topic})
    hit, total = sum(v[0] for v in by.values()), sum(v[1] for v in by.values())
    train_off = sum(plan(index, q).prompt is not None for q in sft_data.OFF_TOPIC)
    return {"n": total, f"recall@{TOP_K}": round(hit / total, 3),
            "by_kind": {k: {"n": v[1], "recall": round(v[0] / v[1], 3)} for k, v in sorted(by.items())},
            "answerable_refused_by_gate": gated, "off_topic_reaching_model": f"{leaked} of {off}",
            "training_off_topic_reaching_model": f"{train_off} of {len(sft_data.OFF_TOPIC)}", "misses": misses[:150]}


def summarize(rows) -> dict:
    marked = [r for r in rows if r["correct"] is not None]
    acc = lambda v: {"n": len(v), "accuracy": round(sum(r["correct"] for r in v) / len(v), 3)} if v else {"n": 0}
    by = defaultdict(list)
    for r in marked:
        by[r["kind"]].append(r)
    out = {k: acc(v) for k, v in sorted(by.items())}
    answerable = [r for r in marked if r["kind"] not in DECLINE]
    for name in ("auto", "hand", "gen"):
        out[f"source:{name}"] = acc([r for r in marked if r["conv"].startswith(name + "-")])
    out["answerable"] = acc(answerable)
    out["overall"] = {**acc(marked), "waiting_for_the_teacher": len(rows) - len(marked),
                      "hit_token_limit": sum(r["capped"] for r in rows),
                      "wrongly_declined": sum(declined(r["pred"]) for r in rows if r["kind"] not in DECLINE),
                      "batman_question_called_off_topic": sum("outside my case files" in r["pred"].lower() for r in rows if r["kind"] not in DECLINE)}
    return out


def og_generate_fn(path, max_new_tokens=160):
    """Greedy generation with a quantized build, through onnxruntime-genai (the builder that makes the browser models)."""
    import onnxruntime_genai as og
    model = og.Model(str(path))
    tok = og.Tokenizer(model)

    def gen(prompt):
        ids = tok.encode(prompt)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(ids) + max_new_tokens, do_sample=False)
        g = og.Generator(model, params)
        g.append_tokens(ids)
        while not g.is_done():
            g.generate_next_token()
        new = list(g.get_sequence(0))[len(ids):]
        return tok.decode(new).replace("<|im_end|>", "").strip(), len(new) >= max_new_tokens
    return gen


def finish(rows, report, name):
    for r in rows:
        r["correct"] = judge(r, r["pred"], r["capped"])
    report["results"] = summarize(rows)
    json.dump(rows, open(OUT / f"rows_{name}.json", "w"), indent=1, ensure_ascii=False)
    json.dump(report, open(OUT / f"report_{name}.json", "w"), indent=1)
    print(json.dumps(report["results"], indent=1))


def teacher_judge(rows_file: str):
    """Have the teacher mark every answer that has a reference answer but no key words, then rewrite the rows and report."""
    from .teacher import Teacher
    rows = json.load(open(rows_file))
    name = pathlib.Path(rows_file).stem.removeprefix("rows_")
    report = json.load(open(OUT / f"report_{name}.json"))
    todo, earlier = [], {}
    for r in rows:
        past = earlier.setdefault(r["conv"], [])
        if r["kind"] not in DECLINE and not r["keys"] and r["ref"] and not r["capped"] and not declined(r["pred"]):
            todo.append((r, "EARLIER TURNS:\n" + "".join(f"User: {q}\nAssistant: {a}\n" for q, a in past[-2:]) + "\n" if past else ""))
        past.append((r["q"], r["ref"] or r["pred"]))
    teacher = Teacher()
    try:
        replies = teacher.ask([JUDGE.format(earlier=e, q=r["q"], ref=r["ref"], pred=r["pred"]) for r, e in todo], max_tokens=6, temperature=0.0)
    finally:
        teacher.close()
    for (r, _), v in zip(todo, replies):
        r["verdict"] = v.strip().upper().startswith("CORRECT")
    finish(rows, report, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "checkpoints" / "batlm-merged"))
    ap.add_argument("--onnx", help="score a quantized build: the folder onnxruntime-genai's builder wrote")
    ap.add_argument("--search", action="store_true", help="score search only; no model needed")
    ap.add_argument("--questions", action="store_true")
    ap.add_argument("--browser", help="rows file from web/scripts/e2e-browser.mjs")
    ap.add_argument("--judge", help="rows file from an earlier run: have the teacher mark the answers without key words")
    ap.add_argument("--name", help="name of this run, used in the output file names")
    ap.add_argument("--only", choices=["auto", "hand", "gen"], help="run only one source of conversations")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.judge:
        return teacher_judge(args.judge)
    index = rag.load()
    convs = [c for c in conversations(index) if not args.only or c["id"].startswith(args.only + "-")][:args.limit]
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
        gen = og_generate_fn(args.onnx) if args.onnx else hf_generate_fn(args.model)
        report["model"] = f"4-bit ONNX, {args.onnx}" if args.onnx else "fp32 PyTorch, " + pathlib.Path(args.model).name
        for c in convs:
            history = []
            for n, t in enumerate(c["turns"]):
                pred, capped, p = respond(index, gen, t["q"], history)
                rows.append({"conv": c["id"], "turn": n, **t, "pred": pred, "capped": capped,
                             "sources": [f"{h['title']}: {h['section']}" for h in p.hits]})
                mark = {True: "✓", False: "✗", None: "?"}[judge(rows[-1], pred, capped)]
                print(f"{mark} [{t['kind']:10s}] {t['q'][:52]:52s} -> {pred[:100]!r}", flush=True)
    finish(rows, report, args.name or ("browser" if args.browser else "onnx" if args.onnx else "pytorch"))


if __name__ == "__main__":
    main()
