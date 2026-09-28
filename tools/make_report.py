"""Assemble evals/report.json (rendered by the site's DIAGNOSTICS tab) from saved per-question rows.

Every config is re-scored with the *current* scorer so numbers are comparable across runs.

    .venv/bin/python tools/make_report.py
"""
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from batlm import rag, sft_data  # noqa: E402
from batlm.agent import MIN_SCORE  # noqa: E402
from batlm.evals import is_refusal, key_hit, nano_eval, retrieval_eval, summarize  # noqa: E402

E = ROOT / "evals"
index = rag.load()
items = {i["q"]: i for i in sft_data.eval_items()}


def rescore(rows, guard=False):
    out = []
    for r in rows:
        it = items.get(r["q"])
        if it is None:
            continue
        pred = r["pred"]
        if guard:
            sc = index.search(r["q"], 3)
            if (sc[0][0] if sc else 0) < MIN_SCORE and not index.mentions_anchor(r["q"]):
                pred = sft_data.REFUSAL
        oos = it["kind"] == "out_of_scope"
        out.append({**r, **it, "pred": pred, "refused": is_refusal(pred),
                    "correct": is_refusal(pred) if oos else (key_hit(pred, it["keys"]) and not is_refusal(pred))})
    assert len(out) == len(items), f"expected {len(items)} rows, got {len(out)}"
    return out


def load(p):
    return json.load(open(E / p))


configs = {
    "shipped": summarize(rescore(load("rows_shipped.json"), guard=True)),     # v1, int4/8 ONNX, + guardrail
    "agent": summarize(rescore(load("v1/rows_ft_rag_77.json"), guard=True)),  # v1, fp32, + guardrail
    "ft+rag": summarize(rescore(load("v1/rows_ft_rag_77.json"))),             # v1, fp32, no guardrail
    "base+rag": summarize(rescore(load("v2/rows_base_rag.json"))),            # stock SmolLM2
}
iterations = []
for name, share, path in (("v1", 0.09, "v1/rows_ft_rag_77.json"), ("v2", 0.29, "v2/rows_ft_rag.json"), ("v3", 0.21, "v3/rows_ft_rag.json")):
    s = summarize(rescore(load(path), guard=True))
    iterations.append({"name": name, "refusal_share": share, "overall": s["overall"]["accuracy"],
                       "in_scope": round((s["reworded"]["accuracy"] * s["reworded"]["n"] + s["unseen_fact"]["accuracy"] * s["unseen_fact"]["n"])
                                         / (s["reworded"]["n"] + s["unseen_fact"]["n"]), 3),
                       "out_of_scope": s["out_of_scope"]["accuracy"], "false_refusals": s["overall"]["false_refusals"]})

report = {"generated": time.strftime("%Y-%m-%d %H:%M"), "n_items": len(items),
          "retrieval": retrieval_eval(index, list(items.values())), "nano": nano_eval(),
          "configs": configs, "iterations": iterations, "shipped": "v1"}
json.dump(report, open(E / "report.json", "w"), indent=1)
for k, v in configs.items():
    print(f"{k:9s}", {kk: vv["accuracy"] for kk, vv in v.items()}, "false_refusals", v["overall"]["false_refusals"])
for it in iterations:
    print(it)
