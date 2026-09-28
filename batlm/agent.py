"""Stage 8 — The harness: an agent loop around BatLM ("ORACLE protocol").

    python -m batlm.agent            # interactive REPL with a visible trace

A 360M model is too small to reliably emit tool-call JSON, so — like many production systems —
planning is done by a deterministic router, and the LLM is used where it is strong: reading
retrieved evidence and writing the answer. The loop:

  PLAN     classify intent -> list of tool calls (quiz / timeline / compare / lookup)
  ACT      run tools: search_case_files, timeline, quiz
  GUARD    if retrieval confidence is below threshold, refuse without calling the LLM
  ANSWER   BatLM generates from the gathered CASE FILES
  TRACE    every step is recorded with timing (the website renders the same trace)

Mirrored in web/src/lib/agent.ts.
"""
import random
import re
import time

from . import rag, sft_data
from .prompt import render

MIN_SCORE = 3.0  # below this (and with no domain anchor like "batman") the files are irrelevant.
# Tuned on 200 in-scope / 60 off-topic questions: blocks 0 in-scope, 24/60 off-topic; the model handles the rest.


class Oracle:
    def __init__(self, generate_fn):
        self.index = rag.load()
        self.generate = generate_fn
        self.pending_quiz = None

    # ---- tools ----------------------------------------------------------------------------
    def search_case_files(self, query, k=3):
        return self.index.search(query, k)

    def timeline(self, year):
        hits = [c for c in self.index.chunks if year in c["text"]]
        return [(1.0 * c["text"].count(year) + 5, c) for c in hits[:3]]

    def quiz(self):
        f = random.choice([f for f in sft_data.F])
        self.pending_quiz = f
        return f["q"][0]

    # ---- planner --------------------------------------------------------------------------
    def plan(self, msg):
        m = msg.lower()
        if self.pending_quiz is not None and not m.startswith("/"):
            return [("grade_quiz", msg)]
        if re.search(r"\b(quiz|test me|challenge me)\b", m):
            return [("quiz", None)]
        if (y := re.search(r"\b(19[3-9]\d|20[0-2]\d)\b", m)) and re.search(r"\b(happen|in|during|year)\b", m):
            return [("timeline", y.group(1)), ("search_case_files", msg), ("answer", msg)]
        if (c := re.search(r"(?:compare|difference between)\s+(.+?)\s+(?:and|vs\.?|with)\s+(.+?)[?.]?$", m)):
            return [("search_case_files", c.group(1)), ("search_case_files", c.group(2)), ("answer", msg)]
        return [("search_case_files", msg), ("answer", msg)]

    # ---- loop -----------------------------------------------------------------------------
    def run(self, msg):
        trace, evidence = [], []

        def step(tool, arg, fn):
            t0 = time.time()
            out = fn()
            trace.append({"tool": tool, "input": arg, "ms": round(1000 * (time.time() - t0)), "output": out})
            return out

        for tool, arg in self.plan(msg):
            if tool == "quiz":
                q = step("quiz", None, self.quiz)
                return f"CHALLENGE: {q}", trace
            if tool == "grade_quiz":
                f, self.pending_quiz = self.pending_quiz, None
                ok = step("grade", arg, lambda: all(k.lower() in arg.lower() for k in f["keys"]))
                return ("CORRECT. " if ok else "NEGATIVE. ") + f["a"], trace
            if tool in ("search_case_files", "timeline"):
                fn = self.search_case_files if tool == "search_case_files" else self.timeline
                hits = step(tool, arg, lambda: fn(arg))
                trace[-1]["output"] = [f"{s:.1f} {c['section']}: {c['text'][:60]}..." for s, c in hits]
                evidence += [(s, c) for s, c in hits if c not in [e[1] for e in evidence]]
            if tool == "answer":
                best = max((s for s, _ in evidence), default=0)
                if best < MIN_SCORE and not self.index.mentions_anchor(msg):
                    step("guardrail", f"max score {best:.1f} < {MIN_SCORE}", lambda: "refuse")
                    return sft_data.REFUSAL, trace
                ctx = [c for _, c in sorted(evidence, key=lambda e: -e[0])[:3]]
                prompt = render(msg, ctx)
                return step("batlm.generate", f"{len(prompt)} chars", lambda: self.generate(prompt)), trace


def hf_generate_fn(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    def gen(prompt):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=96, do_sample=False, repetition_penalty=1.1, eos_token_id=eos)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return gen


def main():
    import pathlib
    path = pathlib.Path(__file__).resolve().parent.parent / "checkpoints" / "batlm-merged"
    oracle = Oracle(hf_generate_fn(str(path)))
    print("BATCOMPUTER ONLINE. Ask about Batman, or say 'quiz me'. Ctrl-D to exit.")
    while True:
        try:
            msg = input("\n> ").strip()
        except EOFError:
            break
        if not msg:
            continue
        answer, trace = oracle.run(msg)
        for t in trace:
            print(f"  · {t['tool']:<18} {str(t['input'])[:40]:<40} {t['ms']:>5}ms")
        print(f"\n{answer}")


if __name__ == "__main__":
    main()
