"""Stage 6 — Answering a question: search -> off-topic gate -> BatLM.

    python -m batlm.agent            # interactive prompt

The gate refuses without calling the model when the question names no Batman subject, is not a
follow-up to one, and search finds nothing relevant. Mirrored in web/src/lib/agent.ts.

A conversation is a list of turns, {"q": question, "a": answer, "topic": subjects}. The last turn tells
search who a follow-up is about, and the last few are shown to the model.
"""
import pathlib
from typing import NamedTuple

from . import rag, sft_data
from .prompt import render

MIN_SCORE = 16.0  # below this, with no Batman subject named or carried over, the question is off-topic
TOP_K = 3         # passages shown to the model

ROOT = pathlib.Path(__file__).resolve().parent.parent


class Plan(NamedTuple):
    prompt: str | None  # None: the off-topic gate answers, the model is not called
    hits: list          # the passages shown to the model
    topic: list         # who this turn is about, for the next turn's follow-up
    named: bool         # does the question itself name a Batman subject?


def plan(index, question, history=()) -> Plan:
    named, carried = index.subjects(question, history[-1] if history else None)
    scored = index.search(question, k=TOP_K, carried=carried)
    if not named and not carried and (scored[0][0] if scored else 0) < MIN_SCORE:
        return Plan(None, [], [], False)
    hits = [c for _, c in scored]
    return Plan(render(question, hits, history), hits, carried[:1] + named if named else carried[:2], bool(named))


def settle(answer: str, named: bool) -> str:
    """A question that names a Batman subject is never 'outside the Batman universe': at worst the files lack the answer."""
    return sft_data.MISSING if named and "outside my case files" in answer.lower() else answer


def respond(index, gen, question, history: list):
    """One turn, exactly as the site runs it: (answer, hit the token limit?, plan). An answered turn joins `history`."""
    p = plan(index, question, history)
    if p.prompt is None:
        return sft_data.REFUSAL, False, p
    pred, capped = gen(p.prompt)
    pred = settle(pred, p.named)
    history.append({"q": question, "a": pred, "topic": p.topic})
    return pred, capped, p


def hf_generate_fn(path, max_new_tokens=160):
    """Greedy generation with the same settings as the browser (web/src/worker.ts)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(device).eval()
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    def gen(prompt):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=1.0, eos_token_id=eos, pad_token_id=eos)
        new = out[0, ids.shape[1]:]
        return tok.decode(new, skip_special_tokens=True).strip(), len(new) >= max_new_tokens
    return gen


def main():
    index, gen = rag.load(), hf_generate_fn(str(ROOT / "checkpoints" / "batlm-merged"))
    print("BatLM ready. Ask about the Batman universe. Ctrl-D to exit.")
    history = []
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            continue
        answer, _, p = respond(index, gen, q, history)
        for c in p.hits:
            print(f"  · {c['title']}: {c['section']}")
        print("\n" + answer)


if __name__ == "__main__":
    main()
