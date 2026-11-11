"""Stage 6 — Answering a question: search -> off-topic gate -> BatLM.

    python -m batlm.agent            # interactive prompt

The gate refuses without calling the model when the question names no Batman subject and search
finds nothing relevant. Mirrored in web/src/lib/agent.ts.
"""
import pathlib

from . import rag, sft_data
from .prompt import render

MIN_SCORE = 10.0 # below this, with no Batman subject named in the question, the question is off-topic

ROOT = pathlib.Path(__file__).resolve().parent.parent


def plan(index, question):
    """Returns (prompt, passages), or (None, []) when the question is off-topic."""
    scored = index.search(question, k=3)
    if not index.mentions(question) and (scored[0][0] if scored else 0) < MIN_SCORE:
        return None, []
    hits = [c for _, c in scored]
    return render(question, hits), hits


def hf_generate_fn(path, max_new_tokens=160):
    """Greedy generation with the same settings as the browser (web/src/worker.ts)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(device).eval()
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    def gen(prompt):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=1.1, eos_token_id=eos, pad_token_id=eos)
        new = out[0, ids.shape[1]:]
        return tok.decode(new, skip_special_tokens=True).strip(), len(new) >= max_new_tokens
    return gen


def main():
    index, gen = rag.load(), hf_generate_fn(str(ROOT / "checkpoints" / "batlm-merged"))
    print("BatLM ready. Ask about the Batman universe. Ctrl-D to exit.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            continue
        prompt, hits = plan(index, q)
        for c in hits:
            print(f"  · {c['title']}: {c['section']}")
        print("\n" + (gen(prompt)[0] if prompt else sft_data.REFUSAL))


if __name__ == "__main__":
    main()
