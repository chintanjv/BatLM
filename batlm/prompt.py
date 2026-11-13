"""The single source of truth for BatLM's prompt format (mirrored in web/src/lib/prompt.ts).

Training, evals, the CLI agent and the website must all build prompts identically —
train/serve skew in prompt formatting is one of the most common silent LLM bugs.

A first question is one system message and one user message (case files + question). In a conversation
the earlier exchanges come in between as ordinary chat turns, without their case files.
"""
import os

from .rag import format_context
from .sft_data import PERSONA

# Any model that uses the ChatML format works: BATLM_BASE is how the larger candidates are trained and exported.
BASE_MODEL = os.environ.get("BATLM_BASE") or "HuggingFaceTB/SmolLM2-360M-Instruct"
HISTORY = 2        # earlier exchanges shown to the model
ANSWER_WORDS = 60  # each earlier answer is cut to this many words


def chat(question: str, hits: list[dict], history=()) -> list[dict]:
    msgs = [{"role": "system", "content": PERSONA}]
    for t in list(history)[-HISTORY:]:
        msgs.append({"role": "user", "content": f"QUESTION: {t['q']}"})
        msgs.append({"role": "assistant", "content": " ".join(t["a"].split()[:ANSWER_WORDS])})
    msgs.append({"role": "user", "content": f"{format_context(hits)}\n\nQUESTION: {question}"})
    return msgs


def render(question: str, hits: list[dict], history=()) -> str:
    # Identical to SmolLM2's chat template with add_generation_prompt=True (verified in tests/test_pipeline.py).
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in chat(question, hits, history))
    return out + "<|im_start|>assistant\n"
