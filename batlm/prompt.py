"""The single source of truth for BatLM's prompt format (mirrored in web/src/lib/prompt.ts).

Training, evals, the CLI agent and the website must all build prompts identically —
train/serve skew in prompt formatting is one of the most common silent LLM bugs.
"""
from .rag import format_context
from .sft_data import PERSONA

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def chat(question: str, hits: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": PERSONA},
        {"role": "user", "content": f"{format_context(hits)}\n\nQUESTION: {question}"},
    ]


def render(question: str, hits: list[dict]) -> str:
    # Identical to SmolLM2's chat template with add_generation_prompt=True (verified in tests/test_prompt.py).
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in chat(question, hits))
    return out + "<|im_start|>assistant\n"
