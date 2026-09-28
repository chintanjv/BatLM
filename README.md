# 🦇 BatLM

**A tiny open-source language model trained on Batman, built end-to-end on a 2019 Intel MacBook Pro and running entirely in your browser.**

BatLM is a learning project that walks through every layer of building an LLM product: data, tokenizer,
pretraining, instruction tuning, retrieval, an agent harness, evals, quantization, and deployment. It is
trained on exactly two web pages:

1. [Wikipedia: Batman](https://en.wikipedia.org/wiki/Batman)
2. [Batman Wiki (Fandom)](https://batman.fandom.com/wiki/Batman_Wiki)

There are two models, because they teach different things:

| | **BatLM-nano** | **BatLM-360M** |
|---|---|---|
| What it is | A GPT written and trained **from scratch** | SmolLM2-360M-Instruct + **LoRA** fine-tune + **RAG** |
| Params | 1.9M | 362M (8.7M trained) |
| Teaches | tokenizers, attention, pretraining, overfitting | SFT, grounding, retrieval, agents, evals |
| Behavior | Continues your text in Batman-ish prose | Answers questions and cites its sources |
| Runs on | Our own ~150-line TypeScript inference engine | transformers.js (WebGPU → WASM fallback), 4-bit |

## The pipeline

```
 scrape.py      tokenizer.py     train_nano.py        sft_data.py + train_lora.py     rag.py        agent.py        evals.py        export_web.py         web/
┌────────┐    ┌───────────┐    ┌─────────────┐      ┌─────────────────────────┐   ┌────────┐    ┌──────────┐    ┌─────────┐    ┌───────────────┐    ┌──────────────┐
│2 pages │ ─▶ │byte-level │ ─▶ │ 4-layer GPT │      │ 129 grounded Q&A + ctx  │   │ BM25 + │ ─▶ │ plan→act │ ─▶ │ held-out│ ─▶ │ ONNX · 4-bit  │ ─▶ │ Vite + React │
│99 chunks│   │BPE (768)  │    │ from scratch│      │ LoRA r=16 on SmolLM2    │   │synonyms│    │→guard→gen│    │ablations│    │ JS nano engine│    │ Web Worker   │
└────────┘    └───────────┘    └─────────────┘      └─────────────────────────┘   └────────┘    └──────────┘    └─────────┘    └───────────────┘    └──────────────┘
```

| Stage | File | What to learn there |
|---|---|---|
| Data | `batlm/scrape.py` | MediaWiki API (Fandom blocks plain scrapers), boilerplate removal, sentence-aligned chunking |
| Tokenizer | `batlm/tokenizer.py` | Byte-level BPE merges, regex pre-splitting, why chars/token matters |
| Model | `batlm/model.py` | Causal self-attention, pre-LN residual blocks, weight tying |
| Pretrain | `batlm/train_nano.py` | AdamW, warmup + cosine LR, grad clipping, **train vs val loss & overfitting** |
| Instruct | `batlm/sft_data.py`, `batlm/train_lora.py` | Chat templates, loss masking, LoRA, "oracle insertion" so the model learns to *read* context |
| Retrieve | `batlm/rag.py` ↔ `web/src/lib/bm25.ts` | BM25 from scratch, stemming, query expansion; measuring recall@k |
| Agent | `batlm/agent.py` ↔ `web/src/lib/agent.ts` | Router-based planning, tools, confidence guardrails, traces |
| Evals | `batlm/evals.py` | Reworded / unseen-fact / off-topic splits, key-fact accuracy, F1, false refusals, ablations |
| Ship | `batlm/export_web.py`, `web/src/worker.ts`, `web/src/lib/nano.ts` | ONNX export, 4-bit quantization, parity tests, in-browser inference |

**Train/serve parity is tested.** `tests/test_prompt.py` checks that the hand-rendered prompt matches the official
chat template, and `web/scripts/nano-parity.mjs` checks that the TypeScript engine reproduces PyTorch's logits (max diff ~1e-5).

## Run it yourself

```bash
# Python 3.11 via uv (torch 2.2.2 is the last build for Intel Macs)
curl -LsSf https://astral.sh/uv/install.sh | sh
make setup
make all        # scrape → nano → lora → evals → export → tests (~2-3 h on a 2019 i5, CPU only)
make web        # http://localhost:5173
python -m batlm.agent   # the same ORACLE agent, in your terminal
```

## Evals

`python -m batlm.evals` writes `evals/report.json`, which the site's **DIAGNOSTICS** tab renders. Each layer is measured on its own:

- **Retrieval recall@k**: do the top-k chunks contain the answer's key facts?
- **Generation** on three held-out splits:
  - *reworded* questions (facts seen in training, new phrasing)
  - *unseen facts* (never trained on, so only answerable through RAG)
  - *off-topic* questions (the model should refuse)
- **Ablations**:
  - `base+rag`: did fine-tuning help?
  - `ft+rag`: the deployed configuration
  - `ft+oracle`: the model given the gold passage, which isolates retrieval errors from generation errors

### Results (77 held-out questions)

| Config | Reworded Qs | Unseen facts | Off-topic refusal | Overall |
|---|---|---|---|---|
| SmolLM2-360M base + RAG | 50% | 67% | 0% | 44% |
| BatLM fp32 + RAG + guardrail | 71% | 56% | 42% | 65% |
| **BatLM in the browser (4/8-bit) + RAG + guardrail** | 61% | 44% | 42% | **56%** |

### What the evals taught us

1. **Retrieval recall was 71%.** Stemming and a small synonym list raised it to 83%.
2. **The first guardrail was broken in both directions.**
   - It refused "Who is Batman?", because BM25 gives near-zero weight to a term that appears everywhere.
   - It let "How do I change a car tire?" through, because "car" expands to "Batmobile".
   - Fix: "domain anchor" terms always pass, and the threshold was tuned on 260 questions.
3. **More refusal training made things worse.** v2 and v3 added "I don't know" examples. The model refused off-topic questions more often, but it also refused 8–9 real questions and lost about 8 points overall. v1 ships, and the guardrail does the refusing.
4. **Default 4-bit quantization cost 28 points** (84% → 56% on a 25-question slice). Mixed 4/8-bit with block size 16 brings the full-set cost down to about 9 points.
5. **The key-fact scorer is lenient.** It marks "created by writer Bob Kane and artist Bill Finger" (roles swapped) as correct. Read the rows in `evals/`, not just the averages.

## Licenses and credits

- Code: MIT.
- Training data comes from Wikipedia and Fandom, both CC BY-SA, so the derived dataset and model weights are CC BY-SA 4.0.
- Base model: [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) (Apache-2.0).
- BatLM is an unofficial, educational fan project. Batman and related characters are trademarks of DC Comics. This project is not affiliated with or endorsed by DC or Warner Bros. The bat emblem was drawn for this project.
