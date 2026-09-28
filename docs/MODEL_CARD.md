---
license: cc-by-sa-4.0
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
library_name: transformers.js
pipeline_tag: text-generation
language: [en]
tags: [batman, rag, lora, onnx, webgpu, educational, tiny-llm]
---

# 🦇 BatLM-360M

A small Batman Q&A model built as an end-to-end learning project: data → SFT → RAG → agent → evals → in-browser deployment.
**[Try it in your browser](https://batlm.vercel.app)** · **[Source code](https://github.com/chintanjv/BatLM)**

- **Base:** [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
- **Fine-tune:** LoRA (r=16, all linear layers), 3 epochs on a laptop CPU, merged
- **Data:** 129 hand-written Q&A pairs, grounded in exactly two pages ([Wikipedia: Batman](https://en.wikipedia.org/wiki/Batman), [Batman Wiki](https://batman.fandom.com/wiki/Batman_Wiki)). Every training prompt includes 3 BM25-retrieved passages ("CASE FILES").
- **Files:** `onnx/model_q4f16.onnx` (WebGPU, 298 MB), `onnx/model_q4.onnx` (WASM, 344 MB)
- **Quantization:** onnxruntime-genai builder with `k_quant_mixed`: sensitive layers and the LM head stay at 8-bit, the rest are 4-bit, in blocks of 16.

This model is meant to be used **with retrieval**: its prompt contains retrieved passages, and it answers from them. Without context, it's just a slightly Batman-flavored SmolLM2.

## Prompt format

```
<|im_start|>system
You are BatLM, the Batcomputer's field intelligence. Answer in one to three short sentences, using only the CASE FILES provided. If the files do not cover the question, say it is outside your case files.<|im_end|>
<|im_start|>user
CASE FILES:
[1] (Section) passage…
[2] (Section) passage…
[3] (Section) passage…

QUESTION: Who created Batman?<|im_end|>
<|im_start|>assistant
```

## Evaluation (77 held-out questions, greedy decoding)

"Correct" means all key facts appear in the answer and the model didn't refuse. For off-topic questions, "correct" means it refused.

| Config | Reworded Qs | Unseen facts | Off-topic refusal | Overall |
|---|---|---|---|---|
| SmolLM2-360M base + RAG | 50% | 67% | 0% | 44% |
| BatLM fp32 + RAG | 71% | 56% | 8% | 60% |
| BatLM fp32 + RAG + guardrail | 71% | 56% | 42% | 65% |
| **BatLM 4/8-bit (these files) + RAG + guardrail** | **61%** | **44%** | **42%** | **56%** |

Zero false refusals on answerable questions. Retrieval recall@3 = 83%.

Two further fine-tunes (v2, v3) added more "I don't know" examples. They refused off-topic questions more often (58%), but also refused 8–9 real questions and scored lower overall, so v1 ships.

## Limitations

This is a toy model. It **hallucinates plausible details**, for example wrong dates, or roles swapped between creators. Always check the cited passages. It only knows what's in the two source pages.

## License and attribution

The training data comes from Wikipedia and Fandom (CC BY-SA), so these weights are released under **CC BY-SA 4.0**. The base model is Apache-2.0. This is an unofficial fan project: Batman is a trademark of DC Comics, and the project is not affiliated with DC or Warner Bros.
