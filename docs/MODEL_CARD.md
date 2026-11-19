---
license: cc-by-sa-4.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: transformers.js
pipeline_tag: text-generation
language: [en]
tags: [batman, rag, lora, onnx, webgpu, educational]
---

# 🦇 BatLM v3 (Qwen2.5-0.5B)

A small model that answers questions about the Batman universe from retrieved passages, built to run in the browser.
**[Try it](https://batlm.vercel.app)** · **[Source code](https://github.com/chintanjv/BatLM)**

- **Base:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Fine-tune:** LoRA (r=16, all linear layers), 2 epochs, merged. The adapter is in `lora/`.
- **Data:** about 20,000 examples:
  - template questions from 286 hand-written, source-checked facts;
  - 17,019 questions and short conversations written by Qwen2.5-14B-Instruct (yes/no, false claims, pushback,
    follow-ups, comparisons, lists, typos, off-topic).

  Every prompt contains the passages the project's own search returns for that question. Every answer was
  written from those passages and checked against them.
- **Files:** `onnx/model_q4f16.onnx` (WebGPU, 591 MB) and `onnx/model_q4.onnx` (CPU/WASM, 1.2 GB). Despite the
  names, every matrix is 8-bit (onnxruntime-genai builder, k-quant, block size 16).

This model is meant to be used **with retrieval**. Its prompt contains retrieved passages and it answers from them.

## Prompt format

```
<|im_start|>system
You are BatLM, an expert on the Batman universe. Answer the question directly in one or two sentences, or a short numbered list, using only the CASE FILES. If the files do not contain the answer, say so.<|im_end|>
<|im_start|>user
QUESTION: an earlier question<|im_end|>
<|im_start|>assistant
the earlier answer<|im_end|>
<|im_start|>user
CASE FILES:
[1] (Title: Section) passage…
[2] (Title: Section) passage…
[3] (Title: Section) passage…

QUESTION: was he an arkham patient?<|im_end|>
<|im_start|>assistant
```

Up to two earlier exchanges come before the case files (each earlier answer cut to 60 words). Decode greedily
and stop at `<|im_end|>`.

## Evaluation

221 hand-written turns in 171 conversations, never trained on, scored on this 8-bit build:

| Question kind | Correct |
|---|---|
| Plain lookups | 96% |
| Yes/no | 81% |
| "Are you sure?" and false claims | 78% |
| Follow-ups | 76% |
| Lists | 89% |
| Comparisons | 60% |
| Two-part questions | 50% |
| Typos | 90% |
| Should decline | 100% |
| **All** | **83.7%** |

Full precision: 85.1%. The previous model, BatLM-360M, scores 75.6% on the same questions. On the full eval (1,261
turns, including teacher-written questions about subjects held out of training) this model scores 76.3%.

## Limitations

A 0.5B model:

- It often half-answers comparisons and two-part questions.
- It loses about one follow-up in four.
- It can agree with a confident false claim.

Check the cited passages.

## License and attribution

The corpus is from Wikipedia (CC BY-SA 4.0), so these weights are released under **CC BY-SA 4.0**. The base model
is Apache-2.0. This is an unofficial fan project: Batman is a trademark of DC Comics, and the project is not
affiliated with DC or Warner Bros.
