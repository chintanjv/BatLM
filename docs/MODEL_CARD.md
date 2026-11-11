---
license: cc-by-sa-4.0
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
library_name: transformers.js
pipeline_tag: text-generation
language: [en]
tags: [batman, rag, lora, onnx, webgpu, educational, tiny-llm]
---

# 🦇 BatLM-360M

A small model that answers questions about the Batman universe from retrieved passages, built to run in the browser.
**[Try it](https://batlm.vercel.app)** · **[Source code](https://github.com/chintanjv/BatLM)**

- **Base:** [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
- **Fine-tune:** LoRA (r=16, all linear layers), 2 epochs, merged
- **Data:** about 2,000 questions generated from 286 hand-written, source-checked facts and the opening
  sentences of 333 Wikipedia pages. Every training prompt contains the three passages the project's own
  search returns for that question.
- **Files:** `onnx/model_q4f16.onnx` (WebGPU, 347 MB), `onnx/model_q4.onnx` (CPU/WASM)
- **Quantization:** onnxruntime-genai builder, k-quant, block size 16. Attention matrices, each MLP's output
  matrix and the LM head are 8-bit; the remaining MLP matrices are 4-bit.

This model is meant to be used **with retrieval**. Its prompt contains retrieved passages and it answers from
them. Without that context it is just SmolLM2 with a habit of short answers.

## Prompt format

```
<|im_start|>system
You are BatLM, an expert on the Batman universe. Answer the question directly in one or two sentences, or a short numbered list, using only the CASE FILES. If the files do not contain the answer, say so.<|im_end|>
<|im_start|>user
CASE FILES:
[1] (Title: Section) passage…
[2] (Title: Section) passage…
[3] (Title: Section) passage…

QUESTION: Who is the Batman?<|im_end|>
<|im_start|>assistant
```

Decode greedily and stop at `<|im_end|>`.

## Evaluation

161 held-out questions, answered by `model_q4f16.onnx` in headless Chrome on WebGPU through the same search
and prompt code as the website. An answer is correct when it contains every required fact, makes no listed
false claim, is not a refusal and ends before the token limit.

| Questions | n | Correct |
|---|---|---|
| Must-pass examples | 13 | 100% |
| Trained facts, new phrasing | 59 | 93.2% |
| Facts never seen in training | 38 | 94.7% |
| Wikipedia pages with no fact card | 18 | 94.4% |
| Batman questions the files cannot answer (should decline) | 13 | 100% |
| Off-topic (should decline) | 20 | 100% |
| **All** | **161** | **95.7%** |

The full-precision model scores the same 95.7%. A build with only attention at 8-bit (306 MB) scores 89.4%.

## Limitations

A 360M model. It answers what the retrieved passages contain and can pick a related but wrong sentence when
the question is phrased unusually. It does not follow a conversation, and compound questions usually get only
one half answered. Check the cited passages.

## License and attribution

The corpus is from Wikipedia (CC BY-SA 4.0), so these weights are released under **CC BY-SA 4.0**. The base
model is Apache-2.0. This is an unofficial fan project: Batman is a trademark of DC Comics, and the project is
not affiliated with DC or Warner Bros.
