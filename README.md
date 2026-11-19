# 🦇 BatLM

**A small language model that answers questions about the Batman universe, running entirely in your browser.**

🦇 **Live: [batlm.vercel.app](https://batlm.vercel.app)** · 🤗 **Model: [chintanjv/BatLM-v3-qwen05-r2](https://huggingface.co/chintanjv/BatLM-v3-qwen05-r2)**

> The first visit downloads the model (about 590 MB with WebGPU, about 1.2 GB in CPU mode); it is cached after that.
> Chrome and Edge with WebGPU answer in a few seconds. Without WebGPU the site falls back to CPU mode, which is much slower.

BatLM is a learning project that covers every layer of a small LLM product: a corpus, checked facts, search,
fine-tuning, evals, quantization and in-browser deployment. Nothing is sent to a server: search and the model
both run on your device.

## How it answers a question

```
question + the last two turns ─▶ search ─▶ off-topic gate ─▶ BatLM ─▶ answer + sources
                                  │                           ▲
                                  └── 3 passages ─────────────┘
```

1. **Search** ranks about 20,000 passages: 286 hand-written facts, 8,599 facts written from the pages by a
   larger "teacher" model and kept only if their key words appear in the source, a short passage for every line
   of every cast list, and 11,332 passages from 805 Wikipedia pages. It combines BM25 with:
   - every subject's names and aliases, with typo correction;
   - the usual ways of asking for each fact;
   - the subject of the previous turn, so "was he an Arkham patient?" stays about the Joker.

   One place is kept for the subject the question names, one for the subject it inherits, and the rest go to
   whatever passages answer best.
2. **The gate** refuses off-topic questions without calling the model. A question that names a Batman subject
   is never called off-topic.
3. **BatLM** ([Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) with a LoRA fine-tune)
   reads the passages and the last two exchanges and answers in one or two sentences, or a short list.

## The pipeline

Everything runs in the cloud: GitHub Actions starts each stage, Kaggle's free GPUs do the heavy work, and
Hugging Face holds the models and the training data. `docs/V3_HANDOFF.md` explains how to run each stage.

| Stage | File | What it does |
|---|---|---|
| Corpus | `batlm/scrape.py` | Downloads 805 Wikipedia pages (a hand-picked list plus the Batman categories) and cuts them into ~80-word passages |
| Facts | `batlm/cards.py`, `batlm/facts.py` | 139 hand-written cards; 8,599 teacher-written facts checked against their source text |
| Search | `batlm/rag.py` ↔ `web/src/lib/search.ts` | BM25, names, phrasings, cast lines and conversation memory, identical in Python and the browser |
| Training data | `batlm/sft_data.py`, `batlm/sft_gen.py` | Template questions, plus 17,019 teacher-written questions and conversations, each answered from the passages search really returns |
| Fine-tune | `batlm/train_lora.py` | LoRA on any ChatML base model |
| Answer | `batlm/agent.py` ↔ `web/src/lib/agent.ts` | Search, off-topic gate, prompt with the last two exchanges |
| Evals | `batlm/evals.py` | Scores PyTorch, the quantized build, or answers from a real browser; ends with a release gate |
| Export | `batlm/export_web.py`, `batlm/quant_build.py` | Search index for the site; ONNX models for WebGPU and CPU |
| Cloud | `cloud/`, `.github/workflows/` | Runs each stage on Kaggle and brings the results back |
| Site | `web/` | Vite + React; inference in a Web Worker with transformers.js |

`tests/test_pipeline.py` and `tests/test_teacher_data.py` guard the failures this project has had: the prompt
drifting between training and the browser, the browser search disagreeing with the Python search, follow-ups
losing their subject, questions that name several subjects crowding out the answer, and the teacher data
pipeline.

## Results

**Hand-written questions** (221 turns in 171 conversations, never generated and never trained on, including the
conversations that failed in real use). Scored on the 8-bit build visitors download:

| Question kind | v3 (this model) | v2 (BatLM-360M) |
|---|---|---|
| Plain lookups | 96% | 85% |
| Yes/no | 81% | 76% |
| "Are you sure?" and false claims | 78% | 56% |
| Follow-ups ("who played him?") | 76% | 64% |
| Lists | 89% | 67% |
| Comparisons | 60% | 50% |
| Two-part questions | 50% | 50% |
| Typos | 90% | 90% |
| Should decline (not in the files, off-topic) | 100% | 100% |
| **All** | **83.7%** | **75.6%** |

The same model at full precision scores 85.1%. In headless Chrome, through the website's own code, it scored
82.9% on the first 40 hand-written conversations, and every turn of the conversations that failed in real use
was answered correctly. On the full eval, which adds about 1,000 generated and template questions, it scores
76.3%. The release gate asks for 90% overall and 85% on follow-ups; this model was released below it as a
clear improvement on the previous one.

### What building it taught

1. **Search was the main problem, not the model.** "Who is the Batman?" reduces to the single word "batman",
   which is in most passages. Name matching fixed that.
2. **A small model answers reliably only when the sentence it needs is in front of it.** One fact per passage,
   and one passage per cast-list line, beat long passages.
3. **Never help the model in training in a way the product cannot.** Every training prompt is built by the same
   search the site uses.
4. **Score the build you ship.** A 4-bit build of the first v3 model lost 7 points; keeping every matrix at
   8-bit won them back for 110 MB more.
5. **An eval that only asks what you trained on will pass.** v2 scored 95.7% on its own questions and failed
   on the first follow-up a friend asked. v3's eval is written separately, by hand and by a teacher, about
   subjects held out of training.
6. **A bigger corpus can make search worse.** Giving every named subject a reserved place filled all three
   places with summaries; titles such as *Batman Live* captured everyday phrases.
7. **Training labels inherit the search's mistakes.** Answers labelled while search was weak taught the model
   to say "I don't have that" and to agree with users. Fixing search and rebalancing the data fixed both.

### Limits

- Comparisons and two-part questions are often half answered.
- About one follow-up in four still loses its thread.
- It can agree with a confident false claim.
- The download is large for phones, and CPU mode is slow.

BatLM answers only from its files; check the sources it shows.

## Licenses and credits

- Code: MIT.
- The corpus is from Wikipedia (CC BY-SA 4.0), so the derived index and model weights are CC BY-SA 4.0.
- Base model: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Apache-2.0). Teacher for facts, training data and judging: Qwen2.5-14B-Instruct (AWQ).
- BatLM is an unofficial, educational fan project. Batman and related characters are trademarks of DC Comics.
  This project is not affiliated with or endorsed by DC or Warner Bros. The bat emblem was drawn for this project.
