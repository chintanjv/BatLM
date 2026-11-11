# 🦇 BatLM

**A small language model that answers questions about the Batman universe, running entirely in your browser.**

🦇 **Live: [batlm.vercel.app](https://batlm.vercel.app)** · 🤗 **Model: [chintanjv/BatLM-360M](https://huggingface.co/chintanjv/BatLM-360M)**

> The first visit downloads the model (about 350 MB with WebGPU, about 580 MB in CPU mode); it is cached after that.
> Chrome and Edge with WebGPU answer in a second or two. Without WebGPU the site falls back to CPU mode, which is much slower.

BatLM is a learning project that covers every layer of a small LLM product: a corpus, hand-checked facts,
search, fine-tuning, evals, quantization and in-browser deployment. Nothing is sent to a server: search and
the model both run on your device.

## How it answers a question

```
question ─▶ search ─▶ off-topic gate ─▶ BatLM-360M ─▶ answer + sources
            │                              ▲
            └── 3 passages ────────────────┘
```

1. **Search** ranks 6,199 passages: 286 hand-written facts and 5,913 passages from 333 Wikipedia pages.
   It combines BM25 with two things plain BM25 lacks: every subject's names and aliases, and the usual ways
   of asking for each fact. The best fact about whatever the question names is shown first.
2. **The gate** refuses off-topic questions without calling the model.
3. **BatLM-360M** ([SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) with a
   LoRA fine-tune) reads the three passages and writes a one- or two-sentence answer, or a short list.

Answers follow mainline comics first, with a clause for a well-known film or TV difference where it matters.

## The pipeline

| Stage | File | What it does |
|---|---|---|
| Corpus | `batlm/scrape.py` | Downloads 333 Wikipedia pages and cuts them into ~80-word passages |
| Facts | `batlm/cards.py` | 139 hand-written cards (286 facts), each checked against its Wikipedia page |
| Search | `batlm/rag.py` ↔ `web/src/lib/search.ts` | BM25 + name matching + question matching, identical in Python and the browser |
| Training data | `batlm/sft_data.py` | About 2,000 questions generated from the facts and page leads, plus held-out eval questions |
| Fine-tune | `batlm/train_lora.py` | LoRA on SmolLM2-360M, using the passages search really returns for each question |
| Answer | `batlm/agent.py` ↔ `web/src/lib/agent.ts` | Search, off-topic gate, prompt |
| Evals | `batlm/evals.py` | Scores the PyTorch model, or answers collected from a real browser |
| Export | `batlm/export_web.py`, `batlm/quant_build.py` | Search index for the site; 4/8-bit ONNX models for WebGPU and CPU |
| Site | `web/` | Vite + React; inference in a Web Worker with transformers.js |

`tests/test_pipeline.py` guards the two failures this project has actually had: the prompt drifting between
training and the browser, and the browser search returning different passages from the Python search.

## Run it yourself

```bash
brew install uv            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
make setup                 # Python environments + web dependencies
make data cards            # download the corpus (rate-limited, ~1 h the first time) and check the facts
make lora evals export     # train, score, build the browser models
make test
make web                   # http://localhost:5173
python -m batlm.agent      # the same pipeline in your terminal
```

Training takes about 20 minutes on a free Kaggle GPU and much longer on a laptop. To train in the cloud, put
`KAGGLE_API_TOKEN=...` in `.env` and run `cloud/kaggle.sh push`, then `cloud/kaggle.sh fetch` when it finishes.

To score the model exactly as visitors get it (the quantized build, in Chrome, on WebGPU):

```bash
python -m batlm.evals --questions
(cd web && npm run dev &) && (cd web && node scripts/e2e-browser.mjs --questions ../evals/questions.json --out ../evals/browser_run.json)
python -m batlm.evals --browser evals/browser_run.json
```

## Results

161 held-out questions, answered by the shipped quantized model in headless Chrome on WebGPU.
An answer is correct when it contains every required fact, makes no listed false claim, is not a refusal
and ends before the token limit. Off-topic and unanswerable questions are correct when the model declines.

| Questions | n | Correct |
|---|---|---|
| Must-pass examples ("who is the batman?", "top 5 Batman villains", …) | 13 | 100% |
| Trained facts, new phrasing | 59 | 93.2% |
| Facts never seen in training | 38 | 94.7% |
| Wikipedia pages with no fact card | 18 | 94.4% |
| Batman questions the files cannot answer | 13 | 100% |
| Off-topic | 20 | 100% |
| **All** | **161** | **95.7%** |

Search puts the answering passage in the top three for 97.7% of the answerable questions.
Every row is in `evals/rows_browser.json`.

### What building it taught

1. **Search was the main problem, not the model.** "Who is the Batman?" reduces to the single word "batman",
   which is in most passages. Name matching fixed that.
2. **A 360M model answers reliably only when the sentence it needs is the first thing it reads.** With long
   fact cards it picked the wrong sentence; with one fact per passage it does not.
3. **Never help the model in training in a way the product cannot.** The first version slipped the correct
   passage into training prompts when search missed it, which hid the search problem entirely.
4. **Score the build you ship.** The full-precision model scored 95.7%; the default 4-bit build of the same
   weights scored 89.4% in the browser, mostly through wrongful refusals. Keeping attention and each MLP
   output matrix at 8-bit costs 49 MB and brings it back to 95.7%.
5. **A substring scorer flatters.** Read the rows, not just the averages.

### Limits

BatLM answers what its files cover. Outside them the right behaviour is to say so, and it sometimes answers
with a related fact instead. It does not hold a conversation: each question is answered on its own.
Compound questions ("who is X and what does he use?") usually get only one half answered.

## Licenses and credits

- Code: MIT.
- The corpus is from Wikipedia (CC BY-SA 4.0), so the derived index and model weights are CC BY-SA 4.0.
- Base model: [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) (Apache-2.0).
- BatLM is an unofficial, educational fan project. Batman and related characters are trademarks of DC Comics.
  This project is not affiliated with or endorsed by DC or Warner Bros. The bat emblem was drawn for this project.
