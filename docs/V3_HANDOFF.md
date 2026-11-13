# BatLM v3: handoff

Written 2026-11-13 for whoever continues this work (a person or an AI assistant with no memory of the
earlier sessions). Everything is on branch `v3`. `main` and batlm.vercel.app are untouched.

## Why v3 exists

The live model (BatLM-360M) scored 95.7% on its own eval but failed in real use:

- "Catwoman is a villian? Are you sure?" got "outside my case files".
- "was he an arkham patient?" (after a Joker question) got Harley Quinn's biography.
- "was joker an arkham patient?" got the same Harley Quinn biography.

Causes: no conversation memory; training data that was only "Who is X?" lookups answered by copying a
sentence; keyword-only search; a thin knowledge base; a small model; and an eval that only asked what
the model was trained on.

## Rules the owner set

- **Nothing runs locally.** Compute runs on GitHub Actions, Kaggle and Hugging Face. Code changes go
  through the GitHub API (`gh api`: blobs, tree, commit, update `refs/heads/v3`), with no clone.
- The owner picks the model from a bake-off table (accuracy, download size, speed).
- A model ships only if it passes the release gate (below). Similar or other problems must not surface.
- Secrets `KAGGLE_API_TOKEN` and `HF_TOKEN` are set as repository secrets. Never print them.

## What is done and working

| Piece | Where | State |
|---|---|---|
| Conversation memory in search and prompt | `batlm/rag.py`, `agent.py`, `prompt.py`; mirrored in `web/src/lib/` | Done, tested |
| Typo correction, subject-aware scoring, short follow-ups | `batlm/rag.py` / `web/src/lib/search.ts` | Done; Python and browser search are checked to match |
| Greetings get a welcome; a Batman question never gets the off-topic refusal | `agent.py` / `agent.ts` | Done |
| Knowledge base: 805 Wikipedia pages, 11,332 passages (was 333 / 5,913) | `batlm/scrape.py`, `data/chunks.json`, workflow `corpus` | Done |
| Hand-written eval: 223 turns, including the three failed conversations | `evals/handwritten.json` | Done |
| Release gate | `batlm/evals.py` (`GATE`, `gate()`) | Done |
| CPU eval of any adapter on a GitHub runner | workflow `eval` | Works |
| Eval of the real browser build in headless Chrome | workflow `browser-eval` | Works (about 1 token/s on the runner, so slow) |
| Teacher pipeline: facts, training conversations, held-out eval questions | `batlm/teacher.py`, `facts.py`, `sft_gen.py` | Proven on a 12-page trial |
| Training, scoring, browser export, upload to Hugging Face | `batlm/train_lora.py`, `export_web.py`, `cloud/kaggle_run.py`, `cloud/pipeline.py` | Proven on Qwen2.5-1.5B |
| Teacher judging of reference-answer questions | `batlm/evals.py --judge` | Fix applied, not yet confirmed in a run |

Baseline to beat (current live model with the v3 search, hand-written eval): 73% in PyTorch, 75.6% in
the real browser build; follow-ups about 50%, challenges 33%. The gate fails it and names the
screenshot turns.

## The release gate

Printed at the end of every eval run and stored in the report as `gate`. To pass:

- overall accuracy at least 0.90;
- follow-ups at least 0.85, every other question kind with 8 or more questions at least 0.80;
- no Batman question answered with the off-topic refusal;
- no answer cut off by the token limit;
- every turn of the `hand-talk-screenshot-*` conversations correct;
- no answers still waiting for the teacher's verdict.

## How to run things

Workflows only exist on `v3`, so they cannot be started by hand from the Actions page. Each one starts
when its trigger file changes in a push to `v3`:

| Workflow | Trigger file | What it does |
|---|---|---|
| `ci` | any push | tests, search report, web build |
| `corpus` | `.github/triggers/corpus.json` | re-download Wikipedia, rebuild passages and the browser index, commit |
| `pipeline` | `.github/triggers/pipeline.json` | one Kaggle stage (see below) |
| `eval` | `.github/triggers/eval.json` | PyTorch eval of an adapter on a CPU runner |
| `browser-eval` | `.github/triggers/browser-eval.json` | eval of a Hugging Face model repo in headless Chrome |

`pipeline.json` fields: `stage` (`teach` or `train`), `action` (`start`, or `collect` to only wait for and
fetch an existing Kaggle run), and for `train`: `name` (short candidate name), `base` (Hugging Face id of
a ChatML model), optional `epochs`, `top_k`. `limit` runs a quick trial on the first N pages.

The GitHub job waits at most 4.6 hours for Kaggle. If the Kaggle run is longer, the job ends with "still
running"; push the same trigger with `"action": "collect"` to pick the results up.

Results land here:

- `teach`: `data/facts.json`, `evals/generated.json`, rebuilt `web/public/data/index.json` (committed);
  training conversations in the private Hugging Face dataset `chintanjv/batlm-data`.
- `train`: browser model in the public repo `chintanjv/BatLM-v3-<name>`; reports in `evals/runs/<name>/`
  (`report_pytorch.json`, `report_onnx.json`, `model.json` with file sizes, `run.json` with failed steps).

## What remains

1. **Full teach run: done** (2026-11-13). 8,599 facts from 804 pages in `data/facts.json`; 19,331 training
   examples in the Hugging Face dataset `chintanjv/batlm-data` (4,167 follow-ups, 1,646 yes/no, 1,471
   challenges, 1,269 false-premise, 1,230 two-part, 984 comparisons, and more); 555 held-out eval
   conversations in `evals/generated.json`. The browser index now has 20,217 passages (8.7 MB).
   Search recall on the eval fell from 0.94 to 0.89 once the new `fact` questions were added: look at the
   `by_kind` numbers in the `ci` search report before blaming the models.
2. **Train the candidates** on the same data, one `train` run each:
   - `360m`: `HuggingFaceTB/SmolLM2-360M-Instruct`. **Started** 2026-11-13 14:35 PST (GitHub run 43520155888,
     Kaggle kernel `chintanjv/batlm-train-360m`).
   - `qwen05`: `Qwen/Qwen2.5-0.5B-Instruct`. **Started** 2026-11-13 14:36 PST (GitHub run 43520850400,
     Kaggle kernel `chintanjv/batlm-train-qwen05`).
   - `qwen15`: `Qwen/Qwen2.5-1.5B-Instruct`. **Not started**: Kaggle runs two GPU sessions at a time. Its
     browser file is 1.43 GB and its CPU fallback 2.6 GB (too big for most browsers). Use `"epochs": 1`;
     at batch size 2 it needs several hours, so expect to need `"action": "collect"`.
   For the two started runs: if `evals/runs/<name>/` exists on `v3`, they finished and committed. If not,
   look at the GitHub run; if it ended with "still running", push the same trigger with
   `"action": "collect"` (keep `stage`, `name` and `base`). Check `run.json` there for failed steps: the
   teacher-judging step had not yet succeeded in any run when this was written, and until it does the
   report shows answers "waiting for the teacher" and the gate fails for that reason alone.
   Consider `HuggingFaceTB/SmolLM2-1.7B-Instruct` only if the others miss the gate.
3. **Bake-off table** for the owner: accuracy by kind, gate verdict, download size, speed.
4. **Preview.** Vercel builds a preview for every push to `v3`. Point it at the chosen model with the
   Vercel environment variable `VITE_BATLM_MODEL` (and `VITE_BATLM_TOP_K` if training used a different
   `top_k`). Run `browser-eval` on the chosen repo.
5. **Release** only after the owner agrees: merge `v3` to `main`, set the production model id
   (`web/src/worker.ts` and `web/scripts/prune-dist.mjs` default to `chintanjv/BatLM-360M`), update
   `README.md` and `docs/MODEL_CARD.md` (limits and eval table), delete `evals/runs/trial-*` and the
   Hugging Face repo `chintanjv/BatLM-v3-trial-qwen15`.
6. **If no candidate passes the gate:** read `rows_pytorch.json` for the failing kinds; likely levers are
   more teacher conversations of that kind, `top_k` 5 for the larger models, and meaning-based
   (embedding) search, which was planned but deliberately not built yet because search already finds
   the answer for 94% of answerable questions.

## Pitfalls already hit (do not rediscover)

- **Kaggle dataset versions go stale.** Updating an existing dataset did not take effect for over
  30 minutes, so runs used an old upload. `pipeline.py` now creates a new dataset per run and the Kaggle
  script checks a run id. Old `batlm-*` datasets on Kaggle can be deleted.
- **The teacher sometimes fails to start** on the two T4s (`custom_all_reduce ... invalid argument`).
  It is now started with `--disable-custom-all-reduce` and three attempts.
- **Wikipedia is slow** from cloud machines (about 100 minutes for the corpus). The `corpus` workflow
  caches the pages; `pipeline.py` bundles them into one file for the teach stage.
- **GPU memory.** Training computes the loss on answer positions only and picks the batch size from the
  model size; without both, a 1.5B model does not fit a 15 GB T4.
- **Disk.** Kaggle's working directory is 20 GB; the teacher is installed under `/tmp` and build folders
  are deleted before judging.
- **Files over 2 GB** cannot be saved as one ONNX file; `export_web.py` falls back to external weights.
- **The `kaggle` command needs `six`** installed beside it on the runner.
- **Commits through the API:** build every file with a checked script and confirm the blob is not empty
  before committing. One commit in this work emptied a file through a shell-quoting mistake.
- Kaggle gives 30 GPU hours a week; roughly 13 were used on 2026-11-13 before the two training runs, so check the quota before starting `qwen15` the same week.
