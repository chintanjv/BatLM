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

### State on 2026-11-18

**First bake-off (2026-11-13 data): both candidates failed the gate.**

| | 360m | qwen05 |
|---|---|---|
| overall (1,261 turns, PyTorch) | 0.713 | 0.715 |
| hand-written (221), PyTorch | 0.783 | 0.824 |
| hand-written, 4-bit build | not scored (config bug, since fixed) | 0.756 |
| browser files | 347 MB WebGPU / 583 MB CPU | 480 MB / 1,091 MB |

The live model scores 0.756 on the hand-written set in its browser build, so neither was an improvement once quantized. Reports are in
`evals/runs/<name>/`. The failures came from five causes:

1. **Search regressed when the corpus grew.** Every named subject got a reserved place, so "Who played Alfred in The Dark
   Knight?" got three summaries and no cast line; "Where does Batman live?" matched the stage show *Batman Live*.
   **Fixed 2026-11-18:**
   - At most two reserved places: one named subject and one carried over from the previous turn.
   - Phrasing matches are scaled by how many of the question's subjects a passage covers.
   - An ambiguous name ("the batman": the hero or the 2022 film) is one subject with two readings.
   - A title that is a character's name plus everyday words is not used as a name.
   - Every "Actor as Role" line of a cast list, and every such pair in the film cards, becomes a short passage
     (`rag.cast_facts`).
2. **A bias toward "yes"** on false premises ("Is Alfred Batman's father?" → "Yes"). The training data had about 3,100
   agreeing replies against 1,270 correcting ones. **Fixed in `sft_gen.py`:**
   - A `wrongclaim` question style.
   - Replies check claims instead of confirming them.
   - At most 1.2 "Yes" replies per "No".
3. **Too many "I don't have that"** replies, learned from labels made with the weak search. Declines are now capped
   at 8%, and the data is being relabelled with the fixed search.
4. **4-bit quantization cost about 7 points** (0.824 to 0.756). Being measured: the `export` stage rebuilds qwen05 with
   every matrix at 8-bit (`"bits": "all"`).
5. **The generated eval was noisy:** unanswerable openers, a strict judge, and phrase-style keys. **Fixed:**
   - Openers that name no subject are dropped (555 to 484).
   - The judge accepts any accurate answer to an open question.
   - Fact questions are judged by the teacher.

### Running when this was written (2026-11-18)

- GitHub run 44447193003: `teach` with `"skip_facts": true`. It regenerates the training conversations with the
  fixed search and rules, and keeps `data/facts.json`.
- GitHub run 44447316744: `export` of qwen05 with `"bits": "all"`. Results go to `evals/runs/qwen05-8bit/` and
  `chintanjv/BatLM-v3-qwen05-8bit`.

### Next

1. When `teach` has committed, train `360m`, `qwen05` and `qwen15` again (`"epochs": 2`; qwen15 `1`). Use the `bits`
   setting the export result supports. A train run with `"bits"` set builds that way.
2. Bake-off table for the owner: hand-written accuracy in the quantized build is the number that matters, plus the
   gate, file sizes and speed.
3. Preview on Vercel with `VITE_BATLM_MODEL`, then `browser-eval`, then release only with the owner's approval
   (release steps below).
4. If no candidate passes, read `rows_*.json` per kind before changing anything.

### Release steps (unchanged)

Merge `v3` to `main` and set the production model id. `web/src/worker.ts` and `web/scripts/prune-dist.mjs` default to
`chintanjv/BatLM-360M`, so both need the new id. Update `README.md` and `docs/MODEL_CARD.md`. Delete
`evals/runs/trial-*`, the Hugging Face repo `chintanjv/BatLM-v3-trial-qwen15`, the `try` branch, `.github/probe.py`
and `.github/workflows/try.yml`.

## Working rule: test on `try` first

Failing workflow runs on `v3` email the owner, who asked for fewer of them. Iterate on the `try` branch: its `try`
workflow runs the tests, the search report and `.github/probe.py` with every step allowed to fail, so it never sends
an email. Fast-forward `v3` to the tested commit only when the tests pass there.

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
- **SmolLM2 leaves `do_sample` empty** in its generation config; onnxruntime-genai refuses it (`evals.og_generate_fn` fills it in).
- **The `kaggle` command needs `six`** installed beside it on the runner.
- **Commits through the API:** build every file with a checked script and confirm the blob is not empty
  before committing. One commit in this work emptied a file through a shell-quoting mistake.
- Kaggle gives 30 GPU hours a week; roughly 13 were used on 2026-11-13 before the two training runs, so check the quota before starting `qwen15` the same week.
