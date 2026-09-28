# BatLM end-to-end pipeline. `make all` reproduces everything from scratch on a laptop CPU.
PY := .venv/bin/python

.PHONY: setup data nano lora evals export test web all

setup:            ## Python 3.11 env (Intel-Mac friendly: torch 2.2.2 is the last x86_64 macOS wheel)
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements.txt
	cd web && npm install

data:             ## scrape + clean + chunk the two source pages
	$(PY) -m batlm.scrape

nano:             ## tokenizer + pretrain BatLM-nano from scratch (~15 min)
	$(PY) -m batlm.train_nano --iters 800

lora:             ## LoRA instruction-tune SmolLM2-360M-Instruct -> BatLM (~1-2 h on CPU)
	$(PY) -m batlm.train_lora --epochs 3

evals:            ## retrieval + generation evals, ablations -> evals/report.json
	$(PY) -m batlm.evals

export:           ## ONNX/quantize + web data
	$(PY) -m batlm.export_web

test:
	$(PY) -m pytest -q tests && cd web && npm run parity

web:              ## local dev server at http://localhost:5173
	cd web && npm run dev

all: data nano lora evals export test
