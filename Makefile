# BatLM pipeline. `make all` rebuilds everything from scratch (about an hour on an Apple Silicon laptop,
# most of it the Wikipedia download, which is rate-limited).
PY := .venv/bin/python

.PHONY: setup data cards lora evals export test web all cloud

setup:            ## Python 3.11 environments (training + ONNX export) and web dependencies
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements.txt
	uv venv --python 3.11 .venv-export && uv pip install --python .venv-export/bin/python -r requirements-export.txt
	cd web && npm install

data:             ## download the Wikipedia pages and cut them into passages -> data/chunks.json
	$(PY) -m batlm.scrape

cards:            ## check every fact card against its Wikipedia page -> data/cards.json
	$(PY) -m batlm.cards

lora:             ## fine-tune SmolLM2-360M-Instruct -> checkpoints/batlm-merged (about 20 min on a free cloud GPU; far longer on a laptop)
	$(PY) -m batlm.train_lora --epochs 2

evals:            ## search and answer accuracy on the held-out questions -> evals/
	$(PY) -m batlm.evals

export:           ## search index for the site + 4-bit ONNX models for the browser
	$(PY) -m batlm.export_web

test:
	$(PY) -m pytest -q tests

web:              ## local dev server at http://localhost:5173
	cd web && npm run dev

all: data cards lora evals export test

cloud:            ## train, score and export on a free Kaggle GPU instead (needs KAGGLE_API_TOKEN in .env)
	cloud/kaggle.sh push
