#!/bin/sh
# Train, score and export BatLM on Kaggle's free GPU, then fetch the results.
#   cloud/kaggle.sh push     upload code + passages as a private dataset and start the run
#   cloud/kaggle.sh status   is it still running?
#   cloud/kaggle.sh fetch    download results into checkpoints/, web/public/models/ and evals/
# Needs KAGGLE_API_TOKEN in .env and a phone-verified Kaggle account (for GPU and internet access).
set -e
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
KAGGLE=.venv/bin/kaggle
USER=$($KAGGLE config view | sed -n 's/^- username: //p')
BUILD=checkpoints/kaggle

case "$1" in
  push)
    rm -rf "$BUILD" && mkdir -p "$BUILD/dataset" "$BUILD/kernel"
    COPYFILE_DISABLE=1 tar -czf "$BUILD/dataset/batlm-src.tar.gz" --exclude __pycache__ batlm data/chunks.json docs
    printf '{"title": "BatLM source", "id": "%s/batlm-src", "licenses": [{"name": "CC-BY-SA-4.0"}]}\n' "$USER" > "$BUILD/dataset/dataset-metadata.json"
    if $KAGGLE datasets status "$USER/batlm-src" >/dev/null 2>&1; then
      $KAGGLE datasets version -p "$BUILD/dataset" -m "update"
    else
      $KAGGLE datasets create -p "$BUILD/dataset"
    fi
    cp cloud/kaggle_run.py "$BUILD/kernel/"
    printf '{"id": "%s/batlm-train", "title": "BatLM train", "code_file": "kaggle_run.py", "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true, "dataset_sources": ["%s/batlm-src"]}\n' "$USER" "$USER" > "$BUILD/kernel/kernel-metadata.json"
    echo "waiting for Kaggle to process the dataset…"
    until $KAGGLE datasets status "$USER/batlm-src" 2>/dev/null | grep -q ready; do sleep 5; done
    $KAGGLE kernels push -p "$BUILD/kernel"
    ;;
  status) $KAGGLE kernels status "$USER/batlm-train" ;;
  fetch)
    rm -rf "$BUILD/output" && $KAGGLE kernels output "$USER/batlm-train" -p "$BUILD/output"
    rm -rf checkpoints/batlm-lora web/public/models/batlm-360m
    mv "$BUILD/output/out/batlm-lora" checkpoints/batlm-lora
    mv "$BUILD/output/out/batlm-360m" web/public/models/batlm-360m
    mkdir -p evals && cp "$BUILD/output/out/evals/"*.json evals/
    echo "fetched: checkpoints/batlm-lora, web/public/models/batlm-360m, evals/"
    ;;
  *) echo "usage: cloud/kaggle.sh push|status|fetch"; exit 1 ;;
esac
