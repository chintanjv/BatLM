"""Starts a stage of the training pipeline on Kaggle and brings its results back. Runs in GitHub Actions
(.github/workflows/pipeline.yml), never on a laptop.

The request is .github/triggers/pipeline.json:
  stage    "teach": the teacher writes facts, training conversations and eval questions
           "train": fine-tune one base model, score it and build its browser files
  action   "start": upload the code and data, start the Kaggle run, wait for it, collect the results
           "collect": only wait and collect (use when the workflow's own time limit ended the wait)
  name     train only: a short name for this candidate, e.g. "360m"
  base     train only: the Hugging Face id of the base model (any ChatML model)
  teacher  the teacher model's Hugging Face id (optional)
  epochs, top_k, limit   optional; limit runs a quick trial on the first N pages/subjects

Needs KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY) and HF_TOKEN in the environment.
Results: teach -> data/facts.json and evals/generated.json in the working tree, training conversations in the
private Hugging Face dataset <user>/batlm-data. train -> the browser model in <user>/BatLM-v3-<name>, eval
reports in evals/runs/<name>/.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
run = json.loads((ROOT / ".github" / "triggers" / "pipeline.json").read_text())
stage, name = run["stage"], run.get("name") or "360m"
slug = f"batlm-{stage}" + (f"-{name}" if stage == "train" else "")
WAIT = 4.6 * 3600  # GitHub ends a job after 6 hours


def kaggle(*args, check=True, quiet=False) -> str:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if not quiet or r.returncode:
        print(f"+ kaggle {' '.join(args)}\n{r.stdout}{r.stderr}".rstrip(), flush=True)
    if check and r.returncode:
        sys.exit(f"kaggle {' '.join(args[:2])} failed")
    return r.stdout + r.stderr


def username() -> str:
    for line in kaggle("config", "view", check=False, quiet=True).splitlines():
        if "username:" in line and line.split("username:")[1].strip() not in ("", "None"):
            return line.split("username:")[1].strip()
    return os.environ.get("KAGGLE_USERNAME") or sys.exit("could not work out the Kaggle username: set the KAGGLE_USERNAME secret")


def hub():
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    return api, api.whoami()["name"]


def start(user: str):
    shutil.rmtree(BUILD, ignore_errors=True)
    src, kernel = BUILD / "dataset", BUILD / "kernel"
    src.mkdir(parents=True)
    kernel.mkdir()
    (ROOT / "cloud" / "run.json").write_text(json.dumps(run))
    paths = ["batlm", "cloud/run.json", "data/chunks.json", "data/facts.json", "docs", "evals/handwritten.json", "evals/generated.json"]
    if stage == "teach":
        paths.append("data/raw")  # the full pages: the teacher reads past the word cap of the search passages
    else:
        api, me = hub()
        try:
            got = api.hf_hub_download(f"{me}/batlm-data", "sft_teacher.jsonl", repo_type="dataset", local_dir=str(ROOT / "checkpoints"))
            print("teacher-written training conversations:", got)
            paths.append("checkpoints/sft_teacher.jsonl")
        except Exception as e:  # no teach stage has run yet
            print(f"WARNING: no teacher-written conversations ({type(e).__name__}); training on the templates only")
    with tarfile.open(src / "batlm-src.tar.gz", "w:gz") as tar:
        for p in paths:
            if (ROOT / p).exists():
                tar.add(ROOT / p, arcname=p, filter=lambda t: None if "__pycache__" in t.name else t)
            else:
                print("not present, left out:", p)
    dataset = f"{user}/{slug}-src"
    (src / "dataset-metadata.json").write_text(json.dumps({"title": f"{slug} src", "id": dataset, "licenses": [{"name": "CC-BY-SA-4.0"}]}))
    if "ready" in kaggle("datasets", "status", dataset, check=False, quiet=True):
        kaggle("datasets", "version", "-p", str(src), "-m", "update")
    else:
        kaggle("datasets", "create", "-p", str(src))
    print("waiting for Kaggle to process the dataset…", flush=True)
    time.sleep(30)
    for _ in range(60):
        if "ready" in kaggle("datasets", "status", dataset, check=False, quiet=True):
            break
        time.sleep(10)
    shutil.copy(ROOT / "cloud" / "kaggle_run.py", kernel / "kaggle_run.py")
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug, "code_file": "kaggle_run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [dataset]}))
    kaggle("kernels", "push", "-p", str(kernel))


def wait(user: str) -> str:
    t0, last = time.time(), ""
    while True:
        s = kaggle("kernels", "status", f"{user}/{slug}", check=False, quiet=True).strip().lower()
        if s != last:
            print(f"[{(time.time() - t0) / 60:5.0f} min] {s}", flush=True)
            last = s
        if "complete" in s:
            return "complete"
        if "error" in s or "cancel" in s:
            return "error"
        if time.time() - t0 > WAIT:
            return "running"
        time.sleep(120)


def collect(user: str):
    out = BUILD / "out"
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{user}/{slug}", "-p", str(out), quiet=True)
    log = out / f"{slug}.log"
    if log.exists():  # Kaggle's log is a JSON list of {"stream_name", "data"}
        try:
            text = "".join(e.get("data", "") for e in json.loads(log.read_text()))
        except ValueError:
            text = log.read_text()
        (BUILD / "kaggle.log").write_text(text)
        print("---- end of the Kaggle log ----\n" + text[-12000:], flush=True)
    res = out / "out"
    status = json.loads((res / "status.json").read_text()) if (res / "status.json").exists() else {"failed": ["no status file"]}
    api, me = hub()
    if stage == "teach":
        for f, dst in (("facts.json", "data/facts.json"), ("generated.json", "evals/generated.json")):
            if (res / f).exists():
                shutil.copy(res / f, ROOT / dst)
        if (res / "sft_teacher.jsonl").exists():
            api.create_repo(f"{me}/batlm-data", repo_type="dataset", private=True, exist_ok=True)
            api.upload_file(path_or_fileobj=str(res / "sft_teacher.jsonl"), path_in_repo="sft_teacher.jsonl", repo_id=f"{me}/batlm-data", repo_type="dataset")
            print(f"training conversations -> https://huggingface.co/datasets/{me}/batlm-data")
    else:
        runs = ROOT / "evals" / "runs" / name
        runs.mkdir(parents=True, exist_ok=True)
        for f in (res / "evals").glob("*.json") if (res / "evals").exists() else []:
            shutil.copy(f, runs / f.name)
        (runs / "run.json").write_text(json.dumps({**run, "failed": status["failed"]}, indent=1))
        if (res / "model").exists():
            repo = f"{me}/BatLM-v3-{name}"
            api.create_repo(repo, exist_ok=True)
            api.upload_folder(folder_path=str(res / "model"), repo_id=repo)
            if (res / "batlm-lora").exists():
                api.upload_folder(folder_path=str(res / "batlm-lora"), repo_id=repo, path_in_repo="lora")
            sizes = {f.name: round(f.stat().st_size / 1e6) for f in (res / "model" / "onnx").glob("*.onnx")}
            (runs / "model.json").write_text(json.dumps({"repo": repo, "base": run.get("base"), "onnx_mb": sizes}, indent=1))
            print(f"browser model -> https://huggingface.co/{repo}  {sizes}")
    (BUILD / "collected").write_text(stage)
    if status["failed"]:
        sys.exit(f"these steps failed on Kaggle: {status['failed']}")


def main():
    user = username()
    if run.get("action", "start") == "start":
        start(user)
        time.sleep(60)
    state = wait(user)
    if state == "running":
        sys.exit(f"{slug} is still running on Kaggle. Start this workflow again with \"action\": \"collect\" to pick up the results.")
    collect(user)
    if state == "error":
        sys.exit(f"{slug} ended with an error on Kaggle")


if __name__ == "__main__":
    main()
