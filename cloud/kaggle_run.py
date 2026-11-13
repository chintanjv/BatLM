"""Runs on Kaggle's free GPU. cloud/pipeline.py uploads the repository as a private dataset and pushes
this script; cloud/run.json inside the dataset says which stage to run:

  teach   the teacher writes facts from every page, then training conversations and held-out eval questions
  train   fine-tune one base model, score it, build the browser models, score the quantized build

Everything worth keeping is left in /kaggle/working/out, which `kaggle kernels output` downloads.
A step that fails is recorded in out/status.json and the run carries on, so finished work is never lost.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback

WORK = "/kaggle/working"
ROOT = f"{WORK}/BatLM"
OUT = f"{WORK}/out"
failed = []


def sh(*args, **kw):
    print("+", " ".join(a if len(a) < 200 else a[:60] + "…" for a in args), flush=True)
    subprocess.check_call(args, **kw)


def step(label, fn, must=False):
    try:
        fn()
        return True
    except Exception:
        traceback.print_exc()
        failed.append(label)
        if must:
            finish()
            sys.exit(0)
        return False


def keep(src, name=None):
    if os.path.exists(src):
        shutil.move(src, f"{OUT}/{name or os.path.basename(src)}")


def finish():
    for f in ("teacher.log",):
        keep(f"{WORK}/{f}")
    json.dump({"failed": failed, "run": run}, open(f"{OUT}/status.json", "w"))
    shutil.rmtree(ROOT, ignore_errors=True)  # merged weights and build caches are large and can be rebuilt
    shutil.rmtree(f"{WORK}/teacher-env", ignore_errors=True)
    for path, _, files in os.walk(OUT):
        for f in files:
            print(f"{os.path.getsize(os.path.join(path, f)) / 1e6:8.1f} MB  {os.path.join(path, f)}")
    print("FAILED STEPS:" if failed else "all steps finished", failed or "", flush=True)


# The dataset may arrive as the archive we uploaded or already unpacked, depending on Kaggle.
found = glob.glob("/kaggle/input/**/batlm/rag.py", recursive=True)
if found:
    shutil.copytree(os.path.dirname(os.path.dirname(found[0])), ROOT)
else:
    archive = glob.glob("/kaggle/input/**/batlm-src.tar.gz", recursive=True)[0]
    os.makedirs(ROOT)
    tarfile.open(archive).extractall(ROOT)
run = json.load(open(f"{ROOT}/cloud/run.json"))
print("run:", run, sorted(os.listdir(ROOT)), flush=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{ROOT}/checkpoints", exist_ok=True)

py = sys.executable
env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "BATLM_BASE": run.get("base") or "",
       "BATLM_TEACHER": run.get("teacher") or "", "BATLM_TOP_K": str(run.get("top_k") or "")}
subprocess.call(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])

TEACH = """
import json, sys
from batlm import facts, sft_gen
from batlm.teacher import Teacher
limit = int(sys.argv[1]) or None
teacher = Teacher()
try:
    facts.OUT.write_text(json.dumps(facts.build(teacher, limit), ensure_ascii=False, separators=(",", ":")))
    sft_gen.main(teacher, limit)
finally:
    teacher.close()
"""

if run["stage"] == "teach":
    step("teach", lambda: sh(py, "-u", "-c", TEACH, str(run.get("limit") or 0), cwd=ROOT, env=env))
    keep(f"{ROOT}/data/facts.json")
    keep(f"{ROOT}/checkpoints/sft_teacher.jsonl")
    keep(f"{ROOT}/evals/generated.json")
else:
    # Kaggle's Python is newer than the pinned one, so use its preinstalled torch/transformers and add only what is missing.
    try:
        import peft  # noqa: F401
    except ImportError:
        sh(py, "-m", "pip", "install", "-q", "peft")
    # peft refuses to start if it finds Kaggle's old torchao, an optional package this project does not use.
    subprocess.call([py, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
    import transformers
    print("transformers", transformers.__version__, "python", sys.version.split()[0], flush=True)
    step("train", lambda: sh(py, "-u", "-m", "batlm.train_lora", "--epochs", str(run.get("epochs") or 2), cwd=ROOT, env=env), must=True)
    shutil.copytree(f"{ROOT}/checkpoints/batlm-lora", f"{OUT}/batlm-lora")  # safe before anything else can go wrong
    step("eval", lambda: sh(py, "-u", "-m", "batlm.evals", "--name", "pytorch", cwd=ROOT, env=env))

    def export():
        try:  # the versions the export environment pins; fall back to current ones if this Python has no build of them
            sh(py, "-m", "pip", "install", "-q", "onnx==1.19.1", "onnxruntime==1.23.2", "onnxruntime-genai==0.11.4", "onnx-ir")
        except subprocess.CalledProcessError:
            sh(py, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxruntime-genai", "onnx-ir")
        sh(py, "-u", "-m", "batlm.export_web", cwd=ROOT, env={**env, "BATLM_EXPORT_PYTHON": py})
        shutil.copytree(f"{ROOT}/web/public/models/batlm-360m", f"{OUT}/model")

    if step("export", export):
        # The build the browser's CPU path runs, scored on the hand-written conversations: does quantization cost accuracy?
        step("eval-onnx", lambda: sh(py, "-u", "-m", "batlm.evals", "--onnx", f"{ROOT}/checkpoints/genai-q4", "--only", "hand",
                                     "--name", "onnx", cwd=ROOT, env=env))
    if os.path.exists(f"{ROOT}/evals/generated.json") and os.path.exists(f"{ROOT}/evals/rows_pytorch.json"):
        step("judge", lambda: sh(py, "-u", "-m", "batlm.evals", "--judge", "evals/rows_pytorch.json", cwd=ROOT, env=env))
    os.makedirs(f"{OUT}/evals", exist_ok=True)
    for f in glob.glob(f"{ROOT}/evals/r*_*.json"):
        keep(f, "evals/" + os.path.basename(f))
finish()
