"""Runs on Kaggle (free GPU): fine-tune BatLM, score it, and build the browser models.

Pushed by cloud/kaggle.sh together with a private dataset holding batlm/, data/chunks.json and docs/.
Everything worth keeping is left in /kaggle/working/out, which `kaggle kernels output` downloads.
"""
import glob
import os
import shutil
import subprocess
import sys
import tarfile

WORK = "/kaggle/working"
ROOT = f"{WORK}/BatLM"


def sh(*args, **kw):
    print("+", " ".join(args), flush=True)
    subprocess.check_call(args, **kw)


# The dataset may arrive as the archive we uploaded or already unpacked, depending on Kaggle.
found = glob.glob("/kaggle/input/**/batlm/rag.py", recursive=True)
if found:
    shutil.copytree(os.path.dirname(os.path.dirname(found[0])), ROOT)
else:
    archive = glob.glob("/kaggle/input/**/batlm-src.tar.gz", recursive=True)[0]
    os.makedirs(ROOT)
    tarfile.open(archive).extractall(ROOT)
print(sorted(os.listdir(ROOT)), flush=True)

py, env = sys.executable, {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
# Kaggle's Python is newer than the local one, so use its preinstalled torch/transformers and add only what is missing.
try:
    import peft  # noqa: F401
except ImportError:
    sh(py, "-m", "pip", "install", "-q", "peft")
# peft refuses to start if it finds Kaggle's old torchao, an optional package this project does not use.
subprocess.call([py, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
import transformers
print("transformers", transformers.__version__, "python", sys.version.split()[0], flush=True)
sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv")
sh(py, "-u", "-m", "batlm.train_lora", "--epochs", "2", cwd=ROOT, env=env)
sh(py, "-u", "-m", "batlm.evals", cwd=ROOT, env=env)

try:  # the versions the local export environment pins; fall back to current ones if this Python has no build of them
    sh(py, "-m", "pip", "install", "-q", "onnx==1.19.1", "onnxruntime==1.23.2", "onnxruntime-genai==0.11.4", "onnx-ir")
except subprocess.CalledProcessError:
    sh(py, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxruntime-genai", "onnx-ir")
sh(py, "-u", "-m", "batlm.export_web", cwd=ROOT, env={**env, "BATLM_EXPORT_PYTHON": py})

out = f"{WORK}/out"
os.makedirs(out, exist_ok=True)
shutil.move(f"{ROOT}/checkpoints/batlm-lora", f"{out}/batlm-lora")
shutil.move(f"{ROOT}/web/public/models/batlm-360m", f"{out}/batlm-360m")
shutil.move(f"{ROOT}/evals", f"{out}/evals")
shutil.rmtree(ROOT)  # the merged fp32 weights and build caches are large and can be rebuilt from the adapter
for path, _, files in os.walk(out):
    for f in files:
        print(f"{os.path.getsize(os.path.join(path, f)) / 1e6:8.1f} MB  {os.path.join(path, f)}")
