"""The teacher: a large open-weight model on a cloud GPU that writes and checks BatLM's training data.

It runs as a vLLM server in its own Python environment, so its packages cannot disturb training, and is
asked many prompts at once. Nothing here ships to the browser.
"""
import concurrent.futures
import json
import os
import re
import subprocess
import time

import requests

MODEL = os.environ.get("BATLM_TEACHER") or "Qwen/Qwen2.5-14B-Instruct-AWQ"
URL = "http://127.0.0.1:8000/v1"


def parse_json(text: str):
    """The JSON value in a model reply, or None. Tolerates code fences and stray words around it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M).strip()
    inner = [text[text.find(a): text.rfind(b) + 1] for a, b in sorted(("[]", "{}"), key=lambda p: text.find(p[0]) % (len(text) + 1))]
    for cand in [text] + inner:  # whichever bracket opens first is the outer value
        try:
            return json.loads(cand)
        except ValueError:
            continue
    return None


class Teacher:
    def __init__(self, model: str = MODEL, context: int = 8192, env_dir: str = "/tmp/teacher-env"):
        import torch
        py = f"{env_dir}/bin/python"
        if not os.path.exists(py):
            subprocess.check_call(["pip", "install", "-q", "uv"])
            subprocess.check_call(["uv", "venv", "--python", "3.11", env_dir])
            subprocess.check_call(["uv", "pip", "install", "-q", "--python", py, "vllm"])
        self.log_path = "/kaggle/working/teacher.log" if os.path.isdir("/kaggle/working") else "teacher.log"
        gpus = max(torch.cuda.device_count(), 1)
        for attempt in range(3):
            self.log = open(self.log_path, "a")
            print(f"starting teacher {model} on {gpus} GPU(s)", flush=True)
            # vLLM's own multi-GPU all-reduce sometimes crashes at start-up on T4s ("invalid argument"); the standard one is reliable.
            self.proc = subprocess.Popen(
                [py, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--served-model-name", "teacher", "--port", "8000",
                 "--max-model-len", str(context), "--tensor-parallel-size", str(gpus), "--gpu-memory-utilization", "0.92",
                 "--disable-custom-all-reduce"],
                stdout=self.log, stderr=subprocess.STDOUT)
            t0, up = time.time(), False
            while self.proc.poll() is None and time.time() - t0 < 2700 and not up:
                try:
                    up = requests.get(URL + "/models", timeout=5).ok
                except requests.RequestException:
                    pass
                if not up:
                    time.sleep(10)
            if up:
                print(f"teacher ready after {time.time() - t0:.0f}s", flush=True)
                return
            self.close()
            print(f"the teacher did not start (attempt {attempt + 1}):\n" + open(self.log_path).read()[-3000:], flush=True)
            time.sleep(30)
        raise RuntimeError("the teacher did not start after 3 attempts; see teacher.log")

    def _one(self, prompt: str, system: str | None, max_tokens: int, temperature: float) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        for attempt in range(4):
            try:
                r = requests.post(URL + "/chat/completions", timeout=1200, json={
                    "model": "teacher", "messages": msgs, "max_tokens": max_tokens, "temperature": temperature})
                if r.ok:
                    return (r.json()["choices"][0]["message"]["content"] or "").strip()
                if r.status_code == 400:  # the prompt does not fit the context window
                    return ""
            except requests.RequestException:
                pass
            time.sleep(5 * (attempt + 1))
        return ""

    def ask(self, prompts: list[str], system: str | None = None, max_tokens: int = 700, temperature: float = 0.7, workers: int = 64) -> list[str]:
        """One reply per prompt, in order. An empty string means the teacher could not answer that prompt."""
        out, t0 = [], time.time()
        with concurrent.futures.ThreadPoolExecutor(workers) as ex:
            for start in range(0, len(prompts), 1000):
                out += list(ex.map(lambda p: self._one(p, system, max_tokens, temperature), prompts[start:start + 1000]))
                print(f"  teacher: {len(out)} of {len(prompts)} prompts, {time.time() - t0:.0f}s", flush=True)
        return out

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()
