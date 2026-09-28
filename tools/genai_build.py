"""Run onnxruntime-genai's model builder on torch 2.2.2 (the last Intel-Mac wheel).

The builder only references torch.uint16/32/64 inside dtype lookup tables; torch 2.2 lacks them,
so we alias placeholders before importing it. Usage mirrors the builder CLI:
  .venv-export/bin/python tools/genai_build.py -i <hf_dir> -o <out_dir> -p int4 -e webgpu
"""
import runpy
import sys

import torch

for name, fallback in (("uint16", torch.int16), ("uint32", torch.int32), ("uint64", torch.int64)):
    if not hasattr(torch, name):
        setattr(torch, name, fallback)

sys.argv = ["builder"] + sys.argv[1:]
runpy.run_module("onnxruntime_genai.models.builder", run_name="__main__")
