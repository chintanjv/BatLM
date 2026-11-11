"""Runs onnxruntime-genai's model builder with a chosen mix of 4-bit and 8-bit layers.

    BATLM_8BIT=mixed|attn|attn+down|all  python -m batlm.quant_build <builder arguments>

The builder's own "k_quant_mixed" recipe keeps about a third of the layers at 8-bit. A model this
small loses a lot of accuracy at 4-bit, so this wrapper can keep more of it at 8-bit:
  mixed      the builder's recipe, unchanged
  attn       every attention matrix at 8-bit, the MLPs at 4-bit
  attn+down  attention and each MLP's output matrix at 8-bit
  all        every matrix at 8-bit
Run with the export environment's Python (see batlm/export_web.py).
"""
import os
import runpy
import sys

import onnxruntime_genai.models
from onnxruntime.quantization.matmul_nbits_quantizer import KQuantWeightOnlyQuantConfig

# The builder imports its classes as the top-level package `builders`; patch that same copy.
sys.path.insert(0, os.path.dirname(onnxruntime_genai.models.__file__))
from builders import base  # noqa: E402

ATTN = ["attn/q_proj", "attn/k_proj", "attn/v_proj", "attn/qkv_proj", "attn/o_proj"]
MLP = ["mlp/gate_proj", "mlp/up_proj", "mlp/down_proj"]
mode = os.environ.get("BATLM_8BIT", "mixed")

if mode != "mixed":
    def eight_bit(self, quant_method):
        names = ATTN + {"all": MLP, "attn+down": ["mlp/down_proj"]}.get(mode, [])
        config = {f"/model/layers.{i}/{n}/MatMul": {"bits": 8} for i in range(self.num_layers) for n in names}
        config["/lm_head/MatMul"] = {"bits": 8}
        return KQuantWeightOnlyQuantConfig(customized_weight_config=config)
    base.Model.make_int4_algo_config = eight_bit

sys.argv = ["builder"] + sys.argv[1:]
runpy.run_module("onnxruntime_genai.models.builder", run_name="__main__")
