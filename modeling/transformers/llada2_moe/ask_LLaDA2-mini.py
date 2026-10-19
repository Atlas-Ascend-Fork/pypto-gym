# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""
LLaDA2.0-mini inference script (baseline vs PyPTO).

LLaDA2 uses block-wise masked diffusion, NOT autoregressive generation.
The generate() API takes gen_length/steps/block_length, not max_new_tokens.

Usage:
    python ask_LLaDA2-mini.py --model-path /path/to/LLaDA2.0-mini [--use_pypto] [--device 14]
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
import torch
import torch_npu  # noqa: F401

# Patch: add 'default' rope type if missing (transformers >= 4.57 compat)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
if "default" not in ROPE_INIT_FUNCTIONS:
    def _compute_default_rope_params(config=None, device=None, seq_len=None, **kwargs):
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq_res = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64)
                     .to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq_res, 1.0
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_params

logging.basicConfig(level=logging.INFO, format="%(message)s")

parser = argparse.ArgumentParser(description="LLaDA2.0-mini inference")
parser.add_argument("--model-path", required=True, help="Path to model weights")
parser.add_argument("--device", default=14, type=int, help="NPU device ID")
parser.add_argument("--prompt", default="Explain the concept of mixture of experts in neural networks.",
                    help="Input prompt")
parser.add_argument("--output_length", default=100, type=int, help="Generation length (tokens)")
parser.add_argument("--steps", default=32, type=int, help="Diffusion steps per block")
parser.add_argument("--block_length", default=32, type=int, help="Block length for masked diffusion")
parser.add_argument("--use_pypto", action="store_true", help="Enable PyPTO fused kernels")
parser.add_argument("--report-file", default=None, help="Output JSON report path")
args = parser.parse_args()

torch.npu.set_device(args.device)
os.environ["TILE_FWK_DEVICE_ID"] = str(args.device)

# ---- PyPTO setup (must be BEFORE model load) ----
if args.use_pypto:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, os.path.join(repo_root, "src"))
    # Install the modified modeling file into model directory
    patched_src = os.path.join(repo_root, "src", "pypto_gym", "transformers",
                               "llada2_moe", "modeling_llada2_moe.py")
    patched_dst = os.path.join(args.model_path, "modeling_llada2_moe.py")
    if os.path.exists(patched_src):
        backup = patched_dst + ".orig"
        if not os.path.exists(backup):
            shutil.copy2(patched_dst, backup)
        shutil.copy2(patched_src, patched_dst)
        logging.info(f"[PyPTO] Installed patched modeling to {patched_dst}")
    # Import real kernel adapter module and register under expected name
    from ops.pypto_tensor import llada2_moe as llada2_kernels
    llada2_kernels.USE_PTO_EXPERT_FFN = True
    sys.modules["llada2_pto_kernels"] = llada2_kernels

from transformers import AutoModelForCausalLM, AutoTokenizer

metrics = {}

# ---- Load model ----
logging.info(f"Device: npu:{args.device}")
logging.info(f"Model:  {args.model_path}")

torch.npu.reset_peak_memory_stats()
t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True,
                                           trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, dtype=torch.bfloat16,
    device_map={"": f"npu:{args.device}"},
    local_files_only=True, trust_remote_code=True,
)
torch.npu.synchronize()
metrics["model_load_s"] = round(time.perf_counter() - t0, 3)
metrics["model_load_peak_mem_mb"] = round(torch.npu.max_memory_allocated() / 1024**2, 1)
torch.npu.reset_peak_memory_stats()

metrics["mode"] = "pypto" if args.use_pypto else "baseline"
logging.info(f"Mode: {metrics['mode']}")

# ---- Tokenize ----
# LLaDA2 generate() takes raw input_ids tensor, not **kwargs
input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(f"npu:{args.device}")
input_len = input_ids.shape[1]
metrics["input_tokens"] = input_len

# ---- Generate (block-wise masked diffusion) ----
gen_kwargs = dict(gen_length=args.output_length, steps=args.steps,
                  block_length=args.block_length, temperature=0.0)

torch.npu.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(input_ids, **gen_kwargs)
torch.npu.synchronize()
metrics["generate_s"] = round(time.perf_counter() - t0, 3)

generated_tokens = outputs.shape[1] - input_len
metrics["generated_tokens"] = generated_tokens
metrics["tokens_per_second"] = round(generated_tokens / metrics["generate_s"], 1) \
    if metrics["generate_s"] > 0 else 0
metrics["generate_peak_mem_mb"] = round(torch.npu.max_memory_allocated() / 1024**2, 1)

# ---- Output ----
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
logging.info(response)
logging.info(f"\n--- Performance ---")
logging.info(f"  Mode:           {metrics['mode']}")
logging.info(f"  Model load:     {metrics['model_load_s']}s (peak {metrics['model_load_peak_mem_mb']}MB)")
logging.info(f"  Generation:     {metrics['generate_s']}s")
logging.info(f"  Tokens:         {metrics['generated_tokens']}")
logging.info(f"  Throughput:     {metrics['tokens_per_second']} tokens/s")
logging.info(f"  Peak memory:    {metrics['generate_peak_mem_mb']}MB")

if args.report_file:
    with open(args.report_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"  Report:         {args.report_file}")
