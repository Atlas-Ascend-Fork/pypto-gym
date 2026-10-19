#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MiniMax (M2.7 / M3) single-prompt (text) inference on Ascend 910B.

One script, two variants selected by ``--variant {m27,m3}``. Both run their text backbone **in-repo**
(no ``trust_remote_code``) and route the **MoE expert FFN** through the PyPTO fused grouped-GEMM
kernel, with the routed experts kept FP8 on the host and dequantized per layer (streaming). Routing /
shared expert / attention stay on the host.

  * **m27** -- MiniMax-M2.7 (228.7B), SiLU-SwiGLU experts (H=3072, I=1536, E=256, top-8).
  * **m3**  -- MiniMax-M3 text backbone, Mixtral-style sparse MoE, swigluoai experts
               (H=6144, I=3072, E=128, top-4). Needs an FP8 checkpoint (e.g. MiniMax-M3-MXFP8) to
               fit one die; ``--max-layers`` gives a quick BF16 smoke test.

    MODEL_PATH=/path/to/MiniMax-M2.7     TILE_FWK_DEVICE_ID=0 python ask_minimax.py --variant m27
    MODEL_PATH=/path/to/MiniMax-M3-MXFP8 TILE_FWK_DEVICE_ID=0 python ask_minimax.py --variant m3 --use_pypto
"""

import argparse
import logging
import os
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("minimax.ask")

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, "src")):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise ImportError("torch_npu required (Ascend NPU).") from exc


# Per-variant constants -- the only places the two models genuinely differ.
_VARIANTS = {
    "m27": {"display": "MiniMax-M2.7", "vec_tile": "128"},   # UB-fitting vector tile (910B)
    "m3": {"display": "MiniMax-M3", "vec_tile": "256"},      # M3-tuned vector tile (UB-fitting; 910B)
}


def parse_args():
    ap = argparse.ArgumentParser(description="MiniMax (M2.7 / M3) single-prompt inference")
    ap.add_argument("--variant", choices=("m27", "m3"), required=True,
                    help="which MiniMax text backbone to run")
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--device", type=int, default=int(os.environ.get("TILE_FWK_DEVICE_ID", 0)))
    ap.add_argument("--prompt", default="Explain mixture-of-experts routing in one paragraph.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-layers", type=int, default=None, help="load only the first N layers (smoke test)")
    ap.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True,
                    help="keep routed experts FP8 on host, dequant per layer (default on)")
    ap.add_argument("--use_pypto", action="store_true", help="route MoE FFN through the PyPTO kernel")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.model_path:
        raise SystemExit("set MODEL_PATH or pass --model-path")
    spec = _VARIANTS[args.variant]
    os.environ.setdefault("PYPTO_VEC_TILE", spec["vec_tile"])
    torch_npu.npu.config.allow_internal_format = True
    torch.npu.set_device(args.device)
    dev = f"npu:{args.device}"

    logger.info("Loading %s (%s, use_pypto=%s, streaming=%s) ...",
                spec["display"], args.model_path, args.use_pypto, args.streaming)
    from _variant import load_variant_model
    model, cfg, n_patched = load_variant_model(args, dev, streaming=args.streaming)
    logger.info("  %d-layer text backbone loaded; patched %d MoE block(s) with the PyPTO kernel",
                cfg.num_hidden_layers, n_patched)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    msgs = [{"role": "user", "content": args.prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    logger.info("=" * 60)
    logger.info(tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
