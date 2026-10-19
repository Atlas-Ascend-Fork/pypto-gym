#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared MiniMax (M2.7 / M3) per-variant model load.

``ask_minimax.py`` and ``bench_minimax.py`` both need to build + load the requested variant on one
die, dispatching to the per-variant model package. That load block is identical between them, so it
lives here as a single entry: ``load_variant_model(args, dev, streaming)`` returns
``(model, cfg, n_patched)``.
"""


def load_variant_model(args, dev, streaming):
    """Build + load the requested MiniMax variant on one die. Returns (model, cfg, n_patched).

    ``streaming`` keeps the routed experts FP8 on host and dequantizes per layer. M2.7 always
    streams the fused grouped-GEMM kernel; M3 dispatches via its ``load_model`` entry point
    (``use_pypto`` selects the fused kernel vs the eager per-expert FFN).
    """
    if args.variant == "m27":
        import ops.pypto_tensor.minimax as _pto
        _pto.USE_PTO_GROUPED_GEMM = True        # enable the fused grouped-GEMM kernel for this run
        from pypto_gym.transformers.minimax_m27.modeling_minimax_m27 import (
            attach_expert_fp8, build_model, load_streaming_state_dict,
            materialize_meta_buffers, patch_moe,
        )
        model, cfg = build_model(args.model_path, max_layers=args.max_layers)
        sd, expert_fp8 = load_streaming_state_dict(args.model_path, max_layers=args.max_layers)
        model.load_state_dict(sd, strict=False, assign=True)
        attach_expert_fp8(model, expert_fp8)
        materialize_meta_buffers(model, "cpu")
        model.to(dev).eval()
        n_patched = patch_moe(model, streaming=streaming)
        return model, cfg, n_patched
    # m3
    from pypto_gym.transformers.minimax_m3.modeling_minimax_m3 import load_model
    return load_model(args.model_path, dev, use_pypto=args.use_pypto,
                      streaming=streaming, max_layers=args.max_layers)
