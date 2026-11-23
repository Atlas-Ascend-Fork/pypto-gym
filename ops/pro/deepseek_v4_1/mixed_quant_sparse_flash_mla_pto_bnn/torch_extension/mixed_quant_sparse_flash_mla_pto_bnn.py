# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Public Torch frontend for the precompiled MixedQuantSparseFlashMlaPtoBnn operator."""
import math

import torch


def mixed_quant_sparse_flash_mla_pto_bnn(
    q,
    *,
    ori_kv,
    ori_sparse_indices,
    ori_block_table,
    cu_seqlens_q,
    sinks,
    metadata,
    tiling_device,
    quant_mode=1,
    softmax_scale=None,
    cmp_kv=None,
    cmp_sparse_indices=None,
    cmp_block_table=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
    out=None,
    lse=None,
    block_dim=None,
    fd_enable=None,
):
    """Sparse mixed-quant Flash MLA (single precompiled ACLNN kernel).

    Runs the installed ACLNN binary on the current NPU stream. ``out``/``lse``
    are allocated internally when omitted; ``lse`` is always written.
    ``metadata``/``tiling_device`` are the AICPU planner's device outputs
    (``int32[1024]`` schedule + ``int32[16]`` tiling params): the kernel reads
    both on device, so this call performs no D2H anywhere. Optional inputs
    pass as ``None``: absent ``*_topk_length`` means the whole K range is valid,
    and an absent cmp group (``cmp_kv`` / ``cmp_sparse_indices`` /
    ``cmp_block_table``) turns the compressed arm off. ``block_dim`` /
    ``fd_enable`` are accepted for source-wrapper compatibility and ignored
    (the launch uses the all-cores superset grid and the FD-capable variant).
    The operator provides forward computation only.

    Returns ``{"out": out, "lse": lse}``.
    """
    if type(quant_mode) is not int or quant_mode != 1:
        raise ValueError("quant_mode must be 1")
    if layout_q != "TND" or layout_kv != "PA_BBND":
        raise ValueError("only layout_q='TND', layout_kv='PA_BBND' are supported")
    for name, tensor in (
        ("seqused_q", seqused_q),
        ("seqused_ori_kv", seqused_ori_kv),
        ("seqused_cmp_kv", seqused_cmp_kv),
        ("cmp_kv", cmp_kv),
        ("cmp_sparse_indices", cmp_sparse_indices),
        ("cmp_block_table", cmp_block_table),
        ("ori_topk_length", ori_topk_length),
        ("cmp_topk_length", cmp_topk_length),
        ("out", out),
        ("lse", lse),
    ):
        if tensor is not None and not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor or None")
    if not isinstance(q, torch.Tensor):
        raise ValueError("q must be a torch.Tensor")
    if not isinstance(tiling_device, torch.Tensor):
        raise ValueError(
            "tiling_device is required: run the AICPU planner "
            "(mixed_quant_sparse_flash_mla_metadata_pto_bnn) on the same device "
            "and pass its tiling_device output here")
    if cmp_kv is not None and (cmp_sparse_indices is None or cmp_block_table is None):
        raise ValueError("cmp_kv is given but cmp_sparse_indices/cmp_block_table is missing")
    scale = 0.0
    if softmax_scale is not None:
        scale = float(softmax_scale)
        if not math.isfinite(scale):
            raise ValueError("softmax_scale must be finite")
    out_t, lse_t = torch.ops.pypto_gym.mixed_quant_sparse_flash_mla_pto_bnn(
        q, ori_kv, ori_sparse_indices, ori_block_table, cu_seqlens_q, sinks,
        metadata, tiling_device, ori_topk_length=ori_topk_length, cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices, cmp_block_table=cmp_block_table,
        cmp_topk_length=cmp_topk_length, seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv, seqused_cmp_kv=seqused_cmp_kv,
        quant_mode=quant_mode, softmax_scale=scale, layout_q=layout_q,
        layout_kv=layout_kv, return_softmax_lse=bool(return_softmax_lse),
        out=out, lse=lse, block_dim=block_dim, fd_enable=fd_enable)
    return {"out": out_t, "lse": lse_t}


# torchair GE/aclgraph 转换器注册(参考 ops-transformer graph_convert_* 模式);
# 需在 torch.ops.pypto_gym 命名空间由 loader 加载原生库之后导入, 故置于模块尾部。
# torchair 缺失或命名空间未就绪时为纯 no-op, 不影响 eager/NPUGraph 路径。
try:
    from . import graph_convert_mixed_quant_sparse_flash_mla_pto_bnn  # noqa: F401,E402
except Exception:  # pragma: no cover - 环境相关(torchair 缺失/无包上下文)的静默降级
    pass
