# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""
DeepSeek-V2-Lite MLA KV Prolog - PyPTO混合优化版本

ITER_14策略：
- Stage 1 (torch_npu原生): kv_a_proj + split + RMSNorm
- Stage 2 (PyPTO融合): kv_b_proj + reshape + split + RoPE + assemble

目标：减少PyPTO内部的vec操作，利用torch_npu原生算子性能优势
"""

import logging
import os
import sys
from dataclasses import dataclass, field
import torch
import torch_npu
import pypto


@dataclass
class HybridConfigs:
    unroll_list: list = field(default_factory=lambda: [8, 4, 2, 1])


def rotate_half_pto(x):
    """rotate_half PyPTO实现"""
    shape = x.shape
    shape_size = len(shape)

    new_shape = list(shape)
    new_shape[shape_size - 1] //= 2

    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = new_shape[shape_size - 1]

    x1 = pypto.view(x, new_shape, offset1)
    x2 = pypto.view(x, new_shape, offset2)

    neg_x2 = pypto.mul(x2, -1.0)
    return pypto.concat([neg_x2, x1], -1)


def rope_2d_pto(x, cos, sin):
    """2D RoPE PyPTO简化实现"""
    shape = x.shape
    half_dim = shape[-1] // 2

    x1 = pypto.view(x, [shape[0], half_dim], [0, 0])
    x2 = pypto.view(x, [shape[0], half_dim], [0, half_dim])
    neg_x2 = pypto.mul(x2, -1.0)
    x_rot = pypto.concat([neg_x2, x1], -1)

    x_embed = pypto.mul(x, cos) + pypto.mul(x_rot, sin)

    return x_embed


def hybrid_stage2_compute(
    compressed_kv_norm_2d,  # [b*s, kv_lora_rank] 来自torch_npu RMSNorm
    k_pe_2d,  # [b*s, rope_dim] 来自split
    kv_b_weight,  # [num_heads * 256, kv_lora_rank]
    cos,
    sin,
    k_nope_out,
    value_out,
    k_pe_embed_out,
    configs
):
    """
    ITER_14: PyPTO Stage 2融合kernel

    输入：compressed_kv_norm_2d (torch_npu RMSNorm输出)
    流程：kv_b_proj + reshape + split + RoPE + assemble
    """
    pypto.set_pass_options(
        cube_l1_reuse_setting={-1: 4},
        cube_nbuffer_setting={3: 4},
    )

    t = compressed_kv_norm_2d.shape[0]
    kv_lora_rank = compressed_kv_norm_2d.shape[1]
    rope_dim = cos.shape[1]
    num_heads = 16
    qk_nope_head_dim = 128
    v_head_dim = 128

    unroll_list = configs.unroll_list

    for tIdx, t_tile in pypto.loop_unroll(0, t, 1, name="HYBRID_STAGE2_LOOP", idx_name="token_offset",
                                           unroll_list=unroll_list):

        # Step 1: view compressed_kv_norm_tile (输入已由torch_npu RMSNorm处理)
        compressed_kv_norm_tile = pypto.view(compressed_kv_norm_2d, [t_tile, kv_lora_rank], 
                                              [tIdx, 0], valid_shape=[t_tile, kv_lora_rank])

        # Step 2: kv_b_proj (PyPTO matmul - Stage 2唯一matmul)
        pypto.set_cube_tile_shapes([16, 16], [256, 256], [64, 64])
        # kv_b_weight已由调用方转置 [kv_lora_rank, num_heads*256]
        # 直接matmul，不需要b_trans
        kv_total_tile = pypto.matmul(compressed_kv_norm_tile, kv_b_weight, pypto.DT_FP16)

        # Step 3: reshape to 3D
        kv_3d_tile = pypto.reshape(kv_total_tile, [t_tile, num_heads, qk_nope_head_dim + v_head_dim])

        # Step 4: split k_nope + value (view操作)
        pypto.set_vec_tile_shapes(128, num_heads, qk_nope_head_dim + v_head_dim)

        k_nope_tile = pypto.view(kv_3d_tile,
                                [t_tile, num_heads, qk_nope_head_dim],
                                [0, 0, 0],
                                valid_shape=[t_tile, num_heads, qk_nope_head_dim])

        value_tile = pypto.view(kv_3d_tile,
                               [t_tile, num_heads, v_head_dim],
                               [0, 0, qk_nope_head_dim],
                               valid_shape=[t_tile, num_heads, v_head_dim])

        # Step 5: RoPE on k_pe (vec计算)
        k_pe_tile = pypto.view(k_pe_2d, [t_tile, rope_dim], [tIdx, 0],
                               valid_shape=[t_tile, rope_dim])

        pypto.set_vec_tile_shapes(128, 128)
        cos_tile = pypto.view(cos, [t_tile, rope_dim], [tIdx, 0],
                             valid_shape=[t_tile, rope_dim])
        sin_tile = pypto.view(sin, [t_tile, rope_dim], [tIdx, 0],
                             valid_shape=[t_tile, rope_dim])

        k_pe_embed_tile = rope_2d_pto(k_pe_tile, cos_tile, sin_tile)

        # Step 6: assemble outputs
        pypto.assemble(k_nope_tile, [tIdx, 0, 0], k_nope_out)
        pypto.assemble(value_tile, [tIdx, 0, 0], value_out)

        k_pe_2d_tile = pypto.reshape(k_pe_embed_tile, [t_tile, rope_dim])
        pypto.assemble(k_pe_2d_tile, [tIdx, 0], k_pe_embed_out)


class HybridManager:
    def __init__(self):
        self.t_vec = [1, 2, 4, 8, 16, 32, 64, 128]
        self.vec_all_shape = {}
        for t in self.t_vec:
            self.vec_all_shape[t] = [t, 512]

    def infer_controlflow_shape(self, *args):
        if not args:
            return [v for v in self.vec_all_shape.values()]
        compressed_kv_shape = args[0]
        for t in self.t_vec:
            if compressed_kv_shape[0] >= t:
                return self.vec_all_shape[t]
        return None

manager = HybridManager()


@pypto.frontend.jit()
def hybrid_stage2_kernel(
    compressed_kv_norm_2d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
    k_pe_2d: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
    kv_b_weight: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    cos: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
    sin: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16),
    k_nope_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    value_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    k_pe_embed_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP16)
):
    configs = HybridConfigs()
    hybrid_stage2_compute(
        compressed_kv_norm_2d, k_pe_2d, kv_b_weight, cos, sin,
        k_nope_out, value_out, k_pe_embed_out, configs
    )


def mla_prolog_hybrid_optimized(hidden_states, kv_a_weight, kv_b_weight, ln_weight, eps, cos, sin, pos_ids):
    """
    ITER_14: 混合优化版本

    Stage 1 (torch_npu原生): kv_a_proj + split + RMSNorm
    Stage 2 (PyPTO融合): kv_b_proj + reshape + split + RoPE + assemble
    """
    torch.npu.set_device(int(os.environ.get('TILE_FWK_DEVICE_ID', 0)))
    bsz, seq_len, hidden_size = hidden_states.shape
    kv_lora_rank = ln_weight.shape[0]
    rope_dim = cos.shape[1]
    num_heads = 16

    # DEBUG: Check device consistency
    devices = {
        'hidden_states': hidden_states.device,
        'kv_a_weight': kv_a_weight.device,
        'kv_b_weight': kv_b_weight.device,
        'ln_weight': ln_weight.device,
        'cos': cos.device,
        'sin': sin.device,
    }
    if len(set(devices.values())) > 1:
        logging.error(f"[ERROR] Device mismatch: {devices}")
        raise RuntimeError(f"Device mismatch: {devices}")

    # Stage 1: kv_a_proj + split + RMSNorm (torch_npu原生算子)
    hidden_2d = hidden_states.reshape(bsz * seq_len, hidden_size).contiguous()

    # 调用方传入已转置的kv_a_weight [hidden_size, kv_dim]
    # 直接matmul: hidden_2d [b*s, hidden_size] @ kv_a_weight [hidden_size, kv_dim]
    compressed_kv_total = torch.matmul(hidden_2d, kv_a_weight)

    # split (torch切片)
    compressed_kv = compressed_kv_total[:, :kv_lora_rank].contiguous()
    k_pe = compressed_kv_total[:, kv_lora_rank:].contiguous()

    # RMSNorm (torch_npu.npu_rms_norm - 原生优化算子)
    compressed_kv_norm, _ = torch_npu.npu_rms_norm(compressed_kv, ln_weight, epsilon=eps)

    # 扩展cos/sin
    if bsz > 1:
        cos_expanded = cos.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seq_len, rope_dim).contiguous()
        sin_expanded = sin.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seq_len, rope_dim).contiguous()
    else:
        cos_expanded = cos.reshape(bsz * seq_len, rope_dim).contiguous()
        sin_expanded = sin.reshape(bsz * seq_len, rope_dim).contiguous()

    # Stage 2: PyPTO融合kernel (kv_b_proj + reshape + split + RoPE + assemble)
    # 确保所有输入tensor在正确设备上
    target_device = hidden_states.device
    k_nope_out = torch.empty(bsz * seq_len, num_heads, 128, dtype=torch.float16, device=target_device)
    value_out = torch.empty(bsz * seq_len, num_heads, 128, dtype=torch.float16, device=target_device)
    k_pe_embed_out = torch.empty(bsz * seq_len, rope_dim, dtype=torch.float16, device=target_device)

    hybrid_stage2_kernel(
        compressed_kv_norm, k_pe, kv_b_weight, cos_expanded, sin_expanded,
        k_nope_out, value_out, k_pe_embed_out
    )

    # reshape到最终输出shape
    k_nope = k_nope_out.reshape(bsz, num_heads, seq_len, 128)
    value = value_out.reshape(bsz, num_heads, seq_len, 128)
    k_pe_final = k_pe_embed_out.reshape(bsz, 1, seq_len, rope_dim)

    return k_nope, value, k_pe_final
