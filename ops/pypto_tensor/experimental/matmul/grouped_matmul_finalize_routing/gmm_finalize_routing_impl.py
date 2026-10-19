#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""grouped_matmul_finalize_routing PyPTO 算子实现（MXFP8）。

Golden 参考实现与单测入口见：
  tests/ops/experimental/matmul/grouped_matmul_finalize_routing/

本模块提供 JIT kernel、配置数据结构与 host 侧 `gen_pypto`。
"""

from dataclasses import dataclass, field

import pypto
import torch
import torch_npu  # type: ignore[reportMissingImports]


@dataclass
class FinalizeRoutingConfig:
    """grouped_matmul_finalize_routing 配置。

    Attributes:
        batch: 输出 batch 维大小（也是 shared_input 的行数基准）。
        topk: 每个 batch 的 token 数。
        m: token 总数，由 batch * topk 自动计算（不可手动指定）。
        k: matmul 的 K 维。
        n: matmul 的 N 维（输出列数）。
        num_experts: expert 数量。
        m_tile_shape: cube M 维 tile 配置。
        k_tile_shape: cube K 维 tile 配置。
        n_tile_shape: cube N 维 tile 配置。
        vector_tile_shape: 向量算子 tile 配置。
        in_dtype: 输入 FP8 数据类型。
        transpose_x1: 是否转置 x1（当前算子仅支持 False）。
        transpose_x2: 是否转置 x2。
        group_list_type: 分组描述类型，0=前缀和，1=每组计数。
        shared_input_weight: shared_input 叠加权重。
        shared_input_offset: shared_input 在 out 上的起始行偏移。
        has_logit: 是否启用 logit 加权。
        has_shared_input: 是否启用 shared_input 叠加。
        description: 用于测试打印的用例描述。
    """

    batch: int
    topk: int
    k: int
    n: int
    num_experts: int
    vector_tile_shape: list
    m_tile_shape: list = field(default_factory=list)
    k_tile_shape: list = field(default_factory=list)
    n_tile_shape: list = field(default_factory=list)
    m: int = field(init=False)
    in_dtype: pypto.DataType = pypto.DT_FP8E4M3
    transpose_x1: bool = False
    transpose_x2: bool = False
    group_list_type: int = 1
    shared_input_weight: float = 1.0
    shared_input_offset: int = 0
    has_logit: bool = True
    has_shared_input: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "m", self.batch * self.topk)
        per_expert_m = self.m // self.num_experts
        if per_expert_m <= 64:
            m_tile_shape = [per_expert_m, per_expert_m]
            k_tile_shape = [256, 512]
            n_tile_shape = [256, 512]
        elif per_expert_m <= 1024:
            m_tile_shape = [128, 128]
            k_tile_shape = [512, 512]
            n_tile_shape = [128, 256]
        else:
            m_tile_shape = [128, 128]
            k_tile_shape = [256, 256]
            n_tile_shape = [256, 256]
        object.__setattr__(self, "m_tile_shape", m_tile_shape)
        object.__setattr__(self, "k_tile_shape", k_tile_shape)
        object.__setattr__(self, "n_tile_shape", n_tile_shape)


@dataclass
class FinalizeRoutingGoldenInputs:
    x1: torch.Tensor
    x2: torch.Tensor
    scale: torch.Tensor
    pertoken_scale: torch.Tensor
    group_list: torch.Tensor
    shared_input: torch.Tensor
    logit: torch.Tensor
    row_index: torch.Tensor
    out: torch.Tensor
    config: FinalizeRoutingConfig


@dataclass
class FinalizeRoutingInputs:
    x1: torch.Tensor
    x2: torch.Tensor
    scale: torch.Tensor
    pertoken_scale: torch.Tensor
    group_list: torch.Tensor
    shared_input: torch.Tensor
    logit: torch.Tensor
    row_index: torch.Tensor
    out: torch.Tensor
    config: FinalizeRoutingConfig


@pypto.frontend.jit(
    pass_options={
        "cube_nbuffer_setting": {-1: 1},
        "vec_nbuffer_setting": {-2: 1, -1: 1},
    },
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1},
)
def gmm_finalize_routing_kernel(
    x1: pypto.Tensor(),
    x2: pypto.Tensor(),
    scale: pypto.Tensor(),
    pertoken_scale: pypto.Tensor(),
    shared_input: pypto.Tensor(),
    shared_row_index: pypto.Tensor(),
    logit: pypto.Tensor(),
    row_index: pypto.Tensor(),
    gmm_out: pypto.Tensor(),
    out: pypto.Tensor(),
    group_list,
    config: FinalizeRoutingConfig,
):
    """Fused grouped matmul finalize routing kernel.

    MXFP8 layout follows the aclnn sample:
    - x1: [m, k]
    - x2: [e, k, n] when transpose_x2=False, otherwise [e, n, k]
    - scale: [ceil(k / 64), n, 2] when transpose_x2=False,
      otherwise [n, ceil(k / 64), 2]
    - pertoken_scale: [m, ceil(k / 64), 2]
    """

    pypto.experimental.auto_mix_partition(1)

    pypto.set_cube_tile_shapes(config.m_tile_shape, config.k_tile_shape, config.n_tile_shape)
    pypto.set_vec_tile_shapes(
        config.vector_tile_shape[0],
        config.vector_tile_shape[1],
        config.vector_tile_shape[2],
        config.vector_tile_shape[3],
    )

    token_num = config.m // config.num_experts

    for expert_idx in pypto.loop(config.num_experts, parallel=True):
        start = expert_idx * token_num
        end = (expert_idx + 1) * token_num

        pypto.experimental.set_operation_options(combine_axis=True)

        x_tile = x1[start:end, :]
        pertoken_scale_tile = pertoken_scale[start:end, :, :]
        weight_tile = x2[expert_idx, :, :]
        weight_tile.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)

        mm_result = pypto.scaled_mm(
            x_tile,
            weight_tile,
            pypto.DT_FP32,
            pertoken_scale_tile,
            scale[:, :, :],
            a_trans=False,
            scale_a_trans=False,
            b_trans=config.transpose_x2,
            scale_b_trans=config.transpose_x2,
        )

        gmm_out[start:end, :] = mm_result

    pypto.set_vec_tile_shapes(config.m_tile_shape[-1], config.n_tile_shape[-1])
    route_tile = 512
    route_tile_num = config.m // route_tile
    route_tail = config.m - route_tile_num * route_tile

    if config.has_logit:
        for tile_idx in pypto.loop(route_tile_num, parallel=False):
            start = tile_idx * route_tile
            end = start + route_tile
            result_tile = gmm_out[start:end, :]
            logit_2d = pypto.unsqueeze(logit[start:end], -1)
            result_tile = pypto.mul(result_tile, logit_2d)
            pypto.index_add_(out, 0, row_index[start:end], result_tile)

        if route_tail > 0:
            start = route_tile_num * route_tile
            end = start + route_tail
            result_tile = gmm_out[start:end, :]
            logit_2d = pypto.unsqueeze(logit[start:end], -1)
            result_tile = pypto.mul(result_tile, logit_2d)
            pypto.index_add_(out, 0, row_index[start:end], result_tile)
    else:
        for tile_idx in pypto.loop(route_tile_num, parallel=False):
            start = tile_idx * route_tile
            end = start + route_tile
            pypto.index_add_(out, 0, row_index[start:end], gmm_out[start:end, :])

        if route_tail > 0:
            start = route_tile_num * route_tile
            end = start + route_tail
            pypto.index_add_(out, 0, row_index[start:end], gmm_out[start:end, :])

    if config.has_shared_input:
        shared_fp32 = pypto.cast(shared_input[:, :], pypto.DT_FP32)
        shared_scaled = pypto.mul(shared_fp32, config.shared_input_weight)
        pypto.index_add_(out, 0, shared_row_index, shared_scaled)

    pypto.set_vec_tile_shapes(config.m_tile_shape[-1], config.n_tile_shape[-1])


def gen_pypto(inputs: FinalizeRoutingInputs) -> torch.Tensor:
    """执行 PyPTO kernel 并返回 FP32 输出。

    注意：
    - kernel 内处理 grouped matmul + logit + row_index accumulate；
    - shared_input 的 cast、缩放与叠加也在 kernel 内完成。
    """
    x1 = inputs.x1.npu()
    x2 = inputs.x2.npu()
    scale = inputs.scale.npu()
    pertoken_scale = inputs.pertoken_scale.npu()
    group_list = inputs.group_list.cpu().tolist()
    shared_input = inputs.shared_input.npu()
    shared_row_index = (
        torch.arange(inputs.shared_input.shape[0], dtype=torch.int32) + inputs.config.shared_input_offset
    ).npu()
    logit = inputs.logit.npu()
    row_index = inputs.row_index.to(torch.int32).npu()
    gmm_out = torch.zeros((inputs.config.m, inputs.config.n), dtype=torch.float32).npu()
    out = inputs.out.clone().npu()

    gmm_finalize_routing_kernel(
        x1,
        x2,
        scale,
        pertoken_scale,
        shared_input,
        shared_row_index,
        logit,
        row_index,
        gmm_out,
        out,
        group_list,
        inputs.config,
    )
    return out.to(torch.float32)
