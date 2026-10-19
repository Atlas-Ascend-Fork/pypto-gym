# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------



"""PyPTO scatter_pa_kv_cache kernel implementation.

实现说明：
    - 使用 pypto.scatter_update API 进行 KV cache 更新
    - num_tokens 标为动态轴，使用 pypto.loop 遍历
    - 使用 2D reshape + loop + valid_shape 模式处理动态 shape
    - 使用 .move() 方法写回原始 4D cache
"""

import pypto
import torch

# ─────────────────────────────────────────────
# JIT Kernel
# ─────────────────────────────────────────────


@pypto.frontend.jit(runtime_options={"device_sched_mode": 0}, pass_options={"vec_nbuffer_setting": {-2: 1, -1: 8}})
def scatter_pa_kv_cache_kernel(
    # Tensor 描述符：动态轴标为 pypto.DYNAMIC，静态轴写常量整数
    # 禁止 pypto.Tensor() / pypto.Tensor([], dtype) 空注解
    # num_tokens 和 num_blocks 都是动态轴（SPEC.md: dynamic_axes: ['num_tokens', 'num_blocks'])
    key: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    key_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    slot_mapping: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),                # [num_tokens]
    value: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    value_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
):
    """PyPTO jit kernel for scatter_pa_kv_cache.

    根据 DESIGN.md §4.2 实现。核心逻辑：
    1. reshape cache to 2D [num_blocks * block_size, num_heads * head_size]
    2. reshape src to 2D [num_tokens, num_heads * head_size]
    3. reshape slot_mapping to 2D [num_tokens, 1]
    4. loop over num_tokens (tile_tokens = 8)
    5. scatter_update key_cache and value_cache
    6. write back using .move()

    关键约束：
    - scatter_update dim=-2，不支持 broadcast
    - TileShape 维度数 = src 维度数（2D）
    - 尾轴 kv_dim = 512 > 16（BF16 对齐要求）
    - inplace=True reshape 的输出不能是函数输出参数（已避免）
    - view 的 shape 参数必须全是 Python int（已满足）
    - valid_shape 处理尾块
    """
    # 获取 shape 信息（编译期常量）
    num_blocks = key_cache.shape[0]      # int = 9760
    block_size = key_cache.shape[1]      # int = 128
    num_heads = key.shape[1]             # int = 2
    head_size = key.shape[2]             # int = 256

    # 计算 kv_dim（编译期常量）
    kv_dim = num_heads * head_size       # int = 512

    # 获取动态轴（SymbolicScalar）
    num_tokens = key.shape[0]            # SymbolicScalar

    # Step 1: reshape cache to 2D [num_blocks * block_size, kv_dim]
    # 注意：inplace=True 的输出不能是函数输出参数，这里 key_cache/value_cache 不是输出参数
    cache_2d_shape = [num_blocks * block_size, kv_dim]  # [1249280, 512]
    key_cache_2d = pypto.reshape(key_cache, cache_2d_shape, inplace=True)     # BF16
    value_cache_2d = pypto.reshape(value_cache, cache_2d_shape, inplace=True)  # BF16

    # Step 2: reshape src to 2D [num_tokens, kv_dim]
    key_2d = pypto.reshape(key, [num_tokens, kv_dim], inplace=True)     # BF16, [num_tokens, 512]
    value_2d = pypto.reshape(value, [num_tokens, kv_dim], inplace=True)  # BF16, [num_tokens, 512]

    # Step 3: reshape slot_mapping to 2D [num_tokens, 1]
    slot_mapping_2d = pypto.reshape(slot_mapping, [num_tokens, 1], inplace=True)  # INT32, [num_tokens, 1]

    # Tiling 配置（TileShape 维度数必须与 src 维度数一致）
    # 性能优化：tile_tokens 按 kv_dim 分段调整，减少大 kv_dim 时的 loop 迭代次数
    tile_tokens = 32  # 每次 scatter 更新 32 个 token
    if kv_dim <= 512:
        tile_tokens = 32
    elif kv_dim <= 4096:
        tile_tokens = 16
    else:
        tile_tokens = 4
    pypto.set_vec_tile_shapes(tile_tokens, kv_dim)  # [tile_tokens, kv_dim]

    # Loop 计算（num_tokens_loop 为 SymbolicScalar）
    num_tokens_loop = (num_tokens + tile_tokens - 1) // tile_tokens

    # Step 4-6: scatter update with loop
    # 关键：loop 必须真实遍历动态轴，trip count 为符号表达式
    for loop_idx in pypto.loop(num_tokens_loop, name="scatter_loop", idx_name="loop_idx", unroll_list=[1]):
        # 计算当前 tile 的 offset 和实际 token 数（处理尾块）
        offset = loop_idx * tile_tokens                                  # SymbolicScalar
        actual_tokens = (num_tokens - offset).min(tile_tokens)           # SymbolicScalar

        # view 切出当前 tile（shape 参数必须全是 Python int）
        # valid_shape 处理动态边界（最后一个 tile 可能少于 tile_tokens）
        index_view = pypto.view(
            slot_mapping_2d, 
            [tile_tokens, 1], 
            [offset, 0],
            valid_shape=[actual_tokens, 1]
        )  # INT32, [tile_tokens, 1]，实际有效数据 [actual_tokens, 1]

        key_view = pypto.view(
            key_2d,
            [tile_tokens, kv_dim],
            [offset, 0],
            valid_shape=[actual_tokens, kv_dim]
        )  # BF16, [tile_tokens, 512]，实际有效数据 [actual_tokens, 512]

        value_view = pypto.view(
            value_2d,
            [tile_tokens, kv_dim],
            [offset, 0],
            valid_shape=[actual_tokens, kv_dim]
        )  # BF16, [tile_tokens, 512]，实际有效数据 [actual_tokens, 512]

        # scatter_update（原地更新 cache）
        # 注意：scatter_update 返回更新后的 input tensor
        # dim=-2 是必需的，沿倒数第二维更新
        key_cache_2d_result = pypto.scatter_update(key_cache_2d, -2, index_view, key_view)   # BF16
        value_cache_2d_result = pypto.scatter_update(value_cache_2d, -2, index_view, value_view)  # BF16

        # 写回原始 4D cache（使用 .move() 方法）
        # .move() 会自动处理 shape 转换，将 2D tensor 数据写回原始 4D tensor
        key_cache.move(key_cache_2d_result)
        value_cache.move(value_cache_2d_result)

# ─────────────────────────────────────────────
# Wrapper 函数（导出接口）
# ─────────────────────────────────────────────


def scatter_pa_kv_cache_wrapper(
    key: torch.Tensor,
    key_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    value: torch.Tensor,
    value_cache: torch.Tensor,
) -> tuple:
    """算子 wrapper，供 test_scatter_pa_kv_cache.py 调用。

    负责：
    1. 构造输出 torch.Tensor（key_cache 和 value_cache 原地更新）
    2. 调用 JIT kernel
    3. 返回更新后的 key_cache 和 value_cache

    Args:
        key: 当前 step 多个 token 的 key 值。
            Shape: [num_tokens, num_heads, head_size]
            Dtype: BFLOAT16
        key_cache: 需要更新的 key cache。
            Shape: [num_blocks, block_size, num_heads, head_size]
            Dtype: BFLOAT16
        slot_mapping: 每个 token key/value 在 cache 中的存储偏移。
            Shape: [num_tokens]
            Dtype: INT32
        value: 当前 step 多个 token 的 value 值。
            Shape: [num_tokens, num_heads, head_size]
            Dtype: BFLOAT16
        value_cache: 需要更新的 value cache。
            Shape: [num_blocks, block_size, num_heads, head_size]
            Dtype: BFLOAT16

    Returns:
        (key_cache_out, value_cache_out): 更新后的 key cache 和 value cache。
            两者都是原地更新（返回引用）。
    """
    # key_cache 和 value_cache 原地更新，不需要额外创建输出 tensor
    # 调用 kernel
    scatter_pa_kv_cache_kernel(key, key_cache, slot_mapping, value, value_cache)

    # 返回原地更新后的 cache
    return key_cache, value_cache