#!/usr/bin/env python3
# coding: utf-8
# ruff: noqa: F821
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""950 算子 IndexerPrologueK（PyPTO-Pro，arch 3510）。

对 ``latent[T, H]`` 做投影，再依次做 RMSNorm、尾部 RoPE、MXFP4 打包，并把打包结果
原地散写进分页 ``k_cache``。中间结果驻留 device：::

    latent -- BF16 GEMM --> projected -- RMSNorm --> 尾部 RoPE
           --> block-32 MXFP4 打包 --> 按 cache_index 原地写 uint8 k_cache

**两种下发形态，按 token 数 ``T`` 派发**（``FUSED_T_MAX``，默认 2048）：

- ``T <= FUSED_T_MAX``：**单 kernel 融合**（``make_fused_kernel``）——同一 launch 内先 ``section_cube()``
  做投影、经 ``sync_all(MIX)`` 全局屏障后 ``section_vector()`` 做后处理。省一次 kernel 下发的固定
  开销，小/中 T 约 1.3–2.1×。（cube+vector 混合核，每 block = 1 AIC + 2 AIV。）
- ``T > FUSED_T_MAX``：**两 kernel**——模块级 ``indexer_projection_kernel``（Cube）+ 工厂化的
  ``indexer_postprocess_kernel``（Vector）。大 T 时融合的 MIX 屏障无收益甚至略慢，故保留两 kernel。

投影是单个模块级 ``@pl.jit`` cube kernel，用 ``tiling_key``（``IndexerProjectionKey``）按 L1 tile
形状类别做编译期特化：``NMode``/``KMode`` 只是解析期常量，模型形状 ``(H, D)`` 与 token 数 ``T`` 都在
运行时从张量 shape 读取，故同一 key 的一份二进制服务该类别下的任意形状与任意 ``T``。

后处理采用**块级 owner**（一个 vector 核负责若干目标行）：每个核按输入顺序扫描 token，
命中自己的目标行时把整行搬入 UB，块内协作完成 RMSNorm（`warp_reduce_add` + `atomic_add`
归约）、尾部 RoPE，再打包 MXFP4 并写回 cache。SIMT 入口要求静态 shape，故后处理/融合变体按
``(T, D, Dr, storage_mode, G, write_scale, cache 形状)`` 固化（投影侧的 ``T`` 仍动态）。

三处 BF16 发布边界与参考实现一致：GEMM 输出、RMSNorm 输出、尾部 RoPE 输出。
MXFP4 使用参考实现的 exponent 位操作（倒数乘 ``safe_amax * (1/6)``，非除法）、E2M1 阈值链、
符号位与 nibble 顺序。``cache_index[t] == -1`` 跳过该 token；重复目标位置以最后一个
token 的结果为准。

接口说明：``wk`` 以 ND ``[D, H]`` 传入（转置在 GM->L1 搬入时完成），
``k_cache`` / ``k_scale_cache`` 必须连续。

用法::

    from indexer_prologue_k import indexer_prologue_k_wrapper
    indexer_prologue_k_wrapper(latent, wk, norm_weight, rope_sin, rope_cos, k_cache,
                               cache_index=..., storage_mode=..., norm_eps=...)
"""

import torch

import pypto_pro.language as pl
from pypto_pro.runtime import platform as pro_platform
from pypto_pro.runtime.tilingkey import TilingKeyField

__all__ = ["indexer_prologue_k_wrapper"]

# ================================================================
#  Tile 尺寸与常量
# ================================================================
BASE_M = 16
BASE_N = 64
BASE_N_WIDE = 128
BASE_K = 128
NZ_ALIGN = 16

FP4_GROUP_SIZE = 32
E2M1_VALUES_PER_BYTE = 2
ROPE_PAIR_SIZE = 2
CACHE_ABI_DIM2 = 1
DEFAULT_CUBE_CORES = 32
DEFAULT_VECTOR_CORES = 64
# 单 kernel 融合的 T 阈值：<= 走融合（省一次 kernel 下发的固定开销，小/中 T 收益明显），
# > 走两 kernel（大 T 时融合的 MIX 屏障无收益甚至略慢）。
FUSED_T_MAX = 2048


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def value_and_scale_bytes(index_head_dim):
    """返回一行的 ``(value_bytes, scale_bytes)``。"""
    return index_head_dim // E2M1_VALUES_PER_BYTE, index_head_dim // FP4_GROUP_SIZE


def projection_axes(index_head_dim, head_dim):
    """返回模型形状对应的 ``(base_n, base_k, n_tiles, k_tiles)``。

    原始 FRACTAL_NZ 的 L1 tile 必须落在物理 16x16 分形边界上，因此对齐的模型形状走
    宽 fast path，未对齐轴退化为一个分形。
    """
    base_n = BASE_N_WIDE if index_head_dim % BASE_N_WIDE == 0 else (
        BASE_N if index_head_dim % BASE_N == 0 else NZ_ALIGN)
    base_k = BASE_K if head_dim % BASE_K == 0 else NZ_ALIGN
    return (
        base_n,
        base_k,
        _ceil_div(index_head_dim, base_n),
        _ceil_div(head_dim, base_k),
    )


# base_n / base_k 类别 -> TilingKey 取值（投影 L1 tile 的编译期形状）。
_N_MODE = {BASE_N_WIDE: 0, BASE_N: 1, NZ_ALIGN: 2}
_K_MODE = {BASE_K: 0, NZ_ALIGN: 1}


class IndexerProjectionKey:
    """投影 kernel 的编译期特化键：L1 tile 形状类别（见 ``projection_axes``）。

    ``NMode`` / ``KMode`` 只编码 tile 形状的类别；模型形状 ``(H, D)`` 与 token 数 ``T``
    都在运行时从张量 shape 读取，因此同一类别组合的一份二进制服务该类别下的任意形状。
    """

    NMode = TilingKeyField(bits=2, values=[0, 1, 2])
    KMode = TilingKeyField(bits=1, values=[0, 1])


def projection_tiling_key(index_head_dim, head_dim):
    """由模型形状 ``(D, H)`` 得到投影 kernel 的 tiling key。"""
    base_n, base_k, _, _ = projection_axes(index_head_dim, head_dim)
    return {"NMode": _N_MODE[base_n], "KMode": _K_MODE[base_k]}


def projection_grid_blocks(total_rows, head_dim, index_head_dim, cube_cores):
    """投影阶段请求的 Cube 核数。

    投影把输出 ``projected[T, D]`` 切成 ``16 × base_n`` 的小块：M 方向
    ``ceil(T/16)`` 个 tile，N 方向 ``ceil(D/base_n)`` 个 tile，工作项数 = 两者相乘。
    每个核以 grid 步长分工处理若干 tile（见 ``indexer_projection_kernel`` 内的
    ``for tile_idx in range(core_id, total_tiles, num_cores)``）。
    起核数取 ``min(设备 AIC 数, 工作项数)``，至少 1，避免起多于工作项的核空转。
    """
    base_n = projection_axes(index_head_dim, head_dim)[0]
    m_tiles = _ceil_div(total_rows, BASE_M)
    n_tiles = _ceil_div(index_head_dim, base_n)
    return max(1, min(cube_cores, m_tiles * n_tiles))


# ================================================================
#  投影 kernel（Cube）——模块级，按 TilingKey 特化 L1 tile 形状类别
# ================================================================
@pl.jit(arch="3510", auto_mutex=True, compile_timeout=200, tiling_key=IndexerProjectionKey)
def indexer_projection_kernel(
    latent: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    wk: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    projected: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    """``projected[T, D] = BF16(GEMM_FP32_ACC(latent, wkᵀ))``。

    ``NMode``/``KMode`` 是解析期常量，决定 L1 tile 形状；模型形状 ``(H, D)`` 与行数 ``T``
    都在运行时从张量 shape 读取，因此同一 key 的一份二进制服务该类别下的任意形状。
    """
    # tiling-key 字段以裸标识符注入（# noqa: F821）。
    if NMode == 0:
        base_n = BASE_N_WIDE
    elif NMode == 1:
        base_n = BASE_N
    else:
        base_n = NZ_ALIGN
    if KMode == 0:
        base_k = BASE_K
    else:
        base_k = NZ_ALIGN

    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    # M 轴是动态的：tile 数由运行时行数推出。
    total_rows = latent.shape[0]
    head_dim = latent.shape[1]
    index_head_dim = wk.shape[0]
    m_tiles = (total_rows + BASE_M - 1) // BASE_M
    n_tiles = (index_head_dim + base_n - 1) // base_n
    k_tiles = (head_dim + base_k - 1) // base_k
    total_tiles = m_tiles * n_tiles

    # L1 buffer 双缓冲；L0A/L0B/L0C 单缓冲。
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[BASE_M, base_k], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
        addrs=0, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[base_k, base_n], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x10000, mutex_ids=[2, 3])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[BASE_M, base_k], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, compact=1),
        addrs=0, mutex_ids=[4, 5])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[base_k, base_n], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, compact=1),
        addrs=0, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[BASE_M, base_n], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, compact=1),
        addrs=0, mutex_ids=[8])

    # =================== CUBE SECTION ===================
    with pl.section_cube():
        for tile_idx in pl.range(core_id, total_tiles, num_cores):
            m_idx = tile_idx // n_tiles
            n_idx = tile_idx % n_tiles
            # M 尾块：最后一个行 tile 可能不足 BASE_M 行。
            valid_m = pl.min(BASE_M, total_rows - m_idx * BASE_M)
            acc_tile = acc.next()
            pl.set_validshape(acc_tile, [valid_m, base_n])
            for k_idx in pl.range(0, k_tiles):
                valid_k = pl.min(base_k, head_dim - k_idx * base_k)
                a_tile = a_l1.next()
                b_tile = b_l1.next()
                pl.set_validshape(a_tile, [valid_m, valid_k])
                # latent 是 ND [T, H] -> L1 NZ，不转置。
                pl.load(a_tile, latent, [m_idx * BASE_M, k_idx * base_k])
                pl.set_validshape(b_tile, [valid_k, base_n])
                # wk 是 ND [D, H]；转置搬入 order=[1, 0] 到 ZN tile 表达 B^T
                # （C = A @ B^T）。
                pl.load(b_tile, wk, [n_idx * base_n, k_idx * base_k], order=[1, 0])
                left_tile = a_left.next()
                right_tile = b_right.next()
                pl.set_validshape(left_tile, [valid_m, valid_k])
                pl.move(left_tile, a_tile)
                pl.set_validshape(right_tile, [valid_k, base_n])
                pl.move(right_tile, b_tile)
                # K 分块：首块初始化 L0C，其余块累加。
                if k_idx == 0:
                    pl.matmul(acc_tile, left_tile, right_tile)
                else:
                    pl.matmul_acc(acc_tile, acc_tile, left_tile, right_tile)
            # FP32 累加器 -> BF16 发布的整块行。
            pl.store(projected, acc_tile, [m_idx * BASE_M, n_idx * base_n])


def _make_postprocess_row(*, dim, rd, storage_mode, write_actual_scale, threads, tail_start):
    """构造后处理的 SIMT 行函数（块内协作处理一行）。

    ``dim`` / ``rd`` / ``storage_mode`` / ``write_actual_scale`` / ``tail_start`` 作为闭包按解析期
    常量折叠，故 mode 分支与列宽均在编译期确定。供后处理 kernel 与融合 kernel 共用，避免重复。
    """
    @pl.vector_function(mode="simt", max_threads=threads)
    def postprocess_row(proj, gamma, norm, pub, red, rope_sin, rope_cos,
                        cache, scale_cache, row, dest0, dest1, data_col, scale_col, epsilon):
        """块内协作处理一行：每线程一个元素，一个 warp 覆盖一个 MXFP4 group。"""
        t = pl.simt.linear_thread_idx()
        lane = pl.simt.lane_id()
        # ---- RMSNorm：warp 内归约 + 每 warp 一次 atomic ----
        if t == 0:
            red[0, 0] = 0.0
        pl.simt.syncthreads()
        x = pl.simt.cast(proj[0, t], pl.DT_FP32)
        partial = pl.simt.warp_reduce_add(x * x)
        if lane == 0:
            pl.simt.atomic_add(red[0, 0], partial)
        pl.simt.syncthreads()
        if t == 0:
            red[1, 0] = pl.simt.rsqrt(red[0, 0] * (1.0 / dim) + epsilon)
        pl.simt.syncthreads()
        inverse = red[1, 0]
        # ---- 发布 RMSNorm（BF16 边界②，每线程一个元素）----
        v = x * inverse * gamma[0, t]
        norm[0, t] = pl.simt.cast(v, pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)
        pl.simt.syncthreads()
        # ---- 尾部相邻偶奇 RoPE（BF16 边界③，每线程一个元素）----
        if t < tail_start:
            pub[0, t] = norm[0, t]
        else:
            local = t - tail_start
            ce = rope_cos[row, local]
            se = rope_sin[row, local]
            if local % 2 == 0:
                a = pl.simt.cast(norm[0, t], pl.DT_FP32)
                b = pl.simt.cast(norm[0, t + 1], pl.DT_FP32)
                r = a * ce - b * se
            else:
                b = pl.simt.cast(norm[0, t], pl.DT_FP32)
                a = pl.simt.cast(norm[0, t - 1], pl.DT_FP32)
                r = b * ce + a * se
            pub[0, t] = pl.simt.cast(r, pl.DT_BF16, mode=pl.RoundMode.CAST_RINT)
        # 每个线程只读回自己写入的 pub[0, t]，此处无需块同步。
        # ---- MXFP4：warp 覆盖一个 group（32 lane = 32 元素），满线程并行 ----
        val = pl.simt.cast(pub[0, t], pl.DT_FP32)
        group_amax = pl.simt.warp_reduce_max(pl.simt.abs(val))
        # safe_amax -> raw_scale = safe_amax * (1/6)：用倒数乘而非除法。
        safe_amax = pl.simt.max(group_amax, 6.0 * (2.0 ** -126))
        raw_scale = safe_amax * (1.0 / 6.0)
        bits = pl.simt.bitcast(raw_scale, pl.DT_UINT32)
        exponent = pl.simt.cast((bits >> 23) & 0xFF, pl.DT_UINT32)
        if (bits & 0x7FFFFF) != 0:
            exponent = pl.simt.cast(exponent + 1, pl.DT_UINT32)
        # scale 恒为 2 的幂，1/scale 也是 2 的幂且可精确表示：用倒数乘替代除法
        # （每元素省一次 FP 除法，结果与除法逐位一致）。
        inv_scale = pl.simt.bitcast(pl.simt.cast((254 - exponent) << 23, pl.DT_UINT32), pl.DT_FP32)
        n = pl.simt.abs(val) * inv_scale
        code = 7
        if n <= 5.0:
            code = 6
        if n <= 3.5:
            code = 5
        if n <= 2.5:
            code = 4
        if n <= 1.75:
            code = 3
        if n <= 1.25:
            code = 2
        if n <= 0.75:
            code = 1
        if n <= 0.25:
            code = 0
        if val < 0.0:
            code = code + 8
        # 偶 lane 与相邻奇 lane 组成一个字节（低 nibble 偶、高 nibble 奇）。
        odd_code = pl.simt.warp_shfl_xor(code, 1)
        if lane % 2 == 0:
            packed = code | (odd_code << 4)
            byte_col = t // 2
            if storage_mode == 0:
                cache[dest0, dest1, 0, byte_col] = pl.simt.cast(packed, pl.DT_UINT8)
            else:
                cache[dest0, dest1, 0, data_col + byte_col] = pl.simt.cast(packed, pl.DT_UINT8)
        if lane == 0:
            group = t // FP4_GROUP_SIZE
            if storage_mode == 1:
                cache[dest0, dest1, 0, scale_col + group] = pl.simt.cast(exponent, pl.DT_UINT8)
            if storage_mode == 0:
                if write_actual_scale:
                    scale_cache[dest0, dest1, 0, group] = pl.simt.cast(exponent, pl.DT_UINT8)

    return postprocess_row


# ================================================================
#  融合后处理 kernel（Vector，块协作 + 整行 UB 缓存）
# ================================================================
def make_postprocess_kernel(
    *,
    total_rows,
    index_head_dim,
    rope_head_dim,
    storage_mode,
    combined_block_size,
    write_scale_cache,
    cache_blocks,
    cache_slots,
    cache_width,
    scale_bytes,
    threads=None,
    scan_blocks=0,
):
    """为某个静态配置构建后处理 kernel。

    调度：**块级 owner**——每个 vector 核（block）按输入顺序扫描全部 token，只处理
    ``owner_key % num_blocks == block_idx`` 的目标行。每个目标行恰好一个 owner 核，
    该核内块协作处理，因此重复 ``cache_index`` 仍是「最后写入」语义。

    每个命中行把 ``projected`` 搬入 UB，随后块内协作完成 RMSNorm（寄存器归约）、
    尾部 RoPE 和 MXFP4 打包，最后写回 cache。
    线程数取 ``D``（每线程一个元素），一个 warp 恰好覆盖一个 32 元素 MXFP4 group。
    """
    dim = int(index_head_dim)
    rd = int(rope_head_dim)
    group_size = int(combined_block_size)
    raw_block_size = int(cache_slots) * group_size
    tail_start = dim - rd
    value_bytes = dim // E2M1_VALUES_PER_BYTE
    write_actual_scale = storage_mode == 0 and write_scale_cache
    scale_boxes = scale_bytes if write_actual_scale else 1
    if threads is None:
        threads = dim  # 每线程一个元素；warp = 一个 group
    if threads > 2048 or threads % FP4_GROUP_SIZE != 0:
        raise ValueError(f"postprocess threads must be a positive multiple of 32 and <= 2048, got {threads}")

    postprocess_row = _make_postprocess_row(
        dim=dim, rd=rd, storage_mode=storage_mode,
        write_actual_scale=write_actual_scale, threads=threads, tail_start=tail_start)

    @pl.jit(arch="3510", auto_mutex=True)
    def indexer_postprocess_kernel(
        projected: pl.Tensor[[total_rows, dim], pl.DT_BF16],
        gamma: pl.Tensor[[1, dim], pl.DT_FP32],
        rope_sin: pl.Tensor[[total_rows, rd], pl.DT_FP32],
        rope_cos: pl.Tensor[[total_rows, rd], pl.DT_FP32],
        cache_index: pl.Tensor[[total_rows], pl.DT_INT64],
        # mode 0: [B, C, 1, D/2]；mode 1: [B, C, 1, G*(D/2 + D/32)]。
        cache: pl.Tensor[[cache_blocks, cache_slots, CACHE_ABI_DIM2, cache_width], pl.DT_UINT8],
        # mode 0 有 scale: [B, C, 1, D/32]；否则为 [1, 1, 1, 1] 别名。
        scale_cache: pl.Tensor[[cache_blocks if write_actual_scale else 1,
                                cache_slots if write_actual_scale else 1,
                                CACHE_ABI_DIM2, scale_boxes], pl.DT_UINT8],
        epsilon: pl.DT_FP32,
        row_count: pl.DT_UINT32,
    ):
        # =================== VECTOR SECTION ===================
        with pl.section_vector():
            proj_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, dim], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
                addrs=[0], mutex_ids=[0])
            norm_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, dim], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
                addrs=[0x1000], mutex_ids=[1])
            pub_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, dim], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
                addrs=[0x2000], mutex_ids=[2])
            gamma_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, dim], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[0x3000], mutex_ids=[3])
            red_g = pl.make_tile_group(
                type=pl.TileType(shape=[8, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[0x4000], mutex_ids=[4])

            gamma_tile = gamma_g.current()
            pl.load(gamma_tile, gamma, [0, 0])

            block_id = pl.get_block_idx()
            # scan_blocks > 0 时核数是编译期常量，``owner_key % num_blocks`` 可强度削减
            # 为位与；运行期值会退化成整数除法。
            num_blocks = scan_blocks if scan_blocks > 0 else pl.get_block_num()
            # 预取下一个 token 的 cache_index：标量 GM 读与当前行的处理重叠，
            # 避免扫描阶段每 token 等待一次 load 延迟（扫描约占总时长一半）。
            next_cr = pl.getval(cache_index, 0)
            for row in pl.range(0, row_count):
                cr = next_cr
                if row + 1 < row_count:
                    next_cr = pl.getval(cache_index, row + 1)
                if cr != -1:
                    if storage_mode == 0:
                        owner_key = cr
                        dest0 = cr // cache_slots
                        dest1 = cr % cache_slots
                        data_col = 0
                        scale_col = 0
                    else:
                        block_id_c = cr // raw_block_size
                        slot = cr % raw_block_size
                        owner_key = block_id_c * cache_slots + slot // group_size
                        dest0 = block_id_c
                        dest1 = slot // group_size
                        data_col = (slot % group_size) * value_bytes
                        scale_col = group_size * value_bytes + (slot % group_size) * scale_bytes
                    if owner_key % num_blocks == block_id:
                        proj_tile = proj_g.next()
                        norm_tile = norm_g.next()
                        pub_tile = pub_g.next()
                        red_tile = red_g.current()
                        pl.load(proj_tile, projected, [row, 0])
                        postprocess_row[threads](
                            proj_tile, gamma_tile, norm_tile, pub_tile, red_tile,
                            rope_sin, rope_cos, cache, scale_cache,
                            row, dest0, dest1, data_col, scale_col, epsilon)

    return indexer_postprocess_kernel


# ================================================================
#  单 kernel 融合（Cube 投影 → MIX 全局屏障 → Vector 后处理）
# ================================================================
def make_fused_kernel(
    *,
    total_rows,
    head_dim,
    index_head_dim,
    rope_head_dim,
    storage_mode,
    combined_block_size,
    write_scale_cache,
    cache_blocks,
    cache_slots,
    scan_blocks=0,
):
    """构建单 kernel 融合实现：同一 launch 内先做 Cube 投影、经 MIX 屏障后做 Vector 后处理。

    与两 kernel 版本相比省一次 kernel 下发与其固定开销（小/中 T 收益明显，约 1.3–2.1×）。
    ``T`` 与 cache 形状因 SIMT 入口要求静态 shape 而固化进本变体（与后处理 kernel 的固化维度一致）。
    """
    dim = int(index_head_dim)
    rd = int(rope_head_dim)
    group_size = int(combined_block_size)
    raw_block_size = int(cache_slots) * group_size
    tail_start = dim - rd
    value_bytes = dim // E2M1_VALUES_PER_BYTE
    scale_bytes = dim // FP4_GROUP_SIZE
    write_actual_scale = storage_mode == 0 and write_scale_cache
    scale_boxes = scale_bytes if write_actual_scale else 1
    cache_width = value_bytes if storage_mode == 0 else group_size * (value_bytes + scale_bytes)
    threads = dim
    if threads > 2048 or threads % FP4_GROUP_SIZE != 0:
        raise ValueError(f"fused threads must be a positive multiple of 32 and <= 2048, got {threads}")

    base_n, base_k, n_tiles, k_tiles = projection_axes(index_head_dim, head_dim)
    postprocess_row = _make_postprocess_row(
        dim=dim, rd=rd, storage_mode=storage_mode,
        write_actual_scale=write_actual_scale, threads=threads, tail_start=tail_start)

    @pl.jit(arch="3510", auto_mutex=True, compile_timeout=200)
    def indexer_fused_kernel(
        latent: pl.Tensor[[total_rows, head_dim], pl.DT_BF16],
        wk: pl.Tensor[[index_head_dim, head_dim], pl.DT_BF16],
        gamma: pl.Tensor[[1, dim], pl.DT_FP32],
        rope_sin: pl.Tensor[[total_rows, rd], pl.DT_FP32],
        rope_cos: pl.Tensor[[total_rows, rd], pl.DT_FP32],
        cache_index: pl.Tensor[[total_rows], pl.DT_INT64],
        # mode 0: [B, C, 1, D/2]；mode 1: [B, C, 1, G*(D/2 + D/32)]。
        cache: pl.Tensor[[cache_blocks, cache_slots, CACHE_ABI_DIM2, cache_width], pl.DT_UINT8],
        # mode 0 有 scale: [B, C, 1, D/32]；否则为 [1, 1, 1, 1] 别名。
        scale_cache: pl.Tensor[[cache_blocks if write_actual_scale else 1,
                                cache_slots if write_actual_scale else 1,
                                CACHE_ABI_DIM2, scale_boxes], pl.DT_UINT8],
        projected: pl.Tensor[[total_rows, index_head_dim], pl.DT_BF16],
        epsilon: pl.DT_FP32,
    ):
        """Cube 段投影 → MIX 全局屏障 → Vector 段 RMSNorm/RoPE/MXFP4 散写。"""
        # =================== CUBE SECTION：BF16 投影 ===================
        with pl.section_cube():
            num_cores = pl.get_block_num()
            core_id = pl.get_block_idx()
            m_tiles = (total_rows + BASE_M - 1) // BASE_M
            total_tiles = m_tiles * n_tiles
            a_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[BASE_M, base_k], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat),
                addrs=0, mutex_ids=[0, 1])
            b_l1 = pl.make_tile_group(
                type=pl.TileType(shape=[base_k, base_n], dtype=pl.DT_BF16,
                                 target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
                addrs=0x10000, mutex_ids=[2, 3])
            a_left = pl.make_tile_group(
                type=pl.TileType(shape=[BASE_M, base_k], dtype=pl.DT_BF16,
                                 target_memory=pl.MemorySpace.Left, compact=1),
                addrs=0, mutex_ids=[4, 5])
            b_right = pl.make_tile_group(
                type=pl.TileType(shape=[base_k, base_n], dtype=pl.DT_BF16,
                                 target_memory=pl.MemorySpace.Right, compact=1),
                addrs=0, mutex_ids=[6, 7])
            acc = pl.make_tile_group(
                type=pl.TileType(shape=[BASE_M, base_n], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Acc, compact=1),
                addrs=0, mutex_ids=[8])
            for tile_idx in pl.range(core_id, total_tiles, num_cores):
                m_idx = tile_idx // n_tiles
                n_idx = tile_idx % n_tiles
                valid_m = pl.min(BASE_M, total_rows - m_idx * BASE_M)
                acc_tile = acc.next()
                pl.set_validshape(acc_tile, [valid_m, base_n])
                for k_idx in pl.range(0, k_tiles):
                    valid_k = pl.min(base_k, head_dim - k_idx * base_k)
                    a_tile = a_l1.next()
                    b_tile = b_l1.next()
                    pl.set_validshape(a_tile, [valid_m, valid_k])
                    pl.load(a_tile, latent, [m_idx * BASE_M, k_idx * base_k])
                    pl.set_validshape(b_tile, [valid_k, base_n])
                    pl.load(b_tile, wk, [n_idx * base_n, k_idx * base_k], order=[1, 0])
                    left_tile = a_left.next()
                    right_tile = b_right.next()
                    pl.set_validshape(left_tile, [valid_m, valid_k])
                    pl.move(left_tile, a_tile)
                    pl.set_validshape(right_tile, [valid_k, base_n])
                    pl.move(right_tile, b_tile)
                    if k_idx == 0:
                        pl.matmul(acc_tile, left_tile, right_tile)
                    else:
                        pl.matmul_acc(acc_tile, acc_tile, left_tile, right_tile)
                pl.store(projected, acc_tile, [m_idx * BASE_M, n_idx * base_n])
            # 投影全部落 GM 后，AIC 与 AIV 在 MIX 屏障汇合。
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

        # =================== VECTOR SECTION：融合后处理 ===================
        with pl.section_vector():
            pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
            proj_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, dim], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
                addrs=[0], mutex_ids=[0])
            norm_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, dim], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
                addrs=[0x1000], mutex_ids=[1])
            pub_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, dim], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
                addrs=[0x2000], mutex_ids=[2])
            gamma_g = pl.make_tile_group(
                type=pl.TileType(shape=[1, dim], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[0x3000], mutex_ids=[3])
            red_g = pl.make_tile_group(
                type=pl.TileType(shape=[8, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                addrs=[0x4000], mutex_ids=[4])
            gamma_tile = gamma_g.current()
            pl.load(gamma_tile, gamma, [0, 0])
            block_id = pl.get_block_idx()
            num_blocks = scan_blocks if scan_blocks > 0 else pl.get_block_num()
            # 预取下一个 token 的 cache_index，隐藏标量 GM 读延迟。
            next_cr = pl.getval(cache_index, 0)
            for row in pl.range(0, total_rows):
                cr = next_cr
                if row + 1 < total_rows:
                    next_cr = pl.getval(cache_index, row + 1)
                if cr != -1:
                    if storage_mode == 0:
                        owner_key = cr
                        dest0 = cr // cache_slots
                        dest1 = cr % cache_slots
                        data_col = 0
                        scale_col = 0
                    else:
                        block_id_c = cr // raw_block_size
                        slot = cr % raw_block_size
                        owner_key = block_id_c * cache_slots + slot // group_size
                        dest0 = block_id_c
                        dest1 = slot // group_size
                        data_col = (slot % group_size) * value_bytes
                        scale_col = group_size * value_bytes + (slot % group_size) * scale_bytes
                    if owner_key % num_blocks == block_id:
                        proj_tile = proj_g.next()
                        norm_tile = norm_g.next()
                        pub_tile = pub_g.next()
                        red_tile = red_g.current()
                        pl.load(proj_tile, projected, [row, 0])
                        postprocess_row[threads](
                            proj_tile, gamma_tile, norm_tile, pub_tile, red_tile,
                            rope_sin, rope_cos, cache, scale_cache,
                            row, dest0, dest1, data_col, scale_col, epsilon)

    return indexer_fused_kernel


# ================================================================
#  Host API
# ================================================================
_FUSED_CACHE = {}
_POSTPROCESS_CACHE = {}


def get_effective_core_counts():
    """查询当前设备 ``(cube_cores, vector_cores)``。

    以 pypto_pro 平台接口为主（读实时 device）；torch_npu 设备属性与静态默认值只在
    无设备时兜底，保证 host 侧 grid 推导仍可用。
    """
    cube = 0
    vector = 0
    try:
        info = pro_platform.get_platform_info()
        cube = int(info.cube_core_num)
        vector = int(info.vector_core_num)
    except Exception:  # noqa: BLE001 - 退化到设备属性。
        pass
    if cube <= 0 or vector <= 0:
        try:
            props = torch.npu.get_device_properties(torch.npu.current_device())
            cube = cube or int(getattr(props, "cube_core_num", 0))
            vector = vector or int(getattr(props, "vector_core_num", 0))
        except Exception:  # noqa: BLE001 - 保留静态默认值。
            pass
    cube = cube or DEFAULT_CUBE_CORES
    vector = vector or DEFAULT_VECTOR_CORES
    if cube <= 0 or vector < cube:
        raise RuntimeError(f"invalid effective NPU core counts: AIC={cube}, AIV={vector}")
    return cube, vector


def _validate_operator_geometry(latent, wk, rope_sin, storage_mode):
    total_rows, head_dim = map(int, latent.shape)
    index_head_dim = int(wk.shape[0])
    rope_head_dim = int(rope_sin.shape[1])
    if total_rows <= 0:
        raise ValueError("T must be positive")
    if head_dim <= 0 or head_dim % NZ_ALIGN != 0:
        raise ValueError(f"H must be a positive multiple of {NZ_ALIGN} for the NZ contract")
    if index_head_dim < FP4_GROUP_SIZE or index_head_dim % FP4_GROUP_SIZE != 0:
        raise ValueError(f"D must be a positive multiple of {FP4_GROUP_SIZE}")
    if rope_head_dim <= 0 or rope_head_dim % ROPE_PAIR_SIZE != 0:
        raise ValueError("Dr must be a positive even number")
    if rope_head_dim > index_head_dim:
        raise ValueError("Dr cannot exceed D")
    if storage_mode not in (0, 1):
        raise ValueError("storage_mode must be 0 or 1")
    return total_rows, head_dim, index_head_dim, rope_head_dim


def _validate_cache_layout(k_cache, k_scale_cache, storage_mode, combined_block_size,
                           value_bytes, scale_bytes):
    if int(k_cache.shape[2]) != CACHE_ABI_DIM2:
        raise ValueError(f"k_cache dimension 2 must equal {CACHE_ABI_DIM2}")
    block_num = int(k_cache.shape[0])
    if storage_mode == 0:
        if int(k_cache.shape[3]) != value_bytes:
            raise ValueError(f"storage_mode=0 requires k_cache.shape[-1] == {value_bytes}")
        if k_scale_cache is not None:
            expected = (block_num, int(k_cache.shape[1]), CACHE_ABI_DIM2, scale_bytes)
            if tuple(k_scale_cache.shape) != expected:
                raise ValueError(f"k_scale_cache must have shape {expected}")
    else:
        if combined_block_size <= 0:
            raise ValueError("storage_mode=1 requires combined_block_size to be positive")
        expected_width = combined_block_size * (value_bytes + scale_bytes)
        if int(k_cache.shape[3]) != expected_width:
            raise ValueError(f"storage_mode=1 requires k_cache.shape[-1] == {expected_width}")


def _get_postprocess_callable(total_rows, index_head_dim, rope_head_dim, storage_mode,
                              combined_block_size, write_scale_cache, cache_blocks, cache_slots,
                              scan_blocks):
    key = (total_rows, index_head_dim, rope_head_dim, storage_mode, combined_block_size,
           write_scale_cache, cache_blocks, cache_slots, scan_blocks)
    compiled = _POSTPROCESS_CACHE.get(key)
    if compiled is None:
        value_bytes, scale_bytes = value_and_scale_bytes(index_head_dim)
        cache_width = value_bytes if storage_mode == 0 else (
            combined_block_size * (value_bytes + scale_bytes))
        compiled = make_postprocess_kernel(
            total_rows=total_rows,
            index_head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            storage_mode=storage_mode,
            combined_block_size=combined_block_size,
            write_scale_cache=write_scale_cache,
            cache_blocks=cache_blocks,
            cache_slots=cache_slots,
            cache_width=cache_width,
            scale_bytes=scale_bytes,
            scan_blocks=scan_blocks,
        )
        _POSTPROCESS_CACHE[key] = compiled
    return compiled


def _get_fused_callable(total_rows, head_dim, index_head_dim, rope_head_dim, storage_mode,
                        combined_block_size, write_scale_cache, cache_blocks, cache_slots,
                        scan_blocks):
    """融合（单 kernel）实现的编译缓存，键与后处理变体一致（另加 head_dim）。"""
    key = (total_rows, head_dim, index_head_dim, rope_head_dim, storage_mode,
           combined_block_size, write_scale_cache, cache_blocks, cache_slots, scan_blocks)
    compiled = _FUSED_CACHE.get(key)
    if compiled is None:
        compiled = make_fused_kernel(
            total_rows=total_rows, head_dim=head_dim, index_head_dim=index_head_dim,
            rope_head_dim=rope_head_dim, storage_mode=storage_mode,
            combined_block_size=combined_block_size, write_scale_cache=write_scale_cache,
            cache_blocks=cache_blocks, cache_slots=cache_slots, scan_blocks=scan_blocks)
        _FUSED_CACHE[key] = compiled
    return compiled


def _resolve_fused_grid():
    """融合 kernel 为 cube+vector 混合核：每 block = 1 AIC + 2 AIV，故 block 数 ≤ min(AIC, AIV/2)。"""
    cube_cores, vector_cores = get_effective_core_counts()
    return max(1, min(cube_cores, vector_cores // 2))


def _resolve_launch_grids(total_rows, head_dim, index_head_dim, storage_mode,
                          combined_block_size):
    """预处理：查设备核数，算出两个 kernel 的 launch 核数。

    投影核数 = ``min(AIC 数, m_tiles·n_tiles)``；后处理核数 = ``min(AIV 数, 工作项数)``：
    mode0 工作项 = token 数 ``T``，mode1 工作项 = ``ceil(T / G)``。工作项不足满核时不空起核。
    该核数同时作为后处理/融合 kernel 的编译期常量（owner 判定的模数），故只按工作量收缩。
    返回 ``(cube_block_num, vec_block_num)``。
    """
    cube_cores, vector_cores = get_effective_core_counts()
    cube_block_num = projection_grid_blocks(total_rows, head_dim, index_head_dim, cube_cores)
    if storage_mode == 0:
        vec_block_num = min(vector_cores, total_rows)
    else:
        vec_block_num = min(vector_cores, _ceil_div(total_rows, combined_block_size))
    return cube_block_num, vec_block_num


def indexer_prologue_k_wrapper(
    latent: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale_cache: torch.Tensor | None = None,
    *,
    cache_index: torch.Tensor,
    storage_mode: int,
    norm_eps: float,
    combined_block_size: int = -1,
) -> torch.Tensor:
    """把 ``latent[T,H]`` 投影并散写 MXFP4 打包行到 ``k_cache``。

    ``storage_mode=0`` 把 E2M1 字节写入 ``k_cache``，可选地把 UE8M0 scale 写入
    ``k_scale_cache``。``storage_mode=1`` 每行 cache 组合 ``G`` 个 token，
    ``G = combined_block_size``。``cache_index[t] == -1`` 跳过该 token。
    返回传入的同一个 ``k_cache``（原地更新）。
    """
    total_rows, head_dim, index_head_dim, rope_head_dim = (
        _validate_operator_geometry(latent, wk, rope_sin, storage_mode))
    value_bytes, scale_bytes = value_and_scale_bytes(index_head_dim)
    _validate_cache_layout(k_cache, k_scale_cache, storage_mode, combined_block_size,
                           value_bytes, scale_bytes)
    if not latent.is_contiguous() or not wk.is_contiguous():
        raise ValueError("latent and wk must be contiguous")
    if k_cache.stride(0) != int(k_cache.shape[1] * k_cache.shape[2] * k_cache.shape[3]):
        raise NotImplementedError(
            "PyPTO-Pro entry currently requires a contiguous k_cache; "
            "materialise strided axis-0 views first")

    projected = torch.empty((total_rows, index_head_dim), dtype=torch.bfloat16, device=latent.device)

    write_scale_cache = storage_mode == 0 and k_scale_cache is not None
    scale_arg = k_scale_cache if write_scale_cache else k_cache[:1, :1, :1, :1]
    gamma1 = norm_weight.reshape(1, index_head_dim)

    if total_rows <= FUSED_T_MAX:
        # ---- 单 kernel 融合：Cube 投影 + MIX 屏障 + Vector 后处理（省一次下发固定开销）----
        fused_grid = _resolve_fused_grid()
        fused_kernel = _get_fused_callable(
            total_rows, head_dim, index_head_dim, rope_head_dim, storage_mode,
            combined_block_size, write_scale_cache, int(k_cache.shape[0]), int(k_cache.shape[1]),
            fused_grid)
        fused_kernel[None, fused_grid](
            latent, wk, gamma1, rope_sin, rope_cos, cache_index,
            k_cache, scale_arg, projected, float(norm_eps))
    else:
        # ---- 两 kernel：BF16 投影（Cube）+ 融合 norm/rope/mxfp4/scatter（Vector）----
        cube_block_num, vec_block_num = _resolve_launch_grids(
            total_rows, head_dim, index_head_dim, storage_mode, combined_block_size)
        indexer_projection_kernel[None, cube_block_num,
                                  projection_tiling_key(index_head_dim, head_dim)](latent, wk, projected)
        postprocess_kernel = _get_postprocess_callable(
            total_rows, index_head_dim, rope_head_dim, storage_mode, combined_block_size,
            write_scale_cache, int(k_cache.shape[0]), int(k_cache.shape[1]), vec_block_num)
        postprocess_kernel[None, vec_block_num](
            projected, gamma1, rope_sin, rope_cos, cache_index,
            k_cache, scale_arg, float(norm_eps), total_rows)
    return k_cache
