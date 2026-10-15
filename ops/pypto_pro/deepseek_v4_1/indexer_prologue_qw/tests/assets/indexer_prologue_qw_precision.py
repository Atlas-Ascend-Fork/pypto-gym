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
"""indexer_prologue_qw 的自包含精度比较器。

职责：把 operator 的量化边界豁免口径固化成仓库内可复用的比较模块，使算子验收不再
依赖任何外部绝对路径。数学 oracle 是本包的 ``indexer_prologue_qw_golden_cpu``
（纯 torch 的独立参考），比较器只负责判定「哪些字节差异是 FP32 累加次序/量化边界
本身无法判定的」。

来源与出处
----------
豁免语义来自 SPEC ``tolerance.policy = pro_scheme_a`` 对独立 CPU Golden 的比较
约定，具体三类边界是源实现固有的：

* **E2M1 一步同号移动**：golden 商值落在 E2M1 相邻档位中点（0.25, 0.75, …）的
  ``ulps`` 窗口内时，浮点累加次序的 1 ulp 差异足以决定舍入方向；
* **``+0`` / ``-0`` 对有符号零**：幅值码同为 0、仅符号位不同，且 golden 商值低于
  第一个 E2M1 中点（下溢符号不是契约）；
* **E8M0 ``±1``**：``amax`` 落在 ``6*2^k`` 边界、其自身点积的累加展布之内时，
  两种求和次序分居边界两侧，指数恰好相差 1。

上述判定与参考比较实现（cannbot-dsl ``samples/indexer_prologue_qw`` 的验证逻辑）
等价——为通过验收所冻结的正是这套规则（21/21 PASS，未放宽容差）；本模块把它改写为
自包含代码，配合 CPU golden 的 post-RoPE ``q_fp32`` 调试返回完成逐位判定。
"""

from __future__ import annotations

__all__ = [
    "E2M1_MIDPOINTS",
    "IndexerPrologueQwInputs",
    "IndexerPrologueQwOutputs",
    "POSITIONAL_NAMES",
    "SOFTMAX_SCALE",
    "compare",
    "descale_accum_ambiguous",
    "e2m1_nibble_to_fp32",
    "fp32_to_e2m1_nibble",
    "indexer_prologue_qw_golden",
    "pack_e2m1_nibbles",
    "q_accum_ambiguous",
    "q_mismatches",
    "select_rows",
    "unpack_e2m1_nibbles",
]

import math
from dataclasses import dataclass, replace

import torch

from indexer_prologue_qw_golden_cpu import (
    E2M1_POS,
    E8M0_BIAS,
    FP4_E2M1_MAX,
    MX_GROUP_SIZE,
    SOFTMAX_SCALE,
    _decode_e8m0_scale,
    _fp32_to_e2m1_nibble,
    _pack_e2m1_nibbles,
    indexer_prologue_qw_golden_cpu,
)

#: Goldens / operator share the same 8 positional tensors plus a scalar scale.
_FIELD_NAMES = (
    "x", "qr", "wqb", "ww", "descale_qr", "descale_wqb", "rope_sin", "rope_cos",
)

#: Halfway between consecutive E2M1 magnitudes: where rounding flips.
E2M1_MIDPOINTS = tuple((low + high) / 2 for low, high in zip(E2M1_POS, E2M1_POS[1:]))


@dataclass
class IndexerPrologueQwInputs:
    """算子与 golden 共用的输入 bundle（权重保持完整，仅行维随 ``T`` 变化）。"""

    x: torch.Tensor
    qr: torch.Tensor
    wqb: torch.Tensor
    ww: torch.Tensor
    descale_qr: torch.Tensor
    descale_wqb: torch.Tensor
    rope_sin: torch.Tensor
    rope_cos: torch.Tensor
    softmax_scale: float


@dataclass
class IndexerPrologueQwOutputs:
    """golden 的完整输出：公开三件套 + 量化前的 post-RoPE FP32 Q（调试用）。"""

    q: torch.Tensor
    descale_q: torch.Tensor
    w: torch.Tensor
    q_fp32: torch.Tensor


def _as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """把 E4M3 视图还原为 uint8 位型，供 CPU golden 的 256 项 LUT 按字节解码。

    参数：``tensor`` 为 E4M3 或 uint8 张量。

    数据流：返回 1 字节 itemsize 的 uint8 视图；已是 uint8 时原样返回，不改数据。
    """
    if tensor.dtype == torch.uint8:
        return tensor
    return tensor.view(torch.uint8)


def select_rows(inputs: IndexerPrologueQwInputs, rows) -> IndexerPrologueQwInputs:
    """把 ``inputs`` 限制到 ``rows`` 指定的 ``T`` 行，两个权重保持完整。

    本算子每个输出行只依赖同下标的 ``x``/``qr``/scale/RoPE 行与整块权重，归约不跨
    ``T``，所以子集的 golden 精确等于子集本身——这正是超长 ``T`` 可校验的原因。

    参数：``inputs`` 输入 bundle；``rows`` 行索引或 slice。

    数据流：返回新的 ``IndexerPrologueQwInputs``，权重字段不变。
    """
    return IndexerPrologueQwInputs(
        x=inputs.x[rows],
        qr=inputs.qr[rows],
        wqb=inputs.wqb,
        ww=inputs.ww,
        descale_qr=inputs.descale_qr[rows],
        descale_wqb=inputs.descale_wqb,
        rope_sin=inputs.rope_sin[rows],
        rope_cos=inputs.rope_cos[rows],
        softmax_scale=inputs.softmax_scale,
    )


def indexer_prologue_qw_golden(inputs: IndexerPrologueQwInputs) -> IndexerPrologueQwOutputs:
    """用 CPU golden 求 ``inputs`` 的数学基准，并附带 post-RoPE ``q_fp32``。

    职责：把 bundle 展开到 ``indexer_prologue_qw_golden_cpu``（``qr``/``wqb`` 以字节
    视图传入），开启 ``return_q_fp32`` 以便比较器在 E2M1 中点与 E8M0 ``6*2^k``
    边界上做自包含判定。

    参数：``inputs`` 为 8 张量 + ``softmax_scale`` 的 bundle。

    数据流：返回 ``IndexerPrologueQwOutputs``（与输入同设备）。
    """
    q, descale_q, w, q_fp32 = indexer_prologue_qw_golden_cpu(
        inputs.x,
        _as_bytes(inputs.qr),
        _as_bytes(inputs.wqb),
        inputs.ww,
        inputs.descale_qr,
        inputs.descale_wqb,
        inputs.rope_sin,
        inputs.rope_cos,
        softmax_scale=inputs.softmax_scale,
        return_q_fp32=True,
    )
    return IndexerPrologueQwOutputs(q=q, descale_q=descale_q, w=w, q_fp32=q_fp32)


def unpack_e2m1_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """把 (..., D/2) 字节拆成 (..., D) 个 nibble（偶列低 4 bit、奇列高 4 bit）。

    参数：``packed`` uint8 的打包字节。

    数据流：返回 uint8 nibble 张量。
    """
    low = packed & 0x0F
    high = (packed.to(torch.int16) >> 4).to(torch.uint8)
    return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def e2m1_nibble_to_fp32(nibbles: torch.Tensor) -> torch.Tensor:
    """把 E2M1 nibble 按其 3 bit 幅值码 + 符号位还原为 FP32。

    参数：``nibbles`` uint8。

    数据流：返回 FP32 值张量。
    """
    codebook = torch.tensor(E2M1_POS, dtype=torch.float32, device=nibbles.device)
    mag = codebook[(nibbles & 0x7).long()]
    sign = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    return mag * sign


def fp32_to_e2m1_nibble(values: torch.Tensor) -> torch.Tensor:
    """FP32 就近舍入到 OCP E2M1（.5 处远离零），复用 CPU golden 的同一 codebook。

    参数：``values`` FP32 张量。

    数据流：返回 uint8 nibble（含符号位）。
    """
    codebook = torch.tensor(E2M1_POS, dtype=torch.float32, device=values.device)
    arange8 = torch.arange(8, device=values.device, dtype=torch.int64)
    return _fp32_to_e2m1_nibble(values, codebook, arange8)


def pack_e2m1_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    """把 (..., D) nibble 打包成 (..., D/2) 字节（复用 CPU golden 的同一布局）。

    参数：``nibbles`` 最后一维为偶数。

    数据流：返回 uint8 打包字节。
    """
    return _pack_e2m1_nibbles(nibbles)


def _dequant_mx_row(values: torch.Tensor, scale_bytes: torch.Tensor, k: int) -> torch.Tensor:
    """解码 MX 操作数的一行到 FP32（E4M3 数值 × 每 32 个 K 的 E8M0 scale）。

    参数：``values`` E4M3/FP32[QLORA]；``scale_bytes`` uint8[.., 2]；``k`` K 长度。

    数据流：返回 FP32[1, k]，供 ``_gemm_uncertainty`` 估算点积展布。
    """
    scale = _decode_e8m0_scale(scale_bytes.reshape(1, -1), 1, k)
    scale = scale.repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :k]
    return values.float().reshape(1, k) * scale


def _gemm_uncertainty(
    inputs: IndexerPrologueQwInputs, row: int, head: int, col: int, d: int
) -> float:
    """同一 Q 点积两种 FP32 累加次序的最大可能差距。

    kernel 与 CPU golden 归约同一组 1280 个乘积，差别只在舍入；用
    ``sqrt(k) * eps * sum|term|`` 作宽松上界。它相对 E2M1 量化步长极小
    （冻结几何下 ~1e-5 步），因此只能豁免本就贴着舍入边界的值。

    参数：``inputs``/``row``/``head``/``col``/``d`` 定位该输出元素对应的点积。

    数据流：返回标量展布。
    """
    k = inputs.qr.shape[1]
    lhs = _dequant_mx_row(inputs.qr[row], inputs.descale_qr[row], k)
    rhs_row = head * d + col
    rhs = _dequant_mx_row(inputs.wqb[rhs_row], inputs.descale_wqb[rhs_row], k)
    terms = float((lhs * rhs).abs().sum())
    return math.sqrt(k) * torch.finfo(torch.float32).eps * terms


def _element_uncertainty(inputs, want, coord, width) -> float:
    """把 ``_gemm_uncertainty`` 经 RoPE 混合传到某个输出元素。

    参数：``coord=(row, head, col)``；``width=(d, dr)``。

    数据流：返回该元素自有的累加展布（RoPE 前的列直接取 base，RoPE 内的列叠加
        rotate-half 伙伴列的展布与一次 eps）。
    """
    row, head, col = coord
    d, dr = width
    base = _gemm_uncertainty(inputs, row, head, col, d)
    if dr == 0 or col < d - dr:
        return base
    local = col - (d - dr)
    half = dr // 2
    partner = local + half if local < half else local - half
    cos = abs(float(inputs.rope_cos[row, local]))
    sin = abs(float(inputs.rope_sin[row, local]))
    partner_unc = _gemm_uncertainty(inputs, row, head, partner + d - dr, d)
    value = abs(float(want.q_fp32[row, head, col]))
    return cos * base + sin * partner_unc + torch.finfo(torch.float32).eps * value


def q_accum_ambiguous(
    got: torch.Tensor,
    want: IndexerPrologueQwOutputs,
    inputs: IndexerPrologueQwInputs,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """在 ``candidates`` 字节中判定哪些 E2M1 决策是 FP32 无法判定的。

    固定 ulp 窗口只是廉价初筛：随 ``T`` 增大，“远离中点仍属纯累加噪声”的样本会
    出现。对每个幸存者重算其点积展布，只在该区间跨过两档中点时豁免。仅极少数字节
    会走到这里，故逐元素代价可忽略。

    参数：``got`` 打包 q；``want`` golden 输出；``inputs``；``candidates`` 待判掩码。

    数据流：返回与 ``candidates`` 同形的 bool 掩码（True = 可豁免）。
    """
    index = torch.nonzero(candidates)
    if index.numel() == 0:
        return torch.zeros_like(candidates)

    t, n, half_d = got.shape
    d = half_d * 2
    dr = int(inputs.rope_sin.shape[-1])
    got_nibbles = unpack_e2m1_nibbles(got)
    want_nibbles = unpack_e2m1_nibbles(want.q)
    scale = torch.pow(2.0, want.descale_q.reshape(t, n, -1).float() - E8M0_BIAS)

    ambiguous = torch.zeros_like(candidates)
    for row, head, byte in index.tolist():
        settled = False
        for half in (0, 1):
            col = 2 * byte + half
            code_got = int(got_nibbles[row, head, col])
            code_want = int(want_nibbles[row, head, col])
            if code_got == code_want:
                continue
            if (code_got & 0x8) != (code_want & 0x8):
                settled = True
                break
            low = min(code_got & 0x7, code_want & 0x7)
            if abs((code_got & 0x7) - (code_want & 0x7)) != 1:
                settled = True
                break
            group_scale = float(scale[row, head, col // MX_GROUP_SIZE])
            window = (
                _element_uncertainty(inputs, want, (row, head, col), (d, dr))
                / group_scale
            )
            quotient = abs(float(want.q_fp32[row, head, col])) / group_scale
            if abs(quotient - E2M1_MIDPOINTS[low]) > window:
                settled = True
                break
        ambiguous[row, head, byte] = not settled
    return ambiguous


def descale_accum_ambiguous(
    got: torch.Tensor,
    want: IndexerPrologueQwOutputs,
    inputs: IndexerPrologueQwInputs,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """在 ``candidates`` scale 字节中判定哪些指数是 FP32 无法判定的。

    ``descale_q`` 取自一组的 amax，故与 golden 逐位相等——除非 ``amax/6`` 落在 2 的
    幂上。当组内最大幅值落在 ``6*2^k`` 边界的累加噪声内，两种求和次序分居两侧，
    指数恰好相差 1。只豁免「相差 1、且该 amax 在自身点积展布内触及边界」的字节。

    参数：``got`` 算子 descale_q；``want`` golden 输出；``inputs``；``candidates`` 掩码。

    数据流：返回与 ``candidates`` 同形的 bool 掩码（True = 可豁免）。
    """
    index = torch.nonzero(candidates)
    if index.numel() == 0:
        return torch.zeros_like(candidates)

    t, n = want.q_fp32.shape[0], want.q_fp32.shape[1]
    d = want.q_fp32.shape[2]
    dr = int(inputs.rope_sin.shape[-1])
    groups_per_head = got.shape[2] * got.shape[3]

    ambiguous = torch.zeros_like(candidates)
    for position in index.tolist():
        row, head = position[0], position[1]
        flat_group = position[2] * got.shape[3] + position[3]
        if abs(int(got[tuple(position)]) - int(want.descale_q[tuple(position)])) != 1:
            continue
        span = d // groups_per_head
        lo = flat_group * span
        group = want.q_fp32[row, head, lo:lo + span].abs()
        amax = float(group.max())
        if amax == 0.0:
            continue
        col = lo + int(group.argmax())
        window = _element_uncertainty(inputs, want, (row, head, col), (d, dr))
        # scale = 2**ceil(log2(amax / 6))，两候选指数间的边界是较小解码 scale 的 6 倍。
        lower_code = min(int(got[tuple(position)]), int(want.descale_q[tuple(position)]))
        boundary = FP4_E2M1_MAX * 2.0 ** (lower_code - E8M0_BIAS)
        ambiguous[tuple(position)] = abs(amax - boundary) <= window
    return ambiguous


def q_mismatches(
    got: torch.Tensor, want: IndexerPrologueQwOutputs, ulps: int = 4
) -> torch.Tensor:
    """打包 ``q`` 中「并非舍入并列」而不同的字节掩码。

    kernel 量化自己 L0C 累加的 FP32 结果，golden 量化 torch 对同一组乘积的求和，
    两者相差 1 ulp，只有当值贴到 E2M1 舍入中点时才改变结果。一个字节被豁免，需其
    每个差异 nibble 满足：

    * 一步同号移动，且 golden 商值落在某中点的 ``ulps`` 窗口内；或
    * ``+0``/``-0`` 对（幅值码同为 0，仅符号位不同）且商值低于第一个中点。

    其余（非零码符号翻转、跳两档、远离中点）都算真实不符。

    参数：``got`` 打包 q；``want`` golden 输出；``ulps`` 中点窗口宽度。

    数据流：返回 bool 掩码（True = 无法解释的差异字节）。
    """
    differs = got != want.q
    if not bool(differs.any()):
        return differs

    t, n, half_d = got.shape
    got_nibbles = unpack_e2m1_nibbles(got)
    want_nibbles = unpack_e2m1_nibbles(want.q)
    nibble_differs = got_nibbles != want_nibbles
    one_step = (
        (got_nibbles & 0x7).to(torch.int32) - (want_nibbles & 0x7).to(torch.int32)
    ).abs() == 1
    same_sign = (got_nibbles & 0x8) == (want_nibbles & 0x8)

    scale = torch.pow(2.0, want.descale_q.reshape(t, n, -1).float() - E8M0_BIAS)
    quotient = (
        want.q_fp32.reshape(t, n, scale.shape[-1], MX_GROUP_SIZE)
        / scale.unsqueeze(-1)
    ).abs().reshape(t, n, half_d * 2)
    eps = torch.finfo(torch.float32).eps
    at_midpoint = torch.zeros_like(quotient, dtype=torch.bool)
    for midpoint in E2M1_MIDPOINTS:
        at_midpoint |= (quotient - midpoint).abs() <= ulps * eps * midpoint

    signed_zero = (
        ((got_nibbles & 0x7) == 0)
        & ((want_nibbles & 0x7) == 0)
        & (quotient < E2M1_MIDPOINTS[0])
    )
    excused = (
        ~nibble_differs
        | (one_step & same_sign & at_midpoint)
        | signed_zero
    )
    # 一个字节的两个 nibble 都要有解释，才豁免该字节。
    return differs & ~(excused[..., 0::2] & excused[..., 1::2])


def compare(got, want: IndexerPrologueQwOutputs, inputs: IndexerPrologueQwInputs):
    """比较算子输出与 golden，按 SPEC 边界豁免口径判定。

    职责：先断言三件套形状，再逐类处理差异——E8M0 字节先过 ``descale_accum_ambiguous``；
    一旦接受某个 scale 边界，就用该已验证 scale 对该组的 ``q_fp32`` 重新量化
    （绝不豁免该组 Q 字节本身）；残留 Q 差异再过 ``q_mismatches`` 与
    ``q_accum_ambiguous``；``w`` 用 ``rtol=atol=2e-2``。

    参数：``got`` 为 ``(q, descale_q, w)``；``want``/``inputs`` 为 golden 输出与输入。

    数据流：返回 ``(raw_q, raw_scales, max_w)``——原始 q/scale 字节差异数与被豁免
        前的 ``w`` 最大绝对误差；任一未解释差异抛 ``AssertionError``。
    """
    q, scales, w = got
    if q.shape != want.q.shape:
        raise AssertionError(f"q shape {q.shape} != {want.q.shape}")
    if scales.shape != want.descale_q.shape:
        raise AssertionError(f"scale shape {scales.shape} != {want.descale_q.shape}")
    if w.shape != want.w.shape:
        raise AssertionError(f"w shape {w.shape} != {want.w.shape}")
    raw_q = int((q != want.q).sum())
    raw_scales = int((scales != want.descale_q).sum())
    bad_scales = scales != want.descale_q
    if bad_scales.any():
        bad_scales &= ~descale_accum_ambiguous(scales, want, inputs, bad_scales)
    if bad_scales.any():
        raise AssertionError(f"{int(bad_scales.sum())} unexplained scale bytes")
    if raw_scales:
        # 接受的 scale 边界会改变该组全部 32 个码；用已验证 scale 重新量化独立 CPU
        # 值，绝不因此豁免该组 Q 字节的比较。
        decoded = torch.pow(2.0, scales.reshape(*q.shape[:2], -1).float() - E8M0_BIAS)
        quotient = want.q_fp32 / decoded.repeat_interleave(MX_GROUP_SIZE, dim=-1)
        want = replace(
            want,
            descale_q=scales,
            q=pack_e2m1_nibbles(fp32_to_e2m1_nibble(quotient)),
        )
    bad_q = q_mismatches(q, want)
    if bad_q.any():
        bad_q &= ~q_accum_ambiguous(q, want, inputs, bad_q)
    if bad_q.any():
        raise AssertionError(f"{int(bad_q.sum())} unexplained Q bytes")
    torch.testing.assert_close(w, want.w, rtol=2e-2, atol=2e-2)
    return raw_q, raw_scales, float((w - want.w).abs().max())


#: Re-exported for callers that construct bundles positionally.
POSITIONAL_NAMES = _FIELD_NAMES
