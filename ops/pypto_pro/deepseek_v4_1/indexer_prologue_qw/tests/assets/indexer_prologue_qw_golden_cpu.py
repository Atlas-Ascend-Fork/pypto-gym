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

"""indexer_prologue_qw 的 CPU golden（对齐 Scriptor 的精确 SPEC 输出 ABI）。

职责：提供一份独立、纯 torch 的参考计算，作为算子精度验收的绝对基准。所有
中间计算都使用标准 dtype：FP8 E4M3 字节经 256 项 LUT 解码，E8M0 scale 字节
按 ``2**(byte-127)`` 还原。按行分块处理，因此完整 ``T`` 范围（最大 262144）
也不会物化 (T, N, D, 8) 的距离张量。

Golden 计算清单：
1. 解码 qr/wqb 的 E4M3 字节 -> FP32，乘以每 32 个元素的 E8M0 块 scale
2. Y = qr_deq @ wqb_deq^T : [c,K]x[K,N*D] -> [c,N*D]，FP32 累加
3. Qp = reshape(Y, [c,N,D])；对末尾 Dr=64 列做 rotate-half RoPE
4. 沿 D 做 MXFP4 量化（每 32 一组）：amax -> E8M0 向上取整 scale -> E2M1
   nibble，每字节打包两个（偶数列在低 nibble）
5. descale_q = E8M0 指数，形状 (c,N,ceil(D/64),2)，不足用 127 填充
6. w = softmax_scale * (x @ ww^T) : [c,dim]x[dim,N] -> [c,N]，FP32 累加
"""

import logging

import torch

logger = logging.getLogger(__name__)

MX_GROUP_SIZE = 32
MX_SCALE_PAIR = 2
MX_K_ALIGN = 64
FP4_E2M1_MAX = 6.0
E8M0_BIAS = 127
FP32_SHIFT_BITS = 23
FP32_MANTISSA_MASK = 0x7FFFFF
E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

N_HEADS = 32
D = 128
DR = 64
ROW_CHUNK = 8192
SOFTMAX_SCALE = 0.08838834764831845

P0_CASES = (
    ("t1", 1), ("t4", 4), ("t6", 6), ("t8", 8), ("t12", 12), ("t16", 16),
    ("t24", 24), ("t32", 32), ("t48", 48), ("t72", 72), ("t96", 96),
    ("t192", 192), ("t1024", 1024), ("t2048", 2048), ("t4096", 4096),
    ("t8192", 8192), ("t16384", 16384), ("t32768", 32768), ("t65536", 65536),
    ("t131072", 131072), ("t262144", 262144),
)


def _ceil_div(value, divisor):
    """计算 ``value`` 除以 ``divisor`` 的向上取整商。

    参数：``value``/``divisor`` 为非负整数。

    数据流：纯标量工具函数，多处用于把 K/D 长度换算成块数。
    """
    return (value + divisor - 1) // divisor


def _e4m3_lut(device):
    """构造 E4M3 字节 -> FP32 的 256 项解码 LUT。

    职责：把所有 256 个 uint8 位型按 ``float8_e4m3fn`` 解释后转 FP32，供
    ``lut[qr]`` 之类的查表解码复用。

    参数：``device`` 指定返回张量所在设备。

    数据流：返回 FP32[256]；供 ``indexer_prologue_qw_golden_cpu`` 解码 qr/wqb。
    """
    return torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().to(device)


def _decode_e8m0_scale(scale_bytes, outer_size, k):
    """把公开的 [outer, G, 2] E8M0 字节解码成每 32 个元素一个的 FP32 scale。

    职责：取每组有效的前 ``ceil(k/32)`` 个 E8M0 指数，按 ``2**(byte-127)``
    还原为 FP32。

    参数：
        scale_bytes (uint8[outer, G, 2])：paired E8M0 字节（每 64 个 K 用 2 字节）。
        outer_size (int)：外层行数。
        k (int)：K 维长度，用于确定有效组数。

    数据流：返回 FP32[outer, ceil(k/32)]，供 ``repeat_interleave`` 展开到
        每个元素后与解码后的权重相乘。
    """
    valid = _ceil_div(k, MX_GROUP_SIZE)
    exponent_bytes = scale_bytes.contiguous().view(torch.uint8).reshape(outer_size, -1)
    exponents = exponent_bytes[:, :valid].to(torch.int16) - E8M0_BIAS
    return torch.pow(2.0, exponents.float())


def _fp32_scale_to_e8m0(scale):
    """把 FP32 scale 向上取整到下一个 2 的幂，并以 IEEE 指数字节返回。

    职责：取 FP32 的指数字段；若尾数非零则指数 +1（即向上取整），非正数记为 0。

    参数：``scale`` 为 FP32 张量。

    数据流：返回 uint8 的 E8M0 指数字节，供 descale_q 输出与量化除 scale。
    """
    bits = scale.to(torch.float32).contiguous().view(torch.int32).to(torch.int64)
    exp_bits = (bits >> FP32_SHIFT_BITS) & 0xFF
    mantissa = bits & FP32_MANTISSA_MASK
    exp_bits = torch.where(mantissa != 0, (exp_bits + 1) & 0xFF, exp_bits)
    zero = scale <= 0
    return torch.where(zero, torch.zeros_like(exp_bits), exp_bits).to(torch.uint8)


def _rotate_half(x):
    """rotate-half：对最后一维取前后半并返回 ``concat(-x2, x1)``。

    参数：``x`` 最后一维长度为偶数，``half = 最后一维/2``。

    数据流：RoPE 内部辅助；在 ``_apply_rope_inplace`` 中与 ``pe`` 组合。
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_inplace(q, rope_sin, rope_cos):
    """对 D 的最后 Dr 列做就地 rotate-half RoPE。q 形状 (T, N, D)。

    职责：``q[..., -Dr:] = pe * cos + rotate_half(pe) * sin``，就地改写。

    参数：
        q (FP32[T, N, D])：待旋转的 Q（会被就地修改并返回）。
        rope_sin / rope_cos (FP32[T, Dr])：sin/cos，内部 unsqueeze 到 (T,1,Dr)。

    数据流：RoPE 在 golden 主循环内紧跟 GEMM reshape 之后调用。
    """
    dr = int(rope_sin.shape[-1])
    if dr == 0:
        return q
    pe = q[..., -dr:]
    cos = rope_cos.to(torch.float32).unsqueeze(1)
    sin = rope_sin.to(torch.float32).unsqueeze(1)
    q[..., -dr:] = pe * cos + _rotate_half(pe) * sin
    return q


def _fp32_to_e2m1_nibble(values, codebook, arange8):
    """把 FP32 舍入到最近的 OCP E2M1 code；.5 处远离零取整。

    参数：
        values (FP32[...])：待量化的幅值或带符号值。
        codebook (FP32[8])：E2M1 正值码表 ``E2M1_POS``。
        arange8 (int64[8])：0..7 的索引，用于在并列时取较大 code。

    数据流：返回 uint8 nibble（含符号位），供 ``_pack_e2m1_nibbles`` 打包。
    """
    sign = values < 0
    mag = values.abs()
    dist = (mag.unsqueeze(-1) - codebook).abs()
    min_dist = dist.min(dim=-1).values.unsqueeze(-1)
    tied = dist == min_dist
    idx = torch.where(tied, arange8, torch.full_like(arange8, -1)).max(dim=-1).values
    nibble = idx.to(torch.uint8)
    return torch.where(sign, nibble | 0x08, nibble)


def _pack_e2m1_nibbles(nibbles):
    """把 (..., D) 个 nibble 打包成 (..., D/2) 字节：偶数列低、奇数列高。

    参数：``nibbles`` 最后一维为偶数。

    数据流：返回 uint8，与算子输出 ``q`` 的字节布局一致。
    """
    low = nibbles[..., 0::2].to(torch.int16)
    high = nibbles[..., 1::2].to(torch.int16)
    return (low | (high << 4)).to(torch.uint8).contiguous()


def _mx_quant_along_last(x, codebook, arange8):
    """把 (T, N, D) 的 FP32 沿 D 做 MXFP4 量化。

    职责：按每 32 列一组求 ``amax``，算 E8M0 向上取整 scale，除 scale 后取
    E2M1 nibble 并打包；同时把每组 E8M0 指数整理成 (T, N, ceil(D/64), 2)
    的 descale（不足补 127）。

    参数：
        x (FP32[T, N, D])：待量化张量。
        codebook (FP32[8]) / arange8 (int64[8])：传给 ``_fp32_to_e2m1_nibble``。

    数据流：返回 ``(q uint8[T,N,D/2], descale uint8[T,N,ceil(D/64),2])``。
    """
    t, n, d = x.shape
    n_groups = _ceil_div(d, MX_GROUP_SIZE)
    pad = n_groups * MX_GROUP_SIZE - d
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    grouped = x.reshape(t, n, n_groups, MX_GROUP_SIZE)
    amax = grouped.abs().amax(dim=-1).clamp_min(0.0)
    raw_scale = torch.where(amax > 0, amax / FP4_E2M1_MAX, torch.ones_like(amax))
    e8m0 = _fp32_scale_to_e8m0(raw_scale)
    decoded = torch.pow(2.0, e8m0.to(torch.float32) - E8M0_BIAS)
    q_fp32 = grouped / decoded.unsqueeze(-1)
    nibbles = _fp32_to_e2m1_nibble(
        q_fp32.reshape(t, n, n_groups * MX_GROUP_SIZE)[..., :d], codebook, arange8
    )
    q = _pack_e2m1_nibbles(nibbles)

    pair_groups = _ceil_div(d, MX_K_ALIGN)
    pair_elems = pair_groups * MX_SCALE_PAIR
    if n_groups < pair_elems:
        e8m0 = torch.nn.functional.pad(e8m0, (0, pair_elems - n_groups), value=E8M0_BIAS)
    descale = e8m0.reshape(t, n, pair_groups, MX_SCALE_PAIR).contiguous()
    return q, descale


def indexer_prologue_qw_golden_cpu(
    x: torch.Tensor,
    qr: torch.Tensor,
    wqb: torch.Tensor,
    ww: torch.Tensor,
    descale_qr: torch.Tensor,
    descale_wqb: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    softmax_scale: float = SOFTMAX_SCALE,
    return_q_fp32: bool = False,
):
    """纯 torch CPU golden：返回 ``(q uint8, descale_q uint8, w float32)``。

    职责：先把 qr/wqb 解码到 FP32 并乘块 scale，再按 ``ROW_CHUNK=8192`` 行分块
    做 GEMM、RoPE 与 MXFP4 量化；W 路径按同一分块计算
    ``softmax_scale * (x @ ww^T)``。整个计算在 ``qr.device`` 上完成。

    参数：
        x (BF16[T, DIM])：W 路径输入。
        qr (uint8 或 E4M3[T, QLORA])：Q 左操作数，按字节查 LUT 解码。
        wqb (uint8 或 E4M3[4096, QLORA])：Q 右操作数。
        ww (BF16[N_HEADS, DIM])：W 权重。
        descale_qr (uint8[T, 20, 2]) / descale_wqb (uint8[4096, 20, 2])：块 scale。
        rope_sin / rope_cos (FP32[T, DR])：RoPE 的 sin/cos。
        softmax_scale (float)：W 输出的缩放系数，默认 ``SOFTMAX_SCALE``。
        return_q_fp32 (bool)：为真时额外返回量化前的 post-RoPE FP32 Q，供精度
            比较器做 E2M1 舍入边界的自包含判定（默认关闭，保持原 ABI）。

    数据流：输出与输入同设备；供 validator 逐块比较算子输出。

    前置条件：``wqb``/``ww`` 按 ND 传入（本函数只做数学，不要求 NZ）。
    """
    device = qr.device
    lut = _e4m3_lut(device)
    codebook = torch.tensor(E2M1_POS, dtype=torch.float32, device=device)
    arange8 = torch.arange(8, device=device, dtype=torch.int64)

    t = qr.shape[0]
    k = qr.shape[1]
    n_heads = ww.shape[0]
    d = wqb.shape[0] // n_heads

    wqb_deq = lut[wqb.long()] * _decode_e8m0_scale(
        descale_wqb, wqb.shape[0], k
    ).repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :k]
    wqb_deq = wqb_deq.contiguous()
    x_f = x.to(torch.float32)
    ww_f = ww.to(torch.float32)

    q_parts, dq_parts, w_parts, qf_parts = [], [], [], []
    for start in range(0, t, ROW_CHUNK):
        stop = min(t, start + ROW_CHUNK)
        chunk = stop - start
        lhs = lut[qr[start:stop].long()] * _decode_e8m0_scale(
            descale_qr[start:stop], chunk, k
        ).repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :k]
        y = lhs.contiguous() @ wqb_deq.transpose(0, 1)
        q_fp32 = y.reshape(chunk, n_heads, d).contiguous()
        _apply_rope_inplace(q_fp32, rope_sin[start:stop], rope_cos[start:stop])
        qc, dqc = _mx_quant_along_last(q_fp32, codebook, arange8)
        q_parts.append(qc)
        dq_parts.append(dqc)
        w_parts.append(softmax_scale * (x_f[start:stop] @ ww_f.transpose(0, 1)))
        if return_q_fp32:
            qf_parts.append(q_fp32)

    q = q_parts[0] if len(q_parts) == 1 else torch.cat(q_parts, dim=0)
    descale_q = dq_parts[0] if len(dq_parts) == 1 else torch.cat(dq_parts, dim=0)
    w = w_parts[0] if len(w_parts) == 1 else torch.cat(w_parts, dim=0)
    q = q.to(torch.uint8)
    descale_q = descale_q.to(torch.uint8)
    w = w.to(torch.float32)
    if return_q_fp32:
        q_fp32 = qf_parts[0] if len(qf_parts) == 1 else torch.cat(qf_parts, dim=0)
        return q, descale_q, w, q_fp32
    return q, descale_q, w


def _make_paired_scale(outer_size, k, device):
    """构造随机的 paired E8M0 块 scale，用于输入构造。

    职责：按每 64 个 K 生成 2 个 E8M0 字节，取值范围
    ``E8M0_BIAS-2 .. E8M0_BIAS+2``。

    参数：
        outer_size (int)：外层行数（T 或 4096）。
        k (int)：K 维长度，决定组数 ``ceil(k/64)``。
        device：返回张量所在设备。

    数据流：返回 uint8[outer_size, ceil(k/64), 2]，作为 ``descale_*`` 输入。
    """
    groups = _ceil_div(k, MX_K_ALIGN)
    exponents = torch.randint(
        0, 5, (outer_size, groups * MX_SCALE_PAIR), dtype=torch.int16, device=device
    )
    return (
        (exponents + (E8M0_BIAS - 2))
        .to(torch.uint8)
        .reshape(outer_size, groups, MX_SCALE_PAIR)
        .contiguous()
    )


def make_case(t, device, seed):
    """按固定几何构造一个 SPEC P0 case 的全部输入张量。

    职责：固定随机种子，生成 qr/wqb（E4M3 字节视图）、x/ww（BF16）、
    descale_qr/descale_wqb（paired E8M0），以及由 theta 的 cos/sin 得到的
    rope_cos/rope_sin。

    参数：
        t (int)：该 case 的 ``T``。
        device：张量所在设备。
        seed (int)：``torch.manual_seed`` 的种子。

    数据流：返回 ``(args, kwargs)``，``args`` 为 8 个位置张量
        ``[x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin, rope_cos]``，
        ``kwargs = {"softmax_scale": SOFTMAX_SCALE}``；供 ``_make_inputs`` 与
        validator 复用。
    """
    torch.manual_seed(seed)
    qr = (
        (torch.randn((t, 1280), dtype=torch.float32, device=device) * 0.5)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    wqb = (
        (torch.randn((4096, 1280), dtype=torch.float32, device=device) * 0.5)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    x = torch.randn((t, 5120), dtype=torch.float32, device=device).to(torch.bfloat16)
    ww = torch.randn((N_HEADS, 5120), dtype=torch.float32, device=device).to(torch.bfloat16)
    descale_qr = _make_paired_scale(t, 1280, device)
    descale_wqb = _make_paired_scale(4096, 1280, device)
    theta = torch.rand((t, DR), dtype=torch.float32, device=device) * 2.0 - 1.0
    rope_cos = torch.cos(theta)
    rope_sin = torch.sin(theta)
    return (
        [x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin, rope_cos],
        {"softmax_scale": SOFTMAX_SCALE},
    )


def _make_inputs(device=torch.device("cpu")):
    """返回全部 SPEC P0，形式为 ``(case_name, 位置张量, 标量 kwargs)``。

    职责：遍历 ``P0_CASES``，对第 ``i`` 个 case 用 ``seed=i+1`` 调 ``make_case``。

    参数：``device`` 指定生成张量所在设备，默认 CPU。

    数据流：返回 case 列表，供 ``_validate`` 与外部 validator 消费。
    """
    cases = []
    for index, (name, t) in enumerate(P0_CASES):
        args, kwargs = make_case(t, device, seed=index + 1)
        cases.append((name, args, kwargs))
    return cases


def _validate():
    """检查每个 P0 的输出 shape/dtype 与有限性。

    职责：在 CPU 上跑完全部 ``P0_CASES``，校验 ``q``/``descale_q``/``w`` 的
    形状、dtype 正确且 ``w`` 无 NaN/Inf，并记录逐 case 报告。

    参数：无。

    数据流：不接外部输入；通过返回 True，任一校验失败即抛 ``AssertionError``。
    """
    logger.info("%s", "=" * 60)
    logger.info("indexer_prologue_qw CPU golden validation report")
    logger.info("%s", "=" * 60)
    cases = _make_inputs(torch.device("cpu"))
    expected = dict(P0_CASES)
    observed = [name for name, _, _ in cases]
    if observed != list(expected):
        raise AssertionError(f"case names mismatch: {observed}")
    for name, args, kwargs in cases:
        t = expected[name]
        q, descale_q, w = indexer_prologue_qw_golden_cpu(*args, **kwargs)
        if tuple(q.shape) != (t, N_HEADS, D // 2):
            raise AssertionError(f"{name} q shape {tuple(q.shape)}")
        if tuple(descale_q.shape) != (t, N_HEADS, 2, 2):
            raise AssertionError(f"{name} descale_q shape {tuple(descale_q.shape)}")
        if tuple(w.shape) != (t, N_HEADS):
            raise AssertionError(f"{name} w shape {tuple(w.shape)}")
        if not (q.dtype == torch.uint8 and descale_q.dtype == torch.uint8
                and w.dtype == torch.float32):
            raise AssertionError(f"{name} dtype mismatch")
        if not torch.isfinite(w).all().item():
            raise AssertionError(f"{name} w not finite")
        logger.info("[%s] q%s descale_q%s w%s PASS", name, tuple(q.shape),
                    tuple(descale_q.shape), tuple(w.shape))
    logger.info("%s", "=" * 60)
    logger.info("validation complete")
    logger.info("%s", "=" * 60)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(0 if _validate() else 1)
