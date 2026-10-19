# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
"""engram_gate PyPTO-Pro kernel and host wrapper (single dynamic-axis kernel).

One module-level ``@pl.jit(auto_mutex=True)`` kernel serves any ``t/hc/dim``
(dynamic axes); a fixed UB row width keeps a whole ``(token, hc)`` row resident so
pass 1 (the two RMS reductions and the weighted dot) and pass 2 (the residual
write) read the same UB tile. Every numeric step runs inside the kernel; the host
side only validates shapes/dtypes, allocates the output and launches once.
"""

import math

import torch

import pypto
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.platform import get_platform_info

CHUNK = 8512  # fixed UB row width; exceeds the largest supported row for one scan

# Gate constants. They are compile-time literals inside the VF, so the wrapper
# can only accept these values.
_EPS = 1e-06
_CLAMP_VALUE = 1e-06

S_BF = CHUNK * 2
A_UX = 0
A_UK = A_UX + 3 * S_BF
A_UV = A_UK + 3 * S_BF
A_UW = A_UV + 3 * S_BF
A_UO = A_UW + CHUNK * 4


def _vector_core_count():
    """Vector core count from the PyPTO-Pro platform info (not hard-coded)."""
    info = get_platform_info()
    count = int(info.vector_core_num)
    if count <= 0:
        raise RuntimeError(
            "cannot determine the NPU vector core count from the PyPTO-Pro platform info"
        )
    return count


@pl.vector_function
def engram_gate_vf(ux, uk, uw, uv, uo, dim: pl.DT_INT64):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pall_u32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    m32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    m16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)

    a_x2 = vf.full(0.0, dtype=pl.DT_FP32)
    a_k2 = vf.full(0.0, dtype=pl.DT_FP32)
    a_dot = vf.full(0.0, dtype=pl.DT_FP32)
    n_full = dim // 64
    tail = dim - n_full * 64
    for s in pl.range(0, n_full, 1):
        mul = s * 64
        xr = vf.load_align(ux, mul, dist=pl.LoadDist.UNPK_B16)
        xf = vf.astype(xr, m32, dtype=pl.DT_FP32, round_mode=pl.VFRoundMode.CAST_RINT,
                       saturate=pl.SaturateMode.OFF)
        kr = vf.load_align(uk, mul, dist=pl.LoadDist.UNPK_B16)
        kf = vf.astype(kr, m32, dtype=pl.DT_FP32, round_mode=pl.VFRoundMode.CAST_RINT,
                       saturate=pl.SaturateMode.OFF)
        wf = vf.load_align(uw, mul)
        a_x2 = vf.mul_add_dst(xf, xf, preg)
        a_k2 = vf.mul_add_dst(kf, kf, preg)
        tmp = vf.mul(xf, wf, preg)
        a_dot = vf.mul_add_dst(tmp, kf, preg)
    if tail != 0:
        mt = vf.update_mask(tail, dtype=pl.DT_FP32)
        mul = n_full * 64
        xr = vf.load_align(ux, mul, dist=pl.LoadDist.UNPK_B16)
        xf = vf.astype(xr, m32, dtype=pl.DT_FP32, round_mode=pl.VFRoundMode.CAST_RINT,
                       saturate=pl.SaturateMode.OFF)
        kr = vf.load_align(uk, mul, dist=pl.LoadDist.UNPK_B16)
        kf = vf.astype(kr, m32, dtype=pl.DT_FP32, round_mode=pl.VFRoundMode.CAST_RINT,
                       saturate=pl.SaturateMode.OFF)
        wf = vf.load_align(uw, mul)
        px = vf.mul(xf, xf, mt)
        a_x2 = vf.add(a_x2, px, preg)
        pk = vf.mul(kf, kf, mt)
        a_k2 = vf.add(a_k2, pk, preg)
        tmp = vf.mul(xf, wf, mt)
        pd = vf.mul(tmp, kf, mt)
        a_dot = vf.add(a_dot, pd, preg)

    sx2 = vf.reduce_sum(a_x2, preg)
    sk2 = vf.reduce_sum(a_k2, preg)
    sdot = vf.reduce_sum(a_dot, preg)

    one = vf.full(1.0, dtype=pl.DT_FP32)
    dimf = vf.full(dim, preg, dtype=pl.DT_FP32)
    inv_dim = vf.div(one, dimf, preg)
    sqrt_dim = vf.sqrt(dimf, preg)
    inv_sqrt_dim = vf.div(one, sqrt_dim, preg)
    xrms = vf.mul(sx2, inv_dim, preg)
    xrms = vf.adds(xrms, _EPS, preg)
    xrms = vf.sqrt(xrms, preg)
    krms = vf.mul(sk2, inv_dim, preg)
    krms = vf.adds(krms, _EPS, preg)
    krms = vf.sqrt(krms, preg)
    t1 = vf.mul(sdot, inv_sqrt_dim, preg)
    t2 = vf.mul(xrms, krms, preg)
    nd = vf.div(t1, t2, preg)
    absr = vf.abs(nd, preg)
    absr = vf.maxs(absr, _CLAMP_VALUE, preg)
    mag = vf.sqrt(absr, preg)
    neg = vf.lt(nd, 0.0, preg)
    nmag = vf.muls(mag, -1.0, preg)
    sr = vf.select(nmag, mag, neg)
    nsr = vf.muls(sr, -1.0, preg)
    ex = vf.exp(nsr, preg)
    den = vf.add(one, ex, preg)
    gate = vf.div(one, den, preg)
    gate = vf.full(gate, preg)

    for s in pl.range(0, n_full, 1):
        mul = s * 64
        ox = vf.load_align(ux, mul, dist=pl.LoadDist.UNPK_B16)
        xf2 = vf.astype(ox, m32, dtype=pl.DT_FP32, round_mode=pl.VFRoundMode.CAST_RINT,
                        saturate=pl.SaturateMode.OFF)
        ov = vf.load_align(uv, mul, dist=pl.LoadDist.UNPK_B16)
        vf2 = vf.astype(ov, m32, dtype=pl.DT_FP32, round_mode=pl.VFRoundMode.CAST_RINT,
                        saturate=pl.SaturateMode.OFF)
        pr = vf.mul(gate, vf2, preg)
        res = vf.add(xf2, pr, preg)
        ob = vf.astype(res, m16, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT,
                       saturate=pl.SaturateMode.OFF)
        ob_pk32 = vf.bit_cast(ob, dtype=pl.DT_UINT32)
        vf.store_align(uo, ob_pk32, pall_u32, mul, dist=pl.StoreDist.PACK)
    if tail != 0:
        mt = vf.update_mask(tail, dtype=pl.DT_FP32)
        mul = n_full * 64
        ox = vf.load_align(ux, mul, dist=pl.LoadDist.UNPK_B16)
        xf2 = vf.astype(ox, m32, dtype=pl.DT_FP32, round_mode=pl.VFRoundMode.CAST_RINT,
                        saturate=pl.SaturateMode.OFF)
        ov = vf.load_align(uv, mul, dist=pl.LoadDist.UNPK_B16)
        vf2 = vf.astype(ov, m32, dtype=pl.DT_FP32, round_mode=pl.VFRoundMode.CAST_RINT,
                        saturate=pl.SaturateMode.OFF)
        pr = vf.mul(gate, vf2, mt)
        res = vf.add(xf2, pr, mt)
        ob = vf.astype(res, m16, dtype=pl.DT_BF16, round_mode=pl.VFRoundMode.CAST_RINT,
                       saturate=pl.SaturateMode.OFF)
        ob_pk32 = vf.bit_cast(ob, dtype=pl.DT_UINT32)
        vf.store_align(uo, ob_pk32, pall_u32, mul, dist=pl.StoreDist.PACK)


@pl.jit(auto_mutex=True)
def engram_gate_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    key: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    value: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    weight: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    image_mask: pl.Tensor[[pl.DYNAMIC], pl.DT_BOOL],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    with pl.section_vector():
        tt = pl.TileType(shape=[1, CHUNK], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
        wt = pl.TileType(shape=[1, CHUNK], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        uw = pl.make_tile_group(type=wt, addrs=[A_UW], mutex_ids=[9], depth=1)[0]
        ux_grp = pl.make_tile_group(type=tt, addrs=[A_UX, A_UX + S_BF, A_UX + 2 * S_BF],
                                    mutex_ids=[0, 1, 2], depth=3)
        uk_grp = pl.make_tile_group(type=tt, addrs=[A_UK, A_UK + S_BF, A_UK + 2 * S_BF],
                                    mutex_ids=[3, 4, 5], depth=3)
        uv_grp = pl.make_tile_group(type=tt, addrs=[A_UV, A_UV + S_BF, A_UV + 2 * S_BF],
                                    mutex_ids=[6, 7, 8], depth=3)
        uo_grp = pl.make_tile_group(type=tt, addrs=[A_UO, A_UO + S_BF, A_UO + 2 * S_BF],
                                    mutex_ids=[10, 11, 12], depth=3)

        hc = x.shape[1]
        dim = x.shape[2]
        core = pl.get_block_idx()
        nc = pl.get_block_num()
        wcore = core % hc
        dtok = nc // hc
        pl.set_validshape(uw, [1, dim])
        pl.load(uw, weight, [wcore, 0])
        slot = 0
        for token in pl.range(core // hc, x.shape[0], dtok):
            flag = pl.getval(image_mask, token)
            si = slot % 3
            if flag != 0:
                pl.set_validshape(ux_grp[si], [1, dim])
                pl.load(ux_grp[si], x, [token, wcore, 0])
                pl.store(y, ux_grp[si], [token, wcore, 0])
            else:
                pl.set_validshape(ux_grp[si], [1, dim])
                pl.load(ux_grp[si], x, [token, wcore, 0])
                pl.set_validshape(uk_grp[si], [1, dim])
                pl.load(uk_grp[si], key, [token, wcore, 0])
                pl.set_validshape(uv_grp[si], [1, dim])
                pl.load(uv_grp[si], value, [token, 0])
                engram_gate_vf(ux_grp[si], uk_grp[si], uw, uv_grp[si], uo_grp[si], dim)
                pl.set_validshape(uo_grp[si], [1, dim])
                pl.store(y, uo_grp[si], [token, wcore, 0])
            slot = slot + 1


def engram_gate_wrapper(x, key, value, weight, image_mask, eps=_EPS, clamp_value=_CLAMP_VALUE):
    """Public wrapper: validate, allocate the output and launch the kernel once."""
    t, hc, dim = x.shape
    if list(key.shape) != [t, hc, dim] or list(weight.shape) != [hc, dim]:
        raise ValueError("key/weight shapes do not match x")
    if list(value.shape) != [t, dim] or list(image_mask.shape) != [t]:
        raise ValueError("value/image_mask shapes do not match x")
    if dim > CHUNK:
        raise ValueError(f"dim {dim} exceeds the supported single-scan width {CHUNK}")
    if not math.isclose(eps, _EPS) or not math.isclose(clamp_value, _CLAMP_VALUE):
        raise ValueError(
            f"this kernel bakes eps={_EPS} and clamp_value={_CLAMP_VALUE}, "
            f"got eps={eps} and clamp_value={clamp_value}")
    y = torch.empty((t, hc, dim), dtype=torch.bfloat16, device=x.device)
    # The kernel strides tokens by nc // hc, so the launch width must be a multiple
    # of hc; for every supported shape this leaves the launched core count unchanged.
    block_dim = min(_vector_core_count(), t * hc)
    block_dim -= block_dim % hc
    with pypto.options(pass_options={"enable_slice": False}):
        engram_gate_kernel[None, block_dim](x, key, value, weight, image_mask, y)
    return y
