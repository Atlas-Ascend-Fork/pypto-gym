# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""quant_mla_prolog 精度判据。

判据限值与 DSL 参考实现的验收契约一致。本文件不 import NPU 实现，
两个参考文件都能脱离实现使用。

__all__ = ["check_precision", "REL_L2_LIMIT", "Q_COSINE_MIN"]
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

REL_L2_LIMIT = 3.0e-2
Q_COSINE_MIN = 0.999


def _rel_l2(actual, expected):
    """相对 L2，只把累加抬到 double。"""
    a, e = actual.double().reshape(-1), expected.double().reshape(-1)
    return float(torch.linalg.vector_norm(a - e)
                 / torch.linalg.vector_norm(e).clamp_min(1e-12))


def _cosine(actual, expected):
    """余弦相似度。"""
    a, e = actual.double().reshape(-1), expected.double().reshape(-1)
    return float((a @ e) / (a.norm() * e.norm()).clamp_min(1e-12))


def _cmp_kv_cache_out(got_kv, want_kv, slots, qs_kv, page, label):
    """写入行比反量化相对 L2，未写入行比逐 bit。"""
    ga8 = got_kv.cpu().contiguous().view(torch.uint8)
    wa8 = want_kv.contiguous().view(torch.uint8)

    def _gather(cache):
        rows = [cache[s // page, :, s % page, :].float().reshape(-1)
                for s in slots if s >= 0]
        return torch.stack(rows) if rows else torch.zeros(1)

    kv_a, kv_e = _gather(got_kv.cpu()) * qs_kv, _gather(want_kv) * qs_kv
    finite = bool(torch.isfinite(kv_a).all() and torch.isfinite(kv_e).all())
    l2 = _rel_l2(kv_a, kv_e)

    touched = {}
    for s in slots:
        if s >= 0:
            touched.setdefault(s // page, set()).add(s % page)
    untouched_ok = True
    for pg in range(ga8.shape[0]):
        keep = [r for r in range(page) if r not in touched.get(pg, set())]
        if keep and not torch.equal(ga8[pg, :, keep, :], wa8[pg, :, keep, :]):
            untouched_ok = False

    nbad = int((ga8.reshape(-1) != wa8.reshape(-1)).sum())
    ok = finite and l2 < REL_L2_LIMIT and untouched_ok
    if not ok:
        raise AssertionError(
            f"{label} kv_cache_out 不达标：写入行 rel_L2={l2:.3e}"
            f"（限 {REL_L2_LIMIT:g}）、"
            f"未写入行逐 bit={'一致' if untouched_ok else '不一致'}、"
            f"有限性={finite}、[诊断] 编码态不同字节={nbad}/{ga8.numel()}")
    return ok


def _cmp_descale_q(got_dsq, want_dsq, c1, label):
    """C1 比相对 L2；C0 只核两侧键都缺席。返回 (gd, wd) 供 q 段反量化。"""
    gd = wd = None
    if c1:
        gd, wd = got_dsq.cpu().float(), want_dsq.float()
        finite = bool(torch.isfinite(gd).all() and torch.isfinite(wd).all())
        l2_dsq = _rel_l2(gd, wd)
        ok = finite and l2_dsq < REL_L2_LIMIT
    else:
        ok = got_dsq is None and want_dsq is None
    if not ok:
        raise AssertionError(
            f"{label} descale_q 不达标："
            + (f"rel_L2={l2_dsq:.3e}（限 {REL_L2_LIMIT:g}）、有限性={finite}"
               if c1 else "C0 应无 descale_q 键"))
    return gd, wd


def _cmp_q(got_q, want_q, gd, wd, c1, label):
    """C1 乘回 descale_q 反量化后比相对 L2 与余弦；C0 直接比。"""
    if c1:
        deq = got_q.cpu().float() * gd.unsqueeze(-1)
        wdeq = want_q.float() * wd.unsqueeze(-1)
    else:
        deq, wdeq = got_q.cpu().float(), want_q.float()
    finite = bool(torch.isfinite(deq).all() and torch.isfinite(wdeq).all())
    l2_q, cos_q = _rel_l2(deq, wdeq), _cosine(deq, wdeq)
    ga = got_q.cpu().contiguous().view(torch.uint8).reshape(-1)
    wa = want_q.contiguous().view(torch.uint8).reshape(-1)
    nbad_q = int((ga != wa).sum())
    ok = finite and l2_q < REL_L2_LIMIT and cos_q > Q_COSINE_MIN
    if not ok:
        raise AssertionError(
            f"{label} q 不达标：反量化 rel_L2={l2_q:.3e}"
            f"（限 {REL_L2_LIMIT:g}）、"
            f"余弦={cos_q:.8f}（限 >{Q_COSINE_MIN:g}）、有限性={finite}、"
            f"[诊断] 编码态不同字节={nbad_q}/{ga.numel()}")
    return ok


def check_precision(actual, golden, slots, qscale_kv, quant_mode_c=1,
                    label=""):
    """全链路精度对比：三个终输出的验收契约。

    参数:
        actual: 算子返回的有序 dict（q / kv_cache_out /（C1）descale_q）。
        golden: CPU 真值返回的 dict（同键集，另含诊断中间量）。
        slots: cache_index 取值列表（用于区分写入行和未写入行）。
        qscale_kv: per-tensor 量化系数（C0 时传 1.0）。
        quant_mode_c: 量化模式（0=Bf16, 1=FP8）。
        label: 日志标签。

    C1 时通过与否只由反量化后的指标与有限性决定，编码态 FP8 的
    mismatch 不参与判定 —— 相邻 FP8 码的差异与单 head 的 scale 边界
    变化不直接代表模型精度。

      - q（E4M3）→ 乘回 descale_q 反量化后比相对 L2 与余弦。
      - descale_q（FP32）→ 相对 L2。
      - kv_cache_out（E4M3）→ 写入的行按 cache_index 取出、乘回 qscale_kv
        反量化后比相对 L2；未写入的行逐 bit 完全不变。

    三者的编码态差异只作为失败消息里的诊断。它对 kv 的写入行固有不可达：
    源头是 `kv_raw` 的 7168 项 FP32 累加次序。

    C0 下没有反量化这一步：q 与 kv_cache_out 都是 BF16 终值，
    直接比 L2 / 余弦；descale_q 键不存在。未写入行仍比逐 bit。
    """
    from quant_mla_prolog_golden_cpu import PA_BLOCK_SIZE as PAGE

    c1 = quant_mode_c == 1
    want_q, want_kv = golden["q"], golden["kv_cache_out"]
    want_dsq = golden.get("descale_q")
    got_q = actual["q"]
    got_kv = actual["kv_cache_out"]
    got_dsq = actual.get("descale_q")


    # ── kv_cache_out：写入行比反量化相对 L2，未写入行比逐 bit ──
    # C0 的 cache 本身就是终值，没有反量化这一步。乘 1.0 让下面的比对代码
    # 两条 mode 共用。
    qs_kv = float(qscale_kv) if c1 else 1.0
    ok_kv = _cmp_kv_cache_out(got_kv, want_kv, slots, qs_kv, PAGE, label)
    gd, wd = _cmp_descale_q(got_dsq, want_dsq, c1, label)
    ok_q = _cmp_q(got_q, want_q, gd, wd, c1, label)
    return ok_kv and ok_q
