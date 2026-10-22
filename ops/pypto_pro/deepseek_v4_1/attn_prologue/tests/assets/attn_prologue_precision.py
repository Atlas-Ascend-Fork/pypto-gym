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

"""attn_prologue 的自包含精度比较器（FP64 值域裁决口径，CPU 可运行）。

职责：把 attn_prologue 验收冻结的判定规则固化成仓库内可复用的比较模块，使算子
验收不依赖任何外部绝对路径。数学 oracle 是本包的 ``attn_prologue_golden_cpu``
（纯 torch 的独立参考）；本模块只负责判定「哪些字节差异是 FP32 累加次序 / E4M3
量化边界本身无法判定的」，其余一律硬失配。本模块不导入 torch_npu / pypto。

判定规则（冻结口径，criterion=``"fp64_value_domain"``）：

* **q**（bf16，容差类）：逐元素 ``|got-ref| > 0.02 + 0.02*|ref|`` 或非有限记
  bad；通过要求 ``q_bad / q_elements <= 1e-3`` 且 ``q_nonfinite == 0``；
  ``q_max_abs`` / ``q_max_rel`` 仅作诊断。
* **qr / descale_qr / KV 命中 slot**（字节精确类）：payload 字节差异先做 FP32
  归因（``attribute_e4m3_mismatch``，``boundary_rel=1e-5`` 冻结）；scale 字节
  单独移动的行经 ``groups_needing_review`` 显式提名；命中行**整行**用 FP64 影子
  重算（``FP64Shadow.recompute_full``）后按值域裁决为 value_identical /
  adjacent_bracket / own_scale_consistent / hard；通过要求 ``qr_hard == 0`` 且
  ``kv_hard == 0``。FP64 裁决本身无 epsilon。
* **未命中 cache slot**：与随机填充的初始字节逐字节一致，``untouched_bad == 0``，
  零容忍。

``validate_outputs`` 是完整校验入口（硬性要求 FP64 shadow，行分块由
``golden_chunks``/``row_chunk`` 提供——所有阶段逐 token 独立，行可任意切分）；
``compare`` 是在其上抛 ``AssertionError(stats.output())`` 的门面。
``test_fp8_value_criterion`` / ``test_full_output_validator`` 是判据本身的两个
CPU 自测（对应源 harness 的 pytest 参数化用例，参数集冻结）。
"""

from __future__ import annotations

__all__ = [
    "BOUNDARY_REL",
    "CRITERION",
    "Comparison",
    "E4M3Attribution",
    "FP64Shadow",
    "Q_ATOL",
    "Q_MAX_BAD_RATE",
    "Q_RTOL",
    "Verdict",
    "adjudicate",
    "attribute_e4m3_mismatch",
    "compare",
    "compare_chunk",
    "decode_payload",
    "encode_at_scale",
    "expand_scale",
    "groups_needing_review",
    "representable_between",
    "test_fp8_value_criterion",
    "test_full_output_validator",
    "validate_outputs",
]

import dataclasses
import logging

import torch

from attn_prologue_golden_cpu import (
    E4M3FN_MAX as E4M3_MAX,
    E8M0_BIAS,
    MX_GROUP_SIZE as MX_GROUP,
    SMALL_PROFILE,
    Case,
    SharedWeights,
    apply_tail_rope,
    attn_prologue_golden,
    dequantize_mxfp8,
    golden_chunks,
    make_case_inputs,
    quantize_mxfp8,
    rms_norm,
)

logger = logging.getLogger(__name__)

#: Frozen acceptance thresholds (recorded with every comparison result). The
#: adjudication code below keeps the original literals verbatim; these constants
#: exist so test entries can record provenance without retyping them.
Q_ATOL = 0.02
Q_RTOL = 0.02
Q_MAX_BAD_RATE = 0.001
CRITERION = "fp64_value_domain"
#: Frozen tie-break window of the FP32 first pass and the FP64 shadow review.
BOUNDARY_REL = 1e-5


# --------------------------------------------------------------------------
# E4M3 失配归因（03 文档 §4.3）
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class E4M3Attribution:
    """把 E4M3 字节失配拆成"量化边界 tie-break"与"真失配"。"""

    total: int
    mismatched: int
    tie_break: int
    hard: int
    hard_index: tuple[torch.Tensor, ...]
    max_code_distance: int

    @property
    def tie_break_ratio(self) -> float:
        return self.tie_break / self.total if self.total else 0.0

    def report(self) -> str:
        lines = [
            f"elements={self.total} mismatched={self.mismatched} "
            f"tie_break={self.tie_break} ({self.tie_break_ratio:.3e}) hard={self.hard}",
            f"max_code_distance={self.max_code_distance}",
        ]
        if self.hard:
            head = [
                tuple(int(axis[i]) for axis in self.hard_index)
                for i in range(min(8, self.hard))
            ]
            lines.append(f"first hard mismatches at {head}")
        return "\n".join(lines)


def _decode_e4m3_bytes(raw: torch.Tensor) -> torch.Tensor:
    """uint8 字节 → FP32 数值（零拷贝 reinterpret）。"""
    return raw.contiguous().view(torch.float8_e4m3fn).to(torch.float32)


def attribute_e4m3_mismatch(
    got_bytes: torch.Tensor,
    expected_bytes: torch.Tensor,
    expected_value: torch.Tensor,
    *,
    boundary_rel: float = 1e-5,
) -> E4M3Attribution:
    """归因 E4M3 payload 的逐字节失配。

    独立 golden **无法**保证 `qr` / `kv_cache` 逐位相等：Cube 按 BK 分块做
    FP32 累加，torch 是单次归约，相对差约 1e-6。这个差要穿过只有 3 位尾数
    的 E4M3 量化，在 round-to-nearest 的判定边界上就会翻到相邻编码。

    一个失配被判为合法 tie-break 需同时满足：
      1. ``expected_value`` 落在两个编码解码值之间（闭区间）——即 got 是
         朝着真值方向的**相邻**编码，不是随机字节；
      2. ``expected_value`` 到两编码中点的相对距离 ≤ ``boundary_rel``——即
         真值本来就贴在判定边界上。

    ``expected_value`` 必须是**实际被 cast 的值**：`qr` 传
    ``trace.qr_normalized``，`kv_cache` 传 ``trace.kv_roped``。

    ``hard_index`` 非空即真 bug，不是浮点问题。
    """
    if got_bytes.shape != expected_bytes.shape:
        raise ValueError("byte tensors must have the same shape")
    if got_bytes.shape != expected_value.shape:
        raise ValueError("expected_value must have the same shape as the bytes")
    got_bytes = got_bytes.cpu().contiguous()
    expected_bytes = expected_bytes.cpu().contiguous()
    reference = expected_value.detach().cpu().to(torch.float32)

    total = int(got_bytes.numel())
    diff_mask = got_bytes != expected_bytes
    mismatched = int(diff_mask.sum())
    if mismatched == 0:
        return E4M3Attribution(total, 0, 0, 0, tuple(), 0)

    index = torch.nonzero(diff_mask, as_tuple=True)
    got_value = _decode_e4m3_bytes(got_bytes)[index]
    exp_value = _decode_e4m3_bytes(expected_bytes)[index]
    truth = reference[index]

    lower = torch.minimum(got_value, exp_value)
    upper = torch.maximum(got_value, exp_value)
    between = (truth >= lower) & (truth <= upper)
    midpoint = (got_value + exp_value) * 0.5
    near = (truth - midpoint).abs() <= boundary_rel * truth.abs().clamp_min(1e-30)
    # NaN 编码（0x7F / 0xFF）永远不是合法 tie-break
    finite = torch.isfinite(got_value) & torch.isfinite(exp_value)
    tie = between & near & finite

    # 编码距离：合法 tie-break 应当是 1；>1 说明差得比"相邻"更远
    code_distance = (got_bytes.to(torch.int16) - expected_bytes.to(torch.int16)).abs()
    max_code_distance = int(code_distance[diff_mask].max())

    hard_mask = ~tie
    hard = int(hard_mask.sum())
    hard_index = tuple(axis[hard_mask] for axis in index)
    return E4M3Attribution(
        total=total,
        mismatched=mismatched,
        tie_break=int(tie.sum()),
        hard=hard,
        hard_index=hard_index,
        max_code_distance=max_code_distance,
    )


def _finite_e4m3_values() -> torch.Tensor:
    """Sorted unique finite E4M3FN values, used as the representable grid.

    ``+0``/``-0`` collapse to one entry, which is what adjacency should see.
    """
    codes = torch.arange(256, dtype=torch.uint8)
    values = codes.view(torch.float8_e4m3fn).to(torch.float64)
    return torch.unique(values[torch.isfinite(values)])


_GRID = _finite_e4m3_values()


def decode_payload(payload: torch.Tensor) -> torch.Tensor:
    """FP8 payload bytes → FP64 values (zero-copy reinterpret, then widen)."""
    return payload.contiguous().view(torch.float8_e4m3fn).to(torch.float64)


def expand_scale(paired: torch.Tensor, groups: int, width: int) -> torch.Tensor:
    """Paired E8M0 ``(rows, ceil(G/2), 2)`` → per-element power-of-two factor.

    The public layout stores consecutive groups in pairs, so a flat view walks
    the groups in order. Element ``col`` belongs to group ``col // 32``.
    """
    flat = paired.contiguous().view(torch.uint8).view(paired.shape[0], -1)[:, :groups]
    factor = torch.ldexp(
        torch.ones_like(flat, dtype=torch.float64), flat.to(torch.int32) - E8M0_BIAS
    )
    return factor.repeat_interleave(MX_GROUP, dim=1)[:, :width]


def representable_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Count representable E4M3 values strictly between ``a`` and ``b``."""
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    first_above = torch.searchsorted(_GRID, lo.contiguous(), right=True)
    first_at = torch.searchsorted(_GRID, hi.contiguous(), right=False)
    return (first_at - first_above).clamp_min(0)


def encode_at_scale(truth: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Encode an FP64 value at a given shared scale, exactly as the op does.

    Divide by the scale, saturate at the E4M3 limit, round to nearest even. Both
    the kernel (``vcast`` with ``RoundingMode.RN``) and the golden
    (``.to(float8_e4m3fn)``) do precisely this, so the result is what a correct
    implementation must emit for that value at that scale.
    """
    normalized = (truth / scale).clamp(-E4M3_MAX, E4M3_MAX)
    return normalized.to(torch.float8_e4m3fn).view(torch.uint8)


class Verdict:
    """Per-element classification counts plus the surviving hard examples."""

    __slots__ = (
        "checked",
        "value_identical",
        "adjacent_bracket",
        "own_scale_consistent",
        "hard",
        "hard_index",
    )

    def __init__(
        self,
        checked,
        value_identical,
        adjacent_bracket,
        own_scale_consistent,
        hard,
        hard_index,
    ):
        self.checked = checked
        self.value_identical = value_identical
        self.adjacent_bracket = adjacent_bracket
        self.own_scale_consistent = own_scale_consistent
        self.hard = hard
        self.hard_index = hard_index

    @property
    def legitimate(self) -> int:
        return self.value_identical + self.adjacent_bracket + self.own_scale_consistent


def adjudicate(got_payload, ref_payload, truth, *, got_scale=None, ref_scale=None):
    """Judge dequantized values against an FP64 truth.

    ``got_payload`` / ``ref_payload`` are FP8 bytes of identical shape.
    ``truth`` is the FP64 value the pair is meant to represent, in the same
    (pre-scale) domain as ``payload * scale``. Scales are per-element
    power-of-two factors, or ``None`` for an output whose scale is fixed at 1
    (the KV cache path).

    Every supplied element is judged, including ones whose payload bytes agree:
    a scale that moved on its own still changes the reconstructed value.
    """
    if got_payload.shape != ref_payload.shape:
        raise ValueError("payload tensors must have the same shape")
    if got_payload.shape != truth.shape:
        raise ValueError("truth must have the same shape as the payloads")

    got_value = decode_payload(got_payload)
    ref_value = decode_payload(ref_payload)
    if got_scale is None:
        got_scale = torch.ones_like(got_value)
    if ref_scale is None:
        ref_scale = torch.ones_like(ref_value)

    dq_got = got_value * got_scale
    dq_ref = ref_value * ref_scale
    truth = truth.to(torch.float64)

    finite = torch.isfinite(dq_got) & torch.isfinite(dq_ref)
    identical = finite & (dq_got == dq_ref)

    # One quantization step apart, and only on a shared scale: two different
    # scales put the values on different grids, where "adjacent" is not defined.
    same_scale = got_scale == ref_scale
    bracket = (truth >= torch.minimum(dq_got, dq_ref)) & (
        truth <= torch.maximum(dq_got, dq_ref)
    )
    adjacent = representable_between(got_value, ref_value) == 0
    step = finite & ~identical & same_scale & adjacent & bracket

    # The two sides can also land in different binades when the group amax sits
    # within FP32 noise of a power of two. With `e = floor(log2(amax)) - 8` the
    # normalized amax lands in [256, 512) while E4M3 stops at 448, so the group
    # saturates whenever that mantissa exceeds 1.75 -- and then the binade the
    # scale picked decides whether it saturates at all, which is why the two
    # reconstructions differ. Neither side is the wrong one: the device is the
    # more accurate of the two as often as not. Accept it when the device
    # payload is exactly what its own scale demands for the FP64 truth, which
    # also covers the saturated endpoint because the encoding clamps.
    own_scale = (
        finite
        & ~identical
        & ~step
        & (
            encode_at_scale(truth, got_scale)
            == got_payload.contiguous().view(torch.uint8)
        )
    )

    hard_mask = ~(identical | step | own_scale)
    return Verdict(
        checked=int(got_payload.numel()),
        value_identical=int(identical.sum()),
        adjacent_bracket=int(step.sum()),
        own_scale_consistent=int(own_scale.sum()),
        hard=int(hard_mask.sum()),
        hard_index=torch.nonzero(hard_mask, as_tuple=False),
    )


def groups_needing_review(got_scale_bytes, ref_scale_bytes, groups):
    """Group indices whose scale byte differs, per row.

    A scale byte that moved while its payload bytes stayed put changes the
    reconstructed value without producing a payload mismatch, so those groups
    have to be pulled into the adjudicated set explicitly.
    """
    got = (
        got_scale_bytes.contiguous()
        .view(torch.uint8)
        .view(got_scale_bytes.shape[0], -1)[:, :groups]
    )
    ref = (
        ref_scale_bytes.contiguous()
        .view(torch.uint8)
        .view(ref_scale_bytes.shape[0], -1)[:, :groups]
    )
    return torch.nonzero(got != ref, as_tuple=False)


class FP64Shadow:
    def __init__(self, weights):
        self.weights = weights
        self._dequantized = {}

    def recompute_rows(self, inputs, rows, output, *, norm_eps=1e-6):
        """Return (FP8 bytes, actual cast values) for complete independent rows."""
        payload, _, cast_values, _ = self.recompute_full(
            inputs, rows, output, norm_eps=norm_eps
        )
        return payload, cast_values

    def recompute_full(self, inputs, rows, output, *, norm_eps=1e-6):
        """Return FP64 (payload, scale, cast values, pre-scale values)."""
        rows = torch.as_tensor(rows, dtype=torch.int64, device="cpu")
        # CPU advanced indexing for float8 is not available in all PyTorch
        # releases. Gather bytes then reinterpret without changing their values.
        x = inputs.x.view(torch.uint8)[rows].contiguous().view(torch.float8_e4m3fn)
        x = dequantize_mxfp8(x, inputs.descale_x[rows], torch.float64)
        if output == "qr":
            projected = x @ self._weight("wqa").T
            normalized = rms_norm(
                projected, inputs.norm_weight_qr, norm_eps, torch.float64
            )
            qr, scale, cast_values = quantize_mxfp8(normalized, torch.float64)
            return qr.view(torch.uint8), scale, cast_values, normalized
        if output == "kv":
            projected = x @ self._weight("wkv").T
            normalized = rms_norm(
                projected, inputs.norm_weight_kv, norm_eps, torch.float64
            )
            roped = apply_tail_rope(
                normalized,
                inputs.rope_sin[rows],
                inputs.rope_cos[rows],
                compute_dtype=torch.float64,
            )
            cast_values = roped.clamp(-448.0, 448.0)
            return (
                cast_values.to(torch.float8_e4m3fn).view(torch.uint8),
                None,
                cast_values,
                cast_values,
            )
        raise ValueError("output must be qr or kv")

    def review(self, inputs, candidates, *, boundary_rel=1e-5, norm_eps=1e-6):
        """Re-evaluate every supplied candidate, preserving original metadata.

        candidates must contain output ('qr'/'kv'), row, column and got_byte;
        may contain FP32 expected_byte and expected_cast_value diagnostics.
        Return exact / adjacent-boundary / hard per candidate, using FP64 values
        throughout the boundary calculation (no FP32 downcast of the reference).
        """
        if boundary_rel != 1e-5:
            raise ValueError("shadow uses the frozen boundary_rel=1e-5")
        result = []
        for output in ("qr", "kv"):
            group = [item for item in candidates if item["output"] == output]
            if not group:
                continue
            rows = sorted({item["row"] for item in group})
            indices = {row: index for index, row in enumerate(rows)}
            expected, values = self.recompute_rows(
                inputs, rows, output, norm_eps=norm_eps
            )
            for item in group:
                index, column = indices[item["row"]], item["column"]
                got_byte, ref_byte = item["got_byte"], int(expected[index, column])
                truth = float(values[index, column])
                decoded = (
                    torch.tensor([got_byte, ref_byte], dtype=torch.uint8)
                    .view(torch.float8_e4m3fn)
                    .double()
                )
                got, ref = float(decoded[0]), float(decoded[1])
                exact = got_byte == ref_byte
                midpoint = (got + ref) * 0.5
                near = abs(truth - midpoint) <= boundary_rel * max(abs(truth), 1e-30)
                adjacent = abs(got_byte - ref_byte) == 1
                finite = bool(torch.isfinite(decoded).all())
                between = min(got, ref) <= truth <= max(got, ref)
                tie = not exact and adjacent and finite and between and near
                result.append(
                    dict(
                        **item,
                        shadow_expected_byte=ref_byte,
                        shadow_cast_value=truth,
                        shadow_classification="exact"
                        if exact
                        else "boundary"
                        if tie
                        else "hard",
                        shadow_midpoint_relative=abs(truth - midpoint)
                        / max(abs(truth), 1e-30),
                        boundary_rel=boundary_rel,
                        diagnostic_only=True,
                    )
                )
        return result

    def _weight(self, name):
        if name not in self._dequantized:
            self._dequantized[name] = dequantize_mxfp8(
                self.weights.fields[name],
                self.weights.fields[f"descale_{name}"],
                torch.float64,
            )
        return self._dequantized[name]


@dataclasses.dataclass
class Comparison:
    q_elements: int = 0
    q_bad: int = 0
    q_nonfinite: int = 0
    q_max_abs: float = 0.0
    q_max_rel: float = 0.0
    scale_elements: int = 0
    scale_bad: int = 0
    qr_elements: int = 0
    qr_mismatched: int = 0
    qr_tie_break: int = 0
    qr_hard: int = 0
    qr_max_code_distance: int = 0
    kv_elements: int = 0
    kv_mismatched: int = 0
    kv_tie_break: int = 0
    kv_hard: int = 0
    kv_max_code_distance: int = 0
    untouched_elements: int = 0
    untouched_bad: int = 0
    hard_examples: list = dataclasses.field(default_factory=list)
    shadow_reviews: list = dataclasses.field(default_factory=list)
    qr_shadow_reviewed: int = 0
    qr_shadow_remaining_hard: int = 0
    kv_shadow_reviewed: int = 0
    kv_shadow_remaining_hard: int = 0
    qr_adjudicated: int = 0
    qr_value_identical: int = 0
    qr_adjacent_bracket: int = 0
    qr_own_scale_consistent: int = 0
    kv_adjudicated: int = 0
    kv_value_identical: int = 0
    kv_adjacent_bracket: int = 0
    kv_own_scale_consistent: int = 0
    scale_groups_reviewed: int = 0

    @property
    def passed(self):
        return (
            self.q_elements > 0
            and self.q_bad / self.q_elements <= 1e-3
            and self.q_nonfinite == 0
            and self.qr_hard == 0
            and self.kv_hard == 0
            and self.untouched_bad == 0
        )

    def output(self):
        return dict(
            **dataclasses.asdict(self),
            q_bad_rate=f"{self.q_bad}/{max(1, self.q_elements)}={self.q_bad / max(1, self.q_elements)}",
            criterion="fp64_value_domain",
            passed=self.passed,
        )


def compare_chunk(stats, got_q, got_qr, got_scales, got_kv, expected, *, start=0):
    candidates = []
    q, ref = got_q.cpu().float(), expected["q"].float()
    diff = (q - ref).abs()
    finite = torch.isfinite(q)
    stats.q_elements += q.numel()
    stats.q_nonfinite += int((~finite).sum())
    stats.q_bad += int(((diff > 0.02 + 0.02 * ref.abs()) | ~finite).sum())
    stats.q_max_abs = max(
        stats.q_max_abs, float(diff.nan_to_num(posinf=float("inf")).max())
    )
    stats.q_max_rel = max(
        stats.q_max_rel, float((diff / ref.abs().clamp_min(1e-6)).nan_to_num().max())
    )
    scales = got_scales.cpu().contiguous().view(torch.uint8)
    stats.scale_elements += scales.numel()
    stats.scale_bad += int((scales != expected["descale_qr"].view(torch.uint8)).sum())
    scale_diff = groups_needing_review(
        scales,
        expected["descale_qr"],
        expected["qr"].shape[1] // MX_GROUP,
    )
    stats.scale_groups_reviewed += int(scale_diff.shape[0])
    for name, got, exp, truth in (
        ("qr", got_qr, expected["qr"], expected["qr_normalized"]),
        ("kv", got_kv, expected["kv_bytes"], expected["kv_roped"]),
    ):
        got = got.cpu().contiguous().view(torch.uint8)
        exp = exp.contiguous().view(torch.uint8)
        attr = attribute_e4m3_mismatch(got, exp, truth, boundary_rel=1e-5)
        stats.__dict__[f"{name}_elements"] += attr.total
        stats.__dict__[f"{name}_mismatched"] += attr.mismatched
        setattr(
            stats,
            f"{name}_max_code_distance",
            max(getattr(stats, f"{name}_max_code_distance"), attr.max_code_distance),
        )
        rows_hit = set(torch.nonzero(got != exp, as_tuple=False)[:, 0].tolist())
        if name == "qr":
            rows_hit |= set(scale_diff[:, 0].tolist())
        for row in sorted(rows_hit):
            candidates.append(dict(output=name, row=start + row, local_row=row))
    return candidates


def _adjudicate_rows(
    stats,
    shadow,
    inputs,
    candidates,
    got_qr,
    got_scales,
    got_kv,
    start,
    *,
    norm_eps=1e-6,
):
    """Adjudicate nominated rows against complete FP64 recomputations."""
    for name in ("qr", "kv"):
        local = sorted({c["local_row"] for c in candidates if c["output"] == name})
        if not local:
            continue
        absolute = [start + row for row in local]
        ref_payload, ref_scale, _, truth = shadow.recompute_full(
            inputs, absolute, name, norm_eps=norm_eps
        )
        if name == "qr":
            device_payload = got_qr.cpu().contiguous().view(torch.uint8)[local]
            groups = device_payload.shape[1] // MX_GROUP
            got_factor = expand_scale(
                got_scales.cpu().contiguous()[local], groups, device_payload.shape[1]
            )
            ref_factor = expand_scale(ref_scale, groups, device_payload.shape[1])
        else:
            device_payload = got_kv.contiguous().view(torch.uint8)[local]
            got_factor = ref_factor = None
        verdict = adjudicate(
            device_payload,
            ref_payload,
            truth,
            got_scale=got_factor,
            ref_scale=ref_factor,
        )
        stats.__dict__[f"{name}_adjudicated"] += verdict.checked
        stats.__dict__[f"{name}_value_identical"] += verdict.value_identical
        stats.__dict__[f"{name}_adjacent_bracket"] += verdict.adjacent_bracket
        stats.__dict__[f"{name}_own_scale_consistent"] += verdict.own_scale_consistent
        stats.__dict__[f"{name}_hard"] += verdict.hard
        stats.__dict__[f"{name}_tie_break"] += verdict.legitimate
        for row, col in verdict.hard_index.tolist():
            if len(stats.hard_examples) >= 16:
                break
            stats.hard_examples.append(
                dict(
                    output=name,
                    row=absolute[row],
                    column=col,
                    got_byte=int(device_payload[row, col]),
                    reference_byte=int(ref_payload[row, col]),
                    fp64_truth=float(truth[row, col]),
                    criterion="fp64_value_domain",
                )
            )


def validate_outputs(
    inputs,
    weights,
    outputs,
    cache_device,
    *,
    row_chunk=256,
    progress=None,
    shadow=None,
    norm_eps=1e-6,
):
    if shadow is None:
        raise ValueError("FP64 reference is required by the value-domain criterion")
    stats = Comparison()
    q, qr, scales = outputs
    d = weights.profile[3]
    # Cache is at most ~128 MiB at requested max T; q remains on device, copied
    # by chunks (~16 MiB for 256 target rows), never as an entire CPU tensor.
    cache = cache_device.cpu().view(-1, d)
    for rows, expected in golden_chunks(
        inputs, weights, row_chunk=row_chunk, norm_eps=norm_eps
    ):
        got_kv = cache[inputs.cache_index[rows]]
        candidates = compare_chunk(
            stats, q[rows], qr[rows], scales[rows], got_kv, expected, start=rows.start
        )
        if candidates:
            _adjudicate_rows(
                stats,
                shadow,
                inputs,
                candidates,
                qr[rows],
                scales[rows],
                got_kv,
                rows.start,
                norm_eps=norm_eps,
            )
        if progress is not None:
            progress(rows.stop, stats)
    untouched = torch.ones(cache.shape[0], dtype=torch.bool)
    untouched[inputs.cache_index] = False
    initial = inputs.kv_cache.view(-1, d)
    stats.untouched_elements = int(untouched.sum()) * d
    stats.untouched_bad = int((cache[untouched] != initial[untouched]).sum())
    return stats


def compare(
    inputs,
    weights,
    outputs,
    cache_device,
    *,
    row_chunk=256,
    progress=None,
    shadow=None,
    norm_eps=1e-6,
):
    """Run full-output validation and enforce the frozen acceptance rule.

    Thin façade over ``validate_outputs``. The pass rule is exactly
    ``Comparison.passed``: ``q_elements > 0`` and ``q_bad / q_elements <= 1e-3``
    and ``q_nonfinite == 0`` and ``qr_hard == 0`` and ``kv_hard == 0`` and
    ``untouched_bad == 0``. Parameters mirror ``validate_outputs``; ``shadow``
    (an ``FP64Shadow``) is mandatory because the value-domain criterion requires
    the FP64 reference.

    数据流：returns the ``Comparison`` stats when passed; otherwise raises
    ``AssertionError`` carrying the complete ``stats.output()`` dict.
    """
    stats = validate_outputs(
        inputs,
        weights,
        outputs,
        cache_device,
        row_chunk=row_chunk,
        progress=progress,
        shadow=shadow,
        norm_eps=norm_eps,
    )
    if not stats.passed:
        raise AssertionError(stats.output())
    return stats


# --------------------------------------------------------------------------
# 判据自检（CPU 可运行；参数集与源 harness 的 pytest 参数化用例一致且冻结）
# --------------------------------------------------------------------------


#: Fixed E4M3 encodings guarding against false accuracy passes.
_FP8_VALUE_CRITERION_CASES = (
    ("signed-zero", (0x00, 0x80), 0.0, (1.0, 1.0), 0),
    ("equivalent-scales", (0x38, 0x30), 1.0, (1.0, 2.0), 0),
    ("adjacent-bracket", (0x38, 0x39), 1.0625, (1.0, 1.0), 0),
    ("nonadjacent-error", (0x38, 0x3A), 1.25, (1.0, 1.0), 1),
    ("wrong-scale", (0x38, 0x38), 1.0, (2.0, 1.0), 1),
    ("nonfinite-output", (0x7F, 0x38), 1.0, (1.0, 1.0), 1),
)

#: Corruption injections exercised against the full-output validator.
_FULL_OUTPUT_CORRUPTIONS = (None, "q", "qr", "scale", "kv", "untouched")


def test_fp8_value_criterion():
    """Use fixed E4M3 encodings to guard against false accuracy passes on CPU."""
    for case_id, encodings, truth, scales, expected_hard in (
        _FP8_VALUE_CRITERION_CASES
    ):
        got, reference = encodings
        got_scale, ref_scale = scales
        verdict = adjudicate(
            torch.tensor([[got]], dtype=torch.uint8),
            torch.tensor([[reference]], dtype=torch.uint8),
            torch.tensor([[truth]], dtype=torch.float64),
            got_scale=torch.tensor([[got_scale]], dtype=torch.float64),
            ref_scale=torch.tensor([[ref_scale]], dtype=torch.float64),
        )
        if verdict.checked != 1:
            raise AssertionError(case_id)
        if verdict.hard != expected_hard:
            raise AssertionError(case_id)
        if verdict.legitimate != 1 - expected_hard:
            raise AssertionError(case_id)
    logger.info("test_fp8_value_criterion: %d/%d PASS",
                len(_FP8_VALUE_CRITERION_CASES), len(_FP8_VALUE_CRITERION_CASES))


def test_full_output_validator():
    """Reject damaged Q, FP8 payloads/scales and written or untouched cache."""
    for corruption in _FULL_OUTPUT_CORRUPTIONS:
        weights = SharedWeights(profile=SMALL_PROFILE)
        inputs = make_case_inputs(Case("decode", 1, 3), weights)
        expected = attn_prologue_golden(inputs, 1e-6, trace=False, row_chunk=2)
        q = expected.q.clone()
        qr = expected.qr.contiguous().view(torch.uint8).clone()
        scales = expected.descale_qr.contiguous().view(torch.uint8).clone()
        cache = expected.kv_cache.clone()
        if corruption == "q":
            q[0, 0, 0] = float("nan")
        elif corruption == "qr":
            qr[0, 0] = 0x7F
        elif corruption == "scale":
            scales[0].fill_(255)
        elif corruption == "kv":
            cache.view(-1, weights.profile[3])[inputs.cache_index[0], 0] = 0x7F
        elif corruption == "untouched":
            cache_rows = cache.view(-1, weights.profile[3])
            unused = next(
                i
                for i in range(cache_rows.shape[0])
                if i not in inputs.cache_index.tolist()
            )
            cache_rows[unused, 0] ^= 1
        stats = validate_outputs(
            inputs,
            weights,
            (q, qr, scales),
            cache,
            row_chunk=2,
            shadow=FP64Shadow(weights),
        )
        if stats.passed != (corruption is None):
            raise AssertionError(stats.output())
        logger.info(
            "test_full_output_validator[corruption=%s]: %s PASS",
            corruption,
            "passed" if stats.passed else "correctly rejected",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_fp8_value_criterion()
    test_full_output_validator()
    raise SystemExit(0)
