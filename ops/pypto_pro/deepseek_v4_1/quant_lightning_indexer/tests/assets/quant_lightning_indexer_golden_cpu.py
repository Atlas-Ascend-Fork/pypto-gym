#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""quant_lightning_indexer CPU Golden for the exact SPEC output ABI.

Pure-torch CPU reference; no PyPTO and no NPU import, so the golden stays
independent from the implementation it validates. Four parts, top to bottom:

1. reference math: ``decode`` / ``scores`` / ``visible_keys`` /
   ``candidate_scores`` - the four SPEC primitives;
2. ``CASES`` - the twelve delivered on-board cases (PA128 / PA64 / TND, ragged
   and causal-compressed visibility, candidates on and off, short sequences,
   empty query / empty key, two B=12 long-context cases), the single source the
   NPU test and the self-check share;
3. ``make_inputs`` - deterministic inputs plus the reference rows, and
   ``expected_scores`` - the expected score tensor for a given selection;
4. a CPU self-check (``python quant_lightning_indexer_golden_cpu.py``) that
   cross-validates the math against an independent FP64 / E2M1-bit-field oracle.

The score chain is a numeric contract, not an optimization target: QK is rounded
to BF16 once, then heads 0..31 are folded with a multiply-add that is rounded to
BF16 after every step. It must not be replaced by a single FP32 reduction.
"""

import logging
from itertools import accumulate
from typing import NamedTuple

import torch

_LOG = logging.getLogger(__name__)

# SPEC 公式（数值契约，自检输出用它）
_SPEC_FORMULA = (
    'q_f = decode_mxfp4(q, descale_q); k_f = decode_mxfp4(k, descale_k); '
    'qk = bf16(q_f @ k_f^T).float(); w_b = bf16(w).float(); '
    'acc_0 = 0; acc_{h+1} = bf16(acc_h + relu(qk[:, h]) * w_b[:, h]).float(), '
    'h = 0..31; score = acc_32; '
    'visible(t) = 0 if t >= Lq, else Lk if mask_mode == 0, else '
    'min(Lk, max(0, (cmp_ratio * Lk + residual - Lq + t + 1) // cmp_ratio)); '
    'candidate block score = max over the 8 tokens of the block, and the '
    'trailing partial block of a query whose visible count is not a multiple '
    'of 8 is +inf'
)

HEADS = 32
CANDIDATE_CAPACITY = 2048
TOPK = 512
SEED = 20260929
_HEAD_DIM = 128
_PACKED_DIM = _HEAD_DIM // 2
_SCALES_PER_HEAD = 4
_SCALE_GROUP = _HEAD_DIM // _SCALES_PER_HEAD
_SCALE_BIAS = 127
# Above this many key rows the FP64 oracle is skipped (host memory / runtime);
# the two long-context typical cases are contract-checked without it.
_ORACLE_KEY_LIMIT = 200000

# ===== 交付算例定义（测试与自检共用的唯一来源）=====
# name -> spec; keys see _SPEC_DEFAULTS. Only these twelve are delivered.
CASES = {
    "pa128_ragged": {"layout": "PA_BBND", "page_size": 128, "mask": 0},
    "pa64_causal_candidates": {
        "layout": "PA_BBND", "page_size": 64, "mask": 3, "candidates": True},
    "tnd_causal": {"layout": "TND", "page_size": 128, "mask": 3},
    "candidate16_ld_tail": {
        "layout": "PA_BBND", "page_size": 64, "mask": 3, "candidates": True,
        "candidate_capacity": 16},
    "invalid_values_PA_BBND": {
        "layout": "PA_BBND", "page_size": 128, "mask": 0,
        "key_lengths": (19, 73)},
    "invalid_values_TND": {
        "layout": "TND", "page_size": 128, "mask": 0, "key_lengths": (19, 73)},
    "empty_query_PA_BBND": {
        "layout": "PA_BBND", "page_size": 128, "mask": 0,
        "key_lengths": (19, 73), "empty": "query"},
    "empty_key_PA_BBND": {
        "layout": "PA_BBND", "page_size": 128, "mask": 0,
        "key_lengths": (19, 73), "empty": "key"},
    "empty_query_TND": {
        "layout": "TND", "page_size": 128, "mask": 0, "key_lengths": (19, 73),
        "empty": "query"},
    "empty_key_TND": {
        "layout": "TND", "page_size": 128, "mask": 0, "key_lengths": (19, 73),
        "empty": "key"},
    "b12_s1_6_n32_s2_64k": {
        "layout": "PA_BBND", "page_size": 128, "mask": 3,
        "query_lengths": (6,) * 12, "key_lengths": (65536,) * 12,
        "signed_weights": True, "uniform_scale": True},
    "b12_s1_6_n32_s2_128k": {
        "layout": "PA_BBND", "page_size": 128, "mask": 3,
        "query_lengths": (6,) * 12, "key_lengths": (131072,) * 12,
        "signed_weights": True, "uniform_scale": True},
}

_SPEC_DEFAULTS = {
    "layout": "PA_BBND",
    "page_size": 128,
    "mask": 0,
    "candidates": False,
    "candidate_capacity": CANDIDATE_CAPACITY,
    "key_lengths": (513, 777),
    "topk": TOPK,
    "empty": None,
    "query_lengths": None,
    "signed_weights": False,
    "uniform_scale": False,
}


def _prefix(lengths):
    return [0, *accumulate(lengths)]


class _Lengths(NamedTuple):
    """Per-batch query, used-query and key lengths of one case."""

    query: list
    used: list
    key: list


class _MaskRule(NamedTuple):
    """Visibility rule of one batch: mask mode, compression ratio, residual."""

    mode: int
    ratio: int
    residual: int


class _RowContext(NamedTuple):
    """The per-batch contract every query row of that batch is checked against."""

    width: int
    used: int
    rule: _MaskRule
    empty: bool


class _RowRef(NamedTuple):
    """One reference row: its scores, its visibility mask and the wanted count."""

    score: torch.Tensor
    visible: torch.Tensor
    want: int


# ===== 参考数学原语（SPEC 的四个独立推导）=====


def decode(packed, scales):
    magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    mantissa = magnitude[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
    exponent = scales.reshape(*packed.shape[:-1], 4).int() - 127
    return mantissa * torch.exp2(exponent.float()).repeat_interleave(32, -1)


def scores(q, k, w, qs, ks):
    qk = (decode(q, qs) @ decode(k, ks).squeeze(1).T).bfloat16().float()
    weight = w.bfloat16().float()
    result = torch.zeros((q.shape[0], k.shape[0]), dtype=torch.float32)
    for head in range(32):
        result = (
            (result + qk[:, head].clamp_min(0) * weight[:, head, None])
            .bfloat16()
            .float()
        )
    return result


def visible_keys(lk, lq, row, rule):
    """How many keys query row ``row`` may see, per the visibility contract."""
    if row >= lq:
        return 0
    if rule.mode == 0:
        return lk
    span = rule.ratio * lk + rule.residual - lq + row + 1
    return min(lk, max(0, span // rule.ratio))


def candidate_scores(expected, allowed):
    """Independent 8-token candidate scores and visibility for one query."""
    padded = torch.full((((expected.numel() + 7) // 8) * 8,), -float("inf"))
    padded[: expected.numel()] = expected.masked_fill(~allowed, -float("inf"))
    block_scores = padded.view(-1, 8).amax(-1)
    valid_blocks = torch.isfinite(block_scores)
    visible = int(allowed.sum())
    if visible % 8:
        block_scores[visible // 8] = float("inf")
    return block_scores, valid_blocks


# ===== 输入构造与参考输出 =====


def make_inputs(case, device=None):
    """Build one case's positional args, keyword attributes and CPU reference.

    Deterministic: every case draws from a fresh generator seeded with ``SEED``
    in a fixed order, so the tensors are reproducible run to run.

    Args:
        case: spec dict; every key is optional (see ``_SPEC_DEFAULTS``).
        device: target device of the returned tensors (CPU when omitted).

    Returns:
        (args, kwargs, refs, allowed): the five positional tensors, the
        sequence/attribute keywords of the public wrapper, one [Lk] float32
        reference score row per query and one [Lk] bool visibility mask per
        query. ``refs``/``allowed`` always stay on CPU.
    """
    args, kwargs, refs, allowed, _ = _build(_resolve(case), device)
    return args, kwargs, refs, allowed


def _resolve(case):
    """Fill one case spec with the defaults and reject unknown keys."""
    unknown = set(case) - set(_SPEC_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown case keys: {sorted(unknown)}")
    spec = dict(_SPEC_DEFAULTS)
    spec.update(case)
    return spec


def _build(spec, device):
    """Deterministic inputs, wrapper keywords and reference rows for one spec.

    Returns (args, kwargs, refs, allowed, logical): the five positional tensors,
    the sequence/attribute keywords, one [Lk] float32 reference score row and
    one [Lk] bool visibility mask per query, and the pre-paging logical tensors
    ``logical = (q, weights, qs, logical_keys, logical_scales)`` that the
    self-check oracle needs (the oracle compares logical key order against the
    reference rows, which are also built pre-paging).
    """
    lengths, ratio, residual = _sequence(spec)
    generator = torch.Generator().manual_seed(SEED)
    logical = _logical_tensors(spec, lengths, generator)
    q, weights, qs, logical_keys, logical_scales = logical
    k, ks, table = _key_storage(spec, generator, logical_keys, logical_scales)
    refs, allowed = _reference_rows(logical, lengths, spec["mask"], ratio, residual)
    kwargs = _wrapper_kwargs(spec, lengths, ratio, device)
    if table is not None:
        kwargs["block_table"] = table.to(device)
    args = [
        q.to(device), k.to(device), weights.to(device), qs.to(device), ks.to(device)
    ]
    if spec["empty"] is not None:
        # An empty query / key batch is expressed through the used-length array;
        # mask_mode=0 would still report Lk visible, so the reference visibility
        # is cleared explicitly, matching the contract that nothing is selected.
        kwargs["seqused_q" if spec["empty"] == "query" else "seqused_k"].zero_()
        for visible in allowed:
            visible.zero_()
    return args, kwargs, refs, allowed, logical


def _logical_tensors(spec, lengths, generator):
    """Pre-paging query, weight, per-head scale and per-batch key tensors."""

    def draw(shape, low=0, high=256):
        return torch.randint(low, high, shape, generator=generator,
                             dtype=torch.uint8)

    rows = sum(lengths.query)
    q = draw((rows, HEADS, _PACKED_DIM))
    qs = (torch.full((rows, HEADS, 2, 2), _SCALE_BIAS, dtype=torch.uint8)
          if spec["uniform_scale"]
          else draw((rows, HEADS, 2, 2), 124, 128))
    weights = torch.rand((rows, HEADS), generator=generator)
    weights = weights * 2 - 1 if spec["signed_weights"] else weights + 0.1
    keys = [draw((n, 1, _PACKED_DIM)) for n in lengths.key]
    scales = [
        torch.full((n, 1, 2, 2), _SCALE_BIAS, dtype=torch.uint8)
        if spec["uniform_scale"]
        else draw((n, 1, 2, 2), 124, 128)
        for n in lengths.key
    ]
    return q, weights, qs, keys, scales


def _key_storage(spec, generator, logical_keys, logical_scales):
    """Key and scale storage in the case's layout, plus its page table."""
    if spec["layout"] == "TND":
        return torch.cat(logical_keys), torch.cat(logical_scales), None
    if spec["layout"] != "PA_BBND":
        raise ValueError(f"unknown layout_k: {spec['layout']}")
    page_size = spec["page_size"]
    per_batch = (max(spec["key_lengths"]) + page_size - 1) // page_size
    pages = len(spec["key_lengths"]) * per_batch
    table = (torch.randperm(pages, generator=generator).int()
             .reshape(len(spec["key_lengths"]), per_batch))
    k = torch.zeros((pages, page_size, 1, _PACKED_DIM), dtype=torch.uint8)
    ks = torch.full((pages, page_size, 1, 2, 2), _SCALE_BIAS, dtype=torch.uint8)
    for batch, length in enumerate(spec["key_lengths"]):
        tokens = torch.arange(length)
        slot = table[batch, tokens // page_size].long()
        physical = slot * page_size + tokens % page_size
        k.view(-1, 1, _PACKED_DIM).index_copy_(0, physical, logical_keys[batch])
        ks.view(-1, 1, 2, 2).index_copy_(0, physical, logical_scales[batch])
    return k, ks, table


def _wrapper_kwargs(spec, lengths, ratio, device):
    """The sequence and attribute keywords of the public wrapper."""

    def meta(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    offset = (torch.arange(sum(lengths.query), dtype=torch.int32, device=device)
              .reshape(-1, 1) * 10000)
    kwargs = dict(
        cu_seqlens_q=meta(_prefix(lengths.query)),
        seqused_q=meta(lengths.used),
        seqused_k=meta(lengths.key),
        topk=spec["topk"],
        quant_mode=1,
        max_seqlen_q=max(lengths.query),
        layout_q="TND",
        layout_k=spec["layout"],
        mask_mode=spec["mask"],
        cmp_ratio=ratio,
        candidate_topk_blocks=(spec["candidate_capacity"]
                               if spec["candidates"] else -1),
        candidate_block_size=8 if spec["candidates"] else -1,
        output_idx_offset=offset,
    )
    if spec["layout"] == "TND":
        kwargs["cu_seqlens_k"] = meta(_prefix(lengths.key))
    if spec["mask"] == 3:
        kwargs["cmp_residual_k"] = meta(_residual_of(spec, lengths))
    return kwargs


def _sequence(spec):
    """Resolve per-batch lengths, the compression ratio and the residual array."""
    lengths = _lengths_of(spec)
    ratio = 2 if spec["mask"] == 3 else 1
    return lengths, ratio, _residual_of(spec, lengths)


def _lengths_of(spec):
    """Per-batch query, used-query and key lengths of one case."""
    typical = spec["query_lengths"] is not None
    query = list(spec["query_lengths"]) if typical else [3, 4]
    used = query if typical else [3, 3]
    key = list(spec["key_lengths"])
    if len(query) != len(key):
        raise ValueError("query_lengths and key_lengths must have equal length")
    return _Lengths(query, used, key)


def _residual_of(spec, lengths):
    """Compression residual per batch; only the typical spec carries a nonzero."""
    if spec["query_lengths"] is None:
        return [1, 0]
    return [0] * len(lengths.query)


def _reference_rows(logical, lengths, mask, ratio, residual):
    """Per-query reference score rows and visibility masks in logical key order."""
    q, weights, qs, logical_keys, logical_scales = logical
    refs, allowed = [], []
    for batch, (start, length) in enumerate(zip(_prefix(lengths.query),
                                                lengths.query)):
        row_scores = scores(q[start:start + length], logical_keys[batch],
                            weights[start:start + length],
                            qs[start:start + length], logical_scales[batch])
        rule = _MaskRule(mask, ratio, residual[batch])
        for row in range(length):
            visible = visible_keys(lengths.key[batch], lengths.used[batch], row,
                                   rule)
            refs.append(row_scores[row])
            allowed.append(torch.arange(lengths.key[batch]) < visible)
    return refs, allowed

# ===== 期望输出（供上板精度对比）=====


def expected_scores(indices, refs):
    """Expected score tensor for the indices the kernel actually selected.

    Row ``r`` holds the reference scores at ``indices[r, 0]`` minus that row's
    ``output_idx_offset`` (row * 10000, the convention ``make_inputs`` uses) and
    ``-inf`` in the invalid slots. Ties make the selected index set
    implementation-defined, so the scores are judged at whatever indices the
    kernel picked; the ``-inf`` slots ride along in the same elementwise check.
    """
    golden = torch.full(indices.shape, float("-inf"), dtype=torch.bfloat16)
    for row, ref in enumerate(refs):
        picked = indices[row, 0]
        present = picked != -1
        golden[row, 0][present] = (
            ref[picked[present].long() - row * 10000].to(torch.bfloat16))
    return golden


# ===== CPU 自检（python quant_lightning_indexer_golden_cpu.py）=====
# 上板测试只验证 kernel 与 golden 一致；golden 自身是否正确只能在这里验证。
# _scores_oracle 用第二种独立推导（E2M1 位域 + FP64 精确折叠）交叉复核
# scores/decode，_check_output 逐算例核对契约与 oracle，_run_property_checks
# 覆盖 oracle 表达不了的性质（E8M0 精确 2 的幂、零/符号权重、候选块规则）。


def _get_device():
    return torch.device("cpu")


def _scores_oracle(q, k, w, qs, ks):
    """Independent higher-precision oracle: E2M1 bit fields, exact-real fold."""
    def decode_bits(packed, scales):
        codes = torch.stack(((packed & 15).long(), (packed >> 4).long()), -1)
        codes = codes.reshape(*packed.shape[:-1], _HEAD_DIM)
        exponent, mantissa = codes >> 1 & 3, codes & 1
        normal = (1.0 + mantissa.float() * 0.5) * torch.exp2((exponent - 1).float())
        magnitude = torch.where(exponent == 0, mantissa.float() * 0.5, normal)
        scale = scales.reshape(*packed.shape[:-1], _SCALES_PER_HEAD).int() - _SCALE_BIAS
        scale = torch.exp2(scale.float()).repeat_interleave(_SCALE_GROUP, -1)
        return torch.where(codes < 8, magnitude, -magnitude) * scale

    qk = decode_bits(q, qs).double() @ decode_bits(k, ks).squeeze(1).T.double()
    qk = qk.to(torch.bfloat16).double()
    weight = w.double().to(torch.bfloat16).double()
    acc = torch.zeros((q.shape[0], k.shape[0]), dtype=torch.float64)
    for head in range(HEADS):
        acc = (acc + qk[:, head].clamp_min(0)
               * weight[:, head].unsqueeze(-1)).to(torch.bfloat16).double()
    return acc.float()


def _bf16_ulp(value):
    scale = value.abs().to(torch.float32)
    return torch.nextafter(scale, torch.full_like(scale, float("inf"))) - scale


def _check_output(name, refs, allowed, spec, logical):
    """Contract check plus the FP64-oracle deviation for one case.

    Returns the worst absolute and worst per-ULP deviation of ``scores`` from the
    independent FP64 oracle, or zeros when the case is too large for the oracle.
    """
    _check_contract(name, refs, allowed, spec)
    if sum(spec["key_lengths"]) > _ORACLE_KEY_LIMIT:
        return 0.0, 0.0
    lengths, _, _ = _sequence(spec)
    return _oracle_deviation(logical, refs, lengths)


def _check_contract(name, refs, allowed, spec):
    """Walk every reference row and enforce the visibility / score contract."""
    lengths, ratio, residual = _sequence(spec)
    if len(refs) != sum(lengths.query) or len(allowed) != len(refs):
        raise AssertionError(f"{name}: {len(refs)} reference rows is inconsistent")
    index = 0
    for batch, length in enumerate(lengths.query):
        context = _RowContext(
            lengths.key[batch],
            lengths.used[batch],
            _MaskRule(spec["mask"], ratio, residual[batch]),
            spec["empty"] is not None,
        )
        for row in range(length):
            index += 1
            want = visible_keys(context.width, context.used, row, context.rule)
            reference = _RowRef(refs[index - 1], allowed[index - 1], want)
            _check_row(name, index, context, reference)
    if index != len(refs):
        raise AssertionError(f"{name}: walked {index} rows, refs hold {len(refs)}")


def _check_row(name, index, context, reference):
    """Check one reference row against the visibility contract."""
    if reference.score.ndim != 1 or reference.score.numel() != context.width:
        raise AssertionError(f"{name}: row {index} score width != Lk")
    if reference.visible.shape != reference.score.shape:
        raise AssertionError(f"{name}: row {index} visibility has the wrong shape")
    if reference.visible.dtype != torch.bool:
        raise AssertionError(f"{name}: row {index} visibility is not a bool mask")
    if context.empty:
        if bool(reference.visible.any()):
            raise AssertionError(f"{name}: row {index} must see no key")
        return
    seen = int(reference.visible.sum())
    if seen != reference.want:
        raise AssertionError(
            f"{name}: row {index} sees {seen} keys, the rule gives {reference.want}"
        )
    prefix = torch.arange(context.width) < reference.want
    if not torch.equal(reference.visible, prefix):
        raise AssertionError(f"{name}: row {index} visibility is not a prefix")
    head = reference.score[:reference.want]
    if reference.want and not bool(torch.isfinite(head).all()):
        raise AssertionError(f"{name}: row {index} has non-finite scores")


def _oracle_deviation(logical, refs, lengths):
    """Worst absolute and worst per-ULP deviation of ``refs`` from the oracle."""
    q, weights, qs, logical_keys, logical_scales = logical
    worst_abs = worst_ulp = 0.0
    start = 0
    for batch, length in enumerate(lengths.query):
        want = _scores_oracle(q[start:start + length], logical_keys[batch],
                              weights[start:start + length],
                              qs[start:start + length], logical_scales[batch])
        for row in range(length):
            diff = (refs[start + row] - want[row]).abs()
            ulp = _bf16_ulp(want[row])
            if diff.numel():
                worst_abs = max(worst_abs, float(diff.max()))
                worst_ulp = max(worst_ulp,
                                float((diff / ulp.clamp_min(1e-30)).max()))
        start += length
    return worst_abs, worst_ulp


def _validate():
    """Run every delivered case on CPU and check the golden contract."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = _get_device()
    _LOG.info("=" * 60)
    _LOG.info("quant_lightning_indexer_golden_cpu validation report")
    _LOG.info("=" * 60)
    _LOG.info("Device: %s", device)
    _LOG.info("Formula: %s", _SPEC_FORMULA)
    if len(CASES) != 12:
        raise AssertionError(f"expected 12 delivered cases, got {len(CASES)}")

    _LOG.info("[delivered cases]")
    for name, case in CASES.items():
        spec = _resolve(case)
        oracle = sum(spec["key_lengths"]) <= _ORACLE_KEY_LIMIT
        _, kwargs, refs, allowed, logical = _build(spec, device)
        worst_abs, worst_ulp = _check_output(name, refs, allowed, spec, logical)
        if oracle and worst_ulp > 1.0:
            raise AssertionError(
                f"{name}: scores deviate from the FP64 oracle by {worst_ulp} ULP")
        _LOG.info("  [PASS] %-26s rows=%-3d topk=%-5d layout=%-8s oracle=%-3s "
                  "max_diff=%.3e (%.2f ULP)", name, len(refs), kwargs["topk"],
                  kwargs["layout_k"], "on" if oracle else "off", worst_abs,
                  worst_ulp)
        del kwargs, refs, allowed, logical

    _run_property_checks()
    _LOG.info("=" * 60)
    _LOG.info("validation complete")
    _LOG.info("=" * 60)
    return True


def _run_property_checks():
    """Contract properties not already covered by the per-case oracle check.

    ``_check_output`` already compares ``scores`` (lookup-table decode) against
    the independent FP64 / bit-field oracle and checks ``visible_keys`` exactly
    on every row of every case, so those are not repeated here.
    """
    _LOG.info("[property checks]")
    generator = torch.Generator().manual_seed(11)
    _check_decode_property(generator)
    _check_weight_property(generator)
    _check_candidate_property()


def _check_decode_property(generator):
    """Every E2M1 code and every E8M0 shift, against the bit-field definition."""
    packed = torch.arange(256, dtype=torch.uint8).reshape(2, 32, 4).repeat(1, 1, 16)
    flat = torch.full((2, 32, 2, 2), _SCALE_BIAS, dtype=torch.uint8)
    codes = torch.stack(((packed & 15).long(), (packed >> 4).long()), -1)
    codes = codes.reshape(*packed.shape[:-1], _HEAD_DIM)
    exponent, mantissa = codes >> 1 & 3, codes & 1
    normal = (1.0 + mantissa.float() * 0.5) * torch.exp2((exponent - 1).float())
    want = torch.where(exponent == 0, mantissa.float() * 0.5, normal)
    want = torch.where(codes < 8, want, -want)
    if not torch.equal(decode(packed, flat), want):
        raise AssertionError("decode disagrees with the E2M1 bit-field derivation")
    packed = torch.randint(0, 256, (4, 1, _PACKED_DIM), dtype=torch.uint8,
                           generator=generator)
    base = decode(packed, torch.full((4, 1, 2, 2), _SCALE_BIAS,
                                    dtype=torch.uint8))
    for offset in (-7, 0, 5, 13):
        shifted = decode(packed, torch.full((4, 1, 2, 2), _SCALE_BIAS + offset,
                                            dtype=torch.uint8))
        if not torch.equal(shifted, base * torch.exp2(torch.tensor(float(offset)))):
            raise AssertionError(f"E8M0 shift {offset} is not an exact 2^x scale")
    _LOG.info("  [PASS] E2M1: all 256 packed bytes decode exactly; E8M0 is an "
              "exact power-of-two multiplier")


def _check_weight_property(generator):
    """Weight sign must carry through to the sign of every score."""
    q = torch.randint(0, 256, (3, HEADS, _PACKED_DIM), dtype=torch.uint8,
                      generator=generator)
    qs = torch.randint(124, 129, (3, HEADS, 2, 2), dtype=torch.uint8,
                       generator=generator)
    k = torch.randint(0, 256, (37, 1, _PACKED_DIM), dtype=torch.uint8,
                      generator=generator)
    ks = torch.randint(124, 129, (37, 1, 2, 2), dtype=torch.uint8,
                       generator=generator)
    w = torch.rand((3, HEADS), generator=generator) * 2 - 1
    if not bool((scores(q, k, torch.zeros_like(w), qs, ks) == 0).all()):
        raise AssertionError("zero weights must give exactly zero scores")
    if not bool((scores(q, k, w.abs(), qs, ks) >= 0).all()):
        raise AssertionError("|w| must give non-negative scores")
    if not bool((scores(q, k, -w.abs(), qs, ks) <= 0).all()):
        raise AssertionError("-|w| must give non-positive scores")
    _LOG.info("  [PASS] zero weights -> 0; |w| -> non-negative; -|w| -> non-positive")


def _check_candidate_property():
    """Block count, full-block max, and the +inf tail on a partial block."""
    expected = torch.arange(20, dtype=torch.float32)
    blocks, valid = candidate_scores(expected, torch.arange(20) < 13)
    if blocks.numel() != 3 or int(valid.sum()) != 2 or not torch.isposinf(blocks[1]):
        raise AssertionError("the trailing partial block must be +inf")
    if float(blocks[0]) != 7.0 or float(blocks[2]) != -float("inf"):
        raise AssertionError(f"block scores {blocks.tolist()} are wrong")
    blocks, valid = candidate_scores(expected, torch.arange(20) < 16)
    if blocks.numel() != 3 or int(valid.sum()) != 2 or bool(torch.isposinf(blocks).any()):
        raise AssertionError("an exact multiple of 8 must not add a +inf tail")
    _LOG.info("  [PASS] candidates: ceil(n/8) blocks, full-block max, +inf tail only "
              "for a partial visible count")


if __name__ == "__main__":
    raise SystemExit(0 if _validate() else 1)
