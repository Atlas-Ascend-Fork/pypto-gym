#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
"""quant_lightning_indexer PyPTO-Pro test: one test function per case, plus a ``main`` entry.

The kernel and the host wrapper live next to this directory, under
``ops/pypto_pro/deepseek_v4_1/quant_lightning_indexer/op_kernel/quant_lightning_indexer.py``.
Every ``test_xxx`` prepares one case's inputs with the independent CPU golden,
runs the public wrapper on the NPU with ``return_value=True`` and ``False`` and
checks all four outputs against that golden.

Two kinds of outputs are judged separately. The index tensors are judged as
sets by ``quant_lightning_indexer_precision.check_topk`` (the contract allows
any output order and any legal tied pick, so they cannot be compared
elementwise), while the returned scores and the candidate counts go through the
repo-wide Scheme A engine ``quant_lightning_indexer_precision.check_precision``.

Run: ``python test_quant_lightning_indexer.py``
"""

import logging
import os
import sys

import pytest
import torch
import torch_npu  # noqa: F401  (registers the torch.npu backend)

# This file sits in <op>/tests/, the golden and the comparator in <op>/tests/assets/,
# and the kernel in <op>/op_kernel/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_HERE)
for _path in (_HERE, os.path.join(_HERE, "assets"), os.path.join(_OP_DIR, "op_kernel")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import quant_lightning_indexer_golden_cpu  # noqa: E402
from quant_lightning_indexer import quant_lightning_indexer_wrapper  # noqa: E402
from quant_lightning_indexer_precision import check_precision, check_topk  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
_LOG = logging.getLogger(__name__)

# The twelve on-board cases, in delivery order. The spec dicts live in the golden
# module so its CPU self-check and this NPU test cannot drift apart.
_CASES = quant_lightning_indexer_golden_cpu.CASES


def _device():
    """Select the NPU this process is allowed to use."""
    name = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(name)
    return name


def _require(result, what):
    """Unpack a Scheme A (passed, summary) result or fail with the summary."""
    passed, summary = result
    if not passed:
        raise AssertionError(f"{what}: {summary}")


def _verify(args, kwargs, refs, allowed, return_value):
    """Run the public wrapper once and verify every output against the golden.

    The index tensors are sets - the contract allows any output order and any
    legal tied pick - so each row is judged by ``check_topk``. The scores and
    the candidate counts are elementwise and go through the repo-wide Scheme A
    engine, one call per output.
    """
    outputs = quant_lightning_indexer_wrapper(
        *args, **kwargs, return_value=return_value)
    torch.npu.synchronize()
    indices, values, candidates, lengths = [x.cpu() for x in outputs]
    for row, (ref, valid) in enumerate(zip(refs, allowed)):
        check_topk(indices[row, 0], ref, valid, row * 10000)
    if return_value:
        _require(check_precision(
            values, quant_lightning_indexer_golden_cpu.expected_scores(indices,
                                                                       refs)),
            "scores")
    capacity = kwargs["candidate_topk_blocks"]
    if capacity < 0:
        return
    counts = []
    for row, (ref, valid) in enumerate(zip(refs, allowed)):
        blocks, valid_blocks = quant_lightning_indexer_golden_cpu.candidate_scores(
            ref, valid)
        counts.append(min(capacity, int(valid_blocks.sum())))
        check_topk(candidates[row, 0, :counts[-1]], blocks, valid_blocks)
        if not bool((candidates[row, 0, counts[-1]:] == -1).all()):
            raise AssertionError("unused candidate slots must be -1")
    _require(check_precision(
        lengths, torch.tensor(counts, dtype=torch.int32).view_as(lengths)),
        "candidate_lengths")


def _run_case(name):
    """Run one case on the NPU through the public wrapper and check its precision."""
    args, kwargs, refs, allowed = quant_lightning_indexer_golden_cpu.make_inputs(
        _CASES[name], _device())
    for return_value in (True, False):
        _verify(args, kwargs, refs, allowed, return_value)
    _LOG.info("[%s] status=PASS :: indices=exact TopK set, scores/counts=Scheme A",
              name)


@pytest.mark.soc("950")
def test_pa128_ragged():
    _run_case("pa128_ragged")


@pytest.mark.soc("950")
def test_pa64_causal_candidates():
    _run_case("pa64_causal_candidates")


@pytest.mark.soc("950")
def test_tnd_causal():
    _run_case("tnd_causal")


@pytest.mark.soc("950")
def test_candidate16_ld_tail():
    _run_case("candidate16_ld_tail")


@pytest.mark.soc("950")
def test_invalid_values_pa_bbnd():
    _run_case("invalid_values_PA_BBND")


@pytest.mark.soc("950")
def test_invalid_values_tnd():
    _run_case("invalid_values_TND")


@pytest.mark.soc("950")
def test_empty_query_pa_bbnd():
    _run_case("empty_query_PA_BBND")


@pytest.mark.soc("950")
def test_empty_key_pa_bbnd():
    _run_case("empty_key_PA_BBND")


@pytest.mark.soc("950")
def test_empty_query_tnd():
    _run_case("empty_query_TND")


@pytest.mark.soc("950")
def test_empty_key_tnd():
    _run_case("empty_key_TND")


@pytest.mark.soc("950")
def test_b12_s1_6_n32_s2_64k():
    _run_case("b12_s1_6_n32_s2_64k")


@pytest.mark.soc("950")
def test_b12_s1_6_n32_s2_128k():
    _run_case("b12_s1_6_n32_s2_128k")


def main():
    test_pa128_ragged()
    test_pa64_causal_candidates()
    test_tnd_causal()
    test_candidate16_ld_tail()
    test_invalid_values_pa_bbnd()
    test_invalid_values_tnd()
    test_empty_query_pa_bbnd()
    test_empty_key_pa_bbnd()
    test_empty_query_tnd()
    test_empty_key_tnd()
    test_b12_s1_6_n32_s2_64k()
    test_b12_s1_6_n32_s2_128k()


if __name__ == "__main__":
    raise SystemExit(main())
