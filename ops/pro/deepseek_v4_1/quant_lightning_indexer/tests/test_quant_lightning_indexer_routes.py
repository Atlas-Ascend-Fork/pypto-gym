# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the JIT and Torch entries against the CPU golden in tests/assets."""
from functools import lru_cache
import importlib.util
import inspect
import itertools
import os
from pathlib import Path
import sys

import pytest
import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).resolve().parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


golden = _load_module("gym_pro_qli_golden", HERE / "assets/quant_lightning_indexer_golden_cpu.py")
precision = _load_module("gym_pro_qli_precision", HERE / "assets/quant_lightning_indexer_precision.py")

pytestmark = pytest.mark.soc("950")
_ENTRY_CYCLE = itertools.cycle(("python", "torch"))


def _torch_entry(*args, **kwargs):
    """Call the raw Torch schema with the same defaults as the public frontend."""
    import pypto_gym
    bound = inspect.signature(pypto_gym.quant_lightning_indexer).bind(*args, **kwargs)
    bound.apply_defaults()
    props = torch.npu.get_device_properties(bound.arguments["q"].device.index)
    return torch.ops.pypto_gym.quant_lightning_indexer(
        **bound.arguments, cube_capacity=props.cube_core_num, vector_capacity=props.vector_core_num)


@lru_cache(maxsize=1)
def _jit_module():
    return _load_module("gym_pro_quant_lightning_indexer", HERE.parent / "op_kernel/quant_lightning_indexer.py")


def runner():
    if os.environ.get("GYM_TEST_MODE", "torch") == "jit":
        return _jit_module().quant_lightning_indexer
    import pypto_gym
    if next(_ENTRY_CYCLE) == "python":
        return pypto_gym.quant_lightning_indexer
    pypto_gym.load("quant_lightning_indexer")
    return _torch_entry


def device():
    name = "npu:" + os.environ.get("TILE_FWK_DEVICE_ID", "0")
    torch.npu.set_device(name)
    return name


def verify(outputs, kwargs, refs, allowed, return_value, offsets=True):
    torch.npu.synchronize()
    idx, val, cand, length = [t.cpu() for t in outputs]
    assert idx.dtype == torch.int32 and idx.shape == (len(refs), 1, kwargs["topk"])
    for row, (ref, valid) in enumerate(zip(refs, allowed)):
        precision.check_topk(idx[row, 0], ref, valid, row * 10000 if offsets else 0)
    if return_value:
        passed, summary = precision.check_precision(val, golden.expected_scores(idx, refs))
        assert passed, summary
    else:
        assert val.shape == (0,) and val.dtype == torch.bfloat16
    capacity = kwargs["candidate_topk_blocks"]
    if capacity < 0:
        assert cand.shape == length.shape == (0,)
        return
    counts = []
    for row, (ref, valid) in enumerate(zip(refs, allowed)):
        score, active = golden.candidate_scores(ref, valid)
        count = min(capacity, int(active.sum()))
        counts.append(count)
        precision.check_topk(cand[row, 0, :count], score, active)
        assert bool((cand[row, 0, count:] == -1).all())
    assert torch.equal(length, torch.tensor(counts, dtype=torch.int32).view_as(length))


@pytest.mark.parametrize("name", list(golden.CASES))
@pytest.mark.parametrize("return_value", [False, True])
def test_default(name, return_value):
    args, kwargs, refs, allowed = golden.make_inputs(golden.CASES[name], device())
    verify(runner()(*args, **kwargs, return_value=return_value), kwargs, refs, allowed,
           return_value)


BOUNDARIES = {
    "page16_topk1": dict(page_size=16, key_lengths=(1, 17), topk=1),
    "page32_topk129": dict(page_size=32, key_lengths=(257, 511), topk=129),
    "page80_causal": dict(page_size=80, key_lengths=(513, 777), mask=3),
    "short_topk8192": dict(key_lengths=(513, 777), topk=8192),
    "long_topk8192": dict(layout="TND", key_lengths=(17001,), query_lengths=(1,),
                          topk=8192, signed_weights=True, uniform_scale=True),
    "large_candidates": dict(page_size=64, mask=3, candidates=True,
                             candidate_capacity=9000, query_lengths=(2, 3),
                             key_lengths=(257, 111)),
    "nonpower_fast_shards": dict(page_size=128, mask=3, query_lengths=(6,),
                                key_lengths=(98304,), signed_weights=True,
                                uniform_scale=True),
    "pa128_128k_ragged_shards": dict(page_size=128, mask=3, query_lengths=(5, 6),
                                   key_lengths=(131072, 1025), signed_weights=True,
                                   uniform_scale=True),
    "padded_pa": dict(page_size=64, mask=3, candidates=True),
    "padded_tnd": dict(layout="TND", mask=3, candidates=True),
    "optional_sequences": dict(layout="TND", query_lengths=(2,), key_lengths=(257,)),
    "ratio1": dict(page_size=64, mask=3),
    "ratio3": dict(page_size=64, mask=3),
    "ratio128": dict(page_size=64, mask=3),
}


@pytest.mark.parametrize("name", list(BOUNDARIES))
@pytest.mark.parametrize("return_value", [False, True])
def test_boundary(name, return_value):
    ratio = int(name[5:]) if name.startswith("ratio") else None
    args, kwargs, refs, allowed = golden.make_inputs(BOUNDARIES[name], device(), cmp_ratio=ratio)
    args = list(args)
    if name.startswith("padded_"):
        for pos in (1, 4):
            source = args[pos]
            shape = list(source.shape)
            shape[0] *= 2
            padded = torch.empty(shape, dtype=source.dtype, device=source.device)[::2]
            padded.copy_(source)
            args[pos] = padded
    if name == "optional_sequences":
        for key in ("cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k"):
            kwargs[key] = None
    verify(runner()(*args, **kwargs, return_value=return_value), kwargs, refs, allowed,
           return_value)


@pytest.mark.parametrize("attribute,value", [
    ("topk", 0), ("topk", 8193), ("quant_mode", 0), ("mask_mode", 1),
    ("cmp_ratio", 0), ("cmp_ratio", 129), ("layout_q", "BSND"),
    ("layout_k", "PA"), ("candidate_topk_blocks", 0), ("candidate_block_size", 8),
])
def test_invalid_attribute(attribute, value):
    args, kwargs, _, _ = golden.make_inputs(golden.CASES["pa128_ragged"], device())
    kwargs[attribute] = value
    with pytest.raises((ValueError, RuntimeError)):
        runner()(*args, **kwargs)


@pytest.mark.parametrize("position", range(5))
def test_invalid_dtype(position):
    args, kwargs, _, _ = golden.make_inputs(golden.CASES["pa128_ragged"], device())
    args = list(args)
    args[position] = args[position].to(torch.int64)
    with pytest.raises((ValueError, RuntimeError)):
        runner()(*args, **kwargs)


def test_invalid_nz_weight():
    args, kwargs, _, _ = golden.make_inputs(golden.CASES["pa128_ragged"], device())
    args = list(args)
    args[2] = torch_npu.npu_format_cast(args[2], 29)
    # The 950 may select a dtype-specific FRACTAL_NZ variant.
    assert torch_npu.get_npu_format(args[2]) not in (0, 2, 30)
    with pytest.raises((ValueError, RuntimeError), match="storage"):
        runner()(*args, **kwargs)


def test_invalid_empty_page_table():
    args, kwargs, _, _ = golden.make_inputs(golden.CASES["pa128_ragged"], device())
    kwargs["block_table"] = kwargs["block_table"][:, :0].contiguous()
    with pytest.raises((ValueError, RuntimeError), match="(block_table|page table)"):
        runner()(*args, **kwargs)
