# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""engram_hash（PyPTO-Pro）测试套件。

纯 CPU 的 ``make_offsets`` 校验，以及 NPU 校验（无 NPU 时自动跳过）：输出为整型，
与 ``tests/assets/engram_hash_golden.py`` 逐位比较（``torch.equal``），不做容差。
用例规格：L=2、Tb=8、G∈{3,4}、mask 开关，T 覆盖 1 / 尾块 / 多块；并额外覆盖负
multiplier（floor-mod 分支）、mask 三态与 T=0。

运行：
    python3 -m pytest ops/pro/deepseek_v4_1/engram_hash/tests/test_engram_hash_routes.py -q

kernel 位于 ``op_kernel/engram_hash.py``，golden 位于 ``tests/assets/``。
"""

import importlib.util
import os
import sys

import pytest
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_THIS_DIR)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _jit_module():
    name = "gym_pro_engram_hash"
    if name in sys.modules:
        return sys.modules[name]
    return _load_module(name, os.path.join(_OP_DIR, "op_kernel", "engram_hash.py"))


if os.environ.get("GYM_TEST_MODE", "jit") == "jit":
    _jit_module()
golden = _load_module("gym_pro_engram_hash_golden", os.path.join(_THIS_DIR, "assets", "engram_hash_golden.py"))
pytestmark = pytest.mark.soc("950")


def entry(*args):
    from op_caller import get_op_caller
    return get_op_caller("engram_hash", jit_module="gym_pro_engram_hash", aclnn_module="pypto_gym")(*args)


def _npu_available():
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:  # noqa: BLE001
        return False


requires_npu = pytest.mark.skipif(
    not _npu_available(), reason="torch_npu / NPU not available"
)


def _set_device():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    return f"npu:{device_id}"


def _make_inputs(num_tokens, max_ngram, num_layers, tables, seed=0, with_mask=False,
                 signed_multipliers=False):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 100000, (num_tokens, max_ngram), generator=g, dtype=torch.int32)
    if signed_multipliers:
        mult = torch.randint(-100000, 100000, (num_layers, max_ngram), generator=g,
                             dtype=torch.int64)
    else:
        mult = torch.randint(0, 100000, (num_layers, max_ngram), generator=g, dtype=torch.int64)
    vocab = torch.randint(100000, 1000000, (num_layers, max_ngram - 1, tables),
                          generator=g, dtype=torch.int32)
    sizes = vocab.flatten(1)
    offsets = torch.cat((torch.zeros_like(sizes[:, :1]), sizes[:, :-1].cumsum(1, dtype=torch.int32)), dim=1)
    mask = ((torch.rand(num_tokens, generator=g) < 0.5).to(torch.uint8)
            if with_mask else None)
    return ids, mult, vocab, offsets, mask


def _run_case(dev, num_tokens, max_ngram, num_layers=2, tables=8, seed=0,
              with_mask=False, signed_multipliers=False):
    ids, mult, vocab, offsets, mask = _make_inputs(
        num_tokens, max_ngram, num_layers, tables, seed, with_mask, signed_multipliers)
    args = tuple(t.to(dev) for t in (ids, mult, vocab, offsets))
    mask_dev = mask.to(dev) if mask is not None else None
    got = entry(*args, mask_dev)

    want = golden.engram_hash_ref(ids, mult, vocab, offsets, mask)
    assert torch.equal(got.cpu(), want), (got.flatten()[:8], want.flatten()[:8])


def test_make_offsets():
    vocab = torch.tensor([[[3, 5], [7, 11]], [[13, 17], [19, 23]]], dtype=torch.int32)
    want = torch.tensor([[0, 3, 8, 15], [0, 13, 30, 49]], dtype=torch.int32)
    pytest.importorskip("pypto_pro", reason="make_offsets belongs to the source JIT module")
    got = _jit_module().make_offsets(vocab)
    assert got.dtype == torch.int32 and torch.equal(got, want)


@requires_npu
def test_contract_errors():
    dev = _set_device()
    ids, mult, vocab, offsets, _ = _make_inputs(8, 3, 2, 8)
    ids, mult, vocab, offsets = (x.to(dev) for x in (ids, mult, vocab, offsets))
    with pytest.raises((ValueError, RuntimeError), match="multipliers"):
        entry(ids, mult.to(torch.int32), vocab, offsets)
    with pytest.raises((ValueError, RuntimeError), match="offsets"):
        entry(ids, mult, vocab, offsets[:1])
    with pytest.raises((ValueError, RuntimeError), match="vocab_sizes"):
        entry(ids, mult, vocab.reshape(-1), offsets)
    with pytest.raises((ValueError, RuntimeError), match="device"):
        entry(ids.cpu(), mult.to(dev), vocab.to(dev), offsets.to(dev))


@requires_npu
def test_rejects_blocked_ids_and_invalid_mask():
    import torch_npu

    device = _set_device()
    args = tuple(t.to(device) for t in _make_inputs(32, 32, 2, 8)[:4])
    blocked = torch_npu.npu_format_cast(args[0], 29)
    with pytest.raises((ValueError, RuntimeError), match="ngram_token_ids.*ND"):
        entry(blocked, *args[1:])
    mask = torch.full((32,), 256, dtype=torch.int32, device=device)
    with pytest.raises((ValueError, RuntimeError), match="image_token_mask.*bool or uint8"):
        entry(*args, mask)


@requires_npu
@pytest.mark.parametrize("num_tokens", [1, 127, 128, 129, 300, 1024])
@pytest.mark.parametrize("max_ngram", [3, 4])
@pytest.mark.parametrize("with_mask", [False, True])
def test_correctness(num_tokens, max_ngram, with_mask):
    dev = _set_device()
    _run_case(dev, num_tokens, max_ngram, with_mask=with_mask,
              seed=num_tokens + max_ngram)


@requires_npu
@pytest.mark.parametrize("num_tokens,max_ngram,with_mask", [(300, 4, True), (128, 3, False)])
def test_correctness_signed_multipliers(num_tokens, max_ngram, with_mask):
    """负 multiplier 让 xor hash 落到负数区间，压 floor-mod 修正。"""
    dev = _set_device()
    _run_case(dev, num_tokens, max_ngram, with_mask=with_mask,
              signed_multipliers=True, seed=7)


@requires_npu
@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("tokens,ngram,tables", [
    (129, 5, 3), (7, 2, 129), (137, 5, 1), (9, 16, 32), (5, 17, 33),
    (9, 17, 1), (128, 64, 8), (129, 17, 1), (2, 65, 8), (2, 17, 9),
])
def test_dynamic_geometry(with_mask, tokens, ngram, tables):
    _run_case(_set_device(), tokens, ngram, num_layers=3, tables=tables, seed=7,
              signed_multipliers=True, with_mask=with_mask)


@requires_npu
@pytest.mark.parametrize("tokens,ngram", [(256, 4), (1, 3), (1, 4)])
def test_mask_dtype_states(tokens, ngram):
    """mask 的 bool / uint8 / None 三态一致。"""
    dev = _set_device()
    ids, mult, vocab, offsets, mask = _make_inputs(tokens, ngram, 2, 8, seed=3, with_mask=True)
    args = tuple(t.to(dev) for t in (ids, mult, vocab, offsets))
    for mask_value in (mask, 1 - mask):
        want = golden.engram_hash_ref(ids, mult, vocab, offsets, mask_value)
        for dtype in (torch.bool, torch.uint8):
            assert torch.equal(entry(*args, mask_value.to(dtype).to(dev)).cpu(), want), dtype
    want_none = golden.engram_hash_ref(ids, mult, vocab, offsets, None)
    assert torch.equal(entry(*args).cpu(), want_none)


@requires_npu
def test_zero_tokens():
    dev = _set_device()
    ids, mult, vocab, offsets, _ = _make_inputs(0, 3, 2, 8)
    out = entry(*(t.to(dev) for t in (ids, mult, vocab, offsets)))
    assert tuple(out.shape) == (2, 0, 16)
