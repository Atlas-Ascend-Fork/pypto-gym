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
    python3 -m pytest ops/pypto_pro/deepseek_v4_1/engram_hash/tests/test_engram_hash.py -q
    ENGRAM_HASH_BENCH=1 python3 -m pytest <本文件> -q -k benchmark -s   # 性能采集

kernel 位于 ``op_kernel/engram_hash.py``，golden 位于 ``tests/assets/``。
"""

import logging
import os
import sys
import time

import pytest
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OP_DIR = os.path.dirname(_THIS_DIR)
# 加入本测试目录、golden（tests/assets/）与 kernel（op_kernel/）。
for _path in (_THIS_DIR, os.path.join(_THIS_DIR, "assets"),
              os.path.join(_OP_DIR, "op_kernel")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import engram_hash as op  # noqa: E402
import engram_hash_golden as golden  # noqa: E402

pytestmark = pytest.mark.soc("950")

logger = logging.getLogger(__name__)


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
    offsets = op.make_offsets(vocab)
    mask = ((torch.rand(num_tokens, generator=g) < 0.5).to(torch.uint8)
            if with_mask else None)
    return ids, mult, vocab, offsets, mask


def _run_case(dev, num_tokens, max_ngram, num_layers=2, tables=8, seed=0,
              with_mask=False, signed_multipliers=False):
    ids, mult, vocab, offsets, mask = _make_inputs(
        num_tokens, max_ngram, num_layers, tables, seed, with_mask, signed_multipliers)
    args = tuple(t.to(dev) for t in (ids, mult, vocab, offsets))
    mask_dev = mask.to(dev) if mask is not None else None
    got = op.engram_hash(*args, mask_dev)

    want = golden.engram_hash_ref(ids, mult, vocab, offsets, mask)
    assert torch.equal(got.cpu(), want), (got.flatten()[:8], want.flatten()[:8])


def test_make_offsets():
    vocab = torch.tensor([[[3, 5], [7, 11]], [[13, 17], [19, 23]]], dtype=torch.int32)
    want = torch.tensor([[0, 3, 8, 15], [0, 13, 30, 49]], dtype=torch.int32)
    got = op.make_offsets(vocab)
    assert got.dtype == torch.int32 and torch.equal(got, want)


@requires_npu
def test_contract_errors():
    dev = _set_device()
    ids, mult, vocab, offsets, _ = _make_inputs(8, 3, 2, 8)
    with pytest.raises(ValueError, match="multipliers"):
        op.engram_hash(ids, mult.to(torch.int32), vocab, offsets)
    with pytest.raises(ValueError, match="offsets"):
        op.engram_hash(ids, mult, vocab, offsets[:1])
    with pytest.raises(ValueError, match="vocab_sizes"):
        op.engram_hash(ids, mult, vocab.reshape(-1), offsets)
    with pytest.raises(ValueError, match="device"):
        op.engram_hash(ids.cpu(), mult.to(dev), vocab.to(dev), offsets.to(dev))


@requires_npu
@pytest.mark.parametrize("num_tokens", [1, op.TILE - 1, op.TILE, op.TILE + 1, 300, 1024])
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
def test_mask_dtype_states():
    """mask 的 bool / uint8 / None 三态一致。"""
    dev = _set_device()
    ids, mult, vocab, offsets, mask = _make_inputs(256, 4, 2, 8, seed=3, with_mask=True)
    args = tuple(t.to(dev) for t in (ids, mult, vocab, offsets))
    want = golden.engram_hash_ref(ids, mult, vocab, offsets, mask)
    for dtype in (torch.bool, torch.uint8):
        assert torch.equal(op.engram_hash(*args, mask.to(dtype).to(dev)).cpu(), want), dtype
    want_none = golden.engram_hash_ref(ids, mult, vocab, offsets, None)
    assert torch.equal(op.engram_hash(*args).cpu(), want_none)


@requires_npu
def test_zero_tokens():
    dev = _set_device()
    ids, mult, vocab, offsets, _ = _make_inputs(0, 3, 2, 8)
    out = op.engram_hash(*(t.to(dev) for t in (ids, mult, vocab, offsets)))
    assert tuple(out.shape) == (2, 0, 16)


@pytest.mark.skipif(os.environ.get("ENGRAM_HASH_BENCH") != "1",
                    reason="set ENGRAM_HASH_BENCH=1 to run the benchmark cases")
@requires_npu
def test_benchmark_spec(capsys):
    """性能采集：固定用例规格 L=2、Tb=8、mask=False、T∈{512, 8001}、G∈{3,4}。

    带宽口径：``bandwidth = count_bytes(输入+输出) / t_us / 1e3``。
    每个用例 warmup 5 + 10 次，墙钟批量取均值；msprof 采集时按同样顺序发 kernel。

    ``ENGRAM_HASH_FLUSH_MB``（默认 0）：设为 256 时，每次调用前用该大小的缓冲刷新
    L2，此时只发 kernel、不打印墙钟，供 msprof 解析；否则打印墙钟。两种模式的解析都按
    ``Task Start Time`` 排序后每 15 行一组逐 case 归属（先 4 个 mask=False 用例，
    再 4 个 mask 开启用例）。
    """
    dev = _set_device()
    flush_mb = int(os.environ.get("ENGRAM_HASH_FLUSH_MB", "0"))
    cache = (torch.empty(flush_mb * 1024 * 1024 // 4, dtype=torch.int32, device=dev)
             if flush_mb > 0 else None)
    lines = []
    for with_mask in (False, True):
        for num_tokens in (512, 8001):
            for max_ngram in (3, 4):
                ids, mult, vocab, offsets, mask = _make_inputs(
                    num_tokens, max_ngram, 2, 8, seed=0, with_mask=with_mask)
                tensors = [ids, mult, vocab, offsets] + ([mask] if mask is not None else [])
                d = tuple(t.to(dev) for t in tensors)
                nbytes = sum(t.numel() * t.element_size() for t in tensors)
                nbytes += 2 * num_tokens * (max_ngram - 1) * 8 * 4

                def call():
                    return op.engram_hash(*d)

                if cache is not None:
                    for _ in range(5):
                        cache.fill_(0)
                        call()
                    torch.npu.synchronize()
                    for _ in range(10):
                        cache.fill_(0)
                        call()
                    torch.npu.synchronize()
                    lines.append(f"T={num_tokens:<5d} G={max_ngram} M={int(with_mask)}  "
                                 f"flush={flush_mb}MB  (msprof only, no wall print)")
                    continue

                for _ in range(5):
                    call()
                torch.npu.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    call()
                torch.npu.synchronize()
                us = (time.perf_counter() - t0) / 10 * 1e6
                lines.append(f"T={num_tokens:<5d} G={max_ngram} M={int(with_mask)}  "
                             f"{us:8.2f} us  {nbytes / 1e6:6.2f} MB  "
                             f"{nbytes / us / 1e3:7.2f} GB/s")
    with capsys.disabled():
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
        logger.info("  benchmark spec (L=2, Tb=8, flush=%dMB):", flush_mb)
        for line in lines:
            logger.info("  %s", line)
