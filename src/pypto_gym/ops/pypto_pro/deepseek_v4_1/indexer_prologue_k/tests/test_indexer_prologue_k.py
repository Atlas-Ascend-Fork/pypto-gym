# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IndexerPrologueK 测试套件。

纯 CPU 的 golden / MXFP4 / cache ABI 校验，以及 NPU 端到端校验（无 NPU 时自动跳过）。
kernel 位于同算子目录的 ``op_kernel/indexer_prologue_k.py``，golden 位于 ``tests/assets/``。
运行：``python3 -m pytest <本文件> -q``。
"""

import math
import os
import struct
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# 加入仓库 src 以导入 PyPTO-Pro kernel（tests/ 上溯 6 级到仓库 src）。
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", "..", ".."))
# golden 位于 tests/assets/。
sys.path.insert(0, os.path.join(_HERE, "assets"))

from pypto_gym.ops.pypto_pro.deepseek_v4_1.indexer_prologue_k.op_kernel import indexer_prologue_k as op  # noqa: E402
import indexer_prologue_k_golden as golden  # noqa: E402

pytestmark = pytest.mark.soc("950")


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
    return device_id


def _make_rope(total_rows, rope_head_dim, generator):
    positions = torch.arange(total_rows, dtype=torch.float32).unsqueeze(1)
    frequencies = 1.0 / (
        40000.0 ** (torch.arange(0, rope_head_dim, 2, dtype=torch.float32) / rope_head_dim)
    )
    base = (positions * frequencies.unsqueeze(0)).repeat_interleave(2, dim=-1)
    angles = base + torch.rand((total_rows, rope_head_dim), generator=generator) * 0.25
    return torch.sin(angles).contiguous(), torch.cos(angles).contiguous()


def _make_inputs(
    total_rows, head_dim, index_head_dim, rope_head_dim, block_num, block_size,
    storage_mode, combined_block_size, with_scale_cache, seed,
    duplicate_indices=False, out_of_order=False, all_skip=False,
):
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(total_rows, head_dim, generator=generator).to(torch.bfloat16)
    wk = (
        torch.randn(index_head_dim, head_dim, generator=generator, dtype=torch.float32)
        / math.sqrt(head_dim)
    ).to(torch.bfloat16)
    norm_weight = torch.randn(index_head_dim, generator=generator, dtype=torch.float32)
    rope_sin, rope_cos = _make_rope(total_rows, rope_head_dim, generator)
    value_bytes = index_head_dim // 2
    scale_bytes = index_head_dim // 32
    if storage_mode == 0:
        cache_shape = (block_num, block_size, 1, value_bytes)
    else:
        cache_shape = (
            block_num,
            block_size // combined_block_size,
            1,
            combined_block_size * (value_bytes + scale_bytes),
        )
    k_cache = torch.randint(0, 256, cache_shape, generator=generator, dtype=torch.uint8)
    k_scale_cache = None
    if storage_mode == 0 and with_scale_cache:
        k_scale_cache = torch.randint(
            0, 256, (block_num, block_size, 1, scale_bytes),
            generator=generator, dtype=torch.uint8,
        )
    slots = torch.randperm(block_num * block_size, generator=generator)[:total_rows].to(torch.int64)
    if total_rows > 1 and seed % 3 == 0:
        slots[-1] = -1
    if duplicate_indices and total_rows > 1:
        slots[1] = slots[0]
        if storage_mode == 1 and total_rows > 3:
            slots[2] = 0
            slots[3] = 1
    if out_of_order:
        slots = slots.flip(0).contiguous()
    if all_skip:
        slots = torch.full((total_rows,), -1, dtype=torch.int64)
    return {
        "latent": latent,
        "wk": wk,
        "norm_weight": norm_weight,
        "rope_sin": rope_sin,
        "rope_cos": rope_cos,
        "k_cache": k_cache,
        "cache_index": slots,
        "k_scale_cache": k_scale_cache,
    }


def _mismatch(actual, expected, name):
    count = int((actual != expected).sum().item())
    first = (actual != expected).nonzero()[:4]
    detail = [(tuple(i.tolist()), int(actual[tuple(i.tolist())]), int(expected[tuple(i.tolist())])) for i in first]
    return f"{name} mismatch count={count}, first={detail}"


def run_device_case(
    total_rows, head_dim, index_head_dim, rope_head_dim, block_num, block_size,
    storage_mode, combined_block_size, with_scale_cache, seed, norm_eps=1e-6,
    duplicate_indices=False, out_of_order=False, all_skip=False,
):
    """在 NPU 上跑一个端到端用例，并与 CPU golden 对比。"""
    inputs = _make_inputs(
        total_rows, head_dim, index_head_dim, rope_head_dim, block_num, block_size,
        storage_mode, combined_block_size, with_scale_cache, seed,
        duplicate_indices, out_of_order, all_skip,
    )
    expected_cache, expected_scale = golden.reference_forward(
        inputs["latent"], inputs["wk"], inputs["norm_weight"], inputs["rope_sin"],
        inputs["rope_cos"], inputs["k_cache"], inputs["cache_index"],
        k_scale_cache=inputs["k_scale_cache"], storage_mode=storage_mode,
        norm_eps=norm_eps, combined_block_size=combined_block_size,
    )
    cache_npu = inputs["k_cache"].npu()
    scale_npu = inputs["k_scale_cache"].npu() if inputs["k_scale_cache"] is not None else None
    kwargs = {} if scale_npu is None else {"k_scale_cache": scale_npu}
    actual = op.indexer_prologue_k_wrapper(
        inputs["latent"].npu(), inputs["wk"].npu(), inputs["norm_weight"].npu(),
        inputs["rope_sin"].npu(), inputs["rope_cos"].npu(), cache_npu, **kwargs,
        cache_index=inputs["cache_index"].npu(), storage_mode=storage_mode,
        norm_eps=norm_eps, combined_block_size=combined_block_size,
    )
    torch.npu.synchronize()
    assert actual is cache_npu
    actual_cache = actual.cpu()
    assert torch.equal(actual_cache, expected_cache), _mismatch(actual_cache, expected_cache, "k_cache")
    if scale_npu is not None:
        assert torch.equal(scale_npu.cpu(), expected_scale), _mismatch(scale_npu.cpu(), expected_scale, "k_scale_cache")
    return actual_cache, expected_cache


# ---------------------------------------------------------------------------
# 纯 CPU 的 golden / MXFP4 / cache ABI 测试
# ---------------------------------------------------------------------------


def _log2_formula_scale(amax):
    safe = max(amax, 6.0 * (2.0 ** -126))
    return float(np.exp2(np.ceil(np.log2(safe / 6.0))))


def test_scale_matches_log2_formula_on_random():
    """纯 CPU：UE8M0 scale 与 log2 公式一致（2000 组随机 amax）。"""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        amax = float(rng.uniform(1e-30, 1e4))
        _, scale = golden.scale_for_amax(amax)
        assert scale == _log2_formula_scale(amax)


def test_scale_byte_round_trip():
    """纯 CPU：scale 的 exponent 字节往返一致（含 0 / 极值）。"""
    for amax in (0.0, 1e-38, 0.125, 0.5, 1.0, 5.0, 6.0, 1e4):
        exp, scale = golden.scale_for_amax(amax)
        assert 0 <= exp <= 0xFF
        bits = struct.unpack("<I", struct.pack("<f", scale))[0]
        assert ((bits >> 23) & 0xFF) == exp
        assert (bits & 0x7FFFFF) == 0


def test_encode_group_known_values():
    """纯 CPU：E2M1 阈值链编码：{6,5,3.5,2.5,1.75,1.25,0.75,0.25} -> 码 {7..0}，scale exp=127。"""
    values = [6.0, 5.0, 3.5, 2.5, 1.75, 1.25, 0.75, 0.25] + [0.0] * 24
    data, exp = golden.encode_group(values)
    codes = []
    for byte in data:
        codes.extend([byte & 0xF, byte >> 4])
    assert codes[:8] == [7, 6, 5, 4, 3, 2, 1, 0]
    assert exp == 127


def test_encode_group_sign_and_extremes():
    """纯 CPU：符号位与极值（-0.0 / ±1e30）编码。"""
    values = [0.0] * 32
    values[0] = -0.0
    values[3] = 1e30
    values[4] = -1e30
    data, _ = golden.encode_group(values)
    codes = []
    for byte in data:
        codes.extend([byte & 0xF, byte >> 4])
    assert all(0 <= code <= 15 for code in codes)
    assert codes[0] == 0  # 参考比较中 -0.0 不 < 0。
    assert codes[3] & 0x8 == 0
    assert codes[4] & 0x8 == 0x8


@pytest.mark.parametrize("value", [6.0, 5.0, 3.5, 2.5, 1.75, 1.25, 0.75, 0.25])
def test_encode_group_threshold_boundaries(value):
    """纯 CPU：E2M1 阈值边界值编码（参数化 8 个阈值）。"""
    values = [value] + [0.0] * 31
    data, _ = golden.encode_group(values)
    assert isinstance(data, (bytes, bytearray))


def test_pack_row_lengths():
    """纯 CPU：D=128 打包后 data=64B、scale=4B（32 元素/组）。"""
    data, scales = golden.pack_mxfp4_row([0.5] * 128)
    assert len(data) == 64
    assert len(scales) == 4


def test_reference_norm_rope_publishes_bf16():
    """纯 CPU：参考 RMSNorm+RoPE 发布边界为 BF16。"""
    torch.manual_seed(0)
    projected = (torch.randn(4, 64) * 2).to(torch.bfloat16)
    gamma = torch.randn(64)
    sin = torch.rand(4, 16)
    cos = torch.rand(4, 16)
    key = golden.reference_norm_rope(projected, gamma, sin, cos, 1e-6)
    assert key.dtype == torch.bfloat16


def test_owner_keys_partition_destinations():
    """纯 CPU：owner_key 分片——mode0（C=64）128 行 -> 128 个唯一目标；mode1（G=8, slots=8）128 行 -> 16 个 combined 行。"""
    mode0 = [golden.owner_key(0, row, 64, 1) for row in range(128)]
    assert len(set(mode0)) == 128
    mode1 = [golden.owner_key(1, row, 8, 8) for row in range(128)]
    # cache_slots=8, G=8 -> 每 block 64 行；128 行 -> 2 block x 8 group。
    assert len(set(mode1)) == 16


def test_scatter_offsets_mode1_layout():
    """纯 CPU：mode1 散布偏移（data_base=5*64、scale_base=8*64+5*4）。"""
    block, dest1, data_base, scale_base = golden.scatter_offsets(1, 5, 8, 8, 128)
    assert block == 0 and dest1 == 0
    assert data_base == 5 * 64
    assert scale_base == 8 * 64 + 5 * 4


# ---------------------------------------------------------------------------
# 投影（Cube）测试
# ---------------------------------------------------------------------------


def _run_projection(total_rows, head_dim, index_head_dim, seed=0):
    _set_device()
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(total_rows, head_dim, generator=generator).to(torch.bfloat16)
    wk = (
        torch.randn(index_head_dim, head_dim, generator=generator, dtype=torch.float32)
        / math.sqrt(head_dim)
    ).to(torch.bfloat16)
    expected = golden.reference_projection(latent, wk)
    cube_cores, _ = op.get_effective_core_counts()
    blocks = op.projection_grid_blocks(total_rows, head_dim, index_head_dim, cube_cores)
    projected = torch.empty(total_rows, index_head_dim, dtype=torch.bfloat16, device=latent.npu().device)
    op.indexer_projection_kernel[None, blocks,
                                 op.projection_tiling_key(index_head_dim, head_dim)](
        latent.npu(), wk.npu(), projected)
    torch.npu.synchronize()
    return projected.cpu(), expected


@requires_npu
@pytest.mark.parametrize("total_rows", [1, 15, 16, 17, 33])
def test_projection_m_tail(total_rows):
    """NPU 投影 M 尾块：T in {1,15,16,17,33}，H=512，D=128（校验尾块 valid_m 与 BF16 逐字节一致）。"""
    actual, expected = _run_projection(total_rows, 512, 128)
    assert torch.equal(actual, expected), (actual.float() - expected.float()).abs().max().item()


@requires_npu
@pytest.mark.parametrize("head_dim,index_head_dim", [(48, 128), (128, 96), (64, 32), (512, 128)])
def test_projection_shape_fallbacks(head_dim, index_head_dim):
    """NPU 投影形状回退：(H,D) in {(48,128),(128,96),(64,32),(512,128)}，T=16（校验 base_n/base_k 回退路径）。"""
    actual, expected = _run_projection(16, head_dim, index_head_dim, seed=7)
    assert torch.equal(actual, expected)


def test_projection_axes_pick_aligned_bases():
    """纯 host：projection_axes 的 (base_n, base_k) 选择（D/H 是否对齐 128/64/16）。"""
    assert op.projection_axes(512, 512)[:2] == (128, 128)
    assert op.projection_axes(128, 512)[:2] == (128, 128)
    assert op.projection_axes(96, 512)[:2] == (16, 128)
    assert op.projection_axes(64, 512)[:2] == (64, 128)
    assert op.projection_axes(128, 48)[:2] == (128, 16)


# ---------------------------------------------------------------------------
# MXFP4 / 后处理 / 端到端测试（NPU）
# ---------------------------------------------------------------------------


@requires_npu
def test_device_encoder_matches_reference_on_random():
    """NPU e2e：T=6，H=512，D=128，Dr=64，blocks=2，block_size=64，mode0 + scale。"""
    cache, expected = run_device_case(6, 512, 128, 64, 2, 64, 0, -1, True, seed=1234)
    assert torch.equal(cache, expected)


@requires_npu
@pytest.mark.parametrize("index_head_dim,rope_head_dim", [(128, 64), (32, 16), (96, 32)])
def test_device_encoder_shapes(index_head_dim, rope_head_dim):
    """NPU e2e：(D,Dr) in {(128,64),(32,16),(96,32)}，T=5，H=512，blocks=2，block_size=64，mode0 + scale。"""
    cache, expected = run_device_case(5, 512, index_head_dim, rope_head_dim, 2, 64, 0, -1, True, seed=55)
    assert torch.equal(cache, expected)


@requires_npu
@pytest.mark.parametrize(
    "rope_head_dim", [2, 16, 62, 64, 122, 128],
    ids=["dr2", "dr16", "dr62", "dr64", "dr122", "dr_full"],
)
def test_norm_rope_boundaries(rope_head_dim):
    """NPU e2e：Dr in {2,16,62,64,122,128}（短尾/全尾），T=13，H=512，D=128，blocks=2，block_size=128，mode0 + scale。"""
    cache, expected = run_device_case(13, 512, 128, rope_head_dim, 2, 128, 0, -1, True, seed=700 + rope_head_dim)
    assert torch.equal(cache, expected)


@requires_npu
def test_norm_rope_small_head_dim():
    """NPU e2e：小头维 T=16，H=64，D=32，Dr=16，blocks=2，block_size=128，mode0 + scale。"""
    cache, expected = run_device_case(16, 64, 32, 16, 2, 128, 0, -1, True, seed=801)
    assert torch.equal(cache, expected)


@requires_npu
def test_duplicate_index_last_token_wins_mode0():
    """NPU e2e：重复 cache_index 取最后写入；T=8，H=512，D=128，Dr=64，blocks=2，block_size=128，mode0 + scale。"""
    cache, expected = run_device_case(8, 512, 128, 64, 2, 128, 0, -1, True, seed=501, duplicate_indices=True)
    assert torch.equal(cache, expected)


@requires_npu
def test_duplicate_index_last_token_wins_mode1():
    """NPU e2e：重复 cache_index 取最后写入（mode1）；T=8，H=512，D=128，Dr=64，blocks=2，block_size=128，G=4。"""
    cache, expected = run_device_case(8, 512, 128, 64, 2, 128, 1, 4, False, seed=502, duplicate_indices=True)
    assert torch.equal(cache, expected)


@requires_npu
def test_out_of_order_indices():
    """NPU e2e：乱序 cache_index；T=10，H=512，D=128，Dr=64，blocks=2，block_size=128，mode0 + scale。"""
    cache, expected = run_device_case(10, 512, 128, 64, 2, 128, 0, -1, True, seed=503, out_of_order=True)
    assert torch.equal(cache, expected)


@requires_npu
def test_skip_all_tokens_leaves_cache_untouched():
    """NPU e2e：cache_index 全为 -1，跳过所有 token、cache 保持不变；T=5，H=512，D=128，Dr=64，blocks=2，block_size=128，mode0 + scale。"""
    cache, expected = run_device_case(5, 512, 128, 64, 2, 128, 0, -1, True, seed=504, all_skip=True)
    assert torch.equal(cache, expected)


@requires_npu
@pytest.mark.parametrize("combined_block_size", [2, 4, 8, 16])
def test_mode1_combined_rows(combined_block_size):
    """NPU e2e：mode1 组合行 G in {2,4,8,16}；T=16，H=512，D=128，Dr=64，blocks=2，block_size=128。"""
    cache, expected = run_device_case(
        16, 512, 128, 64, 2, 128, 1, combined_block_size, False, seed=400 + combined_block_size,
    )
    assert torch.equal(cache, expected)


@requires_npu
def test_untouched_cache_region_sentinel():
    """未命中的 slot 保持原字节；命中的 slot 确实被改写。"""
    inputs = _make_inputs(4, 512, 128, 64, 2, 64, 0, -1, True, seed=909)
    inputs["cache_index"] = torch.tensor([0, 5, 70, -1], dtype=torch.int64)
    k_cache_cpu = inputs["k_cache"].clone()
    cache_npu = inputs["k_cache"].npu()

    op.indexer_prologue_k_wrapper(
        inputs["latent"].npu(), inputs["wk"].npu(), inputs["norm_weight"].npu(),
        inputs["rope_sin"].npu(), inputs["rope_cos"].npu(), cache_npu,
        k_scale_cache=inputs["k_scale_cache"].npu(),
        cache_index=inputs["cache_index"].npu(), storage_mode=0, norm_eps=1e-6,
        combined_block_size=-1,
    )
    torch.npu.synchronize()
    result = cache_npu.cpu()
    assert not torch.equal(result, k_cache_cpu), "operator wrote nothing"

    flat = result.view(-1, result.shape[-1])
    original = k_cache_cpu.view(-1, k_cache_cpu.shape[-1])
    touched = {0, 5, 70}
    for slot in range(flat.shape[0]):
        if slot not in touched:
            assert torch.equal(flat[slot], original[slot])


@requires_npu
@pytest.mark.parametrize("total_rows", [1, 15, 16, 17])
def test_token_boundaries(total_rows):
    """NPU e2e：token 边界 T in {1,15,16,17}；H=512，D=128，Dr=64，blocks=2，block_size=128，mode0 + scale。"""
    cache, expected = run_device_case(total_rows, 512, 128, 64, 2, 128, 0, -1, True, seed=100 + total_rows)
    assert torch.equal(cache, expected)


@requires_npu
@pytest.mark.parametrize(
    "head_dim,index_head_dim,rope_head_dim",
    [(48, 128, 64), (128, 96, 32), (64, 32, 16), (512, 128, 122)],
)
def test_shape_generalisation(head_dim, index_head_dim, rope_head_dim):
    """NPU e2e：形状泛化 (H,D,Dr) in {(48,128,64),(128,96,32),(64,32,16),(512,128,122)}；T=16，blocks=2，block_size=128，mode0 + scale。"""
    cache, expected = run_device_case(16, head_dim, index_head_dim, rope_head_dim, 2, 128, 0, -1, True, seed=211)
    assert torch.equal(cache, expected)


@requires_npu
def test_mode0_without_scale_cache():
    """NPU e2e：mode0 不传 k_scale_cache（仅写 E2M1）；T=9，H=512，D=128，Dr=64，blocks=2，block_size=128。"""
    cache, expected = run_device_case(9, 512, 128, 64, 2, 128, 0, -1, False, seed=301)
    assert torch.equal(cache, expected)


@requires_npu
def test_public_api_returns_same_tensor():
    """NPU：公开接口原地更新并返回同一 k_cache tensor；T=4，H=512，D=128，Dr=64，blocks=2，block_size=64，mode0 + scale。"""
    inputs = _make_inputs(4, 512, 128, 64, 2, 64, 0, -1, True, seed=17)
    cache_npu = inputs["k_cache"].npu()
    returned = op.indexer_prologue_k_wrapper(
        inputs["latent"].npu(), inputs["wk"].npu(), inputs["norm_weight"].npu(),
        inputs["rope_sin"].npu(), inputs["rope_cos"].npu(), cache_npu,
        k_scale_cache=inputs["k_scale_cache"].npu(),
        cache_index=inputs["cache_index"].npu(), storage_mode=0, norm_eps=1e-6,
        combined_block_size=-1,
    )
    assert returned is cache_npu


@requires_npu
def test_large_reference_shape():
    """NPU e2e：模型大规格 T=4，H=7168，D=512，Dr=510，blocks=2，block_size=64，mode0 + scale。"""
    cache, expected = run_device_case(4, 7168, 512, 510, 2, 64, 0, -1, True, seed=900)
    assert torch.equal(cache, expected)


@requires_npu
def test_large_reference_shape_mode1():
    """NPU e2e：模型大规格 + mode1；T=4，H=7168，D=512，Dr=510，blocks=2，block_size=64，G=4。"""
    cache, expected = run_device_case(4, 7168, 512, 510, 2, 64, 1, 4, False, seed=901)
    assert torch.equal(cache, expected)


# ---------------------------------------------------------------------------
# 融合 / 两 kernel 派发一致性（覆盖 T > FUSED_T_MAX 的两 kernel 分支）
# ---------------------------------------------------------------------------


@requires_npu
@pytest.mark.parametrize(
    "storage_mode,group_size,with_scale",
    [
        pytest.param(0, -1, True, id="mode0_with_scale"),
        pytest.param(1, 8, False, id="mode1_g8"),
    ],
)
def test_fused_matches_two_kernel_dispatch(storage_mode, group_size, with_scale):
    """融合与两 kernel 两条路径必须产出逐字节一致的 cache（强制走两个分支）。"""
    total_rows = 4096
    inputs = _make_inputs(
        total_rows, 512, 128, 64, 32, 128, storage_mode, group_size, with_scale, 424242,
    )
    cache_npu = inputs["k_cache"].npu()
    scale_npu = inputs["k_scale_cache"].npu() if inputs["k_scale_cache"] is not None else None
    kwargs = {} if scale_npu is None else {"k_scale_cache": scale_npu}
    tensors = (
        inputs["latent"].npu(), inputs["wk"].npu(), inputs["norm_weight"].npu(),
        inputs["rope_sin"].npu(), inputs["rope_cos"].npu(),
    )

    def run():
        return op.indexer_prologue_k_wrapper(
            *tensors, cache_npu, **kwargs, cache_index=inputs["cache_index"].npu(),
            storage_mode=storage_mode, norm_eps=1e-6, combined_block_size=group_size,
        )

    prev = op.FUSED_T_MAX
    try:
        op.FUSED_T_MAX = 0  # 强制两 kernel 路径
        two = run().clone()
        torch.npu.synchronize()
        op.FUSED_T_MAX = total_rows  # 强制融合路径
        fused = run().clone()
        torch.npu.synchronize()
    finally:
        op.FUSED_T_MAX = prev

    assert torch.equal(two.cpu(), fused.cpu()), _mismatch(two.cpu(), fused.cpu(), "fused vs two")
