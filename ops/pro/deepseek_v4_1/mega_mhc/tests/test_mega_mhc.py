# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end operator checks for typical shapes and shape/value boundaries."""
import math

import pytest
import torch
import torch_npu  # Register the NPU backend.

from mega_mhc_golden_cpu import make_inputs, mega_mhc_golden_cpu
from mega_mhc_precision_compare import assert_outputs

import pypto_gym

# pypto_gym is lazy; register torch.ops before calling it directly.
pypto_gym.load()

pytestmark = pytest.mark.soc('950')

# T, H, input distribution, batch override, and one explicitly selected Torch API.
# Each entry is one complete operator call, not a sub-kernel or framework test.
CASES = [
    # One typical shape for each supported H, including the two larger workloads.
    pytest.param(128, 1024, 'random', None, 'python', id='typical_t128_h1024'),
    pytest.param(128, 2048, 'random', None, 'torch', id='typical_t128_h2048'),
    pytest.param(128, 3072, 'random', None, 'python', id='typical_t128_h3072'),
    pytest.param(200, 4096, 'random', None, 'torch', id='typical_t200_h4096'),
    pytest.param(128, 5120, 'random', None, 'python', id='typical_t128_h5120'),
    pytest.param(96, 5120, 'random', None, 'torch', id='typical_t96_h5120'),
    pytest.param(128, 6144, 'random', None, 'torch', id='typical_t128_h6144'),
    pytest.param(1025, 7168, 'random', None, 'python', id='typical_t1025_h7168'),
    # Empty and smallest nonempty token dimensions.
    pytest.param(0, 1024, 'random', None, 'torch', id='empty_t0_h1024'),
    pytest.param(1, 7168, 'random', None, 'python', id='single_t1_h7168'),
    # Partial/full/next token job (32 tokens in the batch16 configuration).
    pytest.param(31, 4096, 'random', None, 'torch', id='small_tail_t31_h4096'),
    pytest.param(32, 4096, 'random', None, 'python', id='small_tail_t32_h4096'),
    pytest.param(33, 4096, 'random', None, 'torch', id='small_tail_t33_h4096'),
    # Partial/full/next block for the two batch16 block capacities.
    pytest.param(255, 4096, 'random', None, 'python', id='block_t255_h4096'),
    pytest.param(256, 4096, 'random', None, 'torch', id='block_t256_h4096'),
    pytest.param(257, 4096, 'random', None, 'python', id='block_t257_h4096'),
    pytest.param(223, 7168, 'random', None, 'torch', id='block_t223_h7168'),
    pytest.param(224, 7168, 'random', None, 'python', id='block_t224_h7168'),
    pytest.param(225, 7168, 'random', None, 'torch', id='block_t225_h7168'),
    # The automatic batch16 -> batch8 shape threshold, including a batch8 tail.
    pytest.param(4095, 7168, 'random', None, 'python', id='dispatch_t4095_h7168'),
    pytest.param(4096, 7168, 'random', None, 'torch', id='dispatch_t4096_h7168'),
    pytest.param(4097, 7168, 'random', None, 'python', id='dispatch_t4097_h7168'),
    # Degenerate RMS inputs: zero and small nonzero values.
    pytest.param(1, 1024, 'zero', None, 'torch', id='zeros_t1_h1024'),
    pytest.param(5, 3072, 'tiny', None, 'python', id='near_zero_t5_h3072'),
    # A caller may still force batch16 at the automatic switch point.
    pytest.param(4096, 7168, 'random', 16, 'torch', id='force_batch16_t4096_h7168'),
]


def guarded(shape, dtype, device):
    base = torch.full((math.prod(shape) + 256,), 37, dtype=dtype, device=device)
    return base[128:-128].view(shape), base


def check_guards(bases):
    for base in bases:
        assert bool((base[:128] == 37).all()), 'prefix guard overwritten'
        assert bool((base[-128:] == 37).all()), 'suffix guard overwritten'


@pytest.mark.parametrize('tokens,hidden,data_mode,batch,api', CASES)
def test_mega_mhc(tokens, hidden, data_mode, batch, api, device):
    cpu = make_inputs(tokens, hidden, seed=2026 + tokens, mode=data_mode)
    expected = mega_mhc_golden_cpu(*cpu)
    input_pairs = [guarded(tuple(value.shape), value.dtype, device) for value in cpu]
    inputs = tuple(pair[0] for pair in input_pairs)
    for value, original in zip(inputs, cpu):
        value.copy_(original)
    # Exactly one functional API call per case; the binding allocates outputs.
    if api == 'python':
        outputs = pypto_gym.mega_mhc(*inputs, batch=batch)
    elif batch is None:
        outputs = torch.ops.pypto_gym.mega_mhc(*inputs)
    else:
        outputs = torch.ops.pypto_gym.mega_mhc(*inputs, batch=batch)
    torch.npu.synchronize()
    assert_outputs(outputs, expected)
    for value, original in zip(inputs, cpu):
        torch.testing.assert_close(value.cpu(), original, rtol=0, atol=0)
    check_guards([base for _, base in input_pairs])


def test_mega_mhc_rejects_internal_storage(device):
    inputs = [value.to(device) for value in make_inputs(33, 1024, seed=2026)]
    inputs[0] = torch_npu.npu_format_cast(inputs[0], 29)  # FRACTAL_NZ
    assert inputs[0].is_contiguous()  # Logical contiguity cannot detect NZ.
    torch.npu.synchronize()

    # Validation must fail on the caller thread, before any ACLNN work is queued.
    with pytest.raises(RuntimeError, match='x must use ND storage format'):
        torch.ops.pypto_gym.mega_mhc(*inputs)
    torch.npu.synchronize()
