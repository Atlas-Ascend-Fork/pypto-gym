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

"""The 15 default configurations, each validated with the shared precision standard."""
import logging
import os
from pathlib import Path
import sys

import pytest
import torch
import torch_npu

_THIS_DIR = Path(__file__).resolve().parent
_OP_DIR = _THIS_DIR.parent
for _path in (_THIS_DIR, _THIS_DIR / 'assets', _OP_DIR / 'op_kernel'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import kv_compress_epilog_v2 as impl
import kv_compress_epilog_v2_golden_cpu as golden
from kv_compress_epilog_v2_precision import check_precision


_LOGGER = logging.getLogger(__name__)


_CASES = {
    'default2d_e4m3fn_g32_d128_ones': dict(d=128, mode=2, group=32, dtype=torch.float8_e4m3fn),
    'default2d_e4m3fn_g32_d512_ones': dict(d=512, mode=2, group=32, dtype=torch.float8_e4m3fn),
    'default2d_e5m2_g32_d512_ones': dict(d=512, mode=2, group=32, dtype=torch.float8_e5m2),
    'default2d_fp4_g32_d128_ones': dict(d=128, mode=4, group=32, dtype=torch.uint8),
    'default2d_fp4_g32_d512_ones': dict(d=512, mode=4, group=32, dtype=torch.uint8),
    'default2d_fp4_g16_d16_ones': dict(d=16, mode=4, group=16, dtype=torch.uint8),
    'default2d_fp4_g16_d32_ones': dict(d=32, mode=4, group=16, dtype=torch.uint8),
    'default2d_fp4_g16_d48_ones': dict(d=48, mode=4, group=16, dtype=torch.uint8),
    'default2d_fp4_g16_d64_ones': dict(d=64, mode=4, group=16, dtype=torch.uint8),
    'default2d_fp4_g16_d96_ones': dict(d=96, mode=4, group=16, dtype=torch.uint8),
    'default2d_fp4_g16_d128_ones': dict(d=128, mode=4, group=16, dtype=torch.uint8),
    'default2d_fp4_g16_d512_ones': dict(d=512, mode=4, group=16, dtype=torch.uint8),
    'default4d_e4m3fn_g32_d512_ones': dict(d=512, mode=2, group=32, dtype=torch.float8_e4m3fn, layout4=True),
    'default4d_fp4_g32_d512_ones': dict(d=512, mode=4, group=32, dtype=torch.uint8, layout4=True),
    'default4d_fp4_g16_d512_ones': dict(d=512, mode=4, group=16, dtype=torch.uint8, layout4=True),
}


def _run_case(name):
    """Run the public wrapper and compare the entire guarded cache backing."""
    case = _CASES[name]
    inputs = golden.make_inputs(case)
    expected = golden.reference(inputs, case)
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))
    torch.npu.set_device(device_id)
    device = torch.device(f'npu:{device_id}')
    raw = inputs['initial'].to(device)
    cache = raw[golden.GUARD_BYTES:].view(case['dtype']).as_strided(inputs['shape'], inputs['strides'])
    x, slots = inputs['x'].to(device), inputs['slots'].to(device)
    original_shape, original_stride = tuple(cache.shape), tuple(cache.stride())
    returned = impl.kv_compress_epilog_v2_wrapper(
        cache, x, slots, case['group'], case['mode'], True,
        1.0, inputs['block_stride'])
    torch.npu.synchronize()
    assert returned is cache
    assert tuple(cache.shape) == original_shape and tuple(cache.stride()) == original_stride
    assert cache.dtype == case['dtype']
    passed, summary = check_precision(raw.cpu(), expected)
    if not passed:
        raise AssertionError(f'{name} precision failed: {summary}')
    _LOGGER.info('PASS %s: %s', name, summary)


@pytest.mark.soc("950")
def test_default2d_e4m3fn_g32_d128_ones():
    _run_case('default2d_e4m3fn_g32_d128_ones')


@pytest.mark.soc("950")
def test_default2d_e4m3fn_g32_d512_ones():
    _run_case('default2d_e4m3fn_g32_d512_ones')


@pytest.mark.soc("950")
def test_default2d_e5m2_g32_d512_ones():
    _run_case('default2d_e5m2_g32_d512_ones')


@pytest.mark.soc("950")
def test_default2d_fp4_g32_d128_ones():
    _run_case('default2d_fp4_g32_d128_ones')


@pytest.mark.soc("950")
def test_default2d_fp4_g32_d512_ones():
    _run_case('default2d_fp4_g32_d512_ones')


@pytest.mark.soc("950")
def test_default2d_fp4_g16_d16_ones():
    _run_case('default2d_fp4_g16_d16_ones')


@pytest.mark.soc("950")
def test_default2d_fp4_g16_d32_ones():
    _run_case('default2d_fp4_g16_d32_ones')


@pytest.mark.soc("950")
def test_default2d_fp4_g16_d48_ones():
    _run_case('default2d_fp4_g16_d48_ones')


@pytest.mark.soc("950")
def test_default2d_fp4_g16_d64_ones():
    _run_case('default2d_fp4_g16_d64_ones')


@pytest.mark.soc("950")
def test_default2d_fp4_g16_d96_ones():
    _run_case('default2d_fp4_g16_d96_ones')


@pytest.mark.soc("950")
def test_default2d_fp4_g16_d128_ones():
    _run_case('default2d_fp4_g16_d128_ones')


@pytest.mark.soc("950")
def test_default2d_fp4_g16_d512_ones():
    _run_case('default2d_fp4_g16_d512_ones')


@pytest.mark.soc("950")
def test_default4d_e4m3fn_g32_d512_ones():
    _run_case('default4d_e4m3fn_g32_d512_ones')


@pytest.mark.soc("950")
def test_default4d_fp4_g32_d512_ones():
    _run_case('default4d_fp4_g32_d512_ones')


@pytest.mark.soc("950")
def test_default4d_fp4_g16_d512_ones():
    _run_case('default4d_fp4_g16_d512_ones')


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    test_default2d_e4m3fn_g32_d128_ones()
    test_default2d_e4m3fn_g32_d512_ones()
    test_default2d_e5m2_g32_d512_ones()
    test_default2d_fp4_g32_d128_ones()
    test_default2d_fp4_g32_d512_ones()
    test_default2d_fp4_g16_d16_ones()
    test_default2d_fp4_g16_d32_ones()
    test_default2d_fp4_g16_d48_ones()
    test_default2d_fp4_g16_d64_ones()
    test_default2d_fp4_g16_d96_ones()
    test_default2d_fp4_g16_d128_ones()
    test_default2d_fp4_g16_d512_ones()
    test_default4d_e4m3fn_g32_d512_ones()
    test_default4d_fp4_g32_d512_ones()
    test_default4d_fp4_g16_d512_ones()
    _LOGGER.info('PASS all 15 default performance cases')


if __name__ == '__main__':
    main()
