#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""IndexerPrologueQw golden tests; --mode selects the JIT or ACLNN entry."""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch_npu

_THIS_DIR = Path(__file__).resolve().parent
_OP_DIR = _THIS_DIR.parent
for _path in (_THIS_DIR, _THIS_DIR / "assets", _OP_DIR / "op_kernel"):
    _path_str = str(_path)
    while _path_str in sys.path:
        sys.path.remove(_path_str)
    sys.path.insert(0, _path_str)

from op_caller import get_mode, get_op_caller  # noqa: E402
import indexer_prologue_qw_golden_cpu as golden  # noqa: E402
import indexer_prologue_qw_precision as verify  # noqa: E402

pytestmark = pytest.mark.soc("950")

_NAMES = ("x", "qr", "wqb", "ww", "descale_qr", "descale_wqb", "rope_sin", "rope_cos")
_FRACTAL_NZ = 29


def _device():
    device = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}")
    torch.npu.set_device(device)
    return device


def _as_e4m3(args):
    out = list(args)
    out[1] = out[1].view(torch.float8_e4m3fn)
    out[2] = out[2].view(torch.float8_e4m3fn)
    return out


def _base_device_args(args, device):
    dev = [a.contiguous().to(device) for a in _as_e4m3(args)]
    dev[2] = torch_npu.npu_format_cast(dev[2], _FRACTAL_NZ)
    dev[3] = torch_npu.npu_format_cast(dev[3], _FRACTAL_NZ)
    return dev


def _as_e8m0(args):
    out = list(args)
    out[4] = out[4].view(torch.float8_e8m0fnu)
    out[5] = out[5].view(torch.float8_e8m0fnu)
    return out


def _case(t, seed, with_want=True):
    cpu_args, kwargs = golden.make_case(t, torch.device("cpu"), seed=seed)
    e4 = _as_e4m3(cpu_args)
    inputs = verify.IndexerPrologueQwInputs(
        softmax_scale=kwargs["softmax_scale"], **dict(zip(_NAMES, e4))
    )
    want = verify.indexer_prologue_qw_golden(inputs) if with_want else None
    return cpu_args, kwargs, inputs, want


@pytest.mark.parametrize(("t", "seed"), [(512, 13), (1024, 13)])
def test_golden(t, seed):
    device = _device()
    cpu_args, kwargs, inputs, want = _case(t, seed)
    mode = get_mode()
    base = _base_device_args(cpu_args, device)
    dev = _as_e8m0(base) if mode == "aclnn" else base
    got = get_op_caller("indexer_prologue_qw")(
        *dev, softmax_scale=kwargs["softmax_scale"]
    )
    torch.npu.synchronize()
    verify.compare(tuple(o.cpu() for o in got), want, inputs)


def _run_default_scale(t, seed, extra_rows=0):
    device = _device()
    cpu_args, kwargs, inputs, want = _case(t, seed)
    if extra_rows:
        extra = torch.randn((extra_rows, 5120), dtype=torch.float32) * 0.1
        cpu_args[0] = torch.cat([cpu_args[0].float(), extra], dim=0).to(torch.bfloat16)
    mode = get_mode()
    base = _base_device_args(cpu_args, device)
    dev = _as_e8m0(base) if mode == "aclnn" else base
    got = get_op_caller("indexer_prologue_qw")(*dev)
    torch.npu.synchronize()
    verify.compare(tuple(o.cpu() for o in got), want, inputs)


def test_softmax_scale_default():
    _run_default_scale(512, 13)


def test_extra_x_rows_ignored():
    _run_default_scale(512, 13, extra_rows=3)


@pytest.mark.skipif(get_mode() != "aclnn", reason="dtype-B contract is checked on the aclnn bridge")
def test_b_dtype_uint8():
    t, seed = 512, 13
    device = _device()
    cpu_args, kwargs, inputs, want = _case(t, seed)
    dev = [a.contiguous().to(device) for a in cpu_args]
    dev[2] = torch_npu.npu_format_cast(dev[2], _FRACTAL_NZ)
    dev[3] = torch_npu.npu_format_cast(dev[3], _FRACTAL_NZ)
    got = get_op_caller("indexer_prologue_qw")(
        *dev, softmax_scale=kwargs["softmax_scale"]
    )
    torch.npu.synchronize()
    verify.compare(tuple(o.cpu() for o in got), want, inputs)


@pytest.mark.skipif(get_mode() != "aclnn", reason="invalid-input checks need the aclnn bridge")
def test_rejects_t0():
    device = _device()
    cpu_args, kwargs, _, _ = _case(4, 3, with_want=False)
    dev = _as_e8m0(_base_device_args(cpu_args, device))
    for idx in (0, 1, 4, 6, 7):
        dev[idx] = dev[idx][:0]
    with pytest.raises(RuntimeError):
        get_op_caller("indexer_prologue_qw")(*dev, softmax_scale=kwargs["softmax_scale"])
        torch.npu.synchronize()


@pytest.mark.skipif(get_mode() != "aclnn", reason="invalid-input checks need the aclnn bridge")
def test_rejects_short_x():
    device = _device()
    cpu_args, kwargs, _, _ = _case(4, 3, with_want=False)
    cpu_args[0] = cpu_args[0][:3]
    dev = _as_e8m0(_base_device_args(cpu_args, device))
    with pytest.raises(RuntimeError):
        get_op_caller("indexer_prologue_qw")(*dev, softmax_scale=kwargs["softmax_scale"])
        torch.npu.synchronize()


@pytest.mark.skipif(get_mode() != "aclnn", reason="invalid-input checks need the aclnn bridge")
def test_rejects_bad_qr_dtype():
    device = _device()
    cpu_args, kwargs, _, _ = _case(4, 3, with_want=False)
    dev = _as_e8m0(_base_device_args(cpu_args, device))
    dev[1] = dev[1].float()
    with pytest.raises(RuntimeError):
        get_op_caller("indexer_prologue_qw")(*dev, softmax_scale=kwargs["softmax_scale"])
        torch.npu.synchronize()


@pytest.mark.skipif(get_mode() != "aclnn", reason="invalid-input checks need the aclnn bridge")
def test_rejects_bad_descale_shape():
    device = _device()
    cpu_args, kwargs, _, _ = _case(4, 3, with_want=False)
    dev = _as_e8m0(_base_device_args(cpu_args, device))
    dev[4] = dev[4][:, :, :1]
    with pytest.raises(RuntimeError):
        get_op_caller("indexer_prologue_qw")(*dev, softmax_scale=kwargs["softmax_scale"])
        torch.npu.synchronize()


@pytest.mark.skipif(get_mode() != "aclnn", reason="invalid-input checks need the aclnn bridge")
def test_rejects_non_nz_wqb():
    device = _device()
    cpu_args, kwargs, _, _ = _case(4, 3, with_want=False)
    dev = _as_e8m0(_base_device_args(cpu_args, device))
    dev[2] = _as_e4m3(cpu_args)[2].contiguous().to(device)
    with pytest.raises(RuntimeError):
        get_op_caller("indexer_prologue_qw")(*dev, softmax_scale=kwargs["softmax_scale"])
        torch.npu.synchronize()
