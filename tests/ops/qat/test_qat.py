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
"""
QAT Test Module

This module provides test cases and golden reference implementations for QAT operators.

Main Functions:
    - create_asymmetric_qat_golden: Golden reference for asymmetric per-group quantization
    - create_symmetric_qat_nscale_golden: Golden reference for symmetric per-channel quantization
    - create_symmetric_qat_golden: Golden reference for symmetric per-tensor quantization
    - forward_test: Generic forward test interface
    - backward_test_autograd: Generic backward test interface

Example:
    python ai_infra_pypto_qat.py
"""

import logging
import math
import os
import re
from typing import Any, Tuple, Union
import numpy as np
import pytest
import torch
import torch_npu

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))

from numpy.testing import assert_allclose

from qat.qat_impl import (
    ai_infra_qat_asymmetric_per_group,
    ai_infra_qat_asymmetric_per_group_backward,
    ai_infra_qat_symmetric_per_channel,
    ai_infra_qat_symmetric_per_channel_backward,
    ai_infra_qat_symmetric_per_tensor,
    ai_infra_qat_symmetric_per_tensor_backward,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DISTRIBUTION = [
    "uniform_large",
]


# ---------------------------------------------------------------------------
# Helper functions for each distribution type
# ---------------------------------------------------------------------------

def _uniform(shape: Tuple[int, ...], low: float, high: float) -> torch.Tensor:
    """Return a float32 tensor with uniform distribution in [low, high]."""
    return torch.empty(shape, dtype=torch.float32).uniform_(low, high)


def _normal(shape: Tuple[int, ...]) -> torch.Tensor:
    """Return a float32 tensor drawn from a normal distribution.

    μ is sampled uniformly from [-100, 100] and σ from [1, 25], matching the
    original test configuration.
    """
    mu = np.random.uniform(-100, 100)
    sigma = np.random.uniform(1, 25)
    return torch.randn(shape, dtype=torch.float32) * sigma + mu


def _outlier(shape: Tuple[int, ...]) -> torch.Tensor:
    """Generate a normal tensor and inject outliers.

    0.1% of the elements are multiplied by 1000 to create extreme values.
    """
    tensor = _normal(shape)
    mask = torch.rand(shape) < 0.001
    tensor[mask] *= 1000.0
    return tensor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_input(
    shape: Union[Tuple[int, ...], list],
    dtype: str = "float32",
    device: str = "cpu",
    distribution: str = "uniform_large",
    seed: int = 33,
) -> torch.Tensor:
    """Generate a tensor according to the requested distribution.

    Parameters
    ----------
    shape : tuple or list of ints
        Desired tensor shape, e.g. (n, m).
    dtype : str, optional
        Torch dtype name (default "float32").
    device : str, optional
        Device string understood by ``torch.device`` (default "cpu").
    distribution : str, optional
        Supported formats:
        - "uniform[low,high]" where low and high are floats, e.g., "uniform[-5,5]" or "uniform[-0.001,0.001]"
        - "normal"
        - "outlier"
        The default is "uniform_small" (equivalent to "uniform[-0.001,0.001]").

    Returns
    -------
    torch.Tensor
        Tensor on the requested device with the requested dtype.
    """
    torch.manual_seed(seed)

    # Normalise shape input
    if isinstance(shape, list):
        shape = tuple(shape)
    if not isinstance(shape, tuple):
        raise TypeError("shape must be a tuple or list of ints")

    # Choose generation function based on distribution
    if distribution is None:
        distribution = "uniform_large"

    if distribution.startswith("uniform["):
        match = re.match(r"uniform\[\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*\]$", distribution)
        if not match:
            raise ValueError(
                f"Invalid uniform distribution spec '{distribution}'. Expected format uniform[low,high]"
            )
        low, high = map(float, match.groups())
        tensor_fp32 = _uniform(shape, low, high)
    elif distribution == "uniform_small":
        tensor_fp32 = _uniform(shape, -0.001, 0.001)
    elif distribution == "uniform_large":
        tensor_fp32 = _uniform(shape, -5.0, 5.0)
    elif distribution == "normal":
        tensor_fp32 = _normal(shape)
    elif distribution == "outlier":
        tensor_fp32 = _outlier(shape)
    else:
        raise ValueError(
            f"Unsupported distribution '{distribution}'. "
            "Supported: uniform[low,high], uniform_small, uniform_large, normal, outlier"
        )

    tensor = tensor_fp32.to(dtype)

    try:
        tensor = tensor.to(device)
    except Exception as exc:
        raise RuntimeError(f"Failed to move tensor to device '{device}'.") from exc

    return tensor


# Small value threshold
small_value_thres_dict = {
    torch.float16: 2**-11,
    torch.bfloat16: 2**-8,
    torch.float32: 2**-14,
    torch.uint8: 2**-4, torch.float8_e4m3fn: 2**-4
}


# Small value error threshold
small_value_error_thres_dict = {
    torch.float16: 2**-16,
    torch.bfloat16: 2**-16,
    torch.float32: 2**-30,
    torch.uint8: 2**-6, torch.float8_e4m3fn: 2**-6
}


def get_split_index(golden_data, dtype):
    thres = small_value_thres_dict[dtype]
    large_mask = torch.abs(golden_data) >= thres
    small_mask = torch.abs(golden_data) < thres
    return large_mask, small_mask, thres


def compute_matrix_small_value(input_data, golden_data, dtype, small_mask):
    if not torch.any(small_mask):
        return 0
    thres = small_value_error_thres_dict[dtype]

    error_count = torch.sum(torch.abs(input_data[small_mask] - golden_data[small_mask]) > thres).item()
    return error_count


def compute_matrix_large_value(input_data, golden_data, large_mask):
    if not torch.any(large_mask):
        return 0, 0, 0

    input_large = input_data[large_mask]
    golden_large = golden_data[large_mask]

    abs_diff = torch.abs(input_large - golden_large)
    relative_error = abs_diff / (torch.abs(golden_large) + 1e-7)

    mare = torch.max(relative_error).item()
    mere = torch.mean(relative_error).item()
    rmse = torch.sqrt(torch.mean((input_large - golden_large) ** 2)).item()

    return mare, mere, rmse


def compute_re_matrix(input_value, bm_value, small_value_thres):
    if math.isinf(bm_value) or math.isnan(bm_value):
        return 1
    if math.isinf(input_value) or math.isnan(input_value):
        return 1000
    return input_value / max(bm_value, small_value_thres)


def compute_re_triplet_matrix(npu_matrix, golden_matrix, small_value_thres):
    mare_npu, mere_npu, rmse_npu = npu_matrix
    mare_bm, mere_bm, rmse_bm = golden_matrix
    mare_matrix = compute_re_matrix(mare_npu, mare_bm, small_value_thres)
    mere_matrix = compute_re_matrix(mere_npu, mere_bm, small_value_thres)
    rmse_matrix = compute_re_matrix(rmse_npu, rmse_bm, small_value_thres)
    return mare_matrix, mere_matrix, rmse_matrix


def precision_compare_triple(pto_data, bm_data, golden_data, thres=(2, 1.2, 1.2)):
    logger.info(f"{pto_data.dtype=}")
    logger.info(f"{bm_data.dtype=}")
    logger.info(f"{golden_data.dtype=}")
    dtype = pto_data.dtype
    if dtype in ["int8", "int32"]:
        raise NotImplementedError("precision compare triplet only support float")

    if dtype == torch.uint8:
        pto_data = torch_npu.npu_dtype_cast(pto_data, torch.float32, input_dtype=torch_npu.hifloat8)
        bm_data = torch_npu.npu_dtype_cast(bm_data, torch.float32, input_dtype=torch_npu.hifloat8)
        golden_data = torch_npu.npu_dtype_cast(golden_data, torch.float32, input_dtype=torch_npu.hifloat8)
    else:
        pto_data = pto_data.to(torch.float32)
        bm_data = bm_data.to(torch.float32)
        golden_data = golden_data.to(torch.float32)

    pto_data = pto_data.cpu()
    bm_data = bm_data.cpu()
    golden_data = golden_data.cpu()

    large_value_idx, small_value_idx, small_value_thres = get_split_index(golden_data, dtype)

    # Small value scenario
    npu_error_count = compute_matrix_small_value(pto_data, golden_data, dtype, small_value_idx)
    bm_error_count = compute_matrix_small_value(bm_data, golden_data, dtype, small_value_idx)
    small_value_matrix = npu_error_count / max(bm_error_count, 1)

    # Large value scenario
    mare_npu, mere_npu, rmse_npu = compute_matrix_large_value(pto_data, golden_data, large_value_idx)
    mare_bm, mere_bm, rmse_bm = compute_matrix_large_value(bm_data, golden_data, large_value_idx)
    mare_matrix, mere_matrix, rmse_matrix = compute_re_triplet_matrix(
        [mare_npu, mere_npu, rmse_npu], [mare_bm, mere_bm, rmse_bm], small_value_thres)

    is_mare_acceptable = mare_matrix <= thres[0]
    is_mere_acceptable = mere_matrix <= thres[1]
    is_rmse_acceptable = rmse_matrix <= thres[2]

    if all([
        small_value_matrix <= 2,
        is_mare_acceptable,
        is_mere_acceptable,
        is_rmse_acceptable
    ]):
        result = "PASS"
    else:
        result = "FAILED"

    return result, mare_matrix, mere_matrix, rmse_matrix, small_value_matrix


def compare(pto_grad_w, bm_grad_w, golden_grad_w):
    result, mare_matrix, mere_matrix, rmse_matrix, small_value_matrix = precision_compare_triple(
        pto_grad_w, bm_grad_w, golden_grad_w)
    logger.info(f"  precision result: {result}")
    logger.info(f"  mare_matrix: {mare_matrix}")
    logger.info(f"  mere_matrix: {mere_matrix}")
    logger.info(f"  rmse_matrix: {rmse_matrix}")
    logger.info(f"  small_value_matrix: {small_value_matrix}")
    if result != "PASS":
        raise Exception("fail precision check")
    return result, mare_matrix, mere_matrix, rmse_matrix, small_value_matrix


def clone_inputs(inputs):
    """Clone inputs and preserve requires_grad"""
    new_inputs = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            y = x.detach().clone()
            y.requires_grad_(x.requires_grad)
            new_inputs.append(y)
        else:
            new_inputs.append(x)
    return tuple(new_inputs)


def to_double_inputs(inputs):
    new_inputs = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            y = x.detach().cpu().double()
            y.requires_grad_(x.requires_grad)
            new_inputs.append(y)
        else:
            new_inputs.append(x)
    return tuple(new_inputs)


def _to_double_cpu_backward(inputs):
    new_inputs = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            new_inputs.append(x.detach().cpu().double().requires_grad_(x.requires_grad))
        else:
            new_inputs.append(x)
    return tuple(new_inputs)


def normalize_outputs(out):
    if isinstance(out, (tuple, list)):
        return tuple(out)
    return (out,)


def collect_grads(inputs):
    grads = []

    for x in inputs:
        if isinstance(x, torch.Tensor) and x.requires_grad:
            if x.grad is None:
                raise Exception(f"[ERROR]grad is None, shape={tuple(x.shape)}")
            else:
                grads.append(x.grad.detach().clone())
        else:
            grads.append(None)

    return grads


def forward_test(inputs: Tuple[Any, ...], pto_inputs, golden_func, pto_func):
    """
    Generic forward test interface.

    Parameters
    ----------
    inputs : tuple
        All inputs (tensor + non-tensor)

    golden_func : callable
        Reference implementation
        golden_func(*inputs, is_golden=False)
        golden_func(*inputs, is_golden=True)

    pto_func : callable
        Kernel under test
        pto_func(*inputs)

    Constraint
    ----------
    golden_func last parameter must be is_golden
    """

    # benchmark inputs
    bm_inputs = clone_inputs(inputs)

    # golden inputs
    golden_inputs = to_double_inputs(inputs)

    # forward
    bm_out = normalize_outputs(golden_func(*bm_inputs, is_golden=False))
    golden_out = normalize_outputs(golden_func(*golden_inputs, is_golden=True))
    kernel_out = normalize_outputs(pto_func(*pto_inputs))

    assert len(bm_out) == len(kernel_out)

    compare_results = []
    for i, _ in enumerate(bm_out):
        logger.info(f"=== Forward Output[{i}] ===")
        try:
            assert_allclose(kernel_out[i].float().cpu(), bm_out[i].float().cpu(), rtol=1e-3, atol=1e-3)
        except Exception as e:
            logger.error(e)
        result = compare(
            kernel_out[i],
            bm_out[i],
            golden_out[i]
        )
        compare_results.append(result)
    return compare_results


def backward_test_autograd(inputs, pto_inputs, golden_func, pto_func):
    """
    Constraint:
    1. inputs passes all input tensors and other parameters
    2. golden_func last parameter is is_golden
    3. pto_func first parameter is grad_outputs, rest are gradients of corresponding parameters in golden_func
    """

    # ======================
    # benchmark forward
    # ======================

    bm_inputs = clone_inputs(inputs)

    bm_out = golden_func(*bm_inputs, is_golden=False)
    bm_out = normalize_outputs(bm_out)
    bm_out = tuple(o.to(torch.bfloat16) for o in bm_out)
    # Generate grad_outputs
    grad_outputs = tuple(torch.randn(o.shape, device=o.device, dtype=o.dtype) for o in bm_out)

    # ======================
    # benchmark backward
    # ======================

    torch.autograd.backward(bm_out, grad_outputs)

    bm_grads = collect_grads(bm_inputs)

    # ======================
    # golden forward
    # ======================

    golden_inputs = _to_double_cpu_backward(inputs)

    golden_out = golden_func(*golden_inputs, is_golden=True)
    golden_out = normalize_outputs(golden_out)

    grad_outputs_golden = tuple(g.detach().cpu().double() for g in grad_outputs)

    # ======================
    # golden backward
    # ======================

    torch.autograd.backward(golden_out, grad_outputs_golden)

    golden_grads = collect_grads(golden_inputs)

    # ======================
    # pto kernel
    # ======================

    if len(grad_outputs) == 1:
        pto_grads = pto_func(grad_outputs[0], *pto_inputs)
    else:
        pto_grads = pto_func(*grad_outputs, *pto_inputs)

    if not isinstance(pto_grads, (tuple, list)):
        pto_grads = (pto_grads,)

    # ======================
    # compare
    # ======================

    compare_results = []
    idx = 0
    for i, x in enumerate(inputs):
        if isinstance(x, torch.Tensor) and x.requires_grad:

            logger.info(f"=== compare grad of input[{i}] ===")

            pto_grad = pto_grads[idx]
            bm_grad = bm_grads[i]
            golden_grad = golden_grads[i]
            try:
                assert_allclose(pto_grad.float().cpu(), bm_grad.float().cpu(), rtol=1e-3, atol=1e-3)
            except Exception as e:
                logger.error(e)
            result = compare(pto_grad, bm_grad, golden_grad)
            idx += 1
            compare_results.append(result)
    return compare_results


# ---------------------------------------------------------------------------
# Golden Reference Implementations
# ---------------------------------------------------------------------------

def create_asymmetric_qat_golden(group_size, bit, eps=1e-4, clip_val=0.99):

    def asymmetric_qat_golden(weight, scale, offset, is_golden=False):
        """PyTorch reference implementation for Enhanced LSQ+ asymmetric quantization (BF16 I/O, FP32 compute).

        Args:
            weight: Input weight tensor (n, m) in BF16
            scale: Quantization scale tensor (num_groups, 1) in BF16
            offset: Quantization offset tensor (num_groups, 1) in BF16
            group_size: Number of elements per group (default: 128)
            bit: Quantization bit-width (2, 3, or 4)
        """
        if not is_golden:
            weight_in = weight.float()
            scale_in = scale.float()
            offset_in = offset.float()
        else:
            weight_in = weight.double()
            scale_in = scale.double()
            offset_in = offset.double()

        if weight_in.ndim != 2:
            raise ValueError(f"weight must be 2D (n, m), got shape {tuple(weight.shape)}")
        if weight_in.shape[1] % group_size != 0:
            raise ValueError(
                f"weight.shape[1] must be divisible by group_size={group_size}, got m={weight_in.shape[1]}"
            )

        eps_tensor = torch.tensor(eps, device=scale_in.device, dtype=torch.float32)
        protected_scale = torch.where(scale_in > eps_tensor, scale_in, eps_tensor)

        orig_shape = weight.shape
        num_groups = weight.numel() // group_size
        expected_group_shape = (num_groups, 1)
        if tuple(scale.shape) != expected_group_shape:
            raise ValueError(f"scale must have shape {expected_group_shape}, got {tuple(scale.shape)}")
        if tuple(offset.shape) != expected_group_shape:
            raise ValueError(f"offset must have shape {expected_group_shape}, got {tuple(offset.shape)}")

        weight_2d = weight_in.view(num_groups, group_size)

        n_levels = 2 ** (bit - 1)
        shift = 0.5

        weight_shifted = weight_2d - offset_in
        alpha = protected_scale * n_levels

        weight_clipped = torch.clamp(weight_shifted / alpha, -clip_val, clip_val) * n_levels - shift
        weight_rounded = (weight_clipped.round() - weight_clipped).detach() + weight_clipped
        weight_unshifted = weight_rounded + shift
        weight_denorm = weight_unshifted / n_levels
        output_2d = weight_denorm * alpha + offset_in

        output = output_2d.view(orig_shape)
        if is_golden:
            return output
        else:
            return output.to(torch.bfloat16)

    return asymmetric_qat_golden


def create_symmetric_qat_nscale_golden(eps, min_v, max_v):

    def symmetric_qat_golden(weight, scale, is_golden=False):
        """PyTorch reference implementation for embedding head quantization (BF16 I/O, FP32 compute).

        Args:
            weight: Input weight tensor (n, m) in BF16
            scale: Quantization scale tensor (n, 1) in BF16
            eps: Minimum scale threshold (default: 1e-4)
            min_v: Quantization lower bound (default: -128.0)
            max_v: Quantization upper bound (default: 127.0)

        Returns:
            Tuple of (quantized_weight, clamped, protected_scale) all in BF16
        """
        if not is_golden:
            weight_in = weight.float()
            scale_in = scale.float()
        else:
            weight_in = weight.to(torch.float64)
            scale_in = scale.to(torch.float64)
        eps_tensor = torch.tensor(eps, device=weight.device, dtype=torch.float64)
        protected_scale = torch.where(scale_in > eps_tensor, scale_in, eps_tensor)
        weight_normalized = weight_in / protected_scale
        weight_rounded = (weight_normalized.round() - weight_normalized).detach() + weight_normalized
        clamped = torch.clamp(weight_rounded, min_v, max_v)
        output = clamped * protected_scale

        if is_golden:
            return output
        else:
            return output.to(torch.bfloat16)

    return symmetric_qat_golden


def create_symmetric_qat_golden(eps, min_v, max_v):

    def symmetric_qat_golden(weight, scale, is_golden=False):
        """PyTorch reference implementation for embedding head quantization (BF16 I/O, FP32 compute).

        Args:
            weight: Input weight tensor (n, m) in BF16
            scale: Quantization scale tensor (1, 1) scalar in BF16
            eps: Minimum scale threshold (default: 1e-4)
            min_v: Quantization lower bound (default: -128.0)
            max_v: Quantization upper bound (default: 127.0)

        Returns:
            Tuple of (quantized_weight, clamped, protected_scale) all in BF16
        """
        if not is_golden:
            weight_in = weight.float()
            scale_in = scale.float()
        else:
            weight_in = weight.double()
            scale_in = scale.double()
        eps_tensor = torch.tensor(eps, device=scale_in.device, dtype=scale_in.dtype)
        protected_scale = torch.where(scale_in > eps_tensor, scale_in, eps_tensor)
        weight_normalized = weight_in / protected_scale
        # STE
        weight_rounded = weight_normalized + (weight_normalized.round() - weight_normalized).detach()
        clamped = torch.clamp(weight_rounded, min_v, max_v)
        output = clamped * protected_scale

        if is_golden:
            return output
        else:
            return output.to(torch.bfloat16)

    return symmetric_qat_golden


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

# ==================== Asymmetric Per-Group Tests ====================

def run_asymmetric_per_group_test(n, m, group_size, bit, eps, clip_val, distribution, device_id):
    """Run a single test case for asymmetric per-group quantization."""
    device = f"npu:{device_id}"
    seed = 33
    weight_shape = (n, m)
    groups_per_row = m // group_size
    num_groups = n * groups_per_row
    scale_shape = (num_groups, 1)
    offset_shape = (num_groups, 1)

    weight_bm = create_input(weight_shape, torch.bfloat16, device, distribution, seed)
    scale_bm = create_input(scale_shape, torch.bfloat16, device, distribution, seed)
    offset_bm = create_input(offset_shape, torch.bfloat16, device, distribution, seed)

    golden_inputs = [weight_bm, scale_bm, offset_bm]
    pto_inputs = [weight_bm, scale_bm, offset_bm, group_size, bit, eps, clip_val]
    golden = create_asymmetric_qat_golden(group_size, bit, eps, clip_val)
    return forward_test(golden_inputs, pto_inputs, golden, ai_infra_qat_asymmetric_per_group)


@pytest.mark.parametrize(
    ('n', 'm', 'group', 'bit', 'eps', 'clip_val'),
    [
        pytest.param(1024, 2048, 128, 2, 0.0001, 0.99,
                     id="N1024-M2048-group128-bit2-eps0.0001-clip_val0.99",
                     marks=pytest.mark.skip(reason="temporarily disabled")),
        pytest.param(768, 2048, 128, 3, 0.0001, 0.99,
                     id="N768-M2048-group128-bit3-eps0.0001-clip_val0.99"),
    ]
)
def test_asymmetric_per_group(n, m, group, bit, eps, clip_val) -> None:
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    results = []
    for dis in DISTRIBUTION:
        compare_result = run_asymmetric_per_group_test(n, m, group, bit, eps, clip_val, dis, device_id)
        flattened_result = [str(item) for sublist in compare_result for item in sublist]
        str_params = [str(param) for param in [n, m, group, bit, eps, clip_val, dis]]
        results.append(str_params + flattened_result)


# ==================== Symmetric Per-Channel Tests ====================

def run_symmetric_per_channel_test(n, m, bit, eps, distribution, device_id):
    device = f"npu:{device_id}"
    seed = 33
    min_v = float(-2 ** (bit - 1))
    max_v = float(2 ** (bit - 1) - 1)
    weight_shape = (n, m)
    scale_shape = (n, 1)
    weight = create_input(weight_shape, torch.bfloat16, device, distribution, seed)
    scale = create_input(scale_shape, torch.bfloat16, device, distribution, seed)
    golden_inputs = [weight, scale]
    pto_inputs = [weight, scale, eps, min_v, max_v]
    golden = create_symmetric_qat_nscale_golden(eps, min_v, max_v)
    return forward_test(golden_inputs, pto_inputs, golden, ai_infra_qat_symmetric_per_channel)


@pytest.mark.parametrize(
    ('n', 'm', 'bit', 'eps'),
    [
        pytest.param(153376, 2048, 4, 0.0001,
                     id="N153376-M2048-bit4-eps0.0001",
                     marks=pytest.mark.skip(reason="temporarily disabled")),
        pytest.param(38344, 2048, 4, 0.0001,
                     id="N38344-M2048-bit4-eps0.0001"),
    ]
)
def test_symmetric_per_channel(n, m, bit, eps) -> None:
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    results = []
    for dis in DISTRIBUTION:
        compare_result = run_symmetric_per_channel_test(n, m, bit, eps, dis, device_id)
        flattened_result = [str(item) for sublist in compare_result for item in sublist]
        str_params = [str(param) for param in [n, m, bit, eps, dis]]
        results.append(str_params + flattened_result)


# ==================== Symmetric Per-Tensor Tests ====================

def run_symmetric_per_tensor_test(n, m, bit, eps, distribution, device_id):
    device = f"npu:{device_id}"
    seed = 33
    min_v = float(-2 ** (bit - 1))
    max_v = float(2 ** (bit - 1) - 1)
    weight_shape = (n, m)
    scale_shape = (1, 1)
    weight = create_input(weight_shape, torch.bfloat16, device, distribution, seed)
    scale = create_input(scale_shape, torch.bfloat16, device, distribution, seed)
    golden_inputs = [weight, scale]
    pto_inputs = [weight, scale, eps, min_v, max_v]
    golden = create_symmetric_qat_golden(eps, min_v, max_v)
    return forward_test(golden_inputs, pto_inputs, golden, ai_infra_qat_symmetric_per_tensor)


@pytest.mark.parametrize(
    ('n', 'm', 'bit', 'eps'),
    [
        pytest.param(153376, 2048, 8, 0.0001,
                     id="N153376-M2048-bit8-eps0.0001",
                     marks=pytest.mark.skip(reason="temporarily disabled")),
        pytest.param(38344, 2048, 8, 0.0001,
                     id="N38344-M2048-bit8-eps0.0001"),
    ]
)
def test_symmetric_per_tensor(n, m, bit, eps) -> None:
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    results = []
    for dis in DISTRIBUTION:
        compare_result = run_symmetric_per_tensor_test(n, m, bit, eps, dis, device_id)
        flattened_result = [str(item) for sublist in compare_result for item in sublist]
        str_params = [str(param) for param in [n, m, bit, eps, dis]]
        results.append(str_params + flattened_result)


if __name__ == "__main__":
    test_asymmetric_per_group(1024, 2048, 128, 2, 0.0001, 0.99)


# ---------------------------------------------------------------------------
# Backward Test Cases
# ---------------------------------------------------------------------------

# ==================== Asymmetric Per-Group Backward Tests ====================

def run_asymmetric_per_group_backward_test(n, m, group_size, bit, eps, clip_val, distribution, device_id):
    """Run a single backward test case for asymmetric per-group quantization."""
    device = f"npu:{device_id}"
    seed = 33
    weight_shape = (n, m)
    groups_per_row = m // group_size
    num_groups = n * groups_per_row
    scale_shape = (num_groups, 1)
    offset_shape = (num_groups, 1)

    weight_bm = create_input(weight_shape, torch.bfloat16, device, distribution, seed).requires_grad_(True)
    scale_bm = create_input(scale_shape, torch.bfloat16, device, distribution, seed).requires_grad_(True)
    offset_bm = create_input(offset_shape, torch.bfloat16, device, distribution, seed).requires_grad_(True)

    golden_inputs = [weight_bm, scale_bm, offset_bm]
    pto_inputs = [weight_bm, scale_bm, offset_bm, group_size, bit, eps, clip_val]
    golden = create_asymmetric_qat_golden(group_size, bit, eps, clip_val)
    return backward_test_autograd(golden_inputs, pto_inputs, golden, ai_infra_qat_asymmetric_per_group_backward)


@pytest.mark.parametrize(
    ('n', 'm', 'group', 'bit', 'eps', 'clip_val'),
    [
        pytest.param(1024, 2048, 128, 2, 0.0001, 0.99,
                     id="N1024-M2048-group128-bit2-eps0.0001-clip_val0.99",
                     marks=pytest.mark.skip(reason="temporarily disabled")),
        pytest.param(768, 2048, 128, 3, 0.0001, 0.99,
                     id="N768-M2048-group128-bit3-eps0.0001-clip_val0.99"),
    ]
)
def test_asymmetric_per_group_backward(n, m, group, bit, eps, clip_val) -> None:
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    results = []
    for dis in DISTRIBUTION:
        compare_result = run_asymmetric_per_group_backward_test(n, m, group, bit, eps, clip_val, dis, device_id)
        flattened_result = [str(item) for sublist in compare_result for item in sublist]
        str_params = [str(param) for param in [n, m, group, bit, eps, clip_val, dis]]
        results.append(str_params + flattened_result)


# ==================== Symmetric Per-Channel Backward Tests ====================

def run_symmetric_per_channel_backward_test(n, m, bit, eps, distribution, device_id):
    device = f"npu:{device_id}"
    seed = 33
    min_v = float(-2 ** (bit - 1))
    max_v = float(2 ** (bit - 1) - 1)
    weight_shape = (n, m)
    scale_shape = (n, 1)
    weight = create_input(weight_shape, torch.bfloat16, device, distribution, seed).requires_grad_(True)
    scale = create_input(scale_shape, torch.bfloat16, device, distribution, seed).requires_grad_(True)
    inputs = [weight, scale]
    pto_inputs = [weight, scale, eps, min_v, max_v]
    golden = create_symmetric_qat_nscale_golden(eps, min_v, max_v)
    return backward_test_autograd(inputs, pto_inputs, golden, ai_infra_qat_symmetric_per_channel_backward)


@pytest.mark.parametrize(
    ('n', 'm', 'bit', 'eps'),
    [
        pytest.param(153376, 2048, 4, 0.0001,
                     id="N153376-M2048-bit4-eps0.0001",
                     marks=pytest.mark.skip(reason="temporarily disabled")),
        pytest.param(38344, 2048, 4, 0.0001,
                     id="N38344-M2048-bit4-eps0.0001"),
    ]
)
def test_symmetric_per_channel_backward(n, m, bit, eps) -> None:
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    results = []
    for dis in DISTRIBUTION:
        compare_result = run_symmetric_per_channel_backward_test(n, m, bit, eps, dis, device_id)
        flattened_result = [str(item) for sublist in compare_result for item in sublist]
        str_params = [str(param) for param in [n, m, bit, eps, dis]]
        results.append(str_params + flattened_result)


# ==================== Symmetric Per-Tensor Backward Tests ====================

def run_symmetric_per_tensor_backward_test(n, m, bit, eps, distribution, device_id):
    device = f"npu:{device_id}"
    seed = 33
    min_v = float(-2 ** (bit - 1))
    max_v = float(2 ** (bit - 1) - 1)
    weight_shape = (n, m)
    scale_shape = (1, 1)
    weight = create_input(weight_shape, torch.bfloat16, device, distribution, seed).requires_grad_(True)
    scale = create_input(scale_shape, torch.bfloat16, device, distribution, seed).requires_grad_(True)
    inputs = [weight, scale]
    pto_inputs = [weight, scale, eps, min_v, max_v]
    golden = create_symmetric_qat_golden(eps, min_v, max_v)
    return backward_test_autograd(inputs, pto_inputs, golden, ai_infra_qat_symmetric_per_tensor_backward)


@pytest.mark.parametrize(
    ('n', 'm', 'bit', 'eps'),
    [
        pytest.param(153376, 2048, 8, 0.0001,
                     id="N153376-M2048-bit8-eps0.0001",
                     marks=pytest.mark.skip(reason="temporarily disabled")),
        pytest.param(38344, 2048, 8, 0.0001,
                     id="N38344-M2048-bit8-eps0.0001"),
    ]
)
def test_symmetric_per_tensor_backward(n, m, bit, eps) -> None:
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    results = []
    for dis in DISTRIBUTION:
        compare_result = run_symmetric_per_tensor_backward_test(n, m, bit, eps, dis, device_id)
        flattened_result = [str(item) for sublist in compare_result for item in sublist]
        str_params = [str(param) for param in [n, m, bit, eps, dis]]
        results.append(str_params + flattened_result)