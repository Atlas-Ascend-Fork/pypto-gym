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
GLM-4.5 FFN Common Interface Module

This module provides common utility functions for FFN quantization operations,
including symmetric quantization, dequantization, and SwiGLU activation.

Main Functions:
    - symmetric_quantization_per_token: Per-token symmetric quantization
    - dequant_dynamic: Dynamic dequantization with two scale factors
    - swiglu: SwiGLU activation function implementation
"""
from typing import Tuple
import pypto


def symmetric_quantization_per_token(input_tensor) -> Tuple:
    """
    Perform symmetric quantization per token (per row).

    Args:
        input_tensor: Input tensor to quantize

    Returns:
        Tuple of (quantized_int8_tensor, dequantization_scale)
    """
    x_fp32 = pypto.cast(input_tensor, pypto.DT_FP32)
    x_abs = pypto.abs(x_fp32)
    x_max = pypto.amax(x_abs, -1, True)
    shape_0, shape_1 = x_max.shape[:2]
    x_scale = pypto.div(pypto.full([shape_0, shape_1], 127.0, pypto.DT_FP32), x_max)
    x_mul = pypto.mul(x_fp32, x_scale)
    x_int32 = pypto.cast(x_mul, pypto.DT_INT32, pypto.CastMode.CAST_RINT)
    x_fp16 = pypto.cast(x_int32, pypto.DT_FP16, pypto.CastMode.CAST_ROUND)
    x_int8 = pypto.cast(x_fp16, pypto.DT_INT8, pypto.CastMode.CAST_TRUNC, satmode=pypto.SaturationMode.ON)
    x_scale_quant = pypto.div(pypto.full([shape_0, shape_1], 1.0, pypto.DT_FP32), x_scale)
    return x_int8, x_scale_quant


def dequant_dynamic(in_tensor, scale_1, scale_2):
    """
    Perform dynamic dequantization using two scale factors.

    Args:
        in_tensor: Quantized input tensor
        scale_1: First scale factor
        scale_2: Second scale factor

    Returns:
        Dequantized tensor
    """
    in_tensor_fp32 = pypto.cast(in_tensor, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    scale_1_fp32 = pypto.cast(scale_1, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    scale_2_fp32 = pypto.cast(scale_2, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    out_scale_2 = pypto.mul(in_tensor_fp32, scale_2_fp32)
    out = pypto.mul(out_scale_2, scale_1_fp32)
    return out


def swiglu(up_proj):
    """
    Apply SwiGLU activation function: x * sigmoid(x) * right_half.

    Args:
        up_proj: Input tensor with shape [batch, intermediate_size * 2]

    Returns:
        SwiGLU activated tensor with shape [batch, intermediate_size]
    """
    intermediate_size = up_proj.shape[1] // 2
    up_proj_left = pypto.view(up_proj, [up_proj.shape[0], intermediate_size], [0, 0])
    up_proj_right = pypto.view(up_proj, [up_proj.shape[0], intermediate_size], [0, intermediate_size])
    swiglu_mul = pypto.mul(up_proj_left, -1.0)
    swiglu_exp = pypto.exp(swiglu_mul)
    swiglu_add = pypto.add(swiglu_exp, 1.0)
    swiglu_div = pypto.div(up_proj_left, swiglu_add)
    swiglu_out = pypto.mul(swiglu_div, up_proj_right)
    return swiglu_out
