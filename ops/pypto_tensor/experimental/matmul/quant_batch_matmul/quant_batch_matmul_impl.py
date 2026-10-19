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

"""QuantBatchMatmul F-T quantization implemented with PyPTO.

The implementation follows the F-T formula from aclnnQuantMatmulV5:

    out[b, m, n] = quant(x1[b] @ x2[b]^T * x2Scale * x1Scale)

Physical layouts: x1 is [batch, m, k], x2 is [batch, n, k] with ``b_trans=True``.

For F-T FP8E4M3->INT8, x1Scale and x2Scale are scalar tensors with shape (1,).
Each N tile uses ``pypto.matmul`` to INT8 with global ``scale`` passed via ``extend_params``.
"""

import struct
from dataclasses import dataclass, replace

import pypto
import torch
import torch_npu  # type: ignore[reportMissingImports]  # noqa: F401


_FIXPIPE_SCALE_MASK = 0xFFFFE000


@dataclass
class QuantBatchMatmulConfig:
    """Shape, dtype, and tiling configuration for F-T quant batch matmul.

    Attributes:
        ori_shape: Logical matmul shape [batch, m, k, n].
        m_tile_shape: Cube tile shape for the M axis.
        k_tile_shape: Cube tile shape for the K axis.
        n_tile_shape: Cube tile shape for the N axis.
        vector_tile_shape: Vector tile shape used for assemble work.
        n_block_size: N-axis block size for parallel matmul tasks.
        in_dtype: x1/x2 PyPTO dtype. Uses DT_FP8E4M3.
        out_dtype: Output PyPTO dtype. Uses DT_INT8.
        combined_scale: Product of x1Scale and x2Scale used in matmul fixpipe.
        description: Optional testcase description.
    """

    ori_shape: list[int]
    m_tile_shape: list[int]
    k_tile_shape: list[int]
    n_tile_shape: list[int]
    vector_tile_shape: list[int]

    n_block_size: int = 256
    in_dtype: pypto.DataType = pypto.DT_FP8E4M3
    out_dtype: pypto.DataType = pypto.DT_INT8
    combined_scale: float = 1.0
    description: str = ""

    def __post_init__(self):
        if len(self.ori_shape) != 4:
            raise ValueError("ori_shape must be [batch, m, k, n].")

        if self.ori_shape[0] < 1:
            raise ValueError("batch must be >= 1.")

        if self.in_dtype != pypto.DT_FP8E4M3:
            raise ValueError("in_dtype must be DT_FP8E4M3.")

        if self.out_dtype != pypto.DT_INT8:
            raise ValueError("out_dtype must be DT_INT8.")


@dataclass(frozen=True)
class QuantBatchMatmulInputs:
    """Input tensors for the F-T quant batch matmul."""

    x1: torch.Tensor
    x2: torch.Tensor
    x1_scale: torch.Tensor | None
    x2_scale: torch.Tensor


def get_x1_shape(config: QuantBatchMatmulConfig) -> tuple[int, int, int]:
    """Return the physical x1 shape [batch, m, k] for F-T layout."""
    batch, m, k, _ = config.ori_shape
    return batch, m, k


def get_x2_shape(config: QuantBatchMatmulConfig) -> tuple[int, int, int]:
    """Return the physical x2 shape [batch, n, k] for F-T layout."""
    batch, _, k, n = config.ori_shape
    return batch, n, k


def get_x1_scale_shape() -> tuple[int]:
    """Return the x1Scale shape described by the F-T quantization contract."""
    return (1,)


def get_x2_scale_shape() -> tuple[int]:
    """Return the x2Scale shape described by the F-T quantization contract."""
    return (1,)


def get_output_shape(config: QuantBatchMatmulConfig) -> tuple[int, int, int]:
    """Return the logical output shape [batch, m, n]."""
    batch, m, _, n = config.ori_shape
    return batch, m, n


def mask_fixpipe_scale(scale: float) -> tuple[float, float]:
    """Apply fixpipe scale bit-mask and return (kernel_scale, golden_scale)."""
    packed = struct.pack("f", float(scale))
    as_int = struct.unpack("I", packed)[0]

    masked_int = as_int & _FIXPIPE_SCALE_MASK
    golden_scale = struct.unpack("f", struct.pack("I", masked_int))[0]

    return golden_scale, golden_scale


def compute_combined_scale(
    x1_scale: torch.Tensor | None,
    x2_scale: torch.Tensor,
) -> tuple[float, float]:
    """Compute fixpipe scale for kernel and reference golden."""
    x2_value = float(x2_scale.reshape(-1)[0].cpu().item())

    if x1_scale is None:
        raw_scale = x2_value
    else:
        x1_value = float(x1_scale.reshape(-1)[0].cpu().item())
        raw_scale = x1_value * x2_value

    return mask_fixpipe_scale(raw_scale)


@pypto.frontend.jit(
    pass_options={
        "cube_nbuffer_setting": {-1: 2},
        "vec_nbuffer_setting": {-2: 1, -1: 2},
    },
    runtime_options={
        "stitch_function_max_num": 128,
        "device_sched_mode": 1},
)
def quant_batch_matmul_kernel(
    x1: pypto.Tensor(),
    x2: pypto.Tensor(),
    out: pypto.Tensor(),
    config: QuantBatchMatmulConfig,
):
    """Compute F-T quantized batch matmul into ``out``."""
    batch, m, k, n = config.ori_shape

    n_block = config.n_block_size
    m_block = config.m_tile_shape[-1]

    n_loop = (n + n_block - 1) // n_block
    m_loop = (m + m_block - 1) // m_block

    # F-4: flatten batch into M/N so kernel only uses 2D view.
    x1_flat = pypto.reshape(
        x1,
        [batch * m, k],
        inplace=True,
    )
    x2_flat = pypto.reshape(
        x2,
        [batch * n, k],
        inplace=True,
    )

    pypto.set_cube_tile_shapes(
        config.m_tile_shape,
        config.k_tile_shape,
        config.n_tile_shape,
    )
    pypto.set_vec_tile_shapes(*config.vector_tile_shape)

    for b_idx in pypto.loop(batch, name="LOOP_BATCH", idx_name="b_idx"):
        x1_batch_off = b_idx * m
        x2_batch_off = b_idx * n

        for m_idx in pypto.loop(m_loop, name="LOOP_M", idx_name="m_idx"):
            m_off = m_idx * m_block
            valid_m = (m - m_off).min(m_block)

            x1_block = pypto.view(
                x1_flat,
                [m_block, k],
                [x1_batch_off + m_off, 0],
                valid_shape=[valid_m, k],
            )

            for n_idx in pypto.loop(n_loop, name="LOOP_N", idx_name="n_idx"):
                n_off = n_idx * n_block
                valid_n = (n - n_off).min(n_block)

                x2_block = pypto.view(
                    x2_flat,
                    [n_block, k],
                    [x2_batch_off + n_off, 0],
                    valid_shape=[valid_n, k],
                )

                result = pypto.matmul(
                    x1_block,
                    x2_block,
                    pypto.DT_INT8,
                    a_trans=False,
                    b_trans=True,
                    extend_params={
                        "scale": config.combined_scale,
                    },
                )

                pypto.set_vec_tile_shapes(m_block, n_block)

                result_out = pypto.reshape(
                    result,
                    [1, m_block, n_block],
                )

                result_tile = pypto.view(
                    result_out,
                    [1, m, n_block],
                    [0, 0, 0],
                )

                pypto.assemble(
                    result_tile,
                    [b_idx, m_off, n_off],
                    out,
                )


def quant_batch_matmul(
    inputs: QuantBatchMatmulInputs,
    config: QuantBatchMatmulConfig,
) -> torch.Tensor:
    """Launch the PyPTO F-T kernel and return [batch, m, n] INT8."""
    x1 = inputs.x1.npu().contiguous()
    x2 = inputs.x2.npu().contiguous()

    kernel_scale, _ = compute_combined_scale(
        inputs.x1_scale,
        inputs.x2_scale,
    )

    launch_config = replace(
        config,
        combined_scale=kernel_scale,
    )

    out_shape = get_output_shape(launch_config)
    out = torch.zeros(
        out_shape,
        dtype=torch.int8,
    ).npu()

    quant_batch_matmul_kernel(
        x1,
        x2,
        out,
        launch_config,
    )

    return out
