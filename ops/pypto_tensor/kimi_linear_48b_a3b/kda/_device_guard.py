# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared single-NPU bind guard for the KDA PyPTO chunk kernel.

The chunk wrapper must avoid launching a JIT kernel on an NPU other than the one
PyPTO has bound this process to. A single ``_BOUND_NPU`` / ``_FALLBACK_WARNED``
state keeps a single bound device for
the whole process (and emits the partial-coverage warning at most once).
"""

__all__ = ["_bound_device_ok"]

import warnings

import torch
import torch_npu  # noqa: F401  required for NPU device init


# PyPTO binds a JIT kernel to ONE NPU per process (the device of the first
# launch). For a model sharded across NPUs (device_map), a KDA layer whose
# tensors live on a different NPU must not call the kernel — the wrapper raises
# NotImplementedError so the modeling layer falls back to the torch path. Only
# the bound device's KDA layers run on PyPTO.
#
# Note (future work, multi-NPU coverage): in a SINGLE-PROCESS device_map
# deployment this means only the bound NPU's KDA layers are PyPTO-accelerated
# (e.g. ~6 of 20 KDA layers
# with 4-way sharding); the rest fall back to torch. For FULL 20/20 coverage run
# ONE PROCESS PER NPU (pipeline / tensor parallel). The one-time RuntimeWarning
# below ensures this partial coverage is never hit silently.
_BOUND_NPU = None
_FALLBACK_WARNED = False


def _bound_device_ok(device):
    """True if ``device`` is the single NPU this process's PyPTO kernels are
    bound to (lazily fixed to the first NPU a kernel is launched on). On the
    first off-bound-device call, emit a one-time warning so the resulting
    partial PyPTO coverage is visible rather than silent."""
    global _BOUND_NPU, _FALLBACK_WARNED
    if device.type != "npu":
        # The PyPTO kernel is NPU-only; a non-NPU (e.g. CPU) tensor must fall back
        # to the torch path rather than reach the JIT kernel.
        return False
    idx = device.index if device.index is not None else torch.npu.current_device()
    if _BOUND_NPU is None:
        _BOUND_NPU = idx
    if idx == _BOUND_NPU:
        return True
    if not _FALLBACK_WARNED:
        _FALLBACK_WARNED = True
        warnings.warn(
            f"PyPTO KDA is bound to NPU {_BOUND_NPU} (one NPU per process); KDA layers on "
            f"other NPUs (e.g. NPU {idx}) fall back to torch. In a single-process device_map "
            f"deployment only the bound NPU's KDA layers are PyPTO-accelerated (partial "
            f"coverage, e.g. ~6 of 20 KDA layers with 4-way sharding). For full 20/20 "
            f"coverage run one process per NPU; see kda/README.md.",
            RuntimeWarning, stacklevel=2)
    return False
