#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this files except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import os
import sys

import torch
import torch_npu  # noqa: F401
from torch.nn import CrossEntropyLoss

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..", "src",
    ),
)

from ops.pypto_tensor.llada2.liger_cross_entropy.liger_cross_entropy_loss import (  # noqa: E402
    LigerCrossEntropyLoss,
)

DEVICE = os.environ.get("TILE_FWK_DEVICE_ID", "0")
DEV = torch.device("npu:" + str(DEVICE))
torch.npu.set_device(int(DEVICE))

ATOL, RTOL = 1e-4, 7.8125e-3

logger = logging.getLogger(__name__)


def test_liger_cross_entropy_loss():
    """基础正确性：[391, 157184] bf16 mean + backward"""
    torch.manual_seed(0)
    bt, v = 391, 157184
    x = torch.randn(bt, v, dtype=torch.bfloat16, device=DEV).requires_grad_(True)
    target = torch.randint(0, v, (bt,), dtype=torch.int64, device=DEV)

    torch_ce = CrossEntropyLoss(reduction="mean")
    liger_ce = LigerCrossEntropyLoss(reduction="mean")

    _in1 = x.detach().clone().requires_grad_(True)
    _in2 = x.detach().clone().requires_grad_(True)

    out1 = torch_ce(_in1, target)
    out2 = liger_ce(_in2, target)
    assert torch.allclose(out1.float(), out2.float(), atol=ATOL, rtol=RTOL)

    out1.backward(gradient=torch.ones_like(out1))
    out2.backward(gradient=torch.ones_like(out2))
    assert torch.allclose(_in1.grad.float(), _in2.grad.float(), atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    test_liger_cross_entropy_loss()
    logger.info("PASS")
