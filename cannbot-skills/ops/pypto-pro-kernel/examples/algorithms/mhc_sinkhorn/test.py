# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证 4×4 Sinkhorn。GM 只搬运有效矩阵，mutex 保护双槽，寄存器承接迭代。
65 个矩阵覆盖一个满块和一个尾块。两轮归一化后，逐轮写回与寄存器驻留均对照 CPU FP64。
"""

import torch

from .kernel import run
from .reference import ATOL, RTOL, reference


def test_teaching_two_rounds_with_tail(device):
    steps, eps = 2, 1e-4
    x = torch.randn(65, 4, 4, generator=torch.Generator().manual_seed(132))
    x[0, 0] = 0
    x[-1] *= 40  # 让尾块中的矩阵也检验数值稳定性
    expected = reference(x, steps, eps)
    source = x.to(device)
    for resident in (False, True):
        result = run(source, steps, eps, resident=resident)
        torch.testing.assert_close(result.cpu(), expected, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(source.cpu(), x, atol=0, rtol=0)
