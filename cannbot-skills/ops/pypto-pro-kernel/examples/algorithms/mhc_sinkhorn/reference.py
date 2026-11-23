# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""以 FP64 直接计算行 softmax 和交替归一化。精度按 FP32 输出预先固定。"""

import torch

RTOL, ATOL = 2e-5, 2e-6


def reference(x, steps, eps):
    a = torch.softmax(x.double(), dim=-1) + eps
    a = a / (a.sum(dim=-2, keepdim=True) + eps)
    for _ in range(1, steps):
        a = a / (a.sum(dim=-1, keepdim=True) + eps)
        a = a / (a.sum(dim=-2, keepdim=True) + eps)
    return a.float()
