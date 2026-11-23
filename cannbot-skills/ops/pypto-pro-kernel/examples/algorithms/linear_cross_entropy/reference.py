# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU FP64 稠密投影及交叉熵；直接使用已量化的 BF16 输入，按 FP32 输出比较。"""

import torch.nn.functional as F


RTOL, ATOL = 3e-5, 3e-5


def reference(hidden, weight, labels, reduction, ignore_index):
    logits = hidden.double() @ weight.double().T
    return F.cross_entropy(logits, labels.long(), reduction=reduction,
                           ignore_index=ignore_index).float().reshape(1)
