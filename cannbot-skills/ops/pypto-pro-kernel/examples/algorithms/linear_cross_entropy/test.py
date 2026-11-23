# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证线性投影与交叉熵。按词表分块，单 stream 使 Cube 写回后 VF 才消费。
同一独立 CPU 稠密参考核对保留所有块和单块复用，含块边界、后块抬高最大值与全忽略。
用 -k teaching 可只跑 65 个词：标签 63 和 64 分居块的两侧，第二块抬高最大值。
"""

import pytest
import torch

from .kernel import run
from .reference import ATOL, RTOL, reference


@pytest.mark.parametrize("v,h,labels,reduction,ignore", [
    pytest.param(65, 64, [63, -1, 64, 0, 1], "mean", -1, id="teaching-target-in-next-slab"),
    pytest.param(129, 128, [0, 128] * 8, "sum", -100, id="small-terms-across-three-slabs"),
    pytest.param(71, 64, [3, 3], "sum", 3, id="all-ignored-sum-zero"),
    pytest.param(71, 64, [3, 3], "mean", 3, id="all-ignored-mean-nan"),
])
def test_loss(device, v, h, labels, reduction, ignore):
    g = torch.Generator().manual_seed(143)
    x = torch.randn(len(labels), h, generator=g).bfloat16()
    w = torch.randn(v, h, generator=g).bfloat16()
    if v == 65:
        # 最后一个词在第二块。四个有效 token 的均值是 2**127，可表示；
        # 标签 63 对应的单个损失和全体损失之和都会超出 FP32 范围。
        x.zero_()
        x[:, 0] = 1
        w.zero_()
        w[63, 0] = -(2.0**127)
        w[64, 0] = 2.0**127
    elif v == 129:
        # 两个正交向量各用满 H 维，首块和末块交替命中。目标分数为 17，其余为 0。
        # exp(-17) 小于 1.0 的半个 FP32 ULP，但这些小项的总损失可测。
        x.fill_(1)
        x[1::2, 1::2] = -1
        w.zero_()
        w[0] = x[0] * (17 / h)
        w[-1] = x[1] * (17 / h)
    ids = torch.tensor(labels, dtype=torch.int32)
    expected = reference(x, w, ids, reduction, ignore)
    inputs = tuple(t.to(device) for t in (x, w, ids))
    for streamed in (False, True):
        got = run(*inputs, reduction, ignore, streamed=streamed)
        torch.testing.assert_close(got.cpu(), expected, atol=ATOL, rtol=RTOL, equal_nan=True)
