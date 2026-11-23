# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证短批次的原地分页写。两路独立行 DMA 由 mutex 排序；按序或唯一写者保证一致。
逐位比较整块 cache（包含未触碰页、负零和 NaN 载荷），检查返回别名与只读输入。
槽号 [0,9,0,17,9] 让四个 worker 跨页处理两个重复槽和一个唯一槽。槽 0、9、17 的写者依次为 token 2、4、3。
"""

import torch

from .kernel import run
from .reference import reference


def test_teaching_last_writer_across_workers(device):
    slots = [0, 9, 0, 17, 9]
    g = torch.Generator().manual_seed(27)
    cache_shape, new_shape = (3, 8, 1, 8), (len(slots), 1, 8)
    k, v = [torch.randn(cache_shape, generator=g) for _ in range(2)]
    k.view(torch.int32)[0, 1, 0, 0] = 0x7fc01234
    v[0, 1, 0, 0] = -0.0
    nk, nv = [torch.randn(new_shape, generator=g) for _ in range(2)]
    ids = torch.tensor(slots, dtype=torch.int32)
    expected = reference(k, v, nk, nv, ids)
    inputs = [t.to(device) for t in (nk, nv, ids)]
    for workers in (1, 4):
        caches = [t.to(device) for t in (k, v)]
        result = run(*caches, *inputs, workers=workers)
        for got, cache, want in zip(result, caches, expected):
            assert got is cache
            assert torch.equal(got.cpu().view(torch.int32), want.view(torch.int32))
    for src, original in zip(inputs, (nk, nv, ids)):
        assert torch.equal(src.cpu(), original)
