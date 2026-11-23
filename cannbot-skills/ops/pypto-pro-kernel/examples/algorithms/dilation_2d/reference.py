# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""独立的 CPU 坐标定义。每个有效像素先做 FP16 加法，再取逐通道最大值。"""

from itertools import product

import torch


def reference(x, filt, stride, rate, padding):
    n, h, w, c = x.shape
    radius = rate
    if padding == "VALID":
        ys, xs = range(radius, h - radius, stride), range(radius, w - radius, stride)
    else:
        def centers(size):
            last = ((size - 1) // stride) * stride
            left = max(0, last + 2 * radius + 1 - size) // 2
            return range(radius - left, radius - left + size, stride)
        ys, xs = centers(h), centers(w)
    out = torch.full((n, len(ys), len(xs), c), -torch.inf, dtype=x.dtype)
    for (oy, cy), (ox, cx) in product(enumerate(ys), enumerate(xs)):
        for fy, fx in product(range(3), repeat=2):
            iy, ix = cy + (fy - 1) * rate, cx + (fx - 1) * rate
            if 0 <= iy < h and 0 <= ix < w:
                out[:, oy, ox] = torch.maximum(out[:, oy, ox], x[:, iy, ix] + filt[fy, fx])
    return out
