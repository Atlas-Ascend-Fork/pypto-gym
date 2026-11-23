# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证形态学膨胀；每行片段搬入带保护区的 UB，别名 mutex 保证先填充再搬入。
CPU 按坐标枚举九个 tap，逐元素精确核对单行/双行、SAME 负值边界、尾通道及 VALID 的步长与空洞采样。
用 -k teaching 可只跑 3×5 图像：前两行共享输入，四像素一组后还剩一个尾像素。
"""

import pytest
import torch

from .kernel import run
from .reference import reference


@pytest.mark.parametrize("n,h,w,c,stride,rate,padding", [
    pytest.param(1, 3, 5, 17, 1, 1, "SAME", id="teaching-two-rows-four-pixels"),
    pytest.param(2, 11, 17, 32, 2, 2, "VALID", id="strided-dilated-valid"),
])
def test_dilation(device, n, h, w, c, stride, rate, padding):
    g = torch.Generator().manual_seed(142)
    x = -torch.rand(n, h, w, c, generator=g).half() - 1
    filt = -torch.rand(3, 3, c, generator=g).half()
    expected = reference(x, filt, stride, rate, padding)
    src, taps = x.to(device), filt.to(device)
    for rows in (1, 2):
        got = run(src, taps, stride, rate, padding, band_rows=rows)
        torch.testing.assert_close(got.cpu(), expected, atol=0, rtol=0)
