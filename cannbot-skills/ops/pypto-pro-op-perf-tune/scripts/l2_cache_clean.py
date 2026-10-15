#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""采集前刷新目标 NPU 的 L2 cache（按 cannbench 方式）。

cannbench 在每次测量步前跑一次大张量归约清空 L2，使各次测量从一致的 cache 状态起步。
本脚本用同一量级的 buffer（96x1024x1024 fp16 = 192 MiB，大于常见 L2），单独进程执行，
因此它的 kernel 落在 msprof 采集窗口之外，不会进入目标算子的 op_summary，无需保留名过滤。

设备选择与 runner 一致：设置 TILE_FWK_DEVICE_ID 时用它（manifest 协议），
否则在单卡 ASCEND_RT_VISIBLE_DEVICES 掩码下取逻辑 0。
"""

import os
import sys

_SHAPE = (96, 1024, 1024)  # 192 MiB fp16，大于常见 L2


def _device_index() -> int:
    raw = os.environ.get("TILE_FWK_DEVICE_ID", "")
    return int(raw.strip()) if raw.strip().isdigit() else 0


def main() -> int:
    import torch
    import torch_npu  # noqa: F401  # 注册 npu 后端

    device = _device_index()
    torch.npu.set_device(device)
    buffer = torch.ones(_SHAPE, dtype=torch.float16, device=f"npu:{device}")
    torch.max(buffer)  # 完整流过 L2，换出既有驻留数据
    torch.npu.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
