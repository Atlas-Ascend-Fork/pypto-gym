# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""按输入顺序写入 CPU 副本，同槽覆盖自然给出最后写者。按位验证。"""


def reference(k, v, new_k, new_v, slots):
    k, v = k.clone(), v.clone()
    for t, slot in enumerate(slots.tolist()):
        block, offset = divmod(slot, k.shape[1])
        k[block, offset] = new_k[t]
        v[block, offset] = new_v[t]
    return k, v
