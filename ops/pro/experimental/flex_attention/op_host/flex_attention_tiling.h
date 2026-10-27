/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flex_attention_tiling.h
 * \brief FlexAttention host tiling data.
 *
 * The field NAMES, ORDER and TYPES below must match the PyPTO kernel tiling
 * dataclass ``FlexAttnTiling`` in op_kernel/flex_attention.py exactly
 * (PyPTO lowers ``int`` fields to INDEX == int64_t, and ``int[64]`` arrays to a
 * homogeneous int64_t[64] tuple), so the host-filled tiling buffer is
 * byte-layout identical to what the kernel reads.
 * This is enforced at build time by cmake/scripts/tiling_layout_check.py
 * (field-level diff at configure + a runtime layout check program built and
 * executed before the tiling compile).
 */

#ifndef FLEX_ATTENTION_MASK_INFO_TILING_H_
#define FLEX_ATTENTION_MASK_INFO_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FlexAttentionTilingData)
TILING_DATA_FIELD_DEF(int64_t, mask_num)
TILING_DATA_FIELD_DEF(int64_t, n_dim)
TILING_DATA_FIELD_DEF(int64_t, kv_heads)
TILING_DATA_FIELD_DEF(int64_t, batch)
TILING_DATA_FIELD_DEF(int64_t, r_f)
TILING_DATA_FIELD_DEF(int64_t, w_f)
TILING_DATA_FIELD_DEF(int64_t, s1)
TILING_DATA_FIELD_DEF(int64_t, s2)
TILING_DATA_FIELD_DEF(int64_t, d)
TILING_DATA_FIELD_DEF(float, scale)
TILING_DATA_FIELD_DEF_ARR(int64_t, 64, work_starts)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FlexAttention, FlexAttentionTilingData)

} // namespace optiling

#endif // FLEX_ATTENTION_MASK_INFO_TILING_H_