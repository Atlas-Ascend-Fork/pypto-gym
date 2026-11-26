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
 * \file attn_prologue_tiling.h
 * \brief AttnPrologue host tiling data.
 *
 * Field names, order and types must match the PyPTO kernel dataclass
 * ``AttnPrologueTiling`` in op_kernel/attn_prologue.py exactly.
 * This is checked at configure time by cmake/scripts/tiling_layout_check.py.
 */

#ifndef ATTN_PROLOGUE_TILING_H_
#define ATTN_PROLOGUE_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AttnPrologueTilingData)
TILING_DATA_FIELD_DEF(int64_t, t)
TILING_DATA_FIELD_DEF(int64_t, cache_rows)
TILING_DATA_FIELD_DEF(int64_t, block_size)
TILING_DATA_FIELD_DEF(int64_t, block_stride)
TILING_DATA_FIELD_DEF(int64_t, row_stride)
TILING_DATA_FIELD_DEF(int64_t, ws_rows)
TILING_DATA_FIELD_DEF(float, norm_eps)
TILING_DATA_FIELD_DEF(int64_t, emit_qr_norm)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttnPrologue, AttnPrologueTilingData)

} // namespace optiling

#endif // ATTN_PROLOGUE_TILING_H_
