/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(QliTilingData)
TILING_DATA_FIELD_DEF(int64_t, rows)
TILING_DATA_FIELD_DEF(int64_t, batches)
TILING_DATA_FIELD_DEF(int64_t, max_q)
TILING_DATA_FIELD_DEF(int64_t, max_k)
TILING_DATA_FIELD_DEF(int64_t, page_size)
TILING_DATA_FIELD_DEF(int64_t, k_rows)
TILING_DATA_FIELD_DEF(int64_t, s_rows)
TILING_DATA_FIELD_DEF(int64_t, k_page_stride)
TILING_DATA_FIELD_DEF(int64_t, ks_page_stride)
TILING_DATA_FIELD_DEF(int64_t, k_stride)
TILING_DATA_FIELD_DEF(int64_t, ks_stride)
TILING_DATA_FIELD_DEF(int64_t, table_cols)
TILING_DATA_FIELD_DEF(int64_t, cube)
TILING_DATA_FIELD_DEF(int64_t, vector)
TILING_DATA_FIELD_DEF(int64_t, plan_elems)
TILING_DATA_FIELD_DEF(int64_t, capacity)
TILING_DATA_FIELD_DEF(int64_t, flags)
TILING_DATA_FIELD_DEF(int64_t, cmp_ratio)
TILING_DATA_FIELD_DEF(int64_t, ws_scores)
TILING_DATA_FIELD_DEF(int64_t, ws_index_map)
TILING_DATA_FIELD_DEF(int64_t, ws_partial)
TILING_DATA_FIELD_DEF(int64_t, score_cols)
TILING_DATA_FIELD_DEF(int64_t, aux_cols)
TILING_DATA_FIELD_DEF(int64_t, topk)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(GymQuantLightningIndexer, QliTilingData)
struct QliCompileInfo {};
}
