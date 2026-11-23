/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef INDEXER_PROLOGUE_K_FUSED_TILING_H_
#define INDEXER_PROLOGUE_K_FUSED_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(IndexerPrologueKFusedTilingData)
TILING_DATA_FIELD_DEF(int64_t, t)
TILING_DATA_FIELD_DEF(int64_t, head_dim)
TILING_DATA_FIELD_DEF(int64_t, dim)
TILING_DATA_FIELD_DEF(int64_t, dr)
TILING_DATA_FIELD_DEF(int64_t, storage_mode)
TILING_DATA_FIELD_DEF(int64_t, combined_block_size)
TILING_DATA_FIELD_DEF(int64_t, write_scale_cache)
TILING_DATA_FIELD_DEF(int64_t, cache_blocks)
TILING_DATA_FIELD_DEF(int64_t, cache_slots)
TILING_DATA_FIELD_DEF(int64_t, cache_width)
TILING_DATA_FIELD_DEF(int64_t, scale_boxes)
TILING_DATA_FIELD_DEF(float, epsilon)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IndexerPrologueK, IndexerPrologueKFusedTilingData)

} // namespace optiling

#endif
