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

/*!
 * \file indexer_prologue_qw_tiling.h
 * \brief IndexerPrologueQw host tiling data.
 *
 * Field NAMES, ORDER and TYPES must match the PyPTO kernel tiling dataclass
 * ``IndexerPrologueQwTiling`` in op_kernel/indexer_prologue_qw.py exactly.
 */

#ifndef INDEXER_PROLOGUE_QW_TILING_H_
#define INDEXER_PROLOGUE_QW_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(IndexerPrologueQwTilingData)
TILING_DATA_FIELD_DEF(int64_t, t)
TILING_DATA_FIELD_DEF(float, softmax_scale)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IndexerPrologueQw, IndexerPrologueQwTilingData)

} // namespace optiling

#endif // INDEXER_PROLOGUE_QW_TILING_H_
