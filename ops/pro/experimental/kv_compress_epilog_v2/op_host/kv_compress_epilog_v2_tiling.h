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
 * \file kv_compress_epilog_v2_tiling.h
 * \brief KvCompressEpilogV2 host tiling data.
 *
 * The field NAMES, ORDER and TYPES below must match the PyPTO kernel tiling
 * dataclass ``KvCompressTiling`` in op_kernel/kv_compress_epilog_v2.py
 * exactly (PyPTO lowers ``int`` fields to INDEX == int64_t), so the host-filled
 * tiling buffer is byte-layout identical to what the kernel reads.
 * This is enforced at build time by cmake/scripts/tiling_layout_check.py.
 */

#ifndef KV_COMPRESS_EPILOG_V2_TILING_H_
#define KV_COMPRESS_EPILOG_V2_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(KvCompressEpilogV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, tokens)
TILING_DATA_FIELD_DEF(int64_t, dim)
TILING_DATA_FIELD_DEF(int64_t, cache_rows)
TILING_DATA_FIELD_DEF(int64_t, cache_cols)
TILING_DATA_FIELD_DEF(int64_t, block_size)
TILING_DATA_FIELD_DEF(int64_t, block_stride)
TILING_DATA_FIELD_DEF(int64_t, storage_bytes)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(KvCompressEpilogV2, KvCompressEpilogV2TilingData)

} // namespace optiling

#endif // KV_COMPRESS_EPILOG_V2_TILING_H_
