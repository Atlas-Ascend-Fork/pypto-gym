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
 * \file quant_sparse_lightning_indexer_tiling.h
 * \brief QuantSparseLightningIndexer host tiling data.
 *
 * The field NAMES, ORDER and TYPES below must match the PyPTO kernel tiling
 * dataclass ``QsliTiling`` in op_kernel/quant_sparse_lightning_indexer.py exactly
 * (PyPTO lowers ``int`` fields to INDEX == int64_t), so the host-filled tiling
 * buffer is byte-layout identical to what the kernel reads.
 * This is enforced at build time by cmake/scripts/tiling_layout_check.py
 * (field-level diff at configure + a runtime layout check program built and
 * executed before the tiling compile).
 *
 * The first 21 fields are the jit version's ``rt_*`` scalar kernel arguments
 * verbatim; ``num_workers`` is added because the flat workspace size and the
 * staging-ring row count both scale with the launched core quota.
 */

#ifndef QUANT_SPARSE_LIGHTNING_INDEXER_TILING_H_
#define QUANT_SPARSE_LIGHTNING_INDEXER_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(QsliTilingData)
TILING_DATA_FIELD_DEF(int64_t, rt_batch)
TILING_DATA_FIELD_DEF(int64_t, rt_key_stride)
TILING_DATA_FIELD_DEF(int64_t, rt_table_width)
TILING_DATA_FIELD_DEF(int64_t, rt_total_q)
TILING_DATA_FIELD_DEF(int64_t, rt_table_cached)
TILING_DATA_FIELD_DEF(int64_t, rt_tnd_k_stride)
TILING_DATA_FIELD_DEF(int64_t, rt_tnd_s_stride)
TILING_DATA_FIELD_DEF(int64_t, rt_tnd_rows)
TILING_DATA_FIELD_DEF(int64_t, rt_addr32)
TILING_DATA_FIELD_DEF(int64_t, rt_has_cu_q)
TILING_DATA_FIELD_DEF(int64_t, rt_has_seq_q)
TILING_DATA_FIELD_DEF(int64_t, rt_has_seq_k)
TILING_DATA_FIELD_DEF(int64_t, rt_has_residual)
TILING_DATA_FIELD_DEF(int64_t, rt_res_eq_used_q)
TILING_DATA_FIELD_DEF(int64_t, rt_has_idx_off)
TILING_DATA_FIELD_DEF(int64_t, rt_has_table)
TILING_DATA_FIELD_DEF(int64_t, rt_k_capacity)
TILING_DATA_FIELD_DEF(int64_t, rt_cmp_ratio)
TILING_DATA_FIELD_DEF(int64_t, rt_blocks_per_page)
TILING_DATA_FIELD_DEF(int64_t, rt_page_param)
TILING_DATA_FIELD_DEF(int64_t, rt_topk)
TILING_DATA_FIELD_DEF(int64_t, num_workers)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantSparseLightningIndexer, QsliTilingData)

} // namespace optiling

#endif // QUANT_SPARSE_LIGHTNING_INDEXER_TILING_H_
