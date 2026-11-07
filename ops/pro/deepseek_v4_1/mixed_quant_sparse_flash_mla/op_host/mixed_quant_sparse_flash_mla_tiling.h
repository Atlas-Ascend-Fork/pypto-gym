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
 * \file mixed_quant_sparse_flash_mla_tiling.h
 * \brief MixedQuantSparseFlashMla host tiling data.
 *
 * The field NAMES, ORDER and TYPES below must match the PyPTO kernel tiling
 * dataclass ``MqsmlaTiling`` in op_kernel/mixed_quant_sparse_flash_mla.py
 * exactly (PyPTO lowers ``int`` fields to INDEX == int64_t and ``float`` to
 * float32), so the host-filled tiling buffer is byte-layout identical to what
 * the kernel reads. This is enforced at build time by
 * cmake/scripts/tiling_layout_check.py (field-level diff at configure + a
 * runtime layout check program built and executed before the tiling compile).
 *
 * The 1024-int metadata plan is NOT a tiling field: it stays a kernel input
 * tensor. Putting it in tiling inflated the kernel parameter block to 8.3 KB
 * and slowed the kernel by 10%-66%.
 */

#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_TILING_H_
#define MIXED_QUANT_SPARSE_FLASH_MLA_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MixedQuantSparseFlashMlaTilingData)
TILING_DATA_FIELD_DEF(float, scale)
TILING_DATA_FIELD_DEF(int64_t, t1)
TILING_DATA_FIELD_DEF(int64_t, ori_pages)
TILING_DATA_FIELD_DEF(int64_t, cmp_pages)
TILING_DATA_FIELD_DEF(int64_t, b_num)
TILING_DATA_FIELD_DEF(int64_t, k_ori)
TILING_DATA_FIELD_DEF(int64_t, k_cmp)
TILING_DATA_FIELD_DEF(int64_t, bs_ori)
TILING_DATA_FIELD_DEF(int64_t, bs_cmp)
TILING_DATA_FIELD_DEF(int64_t, bt_ori)
TILING_DATA_FIELD_DEF(int64_t, bt_cmp)
TILING_DATA_FIELD_DEF(int64_t, ori_kv_stride0)
TILING_DATA_FIELD_DEF(int64_t, cmp_kv_stride0)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(MixedQuantSparseFlashMla, MixedQuantSparseFlashMlaTilingData)

} // namespace optiling

#endif // MIXED_QUANT_SPARSE_FLASH_MLA_TILING_H_