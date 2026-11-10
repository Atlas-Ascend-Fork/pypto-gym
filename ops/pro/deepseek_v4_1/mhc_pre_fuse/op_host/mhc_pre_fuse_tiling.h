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
 * \file mhc_pre_fuse_tiling.h
 * \brief MhcPreFuse host tiling data.
 *
 * Field NAMES, ORDER and TYPES must match the PyPTO kernel tiling dataclass
 * ``MhcPreTiling`` in op_kernel/mhc_pre_fuse.py exactly. The structural
 * parameters (token block, hidden chunk, residual tail) live in the tiling key;
 * T, H and the five knobs are runtime tiling values.
 */

#ifndef MHC_PRE_FUSE_TILING_H_
#define MHC_PRE_FUSE_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MhcPreFuseTilingData)
TILING_DATA_FIELD_DEF(int64_t, t)
TILING_DATA_FIELD_DEF(int64_t, h)
TILING_DATA_FIELD_DEF(float, inv)
TILING_DATA_FIELD_DEF(float, rms_eps)
TILING_DATA_FIELD_DEF(float, pre_eps)
TILING_DATA_FIELD_DEF(float, sinkhorn_eps)
TILING_DATA_FIELD_DEF(float, post_mult)
TILING_DATA_FIELD_DEF(int64_t, sinkhorn_repeat)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(MhcPreFuse, MhcPreFuseTilingData)

} // namespace optiling

#endif // MHC_PRE_FUSE_TILING_H_
