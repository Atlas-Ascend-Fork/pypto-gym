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
 * \file mhc_post_fwd_tiling.h
 * \brief MhcPostFwd host tiling data.
 *
 * The field NAMES, ORDER and TYPES below must match the PyPTO kernel tiling
 * dataclass ``MhcPostTiling`` in op_kernel/mhc_post_fwd.py exactly
 * (PyPTO lowers ``int`` fields to INDEX == int64_t). The hidden tile width
 * h_blk is a compile-time dimension carried by the tiling key, not a tiling
 * struct field.
 */

#ifndef MHC_POST_FWD_TILING_H_
#define MHC_POST_FWD_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MhcPostFwdTilingData)
TILING_DATA_FIELD_DEF(int64_t, n_tok)
TILING_DATA_FIELD_DEF(int64_t, h)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(MhcPostFwd, MhcPostFwdTilingData)

} // namespace optiling

#endif // MHC_POST_FWD_TILING_H_
