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
 * \file engram_gate_bwd_tiling.h
 * \brief EngramGateBwd host tiling data.
 *
 * The field NAMES, ORDER and TYPES below must match the PyPTO kernel tiling
 * dataclass ``EngramGateBwdTiling`` in op_kernel/engram_gate_bwd.py exactly.
 */

#ifndef ENGRAM_GATE_BWD_TILING_H_
#define ENGRAM_GATE_BWD_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(EngramGateBwdTilingData)
TILING_DATA_FIELD_DEF(int64_t, t)
TILING_DATA_FIELD_DEF(int64_t, dim)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(EngramGateBwd, EngramGateBwdTilingData)

} // namespace optiling

#endif // ENGRAM_GATE_BWD_TILING_H_
