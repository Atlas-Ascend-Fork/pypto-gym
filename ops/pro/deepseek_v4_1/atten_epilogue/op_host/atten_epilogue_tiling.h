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
 * \file atten_epilogue_tiling.h
 * \brief AttenEpilogue host tiling data.
 *
 * The field NAMES, ORDER and TYPES below must match the PyPTO kernel tiling
 * dataclass ``EpilogueTiling`` in op_kernel/atten_epilogue.py exactly
 * (PyPTO lowers ``int`` fields to INDEX == int64_t), so the host-filled tiling
 * buffer is byte-layout identical to what the kernel reads.
 * This is enforced at build time by cmake/scripts/tiling_layout_check.py
 * (field-level diff at configure + a runtime layout check program built and
 * executed before the tiling compile).
 */

#ifndef ATTEN_EPILOGUE_TILING_H_
#define ATTEN_EPILOGUE_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct AttenEpilogueCompileInfo {};

BEGIN_TILING_DATA_DEF(EpilogueTilingData)
TILING_DATA_FIELD_DEF(int64_t, t)
TILING_DATA_FIELD_DEF(int64_t, padded)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttenEpilogue, EpilogueTilingData)

} // namespace optiling

#endif // ATTEN_EPILOGUE_TILING_H_
