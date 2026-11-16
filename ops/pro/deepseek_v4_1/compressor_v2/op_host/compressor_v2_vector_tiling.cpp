/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compressor_v2_vector_tiling.h"
#include "compressor_v2_plan.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingPyptoCompressorV2Vector(gert::TilingContext *context)
{
    compressor_v2_host::StageSelection selection;
    compressor_v2_host::Status status = compressor_v2_host::BuildContextSelection(context, true, selection);
    if (!status) {
        return compressor_v2_host::ReportTilingFailure(context, status.error);
    }
    const auto &key = selection.key;
    uint64_t encoded_key = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(key.Stage), static_cast<uint64_t>(key.Rows),
        static_cast<uint64_t>(key.Ratio), static_cast<uint64_t>(key.Fuse),
        static_cast<uint64_t>(key.Cols), static_cast<uint64_t>(key.Direct),
        static_cast<uint64_t>(key.Parts), static_cast<uint64_t>(key.Dim), static_cast<uint64_t>(key.Mode));
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    return compressor_v2_host::CompleteTiling(context, selection, encoded_key, platform.GetLibApiWorkSpaceSize());
}

static ge::graphStatus TilingParsePyptoCompressorV2Vector([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PyptoCompressorV2Vector)
    .Tiling(TilingPyptoCompressorV2Vector)
    .TilingParse<PyptoCompressorV2VectorCompileInfo>(TilingParsePyptoCompressorV2Vector);

} // namespace optiling
