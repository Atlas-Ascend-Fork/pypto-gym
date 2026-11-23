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
 * \file indexer_prologue_k_fused_infershape.cpp
 * \brief IndexerPrologueK InferShape / InferDataType (in-place cache outputs).
 */

#include <register/op_impl_registry.h>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t K_CACHE_INDEX = 6;
constexpr uint32_t K_SCALE_CACHE_INDEX = 7;
constexpr uint32_t OUT_K_CACHE_INDEX = 0;
constexpr uint32_t OUT_K_SCALE_CACHE_INDEX = 1;

static ge::graphStatus InferShapeIndexerPrologueK(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("IndexerPrologueK", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    const gert::Shape *cacheShape = context->GetInputShape(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheShape);
    gert::Shape *cacheOutShape = context->GetOutputShape(OUT_K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheOutShape);
    *cacheOutShape = *cacheShape;

    const gert::Shape *scaleShape = context->GetInputShape(K_SCALE_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    gert::Shape *scaleOutShape = context->GetOutputShape(OUT_K_SCALE_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleOutShape);
    *scaleOutShape = *scaleShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeIndexerPrologueK(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("IndexerPrologueK", "InferDataTypeContext is nullptr!"), return ge::GRAPH_FAILED);
    context->SetOutputDataType(OUT_K_CACHE_INDEX, ge::DT_UINT8);
    context->SetOutputDataType(OUT_K_SCALE_CACHE_INDEX, ge::DT_UINT8);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IndexerPrologueK)
    .InferShape(InferShapeIndexerPrologueK)
    .InferDataType(InferDataTypeIndexerPrologueK);
} // namespace ops
