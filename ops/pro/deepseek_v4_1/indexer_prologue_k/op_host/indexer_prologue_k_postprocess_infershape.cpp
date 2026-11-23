/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <register/op_impl_registry.h>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t K_CACHE_INPUT = 5;
constexpr uint32_t K_SCALE_CACHE_INPUT = 6;

static ge::graphStatus InferShapeIndexerPrologueKPostprocess(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("IndexerPrologueKPostprocess", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape *cacheShape = context->GetInputShape(K_CACHE_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheShape);
    gert::Shape *cacheOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheOutShape);
    *cacheOutShape = *cacheShape;

    const gert::Shape *scaleShape = context->GetInputShape(K_SCALE_CACHE_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    gert::Shape *scaleOutShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleOutShape);
    *scaleOutShape = *scaleShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeIndexerPrologueKPostprocess(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("IndexerPrologueKPostprocess", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_UINT8);
    context->SetOutputDataType(1, ge::DT_UINT8);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IndexerPrologueKPostprocess)
    .InferShape(InferShapeIndexerPrologueKPostprocess)
    .InferDataType(InferDataTypeIndexerPrologueKPostprocess);
} // namespace ops
