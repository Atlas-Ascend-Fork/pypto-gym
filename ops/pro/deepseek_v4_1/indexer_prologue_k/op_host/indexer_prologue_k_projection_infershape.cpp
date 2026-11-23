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

static ge::graphStatus InferShapeIndexerPrologueKProjection(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("IndexerPrologueKProjection", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape *latentShape = context->GetInputShape(0);
    const gert::Shape *wkShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, latentShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, wkShape);
    gert::Shape *projectedShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, projectedShape);
    projectedShape->SetDimNum(2);
    projectedShape->SetDim(0, latentShape->GetDim(0));
    projectedShape->SetDim(1, wkShape->GetDim(0));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeIndexerPrologueKProjection(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("IndexerPrologueKProjection", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_BF16);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IndexerPrologueKProjection)
    .InferShape(InferShapeIndexerPrologueKProjection)
    .InferDataType(InferDataTypeIndexerPrologueKProjection);
} // namespace ops
