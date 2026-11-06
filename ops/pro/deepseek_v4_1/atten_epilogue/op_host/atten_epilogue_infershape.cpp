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
 * \file atten_epilogue_infershape.cpp
 * \brief AttenEpilogue InferShape / InferDataType.
 *
 * o [T, 64, 512] -> out [T, 5120]; the projection width is a fixed constant of
 * the fused shape contract (GEMM2 N), so only T is carried over from the input.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t O_INDEX = 0;
constexpr uint32_t OUT_FEATURES = 5120;

static ge::graphStatus InferShapeAttenEpilogue(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("AttenEpilogue", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter AttenEpilogue InferShape impl.");
    const gert::Shape *oShape = context->GetInputShape(O_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, oShape);
    OP_CHECK_IF(oShape->GetDimNum() != 3,
                OP_LOGE(context->GetNodeName(), "o requires rank 3 [T, 64, 512], got %zu",
                        oShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    gert::Shape *outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    outShape->SetDimNum(2);
    outShape->SetDim(0, oShape->GetDim(0));
    outShape->SetDim(1, OUT_FEATURES);

    OP_LOGD(context->GetNodeName(), "AttenEpilogue InferShape end.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAttenEpilogue(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("AttenEpilogue", "InferDataTypeContext is nullptr!"), return ge::GRAPH_FAILED);
    const auto *oDtype = context->GetInputDesc(O_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, oDtype);
    // out dtype follows o (BF16 by contract)
    context->SetOutputDataType(0, oDtype->GetDataType());
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AttenEpilogue)
    .InferShape(InferShapeAttenEpilogue)
    .InferDataType(InferDataTypeAttenEpilogue);
} // namespace ops
