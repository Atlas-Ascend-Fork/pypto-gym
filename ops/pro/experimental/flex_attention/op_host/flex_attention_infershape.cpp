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
 * \file flex_attention_infershape.cpp
 * \brief FlexAttention InferShape / InferDataType.
 *
 * Mirrors the Python host wrapper: varlen (q [tq, n, d]) and bshd (q [B, S, n, d])
 * both produce attn_out [tq, n, d] / stats [n, tq] with tq = B * S in bshd mode.
 */

#include <cstring>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;

static bool IsBshd(gert::InferShapeContext *context)
{
    const char *layout = context->GetAttrs() == nullptr ? nullptr : context->GetAttrs()->GetStr(0);
    return layout != nullptr && std::strcmp(layout, "bshd") == 0;
}

static ge::graphStatus InferShapeFlexAttention(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("FlexAttention", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter FlexAttention InferShape impl.");
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    const auto queryShapeDimNum = queryShape->GetDimNum();
    const bool isBshd = IsBshd(context);
    gert::Shape *outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    int64_t totalQ = 0;
    int64_t numHeads = 0;
    if (isBshd) {
        OP_CHECK_IF(queryShapeDimNum != 4,
                    OP_LOGE(context->GetNodeName(), "bshd layout requires query rank 4, got %zu", queryShapeDimNum),
                    return ge::GRAPH_FAILED);
        // bshd q [B, S, H, D] -> delivery-form [B*S, H, D]
        totalQ = queryShape->GetDim(0) * queryShape->GetDim(1);
        numHeads = queryShape->GetDim(2);
        outShape->SetDimNum(3);
        outShape->SetDim(0, totalQ);
        outShape->SetDim(1, numHeads);
        outShape->SetDim(2, queryShape->GetDim(3));
    } else {
        OP_CHECK_IF(queryShapeDimNum != 3,
                    OP_LOGE(context->GetNodeName(), "varlen layout requires query rank 3, got %zu", queryShapeDimNum),
                    return ge::GRAPH_FAILED);
        totalQ = queryShape->GetDim(0);
        numHeads = queryShape->GetDim(1);
        *outShape = *queryShape;
    }

    // softmax_max_out / softmax_sum_out: [num_heads, total_q] fp32
    gert::Shape *maxShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxShape);
    gert::Shape *sumShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumShape);
    maxShape->SetDimNum(2);
    maxShape->SetDim(0, numHeads);
    maxShape->SetDim(1, totalQ);
    *sumShape = *maxShape;

    OP_LOGD(context->GetNodeName(), "FlexAttention InferShape end.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFlexAttention(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("FlexAttention", "InferDataTypeContext is nullptr!"), return ge::GRAPH_FAILED);
    const auto *queryDtype = context->GetInputDesc(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDtype);
    // attn_out dtype follows query (FLOAT16 / BFLOAT16); stats are always FP32
    context->SetOutputDataType(0, queryDtype->GetDataType());
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlexAttention)
    .InferShape(InferShapeFlexAttention)
    .InferDataType(InferDataTypeFlexAttention);
} // namespace ops
