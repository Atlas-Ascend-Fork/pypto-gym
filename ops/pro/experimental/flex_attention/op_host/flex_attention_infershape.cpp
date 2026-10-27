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
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;

static ge::graphStatus InferShapeFlexAttention(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("FlexAttention", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter FlexAttention InferShape impl.");
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    gert::Shape *outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    // attn_out has the same shape as query: [total_q, num_heads, head_dim]
    *outShape = *queryShape;
    OP_LOGD(context->GetNodeName(), "FlexAttention InferShape end.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFlexAttention(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("FlexAttention", "InferDataTypeContext is nullptr!"), return ge::GRAPH_FAILED);
    const auto *queryDtype = context->GetInputDesc(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDtype);
    // attn_out dtype follows query (FLOAT16 / BFLOAT16)
    context->SetOutputDataType(0, queryDtype->GetDataType());
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlexAttention)
    .InferShape(InferShapeFlexAttention)
    .InferDataType(InferDataTypeFlexAttention);
} // namespace ops
