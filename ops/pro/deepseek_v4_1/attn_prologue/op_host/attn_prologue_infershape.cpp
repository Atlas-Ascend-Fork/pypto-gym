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
 * \file attn_prologue_infershape.cpp
 * \brief AttnPrologue InferShape / InferDataType.
 */

#include <register/op_impl_registry.h>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t Q_INDEX = 0;
constexpr uint32_t QR_INDEX = 1;
constexpr uint32_t DESCALE_QR_INDEX = 2;
constexpr int64_t N_HEADS = 64;
constexpr int64_t HEAD_DIM = 512;
constexpr int64_t RANK = 1280;
constexpr int64_t RANK_GROUPS = 20;

static ge::graphStatus InferShapeAttnPrologue(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("AttnPrologue", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape *xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_IF(xShape->GetDimNum() != 2,
                OP_LOGE(context->GetNodeName(),
                        "attn_prologue: x must be rank 2 [T,H], got rank %zu.",
                        xShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    const int64_t t = xShape->GetDim(0);
    OP_CHECK_IF(t < 1,
                OP_LOGE(context->GetNodeName(),
                        "attn_prologue: T must be positive, got %ld.", t),
                return ge::GRAPH_FAILED);

    gert::Shape *qShape = context->GetOutputShape(Q_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    qShape->SetDimNum(3);
    qShape->SetDim(0, t);
    qShape->SetDim(1, N_HEADS);
    qShape->SetDim(2, HEAD_DIM);

    gert::Shape *qrShape = context->GetOutputShape(QR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qrShape);
    qrShape->SetDimNum(2);
    qrShape->SetDim(0, t);
    qrShape->SetDim(1, RANK);

    gert::Shape *scaleShape = context->GetOutputShape(DESCALE_QR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    scaleShape->SetDimNum(3);
    scaleShape->SetDim(0, t);
    scaleShape->SetDim(1, RANK_GROUPS);
    scaleShape->SetDim(2, 2);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAttnPrologue(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("AttnPrologue", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(Q_INDEX, ge::DT_BF16);
    context->SetOutputDataType(QR_INDEX, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(DESCALE_QR_INDEX, ge::DT_FLOAT8_E8M0);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AttnPrologue)
    .InferShape(InferShapeAttnPrologue)
    .InferDataType(InferDataTypeAttnPrologue);
} // namespace ops
