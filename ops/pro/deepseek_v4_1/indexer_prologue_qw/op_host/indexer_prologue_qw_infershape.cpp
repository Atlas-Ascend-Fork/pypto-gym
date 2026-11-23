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
 * \file indexer_prologue_qw_infershape.cpp
 * \brief IndexerPrologueQw InferShape / InferDataType.
 *
 * ``T`` is the first dimension of ``qr``; the three outputs follow it.
 */

#include <register/op_impl_registry.h>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t QR_INDEX = 1;
constexpr uint32_t Q_INDEX = 0;
constexpr uint32_t DESCALE_Q_INDEX = 1;
constexpr uint32_t W_INDEX = 2;
constexpr int64_t N_HEADS = 32;
constexpr int64_t HEAD_DIM = 128;

static ge::graphStatus InferShapeIndexerPrologueQw(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("IndexerPrologueQw", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    const gert::Shape *qrShape = context->GetInputShape(QR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qrShape);
    const int64_t t = qrShape->GetDim(0);

    gert::Shape *qShape = context->GetOutputShape(Q_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    gert::Shape *descaleShape = context->GetOutputShape(DESCALE_Q_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, descaleShape);
    gert::Shape *wShape = context->GetOutputShape(W_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShape);

    *qShape = gert::Shape({t, N_HEADS, HEAD_DIM / 2});
    *descaleShape = gert::Shape({t, N_HEADS, 2, 2});
    *wShape = gert::Shape({t, N_HEADS});
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeIndexerPrologueQw(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("IndexerPrologueQw", "InferDataTypeContext is nullptr!"), return ge::GRAPH_FAILED);
    context->SetOutputDataType(Q_INDEX, ge::DT_UINT8);
    context->SetOutputDataType(DESCALE_Q_INDEX, ge::DT_UINT8);
    context->SetOutputDataType(W_INDEX, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IndexerPrologueQw)
    .InferShape(InferShapeIndexerPrologueQw)
    .InferDataType(InferDataTypeIndexerPrologueQw);
} // namespace ops
