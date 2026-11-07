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
 * \file quant_sparse_lightning_indexer_infershape.cpp
 * \brief QuantSparseLightningIndexer InferShape / InferDataType.
 *
 * q [T, 32, 64] -> indices [T, 1, topk] int32 and values [T, 1, topk] bf16.
 * `topk` is the first registered attribute; both outputs always exist (the
 * WriteValues=0 specialization simply does not store scores, mirroring the jit
 * host which still allocates the score buffer).
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t Q_INDEX = 0;
constexpr uint32_t TOPK_ATTR_INDEX = 0;
constexpr int64_t N2_DIM = 1;

static ge::graphStatus InferShapeQsli(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("QuantSparseLightningIndexer", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter QuantSparseLightningIndexer InferShape impl.");
    const gert::Shape *qShape = context->GetInputShape(Q_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    OP_CHECK_IF(qShape->GetDimNum() != 3,
                OP_LOGE(context->GetNodeName(), "q requires rank 3 [T, 32, 64], got %zu",
                        qShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    const int64_t totalQ = qShape->GetDim(0);

    const auto *attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE(context->GetNodeName(), "QuantSparseLightningIndexer attrs is nullptr!"),
                return ge::GRAPH_FAILED);
    // GetInt hands back a pointer into the attribute blob; reject a null.
    const int64_t *topkAttr = attrs->GetInt(TOPK_ATTR_INDEX);
    OP_CHECK_IF(topkAttr == nullptr,
                OP_LOGE(context->GetNodeName(), "QuantSparseLightningIndexer topk attr is null!"),
                return ge::GRAPH_FAILED);
    const int64_t topk = *topkAttr;
    OP_CHECK_IF(topk < 1,
                OP_LOGE(context->GetNodeName(), "topk must be >= 1, got %ld", topk),
                return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < 2; ++i) {
        gert::Shape *outShape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
        outShape->SetDimNum(3);
        outShape->SetDim(0, totalQ);
        outShape->SetDim(1, N2_DIM);
        outShape->SetDim(2, topk);
    }
    OP_LOGD(context->GetNodeName(), "QuantSparseLightningIndexer InferShape end.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeQsli(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("QuantSparseLightningIndexer", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_BF16);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantSparseLightningIndexer)
    .InferShape(InferShapeQsli)
    .InferDataType(InferDataTypeQsli);
} // namespace ops
