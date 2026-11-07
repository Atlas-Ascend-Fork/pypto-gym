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
 * \file mixed_quant_sparse_flash_mla_infershape.cpp
 * \brief MixedQuantSparseFlashMla InferShape / InferDataType.
 *
 * Mirrors the Python host wrapper: out keeps the q shape [T1, 64, 512] and
 * lse is always [1, T1, 64] float32.
 */

#include <register/op_impl_registry.h>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t Q_INDEX = 0;

static ge::graphStatus InferShapeMixedQuantSparseFlashMla(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("MixedQuantSparseFlashMla", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter MixedQuantSparseFlashMla InferShape impl.");
    const gert::Shape *qShape = context->GetInputShape(Q_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);

    OP_CHECK_IF(qShape->GetDimNum() != 3,
                OP_LOGE(context->GetNodeName(), "q must be rank 3 [T1, 64, 512], got %zu", qShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    int64_t t1 = qShape->GetDim(0);
    int64_t numHeads = qShape->GetDim(1);

    gert::Shape *outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    *outShape = *qShape;

    gert::Shape *lseShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, lseShape);
    lseShape->SetDimNum(3);
    lseShape->SetDim(0, 1);
    lseShape->SetDim(1, t1);
    lseShape->SetDim(2, numHeads);

    OP_LOGD(context->GetNodeName(), "MixedQuantSparseFlashMla InferShape end.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMixedQuantSparseFlashMla(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("MixedQuantSparseFlashMla", "InferDataTypeContext is nullptr!"), return ge::GRAPH_FAILED);
    // out follows q (BF16); lse is always FP32
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MixedQuantSparseFlashMla)
    .InferShape(InferShapeMixedQuantSparseFlashMla)
    .InferDataType(InferDataTypeMixedQuantSparseFlashMla);
} // namespace ops