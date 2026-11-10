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
 * \file mhc_pre_fuse_infershape.cpp
 * \brief MhcPreFuse InferShape / InferDataType.
 *
 * residual is [T, 4, H]; post_mix is [T, 4] FP32, comb_mix [T, 16] FP32 and
 * layer_input [T, H] BF16.
 */

#include <register/op_impl_registry.h>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t RESIDUAL_INDEX = 4;
constexpr uint32_t POST_INDEX = 0;
constexpr uint32_t COMB_INDEX = 1;
constexpr uint32_t LAYER_INDEX = 2;
constexpr int64_t MHC = 4;
constexpr int64_t MHC2 = 16;

static ge::graphStatus InferShapeMhcPreFuse(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("MhcPreFuse", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    const gert::Shape *residualShape = context->GetInputShape(RESIDUAL_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, residualShape);
    OP_CHECK_IF(residualShape->GetDimNum() != 3,
                OP_LOGE(context->GetNodeName(), "residual must be rank 3 [T,4,H]."),
                return ge::GRAPH_FAILED);
    const int64_t t = residualShape->GetDim(0);
    const int64_t h = residualShape->GetDim(2);

    gert::Shape *postShape = context->GetOutputShape(POST_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, postShape);
    postShape->SetDimNum(2);
    postShape->SetDim(0, t);
    postShape->SetDim(1, MHC);

    gert::Shape *combShape = context->GetOutputShape(COMB_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, combShape);
    combShape->SetDimNum(2);
    combShape->SetDim(0, t);
    combShape->SetDim(1, MHC2);

    gert::Shape *layerShape = context->GetOutputShape(LAYER_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, layerShape);
    layerShape->SetDimNum(2);
    layerShape->SetDim(0, t);
    layerShape->SetDim(1, h);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMhcPreFuse(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("MhcPreFuse", "InferDataTypeContext is nullptr!"), return ge::GRAPH_FAILED);
    context->SetOutputDataType(POST_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(COMB_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(LAYER_INDEX, ge::DT_BF16);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcPreFuse)
    .InferShape(InferShapeMhcPreFuse)
    .InferDataType(InferDataTypeMhcPreFuse);
} // namespace ops
