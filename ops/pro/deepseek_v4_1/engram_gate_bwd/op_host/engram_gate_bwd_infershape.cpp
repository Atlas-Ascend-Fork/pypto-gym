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
 * \file engram_gate_bwd_infershape.cpp
 * \brief EngramGateBwd InferShape / InferDataType.
 *
 * grad_x/grad_key mirror x/key and grad_value mirrors value.  grad_w_partial is
 * the kernel's [16, hc, dim] reduction scratch; the bridge sums it over the
 * partial axis to produce the public grad_weight.
 */

#include <register/op_impl_registry.h>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t X_INDEX = 1;
constexpr uint32_t KEY_INDEX = 2;
constexpr uint32_t VALUE_INDEX = 3;
constexpr uint32_t GRAD_X_INDEX = 0;
constexpr uint32_t GRAD_KEY_INDEX = 1;
constexpr uint32_t GRAD_VALUE_INDEX = 2;
constexpr uint32_t GRAD_W_PARTIAL_INDEX = 3;
constexpr int64_t NPART = 16;

static ge::graphStatus InferShapeEngramGateBwd(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("EngramGateBwd", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    const gert::Shape *xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape *valueShape = context->GetInputShape(VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    gert::Shape *gradXShape = context->GetOutputShape(GRAD_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradXShape);
    gert::Shape *gradKeyShape = context->GetOutputShape(GRAD_KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradKeyShape);
    gert::Shape *gradValueShape = context->GetOutputShape(GRAD_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradValueShape);
    gert::Shape *partialShape = context->GetOutputShape(GRAD_W_PARTIAL_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, partialShape);

    *gradXShape = *xShape;
    *gradKeyShape = *keyShape;
    *gradValueShape = *valueShape;
    *partialShape = gert::Shape({NPART, xShape->GetDim(1), xShape->GetDim(2)});
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeEngramGateBwd(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("EngramGateBwd", "InferDataTypeContext is nullptr!"), return ge::GRAPH_FAILED);
    context->SetOutputDataType(GRAD_X_INDEX, ge::DT_BF16);
    context->SetOutputDataType(GRAD_KEY_INDEX, ge::DT_BF16);
    context->SetOutputDataType(GRAD_VALUE_INDEX, ge::DT_BF16);
    context->SetOutputDataType(GRAD_W_PARTIAL_INDEX, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EngramGateBwd)
    .InferShape(InferShapeEngramGateBwd)
    .InferDataType(InferDataTypeEngramGateBwd);
} // namespace ops
