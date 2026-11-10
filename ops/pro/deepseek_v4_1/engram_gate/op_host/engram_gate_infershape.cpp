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
 * \file engram_gate_infershape.cpp
 * \brief EngramGate InferShape / InferDataType.
 *
 * y has exactly the x shape [t, hc, dim] and is BF16.
 */

#include <register/op_impl_registry.h>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t Y_INDEX = 0;

static ge::graphStatus InferShapeEngramGate(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("EngramGate", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    const gert::Shape *xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape *yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeEngramGate(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("EngramGate", "InferDataTypeContext is nullptr!"), return ge::GRAPH_FAILED);
    context->SetOutputDataType(Y_INDEX, ge::DT_BF16);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EngramGate)
    .InferShape(InferShapeEngramGate)
    .InferDataType(InferDataTypeEngramGate);
} // namespace ops
