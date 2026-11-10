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
 * \file mhc_post_fwd_infershape.cpp
 * \brief MhcPostFwd InferShape / InferDataType.
 *
 * out has exactly the residual shape [n0, n1, MHC, h] and is BF16.
 */

#include <register/op_impl_registry.h>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t RESIDUAL_INDEX = 1;
constexpr uint32_t OUT_INDEX = 0;

static ge::graphStatus InferShapeMhcPostFwd(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("MhcPostFwd", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    const gert::Shape *residualShape = context->GetInputShape(RESIDUAL_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, residualShape);
    gert::Shape *outShape = context->GetOutputShape(OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    *outShape = *residualShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMhcPostFwd(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("MhcPostFwd", "InferDataTypeContext is nullptr!"), return ge::GRAPH_FAILED);
    context->SetOutputDataType(OUT_INDEX, ge::DT_BF16);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcPostFwd)
    .InferShape(InferShapeMhcPostFwd)
    .InferDataType(InferDataTypeMhcPostFwd);
} // namespace ops
