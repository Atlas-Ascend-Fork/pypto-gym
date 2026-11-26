/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <limits>
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus Infer(gert::InferShapeContext *ctx)
{
    if (!ctx) return ge::GRAPH_FAILED;
    const auto *ids=ctx->GetInputShape(0), *mult=ctx->GetInputShape(1), *offs=ctx->GetInputShape(3);
    auto *out=ctx->GetOutputShape(0);
    if (!ids || !mult || !offs || !out || ids->GetDimNum()!=2 ||
        mult->GetDimNum()!=2 || offs->GetDimNum()!=2) return ge::GRAPH_FAILED;
    out->SetDimNum(3);
    out->SetDim(0,mult->GetDim(0)); out->SetDim(1,ids->GetDim(0)); out->SetDim(2,offs->GetDim(1));
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Dtype(gert::InferDataTypeContext *ctx)
{
    if (!ctx) return ge::GRAPH_FAILED;
    ctx->SetOutputDataType(0,ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(EngramHash).InferShape(Infer).InferDataType(Dtype);
}
