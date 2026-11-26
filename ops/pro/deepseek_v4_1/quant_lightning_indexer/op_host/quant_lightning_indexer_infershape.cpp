/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include "register/op_impl_registry.h"
namespace ops {
// Required by generated ACLNN InferShape; no separate GEIR route is shipped.
static ge::graphStatus Infer(gert::InferShapeContext *ctx) {
    if (!ctx || !ctx->GetInputShape(0) || !ctx->GetAttrs() ||
        ctx->GetInputShape(0)->GetDimNum()!=3) return ge::GRAPH_FAILED;
    const auto *top=ctx->GetAttrs()->GetInt(0), *cap=ctx->GetAttrs()->GetInt(5);
    if (!top || !cap || *top<1 || *top>8192 || (*cap!=-1 && *cap<=0)) return ge::GRAPH_FAILED;
    const int64_t rows=ctx->GetInputShape(0)->GetDim(0);
    for (size_t i=0;i<4;++i) {
        auto *s=ctx->GetOutputShape(i); if (!s) return ge::GRAPH_FAILED;
        s->SetDimNum(i==3?2:3); s->SetDim(0,rows); s->SetDim(1,1);
        if (i!=3) s->SetDim(2,i==2?std::max(int64_t(1),*cap):*top);
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferType(gert::InferDataTypeContext *ctx) {
    if (!ctx) return ge::GRAPH_FAILED;
    for (size_t i=0;i<4;++i) ctx->SetOutputDataType(i,i==1?ge::DT_BF16:ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(GymQuantLightningIndexer).InferShape(Infer).InferDataType(InferType);
}
