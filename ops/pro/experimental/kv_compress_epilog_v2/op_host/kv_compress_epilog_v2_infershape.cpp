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
 * \file kv_compress_epilog_v2_infershape.cpp
 * \brief KvCompressEpilogV2 InferShape / InferDataType.
 *
 * The cache output is the in-place update target: its shape and dtype follow
 * the cache input exactly.
 */

#include <register/op_impl_registry.h>
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t CACHE_INDEX = 0;
constexpr uint32_t CACHE_OUT_INDEX = 0;

static ge::graphStatus InferShapeKvCompressEpilogV2(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("KvCompressEpilogV2", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    const gert::Shape *cacheShape = context->GetInputShape(CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheShape);
    gert::Shape *cacheOutShape = context->GetOutputShape(CACHE_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheOutShape);
    *cacheOutShape = *cacheShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeKvCompressEpilogV2(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("KvCompressEpilogV2", "InferDataTypeContext is nullptr!"), return ge::GRAPH_FAILED);
    context->SetOutputDataType(CACHE_OUT_INDEX, context->GetInputDataType(CACHE_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvCompressEpilogV2)
    .InferShape(InferShapeKvCompressEpilogV2)
    .InferDataType(InferDataTypeKvCompressEpilogV2);
} // namespace ops
