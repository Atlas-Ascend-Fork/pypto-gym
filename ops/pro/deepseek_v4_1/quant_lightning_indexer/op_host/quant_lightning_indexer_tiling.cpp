/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "quant_lightning_indexer_tiling.h"
#include "quant_lightning_indexer_config.h"
// The common build force-includes this operator's generated data/key headers.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdio>
namespace optiling {
using qli::TensorSpec;
static qli::Dtype Type(ge::DataType t) {
    switch (t) { case ge::DT_UINT8:return qli::U8; case ge::DT_FLOAT:return qli::F32;
    case ge::DT_INT32:return qli::I32; case ge::DT_BF16:return qli::BF16; default:return qli::Invalid; }
}
static TensorSpec Spec(gert::TilingContext *ctx, size_t i) {
    const bool optional=i>=5 && i<=11;
    const auto *shape=optional?ctx->GetOptionalInputShape(i):ctx->GetRequiredInputShape(i);
    const auto *desc=optional?ctx->GetOptionalInputDesc(i):ctx->GetRequiredInputDesc(i);
    TensorSpec s;
    if (!shape || !desc) return s;
    const auto &dims=shape->GetStorageShape();
    if (dims.GetDimNum()==0) return s;
    s.present=true; s.dtype=Type(desc->GetDataType());
    for (size_t j=0;j<dims.GetDimNum();++j) s.dims.push_back(dims.GetDim(j));
    // ACLNN normalizes padded K storage into a contiguous workspace copy.
    // Original caller stride validation belongs to the Torch/API boundary.
    s.strides.resize(s.dims.size());
    int64_t stride=1;
    for (size_t j=s.dims.size();j>0;--j) { s.strides[j-1]=stride; stride*=s.dims[j-1]; }
    return s;
}
static ge::graphStatus ReadAttrs(gert::TilingContext *ctx, qli::Attrs &a) {
    const auto *attrs=ctx->GetAttrs();
    std::array<int64_t,7> v{};
    for (size_t i=0;i<v.size();++i) { const auto *p=attrs->GetInt(i); if (!p) return ge::GRAPH_FAILED; v[i]=*p; }
    const auto *lq=attrs->GetStr(7), *lk=attrs->GetStr(8); const auto *rv=attrs->GetBool(9);
    if (!lq || !lk || !rv) return ge::GRAPH_FAILED;
    a = {v[0],v[1],v[2],v[3],v[4],v[5],v[6],lq,lk,*rv};
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckPlanAndOutputs(gert::TilingContext *ctx, const qli::Config &c) {
    if (!qli::Shape(Spec(ctx,12),{c.plan_elems}) || Spec(ctx,12).dtype!=qli::I32) {
        auto p=Spec(ctx,12);
        std::fprintf(stderr,"GymQuantLightningIndexer: plan must be int32[%ld], actual rank=%zu size=%ld, cores=%ld/%ld\n",
            c.plan_elems,p.dims.size(),p.dims.empty()?-1:p.dims[0],c.cube,c.vector);
        return ge::GRAPH_FAILED;
    }
    for (size_t i=0;i<4;++i) {
        const auto *s=ctx->GetOutputShape(i); const auto *d=ctx->GetOutputDesc(i);
        if (!s || !d || d->GetDataType()!=(i==1?ge::DT_BF16:ge::DT_INT32)) return ge::GRAPH_FAILED;
        const auto &dims=s->GetStorageShape();
        if (dims.GetDimNum()!=(i==3?2:3) || dims.GetDim(0)!=c.rows || dims.GetDim(1)!=1 ||
            (i!=3 && dims.GetDim(2)!=(i==2?std::max(int64_t(1),c.capacity):c.topk))) return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static void WriteConfig(QliTiling *td, const qli::Config &c) {
    td->rows=c.rows;
    td->batches=c.batches;
    td->max_q=c.max_q;
    td->max_k=c.max_k;
    td->page_size=c.page_size;
    td->k_rows=c.k_rows;
    td->s_rows=c.s_rows;
    td->k_page_stride=c.k_page_stride;
    td->ks_page_stride=c.ks_page_stride;
    td->k_stride=c.k_stride;
    td->ks_stride=c.ks_stride;
    td->table_cols=c.table_cols;
    td->cube=c.cube;
    td->vector=c.vector;
    td->plan_elems=c.plan_elems;
    td->capacity=c.capacity;
    td->flags=c.flags;
    td->cmp_ratio=c.cmp_ratio;
    td->ws_scores=c.ws_scores;
    td->ws_index_map=c.ws_index_map;
    td->ws_partial=c.ws_partial;
    td->score_cols=c.score_cols;
    td->aux_cols=c.aux_cols;
    td->topk=c.topk;
}

static ge::graphStatus Tiling(gert::TilingContext *ctx) {
    if (!ctx || !ctx->GetAttrs() || !ctx->GetPlatformInfo()) return ge::GRAPH_FAILED;
    qli::Attrs a{};
    if (ReadAttrs(ctx, a) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    std::array<TensorSpec,12> x;
    for (size_t i=0;i<x.size();++i) x[i]=Spec(ctx,i);
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    qli::Config c; std::string error;
    if (!qli::Derive(x,a,platform.GetCoreNumAic(),platform.GetCoreNumAiv(),c,error)) {
        std::fprintf(stderr,"GymQuantLightningIndexer: %s\n",error.c_str()); return ge::GRAPH_FAILED;
    }
    if (CheckPlanAndOutputs(ctx, c) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    auto *td=ctx->GetTilingData<QliTiling>();
    auto *ws=ctx->GetWorkspaceSizes(1);
    if (!td || !ws || !ctx->GetRawTilingData() || ctx->GetRawTilingData()->GetCapacity()<sizeof(*td)) return ge::GRAPH_FAILED;
    WriteConfig(td, c);
    ctx->GetRawTilingData()->SetDataSize(sizeof(*td));
    const uint64_t key=GET_TPL_TILING_KEY(uint64_t(c.fast),uint64_t(c.tile_n),uint64_t(c.select),uint64_t(c.shard),uint64_t(a.values),uint64_t(c.fixed),uint64_t(a.mask),uint64_t(c.ratio_mode));
    ctx->SetTilingKey(key); ctx->SetBlockDim(c.cube); ws[0]=c.workspace_bytes;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Parse(gert::TilingParseContext *ctx) { return ctx?ge::GRAPH_SUCCESS:ge::GRAPH_FAILED; }
IMPL_OP_OPTILING(GymQuantLightningIndexer).Tiling(Tiling).TilingParse<QliCompileInfo>(Parse);
}
