/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_def_registry.h"

namespace ops {
// Both private operators have the same pointer and attribute order. They use
// separate kernel entries because the projection/fused and reduction stages
// compile to CV and Vector kernel types respectively.
static void DefineCompressorV2Stage(OpDef &op, const char *opFile)
{
    op.Input("x").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();
    op.Input("wkv").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();
    op.Input("wgate").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();
    op.Input("cache_in").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();
    op.Input("table").ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();
    op.Input("cu").ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();
    op.Input("used").ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();
    op.Input("start").ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();
    op.Input("projected_in").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();
    op.Input("history_in").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();
    op.Output("out").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();
    op.Output("cache_out").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();
    op.Output("projected_out").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();
    op.Output("history_out").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();

    op.Attr("stage").AttrType(REQUIRED).Int();
    op.Attr("cmp_ratio").AttrType(REQUIRED).Int();
    op.Attr("has_used").AttrType(REQUIRED).Bool();
    op.Attr("has_start").AttrType(REQUIRED).Bool();

    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("opFile.value", opFile)
        .ExtendCfgInfo("opInterface.value", opFile)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
    op.AICore().AddConfig("ascend950", config);
}
} // namespace ops

namespace ops {
class PyptoCompressorV2Stage : public OpDef {
public:
    explicit PyptoCompressorV2Stage(const char *name) : OpDef(name)
    {
        DefineCompressorV2Stage(*this, "compressor_v2");
    }
};
OP_ADD(PyptoCompressorV2Stage);
} // namespace ops

namespace ops {
class PyptoCompressorV2Vector : public OpDef {
public:
    explicit PyptoCompressorV2Vector(const char *name) : OpDef(name)
    {
        DefineCompressorV2Stage(*this, "compressor_v2_vector");
    }
};
OP_ADD(PyptoCompressorV2Vector);
} // namespace ops
