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
class MegaMhc : public OpDef {
public:
    explicit MegaMhc(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("residual").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("previous_mix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("post_mix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("comb_res_mix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("fn").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("mix_scales").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("mix_bases").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("rmsnorm_weight").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Output("new_residual").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Output("new_post_mix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("new_comb_res_mix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Output("new_previous_mix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("hc_norm_eps").AttrType(OPTIONAL).Float(2e-5);
        this->Attr("hc_pre_eps").AttrType(OPTIONAL).Float(3e-4);
        this->Attr("hc_post_scale").AttrType(OPTIONAL).Float(1.25);
        this->Attr("sinkhorn_eps").AttrType(OPTIONAL).Float(2e-6);
        this->Attr("num_sinkhorn_iters").AttrType(OPTIONAL).Int(10);
        this->Attr("rmsnorm_eps").AttrType(OPTIONAL).Float(7e-6);
        // 0: automatic; 16: force batch16. Public batch8 remains unsupported.
        this->Attr("batch").AttrType(OPTIONAL).Int(0);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "mega_mhc")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(MegaMhc);
} // namespace ops
