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
 * \file mhc_pre_fuse_def.cpp
 * \brief MhcPreFuse operator registration.
 *
 * Fused mHC pre pass:
 *   RMSNorm -> split mixes -> Sinkhorn -> residual pre-apply
 * producing post_mix [T,4] FP32, comb_mix [T,16] FP32 and layer_input [T,H] BF16.
 *
 * The five knobs are runtime attributes; the structural token/hidden chunking is
 * carried by the tiling key.
 */

#include "register/op_def_registry.h"

namespace ops {
class MhcPreFuse : public OpDef {
public:
    explicit MhcPreFuse(const char *name) : OpDef(name)
    {
        this->Input("gemm_out_mul")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gemm_out_sqrsum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mhc_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mhc_base")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("residual")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("post_mix")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("comb_mix")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("layer_input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});

        this->Attr("rms_eps").AttrType(OPTIONAL).Float(1.0e-6F);
        this->Attr("mhc_pre_eps").AttrType(OPTIONAL).Float(1.0e-6F);
        this->Attr("mhc_sinkhorn_eps").AttrType(OPTIONAL).Float(1.0e-6F);
        this->Attr("mhc_post_mult_value").AttrType(OPTIONAL).Float(1.0F);
        this->Attr("sinkhorn_repeat").AttrType(OPTIONAL).Int(10);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "mhc_pre_fuse")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(MhcPreFuse);
} // namespace ops
