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
 * \file mhc_post_fwd_def.cpp
 * \brief MhcPostFwd operator registration.
 *
 * out[n, mo, h] = post_layer_mix[n, mo] * x[n, h]
 *                 + sum_mi comb_res_mix[n, mi, mo] * residual[n, mi, h]
 *
 * All four inputs are contiguous BF16/FP32 tensors; out is a regular output
 * (the kernel only writes it). The host tiling flattens (n0, n1) to n_tok and
 * selects the compile-time hidden tile width through the tiling key.
 */

#include "register/op_def_registry.h"

namespace ops {
class MhcPostFwd : public OpDef {
public:
    explicit MhcPostFwd(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("residual")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("post_layer_mix")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("comb_res_mix")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "mhc_post_fwd")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(MhcPostFwd);
} // namespace ops
