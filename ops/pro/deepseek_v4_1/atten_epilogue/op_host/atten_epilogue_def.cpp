/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file atten_epilogue_def.cpp
 * \brief AttenEpilogue operator registration.
 *
 * Fused attention-output epilogue: inverse RoPE + regroup + OCP MXFP8 quantization
 * + two block-scaled GEMMs, delivered as a single PyPTO kernel launch.
 *
 * Input/output order mirrors the delivery-form kernel signature in op_kernel/
 * (inputs first, then the output `out`, then workspace + tiling).  Only `T`
 * (dim 0 of `o`/`rope_sin`/`rope_cos`/`out`) is dynamic; every other dimension is
 * a compile-time constant of the fused shape contract, so no attribute and no
 * tiling data dependency are needed: the host tiling derives everything from the
 * `o` shape and reports the flat GM workspace the kernel carves its aq/asc/yq/ys
 * payloads out of.
 */

#include "register/op_def_registry.h"

namespace ops {
class AttenEpilogue : public OpDef {
public:
    explicit AttenEpilogue(const char *name) : OpDef(name)
    {
        this->Input("o")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("woa")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("wob")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("descale_woa")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("descale_wob")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rope_sin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rope_cos")
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
            .ExtendCfgInfo("opFile.value", "atten_epilogue")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        // Keep host shape validation active for empty outputs: T must be positive.
        // Otherwise the ACLNN executor skips tiling and silently returns success.
        this->AICore().LaunchWithZeroEleOutputTensors(true);
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(AttenEpilogue);
} // namespace ops
