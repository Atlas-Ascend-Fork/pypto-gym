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
 * \file attn_prologue_def.cpp
 * \brief AttnPrologue operator registration.
 *
 * Input order mirrors the delivery-form kernel signature in op_kernel/
 * (inputs first, then outputs q/qr/descale_qr, then workspace/tiling).
 * The three projection weights are FRACTAL_NZ (caller applies to_nz once at
 * weight-load time); all other tensors are ND.
 */

#include "register/op_def_registry.h"

namespace ops {
class AttnPrologue : public OpDef {
public:
    explicit AttnPrologue(const char *name) : OpDef(name)
    {
        RegisterInput("x", ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, true);
        RegisterInput("wqa", ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ, true);
        RegisterInput("wqb", ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ, true);
        RegisterInput("wkv", ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ, true);
        RegisterInput("descale_x", ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, true);
        RegisterInput("descale_wqa", ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, true);
        RegisterInput("descale_wqb", ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, true);
        RegisterInput("descale_wkv", ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, true);
        RegisterInput("norm_weight_qr", ge::DT_FLOAT, ge::FORMAT_ND, true);
        RegisterInput("norm_weight_kv", ge::DT_FLOAT, ge::FORMAT_ND, true);
        RegisterInput("rope_sin", ge::DT_FLOAT, ge::FORMAT_ND, true);
        RegisterInput("rope_cos", ge::DT_FLOAT, ge::FORMAT_ND, true);
        RegisterInput("cache_index", ge::DT_INT64, ge::FORMAT_ND, true);
        RegisterInput("kv_cache", ge::DT_UINT8, ge::FORMAT_ND, false);
        RegisterInput("qr_norm", ge::DT_BF16, ge::FORMAT_ND, false);

        RegisterOutput("q", ge::DT_BF16);
        RegisterOutput("qr", ge::DT_FLOAT8_E4M3FN);
        RegisterOutput("descale_qr", ge::DT_FLOAT8_E8M0);

        this->Attr("norm_eps").AttrType(OPTIONAL).Float(1e-6f);
        this->Attr("block_stride").AttrType(OPTIONAL).Int(0);
        this->Attr("row_stride").AttrType(OPTIONAL).Int(0);
        this->Attr("storage_bytes").AttrType(OPTIONAL).Int(0);
        this->Attr("emit_qr_norm").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "attn_prologue")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", aicore_config);
    }

private:
    void RegisterInput(const char *name, ge::DataType dtype, ge::Format format, bool contiguous)
    {
        auto &param = this->Input(name).ParamType(REQUIRED).DataType({dtype}).Format({format});
        if (contiguous) {
            param.AutoContiguous();
        } else {
            param.IgnoreContiguous();
        }
    }

    void RegisterOutput(const char *name, ge::DataType dtype)
    {
        this->Output(name).ParamType(REQUIRED).DataType({dtype}).Format({ge::FORMAT_ND});
    }
};
OP_ADD(AttnPrologue);
} // namespace ops
