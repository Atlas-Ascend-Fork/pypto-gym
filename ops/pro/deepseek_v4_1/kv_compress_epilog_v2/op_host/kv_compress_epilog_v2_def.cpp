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
 * \file kv_compress_epilog_v2_def.cpp
 * \brief KvCompressEpilogV2 operator registration.
 *
 * Input order mirrors the delivery-form kernel signature in op_kernel/
 * (cache, x, slot_mapping, then the cache output, then the tiling struct).
 *
 * cache is an in-place output: the bridge passes the same tensor as the input
 * and the output. The cache input is marked IgnoreContiguous because the kernel
 * addresses raw bytes using block_stride; an AutoContiguous copy would break
 * both the physical layout and the in-place update.
 *
 * slot_mapping is registered as INT64 only; the Python bridge converts INT32
 * callers before invoking the aclnn API.
 */

#include "register/op_def_registry.h"

namespace ops {
class KvCompressEpilogV2 : public OpDef {
public:
    explicit KvCompressEpilogV2(const char *name) : OpDef(name)
    {
        this->Input("cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("slot_mapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();

        this->Attr("quant_group_size").AttrType(OPTIONAL).Int(32);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(2);
        this->Attr("round_scale").AttrType(OPTIONAL).Bool(true);
        this->Attr("x_scale").AttrType(OPTIONAL).Float(1.0F);
        this->Attr("block_stride").AttrType(OPTIONAL).Int(0);
        // <=0: derive the geometry lower bound in tiling; >0: caller-declared
        // underlying storage bytes (used to validate the physical write range).
        this->Attr("storage_bytes").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "kv_compress_epilog_v2")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(KvCompressEpilogV2);
} // namespace ops
