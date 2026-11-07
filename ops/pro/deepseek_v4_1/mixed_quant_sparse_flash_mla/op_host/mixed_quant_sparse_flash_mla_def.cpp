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
 * \file mixed_quant_sparse_flash_mla_def.cpp
 * \brief MixedQuantSparseFlashMla operator registration.
 *
 * Input order mirrors the delivery-form kernel signature in op_kernel/
 * (q, ori_kv, ori_sparse_indices, ori_block_table, ori_topk_length, sinks,
 * cmp_kv, cmp_sparse_indices, cmp_block_table, cmp_topk_length, cu_seqlens_q,
 * outputs out/lse, then the tiling struct).
 *
 * The host tiling generates the metadata schedule (uniform whole-row or flat
 * FlashDecode plan) from the per-row TopK lengths and the batch boundaries,
 * so those three inputs are registered ValueDepend(OPTIONAL): when the
 * framework supplies their Host data the tiling plans exactly; otherwise it
 * falls back to a conservative schedule (lengths absent -> uniform, never FD;
 * cu_seqlens_q host data absent is rejected when batch > 1).
 *
 * softmax_scale is a runtime scalar attribute (0 means the default 1/sqrt(d)).
 */

#include "register/op_def_registry.h"

namespace ops {
class MixedQuantSparseFlashMla : public OpDef {
public:
    explicit MixedQuantSparseFlashMla(const char *name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("ori_kv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("ori_sparse_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("ori_block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("ori_topk_length")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous()
            .ValueDepend(OPTIONAL);
        this->Input("sinks")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cmp_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cmp_sparse_indices")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cmp_block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cmp_topk_length")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous()
            .ValueDepend(OPTIONAL);
        this->Input("cu_seqlens_q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous()
            .ValueDepend(OPTIONAL);
        this->Input("metadata")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("lse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        // Host-side contract (mirrors the Python host wrapper arguments):
        // fixed layout strings and the runtime softmax scale.
        this->Attr("layout_q").AttrType(OPTIONAL).String("TND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("PA_BBND");
        this->Attr("softmax_scale").AttrType(OPTIONAL).Float(0.0F);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "mixed_quant_sparse_flash_mla")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(MixedQuantSparseFlashMla);
} // namespace ops