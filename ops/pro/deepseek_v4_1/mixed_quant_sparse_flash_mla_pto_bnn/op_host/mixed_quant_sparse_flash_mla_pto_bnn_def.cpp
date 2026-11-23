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
 * \file mixed_quant_sparse_flash_mla_pto_bnn_def.cpp
 * \brief MixedQuantSparseFlashMlaPtoBnn operator registration.
 *
 * Input order mirrors the delivery-form kernel signature in op_kernel/
 * (q, ori_kv, ori_sparse_indices, ori_block_table, ori_topk_length, sinks,
 * cmp_kv, cmp_sparse_indices, cmp_block_table, cmp_topk_length, cu_seqlens_q,
 * metadata, tiling_device, outputs out/lse, then the tiling struct).
 *
 * 全设备端形态: 值依赖调度全部在设备侧 —— AICPU metadata 算子板上产出
 * int32[1024] 计划与 int32[16] tiling_device(形状标量 + 规划结果), 本算子
 * kernel 执行期从 tiling_device/metadata 张量读取全部 tiling 参数与调度;
 * host tiling 值无关(全核 blockDim + FD workspace 超集)。ori_topk_length /
 * cmp_topk_length / cu_seqlens_q 仍注册 ValueDepend(OPTIONAL) 仅为保留
 * aclnn 的 aclIntArray 兼容变体(aclnn 测试桥使用)。
 *
 * softmax_scale is a runtime scalar attribute (0 means the default 1/sqrt(d)).
 *
 * ori_kv / cmp_kv are IgnoreContiguous: the kernel consumes the page stride
 * from tiling (axis-0 padding allowed), so the framework must not compact them.
 */

#include "register/op_def_registry.h"

namespace ops {
class MixedQuantSparseFlashMlaPtoBnn : public OpDef {
public:
    explicit MixedQuantSparseFlashMlaPtoBnn(const char *name) : OpDef(name)
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
            .IgnoreContiguous();
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
            .IgnoreContiguous();
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
        this->Input("tiling_device")
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
            .ExtendCfgInfo("opFile.value", "mixed_quant_sparse_flash_mla_pto_bnn")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(MixedQuantSparseFlashMlaPtoBnn);
} // namespace ops