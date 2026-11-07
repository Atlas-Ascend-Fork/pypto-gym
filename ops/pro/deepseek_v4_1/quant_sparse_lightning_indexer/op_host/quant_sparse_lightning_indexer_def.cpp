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
 * \file quant_sparse_lightning_indexer_def.cpp
 * \brief QuantSparseLightningIndexer operator registration (MXFP4 sparse indexer).
 *
 * Input order mirrors the kernel entry signature in op_kernel/
 * (inputs first, then the two outputs indices/values, then workspace + tiling).
 *
 * Packed MXFP4 storage: q/k/descale_* are registered as UINT8 because one byte
 * carries two FP4 E2M1 elements (low nibble first) and the E8M0 scales are raw
 * exponent bytes; the kernel reinterprets them as DT_FP4E2M1 / DT_FP8E8M0 views.
 * Registering the byte type keeps the aclnn interface free of sub-byte shape
 * ambiguity while the on-device interpretation stays identical to the jit form.
 *
 * k and descale_k deliberately carry NO AutoContiguous(): the TND path supports
 * axis-0 padding, and the resulting strides select the TndStrideMode tiling key
 * (host tiling reads them via GetInputStride). Forcing a contiguous copy would
 * silently collapse that specialization.
 *
 * metadata is the caller-generated LI/LD schedule table (int32[1024]); the op
 * neither generates nor validates its contents, exactly like the jit host.
 */

#include "register/op_def_registry.h"

namespace ops {
class QuantSparseLightningIndexer : public OpDef {
public:
    explicit QuantSparseLightningIndexer(const char *name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND});
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("descale_q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("descale_k")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND});
        this->Input("candidate_block_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("candidate_block_length")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens_q")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens_k")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused_q")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused_k")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cmp_residual_k")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("output_idx_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("metadata")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Output("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});

        // Attributes follow the public jit signature so the aclnn interface stays
        // a drop-in replacement. topk is the only one without a usable default.
        this->Attr("topk").AttrType(REQUIRED).Int();
        this->Attr("candidate_block_size").AttrType(OPTIONAL).Int(8);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(1);
        this->Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1);
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("cmp_ratio").AttrType(OPTIONAL).Int(1);
        this->Attr("layout_q").AttrType(OPTIONAL).String("TND");
        this->Attr("layout_k").AttrType(OPTIONAL).String("TND");
        this->Attr("return_value").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "quant_sparse_lightning_indexer")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(QuantSparseLightningIndexer);
} // namespace ops
