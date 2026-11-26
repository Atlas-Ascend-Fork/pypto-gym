/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
namespace ops {
class GymQuantLightningIndexer : public OpDef {
public:
    explicit GymQuantLightningIndexer(const char *name) : OpDef(name) {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("w").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("descale_q").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
        this->Input("descale_k").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("cu_seqlens_q").ParamType(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("cu_seqlens_k").ParamType(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("seqused_q").ParamType(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("seqused_k").ParamType(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("cmp_residual_k").ParamType(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("block_table").ParamType(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("output_idx_offset").ParamType(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("metadata").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("values").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Output("candidates").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("lengths").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Attr("topk").AttrType(REQUIRED).Int();
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(1);
        this->Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1);
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("cmp_ratio").AttrType(OPTIONAL).Int(1);
        this->Attr("candidate_topk_blocks").AttrType(OPTIONAL).Int(-1);
        this->Attr("candidate_block_size").AttrType(OPTIONAL).Int(-1);
        this->Attr("layout_q").AttrType(OPTIONAL).String("TND");
        this->Attr("layout_k").AttrType(OPTIONAL).String("TND");
        this->Attr("return_value").AttrType(OPTIONAL).Bool(false);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "quant_lightning_indexer")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950",config);
    }
};
OP_ADD(GymQuantLightningIndexer);

}
