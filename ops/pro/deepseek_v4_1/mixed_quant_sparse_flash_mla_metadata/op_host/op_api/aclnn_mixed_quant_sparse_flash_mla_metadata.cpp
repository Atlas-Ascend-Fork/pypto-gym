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
 * \file aclnn_mixed_quant_sparse_flash_mla_metadata.cpp
 * \brief MQSMLA metadata planner aclnn entry (single AICPU task, no tiling).
 */

#include "aclnn_mixed_quant_sparse_flash_mla_metadata.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MixedQuantSparseFlashMlaMetadata);

const aclTensor *MixedQuantSparseFlashMlaMetadata(
    const aclTensor *cuSeqlensQ, const aclTensor *oriTopkLength, const aclTensor *cmpTopkLength,
    int64_t blocks, int64_t hasCmp, int64_t kOri, int64_t kCmp,
    aclTensor *out, aclOpExecutor *executor)
{
    L0_DFX(MixedQuantSparseFlashMlaMetadata, cuSeqlensQ, oriTopkLength, cmpTopkLength,
           blocks, hasCmp, kOri, kCmp, out);
    static internal::AicpuTaskSpace space("MixedQuantSparseFlashMlaMetadata");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        MixedQuantSparseFlashMlaMetadata,
        OP_ATTR_NAMES({"blocks", "has_cmp", "k_ori", "k_cmp"}),
        OP_INPUT(cuSeqlensQ, oriTopkLength, cmpTopkLength),
        OP_OUTPUT(out),
        OP_ATTR(blocks, hasCmp, kOri, kCmp));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "MixedQuantSparseFlashMlaMetadata ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return out;
}
} // namespace l0op

static aclnnStatus CheckParams(const aclTensor *cuSeqlensQ, const aclTensor *oriTopkLength,
                               const aclTensor *out)
{
    OP_CHECK_NULL(cuSeqlensQ, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(oriTopkLength, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(out->GetViewShape().GetDimNum() == 1 && out->GetViewShape().GetDim(0) == 1024,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "metadata out must be int32[1024], got %s.",
                     op::ToString(out->GetViewShape()).GetString()),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(out->GetDataType() == op::DataType::DT_INT32,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "metadata out must be int32."),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQ, const aclTensor *oriTopkLength, const aclTensor *cmpTopkLengthOptional,
    int64_t blocks, int64_t hasCmp, int64_t kOri, int64_t kCmp, const aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMixedQuantSparseFlashMlaMetadata,
                   DFX_IN(cuSeqlensQ, oriTopkLength, cmpTopkLengthOptional, blocks, hasCmp, kOri, kCmp),
                   DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);

    auto ret = CheckParams(cuSeqlensQ, oriTopkLength, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto md = l0op::MixedQuantSparseFlashMlaMetadata(
        cuSeqlensQ, oriTopkLength, cmpTopkLengthOptional, blocks, hasCmp, kOri, kCmp,
        const_cast<aclTensor *>(out), uniqueExecutor.get());
    CHECK_RET(md != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMixedQuantSparseFlashMlaMetadata(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMixedQuantSparseFlashMlaMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
