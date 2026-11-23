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
 * \file aclnn_mixed_quant_sparse_flash_mla_metadata_pto_bnn.cpp
 * \brief MQSMLA metadata planner aclnn entry (single AICPU task, no tiling).
 */

#include "aclnn_mixed_quant_sparse_flash_mla_metadata_pto_bnn.h"

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
OP_TYPE_REGISTER(MixedQuantSparseFlashMlaMetadataPtoBnn);

const aclTensor *MixedQuantSparseFlashMlaMetadataPtoBnn(
    const aclTensor *cuSeqlensQ, const aclTensor *oriTopkLength, const aclTensor *cmpTopkLength,
    int64_t blocks, int64_t hasCmp, int64_t kOri, int64_t kCmp, int64_t bsOri, int64_t bsCmp,
    int64_t btOri, int64_t btCmp, int64_t oriPages, int64_t cmpPages, int64_t oriKvStride0,
    int64_t cmpKvStride0, aclTensor *out, aclTensor *tilingDevice, aclOpExecutor *executor)
{
    L0_DFX(MixedQuantSparseFlashMlaMetadataPtoBnn, cuSeqlensQ, oriTopkLength, cmpTopkLength,
           blocks, hasCmp, kOri, kCmp, out, tilingDevice);
    static internal::AicpuTaskSpace space("MixedQuantSparseFlashMlaMetadataPtoBnn");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        MixedQuantSparseFlashMlaMetadataPtoBnn,
        OP_ATTR_NAMES({"blocks", "has_cmp", "k_ori", "k_cmp", "bs_ori", "bs_cmp", "bt_ori",
                       "bt_cmp", "ori_pages", "cmp_pages", "ori_kv_stride0", "cmp_kv_stride0"}),
        OP_INPUT(cuSeqlensQ, oriTopkLength, cmpTopkLength),
        OP_OUTPUT(out, tilingDevice),
        OP_ATTR(blocks, hasCmp, kOri, kCmp, bsOri, bsCmp, btOri, btCmp, oriPages, cmpPages,
                oriKvStride0, cmpKvStride0));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "MixedQuantSparseFlashMlaMetadataPtoBnn ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return out;
}
} // namespace l0op

static aclnnStatus CheckParams(const aclTensor *cuSeqlensQ, const aclTensor *oriTopkLength,
                               const aclTensor *out, const aclTensor *tilingDevice)
{
    OP_CHECK_NULL(cuSeqlensQ, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(oriTopkLength, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(tilingDevice, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(out->GetViewShape().GetDimNum() == 1 && out->GetViewShape().GetDim(0) == 1024,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "metadata out must be int32[1024], got %s.",
                     op::ToString(out->GetViewShape()).GetString()),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(out->GetDataType() == op::DataType::DT_INT32,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "metadata out must be int32."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(tilingDevice->GetViewShape().GetDimNum() == 1 && tilingDevice->GetViewShape().GetDim(0) == 16,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tilingDevice out must be int32[16], got %s.",
                     op::ToString(tilingDevice->GetViewShape()).GetString()),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(tilingDevice->GetDataType() == op::DataType::DT_INT32,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tilingDevice out must be int32."),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMixedQuantSparseFlashMlaMetadataPtoBnnGetWorkspaceSize(
    const aclTensor *cuSeqlensQ, const aclTensor *oriTopkLength, const aclTensor *cmpTopkLengthOptional,
    int64_t blocks, int64_t hasCmp, int64_t kOri, int64_t kCmp, int64_t bsOri, int64_t bsCmp,
    int64_t btOri, int64_t btCmp, int64_t oriPages, int64_t cmpPages, int64_t oriKvStride0,
    int64_t cmpKvStride0, const aclTensor *out, const aclTensor *tilingDevice,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMixedQuantSparseFlashMlaMetadataPtoBnn,
                   DFX_IN(cuSeqlensQ, oriTopkLength, cmpTopkLengthOptional, blocks, hasCmp, kOri, kCmp),
                   DFX_OUT(out, tilingDevice));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);

    auto ret = CheckParams(cuSeqlensQ, oriTopkLength, out, tilingDevice);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto md = l0op::MixedQuantSparseFlashMlaMetadataPtoBnn(
        cuSeqlensQ, oriTopkLength, cmpTopkLengthOptional, blocks, hasCmp, kOri, kCmp, bsOri, bsCmp,
        btOri, btCmp, oriPages, cmpPages, oriKvStride0, cmpKvStride0,
        const_cast<aclTensor *>(out), const_cast<aclTensor *>(tilingDevice), uniqueExecutor.get());
    CHECK_RET(md != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMixedQuantSparseFlashMlaMetadataPtoBnn(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMixedQuantSparseFlashMlaMetadataPtoBnn);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
