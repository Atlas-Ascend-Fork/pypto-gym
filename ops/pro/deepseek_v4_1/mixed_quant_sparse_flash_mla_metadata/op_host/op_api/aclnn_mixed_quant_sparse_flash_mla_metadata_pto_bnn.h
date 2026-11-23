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

#ifndef OP_API_INC_ACLNN_MIXED_QUANT_SPARSE_FLASH_MLA_METADATA_PTO_BNN_H_
#define OP_API_INC_ACLNN_MIXED_QUANT_SPARSE_FLASH_MLA_METADATA_PTO_BNN_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnMixedQuantSparseFlashMlaMetadataPtoBnnGetWorkspaceSize(
    const aclTensor *cuSeqlensQ, const aclTensor *oriTopkLength, const aclTensor *cmpTopkLengthOptional,
    int64_t blocks, int64_t hasCmp, int64_t kOri, int64_t kCmp, int64_t bsOri, int64_t bsCmp,
    int64_t btOri, int64_t btCmp, int64_t oriPages, int64_t cmpPages, int64_t oriKvStride0,
    int64_t cmpKvStride0, const aclTensor *out, const aclTensor *tilingDevice,
    uint64_t *workspaceSize, aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnMixedQuantSparseFlashMlaMetadataPtoBnn(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif // OP_API_INC_ACLNN_MIXED_QUANT_SPARSE_FLASH_MLA_METADATA_PTO_BNN_H_
