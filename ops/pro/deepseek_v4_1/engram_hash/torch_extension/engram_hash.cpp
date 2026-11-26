/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gym_aclnn.h"
#include <torch/library.h>

namespace {
at::Tensor Output(const at::Tensor &ids, const at::Tensor &mult, const at::Tensor &vocab,
                  const at::Tensor &offs, const c10::optional<at::Tensor> &mask)
{
    TORCH_CHECK(ids.dim()==2 && ids.scalar_type()==at::kInt, "invalid ngram_token_ids");
    TORCH_CHECK(mult.dim()==2 && mult.scalar_type()==at::kLong && mult.size(1)==ids.size(1),
                "invalid multipliers");
    const auto t=ids.size(0), g=ids.size(1), l=mult.size(0);
    TORCH_CHECK(vocab.dim()==3 && vocab.scalar_type()==at::kInt && l>0 && g>=2 &&
                vocab.size(0)==l && vocab.size(1)==g-1 && vocab.size(2)>0, "invalid vocab_sizes");
    TORCH_CHECK(vocab.size(2)<=std::numeric_limits<int64_t>::max()/(g-1), "too many tables");
    const auto cols=(g-1)*vocab.size(2);
    TORCH_CHECK(offs.dim()==2 && offs.scalar_type()==at::kInt && offs.size(0)==l &&
                offs.size(1)==cols, "invalid offsets");
    TORCH_CHECK(mult.device()==ids.device() && vocab.device()==ids.device() &&
                offs.device()==ids.device(), "inputs must share one device");
    if (mask) {
        TORCH_CHECK(mask->dim()==1 && mask->size(0)==t && mask->device()==ids.device(),
                    "invalid image_token_mask shape or device");
        TORCH_CHECK(mask->scalar_type()==at::kBool || mask->scalar_type()==at::kByte,
                    "image_token_mask must be bool or uint8");
    }
    return at::empty({l,t,cols},ids.options());
}
at::Tensor Run(const at::Tensor &ids, const at::Tensor &mult, const at::Tensor &vocab,
               const at::Tensor &offs, const c10::optional<at::Tensor> &mask)
{
    const c10::OptionalDeviceGuard guard(ids.device());
    auto out=Output(ids,mult,vocab,offs,mask);
    gym_torch::CheckNDStorage(ids, "ngram_token_ids");
    gym_torch::CheckNDStorage(mult, "multipliers");
    gym_torch::CheckNDStorage(vocab, "vocab_sizes");
    gym_torch::CheckNDStorage(offs, "offsets");
    if (mask) gym_torch::CheckNDStorage(*mask, "image_token_mask");
    if (!out.numel()) return out;
    std::vector<at::Tensor> tensors{ids.contiguous(), mult.contiguous(), vocab.contiguous(), offs.contiguous()};
    if (mask) tensors.push_back(mask->to(at::kByte).contiguous());
    tensors.push_back(out);
    using Query=aclnnStatus (*)(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,
                                const aclTensor*,const aclTensor*,uint64_t*,aclOpExecutor**);
    static auto query=gym_torch::Symbol<Query>("aclnnEngramHashGetWorkspaceSize");
    static auto run=gym_torch::Symbol<gym_torch::Execute>("aclnnEngramHash");
    gym_torch::Enqueue("aclnnEngramHash",tensors,[hasMask=mask.has_value()](const auto &infos,auto stream) {
        std::vector<gym_torch::TensorHandle> h;
        for (const auto &info:infos) h.push_back(gym_torch::Tensor(info));
        uint64_t bytes=0;
        aclOpExecutor *executor=nullptr;
        auto status=query(h[0].get(),h[1].get(),h[2].get(),h[3].get(),hasMask?h[4].get():nullptr,
                          h.back().get(),&bytes,&executor);
        TORCH_CHECK(status==0, "aclnnEngramHashGetWorkspaceSize failed: ",status," ",aclGetRecentErrMsg());
        gym_torch::ExecuteOp(run,bytes,executor,stream);
        return 0;
    });
    return out;
}
}
TORCH_LIBRARY_FRAGMENT(pypto_gym,m) {
    m.def("engram_hash(Tensor ngram_token_ids, Tensor multipliers, Tensor vocab_sizes, Tensor offsets, Tensor? image_token_mask=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(pypto_gym,PrivateUse1,m) { m.impl("engram_hash", TORCH_FN(Run)); }
TORCH_LIBRARY_IMPL(pypto_gym,Meta,m) { m.impl("engram_hash", TORCH_FN(Output)); }
