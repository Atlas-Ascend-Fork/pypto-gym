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
#include "quant_lightning_indexer_config.h"
#include "quant_lightning_indexer_metadata_native.h"
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <array>
#include <tuple>
namespace {
using Result=std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
static bool Has(const c10::optional<at::Tensor> &t) { return t.has_value() && t->defined(); }
static void CheckLinearStorage(const at::Tensor &t,const char *name) {
    const auto format=at_npu::native::get_npu_format(t);
    TORCH_CHECK(format==ACL_FORMAT_ND || format==ACL_FORMAT_NCHW || format==ACL_FORMAT_NCDHW,
                name," requires linear ND/NCHW/NCDHW storage, got ",int(format));
}
static qli::TensorSpec Spec(const at::Tensor &t) {
    qli::TensorSpec s;
    if (!t.defined()) return s;
    s.present=true; s.dims=t.sizes().vec(); s.strides=t.strides().vec();
    s.dtype=t.scalar_type()==at::kByte?qli::U8:t.scalar_type()==at::kFloat?qli::F32:
        t.scalar_type()==at::kInt?qli::I32:qli::Invalid;
    return s;
}
static Result Finish(at::Tensor idx,at::Tensor val,at::Tensor cand,at::Tensor len,
                     bool values,int64_t capacity) {
    if (!values) val=val.reshape({-1}).slice(0,0,0);
    if (capacity<0) { cand=cand.reshape({-1}).slice(0,0,0); len=len.reshape({-1}).slice(0,0,0); }
    return {idx,val,cand,len};
}
using NativePlan=decltype(&quant_lightning_indexer_metadataRun);
static NativePlan LoadNativePlanner() {
    const std::string basename="libquant_lightning_indexer_metadata.so";
    std::vector<std::string> names;
    if (const char *vendors=std::getenv("ASCEND_CUSTOM_OPP_PATH")) {
        std::istringstream paths(vendors);
        for (std::string vendor;std::getline(paths,vendor,':');) {
            if (!vendor.empty()) names.push_back(vendor+"/op_api/lib/"+basename);
        }
    }
    names.push_back(basename);
    for (const auto &name:names) {
        if (void *library=dlopen(name.c_str(),RTLD_NOW|RTLD_LOCAL)) {
            if (void *symbol=dlsym(library,"quant_lightning_indexer_metadataRun")) {
                // Keep the handle resident while kernels may be in flight.
                return reinterpret_cast<NativePlan>(symbol);
            }
            dlclose(library);
        }
    }
    TORCH_CHECK(false,"Cannot load offline QLI AICPU library; install the matching .run package");
}

static void LaunchPlanner(const std::array<at::Tensor, 12> &inputs, const at::Tensor &dummy,
                          const at::Tensor &plan, const qli::Config &c) {
    static const NativePlan native_plan = LoadNativePlanner();
    const auto metadata_stream=c10_npu::getCurrentNPUStream(plan.device().index());
    // Empty the host task queue before a direct native launch, matching JIT.
    const auto metadata_raw_stream=metadata_stream.stream(true);
    const auto optional = [&](size_t index) -> const at::Tensor & {
        return inputs[index].defined() ? inputs[index] : dummy;
    };
    const std::array<at::Tensor, 6> metadata_inputs = {
        optional(5), optional(6), optional(7), optional(8), optional(9), plan};
    std::vector<gym_torch::TensorInfo> metadata_info;
    for (const auto &t:metadata_inputs) {
        c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),metadata_stream);
        metadata_info.emplace_back(t);
    }
    const auto ptr=[](const gym_torch::TensorInfo &info) {return reinterpret_cast<int32_t*>(info.data)+info.offset;};
    const auto &mi=metadata_info;
    auto metadata_status=native_plan(ptr(mi[0]),ptr(mi[1]),ptr(mi[2]),ptr(mi[3]),ptr(mi[4]),ptr(mi[5]),
        c.rows,c.max_q,c.max_k,c.schedule,mi[0].sizes[0],mi[1].sizes[0],mi[2].sizes[0],
        mi[3].sizes[0],mi[4].sizes[0],mi[5].sizes[0],metadata_raw_stream);
    TORCH_CHECK(metadata_status==0,"native metadata launch failed: ",metadata_status);
}

static void LaunchCompute(const std::array<at::Tensor, 12> &inputs,
                          const std::array<at::Tensor, 5> &outputs, const qli::Attrs &attrs) {
    std::array<bool,17> present{};
    std::vector<at::Tensor> defined;
    for (size_t i=0;i<inputs.size();++i) {
        present[i]=inputs[i].defined(); if (present[i]) defined.push_back(inputs[i]);
    }
    for (const auto &t:outputs) defined.push_back(t);
    for (size_t i=12;i<present.size();++i) present[i]=true;
    using ComputeWorkspace=aclnnStatus (*)(
        const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,
        int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,char*,char*,bool,
        const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,uint64_t*,aclOpExecutor**);
    static auto compute_ws=gym_torch::Symbol<ComputeWorkspace>("aclnnGymQuantLightningIndexerGetWorkspaceSize");
    static auto compute_run=gym_torch::Symbol<gym_torch::Execute>("aclnnGymQuantLightningIndexer");
    gym_torch::Enqueue("aclnnGymQuantLightningIndexer",defined,
        [present,attrs](const std::vector<gym_torch::TensorInfo> &infos,auto stream) {
            std::vector<gym_torch::TensorHandle> h; h.reserve(infos.size());
            for (const auto &info:infos) h.push_back(gym_torch::Tensor(info));
            std::array<const aclTensor*,17> s{}; size_t cursor=0;
            for (size_t i=0;i<s.size();++i) if (present[i]) s[i]=h[cursor++].get();
            uint64_t bytes=0; aclOpExecutor *executor=nullptr;
            auto status=compute_ws(s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8],s[9],s[10],s[11],s[12],
                attrs.topk,attrs.quant_mode,attrs.max_q,attrs.mask,attrs.ratio,attrs.candidates,attrs.block,
                const_cast<char*>(attrs.layout_q.c_str()),const_cast<char*>(attrs.layout_k.c_str()),attrs.values,
                s[13],s[14],s[15],s[16],&bytes,&executor);
            TORCH_CHECK(status==0,"QLI compute GetWorkspaceSize failed: ",status," ",aclGetRecentErrMsg());
            gym_torch::ExecuteOp(compute_run,bytes,executor,stream); return 0;
        });
}

static Result Run(const at::Tensor &q,const at::Tensor &k,const at::Tensor &w,
    const at::Tensor &qs,const at::Tensor &ks,
    const c10::optional<at::Tensor> &cq,const c10::optional<at::Tensor> &ck,
    const c10::optional<at::Tensor> &uq,const c10::optional<at::Tensor> &uk,
    const c10::optional<at::Tensor> &res,const c10::optional<at::Tensor> &table,
    const c10::optional<at::Tensor> &off,const c10::optional<at::Tensor> &legacy,
    int64_t topk,int64_t quant_mode,int64_t max_q,int64_t mask,int64_t ratio,
    int64_t candidates,int64_t block,std::string layout_q,std::string layout_k,
    bool values,int64_t cube_capacity,int64_t vector_capacity) {
    TORCH_CHECK(q.device().is_privateuseone(),"q must live on an NPU");
    const c10::OptionalDeviceGuard guard(q.device());
    std::array<at::Tensor,12> inputs={q,k,w,qs,ks,
        Has(cq)?*cq:at::Tensor(),Has(ck)?*ck:at::Tensor(),Has(uq)?*uq:at::Tensor(),
        Has(uk)?*uk:at::Tensor(),Has(res)?*res:at::Tensor(),
        Has(table)?*table:at::Tensor(),Has(off)?*off:at::Tensor()};
    std::array<qli::TensorSpec,12> specs;
    for (size_t i=0;i<inputs.size();++i) {
        if (inputs[i].defined()) {
            TORCH_CHECK(inputs[i].device()==q.device(),"all inputs must share the q device");
            CheckLinearStorage(inputs[i],"QLI input");
        }
        specs[i]=Spec(inputs[i]);
    }
    if (Has(legacy)) {
        TORCH_CHECK(legacy->device()==q.device() && legacy->scalar_type()==at::kInt &&
            legacy->is_contiguous() && legacy->sizes()==at::IntArrayRef({1024}),"metadata must be contiguous int32[1024] on q device");
        CheckLinearStorage(*legacy,"metadata");
    }
    qli::Attrs attrs{topk,quant_mode,max_q,mask,ratio,candidates,block,layout_q,layout_k,values};
    qli::Config c; std::string error;
    TORCH_CHECK(qli::Derive(specs,attrs,cube_capacity,vector_capacity,c,error),error);
    auto dummy=at::empty({1},q.options().dtype(at::kInt));
    auto plan=at::empty({c.plan_elems},dummy.options());
    auto idx=at::empty({c.rows,1,topk},dummy.options());
    auto val=at::empty({c.rows,1,topk},q.options().dtype(at::kBFloat16));
    auto cand=at::empty({c.rows,1,std::max(int64_t(1),candidates)},dummy.options());
    auto len=at::empty({c.rows,1},dummy.options());
    LaunchPlanner(inputs, dummy, plan, c);
    LaunchCompute(inputs, {plan, idx, val, cand, len}, attrs);
    return Finish(idx,val,cand,len,values,candidates);
}
static Result Meta(const at::Tensor &q,const at::Tensor &k,const at::Tensor &w,
    const at::Tensor &qs,const at::Tensor &ks,
    const c10::optional<at::Tensor> &cq,const c10::optional<at::Tensor> &ck,
    const c10::optional<at::Tensor> &uq,const c10::optional<at::Tensor> &uk,
    const c10::optional<at::Tensor> &res,const c10::optional<at::Tensor> &table,
    const c10::optional<at::Tensor> &off,const c10::optional<at::Tensor> &legacy,
    int64_t topk,int64_t quant_mode,int64_t max_q,int64_t mask,int64_t ratio,
    int64_t candidates,int64_t block,std::string layout_q,std::string layout_k,
    bool values,int64_t cube_capacity,int64_t vector_capacity) {
    TORCH_CHECK(q.dim()==3 && q.size(1)==32 && q.size(2)==64 && topk>=1 && topk<=8192,"invalid q/topk");
    auto rows=q.sym_size(0);
    auto idx=at::empty_symint({rows,1,topk},q.options().dtype(at::kInt));
    auto val=at::empty_symint({rows,1,topk},q.options().dtype(at::kBFloat16));
    auto cand=at::empty_symint({rows,1,std::max(int64_t(1),candidates)},q.options().dtype(at::kInt));
    auto len=at::empty_symint({rows,1},q.options().dtype(at::kInt));
    return Finish(idx,val,cand,len,values,candidates);
}
}
TORCH_LIBRARY_FRAGMENT(pypto_gym,m) {
    m.def("quant_lightning_indexer(Tensor q, Tensor k, Tensor w, Tensor descale_q, Tensor descale_k, "
          "Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? seqused_q, Tensor? seqused_k, "
          "Tensor? cmp_residual_k, Tensor? block_table, Tensor? output_idx_offset, Tensor? metadata, "
          "int topk, int quant_mode, int max_seqlen_q, int mask_mode, int cmp_ratio, "
          "int candidate_topk_blocks, int candidate_block_size, str layout_q, str layout_k, "
          "bool return_value, int cube_capacity, int vector_capacity) -> (Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(pypto_gym,PrivateUse1,m) { m.impl("quant_lightning_indexer",TORCH_FN(Run)); }
TORCH_LIBRARY_IMPL(pypto_gym,Meta,m) { m.impl("quant_lightning_indexer",TORCH_FN(Meta)); }
