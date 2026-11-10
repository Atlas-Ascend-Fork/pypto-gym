/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_bridge_common.h"
#include <array>
#include <cmath>
#include <limits>

namespace {
using namespace gym_test_bridge;
using GetWorkspace = aclnnStatus (*)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    double, double, double, double, int64_t, double, int64_t,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
using Execute = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
std::vector<torch::Tensor> MegaMhc(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor previous_mix,
    torch::Tensor post_mix,
    torch::Tensor comb_res_mix,
    torch::Tensor fn,
    torch::Tensor mix_scales,
    torch::Tensor mix_bases,
    torch::Tensor rmsnorm_weight,
    double hc, double pre, double post, double sink, int64_t repeat, double norm,
    c10::optional<int64_t> batch, c10::optional<std::vector<torch::Tensor>> out)
{
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.device().is_privateuseone(),
                "x must be a rank-2 BF16 NPU tensor");
    const int64_t t = x.size(0), h = x.size(1);
    TORCH_CHECK(h >= 1024 && h <= 7168 && h % 1024 == 0, "unsupported H");
    TORCH_CHECK(!batch.has_value() || *batch == 16, "public batch must be None or 16");
    for (double v : {hc, pre, post, sink, norm})
        TORCH_CHECK(std::isfinite(v) && v >= 0 && v <= std::numeric_limits<float>::max(), "invalid scalar");
    TORCH_CHECK(hc > 0 && norm > 0 && repeat >= 1 && repeat <= 65535, "invalid RMS epsilon/iterations");
    std::vector<torch::Tensor> inputs = {x, residual, previous_mix, post_mix, comb_res_mix, fn, mix_scales, mix_bases, rmsnorm_weight};
    const std::vector<std::vector<int64_t>> shapes = {
        {t,h},{t,4,h},{t,4},{t,4,1},{t,4,4},{24,4*h},{3},{24},{h}};
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto dtype = (i == 0 || i == 1 || i == 8) ? at::kBFloat16 : at::kFloat;
        TORCH_CHECK(inputs[i].device() == x.device() && inputs[i].is_contiguous() &&
                    inputs[i].scalar_type() == dtype && inputs[i].sizes().vec() == shapes[i], "invalid input ", i);
    }
    const std::vector<std::vector<int64_t>> outShapes = {{t,4,h},{t,4,1},{t,4,4},{t,h},{t,4}};
    std::vector<torch::Tensor> outputs;
    if (out.has_value()) outputs = *out;
    else for (size_t i = 0; i < 5; ++i)
        outputs.push_back(at::empty(outShapes[i], x.options().dtype((i == 0 || i == 3) ? at::kBFloat16 : at::kFloat)));
    TORCH_CHECK(outputs.size() == 5, "out must contain five tensors");
    std::vector<std::pair<uintptr_t, uintptr_t>> occupied;
    for (const auto &value : inputs) {
        const auto start = reinterpret_cast<uintptr_t>(value.data_ptr());
        occupied.emplace_back(start, start + value.numel() * value.element_size());
    }
    for (size_t i = 0; i < 5; ++i) {
        const auto &value = outputs[i];
        TORCH_CHECK(value.device() == x.device() && value.is_contiguous() && value.sizes().vec() == outShapes[i] &&
                    value.scalar_type() == ((i == 0 || i == 3) ? at::kBFloat16 : at::kFloat), "invalid output ", i);
        const auto a = reinterpret_cast<uintptr_t>(value.data_ptr()), b = a + value.numel() * value.element_size();
        if (a != b) {
            for (const auto &span : occupied)
                TORCH_CHECK(span.first == span.second || a >= span.second || span.first >= b, "overlapping output");
            occupied.emplace_back(a, b);
        }
    }
    if (t == 0) return outputs;
    static auto getWorkspace = reinterpret_cast<GetWorkspace>(get_opapi_func("aclnnMegaMhcGetWorkspaceSize"));
    static auto execute = reinterpret_cast<Execute>(get_opapi_func("aclnnMegaMhc"));
    // Destroy descriptors even when GetWorkspaceSize or launch fails.
    struct Descriptors {
        std::vector<aclTensor *> tensors;
        ~Descriptors() { for (auto *p : tensors) destroy_acl_tensor(p); }
    } descriptors;
    for (const auto &value : inputs) descriptors.tensors.push_back(make_acl_tensor(value));
    for (const auto &value : outputs) descriptors.tensors.push_back(make_acl_tensor(value));
    auto &a = descriptors.tensors;
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    auto status = getWorkspace(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
        hc, pre, post, sink, repeat, norm, batch.value_or(0),
        a[9], a[10], a[11], a[12], a[13], &bytes, &executor);
    TORCH_CHECK(status == 0, "aclnnMegaMhcGetWorkspaceSize failed: ", status);
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, x.options().dtype(at::kByte));
    status = execute(bytes ? workspace.data_ptr() : nullptr, bytes, executor, get_current_npu_stream());
    TORCH_CHECK(status == 0, "aclnnMegaMhc failed: ", status);
    return outputs;
}
} // namespace

PYBIND11_MODULE(aclnn_bridge_mega_mhc, m)
{
    m.def("mega_mhc", &MegaMhc,
        pybind11::arg("x"),
        pybind11::arg("residual"),
        pybind11::arg("previous_mix"),
        pybind11::arg("post_mix"),
        pybind11::arg("comb_res_mix"),
        pybind11::arg("fn"),
        pybind11::arg("mix_scales"),
        pybind11::arg("mix_bases"),
        pybind11::arg("rmsnorm_weight"),
        pybind11::kw_only(),
        pybind11::arg("hc_norm_eps") = 2e-5, pybind11::arg("hc_pre_eps") = 3e-4,
        pybind11::arg("hc_post_scale") = 1.25, pybind11::arg("sinkhorn_eps") = 2e-6,
        pybind11::arg("num_sinkhorn_iters") = 10, pybind11::arg("rmsnorm_eps") = 7e-6,
        pybind11::arg("batch") = pybind11::none(), pybind11::arg("out") = pybind11::none());
}
