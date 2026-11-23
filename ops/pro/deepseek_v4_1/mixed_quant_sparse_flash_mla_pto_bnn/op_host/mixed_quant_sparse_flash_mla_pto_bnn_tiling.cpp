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
 * \file mixed_quant_sparse_flash_mla_pto_bnn_tiling.cpp
 * \brief MixedQuantSparseFlashMlaPtoBnn host tiling (value-free, superset).
 *
 * 全设备端形态: tiling 只消费 shape/dtype/属性, 不读任何输入张量的值 ——
 * 值依赖调度全部在设备侧(AICPU metadata 算子板上产出 int32[1024] 计划 +
 * int32[16] tiling_device, kernel 执行期从这两个 GM 张量读取 tiling 参数
 * 并逐槽自分流):
 *   - blockDim = min(cube cores, AIC_CORE_MAX_NUM): FA/FD 槽位上限 36, 计划
 *     内启用核数恒不超过它; 未启用核读到全零 FA 槽(row_count=0)空转, FD 变体
 *     的全核 barrier 由所有已发射核共同到达, 语义安全;
 *   - workspace 恒为 FD 超集(WORKSPACE_BYTES_FD): FD 变体常驻可用, uniform
 *     计划不触碰;
 *   - tilingkey 的 FD_ENABLE 位常置 1: 选择携带 FD 归约段的变体, 运行期按
 *     metadata FD 槽(fen)是否非零自分流, uniform 计划零开销跳过。
 *
 * tiling struct 仍按 shape 填充(保持布局校验有效), kernel 运行期仅从其中读
 * softmax_scale; 其余 tiling 参数由 kernel 从 tiling_device 张量读取。本文件
 * 不消费任何 host 值, 绑定走 Tensor 变体, 长度张量全程留在设备上(无 D2H)。
 */

#include "mixed_quant_sparse_flash_mla_pto_bnn_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#ifndef OP_LOGE
#include <cstdio>
#define OP_LOGE(nodeName, fmt, ...) \
  do { \
    (void)(nodeName); \
    std::printf(fmt "\n", ##__VA_ARGS__); \
  } while (0)
#endif

namespace optiling {

constexpr int64_t AIC_CORE_MAX_NUM = 36;
constexpr int64_t MQSMLA_METADATA_TOTAL_SIZE = 1024;
constexpr int64_t TILING_DEVICE_SIZE = 16;
// FD split partials workspace (72 splits x 64 heads x 512 fp32 + m/l planes);
// 恒定超集分配, uniform 计划不触碰(SetSysWorkspaceForce 路由要求非零)。
constexpr int64_t WORKSPACE_BYTES_FD = (72 * 64 * 512 + 2 * 72 * 64) * 4;

constexpr int64_t D = 512;
constexpr int64_t N1 = 64;
constexpr int64_t ROW_BYTES_ORI = 544;
constexpr int64_t ROW_BYTES_CMP = 320;

enum InputIndex {
    QUERY = 0,
    ORI_KV,
    ORI_SPARSE_INDICES,
    ORI_BLOCK_TABLE,
    ORI_TOP_K_LEN,
    SINKS,
    CMP_KV,
    CMP_SPARSE_INDICES,
    CMP_BLOCK_TABLE,
    CMP_TOP_K_LEN,
    CU_SEQLENS_Q,
    METADATA,
    TILING_DEVICE,
};

enum OutputIndex {
    OUT = 0,
    LSE,
};

class TilingMixedQuantSparseFlashMlaPtoBnnFunc {
public:
    explicit TilingMixedQuantSparseFlashMlaPtoBnnFunc(gert::TilingContext *tilingContext)
        : tilingContext_(tilingContext) {}
    ge::graphStatus RunTiling();

private:
    ge::graphStatus ParseShapesAndAttrs();
    ge::graphStatus ValidateTensors();
    ge::graphStatus ComputeTiling();

    static bool CheckShape(const gert::StorageShape *shape, std::initializer_list<int64_t> dims);
    static ge::DataType GetInputDtype(gert::TilingContext *context, size_t index);
    static ge::DataType GetOutputDtype(gert::TilingContext *context, size_t index);
    ge::graphStatus ResolveKvPageStride(size_t index, const char *name, int64_t bs, int64_t rowBytes,
                                       int64_t *stride0);

    gert::TilingContext *tilingContext_ = nullptr;
    MixedQuantSparseFlashMlaPtoBnnTilingData tilingData_;
    int64_t t1_ = 0;
    int64_t bNum_ = 0;
    int64_t kOri_ = 0;
    int64_t kCmp_ = 0;
    int64_t bsOri_ = 0;
    int64_t bsCmp_ = 0;
    int64_t btOri_ = 0;
    int64_t btCmp_ = 0;
    int64_t oriPages_ = 0;
    int64_t cmpPages_ = 0;
    int64_t oriKvStride0_ = 0;
    int64_t cmpKvStride0_ = 0;
    int64_t blockDim_ = 1;
    int64_t vdep64_ = 0;
    float scale_ = 0.0F;
    bool hasCmp_ = false;
    bool hasOriLen_ = false;
    bool hasCmpLen_ = false;
    uint32_t cubeCoreNum_ = 0;
};

ge::DataType TilingMixedQuantSparseFlashMlaPtoBnnFunc::GetInputDtype(gert::TilingContext *context, size_t index)
{
    const auto *desc = context->GetInputDesc(index);
    return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
}

ge::DataType TilingMixedQuantSparseFlashMlaPtoBnnFunc::GetOutputDtype(gert::TilingContext *context, size_t index)
{
    const auto *desc = context->GetOutputDesc(index);
    return desc == nullptr ? ge::DT_UNDEFINED : desc->GetDataType();
}

bool TilingMixedQuantSparseFlashMlaPtoBnnFunc::CheckShape(const gert::StorageShape *shape,
                                                    std::initializer_list<int64_t> dims)
{
    if (shape == nullptr) {
        return false;
    }
    const auto &storage = shape->GetStorageShape();
    if (dims.size() != storage.GetDimNum()) {
        return false;
    }
    int64_t i = 0;
    for (int64_t dim : dims) {
        if (storage.GetDim(static_cast<size_t>(i)) != dim) {
            return false;
        }
        ++i;
    }
    return true;
}

ge::graphStatus TilingMixedQuantSparseFlashMlaPtoBnnFunc::ResolveKvPageStride(size_t index, const char *name,
                                                                              int64_t bs, int64_t rowBytes,
                                                                              int64_t *stride0)
{
    // 非连续输入只允许页间 padding；连续输入（无 view 信息）按紧凑页长取 stride。
    *stride0 = bs * rowBytes;
    if (!tilingContext_->InputIsView(index)) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Stride *stride = tilingContext_->GetInputStride(index);
    if (stride == nullptr || stride->GetDimNum() != static_cast<size_t>(4)) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: %s view must carry 4-D stride information.", name);
        return ge::GRAPH_FAILED;
    }
    const int64_t slotStride = stride->GetStride(1);
    const int64_t lastStride = stride->GetStride(3);
    if (slotStride != rowBytes || lastStride != 1) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: %s only supports page-axis (axis-0) non-contiguity, got "
                "strides [%ld, %ld, %ld, %ld] for row bytes %ld.",
                name, stride->GetStride(0), slotStride, stride->GetStride(2), lastStride, rowBytes);
        return ge::GRAPH_FAILED;
    }
    const int64_t pageStride = stride->GetStride(0);
    if (pageStride < bs * rowBytes) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: %s page stride(%ld) must cover one page(%ld).", name,
                pageStride, bs * rowBytes);
        return ge::GRAPH_FAILED;
    }
    // kernel 页地址按 int32 计算，页跨度不能超出 int32 表达范围。
    if (pageStride > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: %s page stride(%ld) exceeds int32 range.", name, pageStride);
        return ge::GRAPH_FAILED;
    }
    *stride0 = pageStride;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingMixedQuantSparseFlashMlaPtoBnnFunc::ParseShapesAndAttrs()
{
    // ---- attributes (mirrors the Python wrapper) ----
    const char *layoutQ = tilingContext_->GetAttrs() == nullptr ? nullptr : tilingContext_->GetAttrs()->GetStr(0);
    const char *layoutKv = tilingContext_->GetAttrs() == nullptr ? nullptr : tilingContext_->GetAttrs()->GetStr(1);
    if (layoutQ == nullptr || std::strcmp(layoutQ, "TND") != 0) {
        OP_LOGE(tilingContext_->GetNodeName(), "mixed_quant_sparse_flash_mla_pto_bnn: layout_q must be 'TND'.");
        return ge::GRAPH_FAILED;
    }
    if (layoutKv == nullptr || std::strcmp(layoutKv, "PA_BBND") != 0) {
        OP_LOGE(tilingContext_->GetNodeName(), "mixed_quant_sparse_flash_mla_pto_bnn: layout_kv must be 'PA_BBND'.");
        return ge::GRAPH_FAILED;
    }
    const float *scaleAttr = nullptr;
    if (tilingContext_->GetAttrs() != nullptr) {
        scaleAttr = tilingContext_->GetAttrs()->GetFloat(2);
    }
    scale_ = (scaleAttr != nullptr && *scaleAttr > 0.0F)
        ? *scaleAttr
        : static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));

    const auto *qShape = tilingContext_->GetInputShape(QUERY);
    const auto *oriKvShape = tilingContext_->GetInputShape(ORI_KV);
    const auto *idxShape = tilingContext_->GetInputShape(ORI_SPARSE_INDICES);
    const auto *btShape = tilingContext_->GetInputShape(ORI_BLOCK_TABLE);
    const auto *sinksShape = tilingContext_->GetInputShape(SINKS);
    const auto *cuShape = tilingContext_->GetInputShape(CU_SEQLENS_Q);
    if (qShape == nullptr || oriKvShape == nullptr || idxShape == nullptr ||
        btShape == nullptr || sinksShape == nullptr || cuShape == nullptr) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: a required input shape is null.");
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(tilingContext_, QUERY) != ge::DT_BF16) {
        OP_LOGE(tilingContext_->GetNodeName(), "mixed_quant_sparse_flash_mla_pto_bnn: q must be BF16.");
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(tilingContext_, ORI_KV) != ge::DT_UINT8 ||
        GetInputDtype(tilingContext_, ORI_SPARSE_INDICES) != ge::DT_INT32 ||
        GetInputDtype(tilingContext_, ORI_BLOCK_TABLE) != ge::DT_INT32 ||
        GetInputDtype(tilingContext_, SINKS) != ge::DT_FLOAT) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: input dtype contract violated.");
        return ge::GRAPH_FAILED;
    }
    // ValueDepend planning inputs accept INT32 (Tensor variant) or INT64
    // (aclIntArray variant of the generated aclnn interface).
    for (size_t idx : {static_cast<size_t>(CU_SEQLENS_Q)}) {
        auto dtype = GetInputDtype(tilingContext_, idx);
        if (dtype != ge::DT_INT32 && dtype != ge::DT_INT64) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla_pto_bnn: cu_seqlens_q must be int32/int64.");
            return ge::GRAPH_FAILED;
        }
    }

    // q [T1, 64, 512]
    const auto &qs = qShape->GetStorageShape();
    if (qs.GetDimNum() != 3 || qs.GetDim(1) != N1 || qs.GetDim(2) != D) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: q must be [T1, 64, 512].");
        return ge::GRAPH_FAILED;
    }
    t1_ = qs.GetDim(0);

    // ori_kv [pages, bs, 1, 544]
    const auto &oks = oriKvShape->GetStorageShape();
    if (oks.GetDimNum() != 4 || oks.GetDim(2) != 1 || oks.GetDim(3) != ROW_BYTES_ORI) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: ori_kv must be [pages, bs, 1, 544].");
        return ge::GRAPH_FAILED;
    }
    oriPages_ = oks.GetDim(0);
    bsOri_ = oks.GetDim(1);
    if (ResolveKvPageStride(ORI_KV, "ori_kv", bsOri_, ROW_BYTES_ORI, &oriKvStride0_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // ori_sparse_indices [T1, 1, k_ori]
    const auto &is = idxShape->GetStorageShape();
    if (is.GetDimNum() != 3 || is.GetDim(0) != t1_ || is.GetDim(1) != 1) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: ori_sparse_indices must be [T1, 1, k_ori].");
        return ge::GRAPH_FAILED;
    }
    kOri_ = is.GetDim(2);

    // cu_seqlens_q [B + 1]
    const auto &cus = cuShape->GetStorageShape();
    if (cus.GetDimNum() != 1) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: cu_seqlens_q must be rank 1.");
        return ge::GRAPH_FAILED;
    }
    bNum_ = cus.GetDim(0);

    // ori_block_table [B, bt_ori] with B = b_num - 1
    const auto &bts = btShape->GetStorageShape();
    if (bts.GetDimNum() != 2 || bts.GetDim(0) != bNum_ - 1) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: ori_block_table must be [B, bt_ori].");
        return ge::GRAPH_FAILED;
    }
    btOri_ = bts.GetDim(1);

    // sinks [64]
    if (!CheckShape(sinksShape, {N1})) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: sinks must be [64].");
        return ge::GRAPH_FAILED;
    }

    // ---- optional cmp group: all-or-none ----
    hasCmp_ = tilingContext_->GetOptionalInputShape(CMP_KV) != nullptr;
    if (hasCmp_) {
        const auto *cmpKvShape = tilingContext_->GetOptionalInputShape(CMP_KV);
        const auto *cmpIdxShape = tilingContext_->GetOptionalInputShape(CMP_SPARSE_INDICES);
        const auto *cmpBtShape = tilingContext_->GetOptionalInputShape(CMP_BLOCK_TABLE);
        if (cmpIdxShape == nullptr || cmpBtShape == nullptr) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla_pto_bnn: cmp_kv given but cmp_sparse_indices/cmp_block_table missing.");
            return ge::GRAPH_FAILED;
        }
        if (GetInputDtype(tilingContext_, CMP_KV) != ge::DT_UINT8 ||
            GetInputDtype(tilingContext_, CMP_SPARSE_INDICES) != ge::DT_INT32 ||
            GetInputDtype(tilingContext_, CMP_BLOCK_TABLE) != ge::DT_INT32) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla_pto_bnn: cmp group dtype contract violated.");
            return ge::GRAPH_FAILED;
        }
        const auto &cks = cmpKvShape->GetStorageShape();
        if (cks.GetDimNum() != 4 || cks.GetDim(2) != 1 || cks.GetDim(3) != ROW_BYTES_CMP) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla_pto_bnn: cmp_kv must be [pages, bs, 1, 320].");
            return ge::GRAPH_FAILED;
        }
        cmpPages_ = cks.GetDim(0);
        bsCmp_ = cks.GetDim(1);
        if (ResolveKvPageStride(CMP_KV, "cmp_kv", bsCmp_, ROW_BYTES_CMP, &cmpKvStride0_) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        const auto &cis = cmpIdxShape->GetStorageShape();
        if (cis.GetDimNum() != 3 || cis.GetDim(0) != t1_ || cis.GetDim(1) != 1) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla_pto_bnn: cmp_sparse_indices must be [T1, 1, k_cmp].");
            return ge::GRAPH_FAILED;
        }
        kCmp_ = cis.GetDim(2);
        const auto &cbts = cmpBtShape->GetStorageShape();
        if (cbts.GetDimNum() != 2 || cbts.GetDim(0) != bNum_ - 1) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla_pto_bnn: cmp_block_table must be [B, bt_cmp].");
            return ge::GRAPH_FAILED;
        }
        btCmp_ = cbts.GetDim(1);
        hasCmpLen_ = tilingContext_->GetOptionalInputShape(CMP_TOP_K_LEN) != nullptr;
    } else {
        cmpPages_ = 0;
        bsCmp_ = 0;
        kCmp_ = 0;
        btCmp_ = 0;
        cmpKvStride0_ = 0;
    }
    hasOriLen_ = tilingContext_->GetOptionalInputShape(ORI_TOP_K_LEN) != nullptr;
    if (hasOriLen_) {
        auto dtype = GetInputDtype(tilingContext_, ORI_TOP_K_LEN);
        if (dtype != ge::DT_INT32 && dtype != ge::DT_INT64) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla_pto_bnn: ori_topk_length must be int32/int64.");
            return ge::GRAPH_FAILED;
        }
    }
    if (hasCmpLen_) {
        auto dtype = GetInputDtype(tilingContext_, CMP_TOP_K_LEN);
        if (dtype != ge::DT_INT32 && dtype != ge::DT_INT64) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "mixed_quant_sparse_flash_mla_pto_bnn: cmp_topk_length must be int32/int64.");
            return ge::GRAPH_FAILED;
        }
    }
    vdep64_ = (GetInputDtype(tilingContext_, CU_SEQLENS_Q) == ge::DT_INT64) ? 1 : 0;
    if (hasOriLen_ && GetInputDtype(tilingContext_, ORI_TOP_K_LEN) == ge::DT_INT64) {
        vdep64_ = 1;
    }
    if (hasCmpLen_ && GetInputDtype(tilingContext_, CMP_TOP_K_LEN) == ge::DT_INT64) {
        vdep64_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingMixedQuantSparseFlashMlaPtoBnnFunc::ValidateTensors()
{
    // ---- metadata plan (input tensor, not a tiling field) ----
    const auto *mdShape = tilingContext_->GetInputShape(METADATA);
    if (!CheckShape(mdShape, {MQSMLA_METADATA_TOTAL_SIZE})) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: metadata must be int32[%ld].",
                MQSMLA_METADATA_TOTAL_SIZE);
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(tilingContext_, METADATA) != ge::DT_INT32) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: metadata must be int32.");
        return ge::GRAPH_FAILED;
    }

    // ---- outputs ----
    if (!CheckShape(tilingContext_->GetOutputShape(OUT), {t1_, N1, D})) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: out must be [T1=%ld, 64, 512].", t1_);
        return ge::GRAPH_FAILED;
    }
    if (GetOutputDtype(tilingContext_, OUT) != ge::DT_BF16) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: out must be BF16.");
        return ge::GRAPH_FAILED;
    }
    if (!CheckShape(tilingContext_->GetOutputShape(LSE), {1, t1_, N1})) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: lse must be [1, T1=%ld, 64].", t1_);
        return ge::GRAPH_FAILED;
    }
    if (GetOutputDtype(tilingContext_, LSE) != ge::DT_FLOAT) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: lse must be float32.");
        return ge::GRAPH_FAILED;
    }

    // ---- device-side tiling params (AICPU planner output, input tensor) ----
    const auto *tdShape = tilingContext_->GetInputShape(TILING_DEVICE);
    if (!CheckShape(tdShape, {TILING_DEVICE_SIZE})) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: tiling_device must be int32[%ld].",
                TILING_DEVICE_SIZE);
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype(tilingContext_, TILING_DEVICE) != ge::DT_INT32) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: tiling_device must be int32.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingMixedQuantSparseFlashMlaPtoBnnFunc::ComputeTiling()
{
    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    cubeCoreNum_ = platformInfo.GetCoreNumAic();
    if (cubeCoreNum_ <= 0) {
        cubeCoreNum_ = 1;
    }
    if (cubeCoreNum_ > AIC_CORE_MAX_NUM) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "mixed_quant_sparse_flash_mla_pto_bnn: device cube cores(%u) exceed AIC_CORE_MAX_NUM(%ld).",
                cubeCoreNum_, AIC_CORE_MAX_NUM);
        return ge::GRAPH_FAILED;
    }

    // 值无关超集: 恒发射全部 Cube 核, 计划内未启用的核空转(全零 FA 槽)。
    blockDim_ = static_cast<int64_t>(cubeCoreNum_);

    tilingData_.set_scale(scale_);
    tilingData_.set_t1(t1_);
    tilingData_.set_ori_pages(oriPages_);
    tilingData_.set_cmp_pages(cmpPages_);
    tilingData_.set_b_num(bNum_);
    tilingData_.set_k_ori(kOri_);
    tilingData_.set_k_cmp(hasCmp_ ? kCmp_ : 129);
    tilingData_.set_bs_ori(bsOri_);
    tilingData_.set_bs_cmp(hasCmp_ ? bsCmp_ : 128);
    tilingData_.set_bt_ori(btOri_);
    tilingData_.set_bt_cmp(hasCmp_ ? btCmp_ : 0);
    // 页跨度取输入真实 stride（页间 padding 支持），连续输入由 ResolveKvPageStride 回落紧凑页长。
    tilingData_.set_ori_kv_stride0(oriKvStride0_);
    tilingData_.set_cmp_kv_stride0(hasCmp_ ? cmpKvStride0_ : 0);

    // Tiling key encodes (HAS_ORI_LEN, HAS_CMP_LEN, HAS_CMP, FD_ENABLE, VDEP64)
    // at bits 0..4 in the MqsmlaPtoBnnTilingKey declaration order. FD_ENABLE 常置
    // 1: 变体内常驻 FD 归约段, 运行期按 metadata FD 槽自分流(uniform 计划零跳过);
    // 其余 tiling 参数由 kernel 从 tiling_device 张量读取。
    const int64_t tilingKey = (hasOriLen_ ? 1 : 0) | (hasCmpLen_ ? 1 : 0) << 1 |
                              (hasCmp_ ? 1 : 0) << 2 | (1LL << 3) |
                              (vdep64_ << 4);
    tilingContext_->SetTilingKey(tilingKey);

    // workspace 恒为 FD 超集: FD 变体始终可用, uniform 计划不触碰。
    size_t *workspace = tilingContext_->GetWorkspaceSizes(1);
    workspace[0] = static_cast<size_t>(WORKSPACE_BYTES_FD);

    tilingContext_->SetBlockDim(static_cast<uint32_t>(blockDim_));

    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingMixedQuantSparseFlashMlaPtoBnnFunc::RunTiling()
{
    auto ret = ParseShapesAndAttrs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ValidateTensors();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ComputeTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingMixedQuantSparseFlashMlaPtoBnnFunc tilingObject(context);
    auto ret = tilingObject.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "mixed_quant_sparse_flash_mla_pto_bnn tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MixedQuantSparseFlashMlaPtoBnn)
    .Tiling(TilingFunc)
    .TilingInputsDataDependency({ORI_TOP_K_LEN, CMP_TOP_K_LEN, CU_SEQLENS_Q});

} // namespace optiling