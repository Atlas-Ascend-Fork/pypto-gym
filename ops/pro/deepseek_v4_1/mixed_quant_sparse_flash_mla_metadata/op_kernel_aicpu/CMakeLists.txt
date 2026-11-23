# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED. See the License for the specific
# language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------

# PyPTO AICPU kernel: 本目录 mixed_quant_sparse_flash_mla_metadata_pto_bnn.py 由 pypto_pro codegen 生成 CpuKernel 适配层,
# 编译/链接/json 合并/安装走 cmake/aicpu.cmake 的 custom 包链路(gym_finalize_aicpu_kernels).

# 目录作用域切到 hcc aarch64 交叉编译器(OBJECT 目标链随此编译, 仅影响本目录创建的目标)
gym_resolve_aicpu_toolchain()
set(CMAKE_CXX_COMPILER ${GYM_AICPU_ARM_CXX})

enable_pypto_aicpu_kernel(mixed_quant_sparse_flash_mla_metadata_pto_bnn
        OP_TYPE MixedQuantSparseFlashMlaMetadataPtoBnn)
