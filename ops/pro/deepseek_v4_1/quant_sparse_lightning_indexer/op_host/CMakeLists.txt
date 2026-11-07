# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18.4)
require_pypto_pro(quant_sparse_lightning_indexer)
add_op_to_compiled_list()

target_sources(op_host_aclnn PRIVATE
quant_sparse_lightning_indexer_def.cpp
)
add_modules_sources(OPTYPE quant_sparse_lightning_indexer ACLNNTYPE aclnn)
enable_pypto_kernel(quant_sparse_lightning_indexer)

# ascend950 条件编译——缺少此项会导致 kernel binary 未编译
# ENABLE_CV_COMM_VIA_SSBUF: C/V 跨核通信走硬件 SSBUF, 用户 workspace 即 kernel 收到的
# workspace 实参本身(无 16MB 系统预留), 这正是本算子用 pl.addptr 切分 5 段工作区的前提.
set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME QuantSparseLightningIndexer
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

# ---- 编译哪些 tiling key ----
# QsliKey 经 is_valid 无损剪枝后有 120 个合法 key(原笛卡尔积 320: IsPA=1 时
# TndStrideMode 不被读取、IsPA=0 时 PairMin 不被读取). 但 built-in 路径实际可达的
# 只有 60 个: aclnn/nnopbase 会在 launch 前把非连续输入拷成连续(见 tiling.cpp 的
# 说明与实测 workspace 差值), kernel 永远只看到紧凑 stride, 故 TND 侧只会选中
# TndStrideMode=3; PA 侧 PairMin 仍随确定性等级取 544/0 两档.
#   PA : WriteValues(2) x TopKCapacity(10) x PairMin(2)                  = 40
#   TND: WriteValues(2) x TopKCapacity(10) x TndStrideMode=3 x PairMin=0 = 20
# jit 宿主入口不做这次拷贝, 仍可选到 mode 0/1/2, 所以 is_valid 保留 120 个组合;
# 需要把 120 个全编进 .o 时把 QSLI_COMPILE_REACHABLE_TILING_KEYS 关掉即可.
# 打包值 = WriteValues@0 | IsPA@1 | capIdx@2 | TndStrideMode@6 | PairMinIdx@8
option(QSLI_COMPILE_REACHABLE_TILING_KEYS
       "compile only the 60 tiling keys the built-in tiling can select" ON)
if (QSLI_COMPILE_REACHABLE_TILING_KEYS)
    set(QSLI_REACHABLE_KEYS
        2 3 6 7 10 11 14 15 18 19 22 23
        26 27 30 31 34 35 38 39 258 259 262 263
        266 267 270 271 274 275 278 279 282 283 286 287
        290 291 294 295 448 449 452 453 456 457 460 461
        464 465 468 469 472 473 476 477 480 481 484 485
    )
    add_ops_tiling_keys(
        OP_NAME QuantSparseLightningIndexer
        TILING_KEYS ${QSLI_REACHABLE_KEYS}
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME QuantSparseLightningIndexer
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )
else()
    add_ops_compile_options(
        OP_NAME QuantSparseLightningIndexer
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE quant_sparse_lightning_indexer ACLNNTYPE aclnn)
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
endif()
