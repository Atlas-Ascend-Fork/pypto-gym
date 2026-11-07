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
require_pypto_pro(kv_compress_epilog_v2)
add_op_to_compiled_list()

target_sources(op_host_aclnn PRIVATE
kv_compress_epilog_v2_def.cpp
)
add_modules_sources(OPTYPE kv_compress_epilog_v2 ACLNNTYPE aclnn)
enable_pypto_kernel(kv_compress_epilog_v2)

# ascend950 条件编译——缺少此项会导致 kernel binary 未编译
set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME KvCompressEpilogV2
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME KvCompressEpilogV2
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )
else()
    add_ops_compile_options(
        OP_NAME KvCompressEpilogV2
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE kv_compress_epilog_v2 ACLNNTYPE aclnn)
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
endif()
