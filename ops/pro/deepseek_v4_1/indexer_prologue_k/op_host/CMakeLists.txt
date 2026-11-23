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
require_pypto_pro(indexer_prologue_k)
add_op_to_compiled_list()
set(indexer_prologue_k_projection_dir "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE INTERNAL
    "kernel source dir for IndexerPrologueKProjection")
set(indexer_prologue_k_postprocess_dir "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE INTERNAL
    "kernel source dir for IndexerPrologueKPostprocess")

target_sources(op_host_aclnn PRIVATE
indexer_prologue_k_fused_def.cpp
indexer_prologue_k_projection_def.cpp
indexer_prologue_k_postprocess_def.cpp
)
add_modules_sources(
    OPTYPE indexer_prologue_k indexer_prologue_k_projection indexer_prologue_k_postprocess
    ACLNNTYPE aclnn aclnn aclnn)

function(indexer_prologue_k_add_entry entry source header)
    enable_pypto_kernel(${entry})
    set(_gen "${PYPTO_GEN_DIR}")
    set(_layout_dir "${CMAKE_CURRENT_BINARY_DIR}/${entry}_layout")
    file(MAKE_DIRECTORY "${_layout_dir}")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${header}" "${_layout_dir}/${header}" COPYONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/indexer_prologue_k_plan.h"
                   "${_layout_dir}/indexer_prologue_k_plan.h" COPYONLY)
    set_property(GLOBAL APPEND PROPERTY GYM_TILING_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    set_property(GLOBAL APPEND PROPERTY GYM_PYPTO_GEN_DIRS "${_gen}")
    string(MD5 _source_id "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    set_property(GLOBAL PROPERTY GYM_TILING_GEN_${_source_id} "${_gen}")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PYPTO_GYM_SRC_DIR}/cmake/scripts/tiling_layout_check.py
            --gen-dir "${_gen}" --host-dir "${_layout_dir}"
            --op-file "${entry}" --emit-dir "${CMAKE_BINARY_DIR}/tiling_check"
        COMMAND_ERROR_IS_FATAL ANY)
    set_property(GLOBAL APPEND PROPERTY GYM_TILING_CHECK_SRCS
        "${CMAKE_BINARY_DIR}/tiling_check/${entry}_tiling_layout_check.cpp")
endfunction()

indexer_prologue_k_add_entry(indexer_prologue_k
    indexer_prologue_k_fused_tiling.cpp indexer_prologue_k_fused_tiling.h)
indexer_prologue_k_add_entry(indexer_prologue_k_projection
    indexer_prologue_k_projection_tiling.cpp indexer_prologue_k_projection_tiling.h)
indexer_prologue_k_add_entry(indexer_prologue_k_postprocess
    indexer_prologue_k_postprocess_tiling.cpp indexer_prologue_k_postprocess_tiling.h)

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME IndexerPrologueK
        OPTIONS -Wno-deprecated-declarations
                -Werror
    )
    add_ops_compile_options(
        OP_NAME IndexerPrologueKProjection
        OPTIONS -Wno-deprecated-declarations
                -Werror
    )
    add_ops_compile_options(
        OP_NAME IndexerPrologueKPostprocess
        OPTIONS -Wno-deprecated-declarations
                -Werror
    )
else()
    add_ops_compile_options(
        OP_NAME IndexerPrologueK
        OPTIONS -Wno-deprecated-declarations
    )
    add_ops_compile_options(
        OP_NAME IndexerPrologueKProjection
        OPTIONS -Wno-deprecated-declarations
    )
    add_ops_compile_options(
        OP_NAME IndexerPrologueKPostprocess
        OPTIONS -Wno-deprecated-declarations
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        OPTYPE indexer_prologue_k indexer_prologue_k_projection indexer_prologue_k_postprocess
        ACLNNTYPE aclnn aclnn aclnn)
endif()
