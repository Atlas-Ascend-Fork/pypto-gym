# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.19)

# Build-time wrapper preparation reuses this file in CMake script mode.
if(DEFINED CMAKE_SCRIPT_MODE_FILE)
    file(READ "${ATTEN_EPILOGUE_WRAPPER}" _wrapper)
    string(FIND "${_wrapper}"
        "from pypto_pro.runtime.opc.pypto_compile import pypto_compile_op"
        _wrapper_import)
    if(_wrapper_import EQUAL -1)
        message(FATAL_ERROR "AttenEpilogue requires a PyPTO OPC wrapper")
    endif()
    set(_frontend_path [=[import sys
import sysconfig

# Match configure-time PyPTO instead of the SDK path prepended by OPC.
sys.path.insert(0, sysconfig.get_path("purelib"))

]=])
    string(FIND "${_wrapper}" "${_frontend_path}" _frontend_offset)
    if(NOT _frontend_offset EQUAL 0)
        file(WRITE "${ATTEN_EPILOGUE_WRAPPER}" "${_frontend_path}${_wrapper}")
    endif()
    return()
endif()
require_pypto_pro(atten_epilogue)
add_op_to_compiled_list()

target_sources(op_host_aclnn PRIVATE
atten_epilogue_def.cpp
)
add_modules_sources(OPTYPE atten_epilogue ACLNNTYPE aclnn)
enable_pypto_kernel(atten_epilogue)

# The root dispatcher creates the native compile targets after visiting operators.
# Configure this operator's staged wrapper once those targets exist; shared build
# scripts and the wrappers of other operators remain untouched.
function(atten_epilogue_configure_wrapper)
    set(_copy_targets)
    foreach(_soc ${ASCEND_COMPUTE_UNIT})
        if(TARGET atten_epilogue_${_soc}_py_copy)
            list(APPEND _copy_targets atten_epilogue_${_soc}_py_copy)
        endif()
    endforeach()
    if(_copy_targets)
        add_custom_target(atten_epilogue_prepare_wrapper
            COMMAND "${CMAKE_COMMAND}"
                "-DATTEN_EPILOGUE_WRAPPER=${ASCEND_IMPL_OUT_DIR}/dynamic/atten_epilogue.py"
                -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
            DEPENDS "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
            VERBATIM)
        add_dependencies(atten_epilogue_prepare_wrapper generate_gym_adapt_py)
        foreach(_target ${_copy_targets})
            add_dependencies(${_target} atten_epilogue_prepare_wrapper)
        endforeach()
    endif()
endfunction()
cmake_language(DEFER DIRECTORY "${PYPTO_GYM_SRC_DIR}" CALL atten_epilogue_configure_wrapper)

# ascend950 条件编译——缺少此项会导致 kernel binary 未编译
# ENABLE_CV_COMM_VIA_SSBUF: C/V 跨核通信走硬件 SSBUF, 用户 workspace 即 kernel 收到的
# workspace 实参本身(无 16MB 系统预留), 这正是本算子用 pl.addptr 切分 aq/asc/yq/ys 的前提.
set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    # PyPTO kernel: runs codegen at configure time and exposes PYPTO_GEN_DIR for the host tiling -I below.
    add_ops_compile_options(
        OP_NAME AttenEpilogue
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME AttenEpilogue
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )
else()
    add_ops_compile_options(
        OP_NAME AttenEpilogue
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE atten_epilogue ACLNNTYPE aclnn)
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
endif()
