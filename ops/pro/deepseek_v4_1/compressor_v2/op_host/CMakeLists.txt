# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.19)

if(DEFINED CMAKE_SCRIPT_MODE_FILE)
    foreach(_entry compressor_v2 compressor_v2_vector)
        set(_path "${COMPRESSOR_V2_WRAPPER_DIR}/${_entry}.py")
        file(READ "${_path}" _wrapper)
        set(_frontend [=[import sys
import sysconfig
sys.path.insert(0, sysconfig.get_path("purelib"))
import functools
from pypto_pro.runtime.opc.pypto_compile import pypto_compile_op as _compressor_compile

@functools.wraps(_compressor_compile)
def pypto_compile_op(*args, **kwargs):
    return _compressor_compile(*args, **kwargs)

def _compressor_load_module(spec, module):
    spec.loader.exec_module(module)

]=])
        string(FIND "${_wrapper}" "${_frontend}" _ready)
        if(NOT _ready EQUAL 0)
            set(_import "from pypto_pro.runtime.opc.pypto_compile import pypto_compile_op")
            string(FIND "${_wrapper}" "${_import}" _position)
            string(FIND "${_wrapper}" "_spec.loader.exec_module(_chk_mod)" _load_position)
            if(_position EQUAL -1 OR _load_position EQUAL -1)
                message(FATAL_ERROR "CompressorV2 requires the generated PyPTO OPC wrapper")
            endif()
            string(REPLACE "${_import}" "pass # configure-time PyPTO compiler binding" _wrapper "${_wrapper}")
            string(REPLACE "_spec.loader.exec_module(_chk_mod)" "_compressor_load_module(_spec, _chk_mod)" _wrapper "${_wrapper}")
            file(WRITE "${_path}" "${_frontend}${_wrapper}")
        endif()
    endforeach()
    return()
endif()

require_pypto_pro(compressor_v2)
add_op_to_compiled_list()

target_sources(op_host_aclnn PRIVATE compressor_v2_def.cpp)
add_modules_sources(
    OPTYPE pypto_compressor_v2_stage pypto_compressor_v2_vector
    ACLNNTYPE aclnn aclnn)

# Each entry has a different engine type and its own generated tiling-key header.
# Keep product sources flat; isolate registration metadata only in the build tree
# because the shared layout checker accepts one registration per host directory.
function(compressor_v2_add_entry entry)
    enable_pypto_kernel(${entry})
    set(_source "${CMAKE_CURRENT_SOURCE_DIR}/${entry}_tiling.cpp")
    set(_layout_dir "${CMAKE_CURRENT_BINARY_DIR}/${entry}_layout")
    file(MAKE_DIRECTORY "${_layout_dir}")
    configure_file("${entry}_tiling.h" "${_layout_dir}/${entry}_tiling.h" COPYONLY)
    configure_file("compressor_v2_plan.h" "${_layout_dir}/compressor_v2_plan.h" COPYONLY)
    set_property(GLOBAL APPEND PROPERTY GYM_TILING_SRCS "${_source}")
    set_property(GLOBAL APPEND PROPERTY GYM_PYPTO_GEN_DIRS "${PYPTO_GEN_DIR}")
    string(MD5 _source_id "${_source}")
    set_property(GLOBAL PROPERTY GYM_TILING_GEN_${_source_id} "${PYPTO_GEN_DIR}")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PYPTO_GYM_SRC_DIR}/cmake/scripts/tiling_layout_check.py
            --gen-dir "${PYPTO_GEN_DIR}" --host-dir "${_layout_dir}"
            --op-file ${entry} --emit-dir "${CMAKE_BINARY_DIR}/tiling_check"
        COMMAND_ERROR_IS_FATAL ANY)
    set_property(GLOBAL APPEND PROPERTY GYM_TILING_CHECK_SRCS
        "${CMAKE_BINARY_DIR}/tiling_check/${entry}_tiling_layout_check.cpp")
endfunction()

compressor_v2_add_entry(compressor_v2)
compressor_v2_add_entry(compressor_v2_vector)

function(compressor_v2_configure_wrappers)
    set(_targets)
    foreach(_entry compressor_v2 compressor_v2_vector)
        foreach(_soc ${ASCEND_COMPUTE_UNIT})
            if(TARGET ${_entry}_${_soc}_py_copy)
                list(APPEND _targets ${_entry}_${_soc}_py_copy)
            endif()
        endforeach()
    endforeach()
    if(_targets)
        add_custom_target(compressor_v2_prepare_wrappers
            COMMAND "${CMAKE_COMMAND}" "-DCOMPRESSOR_V2_WRAPPER_DIR=${ASCEND_IMPL_OUT_DIR}/dynamic"
                -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
            DEPENDS "${CMAKE_CURRENT_FUNCTION_LIST_FILE}" VERBATIM)
        add_dependencies(compressor_v2_prepare_wrappers generate_gym_adapt_py)
        foreach(_target ${_targets})
            add_dependencies(${_target} compressor_v2_prepare_wrappers)
        endforeach()
    endif()
endfunction()
cmake_language(DEFER DIRECTORY "${PYPTO_GYM_SRC_DIR}" CALL compressor_v2_configure_wrappers)


foreach(_op PyptoCompressorV2Stage PyptoCompressorV2Vector)
    add_ops_compile_options(OP_NAME ${_op}
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true --cce-auto-sync=off
                -Wno-deprecated-declarations
                --enable-pto-tile-fusion -mllvm -tile-fusion-skip-reduceop-fusion=true
                -mllvm -tile-fusion-skip-legality-check=false)
endforeach()
if(ENABLE_OPS_HOST AND NOT PREPARE_BUILD)
    install(FILES ${ASCEND_IMPL_OUT_DIR}/dynamic/compressor_v2_vector.py
        DESTINATION packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl/dynamic
        COMPONENT ${VENDOR_NAME})
endif()
