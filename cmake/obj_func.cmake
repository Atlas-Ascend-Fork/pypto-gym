# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# gym host 侧源收集宏: 算子 op_host/CMakeLists.txt 消费的宏集合.
# 源统一登记到全局属性, 编译目标(ophost_gym_*_obj / 三 .so)在 custom_build.cmake 顶层统一创建.

function(extract_ops_name SOURCE_DIR OUTPUT_VAR)
    file(GLOB_RECURSE OP_DEF_FILES ${SOURCE_DIR}/*_def*.cpp)
    foreach(OP_DEF_FILE ${OP_DEF_FILES})
        get_filename_component(OP_DEF_NAME ${OP_DEF_FILE} NAME_WE)
        string(REGEX REPLACE "_def$" "" OP_NAME_TMP ${OP_DEF_NAME})
        set(${OUTPUT_VAR} ${OP_NAME_TMP} PARENT_SCOPE)
    endforeach()
endfunction()

# add_modules_sources(OPTYPE <op_types> ACLNNTYPE <aclnn|aclnn_inner|aclnn_exclude> [ACLNN_EXTRA_VERSION ...])
# def 编入 op_build 反射 .so; infershape/手写 opapi 源登记到全局属性.
macro(add_modules_sources)
    set(multiValueArgs OPTYPE ACLNNTYPE ACLNN_EXTRA_VERSION)
    cmake_parse_arguments(MODULE "" "" "${multiValueArgs}" ${ARGN})
    set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

    list(LENGTH MODULE_OPTYPE OpTypeLen)
    if(NOT OpTypeLen)
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/*_def*.cpp)
        if (OPDEF_SRCS)
            message(FATAL_ERROR
                "Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
                "usage: add_modules_sources(OPTYPE optypes ACLNNTYPE aclnntypes)")
        endif()
    endif()

    math(EXPR _idx "${OpTypeLen} - 1")
    if(_idx GREATER -1)
        foreach(i RANGE ${_idx})
            list(GET MODULE_OPTYPE ${i} _op_type)
            list(GET MODULE_ACLNNTYPE ${i} _aclnn_type)
            file(GLOB _def_srcs ${SOURCE_DIR}/${_op_type}_def*.cpp)
            if(_def_srcs)
                if(_aclnn_type STREQUAL "aclnn" AND TARGET op_host_aclnn)
                    target_sources(op_host_aclnn PRIVATE ${_def_srcs})
                elseif(_aclnn_type STREQUAL "aclnn_inner" AND TARGET op_host_aclnnInner)
                    target_sources(op_host_aclnnInner PRIVATE ${_def_srcs})
                elseif(_aclnn_type STREQUAL "aclnn_exclude" AND TARGET op_host_aclnnExc)
                    target_sources(op_host_aclnnExc PRIVATE ${_def_srcs})
                else()
                    message(FATAL_ERROR "ACLNN TYPE UNSUPPORTED: ${_aclnn_type}")
                endif()
            endif()
            set_property(GLOBAL APPEND PROPERTY GYM_ACLNN_TYPE_${_aclnn_type} ${_op_type})

            file(GLOB _infer_srcs ${SOURCE_DIR}/*_infershape*.cpp)
            if(_infer_srcs)
                set_property(GLOBAL APPEND PROPERTY GYM_INFER_SRCS ${_infer_srcs})
            endif()
            # 手写 aclnn 通道: 旧结构 op_host/op_api/ 与新结构同级 op_api/ 均发现
            file(GLOB _opapi_srcs ${SOURCE_DIR}/op_api/*.cpp)
            if(NOT _opapi_srcs)
                file(GLOB _opapi_srcs ${SOURCE_DIR}/../op_api/*.cpp)
            endif()
            if(_opapi_srcs)
                set_property(GLOBAL APPEND PROPERTY GYM_OPAPI_SRCS ${_opapi_srcs})
            endif()
        endforeach()
    endif()
endmacro()

macro(add_modules_sources_with_soc)
    add_modules_sources(${ARGN})
endmacro()

# tiling 源收集(在 enable_pypto_kernel 之后调用, PYPTO_GEN_DIR 指向生成头目录)
macro(add_tiling_modules)
    file(GLOB _tiling_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*_tiling*.cpp)
    if(_tiling_srcs)
        set_property(GLOBAL APPEND PROPERTY GYM_TILING_SRCS ${_tiling_srcs})
    endif()
    file(GLOB _sub_tiling_srcs ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/*.cpp)
    if(_sub_tiling_srcs)
        set_property(GLOBAL APPEND PROPERTY GYM_TILING_SRCS ${_sub_tiling_srcs})
    endif()
    if(DEFINED PYPTO_GEN_DIR)
        set_property(GLOBAL APPEND PROPERTY GYM_PYPTO_GEN_DIRS ${PYPTO_GEN_DIR})
        # tiling 布局校验(configure 期执行, codegen 已由 enable_pypto_kernel 完成).
        # 手写 TilingData 与 kernel dataclass 字段级 diff, 不一致即 FATAL_ERROR;
        # 一致则登记布局核对程序源(构建期编译并执行)供 custom_build.cmake 消费.
        get_filename_component(_tl_op_file ${PYPTO_GEN_DIR} NAME)
        string(REGEX REPLACE "_pypto_gen$" "" _tl_op_file ${_tl_op_file})
        set(_tl_emit_dir ${CMAKE_BINARY_DIR}/tiling_check)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} ${PYPTO_GYM_SRC_DIR}/cmake/scripts/tiling_layout_check.py
                    --gen-dir ${PYPTO_GEN_DIR}
                    --host-dir ${CMAKE_CURRENT_SOURCE_DIR}
                    --op-file ${_tl_op_file}
                    --emit-dir ${_tl_emit_dir}
            RESULT_VARIABLE _tl_rc)
        if(NOT _tl_rc EQUAL 0)
            message(FATAL_ERROR
                "tiling layout check failed for ${_tl_op_file}: "
                "handwritten TilingData != kernel dataclass (see diff above)")
        endif()
        set_property(GLOBAL APPEND PROPERTY GYM_TILING_CHECK_SRCS
                ${_tl_emit_dir}/${_tl_op_file}_tiling_layout_check.cpp)
    endif()
endmacro()

function(add_opapi_modules)
endfunction()

function(add_infer_modules)
endfunction()

# ---------------------------------------- op_build 反射载体 ----------------------------------------
# 三个 stub .so: def.cpp 经 add_modules_sources 按通道收集于此, 供 op_build 反射生成 aclnn/ops-info.
# 顺序约束: 必须在算子 op_host 接入(CMakeLists 顶层 add_subdirectory)之前创建.
add_library(op_host_aclnn SHARED EXCLUDE_FROM_ALL)
add_library(op_host_aclnnInner SHARED EXCLUDE_FROM_ALL)
add_library(op_host_aclnnExc SHARED EXCLUDE_FROM_ALL)
foreach(_aclnn_lib op_host_aclnn op_host_aclnnInner op_host_aclnnExc)
    target_include_directories(${_aclnn_lib} PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime)
    target_link_directories(${_aclnn_lib} PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/lib64)
    target_link_libraries(${_aclnn_lib} PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/lib64/libexe_graph.so
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/lib64/libregister.so
        ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so)
endforeach()
