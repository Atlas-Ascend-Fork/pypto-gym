# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#[[
PyPTO AICPU 自定义算子打包链路
  configure 期: pypto_pro codegen 由 op_kernel_aicpu/<op>.py 生成 CpuKernel 适配 TU + Args 头 + opInfo json
  构建期: ARM 交叉编译适配 TU -> 汇总链接 lib<vendor>_aicpu_kernels.so
  安装期: 合并 cust_aicpu_kernel.json, 与 .so 一起落到 vendors/<v>/op_impl/cpu/{aicpu_kernel/impl,config}
与手写 AICPU 算子同一条 custom 包路径: aclnn 侧 ADD_TO_LAUNCHER_LIST_AICPU 拉起, 不经过 Python.
]]

# 惰性解析 AICPU 交叉编译器(收敛自原 aicpu_func.cmake, 双布局兼容:
# toolkit/toolchain/hcc 优先, 旧布局 tools/hcc 回退; 结果进 CACHE 供全链路消费)。
function(gym_resolve_aicpu_toolchain)
    if (GYM_AICPU_TOOLCHAIN_RESOLVED)
        return()
    endif ()
    set(_gym_cxx_candidates
            ${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++
            ${ASCEND_CANN_PACKAGE_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-g++)
    set(_gym_resolved_cxx)
    foreach (_gym_cand IN LISTS _gym_cxx_candidates)
        if (EXISTS ${_gym_cand})
            set(_gym_resolved_cxx ${_gym_cand})
            break()
        endif ()
    endforeach ()
    if (NOT _gym_resolved_cxx)
        message(FATAL_ERROR "aicpu toolchain not found (aarch64-target-linux-gnu-g++) under ${ASCEND_CANN_PACKAGE_PATH}")
    endif ()
    set(GYM_AICPU_ARM_CXX ${_gym_resolved_cxx} CACHE INTERNAL "hcc aarch64 compiler for aicpu kernels")
    set(GYM_AICPU_TOOLCHAIN_RESOLVED TRUE CACHE INTERNAL "aicpu toolchain resolved")
    message(STATUS "aicpu toolchain: ${GYM_AICPU_ARM_CXX}")
endfunction()

set(GYM_AICPU_CXX_FLAGS
    -std=c++17 -O2 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -fstack-protector-all
    -D_GLIBCXX_USE_CXX11_ABI=1 -Dgoogle=ascend_private -D_FORTIFY_SOURCE=2)

set(GYM_AICPU_INCLUDES
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/external
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/external
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/aicpu
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/aicpu/context/common
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/aicpu
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/aicpu/common
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/aicpu/cpu_kernels
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/aicpu/aicpu_schedule
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/aicpu/queue_schedule
    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/aicpu/tsd)

set(GYM_AICPU_INCLUDE_FLAGS "")
foreach(_gym_aicpu_inc IN LISTS GYM_AICPU_INCLUDES)
    list(APPEND GYM_AICPU_INCLUDE_FLAGS -I${_gym_aicpu_inc})
endforeach()

# enable_pypto_aicpu_kernel(<op_file> [OP_TYPE <type>] [ARCH <arch>])
# 在算子的 op_kernel_aicpu/CMakeLists.txt 中调用一次; <op_file>.py 即 PyPTO AICPU kernel 源.
function(enable_pypto_aicpu_kernel op_file)
    cmake_parse_arguments(PAICPU "" "OP_TYPE;ARCH" "" ${ARGN})
    gym_resolve_aicpu_toolchain()
    if(NOT EXISTS ${GYM_AICPU_ARM_CXX})
        message(FATAL_ERROR "enable_pypto_aicpu_kernel: ARM cross compiler not found: ${GYM_AICPU_ARM_CXX}")
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
                "import os, pypto_pro.runtime.opc as opc; print(os.path.join(os.path.dirname(opc.__file__), 'pypto_aicpu_codegen_cli.py'))"
        OUTPUT_VARIABLE _gym_codegen_py
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _gym_codegen_rc
        ERROR_QUIET)
    if(NOT _gym_codegen_rc EQUAL 0 OR NOT EXISTS "${_gym_codegen_py}")
        message(FATAL_ERROR "enable_pypto_aicpu_kernel: pypto_pro aicpu codegen CLI not found")
    endif()

    set(_py ${CMAKE_CURRENT_SOURCE_DIR}/${op_file}.py)
    if(NOT EXISTS ${_py})
        message(FATAL_ERROR "enable_pypto_aicpu_kernel: kernel python file not found: ${_py}")
    endif()
    set(_gen ${CMAKE_BINARY_DIR}/aicpu_gen/${op_file})
    file(MAKE_DIRECTORY ${_gen})

    set(_cmd ${Python3_EXECUTABLE} ${_gym_codegen_py}
            --py-file ${_py} --out-dir ${_gen} --op-file ${op_file} --soc ${ASCEND_COMPUTE_UNIT})
    if(PAICPU_ARCH)
        list(APPEND _cmd --arch ${PAICPU_ARCH})
    endif()
    if(PAICPU_OP_TYPE)
        list(APPEND _cmd --op-type ${PAICPU_OP_TYPE})
    endif()
    message(STATUS "pypto aicpu codegen: ${op_file} -> ${_gen}")
    execute_process(COMMAND ${_cmd} RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "pypto aicpu codegen failed for ${op_file}\n${_err}\n${_out}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_py})

    # 编译/链接规则统一在顶层 finalize 创建: custom command 的输出仅在定义目录可见,
    # 而汇总 .so 在根作用域生成. 此处只登记 codegen 产物.
    set_property(GLOBAL APPEND PROPERTY GYM_AICPU_KERNELS "${_gen}|${_py}|${op_file}")
    set_property(GLOBAL APPEND PROPERTY GYM_AICPU_ONLY_OPS ${op_file})
    set_property(GLOBAL APPEND PROPERTY GYM_AICPU_JSONS ${_gen}/${op_file}_aicpu.json)
endfunction()

# gym_finalize_aicpu_kernels(): 所有算子接入后调用, 链接单一 AICPU .so 并合并 opInfo json.
function(gym_finalize_aicpu_kernels)
    get_property(_kernels GLOBAL PROPERTY GYM_AICPU_KERNELS)
    if(NOT _kernels)
        return()
    endif()
    gym_resolve_aicpu_toolchain()
    set(_objs "")
    foreach(_entry IN LISTS _kernels)
        string(REPLACE "|" ";" _parts "${_entry}")
        list(GET _parts 0 _gen)
        list(GET _parts 1 _py)
        list(GET _parts 2 _op_file)
        set(_obj ${_gen}/${_op_file}_aicpu_kernel.o)
        add_custom_command(OUTPUT ${_obj}
                COMMAND ${GYM_AICPU_ARM_CXX} -c ${_gen}/${_op_file}_aicpu_kernel.cpp -o ${_obj}
                        -I${_gen} ${GYM_AICPU_INCLUDE_FLAGS} ${GYM_AICPU_CXX_FLAGS}
                DEPENDS ${_gen}/${_op_file}_aicpu_kernel.cpp ${_py}
                COMMENT "Compiling PyPTO AICPU kernel ${_op_file} (aarch64)"
                VERBATIM)
        list(APPEND _objs ${_obj})
    endforeach()
    get_property(_jsons GLOBAL PROPERTY GYM_AICPU_JSONS)
    set(_lib_name lib${VENDOR_NAME}_aicpu_kernels.so)
    set(_so ${CMAKE_BINARY_DIR}/${_lib_name})

    if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ops_base/lib64/libaicpu_context.a)
        set(_aicpu_context ${ASCEND_CANN_PACKAGE_PATH}/ops_base/lib64/libaicpu_context.a)
    else()
        set(_aicpu_context ${ASCEND_CANN_PACKAGE_PATH}/lib64/libaicpu_context.a)
    endif()
    if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ops_base/lib64/libbase_ascend_protobuf.a)
        set(_protobuf ${ASCEND_CANN_PACKAGE_PATH}/ops_base/lib64/libbase_ascend_protobuf.a)
    else()
        set(_protobuf ${ASCEND_CANN_PACKAGE_PATH}/lib64/libbase_ascend_protobuf.a)
    endif()

    add_custom_command(OUTPUT ${_so}
            COMMAND ${GYM_AICPU_ARM_CXX} -shared ${_objs}
                    -Wl,--whole-archive ${_aicpu_context} ${_protobuf} -Wl,--no-whole-archive
                    -Wl,-Bsymbolic -Wl,--exclude-libs=libbase_ascend_protobuf.a -Wl,-z,now -s
                    -o ${_so}
            DEPENDS ${_objs}
            COMMENT "Linking ${_lib_name} (aarch64)"
            COMMAND_EXPAND_LISTS VERBATIM)
    add_custom_target(gym_aicpu_kernels ALL DEPENDS ${_so})

    set(_merged ${CMAKE_BINARY_DIR}/cust_aicpu_kernel.json)
    add_custom_command(OUTPUT ${_merged}
            COMMAND ${Python3_EXECUTABLE} ${PYPTO_GYM_SRC_DIR}/cmake/scripts/merge_aicpu_json.py
                    --output ${_merged} --kernel-so ${_lib_name} ${_jsons}
            DEPENDS ${_jsons} ${PYPTO_GYM_SRC_DIR}/cmake/scripts/merge_aicpu_json.py
            COMMENT "Merging AICPU opInfo json -> ${_merged}"
            COMMAND_EXPAND_LISTS VERBATIM)
    add_custom_target(gym_aicpu_json ALL DEPENDS ${_merged})

    install(FILES ${_so}
            DESTINATION ${GYM_VENDOR_DEST}/op_impl/cpu/aicpu_kernel/impl
            COMPONENT ${VENDOR_NAME})
    install(FILES ${_merged}
            DESTINATION ${GYM_VENDOR_DEST}/op_impl/cpu/config
            COMPONENT ${VENDOR_NAME})
endfunction()
