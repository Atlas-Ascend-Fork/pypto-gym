# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# gym 算子发现 + pypto 桥接 + kernel 编译目标, 机制对齐 ops-transformer cmake/func.cmake (裁剪版)

# 发现 ops/{pro,tensor} 下的算子目录(锚定算子级 CMakeLists.txt, 对齐 ops-transformer op_add_subdirectory);
# 算子契约: <op>/CMakeLists.txt(dispatcher) + <op>/op_host/(算子身份判定, 过滤中间层目录);
# 显式 --ops 时仅编名单内算子, 否则全量
function(gym_op_discovery OP_LIST OP_DIR_LIST)
    set(_OP_LIST)
    set(_OP_DIR_LIST)

    set(_explicit_ops)
    if(DEFINED ASCEND_OP_NAME AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL")
        set(_explicit_ops ${ASCEND_OP_NAME})
    endif()

    # gym 树形: ops/<paradigm>/<network|experimental|common>/<op>/ (实测本 cmake GLOB 的
    # '**' 仅单层, 故按算子深度 1~3 显式展开; 仅 glob pro/tensor 两棵树, 与 setup.py 白名单对账)
    file(GLOB _op_cmake_files
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/pro/*/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/pro/*/*/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/pro/*/*/*/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/tensor/*/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/tensor/*/*/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/tensor/*/*/*/CMakeLists.txt")

    foreach(_cmake_file ${_op_cmake_files})
        get_filename_component(_op_dir "${_cmake_file}" DIRECTORY)

        # 算子身份判定: 必须有 op_host 子目录(过滤 experimental/ 等中间层目录)
        if(NOT EXISTS "${_op_dir}/op_host")
            continue()
        endif()
        get_filename_component(_op_name "${_op_dir}" NAME)

        # 白名单过滤 (与 setup.py _PARADIGM_DIRS 语义对账: 仅 pro/tensor 两棵树被 glob, 历史目录天然不参与)
        if(_explicit_ops AND NOT _op_name IN_LIST _explicit_ops)
            continue()
        endif()

        list(APPEND _OP_LIST ${_op_name})
        list(APPEND _OP_DIR_LIST ${_op_dir})
    endforeach()

    # 孤儿检测: 有 op_host 但缺算子级 CMakeLists(dispatcher)的算子不可见, 违约即报错
    file(GLOB _orphan_op_host_files
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/pro/*/op_host/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/pro/*/*/op_host/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/pro/*/*/*/op_host/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/tensor/*/op_host/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/tensor/*/*/op_host/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/tensor/*/*/*/op_host/CMakeLists.txt")
    foreach(_orphan_file ${_orphan_op_host_files})
        get_filename_component(_orphan_op_host_dir "${_orphan_file}" DIRECTORY)
        get_filename_component(_orphan_op_dir "${_orphan_op_host_dir}" DIRECTORY)
        if(NOT EXISTS "${_orphan_op_dir}/CMakeLists.txt")
            get_filename_component(_orphan_op_name "${_orphan_op_dir}" NAME)
            if(NOT _explicit_ops OR _orphan_op_name IN_LIST _explicit_ops)
                message(FATAL_ERROR "operator '${_orphan_op_name}' has op_host but is missing "
                        "${_orphan_op_dir}/CMakeLists.txt (op-level dispatcher required; "
                        "see flex_attention/CMakeLists.txt for the template)")
            endif()
        endif()
    endforeach()

    list(REMOVE_DUPLICATES _OP_LIST)
    list(REMOVE_DUPLICATES _OP_DIR_LIST)
    list(SORT _OP_LIST)
    list(SORT _OP_DIR_LIST)
    if(NOT _OP_LIST)
        message(FATAL_ERROR "gym_op_discovery: no operator found under ops/{pro,tensor} (explicit ops: '${_explicit_ops}')")
    endif()
    message(STATUS "gym ops discovered: ${_OP_LIST}")
    set(${OP_LIST} ${_OP_LIST} PARENT_SCOPE)
    set(${OP_DIR_LIST} ${_OP_DIR_LIST} PARENT_SCOPE)
endfunction()

macro(add_op_to_compiled_list)
    set(COMPILED_OPS ${COMPILED_OPS} ${OP_NAME} CACHE STRING "Compiled Ops" FORCE)
    set(COMPILED_OP_DIRS ${COMPILED_OP_DIRS} ${PARENT_DIR} CACHE STRING "Compiled Ops Dirs" FORCE)
endmacro()

macro(require_pypto_pro op_name)
    # 获取 pypto_pro 可用状态
    get_property(_pypto_pro_checked GLOBAL PROPERTY PYPTO_PRO_CHECKED)
    if (NOT _pypto_pro_checked)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c
                    "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pypto_pro') else 1)"
            RESULT_VARIABLE _pypto_pro_check_result
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if (_pypto_pro_check_result EQUAL 0)
            set_property(GLOBAL PROPERTY PYPTO_PRO_AVAILABLE TRUE)
        else()
            set_property(GLOBAL PROPERTY PYPTO_PRO_AVAILABLE FALSE)
        endif()
        set_property(GLOBAL PROPERTY PYPTO_PRO_CHECKED TRUE)
    endif()

    get_property(_pypto_pro_available GLOBAL PROPERTY PYPTO_PRO_AVAILABLE)
    if (NOT _pypto_pro_available)
        message(FATAL_ERROR "pypto_pro is unavailable in ${Python3_EXECUTABLE}; operator ${op_name} requires it.")
    endif()

    unset(_pypto_pro_checked)
    unset(_pypto_pro_check_result)
    unset(_pypto_pro_available)
endmacro()

# enable_pypto_kernel(<op_file>): configure 期跑 pypto codegen 生成 tiling/tilingkey 头
function(enable_pypto_kernel op_file)
    set(_compute_units ${ASCEND_COMPUTE_UNIT})
    list(FILTER _compute_units INCLUDE REGEX ".+")
    if (NOT _compute_units)
        message(FATAL_ERROR "enable_pypto_kernel: ASCEND_COMPUTE_UNIT is empty")
    endif ()
    list(GET _compute_units 0 _soc_unit)

    set(_gen "${ASCEND_BINARY_OUT_DIR}/${_soc_unit}/${op_file}_pypto_gen")
    set(_py  "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/${op_file}.py")
    if (NOT EXISTS ${_py})
        message(FATAL_ERROR "enable_pypto_kernel: kernel python file not found: ${_py}")
    endif ()

    message(STATUS "pypto codegen: ${op_file} -> ${_gen}")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PYPTO_GYM_SRC_DIR}/cmake/scripts/pypto_codegen.py
                --py-file "${_py}" --out-dir "${_gen}" --op-file "${op_file}" --soc "${_soc_unit}"
        RESULT_VARIABLE _rc)
    if (NOT _rc EQUAL 0)
        message(FATAL_ERROR "pypto codegen failed for ${op_file} (exit ${_rc})")
    endif ()

    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_py})
    set(PYPTO_GEN_DIR ${_gen} PARENT_SCOPE)
    set(${op_file}_pypto_enabled TRUE    CACHE INTERNAL "pypto-enabled kernel ${op_file}")
    set(${op_file}_pypto_gen_dir ${_gen} CACHE INTERNAL "pypto gen dir for ${op_file}")
    set_property(GLOBAL APPEND PROPERTY PYPTO_ENABLED_OPS ${op_file})
endfunction()

function(add_ops_compile_options)
    cmake_parse_arguments(OP_COMPILE "" "OP_NAME" "COMPUTE_UNIT;OPTIONS" ${ARGN})
    if(NOT OP_COMPILE_OPTIONS)
        return()
    endif()
    execute_process(COMMAND ${Python3_EXECUTABLE} ${GYM_UTIL_SCRIPT_DIR}/ascendc_gen_options.py
            ${ASCEND_CUSTOM_OPTIONS} ${OP_COMPILE_OP_NAME}
            ${OP_COMPILE_COMPUTE_UNIT} ${OP_COMPILE_OPTIONS}
            RESULT_VARIABLE EXEC_RESULT OUTPUT_QUIET ERROR_QUIET)
    if (EXEC_RESULT)
        message(FATAL_ERROR "Error: add ops compile options failed for ${OP_COMPILE_OP_NAME}!")
    endif ()
endfunction()

function(add_ops_tiling_keys)
    cmake_parse_arguments(OP_COMPILE "" "OP_NAME" "COMPUTE_UNIT" "TILING_KEYS" ${ARGN})
    if(NOT OP_COMPILE_TILING_KEYS)
        return()
    endif()
    list(JOIN OP_COMPILE_TILING_KEYS "," STRING_TILING_KEYS)
    add_ops_compile_options(OP_NAME ${OP_COMPILE_OP_NAME} OPTIONS --tiling_key=${STRING_TILING_KEYS})
endfunction()

function(add_compile_cmd_target)
    cmake_parse_arguments(CMD "" "COMPUTE_UNIT" "" ${ARGN})

    set(OP_DEBUG_CONFIG_OPTION --opc-config-file ${ASCEND_CUSTOM_OPC_OPTIONS})
    set(OP_TILING_KEY_OPTION --kernel_template_input)

    set(_OUT_DIR    ${ASCEND_BINARY_OUT_DIR}/${CMD_COMPUTE_UNIT})
    set(GEN_OUT_DIR ${_OUT_DIR}/gen)
    set(COMPILE_CMD_TARGET generate_compile_cmd_${CMD_COMPUTE_UNIT})

    add_custom_target(${COMPILE_CMD_TARGET} ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN_OUT_DIR}
        COMMAND ${Python3_EXECUTABLE} ${GYM_UTIL_SCRIPT_DIR}/ascendc_bin_param_build.py
            ${ASCEND_AUTOGEN_DIR}/aic-${CMD_COMPUTE_UNIT}-ops-info.ini
            ${GEN_OUT_DIR} ${CMD_COMPUTE_UNIT} ${OP_TILING_KEY_OPTION} ${OP_DEBUG_CONFIG_OPTION}
        COMMAND ${Python3_EXECUTABLE} ${GYM_UTIL_SCRIPT_DIR}/ascendc_bin_param_build.py
            ${ASCEND_AUTOGEN_DIR}/inner/aic-${CMD_COMPUTE_UNIT}-ops-info.ini
            ${GEN_OUT_DIR} ${CMD_COMPUTE_UNIT} ${OP_TILING_KEY_OPTION} ${OP_DEBUG_CONFIG_OPTION}
        COMMAND ${Python3_EXECUTABLE} ${GYM_UTIL_SCRIPT_DIR}/ascendc_bin_param_build.py
            ${ASCEND_AUTOGEN_DIR}/exc/aic-${CMD_COMPUTE_UNIT}-ops-info.ini
            ${GEN_OUT_DIR} ${CMD_COMPUTE_UNIT} ${OP_TILING_KEY_OPTION} ${OP_DEBUG_CONFIG_OPTION}
    )
    add_dependencies(${COMPILE_CMD_TARGET} opbuild_gen_default opbuild_gen_inner opbuild_gen_exc)
    add_dependencies(generate_compile_cmd ${COMPILE_CMD_TARGET})
endfunction()

# 源码 staging: 整目录拷贝(与 ops-transformer add_ops_src_copy 语义一致) + stamp
function(add_ops_src_copy)
    cmake_parse_arguments(SRC_COPY "" "TARGET_NAME;SRC;DST;COMPUTE_UNIT" "" ${ARGN})
    set(_stamp ${SRC_COPY_DST}/.${SRC_COPY_TARGET_NAME}.done)
    add_custom_command(OUTPUT ${_stamp}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${SRC_COPY_SRC} ${SRC_COPY_DST}
        COMMAND ${CMAKE_COMMAND} -E touch ${_stamp}
        DEPENDS ${SRC_COPY_SRC}
        VERBATIM)
    add_custom_target(${SRC_COPY_TARGET_NAME} ALL DEPENDS ${_stamp})
endfunction()

# kernel 二进制编译目标: pypto 算子按 ground-truth 命令链执行 opc (asc_opc)
function(add_bin_compile_target)
    cmake_parse_arguments(BINARY "" "COMPUTE_UNIT" "OP_INFO" ${ARGN})

    set(_OUT_DIR ${ASCEND_BINARY_OUT_DIR}/${BINARY_COMPUTE_UNIT})
    set(BIN_OUT_DIR ${_OUT_DIR}/bin)
    set(GEN_OUT_DIR ${_OUT_DIR}/gen)
    set(SRC_OUT_DIR ${_OUT_DIR}/src)
    file(MAKE_DIRECTORY ${BIN_OUT_DIR})

    foreach(_op_info ${BINARY_OP_INFO})
        get_filename_component(_op_name "${_op_info}" NAME)
        set(${_op_name}_dir ${_op_info})
    endforeach()

    set(_ops_target_list)
    file(GLOB compile_scripts ${GEN_OUT_DIR}/*.sh)

    foreach(bin_script ${compile_scripts})
        get_filename_component(bin_file ${bin_script} NAME_WE)
        string(REPLACE "-" ";" bin_sep ${bin_file})
        list(GET bin_sep 0 op_type)
        list(GET bin_sep 1 op_file)
        list(GET bin_sep 2 op_index)

        if (NOT DEFINED ${op_file}_dir)
            continue()
        endif ()

        set(OP_TARGET_NAME ${op_file}_${BINARY_COMPUTE_UNIT})
        if (TARGET ${OP_TARGET_NAME})
            # 同算子多 dtype 脚本共享一组 staging 目标
        else ()
            if (NOT ${op_file}_pypto_enabled)
                message(FATAL_ERROR
                    "op '${op_file}' is not pypto-enabled: gym currently supports PyPTO kernels only")
            endif ()
            add_custom_target(${OP_TARGET_NAME})
            add_dependencies(ops_gym_kernel ${OP_TARGET_NAME})
            list(APPEND _ops_target_list ${OP_TARGET_NAME})

            set(OP_SRC_OUT_DIR ${SRC_OUT_DIR}/${op_file})
            set(OP_BIN_OUT_DIR ${BIN_OUT_DIR}/${op_file})
            file(MAKE_DIRECTORY ${OP_SRC_OUT_DIR})

            # vendors staging: kernel 二进制与配置(COMPONENT 供 CPack 收集)
            set(_kernel_install_dir packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/kernel)
            install(DIRECTORY ${OP_BIN_OUT_DIR}
                    DESTINATION ${_kernel_install_dir}/${BINARY_COMPUTE_UNIT} OPTIONAL
                    COMPONENT ${VENDOR_NAME})
            install(FILES ${BIN_OUT_DIR}/${op_file}.json
                    DESTINATION ${_kernel_install_dir}/config/${BINARY_COMPUTE_UNIT} OPTIONAL
                    COMPONENT ${VENDOR_NAME})

            add_ops_src_copy(
                TARGET_NAME ${OP_TARGET_NAME}_src_copy
                SRC ${${op_file}_dir}
                DST ${OP_SRC_OUT_DIR}
                COMPUTE_UNIT ${BINARY_COMPUTE_UNIT})

            set(_pypto_install_stamp ${OP_SRC_OUT_DIR}/op_kernel/.pypto_install.done)
            add_custom_command(OUTPUT ${_pypto_install_stamp}
                    COMMAND ${CMAKE_COMMAND} -E copy_directory ${${op_file}_pypto_gen_dir} ${OP_SRC_OUT_DIR}/op_kernel
                    COMMAND ${CMAKE_COMMAND} -E touch ${_pypto_install_stamp}
                    DEPENDS ${OP_TARGET_NAME}_src_copy
                    VERBATIM)
            add_custom_target(${OP_TARGET_NAME}_pypto_install DEPENDS ${_pypto_install_stamp})

            set(DYNAMIC_PY_FILE ${OP_SRC_OUT_DIR}/${op_type}.py)
            add_custom_command(OUTPUT ${DYNAMIC_PY_FILE}
                    COMMAND cp -rf ${ASCEND_IMPL_OUT_DIR}/dynamic/${op_file}.py ${DYNAMIC_PY_FILE})
            add_custom_target(${OP_TARGET_NAME}_py_copy DEPENDS ${DYNAMIC_PY_FILE})

            add_custom_command(OUTPUT ${OP_BIN_OUT_DIR} COMMAND mkdir -p ${OP_BIN_OUT_DIR})
            add_custom_target(${OP_TARGET_NAME}_mkdir DEPENDS ${OP_BIN_OUT_DIR})
        endif ()

        set(_BUILD_FLAG ${GEN_OUT_DIR}/${OP_TARGET_NAME}_${op_index}.done)
        # TILINGKEY_PAR_COMPILE/BIN_FILENAME_HASHED 由 CANN 编译工具链消费
        # (BIN_FILENAME_HASHED=1 即 <OpType>_<md5>.o 哈希文件名来源, ascendc_common_utility.py)
        set(_BUILD_COMMAND export TILINGKEY_PAR_COMPILE=1 &&)
        list(APPEND _BUILD_COMMAND export BIN_FILENAME_HASHED=1 &&)
        list(APPEND _BUILD_COMMAND bash ${bin_script} ${SRC_OUT_DIR}/${op_file}/${op_type}.py ${OP_BIN_OUT_DIR})
        if(CMAKE_GENERATOR MATCHES "Unix Makefiles")
            list(APPEND _BUILD_COMMAND && echo $(MAKE))
        endif()

        add_custom_command(OUTPUT ${_BUILD_FLAG}
                COMMAND ${_BUILD_COMMAND}
                COMMAND touch ${_BUILD_FLAG}
                WORKING_DIRECTORY ${GEN_OUT_DIR})
        add_custom_target(${OP_TARGET_NAME}_${op_index} DEPENDS ${_BUILD_FLAG})
        add_dependencies(${OP_TARGET_NAME}_${op_index}
                ${OP_TARGET_NAME}_src_copy ${OP_TARGET_NAME}_py_copy ${OP_TARGET_NAME}_mkdir
                ${OP_TARGET_NAME}_pypto_install generate_compile_cmd)
        add_dependencies(${OP_TARGET_NAME} ${OP_TARGET_NAME}_${op_index})
    endforeach()

    if (_ops_target_list)
        set(OPS_CONFIG_TARGET ops_config_${BINARY_COMPUTE_UNIT})
        set(BINARY_INFO_CONFIG_FILE ${BIN_OUT_DIR}/binary_info_config.json)
        add_custom_command(OUTPUT ${BINARY_INFO_CONFIG_FILE}
                COMMAND ${Python3_EXECUTABLE} ${GYM_UTIL_SCRIPT_DIR}/ascendc_ops_config.py -p ${BIN_OUT_DIR} -s ${BINARY_COMPUTE_UNIT})
        add_custom_target(${OPS_CONFIG_TARGET} DEPENDS ${BINARY_INFO_CONFIG_FILE})
        add_dependencies(ops_gym_config ${OPS_CONFIG_TARGET})
        foreach(_op_target ${_ops_target_list})
            add_dependencies(${OPS_CONFIG_TARGET} ${_op_target})
        endforeach()
        set(_kernel_cfg_dir packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/kernel/config/${BINARY_COMPUTE_UNIT})
        install(FILES ${BINARY_INFO_CONFIG_FILE} DESTINATION ${_kernel_cfg_dir} OPTIONAL
                COMPONENT ${VENDOR_NAME})
        install(FILES ${BIN_OUT_DIR}/relocatable_kernel_info_config.json
                DESTINATION ${_kernel_cfg_dir} OPTIONAL COMPONENT ${VENDOR_NAME})
    endif ()
endfunction()
