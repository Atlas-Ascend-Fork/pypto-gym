# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#[[
构建公共配置, 按四阶段组织:
    1 环境检查
        fail-fast 检查外部依赖(工具链/CANN 包/仓自身), 含就近的路径派生;
    2 CMake 选项缺省  构建系统自身配置的缺省补全(CMAKE_* 系);
    3 入参校验与规范化 gym 业务入参(-D 值)的合法性与规范化;
    4 预处理         构建前置动作(输出目录/ini 初始化/prepare.sh 二次构建).

用户可感知的 option 一律声明在根 CMakeLists.txt, 本文件只做内部变量与检查.
]]

########################################################################################################################
# 环境检查
########################################################################################################################

# 系统名称前缀(内部变量, 供 CANN 包内路径拼接)
if (UNIX)
    set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)
endif()

# Python3 解析
# 1. Python3_EXECUTABLE 用以标识 Python3 可执行文件路径, -D 指定的优先级高于环境变量;
# 2. 刷新 CMake 内部变量 Python3_ROOT_DIR 值, 以便后续 find python 相关 package 使用(主要是 Development 组件);
if (DEFINED Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${Python3_EXECUTABLE}")  # 不可直接获取绝对路径, 避免不同 executable 权限差异导致的 pip 包查找问题
elseif (DEFINED ENV{Python3_EXECUTABLE})
    set(Python3_EXECUTABLE "$ENV{Python3_EXECUTABLE}")  # 不可直接获取绝对路径, 避免不同 executable 权限差异导致的 pip 包查找问题
else ()
    # 当外部未指定 Python3 时, 一般是从 CMake 为入口触发的编译, 此时直接 find.
    find_package(Python3 COMPONENTS Interpreter)
    if ("${Python3_EXECUTABLE}x" STREQUAL "x")
        message(FATAL_ERROR "Can't find python3 Interpreter.")
    endif ()
endif ()
message(STATUS "Python3_EXECUTABLE=${Python3_EXECUTABLE}")
if (NOT (DEFINED Python3_ROOT_DIR OR DEFINED ENV{Python3_ROOT_DIR}))
    get_filename_component(_Python3__EXECUTABLE "${Python3_EXECUTABLE}"     REALPATH)
    get_filename_component(_Python3_ROOT_DIR    "${_Python3__EXECUTABLE}"   DIRECTORY)
    get_filename_component(_Python3_ROOT_DIR    "${_Python3_ROOT_DIR}/../"  REALPATH)
    set(Python3_ROOT_DIR "${_Python3_ROOT_DIR}" CACHE INTERNAL "Python3 root path" FORCE)
    message(STATUS "Python3_ROOT_DIR=${Python3_ROOT_DIR}")
endif ()

# CANN 包路径解析 (优先级 -D 覆盖 > 环境变量 > 默认路径)
set(ASCEND_CANN_PACKAGE_PATH)
if (CUSTOM_ASCEND_CANN_PACKAGE_PATH)
    get_filename_component(ASCEND_CANN_PACKAGE_PATH "${CUSTOM_ASCEND_CANN_PACKAGE_PATH}" REALPATH)
elseif (DEFINED ENV{ASCEND_HOME_PATH})
    get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" REALPATH)
else ()
    get_filename_component(ASCEND_CANN_PACKAGE_PATH  "/usr/local/Ascend/cann/" REALPATH)
endif ()
message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")

# CANN 包完整性: op_build 工具存在性
set(OP_BUILD_TOOL ${ASCEND_CANN_PACKAGE_PATH}/tools/opbuild/op_build CACHE STRING "op_build tool")
if (NOT EXISTS ${OP_BUILD_TOOL})
    message(FATAL_ERROR "op_build tool not found: ${OP_BUILD_TOOL} (requires CANN >= 9.2.0)")
endif()

# CANN 包完整性: tikcpp 目录(内部变量派生)
if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/tikcpp/ascendc_kernel_cmake)
    set(ASCEND_PROJECT_DIR ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/tikcpp/ascendc_kernel_cmake)
endif()

# 仓自身 util 脚本目录(内部变量; gym 自有脚本, 非全部展开以避免路径重复)
set(GYM_UTIL_SCRIPT_DIR ${PYPTO_GYM_SRC_DIR}/cmake/scripts/util)

# 工具链前置检查(fail-fast): bisheng/asc_opc 由 opc gen 脚本消费,
# 在 configure 期检查保证任意入口(build_ci.py / 手工 cmake)均得到相同保护.
# PREPARE_BUILD 二次 configure 会重复执行, 检查幂等, 仅以 guard 避免重复日志.
if (NOT PREPARE_BUILD)
    foreach(_gym_tool bisheng asc_opc)
        find_program(_gym_tool_path ${_gym_tool})
        if(NOT _gym_tool_path)
            message(FATAL_ERROR "'${_gym_tool}' not found in PATH (source CANN set_env.sh first)")
        endif()
        unset(_gym_tool_path CACHE)
    endforeach()
endif()

########################################################################################################################
# 2 CMake 选项, 缺省参数设置 (构建系统自身配置)
#   按 CMake 构建过程对 CMake 选项, CMake 缺省参数进行配置
#   CMake 构建过程: 1) 配置阶段(Configure); 2) 构建阶段(Build); 3) 安装阶段(Install);
########################################################################################################################

# 构建阶段(Build)
#   构建类型
#       CMake中的Generator(生成器)是用于生成本地/本机构建系统的工具. 一般分为两种:
#       1. 单配置生成器(Single-configuration generator):
#          在配置(Configuration)阶段, 仅允许指定一种构建类型, 通过变量 CMAKE_BUILD_TYPE 指定;
#          在构建阶段(Build)无法更改构建类型, 仅允许使用配置(Configuration)阶段通过变量 CMAKE_BUILD_TYPE 指定的构建类型;
#          常见的此类型生成器有: Ninja, Unix Makefiles
#       2. 多配置生成器(Multi-configuration generator) :
#          在配置(Configuration)阶段, 仅指定构建阶段(Build)可用的构建类型列表, 通过变量 CMAKE_CONFIGURATION_TYPES 指定;
#          在构建阶段(Build)通过 "--config" 参数, 指定构建阶段具体的构建类型;
#          常见的此类型生成器有: Xcode, Visual Studio
#       所以:
#           1. 单配置生成器(Single-configuration generator)场景下, 如果构建类型(CMAKE_BUILD_TYPE)未指定, 则默认为 Debug ;
#           2. 多配置生成器(Multi-configuration generator)场景下, 如果构建阶段可选的构建类型(CMAKE_CONFIGURATION_TYPES)未指定,
#              则默认将其指定为CMake允许的构建类型全集 [Debug;Release;MinSizeRel;RelWithDebInfo]
message(STATUS "CMAKE_GENERATOR=${CMAKE_GENERATOR}")
get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (GENERATOR_IS_MULTI_CONFIG)
    if (NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release;MinSizeRel;RelWithDebInfo" CACHE STRING "Configuration Build type" FORCE)
    endif ()
else ()
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type(default Release)" FORCE)
    endif ()
    message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif ()

# 构建阶段(Build)
#   可执行文件运行时库文件搜索路径 RPATH
#       在 UTest 及 STest 场景不略去 RPATH (预留: ENABLE_TESTS 选项落地时生效)
if (ENABLE_TESTS)
    set(CMAKE_SKIP_RPATH FALSE)
else ()
    set(CMAKE_SKIP_RPATH TRUE)
endif ()

# 构建阶段(Build)
#   语言标准
set(CMAKE_C_STANDARD 11)  # 指定 C 语言使用 ISO C11 标准
set(CMAKE_C_STANDARD_REQUIRED ON)  # 要求严格支持 C11 标准
set(CMAKE_C_EXTENSIONS OFF)  # 禁用 C 编译器扩展, 使用纯 ISO C 标准
set(CMAKE_CXX_STANDARD 17)  # 指定 C++ 语言使用 ISO C++17 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # 要求严格支持 C++17 标准
set(CMAKE_CXX_EXTENSIONS OFF)  # 禁用 C++ 编译器扩展, 使用纯 ISO C++17 标准

# 构建阶段(Build)
#   构建选项
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")

# 构建阶段(Build)
#   CCACHE 配置
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM} CACHE PATH "C cache Compiler")
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM} CACHE PATH "CXX cache Compiler")
    message(STATUS "Use ccache, CCACHE_BASEDIR=$ENV{CCACHE_BASEDIR}")
else ()
    message(STATUS "ccache not found.")
endif ()

# 安装阶段(Install)
#   安装路径
#       未显示设置 CMAKE_INSTALL_PREFIX (即 CMAKE_INSTALL_PREFIX 取缺省值)时, 设置与构建树根目录 CMAKE_CURRENT_BINARY_DIR 平级
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    get_filename_component(_Install_Path_Prefix "${CMAKE_CURRENT_BINARY_DIR}/../build_out" REALPATH)
    set(CMAKE_INSTALL_PREFIX    "${_Install_Path_Prefix}"  CACHE STRING "Install path" FORCE)
endif ()
message(STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")


########################################################################################################################
# 3 入参校验与规范化 (gym 业务入参)
########################################################################################################################

# 单 soc 契约: 一次 cmake 调用只处理一个 soc, 多 soc 由外部按 soc 循环驱动(build_ci.py).
# 此处强制契约: 非空、且不含列表分隔符(;/,) —— 违约即报错, 避免编译多 soc 而打包只取其一的不一致状态.
if (NOT ASCEND_COMPUTE_UNIT)
    message(FATAL_ERROR "ASCEND_COMPUTE_UNIT must be a non-empty soc name (e.g. ascend950)")
endif()
if (ASCEND_COMPUTE_UNIT MATCHES "[;,]")
    message(FATAL_ERROR "ASCEND_COMPUTE_UNIT must be a single soc, got '${ASCEND_COMPUTE_UNIT}'; "
            "multi-soc is driven by the external per-soc build loop (build_ci.py)")
endif()

# 当前仅支持 ascend950; soc -> opc soc / pypto arch 映射见 cmake/scripts/util/const_var.py
set(GYM_SUPPORTED_SOC ascend950)
if (NOT ASCEND_COMPUTE_UNIT IN_LIST GYM_SUPPORTED_SOC)
    message(FATAL_ERROR "unsupported soc '${ASCEND_COMPUTE_UNIT}': gym currently supports ${GYM_SUPPORTED_SOC} only")
endif()

# 算子名单规范化: 逗号分隔 -> CMake list(供 gym_op_discovery 白名单过滤与 prepare.sh 转发)
string(REPLACE "," ";" ASCEND_OP_NAME "${ASCEND_OP_NAME}")


########################################################################################################################
# 4 预处理 (构建前置动作)
########################################################################################################################

# [预留能力·原样保留] GCOV 编译器守卫(ENABLE_GCOV 选项落地时生效)
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    if (ENABLE_GCOV)
        set(ENABLE_GCOV OFF)
        message(WARNING "GCov only supported in GNU Compiler, Current Compiler is ${CMAKE_C_COMPILER_ID}, Auto turn off it.")
    endif ()
endif ()

# [预留能力·原样保留] ASAN/UBSAN 用例执行环境配置(ENABLE_TESTS_EXECUTE 选项落地时生效).
# 编译执行用例场景下, 将相关检查在编译前执行, 避免出现编译完成后又无法执行的情况, 影响使用体验.
if ((ENABLE_ASAN OR ENABLE_UBSAN) AND ENABLE_TESTS_EXECUTE)
    # 用于向外传递 XSAN 相关配置, 以保证 whl 包 ASan 场景的兼容性
    set(XSAN_CONFIG_FILE "${PYPTO_GYM_SRC_DIR}/_pypto_gym_xsan_config.txt")
    file(WRITE "${XSAN_CONFIG_FILE}" "")

    # LD_PRELOAD, 仅 GNU 编译器需要设置
    set(XSAN_LD_PRELOAD)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if (ENABLE_ASAN)
            # libasan.so
            execute_process(COMMAND ${CMAKE_C_COMPILER} --print-file-name=libasan.so
                    RESULT_VARIABLE _RST
                    OUTPUT_VARIABLE ASAN_SHARED_PATH)
            if (_RST)
                message(FATAL_ERROR "Can't get libasan.so path with ${CMAKE_C_COMPILER}")
            endif ()
            get_filename_component(ASAN_SHARED_PATH "${ASAN_SHARED_PATH}" DIRECTORY)
            get_filename_component(ASAN_SHARED_PATH "${ASAN_SHARED_PATH}/libasan.so" REALPATH)
            if (NOT EXISTS ${ASAN_SHARED_PATH})
                message(FATAL_ERROR
                        "ASAN_SHARED_PATH=${ASAN_SHARED_PATH} not exist. Please check the completeness of the compiler installation.")
            endif ()
            list(APPEND XSAN_LD_PRELOAD ${ASAN_SHARED_PATH})
        endif ()
        if (ENABLE_UBSAN)
            # libubsan.so
            execute_process(COMMAND ${CMAKE_C_COMPILER} --print-file-name=libubsan.so
                    RESULT_VARIABLE _RST
                    OUTPUT_VARIABLE UBSAN_SHARED_PATH)
            if (_RST)
                message(FATAL_ERROR "Can't get libubsan.so path with ${CMAKE_C_COMPILER}")
            endif ()
            get_filename_component(UBSAN_SHARED_PATH "${UBSAN_SHARED_PATH}" DIRECTORY)
            get_filename_component(UBSAN_SHARED_PATH "${UBSAN_SHARED_PATH}/libubsan.so" REALPATH)
            if (NOT EXISTS ${UBSAN_SHARED_PATH})
                message(FATAL_ERROR
                        "UBSAN_SHARED_PATH=${UBSAN_SHARED_PATH} not exist. Please check the completeness of the compiler installation.")
            endif ()
            list(APPEND XSAN_LD_PRELOAD ${UBSAN_SHARED_PATH})
        endif ()
        # libstdc++.so
        execute_process(COMMAND ${CMAKE_C_COMPILER} --print-file-name=libstdc++.so
                RESULT_VARIABLE _RST
                OUTPUT_VARIABLE STDC_SHARED_PATH)
        if (_RST)
            message(FATAL_ERROR "Can't get libstdc++.so path with ${CMAKE_C_COMPILER}")
        endif ()
        get_filename_component(STDC_SHARED_PATH "${STDC_SHARED_PATH}" DIRECTORY)
        get_filename_component(STDC_SHARED_PATH "${STDC_SHARED_PATH}/libstdc++.so" REALPATH)
        if (NOT EXISTS ${STDC_SHARED_PATH})
            message(FATAL_ERROR
                    "STDC_SHARED_PATH=${STDC_SHARED_PATH} not exist. Please check the completeness of the compiler installation.")
        endif ()
        list(APPEND XSAN_LD_PRELOAD ${STDC_SHARED_PATH})
        # 结果修正
        string(REPLACE ";" ":" XSAN_LD_PRELOAD "${XSAN_LD_PRELOAD}")
        set(XSAN_LD_PRELOAD "LD_PRELOAD=${XSAN_LD_PRELOAD}")
    endif ()
    if (XSAN_LD_PRELOAD)
        file(APPEND "${XSAN_CONFIG_FILE}" "${XSAN_LD_PRELOAD}\n")
    endif ()

    set(ASAN_OPTIONS)
    if (ENABLE_ASAN)
        # 谨慎修改 ASAN_OPTIONS 取值, 当前出现告警会使 GTest 失败.
        # halt_on_error=1, 出现告警时停止运行进而触发构建失败, 避免主进程或 fork 出的子进程出现错误无法发现的情况
        # detect_stack_use_after_return=1, 栈空间返回后使用检测
        # check_initialization_order, 尝试捕获初始化顺序问题
        # strict_init_order, 动态初始化器永远不能访问来自其他模块的全局变量, 及时或者已经初始化
        # strict_string_checks, 检查字符串参数是否正确以 null 终止
        # detect_leaks=1, 内存泄漏检测
        set(ASAN_OPTIONS "ASAN_OPTIONS=halt_on_error=1,detect_stack_use_after_return=1,check_initialization_order=1,strict_init_order=1,strict_string_checks=1,symbolize=1,detect_leaks=1")
        if (ENABLE_FEATURE_PYTHON_FRONT_END)
            set(LSAN_OPTIONS "LSAN_OPTIONS=suppressions=${PTO_FWK_SRC_ROOT}/cmake/asan_suppressions.txt")
            file(APPEND "${XSAN_CONFIG_FILE}" "${LSAN_OPTIONS}\n")
        endif ()
    endif ()
    if (ASAN_OPTIONS)
        file(APPEND "${XSAN_CONFIG_FILE}" "${ASAN_OPTIONS}\n")
    endif ()

    set(UBSAN_OPTIONS)
    if (ENABLE_UBSAN)
        # 谨慎修改 UBSAN_OPTIONS 取值, 当前出现告警会使 UT 失败.
        # halt_on_error=1, 出现告警时停止运行进而触发构建失败, 避免主进程或 fork 出的子进程出现错误无法发现的情况
        # print_stacktrace=1, 出错时打印调用栈
        set(UBSAN_OPTIONS "UBSAN_OPTIONS=halt_on_error=0,print_stacktrace=1")
    endif ()
    if (UBSAN_OPTIONS)
        file(APPEND "${XSAN_CONFIG_FILE}" "${UBSAN_OPTIONS}\n")
    endif ()
endif ()

# 构建树输出目录(prepare.sh 的直接输入)与自定义 opp 目录
set(ASCEND_IMPL_OUT_DIR   ${CMAKE_BINARY_DIR}/impl    CACHE STRING "adapt py output directories")
set(ASCEND_BINARY_OUT_DIR ${CMAKE_BINARY_DIR}/binary  CACHE STRING "kernel binary output directories")
set(ASCEND_AUTOGEN_DIR    ${CMAKE_BINARY_DIR}/autogen CACHE STRING "op_build autogen directories")
set(ASCEND_CUSTOM_OPTIONS     ${ASCEND_AUTOGEN_DIR}/custom_compile_options.ini)
set(ASCEND_CUSTOM_OPC_OPTIONS ${ASCEND_AUTOGEN_DIR}/custom_opc_options.ini)
file(MAKE_DIRECTORY ${ASCEND_AUTOGEN_DIR})
foreach(_ini ${ASCEND_CUSTOM_OPTIONS} ${ASCEND_CUSTOM_OPC_OPTIONS})
    file(REMOVE ${_ini})
    file(TOUCH ${_ini})
endforeach()

# 二次构建机制: 主 configure 期先跑 prepare.sh(PREPARE_BUILD=ON 的第二次 cmake + make prepare_build),
# 产出 autogen(ops-info.ini/aclnn 源) 与 impl(adapt py), 供主构建图消费
if (NOT PREPARE_BUILD AND ENABLE_OPS_KERNEL)
    string(REPLACE ";" "::" EP_ASCEND_OP_NAME "${ASCEND_OP_NAME}")
    execute_process(COMMAND bash ${PYPTO_GYM_SRC_DIR}/cmake/scripts/prepare.sh
            -s ${CMAKE_CURRENT_SOURCE_DIR}
            -b ${CMAKE_BINARY_DIR}/prepare_build
            -p ${ASCEND_CANN_PACKAGE_PATH}
            --autogen-dir ${ASCEND_AUTOGEN_DIR}
            --binary-out-dir ${ASCEND_BINARY_OUT_DIR}
            --impl-out-dir ${ASCEND_IMPL_OUT_DIR}
            --op-build-tool ${OP_BUILD_TOOL}
            --ascend-cmake-dir ${ASCEND_PROJECT_DIR}
            --ascend-compute_unit ${ASCEND_COMPUTE_UNIT}
            --ascend-op_name ${EP_ASCEND_OP_NAME}
            RESULT_VARIABLE result)
    if (result)
        message(FATAL_ERROR "Error: gym prepare build failed.")
    endif ()
endif ()
