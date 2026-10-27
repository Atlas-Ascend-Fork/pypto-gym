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

# 必须在 project() 命令之前执行
include(cmake/prepare_3rd.cmake)

project(pypto_gym
        LANGUAGES C CXX
        VERSION 1.0.0
)
init_cann_project()
add_cann_target_options()

########################################################################################################################
# 基础配置
########################################################################################################################

# 常量
set(VENDOR_NAME                   "pypto_gym"      CACHE STRING "vendor name, installed under opp/vendors/<name>")

# 路径配置
get_filename_component(PYPTO_GYM_SRC_DIR        "${CMAKE_CURRENT_SOURCE_DIR}"      REALPATH)

# 公开编译开关(build)
set(ASCEND_COMPUTE_UNIT           "ascend950"      CACHE STRING "soc that need to be compiled")
set(ASCEND_OP_NAME                "ALL"            CACHE STRING "operators that need to be compiled")
set(PACKAGE_TYPE                  "run"            CACHE STRING "package type: run")
option(ENABLE_OPS_HOST            "Build ops host."                 ON)
option(ENABLE_OPS_KERNEL          "Build ops kernel."               ON)
option(ENABLE_ASAN                "Enable AddressSanitizer (host side)." OFF)
option(ENABLE_UBSAN               "Enable UndefinedBehaviorSanitizer (host side)." OFF)

# 公开编译开关(tests)
# 算子级 dispatcher 据此派发 tests/ 目录; 亦为 config.cmake 预留 XSAN 块的激活条件
option(ENABLE_TESTS_EXECUTE "Enable auto execute tests executable" OFF)

# 流程控制开关(内部机制, 由 prepare.sh 以 -DPREPARE_BUILD=ON 驱动; 手工/CLion 入口保持 OFF)
option(PREPARE_BUILD              "Prepare build (second pass)."    OFF)

# 缺省编译配置, 编译函数, 反射载体 stub
include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/obj_func.cmake)

########################################################################################################################
# 编译流程
########################################################################################################################

# 算子接入: 发现 ops/{pro,tensor} 树, add 算子级目录(dispatcher 派发 op_host/tests).
# 算子侧只做"申报"(宏登记源到全局属性), 目标创建在接入后的 custom_build.cmake 读回完成.
gym_op_discovery(OP_LIST OP_DIR_LIST)
foreach(_op_dir ${OP_DIR_LIST})
    add_subdirectory(${_op_dir})
endforeach()

# 读回算子登记, 创建 host 侧目标(三 .so/核对程序)与 kernel 编译链
include(cmake/custom_build.cmake)       # TODO 看看能不能去掉

########################################################################################################################
# 打包(cann-cmake + makeself)
#   组件/版本定义 -> version.<pkg>.info 生成 -> 脚本骨架安装 -> CPack External 配置(package 目标产出 .run)
#   COMPUTE_UNIT 传入后 .run 名带 chip(如 cann-950-pypto_gym_9.2.0_linux-aarch64.run),
#   多 soc 轮次(build_ci.py 按 soc 循环、build/<soc> 隔离)产物互不冲突
########################################################################################################################
include(version.cmake)
add_cann_version_info_targets()

# run 包脚本骨架安装: install.sh 入口 + help/cleanup + version.info.
# 组件名与 install COMPONENT 一致(set_cann_cpack_config 按组件收集 staging).
set(_GymScriptPrefix "${PYPTO_GYM_SRC_DIR}/scripts/package/${VENDOR_NAME}/scripts")
install(DIRECTORY "${_GymScriptPrefix}/"
        DESTINATION share/info/${VENDOR_NAME}/script
        FILE_PERMISSIONS
                OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
        DIRECTORY_PERMISSIONS
                OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 目录权限
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
        COMPONENT ${VENDOR_NAME}
)

# cann-cmake 公共脚本
set(SCRIPTS_FILES
        ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
        ${CANN_CMAKE_DIR}/scripts/install/common_func_v2.inc
        ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
        ${CANN_CMAKE_DIR}/scripts/install/common_installer.inc
        ${CANN_CMAKE_DIR}/scripts/install/common_interface.csh
        ${CANN_CMAKE_DIR}/scripts/install/common_interface.fish
        ${CANN_CMAKE_DIR}/scripts/install/common_interface.sh
        ${CANN_CMAKE_DIR}/scripts/install/install_common_parser.sh
        ${CANN_CMAKE_DIR}/scripts/install/multi_version.inc
        ${CANN_CMAKE_DIR}/scripts/install/script_operator.inc
        ${CANN_CMAKE_DIR}/scripts/install/version_cfg.inc
        ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
)
install(FILES ${SCRIPTS_FILES}
        DESTINATION share/info/${VENDOR_NAME}/script
        COMPONENT ${VENDOR_NAME}
)

# version 配置
#       由 add_cann_version_info_targets 命令产出
install(FILES ${CMAKE_BINARY_DIR}/version.${VENDOR_NAME}.info
        DESTINATION share/info/${VENDOR_NAME}
        RENAME version.info
        COMPONENT ${VENDOR_NAME}
)

set_cann_cpack_config(${VENDOR_NAME} COMPUTE_UNIT ${ASCEND_COMPUTE_UNIT} PACKAGE_TYPE "${PACKAGE_TYPE}")
