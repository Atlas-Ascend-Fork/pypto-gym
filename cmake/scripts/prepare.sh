#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 二次构建入口: 主 configure 期由 cmake/config.cmake 调用,
# 以 PREPARE_BUILD=ON 再跑一次 cmake 并 make prepare_build, 产出 autogen 与 impl.

CPU_NUM=$(($(grep -c "^processor" /proc/cpuinfo)*2))
JOB_NUM="-j${CPU_NUM}"

while [[ $# -gt 0 ]]; do
    case $1 in
    -s) PATH_TO_SOURCE="$2"; shift 2 ;;
    -b) PATH_TO_BUILD="$2"; shift 2 ;;
    -p) ASCEND_CANN_PACKAGE_PATH="$2"; shift 2 ;;
    --autogen-dir) ASCEND_AUTOGEN_DIR="$2"; shift 2 ;;
    --binary-out-dir) ASCEND_BINARY_OUT_DIR="$2"; shift 2 ;;
    --impl-out-dir) ASCEND_IMPL_OUT_DIR="$2"; shift 2 ;;
    --op-build-tool) OP_BUILD_TOOL="$2"; shift 2 ;;
    --ascend-cmake-dir) ASCEND_CMAKE_DIR="$2"; shift 2 ;;
    --ascend-compute_unit) ASCEND_COMPUTE_UNIT="$2"; shift 2 ;;
    --ascend-op_name) ASCEND_OP_NAME="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

convert_string() { echo "$1" | sed 's/::/;/g'; }

CONVERT_ASCEND_COMPUTE_UNIT="$(convert_string ${ASCEND_COMPUTE_UNIT})"
CONVERT_ASCEND_OP_NAME="$(convert_string ${ASCEND_OP_NAME})"

rm -rf "${PATH_TO_BUILD}"
mkdir -p "${PATH_TO_BUILD}"
cd "${PATH_TO_BUILD}"

cmake ${PATH_TO_SOURCE} \
    -DBUILD_OPEN_PROJECT=ON \
    -DPREPARE_BUILD=ON \
    -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH} \
    -DASCEND_AUTOGEN_DIR=${ASCEND_AUTOGEN_DIR} \
    -DASCEND_BINARY_OUT_DIR=${ASCEND_BINARY_OUT_DIR} \
    -DASCEND_IMPL_OUT_DIR=${ASCEND_IMPL_OUT_DIR} \
    -DOP_BUILD_TOOL=${OP_BUILD_TOOL} \
    -DASCEND_CMAKE_DIR=${ASCEND_CMAKE_DIR} \
    -DASCEND_COMPUTE_UNIT=${CONVERT_ASCEND_COMPUTE_UNIT} \
    -DASCEND_OP_NAME=${CONVERT_ASCEND_OP_NAME} \

make ${JOB_NUM} prepare_build
