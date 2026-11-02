#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

BRed="\033[1;31m"
Color_Off="\033[0m"

# Log error
function LOG_ERROR() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BRed}[ERROR] ${date_time} ${assert_msg}${Color_Off}"
}

function DP_ASSERT_EQUAL() {
    local actual_value=${1}
    local expect_value=${2}
    local assert_msg=${3}
    local log_flag=${4:-"true"}
    local log_path=${5}
    if [ "${actual_value}" != "${expect_value}" ]; then
        if [ -n "${log_path}" ] && [ -f "${log_path}" ]; then
            cat "${log_path}"
        fi
        LOG_ERROR "${assert_msg} is failed."
        exit 1
    else
        if [ "${log_flag}" = "true" ]; then
            echo "${assert_msg} is success."
        fi
    fi
}


echo "WORKSPACE: ${WORKSPACE}"
echo "TARGET_BRANCH: ${TARGET_BRANCH}"
echo "OS_TYPE: ${OS_TYPE}"
echo "task_name: ${task_name}"

cd ${WORKSPACE}
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
if [[ "${task_name}" == *ubuntu24* ]]; then
    if sudo update-alternatives --set gcc /usr/bin/gcc-16 2>/dev/null; then
        echo "Switched to gcc-16"
    elif sudo update-alternatives --set gcc /usr/bin/gcc-15 2>/dev/null; then
        echo "Switched to gcc-15"
    elif sudo update-alternatives --set gcc /usr/bin/gcc-14 2>/dev/null; then
        echo "gcc-16/15 not available, fell back to gcc-14"
    fi
    export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi

if gcc --version | head -n1 | grep -q "15\."; then
    rm -rf /home/jenkins/opensource/lib_cache
    if [ -d /home/jenkins/opensource/gcc15 ]; then
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15/lib_cache/ /home/jenkins/opensource/lib_cache
    elif [ -d /home/jenkins/opensource/gcc15x86 ]; then
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15x86/lib_cache/ /home/jenkins/opensource/lib_cache
    fi
elif gcc --version | head -n1 | grep -q "14\."; then
    gcc --version
else
    gcc --version
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi

gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e
ARCH=$(uname -m)
case "${ARCH}" in
    x86_64)
        ARCH_TAG="x86"
        ;;
    aarch64)
        ARCH_TAG="arm"
        ;;
esac

python3 build_ci.py --clean --pkg-type run
build_ret=$?
DP_ASSERT_EQUAL "${build_ret}" 0 "build ${task_name}"
echo "exec cmd: [python3 build_ci.py --clean --pkg-type run ]"
mkdir -p "${WORKSPACE}/build_whl_run"
find "${WORKSPACE}/build_out" -type f -name "*.run" -exec cp -f {} "${WORKSPACE}/build_whl_run/" \;
python3 build_ci.py --clean --pkg-type whl
build_ret=$?
DP_ASSERT_EQUAL "${build_ret}" 0 "build ${task_name}"
echo "exec cmd: [python3 build_ci.py --clean --pkg-type whl ]"
find "${WORKSPACE}/build_out" -type f -name "*.whl" -exec cp -f {} "${WORKSPACE}/build_whl_run/" \;
PKG_NAME="build_whl_run_${ARCH_TAG}.tar.gz"
tar -zcf "${WORKSPACE}/${PKG_NAME}" -C "${WORKSPACE}" build_whl_run
find "${WORKSPACE}/build_out" -mindepth 1 -delete
cp -f "${WORKSPACE}/${PKG_NAME}" "${WORKSPACE}/build_out/"
