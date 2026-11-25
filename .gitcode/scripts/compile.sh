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

# ---------------------------------------------------------------------------
# Build & install pypto (master) before gym build (gym build depends on pypto).
# Skip the build when this PR's cached archive pypto_run_<arch>.tar.gz
# (.run + master sha, at ci/package/<MR>) matches current master HEAD
# (compile pods are ephemeral, so the cache rides on OBS).
# ---------------------------------------------------------------------------
PYPTO_REPO_DIR="${WORKSPACE}/pypto"
PYPTO_CACHE_TMP="${WORKSPACE}/pypto_cache_tmp"

rm -rf "${PYPTO_REPO_DIR}"
git clone --depth 1 --branch master --single-branch https://gitcode.com/cann/pypto.git "${PYPTO_REPO_DIR}"
DP_ASSERT_EQUAL "$?" 0 "clone pypto master"
PYPTO_SHA=$(git -C "${PYPTO_REPO_DIR}" rev-parse HEAD)

# reuse this PR's previous round archive when its sha matches master HEAD
PYPTO_RUN_PKG=""
if [ -n "${PYPTO_CACHE_URL}" ]; then
    rm -rf "${PYPTO_CACHE_TMP}" && mkdir -p "${PYPTO_CACHE_TMP}"
    if wget -q -O "${PYPTO_CACHE_TMP}/pypto_run.tar.gz" "${PYPTO_CACHE_URL}/pypto_run_${ARCH_TAG}.tar.gz"; then
        tar -zxf "${PYPTO_CACHE_TMP}/pypto_run.tar.gz" -C "${PYPTO_CACHE_TMP}"
        _cached_sha=$(cat "${PYPTO_CACHE_TMP}/pypto_master_sha_${ARCH_TAG}.txt" 2>/dev/null)
        if [ "${_cached_sha}" = "${PYPTO_SHA}" ]; then
            PYPTO_RUN_PKG=$(find "${PYPTO_CACHE_TMP}" -maxdepth 1 -name "cann-pypto_*.run" -print -quit)
            [ -n "${PYPTO_RUN_PKG}" ] && echo "pypto master unchanged (${PYPTO_SHA}), reuse cached run package from ${PYPTO_CACHE_URL}"
        else
            echo "pypto run cache outdated (cache sha: ${_cached_sha:-none}, master: ${PYPTO_SHA})"
        fi
    else
        echo "no cached pypto run package at ${PYPTO_CACHE_URL}/pypto_run_${ARCH_TAG}.tar.gz"
    fi
fi
if [ -z "${PYPTO_RUN_PKG}" ]; then
    echo "no reusable pypto run cache, build pypto"
fi

if [ -z "${PYPTO_RUN_PKG}" ]; then
    cd "${PYPTO_REPO_DIR}"
    python3 -m pip install build "packaging==24.2"
    python3 build_ci.py --clean --build_type=Release --timeout=900 --no_isolation --job_num=16 --pkg-type run
    DP_ASSERT_EQUAL "$?" 0 "build pypto ${task_name}"
    PYPTO_RUN_PKG=$(find "${PYPTO_REPO_DIR}/build_out" -maxdepth 1 -name "cann-pypto_*.run" -print -quit)
    [ -n "${PYPTO_RUN_PKG}" ] || { LOG_ERROR "No pypto run package found in ${PYPTO_REPO_DIR}/build_out"; exit 1; }
fi

# install as jenkins user: a root install aborts because install-path parents
# (/home/jenkins/Ascend/cann) belong to jenkins; --install-path is the Ascend
# root, pypto install.sh appends the "cann" version dir itself
echo "y" | su - jenkins -c "cd '${PYPTO_RUN_PKG%/*}' && bash './${PYPTO_RUN_PKG##*/}' --full --quiet --pylocal --install-path=/home/jenkins/Ascend"
DP_ASSERT_EQUAL "$?" 0 "install pypto run package"

# archive .run + master sha: consumed by the smoke stage and the next round's
# cache read of this PR
echo "${PYPTO_SHA}" > "${WORKSPACE}/pypto_master_sha_${ARCH_TAG}.txt"
PYPTO_PKG_NAME="pypto_run_${ARCH_TAG}.tar.gz"
tar -zcf "${WORKSPACE}/${PYPTO_PKG_NAME}" -C "${PYPTO_RUN_PKG%/*}" "${PYPTO_RUN_PKG##*/}" -C "${WORKSPACE}" "pypto_master_sha_${ARCH_TAG}.txt"

# pylocal-installed python packages for gym build-time pypto/pypto_pro codegen
export PYTHONPATH="/home/jenkins/Ascend/cann/python/site-packages:${PYTHONPATH}"
python3 -c "import pypto, pypto_pro" || { LOG_ERROR "import pypto/pypto_pro failed after pypto install"; exit 1; }

cd "${WORKSPACE}"

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
cp -f "${WORKSPACE}/${PYPTO_PKG_NAME}" "${WORKSPACE}/build_out/"
