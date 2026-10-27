#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# PyPTO SMoke 控制
# 注意: 该脚本可能由 Host 通过远程调用方式, 使其运行在 Docker 中


set -e

echo "======== 脚本原始参数 \$* = [$*]"
echo "======== 脚本参数个数 \$# = [$#]"

# ======================================================================================================================
# 1. 基础函数实现
# ======================================================================================================================

BRed='\e[1;31m'         # Red
BGreen='\e[1;32m'       # Green
Purple='\e[0;35m'       # Purple
BPurple='\e[1;35m'      # Bold Purple
Color_Off='\e[0m'       # Text Reset

function LOG_HEAD() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "\n${BGreen}[INFO] ${date_time} ${assert_msg}${Color_Off}"
}

function LOG_DO() {
   local -a cmd=("$@")
   local cmd_desc
   date_time=$(date +%Y%m%d-%H%M%S)
   printf -v cmd_desc '%q ' "${cmd[@]}"
   echo -e "${BPurple}[Command]${Color_Off} ${date_time} ${Purple}${cmd_desc}${Color_Off}"
   "${cmd[@]}"
}

function LOG_INFO() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "[INFO] ${date_time} ${assert_msg}" >&2
}

function LOG_ERROR() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BRed}[ERROR] ${date_time} ${assert_msg}${Color_Off}"
}


# ======================================================================================================================
# 2. 全局配置
#    全局初始化操作调用
#    自定义全局变量及函数实现
# ======================================================================================================================
LOG_DO source "$HOME/.bashrc"  # 如果是新环境, 注意修改 .bashrc 内容，去除 [ -z "$PS1" ] && return

# 获取当前脚本目录
CUR_DIR=$(cd "$(dirname "$0")" && pwd)

# DATA_DIR 是当前目录的 **上一级目录**
DATA_DIR=$(dirname "${CUR_DIR}")

# 打印确认（可选）
LOG_INFO "CUR_DIR=$CUR_DIR"
LOG_INFO "DATA_DIR=$DATA_DIR"
if [ -d "/usr/local/Ascend/ascend-toolkit/latest/" ]; then
    echo "ascend-toolkit 目录存在"
fi
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

if [ -n "${ASCEND_HOME_PATH:+set}" ]; then
    source "${ASCEND_HOME_PATH}/bin/setenv.bash"
elif [ -f "/usr/local/Ascend/cann/bin/setenv.bash" ]; then
    source "/usr/local/Ascend/cann/bin/setenv.bash"
fi

# 检查升级锁文件，升级锁检查已在外部设置，此处注掉
# ENV_UPGRADE_LOCK_FILE="$DATA_DIR/env_upgrade_flag_v1"
# if [ -f "$ENV_UPGRADE_LOCK_FILE" ]; then
#     LOG_ERROR "Current Environment upgrading, please wait a moment."
#     exit 1
# fi

run_pypto_build() {
    local _python3="$1"
    local gym_repo="$2"
    shift 2

    if ! command -v "$_python3" &>/dev/null; then
        LOG_ERROR "Python interpreter '$_python3' not found!"
        return 1
    fi
    if [ ! -f "$gym_repo/build_ci.py" ]; then
        LOG_ERROR "Build entry '$gym_repo/build_ci.py' not found!"
        return 1
    fi

    local start_time
    local end_time
    local elapsed
    start_time=$(date +%s)
    LOG_HEAD "[BGN] Build PyPTO run package"
    pushd "$gym_repo" > /dev/null || return 1
    if "$_python3" build_ci.py "$@"; then
        :
    else
        local ret=$?
        popd > /dev/null || true
        LOG_ERROR "Build PyPTO run package failed"
        return "$ret"
    fi
    popd > /dev/null || return 1
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    LOG_HEAD "[END] Build PyPTO run package, duration $elapsed secs."
}

install_pypto_run() {
    local pypto_repo="$1"
    local cann_path="$2"
    local timeout_secs="$3"
    local run_pkg

    if [ ! -d "$pypto_repo/build_out" ]; then
        LOG_ERROR "PyPTO output directory '$pypto_repo/build_out' not found"
        return 1
    fi
    run_pkg=$(find "$pypto_repo/build_out" -maxdepth 1 -type f -name 'cann-pypto_*.run' -print -quit)
    if [ -z "$run_pkg" ]; then
        LOG_ERROR "No PyPTO run package found in $pypto_repo/build_out"
        return 1
    fi

    LOG_INFO "PYPTO_RUN_PKG=$run_pkg"
    LOG_HEAD "[BGN] Install PyPTO run package"
    if timeout --signal=INT "$timeout_secs" \
        bash "$run_pkg" --full -q --pylocal --install-path="$cann_path"; then
        :
    else
        local ret=$?
        LOG_ERROR "Install PyPTO run package failed"
        return "$ret"
    fi
    LOG_HEAD "[END] Install PyPTO run package"
}

run_gym_tests() {
    local _python3="$1"
    local gym_repo="$2"
    local timeout_secs="$3"
    shift 3

    local device_ids="0,1,2,3"
    local imported_pypto
    local test_targets=(
        "tests/ops/experimental/ops_transformer/flash_attention_mha/test_flash_attention_mha.py::test_00_910_1b"
        "tests/ops/experimental/ops_transformer/flash_attention_mha_grad/test_flash_attention_mha_grad_a3.py::test_02"
        "tests/ops/qat/test_qat.py::test_asymmetric_per_group"
        "tests/ops/experimental/ops_transformer/engram/test_engram_forward.py::test_engram_forward_pypto"
    )
    local pypto_site="$CANN_PATH/python/site-packages"
    if [ ! -d "$pypto_site" ]; then
        LOG_ERROR "PyPTO Python installation directory '$pypto_site' not found"
        return 1
    fi
    export PYTHONPATH="$pypto_site${PYTHONPATH:+:$PYTHONPATH}"
    local import_probe
    import_probe=$(mktemp)
    if ! "$_python3" -c 'import pypto; print(f"__PYPTO_IMPORT_PATH__={pypto.__file__}", flush=True)' >"$import_probe" 2>&1; then
        LOG_ERROR "Failed to import PyPTO from the run package"
        cat "$import_probe"
        rm -f "$import_probe"
        return 1
    fi
    imported_pypto=$(sed -n 's/^__PYPTO_IMPORT_PATH__=//p' "$import_probe" | tail -n 1)
    rm -f "$import_probe"
    if [ -z "$imported_pypto" ]; then
        LOG_ERROR "Failed to determine the imported PyPTO path"
        return 1
    fi
    LOG_INFO "PyPTO imported from: $imported_pypto"
    if [[ "$imported_pypto" != "$CANN_PATH"* ]]; then
        LOG_ERROR "PyPTO is not imported from the run package installation under $CANN_PATH"
        return 1
    fi

    LOG_HEAD "[BGN] Run pypto-gym tests/ops"
    pushd "$gym_repo" > /dev/null || return 1
    if ASCEND_VISIBLE_DEVICES="$device_ids" \
        PYTEST_AVAILABLE_DEVICES="$device_ids" \
        timeout --signal=INT "$timeout_secs" \
        "$_python3" -m pytest "${test_targets[@]}" -v --durations=0 -s --capture=no \
        --rootdir="$gym_repo" -n 4 --dist=loadscope -p no:skipping "$@"; then
        :
    else
        local ret=$?
        popd > /dev/null || true
        LOG_ERROR "pypto-gym tests failed"
        return "$ret"
    fi
    popd > /dev/null || return 1
    LOG_HEAD "[END] Run pypto-gym tests/ops"
}

cleanup_npu_processes() {
    local _python3="$1"
    local gym_repo="$2"
    if [ -f "$gym_repo/kill_npu_processes.py" ]; then
        "$_python3" "$gym_repo/kill_npu_processes.py" || true
    else
        LOG_INFO "Skip NPU process cleanup: $gym_repo/kill_npu_processes.py not found"
    fi
    npu-smi info || true
}

parse_config_key() {
    # 定义局部变量，避免污染全局
    # local config_file="$DATA_DIR/pypto_cann.cfg"
    CUR_DIR=$(cd "$(dirname "$0")" && pwd)
    local config_file="$CUR_DIR/pypto_cann.cfg"
    local config_query_key="$1"
    local config_query_val=""

    # 1. 校验配置文件是否存在
    if [ ! -f "$config_file" ]; then
        LOG_ERROR "Config file $config_file not exist."
        exit 1
    fi
    # 2. 校验参数：目标键名不能为空
    if [ -z "$config_query_key" ]; then
        LOG_ERROR "Config query key empty."
        exit 1
    fi
    # 3. 核心逻辑：精准匹配键名，提取值并去除所有空白字符
    # -F '='：以等号为分隔符
    # -v key="$config_query_key"：将shell变量传入awk
    # $1 == key：严格匹配键名（避免部分匹配）
    # gsub(/[[:space:]]/, "")：去除值中的空格、制表符、换行等所有空白字符
    # exit：找到后立即退出，提升效率
    config_query_val=$(awk -F '=' -v key="$config_query_key" '
        $1 == key {
            gsub(/[[:space:]]/, "");  # 清洗值的空白字符
            print $2;
            exit;
        }
    ' "$config_file")
    # 4. 直接输出值（无匹配/值为空时，输出空字符串）
    echo "$config_query_val"
    return 0
}

# PYTHON3_EXE=$(which python3)
PYTHON3_EXE=/opt/conda/bin/python3
export PATH="$(dirname "$PYTHON3_EXE"):$PATH"
LOG_INFO "PYTHON3_EXE=$PYTHON3_EXE"

PYPTO_GOLDEN_PATH=$(parse_config_key "PYPTO_GOLDEN_PATH")
if [ -z "$PYPTO_GOLDEN_PATH" ]; then
    LOG_ERROR "Can't get PYPTO_GOLDEN_PATH."
    exit 1
fi
LOG_INFO "PYPTO_GOLDEN_PATH=$PYPTO_GOLDEN_PATH"

PYPTO_3RD_LIB_PATH=$(parse_config_key "PYPTO_3RD_LIB_PATH")
if [ -z "$PYPTO_3RD_LIB_PATH" ]; then
    LOG_ERROR "Can't get PYPTO_3RD_LIB_PATH."
    exit 1
fi
LOG_INFO "PYPTO_3RD_LIB_PATH=$PYPTO_3RD_LIB_PATH"

# ======================================================================================================================
# 3: 解析脚本参数
# ======================================================================================================================
SRC_DIR=""
SRC_TARGET_BRANCH="master"  # 不传入本参数时默认是 master
PTO_ISA_URL=""
CI_MODE=false
CHANGED_FILE_FROM_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)
            SRC_DIR="$2"
            shift 2
            ;;
        --src_target_branch)
            SRC_TARGET_BRANCH="$2"
            shift 2
            ;;
        --pto_isa_url)
            PTO_ISA_URL="$2"
            shift 2
            ;;
        --ci)
            CI_MODE=true
            shift
            ;;
        --f)
            CHANGED_FILE_FROM_ARG="$2"  # <-- 接收 --f 参数
            shift 2
            ;;
        *)
            LOG_ERROR "Unknown argument $1"
            exit 1
            ;;
    esac
done
# 检查必须参数
if [ -z "$SRC_DIR" ]; then
    LOG_ERROR "--src argument is required"
    exit 1
fi
if [ ! -d "$SRC_DIR" ]; then
    LOG_ERROR "$SRC_DIR is not a valid directory"
    exit 1
fi
SRC_DIR=$(cd "$SRC_DIR" && pwd)
# 检查输出可选参数
LOG_INFO "SRC_DIR=$SRC_DIR"
LOG_INFO "SRC_TARGET_BRANCH=$SRC_TARGET_BRANCH"
LOG_INFO "PTO_ISA_URL=$PTO_ISA_URL"

# ======================================================================================================================
# 4: 设置 CANN 环境变量
# ======================================================================================================================
TARGET_CONFIG_KEY="CANN_$SRC_TARGET_BRANCH"
CANN_PATH=$(parse_config_key "$TARGET_CONFIG_KEY")
if [ -z "$CANN_PATH" ]; then
    # 错误提示包含关键信息：缺失的键名、配置文件路径、分支名，方便快速定位
    LOG_ERROR "Failed to get CANN path! "
    LOG_ERROR "  - Target key: $TARGET_CONFIG_KEY (branch: $SRC_TARGET_BRANCH)"
    LOG_ERROR "  Please check if the key exists and has a valid value in the config file."
    exit 1
fi
export CANN_PATH
LOG_INFO "CANN_PATH=$CANN_PATH"

# 安装 PTO-ISA 包
if [ -n "${PTO_ISA_URL}" ]; then

    # 目标目录
    PTO_ISA_PKG_DIR="$DATA_DIR/pto_isa_pkg/$SRC_TARGET_BRANCH"

    # 创建目录（不存在则自动创建）
    mkdir -p "${PTO_ISA_PKG_DIR}"

    # 进入目录，并记住进入前的路径
    pushd "${PTO_ISA_PKG_DIR}" > /dev/null || exit 1

    # 删除所有 .run 文件
    rm -f ./*.run

    # wget .run 包并安装
    PTO_ISA_PKG_NAME="cann-pto-isa_linux.run"
    wget -O "$PTO_ISA_PKG_NAME" "${PTO_ISA_URL}"
    chmod +x ./$PTO_ISA_PKG_NAME
    LOG_DO "bash" "$PTO_ISA_PKG_NAME" "--full" "--quiet" "--install-path=${CANN_PATH}../"

    echo "安装pto-isa包成功：${CANN_PATH}"

    # 自动回到进入前的工作目录
    popd > /dev/null

fi

# 验证setenv.bash文件存在性
SETENV_SH="$CANN_PATH/bin/setenv.bash"
if [ ! -f "$SETENV_SH" ]; then
    LOG_ERROR "$SETENV_SH not found (CANN path: $CANN_PATH)"
    exit 1
fi
LOG_DO source "$SETENV_SH"

# 下载pypto代码
work_dir="$SRC_DIR"
# 定位上级目录
work_parent_dir="${work_dir}/.."
target_repo="${work_parent_dir}/pypto"

# 判断上级目录是否存在pypto文件夹，存在则彻底删除
if [ -d "${target_repo}" ]; then
    echo "检测到上级目录已存在pypto仓库，开始删除..."
    rm -rf "${target_repo}"
fi

# if [ ! -d "${target_repo}" ]; then
#     LOG_INFO "上级目录 ${target_repo} 不存在，开始创建"
#     mkdir -p "${target_repo}"
#     chmod 755 "${target_repo}"
#     LOG_INFO "目录创建完成，权限已设置755"
# fi

# 进入上级目录执行克隆操作
echo "work_parent_dir 变量值为: ${work_parent_dir}"

cd "${work_parent_dir}" || exit 1
git clone --branch "$SRC_TARGET_BRANCH" --single-branch https://gitcode.com/cann/pypto.git

echo "target_repo 变量值为: ${target_repo}"


echo "========== 当前目录文件列表 =========="
ls "${target_repo}"
echo "======================================"

# 进入上级目录执行克隆操作
cd "${work_dir}" || exit 1


# ======================================================================================================================
# 5: 调用任务, 并统计耗时
# ======================================================================================================================
cd "$SRC_DIR" || { LOG_ERROR "Failed to cd to $SRC_DIR"; exit 1; }

# 调用参数准备
if [[ "$CI_MODE" == true ]]; then
    if [ -z "$CHANGED_FILE_FROM_ARG" ]; then
        LOG_ERROR "--ci requires --f <changed-files-file>"
        exit 1
    fi
    changed_file_path="$CHANGED_FILE_FROM_ARG"
    if [ ! -f "$changed_file_path" ]; then
        LOG_ERROR "Changed-files file '$changed_file_path' not found"
        exit 1
    fi
    LOG_INFO "使用外部传入文件: $changed_file_path"
    LOG_INFO "Changed files content as follows:"
    LOG_DO "cat" "$changed_file_path"
fi

# 开始执行任务
total_start=$(date +%s)

# 参数默认值
PYTHON_TOTAL_TIMEOUT=$(parse_config_key "PYTHON_TOTAL_TIMEOUT")
if [ -z "$PYTHON_TOTAL_TIMEOUT" ]; then
    PYTHON_TOTAL_TIMEOUT=900
fi
PYTHON_TOTAL_TIMEOUT="$PYTHON_TOTAL_TIMEOUT"
LOG_INFO "PYTHON_TOTAL_TIMEOUT=$PYTHON_TOTAL_TIMEOUT"

LOG_HEAD "Python Environment:"
LOG_DO "$PYTHON3_EXE" --version
LOG_DO "$PYTHON3_EXE" -m pip list
LOG_DO "npu-smi" "info"

# 使用 PyPTO 仓库自身的构建入口生成 run 包。不要传 --just_build_whl，
# 也不要传 --models/-u/-s，避免在 PyPTO 仓库内提前执行其自身测试。
# $target_repo 是本脚本前面克隆的 pypto 仓(与 gym 同级), gym 的 build_ci.py 职责已修正为
# 构建 gym 自身产物(whl/custom run), 不再代理 pypto 构建.
if run_pypto_build "$PYTHON3_EXE" "$target_repo" \
    --clean \
    --verbose \
    "--cann_3rd_lib_path=$PYPTO_3RD_LIB_PATH" \
    "--timeout=$PYTHON_TOTAL_TIMEOUT"; then
    :
else
    ret=$?
    cleanup_npu_processes "$PYTHON3_EXE" "$SRC_DIR"
    exit "$ret"
fi

# gym 验证对象是 run 安装结果，不直接 pip install build_out 中的 whl。
elapsed_after_build=$(($(date +%s) - total_start))
RUN_INSTALL_TIMEOUT=$((PYTHON_TOTAL_TIMEOUT - elapsed_after_build))
if [ "$RUN_INSTALL_TIMEOUT" -le 0 ]; then
    LOG_ERROR "No timeout budget remains for PyPTO run package installation"
    cleanup_npu_processes "$PYTHON3_EXE" "$SRC_DIR"
    exit 124
fi
LOG_INFO "RUN_INSTALL_TIMEOUT=$RUN_INSTALL_TIMEOUT"

if install_pypto_run "$target_repo" "$CANN_PATH" "$RUN_INSTALL_TIMEOUT"; then
    :
else
    ret=$?
    cleanup_npu_processes "$PYTHON3_EXE" "$SRC_DIR"
    exit "$ret"
fi
LOG_DO source "$SETENV_SH"

elapsed_before_tests=$(($(date +%s) - total_start))
GYM_TEST_TIMEOUT=$((PYTHON_TOTAL_TIMEOUT - elapsed_before_tests))
if [ "$GYM_TEST_TIMEOUT" -le 0 ]; then
    LOG_ERROR "No timeout budget remains for pypto-gym tests"
    cleanup_npu_processes "$PYTHON3_EXE" "$SRC_DIR"
    exit 124
fi
LOG_INFO "GYM_TEST_TIMEOUT=$GYM_TEST_TIMEOUT"

if run_gym_tests "$PYTHON3_EXE" "$SRC_DIR" "$GYM_TEST_TIMEOUT"; then
    :
else
    ret=$?
    cleanup_npu_processes "$PYTHON3_EXE" "$SRC_DIR"
    exit "$ret"
fi
LOG_DO "npu-smi" "info"

# 总耗时统计
total_end=$(date +%s)
total_elapsed=$((total_end - total_start))
LOG_HEAD "All builds completed successfully, Total execution time: $total_elapsed seconds"
echo -e "execute sample success"
