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

log() {
  echo "===================================================================="
  echo "$(date '+%Y%m%d.%H%M%S') : $*"
  echo "===================================================================="
}

# 结果检查: 退出码非 0 或日志中含 FAILED/ERROR 均判失败
check_result() {
  local rc="$1" log_file="$2" label="$3"
  if [ "${rc}" -ne 0 ]; then
    echo "$(date +%Y%m%d.%H%M%S) : ${label} failed, exit code ${rc}"
    exit 1
  fi
  if grep -E '(^|[[:space:]])(FAILED|ERRORS?)([[:space:]]|$)|[0-9]+ (failed|errors?)\b' "${log_file}"; then
    echo "$(date +%Y%m%d.%H%M%S) : ${label} failed"
    exit 1
  fi
}

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0
source /usr/local/Ascend/cann/set_env.sh
pip3 install psutil

log "init test case, please wait ..."
rm -rf /root/ascend/log

# ==============================
# 运行冒烟测试
# pto-isa 包架构由 export 控制: PTO_ISA_ARCH=aarch64(默认)/x86_64;
# PTO_ISA_URL 可直接覆盖完整下载地址(优先级高于 PTO_ISA_ARCH)
# ==============================
case "${PTO_ISA_ARCH:-aarch64}" in
    arm|aarch64)      pto_arch="aarch64" ;;
    x86|x86_64|amd64) pto_arch="x86_64" ;;
    *) echo "invalid PTO_ISA_ARCH=${PTO_ISA_ARCH} (expected aarch64/arm or x86_64/x86)"; exit 1 ;;
esac
pto_url="${PTO_ISA_URL:-https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/pto-isa/daily/cann-pto-isa_linux-${pto_arch}.run}"
echo "[INFO] pto-isa package url: ${pto_url}"

log "start run test case, please wait ..."
echo "==============start smoke testing================="
bash pypto_gym_smoke.sh --src ${WORKSPACE} --pto_isa_url "${pto_url}" --src_target_branch master --f ${WORKSPACE}/pr_filelist.txt --ci 2>&1 | tee -a ./run_test.log
smoke_rc=${PIPESTATUS[0]}

# ==============================
# 打包 log 并上传
# ==============================
mkdir -p /root/ascend
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

# upload plog
# if python3 /home/upload.py --bucket-name "ascend-ci" --action upload  --local-file "slog.tar.gz" --obs-object-key "${repo_name}/package/${pr_id}/${slog_name}"; then
#   echo "::set-output var=plog_url:https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${repo_name}/package/${pr_id}/slog.tar.gz"
# fi

log "checking NPU status ..."
mkdir -p ./npu_log
npu-smi info 2>&1 | tee ./npu_log/npu_info.log

check_result "${smoke_rc}" "./run_test.log" "run test case (smoke)"

# ==============================
# gym ST 看护(aclnn 模式): build_ci.py --stest
# 仅 950(A5) 卡执行, 开关与 pypto_gym_smoke.sh 冒烟一致: export GYM_SMOKE_SOC
#   950/ascend950      执行 stest
#   910/ascend910/未设置  跳过(gym ST 用例及 950 run 包仅适配 A5)
# 可调: GYM_STEST_TIMEOUT(build_ci.py 总超时秒数, 默认 2700)
#       GYM_STEST_FILTER(pytest -k 过滤表达式, 空=全量)
# 注意: 流水 step 的 timeout-minutes 需覆盖 冒烟+gym构建+ST 总耗时
# ==============================
case "${GYM_SMOKE_SOC:-910}" in
    950|ascend950|950pr|950PR) ;;
    910|ascend910)
      log "run gym stest skipped (GYM_SMOKE_SOC=${GYM_SMOKE_SOC:-910})"
      exit 0 ;;
    *) echo "invalid GYM_SMOKE_SOC=${GYM_SMOKE_SOC} (expected 910/ascend910 or 950/ascend950)"; exit 1 ;;
esac

# Python 解释器锁定(/opt/conda/bin/python3 优先, 回退 PATH 中 python3)并导出给 cmake:
# 避免子 configure find_package 误选其它 python, 导致 pybind 测试桥与运行解释器版本错配
GYM_PYTHON3=/opt/conda/bin/python3
[ -x "${GYM_PYTHON3}" ] || GYM_PYTHON3=$(command -v python3)
if [ -z "${GYM_PYTHON3}" ] || [ ! -x "${GYM_PYTHON3}" ]; then
    echo "$(date +%Y%m%d.%H%M%S) : gym stest python3 not found or not executable"
    exit 1
fi
if ! "${GYM_PYTHON3}" -c 'import torch, torch_npu' >/dev/null 2>&1; then
    echo "$(date +%Y%m%d.%H%M%S) : torch/torch_npu import failed under ${GYM_PYTHON3} ($("${GYM_PYTHON3}" -V 2>&1))"
    exit 1
fi
export Python3_EXECUTABLE="${GYM_PYTHON3}"
export Python3_ROOT_DIR="$(dirname "$(dirname "${GYM_PYTHON3}")")"
log "gym stest python check ok: ${GYM_PYTHON3} ($("${GYM_PYTHON3}" -V 2>&1)), Python3_ROOT_DIR=${Python3_ROOT_DIR}"

STEST_ARG="--stest"
[ -n "${GYM_STEST_FILTER}" ] && STEST_ARG="--stest=${GYM_STEST_FILTER}"

cd "${WORKSPACE}" || exit 1
# 三方库不传 cann_3rd_lib_path, 与 compile.sh(stage2) 一致, 由 prepare_3rd.cmake 回退 FetchContent
"${GYM_PYTHON3}" build_ci.py --pkg-type run "${STEST_ARG}" --timeout "${GYM_STEST_TIMEOUT:-2700}" 2>&1 | tee -a ./run_stest.log
stest_rc=${PIPESTATUS[0]}
check_result "${stest_rc}" "./run_stest.log" "run gym stest"
echo "$(date +%Y%m%d.%H%M%S) : run gym stest success"
