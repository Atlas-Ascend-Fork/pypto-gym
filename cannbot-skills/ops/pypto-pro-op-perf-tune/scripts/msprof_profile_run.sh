#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
# msprof runner - 统一性能采集入口
#
# 支持四种模式：
#   1. 标准模式 (默认): 对单个可执行文件/PyPTO-Pro Python runner 采集 7 组 aic-metrics + sample-based
#   2. --compare 模式:  未传 --case-manifest：从 GOLDEN_PERF_REPORT.md 读 golden 数据 + msprof
#                       采集 PyPTO 算子计算加速比（既有行为）；传 --case-manifest：
#                       manifest 证据协议（Golden JSON exact-id join、逐 case/逐 repeat、seed=42）
#   3. --quick 模式:    每次 repeat 只采集 kernel 时间（既有行为）；传 manifest 时逐 case 严格归属
#   4. --batch 模式:    批量扫描目录下的多个算子子目录，并行执行对比测试；算子目录提供
#                       PERFORMANCE_CASES.json 或传入 map 时启用 manifest 证据批，否则保持
#                       既有 batch（flock 锁 + ASCEND_RT_VISIBLE_DEVICES）
#
# Usage (标准模式):
#   bash msprof_profile_run.sh [--warm-up=N] [--output=<dir>] -- python3 test_op.py
#
# Usage (--compare 模式):
#   bash msprof_profile_run.sh --compare --output-dir=/path/to/op_dir [--warm-up=N] [--device=N]
#
# Usage (--quick 模式):
#   bash msprof_profile_run.sh --quick --output-dir=/path/to/op_dir [--warm-up=N] [--device=N]
#
# Usage (--batch 模式):
#   bash msprof_profile_run.sh --batch --base-dir=/path/to/output_dir [--max-jobs=N] [--device-start=N]
#
# Example (标准):
#   bash msprof_profile_run.sh --warm-up=3 --output=./msprof_output -- \
#        python3 test_matmul.py
#
# Example (对比):
#   bash msprof_profile_run.sh --compare --output-dir=./output/GELU --warm-up=3 --device=0
#
# Example (快速):
#   bash msprof_profile_run.sh --quick --output-dir=./output/GELU --warm-up=3 --device=0
#
# Example (批量):
#   bash msprof_profile_run.sh --batch --base-dir=./output_performance --max-jobs=7 --device-start=1
#
# 产出:
#   标准模式: <output_dir>/PROF_GROUP_<timestamp>/PROF_<Metric>/ (7 个子目录 + 1 个 Sample)
#   对比模式: <output_dir>/performance.json + performance.log + perf_report.md
#   快速模式: <output_dir>/performance.json + performance.log + perf_report.md
#   批量模式: <base_dir>/batch_performance.log + 各子目录 performance.json
#   manifest 模式（传 --case-manifest）: quick 产物为 quick_* 三件套；逐 case 证据在 docs/perf/round_NNN/
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_START_TIME=$(date +%s%N)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 格式化耗时输出（纳秒精度入参）
format_elapsed() {
    local ns=$1
    local ms=$(( ns / 1000000 ))
    local s=$(( ms / 1000 ))
    local remain_ms=$(( ms % 1000 ))
    if [[ $s -gt 0 ]]; then
        printf "%d.%03ds" $s $remain_ms
    else
        printf "%dms" $ms
    fi
}

print_elapsed() {
    local now_ns
    now_ns=$(date +%s%N)
    local elapsed_ns=$(( now_ns - SCRIPT_START_TIME ))
    local msg
    msg="[INFO] Total elapsed time: $(format_elapsed "$elapsed_ns")"
    echo ""
    echo "$msg"
    # 如果指定了输出目录，同时写入 performance.log
    if [[ -n "${1:-}" ]]; then
        echo "" >> "$1/performance.log"
        echo "$msg" >> "$1/performance.log"
    fi
}

# 采集前刷新目标 NPU 的 L2 cache（cannbench 式）；失败只告警，不中断采集。
l2_clean() {
    [[ "${L2_CLEAN:-1}" == "1" ]] || return 0
    python3 "${SCRIPT_DIR}/l2_cache_clean.py" >/dev/null 2>&1 || \
        echo "WARN: L2 cache clean failed; continuing without it" >&2
}

# 默认参数
WARM_UP=3
OUTPUT_DIR="./msprof_output"
APP_CMD=()

# 模式开关
MODE="standard"   # standard | compare | batch | quick
OUTPUT_DIR_ARG=""  # --compare 模式用的算子目录
DEVICE_ID=""
BASE_DIR=""
MAX_JOBS=7
DEVICE_START=1
OP_NAME=""
OP_NAME_MAP_FILE=""
CASE_ARG=""
CASE_ENV=""
CASE_MANIFEST=""
CASE_MANIFEST_MAP_FILE=""
REPEATS=1
RETRY=2
KEEP_PROF=0
SEED=0
# 采集前刷新 L2 cache（cannbench 式），默认开启；--no-l2-clean 关闭。
L2_CLEAN=1

usage() {
    cat <<'EOF'
Usage: bash msprof_profile_run.sh [OPTIONS] -- python3 test_op.py [args...]
       bash msprof_profile_run.sh --compare --output-dir=<dir> [OPTIONS]
       bash msprof_profile_run.sh --quick --output-dir=<dir> [OPTIONS]
       bash msprof_profile_run.sh --batch --base-dir=<dir> [OPTIONS]

通用选项:
  --warm-up=N     在正式采集之前，先跑 N 次可执行文件（不采集）预热 DVFS。默认 3。
  --output=<dir>  msprof 结果根目录。默认 ./msprof_output。
  --l2-clean      每次测量前刷新目标 NPU 的 L2 cache（cannbench 式：大张量归约清空 L2，
                  使各次测量从一致的 cache 状态起步）。默认开启。
  --no-l2-clean   关闭测量前的 L2 刷新（与未含该机制的历史协议对齐）。
  -h, --help      Show this help.

标准模式 (默认):
  bash msprof_profile_run.sh [通用选项] -- <executable> [args...]
  （性能证据协议）首次 discovery 采集后先用 msprof_perf_summary.py <PROF_GROUP> --list-op-names
  列出 lowering 后的完整 Op Name；选定目标后再用
  msprof_perf_summary.py <PROF_GROUP> <ops_dir> --op-name=<exact> 正式解析。

 对比模式 (--compare):
  --compare              启用对比模式：未传 --case-manifest 时从 GOLDEN_PERF_REPORT.md 读 golden
                         + msprof 采集 PyPTO 算子计算加速比（既有行为）
  --output-dir=<dir>     算子目录（包含 test_{op}.py；既有模式还需 GOLDEN_PERF_REPORT.md）
  --device=N             指定 NPU 设备 ID；未指定时自动选择空闲卡
  --op-name=<name>       目标 kernel 的 Op Name（多 case 场景必须指定，否则会选到非目标 op）
  --repeats=N            重复采集次数（默认 1）
  --seed=N               传递给测试脚本的 PYPTO_PERF_SEED（默认 0）
  --retry=N              单 case 解析失败重试次数（默认 2）
  --keep-prof            保留 msprof 原始 PROF 目录（用于深度分析）
  （manifest 证据协议，传 --case-manifest 时启用）
  --case-arg=<flag>      逐 case CLI 选择器，例如 --case-id；脚本依次追加 <flag> <case-id>
  --case-env=<name>      逐 case 环境变量选择器，例如 PYPTO_PERF_CASE；与 --case-arg 二选一
  --case-manifest=<path> 可选；传后启用 manifest 证据协议（exact Op Name、逐 case/逐 repeat、
                         seed=42、Golden JSON exact-id join）。selector 精确命中时须输出唯一
                         一行 PYPTO_PERF_SELECTED_CASE=<case-id>，未知 id 非零退出。

 快速模式 (--quick):
  --quick                启用快速模式：每次 repeat 只采集 kernel 时间，不采集 7 个 aic-metrics
  --output-dir=<dir>     算子目录（包含 test_{op}.py；既有模式还需 GOLDEN_PERF_REPORT.md）
  --device=N             指定 NPU 设备 ID；未指定时自动选择空闲卡
  --op-name=<name>       目标 kernel 的 Op Name（多 case 场景必须指定，否则会选到非目标 op）
  --repeats=N            重复采集次数（默认 1）
  --seed=N               传递给测试脚本的 PYPTO_PERF_SEED（默认 0）
  --retry=N              单 case 解析失败重试次数（默认 2）
  --keep-prof            保留 msprof 原始 PROF 目录
  （manifest 证据协议，传 --case-manifest 时启用）--case-arg/--case-env/--case-manifest 同 compare 模式

 批量模式 (--batch):
  --batch                启用批量模式：扫描 base-dir 下所有子目录并行测试
  --base-dir=<dir>       包含多个算子输出子目录的根目录
  --max-jobs=N           最大并发数（默认 7；manifest 批限制 1..8）
  --device-start=N       起始 NPU 设备 ID（默认 1；既有流程按 1..7 轮转）
  （manifest 证据批：任一下属算子目录存在 PERFORMANCE_CASES.json，或传入下列参数时启用证据批）
  --op-name-map=<file>   异构 batch 专用 TSV：每行 <operator-directory><TAB><exact Op Name>
  --case-manifest-map=<file> 可选 TSV：每行 <operator-directory><TAB><manifest path>
  默认读取每个算子目录必需的 PERFORMANCE_CASES.json。
  同构 batch 可传全局 --op-name；异构 batch 必须传 --op-name-map。
  同时接受对比模式的 --case-arg/--case-env、--repeats、可选 --seed 以及 --retry、--keep-prof，
  并原样透传到每个算子。未启用证据批时保持既有行为（flock 锁 + ASCEND_RT_VISIBLE_DEVICES）。

脚本会按顺序用 msprof 采集 7 个 aic-metrics:
  PipeUtilization, ArithmeticUtilization, Memory, MemoryL0, MemoryUB,
  L2Cache, ResourceConflictRatio
所有 PROF 目录放在同一个 PROF_GROUP_<timestamp>/ 下，便于 msprof_perf_summary.py 汇总。

快速模式 (--quick) 每次 repeat 只获取 kernel 时间，不采集 7 个 aic-metrics。
EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --warm-up=*) WARM_UP="${1#*=}"; shift ;;
        --output=*)  OUTPUT_DIR="${1#*=}"; shift ;;
        --output-dir=*) OUTPUT_DIR_ARG="${1#*=}"; shift ;;
        --device=*)  DEVICE_ID="${1#*=}"; shift ;;
        --repeats=*) REPEATS="${1#*=}"; shift ;;
        --seed=*)    SEED="${1#*=}"; shift ;;
        --retry=*)   RETRY="${1#*=}"; shift ;;
        --keep-prof) KEEP_PROF=1; shift ;;
        --l2-clean)  L2_CLEAN=1; shift ;;
        --no-l2-clean) L2_CLEAN=0; shift ;;
        --op-name=*) OP_NAME="${1#*=}"; shift ;;
        --op-name-map=*) OP_NAME_MAP_FILE="${1#*=}"; shift ;;
        --case-arg=*) CASE_ARG="${1#*=}"; shift ;;
        --case-env=*) CASE_ENV="${1#*=}"; shift ;;
        --case-manifest=*) CASE_MANIFEST="${1#*=}"; shift ;;
        --case-manifest-map=*) CASE_MANIFEST_MAP_FILE="${1#*=}"; shift ;;
        --compare)   MODE="compare"; shift ;;
        --quick)     MODE="quick"; shift ;;
        --batch)     MODE="batch"; shift ;;
        --base-dir=*) BASE_DIR="${1#*=}"; shift ;;
        --max-jobs=*) MAX_JOBS="${1#*=}"; shift ;;
        --device-start=*) DEVICE_START="${1#*=}"; shift ;;
        -h|--help)   usage; exit 0 ;;
        --)          shift; APP_CMD=("$@"); break ;;
        -*)          echo "ERROR: unknown option $1" >&2; usage; exit 1 ;;
        *)           APP_CMD=("$@"); break ;;
    esac
done

# 检查 msprof
if ! command -v msprof >/dev/null 2>&1; then
    echo "ERROR: 未找到 msprof，请先 source CANN set_env.sh / setenv.bash" >&2
    exit 1
fi

# ============================================================================
# 标准模式：对单个可执行文件采集
# ============================================================================
run_standard() {
    if [[ ${#APP_CMD[@]} -eq 0 ]]; then
        echo "ERROR: 缺少可执行文件参数 (使用 -- 分隔 msprof 选项与 app 命令)" >&2
        usage
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"
    OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
    TS="$(date +%Y%m%d_%H%M%S)"
    GROUP_DIR="${OUTPUT_DIR}/PROF_GROUP_${TS}"
    mkdir -p "$GROUP_DIR"

    echo "=== msprof profiling ==="
    echo "App      : ${APP_CMD[*]}"
    echo "Output   : ${GROUP_DIR}"
    echo "Warm-up  : ${WARM_UP}"

    if [[ "$WARM_UP" -gt 0 ]]; then
        echo "--- Warm-up x${WARM_UP} ---"
        for i in $(seq 1 "$WARM_UP"); do
            echo "  warm-up ${i}/${WARM_UP}"
            "${APP_CMD[@]}" >/dev/null 2>&1 || true
        done
    fi

    METRICS=(PipeUtilization ArithmeticUtilization Memory MemoryL0 MemoryUB L2Cache ResourceConflictRatio)

    for M in "${METRICS[@]}"; do
        SUB="${GROUP_DIR}/PROF_${M}"
        mkdir -p "$SUB"
        echo "--- aic-metrics=${M} -> ${SUB} ---"
        l2_clean
        msprof --output="$SUB" \
               --ai-core=on \
               --aic-metrics="$M" \
               --task-time=on \
               --ascendcl=on \
               "${APP_CMD[@]}" >"${SUB}/msprof.log" 2>&1 || {
            echo "ERROR: msprof run failed for ${M}. See ${SUB}/msprof.log" >&2
            tail -20 "${SUB}/msprof.log" >&2 || true
            exit 1
        }
    done

    # 追加 sample-based 采集一份，用于逐核负载均衡分析
    SAMPLE_SUB="${GROUP_DIR}/PROF_Sample"
    mkdir -p "$SAMPLE_SUB"
    echo "--- sample-based (per-core load balance) -> ${SAMPLE_SUB} ---"
    l2_clean
    msprof --output="$SAMPLE_SUB" \
           --ai-core=on \
           --aic-metrics=PipeUtilization \
           --aic-mode=sample-based \
           --aic-freq=100 \
           --task-time=on \
           --ascendcl=on \
           "${APP_CMD[@]}" >"${SAMPLE_SUB}/msprof.log" 2>&1 || {
        echo "WARN: sample-based run failed, per-core analysis will be skipped. See ${SAMPLE_SUB}/msprof.log" >&2
        tail -20 "${SAMPLE_SUB}/msprof.log" >&2 || true
    }

    print_elapsed
    echo ""
    echo "=== Done. PROF group = ${GROUP_DIR} ==="
    echo "Next:"
    echo "  python3 ${SCRIPT_DIR}/msprof_perf_summary.py ${GROUP_DIR} <ops_dir>"
    echo "  # 性能证据协议：discovery 后可用 --list-op-names 列出精确 lowering 名称，再带 --op-name 解析"
}

# ============================================================================
# compare：case manifest 驱动 PyPTO target-kernel 采集；Golden 完整覆盖时计算默认目标
# ============================================================================
run_compare() {
    if [[ -z "$OUTPUT_DIR_ARG" ]]; then
        echo "ERROR: --compare 模式必须指定 --output-dir=<dir>" >&2
        usage
        exit 1
    fi
    if [[ -n "$CASE_MANIFEST" && ! -f "$CASE_MANIFEST" ]]; then
        echo "ERROR: --case-manifest 文件不存在: $CASE_MANIFEST" >&2
        exit 1
    fi

    OUT_DIR="$(cd "$OUTPUT_DIR_ARG" && pwd)"

    # 调用 msprof_perf_summary.py --compare（已融合 kernel_perf.py 功能）
    local extra_args=()
    [[ -n "$DEVICE_ID" ]] && extra_args+=("--device" "$DEVICE_ID")
    [[ -n "$OP_NAME" ]] && extra_args+=("--op-name" "$OP_NAME")
    [[ -n "$CASE_ARG" ]] && extra_args+=("--case-arg=${CASE_ARG}")
    [[ -n "$CASE_ENV" ]] && extra_args+=("--case-env=${CASE_ENV}")
    [[ -n "$CASE_MANIFEST" ]] && extra_args+=("--case-manifest=${CASE_MANIFEST}")
    [[ -n "${REPEATS:-}" ]] && extra_args+=("--repeats" "$REPEATS")
    extra_args+=("--seed" "$SEED")
    [[ -n "${RETRY:-}" ]] && extra_args+=("--retry" "$RETRY")
    [[ "${KEEP_PROF:-0}" == "1" ]] && extra_args+=("--keep-prof")
    [[ "${L2_CLEAN:-1}" == "1" ]] && extra_args+=("--l2-clean") || extra_args+=("--no-l2-clean")

    python3 "${SCRIPT_DIR}/msprof_perf_summary.py" --compare \
        --output-dir "$OUT_DIR" \
        --warmup "$WARM_UP" \
        "${extra_args[@]}"

    print_elapsed "$OUT_DIR"
}

# ============================================================================
# 快速模式：每次 repeat 只获取 kernel 时间，不采集 7 个 aic-metrics
# ============================================================================
run_quick() {
    if [[ -z "$OUTPUT_DIR_ARG" ]]; then
        echo "ERROR: --quick 模式必须指定 --output-dir=<dir>" >&2
        usage
        exit 1
    fi
    if [[ -n "$CASE_MANIFEST" && ! -f "$CASE_MANIFEST" ]]; then
        echo "ERROR: --case-manifest 文件不存在: $CASE_MANIFEST" >&2
        exit 1
    fi

    OUT_DIR="$(cd "$OUTPUT_DIR_ARG" && pwd)"

    # 调用 msprof_perf_summary.py --quick（只测时间，不采集 7 个 metrics）
    local extra_args=()
    [[ -n "$DEVICE_ID" ]] && extra_args+=("--device" "$DEVICE_ID")
    [[ -n "$OP_NAME" ]] && extra_args+=("--op-name" "$OP_NAME")
    [[ -n "$CASE_ARG" ]] && extra_args+=("--case-arg=${CASE_ARG}")
    [[ -n "$CASE_ENV" ]] && extra_args+=("--case-env=${CASE_ENV}")
    [[ -n "$CASE_MANIFEST" ]] && extra_args+=("--case-manifest=${CASE_MANIFEST}")
    [[ -n "${REPEATS:-}" ]] && extra_args+=("--repeats" "$REPEATS")
    extra_args+=("--seed" "$SEED")
    [[ -n "${RETRY:-}" ]] && extra_args+=("--retry" "$RETRY")
    [[ "${KEEP_PROF:-0}" == "1" ]] && extra_args+=("--keep-prof")
    [[ "${L2_CLEAN:-1}" == "1" ]] && extra_args+=("--l2-clean") || extra_args+=("--no-l2-clean")

    python3 "${SCRIPT_DIR}/msprof_perf_summary.py" --quick \
        --output-dir "$OUT_DIR" \
        --warmup "$WARM_UP" \
        "${extra_args[@]}"

    print_elapsed "$OUT_DIR"
}

# ============================================================================
# 批量模式：多 NPU 并行执行多个算子
# ============================================================================
run_batch() {
    if [[ -z "$BASE_DIR" ]]; then
        echo "ERROR: --batch 模式必须指定 --base-dir=<dir>" >&2
        usage
        exit 1
    fi

    # 双模式分发：任一带 manifest 输入即走 manifest 证据批；否则既有 batch。
    local use_manifest=0
    if [[ -n "$OP_NAME_MAP_FILE" || -n "$CASE_MANIFEST_MAP_FILE" || -n "$CASE_MANIFEST" ]]; then
        use_manifest=1
    else
        for dir in "$BASE_DIR"/*; do
            if [[ -f "$dir/PERFORMANCE_CASES.json" ]]; then
                use_manifest=1
                break
            fi
        done
    fi
    if [[ "$use_manifest" == "1" ]]; then
        run_batch_manifest
    else
        run_batch_legacy
    fi
}

# ============================================================================
# 批量模式（manifest 证据批）：先完整预检再启动
# ============================================================================
run_batch_manifest() {
    if ! [[ "$MAX_JOBS" =~ ^[1-8]$ ]]; then
        echo "ERROR: --max-jobs 必须是 1..8" >&2
        exit 1
    fi
    if ! [[ "$DEVICE_START" =~ ^[0-7]$ ]]; then
        echo "ERROR: --device-start 必须在当前脚本支持的设备 ID 范围 0..7 内" >&2
        exit 1
    fi
    if [[ -n "$OP_NAME" && -n "$OP_NAME_MAP_FILE" ]]; then
        echo "ERROR: batch 的 --op-name 与 --op-name-map 二选一" >&2
        return 1
    fi
    if [[ -n "$OP_NAME_MAP_FILE" && ! -f "$OP_NAME_MAP_FILE" ]]; then
        echo "ERROR: --op-name-map 文件不存在: $OP_NAME_MAP_FILE" >&2
        return 1
    fi
    if [[ -n "$CASE_MANIFEST" ]]; then
        echo "ERROR: batch 不接受全局 --case-manifest；请用每目录 PERFORMANCE_CASES.json 或 --case-manifest-map" >&2
        return 1
    fi
    if [[ -n "$CASE_MANIFEST_MAP_FILE" && ! -f "$CASE_MANIFEST_MAP_FILE" ]]; then
        echo "ERROR: --case-manifest-map 文件不存在: $CASE_MANIFEST_MAP_FILE" >&2
        return 1
    fi

    BASE_DIR="$(cd "$BASE_DIR" && pwd)"
    operator_dirs=()
    expected_operators=()
    operator_op_names=()
    operator_case_manifests=()
    job_pids=()

    # 第一遍只发现并验证全部任务；任何配置错误都必须在启动采集前失败。
    for dir in "$BASE_DIR"/*; do
        if [[ -d "$dir" ]] && compgen -G "$dir/test_*.py" >/dev/null; then
            folder_name=$(basename "$dir")
            task_op_name="$OP_NAME"
            task_case_manifest=""
            if [[ -n "$OP_NAME_MAP_FILE" ]]; then
                task_op_name=$(awk -F '\t' -v name="$folder_name" '$1 == name { print $2; count++ } END { if (count != 1) exit 1 }' "$OP_NAME_MAP_FILE") || {
                    echo "ERROR: --op-name-map 必须为 $folder_name 提供且只提供一个 exact Op Name" >&2
                    return 1
                }
            fi
            if [[ -z "$task_op_name" ]]; then
                echo "ERROR: batch 必须通过 --op-name 或 --op-name-map 为 $folder_name 提供 exact Op Name" >&2
                return 1
            fi
            if [[ -n "$CASE_MANIFEST_MAP_FILE" ]]; then
                task_case_manifest=$(awk -F '\t' -v name="$folder_name" '$1 == name { print $2; count++ } END { if (count != 1) exit 1 }' "$CASE_MANIFEST_MAP_FILE") || {
                    echo "ERROR: --case-manifest-map 必须为 $folder_name 提供且只提供一个 manifest 路径" >&2
                    return 1
                }
                if [[ "$task_case_manifest" != /* ]]; then
                    task_case_manifest="$(cd "$(dirname "$CASE_MANIFEST_MAP_FILE")" && pwd)/$task_case_manifest"
                fi
                if [[ ! -f "$task_case_manifest" ]]; then
                    echo "ERROR: $folder_name 的 case manifest 不存在: $task_case_manifest" >&2
                    return 1
                fi
            elif [[ -f "$dir/PERFORMANCE_CASES.json" ]]; then
                task_case_manifest="$dir/PERFORMANCE_CASES.json"
            else
                echo "ERROR: $folder_name 缺少必需的 PERFORMANCE_CASES.json" >&2
                return 1
            fi
            # 这里只解析 schema/来源并校验可选 Golden exact-id join；所有算子均通过后才启动任何 msprof。
            case_source_args=(
                --validate-case-source
                --output-dir "$dir"
            )
            case_source_args+=("--case-manifest=$task_case_manifest")
            if ! python3 "${SCRIPT_DIR}/msprof_perf_summary.py" "${case_source_args[@]}"; then
                echo "ERROR: $folder_name 的性能 case 来源预检失败" >&2
                return 1
            fi
            operator_dirs+=("$dir")
            expected_operators+=("$folder_name")
            operator_op_names+=("$task_op_name")
            operator_case_manifests+=("$task_case_manifest")
        fi
    done

    if [[ ${#expected_operators[@]} -eq 0 ]]; then
        echo "ERROR: batch 根目录下没有包含 test_*.py 的算子目录" >&2
        return 1
    fi

    COLLECTION_ID="batch_$(date +%Y%m%d_%H%M%S)_$$"
    export PYPTO_PERF_COLLECTION_ID="$COLLECTION_ID"
    LOG_FILE="${BASE_DIR}/batch_performance_${COLLECTION_ID}.log"
    : > "$LOG_FILE"

    device_id=$DEVICE_START
    for task_index in "${!operator_dirs[@]}"; do
            dir="${operator_dirs[$task_index]}"
            folder_name="${expected_operators[$task_index]}"
            task_op_name="${operator_op_names[$task_index]}"
            task_case_manifest="${operator_case_manifests[$task_index]}"
            # 控制总并发数
            while [[ $(jobs -r | wc -l) -ge $MAX_JOBS ]]; do
                sleep 0.5
            done

            echo ">>> 分配设备 $device_id 并启动算子: $folder_name" | tee -a "$LOG_FILE"

            # 启动后台任务
            (
                unset ASCEND_RT_VISIBLE_DEVICES
                export TILE_FWK_DEVICE_ID=$device_id

                # 调用对比模式；公共证据参数必须与单算子采集完全一致。
                batch_args=(
                    --compare
                    "--output-dir=$dir"
                    "--warm-up=$WARM_UP"
                    "--device=$device_id"
                    "--repeats=$REPEATS"
                    "--retry=$RETRY"
                )
                [[ -n "$SEED" ]] && batch_args+=("--seed=$SEED")
                batch_args+=("--op-name=$task_op_name")
                [[ -n "$task_case_manifest" ]] && batch_args+=("--case-manifest=$task_case_manifest")
                [[ -n "$CASE_ARG" ]] && batch_args+=("--case-arg=$CASE_ARG")
                [[ -n "$CASE_ENV" ]] && batch_args+=("--case-env=$CASE_ENV")
                [[ "$KEEP_PROF" == "1" ]] && batch_args+=(--keep-prof)
                if [[ "${L2_CLEAN:-1}" == "1" ]]; then batch_args+=(--l2-clean); else batch_args+=(--no-l2-clean); fi
                bash "$0" "${batch_args[@]}" >> "$LOG_FILE" 2>&1

                echo ">>> 算子 $folder_name 在设备 $device_id 上执行完毕" >> "$LOG_FILE"
            ) &
            job_pids+=("$!")

            device_id=$(( (device_id + 1) % 8 ))
    done

    batch_failed=0
    for job_pid in "${job_pids[@]}"; do
        if ! wait "$job_pid"; then
            batch_failed=1
        fi
    done
    echo "所有并行任务已执行完毕，日志已保存至 $LOG_FILE"

    # 生成批量汇总报告
    summary_args=(
        --batch "$BASE_DIR"
        --collection-id "$COLLECTION_ID"
        --output-md "${BASE_DIR}/batch_report_${COLLECTION_ID}.md"
        --output-json "${BASE_DIR}/batch_summary_${COLLECTION_ID}.json"
    )
    for operator_name in "${expected_operators[@]}"; do
        summary_args+=(--expect-operator "$operator_name")
    done
    if ! python3 "${SCRIPT_DIR}/msprof_perf_summary.py" "${summary_args[@]}"; then
        batch_failed=1
    fi

    print_elapsed
    if [[ "$batch_failed" == "1" ]]; then
        echo "ERROR: 一个或多个算子未生成完整的本轮性能证据" >&2
        return 1
    fi
}


# ============================================================================
# 批量模式（既有流程）：flock 锁 + ASCEND_RT_VISIBLE_DEVICES 轮转
# ============================================================================
run_batch_legacy() {
    BASE_DIR="$(cd "$BASE_DIR" && pwd)"
    LOG_FILE="${BASE_DIR}/batch_performance.log"
    : > "$LOG_FILE"

    LOCK_DIR="/tmp/ascend_locks_$$"  # kb-integrity: allow-path (a device lock shared across runs; per-cwd would not mutually exclude)
    mkdir -p "$LOCK_DIR"

    device_id=$DEVICE_START

    for dir in "$BASE_DIR"/*; do
        if [[ -d "$dir" ]]; then
            folder_name=$(basename "$dir")

            # 控制总并发数
            while [[ $(jobs -r | wc -l) -ge $MAX_JOBS ]]; do
                sleep 0.5
            done

            # 寻找空闲设备
            while true; do
                lock_file="$LOCK_DIR/device_${device_id}.lock"
                if ! ( set -o noclobber; flock -n 200 ) 200>"$lock_file" 2>/dev/null; then
                    device_id=$(( (device_id % 7) + 1 ))
                    sleep 0.2
                else
                    break
                fi
            done

            echo ">>> 锁定设备 $device_id 并启动算子: $folder_name" | tee -a "$LOG_FILE"

            # 启动后台任务
            (
                lock_file="$LOCK_DIR/device_${device_id}.lock"
                exec 200>"$lock_file"
                flock 200

                export ASCEND_RT_VISIBLE_DEVICES=$device_id

                # 调用对比模式
                bash "$0" --compare --output-dir="$dir" --warm-up="$WARM_UP" --device="$device_id" 2>&1 | \
                    grep -v "tiling struct \[MC2MatmulV3TilingData\] is conflict" | \
                    grep -v "tiling struct \[TileInfo\] is conflict" \
                    >> "$LOG_FILE"

                echo ">>> 算子 $folder_name 在设备 $device_id 上执行完毕" >> "$LOG_FILE"
            ) &

            device_id=$(( (device_id % 7) + 1 ))
        fi
    done

    wait
    rm -rf "$LOCK_DIR"

    echo "所有并行任务已执行完毕，日志已保存至 $LOG_FILE"

    # 生成批量汇总报告
    python3 "${SCRIPT_DIR}/msprof_perf_summary.py" --batch "$BASE_DIR" \
        --output-md "${BASE_DIR}/batch_report.md" \
        --output-json "${BASE_DIR}/batch_summary.json"

    print_elapsed "$BASE_DIR"
}

# ============================================================================
# 主入口
# ============================================================================
case "$MODE" in
    standard)
        run_standard
        ;;
    compare)
        run_compare
        ;;
    quick)
        run_quick
        ;;
    batch)
        run_batch
        ;;
    *)
        echo "ERROR: 未知模式: $MODE" >&2
        usage
        exit 1
        ;;
esac
