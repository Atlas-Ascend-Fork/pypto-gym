#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
#
# restore_model_patch.sh — PYPTO入网适配: 将 HF 下载的原始模型替换为华为修改版
#
# 用法: bash restore_model_patch.sh <model_weight_dir> <model_name>
#
# 示例: bash restore_model_patch.sh /data/models/Phi-3-mini-4k-instruct phi_3_mini_4k_instruct
#
# 前置: 已通过 download_hf_model.py 下载 HF 模型到 <model_weight_dir>
#
# 还原内容:
#   1. 替换 modeling_*.py 为华为修改版 (含 PyPTO 算子注入)
#   2. 写入 {model_name}_pto_kernels/ (PyPTO 融合算子, 若 ops/pypto_tensor/{model_name}/ 存在)
#   3. 确保 config.json 含 auto_map
#   4. 清除 HF 模块缓存

set -e

MODEL_PATH="${1:?用法: bash restore_model_patch.sh <model_weight_dir> <model_name>}"
MODEL_NAME="${2:?用法: bash restore_model_patch.sh <model_weight_dir> <model_name>}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: MODEL_PATH 不存在: $MODEL_PATH"
    echo "请先下载模型: python3 download_hf_model.py --model-id ... --output-dir $MODEL_PATH"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

TRANSFORMERS_DIR="$REPO_ROOT/src/pypto_gym/transformers/$MODEL_NAME"
PYPTO_OPS_DIR="$REPO_ROOT/ops/pypto_tensor/$MODEL_NAME"

echo "=== PyPTO 补丁还原: $MODEL_NAME ==="
echo "MODEL_PATH:  $MODEL_PATH"
echo ""

# ---- Step 1: 替换 modeling 代码 ----
if [ ! -d "$TRANSFORMERS_DIR" ]; then
    echo "[跳过] 未找到 $TRANSFORMERS_DIR — 该模型无需代码覆盖"
else
    for src in "$TRANSFORMERS_DIR"/modeling_*.py "$TRANSFORMERS_DIR"/configuration_*.py; do
        [ -f "$src" ] || continue
        dst_name=$(basename "$src")
        dst="$MODEL_PATH/$dst_name"
        if [ ! -f "${dst}.hf_orig" ]; then
            [ -f "$dst" ] && cp "$dst" "${dst}.hf_orig"
        fi
        cp "$src" "$dst"
        echo "[copy] $dst_name (备份 → ${dst_name}.hf_orig)"
    done
fi

# ---- Step 2: 写入 pto_kernels/ ----
if [ -d "$PYPTO_OPS_DIR" ]; then
    rm -rf "$MODEL_PATH/${MODEL_NAME}_pto_kernels"
    cp -r "$PYPTO_OPS_DIR" "$MODEL_PATH/${MODEL_NAME}_pto_kernels"
    find "$MODEL_PATH/${MODEL_NAME}_pto_kernels" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    echo "[copy] pto_kernels/ ($(find "$PYPTO_OPS_DIR" -name '*.py' | wc -l) py files)"
else
    echo "[跳过] 未找到 $PYPTO_OPS_DIR — 融合算子稍后手动创建"
fi

# ---- Step 3: 确保 config.json 有 auto_map ----
if [ -f "$MODEL_PATH/config.json" ]; then
    python3 -c "
import json
cfg = json.load(open('$MODEL_PATH/config.json'))
cfg.setdefault('auto_map', {})
json.dump(cfg, open('$MODEL_PATH/config.json', 'w'), indent=2)
print('[done] auto_map 已确认')
" 2>&1
fi

# ---- Step 4: 清除 HF 模块缓存 ----
HF_CACHE="$HOME/.cache/huggingface/modules/transformers_modules"
if [ -d "$HF_CACHE" ]; then
    pattern=$(echo "$MODEL_NAME" | tr '_' '|')
    for d in "$HF_CACHE"/*/; do
        if basename "$d" | grep -qiE "$pattern"; then
            rm -rf "$d"
            echo "[cache] 已清除: $(basename "$d")"
        fi
    done
fi

echo ""
echo "=== 补丁还原完成 ==="
echo "  scripts/       → 推理脚本 (需另行部署)"
echo "  modeling_*.py  → 华为修改版"
echo "  pto_kernels/   → PyPTO 融合算子"
echo ""
