# Phi-3-mini-4k-instruct NPU 迁移说明

## 模型信息

| 字段 | 值 |
|------|-----|
| HuggingFace | microsoft/Phi-3-mini-4k-instruct |
| 模型名 | phi_3_mini_4k_instruct |
| 代码来源 | trust_remote_code (HuggingFace仓库自带 + pypto-gym 补丁) |
| 参数量 | ~3.8B (BF16) |
| 精度 | FP16 (运行时转) |

## 环境信息

| 字段 | 版本 |
|------|------|
| conda env | `phi_3_mini_4k_instruct` (clone from `qwen3-1.7b`) |
| Python | 3.11.15 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| transformers | 5.12.0 |
| pypto | 0.2.1 |
| CANN | 9.0.0 |
| NPU | Ascend 910B |

## 环境准备

```bash
conda activate phi_3_mini_4k_instruct
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
export TILE_FWK_DEVICE_ID=0
export PYTHONPATH=$PWD/../../../src:$PYTHONPATH
```

## 下载模型

```bash
export MODEL_PATH=/path/to/Phi-3-mini-4k-instruct

python3 cannbot-skills/model/pypto-fused-op-integration/scripts/download_hf_model.py \
    --model-id microsoft/Phi-3-mini-4k-instruct \
    --output-dir "$MODEL_PATH"
```

## PYPTO入网适配

```bash

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
bash $REPO_ROOT/cannbot-skills/model/pypto-fused-op-integration/scripts/restore_model_patch.sh \
    "$MODEL_PATH" phi_3_mini_4k_instruct
```

脚本执行逻辑

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 备份原始 `modeling_phi3.py` | 保存为 `modeling_phi3.py.hf_orig` |
| 2 | 替换为华为修改版 | 注入 PyPTO RMSNorm 分派逻辑 |
| 3 | 写入 `pto_kernels/` | RMSNorm 融合算子 |
| 4 | 确保 `config.json` 含 `auto_map` | `AutoModelForCausalLM` → `modeling_phi3.Phi3ForCausalLM` |
| 5 | 清除 HF 模块缓存 | 防止旧代码残留 |

## 运行命令

```bash
# 基础推理
python3 scripts/ask_Phi-3-mini-4k-instruct.py --model-path "$MODEL_PATH" --device 0 --prompt "你好"

# PyPTO 融合算子模式
python3 scripts/ask_Phi-3-mini-4k-instruct.py --model-path "$MODEL_PATH" --device 0 --use_pypto --prompt "你好"

# 基准测试
PHI3_MODEL_PATH="$MODEL_PATH" bash bench_Phi-3-mini-4k-instruct.sh
```

> `--model-path` 参数优先级高于 `PHI3_MODEL_PATH` 环境变量。

## 性能对比

| 模式 | 推理耗时 | 吞吐 | 峰值显存 |
|------|---------|------|---------|
| baseline | 4.632s | 10.8 tok/s | 7327.8 MB |
| pto | 12.209s | 4.1 tok/s | 7327.8 MB |

> 测试条件：Prompt "你好"，output_length=50，device=4。单算子替换时 PTO 比基线慢 2-3x 属正常（JIT 首编 + kernel launch 开销），收益来自多算子融合。

## 归档映射

| 来源 (`{model_dir}/`) | 目标 (`{pypto_gym_repo}/`) |
|---|---|
| `scripts/` | `modeling/transformers/phi_3_mini_4k_instruct/` |
| `modeling_phi3.py` / `configuration_phi3.py` | `src/pypto_gym/transformers/phi_3_mini_4k_instruct/` |
| `config.json` | `src/pypto_gym/transformers/phi_3_mini_4k_instruct/` |
| `pto_kernels/` | `ops/pypto_tensor/phi_3_mini_4k_instruct/` |
| `pto_kernels/rms_norm/test/` | `tests/ops/phi_3_mini_4k_instruct/` |
