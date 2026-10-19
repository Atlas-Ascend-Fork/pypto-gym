# Qwen3-VL-8B-Instruct-Unredacted-MAX 迁移说明

## 基本信息

| 项目 | 值 |
|------|------|
| HuggingFace | [prithivMLmods/Qwen3-VL-8B-Instruct-Unredacted-MAX](https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-Unredacted-MAX) |
| 权重目录 | `$MODEL_PATH` |
| 代码来源 | transformers 包（内置，model_type=qwen3_vl） |
| transformers 版本 | 5.12.0 |
| HF 要求 | 无显式版本要求；需 `transformers >= 5.12.0`（依赖 `RopeParameters`、`Qwen3VLTextConfig` 等 5.x API） |
| 运行命令 | `python3 scripts/ask_Qwen3-VL-8B-Instruct-Unredacted-MAX.py --device 5` |

## 目录结构

```
$MODEL_PATH/
├── config.json                      # auto_map 指向根目录 modeling/configuration
├── model.safetensors                # 模型权重
├── modeling_qwen3_vl.py             # 华为修改版（含 PyPTO RMSNorm 注入）
├── modeling_qwen3_vl.py.hf_orig     # 原始 HF 备份
├── configuration_qwen3_vl.py        # 华为修改版（fix_imports）
├── configuration_qwen3_vl.py.hf_orig
├── tokenizer.json / tokenizer_config.json
├── vocab.json / merges.txt
├── pto_kernels/                     # PyPTO 融合算子库
│   ├── __init__.py                  # USE_PTO_RMS_NORM 开关
│   └── rms_norm/
│       ├── README.md
│       └── rms_norm_impl.py
└── scripts/
    ├── ask_Qwen3-VL-8B-Instruct-Unredacted-MAX.py
    ├── bench_Qwen3-VL-8B-Instruct-Unredacted-MAX.sh
    └── README.md
```

## 下载方式

```bash
# 下载模型权重
python3 cannbot-skills/model/pypto-fused-op-integration/scripts/download_hf_model.py \
    --model-id prithivMLmods/Qwen3-VL-8B-Instruct-Unredacted-MAX \
    --output-dir "$MODEL_PATH"
```

## PYPTO入网适配

```bash
# 下载后自动注入 PyPTO 融合代码和算子
bash cannbot-skills/model/pypto-fused-op-integration/scripts/restore_model_patch.sh \
    "$MODEL_PATH" qwen3_vl_8b_instruct_unredacted_max
```

脚本自动完成：备份 HF 原始 modeling 文件 → 替换为华为修改版 → 写入 `pto_kernels/` → 确保 `auto_map` → 清除 HF 缓存。

## 使用方法

```bash
# Baseline（原始 PyTorch 实现，fallback 到 .pow(2) → 需 cann_pow_patch）
python3 scripts/ask_Qwen3-VL-8B-Instruct-Unredacted-MAX.py --device 5 --prompt "你好" --output_length 50

# PyPTO（RMSNorm 替换为 PyPTO 融合算子）
python3 scripts/ask_Qwen3-VL-8B-Instruct-Unredacted-MAX.py --device 5 --prompt "你好" --output_length 50 --use_pypto
```

## 环境信息

| 组件 | 版本 |
|------|------|
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| transformers | 5.12.0 |
| CANN | 9.0.0 |
| NPU | Ascend 910B2 |
| Python | 3.11.15 |
| conda 环境 | `qwen3_vl_8b_instruct_unredacted_max` |

## PyPTO 融合范围

| 算子 | 融合状态 | Fallback | 注入点 |
|------|:---:|----------|--------|
| RMSNorm | ✅ PyPTO | PyTorch fp32 | `Qwen3VLTextRMSNorm.forward()` — `sys.modules.get("pto_kernels")` |
| Attention (Q/K/V/O + RoPE) | 未融合 | eager attention | — |
| MLP (SwiGLU) | 未融合 | PyTorch `nn.Linear` | — |

## 性能对比

| 模式 | 命令 | 模型加载 | 推理耗时 | 吞吐 | 峰值显存 |
|------|------|---------|---------|------|---------|
| baseline | `python3 scripts/ask_Qwen3-VL-8B-Instruct-Unredacted-MAX.py --prompt "你好" --device 5 --output_length 50` | 4.1s | 5.3s | 9.5 tok/s | 16735 MB |
| pto | `python3 scripts/ask_Qwen3-VL-8B-Instruct-Unredacted-MAX.py --prompt "你好" --device 5 --output_length 50 --use_pypto` | 3.4s | 5.1s | 9.9 tok/s | 16765 MB |

> 单算子 RMSNorm 替换，PTO 与基线吞吐持平（差异 < 5% 属测量噪声）。收益来自与相邻算子（pre-attn RMSNorm + QKV 投影）融合后的复合算子。

## 代码修改说明

`modeling_qwen3_vl.py`（华为修改版）相对于 `transformers/5.12.0/models/qwen3_vl/modeling_qwen3_vl.py`（原始 HF）的改动：

1. **导入修复**（`fix_imports.py`）：`from ...xxx` → `from transformers.xxx`
2. **`import sys`**（全局注入通道）
3. **RMSNorm 分发**（`Qwen3VLTextRMSNorm.forward()` 函数开头 4 行）：

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # PyPTO RMS Norm injection
    pto_kernels = sys.modules.get("pto_kernels")
    if pto_kernels is not None and getattr(pto_kernels, "USE_PTO_RMS_NORM", False):
        return pto_kernels.rms_norm_wrapper(hidden_states, self.weight, self.variance_epsilon)

    # ... 原始 PyTorch fallback（保留不变，不传 --use_pypto 时不走 pto 路径）
```

`configuration_qwen3_vl.py` 仅 `fix_imports.py` 修复导入路径，无 PyPTO 逻辑修改。

## 还原重建

按 SKILL.md 步骤 27，从 pypto-gym 归档 + 已下载权重重建可运行模型：

```bash
# ① 环境对齐（见上方环境信息表格）
conda create -n qwen3_vl_8b_instruct_unredacted_max python=3.11 -y
conda activate qwen3_vl_8b_instruct_unredacted_max
pip install torch==2.9.0 torch_npu==2.9.0.post2 --trusted-host pypi.org --trusted-host files.pythonhosted.org
pip install transformers==5.12.0 accelerate --trusted-host pypi.org --trusted-host files.pythonhosted.org

# ② 反向归档映射拷贝
#    modeling/transformers/qwen3_vl_8b_instruct_unredacted_max/          → scripts/
#    src/pypto_gym/transformers/qwen3_vl_8b_instruct_unredacted_max/     → 根目录 (.py) + config.json
#    ops/pypto_tensor/qwen3_vl_8b_instruct_unredacted_max/  → pto_kernels/
#    或直接运行 restore_model_patch.sh（推荐）：
bash cannbot-skills/model/pypto-fused-op-integration/scripts/restore_model_patch.sh \
    "$MODEL_PATH" qwen3_vl_8b_instruct_unredacted_max

# ③ 验证
python3 scripts/ask_Qwen3-VL-8B-Instruct-Unredacted-MAX.py --prompt "你好" --device 5           # baseline
python3 scripts/ask_Qwen3-VL-8B-Instruct-Unredacted-MAX.py --prompt "你好" --device 5 --use_pypto  # PTO
```

## 归档映射

| 来源（`$MODEL_PATH/`） | 目标（`pypto-gym/`） |
|---|---|
| `scripts/` | `modeling/transformers/qwen3_vl_8b_instruct_unredacted_max/` |
| `config.json` | `src/pypto_gym/transformers/qwen3_vl_8b_instruct_unredacted_max/` |
| `modeling_qwen3_vl.py`、`configuration_qwen3_vl.py` | `src/pypto_gym/transformers/qwen3_vl_8b_instruct_unredacted_max/` |
| `pto_kernels/` | `ops/pypto_tensor/qwen3_vl_8b_instruct_unredacted_max/` |
