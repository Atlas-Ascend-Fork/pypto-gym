# Qwen3-1.7B NPU 迁移说明 (transformers 4.51.0)

| 字段 | 说明 |
|------|------|
| HuggingFace | Qwen/Qwen3-1.7B |
| 权重目录 | `{model_weight_dir}`（如 `/path/to/models/Qwen3-1.7B`） |
| 代码来源 | transformers 4.51.0 内置 `/models/qwen3/`，import 已修复为 `from transformers.xxx` |
| transformers版本 | 4.51.0 |
| 代码位置 | modeling_qwen3.py, configuration_qwen3.py（根目录，auto_map指向根目录） |
| 修改内容 | 最小化：仅 `Qwen3Attention.forward()` 增加 6 行 PyPTO dispatch（prefill 用 PTO，decode 退原生） |
| 运行命令 | `python3 scripts/ask_Qwen3-1.7B.py --device 0 --prompt "你好"` |

## 环境信息

| 组件 | 版本 |
|------|------|
| torch | 2.9.0 |
| torch_npu | 2.9.0.post2 |
| torchvision | 0.24.0 |
| transformers | 4.51.0 |
| CANN | 9.0.0 |
| NPU | Ascend 910B2 |
| accelerate | 1.10.1 |
| safetensors | 0.6.2 |
| tokenizers | 0.21.4 |
| numpy | 2.2.6 |

## 下载模型

```bash
python3 cannbot-skills/model/pypto-fused-op-integration/scripts/download_hf_model.py \
    --model-id Qwen/Qwen3-1.7B \
    --output-dir {model_weight_dir}
```

## PYPTO入网适配

```bash
bash cannbot-skills/model/pypto-fused-op-integration/scripts/restore_model_patch.sh \
    {model_weight_dir} qwen3_1_7b
```

## 运行命令

```bash
# 基线推理（原生 PyTorch）
python3 scripts/ask_Qwen3-1.7B.py --device 0 --prompt "你好"

# PyPTO 融合推理
python3 scripts/ask_Qwen3-1.7B.py --device 0 --prompt "你好" --use-pto
```

## 性能对比

| 模式 | 命令 | 模型加载 | 推理耗时 | 吞吐 | 峰值显存 |
|------|------|---------|---------|------|---------|
| baseline | `python3 scripts/ask_Qwen3-1.7B.py --device 0 --prompt "你好"` | 2.5s | 7.2s | 13.9 tok/s | 4470 MB |
| pto | `python3 scripts/ask_Qwen3-1.7B.py --device 0 --prompt "你好" --use-pto` | 2.4s | 8.3s | 12.0 tok/s | 4470 MB |

> 单算子替换时 PTO 比基线慢 ~15%（JIT 首编 + kernel launch 开销），收益来自多算子融合。

## 目录结构

```
Qwen3-1.7B/
├── modeling_qwen3.py              # 模型实现（4.51.0内置 + 6行PTO dispatch）
├── configuration_qwen3.py         # 配置实现
├── config.json                    # 模型配置（auto_map指向根目录）
├── scripts/
│   ├── ask_Qwen3-1.7B.py          # 推理脚本（支持--use-pto）
│   └── README.md                  # 本文档
├── qwen3_pto_kernels/             # PyPTO融合算子模块
│   ├── __init__.py                # USE_PTO_ROPE开关 + qk_rope_wrapper
│   └── rope/
│       ├── __init__.py
│       ├── README.md
│       └── rrms_norm_rope_impl.py # Q/K RMSNorm+RoPE融合kernel
├── model-*.safetensors            # 模型权重
└── tokenizer相关文件              # 分词器
```

## PyPTO 融合范围

| 算子 | 融合? | 说明 |
|------|:---:|------|
| Q/K RMSNorm + RoPE | ✅ | prefill (S>1) 走 PyPTO kernel，decode (S=1) 退原生 |
| Q/K/V projection | ❌ | PyTorch Linear |
| Attention | ❌ | eager/sdpa 原生 |
| MLP | ❌ | PyTorch Linear |
| 整体 RMSNorm | ❌ | PyTorch Qwen3RMSNorm |

## 开关变量

`qwen3_pto_kernels.USE_PTO_ROPE` (bool, 默认 False)

## 归档映射

| 来源 (`{model_weight_dir}/`) | 目标 (`pypto-gym/`) |
|---|---|
| `scripts/` | `modeling/transformers/qwen3_1_7b/` |
| `config.json`, `modeling_qwen3.py`, `configuration_qwen3.py` | `src/pypto_gym/transformers/qwen3_1_7b/` |
| `qwen3_pto_kernels/` | `ops/pypto_tensor/qwen3_1_7b/` |
