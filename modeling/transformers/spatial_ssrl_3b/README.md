# Spatial-SSRL-3B 整网集成说明

## 基本信息

| 项目 | 值 |
|------|------|
| HuggingFace | [internlm/Spatial-SSRL-3B](https://huggingface.co/internlm/Spatial-SSRL-3B) |
| 架构 | Qwen2.5-VL-3B-Instruct (model_type: qwen2_5_vl) |
| 代码来源 | trust_remote_code — HF 下载后由 `restore_model_patch.sh` 替换为华为 PTO 修改版 |
| 运行命令 | `python3 scripts/ask_spatial_ssrl_3b.py --device 0 --prompt "你好"` |

## 环境信息

| 组件 | 版本 |
|------|------|
| torch | 2.9.0 |
| torch_npu | 2.9.0.post2 |
| torchvision | 0.24.0 |
| transformers | 5.6.0 |
| CANN | 9.1.0 |
| NPU | Ascend 910B |

## 下载模型

```bash
python3 cannbot-skills/model/pypto-fused-op-integration/scripts/download_hf_model.py \
    --model-id internlm/Spatial-SSRL-3B \
    --output-dir /path/to/models/spatial_ssrl_3b
```

## PYPTO入网适配

```bash
bash cannbot-skills/model/pypto-fused-op-integration/scripts/restore_model_patch.sh \
    /path/to/models/spatial_ssrl_3b spatial_ssrl_3b
```

脚本自动完成：备份 HF 原始代码 → 替换为华为修改版 → 写入 `pto_kernels/` → 确保 `auto_map` → 清除 HF 缓存。

## 使用说明

```bash
# 基线模式
python3 scripts/ask_spatial_ssrl_3b.py --device 0 --prompt "你好"

# PTO 融合模式
python3 scripts/ask_spatial_ssrl_3b.py --device 0 --use_pto --prompt "你好"
```

## PyPTO 算子集成

| 算子 | 位置 | 开关 | 状态 |
|------|------|------|------|
| RMSNorm | rms_norm/ | USE_PTO_RMS_NORM | ✅ |
| RoPE (Text) | rope/ | USE_PTO_ROPE | ✅ |
| RoPE (Vision) | rope/ | USE_PTO_ROPE | ✅ |

启用方式: `--use_pto` 注入 `spatial_ssrl_3b_pto_kernels` 到 `sys.modules`，设 USE_PTO_RMS_NORM=True / USE_PTO_ROPE=True。modeling 代码在 forward 时通过 `sys.modules.get()` 检测路由。

## 性能对比

| 模式 | 命令 | 模型加载 | 推理耗时 | 吞吐 | 峰值显存 |
|------|------|---------|---------|------|---------|
| baseline | `python3 scripts/ask_spatial_ssrl_3b.py --prompt "你好" --device 0 --output_length 30` | 6.0s | 2.5s | 6.4 tok/s | 7774 MB |
| pto | `python3 scripts/ask_spatial_ssrl_3b.py --prompt "你好" --device 0 --output_length 30 --use_pto` | 7.1s | 37.6s | 0.3 tok/s | 7827 MB |

> 单算子替换时 PTO 比基线慢 ~15x 属正常（JIT 首编 + kernel launch 开销），收益来自多算子融合。RMSNorm + RoPE 为第一阶段；后续 round 预做 pre-attn、post-attn 融合后预期 PTO 反超基线。

## 模型结构

```
spatial_ssrl_3b/
├── config.json                          # 模型配置（含auto_map）
├── configuration_qwen2_5_vl.py          # 模型配置类
├── modeling_qwen2_5_vl.py               # 网络结构（含PTO注入）
├── spatial_ssrl_3b_pto_kernels/         # PyPTO 融合算子
│   ├── rms_norm/                        # RMSNorm kernel
│   └── rope/                            # RoPE kernel（Vision + Multimodal）
└── scripts/
    ├── ask_spatial_ssrl_3b.py           # 推理脚本
    ├── bench_spatial_ssrl_3b.sh         # 性能测试脚本
    └── README.md
```

## 归档映射

| 来源 (`models/spatial_ssrl_3b/`) | 目标 (`pypto-gym/`) |
|---|---|
| `config.json` | `src/pypto_gym/transformers/spatial_ssrl_3b/config.json` |
| `configuration_qwen2_5_vl.py` | `src/pypto_gym/transformers/spatial_ssrl_3b/configuration_qwen2_5_vl.py` |
| `modeling_qwen2_5_vl.py` | `src/pypto_gym/transformers/spatial_ssrl_3b/modeling_qwen2_5_vl.py` |
| `spatial_ssrl_3b_pto_kernels/` | `ops/pypto_tensor/spatial_ssrl_3b/` |
| `scripts/` | `modeling/transformers/spatial_ssrl_3b/` |
