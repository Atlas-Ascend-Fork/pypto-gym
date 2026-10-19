# GutenOCR-3B 整网集成

## HuggingFace

rootsautomation/GutenOCR-3B (https://huggingface.co/rootsautomation/GutenOCR-3B)

## 权重目录

环境变量 `GUTENOCR_MODEL_PATH` 或 `--model-path` 参数指定。

## 代码来源

trust_remote_code（HuggingFace 自带 Qwen2.5-VL 代码） + pypto-gym 补丁（PyPTO RMSNorm / MRoPE 注入）。

## 运行命令

```bash
export GUTENOCR_MODEL_PATH=/path/to/gutenocr_3b
export CANN_POW_PATCH_PATH=/path/to/cann_pow_patch

# Baseline
python3 scripts/ask_gutenocr_3b.py --prompt "你好" --device 0

# PTO 模式
python3 scripts/ask_gutenocr_3b.py --prompt "你好" --device 0 --use_pto
```

## 下载模型

```bash
python3 cannbot-skills/model/pypto-fused-op-integration/scripts/download_hf_model.py \
    --model-id rootsautomation/GutenOCR-3B \
    --output-dir ${GUTENOCR_MODEL_PATH}
```

## PYPTO入网适配

```bash
bash cannbot-skills/model/pypto-fused-op-integration/scripts/restore_model_patch.sh \
    ${GUTENOCR_MODEL_PATH} gutenocr_3b
```

## 环境信息

| 组件 | 版本 |
|------|------|
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| transformers | 5.8.1 |
| CANN | Ascend 910B |

## 归档映射

| 来源 (`models/gutenocr_3b/`) | 目标 (`pypto-gym/`) |
|---|---|
| `scripts/` | `modeling/transformers/gutenocr_3b/` |
| `config.json` | `src/pypto_gym/transformers/gutenocr_3b/` |
| `modeling_qwen2_5_vl.py`, `configuration_qwen2_5_vl.py` | `src/pypto_gym/transformers/gutenocr_3b/` |
| `pto_kernels/` | `ops/pypto_tensor/gutenocr_3b/` |

## 性能对比

| 模式 | 命令 | 推理耗时 | 吞吐 | 峰值显存 |
|------|------|---------|------|---------|
| baseline | `--prompt "你好" --device 0 --output_length 50` | 4.51s | 11.1 tok/s | 7182 MB |
| pto (RMSNorm+MRoPE) | `--prompt "你好" --device 0 --output_length 50 --use_pto` | 5.85s | 8.6 tok/s | 7182 MB |

> 单算子替换时 PTO 比基线慢属正常（kernel launch 开销 > 单算子收益）。收益来自多算子融合。
