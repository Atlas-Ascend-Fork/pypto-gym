# Spatial-SSRL-3B — Model Definition Archive

HuggingFace `internlm/Spatial-SSRL-3B` 的华为修改版模型定义，含 PyPTO RMSNorm + RoPE 算子注入。

## 基本信息

| 项目 | 值 |
|------|------|
| HuggingFace | [internlm/Spatial-SSRL-3B](https://huggingface.co/internlm/Spatial-SSRL-3B) |
| 架构 | Qwen2.5-VL-3B-Instruct (model_type: qwen2_5_vl) |
| 代码来源 | trust_remote_code (HF 下载后由 restore_model_patch.sh 替换) |

## 文件表

| 文件 | 说明 |
|------|------|
| `config.json` | 模型配置（含 auto_map 指向本地 modeling 文件） |
| `configuration_qwen2_5_vl.py` | `Qwen2_5_VLConfig` / `Qwen2_5_VLTextConfig` / `Qwen2_5_VLVisionConfig` |
| `modeling_qwen2_5_vl.py` | 完整网络结构，含 PyPTO kernel dispatch (`sys.modules.get("spatial_ssrl_3b_pto_kernels")`) |

## 环境信息

| 组件 | 版本 |
|------|------|
| torch | 2.9.0 |
| torch_npu | 2.9.0.post2 |
| torchvision | 0.24.0 |
| transformers | 5.6.0 |
| CANN | 9.1.0 |

## 归档映射

| 来源 (`models/spatial_ssrl_3b/`) | 目标 (`pypto-gym/`) |
|---|---|
| `config.json` | `src/pypto_gym/transformers/spatial_ssrl_3b/config.json` |
| `configuration_qwen2_5_vl.py` | `src/pypto_gym/transformers/spatial_ssrl_3b/configuration_qwen2_5_vl.py` |
| `modeling_qwen2_5_vl.py` | `src/pypto_gym/transformers/spatial_ssrl_3b/modeling_qwen2_5_vl.py` |
| `spatial_ssrl_3b_pto_kernels/` | `ops/pypto_tensor/spatial_ssrl_3b/` |
| `scripts/` | `modeling/transformers/spatial_ssrl_3b/` |
