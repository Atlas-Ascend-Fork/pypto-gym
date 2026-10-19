# RMSNorm PyPTO Kernel

## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 概述

融合 RMS Normalization 算子，替代原始 Qwen2RMSNorm 实现。

## 性能收益

- 减少内核调用开销
- 预计性能提升 5-8%

## 调用场景

| 位置 | shape | dtype | eps | 调用频率 |
|------|-------|-------|-----|----------|
| input_layernorm | [batch, seq, 2048] | float16 | 1e-6 | 每层×每步 |
| post_attention_layernorm | [batch, seq, 2048] | float16 | 1e-6 | 每层×每步 |
| final norm | [batch, seq, 2048] | float16 | 1e-6 | 每步1次 |

## 使用方式

```python
# 在 transformers 导入前注入
sys.path.insert(0, model_path)
import spatial_ssrl_3b_pto_kernels as pto_kernels
sys.modules["spatial_ssrl_3b_pto_kernels"] = pto_kernels
pto_kernels.USE_PTO_RMS_NORM = True

from transformers import AutoModel
model = AutoModel.from_pretrained(model_path)
```

## 测试

```bash
export TILE_FWK_DEVICE_ID=2
python3 tests/ops/spatial_ssrl_3b/rms_norm/test_rms_norm_spatial_ssrl_3b.py
```

## 依赖

- pypto >= 0.2.1
- torch-npu >= 2.7.1
- CANN >= 8.5.0