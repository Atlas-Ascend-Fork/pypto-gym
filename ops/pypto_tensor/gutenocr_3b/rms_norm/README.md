# RMSNorm 算子集成 (gutenocr_3b)

## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 概述

将 Qwen2RMSNorm 替换为 PyPTO `rms_norm` 融合算子。D=2048，BF16。

## 测试

```bash
export TILE_FWK_DEVICE_ID=0
python3 tests/ops/gutenocr_3b/test_rms_norm.py
```

## 测试用例来源

从模型打点采集的真实 shape/dtype：
- prefill 主 norm: `[1, seq_len, 2048]`
- decode 主 norm: `[1, 1, 2048]`
- q_norm: `[1, seq_len, 16, 128]`
- k_norm: `[1, seq_len, 8, 128]`

## 技术说明

| 项目 | 说明 |
|------|------|
| 场景 | A — 原始实现只使用 torch 基础算子 |
| 实现 | `pypto.tensor()` 无 shape 声明 + `pypto.rms_norm` 融合 API，tile 根据 dim 动态设置 |
| ACLGraph | 未注册 torch.library |

## 状态

✅ 单算子精度 | ✅ 整网集成 | ⏳ ACLGraph | ⏳ 性能调优
