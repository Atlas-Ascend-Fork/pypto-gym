# MRoPE 算子集成 (gutenocr_3b)

## 概述

将 multimodal rotary position embedding 替换为 PyPTO 等价实现。mrope_section=[16,24,24]，BF16。

## 测试

```bash
export TILE_FWK_DEVICE_ID=0
python3 tests/ops/gutenocr_3b/test_mrope.py
```

## 测试用例来源

从模型打点采集的真实 shape/dtype：
- q/k: `[1, 16, seq_len, 128]` / `[1, 2, seq_len, 128]`
- cos/sin: `[3, 1, seq_len, 128]`

## 技术说明

| 项目 | 说明 |
|------|------|
| 场景 | A — 原始实现只使用 torch 基础算子（cat/split/rotary） |
| 实现 | 等价 torch 路径：`torch.cat` + `torch.split` + `rotate_half`，与 baseline 行为一致 |
| ACLGraph | 未注册 torch.library |

## 状态

✅ 单算子精度 | ✅ 整网集成 | ⏳ ACLGraph | ⏳ 性能调优
