# SwiGLU MLP 算子集成 (gutenocr_3b)

## 概述

将 Qwen2MLP（gate_proj + SiLU gate + up_proj elementwise + down_proj）替换为 PyPTO SwiGLU 融合算子。D=2048，I=11008，BF16。

## 测试

```bash
export TILE_FWK_DEVICE_ID=0
python3 tests/ops/gutenocr_3b/test_swiglu_mlp.py
```

## 测试用例来源

从模型打点采集的真实 shape/dtype：`[1*seq_len, 2048]` → `[1*seq_len, 2048]`

## 技术说明

| 项目 | 说明 |
|---|---|
| 场景 | A — 原始实现只使用 torch 基础算子（nn.Linear + SiLU + mul） |
| 实现 | `swiglu_mlp_fused` / `swiglu_mlp_fused_static` kernel；wrapper 通过 `__init__.py` 桥接 nn.Module → kernel |
| ACLGraph | 未注册 torch.library |

## 状态

🔧 内核适配中 | ✅ 整网集成（fallback torch 路径） | ⏳ 单算子精度 | ⏳ 性能调优
