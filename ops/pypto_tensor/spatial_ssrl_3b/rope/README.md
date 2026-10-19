# RoPE 算子集成 (spatial_ssrl_3b)

## 概述
融合 Rotary Position Embedding，替代原始 `apply_rotary_pos_emb_vision`（Vision）和 `apply_multimodal_rotary_pos_emb`（Text MRoPE, D=128, mrope_section=[16,24,24]）。实现为 PyPTO JIT kernel（tile=32）。GQA 下 q(16 heads) / k(2 heads) 头数不同，拆分两次独立 kernel 调；FP16。

## 测试

```bash
export TILE_FWK_DEVICE_ID=2
pytest tests/ops/spatial_ssrl_3b/rope/test_rope.py -v --forked
```

## 技术说明

| 项目 | 说明 |
|------|------|
| 场景 | A — golden 直接取 PyTorch `apply_rotary_pos_emb_vision` / `apply_multimodal_rotary_pos_emb` |
| 实现 | `@pypto.frontend.jit` + `pypto.mul/add/concat/neg`；wrapper 预展开 cos/sin + contiguous；tile=32 保 UB 不溢 |
| ACLGraph | 未注册 torch.library（暂不支持 --use-acl-graph） |

## 状态
✅ 单算子精度 | ✅ 整网集成 | ❌ ACLGraph | ⏳ 性能调优
