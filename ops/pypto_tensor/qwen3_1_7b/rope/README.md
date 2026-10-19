# Q/K RMSNorm + RoPE 融合算子 (qwen3_1_7b)

## 概述
替代 Qwen3Attention 中 Q/K channel 的 per-head RMSNorm + `apply_rotary_pos_emb`。D=128 head_dim，BF16 精度。

## 测试
```bash
export TILE_FWK_DEVICE_ID=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 tests/ops/qwen3_1_7b/test_rms_norm_rope.py
```

## 测试用例来源
从模型打点采集的真实 shape/dtype：
- Prefill: S=1,4,7,32,128 等变长输入

## 技术说明
| 项目 | 说明 |
|------|------|
| 场景 | A — golden 直接复用 PyTorch eager 等价代码（`rms_norm_per_head() + rotate_half() + RoPE multiply`） |
| 实现 | Tile-based JIT kernel：`BS_TILE=8`，FP32 中间计算，BF16 输入/输出。`_make_qk_rope_kernel(N)` 工厂函数生成 Q(N=16) 和 K(N=8) 两版 kernel |
| ACLGraph | 未注册 torch.library |

## 状态
✅ 单算子精度 | ✅ 整网集成 | ❌ ACLGraph | ⏳ 性能调优
