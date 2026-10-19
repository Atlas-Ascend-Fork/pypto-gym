# flash_attention_score — Flash Attention Score with PSE and Dropout


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子概述

`flash_attention_score` 实现带 PSE（Positional Score Encoding）和 Dropout 的 Flash Attention forward pass，采用 GQA（Grouped Query Attention）架构和 online-softmax 算法。

**核心特性：**
- **GQA**：KV 头数 N_kv（默认 8）小于 Q 头数 N（默认 32），每个 KV 头服务 group = N / N_kv 个 Q 头
- **PSE**：逐元素位置偏置，支持两种计算顺序（pse_type=1: `(scores+pse)*scale`；pse_type=2: `scores*scale+pse`）
- **Dropout**：逐元素 dropout 掩码（keep_prob 控制保留概率）
- **Online Softmax**：按 KV-block 分块累积 softmax 统计量（running max + running sum），避免全局 softmax 的大内存开销
- **BF16 I/O + FP32 内部累积**：输入输出 bf16，所有内部计算 fp32
- **Integrated JIT**：单次 JIT 调用，所有 5 层循环（batch / kv_head / group / Q-block / KV-block）均在 JIT 图内
- **valid_shape**：通过 pypto.view(valid_shape=...) 原生处理尾块，零 host-side padding

## 输入输出规格

### 输入（完整 4D/2D 张量）

| 参数 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| `query` | `[B, N, Sq, D]` | `bfloat16` | Q 张量 |
| `key` | `[B, N_kv, Skv, D]` | `bfloat16` | K 张量 |
| `value` | `[B, N_kv, Skv, D]` | `bfloat16` | V 张量 |
| `atten_mask` | `[Sq, Skv]` | `bfloat16` | Attention mask（0=valid, non-0=masked） |
| `pse` | `[B, N, Sq, Skv]` | `bfloat16` | PSE 张量 |
| `drop_mask` | `[Sq, Skv]` | `bfloat16` | Dropout mask（binary 0/1） |
| `pse_type` | scalar | `int` | PSE 顺序：1=`(scores+pse)*scale`，2=`scores*scale+pse` |
| `keep_prob` | scalar | `float` | Dropout 保留概率；<1.0 时 softmax_sum 乘以 1/keep_prob |
| `scale_value` | scalar | `float` | 注意力缩放因子，通常 1/√D |

### 输出

| 参数 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| `output` | `[B, N, Sq, D]` | `bfloat16` | Attention 输出 |
| `softmax_max` | `[B, N, Sq, 1]` | `float32` | Softmax row max |
| `softmax_sum` | `[B, N, Sq, 1]` | `float32` | Softmax row sum（keep_prob 缩放后） |

## 使用方式

```python
from flash_attention_score_impl import flash_attention_score_wrapper

# 准备输入（所有张量在 NPU 设备上）
output, softmax_max, softmax_sum = flash_attention_score_wrapper(
    query, key, value, atten_mask, pse, drop_mask,
    pse_type=1, keep_prob=1.0, scale_value=0.0883883,
)
```

## 运行测试

```bash
# 从仓库根目录运行
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=/root/pto-isa
python custom/flash_attention_score/test_flash_attention_score.py
```

## 精度标准

| 输出 | dtype | atol | rtol | MARE | MERE | RMSE | 备注 |
|------|-------|------|------|------|------|------|------|
| output | bf16 | 1 | 0 | <10 | <2 | <2 | bf16 < 2⁻⁸ 豁免 MERE |
| softmax_max | fp32 | 1e-5 | 1e-5 | — | — | — | |
| softmax_sum | fp32 | 1e-5 | 1e-5 | — | — | — | |

## 架构说明

### 设计决策
- **模块分解（逻辑）**：3 模块（M1: Q@K^T+PSE+scale, M2: softmax+dropout+P@V, M3: online-softmax accumulation+normalize+output）
- **生产架构**：单一 integrated JIT，5 层 pypto.loop，零 host-side padding
- **Tile 配置**：vec=(16,64), cube_QK=([64,64],[128,128],[64,64]), cube_PV=([64,64],[64,64],[128,128])
- **尾块处理**：pypto.view(valid_shape=[...])，2D→2D 视图，valid_shape 匹配源 tensor rank

### 与旧版本的差异
| 方面 | 旧（Stage 7 版本） | 新（valid_shape 版本） |
|------|-------------------|---------------------|
| JIT 调用次数 | per-KV-block 多次调用 | 单次调用 |
| Host wrapper | ~150 行（pad + slice） | ~25 行 |
| Host-side padding | 大量 torch.zeros/full/ones | 零 |
| 循环位置 | Host Python for loops | JIT 内 pypto.loop |
| 尾块处理 | Host padding → slice | pypto.view(valid_shape) |

## 已知约束

- D 固定为 128（compile-time 常量）
- pse_type 和 keep_prob 为 JIT trace-time 常量（每次改变需重新 JIT）
- BLOCK_Q=64, BLOCK_KV=64 为模块级常量
- 需要在 NPU 环境中运行（不支持纯 CPU SIM 模式）
