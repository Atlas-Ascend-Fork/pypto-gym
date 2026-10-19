# FlashAttentionScoreGrad PyPTO 实现


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子概述

本目录包含 FlashAttentionScoreGrad 算子的 PyPTO 实现，用于计算 Flash Attention 前向传播的反向梯度。

### 数学公式

**前向**: `Y = Softmax(Q @ K^T / sqrt(D)) @ V`

**反向** (Online Softmax):
```
P  = exp(Q @ K^T * scale - softmax_max) / softmax_sum
D  = sum(dY * attention_out, dim=-1, keepdim=True)
dP = dY @ V^T
dS = P * (dP - D)
dV = P^T @ dY
dQ = dS @ K * scale
dK = dS^T @ Q * scale
```

### 特性

- **动态轴**: Batch (B) 和 Sequence (S) 为动态轴
- **Online Softmax**: 利用前向保存的 softmax_max/sum 重算 P，无需存储全量注意力矩阵
- **数据类型**: BFloat16 输入/输出，FP32 中间计算
- **布局**: BNSD [Batch, NumHeads, SeqLen, HeadDim]

## 目录结构

```
flash_attention_score_grad/
├── flash_attention_score_grad_golden.py   # 纯 PyTorch golden 参考实现
├── flash_attention_score_grad_impl.py     # PyPTO JIT kernel + wrapper
├── test_flash_attention_score_grad.py     # 测试入口
└── README.md                              # 本文件
```

## 运行方法

### 环境准备

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
```

### 验证 Golden

```bash
python3 custom/flash_attention_score_grad/flash_attention_score_grad_golden.py
```

### 运行测试

```bash
# 运行所有测试级别
python3 custom/flash_attention_score_grad/test_flash_attention_score_grad.py

# 运行指定级别
python3 custom/flash_attention_score_grad/test_flash_attention_score_grad.py 0

# 查看可用级别
python3 custom/flash_attention_score_grad/test_flash_attention_score_grad.py --list
```

## 输入/输出规格

| Tensor | Shape | DType | 说明 |
|--------|-------|-------|------|
| query | [B, N, S, D] | BF16 | Query 张量 |
| key | [B, N, S, D] | BF16 | Key 张量 |
| value | [B, N, S, D] | BF16 | Value 张量 |
| dy | [B, N, S, D] | BF16 | 输出梯度 |
| softmax_max | [B, N, S, 8] | FP32 | 前向 softmax max |
| softmax_sum | [B, N, S, 8] | FP32 | 前向 softmax sum |
| attention_out | [B, N, S, D] | BF16 | 前向输出 |
| **dQ** (输出) | [B, N, S, D] | BF16 | Query 梯度 |
| **dK** (输出) | [B, N, S, D] | BF16 | Key 梯度 |
| **dV** (输出) | [B, N, S, D] | BF16 | Value 梯度 |

## 简化说明

初版实现聚焦核心 dQ/dK/dV 计算，跳过以下可选功能：
- PSE (位置偏移编码)
- Dropout
- Attention Mask
- RoPE (旋转位置编码)
- FP8 量化
- 稀疏注意力模式
