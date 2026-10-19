# sparse_flash_mla_softmax_l1_norm

## 概述

`sparse_flash_mla_softmax_l1_norm` 是一个基于 `pypto_pro` DSL 在 Ascend NPU 上实现的
**Sparse Flash Mla 注意力 Softmax L1Norm** 算子，支持 Compressed Attention 以及
Sparse Compressed Attention 场景。该算子为 KLLossGrad 反向算子的配套正向接口，输出
可用于反向梯度计算。

本实现为 **JIT 静态形式**：保留 TilingKey + datatype 特化以在编译期剪裁各模式分支，
去掉离线二进制编译所必需的 `workspace` 参数，通过 bracket-launch 语法即时编译启动。

## 接口签名

```python
softmax_l1_norm = sparse_flash_mla_softmax_l1_norm_wrapper(
    q, k, softmax_lse,
    sparse_indices=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    topk_length=None,
    mask_mode=0,
    cmp_ratio=1,
    softmax_scale=None,
    num_cores=4,
)
```

### 输入

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `q` | `[T1, G, D]` (TND) / `[B, Sq, G, D]` (BSND) | FP16/BF16 | Query |
| `k` | `[T2, 1, D]` (TND) / `[B, Sk, 1, D]` (BSND) | FP16/BF16 | Key(V) |
| `softmax_lse` | `[1, T1, G]` (TND) / `[B, Sq, 1, G]` (BSND) | FP32 | softmaxLse |
| `sparse_indices` | `[T1, 1, kLen]` / `[B, Sq, 1, kLen]` | INT32 | 可选，稀疏索引 |
| `cu_seqlens_q/k` | `[B+1]` | INT32 | TND 必传 |
| `seqused_q/k` | `[B]` | INT32 | 可选 |
| `cmp_residual_k` | `[B]` | INT32 | mask_mode=3 且 cmp_ratio>1 时必传 |
| `topk_length` | `[T1,1]` / `[B,Sq,1]` | INT32 | 可选，稀疏 topk |
| `mask_mode` | - | int | 0 (No mask) / 3 (rightDownCausal) |
| `cmp_ratio` | - | int | 压缩率，1~128 |
| `softmax_scale` | - | float | 缩放系数，None 取 1/sqrt(D) |

### 输出

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `softmax_l1_norm` | `[T1, 1, out_len]` / `[B, Sq, 1, out_len]` | FP32 | ReduceSum(P, dim=G) / G |

其中 `out_len` 稀疏场景为 `kLen`，稠密场景为 `max_seqlen_k`（TND）或 `Sk`（BSND）。

## 算法

```text
selectedKv = Gather(K, sparse_indices)   # 稀疏；else 即 K
P[t, k, g] = exp(scale * (q[t,g,:] @ selectedKv[k,:]) - softmaxLse[t,g])
softmaxL1Norm[t, k] = sum_g P[t, k, g] / G
```

`mask_mode` 决定每行 query 实际参与计算的 key 数 `s2_real_size`：
- `0`：使用完整 s2 长度 / k_length；
- `3`：按 `cmp_ratio` / `cmp_residual_k` 计算因果长度。

## 架构设计

- **Cube 段**：QK matmul（`compute_qk` / `compute_qk_dense`），稀疏场景先经
  `gather_k` 把选中的 K 行 gather 到 L1。
- **Vector 段**：对 G 个 head 做 `exp(qk*scale - lse)` 归约求和，除以 G 后原子累加写回。
- **跨核同步**：Cube 与 Vector 通过乒乓事件握手，G 方向按 subblock 均分为两半并行处理。

## Tiling 常量

| 符号 | 值 | 说明 |
|------|-----|------|
| `TG` | 128 | G tile |
| `TKV` | 128 | KV tile |
| `TD` | 128 | D tile |
| `D_TOTAL` | 512 | D 全宽 |
| `GATHER_ROW_NUM` | 32 | 稀疏 gather 行数 |

## 支持的配置

- layout：TND / BSND
- 稀疏 / 稠密
- mask_mode：0 / 3
- dtype：FP16 / BF16
- D 固定 512，G 固定 128

## 文件结构

```text
sparse_flash_mla_softmax_l1_norm/
├── sparse_flash_mla_softmax_l1_norm_impl.py  # JIT kernel + host 封装
└── README.md
```

## 使用方法

```python
from ops.pypto_pro.experimental.ops_transformer.sparse_flash_mla_softmax_l1_norm.sparse_flash_mla_softmax_l1_norm_impl import (
    sparse_flash_mla_softmax_l1_norm_wrapper,
)

# 所有输入需在 NPU 设备上；TND 场景需传 cu_seqlens_q/k
out = sparse_flash_mla_softmax_l1_norm_wrapper(
    q, k, softmax_lse,
    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
    mask_mode=3, cmp_ratio=128,
)
```

首次调用时通过 `@pl.jit` 触发 JIT 编译。

## 精度说明

- **内部计算**：Cube matmul 使用 FP32 累加器；Vector 段 exp/归约全程 FP32。
- **I/O**：q/k 为 FP16/BF16，softmax_lse 与输出为 FP32。
- 参考实现见测试目录 `sparse_flash_mla_softmax_l1_norm_golden.py`（CPU FP32）。