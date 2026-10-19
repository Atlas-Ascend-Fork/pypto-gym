# Sparse Attention Anti-Quantization FP8


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 算子概述

本算子实现了面向 **DeepSeek V32 Multi-head Latent Attention (MLA)** 架构的 **Sparse Flash Attention with FP8 Anti-Quantization (反量化)**，运行于华为昇腾 NPU 上。

核心功能：
1. 基于 PagedAttention 机制，通过 top-k 索引从分页 KV cache 中 gather 选定的 KV 条目
2. 对 FP8 量化的 key-nope 进行在线反量化（per-group FP32 scales）
3. 组装完整的 Q/K 并执行标准 Attention 计算：`O = softmax(Q @ K^T / sqrt(d)) @ V`

本算子专为 DeepSeek V32 的 MLA 架构设计，其中 KV cache 存储压缩的 latent 表示，value 复用反量化后的 key-nope。

## 核心参数

| 参数 | 值 | 说明 |
|---|---|---|
| `kv_lora_rank` | 512 | KV latent 维度 |
| `qk_rope_dim` | 64 | RoPE 维度 |
| `head_dim` | 576 | 完整 head 维度（512 + 64） |
| `nq` | 128 | Query head 数量 |
| `n_kv` | 1 | KV head 数量（GQA） |
| `topk` | 2048 | 每个 token 选取的 top-k KV 数量 |
| `block_size` | 128 | PagedAttention block 大小 |

## 文件结构

```
sparse_attention_antiquant_fp8/
├── README.md
├── sparse_attention_antiquant_fp8_impl.py        # 算子实现
└── deepseekv32_sparse_attention_antiquant_fp8.py  # Golden 参考实现 + 测试
```

## API 签名

算子提供两个入口：`sparse_attention_antiquant_d`（Decode）和 `sparse_attention_antiquant_p`（Prefill），签名一致：

```python
sparse_attention_antiquant_d(
    query_nope,    # (t*nq, 512)           BF16   query nope 部分
    query_rope,    # (t*nq, 64)            BF16   query rope 部分
    nope_cache,    # (block_num*bs, 672)   FP8    分页 KV cache（含 kn + kr + scales）
    topk_indices,  # (t, n_kv*topk)        INT32  top-k 选取的 token 索引
    block_table,   # (b, max_blocknum)     INT32  PagedAttention block 映射表
    kv_act_seqs,   # (b,)                  INT32  每个 batch 的实际序列长度
    attention_out,  # (b*s*nq, 512)        BF16   输出
    nq, n_kv, softmax_scale, topk, block_size, max_blocknum_perbatch, tile_config
)
```

### Decode vs Prefill 差异

| 配置项 | Decode (`_d`) | Prefill (`_p`) |
|---|---|---|
| `vec_nbuffer_setting` | `{-1: 2, 0: 4}` | `{-1: 4, 0: 4}` |
| `cube_l1_reuse_setting` | `{-1: 2}` | `{-1: 4}` |
| `device_sched_mode` | `3` | 未设置 |

## KV Cache 布局（nope_cache）

每个 token 在 `nope_cache` 中占 656 字节（padding 至 672 字节对齐），FP8 视角的数据编排：

| 字节偏移 | 内容 | 逻辑 dtype | 维度 |
|---|---|---|---|
| `[0:512]` | kv_nope（量化后） | FP8_E4M3 | 512 |
| `[512:640]` | key_rope | BF16（以 FP8 视角存储） | 64（128 bytes） |
| `[640:656]` | dequant scales | FP32（以 FP8 视角存储） | 4（16 bytes） |

反量化方式：将 512 维 kv_nope 按 4 组（每组 128 个元素）分组，每组乘以对应的 FP32 scale。

## 计算流程

算子采用 5 层嵌套循环结构：

```
L0: batch_idx      — 遍历 batch
  L1: slc_idx      — 遍历 query 序列
    L2: n_kv_idx   — 遍历 KV head
      L3: group_idx — 遍历 query group（nq / n_kv / g_tile）
        L4: s2_idx  — 遍历 KV 序列 tile
```

每次 L4 迭代的计算步骤：

1. **V0 Gather**：通过 `gather_in_ub` 根据 topk_indices + block_table 从 nope_cache 中取出选定的 KV 条目
2. **Dequant**：提取 FP8 kn -> cast FP32 -> 乘以 per-group scales -> cast BF16
3. **组装 K**：拼接 kn(512) + kr(64) -> kj(576)
4. **组装 Q**：拼接 qn(512) + qr(64) -> qi(576)
5. **C1 MatMul**：`sij = qi @ kj^T`，FP32 累加，shape (g_tile, s2_tile)
6. **V1 Softmax**：scale -> amax -> sub -> exp -> sum -> div -> cast BF16
7. **C2 MatMul**：`q1 = softmax @ vj`，其中 vj = kn（512 维），输出 BF16
8. **写出**：将 q1 写入 attention_out

## Tiling 配置

通过 `SaTileShapeConfig` 数据类配置：

```python
@dataclass
class SaTileShapeConfig:
    g_tile: int          # Group tile 大小（如 128）
    s_kv_tile: int       # KV 序列 tile 大小（如 2048）
    c1_tile_shape: list  # 6 个 int，C1 MatMul cube tile
    v1_tile_shape: list  # 2 个 int，V1 Softmax vector tile
    c2_tile_shape: list  # 6 个 int，C2 MatMul cube tile
    v2_tile_shape: list  # 2 个 int（已定义但未使用）
```

典型配置值：g_tile=128, s_kv_tile=2048, cube tiles 128x128, vector tiles 8x2048 / 64x128。

## 测试用例

### 运行方式

```bash
pytest deepseekv32_sparse_attention_antiquant_fp8.py -v
```

### 测试矩阵

| 用例名 | (b, nq, n_kv, s_q) | actual_seq | 模式 |
|---|---|---|---|
| `sfa_bf16_b4_s2_seq64K_total_fp8_d` | (4, 128, 1, 2) | [65536, 16381, 666, 15] | Decode |
| `sfa_bf16_b4_s2_seq64K_per_fp8_d` | (4, 128, 1, 2) | [65536]*4 | Decode（性能测试，默认 skip） |
| `sfa_bf16_b1_s256_seq64K_fp8_p` | (1, 128, 1, 256) | [65536] | Prefill（默认 skip） |

### 精度验证标准

- `atol = 0.0001`
- `rtol = 0.005`
- `max_error_count = 100`

### 支持芯片

- Ascend 910
- Ascend 950

## 约束与注意事项

1. Golden 参考实现使用 per-tile softmax（非 flash/online softmax），当 topk 超过单个 s2_tile 时精度对齐可能存在偏差
2. nope_cache 的 672 字节 padding 对齐是硬编码要求
3. 算子专为 DeepSeek V32 MLA 架构定制，kv_lora_rank=512 和 qk_rope_dim=64 为固定参数
4. Value 复用反量化后的 key-nope（kv_lora_rank=512 维），这是 MLA 架构的特性
