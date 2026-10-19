# Incre Flash Attention MLA 算子说明文档


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子概述

**算子名称**: incre_flash_attention_mla (Incre Flash Attention with Multi-Head Latent Attention)

**功能描述**:

该算子实现了增量推理阶段的 Multi-Head Latent Attention (MLA) 计算，是 Transformer 模型中注意力机制的优化实现。主要用于大模型推理的增量生成阶段（自回归生成），具有以下特点：

- **增量推理**: 专门针对推理阶段的逐 token 生成优化，query 序列长度固定为 1 (S1=1)
- **RoPE 位置编码分离**: 将位置编码部分 (rope) 与内容部分分离存储，优化内存布局
- **Grouped Query Attention (GQA)**: 多个 query heads 共享同一组 KV heads，减少 KV Cache 存储开销
- **Flash Attention**: 使用 Online Softmax 模式减少 HBM 访问次数，提升性能
- **Paged KV Cache**: 采用分页式 KV Cache，支持非连续内存访问，提高内存利用率

**应用场景**:

- 大语言模型 (LLM) 推理的增量生成阶段
- 支持 MLA 架构的 Transformer 模型
- 需要 Paged Attention 优化的大批量推理服务
- 长上下文场景下的自回归生成


## 数学公式

### 核心计算流程

```
输入:
  query:      [b, n1, s1, q_d]         # 当前步 query (S1=1)
  query_rope: [b, n1, s1, q_rope_d]    # query 的 RoPE 部分
  key:  [block_num, n2, block_size, kv_d]    # paged KV cache 中的 key
  value:[block_num, n2, block_size, kv_d]    # paged KV cache 中的 value (与 key 共享)
  key_rope: [block_num, n2, block_size, k_rope_d]  # key 的 RoPE 部分
  block_table: [b, max_blocks_per_query]      # 逻辑块到物理块的映射表
  kv_actual_seqs: [b]                  # 每个 batch 的实际 KV 序列长度

中间变量:
  key_full:   从 key 和 key_rope 通过 block_table 重建
  value:      与key共享

计算步骤:

1. 拼接 RoPE 部分
   query_full = concat([query, query_rope], dim=-1)
   # Shape: [B, N1, S1, D_q + D_q_rope]

   key_full = concat([key, key_rope], dim=-1)
   # Shape: [B, N1, S2, D_kv + D_k_rope]

2. 计算注意力分数
   softmax_scale = (D_q + D_q_rope) ** -0.5
   scores = matmul(query_full, key_full^T) * softmax_scale
   # Shape: [B, N1, S1, S2]

3. Softmax 归一化
   attention_weights = softmax(scores, dim=-1)
   # Shape: [B, N1, S1, S2]

4. 加权求和
   attention_out = matmul(attention_weights, value)
   # Shape: [B, N1, S1, D_kv]

输出:
  attention_out: [B, N1, S1, D_q]
```

### 关键数学表达式

```
query_full[b, n, :, :] = [query[b, n, :, :], query_rope[b, n, :, :]]  # 拼接

key_full[b, n, :, :] = [key[b, n, :, :], key_rope[b, n, :, :]]  # 拼接

scores[b, n, s1, s2] = Σ_d (query_full[b, n, s1, d] * key_full[b, n, s2, d]) * scale

attention_weights[b, n, s1, s2] = exp(scores[b, n, s1, s2]) / Σ_s exp(scores[b, n, s1, s])

attention_out[b, n, s1, d] = Σ_s (attention_weights[b, n, s1, s] * value[b, n, s, d])
```

## 参数说明

### 输入参数

| 参数名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `query` | torch.Tensor (BF16) | [b, s1, n1, q_d] | 查询张量（非旋转部分） |
| `key` | torch.Tensor (BF16) | [block_num, n2, block_size, kv_d] | Key Cache (Paged 格式) |
| `value` | torch.Tensor (BF16) | [block_num, n2, block_size, kv_d] | Value Cache (Paged 格式，MLA 中与 key 共享) |
| `query_rope` | torch.Tensor (BF16) | [b, s1, n1, q_rope_d] | 查询旋转位置编码部分 |
| `key_rope` | torch.Tensor (BF16) | [block_num, n2, block_size, k_rope_d] | Key 旋转位置编码 Cache |
| `kv_actual_seqs` | torch.Tensor (INT32) | [b] | 每个批次的实际 KV 序列长度 |
| `block_table` | torch.Tensor (INT32) | [b, max_blocks_per_query] | 块表，映射逻辑块到物理块索引 |
| `kernel_config` | MlaConfig | - | 算子配置参数 |
| `tile_config` | AttentionTileConfig | - | Tiling 配置参数 |

### 输出参数

| 参数名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `attention_output` | torch.Tensor (BF16) | [b, n1, s1, q_d] | 注意力输出 |

### 配置参数 (MlaConfig)

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `layout` | "BSND" | 输入张量布局 |
| `b` | 32 | 批次大小 |
| `n1` | 128 | 查询头数量 |
| `s1` | 1 | 查询序列长度（增量生成: 1） |
| `q_d` | 512 | 查询头维度（非旋转部分） |
| `q_rope_d` | 64 | 查询旋转位置编码维度 |
| `n2` | 1 | Key/Value 头数量 |
| `s2` | 4096 | Key/Value 序列长度（最大值） |
| `kv_d` | 512 | Key/Value 头维度 |
| `k_rope_d` | 64 | Key 旋转位置编码维度 |
| `block_size` | 128 | Paged Cache 块大小 |
| `softmax_scale` | 576^(-0.5) | Softmax 缩放因子 |

### Tiling 参数 (AttentionTileConfig)

| 参数名 | 说明 | 推荐值 |
|--------|------|--------|
| `g_tile` | 组维度切分大小 | 128 |
| `s2_tile` | KV 序列维度切分大小 | 2048 |
| `v0_tile` | V0 阶段（组装计算）切分配置 | [128, 576] |
| `c1_tile` | C1 阶段（Q×K^T）切分配置 | [[128, 128], [128, 128], [128, 128]] |
| `v1_tile` | V1 阶段（Softmax）切分配置 | [8, 2048] |
| `c2_tile` | C2 阶段（Softmax×V）切分配置 | [[128, 128], [128, 128], [128, 128]] |
| `v2_tile` | V2 阶段（输出更新）切分配置 | [64, 512] |
| `v2_update_tile` | V2 更新阶段切分配置 | [32, 512] |

## 测试用例

### 测试场景

| 测试用例 | 批次大小 | 查询头数 | KV序列长度 | 说明 |
|---------|----------|----------|-----------|------|
| `test_incre_flash_attention_mla_1b4k` | 1 | 128 | 4096 | 小批次长序列 |
| `test_incre_flash_attention_mla_8b4k` | 8 | 128 | 4096 | 中等批次 |
| `test_incre_flash_attention_mla_16b8k` | 16 | 128 | 8192 | 大批次长序列 |
| `test_incre_flash_attention_mla_32b2k` | 32 | 128 | 2048 | 大批次短序列 |
| `test_incre_flash_attention_mla_32b4k` | 32 | 128 | 4096 | 大批次中等序列 |

### 精度标准

- **绝对容差 (atol)**: 0.0001
- **相对容差 (rtol)**: 0.0078125 (1/128)
- **最大误差比例**: 0.5%

## 约束条件

### 数据类型约束

- 输入/输出张量必须为 `torch.bfloat16`
- Block Table 和 KV 实际序列长度必须为 `torch.int32`
- 中间计算使用 FP32 保证精度

### 形状约束

- `n1` 必须能被 `n2` 整除（分组查询注意力）
- `s1` 通常为 1（增量生成场景）
- `q_d + q_rope_d` = `kv_d + k_rope_d`（完整查询/键维度）
- `block_size` 通常为 128 或 256
- `s2_tile` 应能被 `block_size` 整除

### 内存约束

- Key Cache 和 Value Cache 共享内存（MLA 特性）
- Block Table 中无效索引用 -1 表示
- 需要足够的 workspace 内存存储中间结果

## 性能优化建议

### Tiling 配置优化

根据不同场景选择合适的 Tiling 配置：

| 场景 | 推荐配置 |
|------|---------|
| 小批次长序列 (b=1, s2=4096) | g_tile=128, s2_tile=2048 |
| 中等批次 (b=8-16, s2=4096-8192) | g_tile=128, s2_tile=2048 |
| 大批次短序列 (b=32, s2=2048) | g_tile=128, s2_tile=2048 |
| 大批次长序列 (b=32, s2=4096) | g_tile=128, s2_tile=2048 |

### 内存优化

- 使用 Paged KV Cache 减少内存碎片
- 合理设置 block_size (推荐 128)
- 控制批次大小避免内存溢出

### 计算优化

- 利用分组查询注意力减少计算量 (n1/n2 越大越好)
- 使用在线 Softmax 避免存储完整注意力矩阵
- 合理设置 Tiling 参数平衡计算和内存访问

## 实现细节

### 计算流程

```
Input: query, query_rope, key, key_rope, value_cache, block_table, kv_actual_seqs
  ↓
reshape_qkv_to_2d: 将所有张量 reshape 为 2D
  ↓
Loop Structure: batch_size → s1 → n2 → group → s2_tiles
  ↓
For each s2_tile:
  V0 (MLA_V0): 组装 Key 和 Key_Rope
  C1 (MLA_C1): 计算 Q × K^T
  V1 (MLA_V1): 在线 Softmax
  C2 (MLA_C2): 计算 Softmax × V
  V2 (MLA_V2): 在线 Softmax 输出更新
  ↓
Output: attention_output [batch_size, s1, n1, q_d]
```

### 核心 API

本算子使用以下 PyPTO API：

- `pypto.reshape`: 数据预处理
- `pypto.view`: 非连续内存访问
- `pypto.assemble`: 数据组装
- `pypto.matmul`: 矩阵乘法
- `pypto.mul/sub/add/div`: 向量运算
- `pypto.exp/amax/sum/maximum`: Softmax 相关运算
- `pypto.cast`: 类型转换
- `pypto.loop`: 循环控制
- `pypto.tensor`: 中间张量分配
