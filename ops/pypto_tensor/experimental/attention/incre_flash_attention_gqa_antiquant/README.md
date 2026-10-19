# Incre Flash Attention GQA Anti-Quant 算子说明文档


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 算子概述

**算子名称**: incre_flash_attention_gqa_antiquant (Incre Flash Attention with Grouped Query Attention and Anti-Quantization)

**功能描述**:

该算子实现了增量推理阶段的 Grouped Query Attention (GQA) 计算，并支持 KV Cache 的反量化 (Anti-Quantization)，是 Transformer 模型中注意力机制的优化实现。主要用于大模型推理的增量生成阶段（自回归生成），具有以下特点：

- **增量推理**: 专门针对推理阶段的逐 token 生成优化，query 序列长度固定为 1 (S1=1)
- **Grouped Query Attention (GQA)**: 多个 query heads 共享同一组 KV heads，减少 KV Cache 存储开销
- **KV Cache 反量化**: KV Cache 以 FP8 格式存储，通过反量化缩放因子还原为 BF16 进行计算，内存占用减半
- **Flash Attention**: 使用 Online Softmax 模式减少 HBM 访问次数，提升性能
- **Paged KV Cache**: 采用分页式 KV Cache，支持非连续内存访问，提高内存利用率

**应用场景**:

- 大语言模型 (LLM) 推理的增量生成阶段
- 支持 GQA 架构的 Transformer 模型
- 需要 Paged Attention 优化的大批量推理服务
- 使用 FP8 量化 KV Cache 的推理部署


## 数学公式

### 核心计算流程

```
输入:
  query:               [b, n1, s1, d]              # 当前步 query (BNSD 布局)
  key_cache:           [block_num, n2, block_size, d]    # paged KV cache 中的 key (FP8)
  value_cache:         [block_num, n2, block_size, d]    # paged KV cache 中的 value (FP8)
  key_antiquant_scale: [n2, d]                        # key 反量化缩放因子 (BF16)
  value_antiquant_scale: [n2, d]                      # value 反量化缩放因子 (BF16)
  block_table:         [b, max_blocks_per_query]      # 逻辑块到物理块的映射表 (INT32)
  kv_actual_seqs:      [b]                            # 每个 batch 的实际 KV 序列长度 (INT32)

反量化计算 (FP32 精度执行):
  1) cast K_fp8 [s2_tile, d] → FP32
  2) 取 key_antiquant_scale[n2_idx] → [d] (BF16)，cast → FP32
  3) FP32 × FP32 逐元素相乘，scale 沿 sequence 维度广播至 [s2_tile, d]
  4) cast 结果 → BF16，作为后续 matmul 输入

  同理对 V 执行相同反量化流程，使用 value_antiquant_scale[n2_idx]

  数学表达式:
  K_dequant[s, d] = fp32(K_fp8[s, d]) × fp32(key_antiquant_scale[n2_idx, d])
  V_dequant[s, d] = fp32(V_fp8[s, d]) × fp32(value_antiquant_scale[n2_idx, d])

GQA 扩展:
  group = n1 // n2
  K_expanded = K_dequant.repeat_interleave(group, dim=1)   # [b, n1, s2, d]
  V_expanded = V_dequant.repeat_interleave(group, dim=1)   # [b, n1, s2, d]

计算步骤:

1. 反量化 (FP32 精度执行)
   K_fp8 → cast(FP32)；key_antiquant_scale[n2_idx] → cast(FP32)
   K_dequant = FP32 × FP32 (scale 沿 s 维度广播) → cast(BF16)
   V_fp8 → cast(FP32)；value_antiquant_scale[n2_idx] → cast(FP32)
   V_dequant = FP32 × FP32 (scale 沿 s 维度广播) → cast(BF16)

2. 计算注意力分数
   softmax_scale = d^(-0.5)
   scores = matmul(query, K_expanded^T) × softmax_scale
   # Shape: [b, n1, s1, s2]

3. Softmax 归一化
   attention_weights = softmax(scores, dim=-1)
   # Shape: [b, n1, s1, s2]

4. 加权求和
   attention_out = matmul(attention_weights, V_expanded)
   # Shape: [b, n1, s1, d]

输出:
  attention_out: [b, n1, s1, d]  (BNSD 布局)
```

### 关键数学表达式

```
K_dequant[s, d] = fp32(K_fp8[s, d]) × fp32(key_antiquant_scale[n2_idx, d])

V_dequant[s, d] = fp32(V_fp8[s, d]) × fp32(value_antiquant_scale[n2_idx, d])

scores[b, n1, s1, s2] = Σ_d (query[b, n1, s1, d] × K_dequant[b, n2, s2, d]) × scale
  其中 n2 = n1 // group

attention_weights[b, n1, s1, s2] = exp(scores[b, n1, s1, s2]) / Σ_s exp(scores[b, n1, s1, s])

attention_out[b, n1, s1, d] = Σ_s (attention_weights[b, n1, s1, s] × V_dequant[b, n2, s, d])
```

## 参数说明

### 输入参数

| 参数名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `query` | torch.Tensor (BF16) | [b, n1, s1, d] | 查询张量 (BNSD 布局) |
| `key` | torch.Tensor (FP8 E4M3) | [block_num, n2, block_size, d] | Key Cache (Paged 格式, FP8 存储) |
| `value` | torch.Tensor (FP8 E4M3) | [block_num, n2, block_size, d] | Value Cache (Paged 格式, FP8 存储) |
| `key_antiquant_scale` | torch.Tensor (BF16) | [n2, d] | Key 反量化缩放因子 |
| `value_antiquant_scale` | torch.Tensor (BF16) | [n2, d] | Value 反量化缩放因子 |
| `kv_actual_seqs` | torch.Tensor (INT32) | [b] | 每个批次的实际 KV 序列长度 |
| `block_table` | torch.Tensor (INT32) | [b, max_blocks_per_query] | 块表，映射逻辑块到物理块索引 |

### 输出参数

| 参数名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `atten_out` | torch.Tensor (BF16) | [b, n1, s1, d] | 注意力输出 (BNSD 布局) |

### 配置参数 (自动推导)

| 参数名 | 推导方式 | 说明 |
|--------|----------|------|
| `n1` | query.shape[1] | 查询头数量 |
| `d` | query.shape[3] | 头维度 |
| `block_num` | key.shape[0] | 总 KV 块数 |
| `n2` | key.shape[1] | KV 头数量 |
| `block_size` | key.shape[2] | 块大小 |
| `b` | block_table.shape[0] | 批次大小 |
| `s1` | query.shape[2] | 查询序列长度 |
| `group` | n1 // n2 | 分组数 |
| `softmax_scale` | d^(-0.5) | Softmax 缩放因子 |

### Tiling 参数 (AttentionTileConfig)

| 参数名 | 说明 | 推荐值 |
|--------|------|--------|
| `g_tile` | 组维度切分大小 | group (n1 // n2) |
| `s2_tile` | KV 序列维度切分大小 | 2048 |
| `c1_tile` | C1 阶段（Q×K^T）切分配置 | [[128, 128], [128, 128], [128, 128]] |
| `v1_tile` | V1 阶段（Softmax）切分配置 | [128, 2048] |
| `c2_tile` | C2 阶段（Softmax×V）切分配置 | [[128, 128], [128, 128], [128, 128]] |
| `v2_tile` | V2 阶段（输出更新）切分配置 | [128, 128] |

## 测试用例

### 测试场景

| 测试用例 | 批次大小 | 查询头数 | KV头数 | KV序列长度 | 头维度 | 说明 |
|---------|----------|----------|--------|-----------|--------|------|
| `test_1b2k` | 1 | 8 | 1 | 2048 | 128 | 小批次短序列 |
| `test_8b2kqs2` | 8 | 8 | 1 | 2048 | 128 | 中等批次，s1=2 |
| `test_16b4kqs3` | 16 | 8 | 1 | 4096 | 128 | 大批次，s1=3 |
| `test_32b8k_d256` | 32 | 8 | 1 | 8192 | 256 | 大批次长序列，d=256 |
| `test_64b2k_kvn2` | 64 | 8 | 2 | 2048 | 128 | 大批次，n2=2 |
| `test_4b16k` | 4 | 64 | 8 | 16384 | 128 | 小批次长序列，GQA |
| `test_64b16k` | 64 | 64 | 8 | 16384 | 128 | 大批次长序列，GQA |

### 精度标准

- **绝对容差 (atol)**: 0.0001
- **相对容差 (rtol)**: 0.0078125 (1/128)
- **最大误差比例**: 0.5%

## 约束条件

### 数据类型约束

- `query`, `atten_out` 必须为 `torch.bfloat16` (BNSD 布局)
- `key`, `value` 必须为 `torch.float8_e4m3fn` (PA_BnNBsD 格式)
- `key_antiquant_scale`, `value_antiquant_scale` 必须为 `torch.bfloat16`
- `block_table` 和 `kv_actual_seqs` 必须为 `torch.int32`
- 中间计算使用 FP32 保证精度
- 反量化流程：FP8→cast(FP32)，BF16_scale[n2_idx]→cast(FP32)，FP32×FP32(scale沿s广播)→cast(BF16)

### 形状约束

- `n1` 必须能被 `n2` 整除（分组查询注意力）
- `s1` 通常为 1（增量生成场景）
- `block_size` 为 128
- `s2_tile` (2048) 应能被 `block_size` (128) 整除
- `d` 支持 128 和 256
- query 和 atten_out 使用 BNSD 布局

### 内存约束

- Block Table 中无效索引用 -1 表示，计算时使用 `.max(0)` 处理
- KV Cache 以 FP8 存储，内存占用为 BF16 的一半
- 需要足够的 workspace 内存存储中间结果（out_update, sum_update, max_update）

## 性能优化建议

### Tiling 配置优化

根据不同场景选择合适的 Tiling 配置：

| 场景 | 推荐配置 |
|------|---------|
| 小批次短序列 (b=1, s2=2048) | g_tile=group, s2_tile=2048 |
| 中等批次 (b=8-16, s2=2048-4096) | g_tile=group, s2_tile=2048 |
| 大批次长序列 (b=64, s2=16384) | g_tile=group, s2_tile=2048 |
| d=256 场景 | 调整 cube tile 大小 |

### 内存优化

- 使用 Paged KV Cache 减少内存碎片
- KV Cache 以 FP8 存储，内存占用减半
- 合理设置 block_size (128)
- 控制批次大小避免内存溢出

### 计算优化

- 利用分组查询注意力减少计算量 (n1/n2 越大越好)
- 使用在线 Softmax 避免存储完整注意力矩阵
- 反量化按 tile 即时执行，避免额外全局存储
- 合理设置 Tiling 参数平衡计算和内存访问

## 实现细节

### 计算流程

```
Input: query(BF16), key(FP8), value(FP8), antiquant_scale(BF16),
       block_table, kv_actual_seqs
  ↓
init_kernel_cfg + reshape_qkv_to_2d: 提取参数，reshape 为 2D
  ↓
Loop Structure: batch_size → s1 → n2 → group → s2_tiles
  ↓
For each s2_tile:
Assemble KV: 从 Paged Cache 组装 K/V block (FP8)
  Anti-Quant K: FP8→cast(FP32), scale[n2_idx]→cast(FP32), FP32×FP32→cast(BF16)
  C1: 计算 Q × K_dequant^T (注意力分数)
  V1: 在线 Softmax (缩放、max、exp、sum)
  Anti-Quant V: FP8→cast(FP32), scale[n2_idx]→cast(FP32), FP32×FP32→cast(BF16)
  C2: 计算 Softmax_prob × V_dequant
  V2: 在线 Softmax 输出更新
  ↓
Finalize: output / sum → BF16 → assemble to atten_out (BNSD)
  ↓
Output: atten_out [b, n1, s1, d] (BNSD)
```

### 核心 API

本算子使用以下 PyPTO API：

- `pypto.reshape`: 数据预处理 (inplace)
- `pypto.view`: 非连续内存访问，支持 valid_shape
- `pypto.assemble`: 数据组装（Paged Cache、Query、Output）
- `pypto.cast`: 反量化类型转换 (FP8→FP32, BF16_scale→FP32, FP32→BF16)
- `pypto.matmul`: 矩阵乘法 (C1: Q×K^T, C2: Softmax×V)
- `pypto.mul`: 反量化 FP32×FP32 乘法 + Softmax 缩放
- `pypto.sub/div/exp/amax/sum/maximum`: Online Softmax 计算
- `pypto.loop/cond/is_loop_begin/end`: 循环控制
- `pypto.tensor`: 中间张量分配
- `pypto.ceildiv`: 动态循环次数计算
