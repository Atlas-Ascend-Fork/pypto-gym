# MLA Prolog Quant HiFP8 V3 (PyPTO Kernel)

基于 PyPTO 框架实现的 MLA (Multi-Head Latent Attention) Prolog 量化算子，运行于 Ascend NPU，支持 HiFP8 量化格式，用于 DeepSeek V32 模型推理。


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 文件说明

| 文件 | 说明 |
|------|------|
| `mla_prolog_quant_hifp8_v3_impl.py` | Kernel 实现 + 量化/反量化辅助函数 |
| `test_mla_prolog_quant_hifp8_v3.py` | 测试用例 + Golden reference 实现 |

## 算法概述

MLA Prolog Quant HiFP8 V3 是 MLA 架构的前处理算子量化版本，用于推理场景的 Query 和 Key 预处理。该算子在 MLA Prolog 基础上增加了 HiFP8 量化支持，在保证精度的同时降低计算和存储开销。

### 核心特性

1. **HiFP8 量化**: 使用 High Fidelity 8-bit (HF8) 格式进行权重量化，最大值范围 [-32768, 32768]
2. **两级量化策略**: 支持 quant_a 和 quant_b 两级量化，灵活适配不同精度需求
3. **RmsNorm**: RMS 归一化处理，用于 Query 和 KV latent 的标准化
4. **RoPE (Rotary Position Embedding)**: 旋转位置编码，支持 2D 和 3D 张量
5. **KV Cache 更新**: 通过 scatter_update 实现分页 KV cache 的原地更新

### 计算流程

#### Query 计算路径

```
1. token_x → Quant(HiFP8) → Matmul(w_dq) → Dequant → c^Q
   - 权重 w_dq 使用 HiFP8 格式，per-channel scale
   - 输入使用 per-token scale
   
2. c^Q → RmsNorm(gamma_cq) → c^Q_norm

3. c^Q_norm → Quant(HiFP8) → Matmul(w_uq_qr) → Dequant → q_b
   - 权重 w_uq_qr 使用 HiFP8 格式
   - RmsNorm 输出使用 per-token scale
   
4. q_b → Split → q_nope + q_rope
   - q_nope: [t, n, qk_nope_head_dim]
   - q_rope: [t, n, qk_rope_head_dim]

5. q_nope → BatchMatmul(w_uk) → query_nope_out
   - w_uk: [n, qk_nope_head_dim, kv_lora_rank]
   
6. q_rope → RoPE(cos, sin) → query_rope_out
```

#### Key-Value 计算路径

```
1. token_x → Quant(HiFP8) → Matmul(w_dkv_kr) → Dequant → compressed_kv
   - compressed_kv: [t, kv_lora_rank + qk_rope_head_dim]
   
2. Split → k_nope_raw + k_rope_raw
   - k_nope_raw: [t, kv_lora_rank]
   - k_rope_raw: [t, qk_rope_head_dim]

3. k_nope_raw → RmsNorm(gamma_ckv) → k_nope

4. k_rope_raw → RoPE(cos, sin) → k_rope

5. Cache Update:
   - kv_cache → scatter_update(k_nope) → kv_cache_out
   - kr_cache → scatter_update(k_rope) → kr_cache_out
```

## Kernel 签名

```python
mla_prolog_quant(
    token_x,          # [t, h]              BF16     — 输入 token
    w_dq,             # [h, q_lora_rank]    HF8      — Query 下采样权重 (量化)
    w_dq_scale,       # [q_lora_rank]       FP32     — w_dq 反量化 scale
    w_uq_qr,          # [q_lora_rank, n*q_head_dim]  HF8  — Query 上采样权重 (量化)
    w_uqqr_scale,     # [n*q_head_dim]      FP32     — w_uq_qr 反量化 scale
    w_uk,             # [n, qk_nope_head_dim, kv_lora_rank]  BF16  — Query 最终上采样权重
    w_dkv_kr,         # [h, kv_lora_rank+qk_rope_dim]  HF8  — KV 下采样权重 (量化)
    w_dkvkr_scale,    # [kv_lora_rank+qk_rope_dim]  FP32  — w_dkv_kr 反量化 scale
    gamma_cq,         # [q_lora_rank]       BF16     — Query RmsNorm gamma
    gamma_ckv,        # [kv_lora_rank]      BF16     — KV RmsNorm gamma
    cos,              # [t, qk_rope_head_dim]  BF16  — RoPE cos 参数
    sin,              # [t, qk_rope_head_dim]  BF16  — RoPE sin 参数
    cache_index,      # [t]                 INT64    — Cache 更新索引
    kv_cache,         # [block_num, block_size, n_kv, kv_lora_rank]  INT8/BF16  — KV cache
    kr_cache,         # [block_num, block_size, n_kv, qk_rope_head_dim]  BF16  — KR cache
    query_nope_out,   # [t, n, kv_lora_rank]  BF16  — Query nope 输出
    query_rope_out,   # [t, n, qk_rope_head_dim]  BF16  — Query rope 输出
    kv_cache_out,     # [block_num, block_size, n_kv, kv_lora_rank]  — KV cache 输出 (原地更新)
    kr_cache_out,     # [block_num, block_size, n_kv, qk_rope_head_dim]  — KR cache 输出 (原地更新)
    epsilon_cq,       # float               — Query RmsNorm epsilon
    epsilon_ckv,      # float               — KV RmsNorm epsilon
    tile_config,      # MlaTileConfig       — Tiling 配置
    rope_cfg,         # RopeTileShapeConfig — RoPE Tiling 配置
)
```

其中 `t` (token 数量) 为动态轴 (`pypto.DYNAMIC`)，其他维度为静态轴。

## 量化策略

### HiFP8 量化 (High Fidelity 8-bit)

HiFP8 格式特点：
- **数据范围**: [-32768, 32768]
- **量化公式**: `scale = max(|input|) / 32768.0`
- **量化**: `quantized = input / scale → cast(HF8)`
- **反量化**: `dequantized = quantized * scale`

### 量化位置

| 量化阶段 | 量化对象 | Scale 类型 | 说明 |
|----------|----------|------------|------|
| quant_a | token_x | per-token | 输入激活量化 |
| quant_a | w_dq, w_dkv_kr | per-channel | 权重量化 (按输出通道) |
| quant_b | RmsNorm 输出 | per-token | 归一化后激活量化 |
| quant_b | w_uq_qr | per-channel | 权重量化 |

### Dtype 转换流程

| 阶段 | 操作 | Dtype |
|------|------|-------|
| 输入 | token_x | BF16 |
| Quant_a | token_x → HF8 | BF16 → HF8 |
| Matmul | HF8 @ HF8 → FP32 | HF8 → FP32 |
| Dequant | FP32 * scale | FP32 → BF16 |
| RmsNorm | BF16 → FP32 → BF16 | BF16 → FP32 → BF16 |
| Quant_b | RmsNorm 输出 → HF8 | BF16 → HF8 |
| Matmul | HF8 @ HF8 → FP32 | HF8 → FP32 |
| Dequant | FP32 * scale | FP32 → BF16 |
| 输出 | Query/KV | BF16 |

## 参数说明

### 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `h` | 7168 | Hidden dimension (输入隐藏层维度) |
| `q_lora_rank` | 1536 | Query latent 维度 |
| `kv_lora_rank` | 512 | KV latent 维度 |
| `n` | 128 | Query head 数量 |
| `n_kv` | 1 | KV head 数量 (GQA) |
| `qk_nope_head_dim` | 128 | Query/Key nope 维度 |
| `qk_rope_head_dim` | 64 | Query/Key RoPE 维度 |
| `block_size` | 128 | PagedAttention block 大小 |

### 输入张量

| 参数名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| token_x | (t, h) | BF16 | 输入 token，t 为 token 数量 |
| w_dq | (h, q_lora_rank) | HF8 | Query 下采样权重 (量化) |
| w_dq_scale | (q_lora_rank) | FP32 | w_dq 反量化 scale |
| w_uq_qr | (q_lora_rank, n*q_head_dim) | HF8 | Query 上采样和 RoPE 权重 (量化) |
| w_uqqr_scale | (n*q_head_dim) | FP32 | w_uq_qr 反量化 scale |
| w_uk | (n, qk_nope_head_dim, kv_lora_rank) | BF16 | Query 最终上采样权重 |
| w_dkv_kr | (h, kv_lora_rank+qk_rope_dim) | HF8 | KV 下采样和 RoPE 权重 (量化) |
| w_dkvkr_scale | (kv_lora_rank+qk_rope_dim) | FP32 | w_dkv_kr 反量化 scale |
| gamma_cq | (q_lora_rank,) | BF16 | Query RmsNorm gamma 参数 |
| gamma_ckv | (kv_lora_rank,) | BF16 | KV RmsNorm gamma 参数 |
| cos | (t, qk_rope_head_dim) | BF16 | RoPE 正弦参数 |
| sin | (t, qk_rope_head_dim) | BF16 | RoPE 余弦参数 |
| cache_index | (t,) | INT64 | Cache 更新索引 |
| kv_cache | (block_num, block_size, n_kv, kv_lora_rank) | INT8/BF16 | KV cache |
| kr_cache | (block_num, block_size, n_kv, qk_rope_head_dim) | BF16 | KR cache |

### 输出张量

| 参数名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| query_nope_out | (t, n, kv_lora_rank) | BF16 | Query nope 输出 |
| query_rope_out | (t, n, qk_rope_head_dim) | BF16 | Query RoPE 输出 |
| kv_cache_out | (block_num, block_size, n_kv, kv_lora_rank) | INT8/BF16 | KV cache 更新输出 |
| kr_cache_out | (block_num, block_size, n_kv, qk_rope_head_dim) | BF16 | KR cache 更新输出 |

## Tiling 配置

### MlaTileConfig

```python
@dataclass
class MlaTileConfig:
    tile_b: int = 8              # Batch tile size
    tile_s: int = 1              # Sequence tile size
    tile_bs: int = 8             # Combined batch-sequence tile size
    m_tile: int = 16             # Matmul tile size
    mv_tile: int = 16            # Vector matmul tile size
    pre_quant_cube_tile: list    # Cube tile shapes for pre-quantization matmul
    unroll_list: list            # List of unroll lengths for loop optimization
    q_vec_tile0: int = 16        # Query vector tile dimension 0
    q_vec_tile1: int = 16        # Query vector tile dimension 1
    k_vec_tile0: int = 16        # Key vector tile dimension 0
    k_vec_tile1: int = 16        # Key vector tile dimension 1
    cube_l1_reuse_setting: dict  # L1 reuse configuration
    cube_nbuffer_setting: dict   # N-buffer configuration
```

### RopeTileShapeConfig

```python
@dataclass
class RopeTileShapeConfig:
    two_dim: list      # Tile shape for 2D RoPE operations, e.g., [32, 64]
    three_dim: list    # Tile shape for 3D RoPE operations, e.g., [32, 32, 128]
    four_dim: list     # Tile shape for 4D RoPE operations, e.g., [16, 128, 128, 128]
```

## 测试用例

### 运行方式

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 运行测试
cd models/experimental/ops-transformer/mla_prolog_quant_hifp8_v3
python test_mla_prolog_quant_hifp8_v3.py
```

使用 pytest 运行：

```bash
pytest test_mla_prolog_quant_hifp8_v3.py -v
```

### 测试配置

| 用例名 | batch | s | s2 | n | h | 量化模式 |
|--------|-------|---|----|----|-----|----------|
| test_b4_s64k2_pa_nd_bf16_quant | 4 | 2 | 1024 | 128 | 7168 | quant_a + quant_b |

### 精度校验标准

```python
# Query 输出
rtol = 0.005
atol = 0.0078125
threshold = 0.005

# KV/KR Cache
rtol = 0.0001
atol = 0.0078125
threshold = 0
```

## 实现要点

### 1. 循环结构

采用 `loop_unroll` 进行动态序列处理：

```
MLA_BS_LOOP (unroll) — 遍历 token (t)，使用 unroll_list 优化
  pre_compute_2d()   — Query 和 KV 的预计算
  Query 处理         — Split + BatchMatmul + RoPE
  KV 处理            — Split + RmsNorm + RoPE
  Cache Update       — scatter_update 更新 cache
```

### 2. 量化 Matmul

使用 HiFP8 量化 matmul：
- 输入量化: `quant_hifp8(input) → (quantized, scale)`
- Matmul: `matmul(quantized, weight, DT_FP32)`
- 反量化: `dequant(dtype, result, input_scale, weight_scale)`

### 3. RmsNorm 实现

```python
# Formula: output = gamma * input / sqrt(mean(input^2) + epsilon)
y = input_fp32 * input_fp32
y = y * (1.0 / dim)
y = sum(y, -1, keepdim=True)
y = sqrt(y + epsilon)
y = gamma_fp32 * (input_fp32 / y)
```

### 4. RoPE 实现

#### 2D RoPE (rope_v2)
- 输入: `[seq_size, d_r]`
- 处理: reshape → transpose → rotation → cast

#### 3D RoPE (rope_3d_v2)
- 输入: `[batch, heads, rope_dim]`
- 处理: broadcast cos/sin → reshape → transpose → rotation

#### rotate_half
```
对于维度对 (2i, 2i+1):
out[2i] = -x[2i+1]
out[2i+1] = x[2i]
```

### 5. Cache Update

使用 `scatter_update` 实现 cache 原地更新：
- 通过 `cache_index` 定位要更新的 cache 位置
- axis=-2 表示在序列维度上更新
- 支持分页 KV cache (PagedAttention)

### 6. 性能优化选项

```python
@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {-1: 4, 0: 1, 1: 1, 2: 1},
        "cube_nbuffer_setting": {-1: 4, 0: 1, 1: 1, 2: 1, 3: 3},
    },
    runtime_options={"device_sched_mode": 2}
)
```

## 约束与注意事项

1. **量化权重格式**: w_dq, w_uq_qr, w_dkv_kr 必须为 HiFP8 格式
2. **Cache 布局**: 支持 PA_BSND 格式 (block_num, block_size, n_kv, dim)
3. **动态序列**: t 维度支持动态长度，通过 `pypto.DYNAMIC` 声明
4. **精度容差**: Query 输出的精度容差需考虑量化误差累积
5. **Cache 索引**: cache_index 必须有效 (>= 0 且 < block_num * block_size)
6. **GQA 支持**: n_kv = 1，实现 Grouped Query Attention

## 关键特性

### HiFP8 量化优势

- **高精度**: 最大值 32768，相比 FP8 E4M3 (448) 提供更大的动态范围
- **低内存**: 8-bit 存储，降低 weight 存储开销
- **混合精度**: Matmul 使用 HF8，累加使用 FP32，平衡精度和性能

### MLA 架构适配

- **Latent Compression**: 通过 q_lora_rank 和 kv_lora_rank 实现 latent 表示压缩
- **RoPE 分离**: qk_nope_head_dim 和 qk_rope_head_dim 分离处理
- **KV Cache 效率**: 压缩的 KV latent 减少 cache 存储压力

## 依赖

- PyPTO: 华为昇腾 AI 处理器自定义算子开发框架
- PyTorch: 用于 Golden 实现和数据生成
- torch_npu: NPU 设备支持
- utils.compare: 精度对比工具（位于 `deepseek_v32_exp/utils/compare`）

## 平台支持

- Ascend 910
