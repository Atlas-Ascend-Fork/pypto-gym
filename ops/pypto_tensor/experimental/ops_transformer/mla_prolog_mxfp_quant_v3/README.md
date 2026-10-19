# MLA Prolog MXFP Quant V3 (PyPTO Kernel)

基于 PyPTO 框架实现的 MLA (Multi-Head Latent Attention) Prolog 量化算子，运行于 Ascend NPU，使用 MXFP (Microscaling FP8) 块级量化格式，用于 DeepSeek V32/V4 模型推理的 decode 阶段。

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 文件说明

| 文件 | 说明 |
|------|------|
| `mla_prolog_mxfp_quant_impl.py` | Kernel 实现 + 量化辅助函数 |
| `tests/ops/experimental/ops_transformer/mla_prolog_mxfp_quant_v3/test_mla_prolog_mxfp_quant_v3.py` | 测试用例 + Golden 参考实现 |

## 算法概述

MLA Prolog MXFP Quant V3 是 MLA 架构的前处理算子量化版本，用于推理场景 decode 阶段的 Query 和 Key 预处理。该算子在 MLA Prolog 基础上引入 MXFP (Microscaling FP8) 块级量化，直接消费预量化的 FP8E4M3 输入与权重，通过原生量化矩阵乘 (`pypto.scaled_mm`) 降低计算和存储开销。

### MXFP 量化格式

- **数据格式**: FP8E4M3（非对称，最大值 448）
- **Scale 格式**: FP8E8M0（块级共享指数，每 64 个元素一个 block）
- 量化矩阵乘: `pypto.scaled_mm`
- 块级量化: `pypto.quant_mx`（输入 BF16/FP32 → FP8E4M3 + FP8E8M0 scale）

### 核心特性

1. **MXFP 两级量化**: quant_a（token_x / w_dq / w_dkv_kr）与 quant_b（RmsNorm 输出 / w_uq_qr）
2. **RmsNorm**: RMS 归一化，Query 和 KV latent 标准化，全程 FP32 计算
3. **RoPE (Rotary Position Embedding)**: 旋转位置编码，支持 2D (`rope_v2`) 和 3D (`rope_3d_v2`) 张量
4. **KV Cache 更新**: 通过 `scatter_update` 实现分页 KV cache 的原地更新
5. **k_nope 不量化**: 本版本 k_nope 仅做 RmsNorm，不做额外量化

### 语义约定

- **t**: token 数量（动态轴 `pypto.DYNAMIC`），decode 阶段通常为少量 token
- **h**: 输入隐藏维度
- **n1/n_q**: Query head 数量
- **q_lora_rank**: Query latent 维度
- **kv_lora_rank**: KV latent 维度
- **qk_nope_head_dim / qk_rope_head_dim**: query/key 的 nope 与 rope 部分维度

### 循环结构

```
MLA_BS_LOOP (loop_unroll，unroll_list=[32,16,8,4,2,1]) — 遍历 token (t)
  pre_compute_2d()         — Query/KV 预计算 (scaled_mm)
  Query 处理               — Split(nope/rope) + transposed_batchmatmul(w_uk) + RoPE
  KV 处理                  — Split + RmsNorm(k_nope) + RoPE(k_rope)
  Cache Update             — scatter_update 更新 kv_cache / kr_cache
```

### 计算流程

#### Query 计算路径

```
1. token_x (FP8E4M3) → scaled_mm(w_dq) → q_a_proj            [t, q_lora_rank]    FP32
2. q_a_proj → RmsNorm(gamma_cq) → q_a_proj_norm              [t, q_lora_rank]    BF16
3. q_a_proj_norm → quant_mx(FP8E4M3) → scaled_mm(w_uq_qr)    [t, n*q_head_dim]   BF16
4. q_b → Split → q_nope + q_rope
5. q_nope → transposed_batchmatmul(w_uk) → query_nope_out    [t, n, kv_lora_rank]
6. q_rope → RoPE(cos, sin) → query_rope_out                  [t, n, qk_rope_head_dim]
```

#### Key-Value 计算路径

```
1. token_x (FP8E4M3) → scaled_mm(w_dkv_kr) → compressed_kv   [t, kv_lora_rank + qk_rope_head_dim]  FP32 → BF16
2. Split → k_nope_raw + k_rope_raw
3. k_nope_raw → RmsNorm(gamma_ckv) → k_nope                  [t, kv_lora_rank]    (不量化)
4. k_rope_raw → RoPE(cos, sin) → k_rope                       [t, qk_rope_head_dim]
5. Cache Update (scatter_update, axis=-2):
   - kv_cache → 写回 kv_cache_out (k_nope)
   - kr_cache → 写回 kr_cache_out (k_rope)
```

## Kernel 签名

```python
mla_prolog_quant(
    token_x,          # [t, h]                        FP8E4M3   — 输入 token (预量化)
    x_scale,          # [t, h//64, 2]                 FP8E8M0   — token_x 块级 scale (每 64 元素一 block)
    w_dq,             # [h, q_lora_rank]              FP8E4M3   — Query 下采样权重 (量化)
    w_dq_scale,       # [h//128, q_lora_rank, 2]      FP8E8M0   — w_dq 块级 scale
    w_uq_qr,          # [q_lora_rank, n*q_head_dim]   FP8E4M3   — Query 上采样权重 (量化)
    w_uqqr_scale,     # [n*q_head_dim, 1]             FP8E8M0   — w_uq_qr 块级 scale
    w_uk,             # [n, qk_nope_head_dim, kv_lora_rank]     BF16   — Query 最终上采样权重
    w_dkv_kr,         # [h, kv_lora_rank+qk_rope_dim] FP8E4M3   — KV 下采样权重 (量化)
    w_dkvkr_scale,    # [h//128, q_head_dim, 2]       FP8E8M0   — w_dkv_kr 块级 scale
    gamma_cq,         # [q_lora_rank]                 BF16      — Query RmsNorm gamma
    gamma_ckv,        # [kv_lora_rank]                BF16      — KV RmsNorm gamma
    cos,              # [t, qk_rope_head_dim]         BF16      — RoPE cos 参数
    sin,              # [t, qk_rope_head_dim]         BF16      — RoPE sin 参数
    cache_index,      # [t]                           INT64     — Cache 更新索引
    kv_cache,         # [block_num, block_size, n_kv, kv_lora_rank]          BF16 — KV cache
    kr_cache,         # [block_num, block_size, n_kv, qk_rope_head_dim]      BF16 — KR cache
    query_nope_out,   # [t, n, kv_lora_rank]          BF16      — Query nope 输出
    query_rope_out,   # [t, n, qk_rope_head_dim]      BF16      — Query RoPE 输出
    kv_cache_out,     # 同 kv_cache 形状 — KV cache 输出 (原地更新)
    kr_cache_out,     # 同 kr_cache 形状 — KR cache 输出 (原地更新)
    epsilon_cq,       # float     — Query RmsNorm epsilon
    epsilon_ckv,      # float     — KV RmsNorm epsilon
    tile_config,      # MlaTileConfig       — Tiling 配置
    rope_cfg,         # RopeTileShapeConfig — RoPE Tiling 配置
)
```

其中 `t` (token 数量) 为动态轴 (`pypto.DYNAMIC`)，其余维度为静态轴。

## 量化策略

### MXFP 量化 (Microscaling FP8)

| 量化阶段 | 量化对象 | 量化方式 | Scale |
|----------|----------|----------|-------|
| quant_a | token_x | 预量化 FP8E4M3 | FP8E8M0，每 64 元素一 block，`x_scale` [t, h//64, 2] |
| quant_a | w_dq, w_dkv_kr | 预量化 FP8E4M3 | FP8E8M0，每 128 元素一 block，`dequant_scale_w_dq` [h//128, q_lora_rank, 2] |
| quant_b | RmsNorm 输出 | `pypto.quant_mx` 动态量化 | FP8E8M0，axis=-1，round down |
| quant_b | w_uq_qr | 预量化 FP8E4M3 | FP8E8M0，`dequant_scale_w_uq_qr` |

### Dtype 转换流程

| 阶段 | 操作 | Dtype |
|------|------|-------|
| 输入 | token_x / x_scale | FP8E4M3 / FP8E8M0 |
| Quant_a matmul | scaled_mm(token_x, w_dq) → q_a_proj | FP8E4M3 × FP8E4M3 → FP32 |
| Quant_a matmul | scaled_mm(token_x, w_dkv_kr) → compressed_kv | FP8E4M3 × FP8E4M3 → FP32 |
| 中间 cast | compressed_kv | FP32 → BF16 |
| RmsNorm | 全程 FP32 计算 | BF16 → FP32 → BF16 |
| Quant_b | quant_mx(RmsNorm 输出, FP8E4M3) | BF16 → FP8E4M3 |
| Quant_b matmul | scaled_mm(quant, w_uq_qr) → q_b_proj | FP8E4M3 × FP8E4M3 → BF16 |
| RoPE | FP32 计算 | BF16 → FP32 → BF16 |
| 输出 | query_nope/rope、kv/kr cache | BF16 |

> **说明**: `pre_compute_2d` 将输入维度 `h` 切分为两半分别做 `scaled_mm` 再累加，以优化内存访问模式。

## 参数说明

### 核心参数 (DeepSeek V32 配置)

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
| token_x | (t, h) | FP8E4M3 | 输入 token（预量化），t 为 token 数量 |
| x_scale | (t, h//64, 2) | FP8E8M0 | token_x 块级 scale |
| w_dq | (h, q_lora_rank) | FP8E4M3 | Query 下采样权重 (量化) |
| w_dq_scale | (h//128, q_lora_rank, 2) | FP8E8M0 | w_dq 块级 scale |
| w_uq_qr | (q_lora_rank, n*q_head_dim) | FP8E4M3 | Query 上采样和 RoPE 权重 (量化) |
| w_uqqr_scale | (n*q_head_dim, 1) | FP8E8M0 | w_uq_qr 块级 scale |
| w_uk | (n, qk_nope_head_dim, kv_lora_rank) | BF16 | Query 最终上采样权重 |
| w_dkv_kr | (h, kv_lora_rank+qk_rope_dim) | FP8E4M3 | KV 下采样和 RoPE 权重 (量化) |
| w_dkvkr_scale | (h//128, q_head_dim, 2) | FP8E8M0 | w_dkv_kr 块级 scale |
| gamma_cq | (q_lora_rank,) | BF16 | Query RmsNorm gamma |
| gamma_ckv | (kv_lora_rank,) | BF16 | KV RmsNorm gamma |
| cos / sin | (t, qk_rope_head_dim) | BF16 | RoPE 参数 |
| cache_index | (t,) | INT64 | Cache 更新索引 |
| kv_cache | (block_num, block_size, n_kv, kv_lora_rank) | BF16 | KV cache |
| kr_cache | (block_num, block_size, n_kv, qk_rope_head_dim) | BF16 | KR cache |

### 输出张量

| 参数名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| query_nope_out | (t, n, kv_lora_rank) | BF16 | Query nope 输出 |
| query_rope_out | (t, n, qk_rope_head_dim) | BF16 | Query RoPE 输出 |
| kv_cache_out | (block_num, block_size, n_kv, kv_lora_rank) | BF16 | KV cache 更新输出 |
| kr_cache_out | (block_num, block_size, n_kv, qk_rope_head_dim) | BF16 | KR cache 更新输出 |

## 分块配置

### MlaTileConfig

```python
@dataclass
class MlaTileConfig:
    tile_b: int = 8              # Batch tile size
    tile_s: int = 1              # Sequence tile size
    tile_bs: int = 8             # Combined batch-sequence tile size
    m_tile: int = 16             # Matmul tile size
    mv_tile: int = 16            # Vector matmul tile size
    pre_quant_cube_tile: list    # [16, 16, 256, 256, 128, 128] — 预量化 matmul cube tile
    unroll_list: list            # [32, 16, 8, 4, 2, 1] — 循环展开长度
    q_vec_tile0: int = 16        # Query vector tile 维度 0
    q_vec_tile1: int = 16        # Query vector tile 维度 1
    k_vec_tile0: int = 16        # Key vector tile 维度 0
    k_vec_tile1: int = 16        # Key vector tile 维度 1
    cube_l1_reuse_setting: dict  # {-1: 4} — cube L1 复用配置
    pg_upper_bound: int = 8192   # Pipeline granularity 上界
    cube_nbuffer_setting: dict   # {3: 4} — cube N-buffer 配置
    dynamic_unaligned_enable: bool  # False — 动态非对齐处理开关
```

### RopeTileShapeConfig

```python
@dataclass
class RopeTileShapeConfig:
    two_dim: list      # 2D RoPE tile shape，如 [32, 64]
    three_dim: list    # 3D RoPE tile shape，如 [32, 32, 128]
    four_dim: list     # 4D RoPE tile shape，如 [16, 128, 128, 128]
```

## 测试用例

通过独立的 `test_XX` 函数定义，均标记为 `@pytest.mark.soc("950")`，运行于 Ascend 950PR：

| 用例 | batch | t | s2 (KV 序列) | n1 | h | 量化模式 |
|------|-------|---|--------------|----|-----|----------|
| test_b4_s64k2_pa_nd_bf16_quant | 4 | 8 | 64*1024 | 128 | 7168 | quant_a + quant_b |
| test_b64_s64k2_pa_nd_bf16_quant | 64 | 128 | 64*1024 | 128 | 7168 | quant_a + quant_b |

两个用例均使用：
- dtype: BF16，cache_mode: PA_BSND，block_size: 128，is_nz: False
- 不同的 MlaTileConfig 定制（tile_bs、unroll_list、q/k_vec_tile 等随 batch 调整）
- `(is_quant_a=True, is_quant_b=True)` 两级 MXFP 量化
- 对比输出：`q_nope_out`、`q_rope_out`、`kv_cache_out`、`kr_cache_out`

### 精度校验

使用 `common_utils.compare` 进行精度对比：

```python
# Query 输出 (qNope / qRope)
atol = 0.005
rtol = 0.0078125     # 1/128，等于 BF16 machine epsilon
max_error_ratio = 0.005

# KV / KR Cache 输出
atol = 0.0001
rtol = 0.0078125
max_error_ratio = 0
```

Golden reference 使用 `torch_npu.npu_quant_matmul`（group_sizes=[1, 1, 32]）并严格模拟 MXFP 量化流程（`quant_mx_golden_bytes` 实现 FP8E4M3 编码与 FP8E8M0 共享指数计算），确保对比基准与硬件行为一致。

## 运行方式

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 从仓库根目录运行 (main() 默认执行 test_b4_s64k2_pa_nd_bf16_quant)
python tests/ops/experimental/ops_transformer/mla_prolog_mxfp_quant_v3/test_mla_prolog_mxfp_quant_v3.py
```

使用 pytest 运行：

```bash
pytest tests/ops/experimental/ops_transformer/mla_prolog_mxfp_quant_v3/test_mla_prolog_mxfp_quant_v3.py -v
```

## 实现要点

### 1. 量化 Matmul

使用 MXFP 量化 matmul，全程无独立 dequant 步骤：
- `scaled_mm(token_x_view, w_dq_view, DT_FP32, x_scale_view, dequant_scale_w_dq_view)` → q_a_proj
- `scaled_mm(token_x_view, w_dkv_kr_view, DT_FP32, x_scale_view, dequant_scale_w_dkv_kr_view)` → compressed_kv
- `scaled_mm(quant_view, w_uq_qr, DT_BF16, scale_view, dequant_scale_w_uq_qr)` → q_b_proj

### 2. 动态量化 (quant_b)

RmsNorm 输出通过 `pypto.quant_mx` 在 kernel 内量化：

```python
quant_view, scale_view = pypto.quant_mx(
    norm_res, pypto.DT_FP8E4M3, pypto.ROUND_DOWN, -1, True,
)
```

### 3. RmsNorm 实现

```python
# Formula: output = gamma * input / sqrt(mean(input^2) + epsilon)
y = input_fp32 * input_fp32
y = y * (1.0 / dim)
y = sum(y, -1, keepdim=True)
y = sqrt(y + epsilon)
y = gamma_fp32 * (input_fp32 / y)   # 全程 FP32，避免精度损失
```

### 4. RoPE 实现

#### 2D RoPE (rope_v2)
- 输入: `[seq_size, d_r]`
- 处理: reshape [seq_size, d_r//2, 2] → transpose → reshape → `x*cos + rotate_half(x)*sin` → cast

#### 3D RoPE (rope_3d_v2)
- 输入: `[batch, heads, rope_dim]`
- 处理: broadcast cos/sin 到 head 维度，rotate 后与 x 组合
- `DAV_3510` 平台使用专用 vec_tile shape [1, 128, 64]

#### rotate_half
```
对于维度对 (2i, 2i+1):
out[2i] = -x[2i+1]
out[2i+1] = x[2i]
```

### 5. Cache Update

使用 `scatter_update` 实现分页 cache 原地更新：
- 通过 `cache_index` 定位要更新的 cache 位置
- axis=-2 表示在序列维度上更新
- 支持分页 KV cache (PagedAttention)，`k_nope` 不做量化直接写回

### 6. 性能优化选项

```python
@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {0: 1, 1: 1, 2: 3, 3: 4},
        "cube_nbuffer_setting": {0: 1, 1: 1, 2: 1, 3: 4},
    },
    runtime_options={
        "device_sched_mode": 2,
        "stitch_function_max_num": 128
    }
)
```

## 约束与注意事项

1. **权重格式**: w_dq, w_uq_qr, w_dkv_kr 必须为预量化的 FP8E4M3，scale 为 FP8E8M0
2. **量化对齐**: token_x 的 block 大小为 64 元素 (`x_scale` [t, h//64, 2])，权重 block 大小为 128 元素
3. **Cache 布局**: 支持 PA_BSND 格式 (block_num, block_size, n_kv, dim)
4. **动态序列**: t 维度支持动态长度，通过 `pypto.DYNAMIC` 声明，`loop_unroll` 按 `unroll_list` 展开
5. **精度容差**: Query 输出的精度容差需考虑 MXFP 量化误差累积
6. **Cache 索引**: cache_index 必须有效 (>= 0 且 < block_num * block_size)
7. **k_nope 不量化**: 与 v3 目标一致，k_nope 仅做 RmsNorm 后直接写 cache
8. **GQA 支持**: n_kv = 1，实现 Grouped Query Attention

## 依赖

- Python 3.x
- Pytest (含 soc profile 标记)
- PyTorch + torch_npu
- PyPTO (`pypto` 包)
- NumPy
- `common_utils` 精度对比工具