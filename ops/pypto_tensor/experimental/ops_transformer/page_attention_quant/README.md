# Page Attention Quant FP8 (PyPTO Kernel)

基于 PyPTO 框架实现的 Paged Attention FP8 量化算子，运行于 Ascend NPU，用于 GLM-4.5 模型。


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 文件说明

| 文件 | 说明 |
|------|------|
| `page_attention_quant_fp8_impl.py` | Kernel 实现 + 量化/反量化辅助函数 |
| `test_page_attention_quant_fp8.py` | 测试用例 + Golden reference 实现 |

## 算法概述

实现 GLM-4.5 模型的 Paged Attention 机制，采用分页内存管理策略高效处理变长序列和动态批次，同时使用 FP8 (E4M3) 量化降低内存占用和计算开销。

### 语义约定

- **Q 侧 (s1_size)**: Query 序列长度（生成阶段通常为 1）
- **KV 侧 (s2_size)**: KV cache 序列长度（可变，支持长序列）
- **Paged KV**: KV cache 按 block (block_size=128) 分页管理，通过 block_table 映射
- **FP8 Quantization**: 使用 FP8 E4M3 格式进行量化，scale 采用 per-token 或 per-channel 方式
- **GQA (Grouped Query Attention)**: nq 个 query head 共享 nkv 个 KV head，group = nq // nkv
- **G_TILE**: Group tile 大小，每次处理的 query head group 数量
- **S2_TILE**: KV 序列分块大小，用于迭代计算

### 循环结构

```
batch_loop (parallel)          — 遍历 batch，从 kv_act_seqs 动态获取实际序列长度
  s1_loop                      — 遍历 s1 (query 序列维度)
    n2_loop                     — 遍历 KV head (n2_sym = nkv)
      g_loop                    — 遍历 group，每次处理 g_tile 个 query head
        s2_loop (unroll 8,4,2,1) — KV 序列按 s2_tile 分块迭代
```

### 计算流程 (per batch, per s1, per group)

```
1. 查询分块 Q_tile                [g_tile, D]      FP8 E4M3 + scale
2. 组装 KV 分块 (通过 block_table) 
   - K_assemble                   [s2_tile, D]     FP8 E4M3 + scale
   - V_assemble                   [s2_tile, D]     FP8 E4M3 + scale
3. MM1: S_tile = Q_tile @ K_assemble^T    [g_tile, s2_tile]  FP8 matmul → FP32
4. Dequant: S_fp32 = S_tile * Q_scale * K_scale^T    [g_tile, s2_tile]  FP32
5. Flash Attention Online Softmax:
   - Scale: S_scaled = S_fp32 * softmax_scale       [g_tile, s2_tile]
   - Update: M_new = max(M_old, max(S_scaled))      [g_tile, 1]
   - Exp: P = exp(S_scaled - M_new)                 [g_tile, s2_tile]
   - Accum: Sum_new = Sum_old * exp(M_old - M_new) + sum(P)  [g_tile, 1]
6. Quant: P_fp8, P_scale = quant_per_token(P)      [g_tile, s2_tile] FP8 + scale
7. MM2: O_tile = P_fp8 @ V_assemble                [g_tile, D]  FP8 matmul → FP32
8. Dequant: O_fp32 = O_tile * P_scale * V_scale    [g_tile, D]  FP32
9. Accum: O_accum = O_accum * exp(M_old - M_new) + O_fp32  [g_tile, D]
10. Final: O_final = O_accum / Sum_new             [g_tile, D]  FP32 → BF16
```

## Kernel 签名

```python
ifa_func_kernel(
    q,              # [B*S1, NQ, D]      FP8E4M3  — Query 输入
    q_scale,        # [B, NQ, 1]         FP32     — Query scale (per-token)
    k,              # [num_blocks, block_size, NKV, D]  FP8E4M3  — Key cache (paged)
    k_scale,        # [num_blocks, block_size, NKV, 1]  FP32     — Key scale (per-token)
    v,              # [num_blocks, block_size, NKV, D]  FP8E4M3  — Value cache (paged)
    v_scale,        # [B, NKV, D]        FP32     — Value scale (per-channel)
    block_table,    # [B, max_blocks_per_query]  INT32  — Block mapping table
    kv_act_seqs,    # [B]                INT32    — 每个 batch 的实际 KV 序列长度
    atten_out,      # [B*S1, NQ, D]      BF16     — Attention 输出
)
```

其中 `B` (batch_size) 为动态轴 (`pypto.DYNAMIC`)，`S1`、`NQ`、`NKV`、`D` 为静态轴。

Kernel 入口处将三维/四维输入 reshape 为二维形式以便于 view 操作：
- Q: `[B*S1*NQ, D]` 
- K/V: `[num_blocks*block_size, NKV*D]`

## Dtype 转换流程

Kernel 内部严格控制 FP8/FP32/BF16 转换以平衡精度和性能：

| 阶段 | 操作 | Dtype |
|------|------|-------|
| 输入 | Q/K/V | FP8 E4M3 |
| 输入 | Q_scale/K_scale/V_scale | FP32 |
| MM1 | Q @ K^T → S_quant | FP8 → FP32 |
| Dequant MM1 | S_quant * Q_scale * K_scale^T | FP32 全程 |
| Softmax | scale/max/exp | FP32 全程 |
| Quant P | P → P_fp8 | FP32 → FP8 E4M3 |
| MM2 | P_fp8 @ V → O_quant | FP8 → FP32 |
| Dequant MM2 | O_quant * P_scale * V_scale | FP32 全程 |
| Accum | O_accum (跨 s2_tile 累加) | FP32 全程 |
| 输出 | O_final cast BF16 | FP32 → BF16 |

### 量化策略

1. **Query 量化 (per-token)**: 每个 query token 独立计算 scale
   - `scale = 448.0 / max(|Q|)` (FP8 E4M3 最大值为 448)
   - 输出: Q_fp8, Q_scale

2. **Key 量化 (per-token)**: 每个 KV token 独立计算 scale
   - `scale = 448.0 / max(|K|)`
   - 输出: K_fp8, K_scale

3. **Value 量化 (per-channel)**: 每个 channel (head_dim) 共享 scale
   - `scale = 448.0 / max(|V|, dim=seq)`
   - 输出: V_fp8, V_scale

4. **Attention P 量化 (per-token)**: 每个 softmax 输出 token 独立计算 scale
   - `scale = 448.0 / max(|P|)`
   - 输出: P_fp8, P_scale

## 测试用例

通过独立的 `test_ifa_XX` 函数定义，每个函数指定不同的 `batch_size`、`s1_size`、`s2_size`：

| 用例 | batch | s1 | s2 | 说明 |
|------|-------|----|----|------|
| test_ifa_01 | 16 | 1 | 8192 | 默认配置 (大批次) |
| test_ifa_02 | 8 | 1 | 8192 | 小批次 (skip) |

### 精度校验

使用 `numpy.testing.assert_allclose` 进行精度验证：

```python
rtol = 0.0078125   # 1/128，考虑 FP8/BF16 混合精度误差
atol = 0.001
```

Golden reference 模拟完整的 Paged Attention + FP8 量化流程：
1. 按 block_table 组装 KV cache
2. 执行 QK matmul + softmax + PV matmul
3. 在关键节点进行 FP8 量化/反量化（与 kernel 内部流程一致）

## 运行方式

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 运行全部测试用例
python test_page_attention_quant_fp8.py
```

使用 pytest 运行特定用例：

```bash
# 运行 test_ifa_01
pytest -v test_page_attention_quant_fp8.py::test_ifa_01

# 跳过已标记 skip 的用例
pytest -v test_page_attention_quant_fp8.py -m "not skip"
```

## 添加新用例

```python
@pytest.mark.soc("950")
def test_ifa_03():
    """batch=4, s1=2, s2=4096"""
    ifa_test_impl(b=4, s1=2, s2=4096)
```

`ifa_test_impl` 支持的可选参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `b` | 批次数量 | 16 |
| `s1` | Query 序列长度 | 1 |
| `s2` | KV cache 序列长度 | 8192 |

注意：
- `b` 必须与 `atten_cfg.actual_seq` 的长度相等
- 所有 `actual_seq` 值必须小于等于 `s2`

## 关键特性

### Paged KV Cache

- **Block Size**: 128 tokens per block
- **Block Table**: 每个 batch 维护一个 block 映射表，支持不连续内存布局
- **动态序列长度**: 通过 `kv_act_seqs` 支持变长序列

### FP8 Quantization

- **低精度计算**: 使用 FP8 E4M3 (范围 [-448, 448]) 降低内存占用
- **混合精度策略**: Matmul 使用 FP8，累加/softmax 使用 FP32 保证精度
- **量化位置选择**: 
  - Q/K/V 输入量化
  - Attention P 中间量化 (降低 MM2 计算开销)

### Flash Attention Online Softmax

- **Online Algorithm**: 采用增量式 softmax，避免存储完整的注意力矩阵
- **内存优化**: 只维护 [g_tile, 1] 的 max 和 sum，不存储 [g_tile, s2_tile] 的完整 P 矩阵
- **精度保证**: FP32 累加，最终输出 BF16

## 配置说明

### AttentionConfig

```python
@dataclass
class AttentionConfig:
    b: int = 8                 # batch_size
    s1: int = 1                # query 序列长度
    s2: int = 16384            # KV cache 最大序列长度
    n1: int = 12               # query head 数量
    n2: int = 1                # KV head 数量 (GQA)
    q_d: int = 128             # head dimension
    kv_d: int = 128            # KV head dimension
    block_size: int = 128      # 分页 block 大小
    softmax_scale: float       # softmax scale = 1/sqrt(d)
    kv_layout: str = "PA_BSND" # KV cache 布局格式
    actual_seq: torch.Tensor   # 每个 batch 的实际序列长度
```

### AttentionTileConfig

```python
@dataclass
class AttentionTileConfig:
    g_tile: int = 12           # Group tile 大小
    s2_tile: int = 512         # KV 序列 tile 大小
    c1_tile_shape: list        # MM1 cube tile shapes
    v1_tile_shape: list        # Vector tile shapes (softmax)
    c2_tile_shape: list        # MM2 cube tile shapes
    v2_tile_shape: list        # Vector tile shapes (output)
```

## 依赖

- Python 3.x
- PyTorch + torch_npu
- PyPTO (`pypto` 包)
- NumPy
- pytest

## 平台支持

- DAV_3510 (Ascend 950)

## 性能优化选项

```python
@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128},
    pass_options={
        "cube_l1_reuse_setting": {0: 4},  # Q 常驻 L1，4 次 matmul 合并
        "vec_nbuffer_setting": {-1: 4},   # Vector 双缓冲
        "cube_nbuffer_setting": {-1: 4}   # Cube 双缓冲
    }
)
```

## 注意事项

1. **Block Table 管理**: block_table 中的 block_id 必须 >= 0 且 < num_blocks，-1 表示无效 block
2. **序列长度一致性**: kv_act_seqs 的值不能超过 s2 (最大序列长度)
3. **量化 scale 非零**: 量化前需确保输入 tensor 非零，避免 scale 为无穷大
4. **内存布局**: 所有输入 tensor 必须为 ND 格式 (非 NZ)