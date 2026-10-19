---
schema_version: 2
op_name: qkv_rms_norm_rope_cache
dynamic_axes: [T]
---
# DESIGN

## 1. 当前实现概览

当前 PyPTO 代码只保留 INT8 PA_NZ quant cache 路径，但分为既有快路径和功能兜底路径：

```text
qkv_rms_norm_rope_cache_quant_kernel_tp4
qkv_rms_norm_rope_cache_quant_kernel_tp1
qkv_rms_norm_rope_cache_quant_kernel_generic
```

wrapper 校验 `k_cache/v_cache` dtype：

```text
torch.int8 -> INT8 quant cache 分支
其他 dtype -> TypeError
```

既有两条网络性能用例仍使用 TP4/TP1 快路径；head 规模超出快路径覆盖范围时使用 generic fallback。

## 2. 动态/静态轴设计

JIT 签名只把 token 轴设为动态：

```text
qkv:       [DYNAMIC, STATIC]
cos/sin:   [DYNAMIC, STATIC]
index:     [DYNAMIC]
q_out_out: [DYNAMIC, STATIC]
```

cache、scale、head/dim 相关维度保持 static。TP4/TP1 两组 head/cache shape 分别静态编译；generic fallback 的 `index` 在 wrapper 侧扩展为 `[T, group_heads*D]` 后以 static 二维形态进入 JIT，用于 `pypto.scatter`。

## 3. INT8 分支计算图

```text
qkv [T, Nqkv*D]
  ├─ Q view [T,Nq*D] -> reshape [T,Nq,D]
  │      -> RMSNorm(q_gamma) -> RoPE(cos/sin)
  │      -> reshape [T,Nq*D] -> q_out
  ├─ K view [T,Nk*D] -> reshape [T,Nk,D]
  │      -> RMSNorm(k_gamma) -> RoPE(cos/sin)
  │      -> quant_int8(k_scale)
  │      -> PA_NZ INT8 cache write
  └─ V view [T,Nv*D] -> reshape [T,Nv,D]
         -> quant_int8(v_scale)
         -> PA_NZ INT8 cache write
```

量化公式：

```text
quant = cast_int8_saturate(cast_int32_rint(x / scale))
```

当前实现函数：

```text
_compute_quant
_compute_quant_generic_fallback
_compute_q_grouped
_compute_kv_grouped_fallback
_quant_int8
_scatter_pa_nz_int8_contiguous_page0
_scatter_pa_nz_int8_indexed_group
```

## 4. API 映射

| 逻辑 | 实现 API | 说明 |
| --- | --- | --- |
| SplitVD | `pypto.view` | 按静态 width 切出 Q/K/V |
| reshape | `pypto.reshape` | 2D/3D 转换和 cache 视图转换 |
| RMSNorm | `cast/sum/sqrt/div/mul` | FP32 中间计算 |
| RoPE | `view/concat/mul/add` | half-and-half rotate |
| Q 输出 | `move` 或 `assemble` | 按 token 写回 Q 输出 |
| INT8 quant | `pypto.cast` + saturation | round 后饱和到 int8 |
| INT8 cache fast path | reshape + `move` | 当前 TP4/TP1 网络用例 page0 连续写入快路径 |
| INT8 cache fallback | reshape/transpose + `pypto.scatter` + `assemble` | Generic fallback 按 `index` 写 PA_NZ cache |

## 5. 当前网络 case 的分支

### MTP2-TP4

```text
qkv=[48,2304]
qkv_size=[16,3,18,128]
head_nums=[16,1,1]
k/v_cache=[11898,4,128,32] int8
k/v_scale=[1,128]
```

Q/K/V 大小：

```text
Q = 16*128 = 2048
K = 1*128 = 128
V = 1*128 = 128
```

### MTP2-TP1

```text
qkv=[12,9216]
qkv_size=[4,3,72,128]
head_nums=[64,4,4]
k/v_cache=[11898,16,128,32] int8
k/v_scale=[4,128]
```

Q/K/V 大小：

```text
Q = 64*128 = 8192
K = 4*128 = 512
V = 4*128 = 512
```

### Generic fallback / id15

```text
qkv=[2,49152]
qkv_size=[1,2,384,128]
head_nums=[128,128,128]
k/v_cache=[1,512,128,32] 或 [2,512,128,32] int8
k/v_scale=[128,128]
```

Q/K/V 大小：

```text
Q = 128*128 = 16384
K = 128*128 = 16384
V = 128*128 = 16384
```

该分支按 `q_group_heads=16` 分组计算 Q，按 `kv_group_heads=2` 分组计算 K/V。K cache 和 V cache 写入使用 flattened scatter：

```text
src [T, group_heads, D]
  -> reshape [T, group_heads*D]
cache group [BlockNum, C1, BlockSize, C0]
  -> transpose/reshape [BlockNum*BlockSize, group_heads*D]
scatter(dim=0, index, src)
  -> reshape/transpose/assemble 回 PA_NZ cache
```

## 6. Tiling 与 loop

当前实现使用三套 tile config，由 wrapper 根据 head 规模分支选择：

```text
TP4_TILE_CONFIG:
  q_group_heads=16
  q_vec_token_tile=4
  kv_vec_token_tile=2
  stitch_function_max_num=128
  device_sched_mode=3

TP1_TILE_CONFIG:
  q_group_heads=16
  q_vec_token_tile=4
  kv_vec_token_tile=4
  stitch_function_max_num=128
  device_sched_mode=3

GENERIC_TILE_CONFIG:
  q_group_heads=16
  q_vec_token_tile=2
  kv_vec_token_tile=2
  stitch_function_max_num=128
  device_sched_mode=3
```

Wrapper 分支：

```text
Nq > 64 or Nk > 4 or Nv > 4 -> generic fallback
Nq <= 16                    -> TP4 fast path
otherwise                   -> TP1 fast path
```

Q 路径按 `q_group_heads=16` 分组处理；TP4/TP1 K/V 路径按当前静态 `B*S` 批量处理后写 page0 cache；generic K/V 路径按 `kv_group_heads=2` 分组处理后通过 indexed scatter 写 cache。RMSNorm/RoPE 使用 `set_vec_tile_shapes(tile_t, num_heads, D)`。

历史尝试中，更大 tile 或更通用 scatter 写法出现过编译耗时、性能回退或语义限制，因此当前保留网络 case 的快路径。
AscendC-like 动态 offset `assemble` 已做过 proof：单次/少量 group 可正确运行，QKV 规模下直接逐 group 动态写会触发 AICPU 507018；两段式 full-C1 动态写可通过精度，但泳道图前三条用例劣化。该方案仅作为未来非连续 index 专用兜底模板候选，不替换当前默认计算流。

## 7. 功能限制

- TP4/TP1 INT8 cache 快路径假设 `index=torch.arange(T)`，且 `T <= block_size`。
- TP4/TP1 INT8 fast path 只写 page0 连续区域；不等价于任意 `index` 的通用 PA_NZ scatter。
- Generic fallback 已覆盖 indexed scatter 功能兜底，已用非连续 `index=[129, 3]` 验证，但不承诺替代快路径性能。
- `k_offset/v_offset` 非 `None` 未实现。
- `is_output_qkv=True` 未实现。
- `cache_mode` 只支持 `PA_NZ`。

## 8. 验证

当前验证 case 已通过 NPU precision：

```text
PASS mtp2_tp4_network_quant
PASS mtp2_tp1_network_quant
PASS pypto_qkv_rms_norm_rope_cache_id15
PASS pypto_qkv_rms_norm_rope_cache_id15_indexed
[PRECISION_PASS]
```

已记录的直接 Python benchmark：

```text
mtp2_tp4_network_quant avg_ms=0.342387 repeat=30 warmup=3 tokens=48
mtp2_tp1_network_quant avg_ms=0.684205 repeat=30 warmup=3 tokens=12
2026-06-11 recheck after generic fallback:
mtp2_tp4_network_quant avg_ms=1.575635 repeat=20 warmup=3 tokens=48
mtp2_tp1_network_quant avg_ms=0.711564 repeat=20 warmup=3 tokens=12
```

已记录的泳道图：

```text
mtp2_tp4_network_quant AICore E2E 98.04 us, utilization 37.23%
mtp2_tp1_network_quant AICore E2E 52.66 us, utilization 39.00%
```
