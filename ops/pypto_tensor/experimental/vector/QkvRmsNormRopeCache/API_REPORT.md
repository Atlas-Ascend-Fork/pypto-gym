---
schema_version: 2
op_name: qkv_rms_norm_rope_cache
supported_dtypes: [int8_cache]
axes_list: [T, H, D]
shape_constraints: ["D=128 for current network cases", "Nk=Nv", "Nqkv=Nq+Nk+Nv", "cache_mode=PA_NZ"]
tiling_required: true
feasibility: feasible_with_constraints
---
# API_REPORT

## 1. 概述

`qkv_rms_norm_rope_cache` 拆解为 SplitVD、RMSNorm、RoPE、量化、reshape/transpose 和 PA_NZ cache 写入。当前目标功能是 INT8 cache 对称量化路径，包含既有 TP4/TP1 快路径和超出快路径 head 规模时使用的 generic fallback：

```text
K: RMSNorm -> RoPE -> quant -> INT8 PA_NZ cache
V: quant -> INT8 PA_NZ cache
```

当前代码只保留 INT8 PA_NZ 路径；其他 cache dtype 会在 wrapper 中报错。

## 2. 公式分解

1. `qkv` 按 head 维切分为 Q/K/V。
2. Q/K 沿最后一维做 RMSNorm。
3. Q/K 做 half-and-half RoPE。
4. Q reshape 写出到 `[T, Nq*D]`。
5. INT8 cache：K/V 用 `k_scale/v_scale` 对称量化后写回 INT8 PA_NZ cache。

INT8 量化：

```text
k_quant = saturate_int8(round(k_rope / k_scale))
v_quant = saturate_int8(round(v / v_scale))
```

## 3. API 映射

| 原子操作 | PyPTO API | 说明 |
| --- | --- | --- |
| 切片 | `pypto.view` | 从 `[T, Nqkv*D]` 中视图切出 Q/K/V |
| reshape | `pypto.reshape` | Q/K/V 2D/3D 转换，PA_NZ cache 视图转换 |
| dtype 转换 | `pypto.cast` | BF16/FP32/INT32/INT8 转换 |
| reduction | `pypto.sum` | RMSNorm mean square |
| sqrt/div/mul/add | `pypto.sqrt` / `pypto.div` / 运算符 | RMSNorm 和 RoPE |
| concat | `pypto.concat` | `rotate_half = concat(-x2, x1)` |
| INT8 quant | `pypto.cast(..., CAST_RINT)` + saturation | 对称量化 |
| INT8 cache write fast path | `pypto.reshape` + `move` | 当前 TP4/TP1 网络性能 case 的 page0 连续写入 |
| INT8 cache write fallback | `pypto.reshape` + `pypto.transpose` + `pypto.scatter` + `pypto.assemble` | Generic fallback 按 `index` 写 INT8 PA_NZ cache |
| tiling | `pypto.set_vec_tile_shapes` | Vector tile 配置 |

## 4. 入口约束

- `qkv/q_gamma/k_gamma/cos/sin/q_out` 为 BF16。
- `index` 为 INT64。
- 当前实现要求 `k_cache/v_cache` 为 INT8，并要求 `k_scale/v_scale` 非空。
- `k_cache/v_cache` 非 INT8 时直接报错。
- `k_offset/v_offset` 当前必须为 `None`。
- JIT tensor 参数动态 token 轴使用 `pypto.DYNAMIC`，cache 轴保持 static。

## 5. Shape 约束

- `D` 为偶数，当前网络 case 固定 `D=128`。
- `Nqkv == Nq + Nk + Nv`。
- `Nk == Nv`。
- `cache_mode == "PA_NZ"`。
- 当前验证 case 固定 `C0=32`、`BlockSize=128`。
- TP4/TP1 INT8 fast path 当前要求 `index=torch.arange(T)` 且 `T <= BlockSize`。
- Generic fallback 已验证 indexed scatter，当前覆盖 `D=128`、`C0=32`、`BlockSize=128`。

## 6. 当前验证 case

| Case | qkv | qkv_size | head_nums | cache | scale |
| --- | --- | --- | --- | --- | --- |
| `mtp2_tp4_network_quant` | `[48,2304]` | `[16,3,18,128]` | `[16,1,1]` | `[11898,4,128,32]` INT8 | `[1,128]` |
| `mtp2_tp1_network_quant` | `[12,9216]` | `[4,3,72,128]` | `[64,4,4]` | `[11898,16,128,32]` INT8 | `[4,128]` |
| `pypto_qkv_rms_norm_rope_cache_id15` | `[2,49152]` | `[1,2,384,128]` | `[128,128,128]` | `[1,512,128,32]` INT8 | `[128,128]` |
| `pypto_qkv_rms_norm_rope_cache_id15_indexed` | `[2,49152]` | `[1,2,384,128]` | `[128,128,128]` | `[2,512,128,32]` INT8 | `[128,128]` |

## 7. 泛化边界

功能上可泛化：

```text
T, B, S, Nq, Nk, Nv, D, BlockNum, BlockSize, C0
```

当前实现已经动态化：

```text
T
```

当前实现静态特化：

```text
Nq/Nk/Nv, D, qkv_width, q_out_width, cache shape, scale shape
```

TP4/TP1 快路径仍未泛化到任意 `index`。Generic fallback 已补充基于 `index` 的 flattened scatter 功能兜底，wrapper 在 `Nq>64` 或 `Nk/Nv>4` 时选择 generic kernel，以避免影响既有性能 case 的快路径。

## 8. 验证状态

已记录的 NPU precision：

```text
PASS mtp2_tp4_network_quant
PASS mtp2_tp1_network_quant
PASS pypto_qkv_rms_norm_rope_cache_id15
PASS pypto_qkv_rms_norm_rope_cache_id15_indexed
[PRECISION_PASS]
```

已记录的直接 Python benchmark：

```text
mtp2_tp4_network_quant avg_ms=0.342387 repeat=30 warmup=3 tokens=48 ms_per_token=0.007133
mtp2_tp1_network_quant avg_ms=0.684205 repeat=30 warmup=3 tokens=12 ms_per_token=0.057017
2026-06-11 recheck after generic fallback:
mtp2_tp4_network_quant avg_ms=1.575635 repeat=20 warmup=3 tokens=48 ms_per_token=0.032826
mtp2_tp1_network_quant avg_ms=0.711564 repeat=20 warmup=3 tokens=12 ms_per_token=0.059297
```

已记录的泳道图：

```text
mtp2_tp4_network_quant:
  output/output_20260526_171035_025280_576766_C0A96050/merged_swimlane.json
  AICore E2E 98.04 us, utilization 37.23%

mtp2_tp1_network_quant:
  output/output_20260526_171045_732676_576766_C0A96050/merged_swimlane.json
  AICore E2E 52.66 us, utilization 39.00%
```

## 9. 风险

- TP4/TP1 INT8 fast path 不是通用 PA_NZ scatter，不能覆盖任意 `index`。
- Generic fallback 使用 indexed scatter 作为功能兜底，当前目标不是替代快路径性能。
- `k_offset/v_offset` 非对称量化未实现。
- `is_output_qkv=True` 未实现。
- benchmark 是 host 侧端到端平均耗时；泳道图 AICore E2E 是核侧分析口径，二者不能直接混用。
