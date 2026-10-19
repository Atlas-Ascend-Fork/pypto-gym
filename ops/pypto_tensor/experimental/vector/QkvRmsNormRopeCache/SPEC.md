---
schema_version: 2
op_name: qkv_rms_norm_rope_cache
supported_dtypes: [int8_cache]
dynamic_axes: [T]
network_cases:
  - mtp2_tp4_network_quant
  - mtp2_tp1_network_quant
  - pypto_qkv_rms_norm_rope_cache_id15
  - pypto_qkv_rms_norm_rope_cache_id15_indexed
tolerance:
  atol: 0.004
  rtol: 0.004
---
# SPEC

## 1. 基础信息

- 算子名称：`qkv_rms_norm_rope_cache`
- 算子类型：QKV SplitVD + RMSNorm + RoPE + PA_NZ KV cache 更新融合算子。
- 当前版本范围：当前目标功能为 INT8 对称量化 cache 路径，包含 TP4/TP1 快路径和 generic fallback。
- 当前验证用例：`mtp2_tp4_network_quant`、`mtp2_tp1_network_quant`、`pypto_qkv_rms_norm_rope_cache_id15`、`pypto_qkv_rms_norm_rope_cache_id15_indexed`。

## 2. 数学公式

设 `T = qkv.shape[0]`，`qkv_size=[B,S,Nqkv,D]`，`head_nums=[Nq,Nk,Nv]`。

SplitVD:

```text
q = reshape(qkv[:, 0:Nq*D], [T, Nq, D])
k = reshape(qkv[:, Nq*D:(Nq+Nk)*D], [T, Nk, D])
v = reshape(qkv[:, (Nq+Nk)*D:], [T, Nv, D])
```

RMSNorm:

```text
rms(x, gamma) = x * rsqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma
```

Half-and-half RoPE:

```text
rotate_half(x) = concat(-x[..., D/2:D], x[..., 0:D/2])
rope(x) = x * cos + rotate_half(x) * sin
```

Q 输出：

```text
q_out = reshape(rope(rms(q, q_gamma)), [T, Nq*D])
```

INT8 cache 路径：

```text
k_int8 = saturate_int8(round(rope(rms(k, k_gamma)) / k_scale))
v_int8 = saturate_int8(round(v / v_scale))
k_cache[...] = k_int8
v_cache[...] = v_int8
```

TP4/TP1 INT8 fast path 按当前网络性能用例使用 page0 连续写入，测试输入满足 `index=torch.arange(T)`。
Generic fallback 使用基于 `index` 的 INT8 PA_NZ cache 写入，用于超出 TP4/TP1 快路径 head 规模的功能兜底。

## 3. 输入输出规格

| 名称 | shape | dtype | 说明 |
| --- | --- | --- | --- |
| `qkv` | `[T, Nqkv * D]` | `bfloat16` | QKV 融合输入 |
| `q_gamma` | `[D]` | `bfloat16` | Q RMSNorm gamma |
| `k_gamma` | `[D]` | `bfloat16` | K RMSNorm gamma |
| `cos` | `[T, D]` | `bfloat16` | RoPE cos |
| `sin` | `[T, D]` | `bfloat16` | RoPE sin |
| `index` | `[T]` | `int64` | cache slot |
| `q_out` | `[T, Nq * D]` | `bfloat16` | Q 输出 buffer |
| `k_cache` | `[BlockNum, Nk * D / C0, BlockSize, C0]` | `int8` | K PA_NZ cache |
| `v_cache` | `[BlockNum, Nv * D / C0, BlockSize, C0]` | `int8` | V PA_NZ cache |
| `k_scale` | `[Nk, D]` | `float32` | INT8 K cache 必需 |
| `v_scale` | `[Nv, D]` | `float32` | INT8 V cache 必需 |
| `k_offset/v_offset` | optional | optional | 当前必须为 `None` |

输出为更新后的 `(q_out, k_cache, v_cache)`。

## 4. Shape 约束

- `D` 必须为偶数；当前网络目标为 `D=128`。
- `Nqkv == Nq + Nk + Nv`。
- `Nk == Nv`。
- `cache_mode == "PA_NZ"`。
- 当前重点覆盖 `C0=32`、`BlockSize=128`。
- TP4/TP1 INT8 fast path 要求当前网络输入语义：`index=torch.arange(T)`，`T <= BlockSize`。
- Generic fallback 已验证基于 `index` 的写入，当前覆盖 `D=128`、`C0=32`、`BlockSize=128`、INT8 PA_NZ cache。

## 5. 当前验证场景

| Case | qkv_size | head_nums | qkv shape | q_out | k/v_cache | k/v_scale | cache dtype |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mtp2_tp4_network_quant` | `[16,3,18,128]` | `[16,1,1]` | `[48,2304]` | `[48,2048]` | `[11898,4,128,32]` | `[1,128]` | INT8 |
| `mtp2_tp1_network_quant` | `[4,3,72,128]` | `[64,4,4]` | `[12,9216]` | `[12,8192]` | `[11898,16,128,32]` | `[4,128]` | INT8 |
| `pypto_qkv_rms_norm_rope_cache_id15` | `[1,2,384,128]` | `[128,128,128]` | `[2,49152]` | `[2,16384]` | `[1,512,128,32]` | `[128,128]` | INT8 |
| `pypto_qkv_rms_norm_rope_cache_id15_indexed` | `[1,2,384,128]` | `[128,128,128]` | `[2,49152]` | `[2,16384]` | `[2,512,128,32]` | `[128,128]` | INT8 |

## 6. 精度要求

- Golden 使用 torch CPU 实现。
- `q_out` 与 golden 使用 `rtol=0.004, atol=0.004`。
- INT8 cache 与 golden 的量化结果逐元素对齐到容差检查。
- 未更新位置保持输入 cache 原值。

## 7. 暂不支持

- `cache_mode` 非 `PA_NZ`。
- `is_output_qkv=True` 的 q/k/v before-quant 输出。
- `k_offset/v_offset` 非 `None` 的非对称量化。
- TP4/TP1 快路径不支持任意 `index`，仍是当前网络 case 的连续 page0 快路径。
- Generic fallback 已覆盖 indexed scatter 功能兜底，但未承诺替代快路径性能。
