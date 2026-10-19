---
schema_version: 2
op_name: gather_pa_kv_cache
supported_dtypes: [bfloat16]
cache_mode: Norm
format: ND
dynamic_axes: [num_blocks, Q, total_tokens]
tolerance:
  atol: 0.0
  rtol: 0.0
---
# SPEC

## 1. Scope

`gather_pa_kv_cache` gathers discontiguous paged KV cache blocks into
contiguous key/value outputs.

Current PyPTO scope:

- BF16 only.
- `cache_mode="Norm"` only.
- ND cache layout only.
- INT32 `block_tables`, `seq_lens`, `seq_offset`.
- NPU execution.

## 2. Tensor Contract

Inputs:

| Name | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `key_cache` | `[num_blocks, block_size, key_num_heads, key_dim]` | BF16 | K cache in ND layout |
| `value_cache` | `[num_blocks, block_size, value_num_heads, value_dim]` | BF16 | V cache in ND layout |
| `block_tables` | `[Q, block_table_cols]` | INT32 | Logical-to-physical block table |
| `seq_lens` | `[Q]` or `[Q + 1]` | INT32 | Sequence lengths or cumulative lengths |
| `key_ref` | `[total_tokens, key_num_heads, key_dim]` | BF16 | Optional output buffer |
| `value_ref` | `[total_tokens, value_num_heads, value_dim]` | BF16 | Optional output buffer |
| `seq_offset` | `[Q]` or `None` | INT32 | Optional token offset into `block_tables` |

Outputs:

| Name | Shape | Dtype |
| --- | --- | --- |
| `key_out` | `[total_tokens, key_num_heads, key_dim]` | BF16 |
| `value_out` | `[total_tokens, value_num_heads, value_dim]` | BF16 |

Attributes:

| Name | Supported |
| --- | --- |
| `cache_mode` | `"Norm"` |
| `is_seq_lens_cumsum` | `False` or `True` |
| `run_mode` | `"npu"` |

## 3. Semantics

If `is_seq_lens_cumsum=False`:

```text
seq_len(q) = seq_lens[q]
output_base(q) = sum(seq_lens[:q])
```

If `is_seq_lens_cumsum=True`:

```text
seq_len(q) = seq_lens[q + 1] - seq_lens[q]
output_base(q) = seq_lens[q]
```

If `seq_offset` is provided:

```text
table_offset(q) = seq_offset[q] // block_size
```

Otherwise the wrapper creates a zero offset tensor.

For each sequence and token:

```text
logical_block = token_in_seq // block_size
slot = token_in_seq % block_size
physical_block = block_tables[q, table_offset(q) + logical_block]

key_out[output_base(q) + token_in_seq, :, :]
  = key_cache[physical_block, slot, :, :]

value_out[output_base(q) + token_in_seq, :, :]
  = value_cache[physical_block, slot, :, :]
```

## 4. Shape Constraints

- `key_cache` and `value_cache` must both be rank 4.
- `key_cache.shape[0] == value_cache.shape[0]`.
- `key_cache.shape[1] == value_cache.shape[1]`.
- `key_ref.shape[1:] == key_cache.shape[2:]`.
- `value_ref.shape[1:] == value_cache.shape[2:]`.
- `block_tables.shape[0] == Q`.
- `is_seq_lens_cumsum` must be `True`; `seq_lens.shape == [Q + 1]` and starts with 0.
- `seq_offset` values must be non-negative and divisible by `block_size`.
- Used `block_tables` entries must be in `[0, num_blocks)`.

## 5. Current Test Coverage

`test_cases.json` covers network sweep shapes:

```text
key_cache   [5513,128,1,512]
value_cache [5513,128,1,64]
blockTables [Q,8]
T           6,27,31,35,39,43,47,51,55,44
```

Additional manual validation covered:

```text
[64,128,1,128] / [64,128,1,64]
[64,128,64,128] / [64,128,64,128]
```

## 6. Precision

This is a pure BF16 copy/gather operator. Precision check uses `torch.equal`
against the CPU golden implementation.

## 7. Unsupported

- `PA_NZ` cache layout.
- INT8/FP16/FP32 cache.
- INT64 `block_tables`, `seq_lens`, `seq_offset`.
- Arbitrary non-ND layouts.
- Non-cumsum `seq_lens`.

## 8. 2026-06-11 整改同步

- 当前 ND network sweep 只支持 cumsum `seq_lens`，shape 为 `[Q + 1]`。
- `test_cases.json` 中各 level 的 `seq_lens_shape` 已从 `[Q]` 调整为 `[Q + 1]`，并设置 `is_seq_lens_cumsum=true`。
- `gather_pa_kv_cache_wrapper` 和 golden 默认 `is_seq_lens_cumsum=True`。
