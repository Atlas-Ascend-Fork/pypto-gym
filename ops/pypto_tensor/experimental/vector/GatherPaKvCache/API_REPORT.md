---
schema_version: 2
op_name: gather_pa_kv_cache
supported_dtypes: [bfloat16]
cache_mode: Norm
format: ND
tiling_required: true
feasibility: feasible_with_constraints
---
# API_REPORT

## 1. Overview

The current implementation maps the AscendC `GatherPaKvCache` `Norm` ND branch
to PyPTO `view + assemble` copy operations.

## 2. API Mapping

| Need | PyPTO API | Usage |
| --- | --- | --- |
| JIT kernel | `@pypto.frontend.jit` | NPU execution |
| Dynamic loops | `pypto.loop`, `pypto.loop_unroll` | Iterate sequence and block axes |
| Cache flattening | `pypto.reshape(..., inplace=True)` | `[num_blocks, block_size, H, D] -> [num_blocks * block_size, H, D]` |
| Source tile | `pypto.view` | Read cache block range |
| Writeback | `pypto.assemble` | Write gathered range into output |
| Tiling | `pypto.set_vec_tile_shapes` | Control vector task granularity |

## 3. Dynamic and Static Axes

Dynamic axes:

```text
num_blocks
Q
total_tokens
block_table_cols
```

Static-specialized axes:

```text
block_size
key_num_heads
key_dim
value_num_heads
value_dim
```

The static-specialized axes are read from tensor shape inside the kernel.

## 4. Constraints

- Only `cache_mode="Norm"`.
- Only BF16 cache and output tensors.
- Only INT32 index tensors.
- `seq_offset` must be divisible by `block_size`.
- `seq_lens` must already be cumsum `[Q + 1]`; non-cumsum `seq_lens` is rejected by the wrapper.

## 5. Validation Status

Verified by `test_gather_pa_kv_cache.py` for all entries in `test_cases.json`.
Additional manual target-shape validation covered:

```text
heads1_k128_v64
heads64_k128_v128
```

The latest manual run passed precision for both target shapes:

```text
[CUSTOM_PRECISION_PASS]
```

## 6. Risks

- Small shapes are scheduling-bound rather than compute-bound.
- `PA_NZ` is intentionally out of scope for this implementation.
- The current large-token branch is a performance specialization, not a
  semantic requirement.

## 7. 2026-06-11 整改同步

- Wrapper/golden default `is_seq_lens_cumsum=True`.
- `test_cases.json` now records cumsum `seq_lens_shape=[Q+1]` for all network sweep levels.
- Non-cumsum `seq_lens` is no longer normalized implicitly in this ND network path.
