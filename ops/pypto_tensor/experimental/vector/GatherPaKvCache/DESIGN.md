# DESIGN

## 1. Active Branch

The implementation is aligned with the AscendC `GatherPaKvCache` `Norm` branch:

```text
key_cache   [num_blocks, block_size, key_num_heads, key_dim]
value_cache [num_blocks, block_size, value_num_heads, value_dim]
key_out     [total_tokens, key_num_heads, key_dim]
value_out   [total_tokens, value_num_heads, value_dim]
```

`PA_NZ` is not implemented in the current PyPTO code.

## 2. Wrapper Responsibilities

The Python wrapper performs:

- dtype and rank validation.
- `cache_mode == "Norm"` validation.
- `seq_lens` normalization to cumsum form.
- `seq_offset=None` normalization to zeros.
- output allocation when `key_ref/value_ref` are not provided.
- block table bounds checking on host.
- kernel dispatch selection.

The wrapper derives all layout dimensions from tensor shapes. No separate
Python scalar shape arguments are passed to the JIT kernel.

## 3. JIT Kernels

Two active kernels are implemented:

```text
_gather_pa_kv_cache_nd_kernel_npu
_gather_pa_kv_cache_nd_large_token_kernel_npu
```

The large-token kernel is selected when:

```text
key_num_heads * key_dim > 4096
or value_num_heads * value_dim > 4096
```

This covers the manually validated `[64,128,64,128]` large-head case. The normal
kernel covers the network sweep and smaller token payloads.

Both kernels:

1. reshape cache to `[num_blocks * block_size, heads, dim]`;
2. loop over `Q`;
3. loop over logical cache blocks for each sequence;
4. read `physical_block` from `block_tables`;
5. `view` the valid token range;
6. `assemble` it into `key_ref/value_ref`.

## 4. Tiling

Normal kernel:

```text
K tile: [16, key_num_heads, key_dim]
V tile: [32, value_num_heads, value_dim]
```

Large-token kernel:

```text
K tile: [8, key_num_heads, key_dim]
V tile: [8, value_num_heads, value_dim]
```

The large-token tile was introduced because `[1,64,128]` produced too many
small tasks, while `[8,64,128]` keeps the BF16 tile around 128 KiB.

## 5. Loop Structure

The q loop uses `pypto.loop_unroll(..., unroll_list=[2, 1])`. The block loop
uses a standard dynamic `pypto.loop`.

```text
for q_base, q_unroll in loop_unroll(Q, [2,1]):
  for q_inner in range(q_unroll):
    seq_len = seq_lens_cumsum[q+1] - seq_lens_cumsum[q]
    block_count = ceildiv(seq_len, block_size)
    for block_idx in loop(block_count):
      copy valid token range for K
      copy valid token range for V
```

## 6. Known Performance Notes

Small-token shapes may show low AICore utilization because the operator becomes
a few tiny copy tasks; there is little arithmetic for VF fusion. Large-token
shapes benefit from larger vector tiles and currently use the large-token
kernel.

## 7. Validation Artifacts

Main test entry:

```text
test_gather_pa_kv_cache.py
```

Golden:

```text
gather_pa_kv_cache_golden.py
```

Latest manually recorded target-shape swimlanes:

```text
output/output_20260530_111415_350728_1001774_C0A96050/merged_swimlane.json
output/output_20260530_111420_833310_1001774_C0A96050/merged_swimlane.json
```

## 8. 2026-06-11 整改同步

- Host wrapper now requires cumsum `seq_lens` with shape `[Q + 1]`.
- Kernel path remains the same ND gather path; the change removes the host-side non-cumsum normalization branch.
- Test cases and golden now use cumsum `seq_lens` consistently.
