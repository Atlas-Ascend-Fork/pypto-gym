# gather_pa_kv_cache

PyPTO implementation of the AscendC `GatherPaKvCache` **Norm + ND** branch.


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## Supported Scope

- `cache_mode="Norm"` only.
- `key_cache` and `value_cache` dtype `torch.bfloat16`.
- ND cache layout:
  - `key_cache`: `[num_blocks, block_size, key_num_heads, key_dim]`
  - `value_cache`: `[num_blocks, block_size, value_num_heads, value_dim]`
- Output layout:
  - `key_out`: `[total_tokens, key_num_heads, key_dim]`
  - `value_out`: `[total_tokens, value_num_heads, value_dim]`
- `block_tables`, `seq_lens`, and `seq_offset` dtype `torch.int32`.
- `seq_lens` supports both non-cumsum `[Q]` and cumsum `[Q + 1]`.
- `seq_offset=None` is accepted by the wrapper and is normalized to zeros.

Unsupported:

- `cache_mode="PA_NZ"`.
- Non-BF16 cache tensors.
- INT64 index tensors.
- CPU/sim execution path; current test entry supports NPU mode only.

## Current Network Sweep Cases

`test_cases.json` contains ten ND sweep cases:

```text
key_cache   [5513,128,1,512]
value_cache [5513,128,1,64]
blockTables [Q,8]
seqLens     [Q]
seqOffset   [Q]
key_ref     [T,1,512]
value_ref   [T,1,64]
cache_mode  Norm
dtype       BF16
```

The sweep covers `Q=4, T in {6,27,31,35,39,43,47,51,55}` and `Q=3, T=44`.

Additional manually validated shapes:

```text
heads1_k128_v64:
  key_cache   [64,128,1,128]
  value_cache [64,128,1,64]
  output      [6,1,128], [6,1,64]

heads64_k128_v128:
  key_cache   [64,128,64,128]
  value_cache [64,128,64,128]
  output      [6,64,128], [6,64,128]
```

## Implementation Notes

The wrapper derives `block_size`, head counts, and head dimensions from tensor
shapes. These dimensions are static per PyPTO specialization; only token-related
axes are dynamic in the JIT signature.

The kernel reshapes cache tensors to:

```text
key_cache_3d   [num_blocks * block_size, key_num_heads, key_dim]
value_cache_3d [num_blocks * block_size, value_num_heads, value_dim]
```

Then it loops over `q` and logical blocks, reads the physical block from
`block_tables`, and assembles valid token ranges into the output buffers.

There are two active kernels:

- normal token-size kernel for smaller per-token payloads.
- large-token kernel selected when `heads * dim > 4096`, using larger vector
  tiles for the current large-head scenario.

## Validation

Run precision cases:

```bash
source /mnt/workspace/gitCode/cann/pypto/env_setup.sh
cd /mnt/workspace/zhangsr/pypto-gym-2
PYTHONPATH=/mnt/workspace/zhangsr/pypto-gym-2/src:/tmp/pypto-wheel:${PYTHONPATH} \
  TILE_FWK_DEVICE_ID=0 \
  /opt/buildtools/Python-3.11.4/bin/python3 tests/ops/experimental/vector/GatherPaKvCache/test_gather_pa_kv_cache.py --run-mode npu
```

Run selected cases:

```bash
PYTHONPATH=/mnt/workspace/zhangsr/pypto-gym-2/src:/tmp/pypto-wheel:${PYTHONPATH} \
  /opt/buildtools/Python-3.11.4/bin/python3 tests/ops/experimental/vector/GatherPaKvCache/test_gather_pa_kv_cache.py level0 level9 --run-mode npu
```

## 2026-06-11 Update

- Directory name is `GatherPaKvCache` in pypto-gym.
- Network sweep cases now use cumsum `seq_lens` with shape `[Q + 1]`.
- Wrapper default is `is_seq_lens_cumsum=True`; non-cumsum `seq_lens` is rejected for this ND network path.
