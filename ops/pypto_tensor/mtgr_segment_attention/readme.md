# MTGR Ragged Segment Attention

PyPTO custom-op implementation of variable-length Flash Attention forward
with host-side combined mask support (4-segment structure: causal, full-visibility, diagonal).

**算子名称**: mtgr_ragged_segment_attention

## Current Scope (Phase 1c+, HEAD_DIM=128)

| Item | Value |
|------|-------|
| Operator name | mtgr_ragged_segment_attention |
| Batch size | 1–32 (dynamic via cu_seqlens) |
| Num heads | 8 (configurable) |
| Heads per group | 2 (kernel groups 2 heads together) |
| Head dim | 128 (primary) / 64 (baseline) |
| Group head dim | 256 (HEAD_DIM=128) / 128 (HEAD_DIM=64) |
| Q tile | 256 |
| K tile | 256 |
| Mask template size | 1024 × 1024 (fixed) |
| Segment types | rules: 0=causal, 1=full-visibility, 2=diagonal |
| Test case (C1) | batch=1, segments=[1600,8,200,1200], rules=[0,1,2,2] |
| Mask dtype | FP32 (0=participate, 1=masked) |
| Q/K/V dtype | BF16 |
| Output dtype | BF16 (with FP32 L/M sidecars) |
| Tolerance | atol=1e-3, rtol=0.0078125 |
| Scope design | scope 1 (softmax) → -1 → cube → scope 2 (correction/output) |

## Scope Separation (CV-separate-platform)

The kernel uses **two separate vector scopes** because the compiler enforces CV-separate-platform:
- **Scope 1**: Softmax vector operations (scale, mask, amax, exp, sum, cast BF16)
- **Scope 2**: Correction/output operations (maximum, exp, mul, add, div, cast BF16, assemble)
- **Default scope (-1)**: Cube matmuls (Q@K^T, P@V)

**Important**: Scope 1 and scope 2 must use **different scope numbers** (1 and 2). Using the same number for both causes a CV-separate-platform error.

## Mask Types

| rules value | Name | Behavior on diagonal | Behavior off-diagonal |
|-------------|------|---------------------|----------------------|
| 0 | causal | Apply mask0 (triu, k=1) | No mask, all KV visible |
| 1 | full-visibility | No mask, all KV visible | No mask, all KV visible |
| 2 | diagonal | Apply mask1 (only diagonal=0) | No mask, all KV visible |

## Host Wrapper (KV Cache Support)

`mtgr_ragged_segment_attention_wrapper` provides:
- TND format input/output conversion
- Paged KV cache reading via `_read_kv_cache_host()`
- Mask template generation via `_build_mask_templates()`
- Segment offset conversion to segment_starts tensor

## Run

```bash
export TILE_FWK_DEVICE_ID=0
python3 tests/ops/mtgr_segment_attention/test_mtgr_ragged_segment_attention.py
```

## Three-state Markers

The test prints exactly one of the following on stdout:

- `[PRECISION_PASS]` -- kernel ran AND output matches golden within tolerance
- `[PRECISION_FAIL]` -- kernel ran but precision check failed
- *(no marker, exit != 0)* -- runtime/compile/import/aicore failure
