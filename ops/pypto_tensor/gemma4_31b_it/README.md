# Gemma-4-31B-it fused operators

PyPTO fused-operator implementations for Gemma-4-31B-it (`google/gemma-4-31b-it`,
head_dim 256, bf16). Each operator lives in its own subdirectory with an
`_impl.py` exporting its wrapper function.

## Operators

| Operator | Subdirectory | Description |
|----------|--------------|-------------|
| `attn_softmax` | `attn_softmax/` | 3-pass tiled attention softmax (FP32 internal, bf16 I/O) replacing `F.softmax(scores * scale, dim=-1)` in the eager attention path. Opt-in via `USE_PTO_SOFTMAX`. |
| `gqa_decode_attn` | `gqa_decode_attn/` | GQA decode attention for sliding-window text layers (single-token decode, head_dim 256): averages 16 KV heads into 4, then flash-style online softmax. Opt-in via `USE_PTO_GQA`; falls back to PyTorch for prefill / non-matching shapes. |

## Tests

```bash
export TILE_FWK_DEVICE_ID=0
python3 tests/ops/gemma4_31b_it/test_attn_softmax.py
python3 tests/ops/gemma4_31b_it/test_gqa_decode_attn.py
```
