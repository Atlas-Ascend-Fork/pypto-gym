# LLaDA2.0 MoE fused operators (LLaDA2.0-mini)

PyPTO fused-operator implementation for the LLaDA2.0-mini MoE block
(`LLaDA2MoeSparseMoeBlock`). BF16 input/output, FP32 accumulation. Common
dimensions: H=2048, I=512, E=256.

## Product Support

- Ascend 950PR: not supported
- Atlas A3 Training / Inference series: supported
- Atlas A2 Training / Inference series: supported

## Operators

| Operator | File | Description |
|----------|------|-------------|
| `llada2_moe_grouped_gemm` | `llada2_moe_grouped_gemm_impl.py` | All-experts single grouped GEMM (replaces the per-expert dispatch loop) |

## Tests

```bash
export TILE_FWK_DEVICE_ID=0
python3 -m pytest tests/ops/llada2_moe/
```

Test cases are built from the model's real shapes/dtypes (balanced / uneven /
zero-token / mixed-width); see `tests/ops/llada2_moe/test_cases.json`.

## Integration

The kernel is wired into the model through `_moe_infer_pypto()` on
`LLaDA2MoeSparseMoeBlock` in
[modeling_llada2_moe.py](../../../transformers/llada2_moe/modeling_llada2_moe.py),
gated by the `USE_PTO_EXPERT_FFN` switch (opt-in).
