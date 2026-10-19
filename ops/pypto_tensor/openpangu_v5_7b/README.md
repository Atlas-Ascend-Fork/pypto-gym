# openPangu-Embedded-7B fused operators

PyPTO whole-decoder-layer fused kernel for openPangu-Embedded-7B
(`FreedomIntelligence/openPangu-Embedded-7B`, `model_type="PanguEmbedded"`).
Dense decoder, GQA (32 Q-heads / 8 KV-heads, head_dim=128), 34 layers, bf16 with
FP32 accumulation. The kernel fuses an **entire decoder layer** into a single
`@pypto.frontend.jit` kernel and is registered as a `torch.ops.pypto.*` custom
(`@allow_in_graph`).

## Product Support

- Ascend 950PR: not supported
- Atlas A3 Training / Inference series: supported
- Atlas A5 series products: supported

## Operators

| Operator | File | Description |
|----------|------|-------------|
| `pangu_fused_layer_v2_bsh` | `pangu_fused_layer_dynamic_v2_bsh.py` | One decoder layer per kernel instance, BSH KV-cache `[max_kv_len, kv_size]`, dynamic `actual_kv_len`. Includes QKV/O bias. |

### Fused ops per layer

Residual-add + RMSNorm -> QKV GEMM (+bias) -> RoPE (Q/K) -> KV-cache write
-> tiled GQA attention with online softmax (QK^T -> scale -> softmax -> PV)
-> O GEMM (+bias) -> residual-add + RMSNorm -> gate/up GEMMs + SwiGLU -> down GEMM.

The kernel is **decode-only** (batch=1, q_len=1, `hidden_states=[1,1,4096]`);
prefill stays on the PyTorch `PanguEmbeddedDecoderLayer`.

## Files

| File | Description |
|------|-------------|
| `pangu_fused_layer_dynamic_v2_bsh.py` | Per-layer dynamic kernel + `PanguFusedLayerV2BSHModule`, `DynamicFusedLayerConfigV2BSH`, shared `_pypto_rms_norm` helper |

## Switch

Defined in [`__init__.py`](./__init__.py) (opt-in, default off):

| Switch | Default | Description |
|--------|---------|-------------|
| `USE_PTO_FUSED_LAYER` | `False` | Master switch — wire the PyPTO fused decode path |

## Tests

```bash
source env_setup.sh
# python3 -m pytest tests/ops/openpangu_v5_7b/   # TODO: per-op correctness tests
```

## Integration

The kernel is wired into the model through `PanguEmbeddedModel.forward` in
[modeling_openpangu_dense.py](../../../transformers/openpangu_v5_7b/modeling_openpangu_dense.py),
gated by `USE_PTO_FUSED_LAYER`. Prefill and the disabled case fall back to the
PyTorch `PanguEmbeddedDecoderLayer`; decode (when enabled) iterates over layers
and calls `self.fused_layers[i](...)` per layer.
