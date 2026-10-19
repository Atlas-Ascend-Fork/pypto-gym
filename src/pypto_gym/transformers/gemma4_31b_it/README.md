# Gemma-4-31B-it PyPTO Kernel Integration

In-repo `Gemma4ForConditionalGeneration` model definition (`google/gemma-4-31b-it`)
modified to inject PyPTO fused operators on Ascend NPU hardware. Two compute paths
are routed to PyPTO kernels: the eager attention softmax and the GQA decode
attention on sliding-window text layers.

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| Attention Softmax | Yes | `nn.functional.softmax` (fp32) | Tiled PyPTO softmax inside the eager attention path |
| GQA Decode Attention | Yes | eager / flash_attn / sdpa | Sliding-window text layers, single-token decode, head_dim 256 |
| RMSNorm | No | PyTorch `Gemma4RMSNorm` | Standard implementation |
| MoE FFN | No | HuggingFace experts implementation | Built-in MoE dispatch |
| Full Attention (prefill) | No | eager / flash_attn / sdpa | Standard HuggingFace attention |
| RoPE | No | PyTorch | Full vs sliding layers use different rope types |

The softmax injection happens inside `eager_attention_forward`. The GQA decode
injection happens in `Gemma4TextAttention.forward` and is active only for decode
(sequence length 1) on sliding-window layers (head_dim 256) in inference mode.
Global attention layers (head_dim 512) fall through to the standard attention
interface.

## Switch Variables

The kernel module is registered as `sys.modules["gemma4_pto_kernels"]` and exposes:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_PTO_SOFTMAX` | `bool` | `False` | Enable tiled PyPTO softmax in eager attention forward |
| `USE_PTO_GQA` | `bool` | `False` | Enable GQA decode attention for sliding layers (decode-only, head_dim 256) |

## Related Directories

- **Ops**: `ops/pypto_tensor/gemma4_31b_it/` — PyPTO kernels (`attn_softmax/`, `gqa_decode_attn/`)
- **Tests**: `tests/ops/gemma4_31b_it/` — correctness and performance tests
