# LLaDA2.0-mini MoE PyPTO Kernel Integration

In-repo `LLaDA2MoeForCausalLM` model definition (LLaDA2.0-mini, a diffusion-LM
Mixture-of-Experts model from inclusionAI) modified to route the MoE block
through the PyPTO fused grouped-GEMM operator on Ascend NPU hardware. The
integration replaces the per-expert Python-loop inference in
`LLaDA2MoeSparseMoeBlock` with a single fused grouped GEMM over all experts.

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| Expert FFN (grouped GEMM) | Yes | Per-expert Python loop | All experts in one kernel; FP32 accumulation |
| Router gating | No | PyTorch `LLaDA2MoeGate` | Group-limited top-k + renorm, host-side |
| RMSNorm | No | PyTorch `LLaDA2MoeRMSNorm` | Standard implementation |
| Attention (GQA) | No | eager / flash_attn / sdpa | Standard HuggingFace attention, partial rotary |
| Dense MLP | No | PyTorch `LLaDA2MoeMLP` | Standard SwiGLU MLP |
| RoPE | No | PyTorch `LLaDA2MoeRotaryEmbedding` | `partial_rotary_factor` (0.5) |

The PyPTO path is taken in `_moe_infer_pypto()` on `LLaDA2MoeSparseMoeBlock`:
tokens are sorted by expert assignment, expert boundaries are computed via
cumsum, and a single `grouped_gemm` kernel call processes all experts at once,
eliminating the per-expert Python loop.

## Switch Variable

Exposed in `sys.modules["llada2_pto_kernels"]`, opt-in (default `False`):

| Variable | Description |
|----------|-------------|
| `USE_PTO_EXPERT_FFN` | Route the MoE expert FFN through the fused grouped-GEMM kernel |

## HuggingFace `auto_map`

```json
{
  "auto_map": {
    "AutoConfig": "pypto_gym/transformers/llada2_moe/configuration_llada2_moe.LLaDA2MoeConfig",
    "AutoModelForCausalLM": "pypto_gym/transformers/llada2_moe/modeling_llada2_moe.LLaDA2MoeForCausalLM"
  }
}
```

## Files

| File | Description |
|------|-------------|
| `configuration_llada2_moe.py` | `LLaDA2MoeConfig` — MoE config (routing params, expert/group topology) |
| `modeling_llada2_moe.py` | Model graph; PyPTO path in `LLaDA2MoeSparseMoeBlock._moe_infer_pypto()` |

## Related Directories

- **Ops**: `ops/pypto_tensor/llada2_moe/` — PyPTO fused grouped-GEMM kernel
- **Tests**: `tests/ops/llada2_moe/` — correctness tests
