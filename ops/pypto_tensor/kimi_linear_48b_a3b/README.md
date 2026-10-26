# Kimi-Linear-48B-A3B fused operators

PyPTO fused-operator implementations targeted at Kimi-Linear-48B-A3B-Instruct
(the `KimiDeltaAttention` linear-attention core). Each operator lives in its own
subdirectory with an `_impl.py` exporting a `*_wrapper` function and a `README.md`
describing the contract.

## Operators

| Operator | Subdirectory | Status |
|----------|--------------|--------|
| `kda_fused_prefill` (Kimi Delta Attention, prefill) | [`kda/`](kda/) | Available |
| `kda_fused_decode` (Kimi Delta Attention, decode) | [`kda/`](kda/) | Available |

## Common shape constraints

Kimi-Linear-48B-A3B's KDA (`linear_attn_config`) has:

| Symbol | Meaning | Value |
|--------|---------|-------|
| `D` | Per-head dimension (`head_k_dim == head_v_dim`) | 128 |
| `H` | KDA head count (`num_heads == num_k_heads`)     | 32  |

The kernels keep the per-channel log-gate (`g in [-5, 0]`) numerically stable
in fp32. The wrappers raise `NotImplementedError` outside the supported
envelope (e.g. tensors off the bound NPU) and the modeling layer falls back to
the reference torch path.

## Integration

The kernel is wired into the model through the 
[modeling_kimi.py](../../../transformers/kimi_linear_48b_a3b/modeling_kimi.py).
The ops package is injected as sys.modules["kimi_linear_48b_a3b_pto_kernels"],
exposing USE_PTO_KDA (opt-in switch, default off) together with the PyPTO prefill
and decode entry points.
