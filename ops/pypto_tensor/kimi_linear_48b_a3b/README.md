# Kimi-Linear-48B-A3B fused operators

PyPTO fused-operator implementations targeted at Kimi-Linear-48B-A3B-Instruct
(the `KimiDeltaAttention` linear-attention core). Each operator lives in its own
subdirectory with an `_impl.py` exporting a `*_wrapper` function and a `README.md`
describing the contract.

## Operators

| Operator | Subdirectory | Status |
|----------|--------------|--------|
| `kda_chunk` (Kimi Delta Attention, prefill/chunk) | [`kda/`](kda/) | Available |

PyPTO covers the prefill (`mode == 'chunk'`) path only; decode runs the upstream
torch recurrent path.

## Common shape constraints

Kimi-Linear-48B-A3B's KDA (`linear_attn_config`) has:

| Symbol | Meaning | Value |
|--------|---------|-------|
| `D` | Per-head dimension (`head_k_dim == head_v_dim`) | 128 |
| `H` | KDA head count (`num_heads == num_k_heads`)     | 32  |

The chunk kernel processes each chunk in subchunks of 16 steps for fp32 numerical
stability of the per-channel log-gate (`g in [-5, 0]`). Wrappers mirror the upstream
`fla.ops.kda.chunk_kda` signature and raise `NotImplementedError` outside the
supported envelope (e.g. tensors off the bound NPU).

## Integration

The kernel is wired into the model through the
[modeling_kimi.py](../../../transformers/kimi_linear_48b_a3b/modeling_kimi.py)
dispatch hook: the ops package is injected as
`sys.modules["kimi_linear_48b_a3b_pto_kernels"]`, exposing `USE_PTO_KDA` (opt-in
switch, default off) and `kda_chunk_wrapper`; decode uses the unchanged upstream
`fused_recurrent_kda`. For torch.compile / torchair ACLGraph capture,
`kda_chunk_pypto` is registered as `pypto::kda_chunk_kimi` (Meta + NPU) and is
selected under `USE_PTO_KDA_GRAPH`.
