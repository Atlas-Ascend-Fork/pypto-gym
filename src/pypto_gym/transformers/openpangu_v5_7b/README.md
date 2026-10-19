# openPangu-Embedded-7B PyPTO Fused-Layer Integration

`PanguEmbeddedForCausalLM` model definition (adapted from
[`FreedomIntelligence/openPangu-Embedded-7B`](https://huggingface.co/FreedomIntelligence/openPangu-Embedded-7B)
`modeling_openpangu_dense.py`), modified to route the **whole decoder layer** through
PyPTO fused kernels on Ascend NPU hardware during decode.

Unlike the single-op integrations (qwen3_1_7b RMSNorm+RoPE, gemma4 softmax/GQA), the
Pangu integration fuses an **entire decoder layer** — RMSNorm -> QKV (+bias) -> RoPE
-> KV-cache -> GQA attention -> O proj (+bias) -> RMSNorm -> SwiGLU FFN — into one
`@pypto.frontend.jit` kernel.

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| Decoder layer (decode) | Yes | PyTorch `PanguEmbeddedDecoderLayer` | Decode-only (batch=1, q_len=1); prefill always falls back |
| Attention (prefill) | No | `npu_fused_infer_attention_score` | Standard NPU eager path |
| RMSNorm | Fused-in-layer | `npu_rms_norm` / `npu_add_rms_norm` | Only fused when the layer kernel runs |
| RoPE | Fused-in-layer | `PanguEmbeddedRotaryEmbedding` | Only fused when the layer kernel runs |
| Embedding / LM head | No | `VocabParallelEmbedding` | TP-sharded via `module.linear` |

## Switch Variables

Read dynamically from the ops package (mirrors the minimax_m27 pattern):

| Variable | Location | Default | Description |
|----------|----------|---------|-------------|
| `USE_PTO_FUSED_LAYER` | `ops.pypto_tensor.openpangu_v5_7b` | `False` | Master switch — wire the PyPTO fused decode path |
The modeling reads the switch at `PanguEmbeddedModel.__init__` time and lazily imports
the heavy kernel module only when enabled, so importing the modeling alone does not
pull in `pypto`.

## Kernel API Contract

When `USE_PTO_FUSED_LAYER=True`, the decode branch of `PanguEmbeddedModel.forward`
iterates over decoder layers and calls the per-layer fused kernel:

```python
# Per-layer dynamic kernel
output, new_residual = self.fused_layers[i](
    hidden_states, residual, cos, sin,
    key_cache, value_cache,
    input_ln_weight, post_ln_weight,
    qkv_weight, qkv_bias, o_weight, o_bias,
    gate_weight, up_weight, down_weight,
    actual_kv_len,
) -> (Tensor, Tensor)
) -> (Tensor, Tensor)
```

Non-contiguous (NK-transposed) weight tensors are materialized to contiguous NK format
before being passed in, since pypto requires contiguous tensors.

## Usage

```python
import ops.pypto_tensor.openpangu_v5_7b as _pto
_pto.USE_PTO_FUSED_LAYER = True            # enable BEFORE constructing the model

# The model class is loaded through the cann-recipes PanguEmbeddedRunner, which
# imports ``models.modeling_openpangu_dense``. The ask script injects this
# package's modeling/configuration into ``sys.modules["models.*"]`` so the runner
# picks up the PyPTO-integrated version. See modeling/transformers/openpangu_v5_7b/.
```

## HuggingFace `auto_map`

```json
{
  "auto_map": {
    "AutoConfig": "configuration_openpangu_dense.PanguEmbeddedConfig",
    "AutoModelForCausalLM": "modeling_openpangu_dense.PanguEmbeddedForCausalLM"
  }
}
```

> `model_type` is `"PanguEmbedded"` (non-native). Full inference is driven by the
> cann-recipes `PanguEmbeddedRunner` (needs the `executor`/`module` packages and a
> YAML); it is **not** a standalone `AutoModelForCausalLM.from_pretrained` model —
> see the dependency note below.

## File Table

| File | Description |
|------|-------------|
| `configuration_openpangu_dense.py` | `PanguEmbeddedConfig` — 4096 hidden, 34 layers, 32 Q-heads, 8 KV-heads, head_dim=128, vocab 153376; accepts GPT/LLaMA-style alias kwargs |
| `modeling_openpangu_dense.py` | Full model graph with PyPTO fused-layer dispatch in `PanguEmbeddedModel.forward` (switch-gated, decode-only) |
| `config.json` | Model config with `auto_map` |

## External Dependency Note

`modeling_openpangu_dense.py` is ported faithfully from the cann-recipes-infer repo
and still imports the cann-recipes `executor` / `module` packages
(`QKVParallelLinear`, `ColumnParallelLinear`, `RowParallelLinear`,
`MergedColumnParallelLinear`, `VocabParallelEmbedding`, `FusedMoEGMM`,
`default_weight_loader`, `get_default_group`/`init_comm_group`) and is driven by the
`PanguEmbeddedRunner` + `runner_settings` YAML flow. These packages are **not** part
of pypto-gym; point `PYTHONPATH` at a `cann-recipes-infer` checkout to run it. The
PyPTO fused kernels themselves (in `ops/pypto_tensor/openpangu_v5_7b/`) depend only
on `pypto` + `torch`.

## Environment

| Component | Version |
|-----------|---------|
| torch | 2.6.0 |
| torch_npu | 7.2.RC1.alpha002 |
| transformers | 4.55.0 |
| CANN | 8.3.RC1.alpha002 |
| pypto | 9.1.0 |

## Related Directories

- **Ops**: `ops/pypto_tensor/openpangu_v5_7b/` — PyPTO fused-layer kernels
- **End-to-end**: `modeling/transformers/openpangu_v5_7b/` — `ask_openpangu_v5_7b.py` + reference YAML
