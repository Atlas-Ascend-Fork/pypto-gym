# Qwen3.6-27B PyPTO Kernel Integration

HuggingFace `Qwen3_5ForConditionalGeneration` multimodal model definition (27B variant) modified to inject PyPTO fused operators on Ascend NPU hardware. This is the 27-billion parameter version of the Qwen3.5 vision-language model, sharing the same codebase as qwen3_5_9b but with a distinct kernel module namespace and larger model dimensions.

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| Gated Delta Rule (chunk) | Yes | `chunk_gated_delta_rule` (FLA or torch) | Fused chunk-level GDR kernel for prefill with `qwen3_6_27b_pto_kernels` |
| RMSNorm | No | PyTorch `Qwen3_5RMSNorm` | Standard implementation, not fused |
| Gated RMSNorm | No | `FusedRMSNormGated` (FLA) or `Qwen3_5RMSNormGated` | Post-attention gated normalization |
| Full Attention | No | eager / flash_attn / sdpa | Standard HuggingFace attention |
| MLP (SwiGLU) | No | PyTorch `nn.Linear` | Standard |
| RoPE (MRoPE) | No | PyTorch `Qwen3_5TextRotaryEmbedding` | Interleaved MRoPE |
| Vision Encoder | No | PyTorch `Qwen3_5VisionModel` | Full vision pipeline |
| Conv1D (causal) | No | `causal_conv1d_fn` (optional) or torch | Causal convolution pre-processing |

The PyPTO injection point is identical in structure to qwen3_5_9b but targets a separate kernel namespace (`qwen3_6_27b_pto_kernels`) to allow independent kernel compilation and tuning for the 27B model's specific tensor shapes and parallelism requirements.

## Switch Variables

The kernel module is expected to expose the following in `sys.modules["qwen3_6_27b_pto_kernels"]`:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_PTO_GATED_DELTA_RULE` | `bool` | `False` | Enable PyPTO fused Gated Delta Rule chunk kernel for prefill |

## Kernel API Contract

When `USE_PTO_GATED_DELTA_RULE` is `True`, the kernel must provide:

```python
def gated_delta_rule_wrapper(
    query: torch.Tensor,   # [B, S, Nv, D]  (q/k already broadcast to Nv=48 by repeat_interleave)
    key: torch.Tensor,     # [B, S, Nv, D]
    value: torch.Tensor,   # [B, S, Nv, D]
    *,
    g: torch.Tensor,       # [B, S, Nv]  (fp32 log-gate, <= 0)
    beta: torch.Tensor,    # [B, S, Nv]
    initial_state: torch.Tensor | None,   # [B, Nv, D, D] or None (chunk path: None)
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:   # (core_attn_out [B,S,Nv,D], last_state [B,Nv,D,D])
    ...
```

Envelope (raises `NotImplementedError` otherwise → upstream fallback): `B=1`, `Nv=48`,
`D=128`, `initial_state is None`, `use_qk_l2norm_in_kernel=True`. The modeling hook checks
`sys.modules.get("qwen3_6_27b_pto_kernels")` and falls back to the upstream chunk kernel on
`NotImplementedError` (so the decode/recurrent path always runs upstream).

## 27B vs 9B Model Dimensions

Real values from each model's `config.json` (text config). The only dimension the
gated_delta_rule kernel depends on is `linear_num_value_heads` (Nv, the per-head loop
count); `linear_*_head_dim` is 128 for both.

| Parameter | 9B | 27B |
|-----------|----:|----:|
| `hidden_size` | 4096 | 5120 |
| `intermediate_size` | 12288 | 17408 |
| `num_hidden_layers` | 32 | 64 |
| `num_attention_heads` | 16 | 24 |
| `linear_num_key_heads` | 16 | 16 |
| `linear_num_value_heads` (Nv) | 32 | **48** |
| `linear_{key,value}_head_dim` (D) | 128 | 128 |

The kernel is compiled for `Nv=48`, `D=128` (enforced by the wrapper envelope).

## Usage

The supported entry point is the ask script, which performs the
inject-before-transformers-import + enable-after-NPU-load sequence correctly:

```bash
MODEL_PATH=/path/to/Qwen3.6-27B \
  python3 modeling/transformers/qwen3_6_27b/ask_Qwen3.6-27B.py --use_pypto --device 0
# baseline vs PyPTO timing:
python3 modeling/transformers/qwen3_6_27b/bench_qwen3_6_27b.py --model-path /path/to/Qwen3.6-27B [--use_pypto]
```

Injection contract (what the ask script does):

```python
import sys
import qwen3_6_27b_pto_kernels as pk        # the ops package, before `import transformers`
sys.modules["qwen3_6_27b_pto_kernels"] = pk
# ... load model, move to NPU ...
pk.USE_PTO_GATED_DELTA_RULE = True          # enable AFTER the model is on the NPU
```

The kernel module exposes `USE_PTO_GATED_DELTA_RULE` and `gated_delta_rule_wrapper`
(see `ops/pypto_tensor/qwen3_6_27b/__init__.py`). Set the flag to `False`
to disable at runtime.

> **Note on loading:** the real `Qwen3.6-27B/config.json` has **no `auto_map`**, so the model
> loads the **built-in** `qwen3_5` architecture from the installed `transformers` package
> (`trust_remote_code` has no bundled file to load). `modeling_qwen3_5.py` / `configuration_qwen3_5.py`
> in this repo are **archival snapshots** of that built-in code with the PyPTO injection hook + Huawei
> NOTICE added (for reference/restore; upstream relative imports kept) — they are **not** loaded at
> runtime: the ask script imports the built-in class and injects PyPTO via `sys.modules` + monkey-patch.
> There is no `Qwen3_5Config()`-from-scratch path; always load real weights via `from_pretrained(MODEL_PATH)`.

## File Table

| File | Description |
|------|-------------|
| `__init__.py` | Package init (CANN license header) |
| `configuration_qwen3_5.py` | `Qwen3_5Config`, `Qwen3_5TextConfig`, `Qwen3_5VisionConfig` — archival snapshot, byte-identical to the qwen3_5_9b config class except the reuse NOTICE (its class defaults are the upstream/9B values). The real 27B dimensions (hidden_size 5120, 64 layers, Nv 48) come from the weights `config.json` at load, not from these class defaults |
| `modeling_qwen3_5.py` | Full model graph with GDR chunk injection at `sys.modules.get("qwen3_6_27b_pto_kernels")` — identical code structure to qwen3_5_9b but separate kernel namespace |

## Related Directories

- **Ops**: `ops/pypto_tensor/qwen3_6_27b/` — PyPTO Gated Delta Rule kernel (`gated_delta_rule/`)
- **Tests**: `tests/ops/qwen3_6_27b/` — single-op precision test (`[PRECISION_PASS]`, chunk vs torch golden) + `test_cases.json`
