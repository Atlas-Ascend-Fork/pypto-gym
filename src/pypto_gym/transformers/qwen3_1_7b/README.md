# Qwen3-1.7B PyPTO Kernel Integration

`Qwen3ForCausalLM` model definition (from transformers 4.51.0 `/models/qwen3/`), minimally modified to inject PyPTO fused operator on Ascend NPU hardware.

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| Q/K RMSNorm + RoPE | Yes | PyTorch eager `q_norm/k_norm` + `apply_rotary_pos_emb` | Prefill only (S>1); decode falls back |
| Attention (Q/K/V/O) | No | eager / sdpa | Standard HuggingFace |
| MLP (SwiGLU) | No | PyTorch `nn.Linear` | Standard |
| Rotary Embedding | No | `Qwen3RotaryEmbedding` | Standard |

## Switch Variables

The kernel module in `sys.modules["qwen3_pto_kernels"]` must expose:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_PTO_ROPE` | `bool` | `False` | Enable PyPTO fused Q/K RMSNorm + RoPE |

## Kernel API Contract

When `USE_PTO_ROPE=True`, the module must provide:

```python
def qk_rope_wrapper(
    q_proj_out: torch.Tensor,      # [B, S, num_q_heads * head_dim]
    k_proj_out: torch.Tensor,      # [B, S, num_kv_heads * head_dim]
    cos: torch.Tensor,             # [B, S, head_dim]
    sin: torch.Tensor,             # [B, S, head_dim]
    q_norm_weight: torch.Tensor,   # [head_dim]
    k_norm_weight: torch.Tensor,   # [head_dim]
    q_num_heads: int,              # 16
    kv_num_heads: int,             # 8
    head_dim: int,                 # 128
) -> tuple[torch.Tensor, torch.Tensor]:  # (query_states [B,N,S,D], key_states [B,Nkv,S,D])
    ...
```

## Usage

```python
import sys
from ops.pypto_tensor.qwen3_1_7b import qk_rope_wrapper

# 1. Inject kernel module
sys.modules["qwen3_pto_kernels"] = type("Kernels", (), {
    "USE_PTO_ROPE": True,
    "qk_rope_wrapper": staticmethod(qk_rope_wrapper)
})

# 2. Load model (auto_map resolves to this directory)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16,
    device_map={"": "npu:0"}, trust_remote_code=True
)
```

## HuggingFace `auto_map`

```json
{
  "auto_map": {
    "AutoConfig": "configuration_qwen3.Qwen3Config",
    "AutoModelForCausalLM": "modeling_qwen3.Qwen3ForCausalLM"
  }
}
```

## File Table

| File | Description |
|------|-------------|
| `configuration_qwen3.py` | `Qwen3Config` — 2048 hidden, 28 layers, 16 Q-heads, 8 KV-heads, head_dim=128 |
| `modeling_qwen3.py` | Full model graph with 6-line PTO dispatch in `Qwen3Attention.forward` |
| `config.json` | Model config with auto_map |

## Environment

| Component | Version |
|-----------|---------|
| torch | 2.9.0 |
| torch_npu | 2.9.0.post2 |
| transformers | 4.51.0 |
| CANN | 9.0.0 |
