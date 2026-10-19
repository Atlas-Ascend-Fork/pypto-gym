# fused_recurrent_kda

PyPTO kernel for state-maintaining recurrent Key-Delta-Attention (decode /
recurrent mode).

Token-by-token sequential recurrence maintaining fp32 state matrix S [H, V, K]:

```
S = S * exp(g)                        # gate decay
S += outer(beta * k, v - (k . S))    # KDA delta update
o = q . S                            # output
```

Supports varlen (cu_seqlens), inplace state (ssm_state_indices + NULL slot 0),
spec decoding (num_accepted_tokens + 2D ssm_state_indices), and in-kernel q/k
L2norm.


## 产品支持情况

- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## Function

```python
def fused_recurrent_kda_wrapper(
    q, k, v, g, beta=None, scale=None, initial_state=None,
    inplace_final_state=True, use_qk_l2norm_in_kernel=True,
    cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

### Inputs

| Argument | Shape | Dtype | Notes |
|----------|-------|-------|-------|
| `q` | `[B, T, H, D]` | fp16/bf16 | query |
| `k` | `[B, T, H, K]` | fp16/bf16 | key (K=D) |
| `v` | `[B, T, H, V]` | fp16/bf16 | value (V=D) |
| `g` | `[B, T, H, K]` | float32 | per-dim forget gate (log domain, <=0) |
| `beta` | `[B, T, H]` | float32 | delta update weight (0..1) |
| `scale` | scalar | float | None -> D**-0.5 |
| `initial_state` | `[S, H, V, K]` | float32 | initial state buffer |
| `cu_seqlens` | `[N+1]` | int32 | varlen segment boundaries |
| `ssm_state_indices` | `[N]` or `[N, T]` | int32 | inplace state slots |
| `num_accepted_tokens` | `[N]` | int32 | spec decode accepted token count |

### Outputs

| Return | Shape | Dtype |
|--------|-------|-------|
| `o` | `[B, T, H, D]` | fp16/bf16 |
| `final_state` | `[S, H, V, K]` | float32 |

### Supported dtypes / shapes

- dtypes: float16, bfloat16 (q/k/v); float32 (g/beta/initial_state)
- D = K = V = 128
- B = 1 (varlen packed)

## Testing

Tests under [`tests/ops/ling_3_0_flash/fused_recurrent_kda/`](../../../../../../tests/ops/ling_3_0_flash/fused_recurrent_kda/).
