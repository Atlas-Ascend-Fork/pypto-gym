# Block Attention Residuals Operator Interface Documentation

This document describes the interface for the Block Attention Residuals (BlockAttnRes) operator, covering both the forward and backward operators. BlockAttnRes is a deep residual aggregation operator for Transformer architectures.

## Table of Contents

1. [ai_infra_block_attn_res](#ai_infra_block_attn_res) - Forward operator
2. [ai_infra_block_attn_res_backward](#ai_infra_block_attn_res_backward) - Backward operator

---

# ai_infra_block_attn_res

## Description

Block Attention Residuals forward operator. This operator selectively aggregates historical block representations and the current block partial sum through the softmax attention mechanism, replacing the traditional fixed-weight residual connection. It optionally enables RMSNorm normalization. Use `rms_out_flag` / `alpha_out_flag` to control whether to return them to the caller.

## Interface Definition

```python
def ai_infra_block_attn_res(
    blocks: List[Tensor],                         # BF16/FP16, each shape: (B, T, D)
    proj_weight: Tensor,                          # BF16/FP16, shape: (D,) or (1, D)
    partial_block: Optional[Tensor] = None,       # BF16/FP16, shape: (B, T, D)
    scale: float = 1.0,                           # softmax scaling factor
    rmsnorm_eps: float = 1e-6,                    # RMSNorm epsilon, prevents division by zero
    rmsnorm_gamma: Optional[Tensor] = None,       # BF16/FP16, shape: (D,) or (1, D)
    enable_rmsnorm: bool = True,                  # whether to enable RMSNorm
    rms_out_flag: bool = False,                   # whether to return rms_cache
    alpha_out_flag: bool = False,                 # whether to return alpha_cache
) -> Union[
    Tensor,                                       # both flags are False
    Tuple[Tensor, Optional[Tensor]],              # only rms_out_flag=True: (output, rms_cache)
    Tuple[Tensor, Optional[Tensor]],                        # only alpha_out_flag=True: (output, alpha_cache)
    Tuple[Tensor, Optional[Tensor], Optional[Tensor]],      # both True: (output, rms_cache, alpha_cache)
]
```

## Parameter Description

### Input Parameters

| Parameter | Type | Shape | Data Type | Required/Optional | Format | Description |
| --- | --- | --- | --- | --- | --- | --- |
| blocks | List[Tensor] | Each Tensor shape [B, T, D] | BF16/FP16 | Required | ND | List of completed block representations. List length N satisfies 1 ≤ N ≤ 127, typical value 65 (32 layers * 2 + 1). |
| proj_weight | Tensor | [D,] | Same as blocks | Required | ND | Pseudo-query projection layer weight $w_l$. |
| partial_block | Tensor | [B, T, D] | Same as blocks | Optional | ND | Partial sum of the current block (intra-block partial sum b_n^i). If not provided, the operator aggregates only blocks, L=N. If provided, L=N+1. |
| scale | Scalar | - | float | Optional | - | Softmax scaling factor, default value 1.0, controls the softmax steepness. |
| rmsnorm_eps | Scalar | - | float | Optional | - | RMSNorm epsilon parameter, default value 1e-6, prevents division by zero. |
| rmsnorm_gamma | Tensor | [D,] | Same as blocks | Optional | ND | RMSNorm scaling parameter gamma, required when enable_rmsnorm=True. |
| enable_rmsnorm | Scalar | - | bool | Optional | - | Whether to enable RMSNorm normalization, defaults to True. |
| rms_out_flag | Scalar | - | bool | Optional | - | Whether to return rms_cache, defaults to False; this parameter must be False when enable_rmsnorm=False. |
| alpha_out_flag | Scalar | - | bool | Optional | - | Whether to return alpha_cache, defaults to False. |

### Output Parameters

| Parameter | Type | Shape | Data Type | Required/Optional | Format | Description |
| --- | --- | --- | --- | --- | --- | --- |
| output | Tensor | [B, T, D] | Same as input blocks | Required | ND | Aggregated hidden state. |
| rms_cache | Tensor | [B, T, L, 1] | float | Optional | ND | Denominator cache of RMSNorm normalization in the formula; None when enable_rmsnorm=False. |
| alpha_cache | Tensor | [B, T, L] | float | Optional | ND | Softmax attention weights. |

The return value varies with the `rms_out_flag` / `alpha_out_flag` combination:

| Combination | Return Value |
| --- | --- |
| Both flags are False | `output` |
| Only rms_out_flag=True | `(output, rms_cache)` |
| Only alpha_out_flag=True | `(output, alpha_cache)` |
| Both flags are True | `(output, rms_cache, alpha_cache)` |

## Algorithm Principle

### Step 1: Stack Operation

Stack N historical block representations and the current block partial sum (optional) along a new dimension:

- With partial_block: $\boldsymbol{V} = \text{stack}([b_0, b_1, \ldots, b_{N-1}, b_N^i], \text{dim}=2) \in \mathbb{R}^{B \times T \times (N+1) \times D}$, $L = N+1$
- Without partial_block: $\boldsymbol{V} = \text{stack}([b_0, b_1, \ldots, b_{N-1}], \text{dim}=2) \in \mathbb{R}^{B \times T \times N \times D}$, $L = N$

### Step 2: RMSNorm Normalization

When `enable_rmsnorm=True`, apply RMS normalization to V to obtain K:

$$
\boldsymbol{K} = \frac{\boldsymbol{V}}{\sqrt{\frac{1}{D}\sum_{i=1}^{D} V_i^2 + \epsilon}} \cdot \boldsymbol{\gamma}
$$

When `enable_rmsnorm=False`, $\boldsymbol{K} = \boldsymbol{V}$

### Step 3: Attention Logits Calculation

Compute the dot product between the pseudo-query vector and K, then multiply by scale:

$$
\text{logits}_{b,t,l} = K_{b,t,l,d} \cdot w_d^{T} \cdot \text{scale}
$$

### Step 4: Softmax Normalization

Compute attention weights along the depth dimension L:

$$
\alpha_{b,t,l} = \frac{\exp(\text{logits}_{b,t,l})}{\sum_{l'=0}^{L-1} \exp(\text{logits}_{b,t,l'})}
$$

### Step 5: Weighted Aggregation Output

Use attention weights to compute a weighted sum of V:

$$
h_{b,t,d} = \sum_{l=0}^{L-1} \alpha_{b,t,l} \cdot V_{b,t,l,d}
$$

Where $L = N$ (without partial_block) or $L = N+1$ (with partial_block).

## Constraints

- `blocks` cannot be empty; length N satisfies 1 ≤ N ≤ 127.
- Each tensor in `blocks` must have the same shape [B, T, D] and the same dtype.
- If `partial_block` is provided, its shape must be [B, T, D] and dtype must match blocks.
- `proj_weight` shape must be [D,] and dtype must match blocks.
- When `enable_rmsnorm=True`, `rmsnorm_gamma` is required; if `rmsnorm_gamma` is provided, its shape must be [D,].
- `rms_out_flag=True` requires `enable_rmsnorm=True`.

## Supported Specifications

- Data types: BF16 / FP16 (input/output), FP32 (internal computation, cache)
- Chip platforms: A2 / A3

## Usage Samples

### Sample 1: Output Only (Inference Scenario)

```python
import torch
import pypto

# Set device
torch.npu.set_device(0)

B, T, D = 4, 128, 2048
N = 12

# Construct input
blocks = [torch.randn(B, T, D, dtype=torch.bfloat16, device="npu:0") for _ in range(N)]
proj_weight = torch.randn(D, dtype=torch.bfloat16, device="npu:0")
rmsnorm_gamma = torch.ones(D, dtype=torch.bfloat16, device="npu:0")

# Call operator
output = ai_infra_block_attn_res(
    blocks, proj_weight,
    rmsnorm_gamma=rmsnorm_gamma,
    enable_rmsnorm=True,
)

print(f"Output shape: {output.shape}")  # (4, 128, 2048)
```

### Sample 2: Output Cache for Backward Reuse (Training Scenario)

```python
# Training scenario: output both rms_cache and alpha_cache for backward operator reuse
output, rms_cache, alpha_cache = ai_infra_block_attn_res(
    blocks, proj_weight,
    rmsnorm_gamma=rmsnorm_gamma,
    enable_rmsnorm=True,
    rms_out_flag=True,
    alpha_out_flag=True,
)

print(f"output: {output.shape}")           # (B, T, D)
print(f"rms_cache: {rms_cache.shape}")     # (B, T, L, 1)
print(f"alpha_cache: {alpha_cache.shape}") # (B, T, L)
```

---

# ai_infra_block_attn_res_backward

## Description

Block Attention Residuals backward operator. This operator performs backpropagation for large language model training, computing gradients for each input parameter.

## Interface Definition

```python
def ai_infra_block_attn_res_backward(
    grad_h: Tensor,                               # BF16/FP16, shape: (B, T, D)
    blocks: List[Tensor],                         # BF16/FP16, each shape: (B, T, D)
    proj_weight: Tensor,                          # BF16/FP16, shape: (D,)
    alpha_cache: Tensor,                          # FP32, shape: (B, T, L)
    partial_block: Optional[Tensor] = None,       # BF16/FP16, shape: (B, T, D)
    rmsnorm_gamma: Optional[Tensor] = None,       # BF16/FP16, shape: (D,)
    rms_cache: Optional[Tensor] = None,           # FP32, shape: (B, T, L, 1)
    scale: float = 1.0,                           # softmax scaling factor
    enable_rmsnorm: bool = True,                  # whether to enable RMSNorm
) -> Union[
    Tuple[List[Tensor], Tensor, Tensor],           # enable_rmsnorm=False
    Tuple[List[Tensor], Tensor, Tensor, Optional[Tensor]],   # enable_rmsnorm=True
]
```

## Parameter Description

### Input Parameters

| Parameter | Type | Shape | Data Type | Required/Optional | Format | Description |
| --- | --- | --- | --- | --- | --- | --- |
| grad_h | Tensor | [B, T, D] | Same as blocks | Required | ND | Upstream output gradient $\partial \mathcal{L}/\partial h$. |
| blocks | List[Tensor] | Each Tensor Shape [B, T, D] | BF16/FP16 | Required | ND | List of completed block representations used in forward. |
| proj_weight | Tensor | [D,] | Same as blocks | Required | ND | Pseudo-query projection weight. |
| alpha_cache | Tensor | [B, T, L] | FP32 | Optional | ND | Softmax weights cached in forward. |
| partial_block | Tensor | [B, T, D] | Same as blocks | Optional | ND | Current block partial sum used in forward. |
| rmsnorm_gamma | Tensor | [D,] | Same as blocks | Optional | ND | RMSNorm gamma; required when `enable_rmsnorm=True`. |
| rms_cache | Tensor | [B, T, L, 1] | FP32 | Optional | ND | RMSNorm denominator cached in forward; required when `enable_rmsnorm=True`. |
| scale | Scalar | - | float | Optional | - | Softmax scaling factor, must match forward, default value 1.0. |
| enable_rmsnorm | Scalar | - | bool | Optional | - | Whether to enable RMSNorm, defaults to True. |

### Output Parameters

| Parameter | Type | Shape | Data Type | Required/Optional | Format | Description |
| --- | --- | --- | --- | --- | --- | --- |
| grad_blocks | List[Tensor] | Each Tensor Shape [B, T, D] | Same as input blocks | Required | ND | Gradients of blocks. |
| grad_partial_block | Tensor | [B, T, D] | Same as input blocks | Required | ND | Gradient of partial_block; None if partial_block was not passed in forward. |
| grad_proj_weight | Tensor | [D,] | Same as input blocks | Required | ND | Gradient of pseudo-query projection weight. |
| grad_rmsnorm_gamma | Tensor | [D,] | Same as input blocks | Optional | ND | Gradient of RMSNorm gamma; returned only when `enable_rmsnorm=True`. |

The return value varies with `enable_rmsnorm`:

| enable_rmsnorm | Return Value |
| --- | --- |
| False | `(grad_blocks, grad_partial_block, grad_proj_weight)` |
| True | `(grad_blocks, grad_partial_block, grad_proj_weight, grad_rmsnorm_gamma)` |

## Algorithm Principle

The backward operator propagates gradients backward through the chain rule, divided into the following stages:

### Step 1 Backward (Weighted Summation Backward)

Compute the gradient of alpha (input grad_h and V) and the aggregation gradient component of V:

$$
\frac{\partial \mathcal{L}}{\partial \alpha_{b,t,1,l}} = \frac{\partial \mathcal{L}}{\partial h_{b,t,1,d}} \cdot V_{b,t,l,d}^{T}
$$

$$
\left.\frac{\partial \mathcal{L}}{\partial V_{b,t,l,d}}\right|_{\text{agg}} = \alpha_{b,t,1,l}^{T} \cdot \frac{\partial \mathcal{L}}{\partial h_{b,t,1,d}}
$$

### Step 2 Backward (Attention Softmax Backward)

$$
\frac{\partial \mathcal{L}}{\partial \ell_{b,t,l}} = \alpha_{b,t,l} \left( \frac{\partial \mathcal{L}}{\partial \alpha_{b,t,l}} - \sum_{l'} \frac{\partial \mathcal{L}}{\partial \alpha_{b,t,l'}} \cdot \alpha_{b,t,l'} \right)
$$

### Step 3 Backward (Attention Scores Backward)

Gradient with respect to K:

$$
\frac{\partial \mathcal{L}}{\partial K_{b,t,l,d}} = \frac{\partial \mathcal{L}}{\partial \ell_{b,t,1,l}}^{T} \cdot \text{scale} \cdot w_d
$$

Gradient with respect to proj_weight (sum reduce over (B, T, L)):

$$
\frac{\partial \mathcal{L}}{\partial w_d} = \text{scale} \cdot \sum_{b,t,l} \frac{\partial \mathcal{L}}{\partial \ell_{b,t,l}} \cdot K_{b,t,l,d}
$$

### Step 4 Backward (RMSNorm Backward)

When `enable_rmsnorm=True`, recompute intermediate quantities based on the forward-cached `rms_cache`:

$$
c_{b,t,l} = \sum_{d'} \gamma_{d'} \cdot \text{grad\_K}_{b,t,l,d'} \cdot V_{b,t,l,d'}
$$

$$
\left.\frac{\partial \mathcal{L}}{\partial V_{b,t,l,d}}\right|_{\text{rms}} = \frac{\gamma_d \cdot \text{grad\_K}_{b,t,l,d}}{\text{rms}_{b,t,l}} - \frac{V_{b,t,l,d} \cdot c_{b,t,l}}{D \cdot \text{rms}_{b,t,l}^3}
$$

$$
\frac{\partial \mathcal{L}}{\partial \gamma_d} = \sum_{b,t,l} \text{grad\_K}_{b,t,l,d} \cdot \frac{V_{b,t,l,d}}{\text{rms}_{b,t,l}}
$$

When `enable_rmsnorm=False`, $\left.\partial \mathcal{L}/\partial V\right|_{\text{rms}} = \text{grad\_K}$.

### Step 5 Backward (V Total Gradient and Stack Backward)

Merge the two V gradient paths:

$$
\frac{\partial \mathcal{L}}{\partial V} = \left.\frac{\partial \mathcal{L}}{\partial V}\right|_{\text{agg}} + \left.\frac{\partial \mathcal{L}}{\partial V}\right|_{\text{rms}}
$$

Stack backward: split along the second dimension to obtain block gradients:

$$
\frac{\partial \mathcal{L}}{\partial b_i} = \frac{\partial \mathcal{L}}{\partial V_{:,:,i,:}}, \quad i = 0, 1, \ldots, N-1
$$

If partial_block exists, then:

$$
\frac{\partial \mathcal{L}}{\partial b_N^i} = \frac{\partial \mathcal{L}}{\partial V_{:,:,N,:}}
$$

## Constraints

- `blocks` cannot be empty, and each tensor must have the same shape [B, T, D].
- `grad_h` shape must be [B, T, D].
- `partial_block`, if present, must have shape [B, T, D].
- `proj_weight` shape must be [D,].
- `alpha_cache` is required, shape must be [B, T, L].
- When `enable_rmsnorm=True`, both `rmsnorm_gamma` and `rms_cache` are required; `rmsnorm_gamma` shape is [D,], `rms_cache` shape is [B, T, L, 1].

## Supported Specifications

- Data types: BF16 / FP16 (input/output), FP32 (internal computation, cache)
- Chip platforms: A2 / A3

## Usage Samples

```python
import torch
import pypto

# Set device
torch.npu.set_device(0)

B, T, D = 4, 128, 2048
N = 12

# Construct input
blocks = [torch.randn(B, T, D, dtype=torch.bfloat16, device="npu:0",
                       requires_grad=True) for _ in range(N)]
partial_block = torch.randn(B, T, D, dtype=torch.bfloat16, device="npu:0",
                             requires_grad=True)
proj_weight = torch.randn(D, dtype=torch.bfloat16, device="npu:0",
                           requires_grad=True)
rmsnorm_gamma = torch.ones(D, dtype=torch.bfloat16, device="npu:0",
                            requires_grad=True)

# Forward: output output + rms_cache + alpha_cache
output, rms_cache, alpha_cache = ai_infra_block_attn_res(
    blocks, proj_weight,
    partial_block=partial_block,
    rmsnorm_gamma=rmsnorm_gamma,
    enable_rmsnorm=True,
    rms_out_flag=True,
    alpha_out_flag=True,
)

# Simulate upstream gradient
grad_h = torch.ones_like(output)

# Backward
grad_blocks, grad_partial_block, grad_proj_weight, grad_rmsnorm_gamma = \
    ai_infra_block_attn_res_backward(
        grad_h, blocks, proj_weight, alpha_cache,
        partial_block=partial_block,
        rmsnorm_gamma=rmsnorm_gamma,
        rms_cache=rms_cache,
        scale=1.0,
        enable_rmsnorm=True,
    )

print(f"grad_blocks[0] shape: {grad_blocks[0].shape}")          # (B, T, D)
print(f"grad_partial_block shape: {grad_partial_block.shape}")  # (B, T, D)
print(f"grad_proj_weight shape: {grad_proj_weight.shape}")      # (D,)
print(f"grad_rmsnorm_gamma shape: {grad_rmsnorm_gamma.shape}")  # (D,)
```
