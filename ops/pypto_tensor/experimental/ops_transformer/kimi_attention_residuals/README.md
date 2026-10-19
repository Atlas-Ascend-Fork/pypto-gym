# Block Attention Residuals 算子接口文档

本文档描述 Block Attention Residuals（BlockAttnRes）算子的接口说明，涵盖正向与反向两个算子。BlockAttnRes 是一种用于 Transformer 架构的深度残差聚合算子。

## 目录

1. [ai_infra_block_attn_res](#ai_infra_block_attn_res) - 正向算子
2. [ai_infra_block_attn_res_backward](#ai_infra_block_attn_res_backward) - 反向算子

---

# ai_infra_block_attn_res

## 功能描述

Block Attention Residuals 正向算子。该算子通过 softmax 注意力机制选择性聚合历史 block 表示和当前 block 部分和，替代传统的固定权重残差连接，可选启用 RMSNorm 归一化。通过 `rms_out_flag` / `alpha_out_flag` 控制是否将其返回给调用方。

## 接口定义

```python
def ai_infra_block_attn_res(
    blocks: List[Tensor],                         # BF16/FP16, 每个 shape: (B, T, D)
    proj_weight: Tensor,                          # BF16/FP16, shape: (D,) 或 (1, D)
    partial_block: Optional[Tensor] = None,       # BF16/FP16, shape: (B, T, D)
    scale: float = 1.0,                           # softmax 缩放因子
    rmsnorm_eps: float = 1e-6,                    # RMSNorm epsilon, 防止除零
    rmsnorm_gamma: Optional[Tensor] = None,       # BF16/FP16, shape: (D,) 或 (1, D)
    enable_rmsnorm: bool = True,                  # 是否启用 RMSNorm
    rms_out_flag: bool = False,                   # 是否返回 rms_cache
    alpha_out_flag: bool = False,                 # 是否返回 alpha_cache
) -> Union[
    Tensor,                                       # 两 flag 均为 False
    Tuple[Tensor, Optional[Tensor]],              # 仅 rms_out_flag=True: (output, rms_cache)
    Tuple[Tensor, Optional[Tensor]],                        # 仅 alpha_out_flag=True: (output, alpha_cache)
    Tuple[Tensor, Optional[Tensor], Optional[Tensor]],      # 均为 True: (output, rms_cache, alpha_cache)
]
```

## 参数说明

### 输入参数

| 参数名 | 类型 | 形状 | 数据类型 | 必选/可选 | 格式 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| blocks | List[Tensor] | 每个Tensor shape为[B, T, D] | BF16/FP16 | 必选 | ND | 已完成的 block 表示列表。列表长度 N 满足 1 ≤ N ≤ 127，典型值 65（32 layers * 2 + 1）。 |
| proj_weight | Tensor | [D,] | 与blocks一致 | 必选 | ND | 伪查询投影层权重 $w_l$。 |
| partial_block | Tensor | [B, T, D] | 与blocks一致 | 可选 | ND | 当前 block 的部分和（块内部分和 b_n^i）。如不提供，则仅对blocks进行聚合，L=N，如提供，则 L=N+1。 |
| scale | 标量 | - | float | 可选 | - | softmax 缩放因子，默认值为1.0，用于控制 softmax 陡峭程度。 |
| rmsnorm_eps | 标量 | - | float | 可选 | - | RMSNorm epsilon 参数，默认值为1e-6，防止除零。|
| rmsnorm_gamma | Tensor | [D,] | 与blocks一致 | 可选 | ND | RMSNorm 缩放参数 γ，enable_rmsnorm=True 时必填。 |
| enable_rmsnorm | 标量 | - | bool | 可选 | - | 是否启用 RMSNorm 归一化，默认为True。 |
| rms_out_flag | 标量 | - | bool | 可选 | - | 是否返回 rms_cache，默认为False；enable_rmsnorm=False 时该参数必须为 False。 |
| alpha_out_flag | 标量 | - | bool | 可选 | - | 是否返回 alpha_cache，默认为False。 |

### 输出参数

| 参数名 | 类型 | 形状 | 数据类型 | 必选/可选 | 格式 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| output | Tensor | [B, T, D] | 与输入blocks一致 | 必选 | ND | 聚合后的隐藏状态。 |
| rms_cache | Tensor | [B, T, L, 1] | float | 可选 | ND | 计算公式中 RMSNorm 归一化的分母缓存；enable_rmsnorm=False 时为 None。 |
| alpha_cache | Tensor | [B, T, L] | float | 可选 | ND | softmax 注意力权重。 |

返回值形式随 `rms_out_flag` / `alpha_out_flag` 组合变化：

| 组合 | 返回值 |
| --- | --- |
| 两 flag 均为 False | `output` |
| 仅 rms_out_flag=True | `(output, rms_cache)` |
| 仅 alpha_out_flag=True | `(output, alpha_cache)` |
| 两 flag 均为 True | `(output, rms_cache, alpha_cache)` |


## 算法原理

### Step 1: 堆叠操作

将 N 个历史 block 表示和当前 block 部分和（可选）沿新维度堆叠：

- 有 partial_block：$\boldsymbol{V} = \text{stack}([b_0, b_1, \ldots, b_{N-1}, b_N^i], \text{dim}=2) \in \mathbb{R}^{B \times T \times (N+1) \times D}$，$L = N+1$
- 无 partial_block：$\boldsymbol{V} = \text{stack}([b_0, b_1, \ldots, b_{N-1}], \text{dim}=2) \in \mathbb{R}^{B \times T \times N \times D}$，$L = N$

### Step 2: RMSNorm 归一化

当 `enable_rmsnorm=True` 时，对 V 进行 RMS 归一化得到 K：

$$
\boldsymbol{K} = \frac{\boldsymbol{V}}{\sqrt{\frac{1}{D}\sum_{i=1}^{D} V_i^2 + \epsilon}} \cdot \boldsymbol{\gamma}
$$

当 `enable_rmsnorm=False` 时，$\boldsymbol{K} = \boldsymbol{V}$

### Step 3: 注意力 logits 计算

伪查询向量与 K 做点积并乘以 scale：

$$
\text{logits}_{b,t,l} = K_{b,t,l,d} \cdot w_d^{T} \cdot \text{scale}
$$

### Step 4: Softmax 归一化

沿深度维度 L 计算注意力权重：

$$
\alpha_{b,t,l} = \frac{\exp(\text{logits}_{b,t,l})}{\sum_{l'=0}^{L-1} \exp(\text{logits}_{b,t,l'})}
$$

### Step 5: 加权聚合输出

使用注意力权重对 V 进行加权求和：

$$
h_{b,t,d} = \sum_{l=0}^{L-1} \alpha_{b,t,l} \cdot V_{b,t,l,d}
$$

其中 $L = N$（无 partial_block）或 $L = N+1$（有 partial_block）。

## 约束条件

- `blocks` 不能为空，长度 N 满足 1 ≤ N ≤ 127。
- `blocks` 中每个 tensor 的 shape 必须相同，均为 [B, T, D]，且 dtype 一致。
- `partial_block` 若提供，shape 必须为 [B, T, D]，dtype 与 blocks 一致。
- `proj_weight` 的 shape 必须为 [D,]，dtype 与 blocks 一致。
- `enable_rmsnorm=True` 时，`rmsnorm_gamma` 必填；若提供 `rmsnorm_gamma`，其 shape 必须为 [D,]。
- `rms_out_flag=True` 要求 `enable_rmsnorm=True`。

## 支持规格

- 数据类型：BF16 / FP16（输入/输出），FP32（内部计算、cache）
- 芯片平台：A2 / A3

## 使用示例

### 示例 1：仅计算输出（推理场景）

```python
import torch
import pypto

# 设置设备
torch.npu.set_device(0)

B, T, D = 4, 128, 2048
N = 12

# 构造输入
blocks = [torch.randn(B, T, D, dtype=torch.bfloat16, device="npu:0") for _ in range(N)]
proj_weight = torch.randn(D, dtype=torch.bfloat16, device="npu:0")
rmsnorm_gamma = torch.ones(D, dtype=torch.bfloat16, device="npu:0")

# 调用算子
output = ai_infra_block_attn_res(
    blocks, proj_weight,
    rmsnorm_gamma=rmsnorm_gamma,
    enable_rmsnorm=True,
)

print(f"输出形状: {output.shape}")  # (4, 128, 2048)
```

### 示例 2：输出 cache 供反向复用（训练场景）

```python
# 训练场景：同时输出 rms_cache 与 alpha_cache，供反向算子复用
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

## 功能描述

Block Attention Residuals 反向算子。该算子用于大语言模型训练的反向传播，计算各输入参数的梯度。

## 接口定义

```python
def ai_infra_block_attn_res_backward(
    grad_h: Tensor,                               # BF16/FP16, shape: (B, T, D)
    blocks: List[Tensor],                         # BF16/FP16, 每个 shape: (B, T, D)
    proj_weight: Tensor,                          # BF16/FP16, shape: (D,)
    alpha_cache: Tensor,                          # FP32, shape: (B, T, L)
    partial_block: Optional[Tensor] = None,       # BF16/FP16, shape: (B, T, D)
    rmsnorm_gamma: Optional[Tensor] = None,       # BF16/FP16, shape: (D,)
    rms_cache: Optional[Tensor] = None,           # FP32, shape: (B, T, L, 1)
    scale: float = 1.0,                           # softmax 缩放因子
    enable_rmsnorm: bool = True,                  # 是否启用 RMSNorm
) -> Union[
    Tuple[List[Tensor], Tensor, Tensor],           # enable_rmsnorm=False
    Tuple[List[Tensor], Tensor, Tensor, Optional[Tensor]],   # enable_rmsnorm=True
]
```

## 参数说明

### 输入参数

| 参数名 | 类型 | 形状 | 数据类型 | 必选/可选 | 格式 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| grad_h | Tensor | [B, T, D] | 与blocks一致 | 必选 | ND | 上游传递的输出梯度 $\partial \mathcal{L}/\partial h$。 |
| blocks | List[Tensor] | 每个Tensor Shape为[B, T, D] | BF16/FP16 | 必选 | ND | 正向使用的已完成的 block 表示列表。 |
| proj_weight | Tensor | [D,] | 与blocks一致 | 必选 | ND | 伪查询投影权重。 |
| alpha_cache | Tensor | [B, T, L] | FP32 | 可选 | ND | 正向缓存的 softmax 权重。 |
| partial_block | Tensor | [B, T, D] | 与blocks一致 | 可选 | ND | 正向使用的当前 block 部分和。 |
| rmsnorm_gamma | Tensor | [D,] | 与blocks一致 | 可选 | ND | RMSNorm γ；`enable_rmsnorm=True` 时必填。 |
| rms_cache | Tensor | [B, T, L, 1] | FP32 | 可选 | ND | 正向缓存的 RMSNorm 分母；`enable_rmsnorm=True` 时必填。 |
| scale | 标量 | - | float | 可选 | - | softmax 缩放因子，需与正向一致，默认值为1.0。 |
| enable_rmsnorm | 标量 | - | bool | 可选 | - | 是否启用 RMSNorm，默认为True。 |

### 输出参数

| 参数名 | 类型 | 形状 | 数据类型 | 必选/可选 | 格式 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| grad_blocks | List[Tensor] | 每个Tensor Shape为[B, T, D] | 与输入blocks一致 | 必选 | ND | blocks 的梯度。 |
| grad_partial_block | Tensor | [B, T, D] | 与输入blocks一致 | 必选 | ND | partial_block 的梯度；若正向未传 partial_block 则为 None。 |
| grad_proj_weight | Tensor | [D,] | 与输入blocks一致 | 必选 | ND | 伪查询投影权重的梯度。 |
| grad_rmsnorm_gamma | Tensor | [D,] | 与输入blocks一致 | 可选 | ND | RMSNorm γ 的梯度；仅 `enable_rmsnorm=True` 时返回。 |

返回值形式随 `enable_rmsnorm` 变化：

| enable_rmsnorm | 返回值 |
| --- | --- |
| False | `(grad_blocks, grad_partial_block, grad_proj_weight)` |
| True | `(grad_blocks, grad_partial_block, grad_proj_weight, grad_rmsnorm_gamma)` |


## 算法原理

反向算子通过链式法则反向传递梯度，分为以下阶段：

### Step 1 反向（Weighted Summation Backward）

计算 $\alpha$ 的梯度（输入 grad_h 与 V）与 V 的聚合梯度分量：

$$
\frac{\partial \mathcal{L}}{\partial \alpha_{b,t,1,l}} = \frac{\partial \mathcal{L}}{\partial h_{b,t,1,d}} \cdot V_{b,t,l,d}^{T}
$$

$$
\left.\frac{\partial \mathcal{L}}{\partial V_{b,t,l,d}}\right|_{\text{agg}} = \alpha_{b,t,1,l}^{T} \cdot \frac{\partial \mathcal{L}}{\partial h_{b,t,1,d}}
$$

### Step 2 反向（Attention Softmax Backward）

$$
\frac{\partial \mathcal{L}}{\partial \ell_{b,t,l}} = \alpha_{b,t,l} \left( \frac{\partial \mathcal{L}}{\partial \alpha_{b,t,l}} - \sum_{l'} \frac{\partial \mathcal{L}}{\partial \alpha_{b,t,l'}} \cdot \alpha_{b,t,l'} \right)
$$

### Step 3 反向（Attention Scores Backward）

对 K 的梯度：

$$
\frac{\partial \mathcal{L}}{\partial K_{b,t,l,d}} = \frac{\partial \mathcal{L}}{\partial \ell_{b,t,1,l}}^{T} \cdot \text{scale} \cdot w_d
$$

对 proj_weight 的梯度（按 (B, T, L) 规约求和）：

$$
\frac{\partial \mathcal{L}}{\partial w_d} = \text{scale} \cdot \sum_{b,t,l} \frac{\partial \mathcal{L}}{\partial \ell_{b,t,l}} \cdot K_{b,t,l,d}
$$

### Step 4 反向（RMSNorm Backward）

当 `enable_rmsnorm=True` 时，基于正向缓存的 `rms_cache` 重算中间量：

$$
c_{b,t,l} = \sum_{d'} \gamma_{d'} \cdot \text{grad\_K}_{b,t,l,d'} \cdot V_{b,t,l,d'}
$$

$$
\left.\frac{\partial \mathcal{L}}{\partial V_{b,t,l,d}}\right|_{\text{rms}} = \frac{\gamma_d \cdot \text{grad\_K}_{b,t,l,d}}{\text{rms}_{b,t,l}} - \frac{V_{b,t,l,d} \cdot c_{b,t,l}}{D \cdot \text{rms}_{b,t,l}^3}
$$

$$
\frac{\partial \mathcal{L}}{\partial \gamma_d} = \sum_{b,t,l} \text{grad\_K}_{b,t,l,d} \cdot \frac{V_{b,t,l,d}}{\text{rms}_{b,t,l}}
$$

当 `enable_rmsnorm=False` 时，$\left.\partial \mathcal{L}/\partial V\right|_{\text{rms}} = \text{grad\_K}$。

### Step 5 反向（V 总梯度与 Stack 反向）

合并两路 V 梯度：

$$
\frac{\partial \mathcal{L}}{\partial V} = \left.\frac{\partial \mathcal{L}}{\partial V}\right|_{\text{agg}} + \left.\frac{\partial \mathcal{L}}{\partial V}\right|_{\text{rms}}
$$

Stack 反向：按第二维切分得到 blocks 的梯度：

$$
\frac{\partial \mathcal{L}}{\partial b_i} = \frac{\partial \mathcal{L}}{\partial V_{:,:,i,:}}, \quad i = 0, 1, \ldots, N-1
$$

若存在 partial_block，则：

$$
\frac{\partial \mathcal{L}}{\partial b_N^i} = \frac{\partial \mathcal{L}}{\partial V_{:,:,N,:}}
$$


## 约束条件

- `blocks` 不能为空，且每个 tensor 的 shape 必须相同，均为 [B, T, D]。
- `grad_h` 的 shape 必须为 [B, T, D]。
- `partial_block` 如果存在，则 shape 必须为 [B, T, D]。
- `proj_weight` 的 shape 必须为 [D,]。
- `alpha_cache` 必须传入，shape 必须为 [B, T, L]。
- `enable_rmsnorm=True` 时，`rmsnorm_gamma` 与 `rms_cache` 必须传入；`rmsnorm_gamma` 的 shape 为 [D,]，`rms_cache` 的 shape 为 [B, T, L, 1]。

## 支持规格

- 数据类型：BF16 / FP16（输入/输出），FP32（内部计算、cache）
- 芯片平台：A2 / A3

## 使用示例

```python
import torch
import pypto

# 设置设备
torch.npu.set_device(0)

B, T, D = 4, 128, 2048
N = 12

# 构造输入
blocks = [torch.randn(B, T, D, dtype=torch.bfloat16, device="npu:0",
                       requires_grad=True) for _ in range(N)]
partial_block = torch.randn(B, T, D, dtype=torch.bfloat16, device="npu:0",
                             requires_grad=True)
proj_weight = torch.randn(D, dtype=torch.bfloat16, device="npu:0",
                           requires_grad=True)
rmsnorm_gamma = torch.ones(D, dtype=torch.bfloat16, device="npu:0",
                            requires_grad=True)

# 正向：输出 output + rms_cache + alpha_cache
output, rms_cache, alpha_cache = ai_infra_block_attn_res(
    blocks, proj_weight,
    partial_block=partial_block,
    rmsnorm_gamma=rmsnorm_gamma,
    enable_rmsnorm=True,
    rms_out_flag=True,
    alpha_out_flag=True,
)

# 模拟上游梯度
grad_h = torch.ones_like(output)

# 反向
grad_blocks, grad_partial_block, grad_proj_weight, grad_rmsnorm_gamma = \
    ai_infra_block_attn_res_backward(
        grad_h, blocks, proj_weight, alpha_cache,
        partial_block=partial_block,
        rmsnorm_gamma=rmsnorm_gamma,
        rms_cache=rms_cache,
        scale=1.0,
        enable_rmsnorm=True,
    )

print(f"grad_blocks[0] 形状: {grad_blocks[0].shape}")          # (B, T, D)
print(f"grad_partial_block 形状: {grad_partial_block.shape}")  # (B, T, D)
print(f"grad_proj_weight 形状: {grad_proj_weight.shape}")      # (D,)
print(f"grad_rmsnorm_gamma 形状: {grad_rmsnorm_gamma.shape}")  # (D,)
```
