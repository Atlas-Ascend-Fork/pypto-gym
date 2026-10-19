# Engram 算子接口文档

本文档包含 Engram 门控记忆检索（Gated Memory）算子的接口说明，涵盖正向和反向两个算子。基于 PyPTO 框架实现，BF16 计算，运行于 Ascend NPU。

Engram 是一种多头门控记忆机制：对每个 token 做 Key/Value 投影，对 Key 与 Query 分别 RMSNorm 后点积得到 score，经 sign-sqrt 门控后输出门控后的 value。

## 目录

1. [engram_forward_wrapper](#engram_forward_wrapper) - 多头门控记忆前向算子
2. [engram_backward_wrapper](#engram_backward_wrapper) - 多头门控记忆反向算子

---

# engram_forward_wrapper

多头门控记忆检索（Engram）正向算子。对权重/输入进行 Key/Value 投影、RMSNorm、sign-sqrt 门控，输出门控后的 value 及若干中间结果（供反向使用）。

适用于 Transformer 中的门控记忆层场景，支持任意头数 M，BF16 输入、FP32 中间计算。

## 功能描述

正向算子逐头计算 Key/Value 投影与门控输出。输入/输出为 BF16，matmul 走 FP32 累加，score/gate 等中间量以 FP32 输出。`batch_size*seq_len` 在 kernel 内部合轴（BL）以优化访存。

## 接口定义

```python
from ops.pypto_tensor.experimental.ops_transformer.engram.engram_forward_impl import engram_forward_wrapper

def engram_forward_wrapper(
    hidden_states: Tensor,      # BF16, shape: (B, L, M, Hh)
    embeddings: Tensor,         # BF16, shape: (B, L, De)
    key_proj_weights: Tensor,   # BF16, shape: (M, De, Hh)
    value_proj_weights: Tensor, # BF16, shape: (De, Hh)
    key_gamma: Tensor,          # BF16, shape: (M, Hh)
    query_gamma: Tensor,        # BF16, shape: (M, Hh)
    clamp_value: float = 1e-6,  # sign-sqrt 门控的 clamp 下界
    eps: float = 1e-6,          # RMSNorm 的 ε
    return_cache: bool = True,    # 推理场景置 False: 不分配/不写出 4 个反向 cache, 后 4 项返回 None
) -> Tuple[Tensor, ...]:
    # (value_out, score_back, key_back, value_back, gate_back)
    # return_cache=False 时后 4 项为 None (返回签名一致, 调用方无需分支解包)
```

## 参数说明

### 输入参数

| 参数名 | 类型 | 形状 | 说明 |
|-------|------|------|------|
| hidden_states | Tensor(BF16) | (B, L, M, Hh) | Query 输入，B=batch，L=seq_len，M=头数，Hh=隐藏维度 |
| embeddings | Tensor(BF16) | (B, L, De) | Key/Value 线性层共同输入，De=embedding 维度 |
| key_proj_weights | Tensor(BF16) | (M, De, Hh) | Key 投影权重（逐头） |
| value_proj_weights | Tensor(BF16) | (De, Hh) | Value 投影权重（所有头共享） |
| key_gamma | Tensor(BF16) | (M, Hh) | Key RMSNorm 缩放系数（逐头） |
| query_gamma | Tensor(BF16) | (M, Hh) | Query RMSNorm 缩放系数（逐头） |
| return_cache | bool | - | 推理场景置 False：kernel 不生成 cache 写出代码、wrapper 不分配 cache buffer，节省显存与回传开销，后 4 项 cache 返回 None（默认 True） |

### 输出参数

| 参数名 | 类型 | 形状 | 说明 |
|-------|------|------|------|
| value_out | Tensor(BF16) | (B, L, M, Hh) | 主输出：门控后的 value |
| score_back | Tensor(FP32) \| None | (B, L, M) | 中间量 score（供反向；return_cache=False 时为 None） |
| key_back | Tensor(BF16) \| None | (B, L, M, Hh) | 中间量 key（投影后、归一化前；return_cache=False 时为 None） |
| value_back | Tensor(BF16) \| None | (B, L, Hh) | 中间量 value（投影后、门控前；return_cache=False 时为 None） |
| gate_back | Tensor(FP32) \| None | (B, L, M) | 中间量 gate（供反向；return_cache=False 时为 None） |

## 算法原理

逐头 $m \in [0, M)$ 计算（BL = B*L，下式对单个 token）：

### Step 1: Key / Value 投影

$$K^{(m)} = E \cdot W_k^{(m)}, \quad V = E \cdot W_v$$

matmul 操作数为 BF16，FP32 累加；$K$ 落 BF16，$V$ 落 BF16。

### Step 2: RMSNorm 中间量

$$rms_k = \sqrt{\text{mean}(K^2) + \varepsilon}, \quad rms_q = \sqrt{\text{mean}(Q^2) + \varepsilon}$$

其中 $Q$ = `hidden_states`，$\varepsilon = 10^{-6}$。

### Step 3: 缩放点积 score

$$s = \frac{\sum_h \big(K_h \cdot Q_h \cdot \gamma_k \cdot \gamma_q\big)}{rms_k \cdot rms_q \cdot \sqrt{Hh}}$$

> 注：把 RMSNorm 归一化“吸收”进分母 $rms_k \cdot rms_q$，与 $\text{sum}\big(\text{rmsNorm}(K,\gamma_k) \cdot \text{rmsNorm}(Q,\gamma_q)\big)/\sqrt{Hh}$ 数学等价。

### Step 4: sign-sqrt 门控

$$g = \sigma\Big(\text{sign}(s) \cdot \sqrt{\text{clamp}(|s|,\, c)}\Big)$$

其中 $\sigma$ 为 sigmoid，$c = 10^{-6}$（clamp 形式，非加性 eps）。

### Step 5: 门控输出

$$O = g \cdot V$$

$O$ 即 `value_out`（BF16）；同时输出 $s$→`score_back`、$K$→`key_back`、$V$→`value_back`、$g$→`gate_back`。

## 约束条件

1. 所有 Tensor 输入为 BF16；`score_back`/`gate_back` 输出为 FP32。
2. Key/Value 投影**无 bias**。
3. 头数 `M` 任意；`B*L` 在 kernel 内合轴，无固定上限。
4. `Hh`、`De` 建议为 128 的倍数（对齐 cube/vec tile，性能最佳），非硬性限制。
5. 同一头内 `key_gamma`、`query_gamma` 形状必须为 (M, Hh)。

## 支持规格

- 数据类型：BF16（输入/输出），FP32（内部计算与 score/gate 输出）
- 芯片平台：A2 / A3

## 使用示例

```python
import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)

B, L, M, Hh, De = 1, 4096, 4, 1536, 640
hidden_states     = torch.randn(B, L, M, Hh, dtype=torch.bfloat16, device="npu:0")
embeddings        = torch.randn(B, L, De,    dtype=torch.bfloat16, device="npu:0")
key_proj_weights  = torch.randn(M, De, Hh,  dtype=torch.bfloat16, device="npu:0") * 0.5
value_proj_weights= torch.randn(De, Hh,     dtype=torch.bfloat16, device="npu:0") * 0.5
key_gamma         = torch.ones(M, Hh, dtype=torch.bfloat16, device="npu:0")
query_gamma       = torch.ones(M, Hh, dtype=torch.bfloat16, device="npu:0")

value_out, score_back, key_back, value_back, gate_back = engram_forward_wrapper(
    hidden_states, embeddings, key_proj_weights, value_proj_weights, key_gamma, query_gamma,
)

print(f"value_out: {value_out.shape}, {value_out.dtype}")   # (1,4096,4,1536) BF16
print(f"score_back: {score_back.shape}, {score_back.dtype}")# (1,4096,4) FP32
```

---

# engram_backward_wrapper

多头门控记忆检索（Engram）反向算子。计算门控记忆操作对权重、embeddings、gamma 的梯度，对应正向算子 `engram_forward_wrapper` 的反向传播。

## 功能描述

反向算子沿 Step5→Step1 逆向求导，分两条主路径：Value 路径（grad_out·gate → value_linear_backward → d_embeddings）与 Key/Query 路径（d_gate → d_score → rms_norm_backward → key_linear_backward → d_embeddings 累加）。输入/输出 BF16，score/gate 走 FP32，跨头/跨 tile 梯度用 FP32 累加器保精度。

## 接口定义

```python
from ops.pypto_tensor.experimental.ops_transformer.engram.engram_backward_impl import engram_backward_wrapper

def engram_backward_wrapper(
    grad_out: Tensor,       # BF16, shape: (B, L, M, Hh)
    hidden_states: Tensor,  # BF16, shape: (B, L, M, Hh)
    embeddings: Tensor,     # BF16, shape: (B, L, De)
    weight_key: Tensor,     # BF16, shape: (M, De, Hh)
    weight_value: Tensor,   # BF16, shape: (De, Hh)
    gamma_key: Tensor,      # BF16, shape: (M, Hh)
    gamma_query: Tensor,    # BF16, shape: (M, Hh)
    score: Tensor,          # FP32, shape: (B, L, M)
    gate: Tensor,           # FP32, shape: (B, L, M)
    key_lineared: Tensor,   # BF16, shape: (B, L, M, Hh)
    value_lineared: Tensor, # BF16, shape: (B, L, Hh)
    clamp_value: float = 1e-6,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    # (d_hidden, d_embeddings, d_weight_key, d_weight_value, d_gamma_key, d_gamma_query)
```

## 参数说明

### 输入参数

| 参数名 | 类型 | 形状 | 默认值 | 说明 |
|-------|------|------|--------|------|
| grad_out | Tensor(BF16) | (B, L, M, Hh) | 必填 | 上游梯度 ∂L/∂O |
| hidden_states | Tensor(BF16) | (B, L, M, Hh) | 必填 | 前向 Query 输入 |
| embeddings | Tensor(BF16) | (B, L, De) | 必填 | 前向 Key/Value 线性层输入 |
| weight_key | Tensor(BF16) | (M, De, Hh) | 必填 | Key 投影权重 |
| weight_value | Tensor(BF16) | (De, Hh) | 必填 | Value 投影权重 |
| gamma_key | Tensor(BF16) | (M, Hh) | 必填 | Key RMSNorm gamma |
| gamma_query | Tensor(BF16) | (M, Hh) | 必填 | Query RMSNorm gamma |
| score | Tensor(FP32) | (B, L, M) | 必填 | 前向 `score_back` |
| gate | Tensor(FP32) | (B, L, M) | 必填 | 前向 `gate_back` |
| key_lineared | Tensor(BF16) | (B, L, M, Hh) | 必填 | 前向 `key_back` |
| value_lineared | Tensor(BF16) | (B, L, Hh) | 必填 | 前向 `value_back` |
| clamp_value | float | - | 1e-6 | sign-sqrt 门控的 clamp 下界 |
| eps | float | - | 1e-6 | RMSNorm 的 ε |

### 输出参数

| 参数名 | 类型 | 形状 | 说明 |
|-------|------|------|------|
| d_hidden | Tensor(BF16) | (B, L, M, Hh) | 对 hidden_states 的梯度 |
| d_embeddings | Tensor(BF16) | (B, L, De) | 对 embeddings 的梯度（key + value 路径之和） |
| d_weight_key | Tensor(BF16) | (M, De, Hh) | 对 weight_key 的梯度 |
| d_weight_value | Tensor(BF16) | (De, Hh) | 对 weight_value 的梯度 |
| d_gamma_key | Tensor(BF16) | (M, Hh) | 对 gamma_key 的梯度 |
| d_gamma_query | Tensor(BF16) | (M, Hh) | 对 gamma_query 的梯度 |

## 算法原理

记 $\hat{K}=\text{rmsNorm}(K,\gamma_k)$、$\hat{Q}=\text{rmsNorm}(Q,\gamma_q)$。逐头反向：

### Step 1: 反门控（O = g · V）

$$\frac{\partial L}{\partial V} = \sum_m \frac{\partial L}{\partial O} \odot g, \qquad \frac{\partial L}{\partial g} = \sum_h \frac{\partial L}{\partial O} \odot V$$

### Step 2: 门控反传（STE 近似）

$$\frac{\partial L}{\partial s} = \frac{\partial L}{\partial g} \cdot g(1-g) \cdot \frac{\text{mask}}{2\sqrt{|s|}}, \qquad \text{mask} = \mathbf{1}_{|s| > c}$$

平坦区 $|s| \le c$ 梯度置 0（detach round 的 STE）。

### Step 3: 点积反传

$$\frac{\partial L}{\partial \hat{K}} = \frac{\partial L}{\partial s} \cdot \frac{\hat{Q}}{\sqrt{Hh}}, \qquad \frac{\partial L}{\partial \hat{Q}} = \frac{\partial L}{\partial s} \cdot \frac{\hat{K}}{\sqrt{Hh}}$$

### Step 4: RMSNorm 反传

$$\frac{\partial L}{\partial K} = \frac{1}{rms_k}\left(\frac{\partial L}{\partial \hat{K}}\gamma_k - \hat{K}\cdot\text{mean}\big(\frac{\partial L}{\partial \hat{K}}\gamma_k \odot \hat{K}\big)\right)$$

$$\frac{\partial L}{\partial \gamma_k} = \sum_{B,L} \frac{\partial L}{\partial \hat{K}} \odot \hat{K}$$

对 Query 路径同理，得到 $\frac{\partial L}{\partial Q}$（即 `d_hidden`）与 $\frac{\partial L}{\partial \gamma_q}$（即 `d_gamma_query`）。

### Step 5: 线性反传

$$\frac{\partial L}{\partial E} \mathrel{+}= \frac{\partial L}{\partial K}\cdot W_k^{(m)T} + \frac{\partial L}{\partial V}\cdot W_v^{T}$$

$$\frac{\partial L}{\partial W_k^{(m)}} = E^T \cdot \frac{\partial L}{\partial K}, \qquad \frac{\partial L}{\partial W_v} = E^T \cdot \frac{\partial L}{\partial V}$$

跨头/跨 tile 的 `d_embeddings`、`d_value` 用 FP32 累加器累加，末尾降 BF16。

## 约束条件

1. `grad_out`、`hidden_states`、`weight_key`、`weight_value`、`gamma_key`、`gamma_query`、`key_lineared`、`value_lineared` 为 BF16。
2. `score`、`gate` 必须为 **FP32**（与前向 `score_back`/`gate_back` 直传，不可降 BF16）。
3. `grad_out` 与 `hidden_states` 形状相同 (B, L, M, Hh)。
4. 无 bias；头数 M 任意。
5. `clamp_value`、`eps` 默认 1e-6，应大于 0。

## 支持规格

- 数据类型：BF16（输入/输出），FP32（内部计算、score/gate 通路、跨头累加器）
- 芯片平台：A2 / A3

## 使用示例

```python
import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)

# 前向（同上）得到 cache
value_out, score_back, key_back, value_back, gate_back = engram_forward_wrapper(
    hidden_states, embeddings, key_proj_weights, value_proj_weights, key_gamma, query_gamma,
)

grad_out = torch.randn_like(value_out)

d_hidden, d_embeddings, d_weight_key, d_weight_value, d_gamma_key, d_gamma_query = engram_backward_wrapper(
    grad_out, hidden_states, embeddings,
    key_proj_weights, value_proj_weights, key_gamma, query_gamma,
    score_back, gate_back, key_back, value_back,   # 前向中间量直传
)

print(f"d_hidden:       {d_hidden.shape}")        # (1,4096,4,1536)
print(f"d_embeddings:   {d_embeddings.shape}")    # (1,4096,640)
print(f"d_weight_key:   {d_weight_key.shape}")    # (4,640,1536)
```

---

## engram_autograd（自动求导封装）

`engram_autograd.py`（位于 tests 目录）把 `engram_forward_wrapper` 与 `engram_backward_wrapper` 封装成
`torch.autograd.Function`（`EngramFunc`），使调用方只需 forward + loss + `.backward()` 即可自动反传，
无需手动衔接前向 cache 与反向输入。

**推荐接口**：`engram_autograd(...)`（等价于 `EngramFunc.apply(...)`）。

```python
from engram_autograd import engram_autograd

# 输入设 requires_grad_(True), autograd 自动算 6 个输入梯度
leaves = [t.clone().requires_grad_(True) for t in (
    hidden_states, embeddings, key_proj_weights, value_proj_weights, key_gamma, query_gamma)]
value_out, score_back, key_back, value_back, gate_back = engram_autograd(*leaves)

loss = value_out.sum()          # 或任意 loss
loss.backward()                 # 自动反传; leaves[i].grad 即对应输入梯度

# leaves[0]->d_hidden  leaves[1]->d_embeddings  leaves[2]->d_weight_key
# leaves[3]->d_weight_value  leaves[4]->d_gamma_key  leaves[5]->d_gamma_query
```

要点：
- `forward` 调 `engram_forward_wrapper`，`save_for_backward` 保存 6 个输入与前向中间激活（score/key/value/gate）。
- `backward` 只对 `value_out` 的梯度求导（其余 4 个中间输出的梯度视为 `None`），调
  `engram_backward_wrapper` 输出 6 个输入梯度（顺序与 forward 输入一致）。
- 适合把 npu kernel 接入 autograd 图的场景（如 cascade 三方比对里 npu 路径走自动反向）。

---

## 前向 → 反向衔接

反向算子的 `score`/`gate`/`key_lineared`/`value_lineared` 直接来自前向输出，对应关系：

| 反向输入 | 来源（前向输出） | 形状 | dtype |
|----------|------------------|------|-------|
| score | score_back | (B, L, M) | FP32 |
| gate | gate_back | (B, L, M) | FP32 |
| key_lineared | key_back | (B, L, M, Hh) | BF16 |
| value_lineared | value_back | (B, L, Hh) | BF16 |

## 运行测试

```bash
export PTO_TILE_LIB_CODE_PATH=<pto-isa 路径>   # 必需，否则 kernel JIT 编译失败
export TILE_FWK_DEVICE_ID=0
cd tests/ops/experimental/ops_transformer/engram
python3 test_engram_forward.py     # 或 test_engram_backward.py / test_engram_cascade.py
```

精度：三方比对（Precision Standard 2.1），`ratio = npu误差/benchmark误差`，阈值 MARE≤2.0 / MERE≤1.2 / RMSE≤1.2。默认用例 `b1_s4096_mhc4_h1536_de640`。

## 文件

- src：`engram_forward_impl.py`、`engram_backward_impl.py`
- tests：`engram_golden.py`（golden 参考）、`compare.py`（精度比对）、`engram_autograd.py`（autograd 封装）、`test_engram_{forward,backward,cascade}.py`
