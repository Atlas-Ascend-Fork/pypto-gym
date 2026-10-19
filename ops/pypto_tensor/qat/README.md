# 量化感知训练（QAT）算子接口文档

本文档包含量化感知训练（Quantization-Aware Training, QAT）相关的算子接口说明，涵盖对称量化和非对称量化两种方式。


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 目录

1. [ai_infra_qat_symmetric_per_tensor](#ai_infra_qat_symmetric_per_tensor) - 对称张量级量化算子
2. [ai_infra_qat_symmetric_per_channel](#ai_infra_qat_symmetric_per_channel) - 对称逐通道量化算子
3. [ai_infra_qat_asymmetric_per_group](#ai_infra_qat_asymmetric_per_group) - 非对称分组量化算子

---

# ai_infra_qat_symmetric_per_tensor

对称量化感知训练（Quantization-Aware Training, QAT）算子，包含正向和反向两个算子。该算子用于对权重进行对称量化模拟，在训练过程中引入量化噪声，使模型能够适应量化带来的精度损失。

适用于 Embedding 层场景，scale 为标量形式（shape 为 (1,1)），所有权重元素共享同一个缩放系数。


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 正向算子：ai_infra_qat_symmetric_per_tensor

### 功能描述

对称量化感知训练（Quantization-Aware Training, QAT）正向算子。该算子用于对权重进行对称量化模拟，在训练过程中引入量化噪声，使模型能够适应量化带来的精度损失。

该算子适用于 Embedding 层场景，scale 为标量形式（shape 为 (1,1)），所有权重元素共享同一个缩放系数。

### 接口定义

```python
def ai_infra_qat_symmetric_per_tensor(
    weight: Tensor,      # BF16, shape: (N, M)
    scale: Tensor,       # BF16, shape: (1, 1)
    eps: float,          # 最小scale阈值
    min_v: float,        # 量化下界
    max_v: float,        # 量化上界
) -> Tensor:             # BF16, shape: (N, M)
```

### 参数说明

#### 输入参数

| 参数名 | 类型 | 形状 | 说明 |
|-------|------|------|------|
| weight | Tensor(BF16) | (N, M) | 输入权重张量，N为输出特征维度，M为输入特征维度 |
| scale | Tensor(BF16) | (1, 1) | 量化缩放系数，标量形式，所有权重元素共享同一缩放系数 |
| eps | float | - | scale的最小阈值，防止除零错误，当scale小于eps时使用eps替代 |
| min_v | float | - | 量化下界，对于INT8量化为-128 |
| max_v | float | - | 量化上界，对于INT8量化为127 |

#### 输出参数

| 参数名 | 类型 | 形状 | 说明 |
|-------|------|------|------|
| output | Tensor(BF16) | (N, M) | 伪量化后的权重张量，与输入权重形状相同 |

### 算法原理

对称量化感知训练的核心是使用直通估计器（Straight-Through Estimator, STE）来近似量化操作的梯度。算子通过以下步骤实现：

#### Step 1: Scale 防零保护

$$
s' = \begin{cases} s, & \text{if } s > \varepsilon \\ \varepsilon, & \text{otherwise} \end{cases}
$$

其中 $s$ 为输入的 scale，$\varepsilon$ 为 eps 参数。

#### Step 2: 归一化

$$
W_{\text{norm}} = \frac{W}{s'}
$$

将权重除以缩放系数，得到归一化后的权重。

#### Step 3: 伪量化（STE近似的四舍五入 + 截断）

$$
W_{\text{quant}} = \text{detach}\left(\text{round}(W_{\text{norm}}) - W_{\text{norm}}\right) + W_{\text{norm}}
$$

$$
W_{\text{clamp}} = \text{clamp}(W_{\text{quant}}, V_{\text{min}}, V_{\text{max}})
$$

使用 detach() 阻断四舍五入操作的梯度传播，实现 STE 近似。然后将量化后的值截断到有效范围内。

#### Step 4: 反量化

$$
W_q = W_{\text{clamp}} \times s'
$$

将截断后的值乘以缩放系数，恢复到原始数值范围。

### 约束条件

1. `weight` 必须为 2 维张量，形状为 (N, M)。M∈[128, 3072]，且被128整除。数据类型为BF16。
2. `scale` 必须为 2 维张量，形状为 (1, 1)，数据类型为BF16。
3. `eps` 应在 (0, 1) 范围内，数据类型为float。
4. `min_v` 应小于 `max_v`，并且都为浮点型整数。数据类型为float，小数位为全0。

### 支持规格

- 数据类型: BF16（输入/输出），FP32（内部计算）
- 芯片平台: A2/A3

### 使用示例

```python
import torch
import pypto

# 设置设备
torch.npu.set_device(0)

# 创建输入张量
N, M = 1024, 2048
weight = torch.randn(N, M, dtype=torch.bfloat16, device="npu:0")
scale = torch.tensor([[0.1]], dtype=torch.bfloat16, device="npu:0")

# 创建并调用算子
output = ai_infra_qat_symmetric_per_tensor(weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0)

print(f"输入权重形状: {weight.shape}")
print(f"输出权重形状: {output.shape}")
```

---

## 反向算子：ai_infra_qat_symmetric_per_tensor_backward

### 功能描述

对称量化感知训练（QAT）反向算子。该算子用于计算对称量化操作的梯度，支持权重和缩放系数的梯度计算。

该算子适用于 Embedding 层场景，scale 为标量形式（shape 为 (1,1)），对应正向算子 ai_infra_qat_symmetric_per_tensor 的反向传播。

### 接口定义

```python
def ai_infra_qat_symmetric_per_tensor_backward(
    grad_output: Tensor,  # BF16, shape: (N, M)
    weight: Tensor,       # BF16, shape: (N, M)
    scale: Tensor,        # BF16, shape: (1, 1)
    eps: float = 1e-4,    # 最小scale阈值
    min_v: float = -128.0,  # 量化下界
    max_v: float = 127.0,   # 量化上界
) -> Tuple[Tensor, Tensor]:  # (grad_weight, grad_scale)
```

### 参数说明

#### 输入参数

| 参数名 | 类型 | 形状 | 默认值 | 说明 |
|-------|------|------|--------|------|
| grad_output | Tensor(BF16) | (N, M) | 必填 | 上游传递的梯度张量 |
| weight | Tensor(BF16) | (N, M) | 必填 | 原始输入权重张量 |
| scale | Tensor(BF16) | (1, 1) | 必填 | 量化缩放系数 |
| eps | float | - | 1e-4 | scale的最小阈值 |
| min_v | float | - | -128.0 | 量化下界 |
| max_v | float | - | 127.0 | 量化上界 |

#### 输出参数

| 参数名 | 类型 | 形状 | 说明 |
|-------|------|------|------|
| grad_weight | Tensor(BF16) | (N, M) | 对权重的梯度 |
| grad_scale | Tensor(BF16) | (1, 1) | 对缩放系数的梯度 |

### 算法原理

反向传播需要计算损失函数对权重 W 和缩放系数 s 的梯度。基于正向传播的计算图，梯度通过链式法则反向传递。

#### 第一步：反量化（$W_q = W_{clamp} \times s'$）的梯度

对 $W_{clamp}$ 的梯度：
$$
\frac{\partial\text{Loss}}{\partial W_{\text{clamp}}} = \frac{\partial\text{Loss}}{\partial W_q} \times s'
$$

对 $s'$ 的梯度：
$$
\frac{\partial\text{Loss}}{\partial s'} = \sum_{i=1}^N \sum_{j=1}^M \left( \frac{\partial\text{Loss}}{\partial W_q}[i,j] \times W_{\text{clamp}}[i,j] \right)
$$

#### 第二步：截断的梯度（STE 近似）

$$
\frac{\partial\text{Loss}}{\partial W_{\text{quant}}} = \frac{\partial\text{Loss}}{\partial W_{\text{clamp}}}\odot\begin{cases} 1, & V_{\text{min}}\leq W_{\text{quant}}\leq V_{\text{max}}\\ 0, &\text{otherwise}\end{cases}
$$

#### 第三步：伪量化的梯度

$$
\frac{\partial\text{Loss}}{\partial W_{\text{norm}}} = \frac{\partial\text{Loss}}{\partial W_{\text{quant}}}
$$

#### 第四步：归一化的梯度

对原始权重 W 的梯度：
$$
\frac{\partial\text{Loss}}{\partial W} = \frac{\partial\text{Loss}}{\partial W_{\text{norm}}} \times\frac{1}{s'}
$$

对 s' 的梯度（叠加）：
$$
\frac{\partial\text{Loss}}{\partial s'} \mathrel{+}= -\frac{1}{(s')^2} \times\text{sum}\left( \frac{\partial\text{Loss}}{\partial W_{\text{norm}}} \odot W \right)
$$

#### 第五步：scale 防零的梯度

$$
\frac{\partial\text{Loss}}{\partial s} = \frac{\partial\text{Loss}}{\partial s'}\cdot\begin{cases} 1, & s > \varepsilon\\ 0, &\text{otherwise}\end{cases}
$$

#### 最终合并公式

**对 W 的梯度**：
$$
\frac{\partial\text{Loss}}{\partial W} = \frac{\partial\text{Loss}}{\partial W_q}\odot\begin{cases} 1, & V_{\text{min}}\leq W_{\text{quant}}\leq V_{\text{max}}\\ 0, &\text{otherwise}\end{cases}
$$

**对 s 的梯度**：
$$
\frac{\partial \text{Loss}}{\partial s} = \left[ \text{sum}\left( \frac{\partial \text{Loss}}{\partial W_q} \odot W_{\text{clamp}} \right) - \frac{1}{s'} \cdot \text{sum}\left( \frac{\partial \text{Loss}}{\partial W_q} \odot \mathbf{1}_{\text{mask}} \odot W \right) \right] \cdot \mathbf{1}_{s > \varepsilon}
$$

### 约束条件

1. `grad_output` 和 `weight` 的形状必须相同，为 (N, M)。M∈[128, 3072]，且被128整除。数据类型为BF16。
2. `scale` 的形状必须为 (1, 1)，数据类型为BF16。
3. `eps` 应在 (0, 1) 范围内，数据类型为float。
4. `min_v` 应小于 `max_v`，并且都为浮点型整数。数据类型为float，小数位为全0。

### 支持规格

- 数据类型: BF16（输入/输出），FP32（内部计算）
- 芯片平台: A2/A3

### 使用示例

```python
import torch
import pypto

# 设置设备
torch.npu.set_device(0)

# 创建输入张量
N, M = 1024, 2048
weight = torch.randn(N, M, dtype=torch.bfloat16, device="npu:0", requires_grad=True)
scale = torch.tensor([[0.1]], dtype=torch.bfloat16, device="npu:0", requires_grad=True)

# 前向传播
output = ai_infra_qat_symmetric_per_tensor(weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0)

# 模拟上游梯度
grad_output = torch.ones_like(output)

# 反向传播
grad_weight, grad_scale = ai_infra_qat_symmetric_per_tensor_backward(
    grad_output, weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0
)

print(f"权重梯度形状: {grad_weight.shape}")
print(f"Scale梯度形状: {grad_scale.shape}")
```

---

# ai_infra_qat_symmetric_per_channel

对称量化感知训练（Quantization-Aware Training, QAT）算子（N scale 版本），包含正向和反向两个算子。该算子用于对权重进行对称量化模拟，支持每个输出通道独立的缩放系数。

适用于 Lm Head 层场景，scale 为向量形式（shape 为 (N,1)），每个输出通道对应一个独立的缩放系数。


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 正向算子：ai_infra_qat_symmetric_per_channel

### 功能描述

对称量化感知训练（Quantization-Aware Training, QAT）正向算子（N scale 版本）。该算子用于对权重进行对称量化模拟，支持每个输出通道独立的缩放系数。

该算子适用于 Lm Head 层场景，scale 为向量形式（shape 为 (N,1)），每个输出通道对应一个独立的缩放系数。

### 接口定义

```python
def ai_infra_qat_symmetric_per_channel(
    weight: Tensor,      # BF16, shape: (N, M)
    scale: Tensor,       # BF16, shape: (N, 1)
    eps: float = 1e-4,   # 最小scale阈值
    min_v: float = -128.0,  # 量化下界
    max_v: float = 127.0,   # 量化上界
) -> Tensor:             # BF16, shape: (N, M)
```

### 参数说明

#### 输入参数

| 参数名 | 类型 | 形状 | 默认值 | 说明 |
|-------|------|------|--------|------|
| weight | Tensor(BF16) | (N, M) | 必填 | 输入权重张量，N为输出特征维度，M为输入特征维度 |
| scale | Tensor(BF16) | (N, 1) | 必填 | 量化缩放系数，每个输出通道对应一个独立的缩放系数 |
| eps | float | - | 1e-4 | scale的最小阈值，防止除零错误 |
| min_v | float | - | -128.0 | 量化下界，对于INT8量化为-128 |
| max_v | float | - | 127.0 | 量化上界，对于INT8量化为127 |

#### 输出参数

| 参数名 | 类型 | 形状 | 说明 |
|-------|------|------|------|
| output | Tensor(BF16) | (N, M) | 伪量化后的权重张量，与输入权重形状相同 |

### 算法原理

与 symmetric_qat 算法原理相同，主要区别在于 scale 的形状。当 scale 为 (N, 1) 时，每个输出通道使用独立的缩放系数。

#### Step 1: Scale 防零保护

$$
s'_i = \begin{cases} s_i, & \text{if } s_i > \varepsilon \\ \varepsilon, & \text{otherwise} \end{cases}
$$

其中 $s_i$ 为第 i 个输出通道的 scale。

#### Step 2: 归一化（广播缩放）

$$
W_{\text{norm}}[i, j] = \frac{W[i, j]}{s'_i}
$$

scale 沿 M 维度广播，实现逐通道归一化。

#### Step 3: 伪量化（STE近似的四舍五入 + 截断）

$$
W_{\text{quant}} = \text{detach}\left(\text{round}(W_{\text{norm}}) - W_{\text{norm}}\right) + W_{\text{norm}}
$$

$$
W_{\text{clamp}} = \text{clamp}(W_{\text{quant}}, V_{\text{min}}, V_{\text{max}})
$$

#### Step 4: 反量化

$$
W_q[i, j] = W_{\text{clamp}}[i, j] \times s'_i
$$

### 约束条件

1. `weight` 必须为 2 维张量，形状为 (N, M)。M∈[128, 3072]，且被128整除。数据类型为BF16。
2. `scale` 必须为 2 维张量，形状为 (N, 1)，数据类型为BF16
3. `eps` 应在 (0, 1) 范围内，数据类型为float
4. `min_v` 应小于 `max_v`，并且都为浮点型整数。数据类型为float，小数位为全0。

### 支持规格

- 数据类型: BF16（输入/输出），FP32（内部计算）
- 芯片平台: A2/A3

### 使用示例

```python
import torch
import pypto

# 设置设备
torch.npu.set_device(0)

# 创建输入张量 - Lm Head 场景
# 例如：词表大小 153376，隐藏维度 2048
N, M = 153376, 2048
weight = torch.randn(N, M, dtype=torch.bfloat16, device="npu:0")
scale = torch.abs(torch.randn(N, 1, dtype=torch.bfloat16, device="npu:0")) + 0.01

# 创建并调用算子
output = ai_infra_qat_symmetric_per_channel(weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0)

print(f"输入权重形状: {weight.shape}")
print(f"Scale形状: {scale.shape}")
print(f"输出权重形状: {output.shape}")
```

---

## 反向算子：ai_infra_qat_symmetric_per_channel_backward

### 功能描述

对称量化感知训练（QAT）反向算子（N scale 版本）。该算子用于计算对称量化操作的梯度，支持每个输出通道独立的缩放系数梯度计算。

该算子适用于 Lm Head 层场景，scale 为向量形式（shape 为 (N,1)），对应正向算子 ai_infra_qat_symmetric_per_channel 的反向传播。

### 接口定义

```python
def ai_infra_qat_symmetric_per_channel_backward(
    grad_output: Tensor,  # BF16, shape: (N, M)
    weight: Tensor,       # BF16, shape: (N, M)
    scale: Tensor,        # BF16, shape: (N, 1)
    eps: float = 1e-4,    # 最小scale阈值
    min_v: float = -128.0,  # 量化下界
    max_v: float = 127.0,   # 量化上界
) -> Tuple[Tensor, Tensor]:  # (grad_weight, grad_scale)
```

### 参数说明

#### 输入参数

| 参数名 | 类型 | 形状 | 默认值 | 说明 |
|-------|------|------|--------|------|
| grad_output | Tensor(BF16) | (N, M) | 必填 | 上游传递的梯度张量 |
| weight | Tensor(BF16) | (N, M) | 必填 | 原始输入权重张量 |
| scale | Tensor(BF16) | (N, 1) | 必填 | 量化缩放系数，每个输出通道一个 |
| eps | float | - | 1e-4 | scale的最小阈值 |
| min_v | float | - | -128.0 | 量化下界 |
| max_v | float | - | 127.0 | 量化上界 |

#### 输出参数

| 参数名 | 类型 | 形状 | 说明 |
|-------|------|------|------|
| grad_weight | Tensor(BF16) | (N, M) | 对权重的梯度 |
| grad_scale | Tensor(BF16) | (N, 1) | 对缩放系数的梯度，每个输出通道一个 |

### 算法原理

与 ai_infra_qat_symmetric_per_tensor_backward 算法原理相似，但由于 scale 为 (N, 1) 形状，梯度计算有所不同。

#### 关键差异

由于每个输出通道有独立的 scale，梯度计算无需跨通道求和，只需沿 M 维度求和：

**对 W 的梯度**：
$$
\frac{\partial\text{Loss}}{\partial W}[i,j] = \frac{\partial\text{Loss}}{\partial W_q}[i,j] \cdot \mathbf{1}_{\text{mask}}[i,j] \cdot \frac{1}{s'_i}
$$

**对 s 的梯度**：
$$
\frac{\partial\text{Loss}}{\partial s_i} = \left[ \sum_{j=1}^{M} \left( \frac{\partial\text{Loss}}{\partial W_q}[i,j] \cdot W_{\text{clamp}}[i,j] \right) - \frac{1}{s'_i} \sum_{j=1}^{M} \left( \frac{\partial\text{Loss}}{\partial W_q}[i,j] \cdot \mathbf{1}_{\text{mask}}[i,j] \cdot W[i,j] \right) \right] \cdot \mathbf{1}_{s_i > \varepsilon}
$$

#### 梯度计算步骤

1. **重算前向中间值**：根据 weight 和 scale 重新计算 normalized、rounded、clamped
2. **计算截断掩码**：判断元素是否在 [min_v, max_v] 范围内
3. **计算 scale 掩码**：判断 scale 是否大于 eps
4. **计算 grad_weight**：上游梯度 × 截断掩码
5. **计算 grad_scale**：乘法路径 + 除法路径，沿 M 维度求和

### 约束条件

1. `grad_output` 和 `weight` 的形状必须相同，为 (N, M)。M∈[128, 3072]，且被128整除。数据类型为BF16。
2. `scale` 的形状必须为 (N, 1)，数据类型为BF16。
3. `eps` 应在 (0, 1) 范围内，数据类型为float。
4. `min_v` 应小于 `max_v`，并且都为浮点型整数。数据类型为float，小数位为全0。

### 支持规格

- 数据类型: BF16（输入/输出），FP32（内部计算）
- 芯片平台: A2/A3

### 使用示例

```python
import torch
import pypto

# 设置设备
torch.npu.set_device(0)

# 创建输入张量 - Lm Head 场景
N, M = 153376, 2048
weight = torch.randn(N, M, dtype=torch.bfloat16, device="npu:0", requires_grad=True)
scale = torch.abs(torch.randn(N, 1, dtype=torch.bfloat16, device="npu:0")) + 0.01
scale.requires_grad_(True)

# 前向传播
output = ai_infra_qat_symmetric_per_channel(weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0)

# 模拟上游梯度
grad_output = torch.ones_like(output)

# 反向传播
grad_weight, grad_scale = ai_infra_qat_symmetric_per_channel_backward(
    grad_output, weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0
)

print(f"权重梯度形状: {grad_weight.shape}")
print(f"Scale梯度形状: {grad_scale.shape}")
```

---

## 与 ai_infra_qat_symmetric_per_tensor 的区别

| 特性 | ai_infra_qat_symmetric_per_tensor | ai_infra_qat_symmetric_per_channel |
|-----|--------------------------------|-----------------------------------|
| scale 形状 | (1, 1) | (N, 1) |
| 缩放粒度 | 全局统一缩放 | 逐通道独立缩放 |
| 适用场景 | Embedding 层 | Lm Head 层 |
| 量化精度 | 较低 | 较高 |
| grad_scale 形状 | (1, 1) | (N, 1) |
| 梯度求和维度 | 沿 N 和 M 维度求和 | 仅沿 M 维度求和 |

---

# ai_infra_qat_asymmetric_per_group

非对称量化感知训练（Asymmetric Quantization-Aware Training, QAT）算子，包含正向和反向两个算子。该算子用于对权重进行非对称量化模拟，支持分组量化（Group Quantization）和可学习的偏移量（offset）。

适用于 Transformer Linear 层场景，通过分组量化实现更精细的量化粒度，提高量化后模型的精度。非对称量化相比对称量化能够更好地适应权重分布的不对称性。


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 正向算子：ai_infra_qat_asymmetric_per_group

### 功能描述

非对称量化感知训练（Asymmetric Quantization-Aware Training, QAT）正向算子。该算子用于对权重进行非对称量化模拟，支持分组量化（Group Quantization）和可学习的偏移量（offset）。

该算子适用于 Transformer Linear 层场景，通过分组量化实现更精细的量化粒度，提高量化后模型的精度。非对称量化相比对称量化能够更好地适应权重分布的不对称性。

### 接口定义

```python
def ai_infra_qat_asymmetric_per_group(
    weight: Tensor,      # BF16, shape: (N, M)
    scale: Tensor,       # BF16, shape: (N*M/group_size, 1)
    offset: Tensor,      # BF16, shape: (N*M/group_size, 1)
    group_size: int = 128,  # 分组大小
    bit: int = 4,        # 量化位宽，支持2、3、4
    eps: float = 1e-4,   # 最小scale阈值
    clip_val: float = 0.99,  # 截断值
) -> Tensor:             # BF16, shape: (N, M)
```

### 参数说明

#### 输入参数

| 参数名 | 类型 | 形状 | 默认值 | 说明 |
|-------|------|------|--------|------|
| weight | Tensor(BF16) | (N, M) | 必填 | 输入权重张量，N为输出特征维度，M为输入特征维度 |
| scale | Tensor(BF16) | (N*M/group_size, 1) | 必填 | 量化缩放系数，每组一个缩放系数 |
| offset | Tensor(BF16) | (N*M/group_size, 1) | 必填 | 量化偏移量，每组一个偏移量 |
| group_size | int | - | 128 | 分组大小，每组包含的权重元素数量 |
| bit | int | - | 4 | 量化位宽，支持2、3、4位量化 |
| eps | float | - | 1e-4 | scale的最小阈值，防止除零错误 |
| clip_val | float | - | 0.99 | 截断值，用于限制量化范围 |

#### 输出参数

| 参数名 | 类型 | 形状 | 说明 |
|-------|------|------|------|
| output | Tensor(BF16) | (N, M) | 伪量化后的权重张量，与输入权重形状相同 |

### 算法原理

非对称量化感知训练基于增强型 LSQ+（Learned Step Size Quantization Plus）算法，通过分组量化和可学习的 scale/offset 实现更灵活的量化。

#### 核心公式

##### Step 1: Scale 防零保护

$$
s' = \begin{cases} s, & \text{if } s > \varepsilon \\ \varepsilon, & \text{otherwise} \end{cases}
$$

##### Step 2: 权重重塑为分组形式

$$
W_{\text{group}} = \text{reshape}(W, [G, \text{group\_size}])
$$

其中 $G = \frac{N \times M}{\text{group\_size}}$ 为总组数。

##### Step 3: 计算量化参数

$$
\alpha = s' \times n_{\text{levels}}, \quad n_{\text{levels}} = 2^{(\text{bit}-1)}
$$

$$
\text{shift} = 0.5
$$

##### Step 4: 非对称量化

$$
W_{\text{shifted}} = W_{\text{group}} - \text{offset}
$$

$$
W_{\text{clipped}} = \text{clamp}\left(\frac{W_{\text{shifted}}}{\alpha}, -\text{clip\_val}, \text{clip\_val}\right) \times n_{\text{levels}} - \text{shift}
$$

##### Step 5: 伪量化（STE）

$$
W_{\text{rounded}} = \text{detach}(\text{round}(W_{\text{clipped}}) - W_{\text{clipped}}) + W_{\text{clipped}}
$$

##### Step 6: 反量化

$$
W_{\text{unshifted}} = W_{\text{rounded}} + \text{shift}
$$

$$
W_{\text{denorm}} = \frac{W_{\text{unshifted}}}{n_{\text{levels}}}
$$

$$
W_{\text{out}} = W_{\text{denorm}} \times \alpha + \text{offset}
$$

### 分组量化说明

分组量化将权重矩阵划分为多个小组，每组拥有独立的 scale 和 offset：

```
权重矩阵 (N, M):
┌─────────────────────────────────────┐
│  Group 0  │  Group 1  │ ... │ Group G-1 │
│ (128个元素)│ (128个元素)│     │ (128个元素) │
└─────────────────────────────────────┘
     ↓            ↓              ↓
  scale[0]    scale[1]       scale[G-1]
  offset[0]   offset[1]      offset[G-1]
```

### 约束条件

1. `group_size` 取64、128、256，数据类型为int。
2. `weight` 必须为 2 维张量，形状为 (N, M)，M∈[128, 3072]且被 `group_size` 整除。数据类型为BF16。
3. `scale` 和 `offset` 的形状必须为 (N*M/group_size, 1)。数据类型为BF16。
4. `bit` 只能取 2、3、4，数据类型为int。
5. `eps` 应在 (0, 1) 范围内，数据类型为float。
6. `clip_val` 应在 (0, 1) 范围内，数据类型为float。

### 支持规格

- 数据类型: BF16（输入/输出），FP32（内部计算）
- 芯片平台: A2/A3
- 支持位宽: 2-bit, 3-bit, 4-bit

### 使用示例

```python
import torch
import pypto

# 设置设备
torch.npu.set_device(0)

# 创建输入张量 - Transformer Linear 层场景
# 例如：FFN 层，输入维度 3072，输出维度 768
N, M = 768, 3072
group_size = 128
bit = 4

# 计算组数
num_groups = N * M // group_size

# 创建输入
weight = torch.randn(N, M, dtype=torch.bfloat16, device="npu:0")
scale = torch.abs(torch.randn(num_groups, 1, dtype=torch.bfloat16, device="npu:0")) + 0.01
offset = torch.randn(num_groups, 1, dtype=torch.bfloat16, device="npu:0")

# 创建并调用算子
output = ai_infra_qat_asymmetric_per_group(weight, scale, offset, group_size=group_size, bit=bit, eps=1e-4, clip_val=0.99)

print(f"输入权重形状: {weight.shape}")
print(f"Scale形状: {scale.shape}")
print(f"Offset形状: {offset.shape}")
print(f"输出权重形状: {output.shape}")
```

---

## 反向算子：ai_infra_qat_asymmetric_per_group_backward

### 功能描述

非对称量化感知训练（QAT）反向算子。该算子用于计算非对称量化操作的梯度，支持权重、缩放系数和偏移量的梯度计算。

该算子适用于 Transformer Linear 层场景，对应正向算子 ai_infra_qat_asymmetric_per_group 的反向传播，通过分组量化实现精细的梯度计算。

### 接口定义

```python
def ai_infra_qat_asymmetric_per_group_backward(
    grad_output: Tensor,  # BF16, shape: (N, M)
    weight: Tensor,       # BF16, shape: (N, M)
    scale: Tensor,        # BF16, shape: (N*M/group_size, 1)
    offset: Tensor,       # BF16, shape: (N*M/group_size, 1)
    group_size: int = 128,  # 分组大小
    bit: int = 4,        # 量化位宽
    eps: float = 1e-4,   # 最小scale阈值
    clip_val: float = 0.99,  # 截断值
) -> Tuple[Tensor, Tensor, Tensor]:  # (grad_weight, grad_scale, grad_offset)
```

### 参数说明

#### 输入参数

| 参数名 | 类型 | 形状 | 默认值 | 说明 |
|-------|------|------|--------|------|
| grad_output | Tensor(BF16) | (N, M) | 必填 | 上游传递的梯度张量 |
| weight | Tensor(BF16) | (N, M) | 必填 | 原始输入权重张量 |
| scale | Tensor(BF16) | (N*M/group_size, 1) | 必填 | 量化缩放系数 |
| offset | Tensor(BF16) | (N*M/group_size, 1) | 必填 | 量化偏移量 |
| group_size | int | - | 128 | 分组大小 |
| bit | int | - | 4 | 量化位宽，支持2、3、4 |
| eps | float | - | 1e-4 | scale的最小阈值 |
| clip_val | float | - | 0.99 | 截断值 |

#### 输出参数

| 参数名 | 类型 | 形状 | 说明 |
|-------|------|------|------|
| grad_weight | Tensor(BF16) | (N, M) | 对权重的梯度 |
| grad_scale | Tensor(BF16) | (N*M/group_size, 1) | 对缩放系数的梯度 |
| grad_offset | Tensor(BF16) | (N*M/group_size, 1) | 对偏移量的梯度 |

### 算法原理

反向传播基于 LSQ+ 算法的梯度公式，需要重新计算前向传播的中间变量，并正确处理截断区域和非截断区域的梯度。

#### 前向状态重计算

首先重算以下中间变量：
- `protected_scale`: 经过防零保护的 scale
- `alpha`: = protected_scale × n_levels
- `weight_shifted`: = weight - offset
- `weight_scaled`: = weight_shifted / alpha（未截断）
- `weight_clipped`: 截断后的值
- `weight_denorm`: 反量化后的值

#### 梯度计算

##### 截断掩码（STE 激活区域）

$$
\text{mask}[i,j] = \begin{cases} 1, & -\text{clip\_val} \leq W_{\text{scaled}}[i,j] \leq \text{clip\_val} \\ 0, & \text{otherwise} \end{cases}
$$

##### grad_weight 计算

只有在截断范围内的元素才能传导梯度：

$$
\frac{\partial\text{Loss}}{\partial W} = \frac{\partial\text{Loss}}{\partial W_{\text{out}}} \odot \text{mask}
$$

##### grad_offset 计算

截断区域外的梯度累加到 offset：

$$
\frac{\partial\text{Loss}}{\partial \text{offset}} = \sum_{j \in \text{group}} \left( \frac{\partial\text{Loss}}{\partial W_{\text{out}}} \odot (1 - \text{mask}) \right)
$$

沿 group 维度求和。

##### grad_scale 计算

$$
\frac{\partial\text{Loss}}{\partial s} = \frac{\partial\text{Loss}}{\partial \alpha} \times n_{\text{levels}} \times \mathbf{1}_{s > \varepsilon}
$$

其中：

$$
\frac{\partial\text{Loss}}{\partial \alpha} = \sum_{j \in \text{group}} \left( \frac{\partial\text{Loss}}{\partial W_{\text{out}}} \odot (W_{\text{denorm}} - W_{\text{scaled}} \odot \text{mask}) \right)
$$

### 关键实现细节

#### 无缓存设计

该算子采用无缓存设计，在反向传播时重新计算前向传播的中间变量，避免了存储大量中间状态带来的内存开销。

#### 掩码生成

使用数值方法生成掩码，避免直接使用条件判断：

```python
# 判断是否在截断范围内
diff = weight_norm - weight_clipped
abs_diff = abs(diff)
is_out = clip(abs_diff * big_number, 0.0, 1.0)
mask = 1.0 - is_out
```

#### 梯度累加

对于 scale 和 offset 的梯度，需要沿 group 维度进行求和：

```python
grad_offset = (grad_output * (1 - mask)).sum(dim=1, keepdim=True)
grad_alpha = (grad_output * (weight_denorm - weight_scaled * mask)).sum(dim=1, keepdim=True)
```

### 约束条件

1. `group_size` 取64、128、256，数据类型为int。
2. `grad_output` 和 `weight` 的形状必须相同，为 (N, M)，M∈[128, 3072]且被 `group_size` 整除。数据类型为BF16。
3. `scale` 和 `offset` 的形状必须为 (N*M/group_size, 1)，数据类型为BF16。
4. `bit` 只能取 2、3、4，数据类型为int。
5. `eps` 应在 (0, 1) 范围内，数据类型为float。
6. `clip_val` 应在 (0, 1) 范围内，数据类型为float。

### 支持规格

- 数据类型: BF16（输入/输出），FP32（内部计算）
- 芯片平台: A2/A3
- 支持位宽: 2-bit, 3-bit, 4-bit

### 使用示例

```python
import torch
import pypto

# 设置设备
torch.npu.set_device(0)

# 创建输入张量
N, M = 1024, 2048
group_size = 128
bit = 4
num_groups = N * M // group_size

weight = torch.randn(N, M, dtype=torch.bfloat16, device="npu:0", requires_grad=True)
scale = torch.abs(torch.randn(num_groups, 1, dtype=torch.bfloat16, device="npu:0")) + 0.01
scale.requires_grad_(True)
offset = torch.randn(num_groups, 1, dtype=torch.bfloat16, device="npu:0", requires_grad=True)

# 前向传播
output = ai_infra_qat_asymmetric_per_group(weight, scale, offset, group_size=group_size, bit=bit, eps=1e-4, clip_val=0.99)

# 模拟上游梯度
grad_output = torch.ones_like(output)

# 反向传播
grad_weight, grad_scale, grad_offset = ai_infra_qat_asymmetric_per_group_backward(
    grad_output, weight, scale, offset, group_size=group_size, bit=bit, eps=1e-4, clip_val=0.99
)

print(f"权重梯度形状: {grad_weight.shape}")
print(f"Scale梯度形状: {grad_scale.shape}")
print(f"Offset梯度形状: {grad_offset.shape}")
```

---

## 性能测试结果

> 测试环境: Ascend 910B, CANN 8.5.0
>
> 测试方法: Swimlane (泳道图) — `debug_options={"runtime_debug_mode": 1}`, 解析 `merged_swimlane.json` X 事件 span

| 算子 | 输入 shape | 输入 dtype | kernel 耗时 (μs) | 数据来源 |
|------|-----------|-----------|-----------------|---------|
| qat_symmetric_per_channel (forward) | weight=[1024, 2048] | bfloat16 | 23.1 | Swimlane |

## 与对称量化的对比

| 特性 | 对称量化 | 非对称量化 |
|-----|---------|-----------|
| 量化范围 | [-Q, Q] | [-Q, Q] + offset |
| 参数数量 | scale | scale + offset |
| 适用场景 | 权重分布对称 | 权重分布不对称 |
| 量化精度 | 较低 | 较高 |
| 计算复杂度 | 较低 | 较高 |