# scatter_nd_sub


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

`scatter_nd_sub` 算子实现了对目标张量指定位置的减法更新操作。该算子将 updates 张量中的值从 target 张量的指定索引位置减去，支持对同一位置的多次更新（累加模式）。

该算子对应 TensorFlow 的 `tf.scatter_nd_sub` 操作，常用于稀疏更新场景，如嵌入表更新、梯度更新等。

## 数学公式

$$
\text{target}[\text{indices}[i], :] = \text{target}[\text{indices}[i], :] - \text{updates}[i, :]
$$

其中：
- $\text{target}$：目标张量，将被原地修改
- $\text{indices}$：索引张量，指定要更新的位置
- $\text{updates}$：更新值张量，包含要从目标位置减去的值

对于重复索引的情况，多次减法操作会累加执行。

## 函数原型

```Python
def scatter_nd_sub_kernel(
    target: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    indices: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    updates: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32)
) -> None:
```

## 参数说明

> **说明：**
> - M 表示目标张量的第一维度大小
> - N 表示目标张量的第二维度大小（特征维度）
> - K 表示更新操作的数量（indices 和 updates 的第一维度）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
|--------|-----------|------|----------|----------|-------------|
| target | 输入/输出 | 目标张量，将被原地修改 | float32 | ND | [M, N] |
| indices | 输入 | 索引张量，指定要更新的位置 | int32 | ND | [K, 1] |
| updates | 输入 | 更新值张量，包含要从目标位置减去的值 | float32 | ND | [K, N] |

## 实现原理

该算子通过以下步骤实现：

1. 使用 `pypto.loop_unroll` 对 indices 进行分块处理，支持 [2048, 1024, 512, 256, 1] 等多种分块大小
2. 对每个分块：
   - 提取当前批次的 indices 和 updates
   - 将 updates 取负（乘以 -1）
   - 使用 `pypto.index_put_` 将负值累加到目标位置

## 调用示例

```Python
import torch
import pypto

# 准备数据
target = torch.rand([10000, 16], dtype=torch.float32)
indices = torch.randint(0, 10000, [1024, 1], dtype=torch.int32)
updates = torch.rand([1024, 16], dtype=torch.float32)

# 调用算子
scatter_nd_sub_kernel(target, indices, updates)
```

## 精度验证

本算子使用 TensorFlow 的 `tf.compat.v1.scatter_nd_sub` 作为基准进行精度验证，最大误差容忍度为 1e-1。

## 支持的输入场景

算子已验证以下输入场景：

| target shape | indices shape | updates shape |
|--------------|---------------|---------------|
| [10, 16] | [1024, 1] | [1024, 16] |
| [1000000, 16] | [12900, 1] | [12900, 16] |
| [104007, 8] | [2048, 1] | [2048, 8] |
| [22692, 8] | [2048, 1] | [2048, 8] |
| [295467, 32] | [512, 1] | [512, 32] |
| [3000000, 32] | [201892, 1] | [201892, 32] |
| [301044, 32] | [512, 1] | [512, 32] |
| [32449, 16] | [1024, 1] | [1024, 16] |
| [4635, 8] | [2048, 1] | [2048, 8] |
| [6000000, 32] | [342312, 1] | [342312, 32] |
| [634054, 32] | [512, 1] | [512, 32] |
| [875000, 8] | [22704, 1] | [22704, 8] |
| [9153, 16] | [1024, 1] | [1024, 16] |
| [934708, 64] | [256, 1] | [256, 64] |

## 详细实现

- 详见 [scatter_nd_sub.py](./scatter_nd_sub.py)
