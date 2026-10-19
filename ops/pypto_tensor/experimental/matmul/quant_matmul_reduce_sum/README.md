# quant_matmul_reduce_sum 算子说明

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

`quant_matmul_reduce_sum` 算子实现了量化矩阵乘法与归约求和，主要用于量化推理场景中的批处理矩阵乘法及累加操作。

该算子的核心特性：

1. **INT8 量化**：输入数据使用 INT8 格式，配合 FP32/BF16 缩放因子进行反量化
2. **批处理矩阵乘法**：支持多 batch 的矩阵乘法运算
3. **Reduce Sum**：对多个 batch 的矩阵乘法结果沿 batch 维度求和
4. **格式支持**：支持 ND 和 NZ 两种数据格式

## 计算公式

对于 INT8 量化的输入矩阵 $X_1$ 和 $X_2$，以及对应的缩放因子 $S_1$ 和 $S_2$：

$$
\text{Matmul}(X_1, X_2) = \sum_{b=0}^{B-1} (X_1[b] \cdot S_1[b]) \cdot (X_2[b] \cdot S_2)
$$

详细计算步骤：
1. 矩阵乘法：$\text{result}_{int32} = X_1 \times X_2$
2. 类型转换：$\text{result}_{fp32} = \text{cast}(\text{result}_{int32}, \text{FP32})$
3. 缩放广播：$S_{1\_broadcast} = \text{expand}(S_1, [B, M, N])$, $S_{2\_broadcast} = \text{expand}(S_2, [M, N])$
4. 缩放乘法：$\text{scaled} = \text{result}_{fp32} \times S_{1\_broadcast} \times S_{2\_broadcast}$
5. 归约求和：$\text{output} = \sum_{b=0}^{B-1} \text{scaled}[b]$

## 函数原型

```python
def quant_matmul_reduce_sum_impl(
    x1: pypto.Tensor,
    x2: pypto.Tensor,
    x1_scale: pypto.Tensor,
    x2_scale: pypto.Tensor
) -> pypto.Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) |
|--------|-----------|------|----------|----------|----------|-------------|
| x1 | 输入 | 输入矩阵 1 | INT8 量化数据 | int8 | ND 或 NZ | [batch, M, K] |
| x2 | 输入 | 输入矩阵 2 | INT8 量化数据，支持 ND 或 NZ 格式 | int8 | ND 或 NZ | [batch, K, N] |
| x1_scale | 输入 | X1 的缩放因子 | 用于反量化 | float32 | ND | [batch, M] |
| x2_scale | 输入 | X2 的缩放因子 | 用于反量化 | bfloat16 | ND | [N] |
| 输出 | 输出 | 输出矩阵 | 归约求和后的结果 | bfloat16 | ND | [M, N] |
