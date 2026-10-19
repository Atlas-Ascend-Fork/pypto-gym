# Matrix Multiplication Operators

本目录包含 PyPTO 矩阵乘法相关的实验性算子实现，包括分组矩阵乘法和量化矩阵乘法。

---


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## gmm_mxfp8

### 功能说明

`gmm_mxfp8` 算子实现了分组矩阵乘法与 MXFP8 量化，主要用于大规模推理场景中的高效矩阵运算。

该算子的核心特性：

1. **分组计算**：支持将输入矩阵按不同分组使用不同的权重矩阵进行计算
2. **MXFP8 量化**：使用 MXFP8 格式进行量化，数据使用 E4M3FN 格式，缩放因子使用 E8M0FNU 格式
3. **高效融合**：通过 `scaled_mm` 算子实现缩放和矩阵乘法的融合计算

**说明：**
<blockquote>MXFP8（Microscaling FP8）是一种基于块缩放的量化格式。每 64 个元素共享一个缩放因子，缩放因子仅包含指数部分（E8M0FNU格式），数据部分为 8 位浮点数（E4M3FN格式），适用于推理场景的高效量化计算。</blockquote>

### 计算公式

对于输入矩阵 $A$ 和权重矩阵组 $\{W_1, W_2, ..., W_g\}$：

$$
\text{Output} = \text{Concat}(A_1 \cdot W_1, A_2 \cdot W_2, ..., A_g \cdot W_g)
$$

其中 $A_i$ 是输入矩阵的第 $i$ 个分组。MXFP8 量化后的矩阵乘法为：

$$
\text{ScaledMatmul}(A, W, S_A, S_W) = (A \otimes S_A) \cdot (W \otimes S_W)
$$

其中 $S_A$ 和 $S_W$ 分别为输入和权重的缩放因子，$\otimes$ 表示广播乘法。

### 函数原型

```python
def scaled_matmul_kernel(
    a: pypto.Tensor,
    b: pypto.Tensor,
    scaled_a: pypto.Tensor,
    scaled_b: pypto.Tensor,
    out: pypto.Tensor,
    group_list: list,
    tile_config: ShapeConfig
) -> None
```

### 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) |
|--------|-----------|------|----------|----------|----------|-------------|
| a | 输入 | 输入矩阵 | - | float8_e4m3fn | ND | [M, K] |
| b | 输入 | 权重矩阵组 | 根据 b_trans 决定形状 | float8_e4m3fn | ND | [num_groups, K, N] 或 [num_groups, N, K] |
| scaled_a | 输入 | 输入缩放因子 | 每 64 个元素共享一个缩放因子 | float8_e8m0fnu | ND | [M, K//64, 2] |
| scaled_b | 输入 | 权重缩放因子 | 每 64 个元素共享一个缩放因子 | float8_e8m0fnu | ND | [num_groups, K//64, N, 2] 或 [num_groups, N, K//64, 2] |
| out | 输出 | 输出矩阵 | 分组计算结果拼接 | float32 | ND | [M, N] |

### 调用示例

- 详见 [gmm_mxfp8](./gmm_mxfp8/)

---

## quant_matmul_reduce_sum

### 功能说明

`quant_matmul_reduce_sum` 算子实现了量化矩阵乘法与归约求和，主要用于量化推理场景中的批处理矩阵乘法及累加操作。

该算子的核心特性：

1. **INT8 量化**：输入数据使用 INT8 格式，配合 FP32/BF16 缩放因子进行反量化
2. **批处理矩阵乘法**：支持多 batch 的矩阵乘法运算
3. **Reduce Sum**：对多个 batch 的矩阵乘法结果沿 batch 维度求和
4. **格式支持**：支持 ND 和 NZ 两种数据格式

### 计算公式

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

### 函数原型

```python
def quant_matmul_reduce_sum_impl(
    x1: pypto.Tensor,
    x2: pypto.Tensor,
    x1_scale: pypto.Tensor,
    x2_scale: pypto.Tensor
) -> pypto.Tensor
```

### 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) |
|--------|-----------|------|----------|----------|----------|-------------|
| x1 | 输入 | 输入矩阵 1 | INT8 量化数据 | int8 | ND 或 NZ | [batch, M, K] |
| x2 | 输入 | 输入矩阵 2 | INT8 量化数据，支持 ND 或 NZ 格式 | int8 | ND 或 NZ | [batch, K, N] |
| x1_scale | 输入 | X1 的缩放因子 | 用于反量化 | float32 | ND | [batch, M] |
| x2_scale | 输入 | X2 的缩放因子 | 用于反量化 | bfloat16 | ND | [N] |
| 输出 | 输出 | 输出矩阵 | 归约求和后的结果 | bfloat16 | ND | [M, N] |

### 调用示例

- 详见 [quant_matmul_reduce_sum](./quant_matmul_reduce_sum/)
