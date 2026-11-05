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
