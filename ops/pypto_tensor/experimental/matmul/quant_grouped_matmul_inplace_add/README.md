# quant_grouped_matmul_inplace_add 算子说明

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

---

## 算子语义

`quant_grouped_matmul_inplace_add` 算子实现了基于MXFP8量化的分组矩阵乘法就地加法操作，支持多分组场景下的高效梯度累积计算。将GroupedMatMul和InplaceAdd融合，使用MXFP8量化方式，支持按K轴分组进行矩阵乘法计算并就地累加到输出张量。

### 数学公式

对于每个分组 i（i ∈ [0, num_groups-1]），计算公式为：

$$
y_i = y_i + \left(\left(a_{k_i} \times scale_{a_i}\right) @ \left(b_{k_i} \times scale_{b_i}\right)\right)
$$

其中：
- `k_i` 为第 i 个分组的 K 轴块，范围 `[i×k_block, (i+1)×k_block]`，`k_block = K / num_groups`
- `a_{k_i}` 为左矩阵的第 i 个 K 轴切片：`a[begin:end, :]`，形状 `[k_block, M]`
- `b_{k_i}` 为右矩阵的第 i 个 K 轴切片：`b[begin:end, :]`，形状 `[k_block, N]`
- `scale_{a_i}` 和 `scale_{b_i}` 为对应的 MXFP8 量化缩放因子
- `@` 表示矩阵乘法
- 输出 `y` 形状为 `[num_groups, M, N]`，每个分组对应一个输出矩阵

---

## 核心参数

本算子主要使用以下 4 个核心参数：

| 参数名 | 说明 | 示例值 |
|--------|------|--------|
| **num_groups** | K 轴分组数量 | 32 |
| **M** | 左矩阵的列维度（内轴） | 768 |
| **K** | 矩阵乘的公共维度（切分轴） | 6144 |
| **N** | 右矩阵的列维度（内轴） | 4096 |

**参数关系**：
- 输入矩阵 a：形状 `[K, M]`（左矩阵转置）
- 输入矩阵 b：形状 `[K, N]`（右矩阵不转置）
- 输出矩阵 y：形状 `[num_groups, M, N]`（每个分组一个输出）

---

## 输入输出规格

### 输入参数

| 参数名 | 类型 | 形状 | 描述 |
|--------|------|------|------|
| a | Tensor | `[K, M]` | 左矩阵（FP8E4M3 或 FP8E5M2） |
| b | Tensor | `[K, N]` | 右矩阵（FP8E4M3 或 FP8E5M2） |
| scaled_a | Tensor | `[(K//64)+num_groups, M, 2]` | 左矩阵缩放因子（FP8E8M0） |
| scaled_b | Tensor | `[(K//64)+num_groups, N, 2]` | 右矩阵缩放因子（FP8E8M0） |
| y | Tensor | `[num_groups, M, N]` | 输出矩阵（FP32，同时作为输入初始值） |

### 输出参数

| 参数名 | 类型 | 形状 | 描述 |
|--------|------|------|------|
| y | Tensor | `[num_groups, M, N]` | 累加后的输出矩阵（FP32） |

---

## 约束条件

### 关键约束

**⚠️ 重要：K 轴必须满足双重对齐条件**

```
K 必须是 64 × num_groups 的倍数
```

**约束说明**：
1. **MX 量化约束**：K 轴必须 64 对齐（每 64 个元素共享一个缩放因子）
2. **均匀切分约束**：K 轴必须能被 `num_groups` 整除（均匀分组）
3. **综合约束**：`K = 64 × num_groups × k_block_count`（例如：6144 = 64 × 32 × 3）

### 约束检查清单

调用前必须验证：

1. **K 轴双重对齐**：`K % (64 × num_groups) == 0`
2. **内轴对齐**：`M % 32 == 0` 且 `N % 32 == 0`
3. **矩阵格式**：左矩阵转置（`a_trans=True`），右矩阵不转置（`b_trans=False`）
4. **Scale 形状**：符合 MXFP8 格式要求
5. **输出初始化**：y 需要提供初始值（累加起点）

---

## 实现特点

该算子的核心特性：

1. **MXFP8 量化**：数据支持 FP8E4M3FN 和 FP8E5M2 格式，缩放因子使用 FP8E8M0 格式
2. **K 轴分组**：K 轴按 `num_groups` 均匀切分，每个分组对应不同的权重矩阵块
3. **就地加法**：输出张量同时作为输入，执行就地加法操作 `y = y + result`，减少内存拷贝
4. **融合计算**：通过 `scaled_mm` 算子实现缩放和矩阵乘法的融合计算
5. **新前端写法**：使用 PyPTO 新前端 API，支持类型注解和自动类型推导

---

## MXFP8量化说明

MX 量化（Microscaling Quantization）是一种基于块缩放的量化格式：

- **量化块大小**：每 64 个元素（K 轴）共享一个缩放因子
- **缩放因子格式**：FP8E8M0（8位纯指数），仅包含指数部分，隐含 mantissa=1.0
- **数据格式**：FP8E4M3FN（4位指数+3位尾数）或 FP8E5M2（5位指数+2位尾数）
- **Scale 存储格式**：连续存储，分组间有 1 元素间隔，总长度 `(K//64) + num_groups`

---

## 调用示例

### Case1: 标准配置

```python
test_quant_grouped_matmul_inplace_add(
    ShapeConfig(
        ori_shape=[768, 6144, 4096],  # [M, K, N]
        num_groups=32,                 # 分组数
        m_tile_shape=[128, 128],
        k_tile_shape=[64, 192],
        n_tile_shape=[256, 1024],
        vector_tile_shape=[1, 32, 512, 2],
        in_dtype=pypto.DT_FP8E4M3,
        a_trans=True,                  # 左矩阵转置
        b_trans=False,                 # 右矩阵不转置
        description="Case1"
    )
)
```

**参数验证**：
- K = 6144，num_groups = 32
- K % (64 × num_groups) = 6144 % 2048 = 0 ✅（满足双重对齐）
- k_block = 6144 / 32 = 192（每个分组的 K 轴大小）
- M = 768，N = 4096（均为 32 的倍数，满足内轴对齐）

---

## 性能数据

### Case1 性能

| 指标 | 基准性能 | 优化后性能 | 提升 |
|------|---------|-----------|------|
| 执行时间 | 1709 us | 1342 us | 21.5% |
| 核心利用率 | 32.58% | 52.22% | +19.64% |
| 任务数量 | 7746 | 2370 | -69.4% |

**优化配置**：
- NBuffer 配置：`cube_nbuffer_setting={-1:4}`, `vec_nbuffer_setting={-2:1,-1:4}`
- TileShape 优化：`k_tile_shape=[64,192]`, `n_tile_shape=[256,1024]`

---

## 精度验证

### 容差设置

- **相对容差 (RTOL)**：0.01（MXFP8量化精度）
- **绝对容差 (ATOL)**：0.001

### 验证方法

1. **Golden 实现**：纯 PyTorch 实现，作为精度基准
2. **三态标记**：`[PRECISION_PASS]` 或 `[PRECISION_FAIL]`
3. **对比工具**：`numpy.testing.assert_allclose`

---

## 测试用例

| 测试名称 | M | K | N | num_groups | 说明 |
|---------|---|---|----|-----------|------|
| test_极小规模 | 32 | 64 | 32 | 1 | 极小规模验证 |
| test_小规模 | 128 | 128 | 128 | 2 | 小规模验证 |
| test_中等规模 | 768 | 6144 | 4096 | 32 | 标准场景验证 |
| test_大规模 | 2048 | 16384 | 8192 | 64 | 大规模验证 |

---

## 常见问题

### Q1: K 轴为什么需要双重对齐？

**答**：
- MX 量化要求每 64 个元素共享一个缩放因子，因此 K 必须是 64 的倍数
- 均匀分组要求每个分组大小相同，因此 K 必须能被 num_groups 整除
- 综合约束：K = 64 × num_groups × k_block_count

### Q2: 为什么左矩阵转置，右矩阵不转置？

**答**：
- 左矩阵转置（`a_trans=True`）：矩阵形状 `[K, M]`，便于 K 轴切分（切分第一维）
- 右矩阵不转置（`b_trans=False`）：矩阵形状 `[K, N]`，便于 K 轴切分（切分第一维）
- 这样两个矩阵都在第一维切分，实现均匀分组

### Q3: Scale tensor 的形状为什么有 num_groups 的额外维度？

**答**：
- MXFP8 格式要求分组间有 1 元素间隔
- Scale tensor 总长度 = `(K//64) + num_groups`
- 每个分组的 scale 偏移 = `begin // 64 + i`（begin 为 K 轴起始位置，i 为分组索引）

---

## 参考文档

- **SPEC.md** - 详细需求规格
- **API_REPORT.md** - API映射分析
- **DESIGN.md** - 详细设计文档
- [PyPTO API 文档](../../../docs/api/)
- [MXFP8 量化规范](../../../docs/tutorials/quantization/)
- [性能优化指南](https://gitcode.com/cann/pypto/blob/master/docs/zh/guide/programming_guide/tensor/debug/performance.md)

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2026-04-29 | 初始版本，支持 MXFP8 分组矩阵乘 |
| v1.1 | 2026-04-29 | 性能优化，提升 21.5%，优化 NBuffer 和 TileShape 配置 |