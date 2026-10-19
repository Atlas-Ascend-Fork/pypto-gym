# transpose_quant_batch_matmul 算子说明

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

---

## 算子语义

`transpose_quant_batch_matmul` 算子实现了基于MXFP8量化的批量矩阵乘法（带转置）操作，支持多种perm组合（permX1, permX2, permY）和输出dtype（FP16/BF16）。M轴动态化——同一编译kernel可支持不同M大小的输入。

### 数学公式

对于每个batch索引 b（b ∈ [0, B-1]），计算公式为：

$$
out[:, b, :] = permY\left(permX1(x1)[:, b, :] \times permX2(x2)[b, :, :]\right)
$$

其中：
- `x1` 形状 `[M, B, K]`，FP8格式，M轴动态
- `x2` 形状 `[B, K, N]`（permX2=[0,1,2]）或 `[B, N, K]`（permX2=[0,2,1]）
- `permX1=[1,0,2]`：x1从 `[M,B,K]` 变为 `[B,M,K]`
- `permX2=[0,1,2]` 或 `[0,2,1]`：x2布局决定scaled_mm的b_trans参数
- `permY=[1,0,2]`：输出从 `[B,M,N]` 变为 `[M,B,N]`

---

## 核心参数

本算子主要使用以下 4 个核心参数：

| 参数名 | 说明 | 示例值 |
|--------|------|--------|
| **M** | 左矩阵的行维度（动态） | 8 ~ 32768 |
| **K** | 矩阵乘的公共维度 | 128 |
| **N** | 右矩阵的列维度 | 512 |
| **B** | batch维度（编译期固定） | 128 |

**参数关系**：
- 输入 x1：形状 `[M, B, K]`（FP8，M轴动态）
- 输入 x2：形状 `[B, K, N]` 或 `[B, N, K]`（取决于permX2）
- 输出 out：形状 `[M, B, N]`（FP16 或 BF16）

---

## 输入输出规格

### 输入参数

| 参数名 | 类型 | 形状 | 描述 |
|--------|------|------|------|
| x1 | Tensor | `[M, B, K]` | 左矩阵（FP8E4M3 或 FP8E5M2） |
| x2 | Tensor | `[B, K, N]` 或 `[B, N, K]` | 右矩阵（FP8E4M3 或 FP8E5M2） |
| x1Scale | Tensor | `[M, B, K//64, 2]` | 左矩阵缩放因子（FP8E8M0） |
| x2Scale | Tensor | `[B, K//64, N, 2]` 或 `[B, N, K//64, 2]` | 右矩阵缩放因子（FP8E8M0） |

### 输出参数

| 参数名 | 类型 | 形状 | 描述 |
|--------|------|------|------|
| out | Tensor | `[M, B, N]` | 输出矩阵（FP16 或 BF16） |

---

## 约束条件

### 关键约束

**⚠️ MX量化约束**

```
K 必须是 64 的倍数
```

### 约束检查清单

调用前必须验证：

1. **K 轴对齐**：`K % 64 == 0`
2. **perm组合**：permX2 决定 x2 和 x2Scale 的形状
3. **Scale 形状**：符合 MXFP8 格式要求
4. **输出 dtype**：dtype=1→FP16, dtype=27→BF16
5. **B 轴固定**：batch_size 在编译期固定

---

## 实现特点

该算子的核心特性：

1. **MXFP8 量化**：数据支持 FP8E4M3FN 和 FP8E5M2 格式，缩放因子使用 FP8E8M0 格式
2. **M轴动态化**：同一编译kernel支持不同M大小，运行时通过shape[0]获取
3. **B轴并行**：LOOP_B使用parallel=True，每个batch独立计算
4. **perm组合支持**：通过permX2决定scaled_mm的b_trans参数，硬件内完成转置
5. **输出dtype灵活**：支持FP16和BF16输出
6. **Cache策略**：输入tensor使用NONE_CACHEABLE，减少L1缓存占用

---

## MXFP8量化说明

MX 量化（Microscaling Quantization）是一种基于块缩放的量化格式：

- **量化块大小**：每 64 个元素共享一个缩放因子
- **缩放因子格式**：FP8E8M0（8位纯指数），隐含 mantissa=1.0
- **数据格式**：FP8E4M3FN 或 FP8E5M2
- **x1Scale**：形状 `[M, B, K//64, 2]`
- **x2Scale**：形状取决于permX2

---

## perm组合说明

### permX2=[0, 1, 2]（K,N顺序）

- x2 形状：`[B, K, N]`
- x2Scale 形状：`[B, K//64, N, 2]`
- scaled_mm 参数：无 b_trans
- 计算路径：`[M,K] × [K,N] → [M,N]`

### permX2=[0, 2, 1]（N,K反序）

- x2 形状：`[B, N, K]`
- x2Scale 形状：`[B, N, K//64, 2]`
- scaled_mm 参数：`b_trans=True, scale_b_trans=True`
- 计算路径：`[M,K] × [N,K]^T → [M,N]`

---

## 调用示例

### test1: 小M + b_trans配置

```python
test_transpose_quant_batch_matmul(
    ShapeConfig(
        ori_shape=[8, 128, 512],       # [M, K, N]
        batch_size=128,                 # B轴大小
        m_tile_shape=[256, 256],
        k_tile_shape=[128, 128],
        n_tile_shape=[256, 256],
        vector_tile_shape=[1, 128, 256, 32],
        in_dtype=pypto.DT_FP8E5M2,
        out_dtype=pypto.DT_BF16,
        permX1=[1, 0, 2],
        permX2=[0, 2, 1],              # N,K反序 → b_trans=True
        permY=[1, 0, 2],
        description="test1"
    )
)
```

### test3: 大M + 标准配置

```python
test_transpose_quant_batch_matmul(
    ShapeConfig(
        ori_shape=[8192, 128, 512],     # [M, K, N]
        batch_size=128,
        m_tile_shape=[256, 256],
        k_tile_shape=[128, 128],
        n_tile_shape=[256, 256],
        vector_tile_shape=[1, 128, 256, 32],
        in_dtype=pypto.DT_FP8E4M3,
        out_dtype=pypto.DT_FP16,
        permX1=[1, 0, 2],
        permX2=[0, 1, 2],              # K,N顺序 → 无b_trans
        permY=[1, 0, 2],
        description="test3"
    )
)
```

---

## 性能数据

| 配置 | M | K | N | B | 预估时间 |
|------|---|---|---|----|---------|
| test1 | 8 | 128 | 512 | 128 | ~108 us |
| test2 | 128 | 128 | 512 | 128 | ~122 us |
| test3 | 8192 | 128 | 512 | 128 | ~9583 us |
| test4 | 32768 | 128 | 512 | 128 | ~40635 us |

**优化配置**：
- NBuffer 配置：`cube_nbuffer_setting={-1:2}`, `vec_nbuffer_setting={-2:1,-1:16}`
- Cache策略：`NONE_CACHEABLE` 用于所有输入tensor
- Cube L1复用：`cube_l1_reuse_setting={-1:2}`

---

## 精度验证

### 容差设置

- **相对容差 (RTOL)**：1e-3
- **绝对容差 (ATOL)**：1e-3

### 验证方法

1. **Golden 实现**：纯 PyTorch 实现，作为精度基准
2. **三态标记**：`[PRECISION_PASS]` 或 `[PRECISION_FAIL]`
3. **对比工具**：`numpy.testing.assert_allclose`

---

## 测试用例

| 测试名称 | M | K | N | B | permX2 | 说明 |
|---------|---|---|---|----|--------|------|
| test1 | 8 | 128 | 512 | 128 | [0,2,1] | 小M+b_trans验证 |
| test2 | 128 | 128 | 512 | 128 | [0,1,2] | 中M+标准验证 |
| test3 | 8192 | 128 | 512 | 128 | [0,1,2] | 大M性能验证 |
| test4 | 32768 | 128 | 512 | 128 | [0,1,2] | 超大规模验证 |

---

## 常见问题

### Q1: 为什么permX2=[0,2,1]需要b_trans=True？

**答**：
- permX2=[0,2,1]表示x2的布局是 `[B,N,K]`
- `b_trans=True` 让scaled_mm在硬件内部完成转置
- `scale_b_trans=True` 同时处理scale tensor的转置
- 这样避免了额外的数据搬运操作

### Q2: M轴动态化如何实现？

**答**：
- 编译期不固定M值，同一kernel可处理不同M的输入
- B轴在编译期固定（batch_size参数），通过parallel loop切分
- M轴动态化减少重编译开销，适配不同输入规模

### Q3: 为什么输入tensor使用NONE_CACHEABLE？

**答**：
- MXFP8数据一次性读取，无需缓存复用
- NONE_CACHEABLE减少L1缓存占用，为输出tensor腾出空间
- 对单次读取场景无性能损失

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
| v1.0 | 2026-06-03 | 初始版本，支持MXFP8批量矩阵乘法（带转置） |