# gmm_mxfp8 算子说明

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

---

## 算子语义

`gmm_mxfp8` 算子实现了基于 MXFP8 量化的分组矩阵乘法（Grouped MatMul），对应 MoE 场景中多 expert 权重并行计算。每组使用不同的权重矩阵，通过 `group_list` 指定每组的 token 数量。

### 数学公式

对于每个 expert i（i ∈ [0, num_groups-1]），计算公式为：

$$
out[\text{begin}_i:\text{end}_i, :] = \text{ScaledMatmul}(x_i, w_i, \text{scale}_x_i, \text{scale}_w_i)
$$

其中：
- `x_i = a[begin_i:end_i, :]`：第 i 组的输入 token
- `w_i = b[i]`：第 i 个 expert 的权重
- `scale_x_i`：输入 scale 因子
- `scale_w_i`：权重 scale 因子
- `begin_i = Σ(group_list[0:i])`，`end_i = begin_i + group_list[i]`

---

## 输入输出规格

### 输入张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `a` | `[M, K]` | FP8 E4M3 | 输入 token |
| `b` | `[num_groups, K, N]` 或 `[num_groups, N, K]` | FP8 E4M3 | expert 权重（布局由 `b_trans` 决定） |
| `scaled_a` | `[M, K//64, 2]` | E8M0FNU | 输入 scale 因子 |
| `scaled_b` | `[num_groups, K//64, N, 2]` 或 `[num_groups, N, K//64, 2]` | E8M0FNU | 权重 scale 因子 |
| `group_list` | `[num_groups]` | int | 每组 token 数量 |

### 输出张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `out` | `[M, N]` | FP32 | 分组矩阵乘法结果 |

---

## 约束条件

1. **K 必须是 64 的倍数**：MXFP8 量化要求 `K % 64 == 0`
2. **group_list 合法**：`sum(group_list) == M`
3. **b_trans 决定权重布局**：`b_trans=False` 时 `b` 形状为 `[E, K, N]`；`b_trans=True` 时为 `[E, N, K]`

---

## 实现特点

1. **MXFP8 量化**：输入和权重使用 FP8E4M3FN 格式，缩放因子使用 FP8E8M0 格式
2. **逐组循环计算**：按 expert 顺序循环，每组调用 `pypto.scaled_mm` 计算 FP32 输出
3. **分块配置显式化**：使用 `set_cube_tile_shapes` 和 `set_vec_tile_shapes` 控制 cube/vector tile
4. **Host 侧输出初始化**：输出 tensor 在 host 侧创建并移至 NPU，kernel 内逐组写入

---

## 精度验证

### 容差设置

- **相对容差 (RTOL)**：1e-3
- **绝对容差 (ATOL)**：1e-3

### 验证方法

1. **Golden 实现**：`tests/ops/experimental/matmul/gmm_mxfp8/gmm_mxfp8_golden.py` 中 `gen_golden`
2. **PyPTO 实现**：`gmm_mxfp8_impl.py` 中 `gen_mxfp8` 调用 `scaled_matmul_kernel`
3. **对比工具**：`numpy.testing.assert_allclose`

---

## 测试用例

| 测试名称 | M | K | N | group_list | b_trans | 说明 |
|---------|---|---|---|------------|---------|------|
| testcase | 16 | 512 | 7168 | [7, 9] | False | 基础验证 |

运行方式：

```bash
python3 tests/ops/experimental/matmul/gmm_mxfp8/test_gmm_mxfp8.py
```
