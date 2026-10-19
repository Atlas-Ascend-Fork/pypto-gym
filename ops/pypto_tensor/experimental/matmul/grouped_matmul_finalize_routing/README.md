# grouped_matmul_finalize_routing 算子说明

## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 算子语义

`grouped_matmul_finalize_routing` 是 MoE 场景中的 grouped matmul 后处理融合算子，对应
`aclnnGroupedMatmulFinalizeRoutingV3` 的 MXFP8 路径。该算子将路由后的 token 按 expert 分组执行矩阵乘，并完成 logit 加权、`row_index` 回写累加和 shared expert 叠加。

### 数学公式

```
mm_i = ScaledMatmul(x1_i, x2_i, pertoken_scale_i, scale)
weighted_i = mm_i * logit_i
out[row_index_i] += weighted_i
out[shared_input_offset:shared_input_offset+batch] += shared_input * shared_input_weight
```

**展开形式**：

```
out[row_index[t], n] += logit[t] * Σ(k=0..K-1) dequant(x1[t, k]) * dequant(x2[expert(t), k, n])
```

其中 `dequant` 由 MXFP8 输入值和 E8M0FNU scale 共同决定。

### 计算流程

1. **Grouped Matmul 计算**：按 expert 切分 token，调用 `pypto.scaled_mm` 计算 FP32 输出。
   - `x1_i: [M_i, K]`
   - `x2_i: [K, N]` 或 `[N, K]`
   - 输出：`mm_i: [M_i, N]`

2. **Logit 加权**：当 `has_logit=True` 时，对每个 token 的 matmul 结果乘以对应 `logit`。
   - `logit_i: [M_i]` → unsqueeze → `[M_i, 1]`
   - 广播乘法：`[M_i, N] × [M_i, 1] → [M_i, N]`

3. **Finalize Routing 回写**：根据 `row_index` 将 expert 输出累加到最终输出。
   - `row_index_i: [M_i]`
   - `out[row_index_i] += weighted_i`

4. **Shared Expert 叠加**：当 `has_shared_input=True` 时，将 shared expert 输出按权重叠加到 `out`。
   - `shared_input: [batch, N]`
   - `out[offset:offset+batch] += shared_input * shared_input_weight`

---

## 输入输出规格

### 输入张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `x1` | `[M, K]` | FP8 E4M3/E5M2 | 路由后 token 输入 |
| `x2` | `[E, K, N]` 或 `[E, N, K]` | FP8 E4M3/E5M2 | expert 权重，布局由 `transpose_x2` 决定 |
| `scale` | `[ceil(K/64), N, 2]` 或 `[N, ceil(K/64), 2]` | E8M0FNU | 权重 scale |
| `pertoken_scale` | `[M, ceil(K/64), 2]` | E8M0FNU | token scale |
| `group_list` | `[E]` | int64 | expert 分组信息 |
| `shared_input` | `[batch, N]` | bfloat16 | shared expert 输出 |
| `logit` | `[M]` | float32 | token 对应 expert 权重 |
| `row_index` | `[M]` | int64 | 输出行索引 |
| `out` | `[batch, N]` | float32 | 输出初值 |

### 输出张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `output` | `[batch, N]` | float32 | finalize routing 后的融合结果 |

---

## Shape 范围与约束

### 动态轴

| 轴 | 当前覆盖范围 | 说明 |
|----|--------------|------|
| batch | {64, 128, 256} | 输出行数 |
| M | {128, 256, 768} | 路由后 token 数 |
| K | {5120, 6144, 7168, 8192} | matmul K 维 |
| N | 4096 | 输出列数 |
| E | {8, 16, 32} | expert 数量 |

### 约束条件

1. **transpose_x1 仅支持 False**：当前目标路径不支持转置 `x1`。
2. **group_list 当前测试为均匀分组**：kernel 内按 `M // E` 切分 token。
3. **row_index 范围合法**：`row_index` 中元素必须位于 `[0, batch)`。
4. **shared_input 边界合法**：`shared_input_offset + shared_input.shape[0] <= out.shape[0]`。
5. **MXFP8 scale 布局固定**：K 维按 `ceil(K/64)` 分块，每个 block 包含 2 个 E8M0FNU scale。

---

## 实现特点

### 性能优化

1. **Cube + Vector 融合**：`scaled_mm` 使用 cube 计算，logit 和 scatter-add 使用 vector 路径。
2. **Expert 并行**：通过 `pypto.loop(config.num_experts, parallel=True)` 按 expert 并行执行。
3. **分块配置显式化**：使用 `set_cube_tile_shapes` 和 `set_vec_tile_shapes` 控制 cube/vector tile。
4. **Host 侧 shared input 预处理**：kernel 内聚焦 grouped matmul、logit 和 row_index accumulate。

### 内存访问模式

- `x1` 按 expert token 范围连续切片。
- `x2` 按 expert 维度读取单个 expert 权重。
- `out` 通过 `row_index` 执行非连续 scatter-add。
- shared input 在 host 侧加到 `out` 初值，避免 kernel 内额外分支。

---

## 精度验证

### 容差设置

- **相对容差 (RTOL)**：0.001
- **绝对容差 (ATOL)**：0.001

### 测试用例

| 测试名称 | batch | M | K | N | E | DType | 说明 |
|---------|-------|---|---|---|---|-------|------|
| case1 | 128 | 768 | 6144 | 4096 | 32 | FP8 E4M3 | 大 K、32 experts |
| case2 | 256 | 768 | 8192 | 4096 | 32 | FP8 E4M3 | 更大 K、batch=256 |
| case3 | 64 | 128 | 5120 | 4096 | 8 | FP8 E4M3 | 小 M、8 experts |
| case4 | 64 | 256 | 7168 | 4096 | 16 | FP8 E5M2 | FP8 E5M2 路径 |

### 验证方法

1. **Golden 实现**：`tests/ops/experimental/matmul/grouped_matmul_finalize_routing/gmm_finalize_routing_golden.py` 中 `gen_golden`。
2. **PyPTO 实现**：`gmm_finalize_routing_impl.py` 中 `gen_pypto` 调用 `gmm_finalize_routing_kernel`。
3. **对比工具**：`numpy.testing.assert_allclose`。
