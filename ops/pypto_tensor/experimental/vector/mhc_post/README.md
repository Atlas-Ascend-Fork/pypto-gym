# mhc_post 算子说明


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子语义

`mhc_post` 是 **MHC (Manifold-Constrained Hyper-Connections)** 系统的后处理融合算子，用于注意力机制中的流间混合计算。

### 数学公式

```
h_post_term = h_post.unsqueeze(-1) * h_out.unsqueeze(-2)
h_comb_term = torch.sum(h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=-3)
output = (h_post_term + h_comb_term).to(bfloat16)
```

**展开形式**：
```
output[b*s, n, d] = h_post[b*s, n] * h_out[b*s, d] + Σ(k=0..N-1) h_res[b*s, k, n] * x[b*s, k, d]
```

### 计算流程

1. **h_post_term 计算**：逐样本动态权重广播乘法
   - `h_post: [B*S, N]` → unsqueeze → `[B*S, N, 1]`
   - `h_out: [B*S, D]` → unsqueeze → `[B*S, 1, D]`
   - 广播乘法：`[B*S, N, 1] × [B*S, 1, D] → [B*S, N, D]`

2. **h_comb_term 计算**：逐样本流间混合加权求和
   - `h_res: [B*S, N, N]` → unsqueeze → `[B*S, N, N, 1]`
   - `x: [B*S, N, D]` → unsqueeze → `[B*S, N, 1, D]`
   - 广播乘法：`[B*S, N, N, 1] × [B*S, N, 1, D] → [B*S, N, N, D]`
   - 沿 dim=-3 求和：`[B*S, N, N, D] → [B*S, N, D]`

3. **融合输出**：逐元素相加并转换回 BF16
   - `result = h_post_term + h_comb_term`
   - 转换回 `bfloat16` 精度

---

## 输入输出规格

### 输入张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `x` | `[B, S, N, D]` 或 `[B*S, N, D]` | bfloat16 | 输入 tensor |
| `h_res` | `[B, S, N, N]` 或 `[B*S, N, N]` | float32 | 流间混合权重矩阵 |
| `h_out` | `[B, S, D]` 或 `[B*S, D]` | bfloat16 | 输出项数据 |
| `h_post` | `[B, S, N]` 或 `[B*S, N]` | float32 | 后处理权重 |

### 输出张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `output` | `[B, S, N, D]` 或 `[B*S, N, D]` | bfloat16 | 融合计算结果 |

---

## Shape 范围与约束

### 动态轴

| 轴 | 范围 | 说明 |
|----|------|------|
| B*S | {1024, 2048, 4096} | 动态维度，批大小 × 序列长度 |
| N | 4 | **固定值**，注意力流数量 |
| D | {2560, 5120} | 静态轴，隐藏层维度（变化时触发重编译） |

### 约束条件

1. **N 固定为 4**：注意力流数量不可变
2. **D 为 STATIC 标记**：D 维度变化会触发 kernel 重编译
3. **B*S 为 DYNAMIC**：支持动态 shape，无需重编译
4. **内存连续性**：所有输入 tensor 必须是 contiguous 的
5. **精度约束**：
   - BF16 输入在计算前转为 FP32
   - FP32 中间结果最后转回 BF16 输出
   - sigmoid 和 sum 操作仅支持 FP32

---

## 实现特点

### 性能优化

1. **纯 Vector 算子**：无矩阵乘法操作，纯向量运算
2. **循环展开**：对 BS 轴使用 `pypto.loop_unroll` 进行展开优化（unroll_list=[128]）
3. **向量化分块**：使用 `pypto.set_vec_tile_shapes` 进行细粒度分块
   - 分块大小：`(1, N, 1, 1280)` 或 `(1, N, 1280)`
   - 优化内存访问模式和向量化效率
4. **精度优化**：BF16 → FP32 → BF16 的精度转换路径

### 内存访问模式

- **输入 reshape**：wrapper 将 `[B, S, ...]` reshape 为 `[B*S, ...]`
- **输出 reshape**：计算完成后 reshape 回原始格式 `[B, S, N, D]`
- **原地操作**：使用 `inplace=True` 减少内存拷贝

---

## 精度验证

### 容差设置

- **相对容差 (RTOL)**：0.0078125 (1/128)
- **绝对容差 (ATOL)**：0.0001

### 测试用例

| 测试名称 | B*S | N | D | 说明 |
|---------|-----|---|----|----|
| `test_mhc_post_bs8_n4_d128` | 8 | 4 | 128 | 极小规模验证 |
| `test_mhc_post_bs256_n4_d128` | 256 | 4 | 128 | 小规模验证 |
| `test_mhc_post_bs1024_n4_d5120` | 1024 | 4 | 5120 | 基础验证（大 D） |
| `test_mhc_post_bs4096_n4_d2560` | 4096 | 4 | 2560 | 大规模验证 |

### 验证方法

1. **Golden 实现**：`mhc_post_golden.py` 提供纯 PyTorch 参考实现
2. **三态标记**：`[PRECISION_PASS]` 或 `[PRECISION_FAIL]`
3. **对比工具**：`numpy.testing.assert_allclose`
