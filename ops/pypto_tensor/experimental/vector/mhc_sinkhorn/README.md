# mhc_sinkhorn 算子说明


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子语义

`mhc_sinkhorn` 实现 **Sinkhorn-Knopp 双随机矩阵迭代归一化算法**，用于将矩阵通过交替行列归一化迭代转换为双随机矩阵（每行每列元素和均为 1 的矩阵）。

该算子是 **MHC (Manifold-Constrained Hyper-Connections)** 系统中的残差连接矩阵约束算子，用于确保流间混合权重矩阵满足双随机性约束。

### 核心概念

**双随机矩阵 (Doubly Stochastic Matrix)**：
- 每行元素之和 = 1
- 每列元素之和 = 1
- 所有元素为正数（因为 softmax 输出）

**Sinkhorn-Knopp 算法**：
通过交替对矩阵进行行归一化和列归一化，迭代收敛到双随机矩阵。

### 计算流程

```
输入 x [B*S, N, N]
  ↓
Step 1: softmax(dim=-1) + eps  （行归一化初始）
  ↓
Step 2: 列归一化 (sum dim=-2 + eps)
  ↓
Step 3: 循环 num_iters-1 次
  ├─ 行归一化 (sum dim=-1 + eps)
  └─ 列归一化 (sum dim=-2 + eps)
  ↓
输出 [B*S, N, N] (双随机矩阵)
```

### 数学公式

**Step 1: 初始行归一化**
```
h_comb = softmax(x, dim=-1) + eps
```

**Step 2: 初始列归一化**
```
col_sum = sum(h_comb, dim=-2, keepdim=True)
h_comb = h_comb / (col_sum + eps)
```

**Step 3: 交替归一化（迭代 num_iters-1 次）**
```
for _ in range(num_iters - 1):
    # 行归一化
    row_sum = sum(h_comb, dim=-1, keepdim=True)
    h_comb = h_comb / (row_sum + eps)
    
    # 列归一化
    col_sum = sum(h_comb, dim=-2, keepdim=True)
    h_comb = h_comb / (col_sum + eps)
```

---

## 输入输出规格

### 输入张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `x` | `[B*S, N, N]` | float32 | 输入矩阵 tensor |

### 输出张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `output` | `[B*S, N, N]` | float32 | 双随机矩阵（行列和均为 1） |

### 参数

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `eps` | float | 1e-6 | 数值稳定性参数，防止除零 |
| `num_iters` | int | 20 | Sinkhorn 迭代次数 |

---

## Shape 范围与约束

### 动态轴与静态轴

| 轴 | 范围 | 标记 | 说明 |
|----|------|------|------|
| B*S | {8, 64, 1024, 2048, 4096} | `DYNAMIC` | 动态轴，批大小 × 序列长度，无需重编译 |
| N | 4 或 8 | `STATIC` | 矩阵维度（当前实现使用 N=4，golden 描述中使用 N=8） |

### Shape 约束

**注意**：代码中存在版本差异：
- **Golden 文档描述**：`N = 8`（典型配置）
- **实际实现**：`N = 4`（kernel 和 test 使用）

当前测试和实现使用 `N = 4`，但算法本身支持任意 N 值。

### 约束条件

1. **矩阵维度对称**：输入为 `[B*S, N, N]`，必须是方阵
2. **B*S 为 DYNAMIC**：支持动态 shape，无需重编译
3. **N 为 STATIC**：矩阵维度变化会触发 kernel 重编译
4. **精度约束**：
   - 输入和输出均为 FP32
   - softmax 操作仅支持 FP32
   - 迭代过程保持 FP32 精度
5. **数值稳定性**：
   - 所有除法操作添加 `eps` 防止除零
   - softmax 后添加 `eps` 避免输出为零
6. **收敛性**：
   - 默认迭代 20 次可达到双随机矩阵性质
   - 输出矩阵每行和每列和均近似为 1（误差 < 1e-3）

---

## 实现特点

### 性能优化

1. **核内优化（减少中间 tensor）**：
   ```python
   # 合并 softmax + add
   h = pypto.add(pypto.softmax(h, dim=-1), eps)
   
   # 合并 sum + add + div（列归一化）
   h = pypto.div(h, pypto.add(pypto.sum(h, dim=-2, keepdim=True), eps))
   
   # 合并 sum + add + div（行归一化）
   h = pypto.div(h, pypto.add(pypto.sum(h, dim=-1, keepdim=True), eps))
   ```

2. **Loop Unroll 优化**：
   - 对 B*S 轴使用 `pypto.loop_unroll`
   - `unroll_list=[1, 8]`，分块大小 `s=32`
   - 处理 batch 分块：`(t+s-1)//s` 个分块

3. **Transpose + Reshape 优化**：
   - 输入 reshape 为 `[t, N*N]` → transpose 为 `[N*N, t]` → reshape 为 `[N, N, t]`
   - 优化内存访问模式，提高向量计算效率
   - 输出反向操作：reshape → transpose → reshape

4. **Tile Shape 设置**：
   - Vector 计算：`set_vec_tile_shapes(256, 16)` 或 `(4, 4, 256)`
   - 自适应调整：根据操作类型设置不同的 tile shape

5. **Softmax 手动实现**：
   ```python
   # 手动实现 softmax（替代 pypto.softmax）
   row_max = pypto.amax(comb_flag, 1, True)  # 沿行求最大值
   comb_flag = pypto.exp(comb_flag - row_max)  # 指数化（数值稳定性）
   row_sum = pypto.sum(comb_flag, 1, True)  # 求和
   comb_flag = comb_flag / row_sum + eps  # 归一化 + eps
   ```

### 计算流程详解（Kernel 内部）

**预处理（reshape + transpose）**：
```python
# 输入 [t, N, N] → [t, N*N]
x = pypto.reshape(x, [t, N*N], inplace=True)

# [t, N*N] → [N*N, t]
comb_flag = pypto.transpose(comb_flag, 1, 0)

# [N*N, t] → [N, N, t]（便于行列归一化）
comb_flag = pypto.reshape(comb_flag, [N, N, tile_t], inplace=True)
```

**初始归一化**：
```python
# Softmax（手动实现）
row_max = pypto.amax(comb_flag, 1, True)
comb_flag = pypto.exp(comb_flag - row_max)
row_sum = pypto.sum(comb_flag, 1, True)
comb_flag = comb_flag / row_sum + eps

# 列归一化
col_sum = pypto.sum(comb_flag, 0, True)
comb_flag = comb_flag / (col_sum + eps)
```

**迭代归一化**：
```python
for _ in range(num_iters - 1):
    # 行归一化
    row_sum = comb_flag.sum(1, keepdim=True)
    comb_flag = comb_flag / row_sum + eps
    
    # 列归一化
    col_sum = comb_flag.sum(0, keepdim=True)
    comb_flag = comb_flag / (col_sum + eps)
```

**后处理（transpose + reshape）**：
```python
# [N, N, t] → [N*N, t]
comb_flag = pypto.reshape(comb_flag, [N*N, tile_t], inplace=True)

# [N*N, t] → [t, N*N]
comb_flag = pypto.transpose(comb_flag, 1, 0)

# [t, N*N] → [t, N, N]
out[:] = pypto.reshape(tmp, [t, N, N], inplace=True)
```

---

## 精度验证

### 容差设置

- **相对容差 (RTOL)**：0.0078125 (1/128)
- **绝对容差 (ATOL)**：0.0001

### 测试用例

| 测试名称 | B*S | N | N_out | 说明 |
|---------|-----|---|-------|------|
| `test_mhc_sinkhorn_bs8_n4_n4` | 8 | 4 | 4 | 极小规模验证（Level 0） |
| `test_mhc_sinkhorn_bs64_n4_n4` | 64 | 4 | 4 | 小规模验证（Level 0） |
| `test_mhc_sinkhorn_bs1024_n4_n4` | 1024 | 4 | 4 | 中等规模验证（Level 0） |
| `test_mhc_sinkhorn_bs2048_n4_n4` | 2048 | 4 | 4 | 典型场景验证（Level 1） |
| `test_mhc_sinkhorn_bs4096_n4_n4` | 4096 | 4 | 4 | 大规模验证（Level 2） |

### 验证方法

1. **Golden 实现**：`mhc_sinkhorn_golden.py` 提供纯 PyTorch 参考实现
2. **三态标记**：`[PRECISION_PASS]` 或 `[PRECISION_FAIL]`
3. **对比工具**：`numpy.testing.assert_allclose`
4. **特殊验证项**：
   - 双随机矩阵性质验证（行列和近似为 1）
   - 输出全为正数（softmax + eps）
   - 数值稳定性（无 NaN/Inf）
   - 大值和小值输入测试