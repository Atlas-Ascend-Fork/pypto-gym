# mhc_pre 算子说明


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子语义

`mhc_pre` 是 **MHC (Multi-Head Context)** 系统的前处理算子，用于多头注意力机制中的特征归一化、矩阵变换和三分支分流处理。

### 计算流程

```
输入 x [B*S, N, D] → RMSNorm → MatMul → Split → 三分支处理 → 输出 (h_in, h_post, h_res)
```

**7 个计算步骤**：

1. **Reshape & Float**: `[B*S, N, D] BF16 → [B*S, N*D] FP32`
2. **RMSNorm**: 计算 `inv_rms = rsqrt(mean(X²) + norm_eps)`
3. **MatMul with Normalization**: `h_mix = F.linear(x_flat, phi)`，然后 `weight = h_mix * inv_rms`
4. **Split**: 将结果分流为三路 `[N, N, N²]`
5. **Branch Pre**: `h_pre = sigmoid(X_pre * α₀ + bias) + hc_eps`，加权求和生成 `h_in`
6. **Branch Post**: `h_post = 2 * sigmoid(X_post * α₁ + bias)`
7. **Branch Res**: `h_res = X_comb * α₂ + bias`，reshape 为 `[B*S, N, N]`

### 数学公式

**RMSNorm**:
```
inv_rms = rsqrt(mean(x_flat²) + norm_eps)
```

**Branch Pre (加权求和生成 h_in)**:
```
h_pre = sigmoid(X_pre * alpha[0] + bias[:N]) + hc_eps
h_in = sum(h_pre.unsqueeze(-1) * x.unflatten(-1, (N, D)), dim=1)
```

**Branch Post**:
```
h_post = 2 * sigmoid(X_post * alpha[1] + bias[N:2*N])
```

**Branch Res**:
```
h_res = X_comb * alpha[2] + bias[2*N:].view(N, N)
```

---

## 输入输出规格

### 输入张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `x` | `[B*S, N, D]` | bfloat16 | 输入特征 tensor |
| `phi` | `[N²+2N, N*D]` | float32 | 权重矩阵（MatMul 用） |
| `alpha` | `[3]` | float32 | 缩放系数 [α₀, α₁, α₂] |
| `bias` | `[N²+2N]` | float32 | 偏置向量（三分支共用） |

### 输出张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `h_in` | `[B*S, D]` | bfloat16 | 加权输入（Branch Pre 输出） |
| `h_post` | `[B*S, N]` | float32 | 后处理门控信号（Branch Post 输出） |
| `h_res` | `[B*S, N, N]` | float32 | 组合门控信号（Branch Res 输出） |

### 参数

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `norm_eps` | float | 1e-6 | RMSNorm 的 epsilon，防止除零 |
| `hc_eps` | float | 1e-6 | sigmoid 输出的精度保护，避免饱和区 |

---

## Shape 范围与约束

### 动态轴与静态轴

| 轴 | 范围 | 标记 | 说明 |
|----|------|------|------|
| B*S | {1024, 2048, 4096} | `DYNAMIC` | 动态轴，批大小 × 序列长度，无需重编译 |
| N | 8 | `STATIC` | 注意力头数，固定值 |
| D | 5120 | `STATIC` | 隐藏层维度，变化时触发重编译 |

### 派生常量

| 常量 | 公式 | 典型值 | 说明 |
|------|------|--------|------|
| N*D | N × D | 40960 | 特征维度（x flatten 后） |
| N²+2N | N² + 2N | 80 | 权重矩阵 phi 的第一维 |
| N² | N × N | 64 | Branch Res 输出维度 |

### 约束条件

1. **N 固定为 8**：注意力头数不可变（当前实现硬编码）
2. **D 为 STATIC**：D 维度变化会触发 kernel 重编译
3. **B*S 为 DYNAMIC**：支持动态 shape，无需重编译
4. **内存连续性**：
   - `x` 必须是 contiguous 的
   - `phi` 转置后需 `.contiguous()`（wrapper 中处理）
   - `bias` 需提前切片并 contiguous（wrapper 中处理）
5. **精度约束**：
   - 输入 `x` 为 BF16，计算前转为 FP32
   - `h_in` 输出为 BF16（FP32 → BF16）
   - `h_post` 和 `h_res` 输出为 FP32
   - sigmoid 和 MatMul 仅支持 FP32
6. **MatMul 约束**：
   - `X_flat`: `[B*S, N*D]` FP32
   - `phi_T`: `[N*D, N²+2N]` FP32
   - 输出: `[B*S, N²+2N]` FP32

---

## 实现特点

### 性能优化

1. **MatMul + Vector 混合算子**：
   - Step 3 使用 `pypto.matmul`（Cube 算子）
   - 其他步骤为 Vector 算子（sigmoid、mul、add、sum）

2. **Loop Unroll 优化**：
   - 对 BS 轴使用 `pypto.loop_unroll`，`unroll_list=[128]`
   - 分块处理，减少内存占用

3. **Tile Shape 设置**：
   - Vector 算子：`set_vec_tile_shapes(bs_tile, D_tile)`
   - Cube 算子：`set_cube_tile_shapes([16, 16], [512, 1024], [128, 128], enable_split_k=True)`
   - 自适应调整：根据 D 大小动态设置 `bs_tile` 和 `D_tile`

4. **Bias 预切片**：
   - 在 wrapper 中提前切片 bias：
     - `bias_pre`: `[N]` 用于 Branch Pre
     - `bias_post`: `[N]` 用于 Branch Post
     - `bias_comb`: `[N*N]` 用于 Branch Res
   - 避免 kernel 内部复杂的 view 操作

5. **精度转换路径**：
   - BF16 输入 → FP32 计算 → BF16/FP32 输出
   - `h_in`: FP32 → BF16
   - `h_post`/`h_res`: 保持 FP32

### 计算流程详解

**Step 1-4（归一化与 MatMul）**：
```python
# Reshape & Cast
x_flat = x.reshape([BS, N*D]).float()  # BF16 → FP32

# RMSNorm
inv_rms = rsqrt(mean(x_flat²) + norm_eps)

# MatMul
h_mix = matmul(x_flat, phi_T)  # [BS, N*D] @ [N*D, N²+2N]
X_hat_norm = h_mix * inv_rms

# Split
X_pre, X_post, X_comb = X_hat_norm.split([N, N, N*N], dim=-1)
```

**Step 5（Branch Pre - 加权求和）**：
```python
# sigmoid + 加权
h_pre = sigmoid(X_pre * alpha_0 + bias_pre) + hc_eps

# 扩展维度并加权求和
weighted_X = h_pre.unsqueeze(-1) * x_fp32_3d
h_in = sum(weighted_X, dim=1)  # 沿 N 维求和

# 转回 BF16
h_in_bf16 = h_in.to(bfloat16)
```

**Step 6（Branch Post）**：
```python
# sigmoid + 缩放
h_post = 2 * sigmoid(X_post * alpha_1 + bias_post)
```

**Step 7（Branch Res）**：
```python
# 线性变换
h_res_2d = X_comb * alpha_2 + bias_comb

# reshape 为 3D
h_res = h_res_2d.reshape([BS, N, N])
```

---

## 精度验证

### 容差设置

- **相对容差 (RTOL)**：0.0078125 (1/128)
- **绝对容差 (ATOL)**：0.0001

### 测试用例

| 测试名称 | B*S | N | D | 说明 |
|---------|-----|---|----|----|
| `test_mhc_pre_bs8` | 8 | 4 | 128 | 极小规模验证 |
| `test_mhc_pre_bs128_n4_d5120` | 128 | 4 | 5120 | 基础验证 |
| `test_mhc_pre_bs256` | 256 | 4 | 128 | 小规模验证 |
| `test_mhc_pre_bs1024` | 1024 | 4 | 5120 | 中等规模验证 |
| `test_mhc_pre_bs4096` | 4096 | 4 | 2560 | 大规模验证 |

### 验证方法

1. **Golden 实现**：`mhc_pre_golden.py` 提供纯 PyTorch 参考实现
2. **三态标记**：`[PRECISION_PASS]` 或 `[PRECISION_FAIL]`
3. **对比工具**：`numpy.testing.assert_allclose`
4. **验证项**：
   - 三个输出的 shape 和 dtype 验证
   - 数值对比（最大差异、均值差异）
   - 数值稳定性（无 NaN/Inf）
   - sigmoid 值域检查（h_post ∈ [0, 2]）