# Fused SwiGLU Backward (PyPTO Kernel)

基于 PyPTO 框架实现的 Fused SwiGLU 反向传播算子，运行于 Ascend NPU。


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 文件说明

| 文件 | 说明 |
|------|------|
| `fused_swiglu_grad_impl.py` | Kernel 实现（含 3 个子 kernel） |
| `test_fused_swiglu_grad.py` | 测试用例 + Golden 参考实现 |

## 算法概述

实现 SwiGLU（Swish-Gated Linear Unit）的反向传播，将三个独立的梯度计算 kernel 融合，分别计算：
- **b_kernel**: dg、dfc（中间梯度）+ db_g、db_fc（偏置梯度）
- **w_kernel**: dw_g、dw_fc（权重梯度）
- **x_kernel**: dx（输入梯度）


### 数学公式

给定上游梯度 dy，需计算：

**1. dg 和 dfc（中间梯度）**

$$
dg = dy \cdot FC \cdot \sigma'(Gate)
$$

$$
\text{SiLU}'(g) = \sigma(g) \cdot (1 + g \cdot (1 - \sigma(g)))
$$

$$
dfc = dy \cdot \text{SiLU}(Gate) = dy \cdot g \cdot \sigma(g)
$$

**2. db_g 和 db_fc（偏置梯度）**

$$
db_g = \sum_{i} dg_i \quad \text{(沿 batch 维度求和)}
$$

$$
db_fc = \sum_{i} dfc_i \quad \text{(沿 batch 维度求和)}
$$

**3. dw_g 和 dw_fc（权重梯度）**

$$
dw_g = x^T \cdot dg
$$

$$
dw_fc = x^T \cdot dfc
$$

**4. dx（输入梯度）**

$$
dx = dg \cdot W_g^T + dfc \cdot W_{fc}^T
$$

## Kernel 概览

反向传播拆分为三个独立的 kernel，按顺序调用：

| Kernel | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `fused_swiglu_bwd_b_kernel` | 计算中间梯度 + 偏置梯度 | dy, g, fc | dg, dfc, db_g, db_fc |
| `fused_swiglu_bwd_w_kernel` | 计算权重梯度 | x, dg, dfc | dw_g, dw_fc |
| `fused_swiglu_bwd_x_kernel` | 计算输入梯度 | dg, dfc, w_g, w_fc | dx |

## Kernel 签名

### Kernel 1: fused_swiglu_bwd_b_kernel

```python
fused_swiglu_bwd_b_kernel(
    dy,     # [M, N]    BF16  — 上游梯度（动态 batch 维度）
    g,      # [M, N]    BF16  — Gate 中间值（前向传播输出）
    fc,     # [M, N]    BF16  — FC 中间值（前向传播输出）
    dg,     # [M, N]    BF16  — Gate 梯度输出
    dfc,    # [M, N]    BF16  — FC 梯度输出
    db_g,   # [1, N]    BF16  — Gate 偏置梯度输出（需初始化为 0）
    db_fc   # [1, N]    BF16  — FC 偏置梯度输出（需初始化为 0）
)
```

### Kernel 2: fused_swiglu_bwd_w_kernel

```python
fused_swiglu_bwd_w_kernel(
    x,      # [M, K]    BF16  — 输入张量（前向传播输入）
    dg,     # [M, N]    BF16  — Gate 梯度
    dfc,    # [M, N]    BF16  — FC 梯度
    dw_g,   # [K, N]    BF16  — Gate 权重梯度输出（需初始化为 0）
    dw_fc   # [K, N]    BF16  — FC 权重梯度输出（需初始化为 0）
)
```

### Kernel 3: fused_swiglu_bwd_x_kernel

```python
fused_swiglu_bwd_x_kernel(
    dg,     # [M, N]    BF16  — Gate 梯度
    dfc,    # [M, N]    BF16  — FC 梯度
    w_g,    # [K, N]    BF16  — Gate 权重
    w_fc,   # [K, N]    BF16  — FC 权重
    dx      # [M, K]    BF16  — 输入梯度输出
)
```

## 参数规格

### Kernel 1: b_kernel

| 参数 | 类型 | Shape | 数据类型 | 说明 |
|------|------|-------|----------|------|
| dy | 输入 | [M, N] | BF16 | 上游梯度，M 为动态维度 |
| g | 输入 | [M, N] | BF16 | Gate 中间值（前向传播保存） |
| fc | 输入 | [M, N] | BF16 | FC 中间值（前向传播保存） |
| dg | 输出 | [M, N] | BF16 | Gate 梯度 |
| dfc | 输出 | [M, N] | BF16 | FC 梯度 |
| db_g | 输出 | [1, N] | BF16 | Gate 偏置梯度（需初始化为 0） |
| db_fc | 输出 | [1, N] | BF16 | FC 偏置梯度（需初始化为 0） |

### Kernel 2: w_kernel

| 参数 | 类型 | Shape | 数据类型 | 说明 |
|------|------|-------|----------|------|
| x | 输入 | [M, K] | BF16 | 输入张量（前向传播保存） |
| dg | 输入 | [M, N] | BF16 | Gate 梯度 |
| dfc | 输入 | [M, N] | BF16 | FC 梯度 |
| dw_g | 输出 | [K, N] | BF16 | Gate 权重梯度（需初始化为 0） |
| dw_fc | 输出 | [K, N] | BF16 | FC 权重梯度（需初始化为 0） |

### Kernel 3: x_kernel

| 参数 | 类型 | Shape | 数据类型 | 说明 |
|------|------|-------|----------|------|
| dg | 输入 | [M, N] | BF16 | Gate 梯度 |
| dfc | 输入 | [M, N] | BF16 | FC 梯度 |
| w_g | 输入 | [K, N] | BF16 | Gate 权重 |
| w_fc | 输入 | [K, N] | BF16 | FC 权重 |
| dx | 输出 | [M, K] | BF16 | 输入梯度 |

### 动态轴

- **M (Batch dimension)**: 动态维度，运行时可变

### 分块配置

| Kernel | tile_m | vec_tile | cube_tile (MNK) |
|--------|--------|----------|-----------------|
| b_kernel | 1024 | [128, 128] | - |
| w_kernel | 2048 | [128, 128] | [128, 128], [128, 256], [128, 256] |
| x_kernel | 1024 | [128, 128/256] | [128, 128], [64, 256], [256, 256] |

## 运行测试

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 运行测试
python test_fused_swiglu_grad.py
```

### 测试用例

| 用例 | M | K | N | 说明 |
|------|---|---|---|------|
| test_bwd | 220000 | 512 | 1024 | 大 batch + 标准 FFN 维度 |

### 精度校验

使用 `numpy.testing.assert_allclose` 进行精度验证：

```python
rtol = 0.0078125   # 1/128，等于 BF16 machine epsilon
atol = 0.0001
```

校验输出包括：
- **dx**: 输入梯度
- **dw_g**: Gate 权重梯度
- **dw_fc**: FC 权重梯度
- **db_g**: Gate 偏置梯度
- **db_fc**: FC 偏置梯度

Golden reference 严格模拟 kernel 内部的 dtype 转换流程，确保对比基准与硬件行为一致。

## Dtype 转换流程

| 阶段 | 操作 | Dtype |
|------|------|-------|
| 输入 | dy, g, fc, x, w_g, w_fc | BF16 |
| sigmoid | exp(g) / (1 + exp(g)) | BF16 |
| element-wise | dg, dfc 计算 | BF16 |
| sum | db_g, db_fc | BF16 |
| matmul (w_kernel) | x^T @ dg, x^T @ dfc | BF16 → BF16 |
| matmul (x_kernel) | dg @ w_g^T, dfc @ w_fc^T | BF16 → BF16 |
| 输出 | dg, dfc, dx, dw_g, dw_fc, db_g, db_fc | BF16 |

## 实现细节

### Pass 配置

**b_kernel:**
```python
@pypto.frontend.jit(
    pass_options={
        "pg_upper_bound": 5000000,
        "vec_nbuffer_setting": {-1: 2, 0: 4}
    },
    runtime_options={
        "run_mode": global_run_mode,
        "stitch_function_max_num": 128,
        "device_sched_mode": 3
    }
)
```

**w_kernel:**
```python
@pypto.frontend.jit(
    pass_options={
        "pg_upper_bound": 5000000,
        "vec_nbuffer_setting": {-1: 2, 0: 8},
        "cube_l1_reuse_setting": {-1: 2}
    },
    ...
)
```

**x_kernel:**
```python
@pypto.frontend.jit(
    pass_options={
        "pg_upper_bound": 5000000,
        "vec_nbuffer_setting": {-1: 2, 0: 4},
        "cube_l1_reuse_setting": {-1: 2}
    },
    ...
)
```

### NPU 架构适配

x_kernel 根据 NPU 架构动态调整 vec_tile 配置：

```python
if pypto.platform.npuarch == 'DAV_3510':
    pypto.set_vec_tile_shapes(128, 256)
else:
    pypto.set_vec_tile_shapes(128, 128)
```

## 性能优化

- **分块策略**: Batch 维度分块优化内存访问
- **Cube 复用**: w_kernel 和 x_kernel 启用 L1 复用提升 matmul 性能
- **向量化**: 使用 vector buffer 优化 element-wise 操作
- **梯度累加**: db_g, db_fc, dw_g, dw_fc 支持跨 tile 累加

## 依赖

- Python 3.x
- PyTorch + torch_npu
- PyPTO (`pypto` 包)
- NumPy

