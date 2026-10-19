# Fused SwiGLU Forward (PyPTO Kernel)

基于 PyPTO 框架实现的 Fused SwiGLU 前向传播算子，运行于 Ascend NPU。


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 文件说明

| 文件 | 说明 |
|------|------|
| `fused_swiglu_impl.py` | Kernel 实现 |
| `test_fused_swiglu.py` | 测试用例 + Golden 参考实现 |

## 算法概述

实现 SwiGLU（Swish-Gated Linear Unit）激活函数的前向传播，将门控线性单元（GLU）与 Swish 激活函数融合，并支持 bias 融合。

### 数学公式

$$
\text{Gate} = x @ W_g + b_g
$$

$$
\text{FC} = x @ W_{fc} + b_{fc}
$$

$$
\text{SwiGLU}(x) = \text{SiLU}(\text{Gate}) \times \text{FC} = \text{Gate} \times \sigma(\text{Gate}) \times \text{FC}
$$

其中：
- $\text{SiLU}(x) = x \times \sigma(x)$ (Swish 激活函数)
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ (Sigmoid 函数)

## Kernel 概览

前向传播的 kernel：

| Kernel | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `fused_swiglu_fwd_kernel` | 计算前向传播 | x, w_g, w_fc, b_g, b_fc | y |


## Kernel 签名

### Kernel : fused_swiglu_fwd_kernel

```python
fused_swiglu_fwd_kernel(
    x,      # [M, K]    BF16  — 输入张量（动态 batch 维度）
    w_g,    # [K, N]    BF16  — Gate 权重
    w_fc,   # [K, N]    BF16  — FC 权重
    b_g,    # [1, N]    BF16  — Gate 偏置
    b_fc,   # [1, N]    BF16  — FC 偏置
    y       # [M, N]    BF16  — 输出张量
)
```

## 参数规格

| 参数 | 类型 | Shape | 数据类型 | 说明 |
|------|------|-------|----------|------|
| x | 输入 | [M, K] | BF16 | 输入张量，M 为动态维度 |
| w_g | 输入 | [K, N] | BF16 | Gate 权重矩阵 |
| w_fc | 输入 | [K, N] | BF16 | FC 权重矩阵 |
| b_g | 输入 | [1, N] | BF16 | Gate 偏置 |
| b_fc | 输入 | [1, N] | BF16 | FC 偏置 |
| y | 输出 | [M, N] | BF16 | 输出张量 |

### 动态轴

- **M (Batch dimension)**: 动态维度，运行时可变

### 分块配置

| 参数 | 值 | 说明 |
|------|-----|------|
| tile_m | 1024 | Batch 维度分块大小 |
| matmul tile | [128, 128], [128, 256], [128, 128] | Matmul 分块配置 |
| vec tile | [128, 128] | 向量运算分块配置 |

## 运行测试

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 运行测试
python3 test_fused_swiglu.py
```

### 测试用例

| 用例 | M | K | N | 说明 |
|------|---|---|---|------|
| test_fwd | 220000 | 512 | 1024 | 大 batch + 标准 FFN 维度 |

### 精度校验

使用 `numpy.testing.assert_allclose` 进行精度验证：

```python
rtol = 0.0078125   # 1/128，等于 BF16 machine epsilon
atol = 0.0001
```

Golden reference 严格模拟 kernel 内部的 dtype 转换流程，确保对比基准与硬件行为一致。

## Dtype 转换流程

| 阶段 | 操作 | Dtype |
|------|------|-------|
| 输入 | x, w_g, w_fc, b_g, b_fc | BF16 |
| Matmul | x @ w_g, x @ w_fc | BF16 → BF16 |
| Bias | + b_g, + b_fc | BF16 |
| Sigmoid | sigmoid(g_bias) | BF16 |
| Element-wise | silu = g_bias * sigmoid_g | BF16 |
| Element-wise | y = silu * fc_bias | BF16 |
| 输出 | y | BF16 |

## 实现细节

### Pass 配置

```python
@pypto.frontend.jit(
    pass_options={
        "pg_upper_bound": 5000000,          # Pass group 上界
        "vec_nbuffer_setting": {-1: 2, 0: 4},  # 向量计算 buffer 配置
        "cube_l1_reuse_setting": {-1: 2}    # Cube L1 复用配置
    },
    runtime_options={
        "run_mode": global_run_mode,        # NPU 或 SIM 模式
        "stitch_function_max_num": 128,    # 最大 stitch 函数数
        "device_sched_mode": 3              # 设备调度模式
    }
)
```

## 性能优化

- **Tile 策略**: Batch 维度分块（tile_m=1024）优化内存访问
- **Cube 复用**: 启用 L1 复用提升 matmul 性能
- **向量化**: 使用 vector buffer 优化 element-wise 操作

## 依赖

- Python 3.x
- PyTorch + torch_npu
- PyPTO (`pypto` 包)
- NumPy
