# ApplyRMSProp 算子说明


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 1. 算子概述

`ApplyRMSProp` 实现 RMSProp 优化算法的一次参数更新。当前目录下的核心文件为：

- `apply_rms_prop.py`：golden 实现和功能测试入口
- `apply_rms_prop_impl.py`：PyPTO kernel 实现
- `perf_test.py`：额外的泛化/性能测试脚本

当前实现面向 `fp32`、二维张量输入，主要验证路径为 NPU。

## 2. 数学公式

设输入为：

- `var`：待更新参数
- `ms`：平方梯度移动平均
- `mom`：动量项
- `grad`：当前梯度

标量参数为 `lr`、`rho`、`momentum`、`epsilon`，则更新公式为：

```text
grad_sq = grad * grad
ms_new = ms + (grad_sq - ms) * (1 - rho)
mom_new = mom * momentum + (grad * lr) / sqrt(ms_new + epsilon)
var_new = var - mom_new
```

## 3. PyPTO 实现映射

`apply_rms_prop_impl.py` 中的 kernel 使用如下算子组合：

1. `grad * grad` 计算平方梯度
2. `ms + (grad_sq - ms) * (1.0 - rho)` 更新 `ms`
3. `pypto.sqrt(ms_new + epsilon)` 计算分母
4. `mom * momentum + (grad * lr) / sqrt(...)` 更新 `mom`
5. `var - mom_new` 更新 `var`
6. 通过 `move(...)` 将更新后的结果原地写回 `var/ms/mom`

当前 kernel 还设置了：

- `pypto.experimental.set_operation_options(combine_axis=True)`
- `pypto.set_vec_tile_shapes(32, 512)`

## 4. 输入输出规格

### 输入

| 名称 | 类型 | 形状 | 数据类型 | 说明 |
|------|------|------|----------|------|
| `var` | Tensor | `[rows, cols]` | `fp32` | 待更新参数 |
| `ms` | Tensor | `[rows, cols]` | `fp32` | 平方梯度移动平均 |
| `mom` | Tensor | `[rows, cols]` | `fp32` | 动量项 |
| `grad` | Tensor | `[rows, cols]` | `fp32` | 当前梯度 |
| `lr` | Scalar | `-` | `fp32` | 学习率，默认 `0.001` |
| `rho` | Scalar | `-` | `fp32` | 衰减率，默认 `0.9` |
| `momentum` | Scalar | `-` | `fp32` | 动量系数，默认 `0.9` |
| `epsilon` | Scalar | `-` | `fp32` | 防止除零，默认 `1e-7` |

### 输出

| 名称 | 类型 | 形状 | 数据类型 | 说明 |
|------|------|------|----------|------|
| `var` | Tensor | `[rows, cols]` | `fp32` | 原地更新后的参数 |
| `ms` | Tensor | `[rows, cols]` | `fp32` | 原地更新后的平方梯度移动平均 |
| `mom` | Tensor | `[rows, cols]` | `fp32` | 原地更新后的动量项 |

## 5. 功能测试入口

主测试脚本为：

```bash
python3 apply_rms_prop.py
```

支持的测试项：

- `level0`：`8x8`
- `level1`：`1024x1024`
- `level2`：`16x16`

可用命令示例：

```bash
# 运行全部测试
python3 apply_rms_prop.py

# 运行单个测试
python3 apply_rms_prop.py level0

# 查看测试列表
python3 apply_rms_prop.py --list
```

脚本参数里保留了 `--run_mode`，但当前主要验证和推荐路径为 `npu`。

## 6. 环境准备

请先根据本机环境配置 CANN、`torch_npu` 和 `pto-isa`。典型示例：

```bash
source /path/to/ascend-toolkit/set_env.sh
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=<pto-isa 路径>
```

如果需要重新编译并安装 Python 包：

```bash
python3 build_ci.py -f python3 --disable_auto_execute
```

## 7. 精度检查逻辑

`apply_rms_prop.py` 中的每个 level 都会：

1. 构造随机 `var/ms/mom/grad`
2. 调用 `apply_rms_prop_kernel(...)`
3. 使用 NumPy 版 `apply_rms_prop_golden(...)` 计算 golden
4. 分别计算 `var_diff`、`ms_diff`、`mom_diff`
5. 使用显式阈值检查误差
6. 输出一行精简误差摘要

示例输出格式：

```text
level0: shape=(8, 8), var_diff=0.00000000, ms_diff=0.00000000, mom_diff=0.00000000
```

当前阈值为：

- `atol=1e-5`
- `rtol=1e-5`

## 8. 泛化/性能脚本

目录中保留了 `perf_test.py`、`perf_test_report.md` 和 `performance_analysis_report.md` 作为补充资料。

当前上库前整改已经做了两点清理：

- 删除了输出目录重命名脚本 `rename_output_dirs.py`
- 收敛了测试日志，只保留精简误差摘要和最终汇总

如果后续要继续使用 `perf_test.py` 作为正式入口，建议先确认它与当前 kernel 调用接口保持一致。

## 9. 已知限制

- 当前实现只覆盖 `fp32`
- 输入必须是二维张量
- 当前主验证路径依赖 NPU 环境和 `torch_npu`
- `apply_rms_prop.py` 是当前功能验证的主入口，其他辅助脚本使用前建议再做一次接口核对

## 10. 常见问题

### 运行时报 `TILE_FWK_DEVICE_ID` 未设置

先设置 NPU 设备号，例如：

```bash
export TILE_FWK_DEVICE_ID=0
```

### 运行时报 `Invalid Device`

先检查设备：

```bash
npu-smi info
```

再选择正确的设备号。

### 精度断言失败怎么办

优先检查以下几点：

- 输入是否仍为 `fp32`
- `lr/rho/momentum/epsilon` 是否与 golden 计算一致
- 是否误改了 `apply_rms_prop_impl.py` 中的运算顺序
