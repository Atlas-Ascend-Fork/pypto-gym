# BNTrainingReduce 算子说明


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 1. 算子概述

`BNTrainingReduce` 用于 BatchNorm 训练阶段的归约计算。输入为 `NCHW` 格式的 `fp32` 张量，输出每个通道的：

- `sum_out`：`sum(x)`
- `sq_sum_out`：`sum(x * x)`

当前实现文件为 `bn_training_reduce.py`，算子实现和测试入口都在同一个脚本中。

## 2. 数学公式

对输入 `x[n, c, h, w]`，按 `N/H/W` 三个维度做归约：

```text
sum_out[0, c, 0, 0] = Σ x[n, c, h, w]
sq_sum_out[0, c, 0, 0] = Σ (x[n, c, h, w] ^ 2)
```

其中求和范围为所有 `n/h/w`。

## 3. 实现流程

当前 PyPTO 实现的计算路径如下：

1. 输入 `x` 从 `[N, C, H, W]` reshape 为 `[N, C, H*W]`
2. 计算平方项 `x * x`
3. 将张量转置为按归约维展开更方便的布局
4. reshape 为 `[N*H*W, C]`
5. 在 `dim=0` 上分别执行 `sum`，得到 `sum_raw` 和 `sq_sum_raw`
6. 将结果 reshape 回 `[1, C, 1, 1]`
7. 写入 `sum_out` 和 `sq_sum_out`

代码中使用 `select_bn_reduce_tiles(shape)` 根据输入 shape 选择 tiling 参数，当前策略会根据 `N` 和 `H*W` 自动推导：

- `tile_n`
- `tile_c`
- `tile_hw`
- `tile_reduce`
- `out_tile_c`

## 4. 输入输出规格

### 输入

| 名称 | 类型 | 形状 | 数据类型 | 说明 |
|------|------|------|----------|------|
| `x` | Tensor | `[N, C, H, W]` | `fp32` | 待归约输入 |

### 输出

| 名称 | 类型 | 形状 | 数据类型 | 说明 |
|------|------|------|----------|------|
| `sum_out` | Tensor | `[1, C, 1, 1]` | `fp32` | 每通道元素和 |
| `sq_sum_out` | Tensor | `[1, C, 1, 1]` | `fp32` | 每通道平方和 |

## 5. 代码结构

- `bn_training_reduce.py`
  - `select_bn_reduce_tiles(shape)`：根据输入规模选择 tiling
  - `bn_reduce_kernel(...)`：PyPTO kernel 实现
  - `main()`：测试入口，负责构造输入、执行 kernel、做 golden 对比

## 6. 运行方式

### 环境准备

请先根据本机环境设置 CANN、`torch_npu` 和 `pto-isa` 相关变量。典型示例：

```bash
source /path/to/ascend-toolkit/set_env.sh
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=<pto-isa 路径>
```

### 执行测试

```bash
python3 bn_training_reduce.py
```

当前脚本只支持 `npu` 运行模式，参数保留为：

```bash
python3 bn_training_reduce.py --run_mode npu
```

## 7. 当前测试逻辑

脚本内置了 4 组测试 shape：

```text
(2400, 88, 1, 1)
(400, 8, 1, 1)
(800, 40, 2, 1)
(900, 45, 1, 1)
```

每个 case 的流程为：

1. 构造随机输入 `x_torch`
2. 分配 `sum_out` / `sq_sum_out`
3. 调用 `bn_reduce_kernel(...)`
4. 使用 PyTorch 计算 golden：
   - `torch.sum(x, dim=(0, 2, 3), keepdim=True)`
   - `torch.sum(x * x, dim=(0, 2, 3), keepdim=True)`
5. 使用 `assert_allclose` 做精度校验
6. 输出一行精简误差摘要

示例输出格式：

```text
Case 1: sum_diff=1.220703e-04, sq_sum_diff=1.220703e-04
```

## 8. 精度标准

当前校验阈值为：

- `rtol=1e-3`
- `atol=1e-3`

对应代码位于 `bn_training_reduce.py` 中的 `assert_allclose(...)`。

## 9. 已做整改

针对上库前清理，当前脚本已做如下收敛：

- 删除了冗余调试打印，仅保留精度摘要、失败信息和最终总结
- 删除了修改 `TILE_FWK_OUTPUT_DIR` 的逻辑，不再主动改写输出目录
- 保持 kernel 调用、golden 计算和精度断言逻辑不变

## 10. 已知限制

- 当前实现只覆盖 `fp32`
- 当前测试入口默认依赖 NPU 环境和 `torch_npu`
- 当前文件仍是单脚本组织方式，尚未拆分为独立算子目录

## 11. 常见问题

### 运行时报 `TILE_FWK_DEVICE_ID` 未设置

先设置可用 NPU 设备号，例如：

```bash
export TILE_FWK_DEVICE_ID=0
```

### 运行时报 `Invalid Device`

先检查可用设备：

```bash
npu-smi info
```

再选择正确的设备号。
