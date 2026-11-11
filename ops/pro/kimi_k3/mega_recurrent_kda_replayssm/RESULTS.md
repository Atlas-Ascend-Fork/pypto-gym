# mega_recurrent_kda_replayssm 结果说明

本文给出 `mega_recurrent_kda_replayssm` 的精度与性能结果：用哪些用例检验、按什么标准判定、
实测数据是多少、以及这些数据说明什么。

## 1 用例

**测试环境**

| 项目 | 取值 |
| :--- | :--- |
| 硬件平台 | Ascend 950DT（`Ascend950DT_9582`，`soc_version = 'DAV_3510'`），cube 32 核 / vector 64 核，L2 128 MiB |
| 软件环境 | CANN 9.2.0、PyPTO-Pro、Python 3.12.3、torch 2.13.0、torch_npu 2.13.0 |
| 几何常量 | `S=8`、`H=7168`、`D=128`、`K=4`；`N ∈ {6,12,24}`（TP16/8/4） |

**用例清单**

`tests/test_mega_recurrent_kda_replayssm.py` 里一个用例对应一个 `test_xxx` 函数，共 15 个
（每个都带 `@pytest.mark.soc("950")`）：

| case | N（TP） | B | 起核数 | 说明 |
| :--- | ---: | ---: | ---: | :--- |
| `b16_auto` | 6（TP16） | 16 | 32 | 默认档，N128 退化边界 |
| `b32_g28` | 6 | 32 | 28 | 非整档网格 |
| `b48_g24` | 6 | 48 | 24 | 任务数恰等于核数 |
| `b128_g28` | 6 | 128 | 28 | 8 组，任务数不能被核数整除 |
| `b512_g28` | 6 | 512 | 28 | 最大合法 B（32 组） |
| `b16_conv_layout0` | 6 | 16 | 8 | 最小档网格 + `conv1d_weight` `[3,N,4,D]` |
| `b16_conv_layout2` | 6 | 16 | 8 | 最小档网格 + `conv1d_weight` `[3,4,D]` |
| `n12_b16` | 12（TP8） | 16 | 28 | TP8 映射 |
| `n12_b256` | 12 | 256 | 28 | TP8 大 batch |
| `n24_b16` | 24（TP4） | 16 | 28 | TP4 映射 |
| `n24_b128` | 24 | 128 | 28 | TP4 大 batch |
| `n24_b4` | 24 | 4 | 28 | 最小 B（m-tile 32 行档） |
| `n12_b8` | 12 | 8 | 28 | m-tile 64 行档 |
| `n12_b64_g8` | 12 | 64 | 8 | 8 核的 `m3_k` 规则 |
| `n24_b32_g8` | 24 | 32 | 8 | 8 核的 `m3_k` 规则 |

覆盖了 N=6/12/24 三档、`B=4..512`、m-tile 阶梯的 32/64/128 行档、`conv1d_weight` 的三种
公开 layout（`[3,N,4,D]` / `[4,3P]` / `[3,4,D]`），以及 8/24/28/32 四个核档。

**运行方式**

```bash
python -m pytest -q ops/pro/kimi_k3/mega_recurrent_kda_replayssm/tests/
```

精度判定与参考值都在 `tests/assets/` 下，随算子一起维护。

## 2 精度标准

精度判定分两层：

1. **算子验收契约**（`tests/` 中的 `_assert_precision`）：逐元素
   `|actual - golden| <= atol + rtol*|golden|`，`atol = rtol = 5e-3`；另加
   `recurrent_state` 只读（零容差）与未使用 Conv 槽位零容差两条状态不变量。
2. **方案 A 混合容差比较器**（`tests/assets/mega_recurrent_kda_replayssm_precision.py`，
   补充证据）：`matched_ratio >= 0.99` 且 `max_abs_error <= max(fixed_limit, 32·ULP)`、
   `mean_abs_error <= atol`、`mean_rel_error <= rtol`；阈值表硬编码、不接受外部参数。

比较器的逐 dtype 阈值（`replay_k` 是 FP16，其余输出为 BF16/FP32）：

| dtype | `rtol` | `atol` | `fixed_limit` |
| :--- | ---: | ---: | ---: |
| BF16 | 1.562e-02（2⁻⁶） | 1.562e-02（2⁻⁶） | 1.0 |
| FP16 | 1.953e-03（2⁻⁹） | 1.953e-03（2⁻⁹） | 0.1 |
| FP32 | 9.766e-04（2⁻¹⁰） | 1.526e-05（2⁻¹⁶） | 0.01 |

注意 FP16 的通用默认（`rtol = atol = 2⁻⁹ ≈ 1.95e-3`）比算子契约 `5e-3` 更严：比较器对
`replay_k` 的判定因此更严格；实测它也不超限（下表 `replay_k` 最大绝对误差 2.4e-3 在
`|replay_k|` 较大的元素上，逐元素判定通过），两层判定 15/15 全部一致通过。

### 2.1 结果

**15 / 15 通过。** 三个代表点的逐张量最大绝对误差（当前交付 kernel 实测）：

公开接口可见的输出是 `out` 与四个原地更新张量（`conv_state` / `replay_k` / `replay_u` /
`replay_decay`），下表就是测试实际比较的这五项：

| N | B | `out` | `conv_state` | `replay_u` | `replay_k` | `replay_decay` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 16 | 7.6e-06 | 4.9e-04 | 8.6e-06 | 2.2e-03 | 2.1e-05 |
| 12 | 16 | 7.6e-06 | 2.4e-04 | 8.6e-06 | 2.4e-03 | 2.7e-05 |
| 24 | 16 | 9.5e-06 | 4.9e-04 | 8.6e-06 | 2.4e-03 | 3.3e-05 |

全部满足 `atol=rtol=5e-3`；公开输出里最大的是 FP16 的 `replay_k`（2.4e-3），最终输出 `out`
只有 1e-5 量级。两个状态不变量在每个用例上都检查：

- `recurrent_state` 在调用前后逐位相同（零容差只读）；
- `conv_state` 中未被 `conv_state_indices` 引用的槽位逐位不变（零容差）。

注：`ws_beta_logit` / `ws_decay_a` 走 BF16 atomic add 累加，累加顺序随调度变化，所以同一
输入重复调用的结果可能相差 1 个 BF16 ulp（实测 `out` max_abs 3.8e-6），逐元素判定仍通过；
测试因此不依赖"逐位可复现"，只看上面的容差契约与状态不变量。

### 2.2 覆盖与边界行为

已实测覆盖（15 个交付用例）：`B` 的最小值 4 与最大值 512、m-tile 32/64/128 行档、
N=6/12/24、`conv1d_weight` 三种 layout（`[3,N,4,D]`、`[4,3P]`、`[3,4,D]`）、
8/24/28/32 四个核档、N384 与 N128 两档面板。

声明范围之外的输入由 host 侧校验拦下（`_validate_args`），不会静默给出错误结果：

| 情形 | 行为 |
| :--- | :--- |
| `B ∉ [4, 512]` | `ValueError`：`B must be in [4, 512]` |
| `B*S` 不能被 m-tile 阶梯整除 | `ValueError`：报出 B 与阶梯 |
| `N ∉ {6,12,24}`、`H ≠ 7168`、`S ≠ 8` | `ValueError`：报出期望形状 |
| 权重 dtype 非 BF16 / 非 FRACTAL_NZ | `ValueError`：报出参数名、期望与实际 |
| `conv1d_weight` 三种 layout 以外 | `ValueError`：列出支持的三种形状 |
| `conv_state` 长度 `< S+2` | `ValueError`：`state_length must be >= S+2` |
| `recurrent_state` / `replay_*` 形状、dtype 不符 | `ValueError`：逐参数报出期望 |
| `lower_bound ∉ [-5,0]`、`scale` 非有限、`rms_norm_eps <= 0` | `ValueError`：报出取值 |
| 三个标量不是 Python `float`（如 numpy 标量、int） | `ValueError`：报出参数名、实际类型与取值 |
| 当前 stream 少于 8 个 AIC | `ValueError`：`at least 8 are required` |

## 3 性能结果

单次调用只下发**一个** mixed kernel（workspace 清零已并入 kernel，host 侧没有额外的
memset 小算子），因此本算子的性能即该 kernel 的耗时。

**采集方法**

| 项 | 做法 |
| :--- | :--- |
| 计时 | `msprof` 任务级耗时（`--task-time`，取 `op_summary` 的 `Task Duration(us)`，op 名含 `mega_recurrent_kda_replayssm_kernel`） |
| 冷 L2 | 每次采样前恢复 `conv_state` 初值，再用 224 MiB 缓冲**读取**一次（`evict.sum()`）把 L2 冲干净；不用写入方式驱逐，避免脏行回写影响被测 kernel |
| 统计 | 每点 3 次预热 + 20 次采样，取 20 次的**算术均值**（同时记录 min/median 作稳定性核对）；一个点一个进程 |

### 3.1 48 点kernel时间（µs）

> 注：本表为最终交付实现的 48 点协议复测：3 预热 + 20 采样、**取 20 次算术均值**、
> 一个点一个进程、每次恢复 conv_state 并用 224 MiB **读**驱逐清 L2。

48 点 = TP4/8/16（N=24/12/6）× 模型 batch 1/8/16/32 × AIC 8/16/24/32。每格为该点
20 次采样的算术均值（µs）：

| TP | mb | B | 8 AIC | 16 AIC | 24 AIC | 32 AIC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 4 | 281.4 | 129.3 | 127.1 | 92.1 |
| 4 | 8 | 32 | 955.7 | 526.3 | 402.0 | 284.9 |
| 4 | 16 | 64 | 1841.4 | 1009.1 | 755.1 | 495.3 |
| 4 | 32 | 128 | 3611.8 | 1865.4 | 1297.4 | 921.3 |
| 8 | 1 | 8 | 184.9 | 93.6 | 81.5 | 78.1 |
| 8 | 8 | 64 | 920.0 | 489.3 | 362.3 | 259.1 |
| 8 | 16 | 128 | 1804.1 | 915.5 | 661.8 | 481.6 |
| 8 | 32 | 256 | 3584.4 | 1803.0 | 1238.8 | 931.1 |
| 16 | 1 | 16 | 154.6 | 101.3 | 70.5 | 69.4 |
| 16 | 8 | 128 | 987.3 | 467.0 | 344.1 | 254.8 |
| 16 | 16 | 256 | 1963.4 | 931.4 | 663.1 | 479.5 |
| 16 | 32 | 512 | 3929.5 | 1847.7 | 1266.9 | 953.0 |
