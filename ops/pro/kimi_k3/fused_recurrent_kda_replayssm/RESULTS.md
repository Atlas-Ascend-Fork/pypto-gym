# fused_recurrent_kda_replayssm 结果说明

本文给出 `fused_recurrent_kda_replayssm` 的用例、精度标准与性能结果。文中标「实测」的
数据是在设备上直接测得的；有效带宽等由实测时间与已知数据量按文中算式导出，可逐项复算。

测试环境：

| 项目 | 取值 |
| :--- | :--- |
| 设备 | Ascend 950PR（`Ascend950PR_9589`，`NpuArch=3510`），单卡、device 0 |
| 计算单元 | cube 32 核 / vector 64 核（`[SoCInfo] cube_core_cnt` / `vector_core_cnt`） |
| 片上缓存 | 每个向量核 253952 字节（248 KiB）；L2 134217728 字节（128 MiB） |
| 软件 | CANN 9.3.0、PyPTO-Pro（pypto 0.2.1）、torch 2.13.0、torch_npu 2.13.0rc1 |

## 1 用例

基准形状 `B=16, N=6, S=8, D=128`、BSND、FP32 state。全部用例在
`tests/test_fused_recurrent_kda_replayssm.py` 里，一个用例对应一个 `test_xxx` 函数。

| case | 覆盖 |
| :--- | :--- |
| `test_layouts[BSND/BNSD/TND]` | 三种布局；TND 用不均匀 `cu_seqlens = [5,1,3]` |
| `test_minimal_single_token` | `B=1, S=1, N=1` 最小全量 |
| `test_long_sequence` | `S=17`，跨多个预取周期 |
| `test_max_heads` | `N=96` 上限 |
| `test_record_capacity_suffix` | 记录容量大于序列长度，校验未使用后缀保持原值 |
| `test_rows[16/32/64]` | 三档行块 |
| `test_worker_sweep[1/8/32/64]` | 1 / 8 / 32 / 64 个 AIV |
| `test_repeat_determinism` | 同输入连续 10 次调用逐位一致 |

共 15 例，`TILE_FWK_DEVICE_ID=0` 下全通过。

## 2 精度标准

判据见 `tests/assets/gen_golden.py` 的 `check_precision`，即逐元素

```text
|actual - expected| <= atol + rtol * |expected|
```

`out`、`replay_u`、`replay_k`、`replay_decay` 均取 `atol = rtol = 5e-3`。参考实现为
`tests/assets/gen_golden.py` 的单一 torch 实现：CPU 模式用 FP64，NPU 模式用 FP32 并
按算子输出 dtype 产生 BF16/FP16/FP32；不复用算子代码。`recurrent_state` 在调用前后
逐位比较，未使用记录后缀按零容差比较。

### 2.1 结果

**15/15 PASS**。基准形状（`B=16, N=6, S=8, D=128`、BSND、FP32 state）的逐元素误差：

| 量 | 对 CPU FP64 参考 | 对 NPU FP32 参考 |
| :--- | ---: | ---: |
| `out` (BF16) | 1.214e-04 | 3.052e-05 |
| `replay_u` (FP16) | 9.747e-04 | 2.441e-04 |
| `replay_k` (FP16) | 1.220e-04 | 0 |
| `replay_decay` (FP32) | 1.321e-07 | 0 |
| `recurrent_state` | 逐位不变 | 逐位不变 |

`replay_u` 的误差主要是 FP16 记录舍入；工作状态递推始终在 FP32 上进行。

### 2.2 形状与边界行为

- **布局等价**：BSND / BNSD / TND 走同一 kernel body，仅行 stride 与 `packed` 不同；
  不均匀 `cu_seqlens`、`S=1`、`S=17` 均通过。
- **只读 checkpoint**：所有用例在调用后逐位比较 `recurrent_state`，无写入。
- **记录后缀**：三个 record 只有本轮有效 token 被覆盖，未使用区域保持调用前的值
  （测试用哨兵 `-7` 初始化并按零容差校验）。
- **确定性**：同输入连续 10 次调用，`out` 与 U/K 记录逐位一致（预取的显式 flag 同步
  不依赖调度顺序）。

## 3 性能结果

单次调用只下发一个 kernel，因此本算子的性能即该 kernel 的耗时。对比对象是
`tests/assets/gen_golden.py` 的 NPU 模式：torch + torch_npu，normalize 与激活在 token
循环内但 batch/head 向量化，一次调用约 349 个 device kernel；本实现一次调用 1 个
kernel。

**采集方法**

| 项 | 做法 |
| :--- | :--- |
| 计时 | torch_npu profiler `kernel_details.csv` 的 `Duration(us)`，取一次调用内**全部 device kernel 时长之和**；不含 host 派发与 kernel 间间隙 |
| 统计 | 单卡、单进程；两侧各 3 轮，每轮 10 次预热 + 30 次采样；每轮取中位数与最小值 |
| 达成比 | `Golden 一次调用的 device 时间 / 本实现 kernel Task Duration` |

### 3.1 绝对性能结果

| 实现 | 最小值 (µs) | 中位数 (µs) |
| :--- | ---: | ---: |
| Golden NPU（349 个 device kernel / 次） | 911.122 | 916.842 |
| 本实现（1 个 kernel / 次） | **15.045** | **15.637** |
| 达成比（Golden device 时间 / 本实现 kernel） | **60.6x** | **58.6x** |

Golden 的 3 轮中位数为 917.447 / 916.842 / 916.540 µs，本实现为 15.557 / 15.637 /
15.758 µs。两侧都只统计 device kernel 时间，比值是同一口径下的 device 时间比；差距
来自小算子的固定开销与重复搬运，而不是算术效率。

**Golden NPU 的构成**

一次调用 349 个 device kernel，按 Op 统计（10 次采样平均）：

| op | count | 平均 (µs) | 合计 (µs) |
| :--- | ---: | ---: | ---: |
| Mul | 96 | 3.891 | 373.5 |
| ReduceSum | 32 | 3.469 | 111.0 |
| Cast | 64 | 1.495 | 95.7 |
| Add | 32 | 2.894 | 92.6 |
| ViewCopy | 32 | 2.501 | 80.0 |
| Slice | 40 | 1.752 | 70.1 |
| Sigmoid | 16 | 1.921 | 30.7 |
| Rsqrt | 16 | 1.516 | 24.3 |
| Sub | 8 | 2.141 | 17.1 |
| Exp | 9 | 1.537 | 13.8 |
| ZerosLike | 4 | 2.250 | 9.0 |

合计约 918 µs。Golden 按 token 串行、batch/head 向量化，单个小算子的平均时长只有
1.5–3.9 µs，数量却有 349 个；本实现把同样的数学融合为单次下发的一个 kernel
（15.637 µs）。

### 3.2 理论评估对比

每个 token 对 state 做一次 rank-1 更新并写回记录，算术强度约 1 FLOP/Byte 量级，从
roofline 看落在访存侧。task 级 AIV pipe 计数（10 次采样中位数，基准形状）：

| 计数器 | 时间 (µs) | 占 `aiv_time` |
| :--- | ---: | ---: |
| `aiv_time` | 15.335 | 100% |
| vector | 12.564 | 81.7% |
| scalar | 8.737 | 57.0% |
| MTE2 | 3.770 | 24.5% |
| MTE3 | 3.290 | 21.4% |

vector pipe 占用最高（81.7%），是主瓶颈；scalar 57.0% 对应 reduction 与地址/循环控制；
MTE2/MTE3 已通过跨 token 预取与三槽轮转压到较低占比。剩余空间主要在减少每行 state
的向量指令数（两段点积）或把点积搬到大算力单元。

### 3.3 渐进过程

以下为开发期的历史数据，同一形状、同一 harness 的实测中位数：

| 阶段 | 措施 | 中位数 (µs) |
| :--- | :--- | ---: |
| 初版 | 两段点积 + 更新 pass；token 循环开头即时加载 | 18.4 |
| + 更新 pass unroll | 更新循环 2 行/迭代 | 18.2 |
| + 跨 token 预取 | 奇偶 parity 输入，`V→MTE2` / `MTE2→V` 显式 flag | 16.9 |
| + 预处理与点积融合 | `dk/dq` 留在寄存器，去掉 UB 往返 | 15.9 – 16.0 |
| + 加深 state 槽 | `slots` 2 → 3，同槽写读间隔从 1 个 token 变为 2 个 | **15.5 – 15.8** |

各步骤的判别性证据：

- **预取**：消融（去掉 5 次 token 输入 load）省 1.8 µs，预取后这部分基本被 VF 掩盖。
- **融合**：与初版 kernel 在同一进程交错对比，融合后 15.9–16.0 µs，提升约 4%。
- **state 槽深**：`slots` 2/3/4 在同一进程交错对比为 16.2 / 15.7 / 15.9 µs；三槽把同一槽
  的写读间隔从 1 个 token 拉到 2 个，buffer id 依赖不再紧贴流水，提升 2.5%–3.5%。
- **worker 数**：`48/56/64` 为 22.0 / 22.2 / 16.9 µs；`64` 的突变来自工作项整除
  （`192/64 = 3`，`/48`、`/56` 为 4）。
- **被否定的方案**：把输出点积放回更新行体（`dot(S_new, q)`）为 24.3 µs；去掉
  VST→VLD 同步得到 16.3 µs 但结果非确定；用 `bar_all` 代替 buffer-id 同步为 23.2 µs。
