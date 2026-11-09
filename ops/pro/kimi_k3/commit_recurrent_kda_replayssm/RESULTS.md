# commit_recurrent_kda_replayssm 结果说明

本文给出 `commit_recurrent_kda_replayssm` 的用例、精度标准与性能结果。文中标「实测」的
数据是在设备上直接测得的；有效带宽等由实测时间与已知数据量按文中算式导出，可逐项复算。

测试环境：

| 项目 | 取值 |
| :--- | :--- |
| 设备 | Ascend 950PR（`Ascend950PR_9589`，`NpuArch=3510`），单卡 |
| 计算单元 | cube 32 核 / vector 64 核（`[SoCInfo] cube_core_cnt` / `vector_core_cnt`） |
| 片上缓存 | 每个向量核 253952 字节（248 KiB）；L2 134217728 字节（128 MiB） |
| 软件 | CANN 9.3.0、PyPTO-Pro（pypto 0.2.1）、torch 2.13.0、torch_npu 2.13.0rc1 |

## 1 用例

固定形状 `B=16, N=6, S=8, D=128`、`accepted=4`（性能基准形状），FP32 state。全部用例
在 `tests/test_commit_recurrent_kda_replayssm.py` 里，一个用例对应一个 `test_xxx` 函数。

| case | B | S | N | accepted | 说明 |
| :--- | --: | --: | --: | :--- | :--- |
| `bench_shape_accepted4` | 16 | 8 | 6 | 4 | 基准形状 |
| `accepted1` / `accepted_all` | 16 | 8 | 6 | 1 / 8 | 前缀长度边界 |
| `accepted_zero` | 4 | 16 | 6 | `[0,16,3,0]` | 逐 batch 混合，含 0 与 `S` |
| `mixed_accepted` | 7 | 4 | 6 | `[0,4,1,0,2,3,0]` | 奇数 batch、混合前缀 |
| `multichunk_mixed` | 4 | 32 | 6 | `[0,32,20,5]` | `S > cap` 的多 chunk 路径 |
| `multichunk_tail` | 2 | 40 | 3 | `[40,39]` | 多 chunk + 非整 chunk 尾块 |
| `single_batch_head` | 1 | 8 | 1 | 4 | 最小 B×N |
| `tiny` | 3 | 3 | 2 | 2 | 极小形状 |
| `seq_one` | 16 | 1 | 6 | 1 | `S=1` |
| `wide_heads` | 2 | 8 | 32 | 4 | 多头 |
| `large_batch` | 32 | 8 | 6 | 4 | 大 batch |
| `single_head_full` | 8 | 8 | 1 | 8 | 单 head + 全接受 |

另覆盖 `rows ∈ {8,16,32,64,128}` 的 tiling 扫描，以及重复调用确定性、`accepted=0`
恒等性、非法 `rows` 拒绝。全部用例共 20 个，`TILE_FWK_DEVICE_ID=0` 下全部通过。

## 2 精度标准

判据见 `tests/assets/gen_golden.py` 的 `violations` / `assert_close`：逐元素

```text
|cand - ref| <= atol + rtol * |ref|,  atol = rtol = 1e-6
```

另报告带绝对下限的「floored 相对误差」`max |cand-ref| / max(|ref|, 1e-3)`，使近零元素
可读。参考实现为 `tests/assets/gen_golden.py` 的单一 torch 实现：NumPy 输入在 CPU 上以
FP32 运行并返回 NumPy，torch 输入在设备上以 FP32 运行；不复用算子代码。

### 2.1 结果

**全部 PASS，且与 golden 逐位一致**（floored 相对误差 `0.0`）。多个代表性 case
（含 `S > cap` 的多 chunk、混合 `accepted`、`accepted=0`）：

| case | floored 相对误差 | 越界元素 |
| :--- | --: | --: |
| `bench_shape_accepted4` | 0.0 | 0 |
| `mixed_accepted` | 0.0 | 0 |
| `accepted_zero` | 0.0 | 0 |
| `multichunk_mixed` | 0.0 | 0 |
| `multichunk_tail` | 0.0 | 0 |
| `large_batch` | 0.0 | 0 |

### 2.2 形状与边界行为

- **原地/只读语义**：`replay_u`、`replay_k`、`replay_decay`、`num_accepted_tokens`
  调用前后逐位不变（测试内 `torch.equal` 断言）。
- **`accepted = 0`**：对应 batch 的 checkpoint 逐位不变（含随机初值，非全零）。
- **确定性**：同一输入重复 15 次，结果逐位相同（覆盖 `rows ∈ {8, 64}`），排除了
  loop-carried 缓冲竞态。
- **非法输入**：`rows=96` 被 host 侧拒绝。

## 3 性能结果

本算子不使用 Cube，单次调用只下发一个 kernel，因此本算子的性能即该 kernel 的耗时。
对比对象是 `tests/assets/gen_golden.py` 的 NPU 模式（torch + torch_npu），两侧用完全
相同的 harness 与输入。

**采集方法**

| 项 | 做法 |
| :--- | :--- |
| 计时 | torch_npu profiler `kernel_details.csv` 的 `Duration(us)`，取一次调用内**全部 device kernel 时长之和**；不含 host 派发与 kernel 间间隙 |
| 对比对象 | Golden NPU：torch + torch_npu，batch/head 向量化、只按 token 串行，一次调用 25 个 device kernel；本实现：一次调用 1 个 kernel |
| 统计 | 单卡、单进程；两侧都 10 次预热 + 30 次采样 × 3 轮；采集前把输入 state 重置到同一初值；每轮取中位数与最小值 |
| 达成比 | `Golden 一次调用的 device 时间 / 本实现 kernel Task Duration` |

### 3.1 绝对性能结果

| 实现 | 最小值 (µs) | 中位数 (µs) |
| :--- | ---: | ---: |
| Golden NPU（25 个 device kernel / 次） | 115.455 | 117.883 |
| 本实现（1 个 kernel / 次） | **4.172** | **4.902** |
| 达成比（Golden device 时间 / 本实现 kernel） | **27.7x** | **24.1x** |

Golden 的 3 轮中位数为 117.813 / 117.883 / 118.002 µs，本实现为 4.919 / 4.883 / 4.902 µs。
两侧都只统计 device kernel 时间，比值是同一口径下的 device 时间比；同一组输入上，本
实现与 Golden NPU 的 `recurrent_state` 逐位相等（`max |diff| = 0.0`）。

**Golden NPU 的构成**

一次调用 25 个 device kernel，按 Op 统计（10 次采样平均）：

| op | count | 平均 (µs) | 合计 (µs) |
| :--- | ---: | ---: | ---: |
| Mul | 8 | 6.835 | 54.7 |
| SelectV2 | 4 | 5.968 | 23.9 |
| Add | 4 | 5.320 | 21.3 |
| Greater | 4 | 1.442 | 5.8 |
| Cast | 3 | 1.854 | 5.6 |
| TensorMove | 1 | 4.768 | 4.8 |
| ReduceMax | 1 | 2.575 | 2.6 |

合计 117.9 µs。Golden 已经按 batch/head 向量化，只保留 token 串行；把逐 token 展开成
小算子后，单个算子的固定开销（约 1.4–6.8 µs）仍占主要部分。本实现把同样的数学融合为
单次下发的一个 kernel（4.898 µs），因此 24x 的差距来自「25 个小算子 vs 1 个融合
kernel」的调度与固定开销差异，而不是算术效率差异。

### 3.2 理论评估对比

**Bound 判定。** 每个 accepted token 对 state 做一次 `S ← S⊙decay + u kᵀ`：state
`[N,128,128]` 的每个元素一次乘法、一次加乘，算术强度约 1 FLOP/Byte 量级，从 roofline
看落在访存侧。搬运下界为 state `B*N*128*128*4 = 6.29 MiB` 读一次写一次 = `12.58 MiB`，
加 replay 记录约 `0.38 MiB`，合计约 `12.95 MiB`；按最小值 4.191 µs 折算约 **3.1 TB/s**。
但实测最忙的是 scalar 链，见下表——耗时由访存与标量控制开销共同决定，而不是算术。

**流水线利用率（实测）。** task 级 AIV pipe 计数（10 次采样）：

| 计数器 | 时间 (µs) | 占 `aiv_time` |
| :--- | ---: | ---: |
| `aiv_time` | 4.545 | 100% |
| vector | 1.980 | 43.5% |
| scalar | 2.734 | 60.1% |
| MTE2 | 1.646 | 35.9% |
| MTE3 | 1.274 | 28.4% |

耗时最大的是 scalar 链（60.1%），vector 与两条 DMA 都没有打满。state 的读写流量按
4.191 µs 折算约 3.1 TB/s；由于本套计时是同一组输入重复调用（采集前重置初值），该值
包含片上缓存命中的贡献，应作为「同形状重复调用下的有效带宽」而不是冷 HBM 带宽。
剩余优化空间主要在固定启动开销与 scalar 控制链，而不是算术。

### 3.3 调优过程

以下为开发期在同一形状、同一 harness 下的实测（standalone），用于说明当前
`rows=64, slots=3, cap=16, rdep=2` 的选择依据：

| # | 轮次 | 措施 | 中位数 (µs) | 最小值 (µs) | 结论 |
| --: | :--- | :--- | ---: | ---: | :--- |
| R1 | 基线 | 逐 token 记录、单槽 tile、逐 token 行扫描 | 6.53 | 6.03 | 基线 |
| R2 | 记录双缓冲 | `decay/key/u` tile 改 depth 2 | 5.47 | 4.81 | 采纳 |
| R3 | 行组寄存器驻留 | token 循环移入 4 行循环内，8 个 state 寄存器跨 token 存活 | 5.33 | 4.74 | 采纳 |
| R4 | 融合记录块 + state 预取 | 整段前缀一次搬运；下一 item state 在记录之后入队 | 5.08 | 4.36 | 采纳 |
| R5 | 全预取 | state 与记录同时预取一个 item | 5.13 | 4.45 | 否决（劣于 R4） |
| R6 | 竞态修复 | `accepted==0` 也发起 state 预取；`Rows<64` 的 u tile 物理列宽补齐 | 5.00 | 4.37 | 采纳（正确性修复） |
| R7 | key 每 item 宽化一次 | 行循环内 `UNPK+vcvt` → 单条 `vlds` | 4.90 | 4.18 | 采纳 |
| R8 | `GRP=8` | 8 行寄存器驻留 | 5.34 | 4.69 | 否决（寄存器压力） |
| R9 | `rows` 扫描 | 8/16/32/64/128 | 9.31 / 6.25 / 4.73 / 4.90 / 5.87 | — | 采纳 64 |
| R10 | `cap` 扫描 | 8/16/32/48/64 | 4.69–4.90 | — | 保持 16 |
| R11 | `slots` 扫描 | 2/3/4 | 5.25 / 4.90 / 4.95 | — | 采纳 3 |
| R12 | `rdep` 扫描 | 2/3 | 4.90 / 4.93 | — | 保持 2 |
| R13 | `workers` 扫描 | 16/32/48/56/64 | 16.17 / 9.47 / 7.47 / 4.97 / 4.90 | — | 采纳 64 |
| R14 | 交付形态 | kernel 工厂 → 单模块级 `@pl.jit` + `CommitKey` tiling key | 4.94 | 4.20 | 采纳（性能不变） |

`rows` 的扫描结果就是 `DESIGN.md` 第 3.1 节选用 `64` 的依据：`64` 在每个 token 的常量
开销与负载均衡之间最优，`32` 只在极小 `B*N` 下提供更多工作项。
