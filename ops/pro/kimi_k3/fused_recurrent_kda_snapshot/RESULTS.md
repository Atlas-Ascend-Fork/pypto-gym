# fused_recurrent_kda_snapshot 结果说明

本文给出 `fused_recurrent_kda_snapshot` 的用例、精度标准与性能结果。文中标「实测」的
数据是在设备上直接测得的；有效带宽等由实测时间与已知数据量按文中算式导出，可逐项复算。

测试环境：

| 项目 | 取值 |
| :--- | :--- |
| 设备 | Ascend 950PR（`Ascend950PR_9589`，`NpuArch=3510`），单卡 |
| 计算单元 | cube 32 核 / vector 64 核（`[SoCInfo] cube_core_cnt` / `vector_core_cnt`） |
| 片上缓存 | 每个向量核 253952 字节（248 KiB）；L2 134217728 字节（128 MiB） |
| 软件 | CANN 9.3.0、PyPTO-Pro（pypto 0.2.1）、torch 2.13.0、torch_npu 2.13.0rc1 |

## 1 用例

基准形状 `B=16, N=6, S=8, D=128`、BSND、FP32 state、`accepted=4`（性能基准形状）。
全部用例在 `tests/test_fused_recurrent_kda_snapshot.py` 里，一个用例对应一个 `test_xxx`
函数。

| case | 覆盖 |
| :--- | :--- |
| `test_layouts` | BSND / BNSD / TND × `(B,S) = (1,8) (2,8) (2,1) (3,5)`，含不均匀 TND `cu_seqlens` |
| `test_state_indices_and_accepted` | 非连续置换槽位 + `accepted ∈ {0,5}`，三种布局 |
| `test_bf16_state` | 真实 `torch.bfloat16` state pool |
| `test_row_items` | `rows ∈ {32, 64}` |
| `test_single_token_sequences` | `S=1`（BSND 与 TND） |
| `test_max_heads_and_accepted_edges` | `N=96`；`accepted ∈ {0, S}` 且池大于 token 数 |
| `test_long_sequence` | `S=33` |
| `test_worker_sweep` | workers `1 / 3 / 13 / 56 / 64` |
| `test_head_dim_blocks` | `N=17`、`N=33`（非整除 head 数） |
| `test_smoke_bsnd` | 最小冒烟 |

共 24 例，`TILE_FWK_DEVICE_ID=0` 下全通过。

## 2 精度标准

判据见 `tests/assets/gen_golden.py` 的 `check_precision`，即逐元素

```text
|actual - expected| <= atol + rtol * |expected|
```

`state` 与 `out` 均取 `atol = rtol = 5e-3`（FP32 pool）/ `1e-2`（BF16 pool）。
参考实现为 `tests/assets/gen_golden.py` 的单一 torch 实现：CPU 模式用 FP64、NPU 模式用
FP32；不复用算子代码。

### 2.1 结果

**24/24 PASS**。基准形状（`B=16, N=6, S=8, D=128`、BSND、FP32 state、`accepted=4`）
的带下限相对误差：

| 量 | 误差 |
| :--- | --: |
| `state` | 1.8e-07 |
| `out` | 1.9e-04 |

### 2.2 形状与边界行为

- **布局等价**：BSND / BNSD / TND 走同一 kernel body，仅行 stride 不同；不均匀
  `cu_seqlens` 与 `S=1` 序列均通过。
- **槽位与接受计数**：非连续置换槽位、池大于 token 数、`accepted ∈ {0, S}` 的组合通过；
  只比较被映射的活跃槽位。
- **BF16 pool**：工作状态保持 FP32，快照收窄为 BF16，不截断本轮后续 token 使用的状态；
  容差放宽到 `1e-2` 并按真实 `torch.bfloat16` 张量验证。
- **重构建等价**：拆 pass + 快照解耦相对重构前实现**逐位一致**——同一批输入下 `state`
  与 `out` 均 `torch.equal`（FP32/BF16 pool × `rows` 32/64 × `slots` 3/4），说明这两处
  改动只改调度、不改算术。

## 3 性能结果

本算子不使用 Cube，单次调用只下发一个 kernel，因此本算子的性能即该 kernel 的耗时。
对比对象是 `tests/assets/gen_golden.py` 的 NPU 模式（torch + torch_npu），两侧用完全
相同的 harness 与输入。

**采集方法**

| 项 | 做法 |
| :--- | :--- |
| 计时 | torch_npu profiler `kernel_details.csv` 的 `Duration(us)`，取一次调用内**全部 device kernel 时长之和**；不含 host 派发与 kernel 间间隙 |
| 对比对象 | Golden NPU：torch + torch_npu，normalize 与激活提到 token 循环外、batch/head 向量化，一次调用 170 个 device kernel；本实现：一次调用 1 个 kernel |
| 统计 | 单卡、单进程；两侧都 10 次预热 + 30 次采样 × 3 轮；采集前把输入 state 重置到同一初值；每轮取中位数与最小值 |
| 达成比 | `Golden 一次调用的 device 时间 / 本实现 kernel Task Duration` |

### 3.1 绝对性能结果

| 实现 | 最小值 (µs) | 中位数 (µs) |
| :--- | ---: | ---: |
| Golden NPU（170 个 device kernel / 次） | 538.825 | 544.728 |
| 本实现（1 个 kernel / 次） | **16.638** | **17.338** |
| 达成比（Golden device 时间 / 本实现 kernel） | **32.4x** | **31.4x** |

Golden 的 3 轮中位数为 544.742 / 541.921 / 544.728 µs，本实现为 17.278 / 17.338 /
17.416 µs。两侧都只统计 device kernel 时间，比值是同一口径下的 device 时间比；同一组
输入上，本实现与 Golden NPU 的 `state` 最大绝对误差 1.79e-07、`out` 1.22e-04。

**Golden NPU 的构成**

一次调用 170 个 device kernel，按 Op 统计（10 次采样平均）：

| op | count | 平均 (µs) | 合计 (µs) |
| :--- | ---: | ---: | ---: |
| Mul | 37 | 4.381 | 162.1 |
| BatchMatMulV3 | 16 | 5.207 | 83.3 |
| Slice | 34 | 1.909 | 64.9 |
| Add | 13 | 4.172 | 54.2 |
| ScatterUpdate | 8 | 5.351 | 42.8 |
| Cast | 22 | 1.608 | 35.4 |
| Index | 10 | 2.938 | 29.4 |
| ViewCopy | 8 | 2.487 | 19.9 |
| Sub | 9 | 2.037 | 18.3 |
| RealDiv | 2 | 6.441 | 12.9 |
| 其余 6 类（ReduceSum / Sigmoid / Exp / Range / Sqrt / ClipByValueV2） | 11 | — | 23.0 |

合计约 546 µs。Golden 已经把与递推无关的 normalize 与激活提到 token 循环外，循环内
只剩 rank-1 更新、输出点积和快照写回；每 token 仍有约 21 个小算子，状态张量
`[B, N, 128, 128]` 被反复读写。本实现把同样的数学融合为单次下发的一个 kernel
（17.414 µs），差距来自小算子的固定开销与重复搬运，而不是算术效率。

### 3.2 理论评估对比

**Bound 判定。** 每个 token 对 state 做一次 `S ← S⊙decay + u kᵀ` 并写回快照，算术强度
约 1 FLOP/Byte 量级，属**访存与向量指令混合受限**。流量口径：初态读
`B*N*128*128*4 = 6.29 MB`，逐 token 快照写 `B*S*N*128*128*4 = 50.33 MB`，输入
`mixedqkv` / `g` / `beta` 约 0.79 MB、`out` 约 0.39 MB，合计约 57.8 MB；按最小值
16.717 µs 折算约 **3.46 TB/s**。

**流水线利用率（实测）。** task 级 AIV pipe 计数（10 次采样）：

| 计数器 | 时间 (µs) | 占 `aiv_time` |
| :--- | ---: | ---: |
| `aiv_time` | 16.965 | 100% |
| vector | 14.243 | 84.1% |
| scalar | 6.799 | 40.2% |
| MTE2 | 3.738 | 22.1% |
| MTE3 | 8.461 | 49.7% |

vector pipe 占用最高（84.1%），是主瓶颈；MTE3（49.7%）对应逐 token 的快照写回；scalar
40.2%。由于本套计时是同一组输入重复调用（采集前重置初值），3.46 TB/s 包含片上缓存
命中的贡献，应作为「同形状重复调用下的有效带宽」而不是冷 HBM 带宽。该算子同时接近
vector 与带宽两条上界，剩余空间主要在减少快照写回流量或压缩行体指令。

### 3.3 渐进过程

以下为开发期的历史数据，同一形状、同一 harness 的实测中位数：

| 阶段 | 措施 | 中位数 (µs) |
| :--- | :--- | ---: |
| 原始最终版 | 融合行体 + 延迟写 + `workers=56` | 28.4 |
| + 负载均衡 | worker 上限 56 → 64（`192 = 16*6*2` 恰好 3 个/worker） | 22.10 |
| + 拆分行体 + 快照解耦 | `_step` → `_step_dot`(4 行 unroll) / `_fold_u` / `_step_upd(_snap)`；独立双缓冲快照 tile 即时 DMA | **17.26** |

各步骤的判别性证据：

- **worker 数**：`workers` 扫描 48/56/60/64 在 `rows=64` 下为 29.08 / 27.81 / 27.71 /
  **22.10** µs，`64` 的突变来自工作项整除（`192/64 = 3`，`/56`、`/60` 为 4）。
- **`rows` 扫描**：`32` 为 26.02 µs、`64` 为 22.10 µs，故默认 `64`。
- **`slots` 扫描**：`2 / 3 / 4` 为 28.30 / **22.10** / 22.71 µs，`2` 会让快照搬运与递推
  在同一槽上互卡。
- **拆 pass 与解耦**：重构前后在同一批输入下 `state` 与 `out` 均逐位一致（`torch.equal`，
  覆盖 FP32/BF16 pool × `rows` 32/64 × `slots` 3/4），即收益完全来自调度而非算术；
  行体指令与快照搬运的变化见 §3.2 的 pipe 计数。
