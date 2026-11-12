# block_attn_res_update_rms_norm 设计说明

Kimi K3 块注意力残差模块的融合算子（残差更新 + online-softmax 重标定 + RMSNorm）。

本算子把「残差更新 + softmax 重标定 + RMSNorm」四步融合成**一个 kernel 的一趟行处理**：
每个核独占一整行，行的 `p`、`h` 只在 UB 里算完就写回，中间统计量不往返全局内存。
接口与数值契约见 [README.md](README.md)，实测与取舍证据见 [RESULTS.md](RESULTS.md)。

## 1 计算阶段划分

按数据依赖把一行切成三个阶段，融合的边界就是阶段边界：

| 阶段 | 输入（片上） | 输出 | 归约 |
| :--- | :--- | :--- | :--- |
| P1 | `partial`、`delta`、`query` | `p = partial + fp32(delta)`（留在 `partial` tile）、`dot`/`rms` 两个 uniform 标量（存入 `score` tile） | `dot = Σ p·q`、`ssq = Σ p²` 两路，沿 7168 累加 |
| P2 | `p`、`numerator`、`gamma`、`score`（标量参数）、`logit_max`、`exp_sum` | `h`（BF16 取整后以 FP32 存入 `h` tile）、`rstd` 标量 | `ssq_h = Σ fp32(h)²` |
| P3 | `h`（FP32 tile）、`rstd`、`gamma` | `y`（BF16，写回 GM） | 无 |

- `score = dot / rms` 的精确除法在 P1 与 P2 之间的**外层 SIMD 标量核**完成：P1 把 `dot`/`rms`
  以 uniform 寄存器直存 `score` tile，标量核 `pl.getval` 读出后做 IEEE 除法（与向量精确除法
  同为正确舍入），再作为 VF 标量参数传入 P2。这消除了 VF 内 `vf.div(precision=True)` 约
  60 条指令/行的软件展开，是当前实现的主要优化点之一（见 RESULTS §3.3 剩余空间）。
- P1 里 `p` 既写回 GM（契约要求原地更新）又留在 UB 供 P2 使用，所以 `p` 的 GM 写不是
  后续计算的依赖，唯一的依赖是 UB 里的 tile。
- P2 的 `h` 必须先按 BF16 取整再做 `ssq_h`，与上游 `spec.py` 的取整位置一致；如果直接在
  FP32 的 `h` 上归约，RMSNorm 的分母会与参考实现差一个 BF16 量化台阶。取整经
  `astype(BF16, CAST_RINT)` → `astype(FP32)` 在寄存器内完成，`h` 以**取整后的 FP32 值**存 UB，
  P3 直接按 FP32 读回，整条 h 往返不再出现 `de_interleave`/`or_` 洗牌。
- P3 的 `y` 只取整一次（`fp32(h)·rstd·fp32(gamma)` 全在 FP32 完成），避免两次取整叠加；
  y 经 `astype(BF16, ZERO)` + `store_align(PACK_B32)` 连续直出到独立的 `y` tile。

阶段划分的取舍：不切成 P1/P2/P3 三个 kernel。三个 kernel 要把 `p`、`h` 各多搬一遍 GM
（每行 2×28KB + 14KB），而本算子在 `T = 512` 时已经是访存带宽受限（见 RESULTS 的流量表），
融合后每行 GM 流量是 `partial`(R) + `delta`(R) + `numerator`(R) + `query`(R，每核一次) +
`gamma`(R，每核一次) + `p`(W) + `y`(W)。

## 2 核间切分策略

一行由一个核算完，起核数 `min(向量核数, T)`，行按 `pl.get_block_idx()` 轮流分配：

```text
core c 处理 row = c, c + ncore, c + 2·ncore, ...
```

- **为什么不做拆分**：`D = 7168` 的一行只需要 2 次全行归约，归约本身在单核内就是
  28 轮、每轮 4 对寄存器累加（`N_CHUNKS / 2 = 28` 次迭代）；把一行按 `D` 切到多个核需要额外的
  跨核统计量交换。该方案实现并实测过：8 核各持 896 列，经 GM scratch 传
  `dot` / `ssq` / `ssq_h`，两次 `INTER_BLOCK` 屏障，精度完全正确，但两次 GM 往返加屏障的
  开销超过多核省下的向量时间，已移除。
- **`T` 与核数不整除**：轮流分配天然覆盖，没有尾块；`T < 核数` 时多余的核不启（起核数
  就是 `T`），不会出现空转核。
- **`T = 0`**：host 侧直接返回空张量，不启核。

## 3 核内切分策略

### 3.1 基本块的选择

`D = 7168 = 112 × 64`，按 64 lane 的 FP32 寄存器宽度切成 112 个基础块，VF 里每轮处理
两个 128 段（256 元素），共 `N_CHUNKS // 2 = 28` 轮：

```text
for c in pl.range(0, N_CHUNKS / 2):        # 每轮 256 元素，展开两次 128
    base = c * 256
    ...  load(partial[base : base+128]) + delta_unpk → p，累加 dot/ssq
    ...  load(partial[base+128 : base+256]) 同上
```

一轮内展开两个 128 段，是为了让同一轮的寄存器对（`dot0/dot1`、`ssq0/ssq1`）各自独立
累加，最后合并一次，减少归约树的串行深度。`delta` 用 `LoadDist.UNPK_B16` 直接解包成
FP32 容器再 `astype`，避免额外的中间 tile。

### 3.2 核内 Buffer 划分

UB 静态划分，32 B 对齐，峰值 229632 B / 253952 B：

| Buffer | 大小（B） | 槽数 | 用途 |
| :--- | ---: | :--- | :--- |
| `partial` | 28672 | 2 | `p`，P1 写、P2 读，同时承担 GM 原地更新 |
| `delta` | 14336 | 2 | BF16 增量 |
| `numerator` | 28672 | 2 | 归一化分子 |
| `h` | 28672 | 1 | BF16 取整后的 `h` 以 FP32 存储，P2 写、P3 读 |
| `y` | 14336 | 1 | P3 的 `PACK_B32` 输出，独立于 `h`（dtype 不同，不能复用同一 tile） |
| `query` | 28672 | 1 | 所有行共享，每核搬一次 |
| `gamma` | 14336 | 1 | 所有行共享，每核搬一次（`Path = 0` 时故意延后到首行之后搬，避免压在 P1 的搬入前缀上） |
| `score` | 256 | 1 | P1 存 `dot`（offset 0）与 `rms`（offset 32）两个 uniform 标量 |

双槽只给需要跨行重叠的三个大张量；`h`/`y` 单槽是因为同一行的 P2→P3 是纯行内依赖，
跨行重叠由 `partial`/`delta` 的双槽提供。`h` 以 FP32（28672 B）而非 BF16 存储：消除打包/
解包洗牌后，P3 直接按 64 lane FP32 `load_align` 读回，指令数每行净省且不再依赖
`de_interleave` 布局约定。

### 3.3 寄存器布局

- `dot0/dot1`、`ssq0/ssq1` 四对累加器承载两路归约，每轮各累加 128 元素，循环末
  `add` 合并成一对再 `reduce_sum`。
- P1 的归约结果 `dot`/`rms` 以 `vf.full` 广播成 uniform 寄存器后整条直存 `score` tile；
  `score = dot / rms` 由外层 SIMD 核读回后按标量精确除法完成（见 §1），VF 内不再出现除法
  展开。`Path = 0`（单体 VF）没有可用的标量核间隙，`score` 的除法仍留在 VF 内。
- 行元数据 `logit_max` / `exp_sum` 用 `pl.getval` 在**外层 SIMD kernel** 读成标量再传给
  VF，因此不需要为它们建 tile、做搬入或记账 valid-shape。

## 4 接口选择

- 只有一个模块级 `@pl.jit(auto_mutex=True, tiling_key=RowPath)` 入口：
  `block_attn_res_update_rms_norm_kernel`。没有 kernel 工厂、闭包或按形状生成源码。
- 静态分叉只用一个 1 bit 的 tiling key 字段 `Path ∈ {0, 1}`，由 host 侧
  `select_path(T)`（`T < 3 → 0`，否则 `1`）决定；两条路径的算术、UB 布局、起核方式完全
  相同，差异只是「一次 VF 还是两次 VF」。字段若删除，`T ≥ 3` 会退到单体方案，实测
  小 `T` 场景下的 VF 边界开销消失但 `T ≥ 3` 的 `p` 写回无法与计算重叠（RESULTS §2.2
  的路径交叉实测可见该差异）。
- 同步用 `auto_mutex=True` 的 tile group：同一 tile 的搬入、VF 访问、搬出由框架自动插
  事件，没有手写 `set_flag/wait_flag`。tile group 的 `mutex_ids` 保持在 0~9 的小范围内
  （实测把 id 用到 11 以上会超出每核 mutex 容量，表现为 VF 写回与 GM 搬出的顺序丢失）。
- 行数、隐藏维都由张量元数据给出（`pl.DYNAMIC`），host 侧只传起核数、`Path` 和两个
  epsilon；没有把某个形状表编进 dispatch。

## 5 流水设计

### 5.1 `p` 的写回不挡计算

`p` 必须在 `T ≥ 3` 时尽早写回 GM，才能让后续行的搬入与当前行的 P2/P3 重叠：

- `Path = 0`（`T < 3`）：P1/P2/P3 在同一个 VF 体里顺序执行，`p` 与 `y` 在 VF 结束后一起
  搬出。小 `T` 时没有其他行可重叠，少一次 VF 边界更划算。
- `Path = 1`（`T ≥ 3`）：P1 的 VF 一结束就发 `p` 的搬出，然后才搬 `numerator`、进 P2/P3
  的 VF。这样 `p` 的 GM 写与后续计算时间重叠，实测 `T ≥ 3` 反而比单体方案快。

### 5.2 三张双槽张量让行与行重叠

`partial`/`delta`/`numerator` 各两槽，`auto_mutex` 在第 k 行用 A 槽计算时允许第 k+1 行
往 B 槽搬入，于是 GM 搬入、向量计算、GM 搬出三者在稳态下同时进行。槽深只取 2 而不是
更多：峰值 UB 已经是 229632 / 253952 B，再加一槽 `partial` 就超出（实测把 `h`/`score`
组加深到双槽也无收益，已回退）。

### 5.3 共享张量只搬一次

`query`、`gamma` 与行无关，每个核在首行之前（`query`）或首行之中（`gamma`，`Path = 0`）
搬一次，之后所有行复用同一个 tile，行循环里不再出现这两个张量的搬入。

### 5.4 同步：`auto_mutex`、`mem_bar` 与分支隔离

同一次调用里的依赖分三层，机制各不相同：

| 层次 | 机制 | 覆盖的依赖 |
| :--- | :--- | :--- |
| DMA ↔ VF | `@pl.jit(auto_mutex=True)` + tile group 的 mutex id（0~10） | 搬入→VF 读、VF 写→搬出、槽复用 |
| VF 内 RAW | `vf.mem_bar(mode=pl.MemBarMode.VST_VLD)` | 阶段 1 写的 `p`/`score` → 阶段 2 读；阶段 2 写的 `h` → 阶段 3 读 |
| 核间 | 无 | 每行由单核完成，不需要跨核同步 |

VF 内「写后读」**没有**自动顺序保证（框架文档记录了实测：同址写后读会偶发读到旧值，
5 次同样运行 2 次失败、错误位置稳定但数值随机），所以两条真实 RAW 各用一次
`vf.mem_bar(mode=pl.MemBarMode.VST_VLD)` 排序：

| 位置 | 覆盖的依赖 |
| :--- | :--- |
| `phase23_vf` 开头、`process_row_vf` 阶段 2 之前 | 阶段 1 写回的 `p` 在本 VF 内被重新读入（`Path = 1` 的 `score` 已改为标量核直传，不再走 tile） |
| `phase23_vf` 阶段 3 之前、`process_row_vf` 阶段 3 之前 | 阶段 2 写出的 `h`（FP32）在本 VF 内被重新读入，做第二次归约与 `y` |

要点：

- 用的是最小范围的 `VST_VLD`（只排序 vector store → vector load），不是 `VV_ALL`；一次
  barrier 覆盖它之后的所有相关读，所以每行只有 2 次，且都在 chunk 循环之外，不按元素计。
- 它只排序本核向量流水内部的访存，不涉及 MTE2/MTE3/MTE1，不产生跨核同步，因此不会破坏
  「搬入—计算—搬出」的流水重叠。
- 实测代价为 0：去掉这些 barrier 后耗时差异在噪声内。但去掉后用例仍然通过**不能**证明安全
  （同址写后读的竞争是偶发的），因此保留。
- 另一条思路是把 `h` 留在寄存器里避免这次往返，但 7168 个 BF16 = 14336 B，超过向量寄存器
  容量，写回 UB 再读是这条两遍归约结构固有的。
- `mutex id` 必须留在小范围（这里 0~10）：实测把 id 用到 11 以上会超出每核 mutex 容量，
  表现为 VF 写回与 GM 搬出的顺序丢失。

### 5.5 分支代价被隔离

`Path` 是编译期常量，两条路径各自编译，VF 体里没有运行期 `if`：`tokens`、行号、
`Path` 都不进 VF 循环体，硬件循环与地址生成器可以正常 lowering。
