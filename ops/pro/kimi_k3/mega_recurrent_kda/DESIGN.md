# mega_recurrent_kda 设计说明

> 算子本体：`op_kernel/mega_recurrent_kda.py`；参考实现与精度比较器：`tests/assets/`。
> 运行平台：Ascend 950（平台信息里 `soc_version = 'DAV_3510'`），cube（矩阵计算单元）
> 32 核 / vector（向量计算单元）64 核，UB（片上统一缓冲）容量 256 KiB。
> 默认配额 32 AIC / 64 AIV（AIV = 2 × AIC），支持 8 / 16 / 24 / 28 / 32 AIC。

## 1 计算阶段划分

算子的全部工作由**一个** `@pl.jit` 混合 kernel（`mega_recurrent_kda_kernel`）完成，按 3 个
执行段组织，段间用跨核事件（`set_cross_core` / `wait_cross_core`）衔接。Cube 段跑在 AIC 上，
Vector 段跑在 AIV 上，两段在同一 kernel 内交替：

```text
section_cube   （AIC）
    qkv   = hidden @ Wqkv                       # [Mpad, 3P]，Wqkv 为 [H, 3P]
    proj2 = hidden @ [Wa | Wbeta | Wgate]          # [Mpad, 256+P]，三块权重按 N-tile 选择
    bar_fix(); sync_all(AIC_ONLY); set_cross_core(FIX, 0)     # qkv 就绪
    raw_decay = proj2[:, :128] @ Wb             # [Mpad, P]，Wb 为 [128, P]
    bar_fix(); set_cross_core(FIX, 1)                          # raw_decay 就绪

section_vector（AIV，work item = (batch, head)，共 B·N 个）
    wait_cross_core(MTE2, 0)
    Conv+QK 阶段：因果 Conv1D+SiLU + Q/K RMS norm → qg / kg / vg（[Mpad, P] FP32）
    bar_mte3(); sync_all(AIV_ONLY); wait_cross_core(MTE2, 1)
    递归组 0：heads [0, N/2) 的 activation + KDA 递归 + RMSNorm/gate → gated
               bar_mte3(); sync_all(AIV_ONLY); set_cross_core(MTE3, 2)
    递归组 1：heads [N/2, N) 同上 → gated
               bar_mte3(); sync_all(AIV_ONLY); set_cross_core(MTE3, 3)

section_cube   （AIC）
    wait_cross_core(MTE2, 2)
    n-tile 循环，K ∈ [0, k_half)          → out_pad（普通 store）
    wait_cross_core(MTE2, 3)
    n-tile 循环，K ∈ [k_half, k_gated)    → out_pad（AtomicAdd 累加）
```

之所以合成一个 kernel：拆开做时，`qkv`、`proj2`、`raw_decay`、`convolved`、`qg/kg/vg`、`o`、
`gated` 每一块都要在全局内存与片上之间往返一次，递归状态还要额外读写；融合后中间量
`gexp / beta_act / gate_act / qhat / khat / vconv` 常驻 UB，跨段经全局内存的只剩真正需要
跨计算单元交接的 `qkv / proj2 / raw_decay / qg / kg / vg / gated`。本征递归（Vector）与投影
（Cube）又分属两种计算单元，放在同一个 kernel 里才能用事件把两段接起来，而不是两次 launch。

kernel 本体只保留形状推导、tile 声明和**跨核同步**；计算段各自成函数（DSL 在解析期内联，
生成同一份 IR）：

- `_cube_front_proj`：qkv/proj2 的派发器；7 个分支函数见 2.1（`_cube_qkv_fused4p`、
  `_cube_qkv_n512`、`_cube_qkv_n128`、`_cube_qkv_n384`、`_cube_qkv_n256`、
  `_cube_proj2_1tile`、`_cube_proj2_2tile`）。
- `_cube_decay_proj`：`raw_decay = proj2[:, :128] @ Wb`。
- `_vector_conv_qknorm`：因果 Conv1D+SiLU + Q/K RMSNorm。
- `_vector_kda_recur`：一组 head 的激活 + KDA 递归 + RMSNorm/gate，调用两次。
- `_cube_out_proj`：输出投影；`accum` 参数区分两段 K（第一段普通 store、第二段 AtomicAdd）。

Host 侧由 `_validate_arguments`（参数校验与 `MTile`/`Mpad` 推导）、
`mega_recurrent_kda_pad_input`（A 操作数 padding）、`_pack_conv_weight`（Conv 权重重排）
分工。`set/wait_cross_core` 与段间屏障（AIV 用 `bar_mte3/sync_all`，Cube 用 `bar_fix/sync_all`）
都写在 kernel 本体，只有
`set_cross_core(FIX, 0)` 保留在 `_cube_front_proj` 的 qkv 之后——它是 AIV Conv+QK 阶段的提前放行点，
挪到 proj2 之后会让整个 Vector 段推迟约一个 proj2 的时长。

代价是**递归在 Vector 上逐 token 串行**：`S ← S ⊙ gexp`、`S ← S + (β⊙r)⊗k` 两步依赖上一个
token 的状态，而每个 token 都要把 `128×128` 的状态写回 `recurrent_state`。这部分是硬契约，
也是当前的主要耗时来源。

阶段内部的分工：

```text
Cube 段    qkv / proj2 / raw_decay / 输出投影：NZ 权重 GM→L1→L0 的标准 MMA 链
Vector 段  Conv+QK 阶段：Conv1D(K=4)+SiLU → Q/K RMS 归一化，产出 qg / kg / vg
           递归阶段：decay 合成 gexp、beta/gate 激活、KDA 递归、RMSNorm、输出门控
```

**Conv1D+SiLU**（`_conv_silu_vf`）：每 `(batch, head)` 每个 slice 用 `16×128` 的窗口（3 行
history + 8 行 raw）在 UB 内做 K=4 逐通道卷积 + SiLU，8 个 token 一次算完。四路 tap 权重以 BF16
直接 `UNPK_B16` 搬进寄存器，不再先 cast 成一块 FP32 tile。三个 slice 的面板分别留在
`b_oq/b_ok/b_ov`，由 `_qknorm_vf` 就地做 Q/K RMS 归一化（Q 顺带乘 `scale`），因此没有
`convolved` 全局缓冲的往返。

**KDA 递归**（`_kda_ksum_vf` / `_kda_delta_vf` / `_kda_update_vf`）：状态 `[128 value, 128 key]`
FP32 常驻 UB，逐 token 执行

```text
ksum[r]  = Σ_c S[r,c]·(gexp⊙k)[c]
delta[r] = (v[r] − ksum[r])·β
S[r,c]  ← S[r,c]·gexp[c] + delta[r]·k[c]
o[r]     = Σ_c S[r,c]·q[c]
```

decay / beta / gate 三路激活是三条独立的 `[8,128]` 逐元素链，合并在同一个 `_act_vf` 的一次
行循环里：三块输入各只做一次 `load_align`（BF16 用 `UNPK_B16` 顺带展开），整条链留在寄存器里，
只 store 三块 FP32 结果；`exp(a_log[h])` 用 `vf.full(pl.getval(a_log, h))` + `vf.exp` 在寄存器里
得到。此前用 `pl.*` tile 语义串联（cast/expands/exp/mul/neg/exp/add/recip/muls/exp），每一步都要
把中间 tile 写回 UB。

`ksum` 与 `o` 的行内归约用 `vf.reduce_sum` 加 `store_align(dist=FIRST_ELEMENT_B32)` 落成标量，
`delta` 用 `vf.full` 广播，`S` 的更新用 `vf.load_align(dist=BRC)` 把 `delta[r]` 广播到整行。
每个 token 的 `o` 行由 `_kda_update_vf` 用行偏移（`store_align` 的 `out_off+dv`）**直写进
`[8,128]` UB tile**，RMSNorm 直接读该 UB tile，省掉 `o` 的全局内存往返与逐 work item 的
`bar_all`。循环不变量的寄存器都在循环外只算一次。

RMSNorm + `gamma` + 输出门控由 `_rmsgate_vf` 一次做完：先做一趟行内平方和（`vf.reduce_sum`
落到 `[1,8]` 标量行），再用 `load_align(BRC)` 把每 token 的 `1/sqrt(mean+eps)` 广播回整行，
乘 `gamma`（列广播，寄存器常驻）与门控，同一块 UB tile 进出；只有最后 FP32→BF16 的打包留在
VF 外（该 SDK 的收窄 `vf.astype` 需要未声明的 `CastLayout`）。

数据相关的分支不进入向量函数：`conv_state_indices[b] == 0` 的空槽判断在 kernel 主体里用标量
完成，直通分支只把卷积输出置 0、不改动 `conv_state`，向量函数内部保持直线代码。

最后一步是编译期与运行时的分工。`S`、`D`、`CONV_K`、`BETA_OFF`、`GATE_OFF` 是编译期常量，
它们决定权重面板宽度、UB 字节布局与归一化的立即数；`B`、`N`、`P` 是运行时量。唯一的编译期
形状特化是 `MTile`（`@pl.jit(tiling_key=_MegaTilingKey)`，字段 `MTile ∈ {32, 64, 128}`）：
宿主按 `rows = B·S` 选 32（`rows ≤ 32`）/ 64（`rows ≤ 64`）/ 128，A 操作数由调用方用
`mega_recurrent_kda_pad_input` 补到 `Mpad = ceil(rows / MTile)·MTile` 行后直接传入。
小 batch 不再按 128 行空算投影（`B·S = 32 / 64` 时省掉 4× / 2× 的冗余行），前端权重面板
单独用 `ct_w[128,128]`，不随 M 缩小。pad 行的输出为 0，返回前切掉；wrapper 只校验
`hidden_states` 的形状，不做 pad 也不做拷贝。

## 2 核间切分策略

**Cube 侧**

- 支持的 AIC 配额固定为 **8 / 16 / 24 / 28 / 32**，默认取设备上可用的最大配额
  （`pypto_pro.runtime.platform.get_platform_info().core_num`：Ascend 950 有多种型号，28 核 SKU 默认 28，32 核及以上设备默认 32），可用 `MEGA_KDA_CUBE_CORES` 在集合内覆盖；
  其它值在 host 侧报 `ValueError`。AIV 恒为其 2 倍。
- 投影任务在 `m_tiles × N-tile` 的二维网格上按 **n-major** 映射：`i = t % m_tiles`、
  `j = t // m_tiles`。同一权重面板连续喂给所有 m-tile，使 `3P×H` 的 qkv 权重（N=24 时约
  132 MB / 126 MiB）能在 128 MB L2 内跨 m-tile 复用；m-major 时每个 m-tile 都要重读一遍。
- 前端投影的分支很多，但只由 5 个运行时量决定：`m_tiles`、`nc_c`（AIC 核数）、
  `nt_q = 3N`（qkv 的 N-tile 数）、`nt_p = 2+N`（proj2 的 N-tile 数）、
  `nt_all = nt_q + nt_p`。判定按**优先级**从上到下命中，见 2.1。

### 2.1 前端投影判定树

| 优先级 | 条件 | 采用 | 依据 |
| ---: | :--- | :--- | :--- |
| 1 | `m_tiles==1`、`nc_c≥28`、`nt_all≤nc_c` | **4P 融合**：qkv 与 proj2 共用一个 A 面板和一个任务池 | 单 M tile 的任务本来就少；融合后 N=6 的 18+8 个 N-tile 变成 26 个并发任务，尽量铺满 32 AIC；`nt_all≤nc_c` 保证至少一核一任务、不串行 |
| 2 | 否则 qkv：`nc_c≥28`、`m_tiles==1`、`nt_q%4==0`、`n4_q·2≥nc_c` | **N512**（R=4） | 单 M tile 的宽 N 形状；4 块 `[64,128]` B 面板正好 64 KiB L0B、4 个累加器占满 256 KiB L0C；A 复用 ×4 省 3/4 的 A 字节；任务数至少约核数一半才摊得开更长的 MMA 链 |
| 3 | qkv：`nc_c≥28`、`m_tiles==1`、`((nt_q+1)//2)·2 < nc_c`、`nt_q·2 ≥ nc_c` | **N128**（R=1） | N256 的任务数不到核数，核会空转；该形状延迟受限而非字节受限，一 tile 一任务把任务数翻倍 |
| 4 | qkv：`n384`（见下） | **N384**（R=3） | 一块 A 喂 3 块 B；KS=64 让 3×`[64,128]` B 同驻 48 KiB L0B、3 个累加器 192 KiB L0C；A 复用 ×3 |
| 5 | qkv：其余 | **N256**（R=2） | 默认档：2 块 B + 2 个累加器，字节与任务数平衡 |
| 6 | proj2：`m_tiles==1`、`nc_c≥28`、`n2_p·4≤nc_c` | **1 N-tile/任务** | 2-tile 任务太少（N=6/12 单 M）会把大多数 AIC 空着；1 tile/任务翻倍。N=24 单 M 仍用 2 tile（A 流量更大） |
| 7 | proj2：其余 | **2 N-tile/任务** | 与 qkv 的 N256 对齐，A 面板复用 |

上表每个分支对应一个函数：`_cube_qkv_fused4p` / `_cube_qkv_n512` / `_cube_qkv_n128` /
`_cube_qkv_n384` / `_cube_qkv_n256` / `_cube_proj2_1tile` / `_cube_proj2_2tile`；
`_cube_front_proj` 只负责算判定量、派发和事件 0 前后的同步。

`n384` 的两条依据与一条例外：

- `m_tiles≥2 且 m_tiles·n3_q ≥ nc_c`：任务足够铺满网格，N384 的 A 复用收益能兑现；
- 或 `nc_c≤16 且 n3_q·2 ≥ nc_c`：低核时 Cube 更偏 MTE2 字节受限，即使单 M tile 也划算；但要
  求约一半核能拿到任务，避免 (16,6)@16 这类占核不足的反例；
- 例外：`nc_c≥28` 且 `nt_q≥72`（N=24）且 `m_tiles≥2` 时禁用 N384——满占核时 3 块常驻 B
  面板与更长的 MMA 链输给额外 L2 流量。

**为什么 R ≤ 4、为什么 N384/N512 要 KS=64**：一个累加器是 `[MTile,128]` FP32 = 64 KiB
（MTile=128），L0C 256 KiB 最多放 4 个，所以 R 上限是 4。B 面板若保持 128 宽，R=3 需要
3×32 KiB = 96 KiB > L0B 64 KiB；N384/N512 把 K 切成两段 64，每块 B 变成 16 KiB：3 块 48 KiB、
4 块正好 64 KiB。这就是分支里 `for s in range(0, 2)` 的来源。

**为什么是运行时分支而不是编译期特化**：`m_tiles`、`nc_c`、`nt_*` 都是运行时量；kernel 只按
`MTile` 三档编译。每个核进入时只做几条标量比较，循环边界仍是运行时值，换形状不需要重编译。

- 输出投影（cube2）按输出通道切成 128 宽的 n-tile（`H = 7168` → 56 个），K 切成两段，
  两段分别等待递归阶段组 0 / 组 1 的事件。

**Vector 侧**

- work item 是 `(batch, head)`，共 `B·N` 个，均匀铺到全部 AIV（AIV = 2 × AIC）上。
- 递归阶段按 head 对半拆成两组：组 0 处理 `heads [0, N/2)`，组 1 处理 `[N/2, N)`，每组的
  work item 数为 `B·N/2`。两组之间递归体逐字节相同，只有 work-item 映射不同（head-major：
  组 0 `h = work // batch`，组 1 `h = N/2 + work // batch`）。
- 拆两组的目的是让组 0 先完成、先唤醒 cube2 的上半 K，从而与组 1 重叠（见第 5 节）。

各用例的实际切分（`rows = B·S`，`qkv` 的 N-tile 数 `nt_q = 3N`，Vector work item 数 `B·N`）：

| 用例 | B | N | `rows` | `MTile` | `Mpad` | `m_tiles` | `nt_q` | Vector 工作项 | 递归阶段每组 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tp16_b16` | 16 | 6 | 128 | 128 | 128 | 1 | 18 | 96 | 48 |
| `tp16_b32` | 32 | 6 | 256 | 128 | 256 | 2 | 18 | 192 | 96 |
| `tp16_b48` | 48 | 6 | 384 | 128 | 384 | 3 | 18 | 288 | 144 |
| `tp4_short_b4` | 4 | 24 | 32 | 32 | 32 | 1 | 72 | 96 | 48 |
| `tp4_tail_b12` | 12 | 24 | 96 | 128 | 128 | 1 | 72 | 288 | 144 |
| `tp4_main_b32` | 32 | 24 | 256 | 128 | 256 | 2 | 72 | 768 | 384 |
| `tp8_short_b8` | 8 | 12 | 64 | 64 | 64 | 1 | 36 | 96 | 48 |
| `tp8_tail_b24` | 24 | 12 | 192 | 128 | 256 | 2 | 36 | 288 | 144 |
| `tp8_main_b64` | 64 | 12 | 512 | 128 | 512 | 4 | 36 | 768 | 384 |

`tp4_main_b32` 与 `tp8_main_b64` 的 Vector 工作项最多（768），也最接近把全部 AIV 喂满；
`tp4_short_b4`、`tp8_short_b8`、`tp16_b16` 只有 96 个，属于「每核 2 个」的形态。

## 3 核内切分策略

### 3.1 基本块的选择

| 维度 | 取值 | 选择依据 |
| :--- | :--- | :--- |
| Cube 基本块 | `[MTile, N-tile]` 的 MMA 面板 | 一块 A 面板在 L1 常驻，连续喂给同一任务池里的多个 N-tile |
| `MTile` | 32 / 64 / 128（编译期 tiling key） | 按 `rows = B·S` 选最小档，避免小 batch 按 128 行空算 |
| 前端 N-tile | 128 / 256 / 384 / 512 | 由形状与核数运行时分档，见第 2 节 |
| 输出投影 N-tile | 128 | 输出通道 `H = 7168` 切成 56 个 128 宽 tile |
| K 子切 | 64（N384 的 KS=64） | 3 块 `[64,128]` B 面板共 48 KiB，正好同驻 64 KiB L0B |
| Vector 基本块 | 一个 `(batch, head)` work item | 该 head 的 `128×128` 状态、`gexp`、`k`、`q`、`v` 全在片上，逐 token 只需搬一行 |
| 状态切分 | `[128 value, 128 key]` FP32，按 64 lane 分两半 | 一个 FP32 向量寄存器 64 lane，128 列正好两半 |
| Conv 窗口 | `16×128` BF16（3 history + 8 raw） | 8 个 token 一次算完，K=4 的 3 行历史直接在窗口里 |
| 递归步长 | 1 个 token | 状态更新有跨 token 的写后读依赖，无法并行 |

`MTile` 与 N-tile 都是编译期或运行时分档量：`MTile` 进 tiling key，N-tile 由运行时形状选择
（同一份代码、同一份已编译 kernel）。`S = 8` 与 `D = 128` 是编译期常量，`B`、`N` 是运行时量。

### 3.2 核内 Buffer 划分

| 缓冲 | 容量 / 形状 | 位置 | 用途 |
| :--- | :--- | :--- | :--- |
| `state` | `128×128` FP32 = 64 KiB（`c_st` @0x0A000） | UB | KDA 递归状态，整个 work item 期间常驻 |
| `snap` | `128×128` FP32 × **2 槽** = 128 KiB（`c_snap` @0x1A000/0x2A000） | UB | 状态写回快照，深度 2 环形槽，支撑「延迟一拍」写回 |
| conv 窗口 | `16×128` BF16 = 4 KiB | UB | 3 行 history + 8 行 raw + 对齐 |
| A 面板 | `[MTile, 64]` BF16 ≤ 16 KiB | L1 | hidden 切片；N384 下 R=3、N512 下 R=4 复用 |
| B 面板 | `[64,128]` BF16 = 16 KiB / 块 | L0B | N384 三块同驻 48 KiB（≤ 64 KiB 容量） |
| L0C 累加器 | `[MTile,128]` FP32 = 64 KiB / 块 | L0C | N384 三块 192 KiB、N512 四块 256 KiB |
| `qg` / `kg` / `vg` | `[Mpad, P]` FP32 | GM | Conv+QK 阶段 → 递归阶段的跨线程交接 |
| `qkv` / `proj2` / `raw_decay` / `gated` | `[Mpad, 3P]` / `[Mpad, 256+P]` / `[Mpad, P]` / `[Mpad, P]` | GM | 段间交接 |

UB 是 256 KiB。递归段常驻的 `state`（64 KiB）+ `snap` 双槽（128 KiB）占 192 KiB。为了塞下
第二块快照，UB 布局按「小 scratch 压到低地址、状态三块 64 KiB 连续、conv 段 scratch 上移」重排：
`c_st`@0x0A000、`c_snap` 两槽@0x1A000/0x2A000，conv 的 `b_*` 移到 0x3A000 以上（pass1 用、
pass2 已死，地址相接但不冲突）。L1 ≈512 KiB 用来常驻 A 面板与权重双缓冲；L0B 64 KiB 在 N384 下
被 3 块 B 面板占到 48 KiB；L0C 按 64 KiB/块使用，N384 三块（192 KiB）、N512 四块（256 KiB）。

`state` 与 `snap` 双槽的作用：`_kda_update_vf` 在算下一 token 的 `S` 的同时，把当前 `S` 写进
当前槽（`c_snap[t % 2]`）；GM 写**不存当前槽，而是延迟一拍存 token t−1 的那个槽**。这样
「存 token t−1」读的是上一拍已写好的槽，与当前 token t 的 `_kda_update_vf`（写另一槽）无
`V→MTE3` 当拍依赖，DMA 可与当前 token 的向量计算并发，直到两拍后该槽被复用前才需完成。
这是递归段的主要流水手段（见 §5.4）；相对早期「单 `snap`、当拍写回」结构，它避免了写回与计算串行。

### 3.3 寄存器布局

一个 FP32 向量寄存器是 64 lane 宽（本平台一个 FP32 向量寄存器有 64 个并列通道），128 通道的
状态行正好拆成两半：

| 寄存器 | 用途 |
| :--- | :--- |
| `s0` / `s1` | 状态行 `S[r, 0:64]` / `S[r, 64:128]` |
| `k` / `q` | 本 token 的 `k` / `q` 行（两半） |
| `g` | `gexp` 行（两半） |
| `acc` | `Σ_c S[r,c]·k[c]` 的部分和，段间保持存活 |
| `delta` | `(v[r] − ksum[r])·β`，整行广播 |
| conv：`a0` / `a1` | K=4 卷积的两半累加器 |
| conv：`w0` / `w1` | 4 个 tap 的两半系数 |
| conv：`s0` / `s1` | 本 token 窗口的两半输入（BF16 搬入时展开成 FP32） |
| `preg` | 全 1 掩码（`MaskPattern.ALL`），所有 VF 共用 |

递归热循环里 `ksum` 与 `o` 各需要一次行内归约，用 `vf.reduce_sum` 把 64 lane 的累加器压成
一个标量；`delta` 用 `vf.full` 广播回整行。最后更新 `S` 用 `vf.mul` + `vf.add` 计算
`S ← S·g + delta·k`（`mul_add_dst` 未启用，见 §4）。本平台 FP32 向量寄存器为 64 lane（`_VL` 加宽到 128 会算错），因此状态行按两半 64 lane 处理。

## 4 接口选择

| 环节 | 用法 | 理由 |
| :--- | :--- | :--- |
| 全局内存 ↔ 片上 | `pl.load` / `pl.store` | 权重、激活、中间量的常规搬运 |
| 权重布局 | 公开权重即 `pl.NZ` + 逻辑 `[K, N]` | 六块全部直传，Cube 的 GM→L1 是纯 NZ→NZ 直拷贝，host 零转换 |
| tile 与同步 | `pl.make_tile_group` + `auto_mutex` | 槽位轮转的互斥由框架插入 MTE1/M/MTE2/FIX 信号量 |
| 片上 ↔ 寄存器 | `vf.load_align` / `vf.store_align` | 按 64 lane 对齐寻址，一段一条指令 |
| BF16 展开 | `LoadDist.UNPK_B16` | 搬入时顺带把 BF16 展开成 FP32 |
| 标量归约落盘 | `store_align(dist=FIRST_ELEMENT_B32)` | `ksum` / `o` 的行内归约结果直接成为标量 |
| 标量广播 | `load_align(dist=BRC)` / `vf.full` | `delta[r]` 广播到整行；常数、掩码与 `a_log[h]` 灌满寄存器（`a_log` 的指数也在寄存器里算） |
| 融合乘加 | `vf.mul` + `vf.add`（`vf.mul_dst_add` / `vf.mul_add_dst` 未启用） | `S ← S·g + delta·k` 与 `o` 的累加当前是三条指令；换成 `vmadd`/`vmula` 属指令级精简，不改变算法 |
| 行内归约 | `vf.reduce_sum` | 把 64 lane 的累加器压成标量 |
| 激活与门控 | `vf.exp` / `vf.add` / `vf.mul` / `vf.astype`（升宽） | SiLU、sigmoid、decay 的逐元素计算；FP32→BF16 收窄打包在 VF 外用 `pl.cast`（见 §1） |
| 跨核事件 | `set_cross_core` / `wait_cross_core` | 按 event_id 的计数信号量，段间 1:1 配平 |
| 输出累加 | `pl.AtomicType.AtomicAdd` | cube2 下半 K 与上半 K 并发写同一 `out_pad` |
| 编译期形状 | `@pl.jit(tiling_key=_MegaTilingKey)` | `MTile` 三档，小 batch 不再按 128 行空算 |
| 启动参数 | `pypto.options(pass_options={"enable_slice": False})` | 关闭 slice 变换，保持手写的段结构 |
| 核数 | 支持 8 / 16 / 24 / 28 / 32；默认设备可用最大值；`MEGA_KDA_CUBE_CORES` 在集合内覆盖（其它值报错） | 核数不进公开签名 |

## 5 流水设计

### 5.1 权重零转换直传

六个投影权重都按公开的 `[K, N]` NZ 原样交给 Cube，宿主不做任何格式转换，每一次 GM→L1 都是纯
NZ→NZ 直拷贝。`qkv`、`Wa`、`Wbeta`、`Wgate`、`decay_b`、`Wo` 各自作为一个独立入参；前端按
N-tile 号选择本 tile 属于哪一块（tile 0 是 `Wa`、tile 1 是 128 宽的 `Wbeta`、其余是 `Wgate`），
因此不再需要在 host 侧拼出一张 `[H, 256+P]` 的合成权重。`Wbeta` 的补零由调用方在契约里完成
（见 `README.md` 约束说明）。

权重不做缓存：零转换后 wrapper 只是把调用方的六个张量原样转交给 kernel，不需要持有引用或按
身份查表。scratch buffer（`qkv`、`proj2`、`qg/kg/vg`、`gated` 等）按
`(device, batch, num_heads)` 复用，换 batch 只重新分配 workspace，与权重无关。

### 5.2 Conv+QK 阶段隐藏于 cube1 尾部

cube1 先算 `qkv` 并发事件 0；AIV 只依赖 `qkv`，因此可以立即开始 Conv+QK 阶段（Conv+SiLU + Q/K RMS
归一化），与 cube1 后续的 `proj2` / `raw_decay` 并发。该阶段基本被 cube1 尾段盖住。

Conv+QK 段的收尾要注意一条框架约束：3 块 conv 面板和归一化结果都由 VF 产生，框架的
`auto_mutex` **不为 VF 产出的 tile 插入 V→MTE3 依赖**，所以 `qg/kg/vg` 的三次 TSTORE 之前必须
显式 `bar_all()`（每 item 一次）。去掉这个屏障会在"前一次调用改变了 UB 残留"的序列里读到
NaN（如 `tp4_short_b4 → tp4_main_b32` 序列）。

### 5.3 递归阶段 ‖ cube2

递归阶段按 head 对半拆两组，每组算完发一个事件。cube2 拆成两段 K：等事件 2 后用普通 store 写
上半 K，等事件 3 后用 `AtomicAdd` 把下半 K 累加进 `out_pad`（本次调用新分配的输出缓冲，写完直接返回）。这样递归组 1 与 cube2 上半 K
并发，输出投影的尾段被递归段的重叠吸收。两组之间是同一个递归体的两次调用，只换 work-item
映射，不复制代码。

### 5.4 递归段的写回与预取

递归是逐 token 串行的，但可以做的重叠有两处：

- **状态延迟一拍写回**：`_kda_update_vf` 把当前 token 的 `S` 写进 `c_snap[t % 2]`；GM 写则
  在**当前 token 计算时发上一个 token（t−1）的快照**（`c_snap[prev_buf]`）。这样 MTE3 读的是
  两拍前写好、当前 token 不写的槽，与当前 token 的向量计算无 `V→MTE3` 当拍依赖，可与计算并发，
  直到两拍后该槽被 t+1 复用前才需完成。这是本算子相对「单 `snap` 当拍写回」避免写回串行的结构来源；写回代价仍受 UB→GM 字节带宽与「仅 2 槽、余量有限」的结构约束。
- **槽位索引预取**：token `t` 的 VF 执行期间取 `t+1` 的 `ssm_state_indices`，消除每 token
  一次串行全局内存标量读。

### 5.5 段间同步是全屏障的

`set_cross_core` 默认只覆盖同一 block 的 AIC 与其两个 AIV 子核（`INTRA_BLOCK`），而 Vector 段
有两处**跨 block** 消费：递归阶段读其它线程写的 `qg/kg/vg`，cube2 读其它 block 写的 `gated`。
这两处在 set 前各加 `bar_mte3() + sync_all(AIV_ONLY)`（跨核可见写都走 MTE3；`bar_all` 是超集）。Cube 段用 `bar_fix() + sync_all(AIC_ONLY)`：Acc→GM 的 store
走 fixpipe，排空 FIX 管道即可；它每阶段只执行一次，语义比 `bar_all` 更精确。两条
前端路径各自在最后一次 store 之后只做一次最终屏障：4P 融合路径在 `_cube_front_proj` 末尾、普通
路径在 proj2 之后，kernel 不再重复。否则 32
AIC 上 `(32,24)` 会从 `torch.empty` 工作区读到 NaN 位；24 / 28 AIC 只是未命中该时序窗口。
`INTER_BLOCK` 走 FFTS 全局屏障语义（与 `sync_all` 同机制），不是点对点的核间生产者/消费者信号，故跨 block 交接只能用全屏障。

最后是编译期与运行时的边界。`MTile`、`S`、`D`、`CONV_K` 与各段偏移是编译期常量，它们决定
tiling key、UB 字节布局、掩码宽度与立即数；`B`、`N`、`P`、`m_tiles`、`nt_q` 全部是运行时量，
前端分档与核间映射按运行时值走。代价是分档判断在 kernel 内、每个任务多一条分支；收益是任意
`B` 与 `N ∈ {6,12,24}` 共用同一份 kernel 定义（只有 `MTile` 三档编译期特化），换形状不需要
重新编译。
