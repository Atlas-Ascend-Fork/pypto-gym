# quant_mla_prolog 设计说明（PyPTO-Pro）

本文说明 [quant_mla_prolog.py](op_kernel/quant_mla_prolog.py) 的实现结构。接口与
固定几何见 [README.md](README.md)，用例与性能结果见 [RESULTS.md](RESULTS.md)。

## 计算阶段划分

7 个 Module 分属三个相位。相位内 Cube 与 Vector 协同（MIX kernel），相位之间用
`pl.system.sync_all()` 硬屏障隔开 —— 屏障既保证跨 section 的写后读可见，也让三个
相位分时复用同一片 L1/L0 地址（见「核内切分策略」）。

| 相位 | Module | 计算 | Section | 产物 |
| --- | --- | --- | --- | --- |
| A | 1 QA | `deq(x) @ deq(wqa)ᵀ` MXFP8 下投影 | cube | `qa_raw[T,1536]` FP32（GM ws） |
| A | 2 KVA | `deq(x) @ deq(wkva)ᵀ`，复用同一份 x 的 L1 驻留 | cube | `kva_raw[T,576]` FP32（GM ws） |
| A | 3 Q-requant | `RMSNorm(1536, γ_qa)` → BF16 → 每 32 元素一组 MX 量化 | vector | `qa_mx[T,1536]` E4M3 + `qa_scale` E8M0（GM ws） |
| B | 4 QB | `deq(qa_mx) @ deq(wqb)ᵀ` | cube | `q_head[T,96,192]` BF16（GM ws） |
| B | 5 KV-epilogue | `RMSNorm(512, γ_kva)` ∥ RoPE64 → per-tensor 量化 → PA_NZ 散写 | vector | `kv_cache` 原位 |
| C | 6 WKB | 逐 head `q_head[:,h,0:128] @ wkb[h]` | cube | `q_latent_h[96,T,512]` BF16（GM ws） |
| C | 7 Q-epilogue | `concat(NoPE512, RoPE64)` → per-token-head absmax 量化 | vector | `q[T,96,576]` E4M3 + `descale_q[T,96]` FP32 |

依赖是一条带一处分叉的链：`x` 同时喂 M1 与 M2；M1 → M3 → M4 → {M6, M7 的 RoPE
分支}，M6 → M7；M2 → M5 独立收尾。Module 5 与 Module 4/6 都只依赖相位 A 的产物、
彼此无数据依赖，编号只表达「不前向引用」，不表达执行次序。

**没有 pad 阶段。** front 的 x / descale_x 装载用动态 `valid_shape` 直接读
`[T, 7168]` 的有效行（`m_valid = min(MRows, T − m_idx·MRows)`），装载后把窗口放回
满幅再进 L1→L0 的 move，窗口外的 L1 行保留上一次的字节。收益是省掉一整份 x 的
GM→UB→GM 拷贝与一道 MIX 屏障（屏障从 4 道减到 3 道），x 的 GM 读也只发生在有效
行上。垃圾不外泄的论证见 README 的功能说明。

**补齐一律用 `valid_shape` 收窄，不改物理声明。** 物理声明 ≥ 有效范围：front 按
`m_valid` 收窄；KVA 与 QA 共用 64 宽的 tile、KVA 分支收到 32；`descale_q` 的片上
暂存声明 8 列只用 1 列（FP32 单列 4 B 不满 32 B 对齐）；`q_head` 按二维绝对列号
搬，N=128 段跨 head 边界，`18432/128` 整除无尾块。

三个 Cube Module 的归约轴都超单 tile，按 K 向分块 + L0C 累加链（首块
`matmul_mx`、其余 `matmul_mx_acc`、末块 `AccPhase.Final`）：

| Module | K | 段宽 | 段数 |
| --- | --- | --- | --- |
| 1 QA / 2 KVA | 7168 | `_k_split(m)[0]`：`m ≤ 64` 取 512，`m = 128` 取 256 | 14 / 28 |
| 4 QB | 1536 | `BASE_K_QB = 256` | 6 |
| 6 WKB | 128 | 不分段 | 1 |

Front（Module 1/2）用**单个 L0C 累加器**串行累加全部段（`K_ACCUM_WAYS = 1`）。
`m = 128` 必须把段宽降到 256：`a_l0a` 双槽占 `2·m·base_k`，512 宽需要 131072 B 而
L0A 只有 65536 B。

Vector 侧的行归约（Module 3 的 1536、Module 7 的 576）超单寄存器 64 lane，用两遍
法：第一遍逐寄存器 `vf.reduce_sum` / `vf.reduce_max` 累出行标量，第二遍 `vf.full`
广播回全 lane 做逐元素。`1/1536`、`1/512`、`1/448` 在模块级算成常数，VF 体内禁标量
除法。RMSNorm 写成 `div(1, root)` 而不是 `rsqrt`，因为真值是先 `sqrt` 再除。
## 核间切分策略

一次 kernel launch 完成七个 Module。相位边界的状态全部落 GM workspace，不需要回
host；逐相位 launch 会把屏障换成多次 kernel 启停开销。

**核数来源。** host 侧取 `min(cube_core_num, vector_core_num // 2)` 并再与
`MAX_AIC_CORE_COUNT` 取小，只作为 `block_dim` 传入；kernel 内一律用
`pl.get_block_num()` 读回的 `nc` 做任务划分，因为实际生效值可能小于请求值。混合
kernel 的 `block_dim` 是执行组数而非核数，一个组对应 1 个 AIC 与 2 个 AIV。

各相位的划分方式不同，不是同一套：

| 相位 | work item | 总数 | 映射 |
| --- | --- | --- | --- |
| A：M1 QA | N 段（`BASE_N_QA = 64`） | `N_SEG_QA = 24` | 跨步：`n = core_id + n_local · nc` |
| A：M2 KVA | N 段（`BASE_N_32 = 32`） | `N_SEG_KVA = 18` | 与 M1 共用 tick 循环与 tile，tile 按 64 宽分配、KVA 分支用 `valid_shape` 收窄 |
| B：M4 QB | N 段（`BASE_N_QB = 128`） | `N_SEG_QB = 144` | 跨步 |
| C：M6 WKB | `(head, N 段)` 对 | `96 · n_blocks` | **连续占有**：`per_core = ⌈96·n_blocks / nc⌉`，`gt = core_id · per_core + tick` |
| C：M7 Q-epi | head | 96 | 跨步，另按 sub_id 在两个 AIV 间分半区 |

M6 用连续占有而非跨步，是因为它每个 work item 要带 `wkb[h]` 的 L1 驻留；跨步映射
会让同一个核在相邻拍上跳到不同 head，每拍都得重搬权重。连续占有下同一核的相邻拍
落在同一 head 的 N 段上，`wkb[h]` 只在 head 切换时重载。

**相位 A 的欠载是相位串行加 N 向切分的固有结果，不是映射写法的问题。** M1 的 24 段
与 M2 的 18 段都少于常见的执行组数，尾部组空转；M4 的 144 段与 M6 的 `96·n_blocks`
远多于组数、能填满。M1/M2 共用一条 tick 循环（两者共用同一份 `x_l1`、输出写不同
workspace、无依赖），拍数取
`n_ticks_front = max(rounds_qa, rounds_kva) · n_blocks · K1_WIN_QA`。

**核数改变拍数，不改变数值结果。** 96 个 head 在组数整除时均衡、否则末拍欠载，
所以 [RESULTS.md](RESULTS.md) 的时延只在其标注的核数配置下成立。

## 核内切分策略

### 基本块选择

物理行数 `MRows` 由 tiling key 选，`T` 到档位的映射是 `1-8 → 16`、`9-32 → 32`、
`33-64 → 64`、`65-128 → 128`；`T > 128` 同样用 128 变体、capacity 取
`⌈T/128⌉·128`，kernel 内按 m 块分块。

`16` 是最小合法档：NZ layout 的 tile 行数必须是 inner box rows（16）的倍数，更小的
精确档（如 `T ≤ 8` 用 8 行）会被编译期拒掉。`T ≤ 8` 因此在 16 行的 tile 上算，装载按
`valid_shape` 只搬 8 个有效行，多出的 8 个 pad 行只花 L0A/L0C 与 mac 的空转，不增加
GM 搬运量。

各 Module 的 N 向块宽：QA 取 64、KVA 取 32、QB 取 128、WKB 取 128。K 向窗口
`K_L1_QA = 1024`。

### 核内 Buffer 划分

紧凑排布到 `M = 64` 就要吃掉 97% 的 L1，`M = 128` 无解；不复用共需 644 KB，装不下
L1 容量（平台配置的 `l1_size`，由 `_space_limit("Mat")` 解析，不写字面值）。因此
三个相位分时复用同一片地址，相位之间有 `sync_all` 硬屏障。

下表由 `_l1_rows(64, x_depth=2)` 生成，地址区间是半开的 `[起址, 末端)`：

| buffer | 相位 | layout | 起址 | 槽长 | 槽数 | 末端(不含) | 生命期 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `x` | A | NZ | `0x00000` | 65536 | 1 | `0x10000` | 轮转（与 `x2` 配对） |
| `x2` | A | NZ | `0x10000` | 65536 | 1 | `0x20000` | 轮转（与 `x` 配对） |
| `sx` | A | ZZ | `0x20000` | 14336 | 1 | `0x23800` | 常驻（全 K） |
| `wseg` | A | NZ | `0x23800` | 65536 | 2 | `0x43800` | 轮转 |
| `wsc` | A | NN | `0x43800` | 2048 | 2 | `0x44800` | 轮转 |
| `wqb` | B | NZ | `0x23800` | 65536 | 2 | `0x43800` | 轮转（叠 A 的 `wseg`） |
| `wqbsc` | B | NN | `0x43800` | 2048 | 2 | `0x44800` | 轮转（叠 A 的 `wsc`） |
| `qa` | B | NZ | `0x45800` | 98304 | 1 | `0x5D800` | 常驻（M4 相位内） |
| `qas` | B | ZZ | `0x5D800` | 3072 | 1 | `0x5E400` | 常驻（M4 相位内） |
| `wkb` | C | NZ | `0x45800` | 131072 | 1 | `0x65800` | 常驻（叠 B 的 `qa`+`qas`） |
| `qh` | C | NZ | `0x65800` | 8192 | 2 | `0x69800` | 轮转 |

`x` 与 `x2` 是两个各持一个 mutex 的独立组而不是一个双槽组（单组深度轮转时下一块会
等在同一个 mutex 上）。`M ≥ 128` 时 `x_l1` 退单槽（双槽需 123.8% 的 L1，单槽
98.8%），代价是丢掉 x 的双缓冲、流水换批数减半；`x2` 改叠在 `qa` 上，`wkb` 另得一个
叠在 `x` 上的第二槽。

各档的峰值活跃字节（公式在 `_l1_layout` / `_lr_layout`，断言在 `_l1_tables` /
`_lr_tables`，构造期就报错）：

| `M` | `x` 槽数 | L1 末端 | /512KB | L0A | /64KB | L0C | /256KB | front `base_k` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2 | 314880 | 60.1% | 16384 | 25.0% | 32768 | 12.5% | 512 |
| 32 | 2 | 359424 | 68.6% | 32768 | 50.0% | 65536 | 25.0% | 512 |
| 64 | 2 | 432128 | 82.4% | 65536 | 100.0% | 98304 | 37.5% | 512 |
| 128 | 1 | 518144 | 98.8% | 65536 | 100.0% | 163840 | 62.5% | 256 |

L0A 在 `M ≥ 64` 已占满 —— 这同时是 `M = 128` 的 front 段宽只能取 256 和 `x_l1` 退
单槽的原因。`x` 按 K 窗口流式（整 K 常驻在 `M = 128` 单这一项就 896 KB）；`qa_mx`
常驻单槽、全核共用；`sx` 全 K 常驻，因为它的字节只有 `x` 的 1/32，按窗口轮换反而多
7 次碎事务。

叠址不靠「同址共享 mutex」这条约定成立 —— `mutex_id` 只标识同步槽位，不证明两段
字节可以安全互叠。判据在 `_l1_review`：相位**内**成对完全不重叠，跨相位只允许
**包含**关系，部分重叠一律拒。`MRows = 64` 下有 6 对跨相位叠址，`MRows = 128`
下 8 对。同址复用另有两条硬约束，两条都在编译期拦，但拦它们的不是同一处：

1. **槽数必须相同。** 配对只能 2↔2 或 1↔1；单槽的 `wkb` 叠不到双槽的 `wseg` 上，
   违反会报 `different slot counts (1 vs 2)`，判据来自
   `pypto_pro/runtime/pipeline/_cross_core_scanner.py` 的 `scan_buffer_slot_counts`。
2. **同址的 group 必须共用 mutex id。** section 在源码里的先后顺序**不构成屏障**，
   互斥靠 `auto_mutex` 的核内 Tile 同步，所以 `wkb` 与 `qa` 都用 id 14、`wqb` 与
   `wseg` 都用 `[0, 1]`。这条由 `pypto_pro/runtime/pipeline/_validate.py` 的
   `_check_reuse_mutex_ids` 拦，抛 `InvalidOperation`：mutex 锁的是地址，同一区域上
   挂不同 id 等于没有互斥。

L0A/L0B/L0C 同样三相位复用。L0B 的三组 B 操作数（`b_l0b`、`b_qb`、`b_wkb`）共用
地址 `0x0` 与 `0x8000`、共用 mutex `[8, 9]`。`b_l0b` 的槽距**取 `0x8000` 而不按本组
槽长算**：按槽长排时 `base_k = 256` 的槽 1 落在 `0x4000`，被 `b_qb` 槽 0 覆盖却挂着
不同 id，违反上面第 2 条、编译期报 `MutexMismatch`。取 `0x8000` 后两档的槽 *i* 都
落在 `b_qb` 槽 *i* 区间内且 id 对齐，而它不小于任一档槽长，所以槽 0 吞不掉槽 1。
## 接口选择

| 职责 | 选用 | 理由 |
| --- | --- | --- |
| 三处 MXFP8 收缩（M1/M2/M4） | `pl.matmul_mx` + `pl.matmul_mx_acc` | 直接消费 E4M3 payload 与 E8M0 分组 scale、FP32 累加；设备上不构造完整的反量化 x 与权重 |
| 一处 BF16 收缩（M6 WKB） | `pl.matmul` | 操作数已是 BF16，无分组 scale |
| 归一化、量化、RoPE、输出处理 | 6 个 `@pl.vector_function` | Vector 计算一律走 VF，不使用 tile Vector 计算 |
| GM ↔ 片上搬运 | `pl.load` / `pl.store` | — |
| 权重装载 | 逐 fractal 的 `[K_win, 32]` 视图 tile + 默认 order | 调用方手里的物理 rank-3 `[N/32, K, 32]` 直传 kernel，不用 host 的 rank-2 view；视图组与 `wseg` 同址共享 mutex，与 view 路径逐 bit 等价 |
| γ 的 BF16 绕行 | VF 内 `astype(BF16, CAST_RINT)` → `astype(FP32)` | 与真值的 `.to(BF16).float()` 逐 bit 相同，host 不碰 γ |
| PA_NZ 散写 | `pl.store` 按 fractal 块逐块给 offsets | cache 物理末维 32（C1）或 16（C0），每块恰好 32 B，满足 store 对 UB tile 首地址的对齐要求 |

VF 侧用到的算子：`load_align` / `store_align` 搬运，`astype` 做位宽与编码转换，
`full` 广播行标量，`reduce_max` / `reduce_sum` 做行归约，`div` 完成归一化与反量化，
`create_mask` / `update_mask` 处理尾段，`mem_bar` 划分 VF 内的访存次序。E8M0 的反
scale 用 `254 - raw` 直接拼指数位，不走浮点乘。

**为什么手写 `move` + `matmul_mx` 链而不用 `pl.gemm_mx` 语法糖**：本算子要用
`phase=`（unitFlag）。Front 的 K 段链按 `AccPhase.Partial` 累加、末段
`AccPhase.Final` 收尾，搬出用 `STPhase.Final`；语法糖不发 AccPhase / STPhase，这个
流水机制只能由手写链表达。除此之外没有别的手写理由 —— 其余形态（split-K、MX 的
四次配对搬运）语法糖都能表达。

Cube 的矩阵乘都在 `section_cube`（`matmul_mx` 的操作数空间是 L0A/L0B/L0C），VF 路径
只能 `GM→UB→RegFile→UB→GM`，两类相位因此必须交替出现 —— 这是本算子是混合 kernel
而非单一 Cube 或 Vector kernel 的根本原因。

## 流水设计

`@pl.jit(auto_mutex=True, tiling_key=MlaKey, pipeline=pl.pipeline.PipelineConfig(preload=4))`。
4 个 `@pl.pipeline.stage`：

| stage | 相位 | 职责 |
| --- | --- | --- |
| `stage_w_mm` | A | 一拍 = 一个 `(n, m_idx, k1)`（拍序压成一维、`k1` 最内）：装载本拍权重窗口、预取下一个 `x` 窗口，并在同一拍内交织做 M1 的 `wqa` 与 M2 的 `wkva` 两侧收缩，累加进各自的 L0C。两侧共用同一个 `x` 窗口，这是它们必须在同一条流水上的原因 —— 否则 `x_l1` 的驻留要重建 |
| `stage_wkb` | C | 按连续占有领 `(head, N 段)`，搬 `wkb[h]` 并做 BF16 收缩 |
| `stage_q_epi` | C | C1 的 Q epilogue：concat + per-token-head absmax 量化 |
| `stage_q_epi_c0` | C | C0 的 Q epilogue，BF16 终值 |

C0 与 C1 的 Q epilogue 写成两个 stage 而不是一个带 `if` 的 stage：输出 dtype 进了
kernel 签名类型，不能在运行期分支。

**3 道 MIX 屏障**，每道都是「一个逻辑屏障 = 两侧各一次调用」：cube1→vec1 的
`qa_partial`、vec1→M4 的 `qa_mx`、M6/M7 流水循环前的两侧对齐。MIX 模式下 cube 段与
vector 段各要调用一次 `sync_all`，且必须在循环外，源码共 6 处调用。M4→M6 这道必须
是 MIX 而不是 AIC 内部同步：M4 按 n 段分核写 `q_head`，一个 head 的 NoPE 128 列跨
多个 n 段、落在不同核上，M6 按 head 分核必然读到别核写的列。M6→M7 的逐拍交接走框架
自动 `fwd_ids` / `bwd_ids` 事件，不占屏障。

**跨 section 的中间量只能走 GM。** Cube 只能从 L1 取操作数、L1 靠 `pl.load` 从 GM
搬，所以 vector 段算出的东西要进 cube 必须经 GM 一圈：`qa_partial[1,·,1536]` 与
`kva_partial[1,·,576]`（FP32，承接单累加器 partial）、`qa_mx` 与 `qa_scale_mx`
都是这个原因，可见性由两侧的 `sync_all` 保证。

**E8M0 的窄化在 VF 内按字节落盘，不放 host。** M4 的 A 侧 scale 要在片上读 E8M0，
host 窄化会断掉片上链路。实现上给 VF 写的 UINT8 tile 与给 `pl.store` 读的 E8M0 tile
共用同一个 UB 地址，是同一段字节的纯重解释，无搬运也无算术。

**PA_NZ 散写逐 fractal 块发 store。** 18 个 fractal 块在 page 维相隔 `128*32` 个
元素、彼此不连续，一次搬运取不到，所以在 `kvq_ub` 同一行上按 32 B 步进切出 18 个
`[1,32]` 的 tile 视图逐块写。

槽数按「生产者与消费者的拍距 + 1」定：相位 A 的权重段双槽，让当前 head 的
`n_blocks` 拍期间能预取下一段；`x_l1` 在 `M < 128` 双槽、`M ≥ 128` 退单槽；Front 的
L0C 双槽，让块 A 的 fixpipe 搬出与块 B 的 cube 计算重叠。同一个 n 块的全部 K 段必须
落在同一个 L0C 槽上（累加链跨拍），所以该槽按 n 块的奇偶选，不用 `next()` 轮转。
