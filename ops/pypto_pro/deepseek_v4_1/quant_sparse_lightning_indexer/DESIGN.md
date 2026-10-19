# QuantSparseLightningIndexer 设计说明（PyPTO-Pro）

对应实现：`op_kernel/quant_sparse_lightning_indexer.py`（单模块，一个 `@pl.jit(arch="a5", auto_mutex=True, tiling_key=QsliKey)` 入口 `qsli_kernel` + 一个公开 host 函数）。本文只描述本实现的设备侧结构与流水设计；接口、约束与用例见 `README.md`，精度与性能数据见 `RESULTS.md`。

## 计算阶段划分

一个 kernel 内顺序衔接 5 个阶段，前 4 个按 tile 流水、第 5 个按行收口：

| 阶段 | 执行部件 | 粒度 | 做什么 |
| :--- | :--- | :--- | :--- |
| A 取址与 gather | AIV（两个子块各半） | 每 tile | 行首一次性算出本行 2048 个候选块的物理记录地址（页表查表 + 成对排序）；每 tile 用 16 条 2 行跨步 DMA 把 32 条 544 B 记录收进 UB 槽，再以 2 次 MTE3 发布到本 worker 私有的 GM staging 环（payload `[32,512]`、scale `[32,32]`） |
| B Cube QK | AIC | 每 tile | GM→L1（FP4 转置成 ZN、E8M0 走 phase 轴视图成 NN）→ L1→L0B / ScaleRight → 一次 `matmul_mx` M32×N512×K128 → FP32 Acc |
| C QK 交接 | AIC FIXPIPE | 每 tile | 两次 `pl.move(..., acc_to_vec_mode=SingleModeVec0/Vec1)` 把 FP32 窄化为 BF16，按 N 半区分投两个 AIV 的私有 UB |
| D head 归约 + 键编码 | AIV（两个子块各半） | 每 tile | `_reduce_heads_vf`：8 组 × 4 head 的 BF16 融合乘加链（内联 ReLU），撤销成对 DMA 的地址排序，编码为保序 uint16 键（不可见 token 置 0），MTE3 写 512 B 到本 worker 的 GM key 工作区 |
| E TopK + 输出 | AIV0（子块 0） | 每行 | 回读 joined 键行（最大 32 KB）→ 4 趟 uint16 radix 选择 → `_map_positions_vf` 经候选块号映射回逻辑 token id 并加 offset → 写 `indices/values` |

阶段 E 的三条分支：

- **快路径**：未分片且 `valid_count <= topk` 时，一趟 `_collect_valid_vf` 直接压紧全部有效项，不做直方图。
- **radix 路径**：4 趟遍历键数组 —— 高字节 256 桶累计直方图（`scan//256` 次迭代）→ 在累计分布上二选一扫描定位 `high` bin 并算 `next_k` → 低字节直方图（仅统计高字节命中的键）→ 定位 `kth` → 两趟压缩存储分别收集 `key > kth` 与 `key == kth` 的位置（`scan//128` 次迭代）。位置随后映射为逻辑 token id。
- **空行路径**：`n_tiles == 0` 时直接写哨兵（`-1` / `-inf`），不进入任何计算阶段；`return_value=True` 时两个子块按行轮流分担填充，`False` 时由子块 0 独自完成。

**跨核分片的收口**：一个行区间被切到多个 worker 时（LD 行），各 worker 把自己的键与原始位置发布为 GM 部分结果，在唯一一次 MIX 网格屏障之后，由 LD 记录指定的 AIV 各归并一行：复用与主路径完全相同的 radix 链接与压缩存储接口，再用 `_resolve_merge_pos_vf` 把归并后的位置解析回本行的部分结果数组。归并以 `max(2, min(MERGE_SLOT_CAP, 容量档位/16)) - 1` 为窗口流式进行，超窗分片用 carry 槽回写并进入下一趟。

## 核间切分策略

- **启动形态**：`qsli_kernel[None, num_workers, key]`，`num_workers = min(32, get_platform_info().cube_core_num)`；每个 core group = 1 AIC + `sub_num = 2` 个 AIV；一次公开调用一次启动，kernel 内不再分裂。
- **任务来源**：切分结果由 host 生成的 LI/LD `int32[1024]` 调度表给出，设备侧只消费不重算。每 worker 读自己的 LI 记录（8 字段：enable / 起始 batch,M,tile / 结束 batch,M,tile / 部分结果游标），得到 `(b, m, tile)` 字典序区间 `[begin, end)`。tile 是切分的最小单位（512 token = 64 个候选块），因此尾块粒度可控：目标场景 72 行 × 32 tile = 2304 tile 被 32 个 worker 均分为每核 72 tile，末轮无空转核。
- **角色划分**（不是不相交的核集合，而是同一核组内的部件分工）：AIC 承担阶段 B/C；两个 AIV 各承担 N 维的一半（256 token = 32 块）做阶段 A/D；AIV0 额外承担阶段 E 与本行的输出写入；归并阶段按扁平 AIV 序号（记录号 = `LD_RECORD_START + get_block_idx()`）分配，因此归并可横跨所有 core group 的 AIV。
- **全局同步**：只有 **一次** `pl.system.sync_all(core_type=pl.SyncCoreType.MIX)`，且由调度表的 LD 记录（第 36 行首字段）条件化 —— 该值取自 GM，全部核读到同一份数据，因此「全进 / 全不进」是核间一致的；没有跨核分片行时，归并阶段与 ring UB 的别名复用都不发生，屏障一并跳过（零 tile 场景由此省掉一次 32 块网格同步，实测约占该场景耗时的 20%）。
- **跨核事件预算**：事件 id 只有 0..15，其中 11/12/13 归 `sync_all(MIX)`（AIV 侧 12/13）。当前分配 `EV_K_READY 0..3`、`EV_RING_FREE 4..7`、`EV_QK_READY 8..9`、`EV_JOIN 10`、`EV_QK_FREE 14..15`（`EV_JOINED 11` 仅 AIV 侧可见）。这条预算是 `STAGE_DEPTH=4 + QK_DEPTH=2` 的上限来源 —— 加深 GM 环必须先重排 id 分配。
- **同核两子块之间**：行尾用一对 `CrossCoreSyncMode.INTER_SUBBLOCK` 事件（`EV_JOIN` / `EV_JOINED`）对接键工作区的发布与回读，不占用全局屏障。

## 核内切分策略：基本块选择、核内 Buffer 划分

### 基本块选择

| 维度 | 取值 | 依据 |
| :--- | :--- | :--- |
| `TILE_M` | 32 | 等于 head 数，一个 Query 行的全部 head 一次进 Cube；M 无可切分余地 |
| `TILE_N` | 512 | = 64 个候选块 = 64 token 记录；L0B `[128,512]` FP4 = 32 KB，depth 2 恰好用满 64 KB L0B；Acc `[32,512]` FP32 = 64 KB 在 L0C 之内；每 AIV 承担 256 token = 32 块 × 544 B = 17408 B/槽 |
| `K` | 128 | 等于 head dim，且是 MX 量化组（32）的整数倍；单个 K 块即一次 `matmul_mx`，**核内没有 K 方向累加循环**，流水全部发生在 tile 之间 |
| `STAGE_DEPTH` | 4 | GM 环深度；受上面事件 id 预算约束的上限 |
| `BUF_DEPTH` / `QK_DEPTH` | 2 / 2 | UB gather 槽与 QK 交接槽深度 |
| TopK 容量档 | 16/32/…/8192 | 作为 tiling key 的一个字段，决定 `pos`/`out_i32`/`out_u16` 与归并槽的静态尺寸；实际输出仍按运行期 `topk` 截断（`pl.set_validshape`） |

`TILE_N` 不能再往上翻倍：512→1024 会让 UB 的 ring 与 qk 两侧各翻一倍，TopK=512 档的总占用从 197120 B 涨到约 265728 B，超过 221184 B 的用户 UB 预算；即使把两侧深度都降到 1 也只是勉强贴合，且会同时摧毁 gather↔发布与 Cube↔归约的两级重叠。

### 核内 Buffer 划分

UB（每 AIV）由 `_ub_layout(topk, is_pa)` 以 512 B 对齐一次性静态排布（`functools.lru_cache` 缓存），TopK=512 / PA 档总占用 **197120 B**（用户上限 221184 B）：

| 槽 | 字节 | 用途 |
| :--- | ---: | :--- |
| `ring` | 34816 | 2 × 17408 B gather 槽；成对 DMA 的 `[2,544]` 描述符视图别名其上 |
| `qk` | 32768 | 2 × `[32,256]` BF16 QK 交接槽 |
| `key`（= `key2d`/`keyflat`/`mkey`/`mcarry_key`） | 32768 | joined 键行与归并键，共享地址 |
| `pos` | 36864 | radix 压缩收集的位置（`MAX_TOKENS + 2048`） |
| `cand` / `cand_pf` | 8192 + 8192 | 候选块号与预取副本 |
| `phys`（= `phys64`）/ `swap` | 8192 + 8192 | 物理地址向量（32 位打包 / 64 位交错别名）与成对交换标志 |
| `table` | 8192 | PA 页表 UB 缓存（≤2048 项时启用） |
| `counts` | 8192 | 每候选块的可见 token 数 |
| `out_i32` / `out_u16` | 2048 + 1024 | 输出索引 / 分数暂存 |
| `total`,`validm`,`w_f32`,`w_bf16`,`hist_hi`,`hist_lo`,`sqz`,`idx_hi`,`nk`,`kth`,`kscratch` | ≈5 KB | 标量计数、可见性掩码、权重两态、直方图与 radix 中间量 |

TopK=8192 / PA 的最大占用为 214528 B（`out_i32` → 4096 B、`out_u16` → 16384 B），仍在上限内。

L1 / L0（每 core group）：

| Buffer | 声明 | 深度 | 说明 |
| :--- | :--- | :--- | :--- |
| `q_l1` / `qs_l1` | Mat NZ `[32,128]` FP4 @0 / Mat ZZ `[32,4]` E8M0 @4096 | 1 | Q 与其 scale 整行常驻，跨 tile 复用 |
| `k_l1` / `ks_l1` | Mat ZN `[128,512]` FP4 @8192 / Mat NN `[4,512]` E8M0 @270336 | 2 | K 两槽轮转，承载 GM→L1 转置载入与 E8M0 分形生成 |
| `l0a` / `sa_l0` | Left NZ / ScaleLeft ZZ | 1 | 与 `q_l1` 同步驻留整行 |
| `l0b` / `sb_l0` | Right ZN / ScaleRight NN | 2 | 与 `k_l1` 同步轮转 |
| `acc` | Acc NZ `fractal=1024` `[32,512]` FP32 | 1（64 KB） | 单槽；曾实测 depth 2（128 KB）无收益，故保持 1 |

L1 高水位 286720 B。MX 矩阵乘要求 `addr(scale_a) == addr(lhs) >> 4`、`addr(scale_b) == addr(rhs) >> 4`，由上表的显式地址声明满足（`l0a@0 / sa_l0@0`、`l0b@0 / sb_l0@0`）。

**别名与生命周期**：`ring` 的两个 gather 槽在归并阶段被 `mpos`/`mpos_flat`/`mcarry_pos` 复用；`phys`↔`phys64`、`key`↔`key2d`↔`mkey` 共享地址。别名安全由三件事共同保证：整槽 MTE2 的显式 `mutex_lock/unlock`（16 个成对描述符本身不带 auto mutex id）、归并前的 ring credit 排空 + `pl.system.bar_mte3()`、以及条件化的 MIX 屏障形成的相位边界。编译器可能给出无法识别这些生命周期关系的 `MemSpaceCollision` 提示，属已知现象。

## 接口选择

- **Cube 侧显式三段式**：`pl.load` / `pl.move` / `pl.matmul_mx`，不使用 `gemm`、`gemm_mx`、`mem_space`。原因是需要精确控制 L1/L0 地址以满足 MX scale 的 `>>4` 关系、并让 `k_l1`/`l0b` 的两槽轮转与地址声明解耦。
- **K 的 GM→L1**：`pl.load(k_l1[buf], stg_k_fp4, [row_base, 0], order=[1,0])`，用降序 `order` 直接得到 L0B 需要的 ZN 分形，避免额外的 L1 内转置。
- **E8M0 scale 的 GM→L1**：源张量必须 rank ≥ 3 且**末维是长度为 2 的物理 phase 轴**（host 侧 `.view(-1,2,2)`），phase 轴偏移为 0、`order` 不选该轴 —— 这是 ZZ/NN 分形唯一可用的生成路径。
- **Acc→UB**：两次 `SingleModeVec0` / `SingleModeVec1` 各投半个 N 区，而非一次 `DualModeSplitN`：FP32→BF16 窄化在 dual dst 模式下不可用。
- **Vector 数值计算全部走 VF**（`@pl.vector_function`）：`vf.load_align` + `vf.maxs` + `vf.mul_add_dst`（每步 BF16 舍入）实现 head 归约；权重广播用 `LoadDist.BRC_B16`；撤销成对 DMA 排序用 `vf.gather`；保序键编码用位运算 + `vf.select`；直方图用 `vf.histograms(bin_type=BIN0/BIN1, hist_type=ACCUMULATE)`（累计分布，两条 128-lane 寄存器合覆盖 256 桶）；位置收集用 `vf.squeeze` + `vf.squeeze_store_unalign*` 的非对齐压缩存储；候选块号映射用 `vf.gather`。每个 VF 以 `vf.mem_bar()` 收尾以固定寄存器→UB 的顺序。
- **同步接口分层**：核内 pipe 顺序交给 `auto_mutex=True` + TileGroup 的 `mutex_ids`；**带 `mutex_ids` 的 tile group 上不再叠加手动 `sync_src/sync_dst`**，手动 pipe 同步只保留在 `mutex_ids=None` 的缓冲区（`pos`、`qk`、`validm`、`w_bf16`、`hist_*`、`sqz`、`idx_hi`、`nk`、`kth`）与标量 pipe 依赖（V↔S、S→MTE2）上；跨核用 `pl.system.set_cross_core/wait_cross_core` 计数事件；同核子块间用 `INTER_SUBBLOCK`；全局用 `sync_all(MIX)` + `bar_mte3()`。
- **地址准备按运行期属性选实现**：32 位打包地址向量 / 64 位地址向量（`vf.mull`+`vf.addc` 进位链、`vf.interleave` 存回）/ 标量逐项回退；页大小为 2 的幂时用移位 + 掩码，否则用预计算的倒数乘法 + 过冲修正。**这些属性不进入 tiling key**，因此同一份编译产物服务不同页大小、不同 K 张量规模与不同页表宽度。tiling key 只含 `WriteValues / IsPA / TopKCapacity / TndStrideMode / PairMin` 五个字段。
- **成对 DMA 的地址钳位**：`gap = pl.max(a1 - a0, PACKED_BLOCK_BYTES)` 后无条件发一条 2 行跨步描述符，不使用运行期 `if`。两条不同记录的物理地址必然相差 ≥544 B、且成对排序保证 lane0 ≤ lane1，因此钳位对合法输入是恒等变换，同时保证重复块号（契约外输入）也不会产生行重叠的描述符。这样展开体不再被切成 16 个基本块，标量后段可以把地址计算与描述符发射流水化。

## 流水设计

**三级 buffer 流水**（都在 tile 之间）：

```text
AIV MTE2 gather ──UB 槽(BUF_DEPTH=2, t&1)──▶ AIV MTE3 发布 ──GM 环(STAGE_DEPTH=4, t&3)──▶ AIC MTE2 GM→L1
                                                                                            │
                                       AIV VF 归约 ◀──UB qk 槽(QK_DEPTH=2, t&1)── AIC FIXPIPE ◀┘
```

- GM 环把 gather/发布与 Cube 消费解耦到 4 拍；UB gather 槽把 MTE2 收数与 MTE3 发布解耦到 2 拍；QK 槽把 Cube 的 FIXPIPE 与 AIV 的归约解耦到 2 拍。`k_l1`/`l0b` 同为 depth 2，使 `t+1` 的 GM→L1 载入可与 `t` 的 L1→L0B、`matmul_mx` 重叠。
- **credit 模型**：`EV_K_READY+stage`（AIV MTE3 发布完成 → AIC MTE2 可读环）、`EV_RING_FREE+stage`（AIC 读完环 → 归还 AIV 的 slot credit，启动时由 AIC 预发 `STAGE_DEPTH` 个初值）、`EV_QK_READY+qslot` / `EV_QK_FREE+qslot`（FIXPIPE 交付 / AIV 归约完释放）。两侧在退出前各自排空剩余 credit，从而为环 UB 的别名复用给出一个明确相位边界。
- **tick 循环 = DMA 领先计算一拍**：`for tick in range(tile_begin, tile_end + 1)`，先 `_stage_tile_pa(tick)`（条件 `tick < tile_end 且 tick >= tile_begin + prefetched`），后 `_reduce_heads_vf(tick-1)`（条件 `tick > tile_begin`），因此循环比 tile 数多一次迭代用于收尾。
- **行边界预取**：每行收口前，先为下一行做地址准备（含页表 UB 缓存与候选块 DMA），并把最多 `STAGE_DEPTH` 个 tile 的 gather 提前发出；下一行的 tick 循环据此跳过已预取的 tick。于是下一行的首批 DMA 与本行的 radix TopK VF 计算重叠，行边界不产生 DMA 空窗。`_count_visible_vf` 的结果也提前发布，只在 TopK 真正需要标量计数时才消费。
- **归约与编码在单次 VF 遍历内联完成**：head 归约按 4 个 head 一组手工展开（32 head → 8 次循环控制），两个 128-lane BF16 累加器覆盖本 AIV 的 256 token，中间值留在寄存器不落 UB；ReLU、置换还原、保序键编码、可见性掩码四件事在同一趟里做完，每 tile 只产生一次 512 B 的 MTE3 写。
- **DMA 发射是一级预算**：实测每条描述符的发射成本约 17.7 ns —— 在字节数不变的前提下把每 tile 每 AIV 的 16 条成对 DMA 改成 32 条单记录 DMA，典型场景耗时上升 25%（81.5 → 101.9 µs）。因此本设计选择「成对跨步 + 无分支钳位 + 静态展开 16 次」，并把减少发射条数/分支作为优先于减少字节数的优化方向。
- **归并阶段**是屏障后的独立相位，不与主流水重叠：每个持有 LD 记录的 AIV 载入自己那行的部分结果（键 + 原始位置），流式跑同一套 radix 链，再解析位置并写公开输出。

**已知取舍与已否方案**

- 保留 GM staging 环：每字节 K 因此多一次 L2 写 + 一次 L2 读（环路工作集约 4.5 MB，常驻 128 MiB L2，不额外占用 HBM 带宽）。
- **已实测否决**：用「AIV 的 UB 直接交给 L1」替换该环。载荷腿的能力是可用的（UB 内 `pl.move` 做 ND→NZ 支持 `DT_FP4E2M1`，`Vec→Mat` 走 MTE3，L1 的 ZN `[128,512]` 可由 NZ `[512,128]` 源承接，数值与逐位对照一致），但 UB 内 ND→NZ 重排的代价约 **0.624 µs/tile**，而去掉环只值约 **0.121 µs/tile**；按每核 72 tile 外推为 −8.7 µs 收益对 +44.9 µs 代价，故不采用。此外 E8M0 scale 侧不存在从 UB 生成 ZZ/NN 分形的通路（`FP8E8M0` 不在 UB 内 ND→NZ 支持列表，也没有 UB→ScaleLeft/ScaleRight 的搬运组合），scale 必须保留 GM→L1 路径；而两个 AIV 各写同一个 L1 ZN 槽的半区，在槽复用时会被观测到确定性的数据错位，需要单生产者整体写入才正确 —— 这会把每 tile 的 UB→L1 搬运量翻倍，进一步恶化上述收支。
- **已实测否决**：`acc` 由 depth 1 改 depth 2（L0C 双槽，128 KB）。生成物确认第二个槽真实存在（`TASSIGN(..., 0x10000)`），但典型场景无可测收益 ⇒ MAC 与 FIXPIPE 的串行不在关键路径上，AIC 不是该场景的 pacer。
