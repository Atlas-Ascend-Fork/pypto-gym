# QuantSparseLightningIndexer 设计说明（PyPTO-Pro）

对应实现：[quant_sparse_lightning_indexer.py](op_kernel/quant_sparse_lightning_indexer.py)。模块只有一个公开 host 函数和一个 `@pl.jit(arch="a5", auto_mutex=True, tiling_key=QsliKey)` 入口 `qsli_kernel`。host 校验属性、分配输出及工作区，再启动一次 kernel；设备侧完成候选 K 搬运、QK 点积、head 加权归约、TopK 和分片结果归并。接口见 [README.md](README.md)，测试及历史性能记录见 [RESULTS.md](RESULTS.md)。

本文中，worker 指一个 AIC 及其配合工作的两个 AIV；tile 指一次处理的 512 个候选 token，对应候选数组中连续的 64 个块位置，不要求这些块在逻辑序列或物理内存中连续。GM 是设备全局内存，UB 是 AIV 的本地缓冲区；L1、L0A、L0B 和 L0C 用于 Cube 数据及累加结果。MTE2 负责从 GM 读取数据，MTE3 负责向 GM 等目标搬运数据，FIXPIPE 将 Cube 累加结果交给 AIV。

## 计算阶段划分

单个 Query 的 32 个 head 组成 Cube 矩阵的 M 维，512 个候选 token 组成 N 维，128 维特征组成乘法归约维度。一个 Query 行可能包含多个 tile；同一 worker 处理的该行区间内，Q 和 Q scale 只加载一次。前四个阶段按 tile 流水执行，最后按行或行分片做 TopK。

| 阶段 | 执行部件 | 粒度 | 实现逻辑 |
| :--- | :--- | :--- | :--- |
| A 候选取址与搬运 | 两个 AIV | 每行准备、每 tile 搬运 | 每个 AIV 负责 32 个候选块，即 256 个 token。PA 路径准备页表地址，再将记录搬入 UB；TND 路径分别读取 K 和 scale。两种路径都将数据发布到 worker 私有的 GM 环形工作区 |
| B QK 点积 | AIC | 每 tile | 从 GM 工作区加载 K 为 ZN 布局、scale 为 NN 布局，经 L1→L0B / ScaleRight 后，调用一次 `pl.matmul_mx`，计算 `[32,128] @ [128,512]`，结果为 FP32 |
| C QK 交接 | AIC FIXPIPE | 每 tile | 两次 `pl.move` 使用 `SingleModeVec0/Vec1`，分别将 N 维前后 256 列窄化为 BF16，交给对应 AIV 的 UB |
| D head 归约与键编码 | 两个 AIV | 每 tile 的一半 | `_reduce_heads_vf` 对 QK 做 ReLU，再按 head 0 到 31 顺序与 BF16 权重融合乘加，每一步结果舍入为 BF16；还原 PA 成对搬运的顺序，编码为 uint16 比较键，无效 token 的键设为 0，写入 GM 的 `key_ws` |
| E TopK 与输出 | 主计算阶段的 AIV 子块 0 | 每行或行分片 | 等待两个 AIV 的键写完，将本 worker 的键读回 UB，选择 TopK；整行任务直接写输出，分片任务先写 `part_key` / `part_pos`，随后进入跨核归并 |

**阶段 A 的两种布局**：PA 每条记录为 544 B，包含 8 个 token 的 512 B packed K 和 32 B scale。页表宽度不超过 2048 时在 UB 缓存页表，并用 VF 准备当前行所需的地址。每相邻两条记录按地址升序排列，正常模式下每 AIV 用 16 条 `[2,544]` 跨步 DMA 读取 32 条记录，再用两次 MTE3 分别发布 K 和 scale。`swap` 记录交换位置，阶段 D 恢复候选原顺序。确定性等级为 3 时，`PairMin=0`，每对改用两条单记录 DMA。页表过宽时走标量取址，清零 `swap`，不做地址排序。

TND 不使用 PA 的 544 B 记录或成对 DMA。`_gather_tnd_block` 根据 batch 起点和候选块号分别读取最多 8 个 token 的 K、scale，并用 `pl.set_validshape` 限制实际序列尾部的读取量。scale 的 axis-0 stride 为 4 字节时，整块 scale 先在 UB 拼接，再统一写 GM；stride 含 padding 时，经 `scale_pitch` 缓冲逐块写入紧凑的 GM scale 工作区。未参与有效选择的尾部即使有残留数据，也会被可见性掩码排除。

**可见性与有效数量**：`_count_visible_vf` 为每个候选块计算 `clamp(visible - 8*block_id, 0, 8)`，候选前缀以外设为 0；`_valid_mask_vf` 将块计数展开为逐 token 掩码。只要 Query 有可见 K 且候选长度非零，就按候选长度生成 tile，通常不会根据可见性逐块跳过搬运。因此“最终允许进入 TopK 的 token 数”和“实际进行矩阵乘的 token 槽位数”可能不同。

**BF16 分数到比较键的映射**：设 BF16 原始位模式为 `bits`。负数编码为 `bits ^ 0xffff`，非负数编码为 `bits ^ 0x8000`。在排除 NaN 的范围内，分数越大，uint16 键越大；有效负分数不会被错误地当作无效项。无效键为 0，低于包括 `-inf` 在内的有效数值键。返回分数时对位变换求逆即可恢复 BF16，无需重新计算 QK。

阶段 E 按以下情况处理：

- **有效数量不超过 TopK**：未跨核分片且 `valid_count <= rt_topk` 时，`_collect_valid_vf` 直接收集所有非零键的位置，跳过直方图选择。它扫描键是否非零，不是判断分数是否为正。
- **radix（按字节统计的基数选择）**：先按高字节建立 256 桶累计直方图，再找到目标高字节，仅对该高字节下的低字节建立累计直方图，得到完整的 uint16 边界键 `kth`。代码用 `bottom = max(1, scan - rt_topk + 1)` 将第 K 大换算为升序排名，`scan` 包含补零项。随后两次扫描，先收集 `key > kth`，再收集 `key == kth` 的位置，只取所需数量。这里共四趟遍历键数组：高字节统计、低字节统计、大于边界收集、等于边界收集，并非四轮排序。
- **空行或没有有效 token**：直接写 `-1` / `-inf`。`n_tiles=0` 时跳过 K 搬运和点积；有 tile 但全部候选 token 不可见时，点积已执行，仍按有效数量为 0 写无效输出。写分数且没有 tile 时，两个 AIV 按行轮流填充；其余公开输出由子块 0 写入。

位置映射由 `_map_positions_vf` 完成。普通行中，选择得到的局部位置 `p` 先加 `tile_begin*512`，还原到整行候选 token 位置；再计算 `candidate_ids[p//8]*8 + p%8 + output_idx_offset`。不足 TopK 的位置写 `-1`，不加 offset。输出只保证 TopK 集合正确，不保证按分数排序，同分位置也不规定选择次序。

**跨核分片归并**：每个分片最多保留 `rt_topk` 项，存储比较键和整行候选 token 位置。这样做是充分的：某项若在自己分片内已有至少 K 项分数不低于它，则全行选择 K 项不必保留该项；同分情况仍可选取任意满足边界的集合。所有分片完成后，LD 记录指定的 AIV 复用上述 radix 选择，再通过 `_resolve_merge_pos_vf` 将选择位置映射回原始候选位置，最后输出逻辑 token 编号。

归并窗口按代码计算：`P = align_up(TopKCapacity, 16)`，`S = max(2, min(32, 16384//P))`。每轮最多读 `S-1` 个新分片，剩余一槽保留上轮 TopK（代码中的 carry）；还有分片时将本轮结果写回 carry 工作区，供下一轮读取。例如容量 512 时每轮最多读 31 个新分片，容量 8192 时每轮读 1 个新分片。当前实现每轮均写该行公开输出，最后一轮覆盖为完整归并结果。

## 核间切分策略

- **启动形态**：`qsli_kernel[None, num_workers, key]`，`num_workers = min(32, get_platform_info().cube_core_num)`。每个 worker 的 AIC 处理 QK，两个 AIV 各处理 N 维的一半。kernel 内不再启动其他 kernel。
- **调度来源**：调用方传入 `int32[1024]` 的 `metadata`；公开函数只检查 shape、dtype 和设备，不生成调度。测试中的 `_schedule_cpu` 先计算各行 tile 数，再按 tile 区间分配工作，必要时将同一行拆给多个 worker。其切分还受 `atom = max(1, ceil(max_row_tiles/merge_cap))` 约束，以控制归并分片数。
- **LI 记录**：记录 0–31，每条 8 个 int32，依次为启用标志、起始 batch、起始 Query 行、起始 tile、结束 batch、结束 Query 行、结束 tile、部分结果工作区起点。结束 tile 按排他边界解释；batch 和 Query 行共同定位该边界。AIC 与两个 AIV 读取同一记录，遍历一致的行区间和 tile 区间。调度还要把没有 tile 的行分配给某个 worker，以保证输出被填充。
- **LD 记录**：从记录 36 开始，设备按 `36 + pl.get_block_idx()` 分配给各 AIV。当前归并读取前 5 个字段：启用标志、batch、batch 内 Query 行、首个部分结果槽号、分片数。记录号从 0 开始；第 36 条记录的启用标志同时指示本次调用是否存在归并任务。
- **典型场景**：72 行，每行 2048 个候选块，即 32 tile，总计 2304 tile。32 个 worker 各处理 72 tile，按 tile 数均衡；由于 72 不是每行 tile 数 32 的整数倍，部分行会跨核切分。tile 数相同不等同于包括 TopK、空行填充和归并在内的各核耗时完全相同。
- **全局同步**：存在 LD 任务时，所有 AIC/AIV 进入一次 `pl.system.sync_all(core_type=pl.SyncCoreType.MIX)`。两段代码分别位于 Cube、Vector section，运行时共同形成一次全局同步阶段。各核读取同一个 GM 标志，必须一致进入或跳过；无分片时跳过屏障和归并。
- **同一 worker 的两个 AIV**：`EV_JOIN` 确认键写入完成，`EV_JOINED` 确认子块 0 已读回键行，之后才允许覆盖该 worker 的键工作区。这两类事件使用 `INTER_SUBBLOCK`，用于两个子块之间的局部协作。

跨核事件编号范围为 0–15，与 TileGroup 的 mutex id 是不同的编号空间：

| 事件 | 编号 | 作用 |
| :--- | :--- | :--- |
| `EV_K_READY` | 0–3 | AIV 发布 K/scale 后通知 AIC 可读取相应 GM 槽 |
| `EV_RING_FREE` | 4–7 | AIC 读取后通知 AIV 可以覆盖相应 GM 槽 |
| `EV_QK_READY` | 8–9 | FIXPIPE 写入 QK 后通知 AIV 可归约 |
| `EV_JOIN` / `EV_JOINED` | 10 / 11 | 同一 worker 两个 AIV 的键写入与回读握手；11 只用于 AIV |
| `EV_QK_FREE` | 14–15 | AIV 归约完成后通知 AIC 可覆盖 QK 槽 |

`sync_all(MIX)` 在 AIC 侧保留 11/12/13、AIV 侧保留 12/13。模块导入时通过 `_check_constant` 检查冲突，因此不能只修改 `STAGE_DEPTH` 就直接加深 GM 环。

## 核内切分策略：基本块选择、核内 Buffer 划分

### 基本块选择

| 维度或参数 | 取值 | 与代码实现的关系 |
| :--- | :--- | :--- |
| `TILE_M` | 32 | 一个 Query 的 32 个 head 一次参与点积，M 维不是 32 个 Query |
| `TILE_N` | 512 | 64 个候选块，每条记录含 8 个 token；每 AIV 处理 32 个块。单槽 L0B 为 `128*512/2 = 32768 B`，两槽占 64 KiB |
| `LOGICAL_D` | 128 | head 特征维度，含 4 个 32 元素量化组；一次 `matmul_mx` 覆盖全部特征，没有 K 维累加循环 |
| `STAGE_DEPTH` | 4 | 每 worker 私有 GM 环的槽数，按 `t & 3` 轮转 |
| `BUF_DEPTH` / `QK_DEPTH` | 2 / 2 | UB gather 槽、K 的 L1/L0 槽及 QK 交接槽按 `t & 1` 轮转 |
| `TopKCapacity` | 16、32、…、8192 | `max(16, next_power_of_two(topk))`，决定输出临时缓冲与归并槽尺寸。实际保留和写出数量仍按 `rt_topk` 控制 |

512 个 token 兼顾 DMA 条数、L0B 和 UB 预算。若扩大到 1024，L0B 的单槽需求翻倍，原有双槽需要 128 KiB；UB 中 gather 与 QK 槽也随之增大，无法直接维持当前布局及重叠方式。原文记录的 TopK=512 档 UB 扩容估算约为 265728 B，超过 221184 B 的用户预算；这只是修改 tile 后的容量分析，并非当前代码存在的另一种配置。

### 核内 Buffer 划分

`_ub_layout(topk, is_pa)` 按 512 B 对齐计算 UB 地址。TopK=512、PA 路径每 AIV 共使用 **197120 B**；TopK=8192 时最大占用为 **214528 B**，低于 `USER_UB_BYTES=221184 B`。其中对齐空隙也计入总占用。

| 缓冲区 | TopK=512、PA 的有效分配字节 | 用途 |
| :--- | ---: | :--- |
| `ring` | 34816 | 两个 17408 B gather 槽，每槽容纳 32 条 544 B 记录；TND 在相同槽范围内分别组织 K、scale |
| `qk` | 32768 | 两个 `[32,256]` BF16 槽，承接 FIXPIPE |
| `kscratch` | 1024 | 两个 256 元素 uint16 槽，保存单 tile 半区的编码键并写 GM |
| `key` | 32768 | 最大 16384 项比较键；`key2d`、`keyflat`、`mkey` 和 `mcarry_key` 在不同阶段共享此地址范围 |
| `pos` | 36864 | 收集的位置数组，按 `MAX_TOKENS+2048` 个 uint16 预留，供压缩存储使用 |
| `cand` / `cand_pf` | 各 8192 | 当前行候选和下一行预取候选 |
| `phys` / `swap` | 各 8192 | 物理地址及成对交换标志；`phys64` 是 `phys` 的 int64 别名视图 |
| `table` | 8192 | PA 页表缓存，最多 2048 个 int32；TND 不分配此项，改为分配 256 B `scale_pitch` |
| `counts` | 8192 | 各候选块的可见 token 数 |
| `out_i32` / `out_u16` | 2048 / 1024 | 输出索引与分数或部分结果键；索引按最多 1024 项分块写出 |
| 其他小缓冲 | 5120 | `total`、`validm`、两种 dtype 的权重、两个直方图及边界选择中间量；总布局另含对齐间隙 |

TopK=8192 时，`out_i32` 为 4096 B，按 1024 项分块复用；`out_u16` 为 16384 B，存放完整的分数或比较键。分片阶段即使 `return_value=False` 也需要写内部比较键，不能省掉这块临时缓冲。

L1 / L0 按每个 worker 分配：

| 缓冲区 | 声明布局与地址 | 深度 | 用途 |
| :--- | :--- | :--- | :--- |
| `q_l1` / `qs_l1` | Mat NZ `[32,128]` FP4 @0；Mat ZZ `[32,4]` E8M0 @4096 | 1 | Q 和 Q scale 在当前行区间内复用 |
| `k_l1` / `ks_l1` | Mat ZN `[128,512]` FP4 @8192；Mat NN `[4,512]` E8M0 @270336 | 2 | K 与 K scale 从 GM 加载后的轮转槽 |
| `l0a` / `sa_l0` | Left NZ / ScaleLeft ZZ，起始地址均为 0 | 1 | Q 操作数及 scale，跨 tile 保留 |
| `l0b` / `sb_l0` | Right ZN / ScaleRight NN，起始地址均为 0 | 2 | K 操作数及 scale，共用相应轮转索引 |
| `acc` | Acc NZ `[32,512]` FP32，`fractal=1024`，地址 0 | 1 | 64 KiB 累加结果，再由 FIXPIPE 输出 |

L0A 使用 2048 B，L0B 使用 65536 B，L0C 使用 65536 B；L1 的地址上界为 286720 B。MX 操作数要求相应 scale 与数据地址满足 `addr(scale) == addr(data) >> 4`；这里指 Left/Right 与 ScaleLeft/ScaleRight 的配套地址，不要求 L1 中的 Q、K 和 scale 也满足同样的地址关系。

**GM 工作区**：`_workspace` 为每次调用独立分配 K 环、scale 环、键工作区及两个部分结果数组。32 worker 时，K/scale 环合计 `32*4*512*(64+4) = 4456448 B`，即 4.25 MiB；键工作区为 `32*32*2*256*2 = 1048576 B`。`part_key` 和 `part_pos` 各为 `[128,TopKCapacity]` uint16，前 64 槽供分片发布、后 64 槽供各 AIV 保留阶段性归并结果。

**别名与生命周期**：PA 的成对记录视图与 `ring` 共享地址，单条描述符不带 mutex id，因此 `_stage_tile_pa` 用整槽 MTE2 `mutex_lock/unlock` 与整槽 MTE3 store 协作。归并阶段的 `mpos`、`mpos_flat`、`mcarry_pos` 复用 gather 区域，必须等环槽事件排空、`bar_mte3()` 完成发布、MIX 全局同步结束后才可访问。仅凭地址别名或相同 mutex id 无法替代这道阶段边界。

## 接口选择

- **显式 Cube 操作**：使用 `pl.load`、`pl.move`、`pl.matmul_mx`，便于控制 L1/L0 地址、scale 配对与轮转深度。测试 `test_no_sugar_dependency` 会移除 `gemm`、`gemm_mx`、`mem_space` 后重新导入并执行两种布局，以确认不依赖这三个接口。
- **K 的布局转换**：GM 中 K 按 token 连续存放，`pl.load(..., order=[1,0])` 将其载入为 Cube 右操作数使用的 ZN 布局，避免额外的 L1 内转置。
- **scale 的 phase 轴**：host 将 scale 视作 `[-1,2,2]`，最后的长度 2 维是物理 phase 轴。GM→L1 加载时 phase 偏移为 0，`order` 仅排列其他维度，生成 Q 使用的 ZZ 与 K 使用的 NN 布局；这里的 `view` 只改变视图，不做数值解码。
- **FP4 dtype 解释**：外部 Q/K 为 uint8 packed 数据，JIT 参数和 tile 以 `DT_FP4E2M1` 使用这些存储位。设备侧通过 `matmul_mx` 联合 FP4 与 E8M0 计算，不先展开成完整 FP32 Q/K 张量。
- **Acc→UB**：用两次 `SingleModeVec0/Vec1` 完成 FP32→BF16 和 N 维分配，采用这条路径是因为当前工具链的双目标模式不能同时完成这里要求的窄化。
- **VF 计算**：`_weights_to_bf16_vf` 转换权重；`vf.maxs` 和 `vf.mul_add_dst` 完成逐 head 计算；`vf.gather` 完成位置置换和索引映射；`vf.histograms(..., ACCUMULATE)` 生成累计直方图；`vf.squeeze` 和 `vf.squeeze_store_unalign*` 将命中位置连续压缩存入 UB。多个辅助函数以内存屏障固定读写顺序，调用方仍需满足跨 pipe 依赖。
- **同步职责**：带 `mutex_ids` 的 TileGroup 由 `auto_mutex=True` 管理相应数据依赖；别名描述符用整槽锁；无自动 mutex 的 QK 由 ready/free 跨核事件保护。VF、标量和搬运之间还有显式 `sync_src/sync_dst`，例如标量读取 VF 计数前的 V→S，以及复用键暂存前的 MTE3→V。跨 worker 归并使用 MIX 全局同步。
- **PA 地址选择**：页表宽度 `<=2048` 时使用 VF；物理 K 存储跨度 `<2^31` 字节时用 32 位地址，否则用 `vf.mull`、`vf.addc` 组成 64 位运算并按高低字比较地址。页中块数为 2 的幂时用移位、掩码，否则用预计算倒数乘法及过冲修正。页表无法缓存时回退到标量取址，该路径仍有整数除法，不能把向量路径的“无除法”推广到所有分支。
- **成对 DMA 的地址间距**：正常 PA 路径用 `gap = max(a1-a0, 544)` 构造两行跨步视图。向量路径已将成对地址升序排列，不同物理记录间距至少 544 B，因而钳位不改变合法记录的读取。候选前缀之外的补齐位置可能重复读取安全记录，但这些 token 必须被掩码排除；钳位不意味着支持重复的有效候选块或非法页表。

`QsliKey` 只有 `WriteValues`、`IsPA`、`TopKCapacity`、`TndStrideMode`、`PairMin` 五个字段。`TndStrideMode` 的 bit 0 表示 K stride 为 64 字节，bit 1 表示 scale stride 为 4 字节。batch、序列长度、页大小、压缩比例、地址宽度和页表是否缓存均作为运行时标量或数据处理，不因这些具体数值建立新的 tiling key。

## 流水设计

三组缓冲区将搬运、点积和归约连接起来：

```text
AIV MTE2：从候选地址读取 K / scale
    ↓ UB gather 双槽（t & 1）
AIV MTE3：发布到每个 worker 的 GM 环
    ↓ GM 四槽（t & 3）
AIC MTE2 → L1 双槽 → MTE1 → L0B / scale 双槽 → matmul_mx
    ↓ FP32 Acc 单槽 → FIXPIPE
AIV QK 双槽（t & 1，BF16）→ VF head 归约 → key_ws
    ↓ 两个 AIV 握手 → 子块 0 回读键行
TopK → 整行输出，或发布分片结果 → MIX 同步 → LD 归并
```

- **GM 槽的生产与消费**：AIC 启动时为 4 个槽发出 `EV_RING_FREE` 初始事件。AIV 取得空槽事件后搬运和发布，MTE3 完成时发 `EV_K_READY`；AIC 等待后读取 K/scale，并在 MTE2 读取完成时归还空槽。这样 GM 槽可以在后续 L0 搬运及点积尚未结束时被生产者复用。
- **QK 槽的生产与消费**：AIV 为两个 QK 槽发出初始 `EV_QK_FREE`。AIC 在 FIXPIPE 写入前取得空槽，交付后发 `EV_QK_READY`；AIV 等待、归约，随后归还空槽。QK 的生命周期与 GM 环分别管理，不要求两个环采用同样深度。
- **tile 循环**：AIV 遍历 `tick = tile_begin .. tile_end`，先发布 `tick` 的 K，再归约 `tick-1` 的 QK。末次迭代仅收尾，故循环次数比 tile 数多 1。UB gather 双槽允许 MTE2 读取与 MTE3 发布重叠；L1/L0B 双槽允许后续 tile 的加载与当前 tile 的计算交叠。
- **行间预取**：当前行键已回读后，如果同一 batch、同一 LI 区间内还有下一行，且下一行确有 tile，则提前加载下一行候选、准备 PA 地址并发布最多 4 个 tile。随后子块 0 处理当前行 TopK；下一行循环根据 `prefetched` 跳过已发布的 tile。该机制为搬运与 TopK 提供重叠机会，不覆盖跨 batch 或所有空行边界，也不能仅由源码保证完全没有等待。
- **归约融合**：32 个 head 按每组 4 个手工展开，两个 128-lane BF16 累加器覆盖每 AIV 的 256 个 token。ReLU、逐步融合乘加、顺序恢复、键编码和有效掩码在同一次 VF 调用中完成，每 tile 每 AIV 向 `key_ws` 写 512 B。
- **退出与归并**：AIC 排空 QK 空槽事件，AIV 排空 GM 空槽事件并执行 `bar_mte3()`。存在 LD 任务时所有核再进入 MIX 同步；此后 gather 槽不再用于搬运，才可复用为归并位置数组。归并是主流水结束后的独立阶段。
- **DMA 条数的影响**：原有对照记录中，每 tile 每 AIV 从 16 条成对 DMA 改为 32 条单记录 DMA 后，典型场景由 81.5 µs 增至 101.9 µs。按每核 72 tile 折算，新增一条 DMA 对应约 `20.4 µs/(72*16)=17.7 ns` 的平均时间增量。它包含实际流水调度影响，不能视为脱离场景的单条指令固定延迟。

**已知取舍与已否方案**

- **保留 GM 环**：它让 AIV 先把离散候选整理成连续数据，再由 AIC 使用 GM→L1 的布局转换接口；代价是 K/scale 相比只读原输入多一次 GM 写和一次 GM 读。32 worker 的环工作集为 4.25 MiB，小于记录环境的 128 MiB L2，具备较好的缓存复用条件；是否全部命中 L2、是否产生额外 HBM 流量，仍需缓存及内存计数器确认。
- **原有 UB→L1 对照记录**：尝试在 UB 将 FP4 做 ND→NZ，再直接交 L1。记录中重排约需 0.624 µs/tile，而省掉 GM 中转约节约 0.121 µs/tile，故未采用。E8M0 scale 的布局生成仍依赖 GM→L1；两个 AIV 写同一 L1 槽的不同半区还需要进一步解决复用同步。上述数字和能力结论来自历史实验，本次只核对当前采用的 GM 路径，未重新运行该方案。
- **原有 L0C 双槽对照记录**：`acc` 从 depth 1 改成 depth 2 后，典型场景记录为 81.35→81.51 µs，没有显示收益，因此当前仍使用单槽。该结果支持当时的瓶颈主要在其他阶段；不能据此保证所有输入上双槽都没有收益。
