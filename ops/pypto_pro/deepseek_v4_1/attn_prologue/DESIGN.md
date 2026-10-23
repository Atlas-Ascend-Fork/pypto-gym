# attn_prologue 设计说明（PyPTO-Pro）

对应实现：[attn_prologue.py](op_kernel/attn_prologue.py)。单模块包含 MIX kernel `attn_prologue_kernel`、公开 host 函数 `attn_prologue` 和权重转换工具 `to_nz`。本文根据当前 kernel 描述计算、分核、缓冲区与同步；接口见 [README.md](README.md)，测试及历史性能见 [RESULTS.md](RESULTS.md)。

一个核组包含 1 个 AIC 和 2 个 AIV。本文的 tile 是连续的最多 128 个 token；输出列块为 128 列。QA、QB 分别指代码中的第一次、第二次 Q 投影，QR 是两者之间经过 RMSNorm 和量化的中间表示。阶段编号 A1、A2、B、B-epi 沿用源码注释，分别对应第一次投影、归一化与量化、第二次 Q 投影及最终 Q 输出处理。

## 计算阶段划分

`section_cube` 和 `section_vector` 定义 AIC、AIV 的两条执行流；它们通过两次 MIX 全局同步连接，只有后半段 QB 与 Q 输出处理使用逐输出块的流水握手。

| 阶段 | 执行部件 | 输入、计算与写出 |
| :--- | :--- | :--- |
| QA/KV 投影（A1） | AIC | 同一 X 分别乘 Wqa、Wkv 的转置。QA 有 10 个输出列块，KV 有 4 个；每块沿 H=5120，以 K0=256 做 20 步 MXFP8 乘加。首步 `matmul_mx` 初始化 FP32 累加器，后续 `matmul_mx_acc` 累加。结果从 Acc 写入 GM 工作区 `ws_qa/ws_kv`；小 T 的归约维二分路径每分区仅做 10 步，并分别写部分和 |
| 全局同步 1 | 全部 AIC/AIV | 所有投影块写完后，AIV 才能读取完整 QA/KV 行。Cube 的工作区结果由 Acc 写出，不能将其描述为 Vector 的 MTE3 写回 |
| 归一化、量化与 cache 写回（A2） | AIV | QR 行沿 1280 个特征做 RMSNorm，再按 32 元素组量化，写公开 `qr/descale_qr`。KV 行沿 512 个特征做 RMSNorm，末尾 64 维做位置变换，再按 scale=1 转 E4M3FN，写入 `cache_index` 指定的 slot。归约维二分时，先对两个 FP32 部分和做一次 `pl.add` |
| 全局同步 2 | 全部 AIC/AIV | QR payload 与 scale 写完后，AIC 回读作为 QB 的 A 操作数；KV 写回也在 AIV 到达该同步之前发出 |
| QB 投影（B） | AIC | `qr @ Wqb.T`，每个 head 的 512 列切为 4 块，全 tile 共 `64*4=256` 块。每块沿 R=1280 做 5 步乘加，FP32 Acc 经 `DualModeSplitM` 向两个 AIV 的 `cv_ub` 各交付 64 行 |
| Q 输出处理（B-epi） | AIV | 等待对应 `cv_ub` 就绪，普通列块转 BF16；每 head 最后一个列块还需对后 64 列做 RoPE，再转 BF16，写入 q，并通知 Cube 可以复用交接槽 |

Q 的 RoPE 使用 FP32 QB 结果，随后才舍入成 BF16；KV 则先完成归一化和 RoPE，再饱和量化。所有 head 使用同一 token 的位置表，但各 head 的 Q 数据独立。

`qr_row_vf` 先按 64-lane FP32 向量求平方和，再将行归一化系数广播。每个量化组只激活 32 lane，先计算 `(qa*rstd)*gamma`，取最大绝对值，再提取 FP32 指数字段。最后一组起点为 1248，仍会做 64-lane 加载，因此 `qa_row/gamma_qr` 按 `QR_PAD=1344` 预留；额外 lane 不参与组内归约与输出。

RoPE 的 `INTERLEAVE_HALF` 表示相邻输入 `(z[2*i], z[2*i+1])` 分别产生两个输出半区的第 i 项。代码分别做两次乘法和一次加法，没有额外负号，也不把它们改成融合乘加。`vf.gather` 在寄存器内取偶数/奇数位置，未激活 lane 的索引也钳位在 0–63，避免使用越界的寄存器索引。

## 核间切分策略

**启动网格**：host 正常查询路径取 `block_dim=max(1,min(cube_core_num,vector_core_num//2))`，再调用 `attn_prologue_kernel[None,block_dim]`。平台查询失败或返回异常核数时回退到 32/64 配置。kernel 由 block 与 subblock 索引取得 AIC 核组号及 AIV 子块号；以下具体核数示例按原有 32 核组、64 AIV 环境说明。

**阶段 A1 独立选择三种分配方式**：令 `n_tiles=ceil(T/128)`，`NA1=14`，`NA1G=7`。

| 条件与顺序 | 分配单元 | 每单元工作 | 32 核组上的范围 |
| :--- | :--- | :--- | :--- |
| 先判断 `n_tiles*7 >= num_cores` | `(tile, 两个相邻输出列块)` | `_a1_group` 中两个累加器共享每个 X 的 K 窗口，完成 20 步归约 | `n_tiles>=5` |
| 否则判断 `n_tiles*14*2 <= num_cores` | `(输出列块, 归约分区)` | `_a1_block_sk` 把 H 分为两个 2560 维分区，各算 10 步并写独立行带 | `n_tiles=1`，共 28 个单元 |
| 其余情况 | `(tile, 单个输出列块)` | `_a1_block` 对完整 H 做 20 步归约 | `n_tiles=2..4` |

这些单元均以 `pl.range(core_id,total_work,num_cores)` 跨步分配。分组减少 X 的重复加载，却把并行工作量从每 tile 14 项变为 7 项；只有工作足够时才启用，以免小 T 下可用核更少。归约维二分分支的寻址固定从 X 第 0 行开始，当前 32 核组下只在一个 tile 时进入；平台查询和实际网格不应脱离本实现已验证的架构配置。

归约维二分时，部分和落在 `ks*n_tiles*128 + row`，ks 为 0 或 1。A2 载入两个行带并以一次 FP32 加法合并，再做归一化。这改变了与完整 H 单次归约相比的求和顺序，不能从数学相等推导所有输入逐字节相等；精度测试通过独立 FP64 复核处理量化边界差异。实现没有执行通用的 Kahan 补偿求和。

**阶段 A2 的 AIV 分配**：`worker=core_id*subblock_num+sub`，`n_aiv=num_cores*subblock_num`。

- 归约维二分时，将 QR 与 KV 拆成 `2*T` 个任务，以 worker 号跨步执行；task 小于 T 对应 QR 行，否则对应 KV 的 `task-T` 行，每项先合并两个部分和。
- 非二分且 `n_tiles<=2` 时，仍按同样的 QR/KV 任务集合分配，但将 QR 循环和 KV 循环分开。首批 KV 位置表来自屏障前预载槽，超过预载数量的任务回到 GM 流式读取。
- `n_tiles>=3` 时按 row 跨步分配，同一个 AIV 依次完成该行 QR 与 KV。源码保留此路径，以避免把大 T 也拆成任务后增加控制和搬运成本。

任务数少于 64 时不会占满全部 AIV，例如 T=1 只有 QR、KV 两项，不能将“按 64 个 AIV 分配”写成“64 个 AIV 都有工作”。

**阶段 B 与 B-epi 独立选择两种分配方式**：没有 tiling key，kernel 根据运行时 `n_tiles` 分支，不能把 A1 与 B 的门槛混为一谈。

- `n_tiles<32`：按 `(tile,head,列块)` 展平为 `wu=tile*256+head*4+block`，以核组数跨步分配。相邻核组访问相邻的 Wqb 128 行权重块；每核只有首次或 tile 改变时才回读 `qr/descale_qr`。以 32 核组为例，一个 tile 的 256 块由每核 8 块承担，每核都会读取该 tile 的 QR。
- `n_tiles>=32`：每核按 tile 跨步分配，独占所分 tile 的全部 64 个 head 和 256 块 QB 输出。QR 与其 scale 每 tile 只从 GM 载入一次，保留在 L1 中跨输出块复用。

准确的 B 分支边界是 T=3969（第 32 个 tile 出现），不是 T=4096；原生产测试矩阵中相邻取值为 2048、4096，因此观测为从 4096 起进入逐 tile 分配。A1 分组在 32 核组上则从 T=513 起启用。

**工作区分配与分支一致**：host 用同一 `n_tiles*14*2 <= grid` 条件决定 `sk_bands`。二分时 `ws_rows=2*n_tiles*128`，否则 `ws_rows=T`；公开 QR/Q 输出始终只分配 T 行。全局同步使 A1 的多核列块生产、A2 的按行消费以及 B 的多核回读可以采用不同的任务划分。

**事件与 mutex**：两次 `sync_all(MIX)` 在 Cube、Vector section 中各有对应调用点，运行时形成两道全局同步。跨核事件 id 为 0–15，当前 `EV_READY` 用 0/1，`EV_FREE` 用 2/3；AIC 侧 11/12/13、AIV 侧 12/13 留给 MIX 同步。

mutex id 是另一套编号空间，不能与跨核事件 id 混用。Cube section 内部 TileGroup 使用 0–13、16–28，共 27 个 id；外部声明、供两段共用的 `cv_ub` 使用 14/15。Vector 加上共享 `cv_ub` 使用 0–17，预载槽再用 18–25。同一物理地址的 `b_nz/b_zn` 视图共享对应 mutex id。导入检查覆盖预载 id 之间及其与已占 Vector id 的冲突，并非逐个自动扫描全部 TileGroup 声明。

## 核内切分策略：基本块选择、核内 Buffer 划分

### 基本块选择

| 参数 | 取值 | 与实现的关系 |
| :--- | :--- | :--- |
| `BASE_M` | 128 | Cube 的 token 行窗口；QB 交接后每 AIV 接收 64 行，逻辑尾部另限制存储 |
| `K0` | 256 | 每次乘法的归约维宽度，满足 MX 的 64 倍数要求；完整 X/权重窗口 `[128,256]` E4M3FN 为 32768 B，L0A/L0B 双槽各占满 64 KiB |
| `BN_A/BN_B` | 128/128 | A1 与 QB 的输出列块；每个 `[128,128]` FP32 Acc 为 64 KiB。若直接把 BN_B 改为 256，右操作数双槽和累加器都需重新规划 |
| `NK_A/NK_B` | 20/5 | H/K0 与 R/K0；归约维二分的每个 A1 分区为 10 步 |
| `A1_GRP` | 2 | 两输出块共享一次 X GM→L1 载入，分组路径每 tile 的完整 X 读取等效由 14 次降为 7 次 |
| `KV_PRE` | 4 | 非归约维二分、`n_tiles<=2` 的 A2 路径预载位置表槽数；32 核组时每 AIV 最多 4 个 KV 任务，可全部覆盖 |
| `b_nz/b_zn` 深度 | 4 | 同址视图各自按相同顺序 `.next()`；同时容纳 4 个权重 K 窗口，供流水及首个 QB 块预取使用 |

三个矩阵乘沿 K 方向分块，每一步 K0 是矩阵乘归约的特征数；它与 cache 页大小、量化组 32、RoPE 宽度 64 是不同概念。

### 核内 Buffer 划分

以下容量使用 KiB（1024 B），高水位是地址上界，含对齐空隙，不等于各声明的有效元素字节之和。L1 高水位 `0x76000=483328 B=472 KiB`，预算为 512 KiB。

| 缓冲区 | 逻辑形状、布局与起始地址 | 深度 | 有效数据字节合计 | 用途 |
| :--- | :--- | :--- | ---: | :--- |
| `a_l1` | `[128,256]` E4M3 NZ，0x00000/0x08000 | 2 | 65536 | X 的 K 窗口 |
| `sa_l1` | `[128,160]` E8M0 ZZ，0x10000/0x6E000 | 2 | 40960 | 一个 A1 单元的完整 X scale，按 K 步切片搬到 ScaleLeft |
| `b_nz/b_zn` | `[128,256]` NZ / `[256,128]` ZN，0x16000/0x1E000/0x26000/0x2E000 | 4，共享存储 | 131072 | NZ 视图加载 Wqa/Wkv/Wqb，ZN 视图作为转置右操作数搬出 |
| `sb_l1` | `[160,128]` E8M0 NN，0x36000/0x3B000 | 2 | 40960 | QA/KV 权重 scale；分组时两个槽保留两个输出块的 scale |
| `qr_l1` | `[128,1280]` E4M3 NZ，0x40000 | 1 | 163840 | QB 的 QR 操作数，跨输出块复用 |
| `sq_k` | `[128,40]` E8M0 ZZ，0x68000/0x74000 | 2 | 10240 | QR scale |
| `sw_k` | `[40,128]` E8M0 NN，0x6A000/0x6C000 | 2 | 10240 | QB 权重 scale |

L0A 的 `[128,256]` 与 L0B 的 `[256,128]` 均为 E4M3，槽地址 0/0x8000，各双槽占 65536 B。ScaleLeft/ScaleRight 的有效形状分别为 `[128,8]`、`[8,128]`，槽地址 0/0x800；每个 scale 存储空间按 4 KiB 预算检查。数据槽与 scale 槽地址满足 `scale_addr=data_addr>>4`，两者使用配套轮转索引。这是 L0 操作数与 MX scale 的约束，不要求 L1 中独立区域也按同一关系排布。

L0C 共 256 KiB：`a1_acc[0/1]` 位于 0x00000/0x10000，供 A1 两列块分组同时保留部分和；`q_acc[0/1]` 位于 0x20000/0x30000，供 QB 轮转。四槽都是 `[128,128]` FP32。A1 与 QB 分属不同阶段，但当前声明为不同地址；不能因为运行时不重叠就直接缩小高水位而不改代码。

每个 AIV 的 UB 高水位为 `0x27E00=163328 B=159.5 KiB`，预算为 216 KiB。原文 163.5 KiB 的换算与该十六进制地址不符。

| 缓冲区 | 有效数据字节 | 用途 |
| :--- | ---: | :--- |
| `cv_ub` 双槽 | 65536 | 每槽 `[64,128]` FP32，由 FIXPIPE 写入，VF 读取 |
| `qa_row/qa_row2/gamma_qr` | 各 5376 | QR 行、第二部分和与权重，按 1344 元素预留 |
| `qr_payload/exp_tile/scale_u8` | 1280/256/64 | 量化 payload、每组指数暂存、压紧后的 scale 字节 |
| `kv_row/kv_row2/gamma_kv` | 各 2048 | KV 行、第二部分和与权重 |
| `kvb/cos_row/sin_row` | 512/256/256 | KV 编码与当前行位置表 |
| `q_bf16` 双槽 | 32768 | 每槽 `[64,128]` BF16，供 Q 写出 |
| `cos_ub/sin_ub` | 各 16384 | QB 当前 tile 半区的 `[64,64]` FP32 位置表 |
| `cos_pre/sin_pre` | 各 1024 有效数据 | 各 4 个 256 B 行，槽距 512 B，各预留 2048 B 地址范围 |

GM 工作区按每次调用独立分配：`ws_qa` 为 `[ws_rows,1280]` FP32，`ws_kv` 为 `[ws_rows,512]` FP32。普通路径合计 `T*7168 B`，二分路径按两行带分配；三个公开输出合计 `T*(65536+1280+40) B`。cache 沿用调用方存储，不额外复制。

## 接口选择

- **Cube 显式分段操作**：使用 `pl.load`、`pl.move`、`pl.matmul_mx` 和 `pl.matmul_mx_acc`，以控制片上地址、分组 scale、同址视图和不同深度的缓冲轮转。
- **NZ 权重转置视图**：权重逻辑为 `[输出列,归约维]`。先按 NZ `[128,256]` 加载，再用同地址 ZN `[256,128]` 视图搬到 L0B，实现 `A @ W.T` 的右操作数方向；这不是重新复制一份转置权重。NZ/ZN 的地址、槽顺序和 mutex id 必须一一对应。
- **E8M0 加载**：公开 scale 的末维长度为 2，作为物理 phase 轴。加载时该轴偏移为 0，`order` 仅排列其余轴，A 侧落 ZZ、B 侧以 `order=[1,0]` 落 NN，之后按 K0 对应的 8 个量化组切片进入 L0 scale 空间。
- **QR 经 GM 反馈**：A2 直接写公开 `qr/descale_qr`，屏障后 AIC 用 GM→L1 的布局生成能力回读。这样 A2 可按行生产，B 可按输出列块或 tile 消费，无需采用相同的分核方式；当前实现不使用 UB→L1 生成 E8M0 的 ZZ/NN 布局。
- **Acc→UB 保留 FP32**：`DualModeSplitM` 将 `[128,128]` Acc 按 M 对半交给两个 AIV；此处没有 FP32→BF16 窄化，后续 RoPE 仍需 FP32。BF16 转换在 VF 内完成。
- **VF 的归一化与编码**：`reduce_sum` 求平方和，`DuplicatePos.LOWEST` 广播归约结果，`sqrt` 和 `div` 求归一化系数。只有量化 scale 的倒数通过指数字段构造，不能称所有 VF 都没有除法。`astype(...,CAST_RINT,SaturateMode.ON)` 转 E4M3，`PACK4` 从 32 位容器压紧成字节；BF16 用 `PACK` 压紧。指数暂存写后再读之前有 `vf.mem_bar(VST_VLD)`。
- **分页写回**：`cache_index` 为扁平 slot，kernel 将 cache 无复制地视作 `[P*BS,512]`，以 `pl.getval` 读取 int64 索引，再 `pl.store(...,[slot,0])` 写整行。故不用另查 block table；页号和页内偏移已经由扁平索引隐含表达。
- **同步职责**：TileGroup 的 `mutex_ids` 与 `auto_mutex=True` 管理核内依赖及槽复用；Cube/AIV 交接另用 ready/free 事件，阶段间用 MIX 屏障。自动 mutex 与跨核事件负责不同层次，不能互相替代。

## 流水设计

**QB 与 Q 输出处理的双槽交接**：代码中的 `beat` 是本核处理的 QB 输出列块序号；每块由 `_qb_beat` 完成 5 步矩阵乘，再由 `_q_epi_beat` 输出对应 Q。它不是 K 归约循环的步号。

```text
AIC: 5 步 MX 乘加 → q_acc[beat%2]
     → 若 beat>=2，等待 EV_FREE+(beat%2)
     → FIXPIPE 把 FP32 结果分给两个 AIV 的 cv_ub[beat%2]
     → 发布 EV_READY+(beat%2)

AIV: 等待对应 EV_READY → 读取 cv_ub → RoPE/BF16
     → 按逻辑行数存储 q → 发布对应 EV_FREE
```

前两个输出块使用尚未占用的槽，无需等待 free；从第三块起，覆盖同一槽前必须等上一消费者读完。AIV 的 free 由 V pipe 发出，表示 VF 已读完 `cv_ub`；独立的 BF16 输出缓冲还受自动 mutex 保护，避免 MTE3 写出前被覆盖。函数尾部没有显式排空最后两个 free 事件的循环，不能将其描述为主动清空全部事件。

Cube 和两个 AIV 采用完全对应的任务枚举，保证同一 `beat` 指向相同 tile/head/列块。逐 tile 分支通过 `beat0=(tile//num_cores)*256` 延续每核序号；分布分支通过 `(wu-core_id)//num_cores` 生成本地序号。即使某个 AIV 在尾 tile 的 `store_rows=0`，仍要等待 ready 并发布 free，避免 Cube 因无人响应而停住。

**围绕加载与计算依赖安排重叠**：

- 权重 L1 四槽、L0A/L0B 双槽和配套 scale 轮转，为后续加载与当前搬运、矩阵乘提供重叠空间；同一槽必须在前一次消费者读取后才能覆盖。是否完全隐藏加载等待需由 profiler 验证，不能仅凭槽深度断言带宽已满。
- 分组 A1 在每个 K 步只读一次 X 窗口，再喂两个输出块；减少的是 X 的 GM→L1 重复读取，权重仍按各列块分别读取，X 的 L1→L0 搬运也分别执行。
- AIV 在全局同步 1 之前加载两条 RMSNorm 权重。非二分且 `n_tiles<=2` 时，还预载最多 4 个 KV 任务的 sin/cos；加载行号和实际 KV 任务号对应，并用范围守卫保证不读未加载槽。归约维二分路径不使用该位置表预载。
- AIC 在两道全局同步之间，为分布式 B 的本核首个输出块预取完整权重 scale 和 K 步 0–3 的四个 Wqb 窗口。首块调用 `_qb_beat(...,pre=True)` 跳过这些 GM 加载，实际使用预取槽；第 4 个 K 步仍正常加载。`sw_k` 提前推进两次恢复奇偶位置，`b_nz` 提前推进 4 次恢复模 4 位置，使后续 `b_zn.next()` 对齐相同物理数据。
- B 的 QR 和 B-epi 的位置表都只在本核首次或 tile 变化时加载；逐 tile 分支在一整个 tile 内复用，分布分支在本核所分配的同 tile 输出块之间复用。不同核组依然有各自独立的 L1/UB 副本。
- `rows_full=min(128,T-tile_row)` 限制 X、scale、QR 的 GM 加载及 A1 工作区写出。QB 的 Acc→UB 仍按完整 128 行窗口交接，未执行原文所述的 valid_M 向 2 对齐再分半；两个 AIV 分别按 `min(64,max(T-ro,0))` 计算实际 VF 处理及写出行数。片上矩阵窗口大于逻辑尾部，不表示 host 为普通工作区或公开输出分配了补齐行。

两个全局同步分别保护整行 RMSNorm 所需的多列块结果，以及量化 QR 对 QB 的反馈。在它们之间可以预取纯输入，但 AIC 不能提前消费尚未写完的 QR。源码结构因此提供部分重叠机会，并没有把全部 A2 归一化与 QB 计算放在同一条逐 tile 流水中。
