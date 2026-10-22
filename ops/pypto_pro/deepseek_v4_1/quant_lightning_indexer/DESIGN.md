# quant_lightning_indexer 设计说明（DESIGN）

> 实现：`op_kernel/quant_lightning_indexer.py`。整份设计只用一个 `@pl.jit` kernel 覆盖全部形状组合，
> 形状里只有物理分块容量与布局选择会参与编译（tiling key），批次、query 数、key 数都是运行期动态轴。
>
> 本文用到的硬件侧术语先在此说明。**AIC**（Cube 核）负责矩阵乘，**AIV**（Vector 核）负责
> 向量计算；两者在同一个 MIX 块里配对，用跨核事件互相通知。**L1** 是 Cube 的输入缓冲，
> **L0A / L0B** 是它的操作数缓冲，**L0C** 是矩阵乘的累加器；**UB** 是 Vector 侧的工作缓冲。
> **MTE1 / MTE2 / MTE3** 是三条搬运流水：MTE1 走 L1→L0，MTE2 走显存→片上，MTE3 走片上→显存。
> 文中常量与分支均与 `op_kernel/quant_lightning_indexer.py` 一致。

## 计算阶段划分

一次调用在设备上分成三个逻辑阶段，全部由一个 kernel 承载：

```text
Q/K 反量化 ─┐
            ├─► AIC：MXFP4 的 QK 矩阵乘（只舍入一次到 BF16）
            │
            ├─► AIV：逐 head 的 BF16 累加折叠 + mask 生效
            │
            └─► AIV：精确 Top-K 选择（可选：分数回写、候选块归约）
        host：回读序列数组 → 预计算分核表 plan → 提交 kernel
```

**阶段 1 — 打分（AIC 与 AIV 协作）。** AIC 取「本工作项负责的若干 query 行 × 32 个 head」作为
矩阵乘的 M 轴、head 维度 128 作为 K 轴、一段 key 作为 N 轴，用 `pl.matmul_mx` 直接吃 MXFP4 数据
与 E8M0 块缩放因子，做一次 QK 矩阵乘。产物是 FP32 的 L0C 累加块，搬进 UB 时按 128 个 key 一档
折成 BF16 分数面。AIV 随后对 32 个 head 做**逐步 BF16 舍入**的累加折叠
（`acc ← bf16(acc + relu(qk_h) * bf16(w_h))`）；这一步同时让 mask 生效：不可见的 key 不参与折叠，
收尾时把无效槽写成 `-inf`。

**阶段 2 — 选择（AIV）。** 在分数面上用 BF16 位型做两级直方图，定位第 `topk` 大的阈值，再按阈值
收集索引。选择是**精确**的，不是近似：先做局部剪枝再归并的正确性来自「被丢弃的元素至少有 `topk`
个不劣于它的前驱」这一不变量。

**阶段 3 — 收尾（AIV）。** 三种可选的收尾，按请求开启：把选中位置的分数回写为 `values`；
把 8-token 候选块的块内最大分归约成 `candidates` 与 `candidate_lengths`；把索引加上
`output_idx_offset`。

**host 侧。** wrapper 做形状 / dtype / stride 校验、分配输出与 workspace、组装 tiling key、编排阶段；
并在 host 上**预计算整张分核表**：回读一次序列数组（`cu_seqlens_*` / `seqused_*` / `cmp_residual_k`），
推出每个工作项的 `(qbase, qvalid, kl, kstart)` 和每个 query 行的 `(visible, partial, count)`
与全空标志，打成一个 int32 张量 `plan` 传给 kernel。kernel 因此**不做任何数据依赖的标量推导**，
只读 `plan` 与 `block_table` / `output_idx_offset`。代价是每次调用多出几次 device→host 同步与框架
提交开销，这部分**不计入**核内耗时；二者的占比见 RESULTS.md 的「绝对性能结果」。

## 核间切分策略

设备是 32 个 Cube 核 + 64 个 Vector 核（`cube_vector_combine=split`，即一个 Cube 核配一对 AIV）。
算子用同一个 kernel 入口、以 `cube_core_num` 为块数（Block Num = 32）启动，一个 MIX 块内
Cube 与 AIV 用跨核事件协作。

**工作项的划分**统一按「一段 query × 一段 key」切，一段 query 恒为 6 行：

| 路径 | 工作项数 | 每项的 query | 每项的 key 范围 |
| :--- | :--- | :--- | :--- |
| 通用（Operation 0） | `B × ceil(max_q/6) × split_h` | 6 行 | `split_h = min(7, ceil(max_k / tile_n))` 个 key 分片，`tile_n = gcd(page, 128)` |
| 分页快速（Operation 1） | `B × ceil(max_q/6) × parts` | 6 行 | `parts = ceil(ceil(max_k/256) / shard_tiles)` 个 key 分片，`shard_tiles = ceil(max_k / (256 × 7))` |

两者都把 key 轴切开分给不同工作项，区别在分片粒度：通用路径按 `tile_n`（64 或 128）分，
快速路径按 256 对齐的 shard 分（`shard_width = shard_tiles × 256`）。分片结果在 kernel 内按行归并，
快速路径经 `index_map` / `partial_values` 在显存上归并一次。

**快速路径的启用条件。** 快速路径不是无条件走的。host 侧要同时满足下面几条才会选 `Operation 1`：
K 是 PA 布局、page 恰好是 128、单个序列的 `max_k` 落在 8192–131072 且是 256 的倍数、
`topk = 512`、关闭候选块、传了 `block_table` 与三个序列数组、并且 `parts × 512 ≤ shard_width`。
其余组合走通用路径：`max_k ≤ 16384` 时选择由 `Operation 0` 在同一次 launch 里一并做完，只有
`max_k > 16384` 才把分数写回显存、再单独启动一次 `Operation 2`。

**为什么 query 段固定 6 行。** 6 行 query × 32 个 head = 192，正好是矩阵乘 M 轴的规模，一次矩阵乘
就把一个工作项的全部 head 算完。M 轴定成 192 之后，L0C 的占用只由 key 方向宽度 N 决定。快速路径
的 N 固定 256，累加块是 `96 × 256`，单块 `96 × 256 × 4 B = 98 304 B`，两槽轮转合计 196 608 B。
通用路径的 N 取 page 与 128 的公约数（page 128 时 N = 128，page 64 时 N = 64），累加块是
`192 × N`，同样两槽轮转：N = 128 时单槽 98 304 B、合计 196 608 B，N = 64 时合计 98 304 B。
最坏情况是 N = 128 的通用路径与快速路径，都要占掉 262 144 B 的 L0C 的 75%，只剩 25% 余量，
所以 query 段再翻一倍（M = 384）时累加块要到 `384 × 128 × 4 B × 2 槽 = 393 216 B`，放不下。
序列的 query 行数不足 6 时 query 段装不满，这时用 `set_validshape` 把无效行夹掉，不额外开分支。

**块数固定 32 与负载不均。** 工作项数一般不是 32 的整数倍，因此各 Cube 核要跑的轮数
`ceil(工作项数 / 32)` 相同，但最后一轮可能只有部分核有活。例如批次 12、每条序列切成 7 片时
共有 `12 × 1 × 7 = 84` 个工作项，32 个核各跑 3 轮。此时决定耗时的是
`轮数 × 每个 shard 的 key tile 数`，也就是**最长那个核的关键路径**——这正是分片粒度必须选对的原因。

**空输入短路。** 所有批次都为空（`seqused_q` 或 `seqused_k` 为 0）时，host 把 `plan[0]` 置 1，
kernel 直接在设备端写 `-1` / `-inf` / 零候选长度，跳过矩阵乘搬运与逐 tile 握手。

## 核内切分策略、基本块选择与核内 Buffer 划分

**基本块。** 几个决定切分方式的常量（名字与取值都取自 kernel 源码）：

| 常量 | 值 | 含义与理由 |
| :--- | ---: | :--- |
| `GENERIC_MQ` / `PAGED_MQ` | 6 | 每个工作项负责的 query 行数；与 32 个 head 组成 M=192 的矩阵乘基本块 |
| `GENERIC_MM` / `PAGED_MM` | 192 | 矩阵乘的 M 轴规模 = `MQ × 32`，一次矩阵乘覆盖全部 head |
| `GENERIC_HALF` / `PAGED_HALF` | 96 | 把 192 行对半分给两个 AIV 子核（各 3 个 query × 32 head） |
| `PAGED_KR` | 128 | QK 的 K 轴 = head 维度（逻辑 128，MXFP4 下 64 字节） |
| `PAGED_KN` / `PAGED_TILE_N` | 256 | 快速路径每个 key panel 的宽度（对齐 256，正好是两个 128 宽的向量档） |
| `GENERIC_VEC_N` | 128 | 向量侧一次处理的 key 宽度（一个向量寄存器的元素数） |
| `SELECTION_CHUNK` | 16384 | 通用路径选择器一次处理的最大元素数 |
| `STREAMING_SELECTION_CHUNK` | 8192 | 流式选择（`Operation 3`）的 DMA 块大小；请求的输出容量超过 8192 时启用 |
| `CANDIDATES_WIDTH` | 2048 | 候选块归约的分块宽度 |
| `PAGED_HANDOVER_DEPTH` | 2 | 快速路径 Cube↔Vector 的交接深度 |

`SELECTION_CHUNK = 16384` 是从 UB 预算反推出来的。通用路径的选择缓冲按元素数线性增长，
每个元素要占 6 字节（分数面 BF16 2 B + 排列键 UINT16 2 B + 收集键 UINT16 2 B），
另有 4 352 B 的直方图与计数标量（三块 256 项 UINT32 + 两块 128 项 UINT32 + 一块 128 项 UINT16），
再加上收集缓冲多留的 128 个 UINT16（256 B），正好凑成地址式里的 4 608 B；输出区再按 6 B / 元素
（索引 INT32 4 B + 分数 BF16 2 B）算。从 kernel 里这些 tile 的显式地址可以算出选择区的最高地址是
`GENERIC_SELECTION_BASE + 6 × selection_capacity + 4608 + 6 × output_width`，
其中 `GENERIC_SELECTION_BASE = 396 × 128 + 384 = 51 072 B`、`selection_capacity = max(256, SelectWidth)`。
取 `selection_capacity = 16384`、`output_width = 512` 时最高地址到 157 056 B，约占 UB 的 62%；
把块大小翻倍到 32768 时该地址会到 255 360 B，越过 253 952 B 的 UB 容量。
所以 16384 是能放下的最大 2 的幂。可见 key 数超过 `SELECTION_CHUNK` 时，先把分数按 16384 一段
切开、每段各选一遍再归并，每一轮把宽度缩小 `topk` 倍，直到一轮就能装下（`Operation 2`）。
如果请求的输出容量本身超过 8192，就改用流式选择（`Operation 3`）：它一轮只处理 8192 个元素，
UB 占用与请求容量解耦。公开接口的 `topk` 限在 8192 以内，所以这条路只有**候选容量
大于 8192** 时才会走；而候选容量一旦超过 8192，融合路径也已经关闭，于是整个候选阶段变成
「`Operation 4` 归约块内最大分 + `Operation 3` 选择」两次独立启动。

**Buffer 划分。** 硬件侧可用容量（`Ascend950PR_9589.ini`）：UB 253 952 B、L1 524 288 B、
L0A 与 L0B 各 65 536 B、L0C 262 144 B、L2 134 217 728 B。

AIC 侧（每个 MIX 块）。下表按快速路径列；通用路径的 L0A 与累加块都是一整块 `192 × N`（N 取
page 与 128 的公约数），key 方向宽度从 256 换成 N，其余形状不变：

| 缓冲 | 内容 | 说明 |
| :--- | :--- | :--- |
| L1 query | `[qvalid × 32, 128]` FP4 + `[qvalid × 32, 4]` E8M0 | 每个工作项在第一步载入一次，全程驻留 |
| L1 key panel | `[128, 256]` FP4 + `[4, 256]` E8M0 | 双缓冲，每步换一份 |
| L0A | `96 × 128` 的基本块，两槽轮转 | 由 L1 的 query 切成两半分别搬入，两块共用同一份 key panel |
| L0B | `128 × 256`，两槽轮转 | key panel 搬入 |
| L0C | `96 × 256` 累加块，两槽轮转 | 一次矩阵乘的产物，随后折进 UB；两槽合计 196 608 B |

**关键设计：query 常驻 L0A，每个 key panel 喂两个累加器。** query 块只在第一步搬一次，
之后每换一个 key panel 只搬 L0B 与它的 scale，一次搬运服务两次矩阵乘。这样把「K 轴只有 128 的
小矩阵乘」的搬运开销摊到最薄——本算子的 M 轴只有 192，搬运次数比乘加次数更容易成为瓶颈。

AIV 侧（UB）。两条路径的缓冲形状不一样，下面按路径分列：

| 缓冲 | 路径 | 内容 | 说明 |
| :--- | :--- | :--- | :--- |
| 分数面 | 通用 | `[96, 128]` BF16，两块 | 一个 AIV 子核负责 3 个 query × 32 个 head = 96 行 |
| 分数面 | 快速 | `[192, 256]` BF16，两槽交错 | 每槽 128 列，一个槽装半个 key panel |
| 权重 | 通用 | `[3, 32]` F32 | 本子核负责的 3 个 query 的权重 |
| 权重 | 快速 | `[4, 32]` F32 + `[4, 32]` BF16 | 第一步把 F32 权重转成 BF16，之后按 BF16 逐 head 累加 |
| 排序键 | 快速 | `[3, shard_width]` UINT16 | 分数按 BF16 位型异或成单调可比的键后**留在 UB**，不回写显存 |
| 直方图 | 通用 | `[1, 256]` UINT32 × 2 | 高位桶与低位桶各一份 |
| 直方图 | 快速 | `[3, 256]` UINT16 + `[1, 256]` UINT32 × 2 | 每个 query 一份高位桶（跨 key panel 累加），定位时再用高低位桶 |
| 选择工作区 | 两条 | — | 由 `SELECTION_CHUNK` / `STREAMING_SELECTION_CHUNK` 界定 |

定位第 `topk` 大的键分两级：先用键的高 8 bit 建直方图，找出阈值落在哪个高位桶；再只对高位桶相同的
键用低 8 bit 建一次直方图，得到精确阈值。整个过程不需要真排序。`vf.histograms` 一次只覆盖 128 个桶，
所以每一级都要对同一批键调两次（`BIN0` 一次、`BIN1` 一次）才凑成 256 桶。

**快速路径的 UB 账。** 快速路径把一个 key shard 的全部排序键缓在 UB 里，等最后一片到齐后直接在本核
做选择，省掉一次显存往返。这份排序键缓冲是 `3 × shard_width` 个 UINT16，即 6 字节 / key；收集索引
用的缓冲再要 `(shard_width + 128)` 个 UINT16，即 2 字节 / key。AIV 侧 tile 声明的地址跨度是
`108544 + 8 × shard_width` 字节，也就是**每多一个 key 就多占 8 个字节**，shard 宽度因此不能随便放大。
这个跨度是保守上界：分数面的两个槽位用交错寻址，声明的包围盒互相重叠
（`address(slot, row, col) = 256 × slot + 512 × row + 2 × col`），实际访问到的字节比它少。
`max_k = 65536` 时 `shard_width` 为 9472，`max_k = 131072` 时为 18944。

**L0C 到 UB 的折叠。** 每个 AIV 子核只管 96 行分数面（3 个 query × 32 个 head）。通用路径用
`set_validshape` 把工作面夹到 `[96, 128]`（两块合起来覆盖 L0C 的 192 行），快速路径的工作面是
`[192, 256]`。两条路径都把 L0C 的 key 按 128 为一档折进这些行，快速路径每个子核因此同时持有
6 个 128 宽的累加器（3 个 query × 2 档 key），逐 head 累加完再写排序键。

## 接口选择

| 需求 | 采用 | 理由 |
| :--- | :--- | :--- |
| MXFP4 矩阵乘 | `pl.matmul_mx` | 需要显式传 A / B 两侧的 E8M0 块缩放因子（`scale_a` / `scale_b`） |
| 数据搬运 | `pl.move` + `pl.load` / `pl.store` | 需要精确控制 query 常驻、key panel 双缓冲、L0C 的向量折叠顺序 |
| tile 声明 | `pl.make_tile_group` 显式声明地址 / 布局 / mutex ID | 长序列下 UB 预算很紧，必须自己算占用；显式 mutex ID 让跨核事件与 tile 生命周期可控 |
| 向量计算 | `@pl.vector_function` 原生向量函数形式 | 选择器需要 `bit_cast` / `xor` / `shift_right` / `pack` / `histograms` 这类位级与直方图原语 |
| 16 bit 基数选择 | `vf.histograms` + `vf.shift_right` + `vf.pack` | 要比的是 BF16 的位型而不是数值：先异或成单调的 UINT16 键，再按高低 8 bit 两级直方图定位第 k 大，不需要真排序 |
| Cube↔Vector 同步 | `pl.system.set_cross_core` / `wait_cross_core`，收尾用 `sync_all(SyncCoreType.MIX)` | 需要一个不预取的双向握手，且要在工作网格结束后显式排空 |
| 编译期分支 | tiling key 的 22 个 `TilingKeyField`，`Operation` 选算法 | 未选中的算法**不生成设备代码**；`topk = 512` 时置 `FixedTopK = 512`，保留编译期循环上界 |

全部 tile 地址、布局、mutex ID 与跨核事件都由算子显式声明，不依赖编译器的自动流水变换。

## 流水设计

**双向握手。** 打分阶段天然是一条「AIC 产分数、AIV 消费分数」的生产者—消费者链。算子用一对
**反向 token** 显式实现握手，而不是依赖编译器预取：

```text
起始：两个反向 token 各初始化一次
每一步（slot）：
    AIC：等前向 token → 搬 key panel → 矩阵乘 → 置后向 token
    AIV：等后向 token → 折 head / 写排序键 → 置前向 token
结束：工作网格跑完后，AIC 与 AIV 各做一次 sync_all(MIX) 把两个 token 排空
```

- 前向 / 后向 token 用固定的 event id 对：通用路径前向 `[0, 1]`、后向 `[0, 1]`；快速路径前向
  `[0, 1]`、后向 `[2, 3]`，另有向量组 id `[24, 25]`。
- **槽位计数器跨工作项连续推进**，包括 query 段不满 6 行的部分工作项，因此不同工作项之间不需要
  重新对齐，也就没有额外的气泡。
- 「不预取」的含义是两侧都不预取第一步：AIC 先等 AIV 交还缓冲，AIV 先等 AIC 交出第一份分数面。
  这样 UB 与 L0C 的实际占用是确定的，不依赖调度器的推测。
- 交接深度为 2（`PAGED_HANDOVER_DEPTH = 2`）：AIC 在算当前 key panel 时，AIV 可以同时在折上一份
  分数面。

**两级流水叠加。** 对快速路径，实际叠加了三级并行：

1. **AIC 内**：key panel 双缓冲，L0B 搬当前 panel 与算上一个 panel 重叠；
2. **AIC↔AIV**：上面的双向握手，让打分与折 head 重叠；
3. **AIV 内**：写排序键与更新直方图在同一趟遍历里完成——每累加完一个 head，就对 3 个 query 的
   分数各走一遍 `bit_cast`（取 BF16 位型）→ `xor`（翻成单调键）→ `store`（写进 UB 的排序键缓存）
   → `shift_right` → `pack` → `histograms`；最后一个 key shard 到齐后立刻在本核做选择，
   不需要第二遍扫描。

**尾块与无效数据。**

- query 行不满 6：`set_validshape` 夹住有效行，AIC 与 AIV 都以运行期有效行数为循环上界；
- key 不满一个 panel：同样按运行期有效宽度夹住；
- 不可见的 key：置成 0（比任何合法键都小），永远不会被选中，因此不需要额外分支；
- 全空序列：`plan[0]` 置位后走设备端短路，不进入流水。

**取舍。** 打分与选择在快速路径里是**同一个 kernel、同一次 launch**，中间分数留在 UB，不做显存
往返；代价是 UB 预算要被排序键缓冲占用，`shard_width` 因此有上限，分片粒度必须与 UB 容量匹配。
通用路径在 `max_k > 16384` 时把分数写回显存（`scores`），选择阶段作为独立的 `Operation 2` 再
启动，候选块的归约也有独立的 `Operation 4`；换来的是超大 `max_k` 与任意候选容量的通用性。

快速路径的 `Split` 字段取 7，片宽按 `shard_width = ceil(max_k / (256 × 7)) × 256` 由 `max_k`
决定，因此上下文变长时增大的是每个工作项覆盖的 key 数，而不是分片的数量级。
