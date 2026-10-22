# attn_prologue 设计说明（PyPTO-Pro）

对应实现：`op_kernel/attn_prologue.py`（单模块：一个 `@pl.jit(auto_mutex=True)` 的 MIX kernel
`attn_prologue_kernel`、一个公开 host 函数 `attn_prologue`、一个权重加载期工具 `to_nz`）。本文
描述设备侧结构与流水设计；接口与约束见 `README.md`，精度与性能数据见 `RESULTS.md`。

## 计算阶段划分

kernel 内有两条并发指令流：`section_cube`（AIC，矩阵乘）与 `section_vector`（AIV，向量处理），
由两次 `pl.system.sync_all(core_type=MIX)` 全局屏障衔接为四个逻辑阶段。代码注释中四个阶段分别
记作 A1、A2、B、B-epi，本文沿用该编号并在首次出现时给出含义：

| 阶段 | 执行部件 | 内容 |
| :--- | :--- | :--- |
| 阶段 1：QA/KV 投影（A1） | AIC | `qa = X @ Wqa^T`（输出 R=1280 列，切成 10 个 128 列的输出块）与 `kv = X @ Wkv^T`（输出 D=512 列，切成 4 个输出块）。两个矩阵乘都用 MXFP8 的 `matmul_mx`/`matmul_mx_acc` 沿完整的 K=H=5120 归约（每步 K0=256，共 20 步），FP32 部分和经 FIXPIPE 直接写入 GM 上的中间缓冲 `ws_qa[T,R]` 与 `ws_kv[T,D]` |
| 全局屏障 1 | 全部 AIC+AIV | `sync_all(MIX)`，保证中间缓冲对 Vector 可见（MTE3 写与 MTE2 读之间的全局顺序） |
| 阶段 2：归一化与量化（A2） | AIV（64 个行条带） | QR 支路：读 `ws_qa` 的一行（归约维切分场景下先合并两个部分和），沿 R 做 RMSNorm，再做动态 MXFP8 量化，写公开输出 `qr`（E4M3 payload）与 `descale_qr`（成对 E8M0 scale 字节）。KV 支路：读 `ws_kv` 的一行，沿 D 做 RMSNorm，做尾部 RoPE（INTERLEAVE_HALF 约定），饱和到 ±448 后按 scale=1 转 E4M3，最后按 `cache_index` 把整行写入 `kv_cache` 对应 slot |
| 全局屏障 2 | 全部 AIC+AIV | `sync_all(MIX)`，保证 `qr/descale_qr` 的 GM 写入对 Cube 的回读可见 |
| 阶段 3：QB 投影（B） | AIC | `q_proj = qr @ Wqb^T`。A 操作数是从 GM 回读的量化结果：`qr` 以 ND→NZ 转换载入 L1，`descale_qr` 以 ZZ 分形载入；B 操作数是 `Wqb` 的 NZ→ZN 同址别名视图加 NN 分形的 `descale_wqb`；K=R=1280（每步 256，共 5 步）。`[128,128]` 的 FP32 累加结果经 `DualModeSplitM` 搬到跨段共享的 UB 缓冲 `cv_ub`，随后 `set_cross_core(FIX)` 通知 Vector |
| 阶段 4：Q 收尾（B-epi） | AIV（每核 2 个半区，各 64 行） | `wait_cross_core(V)` 等到数据后消费 `cv_ub`：非尾部输出块直接转 BF16；每个 head 的最后一个输出块还要对末尾 Dr=64 列做 INTERLEAVE_HALF RoPE 再转 BF16；结果写 `q[T,N,D]`，然后 `set_cross_core(V)` 归还缓冲槽位 |

阶段 3 与阶段 4 之间以「拍」为粒度做跨核事件握手：Cube 每完成一次 L0C→UB 的搬运并发布，称为
一拍；两个 UB 槽位轮转使用，构成 Cube 与 Vector 之间深度 2 的乒乓缓冲。

两条数值契约贯穿阶段划分（见 `README.md` 功能说明）：QB 投影消费的是**量化后**的 `qr` 而不是
FP32 归一化值；KV 支路**先归一化和 RoPE、再量化**且 scale 固定为 1。

## 核间切分策略

**启动形态**：`attn_prologue_kernel[None, block_dim]`，其中
`block_dim = min(cube_core_num, vector_core_num//2)`，由运行期平台查询得出（32 Cube / 64 AIV
的机器上为 32），每个 block 对应 1 个 AIC 和 2 个 AIV。一次公开调用只启动一次。kernel 内用
`pl.get_block_num()`、`get_block_idx()`、`get_subblock_num()` 推导逻辑核号，AIV 半区号取
`get_subblock_idx()`。

**单二进制、运行期选择工作分配方式**：不使用 tiling key。设 `n_tiles = ceil(T/128)`，运行期在
两种模板中二选一；两个分支都会被 trace，但只有一个执行：

- **单元分布模板**（`n_tiles < 32`，decode 与中小 prefill；代码注释记作 distributed）：把工作
  单元跨步铺到所有核上。阶段 1 以 `(tile, 输出块)` 为单元；当单元总数不少于核数时启用组宽 2
  的列块分组（代码常量 `A1_GRP=2`），让 2 个输出块共享一次 X 的 K 窗口载入。阶段 3 以
  `(tile, head, 输出块)` 为单元、按核数跨步分配，使相邻核在同一时刻读取相邻的 128 行权重带，
  单次 L2 访问的有效字节密度最大。
- **逐 tile 模板**（`n_tiles >= 32`，大 prefill；代码注释记作 split-T）：一个核独占一个 tile 的
  全部 64 个 head × 4 个输出块，`qr` 每个 tile 只从 GM 载入 L1 一次，跨 256 拍复用。

**decode 小 T 的归约维切分**：当 `n_tiles*NA1*2 <= num_cores`（32 核网格上即 `n_tiles==1`，
代码常量 `SK_A=2`）时，阶段 1 把 K=H 沿归约维切成 2 个分区（split-K）：28 个 `(输出块, 分区)`
单元铺满核，各分区把部分和写入中间缓冲的第 `ks*n_tiles*128` 行带，阶段 2 读取时合并。分区宽
H/2=2560 恰为 10 个 K0，NZ 分形网格整除。

**阶段 2 的行条带**：行按扁平 AIV 号（`core_id*2+sub`，共 64 个）跨步分配。`n_tiles<=2` 时把
行循环进一步拆成 `2*t_live` 个独立的 QR/KV 任务铺满 64 个 AIV，缩短屏障之间暴露的向量时间；
prefill 保持逐行循环。归约维切分场景下，两支路各载入两个部分和行带做一次 FP32 `pl.add` 合并
（只有 2 个部分和时，该合并与 Kahan 补偿求和逐位一致，因为补偿项从 0 开始）。

**host 与 kernel 的门控逐字节一致**：中间缓冲的行数（归约维切分时为 `2*n_tiles*128`，否则为
物理行数）在 host 侧用与 kernel 内完全相同的表达式派生，杜绝分配与写入错位。

**全局同步**：恰好两次 `sync_all(MIX)`。阶段 1→2 的中间缓冲依赖与阶段 2→3 的 `qr` 回读依赖都
是 all-to-all 形态（一行 `ws_qa` 由多个核写、`qr` 行被多核读），全局屏障是这类依赖的原生同步；
两侧 section 以相同条件、相同次数、相同顺序到达屏障。

**跨核事件预算**：事件 id 只有 0..15，其中 11/12/13 由 `sync_all(MIX)` 保留（AIV 侧为
12/13）。当前分配：`EV_READY 0..1`（Cube FIXPIPE 发布 → Vector 消费）、`EV_FREE 2..3`
（Vector 归还 → Cube 复用槽位）；导入期 `_check_constant` 校验不与保留 id 冲突。

**mutex id 预算**：Cube 侧 tile group 占用 0..17（NZ/ZN 别名**共享**同一组 id，使 auto_mutex
能看见别名依赖），Vector 侧占用 0..17 加预载槽 18..25；容量与冲突均在导入期校验。

## 核内切分策略：基本块选择、核内 Buffer 划分

### 基本块选择

| 维度 | 取值 | 依据 |
| :--- | :--- | :--- |
| `BASE_M` | 128 | token tile 高度（GEMM 的 M）；经 `DualModeSplitM` 对半分发后每个 AIV 承担 64 行（`ROWS_VEC`） |
| `K0` | 256 | L0 的 K 步长。`matmul_mx` 要求 K 是 64 的倍数；取 256 使每次 GM→L1 的权重/X 突发为 32 KB（MTE2 事务少而大——本 kernel 为 MTE2 带宽受限），同时 L0A/L0B 的 `[128,256]` E4M3 槽为 32 KB，深度 2 恰好占满 64 KB，L0_MX 的 2×2 KB 恰好占满 4 KB |
| `BN_A` / `BN_B` | 128 / 128 | 阶段 1 与阶段 3 的输出列块宽。`[128,128]` FP32 累加器为 64 KB；QB 侧深度 2 乒乓（128 KB）加阶段 1 侧深度 2（128 KB）恰好合占 256 KB L0C。若取 `BN_B=256`，L0B（`[256,256]` E4M3 为 64 KB，深度 2 需 128 KB > 64 KB）与 L0C 预算都会超出 |
| `NK_A` / `NK_B` | 20 / 5 | K 归约步数，分别为 H/K0 与 R/K0 |
| `A1_GRP` | 2 | prefill 下阶段 1 的 X 复用组宽：2 个输出块共享一次 X 的 K 窗口载入，X 的 L2→L1 流量减半（每 tile 14 次降为 7 次）；QA 的 10 块与 KV 的 4 块都是偶数，分组无残块 |
| `SK_A` | 2（门控启用） | decode（`n_tiles==1`）下阶段 1 的归约维分区数：28 个单元铺满 28 个核；分区宽 H/2=2560 为 10 个 K0，NZ 网格整除 |
| `KV_PRE` | 4 | 全局屏障 1 之前预载的阶段 2 KV 任务 cos/sin 行数。门控域内（`n_tiles<=2`，即 `t_live<=256` 行、64 个 AIV 条带）每个 AIV 至多 4 个 KV 任务，预载恰好全覆盖 |
| `b_nz` 深度 | 4 | 权重 K 窗口的轮转深度：4 个 32 KB 连续槽，使 MTE2 载入领先 MTE1 搬出与 MAC 消费至多 3 拍；也等于阶段 3 消费型预取的窗口数（`NK_B-1=4`，保持 ZN 视图的同余对齐） |

### 核内 Buffer 划分

L1（Mat 侧，512 KB，高水位 `L1_END=0x76000` 即 472 KB；各区域起点 8 KB 对齐，导入期校验不重叠、
不越界）：

| Buffer | 声明 | 深度 | 字节 | 用途 |
| :--- | :--- | :--- | ---: | :--- |
| `a_l1` | NZ `[128,256]` E4M3 @0x00000 | 2 | 64 KB | X 的 K 窗口（阶段 1 的 A 操作数），GM 的 ND 数据载入时就地转换为 NZ |
| `sa_l1` | ZZ `[128,160]` E8M0 @0x10000(+0x6E000) | 2 | 40 KB | `descale_x` 整行驻留，逐 K 步切片进 L0A_MX |
| `b_nz`/`b_zn` | NZ `[128,256]` / ZN `[256,128]` E4M3 @0x16000（槽 2,3 @0x26000） | 4 | 128 KB | 权重的 K 窗口。**同一块字节的 NZ/ZN 双视图别名，共享 mutex id**：`A @ W^T` 的 B 操作数经 ZN 视图 `pl.move` 进 L0B |
| `sb_l1` | NN `[160,128]` E8M0 @0x36000 | 2 | 40 KB | `descale_wqa/wkv`（阶段 1 的 B 侧 scale，以 `order=[1,0]` 转置载入） |
| `qr_l1` | NZ `[128,1280]` E4M3 @0x40000 | 1 | 160 KB | 阶段 3 的 A 操作数：公开输出 `qr` 的 GM 回读；逐 tile 模板下每个 tile 只驻留一次 |
| `sq_k` | ZZ `[128,40]` E8M0 @0x68000(+0x74000) | 2 | 10 KB | `descale_qr` 整行（阶段 3 的 A 侧 scale） |
| `sw_k` | NN `[40,128]` E8M0 @0x6A000(+0x6C000) | 2 | 10 KB | `descale_wqb`（阶段 3 的 B 侧 scale） |

L0A/L0B/L0_MX（每个 core group）：`l0a` 为 NZ `[128,256]`、`l0b` 为 ZN `[256,128]`，各深度 2
（槽 0x0000/0x8000，各 64 KB 恰好占满）；`scale_a_l0` 为 ZZ `[128,8]`、`scale_b_l0` 为 NN
`[8,128]`，深度 2（槽 0x0000/0x800）。MX 矩阵乘的硬约束「scale 地址 = 数据地址 >>4」由显式地址
声明满足（0x8000>>4=0x800）；轮转时数据槽与 scale 槽按同一 `.next()` 序联动。

L0C（累加器，256 KB）：`a1_acc` 为 NZ fractal=1024 的 `[128,128]` FP32、深度 2
（0x00000/0x10000，阶段 1 的列块分组需要 2 个累加器），`q_acc` 同形、深度 2（0x20000/0x30000，
供阶段 3/4 乒乓）。`compact=1` 的累加器窗口在首个 matmul 之前一次设满物理窗，matmul 与搬运之间
不再改窗。

UB（Vec 侧，用户预算 216 KB，高水位 `UB_END=0x27E00` 即 163.5 KB）：

| 槽 | 字节 | 用途 |
| :--- | ---: | :--- |
| `cv_ub` ×2 | 64 KB | 跨段握手槽 `[64,128]` FP32（声明在两个 section 之外，深度 2 乒乓） |
| `qa_row`/`qa_row2`、`gamma_qr` | 5.25+5.25+5.25 KB | 阶段 2 QR 支路：中间缓冲行（归约维切分时为两个部分和行带）与归一化权重（`QR_PAD=1344` 保证末组 64 lane 载入不越 tile） |
| `qr_payload`、`exp`、`scale_u8` | 1.25+0.25+0.06 KB | 量化产物与指数字段的暂存 |
| `kv_row`/`kv_row2`、`gamma_kv`、`kvb`、`cos_row`/`sin_row` | 2+2+2+0.5+0.5 KB | 阶段 2 KV 支路 |
| `q_bf16` ×2 | 32 KB | 阶段 4 输出缓冲 `[64,128]` BF16（深度 2，与 `cv_ub` 同拍） |
| `cos_ub`/`sin_ub` | 16+16 KB | 阶段 4 的 RoPE 表 `[64,64]` FP32（按 tile 变化点提升载入） |
| `cos_pre`/`sin_pre` ×4 | 2+2 KB | 全局屏障 1 之前预载的阶段 2 KV cos/sin 行（512 B 槽距） |

## 接口选择

- **Cube 侧显式三段式**：`pl.load`（GM→L1）、`pl.move`（L1→L0、L0C→UB）、
  `pl.matmul_mx[_acc]`，不使用高层 gemm 封装。原因是需要精确控制 L1/L0 地址以满足 MX scale 的
  「地址 >>4」耦合、让 NZ/ZN 别名共享 mutex id，并独立管理各缓冲的轮转深度。
- **`A @ W^T` 与 NZ 权重**：FRACTAL_NZ 权重按 `[N,K]` 存储，不能经 `order` 转置读取。实现用
  同址双视图：以 NZ `[BN,K0]` 载入，再以 ZN `[K0,BN]` 视图搬出，得到 L0B 需要的转置分形；两个
  视图共享 mutex id，使 auto_mutex 看见别名依赖。
- **E8M0 scale 的 GM→L1**：源张量 rank≥3 且末维是长度为 2 的物理 phase 轴（成对存储的公开布
  局）；phase 轴偏移恒为 0，`order` 不选该轴。A 侧 scale 落 ZZ 分形、B 侧 scale 以
  `order=[1,0]` 转置落 NN 分形——这是 ZZ/NN 分形唯一可用的生成路径。L1→L0_MX 用 `pl.move`
  （ZZ→ZZ / NN→NN）。
- **量化结果经 GM 回读反馈**：阶段 2 把 `qr/descale_qr` 写入公开输出张量（GM），全局屏障 2 之后
  Cube 回读作为阶段 3 的 A 操作数（payload 走 ND→L1 NZ，scale 走 phase 轴规则）。选择该路径的
  原因：UB→L1 的 ND→NZ 就地转换对 E8M0 scale 不存在 ZZ/NN 生成通路；且 GM 路径让阶段 2 的行
  条带与阶段 3 的单元分配完全解耦（生产者与消费者的形状无需匹配），全局屏障同时提供写后读的
  全局一致性。
- **累加器到 UB**：`pl.move(..., acc_to_vec_mode=DualModeSplitM)`，valid_M 先向上对齐到 2 再
  对半分，两个 AIV 各得 64 行。被 `pl.move` 排空的 matmul 不传 `phase=`（没有 STPhase 消费者时
  传 phase 会挂起 unit_flag）。
- **Vector 数值全部走 VF**（`@pl.vector_function`）：RMSNorm 用 `vf.reduce_sum` 求行平方和，
  `vf.full(reg, preg, pos=DuplicatePos.LOWEST)` 把 lane0 广播到全寄存器，再 `vf.sqrt` 与
  `vf.div(one, root)` 求倒数（VF 没有 rsqrt；`1/R`、`1/D` 是模块级常量，VF 内禁止标量浮点除
  法）。量化组的共享指数取 `bit_cast` 到 i32 后右移 23 位、减 `127+8`、再 `maxs(-127)` 截断；
  全零组用 `select` 把 scale 字节置 0。倒数 `2^-e` 由 FP32 指数字段直接构造（取负、加 127、
  左移 23、`bit_cast` 回浮点），不出现除法。E4M3 转换用
  `vf.astype(..., CAST_RINT, SaturateMode.ON)` 且先显式饱和到 ±448；payload 密排用
  `StoreDist.PACK4`（b32 取低字节），BF16 输出用 `bit_cast(u32)` 后 `StoreDist.PACK`。RoPE 的
  去交错用钳位到 0..63 的寄存器 `gather` 索引（避免 DINTLV 指令末行过读），两次乘法与一次加法
  分开执行以保持舍入次序。每次 UB 暂存的写后读之间插 `vf.mem_bar(VST_VLD)`。
- **分页 KV 写回**：`slot = pl.getval(cache_index, row)`（INT64）后
  `pl.store(kv2d, kvb, [slot, 0])`，其中 `kv_cache` 以 `make_tensor` 展平为 `[P*BS, D]` 视图；
  整行 512 B 一次落盘（不小于 64 B 的安全粒度）。`cache_index` 的唯一性由接口契约保证。
- **同步分层**：核内 pipe 顺序交给 `auto_mutex=True` 加 tile group 的 `mutex_ids`（跨 op 依赖
  一律用 `make_tile_group` 声明，不用裸 `make_tile`）；Cube 与 Vector 的握手用
  `set/wait_cross_core`（FIX↔V，深度 2 的信用额度，`beat>=2` 守卫，退出前自然配平）；全局相位
  用两次 `sync_all(MIX)`。

## 流水设计

**阶段 3 与阶段 4 之间深度 2 的乒乓**（「拍」指本核处理的第几个 `(head, 输出块)` 单元）：

```text
AIC: matmul_mx×5 → acc[beat%2] → move(cv_ub[beat%2], DualModeSplitM) → set(FIX, beat%2)
AIV: wait(V, beat%2) → RoPE/BF16 → store q → set(V, 2+beat%2)
AIC: wait(FIX, 2+beat%2)  (beat>=2 时) → 复用 cv_ub 槽
```

Cube 的第 beat+1 个 QB 矩阵乘与 Vector 的第 beat 个收尾重叠。两侧 section 以相同的单元枚举顺序
（同一个 `pl.range(core_id, n_work, num_cores)` 解码）保证拍号对齐。

**MTE2 预算下的载入调度**。kernel 为 GM→L1 载入带宽受限，流水设计围绕「减少字节、增大突发、
隐藏延迟」三点展开：

- **权重窗口深度 4 轮转**：`b_nz` 的 4 个槽使 MTE2 载入领先 MTE1 搬出与 MAC 消费至多 3 拍，
  打断单槽/双槽下「第 k+1 步载入必须等第 k 步读出完成」的写后读冲突（WAR）串行。
- **小 scale 缓冲深度 2 轮转**（`sw_k/sa_l1/sq_k`）：消除逐拍、逐单元边界上「载入等待上一拍
  L1→L0_MX 搬出完成」的停顿，机制与权重轮转相同。
- **单元分配的带局部性**：单元分布模板下阶段 3 以 `(tile, head, 输出块)` 为单元、按核数跨步，
  相邻核在同一时刻读取相邻的 128 行权重带，单次 L2 访问的有效字节密度最大，41.9 MB 的权重流
  以聚合流带宽满速运行。
- **阶段 1 的 X 复用（组宽 2）**：一个 X 的 K 窗口喂 2 个输出块的矩阵乘，X 的 L2→L1 流量减半；
  decode 退回逐块分配（更多核发 MTE2、墙钟更短），由 `n_tiles*NA1G >= num_cores` 门控自适应。
- **屏障窗口的载入隐藏**：两个全局屏障各有一段空闲窗口。阶段 1 期间 64 个 AIV 空闲：归一化权重
  与阶段 2 首批 `KV_PRE=4` 行 cos/sin 在屏障 1 **之前**预载（它们是纯输入，无依赖）。阶段 2 期间
  32 个 AIC 空闲：阶段 3 首批 4 个权重窗口加首个 `descale_wqb` 在两个屏障之间预取，且首拍
  **实际消费**预取槽（跳过对应载入；`.next()` 全部无条件推进，两条路径的轮转计数一致；逐 tile
  分支以 `pre=False` 折叠为与不开预取逐指令相同）。
- **冗余载入按变化点提升**：单元分布模板下阶段 3 的 `qr_l1/sq_k` 仅在本核首个单元或 tile 变化
  时装载；阶段 4 的 `cos_ub/sin_ub` 遵循同一规则（RoPE 表只依赖行偏移）。
- **阶段 2 任务化（decode）**：`n_tiles<=2` 时把 QR/KV 拆成 `2*t_live` 个独立任务铺满 64 个
  AIV，缩短屏障之间暴露的向量时间；任务到行的映射是确定性条带化，与预载槽一一对应。
- **尾块处理**：`set_validshape` 在每次观察 extent 的操作前设置（GM 面用逻辑行数 `rows_full`，
  NZ 分形面用 16 对齐的 `rows_r`，累加器保持全物理窗）。`DualModeSplitM` 的 valid_M 先对齐到 2
  再对半，Vector 侧按半区重算自己的 `store_rows` 并守卫 `store_rows>0`。物理行数
  `ceil(T/128)*128` 只服务中间缓冲与 NZ 对齐，公开输出与 KV 写回严格按逻辑 T。
