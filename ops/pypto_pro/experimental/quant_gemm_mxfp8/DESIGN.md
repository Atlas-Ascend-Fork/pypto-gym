# QuantMatmul MXFP8 设计说明（PyPTO-Pro）

对应实现：[quant_matmul_mxfp8.py](op_kernel/quant_matmul_mxfp8.py)。模块有一个公开 host 函数、**一个** `@pl.jit(arch="a5", auto_mutex=True, tiling_key=MxGemmKey)` 入口 `_quant_matmul_mxfp8_kernel`，以及两个 `@pl.vector_function` 辅助函数 `_add_partial_in_place` 与 `_cast_first_partial_in_place`（**仅 Profile 0** 的归并使用；其余 profile 不含 Vector 段）。kernel 内按 tiling key 的 `Profile` 字段在**编译期**选择四种调度之一，因此每个编译变体只 trace 自己那一条分支。host 校验属性、推导 shape、选定 profile 与启动块数、按 profile 分配输出（Profile 0 另有 FP32 workspace 用于跨核部分和，其余 profile 无 workspace），再启动 kernel。接口见 [README.md](README.md)，测试及实测记录见 [RESULTS.md](RESULTS.md)。

支持域是**单一**的——`M`/`N` 任意正整数、`K` 为 128 的倍数、rank-2（WeightNZ 另需 `N` 为 32 的倍数）。一个 kernel 服务整个支持域，调度由 `Profile` 这个 tiling key 字段在编译期选定（`Profile` 占 2 bit，取值 0–3）：

| Profile | 适用 shape | 调度 | 启动块数 |
| :--- | :--- | :--- | :--- |
| 0 | `(96,1792,5120)` | split-K：7 个 N 块 × 4 段 K、FP32 workspace、全 AIC FIX 屏障、56 个 AIV 归并（MIX kernel） | 32（28 有效） |
| 1 | `(96,32768,1280)` | A 常驻：`a_l1` depth=5、`sa_l1` depth=1，每 worker 4 个 N 块 | 32 |
| 2 | `(96,5120,8192)` | 滑动窗口：每 worker 1 个 N 块，沿完整 K 归约 | 20 |
| 3 | 支持域内的其余 shape | 纯 Cube，复用 Profile 2 的 tiling 与地址表；M/N 尾块、K 尾窗、grid-stride 多轮 | `min(总块数, 32)` |

> **⚠ Profile 的选择不可改为「一律用 Profile 3」。** 把 host 的判据置空（`SHAPES = ()`）、强制三组已实测 shape 走 Profile 3，功能仍然正确（全部在 CPU FP64 golden 门限内通过），但会失去两样东西：
>
> 1. **`(96,1792,5120)` 的 K 并行度**。该 shape 只有 `ceil(96/96) × ceil(1792/256) = 7` 个输出块，Profile 3 按 `min(总块数, 32)` 只启动 **7 个 worker**、每个沿完整 `K=5120` 端到端归约；Profile 0 把 K 切成 4 段，用 **28 个有效 AIC**（7 个 N 块 × 4 段）并行，再由 56 个 AIV 归并。改用 Profile 3 等于把该 shape 的 Cube 占用从 28/32 降到 7/32。
> 2. **该 shape 的 FP32 累加分组**。split-K 是「4 段各自 FP32 累加 → 归并」，Profile 3 是「1 个 worker 沿完整 K 累加」，两者分组不同则末位 BF16 舍入可能不同。实测该 shape 有 `36/172032` 个元素的末位 BF16 不同（两种调度都在门限内通过）。另两组 shape 完全一致：Profile 3 的 grid-stride 恰好复现了 Profile 1「32 块各 4 轮」与 Profile 2「20 块各 1 轮」的工作划分，且它们的 K 归约本来就是单 worker 完成的。
>
> 同级 `quant_bmm_mxfp8` 不存在这个取舍：它只有一种调度，性能实测 shape 是其动态遍历的特例（总块数恰为 32、单轮满波、所有 `pl.min` 取到上界）。

本文中，worker 指一个 AIC；AIV 指与 AIC 配合的向量核，本机每 AIC 配 2 个、共 64 个。GM 是设备全局内存，UB 是 AIV 的本地缓冲区；L1、L0A、L0B 和 L0C 用于 Cube 数据及累加结果。MTE2 负责 GM→L1，MTE1 负责 L1→L0，FIXPIPE 将 Cube 累加结果交给 AIV 或 GM。tile 指一个 `[96,256]` 的输出块（Profile 3 的尾块处按 `set_validshape` 收窄）。

**四个 profile 不是同一算法的四组参数**：它们在 L1 布局、驻留策略、启动几何、是否有 Vector epilogue 与是否产生跨核部分和上都不同。下文按 profile 分别说明，不做统一抽象；Profile 0/1/2 的 shape 由 `SHAPES[Profile]` 给出并写进调度本身，Profile 3 的 shape 是运行时值。

**Profile 3 不引入第五种算法**：它直接复用 **Profile 2 的那一份 tiling 与 L1/L0 地址表**（`G_BM/G_BN/G_BK = 96/256/128`、`G_KL1 = 256`、`G_SCALE_KL1 = 4096`），并**去掉 split-K**，因此是纯 Cube、无 workspace、无 Vector 段。选 Profile 2 的理由有两条：其一，它的地址表与同级 `quant_bmm_mxfp8` 完全相同，而后者已在板上验证过部分 M / 部分 N 可行（最小到单个 16×16 fractal）；其二，Profile 0 的 split-K 与 Profile 1 的 A 常驻都把 shape 写进了调度本身（N 块数 7、K 分片数 `⌊32/7⌋=4`、`a_l1` depth=5、`sa_l1` depth=1），泛化会直接破坏其前提。代价是 Profile 3 对小 N 大 K 的 shape 拿不到 split-K 并行度——这符合「支持域内除已实测的 18 个 shape 外不做性能声明」的约定。

## 计算阶段划分

M=96 恰好等于基本块 `BM`，因此 M 维不分块；N 按 256 分块；K 是归约维。三组的 K 处理方式不同：Profile 0 把 K 切成 4 段分给 4 个 worker 并产生部分和，Profile 1 与 2 由单个 worker 沿完整 K 归约。

| 阶段 | 执行部件 | 粒度 | 实现逻辑 |
| :--- | :--- | :--- | :--- |
| A BiasTable 就位 | AIC | 每个需要 bias 的输出块一次 | `pl.load(bias_mat, bias, [0, n_offset])` 把 FP32 bias 从 GM 载入 L1，再 `pl.move(bt, bias_mat)` 送入 `MemorySpace.Bias`。Profile 0 只在 split 0（`worker%4==0`）执行。本阶段不产生任何乘加 |
| B scale 窗口载入 | AIC MTE2 | 每 `scale_kl1` 窗口一次 | A scale 以 ZZ、B scale 以 NN 布局载入。Profile 0 每分片 2 个窗口（1024 + 尾窗 256）；Profile 1 的 `scale_kl1=1280` 覆盖全 K 故只 1 次，且 A scale 只在 `output_iter==0` 载入后跨 4 个 N 块复用；Profile 2 为 2 个 4096 窗口 |
| C 数据载入 | AIC MTE2 | 每 `kl1` 一次 | 载入顺序固定为 scale、A、bias（仅 iter0）、B。Profile 1 的 A 只在 `output_iter==0` 载入 5 个面板后复用；WeightNZ 时 B 走原生 GM NZ→L1 NZ 搬运。**B 的两条载入路径都带 `l2_cache_mode=TLoadL2Hint.NotAllocKeep`**；A、两份 scale、bias 与输出存储都不带 hint |
| D L1→L0 提取 | AIC MTE1 | 每 `bk=128` 一次 | A、B 数据与其 E8M0 plane 成对提取。plane 组偏移在 Profile 1 为 `(data_k+l0_k)//32`（因 A scale 覆盖全 K、无窗口起点），其余为 `(data_k-scale_k+l0_k)//32`（当前 scale 窗口内的组号） |
| E MX 乘加 | AIC Cube | K 升序，次数见下表 | 首块首段用 `pl.matmul_mx`：需要 bias 时该指令同时消费 FP32 BiasTable（`init_with_zero=false`），否则省略该操作数；其余各次为 `pl.matmul_mx_acc`（无 bias 形参）。`data_k+l0_k+bk == local_k` 时用 `AccPhase.Final`，否则 `Partial` |
| F 结果写出 | AIC FIXPIPE | 每输出块一次 | Profile 0 写 GM workspace 的 `[n_idx, worker%4, 0, 0]`；Profile 1/2 以 `STPhase.Final` 直接写 BF16 输出 |
| G 跨核完成屏障 | 全部 32 个 AIC | 仅 Profile 0，一次 | `set_cross_core(FIX, event 0, INTER_BLOCK)` 后立即 `wait_cross_core(FIX, event 0, INTER_BLOCK)`，再 `set_cross_core(FIX, event 1)`。**空闲的 4 个块也必须参与**，否则配对交接会让 AIV 过早读到别的块尚未写完的部分和 |
| H Vector ICache 预取 | 仅 Profile 0 的 56 个有效 AIV | 一次 | 在跨核等待与有效 AIV 判断之后、GM 绑定之后、归并之前执行 `icache_preload(2)` |
| I Vector 归并 | 仅 Profile 0 的 56 个有效 AIV | 每 AIV 一次 | 载入四份 `[12,256]` FP32 部分和，三次整段 UB 原地 Add，再同址 BF16 RNE Cast，写回对应输出行 |

**阶段 I 的三次原地 Add**：归并不是把四份部分和读进寄存器连续求和后写入独立 BF16 缓冲，而是三次「整段 UB 原地累加」再同址转换：`_add_partial_in_place(src, 1/2/3)` 各遍历 `off = 0,64,...,3072`，读 `[0,off]` 与 `[i,off]`、相加后写回 `[0,off]`。三次 Add 由同一原生 VF 分别调用，**跨调用的 UB 读写依赖交给 `auto_mutex`**——用独立的 VF 调用而不是一个融合循环，正是为了让每次 UB 读/写边界都暴露给框架。这样做的收益是 UB 只需一份 `[4,3072]` FP32 声明（49152 字节），不必再为 BF16 结果另开 6144 字节。

**同址窄化转换**：`_cast_first_partial_in_place` 用 `pl.reinterpret(src, dtype=BF16, shape=[12,256])` 得到与 FP32 源**共享地址 0 及同一 allocation/mutex 所有权**的目标视图，因此不声明额外的 BF16 TileGroup。正向遍历时第 `off` 次迭代读字节 `[4*off, 4*(off+64))`、写字节 `[2*off, 2*(off+64))`，写区间的上界 `2*off+128` 恒小于后续迭代的读起点 `4*(off+64)`，故不会覆盖后续迭代尚需读取的输入。缩窄 store 前显式使用 `vf.mem_bar(mode=VLD_VST)` 固定读→写的重叠依赖，**不能把 `auto_mutex` 对 VF 调用之间的同步误当成 VF 内部别名访问自动安全**。转换后每个 b32 容器只含一个 BF16，因此必须用 `dist=StoreDist.PACK_B32` 精确压实 64 个 BF16，普通全宽 b16 store 不等价。

- **原生 WeightNZ 搬运**：WeightNZ 使用原生 `pl.Tensor[..., pl.DT_FP8E4M3FN, pl.NZ]` 和 `pl.load`，以逻辑 K/N 偏移直接完成 GM NZ→L1 NZ 搬运；两份 scale 保持 ND。不做数值转换、不分配中间权重。

**Profile 3 的阶段序列**是上表去掉 G/H/I、并把 B/C/D/E 的循环边界换成运行时值：

```text
m_tiles = ceil(M/96)   n_tiles = ceil(N/256)   total = m_tiles * n_tiles
for tile_id in pl.range(core_id, total, num_cores):        # grid-stride，可多轮
    m_offset, n_offset = 解码(tile_id)
    vm = min(96, M - m_offset)     vn = min(256, N - n_offset)      # M/N 尾块
    for scale_k in pl.range(0, K, 4096):                            # scale 窗口
        scale_len = min(4096, K - scale_k)                          # 末窗收缩
        for data_k in pl.range(scale_k, scale_k + scale_len, 256):   # data_k 为绝对偏移
            data_len = min(256, K - data_k)                          # K 尾窗
            for l0_k in pl.range(0, data_len, 128):
                s_off = (data_k - scale_k + l0_k) // 32               # 窗口内 E8M0 列
```

尾块一律用 `pl.set_validshape` 收窄对应 tile：L1 的 `a_l1[vm,data_len]`、`b_l1[data_len,vn]`、`sa_l1[vm,scale_len/32]`、`sb_l1[scale_len/32,vn]`，L0 的 `a_l0[vm,128]`、`b_l0[128,vn]` 与两个 scale plane，以及 `acc[vm,vn]` 与 `bias[1,vn]`。Cube 接受部分 M 与部分 N，最小可到单个 16×16 fractal，因此**不需要 host 侧 padding**。

**K 尾不补零**：`K % 128 == 0` 已隐含 `K % 64 == 0` 与 `K % 32 == 0`，因此打包 scale 的 64 元素窗口整除、MX 组宽 32 也对齐，kernel 不需要 K 方向补零。`K` 不是 `G_KL1` 或 `G_SCALE_KL1` 整数倍时，末窗由 `pl.min` 收缩并用 `set_validshape` 收窄，同样不需要补零。

## 核间切分策略

- **启动形态**：只有一个入口 `_quant_matmul_mxfp8_kernel[None, workers, key]`，`workers` 由 host 按 profile 给出：Profile 0/1 为 32、Profile 2 为 20、Profile 3 为 `min(总块数, 32)`。核数是编译期常量或 host 侧算出的常量，不运行期查询平台。
- **Profile 0（split-K）**：`N=1792` 只给出 `1792/256 = 7` 个输出块，远少于 32 个 AIC，若每块由一个 worker 沿完整 K 归约则只有 7/32 的 Cube 占用。因此把 K 再切一刀：K 分片数 `= ⌊32/7⌋ = 4`，每分片长度 `= ⌈5120/4⌉ = 1280`（已是 256 的倍数，无需再对齐）。得 7 个 N 块 × 4 个 K 分片 = **28 个有效 AIC**，在 32 块的混合启动中余 4 个空闲块。`n_idx = worker//4`、`k_start = (worker%4)*1280`，bias 只在 `worker%4 == 0` 的分片挂载（否则会被加 4 次）。每分片沿 1280 做 10 次 `BK=128` 的 MX，FP32 结果经 FIXPIPE 写 workspace；workspace 有效布局为 `[7,4,96,256]`，共 2752512 字节，块 `(n_idx, split)` 的偏移是 `(n_idx*4 + split)*96*256` 个 FP32 元素。归并阶段 56 个 AIV 各负责 12 行 × 256 列：`n_tile = vector_idx//8`、`row = (vector_idx%8)*12`，`7 × 8 × 12 = 96` 行正好覆盖全部输出。
- **Profile 1（A 常驻）**：`N=32768` 给出 128 个输出块，32 个 AIC 全部有效、每核顺序处理 4 个（`rounds=4`，`n_offset = (worker + output_iter*32)*256`），四轮覆盖 128 块。`K=1280` 使整份 A 只有 5 个 `[96,256]` NZ 面板，可以一次全载并跨 4 个输出块复用，A 的 GM 读取量因此降到 1/4；B data 四槽、B scale 双槽、bias 与 L0C 的槽位游标跨 N 块连续推进，不随 `output_iter` 复位。
- **Profile 2（滑动窗口）**：`N=5120` 给出 20 个输出块，少于 32，故**启动 20 个 AIC**、每核 1 块、`rounds=1`，沿完整 `K=8192` 归约；A、B 各四槽流式推进，scale 分两个 4096 窗口刷新。启动核数取 20 而非 32，是因为多出的 12 块无工作可做。
- **Profile 3 的块数**：`total = ceil(M/96) × ceil(N/256)`，取 `min(total, 32)`，即不写死为 20 或 32，而是按输出块数取小，避免块数少于核数时空转。
- **Profile 3 的工作分配**：`for tile_id in pl.range(core_id, total, num_cores)`，`num_cores` 由 `pl.get_block_num()` 运行期取得，块号按 `m_tile → n_tile` 解码，即 `m_offset = (tile_id // n_tiles)*96`、`n_offset = (tile_id % n_tiles)*256`。这是通用的满波/多轮调度，**不使用 split-K**：没有 `k_start` 分片、没有空闲块、没有部分和、没有 Vector 段。Profile 2 的「20 块各 1 个 N 块」恰好是 `total=20、num_cores=20、每核一轮、m_tiles=1` 的特例。
- **跨核同步**：只有 Profile 0 需要，即阶段 G 的全 AIC FIX 完成屏障与阶段 I 前的 AIV 等待。Profile 1/2/3 每核独占输出块、不产生部分和，因此没有 workspace、跨核事件或全局屏障。
- **负载均衡**：Profile 0 的 28 个有效块各承担一个 `[96,256]×1280` 分片，工作量相同，但 4 个空闲块使按启动块数计算的均衡度为 `28/32`；归并阶段 56 个 AIV 各承担同形状子块，按构造均衡。Profile 1 每核 4 个同形状输出块、Profile 2 每核 1 个，均按构造均衡。Profile 3 下**均衡不由构造保证**：尾块的 `vm`/`vn` 更小、工作量更少，且 `total` 不是核数整数倍时最后一轮会有 worker 空转。分核时间离散度未测。

## 核内切分策略：基本块选择、核内 Buffer 划分

### 基本块选择

| 维度或参数 | Profile 0 | Profile 1 | Profile 2 | 与代码实现的关系 |
| :--- | :--- | :--- | :--- | :--- |
| shape `(M,N,K)` | 96×1792×5120 | 96×32768×1280 | 96×5120×8192 | `SHAPES` 常量，由 `SHAPES.index((m,n,k))` 选定 profile |
| 路径 | split-K | A 常驻 + 多轮 | 滑动窗口 | 见「核间切分策略」 |
| `BM/BN/BK` | 96/256/128 | 96/256/128 | 96/256/128 | `BK=128` 来自 L0 半区最大 K = `32768/max(96,256)`；`BM=96=M` 故 M 不分块；`BN=256` 与 `BM` 一起决定 L0C 的 `[96,256]` FP32 tile |
| `KL1` | 512 | 256 | 256 | 由 L1 容量反解，见下 |
| `SCALE_KL1` | 1024（尾窗 256） | 1280（覆盖全 K） | 4096（两个窗口） | 由 L1 容量与「窗口数越少、scale 重载越少」共同决定，见下 |
| L1 data 槽数 | A、B 各 2 | A 5、B 4 | A/B 各 4 | Profile 0 的 L1 面板最大（`KL1=512`），只放得下双槽；Profile 1 的 A 全载需 5 槽；Profile 2 用四槽换取更宽的搬运/计算重叠 |
| 启动块数 / 有效 AIC | 32 / 28 | 32 / 32 | 20 / 20 | 见「核间切分策略」 |
| `rounds`（每核输出块数） | 1 | 4 | 1 | 代码中的 `output_iter` 循环次数 |
| `local_k`（每核归约长度） | 1280 | 1280 | 8192 | Profile 0 是 K 分片长度，其余是完整 K |
| MX 次数（全核合计） | 280 | 1280 | 1280 | P0：28×10；P1：32×4×10；P2：20×64 |
| bias 挂载次数 | 7 | 128 | 20 | P0 只有 `worker%4==0` 的 7 个分片；P1 每核每输出块一次；P2 每核一次。均表现为该输出块首条 MX MMAD 携带 BiasTable 操作数，**没有独立的 seed MMAD**，故 MMAD 总数恰等于上行的 MX 次数 |
| A 载入次数 | 84 | 160 | 640 | P0：28×3（`⌈1280/512⌉=3`）；P1：32×5（仅 `output_iter==0`）；P2：20×32（`8192/256`） |

**`KL1` 由 L1 容量反解**。L1 是 524288 字节，data 面板按 `KL1` 线性放大：A 面板 `96×KL1` 字节、B 面板 `KL1×256` 字节，两者之和记为「A+B 单槽」。

| `KL1` | A+B 单槽 | Profile 0（双槽）+ scale/bias | Profile 2（四槽）+ scale/bias |
| ---: | ---: | ---: | ---: |
| 1024 | 98304 + 262144 = 360448 | 720896 ✗ 超 L1 | — |
| 512 | 49152 + 131072 = 180224 | 360448 + 22528 + 2048 = **385024** ✓ | 720896 ✗ 超 L1 |
| 256 | 24576 + 65536 = 90112 | — | 360448 + 90112 + 2048 = **452608** ✓ |

故 Profile 0 取 512（再大一档就放不下双槽），Profile 2 取 256（`KL1=512` 时四槽已超）。Profile 1 的 A 要全载 5 个面板、B 要四槽，`KL1=512` 时仅 B 四槽就需 524288 字节，故同样取 256。

**`SCALE_KL1` 由 L1 余量与窗口数权衡决定**。scale 面板是 `96×(SCALE_KL1/32)`（A，ZZ）与 `(SCALE_KL1/32)×256`（B，NN），各双槽。Profile 2 若取 full-cover 8192，A/B scale 双槽合计 `2×24576 + 2×65536 = 180224` 字节，加上 data 四槽的 360448 得 540672 > 524288，放不下；取 4096 则 scale 双槽合计 90112，总量 452608 ✓，代价是 `K=8192` 分成两个窗口、scale 重载一次。Profile 1 的 `K=1280` 恰好能被 `SCALE_KL1=1280` 全覆盖（A scale 单槽 3840、B scale 双槽 20480，总量 411392 ✓），因此每个输出块只载一次 scale，且 A scale 还能跨 4 个 N 块复用。Profile 0 的 `local_k=1280` 取 `SCALE_KL1=1024`，得 1024 + 尾窗 256 两个窗口。

**尾块必须用有效 shape**：Profile 0 每分片 `local_k=1280` 按 `kl1=512` 切分，最后一段只有 256；`scale_kl1=1024` 的最后一个窗口同样只有 256。代码用 `pl.min(scale_kl1, local_k - scale_k)` 与 `pl.min(kl1, local_k - data_k)` 求出 `scale_len` / `data_len`，再以 `pl.set_validshape` 收紧描述符，**不能从已分配的 512/1024 长度读取到下一个分片**。

上述常量都是编译期立即数，**不是板上 tiling dump**：host 不做 tiling 计算、不查询平台、不读 SDK ini。好处是没有 host tiling 开销、行为可复现；代价是它们绑定目标 SKU（32 AIC / 64 AIV、L1 524288、L0A/B 每槽 32768、L0C 262144、L2 128 MiB 字节），换 SKU 或受限核数配置需要按上面的容量算式重新核算。

### 核内 Buffer 划分

**L1（按 profile 不同）**。Profile 0（`kl1=512`、`scale_kl1=1024`，A、B 各双槽）：

| 缓冲区 | 声明布局与地址 | 深度 | 单槽字节 |
| :--- | :--- | :--- | ---: |
| `a_l1` | Mat `[96,512]` FP8 @ `[0, 262144]` | 2 | 49152 |
| `b_l1` | Mat `[512,256]` FP8 @ `[49152, 311296]` | 2 | 131072 |
| `sa_l1` | Mat ZZ `[96,32]` E8M0 @ `[180224, 442368]` | 2 | 3072 |
| `sb_l1` | Mat NN `[32,256]` E8M0 @ `[183296, 445440]` | 2 | 8192 |
| `bias_l1` | Mat `[1,256]` FP32 @ `[191488, 453632]` | 2 | 1024 |

前半区依次为 A 两槽 `0..49151`、`49152..98303`，B 两槽 `49152` 起（与 A 交错排布，槽间隔 262144 把同名槽分到两个半区），A scale @180224、B scale @183296、bias @191488，末端 192512；后半区是同样五个缓冲区各 `+262144`，末端为 bias 槽的 454656。两者均低于 524288 字节的 L1，且各段互不交叠。

Profile 1（`kl1=256`、`scale_kl1=1280`，A 五面板全载、B 四槽；`bias_bytes = HasBias*1024`）：

| 缓冲区 | 声明布局与地址（带 bias） | 深度 | 单槽字节 |
| :--- | :--- | :--- | ---: |
| `a_l1` | Mat `[96,256]` FP8 @ `[142336, 166912, 191488, 216064, 240640]` | 5 | 24576 |
| `b_l1` | Mat `[256,256]` FP8 @ `[0, 269056, 65536, 334592]` | 4 | 65536 |
| `sa_l1` | Mat ZZ `[96,40]` E8M0 @ `265216` | 1 | 3840 |
| `sb_l1` | Mat NN `[40,256]` E8M0 @ `[131072, 400128]` | 2 | 10240 |
| `bias_l1` | Mat `[1,256]` FP32 @ `[141312, 410368]` | 2 | 1024 |

地址由容量算式给出：B 单槽 65536、B scale 单槽 10240、bias 单槽 1024、A 五面板合计 122880、A scale 3840。前半区先放 B 的两槽 `0..131071`，再放 B scale 首槽 @131072、bias 首槽 @141312，然后 A 五面板从 `2×65536 + 10240 + 1024 = 142336` 起每 24576 递进到 240640，A scale 单槽 @265216；后半区的 B 第二组槽起点取 `max(L1/2, A scale 末尾) = max(262144, 269056) = 269056`，**不把 B 槽错放回 262144**（那会与 A scale 交叠）。五个 A 面板在 K 上连续（NZ 物理布局 `[K/32, M/16, 16, 32]`），因此五段 `[96,256]` 面板的物理字节与整块 `[96,1280]` NZ A 一致；代码用 `depth=5` 的组按相同地址显式管理这五段。带 bias 时最高使用地址为 `bias_l1` 后半区槽末端 **411392**。

不带 bias 时 `bias_bytes=0`，其后所有地址前移 1024 字节：A 五面板起点变为 `[141312, 165888, 190464, 215040, 239616]`、`sa_l1=264192`、B 四槽 `[0, 268032, 65536, 333568]`、B scale `[131072, 399104]`，并删除全部 bias 声明，L1 逻辑用量 409344 字节。这是**按同一算式重算**的结果，不是沿用带 bias 的地址。

Profile 2（`kl1=256`、`scale_kl1=4096`，A、B 各四槽）：

| 缓冲区 | 声明布局与地址 | 深度 | 单槽字节 |
| :--- | :--- | :--- | ---: |
| `a_l1` | Mat `[96,256]` FP8 @ `[0, 262144, 24576, 286720]` | 4 | 24576 |
| `b_l1` | Mat `[256,256]` FP8 @ `[49152, 311296, 114688, 376832]` | 4 | 65536 |
| `sa_l1` | Mat ZZ `[96,128]` E8M0 @ `[180224, 442368]` | 2 | 12288 |
| `sb_l1` | Mat NN `[128,256]` E8M0 @ `[192512, 454656]` | 2 | 32768 |
| `bias_l1` | Mat `[1,256]` FP32 @ `[225280, 487424]` | 2 | 1024 |

前半区排布是 A 两槽 `0..24575`、`24576..49151`，紧接 B 两槽 `49152..114687`、`114688..180223`，然后 `sa_l1` @180224、`sb_l1` @192512、`bias_l1` @225280；后半区是同样五个缓冲区各 `+262144`。带 bias 时活跃分配合计 **452608 字节**、物理高水位 488448（由 `bias_l1` 后半区槽末端决定）；不带 bias 为 450560 / 487424 字节，均低于 524288，两者只差 `bias_l1` 两个槽的 2048 字节。**这张表与同级 `quant_bmm_mxfp8` 的 L1 布局完全相同**，也是 Profile 3 选用它的理由之一。

**L0A / L0B / L0C（四个 profile 相同）**：

| 缓冲区 | 声明布局与地址 | 深度 | 用途 |
| :--- | :--- | :--- | :--- |
| `a_l0` | Left NZ `[96,128]` FP8 @ `[0, 32768]` | 2 | A 操作数；槽间隔为 L0A 的一半 |
| `b_l0` | Right ZN `[128,256]` FP8 @ `[0, 32768]` | 2 | B 操作数 |
| `plane_a` | ScaleLeft ZZ `[96,4]` E8M0 @ `[0, 2048]` | 2 | A 的 E8M0 plane，地址恒为对应 data 地址 `>>4` |
| `plane_b` | ScaleRight NN `[4,256]` E8M0 @ `[0, 2048]` | 2 | B 的 E8M0 plane，同上 |
| `acc_group` | Acc NZ `[96,256]` FP32，`fractal=1024` @ `[0, 131072]` | 2 | 累加结果（`96·256·4·2 = 196608 ≤ 262144`） |
| `bias_bt` | Bias `[1,256]` FP32 @ `[0, 1024]` | 2 | 仅 `HasBias=1`；BiasTable |

MX 操作数要求相应 scale 与数据地址满足 `addr(scale) == addr(data) >> 4`；这里指 Left/Right 与 ScaleLeft/ScaleRight 的配套地址（FP8 的除数是 16）。`bias_bt` 位于独立的 Bias 平面，不与 L0A/L0B 争用地址。BiasTable 搬运放在 `l0_k` 循环之前；由于 L1→Bias 搬运既不触碰 L0A 也不触碰 L0B，它与数据提取之间没有顺序约束，唯一要求是它在消费 bias 的首条 MX MMAD 之前完成，故其位置不可观测。

**UB（仅 Profile 0 的 Vector 段）**：

| 缓冲区 | 声明布局与地址 | 深度 | 字节 | 用途 |
| :--- | :--- | :--- | ---: | :--- |
| `src_group` | Vec `[4,3072]` FP32 @ `0` | 1 | 49152 | 四份部分和；行 0 同时是运行和，行 1/2/3 为待加项 |
| `dst` | `reinterpret(src_group.current(), BF16, [12,256])` @ `0` | — | 6144（落在上项内） | 同址 BF16 目标，不额外声明 |

每个 AIV 处理 12 行 × 256 列 = 3072 个元素，恰为 UB tile 的一行。Profile 1/2/3 没有 Vector 段，UB 占用为 0。

**GM 侧**：输出 `[96,N]` BF16 由 host `torch.empty` 分配、不做设备侧预置。Profile 0 另有 `[7,4,96,256]` FP32 workspace 共 2752512 字节——只分配实际消费的部分，不为未访问的预留区增加填充。该 workspace 同样由 `torch.empty` 分配、无需预置，因此不产生额外设备任务。bias 由首条 MX MMAD 直接消费，host 不分配零缓冲、不产生附带的清零 task，因此四个 profile 都是**每次公开调用恰好一个设备任务**。仅 Profile 0 需要 keepalive：输出张量通过 `out._mxfp8_keepalive` 持有 workspace 引用，其余路径不分配内部缓冲因而不挂该属性。

## 接口选择

- **显式 Cube 操作**：使用 `pl.load`、`pl.move`、`pl.matmul_mx` / `pl.matmul_mx_acc`、`pl.store`，以便显式控制 L1/L0 地址、scale 配对、轮转深度与 `AccPhase`/`STPhase`。
- **bias 直接挂到首条 MX MMAD**：`pl.matmul_mx(dst, lhs, rhs, scale_a, scale_b, bias_tile=None, *, phase)` 接受可选的 FP32 BiasTable 操作数。IR 层 `_check_mx_operands` 要求它位于 `MemorySpace.Bias`、dtype 为 FP32、shape `[1, N]` 行广播且列数等于 rhs 的 N——本实现的 `bias_bt`（`Bias [1,256]` FP32）逐项满足。`matmul_mx_acc` 无 bias 形参，因此 bias 只由首条 MMAD 以 `init_with_zero=false` 消费、其后各步是纯累加；split-K 下这条首指令只出现在 split 0。这样带 bias 的调用不需要任何内部 GM 缓冲，也不产生额外的设备任务。
- **`set_mm_layout_transform`**：整个 cube section 包在 `pl.system.set_mm_layout_transform(enabled=True)` 内、末次 store 之后恢复 `False`，使 fixpipe 沿 N 方向列优先排出 L0C。
- **`pl.set_validshape` 表达紧凑尾块**：Profile 0 的末段 `data_len=256`、末窗 `scale_len=256` 与 WeightNZ 的 `data_len*4`，以及 Profile 3 的全部 M/N 尾块，都靠它收紧描述符，从而复用同一组 L1 槽而不必为尾块单独声明缓冲。
- **跨核交接用公开 system 原语**：`pl.system.set_cross_core` / `wait_cross_core` 配合 `CrossCoreSyncMode.INTER_BLOCK` 与 `PipeType.FIX` / `MTE2`。这是一次性交接，不是环形流水；event 0 用于全 AIC 完成屏障，event 1 用于通知 AIV。
- **归并用 VF 而非 tile Vector API**：`vf.load_align` / `vf.add` / `vf.astype(round_mode=CAST_RINT)` / `vf.mem_bar(VLD_VST)` / `vf.store_align(dist=PACK_B32)`。这是 Pro 的原生表达方式，同级 `quant_bmm_mxfp8` 也使用同一套 VF 原语。
- **FP8 dtype 解释**：外部 x1/x2 为 FP8 张量，kernel 内以 `DT_FP8E5M2` 或 `DT_FP8E4M3FN` 使用其存储位，由 `matmul_mx` 联合 E8M0 完成分组反量化，不先展开成 FP32 张量。
- **x2 侧 L2 cache hint**：两处 x2 载入（ND 与 WeightNZ）都传 `pl.load(..., l2_cache_mode=pl.TLoadL2Hint.NotAllocKeep)`，即权重流不在 L2 分配。理由是 x2 是逻辑流量的主体且每个输出块只流过一次，而 L2 只有 128 MiB，让它参与分配会挤掉真正被跨块复用的张量（Profile 1 的 A 面板正是要跨 4 个 N 块复用的）。`pl.TLoadL2Hint` 的取值是 `NormalFirstVictim=0, NormalLastVictim=1, NormalPersistent=2, NotAllocKeep=3, NotAllocClean=4, NotAllocDrop=5`；`pl.TStoreL2Hint` 少两项 `NotAlloc*`，为 `NormalFirstVictim=0, NormalLastVictim=1, NormalPersistent=2, NotAllocClean=3`。本算子只 hint x2 的载入，不使用 `TStoreL2Hint`。**只有 x2 被设 hint**：x1、两份 scale 与 bias 都小且被复用，输出存储保持默认策略。`l2_cache_mode` 是编译期 lowering 属性（落成 `TLOAD<TLoadL2Hint::NotAllocKeep>` 模板实参），因此一个值适用于全部块；它只影响 L2 的分配与淘汰、不影响数值。选择把它冻结成模块常量 `B_LOAD_L2_HINT` 而不是运行期判定，是因为 Profile 0/1/2 三组 shape 的 `N/BN` 分别为 7 / 128 / 20 均整除、无 N 尾块，对齐条件恒为真。生成代码核实：每个 tiling-key 变体恰好 1 处 `TLoadL2Hint::NotAllocKeep`，无其他 `TLoadL2Hint`、无 `TStoreL2Hint`。
- **ICache 预取的执行范围与位置**：顺序是 `wait_cross_core(MTE2, event 1)` → `vector_idx < 56` → 绑定 GM → `icache_preload(2)` → 归并工作。即**只有 Profile 0 的 56 个有效 AIV 预取**，8 个无效 AIV 只等待、不预取；Profile 1/2/3 以及同级 `quant_bmm_mxfp8` 都不预取（它们没有 Vector 段）。预取长度 2 由 `ICACHE_PREFETCH_LEN` 给出。
- **同步职责**：带 `mutex_ids="auto"` 的 TileGroup 由 `auto_mutex=True` 管理核内数据依赖，不使用手工事件编号；跨核依赖只由上述显式 cross-core 事件表达。代价是**不等价于逐条指令一致**，无法逐条比对指令序列。

`MxGemmKey` 有 `Profile`（2 bit）、`InputFormat`、`HasBias`、`WeightNz` 四个字段。`Profile` 之所以必须进 key，是因为它改变 L1 布局、驻留策略、启动几何与 split-K 算术，复用错误的编译变体会直接算错；`HasBias` 进 key 是为了防止带/不带 bias 两类初始化方式复用同一变体。具体 shape 数值不进 key。

**Profile 3 的接口选择，以及单一 kernel 的统一签名**：

- **共用一个 tiling key**：四种调度同属 `MxGemmKey`，`Profile` 取 0/1/2/3（2 bit 恰好容纳）。因为 `Profile` 是编译期常量，每个变体只 trace 自己那一条分支，所以合并成一个 kernel **不会**让任何变体多编译代码，也不会让变体数量增加——Profile 0/1/2 的 18 个变体 tiling key 编码不变（`WeightNz<<4 | HasBias<<3 | InputFormat<<2 | Profile`），缓存目录名也不变。
- **统一签名是两条分支参数的并集**：`x1_ptr, x2_ptr, x2_nz, sa_ptr, sb_ptr, bias_ptr, workspace_ptr, out_ptr, m_dim, n_dim, k_dim` 共 11 个。Profile 0/1/2 只读前 8 个（`workspace_ptr` 仅 Profile 0 真正使用），Profile 3 只读除 `workspace_ptr` 外的 10 个。host 对不需要的指针传一个存活的输入指针作 ABI 占位（与 no-bias 场景同一手法），不额外分配。代价是 Profile 0/1/2 的生成码入口签名多 3 个从不读取的 `int64_t`；实测其设备侧函数体与合并前**逐字节相同**，18 个性能用例的净耗时差中位 +0.32%（小于 4.1% 的跨采集噪声）。
- **shape 以运行时标量传入**：`m_dim / n_dim / k_dim` 是 `pl.DT_INT64` 形参，张量沿用 Profile 0/1/2 的 `pl.Ptr[pl.DT_UINT8]` + `pl.make_tensor` 写法（这样原生 E8M0 与 uint8+override 两种 scale 载体都仍能工作），只有 `x2_nz` 保持 `pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], ..., pl.NZ]`。因为 shape 不进 key，Profile 3 对同一 key **只编译一次**即可服务所有 shape。
- **动态 shape 接口**：张量形参用 `pl.Tensor[[pl.DYNAMIC, ...], dtype]` 声明；尾块用 `pl.set_validshape` 收窄 tile，窗口长度用 `pl.min(窗口, 剩余)` 计算，多轮遍历用 `pl.range(core_id, total, num_cores)` 配合 `pl.get_block_num()`。**Cube 侧对部分 M / 部分 N 的支持在框架官方用例中没有先例**，故用独立探针在板上验证：`96x100`、`50x256`、`50x100`、`16x256`、`96x16` 五种部分块组合都与 CPU FP64 golden 相符，最小可到单个 16×16 fractal，因此**不需要 host 侧 padding**。
- **单步 K 的相位判定**：`AccPhase` 的选择必须把「首步」与「末步」**分层嵌套**，不能写成 `if 首步 / elif 末步` 的互斥链。`K == G_BK` 时两者重合，互斥写法会只发 `Partial` 而永不发 `Final`，累加器不被终结，随后 `STPhase.Final` 的写出就没有可等待的完成事件，板上表现为 `The aicore execution times out`。Profile 3 分支采用嵌套写法。Profile 0/1/2 分支保持互斥写法，因为三组 shape 的 K 分别是 40、10、64 个 `BK` 步，不会触发这一边界。

内部 JIT 增加 `x2_nz` Tensor 参数：WeightNZ 场景传原始 E4M3FN NZ tensor，ND 场景传 `None`，对应分支由 `WeightNz` 编译期常量删除。公开接口、测试调用方式与 tiling key 不变。Pro 的 nullable Tensor 参数绑定会为 `None` 传空指针，动态维度槽补零，不读取该参数。

## 流水设计

Profile 1/2/3 是单段 Cube 流水：

```text
AIC MTE2：GM → L1（scale 窗口；A、B 按 kl1 轮转，四槽或双槽）
    ↓ L1 data 槽
AIC MTE1：L1 → L0A/L0B 及其 E8M0 plane（双槽轮转）
    ↓ L0 双槽
AIC Cube：matmul_mx / matmul_mx_acc（K 升序）
    ↓ FP32 Acc 双槽
AIC FIXPIPE：STPhase.Final → GM 输出 BF16
```

Profile 0 在 Cube 段之后追加一次跨核交接与 Vector 归并：

```text
28 个有效 AIC：各自 1280 长的 K 分片 → FP32 部分和 → FIXPIPE 写 GM workspace[n_idx, split]
    ↓ 全部 32 个 AIC 参与 FIX 完成屏障（event 0，INTER_BLOCK）→ set event 1
56 个 AIV：wait event 1（MTE2）→ ICache 预取 → 载入 [4,3072] FP32 到 UB
    ↓ 三次整段原地 Add（auto_mutex 管跨调用依赖）
同址 RINT Cast 为 BF16（VLD_VST 屏障 + PACK_B32 压实）→ 写 out[row, n_tile*256]
```

- **载入与计算的重叠**：Profile 1/2/3 的 L1 四槽允许后续 `data_k` 面板的 GM→L1 搬运与当前面板的 L0 提取及乘加交叠；L0 双槽允许下一次提取与当前乘加交叠。Profile 0 只有双槽，重叠窗口更窄。
- **Profile 1 的驻留复用**：A 的五个面板与整份 A scale 只在 `output_iter==0` 载入，之后三个输出块直接复用，因此 A 的 GM 读取量降到 1/4；B data、B scale、bias 与 L0C 的游标跨 N 块连续推进，不随 `output_iter` 复位。
- **Profile 2 的 scale 窗口**：`scale_kl1=4096` 把 `K=8192` 分成两个窗口，每窗口内 data 仍按 256 载入 16 次；A、B scale 每窗口各刷新一次。
- **Profile 0 的屏障位置**：屏障必须在**所有** AIC 完成 FIX 之后、任何 AIV 读取之前。代码把 `set_cross_core` 与 `wait_cross_core` 都放在 `if worker < active` 之外，使 4 个空闲块也进入，这是正确性要求而非冗余。
- **归并不是流水的一部分**：它是一次性的收尾阶段。56 个 AIV 各自独立处理一个 `[12,256]` 子块，彼此无依赖；UB 内的三次 Add 之间才有依赖，交给 `auto_mutex`。

**已知取舍**

- **tiling 是编译期常量而非运行期复算**：host 不做 tiling 计算、不查询平台、不读 SDK ini，几何与片上地址都写死为常量与立即数。好处是没有 host tiling 开销、行为可复现；代价是这些常量绑定目标 SKU，换 SKU 或受限核数配置需要按「基本块选择」里的容量算式重新核算。
- **Profile 3 不做 split-K**：每个输出块由一个 worker 沿完整 K 端到端归约。当 N 相对 K 很小、输出块数少于核数时，空闲核无法参与该块的归约；这是为「一套 tiling 覆盖任意 shape」接受的取舍。
- **Profile 3 的 L2 hint 不区分尾块**：`l2_cache_mode` 是编译期属性，无法按块取不同值。N 尾块的 `baseN` 可能不满足 128 元素对齐（FP8 的对齐掩码是 `0x7f`），此时更合适的取值是普通分配类（`NormalFirstVictim` / `NormalLastVictim` / `NormalPersistent`），但 Profile 3 对**全部块一律传 `NotAllocKeep`**。cache hint 只影响 L2 的分配与淘汰、不影响数值，因此这是**性能层面的取舍而非正确性问题**；其代价未量化，需对尾块补采 L2 命中率计数器才能评估。Profile 0/1/2 不受此影响：三组 shape 的 `N/BN` 均整除、无尾块，故 `B_LOAD_L2_HINT` 可以冻结为常量。
- **不使用手工事件编号**：核内同步交给 `auto_mutex=True`，依赖关系由框架按 TileGroup 的 mutex 所有权推导。Profile 0 的跨核交接是唯一例外，用显式 cross-core 事件表达。
- **split-K 的归并顺序是固定的左结合**：`((p0 + p1) + p2) + p3`，不做树形求和、不后加 bias、不转换中间部分和为 BF16。这决定了 Profile 0 的 FP32 累加分组，也是它与 Profile 3 在同一 shape 上可能出现末位 BF16 差异的原因。

支持域尚未覆盖：`K % 128 != 0`（需 K 方向尾块掩码或补零）、ScaleNZ、`x1` 的 NZ、非 NN 逻辑 stride 与 rank≠2 输入。**性能实测只覆盖 18 个 shape**（`RESULTS.md` 列出的三组 `M/N/K` × dtype × bias × 权重布局，是支持域的子集）；支持域内其余 shape 的性能**尚未按测量协议采集**，故本文不对它们做任何快慢声明。此外 Profile 3 下尾块工作量更小、且输出块数不是核数整数倍时末轮有 worker 空转，负载均衡不由构造保证，这属于为支持任意 shape 接受的代价。
