# QuantBatchMatmul MXFP8 设计说明（PyPTO-Pro）

对应实现：[quant_batch_matmul_mxfp8.py](op_kernel/quant_batch_matmul_mxfp8.py)。模块有一个公开 host 函数和**一个** `@pl.jit(arch="a5", auto_mutex=True, tiling_key=MxKey)` 入口 `_quant_batch_matmul_mxfp8_kernel`，它服务整个支持域。host 校验属性、推导 shape、分配输出并确定启动块数；设备侧完成 GM→L1 搬运、L1→L0 提取、MX 乘加与 L0C→GM 写出。它是**纯 Cube kernel**：只有 `pl.section_cube()`，没有 vector section，因此不占用 UB。接口见 [README.md](README.md)，测试及实测记录见 [RESULTS.md](RESULTS.md)。

**tile 几何与整张 L1/L0 地址表都是编译期常量**（`BM/BN/BK`、`KL1/SCALE_KL1` 与立即数偏移）；只有「把这些固定 tile 铺到 shape 上」的遍历是运行期的。shape 在 kernel 内由 `out.shape[i]` 与 `x1_scale.shape[2] * 64` 读出，是运行时值而非 tiling key 字段，因此**同一 key 只编译一次即可服务所有 shape**，变体数量不随 shape 增长。

| 需要泛化的维度 | 处理方式 |
| :--- | :--- |
| M/N 尾块 | `pl.set_validshape` 就地收窄 L1、L0、scale plane、bias 与 Acc 的描述符；Cube 接受部分 M/N，最小到单个 16×16 fractal，故不需要 host 侧 padding |
| K 尾窗 | `scale_len / data_len = pl.min(窗口, 剩余)` 收缩，同样用 `set_validshape` 收窄 |
| 输出块数 ≠ 核数 | grid-stride 循环 `pl.range(core_id, total, num_cores)` 分多轮，不要求恰好满波 |
| 启动块数 | host 取 `min(总块数, 32)`，块数少于核数时不空转 |

性能实测所用的 `(8,96,1024,4096)` 是这套动态遍历的一个**特例**：总块数 `8 × 1 × 4 = 32`，启动 32 块、grid-stride 只跑一轮、每核恰好一个 `[96,256]` 输出块；`vm=96`、`vn=256` 均等于满 tile，`scale_len=4096`、`data_len=256` 均等于满窗口，即所有 `pl.min` 都取到上界、所有 `set_validshape` 都写成满形状。动态遍历在此退化为单轮满波，且 kernel 本身是访存受限（`mte2` 占比 0.94、达 roofline 上限约 90%），因此循环边界由编译期常量改为运行期求值所带来的标量开销被搬运掩盖，不落在关键路径上。

本文中，worker 指一个 AIC；tile 指一个 worker 独占的 `[96,256]` 输出块（尾块处按 `set_validshape` 收窄）。GM 是设备全局内存。L1、L0A、L0B 和 L0C 用于 Cube 数据及累加结果。MTE2 负责从 GM 读取数据到 L1，MTE1 负责 L1→L0 提取，FIXPIPE 将 Cube 累加结果写回 GM。

## 计算阶段划分

M 按 `BM=96` 分块、N 按 `BN=256` 分块，K 是归约维：先按 `SCALE_KL1=4096` 分窗，窗内按 `KL1=256` 载入，每次载入再按 `BK=128` 分 2 次乘加。一个 worker 独占一个输出块并沿完整 K 归约，全程不与其他 worker 交换数据。以性能实测的 `(8,96,1024,4096)` 为例：M 恰好等于 `BM` 故不分块、N 分为 4 块、K 分 16 次载入、共 32 次 MX，且 `SCALE_KL1` 覆盖全 K 故 scale 窗口只有一轮。

三层循环的边界都是运行时值，外面再套一层输出块遍历：

```text
m_tiles = ceil(M/96)   n_tiles = ceil(N/256)   total = BATCH * m_tiles * n_tiles
for tile_id in pl.range(core_id, total, num_cores):        # grid-stride，可多轮
    batch_idx, m_offset, n_offset = 解码(tile_id)
    vm = min(96, M - m_offset)     vn = min(256, N - n_offset)      # M/N 尾块
    for scale_k in pl.range(0, K, 4096):                            # scale 窗口
        scale_len = min(4096, K - scale_k)                          # 末窗收缩
        for data_k in pl.range(scale_k, scale_k + scale_len, 256):   # data_k 为绝对偏移
            data_len = min(256, K - data_k)                          # K 尾窗
            for l0_k in pl.range(0, data_len, 128):
                s_off = (data_k - scale_k + l0_k) // 32               # 窗口内 E8M0 列
```

尾块一律用 `pl.set_validshape` 收窄对应 tile：L1 的 `a_l1[vm,data_len]`、`b_l1[data_len,vn]`、`sa_l1[vm,scale_len/32]`、`sb_l1[scale_len/32,vn]`，L0 的 `a_l0[vm,128]`、`b_l0[128,vn]` 与两个 scale plane，以及 `acc[vm,vn]` 与 `bias[1,vn]`。Cube 接受部分 M 与部分 N，最小可到单个 16×16 fractal，因此**不需要 host 侧 padding**。

下表给出各阶段的粒度，括号内是性能实测 shape `(8,96,1024,4096)` 下的具体次数。阶段顺序与依赖不随 shape 改变。

| 阶段 | 执行部件 | 粒度 | 实现逻辑 |
| :--- | :--- | :--- | :--- |
| A BiasTable 就位 | AIC | 每输出块一次，仅 `HasBias=1` | `pl.load(bias_mat, bias, [batch_idx*bias_per_batch, n_offset])` 把 FP32 bias 从 GM 载入 L1，再 `pl.move(bias_table, bias_mat)` 送入 `MemorySpace.Bias`。本阶段不产生任何乘加 |
| B scale 窗口载入 | AIC MTE2 | 每输出块每窗口一次 | A scale 以 ZZ、B scale 以 NN 布局各载入一份窗口，窗口数为 `ceil(K/SCALE_KL1)`（K=4096 时为 1） |
| C 数据载入 | AIC MTE2 | 每 `KL1=256` 一次 | 载入顺序为 scale、A、bias（仅首个 `data_k`）、B。WeightNZ 时 B 走原生 GM NZ→L1 NZ 搬运。**B 的两条载入路径都带 `l2_cache_mode=TLoadL2Hint.NotAllocKeep`**；A、两份 scale、bias 与输出存储都不带 hint |
| D L1→L0 提取 | AIC MTE1 | 每 `BK=128` 一次 | A、B 数据与其 E8M0 plane 成对提取；plane 的组偏移是 `(data_k - scale_k + l0_k)//32`，即当前 scale 窗口内的第几个 32 元素组 |
| E MX 乘加 | AIC Cube | 每 `BK` 一次，K 升序 | 首次为 `pl.matmul_mx`：带 bias 时该指令同时消费 FP32 BiasTable（`init_with_zero=false`），不带 bias 时省略该操作数。其后各次为 `pl.matmul_mx_acc`（无 bias 形参）。末次用 `AccPhase.Final`，其余 `Partial` |
| F 结果写出 | AIC FIXPIPE | 每输出块一次 | `pl.store(out, acc, [batch_idx, m_offset, n_offset], phase=STPhase.Final)`，L0C 直接写 GM 的 BF16 |

**BiasTable 搬运的位置**：它放在首个 `l0_k` 迭代内、A/scaleA 提取之后、B/scaleB 提取之前。L1→Bias 搬运既不触碰 L0A 也不触碰 L0B，与数据提取之间没有顺序约束，唯一要求是它在消费 bias 的首条 MX MMAD 之前完成。GM→L1 的载入顺序则固定为 scale、A、bias（仅首个 `data_k`）、B。

- **原生 WeightNZ 搬运**：WeightNZ 使用原生 `pl.Tensor[..., pl.DT_FP8E4M3FN, pl.NZ]` 和 `pl.load`，以逻辑 K/N 偏移直接完成 GM NZ→L1 NZ 搬运；两份 scale 保持 ND。不做数值转换、不分配中间权重。

**K 尾不补零**：`K % 128 == 0` 已隐含 `K % 64 == 0` 与 `K % 32 == 0`，因此打包 scale 的 64 元素窗口整除、MX 组宽 32 也对齐，kernel 不需要 K 方向补零。`K` 不是 `KL1` 或 `SCALE_KL1` 整数倍时，末窗由 `pl.min` 收缩并用 `set_validshape` 收窄，同样不需要补零。

## 核间切分策略

- **启动形态**：`_quant_batch_matmul_mxfp8_kernel[None, max(1, min(total, 32)), key]`，其中 `total = BATCH × ceil(M/96) × ceil(N/256)`。块数不超过输出块总数以免空转；`total > 32` 时由 kernel 内的 grid-stride 循环分多轮完成。kernel 内不启动其他 kernel。
- **工作分配**：`for tile_id in pl.range(core_id, total, num_cores)`，`num_cores` 由 `pl.get_block_num()` 运行期取得，块号按 `batch → m_tile → n_tile` 的顺序解码，即 `batch_idx = tile_id // (m_tiles*n_tiles)`、`m_offset = (rem // n_tiles)*96`、`n_offset = (rem % n_tiles)*256`。块数多于核数时自动分多轮，不要求恰好满波。性能实测 shape 下 `total = num_cores = 32`，故每核一轮，解码退化为 `batch_idx = tile_id//4`、`n_offset = (tile_id%4)*256`、`m_offset = 0`。
- **无跨核同步**：每个 worker 独占一个输出块并独立沿完整 K 归约，不产生部分和，因此不需要 workspace、跨核事件或全局屏障。grid-stride 只增加每个 worker 的轮数，不引入 worker 间依赖。
- **负载均衡**：性能实测 shape 下 32 个 worker 各承担一个形状相同的 `[96,256]×4096` 输出块，工作量按构造完全一致。一般情况下**均衡不由构造保证**：尾块的 `vm`/`vn` 更小、工作量更少，且 `total` 不是核数整数倍时最后一轮会有 worker 空转。这是为支持任意 shape 接受的代价。
- **核内同步**：全部由 `auto_mutex=True` 按 TileGroup 的 mutex 所有权推导，不使用手工事件编号。

## 核内切分策略：基本块选择、核内 Buffer 划分

### 基本块选择

| 维度或参数 | 取值 | 依据 |
| :--- | :--- | :--- |
| `BM` | 96 | 输出块的 M 高度。等于所测 shape 的 M，故该维不分块；M 更大时按 `ceil(M/96)` 分块并可能有尾块 |
| `BN` | 256 | 输出块的 N 宽度。与 `BM` 一起决定 L0C 的 `[96,256]` FP32 累加 tile（98304 字节，双槽 196608 ≤ L0C 的 262144） |
| `BK` | 128 | L0 的 K 步长，也是 K 的最小支持粒度。L0A/L0B 每槽 32768 字节，`[96,128]` 与 `[128,256]` 的 FP8 NZ 面板各占一半 |
| `KL1` | 256 | GM→L1 的数据暂存粒度。一个 L1 数据面板是 `[96,256]`（A，24576 字节）与 `[256,256]`（B，65536 字节），各喂 2 次 `BK=128` 的 MX 乘加 |
| `SCALE_KL1` | 4096 | E8M0 scale 窗口宽度。取 4096 使一个窗口覆盖所测 shape 的整个 K，因此每个输出块只载入一次 scale；A 窗口是 `[96,128]`（12288 字节），B 窗口是 `[128,256]`（32768 字节） |
| L1 data 槽数 | 4 | A、B 各四槽轮转，槽 0/2 在 L1 前半区、1/3 在后半区，使搬运与提取/乘加可以交叠 |
| L1 scale 槽数 | 2 | 双槽；当窗口覆盖全 K 时每个输出块只载入一次，实际恒用首槽 |
| L0 / Acc 槽数 | 2 / 2 | 双槽使下一次提取与当前乘加交叠。输出块数不超过核数时每核只有一个输出块，故 Acc 实际恒取首槽（性能实测 shape 即如此） |
| MX 次数 | `K/BK` | 性能实测 shape 为 `16 (data_k) × 2 (l0_k) = 32`，K 升序 |

A、B 各四槽的 L1 数据 + 双槽 scale + 双槽 bias 的活跃分配合计 **452608 字节**（带 bias），低于 524288 字节的 L1；这是 `KL1=256` 配四槽而不配更大面板或更多槽的容量约束。若把 `KL1` 提到 512，A/B 单槽翻倍为 49152/131072 字节，四槽合计 720896 字节，已超出 L1。

### 核内 Buffer 划分

本 kernel 不使用 UB（无 vector section），因此没有 UB 预算表。L1 按两个半区（相隔 262144 字节）布置，每半区放两个数据槽与一份 scale/bias 槽：

| 缓冲区 | 声明布局与地址 | 深度 | 单槽字节 | 用途 |
| :--- | :--- | :--- | ---: | :--- |
| `a_l1` | Mat `[96,256]` FP8 @ `[0, 262144, 24576, 286720]` | 4 | 24576 | x1 数据轮转槽 |
| `b_l1` | Mat `[256,256]` FP8 @ `[49152, 311296, 114688, 376832]` | 4 | 65536 | x2 数据轮转槽；WeightNZ 时原生 GM NZ→L1 NZ |
| `sa_l1` | Mat ZZ `[96,128]` E8M0 @ `[180224, 442368]` | 2 | 12288 | x1 的 scale 窗口 |
| `sb_l1` | Mat NN `[128,256]` E8M0 @ `[192512, 454656]` | 2 | 32768 | x2 的 scale 窗口 |
| `bias_l1` | Mat `[1,256]` FP32 @ `[225280, 487424]` | 2 | 1024 | 仅 `HasBias=1`；首个 `data_k` 载入一次后整个输出块复用 |

前半区的排布是 A 两槽 `0..24575`、`24576..49151`，紧接 B 两槽 `49152..114687`、`114688..180223`，然后 `sa_l1` @180224、`sb_l1` @192512、`bias_l1` @225280；后半区是同样五个缓冲区各 `+262144`。L1 活跃分配合计：带 bias **452608 字节**（物理高水位 488448，由 `bias_l1` 后半区槽末端决定），不带 bias **450560 字节**（高水位 487424，由 `sb_l1` 后半区槽末端决定），两者都低于 524288 字节的 L1，且只差 `bias_l1` 两个槽的 2048 字节。

L0A / L0B / L0C 按每个 worker 分配：

| 缓冲区 | 声明布局与地址 | 深度 | 用途 |
| :--- | :--- | :--- | :--- |
| `a_l0` | Left NZ `[96,128]` FP8 @ `[0, 32768]` | 2 | A 操作数；槽间隔为 L0A 的一半 |
| `b_l0` | Right ZN `[128,256]` FP8 @ `[0, 32768]` | 2 | B 操作数 |
| `sa_l0` | ScaleLeft ZZ `[96,4]` @ `[0, 2048]` | 2 | A 的 E8M0 plane，地址恒为对应 data 地址 `>>4` |
| `sb_l0` | ScaleRight NN `[4,256]` @ `[0, 2048]` | 2 | B 的 E8M0 plane，同上 |
| `acc` | Acc NZ `[96,256]` FP32，`fractal=1024` @ `[0, 131072]` | 2 | 累加结果；每核一个输出块时（如性能实测 shape）实际恒用首槽 |
| `bias_bt` | Bias `[1,256]` FP32 @ `[0, 1024]` | 2 | 仅 `HasBias=1`；BiasTable |

MX 操作数要求相应 scale 与数据地址满足 `addr(scale) == addr(data) >> 4`；这里指 Left/Right 与 ScaleLeft/ScaleRight 的配套地址（FP8 的除数是 16）。`bias_bt` 位于独立的 Bias 平面，不与 L0A/L0B 争用地址。

**GM 侧**：输出 `[B,M,N]` BF16，由 host `torch.empty` 分配、不做设备侧预置，是本算子**唯一**的设备分配。bias 由首条 MX MMAD 直接消费，host 不分配任何内部缓冲，因此返回张量上不挂内部缓冲引用，一次公开调用恰好对应一个设备任务。

## 接口选择

- **显式 Cube 操作**：使用 `pl.load`、`pl.move`、`pl.matmul_mx` / `pl.matmul_mx_acc`、`pl.store`，以便显式控制 L1/L0 地址、scale 配对、轮转深度与 `AccPhase`/`STPhase`。
- **bias 直接挂到首条 MX MMAD**：`pl.matmul_mx(dst, lhs, rhs, scale_a, scale_b, bias_tile=None, *, phase)` 接受可选的 FP32 BiasTable 操作数。IR 层 `_check_mx_operands` 要求它位于 `MemorySpace.Bias`、dtype 为 FP32、shape `[1, N]` 行广播且列数等于 rhs 的 N——本实现的 `bias_bt`（`Bias [1,256]` FP32）逐项满足。`matmul_mx_acc` 无 bias 形参，因此 bias 只由首条 MMAD 以 `init_with_zero=false` 消费，其后各步是纯累加。这样带 bias 的调用不需要任何内部 GM 缓冲，也不产生额外的设备任务。
- **`set_mm_layout_transform`**：整个 cube section 包在 `pl.system.set_mm_layout_transform(enabled=True)` 内、末次 store 之后恢复 `False`，使 fixpipe 沿 N 方向列优先排出 L0C。
- **原生 WeightNZ 搬运**：WeightNZ 使用原生 `pl.Tensor[..., pl.DT_FP8E4M3FN, pl.NZ]` 和 `pl.load`，以逻辑 K/N 偏移直接完成 GM NZ→L1 NZ 搬运；两份 scale 保持 ND。不做数值转换、不分配中间权重。
- **scale 的 phase 轴**：host 把 scale 视作 `[..., 2]`，最后的长度 2 维是物理 phase 轴；GM→L1 载入时生成 A 使用的 ZZ 与 B 使用的 NN 布局。`view` / `reinterpret` 只改变视图，不做数值解码。
- **FP8 dtype 解释**：外部 x1/x2 为 FP8 张量，kernel 内以 `DT_FP8E5M2` 或 `DT_FP8E4M3FN` 使用其存储位，由 `matmul_mx` 联合 E8M0 完成分组反量化，不先展开成 FP32 张量。
- **x2 侧 L2 cache hint**：两处 x2 载入（ND 与 WeightNZ）都传 `pl.load(..., l2_cache_mode=pl.TLoadL2Hint.NotAllocKeep)`，即权重流不在 L2 分配。理由是 x2 占逻辑流量的绝大部分且每个输出块只流过一次：在 `B/M/N/K = 8/96/1024/4096` 下它是 39.45 MB 中的 33.55 MB，而 L2 只有 128 MiB，让它参与分配会挤掉真正被跨块复用的张量。**只有 x2 被设 hint**：x1、两份 scale 与 bias 都小且被复用，输出存储保持默认策略。`l2_cache_mode` 是编译期 lowering 属性（落成 `TLOAD<TLoadL2Hint::NotAllocKeep>` 模板实参），因此一个值适用于全部块，N 尾块也不例外；它只影响 L2 的分配与淘汰、不影响数值。
- **同步职责**：带 `mutex_ids="auto"` 的 TileGroup 由 `auto_mutex=True` 管理数据依赖；无跨核依赖，故不使用 `set_cross_core` / `sync_all`。

`MxKey` 只有 `InputFormat`、`HasBias`、`WeightNz` 三个字段，分别选择 FP8 类型、首条 MX MMAD 是否携带 BiasTable 操作数、x2 走 ND 还是 WeightNZ 搬运。batch、M、N、K **不进入 key**，也不因具体数值建立新的 tiling key：它们在 kernel 内是运行时值。因此对同一 key **只编译一次**即可服务所有 shape，避免变体数量随 shape 线性增长。

- **动态 shape 接口**：张量形参用 `pl.Tensor[[pl.DYNAMIC, ...], dtype]` 声明，shape 在 kernel 内由 `out.shape[i]` 与 `x1_scale.shape[2] * 64` 读出（`x2_nz` 在 ND 场景为 `None`，故不从它取维）。尾块用 `pl.set_validshape` 收窄 tile，窗口长度用 `pl.min(窗口, 剩余)` 计算，多轮遍历用 `pl.range(core_id, total_tiles, num_cores)` 配合 `pl.get_block_num()`。
- **单步 K 的相位判定**：`AccPhase` 的选择把「首步」与「末步」**分层嵌套**判定，而不是写成 `if 首步 / elif 末步` 的互斥链。`K == BK` 时两者重合，该步必须直接携带 `AccPhase.Final`；若只发 `Partial`，累加器不会被终结，随后的 `STPhase.Final` 写出就没有可等待的完成事件。性能实测 shape 的 K 有 32 个 `BK` 步，不会触发这一边界；触发它的是 `K == 128` 的用例。

## 流水设计

四槽 L1 数据与双槽 L0 把搬运、提取和乘加连接起来（次数按性能实测 shape `(8,96,1024,4096)` 给出）：

```text
AIC MTE2：GM → L1（scale 窗口一次；A、B 各 16 次，四槽轮转）
    ↓ L1 data 四槽
AIC MTE1：L1 → L0A/L0B 及其 E8M0 plane（32 次，双槽轮转）
    ↓ L0 双槽
AIC Cube：matmul_mx / matmul_mx_acc（32 次，K 升序）
    ↓ FP32 Acc 双槽（每核一个输出块，实际用首槽）
AIC FIXPIPE：STPhase.Final → GM 输出 BF16
```

- **载入与计算的重叠**：L1 数据四槽允许后续 `data_k` 面板的 GM→L1 搬运与当前面板的 L0 提取及乘加交叠；L0 双槽允许下一次提取与当前乘加交叠。
- **scale 只载一次**：因 `SCALE_KL1=4096` 覆盖全 K，`scale_k` 循环只有一轮，每个输出块只载入一次 scale 窗口，消除了每 `KL1` 重载 scale 的流量。
- **bias 只在首个 `data_k` 载入**：条件为 `HasBias == 1 and scale_k == 0 and data_k == 0`，随后整个输出块复用同一份 L1 bias。
- **无跨阶段屏障**：一个 worker 从头到尾只处理自己的输出块，流水内部没有全局同步点，也没有归并阶段。
- **bias 与首条 MX 的关系**：在首轮 `l0_k == 0` 中，依次提取 A、scaleA，搬入 BiasTable，再提取 B、scaleB，最后由首条 MX 消费 bias；之后各步纯累加。

**已知取舍**

- **tiling 是编译期常量而非运行期复算**：host 不做 tiling 计算、不查询平台、不读 SDK ini，几何与片上地址都写死为常量与立即数。好处是没有 host tiling 开销、行为可复现；代价是这些常量绑定目标 SKU（32 Cube / 64 AIV、L1 512 KiB、L0 每槽 32768 字节、L2 128 MiB），换 SKU 或受限核数配置需要重新核算。
- **L2 hint 不区分尾块**：`l2_cache_mode` 是编译期属性，无法按块取不同值，因此 N 尾块与满块使用同一个 `NotAllocKeep`。这只影响缓存行为、不影响数值；其对尾块的性能影响未量化，需要 L2 命中率计数器才能评估。
- **不使用手工事件编号**：核内同步交给 `auto_mutex=True`，依赖关系由框架按 TileGroup 的 mutex 所有权推导，代价是无法逐条比对指令序列。
- **不做 split-K**：每个输出块由一个 worker 沿完整 K 端到端归约。当 N 相对 K 很小、输出块数少于核数时，空闲核无法参与该块的归约；这是为「一套 tiling 覆盖任意 shape」接受的取舍。

内部 JIT 增加 `x2_nz` Tensor 参数：WeightNZ 场景传原始 E4M3FN NZ tensor，ND 场景传 `None`，对应分支由 `WeightNz` 编译期常量删除。公开接口、测试调用方式与 tiling key 不变。Pro 的 nullable Tensor 参数绑定会为 `None` 传空指针，动态维度槽补零，不读取该参数。

本算子**只有 Cube 段**——没有 vector epilogue、没有 split-K、没有跨核部分和 workspace，因此也不含 ICache 预取；输出由 `pl.store(..., phase=STPhase.Final)` 从 L0C 直接写出 BF16。

支持域尚未覆盖：`K % 128 != 0`（需 K 方向尾块掩码或补零）、ScaleNZ、`x1` 的 NZ、非 NN 逻辑 stride。**性能实测只覆盖九个 shape**（`B/M/N/K = 8/96/1024/4096`，是支持域的子集）；支持域内其余 shape 的性能**尚未按测量协议采集**，故 [RESULTS.md](RESULTS.md) 的性能章节只对应这九个 shape，本文不对其余 shape 的耗时做任何声明。此外尾块工作量小于满块、且输出块数不是核数整数倍时末轮有 worker 空转，负载均衡不由构造保证，这属于为支持任意 shape 接受的代价。
