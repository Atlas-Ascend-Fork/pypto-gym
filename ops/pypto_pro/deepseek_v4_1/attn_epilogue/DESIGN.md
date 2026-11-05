# AttnEpilogue 设计说明

单文件 kernel（`op_kernel/attn_epilogue.py`）每次调用只启动 **一个** `@pl.jit` kernel，核心计算全部在 kernel 内；wrapper 只做参数校验、只读元数据读取、纯 Python 整数推导与 scratch 分配。

## 计算阶段划分

| 阶段 | 执行域 | 内容 | 数据交接 |
|---|---|---|---|
| **M1** | Vector（56/64） | inverse RoPE（按 512 维 head 切片）+ 回归组为 8 group + MXFP8 量化 | 写出 GM 载荷 `aq`（FP8E4M3）与 `asc`（E8M0） |
| **M2** | Cube（28/32） | GEMM1：`y[t, g*1024+n] = Σ_k aq[t,g,k]·woa[g,n,k]`，K=4096，FP8×FP8→FP32 | 累加器经 C→V 通道交给 Vector |
| **M3** | Vector（56/64） | 对 `y` 再做 MXFP8 量化 | 写出 GM 载荷 `yq`（FP8E4M3）与 `ys`（E8M0） |
| **M4** | Cube（28/32） | GEMM2：`out[t,m] = Σ_k yq[t,k]·wob[m,k]`，K=8192，bf16 写回 | 直接写 `out` |

四个阶段同属一个 native task。M1 的 `aq/asc` 与 M3 的 `yq/ys` 经 GM 传递，M2 的累加器经 C→V 共享环交给 M3；第二级首块权重在 L1 中跨阶段保留。分段发布、共享环事件及阶段交接分别保护这些数据路径，不能仅凭一个全局 barrier 推断全部片上状态的生命周期。

## 核间切分策略

- **核配额**：按实际硬件选择 28 Cube + 56 Vector 或 32 Cube + 64 Vector，每个 Cube 固定配 2 个 Vector subblock。
- **第一级工作空间**：按 **M → group → N** 排列，任务数 `8 × nm × ⌈1024/BN1⌉`（`nm = ⌈padded/BM⌉`）。
  按 M 优先可让同一 group 的 N 分块落在相邻核上，长输入时每个 Cube 拿到互不重叠的输出 tile。
- **第二级工作空间**：任务数 `nm × ⌈5120/BN⌉`，每个 Cube 负责不重叠的输出 tile。
- **Vector 内部**：同一 Cube 的两个 Vector 按**补齐行对半**分工（各自处理该 tile 的一半行）。
- **量化分段发布**：32 核布局把激活量化切成两个 **2048 列**分段，每段全部 Vector 写完后再通知 Cube，
  使 M1 与 M2 重叠。
- **不切 K**：两级 GEMM 都不做 K 维跨核拆分（无 split-K/stream-K 合并），K 在核内流式消费——
  这避免了部分和合并的额外搬运与 cast-order 风险。
- **布局自适应**：`tiling_key_for(t, cube_cores)` 以 `cube_cores == 32` 区分两套布局（32 核用
  `BN1=256/BN=160/K0=128`，少数核布局用 `BN1=160/BN=192`），tiling key schema 共 6 个合法 key，
  公开调用在 32 核机器上只产生 3 个编译版本，且签名不含 `T` 的静态取值。

## 核内切分策略

### 基本块选择（32 核布局）

| BM | BK | VSlots | K0 | BN1 | BN | 适用 T |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 512 | 2 | 256 | 256 | 160 | `T ≤ 64` |
| 128 | 256 | 2 | 128 | 256 | 160 | `64 < T ≤ 128` |
| 256 | 256 | 1 | 128 | 256 | 160 | `T > 128` |

基本块同时受 L0C、L0B、L1 和 UB 约束。容量与布局由所选工具链的原生声明检查，采用核心 skill 的[容量契约](../../../../cannbot-skills/ops/pypto-pro-kernel/references/api/memory-limits.md)，不以单个 SDK 配置文件替代当前设备校验。

- **L0C**：每槽预留 `4 * BM * max(BN1, BN)` 字节，深度 `min(2, 65536 // (BM * max(BN1, BN)))`；六档配置中，`BM=256` 使用一槽，其他档使用两槽。
- **L0B**：每槽按 `K0 * max(BN1, BN)` 字节分隔，槽数为 `256 // K0`，并为对应 scale 平面保留配对地址。
- **L1**：A/B 双缓冲按 `2 * (BM + max(BN1, BN)) * BK` 字节预留；随后 A-scale 区预留 `BM * 256` 字节，B-scale 区容纳两阶段中较大的声明。
- **UB**：C→V 共享环占 `2 * BM * BN1 * VSlots` 字节，另计量化 scratch 与偏移。inverse RoPE 以 `R_ROWS=8` 行处理，32 核布局的 `y` 量化以 `Q_ROWS=64` 行处理，使工作区峰值受行块控制。

### 核内 Buffer 划分

- **L1**：A/B 条带各双缓冲，两个投影共享每个物理槽的 mutex ID。B 槽按 `max(BN1, BN) * BK` 预留间距；A/B scale 起点也在两阶段保持一致，A-scale 区按第二级 `K=8192` 的 256 个 scale 列预留。各阶段保留自己的有效形状和布局，B 条带经 `ZN` 转置视图供 MX matmul 消费。
- **L0**：L0A/L0B 环各 `256 // K0` 槽。L0B 的槽距为 `max(BN1, BN) * K0`，对应 ScaleLeft/ScaleRight 地址等于数据槽地址 `>> 4`。两个阶段的对应槽保留相同 mutex ID。
- **L0C**：两阶段使用相同槽距 `4 * BM * max(BN1, BN)`，深度为 `min(2, 65536 // (BM * max(BN1, BN)))`。Acc 的 `phase=`/`unit_flag` 依赖仍需在阶段切换时排空；统一 mutex ID 不替代该完成协议。
- **UB**：C→V 共享环（`VSlots` 槽，每槽 `[BM/2, BN1]` fp32）供 `y` 交接；量化 scratch 依次为
  bf16 中间、量化载荷、E8M0 序列、倒数表、偶/奇列展开缓冲与 scale 打包缓冲，全部落在同一个
  静态 UB 布局内。
- **GM 工作区**：4 个缓冲区（`aq/asc/yq/ys`）合计 `42240 × ceil(T/16)×16` 字节，按
  (device, stream, padded) 键缓存复用，图捕获期间只分配一次。

## 接口选择

- **对外 API**：`attn_epilogue(o, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos) -> bf16 [T,5120]`，
  与原生参考语义一致（ND 布局 + 独立 E8M0 scale，不做权重量化）。
- **kernel 入口**：单一 `@pl.jit(arch="a5", auto_mutex=True, tiling_key=EpilogueKey)`，`T` 与地址在
  运行期传入；`EpilogueKey` 用 `TilingKeyField` 声明 `BM/BK/VSlots/K0/BN1/BN`，`is_valid` 枚举两套
  硬件布局的合法组合，使编译产物数量可控且可枚举校验。
- **wrapper 边界**：只做 shape/dtype/device 校验、`scratch_buffers` 分配与**一次** `launch_kernel`；启动不在 host
  循环内，也不做任何张量运算。
- **精度接口**：输出与中间载荷都可与参考逐字节比较（`tests/assets/precision_compare.py`），
  支撑 `rtol=0, atol=0` 的验收方式。

## 流水设计

- **首块权重预取**：进入任务循环前先把 GEMM1 的首块 B tile（含 scale）搬入 L1，与 M1 的尾部 Vector
  工作重叠；任务循环内首个工作项跳过该块的重复加载。
- **K 条带双缓冲**：A/B 条带按 K 交替使用两个 L1 槽，由 `auto_mutex` 保证"载入 vs 消费"的依赖，
  不额外叠加手写同步。
- **L0 深环**：`K0=128` 时 L0A/L0B 各有 2 槽，使 L0 载入与 MMA 重叠；`K0=256` 时单槽但减少 inner 步数。
- **C→V 通道**：GEMM1 的 FP32 累加器经 C→V 共享环交给 Vector，环深 `VSlots` 配合自动事件
  （0/1）实现生产者/消费者推进，`VSlots=2` 允许 drain 与下一次 MMA 重叠。
- **Vector↔Cube 重叠**：量化按 2048 列分段发布，全 Vector 写入完成用事件 5/6 通知，
  Vector→Cube 用事件 3/4；由此 M1 的量化与 M2 的 GEMM1 在时间上重叠，而不是串行。
- **跨投影预取的一致性**：GEMM2 的首块权重在 GEMM1 最后一次读取后写入 L1 的 B 槽 0，使用 `reinterpret(bg, shape=[BN, BK])` 保持 GEMM2 的物理 NZ 行距，并继承槽 0 的 ID。容量按 `max(BN1, BN)` 预留，两个有效 N 宽度无需相等。
- **阶段交接**：Cube 在两级投影之间执行 `bar_all`，排空预取、操作数消费和 MMA/FIX；Vector 端须完成 `yq/ys` 的发布。28 核布局通过 MIX collective 交接，32 核布局通过 Vector 的 MTE3 全局事件 8 和向 Cube MTE2 的事件 7 交接。测试把 Cube 存储域的 `MutexMismatch` 视为错误，同时保留逐字节中间量、输出守卫及交替输入的 graph replay 验证。

32 核布局的 RoPE 临时区与 C→V 共享区存在串行复用，原生声明检查可能报告 `Vec` 重叠；该提示保持可见。最后一个 slab 的 MTE3 全局事件 6 完成后，事件 4 才允许 Cube 加载后半 K；GEMM1 消费全部 K 后才能写共享 UB。这个跨核完成链保护复用，不能用同一个 per-core mutex ID 替代。声明告警、编译成功和实际数值回归分别判断。

## 命名与符号约定

为便于阅读，kernel 内部函数一律用语义化命名（无 `_a`/`_b` 这类字母后缀）：

| 函数 | 职责 |
|---|---|
| `_apply_inverse_rope` / `_apply_inverse_rope_slab` | inverse RoPE（单 head 切片 / 按行 slab） |
| `_compute_mx_block_scales` | 每 32 个元素的 E8M0 块 scale 与倒数 |
| `_quantize_to_fp8_payload` | 用倒数把 bf16 量化为 FP8E4M3 载荷 |
| `_cast_accumulator_to_bf16` | GEMM 累加器（FP32）按行转 bf16 |
| `_pack_mx_scale_bytes` | 把 per-32 scale 字节打包成 GEMM2 读取的 E8M0 形态 |
| `_zero_fill_quantized_tile`（+ `_tail` / `_batch` / `_batch_tail`） | 空 tile / 尾行的载荷与 scale 置零 |
| `_gemm1_tile` | M2：一个 (group, m, n) tile 的 GEMM1，累加器交给 C→V 环，并预取 GEMM2 首块权重 |
| `_quantize_gemm1_output` | M3：把 GEMM1 输出 tile 量化为 `yq`/`ys` |
| `_gemm2_tile` | M4：GEMM2 的一个输出 tile |
| `_prefetch_gemm1_weights` | 任务循环前把 GEMM1 首块 B（含 scale）搬入 L1 |
| `_mx_matmul_strip` / `_mx_matmul_step` | MX 矩阵乘：一个 L1 K 条带 / 其中一个 L0 子步 |
| `_rope_quantize_slabbed` / `_rope_quantize_whole` | M1 整体（slab 分块 / 整段两种变体） |
| `scratch_buffers` / `BUFFER_CACHE` | GM scratch 的取用函数 / 其按 (device, stream, padded) 的缓存 |
| `tiling_key_for` / `launch_kernel` | 按 `(T, cube_cores)` 选档 / 启动一次 kernel |

片上 Buffer 与 GM 载荷的符号：

| 符号 | 含义 |
|---|---|
| `aq` / `asc` | M1 输出的激活 FP8E4M3 载荷 / E8M0 scale（GM 载荷） |
| `yq` / `ys` | M3 输出的 `y` 载荷 / E8M0 scale（GM 载荷） |
| `ag` / `bg` | L1 上 A / B 操作数 tile 组（双缓冲，`Mat` 空间） |
| `asg` / `bsg` | A / B 的 block scale 暂存 tile（`ZZ` / `NN` 布局） |
| `al` / `als` | L0A 上的 A 条带 / 其 scale 条带 |
| `bl` / `bls` | L0B 上的 B 条带 / 其 scale 条带 |
| `accg` | L0C FP32 累加器组（槽数 `acc_slots`） |
| `shared` | C→V（L0C→UB）共享环，深度 `VSlots` |
| `bfg`/`qg`/`esg`/`rg`/`eg`/`og`/`spg` | 量化阶段 UB scratch：bf16 中间、FP8 载荷、E8M0 序列、倒数表、偶/奇列展开、scale 打包 |
