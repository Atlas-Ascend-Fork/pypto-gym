# indexer_prologue_qw 设计说明

单文件 kernel（`op_kernel/indexer_prologue_qw.py`，1224 行）每次调用只启动 **一个**
`@pl.jit(auto_mutex=True)` MIX kernel；公开 wrapper 仅做校验、分配与一次启动，不含 host 计算。

## 计算阶段划分

kernel 内以 `with pl.section_cube()` / `with pl.section_vector()` 划分 CV 边界：

| 阶段 | 执行域 | 内容 |
|---|---|---|
| **Cube：Q 路径** | Cube | `qr @ wqb^T`，MXFP8（E4M3 字节 + 每 32 个 K 一个 E8M0）→ FP32 累加，L0 K 步 `K0=128`（`NK=10` 步，每步 4 个 scale
组） |
| **Cube：W 路径** | Cube | `x @ ww^T`，BF16，K=`dim=5120`（`WK0=128`，`NWK=40` 步）→ FP32 累加 |
| **CV 握手** | Cube↔Vector | 累加器经 L0C → UB 交接，向量阶段接手做收尾 |
| **Vector 收尾** | Vector | 只对末尾 `Dr=64` 列做 rotate-half RoPE（两半各 32 维）→ 每 32 列一组求 `amax`/E8M0 → E2M1 打包（偶列低 4 bit）→ 写回 GM
|
| **Vector：W 归约** | Vector | 仅 split-K 且 `n_tiles ≤ 12` 时：在 grid barrier 之后对各 K 分片做确定性求和 |

数据流：`GM → L1 → L0A/L0B → L0C → UB → GM`。Q 在 UB 内完成 RoPE 与 MXFP4 打包后写回 GM；
W 要么直接写结果，要么先把分片写入 workspace 待归约。

## 核间切分策略

- **T 维切块**：沿 `T` 轴按 `BASE_M=128` 行切成 `n_tiles = ceil(T/128)` 个 tile。
- **split-T（`n_tiles ≥ SPLIT_T_TILES=32`）**：一个核独占一个 `T` tile 的**全部 32 个 head**，
  即该核负责 `[128, 4096]` 的 Q 输出 tile；W 路径的 K 归约（`NWK=40` 片）**交错进 Q head 循环**
  完成，避免额外的 kernel/归约阶段。
- **split-K（否则）**：head 轴切成 `HB_K=16` 个 **2-head block**（`HPB_K=2`），提高并行度；
  当 `n_tiles ≤ SPLIT_W_TILES=12` 时，W 的 K 轴也按同一 head block 划分（`WSW=160` 的 L1 K 窗口），
  各 block 把 K 分片写入 workspace，向量单元在 **grid barrier 之后求和**（确定性，非原子加）。
- **启动几何**：按 `_cube_cores()` 选择（32 Cube / 少数核布局）。
- **Vector 核内分工**：一个 Cube 配 2 个 Vector，`ROWS_VEC=64`——即 128 行的 tile 按 **AIV 半边**
  分工；尾块时按有效行夹紧 store。

## 核内切分策略

### 基本块选择

| 常量 | 值 | 含义 |
|---|---|---|
| `BASE_M` | 128 | `T` 轴 tile 高度（GEMM M） |
| `K0` / `NK` | 128 / 10 | Q 路径 L0 K 步与步数（`q_lora=1280`） |
| `SK0` / `SQ_ALL` | 4 / 40 | 每个 Q L0 步的 scale 组数 / 整行 scale 组数（`MX_GROUP=32`） |
| `WK0` / `NWK` | 128 / 40 | W 路径 L0 K 步与步数（`dim=5120`） |
| `WSW` / `WSUB` | 160 / 80 | split-K W 的 L1 K 切片窗口 / L0 子块（受 64 KB L0A 预算约束） |
| `ROWS_RED` | 2 | split-K W 归约的带宽高度（每次归约行数） |
| `VL` / `MX_GROUP` / `GROUPS` | 64 / 32 / 4 | 每寄存器 fp32 lane 数 / MX 组宽 / 每行组数 |
| `ENC_REGS` / `AMAX_LEN` | 4 / 256 | 每 AIV 半边的量化寄存器数 / `amax` 缓冲长度 |
| `LUT_LEN` / `LUT_REGS` / `LUT_SHIFT` | 1088 / 17 / 20 | 片内 E2M1 桶表长度 / 寄存器数 / 索引位移 |

选择依据：Q 路径的 `K0=128` 使每个 L0 步恰好覆盖 4 个 MX scale 组（`SK0=4`），scale 平面与
数据平面按 `addr(scale) = addr(data) >> 4` 对齐；W 路径按 `WK0=128` 切成 40 片，使 split-T 与
split-K 两种调度共享同一 K 步长。`WSW=160` / `WSUB=80` 由 L1 切片与 64 KB L0A 预算反推。

### 核内 Buffer 划分

- **L1**：Q/W 两条路径各自的 A/B 操作数 tile（`wqb`/`ww` 以 FRACTAL_NZ 行连续搬入）+ block scale
  平面（`SQ_ALL=40` 组）；split-K 下另有 `WSW=160` 宽的 W K 切片窗口。
- **L0A/L0B**：按 L0 步长切环（Q 路径 `K0=128`、W 路径 `WK0=128`，子块 `WSUB=80`）；
  `set_validshape` 把 L1 载入夹到有效行，避免尾块越界读。
- **L0C**：两条路径各自的 FP32 累加器——Q 为 `[BASE_M, D]`（保持**全物理窗口**，不做运行期收缩，
  以免 L0C mis-layout），W 为 `[BASE_M, N=32]`。
- **UB**：向量收尾工作集——RoPE 的 `Dr=64`（两半各 32 维）在寄存器内完成；量化侧每 AIV 半边
  `ROWS_VEC=64` 行 × `GROUPS=4` 组，占 `ENC_REGS=4` 个寄存器，`amax` 用 `AMAX_LEN=256`
  缓冲；`LUT_LEN=1088` 的 E2M1 桶表占 `LUT_REGS=17` 个寄存器，由 `ipqw_build_lut_vf` 在片内生成。
- **GM workspace**：仅 split-K 的 W K 分片需要，由 wrapper 一次性分配、随 kernel 生命周期复用。

## 接口选择

- **对外 API**：`indexer_prologue_qw(x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin, rope_cos,
  softmax_scale=SOFTMAX_SCALE) -> (q, descale_q, w)`；`SOFTMAX_SCALE` 默认 `128**-0.5`。
- **kernel 入口**：唯一 `indexer_prologue_qw_kernel`，`T` 与地址运行期传入；冻结几何以模块常量
  表达（`DIM/QLORA/N_HEADS/D/DR/BASE_M/…`），编译产物不随 `T` 变化。
- **数据类型选择**：Q 路径输入与权重为 FP8E4M3FN（也接受 uint8 视图），块 scale 为 E8M0 成对
  布局（`[..., 20, 2]`）；W 路径输入/权重为 BF16、输出 FP32（缩放由 kernel 乘入）；Q 输出为打包
  E2M1（uint8）+ E8M0（uint8），避免 host 侧再做打包。
- **片内生成量化表**：不接收任何 host 私有输入，`ipqw_build_lut_vf` 按 `LUT_SHIFT=20` 构造
  幅值桶表，使收尾可用截断索引得到精确最近邻（含 `.5` 处远离零）。
- **wrapper 边界**：只做 `_validate_metadata`（shape/dtype/格式）、`torch.empty` 分配
  （`q`/`descale_q`/`w`/workspace）、按 `_cube_cores()` 启动一次 kernel；host 调用链无循环、
  无计算、无 reshape。

## 流水设计

- **CV 握手**：Cube 阶段把累加器经 L0C→UB 交给向量阶段，向量收尾（RoPE/量化/打包）与下一 tile 的
  GEMM 推进重叠，而不是阶段串行。
- **split-T 的 K 归约内联**：W 路径的 40 片 K 归约交错进 Q head 循环，靠同一核内的 L0 环与
  累加器交替推进，省掉独立归约阶段与额外同步。
- **split-K 的确定性归约**：K 分片写 workspace，grid barrier 之后由向量单元求和（非原子），
  避免浮点原子加的次序不确定。
- **尾块流水**：Q 累加器保持全物理窗口、L1 载入 `set_validshape` 夹有效行、store 端按 AIV 半边
  夹紧；W 累加器按运行期有效 `M` 裁剪 store——尾块因此不需要 host 侧补齐或额外 kernel。
- **量化收尾的查表化**：E2M1 收尾以一次查表（截断索引）替代 7 次中点比较，缩短向量关键路径。
- **单 launch、图友好**：一次 kernel 启动、静态分配、无 host 侧计算，因此可以直接放进 NPU-graph
  重放（`RESULTS.md` 的冻结计时口径即基于此）。
