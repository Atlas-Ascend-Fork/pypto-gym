# mega_recurrent_kda_replayssm 设计说明

> 算子本体：`op_kernel/mega_recurrent_kda_replayssm.py`；参考实现与精度比较器：`tests/assets/`。
> 运行平台：Ascend 950（平台信息 `soc_version = 'DAV_3510'`），
> cube（矩阵计算单元）32 核 / vector（向量计算单元）64 核；L0A / L0B 各 64 KiB、
> L0C 256 KiB、L1 512 KiB、UB（片上统一缓冲）248 KiB、L2 128 MiB。

## 0 代码结构

`op_kernel/mega_recurrent_kda_replayssm.py` 自上而下分 9 段，对应下文的四个计算阶段：

| 段落 | 内容 | 关键符号 |
| :--- | :--- | :--- |
| 几何常量 | 固定维度、面板深度、事件 id、m-tile 阶梯 | `H/D/S/CONV_K`、`READY_IDS` / `RELEASE_IDS`、`PUB_VEC` / `PUB_ACK` |
| `GeometryKey` | 编译期 N / m-tile 特化字段（§5.5） | `KEY_NUM_HEADS`、`KEY_PROJ_WIDTH`、`KEY_PANEL_TASKS`、`KEY_M_TILE_ROWS`、`KEY_K_SPLIT`、`KEY_PREFETCH` |
| Cube tile helpers | M1 面板与 M4 输出投影的 tile 级步骤 | `_mm_head`、`_handoff`、`_n384_task`、`_n128_task`、`_aux_gemm*` |
| M2 VF + consumers | Conv1D / SiLU / L2 归一化的 VF 与面板消费 | `conv_tap_vf`、`silu_vf`、`l2norm_vf`、`_load_filter`、`_consume_channel`、`_consume_gate`、`_m2_loop` |
| M3 VF | 递推、decay 激活、RMSNorm + gate 的 VF | `rec_token_vf2`、`rec_pair_vf`、`exp_tile_vf`、`decay_vf`、`rmsnorm_gate_vf` |
| M3 slot loop | item 装入与两个 `KEY_PREFETCH` 变体 | `_m3_load_item`、`_m3_loop_prefetch`、`_m3_loop_inline` |
| loop A 调度 | N384 / N128 任务表与 aux 分派 | `_aux_beta`、`_aux_fa`、`_aux_split`、`_aux_task`、`_loopa_n384`、`_loopa_n128` |
| 模块封装 | 每个阶段一个 host 侧组合函数：声明本段 tile、算几何、调下一层 | `_make_panels`、`_module1_cube`、`_module2_vector`、`_module3_vector`、`_module4_cube` |
| kernel + host | 单个 `@pl.jit` MIX kernel（只做 section 划分与 4 次模块调用）、公开 host wrapper | `mega_recurrent_kda_replayssm_kernel`、`mega_recurrent_kda_replayssm_wrapper` |

两个 M3 变体共用 `_m3_load_item` 的 item 装入逻辑和同一段 3 步 slot 协议注释：`KEY_PREFETCH=1`
（默认）走 `_m3_loop_prefetch` + `rec_pair_vf`（2-token 融合），`KEY_PREFETCH=0`（`≤8 AIC 且 N=6
多组`）走 `_m3_loop_inline` + `rec_token_vf2`（逐 token）。模块导入只定义符号，不注册
`torch.library` op，也没有 `torch.compile` 图模式；所有调用都经过公开入口。

## 1 计算阶段划分

算子的全部工作由**一个** mixed AIC/AIV 的 `@pl.jit` kernel 完成（1 次 launch），
Cube 与 Vector 两个 section 在同一 block 网格内各跑各的阶段，通过面板交接与
2 个阶段 `sync_all(MIX)` 屏障衔接（split-K 启用时另有 1 个 init 屏障）。阶段划分：

```text
[Vector] Zero init  ws_beta_logit / ws_decay_a 清零（仅 KEY_K_SPLIT>1，AIV grid-stride）
[Cube+Vector]  sync_all(MIX) #0（仅 KEY_K_SPLIT>1，保证清零先于 loop A 的 atomic add）

[Cube]   M1 前端  loop A：n_groups × (KEY_PANEL_TASKS + 2·KEY_K_SPLIT) 个任务（扁平 grid-stride）
         ├ N384 面板（3 head/面板，PANEL_N=384）：KC=256 L1 双缓冲 + KS=128 L0 步长，
         │   3 个 head 的累加器 L0C0/1/2，完成后经 DualModeSplitM 写入 UB 面板槽
         ├ gate 面板（同 N384 形状，投影到 output gate）
         └ β / decay-A：按 KEY_K_SPLIT=8 拆 K 的 aux 投影，BF16 atomic add 累加到工作区
         面板级交接：wait RELEASE → move → set READY（每个 (head,slot) 一条事件）

[Cube]   drain：等 6 个 (head,slot) 的 RELEASE，确认面板槽可复用
         sync_all(AIC_ONLY) → decay-B：ws_decay_a @ W_bᵀ（按 (group,head) 平铺）

[Cube+Vector]  sync_all(MIX) #1

[Cube]   M4 输出   slot 循环：wait PUB_VEC → set PUB_ACK → sync_all(AIC_ONLY)
         → 多组：对已就绪的 chunk（每个 m-tile 一个）做输出投影 H/BN 平铺
         → 单组：按 slot 新增的 head 区间增量累积到 L0C，最后一个 head 后 store
[Vector] M2 消费   与 M1 同一批面板的 grid-stride 消费：
         Conv1D(K=4)+SiLU、BF16 边界、Q/K L2 归一化、ws_qkv/ws_gate/replay_k、
         conv_state 原地滑动
[Vector] M3 后端   slot 循环：每 (batch,head) 做 decay/beta 激活、delta-rule 递推
         （KEY_PREFETCH=1 时 2 token/次）、RMSNorm+sigmoid gate → ws_gated/replay_u/replay_decay；
         slot 末 set PUB_VEC → wait PUB_ACK
```

执行顺序：Cube 先跑 M1，Vector 同时在 M2 里消费已就绪的面板（同一任务表的两种切法）；
MIX 屏障 #1 之后，M3 在 AIV 上递推、M4 在 Cube 上按 slot 追赶 M3 的发布，两者以 3 步
握手重叠（细节见第 5 节）。M1→M2 的衔接是**面板交接**而不是屏障，所以两段 Cube/Vector
计算可以在同一个 launch 内前后错开。

## 2 核间切分策略

- **block 结构**：每个 block = 1 个 AIC + 2 个 AIV；AIV 在 block 内用 `sub`（0/1）
  区分自己那份 batch 子块（`KEY_BATCH_PER_SUBBLOCK = KEY_BATCH_PER_GROUP/2` 个 batch）。
- **起核数**：`_GRID_TIERS = (8, 16, 24, 28, 32)`，取不超过
  `min(cube_core_num, vector_core_num // 2)` 的最大档（本平台即 32）；少于 8 个 AIC 时
  host 入口报错。用例可显式指定其中一档。
- **M1 任务表**：`n_groups = B / KEY_BATCH_PER_GROUP`（`KEY_BATCH_PER_GROUP = KEY_M_TILE_ROWS/S` 由 m-tile 档推出），
  `wi ∈ [0, n_groups·(KEY_PANEL_TASKS + 2·KEY_K_SPLIT))`，`nd = wi // n_groups`、`m_group = wi % n_groups`；
  同一组的任务依次是 qkv 面板（`KEY_QKV_PANELS` 个）、gate 面板（`KEY_GATE_PANELS` 个）、β 的 8 个 K-slice、
  decay-A 的 8 个 K-slice。按 `core_id` 做 grid-stride。
- **M2 任务表**：与 M1 的 qkv/gate 面板一一对应（不含 aux）——N384 档每组 `KEY_PANEL_TASKS` 个任务
  （每个任务消费 3 个 head 的面板），N128 档每组 `KEY_N128_TASKS = 4N` 个任务（3 个 qkv 投影 × N 个
  head + gate 的 N 个 head）；同样 grid-stride，`nd = wi // n_groups`。
- **M3 切分**：worker = `block_id·2 + sub`（共 `cores·2` 个），
  `item = (slot·k + kk)·workers + worker`，item 即一个 `(batch, head)`。
- **M4 切分**：`chunk` 是一个 m-tile（`KEY_BATCH_PER_GROUP·KEY_NUM_HEADS` 个 item），chunk 内按
  `(chunk, n_tile)` 平铺，`n_tile` 由 `core_id` 轮转。

各用例的实际起核数（`tests/` 显式指定的档位；`B=16` 的默认用例走 32 档）：

| 用例 | B | N（TP） | 起核数 | 说明 |
| :--- | ---: | ---: | ---: | :--- |
| `b16_auto` | 16 | 6（TP16） | 32 | 默认档，N128 退化边界 |
| `b32_g28` | 32 | 6 | 28 | 非整档网格 |
| `b48_g24` | 48 | 6 | 24 | 任务数恰等于核数 |
| `b128_g28` | 128 | 6 | 28 | 8 组，任务数不能被核数整除 |
| `b512_g28` | 512 | 6 | 28 | 最大合法 B（32 组） |
| `b16_conv_layout0` | 16 | 6 | 8 | 最小档网格 + conv layout 0 |
| `b16_conv_layout2` | 16 | 6 | 8 | 最小档网格 + conv layout 2 |
| `n12_b16` | 16 | 12（TP8） | 28 | TP8 映射 |
| `n12_b256` | 256 | 12 | 28 | TP8 大 batch |
| `n24_b16` | 16 | 24（TP4） | 28 | TP4 映射 |
| `n24_b128` | 128 | 24 | 28 | TP4 大 batch |
| `n24_b4` | 4 | 24 | 28 | 最小 B（m-tile 32 行档） |
| `n12_b8` | 8 | 12 | 28 | m-tile 64 行档 |
| `n12_b64_g8` | 64 | 12 | 8 | 8 核的 m3_k 规则 |
| `n24_b32_g8` | 32 | 24 | 8 | 8 核的 m3_k 规则 |

## 3 核内切分策略

### 3.1 基本块的选择

| 维度 | 取值 | 选择依据 |
| :--- | :--- | :--- |
| M1 基本块 | N384 面板（3 个 head，384 列） | 3 个 head 共享一次 x 搬运；3 个 `[KEY_M_TILE_ROWS,128]` 累加器正好占 L0C 的 192 KiB（256 KiB 容量的 75%） |
| L1 K 面板 `KC` | 256 | `x_l1 [KEY_M_TILE_ROWS,256]` BF16 双槽 128 KiB + 权重 `[384,256]` BF16 双槽 384 KiB = 512 KiB，正好占满 L1 |
| L0 步长 `KS` | 128 | `[128,128]` BF16 = 32 KiB，L0A / L0B 各 64 KiB 恰好放 2 槽；`KS=256` 时单 tile 就占满 64 KiB，无法双缓冲 |
| m-tile `KEY_M_TILE_ROWS` | `M_TILE_LADDER` 最大整除档 | 覆盖 `B=4..512` 的 `rows=B*S`；`KEY_M_TILE_ROWS` 越小每核任务越细、越小档位并行度越高 |
| K 拆分 `KEY_K_SPLIT` | 8 | β / decay-A 的 768 长 K 拆成 8 段摊到空闲核，K=96/段 |
| 发布粒度 `KEY_M3_K` | 见 5.2 | 每个 slot 的 item 数与 M4 就绪块（`KEY_BATCH_PER_GROUP·KEY_NUM_HEADS`）对齐 |
| 输出投影宽度 `BNOut` | 128（≤8 核）/ 256 | 小网格下 128 让 `H/128` 的尾块更少，大网格下 256 减少任务数 |

### 3.2 核内 Buffer 划分

tile 指片上的一块数据缓冲，地址是**静态字节偏移**（编译期算好，运行时不改）：

**L1（Mat，512 KiB）**

| 缓冲 | 形状 / 类型 | 起始字节 | 槽数 | 用途 |
| :--- | :--- | ---: | ---: | :--- |
| `x_l1` | `[KEY_M_TILE_ROWS, KC]` BF16 NZ | 0x00000 | 2 | 面板的 x 切片（`KC=256` 双缓冲） |
| `w_l1` / `w_zn` | `[PANEL_N, KC]` NZ / `[KC, PANEL_N]` ZN | 0x20000 | 2 | 权重面板（同一块字节的两种视图） |
| `w_l1n` / `w_znn` | `[128, KC]` / `[KC, 128]` | 0x20000 | 2 | N128 退化档的单 head 权重面板 |
| aux 别名 | `x_b [KEY_M_TILE_ROWS,128]`、`w_beta [KEY_BETA_WIDTH,128]`、`w_fa [D,128]`、`w_fb [D,128]` 等 | 0x00000 / 0x20000 | 2 | β / decay-A / decay-B 复用 x/w 区域（同一 mutex），阶段间屏障排空 |
| `wout_res` / `wout_res_zn` | `[256, 768]` NZ / `[768, 256]` ZN | 0x00000 | 1 | N=6 且 ≥28 AIC 时 M4 的常驻 Wout 分片（384 KiB），跨 chunk 复用 |
| `a_o_res` | `[KEY_M_TILE_ROWS, TILE]` BF16 | 0x60000 | 2 | 常驻 Wout 路径的 A 操作数 |

**L0（A/B 各 64 KiB，C 256 KiB）**

| 缓冲 | 形状 / 类型 | 起始字节 | 槽数 | 用途 |
| :--- | :--- | ---: | ---: | :--- |
| `l0a` | `[KEY_M_TILE_ROWS, KS]` BF16 | 0x0 | 2 | x 的 L0 切片（32 KiB/槽） |
| `l0b` | `[KS, TILE]` BF16 | 0x0 | 2 | 权重的 L0 切片（32 KiB/槽） |
| `l0c0/1/2` | `[KEY_M_TILE_ROWS, TILE]` FP32 | 0x0 / 0x10000 / 0x20000 | 各 1 | N384 面板的三个 head 累加器（64 KiB 各） |
| aux `a_l0` / `acc` | `[KEY_M_TILE_ROWS,128]` / `[KEY_M_TILE_ROWS,256]` | 0x0 / 0x20000 | 1 | aux 与 M4 复用；M4 的 `BNOut=256` 时用 `acc_b`（0x20000） |

**UB（248 KiB）**

| 缓冲 | 形状 / 类型 | 起始字节 | 槽数 | 用途 |
| :--- | :--- | ---: | ---: | :--- |
| `panel_h0/1/2` | `[KEY_M_TILE_ROWS_HALF, TILE]` FP32 | 0x00000 / 0x10000 / 0x20000 | 各 2 | M1→M2 面板交接（每个 head 位置 2 槽） |
| M2 tiles | `pf/fb/filt/h3b/h3/r8b/r8/seq/sc/ob/sv8/bh` | 0x30000 起 | 1 | Conv/归一化/存储的临时行 |
| `st` | `[D, D]` FP32 | 0x00000 | 1 | M3 的递推状态（MIX 屏障后复用面板区） |
| `q8/k8/v8/o8/sq/d8` | `[S, D]` FP32 | 0x10000 起（每 0x1000） | 1 | M3 的 token 行 |
| `t8` / `t8p` | `[S,D]` BF16 | 0x16000 / 0x18600 | 各 1 | cast 暂存；`t8p` 只给预取路径用 |
| `uh/b18/b18f/kvr/dr/alpha/dtb/r81/gm` | `[1,D]` FP16 / `[S,KEY_BETA_WIDTH]` / `[1,D]` FP32 / `[S,1]` FP32 | 0x17000 起 | 各 1 | replay_u、β 暂存、递推 dot、decay 参数、RMSNorm 中间量 |
| `kvr1/op0/op1/dr1/scs` | `[1,D]` FP32 / `[1,8]` FP32 | 0x18E00 起 | 各 1 | 2-token 融合的 dot / delta / 交叉标量 |

### 3.3 寄存器布局

一个 FP32 向量寄存器是 64 lane 宽（`DH = D/2 = 64`），`[*,D]` 的行按两个 64 元素半宽处理：

| 寄存器 | 用途 |
| :--- | :--- |
| `x` / `k` / `w` | Conv 一段的输入、卷积核、以及投影搬入时的 64 元素半行 |
| 状态行 | M3 的 `[V,K]` 状态按行寻址：每次读 2×64 lane（`st0/st1`）算 kv、delta、更新 |
| 标量链 | `beta`、`decay`、`scale` 等由 `vf.full` / 广播 load 灌入整行 |

M3 的递推是逐 `dv` 行的向量循环（每 token 约 3k 条指令），每 token 对状态读 2 遍、写 1 遍
（每 item 8 token ≈ 1.5 MB UB 流量）；实测单 item ≈ 10 µs，是"指令吞吐 + UB 流量"受限，
不是 `reduce_sum` 的延迟受限。

## 4 接口选择

| 环节 | 用法 | 理由 |
| :--- | :--- | :--- |
| 全局内存 ↔ L1 | `pl.load` / `pl.store` + `pl.set_validshape` | 只搬有效行/列；NZ / ZN 由 `load(..., order=...)` 直接得到，代价为零 |
| Conv 权重的三种 layout | `pl.load(..., order=[0,2])` / `[1,2]` | `[3,N,4,D]` 与 `[3,4,D]` 的两种视图都由同一份字节转出 |
| L1 → L0 | `pl.move` | `[128,128]` 的 32 KiB 切片，与 `KS=128` 的 2 槽双缓冲配合 |
| 矩阵乘 | `pl.matmul` / `pl.matmul_acc` | 累加器常驻 L0C，K 方向逐步累加 |
| L0C → UB 面板 | `pl.move(..., acc_to_vec_mode=DualModeSplitM)` | 一次把 128 行拆成两个 AIV 子块（`KEY_M_TILE_ROWS_HALF`）写入面板槽 |
| 面板交接 | `pl.system.set_cross_core` / `wait_cross_core` | 6 条 READY（Cube→Vector）+ 6 条 RELEASE（Vector→Cube），不占 MIX 屏障 |
| 后端 3 步握手 | `set/wait_cross_core(PUB_VEC/PUB_ACK)` + `sync_all(AIC_ONLY)` | 每 slot 的发布/确认 + 全 AIC 硬件屏障，让 M4 与下一 slot 的 M3 重叠 |
| 分段屏障 | `pl.system.sync_all(core_type=MIX)` | 阶段屏障分隔 M1/M2 与 M4/M3；split-K 启用时另有 init 屏障，保证 kernel 内清零先于 loop A 的 atomic add |
| 累加写 | `pl.store(..., atomic=AtomicType.AtomicAdd)` | β / decay-A 的 K-slice 并发累加（BF16 atomic add） |
| 向量计算 | `vf.*`（`conv_tap_vf` / `silu_vf` / `l2norm_vf` / `exp_tile_vf` / `decay_vf` / `rmsnorm_gate_vf` / `rec_token_vf2` / `rec_pair_vf`） | 全部在 `@pl.vector_function` 内，一条向量指令对应一段 64 lane |
| tile 与同步 | `pl.make_tile_group` + `auto_mutex=True` | 槽位轮转的互斥由 mutex id 自动管理 |
| 编译期特化 | `@pl.jit(tiling_key=GeometryKey)` + `pl.STATIC` extent | N 与 m-tile 折进编译变体，batch 维保持动态 |
| 平台信息 | `get_platform_info()` 的 `cube_core_num` / `vector_core_num` | 起核数从平台问，不写死 |

## 5 流水设计

### 5.1 面板交接（M1 → M2）

每个 head 位置有 2 个 UB 面板槽（`PANEL_DEPTH=2`），事件按 `h*2+slot` 编号：
Cube 写槽前 `wait RELEASE`（Vector 已消费完），写完 `set READY`；Vector 消费前
`wait READY`，消费完 `set RELEASE`。6 条 READY（0..5）与 6 条 RELEASE（6..10、14，跳过
MIX 屏障占用的 11..13）互不干扰。loop A 结束后 Cube 显式 drain 这 6 条 RELEASE，
才允许复用面板区。

### 5.2 两段屏障 + 3 步 slot 握手（M3/M4）

M3 与 M4 的依赖是**按 slot** 的，不是全网格同步：

1. M3 每个 slot 处理 `m3_k·workers` 个 item，末 item 的 `ws_gated` 落盘后
   `set PUB_VEC(MTE2, INTRA_BLOCK)`；
2. M4 等 `PUB_VEC` 后 `set PUB_ACK(FIX)`，M3 等 `PUB_ACK` 才能进入下一 slot；
3. M4 再 `sync_all(AIC_ONLY)`，让 `completed` 台账对全网格成立，然后只处理
   "已完整就绪"的 chunk（`KEY_BATCH_PER_GROUP·KEY_NUM_HEADS` 个 item = 一个 m-tile）。

这样 M4 处理当前 chunk 时，M3 已经在算下一个 slot。`m3_k` 由 host 查表后写入 `KEY_M3_K` tiling key（编译期常量，kernel 不再接收标量参数），取值：

- `≤8 AIC`：`N≥12 → 8`（8 AIC 只有 16 个 AIV，slot 容量太小时 M4 就绪块被切得过碎）；
  `N=6` 沿用 `B≤32→1、B≤64→2、其余 4`。8 核 N=6 多组改用 64-item 细 slot 会因握手
  过密慢约 7%，因此不在 8 核上套用 N=6 的细 slot 规则。
- `>8 AIC`：`N=6` 且 `B>32`：16/24/28 AIC → 2，32 AIC → 1；N=6 的 B≤32 与
  N=12/24 的 `B≤32→1、B≤64→2、其余4` 不变。N=6 的输出投影 K 只有 768、chunk 多，
  16/24 AIC 用 `m3_k=2` 比 1 快 5–6.5%（B=128/256/512），32 AIC 用 1 更稳；N=12/24
  的 K 更长、chunk 少，`m3_k=1` 反而因握手过密慢约 5%/11%，保持 4。

#### 5.2.1 单组增量输出（`n_groups == 1` 且 `>8 AIC`）

单组时 `chunk_items = KEY_BATCH_PER_GROUP·KEY_NUM_HEADS` 等于 `total_items`，按 chunk 发布会一直等到最后一个
slot，M3 与 M4 串行。现在改为按 head 区间增量累积：

- M3 的 item 顺序在单组内改为 **head-major**（`head = item // B`、`b = item % B`），
  每个 slot 完成的是“所有 batch 的连续若干个 head”；
- M4 每个 slot 只处理该 slot 新完成的 head 区间
  `[completed_before // B, completed // B)`，把
  `ws_gated[:, head*D:(head+1)*D] @ Wout` 的贡献用 `matmul_acc` 累加到常驻 L0C 的
  `acc_b`，到最后一个 head 才 `set_validshape` + store；
- 这样 slot `s` 的 M4 与 slot `s+1` 的 M3 重叠，单组小 batch 的输出投影不再整体暴露。

N=6 且 ≥28 AIC 时，Wout 的 256 列分片（384 KiB）在 M4 开始时一次搬入 L1 的
`wout_res`；之后每个 head 只做 L1→L0 切片，避免每个 chunk 重读整块 Wout。多组路径
仍按原 chunk 逻辑处理。

#### 5.2.2 M3 跨 item 预取

每个 worker 的 item 序列是 `item + n_workers`，所以当前 item 的最后一个 token 完成后，
立即把下一个 item 的 checkpoint（64 KiB）和 raw q/k/v/decay 装入 `st` 与
`q8/k8/v8/d8`（raw BF16 先落到 `t8p`），与当前 item 的 RMSNorm/gate/store 以及下一
item 的 decay/beta 激活重叠。第一个 item 的这批数据在 slot 循环前做一次 prologue
预取。gate 的加载从 token 循环之后提到循环之前，与递推重叠；beta / dt_bias / alpha
仍在 item 开头加载。

`KEY_PREFETCH=0` 的路径（`≤8 AIC 且 N=6 多组`）走 `_m3_loop_inline`（item 装入与循环体共用
`_m3_load_item`），保留原 inline 加载与 `m3_k=4`；其余路径 `KEY_PREFETCH=1` 走
`_m3_loop_prefetch`。这样 8 核 TP16 大 batch 不因预取/细 slot 回退：
B=128/256/512 为 970.1/1928.8/3849.0 µs，与改前 971.6/1929.4/3851.4 一致。

`dv` 循环的 2× 手动展开实测为中性到略差（B=512 971.5 vs 970.9 µs，B=16 68.4 vs
67.0 µs），未保留。

#### 5.2.3 2-token 状态融合（`KEY_PREFETCH=1`，`rec_pair_vf`）

每两个 token 共享一次 state 载入/写出：

- pass 1 同时算 `<S, d_t k_t>`、`<S, d_t d_{t+1} k_{t+1}>`、`<S, d_t q_t>`、
  `<S, d_t d_{t+1} q_{t+1}>` 四个点积，并用 4 个 D 长交叉标量
  `s_k=<k_t,d_{t+1}k_{t+1}>`、`s_q0=<k_t,q_t>`、`s_q1=<k_t d_{t+1},q_{t+1}>`、
  `s_q2=<k_{t+1},q_{t+1}>` 组合出两个 delta 和两个 o；
- pass 2 写
  `S_{t+1} = S*diag(d_t d_{t+1}) + delta_t (k_t d_{t+1})^T + delta_{t+1} k_{t+1}^T`。

state 读写从每 token 4 载/2 存降到每 2 token 4 载/2 存。数值上 `replay_u` 最大
绝对误差 2.8e-3（契约 5e-3），`ws_gated` 2.6e-3（BF16），`out` 4.0e-4；8 核 N=6
多组的 `KEY_PREFETCH=0` inline 路径仍逐 token。

### 5.3 split-K aux（KEY_K_SPLIT = 8）

β / decay-A 的 K 长度为 768（= 56 个 `TILE` 的 3 个 `KC` 块）。按 `KEY_K_SPLIT=8` 拆成 8 个
K-slice（每段 96），每个 slice 是一个独立任务，摊到 loop A 的空闲核上并行做
`[128,96]×[96,KEY_BETA_WIDTH/TILE]` 的矩阵乘，再以 BF16 atomic add 累加进 `ws_beta_logit` /
`ws_decay_a`（AIV 在 kernel 内先清零，经 MIX 屏障后才进入 loop A）。实测不拆（`KEY_K_SPLIT=1`）时 1 组小 batch
慢 3%~10%，所以单组也保持拆分。

### 5.4 N384 / N128 退化

`use_n384 = (n_groups·KEY_PANEL_TASKS·4) >= (num_cores·3)`：任务数不足以铺满 3/4 网格时退化为
单 head 的 N128 面板（`4N + 2·KEY_K_SPLIT` 个任务/组），保证每个核都有活干；否则用 3 head 的
N384 面板以减少 x 重复搬运（x 每面板只搬一次、3 个 head 复用）。

### 5.5 编译期特化（`GeometryKey`）

一个 `@pl.jit(auto_mutex=True, tiling_key=GeometryKey)` kernel 覆盖
N ∈ {6,12,24} × m-tile 阶梯。host 从权重形状推断 N、从 `B*S` 选 m-tile 档，把全部
编译期常量通过 tiling key 传入，parser 折成常量：

| 字段 | 含义 |
| :--- | :--- |
| `KEY_NUM_HEADS` | `N = 96/TP`（6/12/24） |
| `KEY_PROJ_WIDTH` / `KEY_QKV_WIDTH` | `P = N*D` / `P3 = 3*P` |
| `KEY_QKV_PANELS` / `KEY_GATE_PANELS` / `KEY_PANEL_TASKS` | N384 面板数（`P3/384`）、gate 面板数（`P/384`）、二者之和 |
| `KEY_N128_TASKS` | N128 每组的单头任务数（`4N`） |
| `KEY_BETA_WIDTH` | β 工作区/GEMM 的 N 宽度（N≤16 → 16，N=24 → 32；Cube 的 N 粒度为 16） |
| `KEY_M_TILE_ROWS` / `KEY_M_TILE_ROWS_HALF` / `KEY_BATCH_PER_GROUP` / `KEY_BATCH_PER_SUBBLOCK` | m-tile 行数 / 半行（每个 AIV 的面板半宽）/ 每组 batch 数 / 每 AIV 子块 batch 数 |
| `KEY_K_SPLIT` | β / decay-A 的 K 拆分因子（8） |
| `KEY_M3_K` | M3 每个 slot 中每个 worker 的 item 数（发布粒度，1/2/4/8） |

N 与 m-tile 相关的参数 extent 一律声明为 `pl.STATIC`，由 launch 的实际 extent 折进
编译变体；batch 维保持 `pl.DYNAMIC`。
