# attn_prologue 设计说明（PyPTO-Pro）

对应实现：`op_kernel/attn_prologue.py`（单模块，一个 `@pl.jit(auto_mutex=True)` MIX kernel
`attn_prologue_kernel` + 一个公开 host 函数 `attn_prologue` + 权重加载期工具 `to_nz`）。本文只描述
本实现的设备侧结构与流水设计；接口、约束与用例见 `README.md`，精度与性能数据见 `RESULTS.md`。

## 计算阶段划分

一个 kernel 内两条并发指令流（`section_cube` AIC / `section_vector` AIV），由两次
`pl.system.sync_all(core_type=MIX)` 全局屏障衔接为四个逻辑阶段；B 与 B-epi 之间用逐 beat 的
跨核事件握手形成深度 2 的 CV ping-pong：

| 阶段 | 执行部件 | 粒度 | 做什么 |
| :--- | :--- | :--- | :--- |
| A1 投影 | AIC | 每工作单元 | `qa = X @ Wqa^T`（R=1280 → 10 个 128 列块）与 `kv = X @ Wkv^T`（D=512 → 4 个 128 列块），MXFP8 `matmul_mx`/`matmul_mx_acc` 全 K=H=5120 归约（K0=256 × 20 步），FP32 部分和经 FIXPIPE 直写 GM workspace `ws_qa[T,R]` / `ws_kv[T,D]` |
| barrier1 | 全部 AIC+AIV | 一次 | `sync_all(MIX)`：workspace 对 Vector 可见（MTE3→MTE2 全局序） |
| A2 归约/归一化/量化 | AIV（64 个扁平条带） | 每行/每任务 | QR 臂：读 `ws_qa` 行 →（SK 域先合并两个 K 分带）→ RMSNorm(沿 R) → 动态 MXFP8 量化 → 写公开 `qr`(e4m3) 与 `descale_qr`(e8m0 paired)；KV 臂：读 `ws_kv` 行 → RMSNorm(沿 D) → 尾部 RoPE(INTERLEAVE_HALF) → clamp(±448) → E4M3(scale=1) → 按 `cache_index` 整行 scatter 到 `kv_cache` |
| barrier2 | 全部 AIC+AIV | 一次 | `sync_all(MIX)`：`qr/descale_qr` 对 Cube 可见（GM store→reload 一致性） |
| B QB 投影 | AIC | 每 beat（head×col-block） | `q_proj = qr @ Wqb^T`：A 操作数为 GM 回读的量化 `qr`（ND→L1 NZ）+ `descale_qr`（ZZ），B 为 `Wqb` NZ→ZN 别名 + `descale_wqb`（NN），K=R=1280（K0=256 × 5 步）；acc `[128,128]` FP32 经 `DualModeSplitM` 搬到跨段 UB `cv_ub`，`set_cross_core(FIX)` 发布 |
| B-epi Q 收尾 | AIV（每块 2 个半区各 64 行） | 每 beat | `wait_cross_core(V)` 消费 `cv_ub`：非尾块直接 BF16；每 head 最后一个 col-block 的末尾 Dr=64 列做 INTERLEAVE_HALF RoPE 后 BF16；写 `q[T,N,D]`，`set_cross_core(V)` 归还槽位 |

QB 消费的是**量化后**的 `qr`（不是 FP32 归一化值），KV 是**先 norm+rope 再量化**且 scale 固定
为 1——两者都是数值契约（见 `README.md` 功能说明）。

## 核间切分策略

- **启动形态**：`attn_prologue_kernel[None, block_dim]`，`block_dim = min(cube_core_num,
  vector_core_num//2)`（运行期平台查询，32 Cube/64 AIV 时为 32）；每个 block = 1 AIC + 2 AIV。
  一次公开调用一次启动；kernel 内以 `pl.get_block_num()`/`get_block_idx()//get_subblock_num()`
  推导逻辑核号，AIV 半区号取 `get_subblock_idx()`。
- **单二进制、运行期模板**：无 tiling-key；按 `n_tiles = ceil(T/128)` 在运行期二选一工作分配，
  两个分支都被 trace、只有一个执行：
  - **distributed（`n_tiles < 32`，decode 与中小 prefill）**：A1 按 `(tile, col-block)` 单元
    跨步铺开（`n_tiles*NA1G >= num_cores` 时启用 G=2 分组：2 个输出块共享一次 X K-窗口载入）；
    decode 小 T（门控 `n_tiles*NA1*2 <= num_cores`，即 32 核网格上的 `n_tiles==1`）进一步把
    K=H 切成 SK=2 个分区，28 个 `(blk,ks)` 单元铺满核，部分和写 `ws` 的第 `ks*n_tiles*128`
    行带，A2 合并；B 按 `(tile, head, col-block)` 单元以 `num_cores` 跨步铺开，**相邻核同一
    时刻读相邻 128 行 Wqb 带**（NZ k-frac 页内密度最大化）。
  - **split-T（`n_tiles >= 32`，大 prefill）**：一核独占整 tile 的全部 64 head × 4 col-block，
    `qr` 每 tile 只从 GM 驻留 L1 一次并跨 256 个 beat 复用。
- **A2 条带**：行按扁平 AIV 号（`core_id*2+sub`，共 64）跨步；`n_tiles<=2` 时行循环进一步拆成
  `2*t_live` 个独立 QR/KV 任务铺满 64 AIV（prefill 保持行串行循环）。SK 域的 A2 双臂各载两个
  ws 分带做一次 FP32 `pl.add` 合并（2 个部分和时与 Kahan 补偿求和逐位一致：补偿项从 0 起）。
- **wrapper 与 kernel 的门控逐字节一致**：ws 行数（SK 域 `2*n_tiles*128`，否则 `t`）在 host 侧
  用与 kernel 内相同的表达式派生，杜绝分配与写入错位。
- **全局同步**：恰好两次 `sync_all(MIX)`（barrier1/2），两侧 section 同条件同次数同顺序；
  A1→A2 的 workspace 依赖与 A2→B 的 `qr` GM 回读依赖都是 all-to-all（一行 `ws_qa` 由多个核
  写、`qr` 行被多核读），全局屏障是该依赖形态的原生同步。
- **跨核事件预算**：事件 id 仅 0..15，其中 11/12/13 归 `sync_all(MIX)`（AIV 侧 12/13）。
  当前分配：`EV_READY 0..1`（Cube FIXPIPE 发布 → Vector V 消费）、`EV_FREE 2..3`（Vector 归还
  → Cube FIX 复用），导入期 `_check_constant` 断言不与保留 id 冲突。
- **mutex id 预算**：cube 侧 tile group 占用 0..17（NZ/ZN 别名**共享**同一组 id，使 auto_mutex
  看见别名依赖），vector 侧 0..17 + 预载槽 18..25；容量与冲突均在导入期断言。

## 核内切分策略：基本块选择、核内 Buffer 划分

### 基本块选择

| 维度 | 取值 | 依据 |
| :--- | :--- | :--- |
| `BASE_M` | 128 | T-tile 高度（GEMM M）；`DualModeSplitM` 后每 AIV 承担 `ROWS_VEC=64` 行 |
| `K0` | 256 | L0 K 步长。`matmul_mx` 要求 K%64==0；256 使每次 GM→L1 权重/X 突发为 32 KB（MTE2 少而大的事务，kernel 为 MTE2 带宽受限）；同时 L0A/L0B 的 `[128,256]` e4m3 槽 = 32 KB，depth-2 恰好占满 64 KB，L0_MX 2×2 KB 恰好占满 4 KB |
| `BN_A` / `BN_B` | 128 / 128 | QA/KV 与 QB 的输出列块。acc `[128,128]` FP32 = 64 KB；QB 侧 depth-2 ping-pong（128 KB）+ A1 侧 depth-2（128 KB）恰好合占 256 KB L0C。`BN_B=256` 会同时超出 L0B（`[256,256]` e4m3 = 64 KB，depth-2 = 128 KB > 64 KB）与 L0C 预算 |
| `NK_A` / `NK_B` | 20 / 5 | K 归约步数：H/K0 与 R/K0 |
| `A1_GRP` | 2 | prefill 的 A1 X 复用组宽：2 个输出块共享一次 X K-窗口载入，X 的 L2→L1 流量减半（14 次/tile → 7 次）；QA_BLOCKS(10) 与 KV_BLOCKS(4) 均为偶数，分组无残块 |
| `SK_A` | 2（门控） | decode（`n_tiles==1`）的 A1 K 分区数：28 单元铺满 28 核；分区宽 H/2=2560 = 10×K0，NZ 网格整除 |
| `KV_PRE` | 4 | barrier1 前预载的 A2 KV 任务 cos/sin 行数；门控域（`n_tiles<=2` ⇒ `t_live<=256`，64 AIV 条带）内每 AIV 至多 4 个 KV 任务，恰好全覆盖 |
| b_nz 深度 | 4 | 权重 K-窗口轮转深度：4×32 KB 连续槽，MTE2 载入领先 MTE1/MAC 消费至多 3 拍；等于 B 阶段消费型预取的窗口数（`NK_B-1=4`，保持 b_zn 同余对齐） |

### 核内 Buffer 划分

L1（Mat，512 KB，高水位 `L1_END=0x76000=472 KB`；区域起点 8 KB 对齐，导入期断言不重叠不越界）：

| Buffer | 声明 | 深度 | 字节 | 用途 |
| :--- | :--- | :--- | ---: | :--- |
| `a_l1` | NZ `[128,256]` e4m3 @0x00000 | 2 | 64 KB | X K-窗口（A1 的 A 操作数），GM ND→L1 NZ 由 load 就地转换 |
| `sa_l1` | ZZ `[128,160]` e8m0 @0x10000(+0x6E000) | 2 | 40 KB | `descale_x` 整行驻留，逐 K 步切片进 L0A_MX |
| `b_nz`/`b_zn` | NZ `[128,256]` / ZN `[256,128]` e4m3 @0x16000（slots 2,3 @0x26000） | 4 | 128 KB | 权重 K-窗口；**同一字节的 NZ/ZN 双视图别名，共享 mutex id**——`A @ W^T` 的 B 操作数由 ZN 视图 `pl.move` 进 L0B |
| `sb_l1` | NN `[160,128]` e8m0 @0x36000 | 2 | 40 KB | `descale_wqa/wkv`（A1 B-scale，`order=[1,0]` 转置载入） |
| `qr_l1` | NZ `[128,1280]` e4m3 @0x40000 | 1 | 160 KB | QB 的 A 操作数：公开 `qr` 输出的 GM 回读，split-T 下每 tile 驻留一次 |
| `sq_k` | ZZ `[128,40]` e8m0 @0x68000(+0x74000) | 2 | 10 KB | `descale_qr` 整行（QB A-scale） |
| `sw_k` | NN `[40,128]` e8m0 @0x6A000(+0x6C000) | 2 | 10 KB | `descale_wqb`（QB B-scale） |

L0A/L0B/L0_MX（每 core group）：`l0a` NZ `[128,256]` 与 `l0b` ZN `[256,128]` 各 depth-2
（槽 0x0000/0x8000，各 64 KB 恰好占满）；`scale_a_l0` ZZ `[128,8]` 与 `scale_b_l0` NN `[8,128]`
depth-2（槽 0x0000/0x0800）。**MX 硬约束 `addr(scale)==addr(data)>>4` 由显式地址声明满足**
（0x8000>>4=0x800），轮转时数据槽与 scale 槽按同一 `.next()` 序联动。

L0C（Acc，256 KB）：`a1_acc` NZ fractal=1024 `[128,128]` FP32 depth-2（0x00000/0x10000，
A1 的 G=2 分组需要 2 个累加器）+ `q_acc` 同形 depth-2（0x20000/0x30000，CV ping-pong）。
`compact=1` 的 acc 窗口在首个 matmul 前一次设满物理窗，matmul 与 drain 之间不再改窗。

UB（Vec，用户预算 216 KB，高水位 `UB_END=0x27E00=163.5 KB`）：

| 槽 | 字节 | 用途 |
| :--- | ---: | :--- |
| `cv_ub` ×2 | 64 KB | 跨段 CV 握手槽 `[64,128]` FP32（声明在两个 section 之外） |
| `qa_row`/`qa_row2`、`gamma_qr` | 5.25+5.25+5.25 KB | A2 QR 臂：ws 行（SK 域双带）、gamma（QR_PAD=1344 保证末组 64-lane 载入不越 tile） |
| `qr_payload`、`exp`、`scale_u8` | 1.25+0.25+0.06 KB | 量化产物与指数字段 scratch |
| `kv_row`/`kv_row2`、`gamma_kv`、`kvb`、`cos_row`/`sin_row` | 2+2+2+0.5+0.5 KB | A2 KV 臂 |
| `q_bf16` ×2 | 32 KB | B-epi 输出缓冲 `[64,128]` BF16（depth-2 与 cv_ub 同拍） |
| `cos_ub`/`sin_ub` | 16+16 KB | B-epi RoPE 表 `[64,64]` FP32（tile-change 提升载入） |
| `cos_pre`/`sin_pre` ×4 | 2+2 KB | barrier1 前预载的 A2 KV cos/sin 行（512 B 槽距） |

## 接口选择

- **Cube 侧显式三段式**：`pl.load`（GM→L1）/ `pl.move`（L1→L0、L0C→UB）/ `pl.matmul_mx[_acc]`，
  不使用高层 gemm 封装。原因是需要精确控制 L1/L0 地址以满足 MX scale 的 `>>4` 耦合、让 NZ/ZN
  别名共享 mutex id、并独立管理各缓冲的轮转深度。
- **`A @ W^T` 的 NZ 权重**：FRACTAL_NZ 权重按 `[N,K]` NZ 存储，不可经 `order` 转置读取；用
  **同址双视图**（NZ `[BN,K0]` 载入 + ZN `[K0,BN]` 搬出）得到 L0B 需要的转置分形，两视图共享
  mutex id 使 auto_mutex 看见别名依赖。
- **E8M0 scale 的 GM→L1**：源张量 rank≥3 且末维为长度 2 的物理 phase 轴（paired 公开布局），
  phase 轴偏移恒 0、`order` 不选该轴；A-scale 落 ZZ、B-scale 以 `order=[1,0]` 转置落 NN——这是
  ZZ/NN 分形唯一可用的生成路径。L1→L0_MX 用 `pl.move`（ZZ→ZZ / NN→NN）。
- **QR 反馈走 GM 回读**：A2 把 `qr/descale_qr` 写入公开输出张量（GM），barrier2 后 Cube 以
  ND→L1 NZ（payload）与 phase 轴规则（scale）回读作为 QB 的 A 操作数。选择该路径的原因：
  UB→L1 的 ND→NZ 就地转换对 e8m0 scale 不存在 ZZ/NN 生成通路，且 GM 路径让 A2 的行条带与 B 的
  单元分配完全解耦（生产者/消费者形状无需匹配），barrier2 同时提供 store→reload 全局一致性。
- **Acc→UB**：`pl.move(..., acc_to_vec_mode=DualModeSplitM)`，valid_M 先向上对齐到 2 再对半，
  两个 AIV 各得 64 行；被 `pl.move` 排空的 matmul **不传 `phase=`**（无 STPhase 消费者会挂起
  unit_flag）。
- **Vector 数值全部走 VF**（`@pl.vector_function`）：RMSNorm 用 `vf.reduce_sum` → `vf.full(reg,
  preg, pos=DuplicatePos.LOWEST)` 广播 lane0 → `vf.sqrt` + `vf.div(one, root)`（VF 无 rsqrt；
  `1/R`、`1/D` 为模块级常量，VF 内禁止标量浮点除法）；共享指数取 `bit_cast→i32 >>23` 减
  `127+8` 再 `maxs(-127)`，零组 `select` 到字节 0；倒数 `2^-e` 由 FP32 指数字段构造
  （`neg→+127→<<23→bit_cast`），无除法；E4M3 转换 `vf.astype(..., CAST_RINT, SaturateMode.ON)`
  且先显式 clamp(±448)；payload 密排用 `StoreDist.PACK4`（b32→b8 低字节），BF16 用
  `bit_cast(u32)` 后 `StoreDist.PACK`；RoPE 去交错用钳位到 0..63 的寄存器 `gather` 索引
  （避免 DINTLV 末行过读），两次 mul 与 add 分开保持舍入次序；每次 UB scratch 写后读之间
  `vf.mem_bar(VST_VLD)`。
- **分页 KV scatter**：`slot = pl.getval(cache_index, row)`（INT64）后
  `pl.store(kv2d, kvb, [slot, 0])`（`kv_cache` 以 `make_tensor` 展平为 `[P*BS, D]` 视图），
  整行 512 B 一次落盘（≥64 B 安全粒度）；`cache_index` 唯一性由接口契约保证。
- **同步分层**：核内 pipe 序交给 `auto_mutex=True` + tile group `mutex_ids`（跨 op 依赖一律
  `make_tile_group`，不用裸 `make_tile`）；CV 握手用 `set/wait_cross_core`（FIX↔V，深度 2
  credit，`beat>=2` 守卫 + 退出前自然配平）；全局相位用两次 `sync_all(MIX)`。

## 流水设计

**B ↔ B-epi 深度 2 CV ping-pong**（beat = 本核处理的第几个 (head,col-block) 单元）：

```text
AIC: matmul_mx×5 → acc[beat%2] → move(cv_ub[beat%2], DualModeSplitM) → set(FIX, beat%2)
AIV: wait(V, beat%2) → RoPE/BF16 → store q → set(V, 2+beat%2)
AIC: wait(FIX, 2+beat%2)  (beat>=2 时) → 复用 cv_ub 槽
```

Cube 的第 beat+1 个 QB matmul 与 Vector 的第 beat 个收尾重叠；两 section 以相同的单元枚举序
（同一 `pl.range(core_id, n_work, num_cores)` 解码）保证 beat 对齐。

**MTE2 一级预算下的载入调度**（kernel 为 GM→L1 带宽受限，流水设计以「减少字节 × 增大突发 ×
隐藏延迟」为轴）：

- **权重窗口 depth-4 轮转**：`b_nz` 4 槽使 MTE2 载入领先 MTE1 搬出与 MAC 消费至多 3 拍，
  打断单槽/双槽下 step k+1 载入对 step k 读出的 WAR 串行。
- **小 scale 缓冲 depth-2 轮转**（`sw_k/sa_l1/sq_k`）：消除逐 beat / 逐单元边界的
  「载入 WAR 等待上一拍 L1→L0_MX 搬出」停顿，机制与权重轮转同类。
- **单元跨步的带局部性**：distributed B 以 `(tile,head,col-block)` 为单元、`num_cores` 跨步，
  使相邻核同一时刻读相邻 128 行 Wqb 带（NZ k-frac 页内密度最大化），41.9 MB 权重流以聚合
  L2→L1 流带宽满速运行。
- **A1 X 复用（G=2 分组）**：一个 X K-窗口喂 2 个输出块的 MMA，X 的 L2→L1 流量减半；decode
  退回逐块（更多核发 MTE2 更短墙钟），由 `n_tiles*NA1G >= num_cores` 门控自适应。
- **屏障窗口载入隐藏**：两个全局屏障各有一个空闲窗口——A1 期间 64 AIV 空闲：gamma 与 A2 首批
  KV_PRE=4 行 cos/sin 在 barrier1 **之前**预载（纯输入，无依赖）；A2 期间 32 AIC 空闲：B 首批
  4 个 Wqb 窗口 + 首个 `descale_wqb` 在两屏障之间预取，且 B 的首 beat **实际消费**预取槽
  （跳过对应载入；`.next()` 全部无条件推进，两路径轮转计数一致，split-T 分支以 `pre=False`
  折叠为与不开预取逐指令相同）。
- **冗余载入按变化点提升**：distributed B 的 `qr_l1/sq_k` 仅在本核首单元或 tile 变化时装载；
  B-epi 的 `cos_ub/sin_ub` 镜像同一规则（表只依赖行偏移 ro）。
- **A2 任务化（decode）**：`n_tiles<=2` 时 QR/KV 拆成 `2*t_live` 个独立任务铺满 64 AIV，
  缩短屏障间暴露的向量墙钟；任务→行映射确定性条带化，与预载槽一一对应。
- **尾块**：`set_validshape` 在每次观察 extent 的操作前设置（GM 面用 `rows_full`，NZ fractal 面
  用 16 对齐的 `rows_r`，acc 保持全物理窗）；`DualModeSplitM` 的 valid_M 先对齐到 2 再对半，
  Vector 侧按半区重算自己的 `store_rows` 并守卫 `store_rows>0`；物理行 `ceil(T/128)*128` 只服务
  workspace/NZ，公开输出与 scatter 严格按逻辑 T。
