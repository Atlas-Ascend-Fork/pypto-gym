# mixed_quant_sparse_flash_mla 设计说明（PyPTO-Pro）

对应实现：`op_kernel/mixed_quant_sparse_flash_mla.py`。接口与约束见
[`README.md`](README.md)，精度与性能结果见 [`RESULTS.md`](RESULTS.md)。

> 执行单元是**单个 `@pl.jit(auto_mutex=True)` kernel**，不做 L1 分 Module 分段开发；
> 下文 §3 的"阶段"是 kernel 内的逻辑阶段，不是交付 Module。

## 0. 目标与约束回顾

- 13 个 P0（README.md「约束说明」的合约）真机精度 PASS（pro_scheme_a + LSE 1e-5 +
  空行严格 0/sink + 输出有限）。
- 平台：Ascend950PR_9579（950PR_957x.ini：UB 248KB/AIV、L1 512KB、L0A/L0B 64KB、
  L0C 256KB、28 Cube/56 Vector）。launch `block_dim = metadata.planner blocks`（≤`AIC_CORE_MAX_NUM=36`；host 规划器缺省取当前设备 `cube_core_num`）。
- host 边界：wrapper 只校验、`torch.empty`、派生标量、一次 launch；metadata 由
  host 入口 `mixed_quant_sparse_flash_mla_metadata` 生成（用户指令）。

## 1. Kernel 签名（typed + 标量）

```python
@pl.jit(auto_mutex=True, tiling_key=MqsmlaTilingKey)
def mixed_quant_sparse_flash_mla_kernel(
    q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    ori_kv: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
    ori_sparse_indices: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    ori_block_table: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    ori_topk_length: pl.Ptr[pl.DT_INT32],
    sinks: pl.Tensor[[pl.DYNAMIC], pl.DT_FP32],
    metadata: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    cmp_kv: pl.Ptr[pl.DT_UINT8],
    cmp_sparse_indices: pl.Ptr[pl.DT_INT32],
    cmp_block_table: pl.Ptr[pl.DT_INT32],
    cmp_topk_length: pl.Ptr[pl.DT_INT32],
    cu_seqlens_q: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    lse: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    fd_partial_ws: pl.Tensor[[pl.DYNAMIC], pl.DT_UINT8],
    scale_arg: pl.DT_FP32,
    ori_kv_stride0: pl.DT_INT64,
    cmp_kv_stride0: pl.DT_INT64,
    b_num: pl.DT_INT64,
    k_ori: pl.DT_INT64,
    k_cmp: pl.DT_INT64,
    bs_ori: pl.DT_INT64,
    bs_cmp: pl.DT_INT64,
    bt_ori: pl.DT_INT64,
    bt_cmp: pl.DT_INT64,
):
```

启动形式为 `mixed_quant_sparse_flash_mla_kernel[None, block_dim, key](*args, *runtime_args)`；
`block_dim`（`metadata` 规划出的核数）是 grid 维、不是 kernel 参数。

- **前 14 个参数 = 公开张量**，顺序即 `ARG_ORDER`、也就是 README 接口表 inputs+outputs 的顺序：
  KV 池、稀疏索引与块表在 kernel 内用 `pl.make_tensor` 重建字节 / 二维视图（页间 padding 由
  `ori_kv_stride0` / `cmp_kv_stride0` 表达）。可选输入声明为 `pl.Ptr[...]`，缺省时 wrapper 传
  空指针，是否解引用由 tilingkey 条件（`HAS_ORI_LEN` / `HAS_CMP_LEN` / `HAS_CMP`）在编译期决定。
- **第 15 个参数 `fd_partial_ws`** 是 wrapper 分配、**不属于公开接口**的 FlashDecode partial
  工作区（`uint8[MQSMLA_WORKSPACE_BYTES]`，`(72·64·512 + 2·72·64) × 4` B）。FD 打开时 kernel 在
  其中切出 `fd_o [72,64,512]`(FP32, +0)、`fd_m [72,1,64]`(+9437184)、`fd_l [72,1,64]`(+9455616)
  三块；FD 关闭时整段不编译。
- **10 个运行时标量**（顺序 = 签名尾部）：`scale_arg`（FP32，softmax scale）、
  `ori_kv_stride0` / `cmp_kv_stride0`（INT64，KV 池页距）、`b_num`（= `B+1`）、`k_ori` /
  `k_cmp`（两侧 TopK）、`bs_ori` / `bs_cmp`（页 slot 数）、`bt_ori` / `bt_cmp`（块表宽度）。
  全部运行期传入且形状无关 ⇒ **改数值不触发重编译**。
- **tilingkey** 只承载 4 个 1-bit 编译期条件：`HAS_ORI_LEN` / `HAS_CMP_LEN` / `HAS_CMP` /
  `FD_ENABLE`（见 `MqsmlaTilingKey`）。`HAS_CMP=0` 时压缩侧槽位为空，不得解引用。
- LSE 恒输出（`return_softmax_lse=False` 时由 wrapper 丢弃，见 README 接口说明）。

## 2. Metadata 消费（host 生成，int32[1024]）

- 本核 `fa = 9*block_idx`：`enable=md[fa]`；`bn2_start=md[fa+1]`、`m_start=md[fa+2]`、
  `s2_start=md[fa+3]`(=0)、`bn2_end=md[fa+4]`、`m_end=md[fa+5]`、`s2_end=md[fa+6]`(=0)。
- 行区间：`m_global_first = cu_q[bn2_start] + m_start`；行数
  `row_count = m_end - m_start (+1 if s2_end>0)`；uniform 下 s2_end=0 → 行数 = m_end-m_start。
- `fd_used_vec_num = md[900]`（host 规划恒 0）：非 0 属合同违约，kernel 直接
  断言式跳过 FD（本版本不实现 FD 归约路径）。
- `enable==0` 的核无行任务：直接返回（无跨核事件，不存在全局 barrier）。

## 3. 核心数据流（每行每 KV tile）

行内 tile 序列 = ORI ceil(W_ori/128) 个 + CMP ceil(W_cmp/128) 个（连续在线 softmax）。

### 阶段 A（AIV，每 tile）
1. 取本 tile 128 行中本 AIV 的 64 行（AIV0 行 0-63、AIV1 行 64-127）；
   tile 内无效列（pos ≥ actual_n）地址 clamp 到 `pos = actual_n-1`。
   每行地址：`idx = getval(idx_flat, m*K+seg+pos)`（int64）→ `page=idx//bs; off=idx%bs`
   → `phys = getval(bt_flat, b*BT+page)` → `addr = phys*stride0 + off*row_bytes`；
   `pl.load(bytes_tile[row], kv_flat, [addr])`（[1,row_bytes] u8，1D 视图）。
2. VF 反量化（ORI：512B FP8→FP32 ×16 scale BF16→FP32 → BF16；CMP：256B FP4 低 nibble
   在前 → BF16 值 ×32 scale → BF16），写 UB `kv_bf16[row, 512]`。CMP 的每个 scale 通过
   `LoadDist.E2B_B16` 在 load 时直接扩展成 16 lanes，并提前到 FP4 load/cast 前覆盖依赖延迟；
   不再构造 lane index 或执行 `vf.gather`。
3. `pl.move(UB kv_bf16 → L1 kv_half[本 AIV 半区])`（ND 字节拷贝）→ `set_cross_core(KV_READY)`。

### 阶段 B（AIC，每 tile）
1. `wait_cross_core(KV_READY)`。
2. QK：4 个 d-chunk：`L0A K[128,128] NZ ← L1 kv chunk（ND→NZ）`；
   `L0B Q^T[128,64] ZN ← L1 qT chunk`；`matmul_acc(S_l0c[128,64], ...)`（chunk0 init）。
3. `pl.move(S_ub, S_l0c, acc_to_vec_mode=DualModeSplitM)`：L0C NZ→UB DN [64,128]，
   AIV0 得 head 0-31、AIV1 得 head 32-63 → 每 AIV [32,128] FP32。
4. `set_cross_core(S_READY)`。

### 阶段 C（AIV，每 tile）
1. `wait_cross_core(S_READY)`。
2. 行 softmax（head = 行，kv = 列）：`m_chunk = row_max(S[:, :valid_n])`（mask 无效列）；
   首 tile 以 sinks 播种（m=sinks_h, l=1, res_o=0）；`m_new=max(m_old,m_chunk)`；
   `alpha=exp(m_old-m_new)`；`p=exp(S-m_new)`（无效列写 0）；`l=l_old*alpha+row_sum(p)`。
3. P → BF16 → UB NZ [32,128] → `pl.move(UB → L1 p_half[本 AIV])` → `set_cross_core(P_READY)`。
   （alpha 缓存到 UB 供阶段 E 使用。）

### 阶段 D（AIC，每 tile）
1. `wait_cross_core(P_READY)`。
2. 每半区 h∈{0,1}、每 d-chunk c∈0..3：`L0A p_half_h[32,pv_k] NZ`（validshape K=pv_k=
   ceil(actual_n/16)*16）、`L0B V[128,128] ZN ← L1 kv chunk c`、
   `matmul_acc(O_l0c_h_c[32,128])` → `pl.move(O_ub_h_c, O_l0c_h_c)`（L0C→UB，ND）。
3. 4 chunk 全部搬完 → `set_cross_core(O_READY)`。

### 阶段 E（AIV，每 tile）
1. `wait_cross_core(O_READY)`。
2. 4 chunk：`res_o[:, c] = res_o[:, c]*alpha + O_ub_c`（VF，FP32）。
3. 末 tile：`out_tile = res_o / l` → cast BF16 → `pl.store(out, ...,
   [m, half*32, 0])`；`lse[m, half*32+h] = log(l)+m`（FP32 store）。

### 空行（W_ori==0 且 W_cmp==0）
两 AIV 直接写 out=0、lse=sinks（不经 Cube）。

## 4. 存储布局

### L1（512KB）
| 区域 | 形状/dtype | 大小 | 说明 |
|---|---|---|---|
| kv_half0/1 | [64,512] BF16 ND ×2 slot | 2×64KB×2slot=256KB | 每 slot 两 AIV 半区连续；alias 4 个 [128,128] chunk 视图供 L0A/L0B |
| qT | [512,64] ZN | 64KB | GM Q 降序 load 一次/行；alias 4 个 [128,64] chunk |
| p_half0/1 | [32,128] NZ ×2 slot | 4×8KB=32KB | 两 AIV 各写自己的 tile |

（原型可用单 slot：KV 128KB + qT 64KB + P 16KB = 208KB。）

### UB（248KB/AIV）
| 区域 | 形状/dtype | 大小 |
|---|---|---|
| res_o | [32,512] FP32 | 64KB |
| kv_bf16 | [64,512] BF16 | 64KB |
| bytes | [16,640] u8 ×2 | 20KB |
| S_ub | [32,128] FP32 | 16KB |
| p_ub / p_nz | [32,128] BF16 ×2 | 16KB |
| O_ub | [32,128] FP32（单缓冲，每 chunk 复用） | 16KB |
| 状态（m/l/alpha/sinks/exp） | [32,1] FP32 若干 + [1,32] | ~4KB |
合计 ≈ 238KB（超限时 bytes 降为 [16,608]×2）。

### GM workspace
无（本设计不需要 GM 中转；FD 字段为 0）。

## 5. 同步（4 条流水通道 / 10 个 INTRA_BLOCK 事件 id）

AIC 与 AIV 之间的握手全部用 `pl.system.set_cross_core` / `wait_cross_core`，`sync_mode`
一律 `CrossCoreSyncMode.INTRA_BLOCK`；`pipe=` 指的是**本核**的管道（AIC 侧用 `FIX` / `MTE1`，
AIV 侧用 `V` / `MTE3`）。16 个事件 id 使用 0–9，10–15 预留。

每条数据流占一对 id：**偶数 = 就绪（生产者 set、消费者 wait），奇数 = 回收（消费者 set、
生产者 wait）**。§3 阶段描述里沿用的事件符号名对应到**就绪**边：`KV_READY` = id 0、
`P_READY` = id 2、`S_READY` = id 4（slot0）/ 8（slot1）、`O_READY` = id 6。

| 流（缓冲 / 深度） | 就绪（set → wait） | 回收（set → wait） | 关联 |
| :--- | :--- | :--- | :--- |
| L1 KV 环 `l1k_grp`（3） | **0** AIV `MTE3` → AIC `MTE1` | **1** AIC `MTE1` → AIV `MTE3` | 反量化后的 KV 发布到 L1 / 槽位归还 |
| L1 P 双缓冲 `l1p0` / `l1p1`（各 depth=1，共 2 槽） | **2** AIV `MTE3` → AIC `MTE1` | **3** AIC `MTE1` → AIV `MTE3` | P 发布到 L1 / 槽位归还 |
| S 双缓冲 slot0（`ub_s_grp[0]`、`l0c_s_grp[0]`） | **4** AIC `FIX` → AIV `V` | **5** AIV `V` → AIC `FIX` | S 分块 L0C→UB 搬完 / UB 槽位归还 |
| S 双缓冲 slot1 | **8** AIC `FIX` → AIV `V` | **9** AIV `V` → AIC `FIX` | 同上，另一个槽 |
| L0C O 环 `l0c_o_grp`（6）+ `t_4..t_7` | **6** AIC `FIX` → AIV `V` | **7** AIV `V` → AIC `FIX` | O 的 4 个 chunk 就绪 / 分块归还 |

- **S 环按 `4 + 4·slot` 取 id**（slot0 → 4/5，slot1 → 8/9），即最初的 `4·t%2 + phase` 轮转
  约定；QK 用 `clk & 1` 选槽、softmax 用 `clk_1 & 1` 选到同一槽，从而让 QK(t+1) 的搬出不再
  等 softmax(t) 释放 `ub_s`。
- **预热 credit 数 = 缓冲深度**：进循环前两侧各注一次——AIC 侧 `1`×3、`3`×2，AIV 侧
  `5`/`9`/`7` 各 1。这样每轮的 `wait → 消费 → set` 不依赖前序 tile 即可启动，跨 tile 流水的
  首轮串行被消除。
- 每条流每 tile 恰好 1 次 set / 1 次 wait。id `7` 在源码里有 4 处 `wait`，但落在
  `mod_29` × `actnm <= 16` 的**互斥分支**中，每 tile 只执行一处。
- 不占事件 id 的同步：AIV 本地的 `pl.system.bar_all()`（store / 写回排序，`PIPE_ALL`），以及
  FD 收尾的 `pl.system.sync_all(core_type=MIX)`。
- 成对与到达规则沿用知识库约束 `sync-stitch.md`
  （`cannbot-skills/ops/pypto-pro-op-kb/constraints/`）。

## 6. 尾块与边界

- `actual_n = min(128, W_pool - 128*tile_in_pool)`；行 tile 数由 W_ori/W_cmp 决定。
- softmax 行 max/sum 只统计有效列（VF mask）；P 无效列存 0（显式清零后写有效值）。
- PV K 维 = pv_k=ceil(actual_n/16)*16，L0A/L0B validshape 裁剪；[actual_n,pv_k) 行
  来自 clamp 后的真实 KV（有限值）×P=0。
- T1 尾行/空行由 metadata 行区间覆盖；out/lse 全量写后读（torch.empty 初始化合规）。
- block_table/索引越界属非法输入（见 README「约束说明」），kernel 不做值校验。

## 7. 与知识库约束的对应

- online-softmax-tail：§3 阶段 C/E 的递推顺序与 sinks 播种、终局归一化。
- buffer-reuse-lifetime：§4 UB/L1 区域生命周期；O_ub 每 chunk 复用；KV slot 轮转。
- vec-row-reduce-broadcast：m/l/alpha 的行归约与按 head 广播缩放。
- wrapper-boundary：§0；metadata host 入口与主 wrapper 分离。
- 其余约束在实现中逐条落实。

## 8. 验证策略

- 原型：单文件 `test_mixed_quant_sparse_flash_mla.py`，13/13 P0 上板，
  逐 case 与 CPU golden 比较（pro_scheme_a 判据 + LSE/空行附加检查）。
- 精度定位：FP4/FP8 位语义失败时，用 golden 的三套独立解码（torch view/位数学/LUT）
  对比单行/单 tile dump（pl.printf/dump_data 可用则用）。
- 性能：msprof `task duration`，`warmup 2 + repeat 10` 取后 10 次中位；任何改动走同会话
  交错 A/B（口径与结果见 RESULTS.md「性能结果」）。
