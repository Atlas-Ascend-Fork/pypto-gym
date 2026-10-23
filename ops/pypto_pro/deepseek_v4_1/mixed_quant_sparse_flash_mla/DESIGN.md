# DESIGN — mixed_quant_sparse_flash_mla（PyPTO-Pro，单 kernel CV 融合）

> Scriptor 模式：本设计的执行单元是**单个 `@pl.jit(auto_mutex=True)` kernel**；
> 不做 L1 分 Module 分段开发。§3 的"阶段"是 kernel 内的逻辑阶段，不是交付 Module。

## 0. 目标与约束回顾

- 10 个 P0（SPEC machine-contract）真机精度 PASS（pro_scheme_a + LSE 1e-5 +
  空行严格 0/sink + 输出有限）。
- 平台：Ascend950PR_9579（950PR_957x.ini：UB 248KB/AIV、L1 512KB、L0A/L0B 64KB、
  L0C 256KB、28 Cube/56 Vector）。launch `block_dim = metadata.planner blocks`（≤28）。
- host 边界：wrapper 只校验、`torch.empty`、派生标量、一次 launch；metadata 由
  host 入口 `mixed_quant_sparse_flash_mla_metadata` 生成（用户指令）。

## 1. Kernel 签名（typed + 标量）

```python
@pl.jit(auto_mutex=True)
def mqsmla_kernel(q, ori_kv, ori_sparse_indices, ori_block_table,
                  ori_topk_length, sinks, metadata,
                  cmp_kv, cmp_sparse_indices, cmp_block_table, cmp_topk_length,
                  cu_seqlens_q, out, lse,
                  softmax_scale, has_cmp, ori_kv_stride0, cmp_kv_stride0,
                  block_dim)
```

- 张量 14 个：顺序即 SPEC inputs+outputs 顺序；KV 池/索引/页表在 kernel 内用
  `pl.make_tensor` 重建一维/二维字节视图（页间 padding 由 stride0 标量表达）。
- 标量：`softmax_scale`(FP32)、`has_cmp`、`ori_kv_stride0`、`cmp_kv_stride0`、
  `block_dim`(int)。`has_cmp=0` 时 cmp 槽位为占位张量，不得解引用。
- LSE 恒输出（SPEC 偏差已声明）。

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
   在前 → LUT 值 ×32 scale → BF16），写 UB `kv_bf16[row, 512]`。
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

## 5. 同步（每 tile 4 对 INTRA_BLOCK 事件，全用例统一）

| 事件 | 生产 | 消费 | ID |
|---|---|---|---|
| KV_READY | AIV0+AIV1（move 完成后 set） | AIC（wait 后读 L1 KV） | 0 |
| S_READY | AIC（L0C→UB 后） | AIV0+AIV1 | 1 |
| P_READY | AIV0+AIV1 | AIC | 2 |
| O_READY | AIC（4 chunk 搬完） | AIV0+AIV1 | 3 |

- 原型为全同步握手（不跨 tile 流水）；tile 间同 buffer 复用由事件顺序保证。
- 优化轮可引入双缓冲 + 事件 ID 轮转（ID = 4*t%2 + phase）做 tile 间流水，
  遵守 sync-stitch.md 的成对与到达规则。

## 6. 尾块与边界

- `actual_n = min(128, W_pool - 128*tile_in_pool)`；行 tile 数由 W_ori/W_cmp 决定。
- softmax 行 max/sum 只统计有效列（VF mask）；P 无效列存 0（显式清零后写有效值）。
- PV K 维 = pv_k=ceil(actual_n/16)*16，L0A/L0B validshape 裁剪；[actual_n,pv_k) 行
  来自 clamp 后的真实 KV（有限值）×P=0。
- T1 尾行/空行由 metadata 行区间覆盖；out/lse 全量写后读（torch.empty 初始化合规）。
- block_table/索引越界属非法输入（SPEC shape_constraints），kernel 不做值校验。

## 7. 与约束/模式的对应（KB_USAGE 摘要）

- online-softmax-tail：§3 阶段 C/E 的递推顺序与 sinks 播种、终局归一化。
- buffer-reuse-lifetime：§4 UB/L1 区域生命周期；O_ub 每 chunk 复用；KV slot 轮转。
- vec-row-reduce-broadcast：m/l/alpha 的行归约与按 head 广播缩放。
- wrapper-boundary：§0；metadata host 入口与主 wrapper 分离。
- 其余约束在实现中逐条落实（见 KB_USAGE.json）。

## 8. 验证策略

- 原型：单文件 `test_mixed_quant_sparse_flash_mla.py`，10/10 P0 上板，
  逐 case 与 CPU golden 比较（pro_scheme_a 判据 + LSE/空行附加检查）。
- 精度定位：FP4/FP8 位语义失败时，用 golden 的三套独立解码（torch view/位数学/LUT）
  对比单行/单 tile dump（pl.printf/dump_data 可用则用）。
- 性能：优化轮以 msprof latency（32 次 warmup 5）评估，口径为冻结基线。
