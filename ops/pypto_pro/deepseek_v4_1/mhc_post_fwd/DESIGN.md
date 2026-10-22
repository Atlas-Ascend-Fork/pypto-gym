# MHCPostFwd 设计说明

## 计算阶段划分

单个 `@pl.jit` kernel，全部计算在一个 `with pl.section_vector()` 内完成（无 Cube、无多阶段状态传递）。
每个工作项沿 hidden 分块执行同一段寄存器级数据流，分四步：

1. **系数广播**：每 token 的 16 个 `comb`（FP32）与 4 个 `post`（FP32）从 UB 用 `vf.load_align(dist=BRC_B32)`
   广播进寄存器（各一次，位于 chunk 循环之外）；
2. **提升**：`x` 与 4 行 `residual`（BF16）逐 chunk 用 `vf.load_align(dist=UNPK_B16)` +
   `vf.astype(CastLayout.ZERO, DT_FP32)` 位精确提升到 FP32；
3. **乘加**：每个输出份 `mo` 一条 `vf.mul(x, post[mo])` + 4 条 `vf.mul_add_dst(res[mi], comb[mi][mo])`，
   累加顺序固定为 `mi = 0,1,2,3`，全程 FP32；
4. **收窄写回**：`vf.astype(DT_BF16)` 一次性舍入，`vf.store_align(dist=PACK_B32)` 写回 GM。

阶段之间没有中间 GM 往返，也没有跨 chunk 的归约或前缀依赖 —— 因此不存在阶段级同步点，
只有 tile group 的槽位轮转。

## 核间切分策略

- **工作项** = 一个 `(token, hidden tile)` 对，总数为 `n_tok × n_ht`（`n_tok = n0·n1`，
  `n_ht = ceil(h / h_blk)`）。
- **网格**：`num_cores = min(vector_core_num, n_tok × n_ht)`（本机 `vector_core_num = 64`）。
  按工作项总数而非 token 数收敛，因此在 `n_ht > 1` 且 token 较少时仍能占满核。
- **分配**：`for work in pl.range(core_id, total, num_cores)` 的 grid-stride，
  每个核处理 `ceil(total / num_cores)` 或 `floor(...)` 个工作项，无跨核依赖、无原子累加。
- 平台核数由 `get_platform_info().vector_core_num` 读取（约 0.3 µs），**不用**
  `torch.npu.get_stream_limit(current_stream())`（每次约 18.8 µs，需先物化当前 stream）。

## 核内切分策略

### 基本块选择

`h_blk` 是**编译期常量**，由 wrapper 按参考公式现算，`_kernel_for(h_blk)` 惰性构建并缓存：

```text
h_blk = align(h if h <= 4096 else min(ceil(h / 2), 4096), 64)
```

P0 的四个 hidden 得到 3 个不同 `h_blk`（1280 / 2560 / 3584）。
`h_blk` 上界 4096 由 UB 预算决定（见下）。

核内 tile 形状（`MHC = 4`，`VL = 64` 为 FP32 lane 宽）：

| tile | 形状 | dtype | 说明 |
|---|---|---|---|
| `xbf` | `[1, h_blk]` | BF16 | 一个 token 的 hidden 分块 |
| `rbf` | `[4, h_blk]` | BF16 | 同一 token 的 4 行残差 |
| `obf` | `[4, h_blk]` | BF16 | 4 份输出 |
| `cf` | `[4, 8]` | FP32 | `comb`；valid `[4,4]`，行内补到 8 使每行 32 B 对齐 |
| `pc` | `[8, 1]` | FP32 (DN) | `post`；valid `[4,1]`，补到 8 行使 32 B 对齐 |

常数掩码即足够：`h` 与 `h_blk` 都是 64 的倍数 ⇒ 每个工作项的 `valid = min(h_blk, h - h0)`
恒为 64 的倍数 ⇒ `n_chunks = valid // VL` 的每一拍都是满 64 lane，**不需要**逐拍 `update_mask`。
`h % h_blk != 0`（如 `h = 4160`、`h_blk = 2112`）只是末块变窄，仍然 64 对齐，该结论不变。

### 核内 Buffer 划分

五组 `pl.make_tile_group` 轮转缓冲，每组 `STAGES = 3` 个槽（与参考 `num_stages = 3` 对齐），
**每槽 1 KB 对齐**；`auto_mutex` 按 tile group 的 `mutex_ids` 自动插入核内同步。

| tile group | 单槽字节 | 对齐后 | `h_blk = 1280` 末槽地址 |
|---|---:|---:|---|
| `cf` | 128 | 1024 | `0x800` |
| `pc` | 32 | 1024 | `0x1400` |
| `xbf` | `2·h_blk` | — | `0x3000` |
| `rbf` | `8·h_blk` | — | `0x8c00` |
| `obf` | `8·h_blk` | — | `0x10400` |

总占用 = `6144 + 3·slot(2·h_blk) + 3·slot(8·h_blk) + 3·slot(8·h_blk)`：

| `h_blk` | 对应 `h` | `h_tiles` | 总占用 | 占 UB |
|---:|---|---:|---:|---:|
| 1280 | 1280 | 1 | 76800 B | 29.3% |
| 2560 | 2560 / 5120 | 1 / 2 | 144384 B | 55.1% |
| 3584 | 7168 | 2 | 199680 B | 76.2% |
| 4096 | 公式上限 | — | 227328 B | 86.7% |

`_make_kernel` 内有**编译期 UB 预算闸门**，超限直接 `raise ValueError`（不用 `assert`，
`python -O` 下依然生效）。

## 接口选择

- **单 kernel**：一个 `@pl.jit(auto_mutex=True)` kernel，计算体是 `@pl.vector_function`
  的寄存器级实现；`make_tile_group` + `auto_mutex` 管理轮转与核内同步，**不叠加**手工
  `sync_src`/`sync_dst`。
- **ABI 保留原始 4-D rank**：token 轴的展平在核内用 `pl.make_tensor(x, [n_tok, h], [h, 1])`
  等 5 条视图表达（只重算偏移，无数据搬运）；wrapper 因此**不做任何形状处理**——不
  reshape/view/cast/`.contiguous()`，只做校验、整数推导、`torch.empty`、一次启动。
  这是 `wrapper-boundary`（mandatory 约束）的硬要求：*"A host-side `.view()` still violates
  the delivery boundary even when it is only a view"*，且 host 侧补一次设备拷贝既是额外开销
  也是兼容性风险。
- **为什么用 `vf.*` 而不是 tile 级算子**：本算子需要显式 lane 语义（bf16→fp32 位精确提升、
  fp32→bf16 偶数 lane 收窄后连续打包），tile 级写法表达不了；且实测 tile 级（`pl.getval`）
  比寄存器级慢约 4 倍（每次 `pl.getval` 生成一对 `get_buf/rls_buf`）。
- **领域语言（DSL）对代码组织的硬约束**（由 `pypto_pro` parser 报错实测，决定了下面的写法）：
  1. `@pl.vector_function` 体内**不能调用非 vector 内联函数**（`_call_parser: Vector function cannot
     call non-vector inline function`），因此 VF 的逐输出份计算必须内联展开，不能抽成宏；
  2. traced 函数体内的 `for` **必须用 `pl.range()`**（`_control_flow_parser: For loop must use
     pl.range()`），普通 Python `for` 与列表推导式均被拒绝，因此无法用循环压缩 20 次系数广播、
     也无法把 4 个输出份收成循环；
  3. 具名元组 / dataclass 等构造器调用在 traced 体内不被接受，因此 VF 的 tile 参数只能逐个列出；
  4. `pl.load(dst_tile, src_tensor, offsets)` **没有目标偏移参数**，因此无法把 `comb` 与 `post`
     合进同一个 UB tile 来减少参数与 tile group 数。

  受此约束，VF 保持展开写法（20 次系数广播 + 4 段输出份），并把编译期常量 `h_blk` 以闭包引用
  （不占参数）；kernel 侧则用模块级宏 `_load_tile` 把「夹有效形状 → 搬运 → 恢复物理形状」三段
  收成一行。两者生成的目标代码与展开前**逐指令一致**。

- **上板实测的工具链约束**（各条均由探针确定）：
  1. 4 行 FP32 输出必须用 4 个独立累加寄存器；单个累加 tile 在 `pl.range` 内复用会算错
     （`test_multi_item_per_core` 固化该回归）。
  2. `[4,4]` / `[4,1]` FP32 tile 非法（`Cols × sizeof(dtype)` 需 32 B 对齐）；
     故 `cf` 用 `[4,8]`、`pc` 用 `[8,1]`（DN）。
  3. `load_align(dist=UNPK_B16)` + `astype(CastLayout.ZERO, DT_FP32)` 位精确还原 bf16→fp32；
     fp32→bf16 收窄把结果放在**偶数** lane，只有 `StoreDist.PACK_B32` 能连续打包。
  4. `vf.load_align` 不能作为调用实参嵌套，必须先赋值给局部变量。
  5. `vf.axpy` 的 multiplicand 必须是标量；`vmula` 对应 `vf.mul_add_dst`。
  6. `astype` 需要新的目标寄存器名（寄存器类型在首次赋值时固定，承接加宽结果的 FP32 名不能复用 BF16 名）。
  7. UB 编译期容量 `0x40000`（256 KB）。

## 流水设计

- **轮转深度 3**：五组 tile group 各 `STAGES = 3` 个槽，`g_*.next()` 取下一个槽，
  `auto_mutex` 按 `mutex_ids` 在生产者/消费者之间插同步。3 深使第 `N+1` 个工作项的
  `load`（MTE2）能与第 `N` 个工作项的 VF 计算、以及第 `N-1` 个的 `store`（MTE3）重叠。
- **无手工同步**：没有跨核依赖、没有跨 section 的 barrier，全部依赖边由 `auto_mutex` 托管。
- **重叠证据**（NPU 指令时间线，逐 P0 case，`instr-profiling=on`）：目标 task 时间窗内
  12 个受监控 lane 的 `compute+movement` 同-lane 区间重叠为任务时长的 27.5%–136.7%
  （中位 108%，跨 lane 求和故可 >100%）。源码 Tile DAG 中的唯一合法重叠边就是相邻工作项的
  slot 轮转，因此该重叠对应「相邻工作项的 load/store 与当前工作项的 VF 计算重叠」。
- **未被流水掩盖的残留**：MTE2 占用 0.948–0.989（见 `RESULTS.md`）说明搬运单元已饱和，
  重叠已足够；小 shape 的带宽利用率偏低（62%）来自每工作项数据量小，与流水深浅无关
  （改 `STAGES` 与批量装载系数的 A/B 都为中性）。
