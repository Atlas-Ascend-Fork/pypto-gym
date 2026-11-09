# fused_recurrent_kda_snapshot 设计说明

> 算子本体：`op_kernel/fused_recurrent_kda_snapshot.py`；参考实现与精度判据：`tests/assets/`。
> 运行平台：Ascend 950PR（平台信息里 `soc_version = 'DAV_3510'`），cube（矩阵计算单元）
> 32 核 / vector（向量计算单元）64 核，UB（片上统一缓冲）容量 256 KiB。

## 1 计算阶段划分

工作项 = 一个 `(batch, head)` 的 `rows` 行 state，按 token 顺序：

| 阶段 | 内容 | 位置 |
|:---|:---|:---|
| 预处理 | Q/K 行 L2 normalize（`1e-6`）、gate/beta sigmoid 链、`decay = exp(lower*σ(α(g+dt_bias)))`、折叠权重 `kd = beta*decay*key` 与 `vf32 = beta*V` | VF `_prep` |
| key 点积 | 逐行 `sk[r] = dot(S_row_r, kd)`，**一次 4 行在飞**，结果 `FIRST_ELEMENT_B32` 紧凑写回 | VF `_step_dot` |
| 扁平 delta | `u = vf32 - sk`，**一条向量指令覆盖全部持有行** | VF `_fold_u` |
| 递推 + 输出 | 逐行 rank-1 更新 `S_new = S*decay + u⊗key`、输出点积 `o = dot(S_new, q)`、紧凑写回；同时把 `S_new` 写入快照 tile（BF16 池 `write_snap=False`，活槽仅作占位、不写快照，随后 `_narrow_state` 生成真实快照） | VF `_step_upd_snap` |
| 输出打包 | FP32→BF16 `astype(ZERO)` + dense `PACK_B32` | VF `_pack_out` |
| 快照写回 | 快照 tile 即时 DMA 到 `state[ssm_state_indices[t]]` | MTE3 |
| 输出写回 | `out` BF16 tile DMA | MTE3 |

## 2 核间切分策略

- 工作项 = `(batch, head, 行块)`，共 `B*N*(128/rows)` 个；按
  `pl.range(block_idx, total_items, block_num)` 轮转，各 worker 计数差不超过 1。
- 启动 worker 数 `min(vector_core_num, 64)`。基准形状 `B=16, N=6, rows=64` 给出
  `192` 个工作项，恰好 3 个/worker；`56`/`60` 会留下 4 个的临界路径（实测慢 20%+）。
- 三种布局（BSND / BNSD / TND）共用同一 kernel body：布局差异只体现为 host 侧算出的
  行 stride 标量与 `packed` 标志，kernel 内不做布局分支。

## 3 核内切分策略

### 3.1 基本块的选择

`rows ∈ {32, 64}`。`64` 使每个 token 的常量开销（`_prep` 的归一化/激活/折叠）在 2 倍
行数上摊薄；`32` 对小的 `B*N` 给出更细的负载粒度。基准形状下 `64` 更优。

### 3.2 核内 Buffer 划分

`rows=64, slots=3` 时：

| Buffer | 大小 | 说明 |
|:---|---:|:---|
| `state_g` | 3 × 32 KiB | 工作 state 槽，token `t` 读槽 `t%3`、写槽 `(t+1)%3` |
| 快照 / staging | 64 KiB | 独立双缓冲快照 tile（FP32 直写；BF16 由 `_narrow_state` 收窄） |
| BF16 staging | 3 × 16 KiB | 输入 tile（q/k/g/v/beta） |
| misc | < 16 KiB | `qn/kn/dec/kd/vf32/outf` 等一维暂存 |

峰值在 256 KiB Vec 容量内，不使用 Cube 资源。

### 3.3 寄存器布局

`_step_dot` 一次持有 4 行 × 2 个半行寄存器（`s0..s3`、`s0h..s3h`）共 8 个，配 2 个
`kd` 寄存器，形成 4 条独立的 reduction 链；`_step_upd_snap` 逐行持有 `s_pre/s_post`、4 个
`kn/dec/qn` 寄存器与 1 个 `u` 广播寄存器。

## 4 接口选择

- 单个 `@pl.jit(auto_mutex=True)` 入口 `kda_snapshot_r{rows}[_bf]`：`batch / seq /
  heads /` 各布局行 stride / `packed` 都是运行时标量或动态轴，`rows` 与 state dtype
  为静态分类（`_bf` 后缀）。
- host wrapper `fused_recurrent_kda` 只做 shape/dtype/device 校验、
  `torch.empty` 分配与单次 launch；worker 数由平台信息取。
- BF16 pool 需要「不截断本轮后续 token 使用的 FP32 工作状态」，因此工作 state 始终
  FP32，收窄只发生在写快照前。

## 5 流水设计

### 5.1 三槽轮转，让行与行重叠

工作 state 常驻 3 槽 UB；每个 token 读一槽写另一槽。快照 tile 与活状态槽解耦后，
MTE3 快照搬运不再与 VF 递推经槽 mutex 互卡。

### 5.2 融合行体拆成独立 pass

初版是在一个 `_step` 里逐行完成 `key 点积 → BRC 广播 → rank-1 更新 → 输出点积`，
每行的 `reduce_sum` 结果直接喂给同一迭代的更新，形成串行链，vector pipe 成为瓶颈。

拆成 `_step_dot`（4 行 unroll，只做点积并紧凑写回）/ `_fold_u`（一条扁平 `sub` 完成
全部行的 delta）/ `_step_upd_snap`（更新 + 输出点积 + 快照写）后，`reduce → 广播 → update` 的
串行依赖被拆开，`_fold_u` 还把每行的 `BRC + vf.full + sub` 三条指令压成一条。

### 5.3 快照解耦

初版用「延迟写」：第 `t` 个 token 的快照 DMA 读轮转状态槽，靠 `slots=3` 给两个 token
周期的余量。改成独立双缓冲快照 tile 后，VF 递推只碰活状态槽、MTE3 只读快照 tile，
两者不再共享 mutex，消除了最后一段 VF/MTE3 耦合。

### 5.4 同步

`auto_mutex` 依 tile group 的 mutex id 自动插入 get_buf/rls_buf。工作 state 槽与快照
tile 使用不同的 mutex id，因此「VF 写 state」与「MTE3 读快照」可以并发。
