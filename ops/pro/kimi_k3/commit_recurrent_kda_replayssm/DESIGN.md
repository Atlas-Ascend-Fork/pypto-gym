# commit_recurrent_kda_replayssm 设计说明

> 算子本体：`op_kernel/commit_recurrent_kda_replayssm.py`；参考实现与精度判据：`tests/assets/`。
> 运行平台：Ascend 950PR（平台信息里 `soc_version = 'DAV_3510'`），cube（矩阵计算单元）
> 32 核 / vector（向量计算单元）64 核，UB（片上统一缓冲）容量 256 KiB。

## 1 计算阶段划分

对每个工作项（一个 `(batch, head)` 的 `Rows` 行 state），按 token 顺序：

| 阶段 | 内容 | 位置 |
|:---|:---|:---|
| 记录搬运 | 一次取回整段接受前缀的记录块：`[Cap,128]` decay(FP32) / key(FP16) / u(FP16) | MTE2 |
| 记录宽化 | key、u 由 FP16 转 FP32（保序 `UNPK` + `astype(ZERO)`），每 item 一次 | VF |
| 递推 | 行组寄存器驻留的 `S_row = S_row * decay + u r * key`，token 循环在行循环内部 | VF |
| 写回 | state tile 一次写回 GM | MTE3 |
| 预取 | 下一 item 的 state tile | MTE2 |

每个 state 字节只读一次写一次；token 循环内部没有任何搬运与类型转换。

## 2 核间切分策略

- 工作项 = `(batch, head, 行块)`，共 `B*N*(128/Rows)` 个；按
  `pl.range(block_idx, total_items, block_num)` 轮转分配，各 worker 计数差不超过 1。
- 启动 worker 数 `min(vector_core_num, 64)`。基准形状
  `B=16, N=6, Rows=64` 给出 `192` 个工作项，恰好 `192/64 = 3` 个/worker（`56` 或 `60`
  会留下 4 个的临界路径，实测慢 20%+）。
- 行在递推中相互独立，因此一个 head 可自由切分，不需要跨核通信。

## 3 核内切分策略

### 3.1 基本块的选择

`Rows ∈ {8,16,32,64,128}`。`64` 在每个 token 的常量开销与负载均衡之间最优（见
`RESULTS.md` §3.3 的扫描）；`32` 对极小 `B*N` 给出更多工作项；`8/16` 因每 item 固定
开销占比过高而明显更慢；`128` 只给 `96` 个工作项，负载不均。

### 3.2 核内 Buffer 划分

`rows=64, slots=3, cap=16, rdep=2` 时（字节）：

| Buffer | 大小 | 说明 |
|:---|---:|:---|
| `state_g` | 3 × 32 KiB | 工作 state 槽，3 槽轮转 |
| `dec_g` | 2 × 8 KiB | `[Cap,128]` FP32 decay，双缓冲 |
| `key_g` | 2 × 4 KiB | `[Cap,128]` FP16 key，双缓冲 |
| `k32_g` | 2 × 8 KiB | `[Cap,128]` FP32 key（每 item 宽化一次） |
| `u16_g` | 2 × 2 KiB | `[Cap,Rows]` FP16 u |
| `u32_g` | 2 × 4 KiB | `[Cap,Rows]` FP32 u |

峰值 < 130 KiB，在 256 KiB Vec 容量内。未知 `B/S/N` 不改变物理布局；`Rows < 64` 时
u tile 的物理列宽取 `ceil(Rows/64)*64`，保证 `UNPK` 读宽不越界（有效 shape 由
`set_validshape` 限定为 `Rows`）。

### 3.3 寄存器布局

行组 `GRP = 4`：`a0..a3` / `b0..b3` 共 8 个向量寄存器跨整个 token 循环持有该组的
state 前后半行；另需 `dec`(2) + `k32`(2) + `u` 广播(4) + `full` mask。因此 state 的 UB
往返是「每行每 chunk 一次」而不是「每行每 token 一次」。

## 4 接口选择

- 单个 `@pl.jit(auto_mutex=True, tiling_key=CommitKey)` 入口
  `commit_recurrent_kda_replayssm_kernel`，覆盖全部 `Rows/Slots/Cap/Rdep` 组合。
- `CommitKey` 字段：`Rows ∈ {8,16,32,64,128}`、`Slots ∈ {2,3,4}`、
  `Cap ∈ {8,16,32,48,64}`、`Rdep ∈ {2,3}`。
- host wrapper 只做 shape/dtype/device 校验 + 单次 `kernel[None, workers, route](...)`；
  worker 数由 `get_platform_info().vector_core_num` 取。
- `B/S/N` 通过 `pl.DYNAMIC` 轴 + 运行时标量传入，同一份二进制服务任意形状。

## 5 流水设计

### 5.1 逐 token 记录 → 融合记录块

初版每个 token 单独搬 `decay/key/u`（3 次 MTE2）并在 token 循环内做 FP16→FP32 转换。
改为每个 item 搬一个 `[Cap,128]` 融合块、只宽化一次后，token 循环内零搬运、零转换；
`key` 顺带一次宽化成 FP32，行循环里由 `UNPK+vcvt` 变为单条 `vlds`（配对实测
+2.4%/+4.6%，胜率 68–70%）。

### 5.2 行组寄存器驻留，让行与行重叠

token 循环从「按 token 展开、每 token 扫一遍全部行」改成「按行组展开、组内跨 token
驻留寄存器」。这把每行的 `reduce → 广播 → update` 串行链改成单纯的行内累加，state 的
UB 往返与 token 数无关。

### 5.3 下一 item 的 state 预取

`state_g` 环 + 显式 `state_g[i]` 索引：在第 `k` 个 item 的记录搬运已经进入 MTE2 队列
**之后**再发起第 `k+1` 个 item 的 state 搬运，既不让 32 KiB state 挡住当前 VF 需要的
记录，又能与当前 VF 重叠。`accepted == 0` 的 item 也照常发起预取（否则下一 item 会读到
陈旧 tile —— 这是实测到的一个真实竞态，已修复并加了回归测试）。

### 5.4 同步

`auto_mutex` 依 tile group 的 mutex id 自动插入 get_buf/rls_buf；state 槽 3 个、
记录块 2 个，保证「写→读」「读→写」不会在同一 slot 上相邻排队。
