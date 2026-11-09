# fused_recurrent_kda_replayssm 设计说明

> 算子本体：`op_kernel/fused_recurrent_kda_replayssm.py`；参考实现与精度判据：
> `tests/assets/`。运行平台：Ascend 950PR（平台信息里 `soc_version = 'DAV_3510'`），
> cube 32 核 / vector 64 核，UB 容量 253952 字节（248 KiB）。

## 1 计算阶段划分

工作项 = 一个 `(batch, head)` 的 `rows` 行 state，按 token 顺序：

| 阶段 | 内容 | 位置 |
|:---|:---|:---|
| 预处理 + key 点积 | Q/K 行 L2 normalize（`1e-6`）、gate/beta sigmoid 链、`d = exp(lower*σ(α(g+dtb)))`、折叠权重 `beta*d*k̂` 与 `d*q̂*scale`、`kq = k̂·(q̂*scale)`；紧接着在同一 VF 调用内逐行 `sk[r] = dot(S_row_r, beta*d*k̂)`、`sq[r] = dot(S_row_r, d*q̂*scale)`，一次 4 行在飞 | VF `_prep_dot` |
| 扁平 delta 与输出 | `u = beta*V - sk`、`o = sq + u*kq`，两条扁平向量指令覆盖全部持有行；同时收窄 FP16 U 记录与 BF16 输出 | VF `_delta_out` |
| 递推更新 | 逐行 rank-1 更新 `S_new = d*S_row + u⊗k̂`，两行 unroll | VF `_step_upd` |
| 记录写回 | `replay_u` 按行块写、`replay_k` / `replay_decay` 由 `rblk == 0` 的工作项写 | MTE3 |
| 输出写回 | `out` BF16 tile 按行块 DMA | MTE3 |

点积权重在 `_prep_dot` 内部以寄存器形式直接消费，不经过 UB：见 §3.3。

## 2 核间切分策略

- 工作项 = `(batch, head, 行块)`，共 `B*N*(128/rows)` 个；按
  `pl.range(block_idx, total_items, block_num)` 轮转，各 worker 计数差不超过 1。
- 启动 worker 数 `min(vector_core_num, 64)`，再截断到工作项数。基准形状
  `B=16, N=6, rows=64` 给出 `192` 个工作项，恰好 3 个/worker；`48/56` 会留下 4 个的
  临界路径（实测慢 20%+）。
- `rows` 由 `B*N` 自适应：`B*N >= 24` 用 64，`>= 6` 用 32，否则用 16。大形状下 64 让
  `_prep_dot` 的每 token 常量开销在 2 倍行数上摊薄；小形状下更细的行块才能填满 64 个
  AIV。
- 三种布局（BSND / BNSD / TND）共用同一 kernel body：布局差异只体现为 host 侧算出的
  行 stride 标量与 `packed` 标志，kernel 内不做布局分支。

## 3 核内切分策略

### 3.1 基本块的选择

`rows ∈ {16, 32, 64}`。`64` 是基准形状的默认；`32/16` 只在 `B*N` 较小、需要更细的
负载粒度时使用。`rows` 只影响 state 行块大小，Q/K/g/beta 始终是完整 128 通道。

### 3.2 核内 Buffer 划分

`rows=64, slots=3` 时：

| Buffer | 大小 | 说明 |
|:---|---:|:---|
| `state_g` | 3 × 32 KiB | 工作 state 槽，token `t` 读槽 `t%3`、写槽 `(t+1)%3` |
| 输入 parity 组 | 2 × ~4 KiB | q/k/g/v/beta 各两份（奇偶 token 交错预取） |
| 每 token 常量/暂存 | ~8 KiB | `kn/dec/kqt/vf32/sk/sq/ug/outf` 等一维 tile |
| 输出 / 记录 staging | ~4 KiB | `outb/ru/rk/rd` |

峰值远小于 248 KiB Vec 容量，不使用 Cube 资源。

### 3.3 寄存器布局与折叠

每个 token 的两组点积权重在 `_prep_dot` 内构造后立即使用：

```
dk  = beta * d * k_hat          (key 点积权重)
dq  = d * q_hat * scale         (输出点积权重)
kq  = k_hat . (q_hat * scale)   (每 token 一次标量 reduction)
```

`dk_pre/post`、`dq_pre/post` 与 4 行 × 2 个半行 state 寄存器同时活跃（约 16 个），
形成 4 条独立 reduction 链。把权重留在寄存器里消掉了原先 `dk/dq` 的 UB 往返
（2 次 store + 4 次 load / token）和一组 buffer 同步。

### 3.4 输出不与更新串在同一行体

输出 `o = scale * S_new q` 展开为 `dot(S_old, d*q̂*scale) + u * (k̂·q̂*scale)`：输出
点积与 key 点积共享同一行 state 的 load，`u` 与 `o` 都是扁平向量指令，更新 pass 内不再
含 reduction。把输出点积放回更新行体（`dot(S_new, q)`）会让更新循环重新出现
`reduce → 依赖` 的串行链，实测慢约 50%。

## 4 接口选择

- 单个 `@pl.jit(auto_mutex=True, tiling_key=ReplayKey)` 入口
  `fused_recurrent_kda_replayssm`：`batch / seq / heads /` 各布局行 stride / `packed`
  都是运行时标量或动态轴，`Rows` / `Slots` / `Unroll` 由 tiling key 选择静态分类。
- host wrapper `fused_recurrent_kda` 只做 shape/dtype/device 校验、常量准备
  （默认 `cu_seqlens` 占位）、`torch.empty` 输出分配与单次 launch。
- `recurrent_state` 只读，因此不需要快照或池索引；`out` 由算子内部新建。

## 5 流水设计

### 5.1 跨 token 预取

每 token 需要 q/k/g/V/beta 五块输入。初版在 token 循环开头加载并立刻消费，MTE2 延迟
完全暴露（消融测得约 1.8 µs）。

现在输入 tile 做成两组奇偶 parity：token `t+1` 的 5 次 load 在 token `t` 计算前发起，
用两对 `set_flag`/`wait_flag` 保证正确性：

- `V → MTE2`：覆盖 parity 前，先等上一轮读该 parity 的向量指令结束；
- `MTE2 → V`：下一轮读该 parity 前，先等 load 完成。

这样 MTE2 与 VF 重叠，且不依赖 loop-carried 的自动同步推断。首 token 的数据在 item
prologue 加载，并用一次 `bar_mte2` 与 VF 分开。

### 5.2 三槽 state 轮转

工作 state 常驻 3 槽 UB；token `t` 读槽 `t%3`、写槽 `(t+1)%3`。state 在初态 load 之后
只被 VF 读写，由向量流水按程序序保证顺序，不需要跨 pipe 同步。初态 load 与首个 token
之间用 `bar_mte2` 分隔。

三槽而不是两槽：两槽时 token `t+2` 会立刻复用 token `t` 的槽，`_step_upd` 的写与下一次
`_prep_dot` 的读只隔一个 token，buffer id 依赖更容易在流水上产生停顿；加深到三槽后
同一槽的写读间隔变为两个 token，实测比两槽快 2.5%–3.5%（三槽与四槽基本持平，取更省
UB 的三槽）。

### 5.3 同步

`auto_mutex` 依 tile group 的 mutex id 自动插入 get_buf/rls_buf。输入 parity 组关闭
自动同步（由 §5.1 的显式 flag 管理），每 token 常量、state 和输出/记录组保留自动
mutex：这些 tile 在相邻 VF 调用之间存在 VST→VLD 依赖，去掉同步实测会出现非确定性
错误。

### 5.4 数值边界

`lower_bound ∈ [-5, 0]` 且 sigmoid < 1，因此 `d_t ∈ (0, 1]`；递推中的乘性衰减不会放大
状态。工作状态始终 FP32，只有导出的 U/K 记录收窄到 FP16、输出收窄到 BF16。
