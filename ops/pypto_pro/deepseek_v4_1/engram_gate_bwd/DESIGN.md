# EngramGateBwd 设计说明（DESIGN）

> 实现：`op_kernel/engram_gate_bwd.py`。整份设计只用一个 `@pl.jit(auto_mutex=True)` kernel
> 服务全部用例（`t`/`dim` 都是运行时动态轴）。
>
> 后文会反复用到几个硬件侧的词，先说明：**MTE2** 是把数据从显存搬到片上（GM→UB）的搬运流水，
> **MTE3** 是反向（UB→GM）的搬运流水，**V** 是向量计算流水（真正做乘加的那个）；
> 三者可以同时进行，一个 kernel 快不快取决于哪条流水最先忙满。
> **VF** 指一次 `pl.vector_function` 调用，也就是向量单元上的一段寄存器级计算。

## 计算阶段划分

一个工作项 = 一个 `(token, head)` 行。它的全部数学在**一次** `engram_gate_bwd_vf` 调用里完成，
中间标量（`beta`、`c·rstd_x²`、`c·rstd_k²`、`gate`）留在寄存器里，不落 UB、不重读。

| 阶段 | 做什么 | 数据 |
| :--- | :--- | :--- |
| S1 | 沿 `dim` 归约求 `dgate = Σ(grad_out · value)` | 读 `grad_out` 行 + `value` 行 |
| S2 | 由 `dgate` 与四个保存量算出 `beta`、`c`、以及 `grad_value` 要用的权重 | 只读寄存器 + `saved` 行 |
| S3 | 算 `grad_value` 属于本 head 的四分之一 | 读 `grad_out` 的四个 head 的四分之一 |
| S4 | 沿 `dim` 扫一遍，算 `grad_x`、`grad_key`，并累加 `grad_weight` | 读 `grad_out`/`x`/`key`/`weight` 行 |

阶段顺序是数学决定的，不能换：

- **S1 必须在 S2 之前**：`beta` 依赖整个 `dim` 的归约结果 `dgate`，归约不完成拿不到 `beta`。
- **S3 必须在 S4 之前**：S4 把结果直接写回搬入 `grad_out` 用的那块 UB 缓冲，
  而 S3 需要的是还没被改写的 `grad_out`（`grad_value = Σ_hc(gate · grad_out)`）。

一个 `(token, head)` 行只需要四个保存量里的第 `head` 列，因此 `saved` 按 `[t, 4*hc]` 打包，
一次 64 B 的搬入就能拿到本 head 需要的 `raw_dot` / `gate` / `rstd_x` / `rstd_k`。

## 核间切分策略

启动 64 个 vector 核（平台信息给出，不写死），按 **16 组 × 4 头** 二维切：

```text
head = c % 4            该核固定负责的 head，整个 launch 不变
grp  = c // 4           组号，0..15
```

- **每个核负责一个 head 的全部 `dim` 宽度**，所以该 head 的 `grad_x` / `grad_key` 整行只由一个核写，
  不需要任何行内归约或多核拼接。
- **token 按连续切片归属**：`nq = t // 16`，`nr = t - nq*16`；核 `grp` 拿到从
  `base = grp*nq + nr` 开始的 `nq` 个 token（前 `nr` 个组各多拿一个），覆盖恰好不重不漏。

token 归属方式的选择：**不采用跨步归属**。跨步归属下每个核固定拿 `t = grp + i*ng`，步长是 `16`；
只要 `image_mask` 的重复周期能整除这个步长（按固定间隔标记时，周期 2、周期 4 均可整除 `16`），
每个核就会只拿到同一类 token——有的核全是 image token、有的核全是真实 token。
此时整 kernel 的耗时由最慢的核决定，逐 token 短路（见下文「image token 的短路」）节省的时间落在快核上，
不会反映到总耗时中。连续切片不受这一模式影响：每个核拿到的 image / 真实 token 比例与全局一致。

`grad_weight` 是跨 token 的累加：每个核维护自己 `(grp, head)` 的 FP32 累加器，最后写
`grad_w_partial[grp, head, :]`（16×4 份）。框架没有跨核原子加，这 16 份由 host 侧求和合并成
`[hc, dim]`，代价是每调用多一次 `aclnnReduceSum`（见 RESULTS.md 的性能）。

## 核内切分策略、基本块选择与核内 Buffer 划分

**基本块。**

- **一次处理 2 个工作项**（同核相邻的两个 token，`t0` / `t1`），两者的搬入**全部先发出**、再分别计算。
  这样第 2 个 token 的 MTE2 会压在第 1 个 token 的向量计算后面，把访存延迟藏起来。
- 行宽固定为 `MAXC = 7168`（编译期常量，等于最大 `dim`）：所有 buffer 都按最大形状分配，
  小 `dim` 用例通过 `pl.set_validshape` 只处理有效部分，因此**一份二进制服务全部用例**，
  不需要按形状重新编译。
- VF 内部沿 `dim` 以 **128 元素**为一步（`STEP = 128`）遍历：fp32 寄存器是 64 通道，
  128 元素正好一次拿到偶/奇两个平面，配合 `pl.LoadDist.DINTLV_B32` 与 `pl.StoreDist.INTLV_B32`
  在寄存器边界完成 bf16↔fp32 转换，**不经过额外的 fp32 UB 缓冲**。

**Buffer 划分。**

UB 总量 253952 B，实际使用 **209920 B**（82.7%）。需要搬入搬出的 per-token buffer 都开
**2 槽轮转**（「基本块」里一次处理两项需要），常驻的 `w` / `gw` 与只读的零源 `zb` 各 1 槽。

| buffer | 类型 / 形状 | 槽数 | 字节 | 作用 |
| :--- | :--- | ---: | ---: | :--- |
| `go_b` | BF16 `[1, 7168]` | 2 | 28672 | 搬入 `grad_out` 行；S4 后该 buffer 里就是 `grad_x` |
| `x_b` | BF16 `[1, 7168]` | 2 | 28672 | 搬入 `x` 行 |
| `k_b` | BF16 `[1, 7168]` | 2 | 28672 | 搬入 `key` 行；S4 后该 buffer 里就是 `grad_key` |
| `v_b` | BF16 `[1, 7168]` | 2 | 28672 | 搬入 `value` 行；S3 后该 buffer 里就是本 head 的 `grad_value` 四分之一 |
| `og0`/`og1`/`og2` | BF16 `[1, 1792]` | 各 2 | 3×7168 | `grad_out` 另外三个 head 在本 v 片上的四分之一 |
| `w` | FP32 `[1, 7168]` | 1 | 28672 | 本 head 的 `weight` 行，常驻 |
| `gw` | FP32 `[1, 7168]` | 1 | 28672 | `grad_weight` 累加器，常驻 |
| `zb` | BF16 `[1, 7168]` | 1 | 14336 | image token 写 0 用的零源 |
| `s42` | FP32 `[1, 256]` | 2 | 2048 | 本 token 的四个保存量 |

两点说明：

- **`go_b` 与 `k_b` 是原地复用**的：S4 把结果直接写回搬入用的同一块 UB，再整行搬出到 `grad_x` / `grad_key`，
  省掉两块同宽的目的 buffer，也少一次 UB 读写。
- **`og0..og2` 只放"别的 head 在本核 v 片上的那四分之一"**，宽度是 `MAXC/4 = 1792`，不是整行。
  `grad_value` 按 head 拆成 4 份、每份由一个核算出，所以本核只需要别的 head 在同一 v 片上的数据。

## 接口选择

| 选择 | 原因 |
| :--- | :--- |
| 反向把 `rstd_x`/`rstd_k`/`raw_dot`/`gate` 作为**输入** | 正向已经算过并保存（`torch.autograd` 的保存张量约定）。反向重算它们要多扫一遍 `x`/`key`/`weight`，是纯粹的重复开销。 |
| `key` 与 `value` 分成两个张量传入 | 两者形状不同（`[t,hc,dim]` 与 `[t,dim]`），各自按原布局直读，host 侧不需要任何额外拷贝。若强行合并成一个张量，host 侧要多做 `t*(hc+1)*dim*2` 字节的拷贝——`t=8001, dim=2048` 即 164 MB，代价远超算子本身。 |
| `image_mask` 用 BOOL `[t]` | 与正向接口保持一致；kernel 内逐 token 读一个 bit 决定走哪条路径。 |
| `grad_weight` 在 host 侧合并 16 份 partial | 框架没有跨核原子加。16 份 partial 是 `[16, hc, dim]` 的 FP32，合并只读 16 行，开销可忽略。 |
| `@pl.jit(auto_mutex=True)` + `make_tile_group` | 让框架按 buffer 槽位自动插入 MTE2→V→MTE3 的顺序，kernel 里不写显式屏障，也就不容易写出漏屏障的静默错误。 |
| kernel 内**只发一个 kernel**，host 只做校验/分配/launch | 避免多 kernel 串联带来的额外下发开销；`1/sqrt(dim)` 与 `1/dim` 也在 kernel 内由 `dim` 推出，host 只传形状。 |
| `pass_options={"enable_slice": False}` | 与同族的 pypto-pro 正向算子 `engram_gate` 保持一致（该算子同样用这个 launch 选项）。 |

## 流水设计

**一轮两条工作项。**

```text
for i in range(n_pair):                 # n_pair = nitems // 2
    t0, t1 = base + 2i, base + 2i + 1
    ── 搬入 ──
    搬 grad_out[t0]、grad_out[t1]                                     （无条件）
    if 非 image：搬 t0 的 saved / value / x / key / 三个 og
    if 非 image：搬 t1 的 saved / value / x / key / 三个 og
    ── 计算 ──
    if 非 image：对 t0 跑一次 VF，写回 grad_x / grad_key / grad_value
    else       ：grad_out→grad_x，零→grad_key / grad_value
    if 非 image：对 t1 跑一次 VF，写回 grad_x / grad_key / grad_value
    else       ：同上
```

关键点：**两个 token 的搬入都排在计算之前**。这样第 2 个 token 的搬入在第 1 个 token 计算期间
仍然在飞，访存与计算重叠；如果按「搬一个、算一个」写，每一轮都要等搬入落地，两者无法重叠。

余下的 `nitems % 2`（0 或 1）个 token 由尾部单条循环处理，结构与上面单项分支一致。

**image token 的短路。**

正向对 image token 直接返回 `x`，反向里这些行是常数解：`grad_x = grad_out`、`grad_key = grad_value = 0`。
所以 image token **只搬 `grad_out` 一行**，然后写 `grad_out → grad_x` 与 `0 → grad_key`、`0 → grad_value`，
`value`/`x`/`key`/`saved`/`og` 一行都不搬、VF 也不跑。零源用常驻的 `zb`，避免每次现算。

因为「核间切分策略」里的连续切片保证了每个核拿到的 image 比例与全局一致，这部分省下来的时间能真正反映到总耗时上。

**依赖链。**

```text
        ┌── grad_out[t] ──┐
MTE2 ───┼── value[t] ─────┼──► V: S1(dgate) ──► S2(beta) ──► S3(grad_value)
        └── x/key/... ─────┘                                  └──► S4(grad_x, grad_key, grad_weight)
MTE3 ◄──── grad_x / grad_key / grad_value（整行搬出）
```

- S1 是唯一一处跨整个 `dim` 的归约，它是这条链上最长的等待；它读的 `grad_out` 与 `value` 都在本轮
  已经搬入，不需要额外访存。
- `gw` 常驻 UB，S4 每步读改写一次，整个 launch 期间只在最后写回一次 GM（写 `grad_w_partial`）。
- `w` 常驻 UB，整个 launch 只搬一次。
