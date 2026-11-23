# kda_cp_main 设计说明

Kimi Delta Attention CP prefill 后半段主算子（残差更新 + 逐块状态递推 + 块内
attention 输出）的 PyPTO-Pro 实现。接口与数值契约见 [README.md](README.md)，
实测数据见 [RESULTS.md](RESULTS.md)。

## 1 计算阶段划分

序列按 `CHUNK = 64` 分块；每个 (batch, head, dv-half) item 沿块序串行递推。
第 c 块按数据依赖切成三个阶段，跨越 cube / vector 两侧：

| 阶段 | 执行侧 | 输入 | 输出 |
| :--- | :--- | :--- | :--- |
| Δ 链 | cube | `w`、`u`、`S_{c-1}`（L1）、恒等矩阵 | `Δ = w·S^T + I@u`（FIX 在途 bf16 截断落 L1） |
| O / T | cube | `qg`、`mqk`、`Δ`、`kg` | `O = qg·S^T + Mqk·Δ`（FIX→UB）；`T = Δ^T·kg`（FIX→UB） |
| S 更新与发布 | vector | `S_{c-1}`（UB 常驻）、`decay`、`T` | `S_c = S·decay + T`（f32 原地）；bf16 快照发布到 L1 |

- Δ 链完全在 cube 侧闭合：`+u` 用恒等矩阵乘 `I@u` 摄入 cube 累加器，`Δ` 经
  FIX 通道 bf16 在途截断直接落 L1（NZ/ZN 双拼写视图），不经向量侧往返，
  每块只有 2 次跨核事件跳变（T 下发 / S 发布）。
- `O` 的 GM 写回不在 cube 关键环上：acc_o 经 `DualModeSplitM` 落 UB 后由
  向量侧异步 cast + 写 GM，与下一块的 cube 计算重叠（见 §5）。
- 数值契约的三处截断：`S` 每块起点 bf16 截断作 matmul 操作数；`Δ` fp32 求和后
  截断一次；`O` / `final_state` 按 fp32 累加输出。与独立 CPU golden 一致。

## 2 核间切分策略

任务粒度 = (batch, head, dv_idx)，item 之间无任何数据交换（状态切片独立，
无 GM workspace、无跨 item 通信），因此 item 可以任意映射到核：

```text
core c 处理 item = c, c + nblocks, c + 2·nblocks, ...
```

- dv 二分（`DVH = 64`）：`2·B·Nv` 个 item，每 item 持有 `S[64, 128]` 半区。
  单一变体，host 起核数 `min(32, 2·B·Nv)`，item 数不足核数时多余核不启用。
- item 数超过核数时 item 跨步分配，任务边界经奇偶槽位地址隔离。

## 3 核内切分策略、基本块选择与核内 Buffer 划分

### 3.1 基本块与子块

- token 块固定 `[64, 128]`（CHUNK × D）；`S` 切片 `[64, 128]`。
- 每个 AIV 子块负责 dk 半区（`col0 = 64·sub_id`）与 O 的 32 行窗口
  （`row0 = 32·sub_id`），两 AIV 对同一 item 分工不重叠。
- 状态更新 VF 以 128 元素为寄存器粒度，`S·decay + T` 与 bf16 快照发射在同遍
  完成（乘加经 `vmadd` 融合指令实现），每块每 AIV 一次 VF 调用。

### 3.2 地址表（每核）

L1（512KB）：

| tile | 形状 | 布局 | addr | 槽 |
| :--- | :--- | :--- | :--- | ---: |
| w / qg / kg | [64,128]bf16 | NZ | 0x00000 / 0x10000 / 0x20000 | 各 4 |
| mqk | [64,64]bf16 | NZ | 0x30000 | 4 |
| S（NZ + ZN 双拼写） | [64,128]bf16 | NZ/ZN | 0x38000 | 2 |
| u | [64,64]bf16 | NZ | 0x40000 | 4 |
| Δ（NZ + ZN 双拼写） | [64,64]bf16 | NZ/ZN | 0x48000 | 2 |
| 恒等矩阵 | [64,64]bf16 | NZ | 0x4C000 | 1（每核装载一次，跨 item 复用） |

L0A（64KB）：`la_w`/`la_q` [64,128] @0x0/0x4000；`la_mqk` [64,64] @0x8000；
`la_dt` [64,64] @0xA000；`la_i` [64,64] @0xC000。
L0B（64KB）：`lb_s` [128,64] @0x0；`lb_kg` [64,128] @0x4000；
`lb_u`/`lb_d` [64,64] @0x8000/0xA000。
L0C（256KB）：`acc_d`/`acc_o` [64,64]f32 @0x0/0x4000；`acc_t` [64,128]f32 @0x8000。

UB（每 AIV）：

| tile | 形状 | addr | 说明 |
| :--- | :--- | :--- | :--- |
| decay | [1,64]f32 | 0x00000（双槽） | 本块衰减行 |
| s_state | [64,64]f32 | 0x01000 / 0x05000 | item 状态 f32 常驻（奇偶任务槽） |
| sd_ub | [64,64]f32 | 0x09000（双槽） | `T` 的 SplitN 落点 |
| ws_ub | [32,64]f32 | 0x12000（双槽） | `O` 的 SplitM 落点 |
| u_nd | [32,64]bf16 | 0x16000（双槽） | O 排水 cast |
| s_nd / s_nz | [64,64]bf16 | 0x18000 / 0x1A000 | S 快照自然序 → NZ 中转 |

### 3.3 事件协议（INTRA_BLOCK）

| 事件 | 方向 | 含义 |
| :--- | :--- | :--- |
| `EV_SD` | FIX(set) → V(wait) | `T` 已落 UB，向量侧可更新 S |
| `EV_S` | MTE3(set) → MTE1(wait) | `S` 已插入 L1，cube 可消费 |
| `EV_WS` | FIX(set) → V(wait) | `O` 已落 UB，向量侧可排水 |

三者均按 chunk 奇偶分槽对（成对 id 0–15），避免相邻块复用同一事件号。
`O` 排水（`ws_ub` 读 + cast + GM 写）在 S 发布之后执行，消费早已置位的
`EV_WS`，不占用环上的等待。

### 3.4 尾块

`valid = min(64, end - t0)`：w/u 的无效行经 `expands` 显式清零后进入递推
（Δ 的零尾使 kg 的裁剪尾行无害）；`O` 的尾行经 `set_validshape` 窗口化排水；
`vr == 0` 的 AIV 跳过排水。Acc 不设 validshape（满窗口），垃圾行只出现在
被裁剪/清零的位置。

## 4 接口选择

- 单个模块级 `@pl.jit(auto_mutex=True)` 入口（`kda_cp_main_kernel`），
  host 侧 `flash_kda_cp_main` 完成 input 校验后一次启动，无 if/else 分派。
- token 张量经 `make_tensor` 以 rank-erased 视图统一为 `[B, Nv, S, D]`
  逻辑形状 + 运行时 stride（TND 用打包 stride 与 `cu_seqlens`），BSND 仅靠
  输出 stride 表达布局差异，无 host 侧 transpose/pad/cast。
- `transition` / `state_ext` / `metadata` 为接口兼容参数，仅校验不消费。
- 输出经 `torch.empty` 分配；校验失败一律 `ValueError`。

## 5 流水设计

### 5.1 2 跳串行环

块内依赖决定每 chunk 的环形状：`S_c` 依赖 `T_c`（cube 块尾产出），
chunk `c+1` 的全部 matmul 依赖 `S_c`。因此环周期为

```text
cube(c)：stage → Δ 链 → O matmul → T → [EV_SD]
vector(c)：S 更新 → S 发布 → [EV_S]
cube(c+1)：...
```

环上不可再分的串行段只有 Δ 链与 S 更新/发布；其余工作全部搬进等待窗或
对侧：

- **L0 staging 前置**：chunk c 的 5 个 L1→L0 搬移在 `EV_S` 等待之前 issue，
  排空与向量侧段重叠；
- **O 排水移出环**：`O` 只在 cube 环上保留 FIX→UB + 事件，cast 与 GM 写在
  向量侧执行，与下一块的 cube 计算重叠；
- **GM→L1 预取**：chunk c 尾部 issue 下块 5 个操作数的 GM 读（奇偶槽隔离）；
- **跨 item 预取**：item 最后一块的尾部预取下一 item 的初始状态（含 cast 与
  L1 发布），任务边界无冷启动气泡；恒等矩阵每核装载一次跨 item 复用。

### 5.2 同步层次

| 层次 | 机制 | 覆盖 |
| :--- | :--- | :--- |
| DMA ↔ VF / matmul | `auto_mutex` tile 组互斥 | 搬入→消费、生产→搬出、槽复用 |
| VF 内 RAW | `vf.mem_bar(VST_VLD)` | S 更新前后同址读写的顺序 |
| cube ↔ vector | `EV_SD` / `EV_S` / `EV_WS` 事件对 | T 下发、S 发布、O 排水 |

同一 tile 的跨 pipe 访问由 auto_mutex 插入事件；VF 内同址写后读（`s_state`
原地更新与快照）显式 `mem_bar` 排序。尾块与多 item 的槽位复用经奇偶地址
隔离，无别名。
