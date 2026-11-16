# KDA

Ascend 950 上的 PyPTO-Pro KDA 正向、反向、自动求导与仿射摘要 CP 实现。
目录遵循相邻算子 `quant_mla_prolog` 的组织：实现放在 `op_kernel/`，
测试和 CPU 参考放在 `tests/`，接口和设计分别记录在 README、DESIGN。

```text
flash_kda/
├── README.md
├── DESIGN.md
├── op_kernel/
│   ├── __init__.py         # 公开入口
│   ├── flash_kda.py        # 正向两个 stage
│   ├── flash_kda_bwd.py    # 反向两个 stage
│   ├── chunk_kda.py        # 自动求导与 head bucket 调度
│   └── kda_cp.py           # CP 元数据、摘要通信与参数梯度归约
└── tests/
    ├── test_flash_kda.py
    └── assets/
        ├── kda_golden_cpu.py
        └── kda_precision.py
```

[设计说明](DESIGN.md)描述计算与通信流程。
以下四节保留完整 wrapper 接口定义：

- [正向 Stage1](#flash_kda_stage1_wrapper)
- [正向 Stage2](#flash_kda_stage2_wrapper)
- [反向 Stage1](#flash_kda_bwd_stage1_wrapper)
- [反向 Stage2 与自动求导](#flash_kda_bwd_stage2_wrapper)

## 导入与迁移

配置好 CANN、PyPTO-Pro 和 torch_npu 后，从仓库根目录使用源码包：

```python
from ops.pypto_pro.kimi_k3.flash_kda.op_kernel import chunk_kda, KDACPContext
```

原 `from ops.pypto_pro.kimi_k3 import ...` 由包级转发保留兼容，指向同一组对象。
直接引用旧子模块的代码需把 `ops.pypto_pro.kimi_k3.<模块>` 改为
`ops.pypto_pro.kimi_k3.flash_kda.op_kernel.<模块>`。将 `op_kernel/` 加入 Python 搜索路径后，
也可直接使用 `from chunk_kda import chunk_kda`。这里描述的是源码入口，不承诺安装 wheel 后可用。

原四份 `flash_kda_*_wrapper.md` 已合并到本文，原测试入口迁至算子目录。

## 运行测试

以下命令从仓库根目录执行，单次超时不超过 280 秒。本算子测试位于 `ops/` 下，
需要显式传入路径；仓库默认 `testpaths=tests/ops` 不会自动收集它。
先列出参数 ID，再逐项串行运行，避免将全部编译时间叠加到一次 280 秒调用中。

```bash
timeout --kill-after=3s 280s python -m pytest --collect-only -q \
  ops/pypto_pro/kimi_k3/flash_kda/tests/test_flash_kda.py
timeout --kill-after=3s 280s python -m pytest -q -s \
  ops/pypto_pro/kimi_k3/flash_kda/tests/test_flash_kda.py::test_wrappers_threeway
timeout --kill-after=3s 280s python -m pytest -q -s \
  'ops/pypto_pro/kimi_k3/flash_kda/tests/test_flash_kda.py::test_autograd[dense]'
```

使用 `TILE_FWK_DEVICE_ID` 选择 NPU；测试生成文件写入 pytest 临时目录。
[CPU 参考](tests/assets/kda_golden_cpu.py)与[精度判据](tests/assets/kda_precision.py)独立于设备实现。
本次目录迁移不改变算子数值、接口参数或测试阈值。

<a id="flash_kda_stage1_wrapper"></a>

## flash_kda_stage1_wrapper

本目录的 KDA 正向由两个 wrapper 组成。本接口在一个 kernel 内完成 L2Norm、gate 激活、beta sigmoid、chunk 预计算和局部仿射摘要；[正向 Stage2](#flash_kda_stage2_wrapper) 合并前缀摘要并输出结果和训练缓存。

入口来自 `ops.pypto_pro.kimi_k3.flash_kda.op_kernel`，实现见 [flash_kda.py](op_kernel/flash_kda.py)。以下说明对应本目录的摘要 CP 接口，不能混用历史顺序传递 final-state 的接口。

<a id="flash_kda_stage1_wrapper-函数原型"></a>

### 函数原型

```python
def flash_kda_stage1_wrapper(
    q, k, v, g, beta, scale, a_log, dt_bias, lower_bound,
    sequence_initial_state=None, *, cp_meta, layout_qkv='BSND',
    is_training=True, num_cores=None, allow_neg_eigval=False,
):
    ...
```

返回 `(prepared, summary)`。

<a id="flash_kda_stage1_wrapper-维度与支持范围"></a>

### 维度与支持范围

所有计算 Tensor 均位于同一 NPU。`B/S/Nk/Nv` 为本 rank、本次 head bucket 的维度；`Nk` 为 Q/K head 数，`Nv` 为 V/g/beta head 数。端到端正反向的共同范围为：

- `1 <= B <= 64`，`1 <= Nk <= Nv <= 96`，`Nv % Nk == 0`，`K=V=D=128`。
- chunk size 固定为 `64`；状态固定为 V-first，内层顺序为 `[V,K]`。
- dense 本地 `1 <= S <= 131072`，`NC=ceil(S/64)`，`P=64*NC`。
- packed 使用 `TND`：`T` 个 token，`M` 个非空本地片段，长度为 `L_i`；`J=sum(ceil(L_i/64))`，`P=64*J <= 131072`。缓存将这些 chunk 按片段顺序紧密排列，内部 `B=1`、`NC=J`。
- `lower_bound` 是有限标量，范围 `[-5,0]`；`scale` 是有限标量，通常取 `128**-0.5`，底层 wrapper 没有默认值。
- L2Norm、gate 激活、beta sigmoid 固定启用；`allow_neg_eigval` 只能为 `False`，即 beta sigmoid 不乘 2。

以上是接口支持域，不表示所有尺寸都已完成性能或精度验收。

<a id="flash_kda_stage1_wrapper-输入"></a>

### 输入

| 输入 | dtype | BSND | BNSD | TND |
|---|---|---|---|---|
| `q`, `k` | BF16 | `[B,S,Nk,128]` | `[B,Nk,S,128]` | `[T,Nk,128]` |
| `v`, `g` | BF16 | `[B,S,Nv,128]` | `[B,Nv,S,128]` | `[T,Nv,128]` |
| `beta` | BF16 | `[B,S,Nv]` | `[B,Nv,S]` | `[T,Nv]` |
| `a_log` | FP32 | `[Nv]` | 同左 | 同左 |
| `dt_bias` | FP32 | `[Nv,128]` | 同左 | 同左 |
| `sequence_initial_state` | FP32，可为 `None` | `[B,Nv,128,128]` | 同左 | `[M,Nv,128,128]` |

`q/k` 是归一化前的原始输入；`g/beta` 是激活前的原始输入。gate 使用 `lower_bound * sigmoid(exp(a_log) * (g + dt_bias))`。GVA 将一个 key head 对应到连续的 `Nv/Nk` 个 value heads。

raw 输入支持 row-major 存储上的非重叠切片：stride 为正、最后一维连续、外层 stride 满足 row-major 非重叠要求；不自动做 `.contiguous()`。`a_log/dt_bias` 必须连续。初始状态的内层矩阵必须连续，stride 为 `(128,1)`。

`sequence_initial_state=None` 表示逻辑序列起点使用零状态。延续前一 rank 的片段由 Stage2 通过摘要恢复起始状态，不把前一 rank 的 final-state 作为这个输入传递。packed 中初始状态逐片段提供，仅逻辑起点对应的项被使用。

`is_training=True` 保存反向所需缓存；`False` 不保存训练缓存，仍执行相同的前向输出计算。`num_cores=None` 使用设备可用 Cube 核数，再受任务数限制；dense 实际核数最多 `B*Nk`，packed 最多 `Nk`。显式值应为当前设备支持范围内的正整数。

<a id="flash_kda_stage1_wrapper-meta-与-cp_meta"></a>

### `meta` 与 `cp_meta`

`prepared['meta']` 是 wrapper 自动生成的**算子元数据**：布局、尺寸、chunk 数、head bucket、scale、lower_bound、缓存版本等。调用方保留它，不自行修改。

`cp_meta` 是调用方提供的**CP 分区元数据**，不包含通信 group，也不执行通信：

| 字段 | 类型 / 含义 |
|---|---|
| `cp_rank`, `cp_size` | CP group 内的 rank 和大小；`0 <= cp_rank < cp_size` |
| `first_continues` | 本 rank 首片段是否延续前一 rank；dense 缺省为 `cp_rank>0`，packed 必填 |
| `last_continues` | 本 rank 尾片段是否延续到后一 rank；dense 缺省为 `cp_rank+1<cp_size`，packed 必填 |
| `global_Nk`, `global_Nv` | 可选，全 head 数；缺省为本次 `Nk/Nv` |
| `hk_start`, `hv_start` | 可选，当前 bucket 在全 head 中的起点；缺省为 0；必须满足 `hv_start=hk_start*(Nv/Nk)` |

bucket 必须包含完整、连续的 key/value-head 组，并保持全局 GVA 比例。dense 的延续标志对整个本地 batch 生效。跨 rank 延续边界必须位于逻辑 chunk 边界：dense 的 `S` 或 packed 最后片段的长度在 `last_continues=True` 时须为 64 的倍数。rank 0 不能向前延续，最后 rank 不能向后延续；相邻 rank 的标志和逻辑序列 ID 必须一致。

TND 额外需要以下字段。所有索引 Tensor 都必须是输入设备上的连续 INT64；偏移从 0 开始：

| 字段 | shape / 含义 |
|---|---|
| `sequence_lengths` | CPU tuple/list，长度 `M`；正数且总和为 `T` |
| `sequence_ids` | CPU 整数序列，长度 `M`；供 CP adapter 校验逻辑序列延续关系 |
| `cu_seqlens` | `[M+1]`，本地原始 token 的片段边界 |
| `chunk_starts` | `[J]`，每个 chunk 在本地 raw token 中的起点 |
| `chunk_lengths` | `[J]`，有效 token 数，范围 `[1,64]` |
| `chunk_offsets` | `[M+1]`，每个片段在 chunk 缓存中的起止偏移 |
| `h_offsets` | `[M+1]`，`h_offsets[i]=chunk_offsets[i]+i`，为每个片段额外预留一个状态 |
| `chunk_indices` | `[J,2]`，每个 chunk 的 `(片段序号,片段内 chunk 序号)` |

建议由 [kda_cp.py](op_kernel/kda_cp.py) 的 `prepare_cp_metadata` 生成这些字段，或直接使用 [自动求导入口](#flash_kda_bwd_stage2_wrapper-自动求导入口)。底层 wrapper 不替代 adapter 的跨 rank 一致性检查。四个 wrapper 应使用同一份逻辑 `cp_meta`，并保留 packed 索引 Tensor 的对象引用。

例如 `[65,127,1]` 三个片段对应 `cu_seqlens=[0,65,192,193]`，`chunk_offsets=[0,2,4,5]`，`h_offsets=[0,3,6,8]`，`J=5`、`P=320`。

<a id="flash_kda_stage1_wrapper-输出-prepared"></a>

### 输出 `prepared`

`prepared` 是进程内字典，必须整体交给 Stage2。下表以 dense 为主；packed 将 `B=1,NC=J`，内部 prepared 布局固定为 BSND。

| 字段 | dtype | shape / 用途 |
|---|---|---|
| `qd`, `kr`, `w` | BF16 | BSND `[B,P,Nv,128]`，BNSD `[B,Nv,P,128]`；状态递推及输出计算的预计算量 |
| `gam` | FP32 | `[B,Nv,NC,128]`，chunk 状态衰减系数 |
| `mqk` | BF16 | `[B,Nv,NC,64,64]`，chunk 内输出系数 |
| `upre` | FP32 | `[B,Nv,128,P]`，状态修正前的 value 中间量 |
| `q_c/k_c/v_c/gk_c/beta_c/akk` | 见缓存表 | 训练时保存；推理时对应值为 `None` |
| `qk_rstd` | FP32，可选 | `[B,NC,Nk,2,64]`，Q/K reciprocal norm；当前仅完整 dense chunk 训练路径生成 |
| `meta`, `cp_meta` | 字典 | 算子及 CP 元数据 |
| `sequence_initial_state` | FP32 Tensor 或 `None` | 保留调用方状态引用 |

训练字段的精确 shape 见 [Stage2 缓存定义](#flash_kda_stage2_wrapper-训练缓存-train_cache)。

<a id="flash_kda_stage1_wrapper-输出-summary-与通信"></a>

### 输出 `summary` 与通信

`summary` 是连续 FP32 Tensor：dense `[B,Nv,2,128,128]`，packed `[1,Nv,2,128,128]`。slot 0 为 `E`，slot 1 为 `T`，描述 V-first 状态的仿射关系 `H_out = H_in @ T + E`；实际计算包含 BF16 操作数舍入，不承诺浮点逐位满足代数等式。

逻辑起点的初始状态贡献已包含在 `E` 中，对前序状态的 `T` 置零。packed 的 packet 描述**最后一个本地片段**；当该片段不延续前一 rank 时，`T` 同样置零。

CP>1 时，调用方按 **group rank 顺序**对 packet 做 AllGather，得到 `[CP*B,Nv,2,128,128]`；packed 为 `[CP,Nv,2,128,128]`。仅摘要参与通信，`prepared` 和训练缓存保留在本 rank。CP=1 时直接传原 packet 即可。具体调用顺序见 [正反向直调示例](#flash_kda_bwd_stage2_wrapper-单卡四个-wrapper-直调示例)。

<a id="flash_kda_stage2_wrapper"></a>

## flash_kda_stage2_wrapper

本接口接收 [Stage1](#flash_kda_stage1_wrapper) 的本地预计算结果和 AllGather 后的正向摘要，在一个 kernel 内完成前缀合并、真实状态递推、输出和训练缓存写回。实现见 [flash_kda.py](op_kernel/flash_kda.py)。

<a id="flash_kda_stage2_wrapper-函数原型"></a>

### 函数原型

```python
def flash_kda_stage2_wrapper(
    prepared, all_fwd_summary, *, cp_meta, num_cores=None,
    out=None, final_state_out=None,
):
    ...
```

返回 `(out, final_state, train_cache)`。

<a id="flash_kda_stage2_wrapper-输入与输出"></a>

### 输入与输出

| 参数 | 约定 |
|---|---|
| `prepared` | 当前 rank、当前 head bucket 的 Stage1 原始返回字典 |
| `all_fwd_summary` | 同设备、连续 FP32；dense `[CP*B,Nv,2,128,128]`，packed `[CP,Nv,2,128,128]`；按 group rank 顺序拼接，不能按其他 rank 编号排序 |
| `cp_meta` | 与 Stage1 相同的 CP 信息；packed 索引 Tensor 引用也必须保持一致 |
| `num_cores` | 与正向 Stage1 相同的核数规则；不要求固定 36 核 |
| `out` | 可选 BF16 输出 buffer，shape/layout 与 raw `v` 相同 |
| `final_state_out` | 可选 FP32 V-first 状态 buffer；dense `[B,Nv,128,128]`，packed `[M,Nv,128,128]` |

不传输出 buffer 时由 wrapper 分配；传入时返回同一个对象。支持合法 row-major 非重叠切片，最后一维连续；状态内层 `[V,K]` 必须连续。输出 buffer 不得与 prepared 中仍在使用的 Tensor 或另一个输出的存储范围重叠。

`final_state` 始终返回：dense 为每个本地 batch 的最终状态，packed 为每个本地片段的最终状态，均为 FP32。它不一定是完整跨 rank 逻辑序列的最终状态。`is_training=False` 时 `train_cache=None`。

若本 rank 延续前序 rank，Stage2 按前序 group rank 顺序执行 `H = MM16(H,T_rank) + E_rank`，恢复入站状态。否则从本地逻辑起点的 initial state（或零）开始。这里 `MM16` 表示 BF16 操作数、FP32 累加的矩阵乘；摘要自身存储为 FP32。packed 仅首个延续片段使用前缀，后续片段从各自初始状态开始。

<a id="flash_kda_stage2_wrapper-训练缓存-train_cache"></a>

### 训练缓存 `train_cache`

缓存版本固定为 `flash_kda_cp_train_v3_bf16_state`；必须与本目录反向成对使用。旧 FP32 H 缓存版本会被反向拒绝。

下表中的 `NC=ceil(S/64)`；packed 使用 `B=1,NC=J,P=64*J`，`J` 为所有本地片段的 chunk 数之和。

| 字段 | dtype | dense shape | packed shape |
|---|---|---|---|
| `q_c`, `k_c` | BF16 | `[B,NC,Nk,64,128]` | `[1,J,Nk,64,128]` |
| `v_c` | BF16 | `[B,NC,Nv,64,128]` | `[1,J,Nv,64,128]` |
| `gk_c` | FP32 | `[B,NC,Nv,64,128]` | `[1,J,Nv,64,128]` |
| `beta_c` | FP32 | `[B,NC,Nv,64]` | `[1,J,Nv,64]` |
| `akk` | BF16 | `[B,NC,Nv,64,64]` | `[1,J,Nv,64,64]` |
| `h` | BF16 | `[B,NC+1,Nv,128,128]` | `[1,J+M,Nv,128,128]` |
| `v_new` | BF16 | `[B,NC,Nv,128,64]` | `[1,J,Nv,128,64]` |
| `mqk` | BF16 | `[B,Nv,NC,64,64]` | `[1,Nv,J,64,64]` |
| `w`, `kr` | BF16 | BSND `[B,P,Nv,128]` 或 BNSD `[B,Nv,P,128]` | `[1,P,Nv,128]` |
| `gam` | FP32 | `[B,Nv,NC,128]` | `[1,Nv,J,128]` |
| `qk_rstd`（可选） | FP32 | `[B,NC,Nk,2,64]` | 当前不生成 |
| `meta`, `cp_meta` | 字典 | 保留 Stage1 元数据 | 同左 |

`q_c/k_c` 是归一化后的 Q/K，`gk_c` 是 chunk 内累计 gate，`beta_c` 是 sigmoid 后的 beta。`akk` 是反向需要的三角逆矩阵缓存，dtype 为 BF16。`v_new` 是状态修正后的 value，**物理布局为 `[V,C]`**，不是 `[C,V]`。

`h` 为 **V-first `[V,K]`** 状态边界缓存。dense 的第 0 项是第一个 chunk 的入站状态，第 `c+1` 项是 chunk `c` 后的状态。packed 的片段 `i` 使用 `h_offsets[i]` 至 `h_offsets[i+1]-1`，各片段都有独立的起始状态和终态项。

`h` 按 BF16 保存，外部 `final_state` 和状态递推仍为 FP32；不能用 `h` 的最后一项替代 FP32 `final_state` 并声称数值逐位一致。

所有缓存 Tensor 均为 wrapper 定义的连续物理布局；不要自行转置、改 dtype、重建 `meta` 或将一个 bucket 的缓存用于另一个 bucket。正向缓存和 raw `q/k/g` 必须保留到反向 Stage2 完成。

反向允许同版本缓存同时缺少 `kr/gam` 并走重建路径，但只缺其中一项会报错。`qk_rstd` 缺省时反向从 raw Q/K 重建归一化信息。当前正向正常输出的缓存应直接整体传递，无需主动删减这些字段。

<a id="flash_kda_stage2_wrapper-衔接反向"></a>

### 衔接反向

```text
正向 Stage1 → AllGather(E,T) → 正向 Stage2 → out / final_state / train_cache
反向 Stage1 → AllGather(E_grad,R) → 反向 Stage2 → 输入梯度 / 参数梯度
```

四个计算 wrapper 本身不执行分布式通信。[chunk_kda 自动求导入口](#flash_kda_bwd_stage2_wrapper-自动求导入口)负责保存缓存、通信与梯度调度。`prepared` 中的 `qd/upre` 供当前正向 Stage2 使用，不是反向公开输入；反向所需字段已经包含在 `train_cache` 中。

<a id="flash_kda_bwd_stage1_wrapper"></a>

## flash_kda_bwd_stage1_wrapper

本接口在一个 kernel 内生成本 rank 的反向仿射摘要 `(E_grad,R)`，并清零当前 head bucket 的参数梯度区间。它不公开返回 `du/dh/dh0`；这些中间量在反向内部生成。实现见 [flash_kda_bwd.py](op_kernel/flash_kda_bwd.py)。

<a id="flash_kda_bwd_stage1_wrapper-函数原型"></a>

### 函数原型

```python
def flash_kda_bwd_stage1_wrapper(
    train_cache, do, scale, d_final_state=None, *, cp_meta,
    num_cores=None, param_grad_buffer=None,
):
    ...
```

返回 `(summary, parameter_buffer)`。

<a id="flash_kda_bwd_stage1_wrapper-输入"></a>

### 输入

| 参数 | dtype / shape / 语义 |
|---|---|
| `train_cache` | [正向 Stage2 缓存](#flash_kda_stage2_wrapper-训练缓存-train_cache)，版本必须为 `flash_kda_cp_train_v3_bf16_state` |
| `do` | BF16，shape/layout 与本地 forward 的 `out/v` 相同；`None` 表示零输出梯度 |
| `scale` | 必须与 `train_cache['meta']['scale']` 相同 |
| `d_final_state` | 本地 forward `final_state` 的上游梯度；FP32 V-first；dense `[B,Nv,128,128]`，packed `[M,Nv,128,128]`；`None` 表示零 |
| `cp_meta` | 与正向相同的 CP、head bucket 和 packed 索引；含义见 [正向 Stage1](#flash_kda_stage1_wrapper-meta-与-cp_meta) |
| `num_cores` | 正整数或 `None`，范围为当前设备可用 Cube 核数；`None` 优先读取 `KDA_BWD_NUM_CORES`，否则使用设备可用核数 |
| `param_grad_buffer` | 可选同设备连续 FP32 一维 Tensor，本 bucket `[Nv*129]` 或全 head `[global_Nv*129]` |

`do/d_final_state` 支持非重叠 stride 视图，最后维连续；状态内层矩阵必须连续。缓存布局和 dtype 必须与正向一致。`do=None`、`d_final_state=None` 在 kernel 内按零处理。

`d_final_state` **仅包含本地输出的 loss 梯度**，不能预先加上后继 rank 的状态梯度；后继贡献由 Stage2 合并，否则会重复计算。Stage1 和 Stage2 必须传入相同的本地 `do/d_final_state`。

<a id="flash_kda_bwd_stage1_wrapper-输出摘要"></a>

### 输出摘要

`summary` 为连续 FP32：dense `[B,Nv,2,128,128]`，packed `[1,Nv,2,128,128]`。

| slot | 内容 | 含义 |
|---|---|---|
| 0 | `E_grad` | 由本地 `do` 和本地 `d_final_state` 产生的入站状态梯度贡献 |
| 1 | `R` | 将后继传入的状态梯度映射回本地起点的线性传递矩阵 |

以 V-first 状态梯度表示，摘要描述 `D_in = D_suffix @ R + E_grad`。`E_grad` 由实际上游梯度递推得到；`R` 对应单位阵种子、零输出梯度的递推。摘要存储 FP32，内部矩阵乘使用 BF16 操作数、FP32 累加；仿射表达不代表浮点运算逐位满足线性。

`last_continues=False` 时不向后继传播依赖，`R` 写零。packed 的反向摘要描述**第一个本地片段**；若本地有多个片段，后继尾片段无法影响首片段，公开 `R` 也写零。首片段自己的 `d_final_state` 仍计入 `E_grad`。

CP>1 时，在 wrapper 外按 group rank 顺序 AllGather，得到 `[CP*B,Nv,2,128,128]`，packed 为 `[CP,Nv,2,128,128]`，传给 [反向 Stage2](#flash_kda_bwd_stage2_wrapper)。每个 bucket 的 packet 必须在对应 bucket 内通信，不能跨 bucket 混用。

CP=1 直接将原 `summary` Tensor 对象传到 Stage2。部分形状下 Stage1 会在该对象上保留私有 `du/dH` 缓存；不要把这些属性当作 ABI。clone/gather 丢失私有属性时，Stage2 可以重新递推，数值正确性合同不变，但执行路径和性能可能变化。

<a id="flash_kda_bwd_stage1_wrapper-参数梯度-buffer"></a>

### 参数梯度 buffer

令 `H` 为 buffer 覆盖的 value-head 总数，布局为：

```text
buffer[:H]                    = dA_log        [H]
buffer[H:].view(H, 128)        = ddt_bias      [H,128]
```

这里的 `dA_log` 是 gate 参数 `a_log` 的梯度，不是 CP 摘要中的矩阵，也不是 `akk` 的梯度。

Stage1 只清零，**此时 buffer 还不是最终参数梯度**；Stage2 向同一 buffer 的当前 bucket 区间累积结果。本地 buffer 从 head 0 开始；全 head buffer 使用 `hv_start`：`dA_log` 区间为 `[hv_start:hv_start+Nv]`，`ddt_bias` 区间起点为 `global_Nv+128*hv_start`。

省略 `param_grad_buffer` 时分配本 bucket 大小；传入时原对象返回。多个 bucket 可以共享全 head buffer，每个 bucket 串行执行自己的 Stage1/Stage2，只清零并累积其负责的区间。跨 CP rank 的 SUM AllReduce 应在各 bucket 都写完后执行，规则见 [Stage2 参数梯度通信](#flash_kda_bwd_stage2_wrapper-cp-通信与参数梯度)。

<a id="flash_kda_bwd_stage2_wrapper"></a>

## flash_kda_bwd_stage2_wrapper

本接口在一个 kernel 内完成反向 suffix 摘要合并、状态递推、chunk 梯度，以及 L2Norm/gate/beta sigmoid 的梯度链。返回的是 **raw 输入的梯度**，调用方不应再次应用这些激活的梯度。实现见 [flash_kda_bwd.py](op_kernel/flash_kda_bwd.py)。

<a id="flash_kda_bwd_stage2_wrapper-函数原型"></a>

### 函数原型

```python
def flash_kda_bwd_stage2_wrapper(
    q_raw, k_raw, g_raw, a_log, dt_bias,
    train_cache, do, all_bwd_summary, param_grad_buffer,
    scale, lower_bound, d_final_state=None, *, cp_meta,
    num_cores=None, output_buffers=None,
):
    ...
```

返回顺序为 `(dq, dk, dv, dbeta, dg, dh_initial_state)`。注意 `dbeta` 位于 `dg` 前面。

<a id="flash_kda_bwd_stage2_wrapper-输入与返回值"></a>

### 输入与返回值

| 输入 | 约定 |
|---|---|
| `q_raw`, `k_raw` | 正向原始 Q/K，BF16，`Nk` heads，归一化前的数据 |
| `g_raw` | 正向原始 gate 输入，BF16，`Nv` heads，激活前的数据 |
| `a_log`, `dt_bias` | 与正向相同的连续 FP32 参数，`[Nv]` / `[Nv,128]` |
| `train_cache` | [正向 Stage2](#flash_kda_stage2_wrapper-训练缓存-train_cache) 的完整训练缓存和元数据 |
| `do`, `d_final_state` | 与反向 Stage1 相同的本地 loss 梯度；可为 `None`，不预加后继 rank 贡献 |
| `all_bwd_summary` | 同设备、连续 FP32；dense `[CP*B,Nv,2,128,128]`，packed `[CP,Nv,2,128,128]`；按 group rank 排序 |
| `param_grad_buffer` | Stage1 已清零的同一 FP32 buffer，本 bucket 或全 head 布局 |
| `scale`, `lower_bound` | 必须分别等于 forward meta 中的值；`lower_bound` 范围 `[-5,0]` |
| `cp_meta`, `num_cores` | 与 [反向 Stage1](#flash_kda_bwd_stage1_wrapper) 相同的规则 |
| `output_buffers` | 可选 dict，允许键为 `dq/dk/dv/dbeta/dg/dh_initial_state`；缺省项由 wrapper 分配 |

`q_raw/k_raw/g_raw/do` 的 shape/layout 与对应正向 raw 输入一致，支持 BSND、BNSD、TND。无需再次传入 raw `v/beta`：反向从 `v_c/beta_c/v_new` 等训练缓存取得所需值。

| 返回值 | dtype | shape |
|---|---|---|
| `dq`, `dk` | BF16 | 分别与 raw `q/k` 相同，GVA 下仍是 `Nk` heads |
| `dv`, `dg` | BF16 | 分别与 raw `v/g` 相同，`Nv` heads |
| `dbeta` | BF16 | 与 raw `beta` 相同，移除特征维 |
| `dh_initial_state` | FP32 | V-first；dense `[B,Nv,128,128]`，packed `[M,Nv,128,128]` |

`dh_initial_state` 在实际逻辑起点写入梯度，在延续位置写零。底层 wrapper 始终返回它；自动求导入口只有在调用方提供了 initial state 时才把它作为输入梯度返回。

`dA_log/ddt_bias` 写入 `param_grad_buffer`，不额外返回；布局见 [Stage1 参数 buffer](#flash_kda_bwd_stage1_wrapper-参数梯度-buffer)。

raw 输入和绑定输出允许合法非重叠 stride，末维连续；状态内层矩阵连续。反向输出绑定不得与任何 live 输入或其他输出共享底层 storage，即使是同一 storage 中不重叠的切片也会被保守检查拒绝。合法绑定会返回原对象，TND 也保留三维输出布局。不要复用或修改仍被缓存引用的输入存储。

<a id="flash_kda_bwd_stage2_wrapper-cp-通信与参数梯度"></a>

### CP 通信与参数梯度

Stage2 从后往前合并本 rank 后面的摘要：`D = MM16(D,R_rank) + E_grad_rank`。合并结果用于本地尾部延续片段，再加一次本地 `d_final_state`。packed 的其他片段独立反传；通信包不会把不同逻辑序列混合。

完整运行顺序为：

```text
正向 Stage1 → AllGather(E,T) → 正向 Stage2
反向 Stage1 → AllGather(E_grad,R) → 反向 Stage2
           → 参数梯度 SUM AllReduce（所有 head bucket 完成后）
```

四个 wrapper 本身只负责计算，不执行通信。自动求导的 [kda_cp.py](op_kernel/kda_cp.py) adapter 在 CP>1 时每个 bucket、每个方向各做一次 Tensor AllGather；默认对完整 `dA_log/ddt_bias` buffer 做一次 **SUM** AllReduce。配置 `param_grad_reduce='external'` 时由外部负责归约，避免重复求和。token 输入梯度不在此 adapter 内 AllReduce。

adapter 的准备阶段还会用 `all_gather_object` 校验各 rank 的布局、参数和 packed 逻辑序列关系；这与上面的两个 Tensor 摘要 AllGather 不同。HCCL 用于 NPU 通信；Gloo 分支会经 CPU 中转，仅用于验证。当前仓内 CP2/CP4 测试在单 NPU 上模拟通信，不能当作真实 HCCL 多卡验收。

<a id="flash_kda_bwd_stage2_wrapper-单卡四个-wrapper-直调示例"></a>

### 单卡四个 wrapper 直调示例

以下函数假设调用方已创建同设备、满足上述 shape/dtype 的 Tensor。dense 单卡使用 `cp_size=1`，两个摘要直接透传。

```python
from ops.pypto_pro.kimi_k3.flash_kda.op_kernel import (
    flash_kda_stage1_wrapper, flash_kda_stage2_wrapper,
    flash_kda_bwd_stage1_wrapper, flash_kda_bwd_stage2_wrapper,
)

def run_kda_with_grads(q, k, v, g, beta, a_log, dt_bias, do,
                       initial_state=None, d_final_state=None,
                       lower_bound=-5.0, num_cores=None):
    scale = 128**-0.5
    cp_meta = dict(cp_rank=0, cp_size=1,
                   first_continues=False, last_continues=False)
    prepared, fwd_summary = flash_kda_stage1_wrapper(
        q, k, v, g, beta, scale, a_log, dt_bias, lower_bound,
        initial_state, cp_meta=cp_meta, is_training=True,
        num_cores=num_cores,
    )
    out, final, cache = flash_kda_stage2_wrapper(
        prepared, fwd_summary, cp_meta=cp_meta, num_cores=num_cores,
    )
    bwd_summary, params = flash_kda_bwd_stage1_wrapper(
        cache, do, scale, d_final_state,
        cp_meta=cp_meta, num_cores=num_cores,
    )
    grads = flash_kda_bwd_stage2_wrapper(
        q, k, g, a_log, dt_bias, cache, do, bwd_summary, params,
        scale, lower_bound, d_final_state,
        cp_meta=cp_meta, num_cores=num_cores,
    )
    nv = a_log.numel()
    dA_log = params[:nv]
    ddt_bias = params[nv:].view(nv, 128)
    return out, final, grads, dA_log, ddt_bias
```

<a id="flash_kda_bwd_stage2_wrapper-自动求导入口"></a>

### 自动求导入口

一般训练代码使用本目录导出的 `chunk_kda`，由它保存缓存并调度四个 wrapper：

```python
def chunk_kda(
    q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True, allow_neg_eigval=False,
    safe_gate=True, lower_bound=-1.0, disable_recompute=True,
    return_intermediate_states=False, state_v_first=True,
    cu_seqlens=None, cu_seqlens_cpu=None, cp_context=None, *,
    A_log, dt_bias, layout_qkv='BSND', num_cores=None,
    chunk_size=64, head_bucket_size=None,
):
    ...
```

`A_log` 是自动求导入口的参数名，对应底层 wrapper 的 `a_log`。需要梯度的 Tensor 应设置 `requires_grad=True`。实现见 [chunk_kda.py](op_kernel/chunk_kda.py)。

- 返回 `(out, final_state)`；`output_final_state=False` 时第二项为 `None`。`scale=None` 使用 `128**-0.5`。
- `use_qk_l2norm_in_kernel/use_gate_in_kernel/use_beta_sigmoid_in_kernel/safe_gate/state_v_first` 当前必须为 `True`；`allow_neg_eigval=False`、`chunk_size=64`。
- `disable_recompute=True` 保存训练缓存，每个 bucket 正向两个 kernel、反向两个 kernel；设为 `False` 会在反向额外运行两个正向 kernel 重建缓存。
- `head_bucket_size` 按 key-head 组计数，范围 `[1,Nk]`，缺省一个完整 bucket；不能拆开一个 key-head 对应的 value-head 组。
- `return_intermediate_states=True` 增加第三个返回值 `h`（BF16 V-first 训练状态缓存），它不可微，且要求单个 head bucket。`h` 不是额外的 FP32 final state。
- TND 通过 `cu_seqlens` 指定非空片段；也接受 `B=1` 的 BSND 加 `cu_seqlens`，内部用 TND，输出恢复 batch 维。`cu_seqlens_cpu` 可提供一致的 CPU 边界副本。
- `cp_context=None` 为单卡，无分布式初始化要求。CP 使用 `KDACPContext(group=..., ...)`，须先初始化 process group；该 dataclass 的字段为 `group/cu_seqlens/cu_seqlens_cpu/first_continues/last_continues/sequence_ids/param_grad_reduce`。packed CP 必须提供延续标志与逻辑序列 ID，规则见 [正向 Stage1](#flash_kda_stage1_wrapper-meta-与-cp_meta)。
- 当前自动求导只支持一阶反向，不支持二阶梯度。

<a id="flash_kda_bwd_stage2_wrapper-测试口径"></a>

### 测试口径

仓内回归入口为 [test_flash_kda.py](tests/test_flash_kda.py)，数学参考为 [kda_golden_cpu.py](tests/assets/kda_golden_cpu.py)。测试从仓库根目录运行；该路径是源码目录入口。

当前脚本以 CPU FP32 Golden 的整体相对 RMS/L2 `<0.008` 且结果有限作为判据；三方检查中的第三方 matmul 将操作数先舍入到 BF16，再转 FP32 做 CPU matmul。这个历史测试口径**不是“双千分之三”验收标准**，不能据此声称通过双千三。

`test_autograd[packed]` 使用 `lower_bound=-5`，其余自动求导用例使用 `-0.1`；该配置调整不表示原 `-0.1` packed 参数梯度问题已被算法修复。本次接口文档不改变测试阈值或 kernel 行为。
