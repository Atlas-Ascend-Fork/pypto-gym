# fused_recurrent_kda_replayssm

## 产品支持情况

| 项目 | 说明 |
| :--- | :--- |
| 硬件平台 | Ascend 950PR（`Ascend950PR_9589`，`NpuArch=3510`）；cube 32 核 / vector 64 核 |
| 使用的计算单元 | AI Vector（AIV）核，不使用 Cube |
| 软件环境 | CANN 9.3.0、PyPTO-Pro（pypto 0.2.1）、torch 2.13.0、torch_npu 2.13.0rc1 |
| 设备侧 kernel 名 | `PYPTO_fused_recurrent_kda_replayssm_kernel`（入口 `fused_recurrent_kda_replayssm`，按 `Rows` / `Slots` / `Unroll` tiling key 编译） |
| 输入数据类型 | `mixedqkv`、`beta`、`g` 为 BF16；`a_log`、`dt_bias`、`recurrent_state` 为 FP32 |
| 输出数据类型 | `out` 为 BF16；`replay_u`、`replay_k` 为 FP16；`replay_decay` 为 FP32 |
| 支持的形状 | `B > 0`、序列长度为正、`1 <= N <= 96`、`Dk = Dv = 128`；布局为 BSND / BNSD / TND |

不接受的输入由 wrapper 直接拒绝并给出错误信息，不做静默降级：dtype 或 shape 与参数表
不符；`recurrent_state` 不是连续 FP32 `[B,N,128,128]`；三个 replay record 的 dtype /
shape 与 state 不匹配；`lower_bound` 不在 `[-5, 0]`；`rows` 不在 `{16, 32, 64}`；
`slots` 不在 `[2, 4]`；TND 缺少 `cu_seqlens`；输入不在同一 NPU 或不连续。

## 功能说明

Kimi Delta Attention ReplaySSM verify 算子：读入只读 FP32 checkpoint，计算全部候选
token 的 KDA 输出，并写出后续 Commit 使用的 replay record。算子融合 Q/K 行级 L2
normalize、gate 与 beta 激活、FP32 状态递推和 BF16 输出 / FP16 U、K / FP32 decay
记录写出。

对一个 batch、一个 head，令 $Q_t,K_t,V_t\in\mathbb{R}^{D}$ 为第 $t$ 个 token 的原始
输入，$S_t\in\mathbb{R}^{D_v\times D_k}$ 为处理该 token 前的递归状态，
$D_k=D_v=D=128$：

$$
q_t=\frac{Q_t}{\sqrt{\sum_{j}Q_{t,j}^{2}+10^{-6}}},\qquad
k_t=\frac{K_t}{\sqrt{\sum_{j}K_{t,j}^{2}+10^{-6}}}
$$

$$
\beta_t=\sigma(\beta_t^{\mathrm{raw}}),\qquad
\ell_t=\operatorname{lower\_bound}\cdot
\sigma\!\left(e^{A_{\log}}(g_t+\operatorname{dt\_bias})\right),\qquad d_t=e^{\ell_t}
$$

$$
\begin{aligned}
\bar S_t&=S_t\operatorname{diag}(d_t),\\
u_t&=\beta_t\left(V_t-\bar S_t k_t\right),\\
S_{t+1}&=\bar S_t+u_t k_t^{\top},\\
o_t&=\operatorname{scale}\cdot S_{t+1}q_t
\end{aligned}
$$

其中 $\sigma(x)=(1+e^{-x})^{-1}$，$S_0=$ `recurrent_state[b]`。gate 与 decay 按 key
通道计算，$\beta_t$ 是每个 token、每个 head 的标量。递推使用 FP32，输出转换为 BF16。
记录保存为

$$
\operatorname{replay\_u}_t=\operatorname{FP16}(u_t),\quad
\operatorname{replay\_k}_t=\operatorname{FP16}(k_t),\quad
\operatorname{replay\_decay}_t=\operatorname{FP32}(d_t)
$$

保存的 K 是归一化后的 key，decay 是激活后的指数值，U 是已乘 beta 的修正向量。FP16
U/K 只用于导出记录；本轮后续 token 的递推继续使用 FP32 工作状态。
`recurrent_state` 在算子内始终只读。

沿 token 维的递推是唯一的串行轴；不同 `(batch, head)` 之间相互独立，state 行之间也
相互独立，因此一个 head 可以按行切分。切分与流水细节见 `DESIGN.md`。

## 函数原型

```python
# 把 op_kernel 目录加入 sys.path 后：
from fused_recurrent_kda_replayssm import fused_recurrent_kda

out = fused_recurrent_kda(
    mixedqkv, recurrent_state, beta, g, scale, a_log, dt_bias,
    lower_bound=-5.0, layout_qkv="BSND", *,
    replay_u, replay_k, replay_decay,
    cu_seqlens=None, rows=None, workers=None, slots=3, unroll=2,
)
```

### 参数说明

`B` 是 batch 数，`S` 是稠密布局每条序列长度，`T` 是 TND 的 packed token 总数，`N` 是
Q/K/V 共用的头数，`D = 128`，`P = N*D`。

| 参数 | BSND shape | 类型 | 必选/默认 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `mixedqkv` | `[B,S,3*P]` | BF16 | 必选 | 按完整 Q、K、V 三段打包；输入已完成投影与卷积 |
| `recurrent_state` | `[B,N,D,D]` | FP32 | 必选 | 只读 checkpoint；后两维依次为 Dv、Dk |
| `beta` | `[B,S,N,1]` | BF16 | 必选 | 原始 logits，算子内做 sigmoid；末维 1 必须保留 |
| `g` | `[B,S,N,D]` | BF16 | 必选 | 原始 gate，由 `a_log`、`dt_bias`、`lower_bound` 激活 |
| `scale` | — | float / None | 必选 | Query 缩放；`None` 使用 `1/sqrt(D)` |
| `a_log` | `[N]` | FP32 | 必选 | 每个 head 的 log 时间尺度 |
| `dt_bias` | `[N,D]` | FP32 | 必选 | 每个 head、每个 key 通道的 gate 偏置 |
| `lower_bound` | — | float | `-5.0` | gate 下界，范围 `[-5, 0]` |
| `layout_qkv` | — | str | `"BSND"` | `"BSND"` / `"BNSD"` / `"TND"` |
| `replay_u` | `[B,S,N,D]` | FP16 | 必选，关键字参数 | 调用方分配，原地写已乘 beta 的修正向量 U |
| `replay_k` | `[B,S,N,D]` | FP16 | 必选，关键字参数 | 调用方分配，原地写归一化后的 K |
| `replay_decay` | `[B,S,N,D]` | FP32 | 必选，关键字参数 | 调用方分配，原地写激活后的 decay 指数值 |
| `cu_seqlens` | `[B+1]` | INT32 / None | `None`；TND 必传 | packed token 前缀和 |
| `rows` | — | int / None | `None` | 一个工作项持有的 state 行数，取值 `{16, 32, 64}`；`None` 按 `B*N` 自适应 |
| `workers` | — | int / None | `None` | 启动 AIV 数，默认 `min(vector_core_num, 64)` |
| `slots` | — | int | `3` | 工作 state 的 UB 槽位数，取值 `[2, 4]` |
| `unroll` | — | int | `2` | 更新 pass 的行 unroll，取值 `{1, 2, 4}` |

三种布局的 `mixedqkv`：BSND `[B,S,3*N*D]`、BNSD `[B,3*N,S,D]`、TND `[T,3*N*D]`；
`g`、`out` 与之对齐（`[B,S,N,D]` / `[B,N,S,D]` / `[T,N,D]`）；`beta` 末维为 1。
Replay record 始终按 `[B,S,N,D]` 存储，不随输入布局变化；TND 下 token
`cu_seqlens[b]+t` 的记录写入 `[b,t]`，各 batch 从记录位置 0 开始。

`rows=None` 时的选择规则（按实测负载粒度）：`B*N >= 24` 用 64，`B*N >= 6` 用 32，
否则用 16；`workers` 再截断到 `B*N*(128/rows)`。

### 返回值说明

返回 `out`：与 `g` 同 shape、dtype 为 BF16，算子内部新建。三个 replay record 通过传入
的缓冲区原地输出，不在返回值中重复返回。

## 约束说明

- `B > 0`、序列长度为正、`1 <= N <= 96`、`Dk = Dv = 128`；Q/K/V 头数必须相同。
- 所有张量必须连续并位于同一 NPU，dtype 和 shape 须满足上表。
- `recurrent_state` 在调用前后逐位不变；算子不写 checkpoint。
- 三个 replay record 必须由调用方预先分配且 shape 一致；只有本轮有效 token 的记录被
  覆盖，未使用的后缀保持原值（TND 下为 `[L_b, S)` 后缀）。
- TND 的 `cu_seqlens` 必须为 INT32 `[B+1]`，从 0 严格递增到 `T`，每条序列长度位于
  `[1, S]`；NPU 前缀和的数值由调用方保证合法。
- `scale` 必须有限，`lower_bound` 必须位于 `[-5, 0]`。
- 单次调用下发一个 kernel。wrapper 只做形状校验、常量准备与输出分配，不做数值计算。

## 调用示例

```python
import torch
import torch_npu

from fused_recurrent_kda_replayssm import fused_recurrent_kda

B, S, N, D = 16, 8, 6, 128
q = torch.randn(B, S, N, D, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
mixedqkv = torch.cat((q, k, v), dim=2).flatten(2).contiguous()
g = torch.randn_like(q)
beta = torch.randn(B, S, N, 1, device="npu", dtype=torch.bfloat16)
a_log = torch.zeros(N, device="npu", dtype=torch.float32)
dt_bias = torch.zeros(N, D, device="npu", dtype=torch.float32)
checkpoint = torch.zeros(B, N, D, D, device="npu", dtype=torch.float32)
replay_u = torch.empty(B, S, N, D, device="npu", dtype=torch.float16)
replay_k = torch.empty_like(replay_u)
replay_decay = torch.empty(B, S, N, D, device="npu", dtype=torch.float32)

out = fused_recurrent_kda(
    mixedqkv, checkpoint, beta, g, None, a_log, dt_bias, -5.0, "BSND",
    replay_u=replay_u, replay_k=replay_k, replay_decay=replay_decay,
)
# out.shape == g.shape；checkpoint 未被修改；三个 record 已原地写入。
```

示例依赖的文件分布在算子目录下：

```text
fused_recurrent_kda_replayssm/
├── README.md                                       # 本文件：接口与用法
├── DESIGN.md                                       # 设计说明：阶段划分、切分与流水
├── RESULTS.md                                      # 结果说明：用例、精度与性能
├── __init__.py
├── op_kernel/
│   ├── __init__.py
│   └── fused_recurrent_kda_replayssm.py            # 算子本体（kernel + 公开 wrapper）
└── tests/
    ├── test_fused_recurrent_kda_replayssm.py       # 测试入口
    └── assets/
        └── gen_golden.py                          # 参考实现（CPU/NPU 双模式）与精度判据
```

测试与参考实现自检都可以在仓库根目录直接运行：

```bash
# 全部用例（conftest.py 统一注入 tests/、tests/assets/、op_kernel/ 路径）
TILE_FWK_DEVICE_ID=0 pytest ops/pro/kimi_k3/fused_recurrent_kda_replayssm/tests --mode jit -q

# 参考实现自检（CPU FP64 与 NPU FP32 交叉校验）
python ops/pro/kimi_k3/fused_recurrent_kda_replayssm/tests/assets/gen_golden.py
```

用例清单、精度标准与性能结果见 `RESULTS.md`。
