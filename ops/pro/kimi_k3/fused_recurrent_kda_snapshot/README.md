# fused_recurrent_kda_snapshot

## 产品支持情况

| 项目 | 说明 |
| :--- | :--- |
| 硬件平台 | Ascend 950PR（`Ascend950PR_9589`，`NpuArch=3510`）；cube 32 核 / vector 64 核 |
| 使用的计算单元 | AI Vector（AIV）核，不使用 Cube |
| 软件环境 | CANN 9.3.0、PyPTO-Pro（pypto 0.2.1）、torch 2.13.0、torch_npu 2.13.0rc1 |
| 设备侧 kernel 名 | `PYPTO_kernel`（前端入口按 `rows` 与 state dtype 特化为 `kda_snapshot_r64` 等） |
| 输入数据类型 | `mixedqkv`、`beta`、`g` 为 BF16；`a_log`、`dt_bias` 为 FP32；`state` 为 BF16 / FP32；索引为 INT32 |
| 输出数据类型 | `out` 为 BF16；`state` 原地更新，dtype 与输入相同 |
| 支持的形状 | `B > 0`、序列长度为正、`1 <= N <= 96`、`Dk = Dv = 128`；布局为 BSND / BNSD / TND |

不接受的输入由 wrapper 直接拒绝并给出错误信息，不做静默降级：dtype 或 shape 与参数表
不符；`rows` 不在 `{32, 64}`；`slots` 不在 `[2, 5]`；`lower_bound` 不在 `[-5, 0]`；
BSND/BNSD 下 `C < B*S` 或 TND 下 `C < T`；`ssm_state_indices` 越界或同一次调用内重复；
输入不在同一 NPU 或不连续。

## 功能说明

打包 QKV 的 Kimi Delta Attention decode 融合算子：Q/K 行级 L2 normalize、gate 与 beta
激活、逐 token KDA 递推，并把每个 token 的完整递归状态写回 state pool。算子原地更新
`state`，`out` 内部新建。

对一个 batch、一个 head，令 `Q_t, K_t, V_t ∈ R^D` 为第 `t` 个 token 的原始输入，
`S_t ∈ R^{Dv×Dk}` 为处理该 token 前的递归状态，`Dk = Dv = D = 128`：

$$
q_t=\frac{Q_t}{\sqrt{\sum_{j}Q_{t,j}^{2}+10^{-6}}},\qquad
k_t=\frac{K_t}{\sqrt{\sum_{j}K_{t,j}^{2}+10^{-6}}}
$$

$$
\beta_t=\sigma(\beta_t^{\mathrm{raw}}),\qquad
\ell_t=\operatorname{lower\_bound}\cdot\sigma\!\left(e^{A_{\log}}(g_t+\operatorname{dt\_bias})\right)
$$

$$
\begin{aligned}
\bar S_t&=S_t\operatorname{diag}(e^{\ell_t}),\\
u_t&=\beta_t\left(V_t-\bar S_t k_t\right),\\
S_{t+1}&=\bar S_t+u_t k_t^{\top},\\
o_t&=\operatorname{scale}\cdot S_{t+1}q_t
\end{aligned}
$$

`S_{t+1}` 写入槽位 `ssm_state_indices[p_b + t]`；Q/K 归一化、激活与工作状态递推在
FP32，`o_t` 输出为 BF16，快照按 `state` 的 dtype 写回。

沿 token 维的递推是唯一的串行轴；不同 `(batch, head)` 之间相互独立，state 行之间也
相互独立。初态由 `num_accepted_tokens` 从池中选取，本轮仍计算全部候选并覆盖对应槽位。
切分与流水细节见 `DESIGN.md`。

## 函数原型

```python
# 把 op_kernel 目录加入 sys.path 后：
from fused_recurrent_kda_snapshot import fused_recurrent_kda

fused_recurrent_kda(
    mixedqkv, state, beta, g, scale, a_log, dt_bias,
    lower_bound=-5.0, layout_qkv="BSND", *,
    cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
    rows=64, workers=None, slots=3,
) -> (state, out)
```

### 参数说明

`B` 是 batch 数，`S` 是稠密布局每条序列长度，`T` 是 TND 的 packed token 总数，`N` 是
Q/K/V 共用的头数，`D = 128`，`P = N*D`，`C` 是 state pool 槽位数，`L = B*S`（BSND/BNSD）
或 `T`（TND）。

| 参数 | BSND shape | 类型 | 必选/默认 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `mixedqkv` | `[B,S,3*P]` | BF16 | 必选 | 按完整 Q、K、V 三段打包；输入已完成投影与卷积 |
| `state` | `[C,N,D,D]` | BF16 / FP32 | 必选 | 初态与逐 token 快照共用的池，原地更新；后两维为 Dv、Dk |
| `beta` | `[B,S,N,1]` | BF16 | 必选 | 原始 logits，算子内做 sigmoid；末维 1 必须保留 |
| `g` | `[B,S,N,D]` | BF16 | 必选 | 原始 gate，由 `a_log`、`dt_bias`、`lower_bound` 激活 |
| `scale` | — | float / None | 必选 | Query 缩放；`None` 使用 `1/sqrt(D)` |
| `a_log` | `[N]` | FP32 | 必选 | 每个 head 的 log 时间尺度 |
| `dt_bias` | `[N,D]` | FP32 | 必选 | 每个 head、每个 key 通道的 gate 偏置 |
| `lower_bound` | — | float | `-5.0` | gate 下界，范围 `[-5, 0]` |
| `layout_qkv` | — | str | `"BSND"` | `"BSND"` / `"BNSD"` / `"TND"` |
| `cu_seqlens` | `[B+1]` | INT32 / None | `None`；TND 必传 | packed token 前缀和 |
| `ssm_state_indices` | `[L]` | INT32 / None | `None` | 每个 token 的状态槽位；默认连续映射 |
| `num_accepted_tokens` | `[B]` | INT32 / None | `None`；默认 1 | 用于选择本轮初态快照 |
| `rows` | — | int | `64` | 一个工作项持有的 state 行数，取值 `{32, 64}` |
| `workers` | — | int | `None` | 启动 AIV 数，默认 `min(vector_core_num, 64)` |
| `slots` | — | int | `3` | 工作 state 槽位数，取值 `[2, 5]` |

三种布局的 `mixedqkv`：BSND `[B,S,3*N*D]`、BNSD `[B,3*N,S,D]`、TND `[T,3*N*D]`；
`g`、`out` 与之对齐（`[B,S,N,D]` / `[B,N,S,D]` / `[T,N,D]`）；`beta` 末维为 1。

### 返回值说明

返回 `(state, out)`：`state` 与传入的是同一张量，映射槽位已写入本轮逐 token 快照；
`out` 与 `g` 同 shape，dtype 为 BF16。

## 约束说明

- `B > 0`、序列长度为正、`1 <= N <= 96`、`Dk = Dv = 128`；Q/K/V 头数必须相同。
- 所有公开输入必须连续并位于同一 NPU，dtype 与 shape 满足上表。
- BSND/BNSD 要求 `C >= B*S`；TND 要求 `C >= T`。`ssm_state_indices` 每项须位于
  `[0, C)`，同一次调用内不得重复，否则快照互相覆盖。
- `num_accepted_tokens[b]` 须位于 `[0, L_b]`，只用于选择历史快照，不表示本轮只计算这些
  token；本轮仍计算全部候选并覆盖对应槽位。
- TND 的 `cu_seqlens` 必须从 0 严格递增到 `T`，每条序列非空。
- `lower_bound` 必须位于 `[-5, 0]`。索引、接受数与 NPU 前缀和不拷回 host 校验。
- `rows` 只支持 `32/64`；`slots` 只支持 `[2, 5]`。
- 单次调用下发一个 kernel。wrapper 只做形状校验与输出分配，不做数值计算。

## 调用示例

```python
import torch
import torch_npu

from fused_recurrent_kda_snapshot import fused_recurrent_kda

B, S, N, D = 16, 8, 6, 128
q = torch.randn(B, S, N, D, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
mixedqkv = torch.cat((q, k, v), dim=2).flatten(2).contiguous()
g = torch.randn_like(q)
beta = torch.randn(B, S, N, 1, device="npu", dtype=torch.bfloat16)
a_log = torch.zeros(N, device="npu", dtype=torch.float32)
dt_bias = torch.zeros(N, D, device="npu", dtype=torch.float32)
state = torch.zeros(B * S, N, D, D, device="npu", dtype=torch.float32)
ssm_state_indices = torch.arange(B * S, device="npu", dtype=torch.int32)
num_accepted_tokens = torch.full((B,), 4, device="npu", dtype=torch.int32)

updated_state, out = fused_recurrent_kda(
    mixedqkv, state, beta, g, None, a_log, dt_bias,
    layout_qkv="BSND",
    ssm_state_indices=ssm_state_indices,
    num_accepted_tokens=num_accepted_tokens,
)
# updated_state is state; out.shape == g.shape
```

示例依赖的文件分布在算子目录下（`op_kernel/` 放 kernel 与公开 wrapper，`tests/assets/`
放 golden 与精度判据，`tests/` 直接放测试入口）：

```text
fused_recurrent_kda_snapshot/
├── README.md                                  # 本文件：接口与用法
├── DESIGN.md                                  # 设计说明：阶段划分、切分与流水
├── RESULTS.md                                 # 结果说明：用例、精度与性能
├── __init__.py
├── op_kernel/
│   ├── __init__.py
│   └── fused_recurrent_kda_snapshot.py        # 算子本体（kernel + 公开 wrapper）
└── tests/
    ├── test_fused_recurrent_kda_snapshot.py   # 测试入口（pytest 与命令行都能跑）
    └── assets/
        └── gen_golden.py                      # 参考实现（CPU/NPU 双模式）与精度判据
```

测试与参考实现自检都可以在仓库根目录直接运行，不需要额外的驱动脚本：

```bash
# 全部用例（conftest.py 统一注入 tests/、tests/assets/、op_kernel/ 路径）
TILE_FWK_DEVICE_ID=0 pytest ops/pro/kimi_k3/fused_recurrent_kda_snapshot/tests --mode jit -q

# 参考实现自检（覆盖 BSND / BNSD / TND，NPU 模式与 CPU 模式交叉校验）
python ops/pro/kimi_k3/fused_recurrent_kda_snapshot/tests/assets/gen_golden.py
```

用例清单、精度标准与性能结果见 `RESULTS.md`。
