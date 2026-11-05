# mega_recurrent_kda_replayssm

## 产品支持情况

| 项目 | 说明 |
| :--- | :--- |
| 硬件平台 | Ascend 950（平台信息 `soc_version = 'DAV_3510'`）；cube（矩阵计算单元）32 核 / vector（向量计算单元）64 核，L2 128 MiB |
| 使用的计算单元 | Cube 与 AI Vector 混用：投影（M1）与输出投影（M4）在 Cube，Conv/归一化（M2）与递推（M3）在 AIV，单个 mixed kernel 内以一个 launch 完成 |
| 软件环境 | CANN 9.2.0、PyPTO-Pro（`pypto_pro.language`）、Python 3.11、torch / torch_npu 2.12 |
| 输入数据类型 | `hidden_states`、各投影权重、`conv1d_weight`、`conv_state` 为 BF16；`output_norm_weight`、`recurrent_state`、`replay_decay`、`a_log`、`dt_bias` 为 FP32；`replay_u` / `replay_k` 为 FP16；`conv_state_indices` / `conv_num_accepted_tokens` 为 INT32 |
| 输出数据类型 | `out` 为 BF16；`conv_state`、`replay_u`、`replay_k`、`replay_decay` 原地更新 |
| 支持的形状 | `S=8`、`H=7168`、`D=128`；`N = 96/TP ∈ {6, 12, 24}`（TP = 16/8/4），`P = N*D`，`P3 = 3*P`；`B ∈ [4, 512]`，且 `rows = B*S` 可被 m-tile 阶梯 `{128,112,96,80,64,48,32}` 中某一档整除 |
| 参考实现 | 数学契约的独立 CPU 参考实现（FP32 计算、BF16 输出），见 `tests/assets/mega_recurrent_kda_replayssm_golden_cpu.py`；精度比较器见 `tests/assets/mega_recurrent_kda_replayssm_precision.py` |

未覆盖的情况（当前不支持）：`B < 4`、`B > 512`、`B*S` 不能被 m-tile 阶梯整除、`N ∉ {6,12,24}`、
`H ≠ 7168`、`conv1d_weight` 三种公开 layout 以外的形状、输入含 NaN / Inf。

前两类在 host 侧直接抛 `ValueError`，不会静默给出错误结果；dtype 或内存格式与上表
不符时，同样在 host 侧报错并指出是哪个入参、期望与实际分别是什么。含 NaN / Inf 的输入不在
支持范围内，输出未定义。

## 功能说明

`mega_recurrent_kda_replayssm` 是 Kimi K3 ReplaySSM **verify 片段**的融合算子，使用 PyPTO-Pro
实现为**一次** mixed AIC/AIV kernel 调用。输入 `hidden_states` 依次经过四段计算：

```text
1. 投影          raw_qkv = X·W_qkvᵀ（FP32 累加）      gate = X·W_gateᵀ
                 raw_decay = (X·W_aᵀ → BF16)·W_bᵀ    beta_logit = X·W_betaᵀ
2. Conv/激活     qkv = SiLU(causal Conv1D(K=4)(历史 3 行 + 原始 8 行))；Q/K 做 L2 归一化
                 decay = exp(lb · sigmoid(exp(a_log)·(raw_decay + dt_bias)))
                 beta  = sigmoid(beta_logit)
3. 递推          每 (batch, head) 一个 D×D 状态：
                 S_t = S_{t-1}·diag(decay_t) + delta_t·k_tᵀ
                 delta_t = (v_t − S_{t-1}·diag(decay_t)·k_t)·beta_t
                 o_t = (S_t·q_t)·scale
4. 输出           gated = bf16(rmsnorm(o)·gamma · sigmoid(gate))；out = gated·W_outᵀ
```

把这四段各写成一个 kernel，就要在全局内存与片上之间往返搬运中间张量（q/k/v、decay、beta、
gate、gated 等）。本算子把 QKV/decay/beta/output-gate 投影、causal Conv1D+SiLU、recurrent KDA、
RMSNorm、output gating 与输出投影组合成**一个** kernel：中间张量只经工作区（`ws_qkv`、
`ws_gate`、`ws_gated` 等）在同一个 launch 内传递，host 侧不做任何数值计算，也不做额外的同步
或后处理。两段 Cube 计算（M1 投影、M4 输出投影）与两段 AIV 计算（M2 消费、M3 递推）通过面板
交接与 3 步 slot 握手重叠，具体切分见 `DESIGN.md`。

## 函数原型

```python
# 把 op_kernel 目录加入 sys.path 后：
from mega_recurrent_kda_replayssm import mega_recurrent_kda_replayssm_wrapper

out = mega_recurrent_kda_replayssm_wrapper(
    hidden_states, qkv_projection_weight, decay_projection_a_weight,
    decay_projection_b_weight, beta_projection_weight, output_gate_projection_weight,
    output_norm_weight, output_projection_weight, conv1d_weight, conv_state,
    recurrent_state, replay_u, replay_k, replay_decay,
    conv_state_indices, conv_num_accepted_tokens, a_log, dt_bias,
    scale, lower_bound, rms_norm_eps=1e-5,
)   # out: BF16 [B, S, H]
```

### 参数说明

| 参数 | 形状 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `hidden_states` | `[B, S, H]` | BF16 | 主输入；`B ∈ [4,512]` 且 `B*S` 可被 m-tile 阶梯整除 |
| `qkv_projection_weight` | `[3P, H]` | BF16, NZ | Q/K/V 投影权重 |
| `decay_projection_a_weight` | `[D, H]` | BF16, NZ | decay 的两段投影之一 |
| `decay_projection_b_weight` | `[P, D]` | BF16, NZ | decay 的两段投影之二 |
| `beta_projection_weight` | `[N, H]` | BF16, NZ | beta 投影权重 |
| `output_gate_projection_weight` | `[P, H]` | BF16, NZ | output gate 投影权重 |
| `output_norm_weight` | `[D]` | FP32 | RMSNorm 的 gamma |
| `output_projection_weight` | `[H, P]` | BF16, NZ | 输出投影权重 |
| `conv1d_weight` | `[3, N, 4, D]` / `[4, 3P]` / `[3, 4, D]` | BF16 | 三种公开 layout 均可，host 侧由形状推断；kernel 收到原始张量 + `conv_layout`，在 kernel 内建视图（wrapper 不做 reshape / 重排 / 拷贝） |
| `conv_state` | `[C, S+2, 3P]` | BF16 | 卷积历史，原地更新；`C` 需覆盖 `conv_state_indices` 的最大槽位（测试用 `B+1`） |
| `recurrent_state` | `[B, N, D, D]` | FP32 | 递推初态，**只读** |
| `replay_u` | `[B, S, N, D]` | FP16 | 递推中间量，原地更新 |
| `replay_k` | `[B, S, N, D]` | FP16 | 递推中间量，原地更新 |
| `replay_decay` | `[B, S, N, D]` | FP32 | 递推中间量，原地更新 |
| `conv_state_indices` | `[B]` | INT32 | 每个 batch 的 `conv_state` 槽位号 |
| `conv_num_accepted_tokens` | `[B]` | INT32 | 每个 batch 的有效历史长度 |
| `a_log` | `[N]` | FP32 | decay 的逐 head 缩放 |
| `dt_bias` | `[N, D]` | FP32 | decay 的逐 head 偏置 |
| `scale` | 标量 | float | 递推输出的缩放，必须是有限 Python `float`；`128**-0.5` |
| `lower_bound` | 标量 | float | decay 的指数下界，Python `float` 且取 `[-5, 0]`；`-5.0` |
| `rms_norm_eps` | 标量 | float | RMSNorm 的稳定项，Python `float` 且 `> 0`；默认 `1e-5` |

### 返回值说明

| 返回值 | 形状 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `out` | `[B, S, H]` | BF16 | 最终输出，由本函数内部分配并返回；`conv_state` / `replay_*` 原地更新，`recurrent_state` 不变 |

公开入口是 `mega_recurrent_kda_replayssm_wrapper`：模块内只有这一个对外符号（另有
`@pl.jit` kernel `mega_recurrent_kda_replayssm_kernel`）。wrapper 完成参数校验、
工作区分配与唯一一次 kernel 下发，返回 `out`；Module 1..4 的中间工作区（`ws_qkv`、
`ws_gated` 等）是 kernel 内部量，不作为返回值暴露。

## 约束说明

- `B` 必须在 `[4, 512]`，且 `rows = B*S` 能被 m-tile 阶梯 `{128,112,96,80,64,48,32}` 中
  某一档整除；host 取能整除的最大档作为 `KEY_M_TILE_ROWS`，不符合时入口抛 `ValueError`，不做静默降级。
- `N` 由 `beta_projection_weight` / 各投影权重形状推断，只支持 `6 / 12 / 24`；`H` 固定 `7168`，
  `S` 固定 `8`，`D` 固定 `128`，`conv1d_weight` 的 K 固定 `4`。
- 各权重的 dtype / NZ 格式必须与参数表一致：入口不做 host 侧 format cast，也不做隐式 dtype
  转换；不符合时报错而不是静默降级。
- `scale` / `lower_bound` / `rms_norm_eps` 必须是 Python `float`（`type(x) is float`）；numpy
  标量请先 `float(...)` 转换，否则入口抛 `ValueError`（这三个标量按 builtin float 绑定到 kernel）。
- `conv_state`、`replay_u`、`replay_k`、`replay_decay` 是原地输出：调用后内容被更新；
  `recurrent_state` 只读，kernel 不写回（零容差只读校验）。
- 单次调用只下发一个 kernel（一个 mixed AIC/AIV block 网格），host 侧不做数值计算、不做额外
  同步或后处理；`out` 是新分配的张量。
- 起核数取 `{8,16,24,28,32}` 中不超过 `min(cube, vector//2)` 的最大档；测试用例可显式指定
  其中一档（`select_grid` 覆盖）。当前 stream 少于 8 个 AIC 时入口报错。
- 模块导入无副作用：不注册 `torch.library` op，也不提供 `torch.compile` 图模式；调用只经
  公开入口 `mega_recurrent_kda_replayssm_wrapper`。

## 调用示例

```python
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "ops/pro/kimi_k3/mega_recurrent_kda_replayssm/op_kernel")
from mega_recurrent_kda_replayssm import mega_recurrent_kda_replayssm_wrapper

torch.npu.set_device(0)
B, S, H, D, N = 16, 8, 7168, 128, 6
P, P3 = N * D, 3 * N * D
NZ = torch_npu.Format.FRACTAL_NZ

bf = lambda *s: torch.randn(*s, dtype=torch.bfloat16).npu()
out = mega_recurrent_kda_replayssm_wrapper(
    bf(B, S, H),
    torch_npu.npu_format_cast(bf(P3, H), NZ),
    torch_npu.npu_format_cast(bf(D, H), NZ),
    torch_npu.npu_format_cast(bf(P, D), NZ),
    torch_npu.npu_format_cast(bf(N, H), NZ),
    torch_npu.npu_format_cast(bf(P, H), NZ),
    torch.randn(D, dtype=torch.float32).npu(),
    torch_npu.npu_format_cast(bf(H, P), NZ),
    bf(4, P3),                                   # [4, 3P] layout
    bf(B + 1, S + 2, P3),                        # conv_state
    torch.randn(B, N, D, D, dtype=torch.float32).npu(),   # recurrent_state, read-only
    torch.empty(B, S, N, D, dtype=torch.float16).npu(),
    torch.empty(B, S, N, D, dtype=torch.float16).npu(),
    torch.empty(B, S, N, D, dtype=torch.float32).npu(),
    (torch.arange(B, dtype=torch.int32) + 1).npu(),
    (torch.arange(B, dtype=torch.int32) % (S + 1)).npu(),
    torch.randn(N, dtype=torch.float32).npu(),
    torch.randn(N, D, dtype=torch.float32).npu(),
    128 ** -0.5, -5.0,
)
torch.npu.synchronize()
print(out.shape, out.dtype)
# torch.Size([16, 8, 7168]) torch.bfloat16
```

示例依赖的文件分布在算子目录下（`op_kernel/` 放 kernel 与 host 入口，`tests/assets/` 放
golden 与精度比较器，`tests/` 直接放测试入口）：

```text
mega_recurrent_kda_replayssm/
├── README.md                                         # 本文件：接口与用法
├── DESIGN.md                                         # 设计说明：阶段划分、切分与流水
├── RESULTS.md                                        # 结果说明：用例、精度与性能
├── op_kernel/
│   └── mega_recurrent_kda_replayssm.py               # 算子本体（kernel + 公开 host wrapper）
└── tests/
    ├── test_mega_recurrent_kda_replayssm.py          # 测试入口（pytest 与命令行都能跑）
    └── assets/
        ├── mega_recurrent_kda_replayssm_golden_cpu.py   # 独立 CPU 参考实现
        └── mega_recurrent_kda_replayssm_precision.py    # 精度比较器（方案 A）
```

## 运行测试

```bash
# 全部 15 个交付用例（需要 950 设备）
python -m pytest -q ops/pro/kimi_k3/mega_recurrent_kda_replayssm/tests/
# 或直接跑测试文件
python ops/pro/kimi_k3/mega_recurrent_kda_replayssm/tests/test_mega_recurrent_kda_replayssm.py
```

代码分层（8 段：几何常量 / `GeometryKey` / Cube helpers / M1–M4 各段 / kernel / host）见
`DESIGN.md` 第 0 节，切分与流水见第 1–5 节，精度与性能结果见 `RESULTS.md`。
