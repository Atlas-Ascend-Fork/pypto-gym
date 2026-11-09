# commit_recurrent_kda_replayssm

## 产品支持情况

| 项目 | 说明 |
| :--- | :--- |
| 硬件平台 | Ascend 950PR（`Ascend950PR_9589`，`NpuArch=3510`）；cube 32 核 / vector 64 核 |
| 使用的计算单元 | AI Vector（AIV）核，不使用 Cube |
| 软件环境 | CANN 9.3.0、PyPTO-Pro（pypto 0.2.1）、torch 2.13.0、torch_npu 2.13.0rc1 |
| 设备侧 kernel 名 | `PYPTO_commit_recurrent_kda_replayssm_kernel` |
| 输入数据类型 | `recurrent_state`、`replay_decay` 为 FP32；`replay_u`、`replay_k` 为 FP16；`num_accepted_tokens` 为 INT32 |
| 输出数据类型 | 无独立输出；原地更新 `recurrent_state`（FP32） |
| 支持的形状 | `D = 128` 固定；`B`、`S`、`N` 为正整数，无上限；`num_accepted_tokens[b] ∈ [0, S]` |

不接受的输入由 wrapper 直接拒绝并给出错误信息，不做静默降级：dtype 或 shape 与上表
不符；`rows` 不在 `{8, 16, 32, 64, 128}`；`slots` 不在 `{2, 3, 4}`；`cap` 不在
`{8, 16, 32, 48, 64}`；`rdep` 不在 `{2, 3}`；输入不在同一 NPU 或不连续。

## 功能说明

`commit_recurrent_kda_replayssm` 根据 ReplaySSM 保存的 replay record，把每个 batch 已
接受的 token 依次提交到递归 checkpoint。算子原地更新 `recurrent_state`，不修改 replay
record 与接受计数。

### 计算过程

对 batch `b` 的前 `num_accepted_tokens[b]` 个 token，按 `t` 递增顺序执行：

$$
S \leftarrow S \odot \text{replay\_decay}[b,t] + \text{replay\_u}[b,t]\,\text{replay\_k}[b,t]^{\top}
$$

`D = 128` 固定。`num_accepted_tokens[b] = 0` 时该 batch 的 checkpoint 逐位不变。接受
计数只读，不拷回 host 校验，调用方必须保证每个值位于 `[0, S]`。

一个 `(batch, head, 行块)` 是一个工作项：记录块与 state 各搬一次，token 循环全在
向量寄存器与 UB 之间完成。沿 token 维的递推是唯一的串行轴，batch、head 与 state 行
之间相互独立。切分与流水细节见 `DESIGN.md`。

## 函数原型

```python
# 把 op_kernel 目录加入 sys.path 后：
from commit_recurrent_kda_replayssm import commit_recurrent_kda_replayssm

commit_recurrent_kda_replayssm(
    recurrent_state, replay_u, replay_k, replay_decay, num_accepted_tokens,
    block_num=None, rows=64, workers=None, slots=3, cap=16, rdep=2,
) -> None
```

### 参数说明

| 参数 | 形状 | 类型 | 必选/默认 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `recurrent_state` | `[B, N, D, D]` | FP32 | 必选 | 输入 checkpoint；算子原地提交已接受 token |
| `replay_u` | `[B, S, N, D]` | FP16 | 必选 | ReplaySSM 保存的 U，只读 |
| `replay_k` | `[B, S, N, D]` | FP16 | 必选 | ReplaySSM 保存的 K，只读 |
| `replay_decay` | `[B, S, N, D]` | FP32 | 必选 | ReplaySSM 保存的 decay，只读 |
| `num_accepted_tokens` | `[B]` | INT32 | 必选 | 每个 batch 提交的 token 数，取值 `[0, S]` |
| `block_num` | — | int | `None` | 兼容调用方签名；本实现不使用 |
| `rows` | — | int | `64` | 一个工作项持有的 state 行数，取值 `{8, 16, 32, 64, 128}` |
| `workers` | — | int | `None` | 启动 AIV 数，默认 `min(vector_core_num, 64)` |
| `slots` | — | int | `3` | state 槽位数，取值 `{2, 3, 4}` |
| `cap` | — | int | `16` | 融合记录块的 token 数，取值 `{8, 16, 32, 48, 64}` |
| `rdep` | — | int | `2` | 记录块槽位数，取值 `{2, 3}` |

### 返回值说明

返回 `None`。计算结果原地写入 `recurrent_state`。

## 约束说明

- `D = 128` 固定；`B`、`S`、`N` 为正数运行时维度，无人为上限。
- 所有输入张量必须连续、位于同一 NPU，并满足参数表中的 dtype 与 shape。
- `num_accepted_tokens` 的每个值必须位于 `[0, S]`，且不拷回 host 校验。
- `rows` 必须是 4 的倍数且属于 `{8, 16, 32, 64, 128}`；非法取值由 host 侧拒绝。
- `S > cap` 时算子在同一块常驻 state 上多走若干 chunk，结果不变、耗时增加。
- 单次调用下发一个 kernel。wrapper 只做形状校验，不做数值计算，也不分配 GM workspace。

## 调用示例

```python
import torch
import torch_npu

from commit_recurrent_kda_replayssm import commit_recurrent_kda_replayssm

B, S, N, D = 16, 8, 6, 128
recurrent_state = torch.zeros(B, N, D, D, dtype=torch.float32).npu()
replay_u = (torch.randn(B, S, N, D) * 0.02).half().npu()
replay_k = (torch.randn(B, S, N, D) * 0.02).half().npu()
replay_decay = torch.sigmoid(torch.randn(B, S, N, D)).npu()
num_accepted_tokens = torch.full((B,), 4, dtype=torch.int32).npu()

commit_recurrent_kda_replayssm(
    recurrent_state, replay_u, replay_k, replay_decay, num_accepted_tokens)
torch.npu.synchronize()
```

示例依赖的文件分布在算子目录下（`op_kernel/` 放 kernel 与公开 wrapper，`tests/assets/`
放 golden 与精度判据，`tests/` 直接放测试入口）：

```text
commit_recurrent_kda_replayssm/
├── README.md                                  # 本文件：接口与用法
├── DESIGN.md                                  # 设计说明：阶段划分、切分与流水
├── RESULTS.md                                 # 结果说明：用例、精度与性能
├── __init__.py
├── op_kernel/
│   ├── __init__.py
│   └── commit_recurrent_kda_replayssm.py      # 算子本体（kernel + 公开 wrapper）
└── tests/
    ├── test_commit_recurrent_kda_replayssm.py # 测试入口（pytest 与命令行都能跑）
    └── assets/
        └── gen_golden.py                      # 参考实现（CPU/NPU 双模式）与精度判据
```

测试与参考实现自检都可以在仓库根目录直接运行，不需要额外的驱动脚本：

```bash
# 全部用例（conftest.py 统一注入 tests/、tests/assets/、op_kernel/ 路径）
TILE_FWK_DEVICE_ID=0 pytest ops/pro/kimi_k3/commit_recurrent_kda_replayssm/tests --mode jit -q

# 参考实现自检（NPU 模式与 CPU 模式交叉校验）
python ops/pro/kimi_k3/commit_recurrent_kda_replayssm/tests/assets/gen_golden.py
```

用例清单、精度标准与性能结果见 `RESULTS.md`。
