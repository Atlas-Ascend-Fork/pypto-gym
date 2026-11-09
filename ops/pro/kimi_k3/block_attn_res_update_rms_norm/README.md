# block_attn_res_update_rms_norm

## 产品支持情况

- Ascend 950：支持（`get_platform_info()` = `PlatformInfo(soc_version='DAV_3510', vector_core_num=56)`）。
- 纯 Vector kernel，不使用 Cube；无 GM workspace，无跨核交接，无 scratch 张量。
- 软件环境（验收环境）：CANN 9.3.0（`/data/zsj/cann930/cann-9.3.0`）、pto-isa `1a25714`、pypto 0.3.0、Python 3.10.20、torch 2.10.0+cpu、torch_npu 2.10.0。

## 功能说明

Kimi K3 块注意力（block attention）残差模块：残差更新 + online-softmax 重标定 + RMSNorm 的融合算子。
算子原地更新 `partial_block` 为 `p`，并返回 `y`；`delta`、`pseudo_query`、`numerator`、
`logit_max`、`exp_sum`、`gamma` 均为只读。

### 计算过程

对每一行（token）`t`，`D = 7168` 固定：

$$
p = \text{partial\_block} + \text{fp32}(\text{delta})
$$

$$
\text{score} = \frac{\sum_j p_j q_j}{\sqrt{\frac{1}{D}\sum_j p_j^2 + \text{score\_eps}}},
\qquad m = \max(\text{logit\_max}, \text{score})
$$

$$
\alpha = \frac{e^{\text{logit\_max}-m}}{\text{exp\_sum}\cdot e^{\text{logit\_max}-m} + e^{\text{score}-m}},
\qquad
\beta = \frac{e^{\text{score}-m}}{\text{exp\_sum}\cdot e^{\text{logit\_max}-m} + e^{\text{score}-m}}
$$

$$
h = \text{bf16}\big(\text{numerator}\cdot\alpha + p\cdot\beta\big),
\qquad
\text{rstd} = \frac{1}{\sqrt{\frac{1}{D}\sum_j \text{fp32}(h_j)^2 + \text{norm\_eps}}}
$$

$$
y = \text{bf16}\big(\text{fp32}(h)\cdot \text{rstd}\cdot \text{fp32}(\text{gamma})\big)
$$

数值细节：`delta` 由 BF16 精确加宽到 FP32 后相加（不额外取整）；`h` 先按 BF16 就近取整，
再用取整后的值做第二次归约；`y` 只在最后取整一次；三处除法都用精确除法（非倒数近似）。

## 函数原型

```python
# 把 op_kernel 目录加入 sys.path 后：
from block_attn_res_update_rms_norm import block_attn_res_update_rms_norm

y = block_attn_res_update_rms_norm(
    partial_block, delta, pseudo_query, numerator, logit_max, exp_sum, gamma,
    score_eps=1e-6, norm_eps=1e-6,
)
```

### 参数说明

| 参数 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `partial_block` | Tensor | 必选 | 局部注意力结果；算子原地更新为 `p` | float32 | `[T, 7168]` |
| `delta` | Tensor | 必选 | 本次块的残差增量，只读 | bfloat16 | `[T, 7168]` |
| `pseudo_query` | Tensor | 必选 | 伪 Query 向量 `q`，所有行共享，只读 | float32 | `[7168]` |
| `numerator` | Tensor | 必选 | online-softmax 加权和，只读 | float32 | `[T, 7168]` |
| `logit_max` | Tensor | 必选 | 每行历史最大 logit，只读 | float32 | `[T]` |
| `exp_sum` | Tensor | 必选 | 每行历史指数和，只读 | float32 | `[T]` |
| `gamma` | Tensor | 必选 | RMSNorm 缩放因子，所有行共享，只读 | bfloat16 | `[7168]` |
| `score_eps` | float | 可选 | `score` 分母稳定项，默认 `1e-6`，须有限且 `> 0`；int 也接受并在 host 侧归一化为 float | float | 标量 |
| `norm_eps` | float | 可选 | `rstd` 分母稳定项，默认 `1e-6`，须有限且 `> 0`；int 也接受并在 host 侧归一化为 float | float | 标量 |

### 返回值说明

返回 `y`：`[T, 7168]` bfloat16 连续 Tensor。调用后 `partial_block` 已被原地写为 `p`（FP32）；
其余输入逐位不变。

## 约束说明

- 所有输入张量必须连续、位于同一 NPU，并满足参数表中的 dtype 与 shape。
- `D = 7168` 固定（112 个 64 lane 的 FP32 寄存器对）；其他隐藏维在 host 侧抛 `ValueError`。
- `T ≥ 0` 是唯一动态轴；`T = 0` 不启核，直接返回空张量。起核数为 `min(56, T)`，行按
  `pl.get_block_idx()` 步长轮流分配，每行恰好一个核，无尾块。
- 三个除法都用精确除法；`den = exp_sum·a + b ≥ b > 0`，`alpha`/`beta` ∈ `[0, 1]`；
  `score_eps`/`norm_eps` 为正保证两个 `sqrt` 的被开方数 ≥ `eps`，即使 `p`、`h` 全零或被
  FTZ 冲成 `+0` 结果仍有限。
- 不支持带 stride 的视图、跨 device 输入、非有限 `eps`；非法输入一律 `ValueError`，
  不静默给出错误结果。
- 输入含 NaN / Inf 时输出未定义（不在支持范围内）。

## 调用示例

```python
import torch
import torch_npu

from block_attn_res_update_rms_norm import block_attn_res_update_rms_norm

T, D = 64, 7168
dev = "npu:0"
f32 = lambda *s: torch.randn(*s, dtype=torch.float32, device=dev)
bf16 = lambda *s: torch.randn(*s, dtype=torch.bfloat16, device=dev)

partial_block = f32(T, D)
delta = bf16(T, D)
pseudo_query = f32(D)
numerator = f32(T, D)
logit_max = f32(T)
exp_sum = torch.rand(T, dtype=torch.float32, device=dev) + 1.0
gamma = bf16(D)

y = block_attn_res_update_rms_norm(
    partial_block, delta, pseudo_query, numerator, logit_max, exp_sum, gamma)
torch.npu.synchronize()
# torch.Size([64, 7168]) torch.bfloat16
print(y.shape, y.dtype)
# partial_block 已被原地更新为 p
```

## 精度测试

在仓库根目录执行（`conftest.py` 统一注入 `tests/`、`tests/assets/`、`op_kernel/` 路径）：

```bash
TILE_FWK_DEVICE_ID=5 pytest ops/pro/kimi_k3/block_attn_res_update_rms_norm/tests \
    --mode jit -q
# 结果：21 passed，y / p 与 CPU golden 逐位一致（max|diff| = 0.0）
```

与独立 CPU golden（`tests/assets/block_attn_res_update_rms_norm_golden_cpu.py`）按固定判据
（`tests/assets/block_attn_res_update_rms_norm_precision.py`：`np.isclose(rtol=1e-3, atol=0)`
的不满足比例 ≤ 1e-3，非有限值判失败）比对，覆盖：

| 用例组 | 内容 |
|:---|:---|
| `test_shape` | `T = 1, 2, 4, 8, 32, 64, 128, 254, 512, 1024`（含阈值、非整除、超核数） |
| `test_unlisted_legal_shape` | `T = 3, 5`：不参与特化选择的合法尺寸 |
| `test_cancellation` | `T = 8`、两个种子，`dot ≈ 0` 的病态消去（2 倍预言机规则） |
| `test_denormal_input_modes` | `ftz` / `recip_ftz` 输入，与 `T = 254` 的 `ftz` |
| `test_route_selection` / `test_class_transitions` | 路由阈值两侧、同一 kernel 对象来回切换 |
| `test_empty_token` / `test_rejects_invalid_inputs` | `T = 0` 与 5 类非法入参 |

CPU 参考实现可脱离 NPU 自检：`python tests/assets/block_attn_res_update_rms_norm_golden_cpu.py`。

## 性能

> 下表为**参考数值**，实际耗时每次运行可能有所波动。

口径：msprof application 采集的内核 `Task Duration`（原生 task 时长，不含 host 派发与输入恢复），
50 次预热 + 200 次采样，同机同 harness、同一会话，`D = 7168`、seed 42，Ascend 950PR / 56 向量核。

| 用例 | 本实现 med（µs） |
| :--- | ---: |
| T1 | 3.025 |
| T2 | 3.129 |
| T4 | 3.634 |
| T8 | 3.727 |
| T32 | 4.148 |
| T64 | 5.704 |
| T254 | 10.626 |
| T512 | 17.332 |

**性能倍数（> 1 表示本实现更快）**

| 对比对象 | 口径 | 本实现性能 = 对方的 |
| :--- | :--- | ---: |
| AscendC `block_attn_res_update` + `RmsNorm` | 同功能，两算子合计 | **1.45 倍** |
| AscendC `block_attn_res_update` 单算子 | 半程（我们多一遍 RMSNorm） | 0.80 倍 |

相对 AscendC 两段流水，`T1/8/64/512` 分别为 1.58 / 1.62 / 1.52 / 1.15 倍。AscendC 参考数据
在不同环境测得，仅作量级参考。
实现逐层走读与 AscendC 对比细节、计时入口留在仓库外的本地目录
`/data/n00947604/block_attn_res_update_rms_norm_*`，仅作本地阅读。

## 文件结构

```
ops/pro/kimi_k3/block_attn_res_update_rms_norm/
├── DESIGN.md                                     # 设计说明
├── README.md
├── RESULTS.md                                    # 用例 / 精度 / 性能结果
├── __init__.py
├── op_kernel/
│   ├── __init__.py
│   └── block_attn_res_update_rms_norm.py         # kernel + host wrapper
└── tests/
    ├── test_block_attn_res_update_rms_norm.py    # 逐 case 测试
    └── assets/
        ├── block_attn_res_update_rms_norm_golden_cpu.py   # 参考实现 + 输入生成 + 自检
        └── block_attn_res_update_rms_norm_precision.py    # 精度判据
```

> 当前交付为 **JIT 形态**（op_kernel + tests）；`op_host/` 与 run 包暂不提供。
