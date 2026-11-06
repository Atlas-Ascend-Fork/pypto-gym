# mega_recurrent_kda

## 产品支持情况

| 项目 | 说明 |
| :--- | :--- |
| 硬件平台 | Ascend 950（平台信息 `soc_version = 'DAV_3510'`）；cube（矩阵计算单元）32 核 / vector（向量计算单元）64 核 |
| 使用的计算单元 | AI Cube（AIC）与 AI Vector（AIV）混合的**一个** kernel；默认 32 AIC（可配 8 / 16 / 24 / 28 / 32）+ 64 AIV（AIV = 2 × AIC 配额） |
| 软件环境 | CANN 9.2.0-beta.2、PyPTO-Pro |
| 输入数据类型 | `hidden_states`、`conv1d_weight`、`conv_state` 为 BF16；六个投影权重为 BF16 FRACTAL_NZ（逻辑 `[K,N]`，非 NZ 或方向不符会被拒绝）；`output_norm_weight`、`recurrent_state`、`a_log`、`dt_bias` 为 FP32；四个索引 / token 计数为 INT32；`scale`、`lower_bound`、`rms_norm_eps` 为 FP32 标量 |
| 输出数据类型 | `out` 为 BF16；`conv_state` 为 BF16、`recurrent_state` 为 FP32（两者原地更新） |
| 支持的形状 | `hidden_states` 为 `[Mpad, 7168]`（调用方按 `MTile` 对齐的 A 操作数），`out` 为 `[B, 8, 7168]`；`B = 模型 batch × TP`；`N ∈ {6, 12, 24}` 为 head 数，`P = N·128`、`H = 7168`、`S = 8` |

未覆盖的情况（当前不支持）：`S != 8`（`hidden_states` 行数必须与 `B·8` 一致）、`H != 7168`、奇数 head 数、
`output_gate_projection_weight` 的列数非 128 的倍数、head 数超过 128，以及六个投影权重以 ND 而非
FRACTAL_NZ 传入。这些都在 host 侧由 wrapper 直接报 `ValueError`，不会静默给出错误结果。输入含
NaN / Inf 不在支持范围内，输出未定义。

## 功能说明

`mega_recurrent_kda` 是 Kimi K3 的融合线性注意力算子，用 PyPTO-Pro 在昇腾 NPU 上实现。一次
调用完成 **QKV 投影 + decay 低秩投影 + beta 投影 + 输出门控投影 + 因果 Conv1D + SiLU +
Q/K RMS 归一化 + recurrent KDA（delta 规则）+ RMSNorm + 输出门控 + 输出投影**：

```text
qkv   = h @ Wqkv                                       # QKV 投影，列序 [Q|K|V]
f_a   = bf16(h @ Wa);  f_b = f_a @ Wb                  # decay 低秩投影（中间结果舍入 BF16）
beta  = sigmoid(h @ Wbeta)                             # 逐 head 的更新强度
z     = h @ Wgate                                      # 输出门控 logits
conv  = silu(w0·h0 + w1·h1 + w2·h2 + w3·cur)           # 因果 K=4 卷积，逐通道
q_hat = q / sqrt(‖q‖² + 1e-6);  k_hat = k / sqrt(‖k‖² + 1e-6)   # Q/K RMS 归一化
decay = lower_bound · sigmoid(exp(a_log) · (f_b + dt_bias))    # 逐 (head, channel) 的衰减
gexp  = exp(decay)
# KDA delta 规则，逐 (batch, head)，状态 S ∈ R^{128×128}
for t:  S ← S ⊙ gexp_t
        r = v_t − S·k_t
        S ← S + (beta_t ⊙ r) ⊗ k_t
        o_t = scale · (S·q_t)
o'    = o · rsqrt(mean(o²) + rms_eps) · gamma           # 输出 RMSNorm
gated = bf16(o' · sigmoid(z))                          # 输出门控
out   = gated @ Wo                                     # 输出投影
```

其中 `h` 是展平成 `B·S` 行的 `hidden_states`（调用方按 `MTile` 补到 `Mpad` 行），`v` 取卷积后的
V 通道，`S` 每个 token 都写回 `recurrent_state`。

如果把这十步拆成多个算子，`qkv`、`proj2`、`raw_decay`、`convolved`、`qg/kg/vg`、`o`、`gated`
都要在全局内存（HBM）与片上之间往返，递归状态还要额外读写。本算子把它们融合成**一个混合
AIC/AIV kernel 的三段执行**：Cube 段做投影，Vector 段做卷积、归一化与递归，Cube 段再做输出
投影，三段用跨核事件衔接，一次 launch 完成。中间量 `gexp / beta_act / gate_act / qhat /
khat / vconv` 常驻 UB（片上统一缓冲），跨段经全局内存的只有 `qkv`、`proj2`、`raw_decay`、
`qg/kg/vg`、`gated` 这几块真正需要跨计算单元交接的张量。

代价是递归在 Vector 上**逐 token 串行**：每个 token 的状态更新依赖上一个 token 的状态，
而每 token 都要把 `128×128` 的状态写回 `recurrent_state`。这部分是硬契约，也是当前的主要
耗时来源（见 `RESULTS.md` 第 3 节）。

## 函数原型

```python
# 把 op_kernel 目录加入 sys.path 后：
from mega_recurrent_kda import mega_recurrent_kda_pad_input, mega_recurrent_kda_wrapper

hidden_pad = mega_recurrent_kda_pad_input(hidden_states)   # [B, S, H] -> [Mpad, H]
out = mega_recurrent_kda_wrapper(
    hidden_pad, qkv_projection_weight, decay_projection_a_weight, decay_projection_b_weight,
    beta_projection_weight, output_gate_projection_weight, output_norm_weight,
    output_projection_weight, conv1d_weight, conv_state, recurrent_state,
    conv_state_indices, ssm_state_indices, conv_num_accepted_tokens, num_accepted_tokens,
    a_log, dt_bias, scale, lower_bound, rms_norm_eps=1e-5)
```

20 个位置参数，投影权重方向为本 kernel 的 `[K, N]`（见参数表）；核数不在签名内。

### 参数说明

| 参数 | 形状 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `hidden_states` | `[Mpad, 7168]` | BF16 ND | 已按 `MTile` 对齐的 A 操作数，`Mpad = ceil(B·S / MTile)·MTile`；用 `mega_recurrent_kda_pad_input` 从 `[B, 8, 7168]` 生成 |
| `qkv_projection_weight` | `[H, 3P]` | BF16 **FRACTAL_NZ** | QKV 投影权重，列序 Q/K/V；逻辑 `[K, N]` |
| `decay_projection_a_weight` | `[H, 128]` | BF16 **FRACTAL_NZ** | decay 低秩投影的第一级 |
| `decay_projection_b_weight` | `[128, P]` | BF16 **FRACTAL_NZ** | decay 低秩投影的第二级 |
| `beta_projection_weight` | `[H, 128]` | BF16 **FRACTAL_NZ** | beta 投影，前 `N` 列有效、其余补零（按 head 取一列） |
| `output_gate_projection_weight` | `[H, P]` | BF16 **FRACTAL_NZ** | 输出门控投影 |
| `output_norm_weight` | `[128]` | FP32 ND | 输出 RMSNorm 的 `gamma`，各 head 共享 |
| `output_projection_weight` | `[P, H]` | BF16 **FRACTAL_NZ** | 输出投影 |
| `conv1d_weight` | `[3,N,4,128]` / `[3,4,128]` / `[4,3P]` | BF16 ND | 三种布局都接受，列序统一为 `[Q(P)\|K(P)\|V(P)]` |
| `conv_state` | `[C, L≥S+2, 3P]` | BF16 ND | 卷积历史状态，**原地更新**；槽 0 不可写 |
| `recurrent_state` | `[R≥B·S, N, 128, 128]` | FP32 ND | KDA 递归状态，**原地更新** |
| `conv_state_indices` | `[B]` | INT32 | 每个 batch 使用的卷积槽号，取值 `0..C-1`；真实 batch 用非零且唯一的值，`0` 表示空槽 |
| `ssm_state_indices` | `[B·8]` | INT32 | 每个 token 使用的递归状态槽号，batch-major token 顺序 |
| `conv_num_accepted_tokens` | `[B]` | INT32 | 卷积历史的起点，`max(c-1, 0)` |
| `num_accepted_tokens` | `[B]` | INT32 | 选择初始递归状态的 token 数 |
| `a_log` | `[N]` | FP32 ND | 衰减的逐 head 对数尺度 |
| `dt_bias` | `[N, 128]` | FP32 ND | 衰减的逐 (head, channel) 偏置 |
| `scale` | 标量 | FP32 | 输出缩放，通常 `128 ** -0.5` |
| `lower_bound` | 标量 | FP32 | 衰减的下界，通常 `-5.0` |
| `rms_norm_eps` | 标量 | float | 输出 RMSNorm 的稳定项，默认 `1e-5` |

### 返回值说明

| 返回值 | 形状 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `out` | `[B, 8, 7168]` | BF16 | 融合算子的输出；每次调用新分配，kernel 直接写入，不再做额外的 D2D 拷贝 |
| `conv_state` | 原地 | BF16 | 写回更新后的卷积历史；`conv_state_indices[b] == 0` 的行不改动 |
| `recurrent_state` | 原地 | FP32 | 每个 token 的 `128×128` 状态都写回；槽 0 保持不变 |

## 约束说明

- `S` 必须为 8、`H` 必须为 7168、head 数必须为偶数（已验证的取值为 6 / 12 / 24，对应
  TP16 / TP8 / TP4），否则 wrapper 直接报 `ValueError`，不做静默降级。
- 六个投影权重必须是 **BF16 FRACTAL_NZ allocation-root** 张量，逻辑形状为 `[K, N]`（归约维在前、
  输出通道在后），即 kernel 直接消费的方向；传 ND 或传成 `[N, K]` 都会抛 `ValueError`。
  六块全部按原样直传，Cube 的 GM→L1 是纯 NZ→NZ 直拷贝，宿主不做任何格式转换。
- `beta_projection_weight` 必须补零到 `[H, 128]`：kernel 的 proj2 N-tile 固定 128 宽，beta 块占
  其中一列 tile，只读前 `N` 列；补零列的值不影响结果，但物理上必须存在。
- `conv1d_weight` 接受 `[3,N,4,128]`（Q/K/V、head、tap、channel）、head 共享的 `[3,4,128]`
  与扁平 `[4,3*N*128]`，内部统一重排成 `[4, 3P]`、列序 `[Q(P)|K(P)|V(P)]`。
- `hidden_states` 必须是调用方已 padding 的 `[Mpad, 7168]`，`Mpad = ceil(B·S / MTile)·MTile`
  （`MTile ∈ {32, 64, 128}`，由 `B·S` 选档）；wrapper 不做任何 pad 或拷贝，形状不对直接报错。
  `mega_recurrent_kda_pad_input` 可以完成这一步，`B·S` 已对齐时返回零拷贝视图。
- `conv_state_indices[b] == 0` 表示不可写的空槽：该行不改动 `conv_state`，卷积输出全 0；
  两个状态池的槽 0 在任何情况下都保持逐位不变。
- `S`、`H`、`N` 与 `B` 都是运行时量，同一个 kernel 定义服务全部形状；只有 `MTile`
  （32 / 64 / 128）随 `B·S` 在三档编译期特化间选择，换档会触发一次重编译（见 `DESIGN.md` 第 1 节）。
- 全部张量需同设备且连续；`conv_state` 与 `recurrent_state` 必须由调用方提供可写张量。
- 核数不在签名内；支持的 AIC 配额固定为 **8 / 16 / 24 / 28 / 32**，默认取设备上可用的
  最大配额（由 `pypto_pro.runtime.platform.get_platform_info().core_num` 读取：Ascend 950 有多种型号，28 核 SKU 默认 28，32 核及以上设备默认 32）。可用 `MEGA_KDA_CUBE_CORES` 在支持集合内
  覆盖，其它值在 host 侧报 `ValueError`；AIV 数恒为其 2 倍。
- 输出 `out` 是每次调用新分配的张量，不别名跨调用缓存，调用方可长期持有。
- 单次调用只下发一个 kernel；host 侧只做参数校验和缓存查找，不做权重格式转换、A 操作数 padding 或输出拷贝。
- 已知非契约差异：`out` 的 K 段累加走 `AtomicAdd`，同一输入两次运行的输出可有 1–2 个 BF16
  ULP 的差异，远小于 `5e-3` 容差。

## 调用示例

```python
import sys

import torch
import torch_npu

sys.path.insert(0, "ops/pro/kimi_k3/mega_recurrent_kda/op_kernel")
from mega_recurrent_kda import mega_recurrent_kda_pad_input, mega_recurrent_kda_wrapper

torch.npu.set_device(0)

B, S, H, N, D = 4, 8, 7168, 24, 128
P = N * D
C, L = B + 1, S + 2
nz = torch_npu.Format.FRACTAL_NZ


def bf16(shape):
    return (torch.randn(shape) * 0.02).bfloat16()


def nz_weight(shape):
    return torch_npu.npu_format_cast(bf16(shape).npu(), nz)


hidden = bf16((B, S, H)).npu()
hidden = mega_recurrent_kda_pad_input(hidden)
wqkv = nz_weight((H, 3 * P))
wa = nz_weight((H, D))
wb = nz_weight((D, P))
wbeta = nz_weight((H, 128))   # 只有前 N 列被读取
wgate = nz_weight((H, P))
wo = nz_weight((P, H))
norm_weight = (torch.randn(D) * 0.2).float().npu()
conv_weight = bf16((4, 3 * P)).npu()

conv_state = bf16((C, L, 3 * P)).npu()
recurrent_state = (torch.randn(B * S + 1, N, D, D) * 0.002).float().npu()
conv_state_indices = (torch.randperm(B) + 1).int().npu()
ssm_state_indices = (torch.randperm(B * S) + 1).int().npu()
conv_num_accepted = (torch.arange(B) % (S + 1)).int().npu()
num_accepted = (S - conv_num_accepted).int().npu()
a_log = (torch.randn(N) * 0.1).float().npu()
dt_bias = (torch.randn(N, D) * 0.1).float().npu()

out = mega_recurrent_kda_wrapper(
    hidden, wqkv, wa, wb, wbeta, wgate, norm_weight, wo, conv_weight,
    conv_state, recurrent_state, conv_state_indices, ssm_state_indices,
    conv_num_accepted, num_accepted, a_log, dt_bias,
    128 ** -0.5, -5.0)

torch.npu.synchronize()
print(out.shape, out.dtype)
# torch.Size([4, 8, 7168]) torch.bfloat16
print(conv_state.shape, recurrent_state.shape)
# torch.Size([5, 10, 9216]) torch.Size([33, 24, 128, 128])
```

示例依赖的文件分布在算子目录下（`op_kernel/` 放 kernel 与公开 wrapper，`tests/assets/` 放
golden 与精度比较器，`tests/` 直接放测试与基准入口）：

```text
mega_recurrent_kda/
├── README.md                           # 本文件：接口与用法
├── DESIGN.md                           # 设计说明：阶段划分、切分与流水
├── RESULTS.md                          # 结果说明：用例、精度与性能
├── op_kernel/
│   └── mega_recurrent_kda.py           # 算子本体（mixed kernel + 公开 wrapper）
└── tests/
    ├── test_mega_recurrent_kda.py      # 精度/契约测试 + 计时入口（--bench）
    └── assets/
        ├── mega_recurrent_kda_golden_cpu.py   # 独立 CPU 参考实现
        └── mega_recurrent_kda_precision.py    # 精度比较器（方案 A）
```

测试与计时都可以在仓库根目录直接运行，不需要额外的驱动脚本：

```bash
# 精度用例（pytest 会按 soc 标记过滤）
pytest ops/pro/kimi_k3/mega_recurrent_kda/tests/test_mega_recurrent_kda.py -q

# 只跑其中一个用例
pytest ops/pro/kimi_k3/mega_recurrent_kda/tests/test_mega_recurrent_kda.py -q -k tp4_main_b32

# 直接运行（不经过 pytest）
python ops/pro/kimi_k3/mega_recurrent_kda/tests/test_mega_recurrent_kda.py

# 计时（配额默认取设备支持的最大配额，可用 MEGA_KDA_CUBE_CORES 指定）
MEGA_KDA_CUBE_CORES=24 python ops/pro/kimi_k3/mega_recurrent_kda/tests/test_mega_recurrent_kda.py --bench
```
