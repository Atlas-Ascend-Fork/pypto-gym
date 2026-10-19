# EngramGateBwd（PyPTO-Pro）

`engram_gate_bwd` 是 deepseek_v4_1 中 Engram residual gate 正向算子 `engram_gate` 的**解析反向**，
用 PyPTO-Pro 在昇腾 950 上以**单 kernel** 实现。

- 实现（kernel + host wrapper）：`op_kernel/engram_gate_bwd.py`
- 测试与精度基准：`tests/`（golden 与精度对比脚本在 `tests/assets/`）
- 设计说明：`DESIGN.md`；用例、精度标准与性能结果：`RESULTS.md`

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :--- |
| Ascend 950PR（DAV_3510，32 AIC / 64 AIV） | 支持 |
| 其它昇腾产品 | 不支持 |

kernel 按 arch 3510 编译，启动核数在运行时取自平台信息，未针对其它 SoC 做过验证。

## 功能说明

先约定几个记号：`t` 是 token 数，`hc` 是同一个 token 上并行的路数（本算子固定为 4），
`dim` 是每路的特征宽度；`image_mask` 标记的是「正向不参与门控、原样透传」的那些 token。

正向算子 `engram_gate` 对每个 `(token, hc)` 行做双路 RMS、加权点积、signed-sqrt sigmoid 门控，
再把门控值乘到 `value` 上做残差更新：

```text
rstd_x  = rsqrt(mean(x²) + eps)
rstd_k  = rsqrt(mean(key²) + eps)
raw_dot = sum_d(x · weight · key)
dot     = raw_dot · rstd_x · rstd_k / sqrt(dim)
gate    = sigmoid(copysign(sqrt(max(|dot|, clamp_value)), dot))
y       = bf16(x + gate · value)          # image_mask 为 True 的 token：y = x
```

`engram_gate_bwd` 接收 `y` 的梯度 `grad_out`，按同一套公式求反向，输出对 `x`、`key`、
`value`、`weight` 的四个梯度：

```text
dgate = sum_d(grad_out · value)
beta  = dgate · gate · (1 - gate) · 0.5 · sqrt(rstd_x · rstd_k / sqrt(dim) / |raw_dot|)
        （|dot| < clamp_value 时为 0；image token 为 0）
c     = -beta · raw_dot / dim

grad_x      = grad_out + beta · weight · key + c · rstd_x² · x
grad_key    =            beta · weight · x   + c · rstd_k² · key
grad_value  = sum_hc(gate · grad_out)
grad_weight = sum_t(beta · x · key)
```

正向把 `rstd_x`、`rstd_k`、`raw_dot`、`gate` 这四个逐 `(token, hc)` 的中间量保存下来供反向使用
（与 `torch.autograd` 保存张量的约定一致），因此本算子把它们作为**输入**接收，不重复计算。

`image_mask` 为 True 的 token 在正向被原样透传（`y = x`），反向对应地
`grad_x = grad_out`（逐位相等）、`grad_key = grad_value = 0`，且这些 token 对 `grad_weight` 没有贡献。

## 函数原型

本算子以模块形式提供，公开入口为 `engram_gate_bwd_wrapper`：

```python
from engram_gate_bwd import engram_gate_bwd_wrapper

grad_x, grad_key, grad_value, grad_weight = engram_gate_bwd_wrapper(
    grad_out, x, key, value, weight,
    rstd_x, rstd_k, raw_dot, gate, image_mask,
    clamp_value=1e-6,
)
```

### 参数说明

| 参数 | 类型 / shape | 必填 | 说明 |
| :---- | :---- | :---- | :---- |
| `grad_out` | BF16 `[t, hc, dim]` | 是 | 正向输出 `y` 的梯度 |
| `x` | BF16 `[t, hc, dim]` | 是 | 正向的 query 激活 |
| `key` | BF16 `[t, hc, dim]` | 是 | 正向的 key 激活 |
| `value` | BF16 `[t, dim]` | 是 | 正向的 value，按 token 共享，不随 `hc` 变化 |
| `weight` | FP32 `[hc, dim]` | 是 | 正向的逐 `(hc, dim)` 权重（即正向点积里用的那个权重） |
| `rstd_x` | FP32 `[t, hc]` | 是 | 正向保存的 `rsqrt(mean(x²) + eps)` |
| `rstd_k` | FP32 `[t, hc]` | 是 | 正向保存的 `rsqrt(mean(key²) + eps)` |
| `raw_dot` | FP32 `[t, hc]` | 是 | 正向保存的 `sum_d(x · weight · key)`（未乘 `rstd` 与 `1/sqrt(dim)`） |
| `gate` | FP32 `[t, hc]` | 是 | 正向保存的门控值 |
| `image_mask` | BOOL `[t]` | 是 | `True` 表示该 token 是 image token，正向原样透传 |
| `clamp_value` | float | 否 | `|dot|` 的软阈值，默认 `1e-6`；kernel 内是编译期常量，只接受该默认值 |

### 返回值说明

返回 `(grad_x, grad_key, grad_value, grad_weight)` 四个张量：

| 返回值 | 类型 / shape | 说明 |
| :---- | :---- | :---- |
| `grad_x` | BF16 `[t, hc, dim]` | 与 `x` 同形状 |
| `grad_key` | BF16 `[t, hc, dim]` | 与 `key` 同形状 |
| `grad_value` | BF16 `[t, dim]` | 与 `value` 同形状 |
| `grad_weight` | FP32 `[hc, dim]` | 与 `weight` 同形状 |

`grad_weight` 由 per-core 的部分和合并得到（框架没有跨核原子加），因此是 FP32。

## 约束说明

- `hc` 固定为 4（kernel 按 `hc = 4` 编译，其它值会被 wrapper 拒绝）。
- `dim` 必须满足 `0 < dim <= 7168` 且 `dim % 512 == 0`（`512 = hc × 128`：整行按 128 元素步长扫描，
  `grad_value` 的四分之一是 `dim / hc`，也要是 128 的整数倍）；`7168` 是编译期固定的 UB 行宽。
  不满足时 wrapper 会直接报错，不会静默算错。
- `t > 0`；所有输入必须在同一个 NPU 上，`grad_out`/`x`/`key` 为 `[t, hc, dim]`、
  `value` 为 `[t, dim]`、`weight` 为 `[hc, dim]`、`image_mask` 为 `[t]`，且要互相自洽。
- `grad_out` / `x` / `key` / `value` / `weight` / `image_mask` 必须是**连续**张量。kernel 按 shape
  算地址、不读 stride，非连续视图会被读错，因此 wrapper 会直接报错而不是给出错误结果。
- `rstd_x` / `rstd_k` / `raw_dot` / `gate` 必须是 `[t, hc]` 的 FP32 张量，且要与 `x`/`key`/`weight`
  来自同一次正向计算。这四个由 wrapper 打包成一块连续缓冲，所以它们本身不需要连续。
- 只支持 `clamp_value = 1e-6`（该值在 kernel 内是编译期常量）。

## 调用示例

```python
import torch
import torch_npu  # noqa: F401

from engram_gate_bwd import engram_gate_bwd_wrapper

t, hc, dim = 512, 4, 2048
dev = "npu"

grad_out = torch.randn(t, hc, dim, dtype=torch.bfloat16, device=dev)
x = torch.randn(t, hc, dim, dtype=torch.bfloat16, device=dev)
key = torch.randn(t, hc, dim, dtype=torch.bfloat16, device=dev)
value = torch.randn(t, dim, dtype=torch.bfloat16, device=dev)
weight = torch.randn(hc, dim, dtype=torch.float32, device=dev)

# 这四个是正向保存下来的中间量；这里按正向公式重算一份用于演示。
xf, kf = x.float(), key.float()
rstd_x = torch.rsqrt(xf.pow(2).mean(-1) + 1e-6)
rstd_k = torch.rsqrt(kf.pow(2).mean(-1) + 1e-6)
raw_dot = (xf * weight.unsqueeze(0) * kf).sum(-1)
dot = raw_dot * rstd_x * rstd_k * (dim ** -0.5)
gate = (dot.abs().clamp_min(1e-6).sqrt() * dot.sign()).sigmoid()
image_mask = torch.zeros(t, dtype=torch.bool, device=dev)

grad_x, grad_key, grad_value, grad_weight = engram_gate_bwd_wrapper(
    grad_out, x, key, value, weight,
    rstd_x, rstd_k, raw_dot, gate, image_mask,
)
torch.npu.synchronize()
print(grad_x.shape, grad_key.shape, grad_value.shape, grad_weight.shape)
# torch.Size([512, 4, 2048]) torch.Size([512, 4, 2048]) torch.Size([512, 2048]) torch.Size([4, 2048])
```
