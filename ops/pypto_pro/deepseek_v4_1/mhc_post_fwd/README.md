# MHCPostFwd（PyPTO-Pro）

mHC（manifold hyper-connection）后处理前向：把本层新算出的隐藏状态 `x` 与 4 份历史残差
`residual`，用两组按 token 变化的学习系数重新混合成 4 份输出。纯逐元素 + 定长小归约的
向量算子（无矩阵乘）。

## 产品支持情况

| 产品 | 是否支持 |
|---|---|
| Ascend 950PR（`Ascend950PR_9589`，32 Cube / 64 Vector） | **支持**（本机实测：精度 8/8 P0 通过，性能见 `RESULTS.md`） |
| 其它产品 | 不支持（kernel 按 A5 的 UB 容量与 64-lane 向量寄存器宽度编译） |

## 功能说明

对每个 token `n`（展平后的 `(b, s)`）、每个输出份 `mo ∈ [0,4)`、每个隐藏维 `h`：

```text
out[n, mo, h] = post_layer_mix[n, mo, 0] * x[n, h]
                + sum_{mi=0}^{3} comb_res_mix[n, mi, mo] * residual[n, mi, h]
```

求和只跨 `mi`，**不跨 token、不跨 hidden**；`h` 只承担逐元素索引。`x`/`residual`/`out` 为
BF16，`post_layer_mix`/`comb_res_mix` 为 FP32：乘积与 4 项求和在 FP32 寄存器中累加，
落盘前**一次性**舍入到 BF16。

形状记号（本仓文档统一）：`AxBxC` 表示 `n0 = A`、`n1 = B`、`h = C`。
`n0` 为 batch 维、`n1` 为 sequence 维、`h` 为 hidden 维（`x` 的第 3 轴，也是 `residual`/`out`
的第 4 轴）；`mhc = 4` 固定，为 `residual`/`out` 的中间轴。用例清单见 `RESULTS.md`。

## 函数原型

```python
mhc_post_fwd(x, residual, post_layer_mix, comb_res_mix, out=None) -> torch.Tensor
```

## 参数说明

| 参数 | 方向 | 形状 | dtype | 说明 |
|---|---|---|---|---|
| `x` | 输入 | `[n0, n1, h]` | BF16 | 本层新算出的隐藏状态 |
| `residual` | 输入 | `[n0, n1, 4, h]` | BF16 | 4 份历史残差 |
| `post_layer_mix` | 输入 | `[n0, n1, 4, 1]` | FP32 | 每个 token 的 4 个 `x` 权重 |
| `comb_res_mix` | 输入 | `[n0, n1, 4, 4]` | FP32 | 每个 token 的 4×4 混合矩阵，下标序 `(mi, mo)` |
| `out` | 输出 | `[n0, n1, 4, h]` | BF16 | 可选。省略时按 `residual` 形状新建；传入时**原地填充** |

`n0`/`n1` 任意；`mhc = 4` 固定；`h` 为 64 的倍数。

## 返回值说明

返回 `out`（`BF16[n0, n1, 4, h]`）。传入 `out` 时返回同一张量（`data_ptr` 不变），否则返回新建张量。

## 约束说明

1. `h` 必须是 **64 的倍数**（向量寄存器 lane 宽）；`mhc` 固定为 4。
2. 四入一出必须**连续存储**（`is_contiguous()`）；kernel 的扁平 token 视图按显式连续 stride 构造，
   wrapper 不做 `.contiguous()` 兜底，不满足即报 `ValueError`。
3. dtype 固定：`x`/`residual`/`out` 为 BF16，`post_layer_mix`/`comb_res_mix` 为 FP32。
4. `out` 传入时形状/dtype/device 必须与 `residual` 一致。
5. wrapper **不含任何 host 侧形状处理**（不 reshape/view/cast/`.contiguous()`），token 轴展平
   由核内 `pl.make_tensor` 完成；单次调用只启动一个 `@pl.jit` kernel。
6. 参数非法时抛 `ValueError`（用 `raise` 而非 `assert`，`python -O` 下依然生效）。

## 调用示例

```python
import torch
import torch_npu
from mhc_post_fwd import mhc_post_fwd          # op_kernel 在 sys.path 上

n0, n1, h = 2, 4096, 7168
x        = torch.randn(n0, n1, h,        dtype=torch.bfloat16, device="npu:0")
residual = torch.randn(n0, n1, 4, h,     dtype=torch.bfloat16, device="npu:0")
post     = torch.randn(n0, n1, 4, 1,     dtype=torch.float32,  device="npu:0")
comb     = torch.randn(n0, n1, 4, 4,     dtype=torch.float32,  device="npu:0")

out = mhc_post_fwd(x, residual, post, comb)    # 也可传入预置 out 原地填充
```

运行测试与复现见 `RESULTS.md`；设计与切分见 `DESIGN.md`。
