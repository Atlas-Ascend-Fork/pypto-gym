# liger_cross_entropy

## 概述

`liger_cross_entropy` 是 [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) 融合交叉熵损失的 **正向 + 反向** 算子对，基于 `pypto_pro` DSL 在 Ascend NPU 上实现，当前提供 **float32** 变体。

算子的关键设计沿用上游 triton 实现的省显存技巧：**正向直接把梯度中间量原地写回 logits**，因此反向只是对它做一次链式缩放，不需要重算 softmax，也不需要额外保存一份 `[BT, V]` 张量。

- **正向**为向量侧两趟流式结构。第一趟按行计算 online softmax 统计量并写入 workspace，经一次跨核 rendezvous 让所有 lane 拿到全局归一化因子（`n_non_ignore` / `sum_non_ignore_weight`），第二趟写回梯度；首次出现的 argmax 只在调用方要 `token_accuracy` 或 `predicted_tokens` 时才扫描。两趟都用双缓冲承载 GM tile，下一列块的搬运与当前列块的向量计算重叠。
- **反向**为纯带宽型逐元素算子，一读一写，使用 `auto_mutex` + tile group 流水，实测达到该卡对齐平面屋顶的约 85%。

## 接口签名

四个公开入口，与 `ops/pypto_tensor/llada2/liger_cross_entropy` 的适配层同名同签名，
两个 backend 可以直接互换：

```python
from ops.pypto_pro.llada2.liger_cross_entropy_loss import (
    CrossEntropyOutput,        # 需要可选输出时的返回类型
    LigerCrossEntropyFunction, # torch.autograd.Function
    liger_cross_entropy,       # functional 形式
    LigerCrossEntropyLoss,     # nn.Module 形式
)
```

日常用法就是普通 autograd：

```python
loss = liger_cross_entropy(logits, target, reduction="mean")
loss.backward()                      # 梯度落在 logits.grad

criterion = LigerCrossEntropyLoss(label_smoothing=0.1)
out = criterion(logits, target)
```

需要直接调 kernel 时，两个 wrapper 的签名如下：

```python
from ops.pypto_pro.llada2.liger_cross_entropy_loss.liger_cross_entropy_loss_fwd_impl import (
    liger_cross_entropy_loss_fwd_wrapper,
)
from ops.pypto_pro.llada2.liger_cross_entropy_loss.liger_cross_entropy_loss_bwd_impl import (
    liger_cross_entropy_loss_bwd_wrapper,
)

loss, z_loss, token_accuracy, predicted_tokens, saved_input = \
    liger_cross_entropy_loss_fwd_wrapper(
        logits,                  # [BT, V]  FP32，contiguous，原地写回梯度中间量
        target,                  # [BT]     INT64
        weight=None,             # [V]      FP32，可选类别权重
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",        # "mean" | "sum" | "none"
        softcap=None,
        return_z_loss=False,
        return_token_accuracy=False,
        return_predicted_tokens=False,
    )

dx = liger_cross_entropy_loss_bwd_wrapper(
    saved_input,             # [BT, V]  FP32，即正向的第五个返回值
    grad_output,             # 标量、0 维张量，或 [BT] 向量
)
```

`saved_input` 是梯度中间量所在的 buffer。本 kernel 原地写回，所以它**就是** `logits`
本身（按 `[BT, V]` 视图），不是第二份分配 —— 这正是 `LigerCrossEntropyFunction` 存进
context 里的东西，也是省掉一份 `[BT, V]` 的原因。无论 `requires_grad` 与否都会返回，
由调用方决定留不留。

反向**没有** `inplace` 参数：返回哪块 buffer 由 `grad_output` 的形态决定，与上游 triton
一致 —— 见下表。

### 正向输入

| 名称                 | 形状       | 类型   | 说明                                                     |
|----------------------|------------|--------|----------------------------------------------------------|
| `logits`             | `[BT, V]`  | FP32   | 输入 logits，必须 contiguous；`requires_grad` 为真时被梯度中间量原地覆盖，为假时原值写回、但被 `ignore_index` 命中的行仍置零（与上游一致） |
| `target`             | `[BT]`     | INT64  | 类别索引，取值 `[0, V)` 或等于 `ignore_index`            |
| `weight`             | `[V]`      | FP32   | 可选的按类别权重                                         |
| `ignore_index`       | 标量       | INT    | 该值对应的行对所有输出贡献为零，其 `dx` 行被置零         |
| `lse_square_scale`   | 标量       | FLOAT  | z-loss 系数，为训练稳定性加上 `scale * lse²`             |
| `label_smoothing`    | 标量       | FLOAT  | 标签平滑系数，取值 `[0, 1]`                              |
| `reduction`          | 字符串     | —      | `"mean"` / `"sum"` / `"none"`                            |
| `softcap`            | 标量       | FLOAT  | 为真时先做 `softcap * tanh(x / softcap)`，且反向带链式项 |

### 正向输出

| 名称                | 形状                        | 类型   | 说明                                                   |
|---------------------|-----------------------------|--------|--------------------------------------------------------|
| `loss`              | 标量 或 `[BT]`              | FP32   | `reduction="none"` 时为逐行                            |
| `z_loss`            | 标量 或 `[BT]`              | FP32   | 仅在 `return_z_loss` 时返回                            |
| `token_accuracy`    | 标量 或 `[BT]`              | FP32   | 仅在 `return_token_accuracy` 时返回                    |
| `predicted_tokens`  | `[BT]`                      | INT64  | 逐 token argmax，忽略行为 `-1`                         |
| （原地）`logits`    | `[BT, V]`                   | FP32   | 梯度中间量，交给反向                                   |

### 反向的 `grad_output` 三种形态

| 形态             | 来源                        | 行为                                     | 返回 buffer |
|------------------|-----------------------------|------------------------------------------|-------------|
| 恰好 `1.0`       | 交叉熵是最后一层            | 短路，不下发 kernel                      | 原输入      |
| 0 维张量         | `reduction` 为 mean / sum   | 每行统一缩放                             | 原地        |
| `[BT]` 向量      | `reduction="none"`          | 第 `r` 行按 `grad_output[r]` 缩放        | 新张量      |

`[BT]` 向量那条返回新张量，是为了对齐上游 —— 它走的是会分配的
`_input * grad_output.unsqueeze(1)`。本 kernel 只会原地写，所以这份拷贝是显式的：
比原地路径多一遍 `[BT, V]` 的读写。

主机侧不把 0 维标量摊成 `[BT]` 向量：`g` 以 `[1, ng]` 传入（`ng` 为 1 或 `BT`），第 `r` 行读取 `min(r, ng - 1)`，一条无分支路径同时服务两种形态。

## 数学公式

对 `x[BT, V]` 的每一行 `r`，所有中间量为 FP32：

```
x_cap = softcap * tanh(x / softcap)                   (关闭时为恒等)
m     = max(x_cap)   d = sum(exp(x_cap - m))   lse = m + ln(d)
loss  = ((lse - x_cap[y]) * w[y]) * (1 - ls)
        + (-eps * sum(x_cap * w) + eps * lse * sum(w))
z     = lse_square_scale * lse²
loss  = loss / D_loss + z / D_z                       (仅 mean，否则 D = 1)

dx    = A * exp(x_cap - lse) - B * w,  dx[y] -= C
dx   *= 1 - (x_cap / softcap)²                        (softcap 链式项)
dx   *= grad_output[r]                                (反向)
```

其中 `eps = ls / V`，`D_loss = sum_non_ignore_weight`，`D_z = n_non_ignore`。

三个逐行系数（`w_y` 为真类权重，`sw = sum(w)`）：

```
A = (w_y * (1 - ls) + eps * sw) / D_loss + 2 * lse_square_scale * lse / D_z
B =  eps / D_loss
C = (w_y * (1 - ls)) / D_loss
```

被 `ignore_index` 命中的行不走这条公式：它的 loss 与 z-loss 直接取零，`dx` 整行在梯度趟统一置零，等价于上游 triton kernel 的提前返回。

不带权重时无需单独分支：代入 `w = 1` 即可精确退化为无权重公式（`w_y = 1`、`sum(w) = V`、`eps * sum(w) = ls`），因此 kernel 只保留一条带权路径，并把权重 tile 预填为 1。

## 测试

```bash
pytest tests/ops/pypto_pro/llada2/liger_cross_entropy/test_liger_cross_entropy_pro.py -v
```

`test_liger_cross_entropy_pro.py` 一个文件，分四段：

| 段 | 内容 |
|------|------|
| golden | 纯 torch 参考实现（正反向）。它带着 `torch.nn.CrossEntropyLoss` 没有的部分：z-loss、token accuracy、首次出现的 predicted tokens，以及正向留在输入 buffer 里的梯度中间量 |
| 正向 | 上游 shape 集 + 宽词表集；12 组语义开关组合；argmax 关闭时的梯度校验（V > 64） |
| 反向 | 同样的 shape 集，逐位相等；`grad_output` 全部形态与各自的 buffer 契约；正反向串联（跨 1 / 2 / 4 个 64 lane 分组三种词表宽度） |
| autograd | `loss.backward()` 打通到反向 kernel，三种 reduction 各自触发对应的 `grad_output` 形态；functional / Module 的返回契约与参数校验 |

**两个参照，是有意的**：golden 覆盖完整的上游语义，但它对语义的理解与 kernel 同源；所以串联与 autograd 两段改对照 `F.cross_entropy`，它与两者都不共用代码，能抓住 kernel 与 golden 共同误读上游的情况。代价是它没有 z-loss 项，因此 `lse_square_scale` 只由 golden 覆盖。

## 性能

Ascend950PR，card 0，CANN 9.2.0，测量时 `block_dim = 64`，取 3 轮 sweep 的逐格最小值，单位微秒/次。

| Shape | 正向 µs | 正向 GB/s | 反向 µs | 反向 GB/s |
|---|---:|---:|---:|---:|
| BT=8192  V=32000  | 2455.2 | 1281 | 1469.9 | 1427 |
| BT=2048  V=32000  |  719.8 | 1093 |  374.8 | 1399 |
| BT=1002  V=157184 | 1435.1 | 1317 |  891.1 | 1414 |
| BT=810   V=157184 | 1200.9 | 1272 |  719.3 | 1416 |
| BT=618   V=157184 |  956.8 | 1218 |  551.2 | 1410 |
| BT=391   V=157184 |  686.8 | 1074 |  358.0 | 1373 |

`GB/s` 只计 logits 的搬运：正向流三遍（统计、梯度、写回），反向一读一写。

反向为带宽受限，稳定在 1373–1427 GB/s，约为该卡对齐平面屋顶（~1.7 TB/s）的 85%；正向不在带宽曲线上 —— 它每元素两次 `exp`（统计趟的 online softmax 一次，梯度趟一次），开启 softcap 再加一次 `exp` 与一次除法，要 metrics 再加一次比较与选择。换 BF16 让字节减半后耗时反而略增，可佐证它受限于向量计算而非带宽。

### 正向的一次优化

宽词表场景（`V = 157184`）下正向做过一轮优化，同卡同方法（3 轮逐格最小值）前后对比：

| Shape | 优化前 µs | 优化后 µs | 加速 |
|---|---:|---:|---:|
| BT=391  V=157184 |  824.0 |  686.8 | 1.20x |
| BT=618  V=157184 | 1177.8 |  956.8 | 1.23x |
| BT=810  V=157184 | 1516.5 | 1200.9 | 1.26x |
| BT=1002 V=157184 | 1845.3 | 1435.1 | 1.29x |

四处改动，都在正向：统计趟不再多写一份用不到的 fp32 tile；softcap 链式项与 argmax 扫描改为按零/一次的循环次数执行，而不是无条件执行；两趟流式结构从"每个 tile 一次全局 barrier"换成双缓冲流水，让搬运与计算重叠。反向未改动。

反向的数字在这次重测中相对上一次波动 ±2%，与代码无关 —— 它一个字节都没改，可以拿来当这台机器当时负载的对照量。

## 已知限制

- **仅 FP32。** BF16 / FP16 变体尚未随本次提交一并上库。
- **动态选择 block 数。** 正反向均按 `min(BT, get_platform_info().vector_core_num)` 启动。正向跨两次 `INTER_BLOCK` rendezvous，不能超过当前设备的 Vector 核数，否则未能共同驻留的 block 无法到达屏障。
- **全部行被 ignore 时**，`mean` 归约为 `0/0`：梯度按上游约定给全零（与 `F.cross_entropy` 的梯度一致），但 loss 为 0 而非 `NaN`。
- **`logits` / `saved_input` 必须 contiguous。** 梯度中间量原地写回的是调用方那块 buffer，非 contiguous 的输入在下发路径上会被复制，kernel 写到副本上，调用方拿回原值 —— loss 看着正常而梯度悄悄丢失。两个 wrapper 都会直接报错而不是默默复制（反向仅在 `inplace=True` 时检查）。
- **kernel 主体为机器生成**，请勿手工修改 `@pl.jit` 函数体。
- **`lse_square_scale` 不在 autograd 用例的对照范围内**：那一层拿 `F.cross_entropy`
  当独立参照，而它没有 z-loss 项，非零 scale 下 loss 与梯度都会合理地不同。该开关由
  正向用例对照 golden 覆盖。

## 重新生成 kernel 主体

两个 impl 文件里 `@pl.vector_function` 与 `@pl.jit` 的函数体不是手写的。它们的来源是一份 EasyASC kernel，经 `easyasc.targets.pypto` 转译为 `pypto_pro` 源码后，与本文件同目录的 host wrapper 拼接而成。生成器不在本仓库内，流程是：

1. 在 EasyASC 侧把 kernel 以 `globvars.target = "pypto"` 追踪一遍；
2. 调用 `easyasc.targets.pypto.transpiler.transpile(kern, list(kern.used_micros), jit_name)`，它返回 `(source, gaps, ...)`，`gaps` 非空表示存在没有映射的转译路径，必须先补齐；
3. 把 `source` 原样贴回 impl 文件，只替换 `@pl.jit` 函数名（`liger_cross_entropy_forward_kernel` / `liger_cross_entropy_backward_kernel`），并保留文件末尾的 host wrapper；
4. 重跑本目录下的测试。

`split_workspace` 在 AscendC 侧是 kernel 内部分配，`pypto_pro` 没有对应物，所以转译后的签名会在公开张量之后追加每个 workspace 一个 `pl.Tensor`。wrapper 里的 `_FWD_WORKSPACES` 必须与之一致。
