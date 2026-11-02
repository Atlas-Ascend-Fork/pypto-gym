# CompressorV2（PyPTO-Pro）

## 产品支持情况

本实现面向 Ascend 950PR，使用矩阵计算单元（Cube）与向量计算单元（Vector）协同执行。启动核数从平台信息读取，调用前需配置匹配的 CANN、PyPTO-Pro 和 torch_npu 运行环境。

## 功能说明

CompressorV2 将连续若干个 token 的投影结果压缩为一个向量，并原位更新历史状态缓存。记输入特征宽度为 `H`、输出特征宽度为 `D`、压缩率为 `r = cmp_ratio`。每个 token 先计算两路投影：

```text
K = X @ wkv.T
G = X @ wgate.T
```

两路权重均按 `[D, H]` 存储。FP16 和 BF16 输入直接参与矩阵乘，投影以 FP32 累加；沿 H 分区计算时，各分区的 FP32 部分和先合并，再用于压缩与缓存更新。

随后按绝对位置划分长度为 `r` 的组，在每个通道内沿 token 轴进行 softmax 加权：

```text
m[d] = max_i G[i, d]
a[i, d] = exp(G[i, d] - m[d])
Y[d] = sum_i(a[i, d] * K[i, d]) / sum_i a[i, d]
```

`i` 遍历一个完整组内的 `r` 个 token。实现先将 KV 乘以 `1 / r`，加权归一化后再乘回 `r`；指数、加权累加和除法均使用 FP32。输出转换为与 `x` 相同的类型，舍入到最近的可表示值，中点取偶数。

`start_pos[b]` 为第 `b` 条序列本次调用的绝对起点。若起点未对齐组边界，首组先从调用前缓存读取 `start_pos[b] % r` 个历史 token，再拼接本次新输入。仅完整组产生有效输出；末尾不足一组不产生有效输出，其投影仍参与缓存更新。

缓存采用环形寻址：第 `b` 条序列使用物理块 `state_block_table[b]`，绝对位置 `p` 对应槽位 `p % block_size`。前 `D` 个通道保存 `K`，后 `D` 个通道保存 `G`。每条序列仅写入本次最后 `min(L_b, block_size)` 个 token 的投影，`L_b` 为实际参与计算的新输入长度；其余槽位及未使用的物理块保持原值。

## 函数原型

公开入口位于 `op_kernel/compressor_v2.py`，同一模块包含 kernel、输入校验、输出分配和启动逻辑：

```python
def compressor(
    x, wkv, wgate, state_cache, cmp_ratio=4, *,
    state_block_table=None, cu_seqlens=None,
    seqused=None, start_pos=None,
):
    ...
```

`state_block_table`、`cu_seqlens`、`seqused` 和 `start_pos` 只能按关键字传入。

### 参数说明

| 参数 | 类型与形状 | 说明 |
| :--- | :--- | :--- |
| `x` | FP16 或 BF16，`[B, S, H]` 或 `[T, H]` | 输入激活。`B` 为序列数，`S` 为每条序列本次输入长度，`T` 为所有序列本次输入长度之和。 |
| `wkv` | 与 `x` 同类型，`[D, H]` | KV 投影权重。 |
| `wgate` | 与 `x` 同类型，`[D, H]` | softmax 分数的投影权重。 |
| `state_cache` | FP32，`[block_count, block_size, 2 * D]` | 可读写的历史状态，调用时原位更新。 |
| `cmp_ratio` | 整数，默认 `4` | 一个完整压缩组包含的 token 数。 |
| `state_block_table` | INT32，`[B]` | 序列到缓存物理块的映射，必须显式传入。 |
| `cu_seqlens` | INT32，`[B + 1]` 或 `None` | `[T, H]` 输入必填的序列长度前缀和；`[B, S, H]` 输入必须为 `None`。 |
| `seqused` | INT32，`[B]` 或 `None` | 各序列实际参与压缩和缓存更新的新输入 token 数；默认使用各序列完整长度。 |
| `start_pos` | INT32，`[B]` 或 `None` | 各序列非负的绝对起点，默认全为 `0`。 |

### 返回值说明

返回一个连续输出张量 `out`，设备与数据类型均与 `x` 相同：

| 输入布局 | 输出分配形状 |
| :--- | :--- |
| `[B, S, H]` | `[B, ceil(S / r), D]` |
| `[T, H]` | `[min(T, T // r + B), D]` |

对第 `b` 条序列，提供 `seqused` 时取 `L_b = seqused[b]`，否则取该序列完整长度。当 `L_b > 0` 时，有效输出组数为 `(start_pos[b] % r + L_b) // r`；当 `L_b = 0` 时为 `0`。省略 `start_pos` 时按 `0` 计算。

`[B, S, H]` 布局中，每条序列的有效结果从该序列输出区域首行开始；`[T, H]` 布局中，各序列有效结果按序列顺序紧密排列。分配容量中未使用行的数值不作规定，调用方应按有效组数读取结果。`state_cache` 的更新通过传入张量体现，不另行返回。

## 约束说明

- `H` 在 `[1024, 10240]` 内且为 `512` 的整数倍；`D` 为 `128` 或 `512`。
- `cmp_ratio` 为 `2、4、8、16、32、64、128` 之一。
- `block_count > 0`，`1 <= block_size <= 1024`。缓存最后两轴连续，第 `0` 轴步长不得小于 `block_size * 2 * D`，允许块间留有间隔。
- 所有张量位于同一个 NPU。除缓存第 `0` 轴外，各输入必须连续。
- `state_cache` 的可写区域不得与 `x`、`wkv`、`wgate` 或任一元数据张量的存储区域重叠；这些只读输入之间允许共享数据。
- `state_block_table` 的块号互不相同，且满足 `0 <= block_id < block_count`；块号 `0` 为有效物理块。
- `cu_seqlens` 单调不减，首元素为 `0`，末元素为 `T`。`seqused[b]` 在 `0` 到第 `b` 条序列实际长度之间。
- `start_pos` 非负。首次使用历史 token 前，调用方需保证对应缓存槽位包含所需状态。若环形容量不足以保留全部历史，首组读取的是传入槽位当前实际保存的值。
- 输入校验检查形状、类型、步长和设备；设备侧元数据的数值约束及存储不重叠约束由调用方保证。`B`、`S` 或 `T` 可以为 `0`，总 token 数为 `0` 时不启动 kernel。

## 调用示例

在算子目录已加入 Python 模块搜索路径的环境中，可按以下方式调用：

```python
import torch
import torch_npu  # noqa: F401

from op_kernel.compressor_v2 import compressor

batch, sequence, hidden, dim = 1, 128, 1024, 128
device = "npu"
dtype = torch.bfloat16
x = torch.randn(batch, sequence, hidden, device=device, dtype=dtype)
wkv = torch.randn(dim, hidden, device=device, dtype=dtype)
wgate = torch.randn(dim, hidden, device=device, dtype=dtype)
state_cache = torch.zeros(2, 128, 2 * dim, device=device, dtype=torch.float32)
state_block_table = torch.tensor([1], device=device, dtype=torch.int32)

out = compressor(
    x, wkv, wgate, state_cache,
    cmp_ratio=4,
    state_block_table=state_block_table,
)
torch.npu.synchronize()
assert out.shape == (1, 32, 128)
```
