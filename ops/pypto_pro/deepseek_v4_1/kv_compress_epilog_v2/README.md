# KvCompressEpilogV2（PyPTO-Pro）

## 产品支持情况

| 产品 | 支持情况 |
|---|---|
| Ascend 950PR | 支持 |

## 功能说明

本算子将 BF16 输入按组量化为 FP8 或 FP4，将量化数据和每组的 BF16 scale 写入缓存，并通过 `slot_mapping` 指定每行的缓存位置。缓存原地更新，未命中的区域保持原值。

设输入形状为 `[T,D]`，分组大小为 `g`，每行有 `G=D/g` 个组。FP8 每个值占 1 字节；FP4 每两个值打包为 1 字节，偶数位置在低 4 位，奇数位置在高 4 位。所有组的量化数据连续保存，随后保存 `G` 个 BF16 scale。

FP8 支持 E4M3FN 和 E5M2。设格式的最大有限值为 `M`：E4M3FN 为 448，E5M2 为 57344。每组按以下顺序计算：

1. 在 FP32 中计算 `s=max(max(abs(x)),1e-4) × FP32(1/M)`。
2. `round_scale=True` 时，将有限正 scale 向上舍入到 2 的整数次幂；否则保留原始 FP32 scale。
3. 使用这个 FP32 scale 对数据做除法，将结果限制在 `[-M,M]`，再按最近偶数舍入转换为 FP8。
4. 将 scale 另行按最近偶数舍入保存为 BF16。数据除法不会先使用已舍入的 BF16 scale。

FP8 转换中出现的 NaN 数据编码为 `0x7F`。NaN scale 在 `round_scale=False` 时保存为 BF16 `0x7FFF`；`round_scale=True` 时按指数位规则保存为负零 `0x8000`。

FP4 从 BF16 指数位确定 scale。`bits(x)` 表示 BF16 的 16 位编码。令 `A=max(bits(x) & 0x7F80)`，`E=max(A,0x0100)-0x0100`。有限且 `A>0` 的组，scale 的 BF16 位模式为 `E`，逆 scale 的位模式为 `0x7F00-E`；`A=0` 时两者均为正零。输入先与逆 scale 做 BF16 乘法，再按最近偶数舍入转换为 E2M1。含 NaN 或 Inf 的组输出正零数据，scale 位模式为 `0x7F81`。`round_scale` 不改变 FP4 的规则。

令数据字节数 `Q=D`（FP8）或 `D/2`（FP4）。一条记录的有效长度为 `bundle=Q+2G` 字节，向上对齐到 32 字节后的长度为 `K=ceil(bundle/32)×32`：

| 缓存布局 | 合法 slot 的起始字节地址 | 写入范围 |
|---|---|---|
| 二维 `[N,C]` | `slot×C` | 数据、BF16 scale，以及清零的 `[bundle,K)` 行尾区域 |
| 四维 `[B,S,1,C]` | `(slot//S)×block_stride+(slot%S)×bundle` | 仅写数据和 BF16 scale，共 `bundle` 字节 |

四维页内记录使用紧凑的 `bundle` 步长，地址不是 `(slot%S)×C`。二维和四维缓存均按上述物理字节地址更新，张量 view 的逻辑 stride 不参与寻址。二维 `[K,C)`、四维页间空隙和其它未命中区域均保持原值。

## 函数原型

```python
kv_compress_epilog_v2_wrapper(
    cache,
    x,
    slot_mapping,
    quant_group_size=32,
    quant_mode=2,
    round_scale=True,
    x_scale=1.0,
    block_stride=0,
)
```

### 参数说明

| 参数 | 类型 | 说明 |
|---|---|---|
| `cache` | `torch.Tensor` | 原地更新的二维或四维缓存。FP8 使用 `torch.float8_e4m3fn` 或 `torch.float8_e5m2`；FP4 使用 `torch.uint8`。 |
| `x` | `torch.Tensor` | 连续 BF16 输入，形状为 `[T,D]`。 |
| `slot_mapping` | `torch.Tensor` | 连续 INT32 或 INT64，形状为 `[T]`。负数或超出缓存行数的 slot 跳过。 |
| `quant_group_size` | `int` | FP8 仅支持 32；FP4 支持 16 或 32。 |
| `quant_mode` | `int` | `2` 表示 FP8，`4` 表示 FP4。FP8 格式由 `cache.dtype` 决定。 |
| `round_scale` | `bool` | FP8 是否将 scale 向上舍入到 2 的整数次幂；FP4 忽略此参数。 |
| `x_scale` | `float` | 保留参数，必须为 `1.0`。 |
| `block_stride` | `int` | 四维缓存的页间字节步长。`0` 表示使用 `S×C`；二维布局不使用它寻址。 |

### 返回值说明

返回与输入 `cache` 相同的对象，其 shape、dtype 和 stride 保持不变。量化数据与 BF16 scale 保存在上述缓存字节布局内，不另行返回张量。

## 约束说明

- 三个张量必须位于同一 NPU，并在该设备上下文执行。
- `T>0`，`0<D≤8192`，且 `D` 能被 `quant_group_size` 整除。
- 二维缓存要求 `N>0`、`C≥K`；slot 有效范围为 `[0,N)`。
- 四维缓存要求 `B>0`、`S>0`、第三维为 1、`C≥bundle`；slot 有效范围为 `[0,B×S)`。显式或默认推导的 `block_stride` 均须不小于 `S×C`，且为 32 字节的整数倍。
- 缓存底层存储空间必须覆盖实际写入地址。四维寻址遵循 `block_stride` 和紧凑 `bundle` 协议，张量 view 的逻辑 stride 不替代此协议。
- 同一次调用中，合法 slot 应互不重复，避免多个输入行并发写入同一缓存位置。

## 调用示例

以下示例在算子目录下使用 FP8 E4M3FN，`D=512`、group 为 32。每行保存 512 字节数据和 32 字节 BF16 scale：

```python
import torch
import torch_npu

from op_kernel.kv_compress_epilog_v2 import kv_compress_epilog_v2_wrapper

torch.npu.set_device(0)
device = torch.device("npu:0")
x = torch.randn((4, 512), dtype=torch.bfloat16, device=device)
slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
cache = torch.full(
    (8, 544), 0x5A, dtype=torch.uint8, device=device
).view(torch.float8_e4m3fn)

result = kv_compress_epilog_v2_wrapper(
    cache, x, slots,
    quant_group_size=32,
    quant_mode=2,
    round_scale=True,
)
torch.npu.synchronize()

assert result is cache
assert cache.shape == (8, 544)
```
