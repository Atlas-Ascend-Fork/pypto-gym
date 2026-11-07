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

`kv_compress_epilog_v2` 是同一入口的别名。

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

## aclnn 交付形态

| 调用方式 | 说明 |
| --- | --- |
| jit 直调 | 经 `op_kernel/kv_compress_epilog_v2.py` 的宿主入口构造 tiling 并启动 kernel。 |
| aclnn API | `aclnnKvCompressEpilogV2GetWorkspaceSize` + `aclnnKvCompressEpilogV2`。测试桥见 `tests/aclnn_bridge/aclnn_bridge_kv_compress_epilog_v2.cpp`，`pytest --mode aclnn`。 |

aclnn 路径与 jit 路径的差异：

- `cache` 是原地更新对象：def 中同时注册为输入和输出，桥把同一个张量传给两个形参；kernel 从输出指针写回。
- `cache` 输入使用 `IgnoreContiguous`，保留调用方的 storage offset / stride 与四维页间空洞；不能用连续化副本替代。
- `slot_mapping` 在 aclnn 侧统一为 INT64；桥会把 INT32 输入转换为 INT64。
- `storage_bytes` 是 aclnn 接口的附加属性：桥从 `torch` storage 计算实际值，host tiling 用它校验物理写入范围；不传（0）时退化为由 shape/stride 推导的下界。
- tiling key 保留会改变生成代码/物理 tile 的维度：`Kind/Group/RoundScale/Slot64/Width/LargeScale/Layout4`。`Width` 决定片上 tile 宽度（固定 8192 会让小 `D` 行变慢 19%–44%），`Slot64` 决定 slot 的载入 dtype（保留 INT32 直读，避免每次 wrapper 触发一次 cast kernel）；FP4 的 `RoundScale` 固定为 1。共 228 个有效模板组合，按 cache dtype 分成 3 个二进制；本机构建约 85 s。
- jit 热路径用 `tiling_instance_to_bytes` + device uint8 tensor + `lru_cache` 预打包 tiling，并缓存 1 B workspace，避免每次 launch 的 2–3 ms H2D。
- aclnn 侧 `slot_mapping` 仍统一为 INT64（`Slot64=1`），桥会转换 INT32 输入；kernel 二进制同时保留 `Slot64=0/1`。

## 约束说明

- 三个张量必须位于同一 NPU，并在该设备上下文执行。
- `T>0`，`0<D≤8192`，且 `D` 能被 `quant_group_size` 整除。
- 二维缓存要求 `N>0`、`C≥K`；slot 有效范围为 `[0,N)`。
- 四维缓存要求 `B>0`、`S>0`、第三维为 1、`C≥bundle`；slot 有效范围为 `[0,B×S)`。显式或默认推导的 `block_stride` 均须不小于 `S×C`，且为 32 字节的整数倍。
- 缓存底层存储空间必须覆盖实际写入地址。四维寻址遵循 `block_stride` 和紧凑 `bundle` 协议，张量 view 的逻辑 stride 不替代此协议。
- aclnn 路径无法从 tensor 描述符直接得知底层存储大小；需要校验时请传 `storage_bytes` 属性（测试桥会自动传入）。
- 同一次调用中，合法 slot 应互不重复，避免多个输入行并发写入同一缓存位置。

## 性能对照

修复后同机测量（host 5 次执行取中位数；device 为冷 L2 A/B、3 次执行 × 20 样本）：

| 路径 | 原 jit | 新 jit | 变化 |
| --- | ---: | ---: | ---: |
| wrapper host（15 用例范围） | 135.8–138.5 us | 140.8–145.0 us | ≤ 1.047× |
| device c01 D128 FP8 | 1.930 us | 1.698 us | -12% |
| device c02 D512 FP8 | 2.115 us | 1.887 us | -11% |
| device c06 D16 FP4 | 1.794 us | 1.534 us | -15% |
| device c15 4D D512 FP4 | 1.986 us | 1.662 us | -16% |
| device stress D8192 FP8 | 7.978 us | 8.004 us | +0.3% |

修复前的两个回退点：dataclass tiling 每次 launch 重打包使 host 到 ~3.3 ms；`Width` 固定 8192
使小 D kernel 慢 19%–44%。修复后均回到原 jit 的 5% 以内。

## 目录结构

```
kv_compress_epilog_v2/
├── op_kernel/
│   └── kv_compress_epilog_v2.py   # delivery-form kernel（raw Ptr + OpTiling）
├── op_host/
│   ├── kv_compress_epilog_v2_def.cpp
│   ├── kv_compress_epilog_v2_infershape.cpp
│   ├── kv_compress_epilog_v2_tiling.cpp/.h
│   └── CMakeLists.txt
└── tests/
    ├── aclnn_bridge/
    ├── assets/
    └── test_kv_compress_epilog_v2.py
```

验证：

```bash
pytest ops/pro/experimental/kv_compress_epilog_v2/tests -q --mode jit
pytest ops/pro/experimental/kv_compress_epilog_v2/tests -q --mode aclnn
```
