# engram_gate

## 产品支持情况

| 项目 | 说明 |
| :--- | :--- |
| 硬件平台 | Ascend 950PR |
| 使用的计算单元 | AI Vector（AIV）核，不使用 Cube |
| 输入数据类型 | `x`、`key`、`value` 为 BF16；`weight` 为 FP32；`image_mask` 为 BOOL |
| 输出数据类型 | `y` 为 BF16 |
| 支持的形状 | `x`/`key`/`y` 为 `[t, hc, dim]`，`value` 为 `[t, dim]`，`weight` 为 `[hc, dim]`，`image_mask` 为 `[t]`；`t ≥ 1`，`hc` 为 `1 ~ 向量核数`，`dim` 最大 8512 |

## 功能说明

DeepSeek V4.1 Engram 的残差门控算子，沿最后一维 `dim`、逐 `(token, head)` 行计算：

```text
rstd = rsqrt(mean(x²)+eps) · rsqrt(mean(key²)+eps)
dot  = Σ(x · weight · key) · rstd / √dim
gate = sigmoid(copysign(√max(|dot|, clamp_value), dot))
y    = bf16(x + gate · value)
```

`image_mask == True` 的 token 直接原样返回 `x`（逐位相等）。`value` 按 token 共享，在 head
维广播。融合后每个 `(token, head)` 行的 `x`/`key`/`value` 各只搬入一次、`y` 只写出一次。

## 函数原型

```python
y = engram_gate_wrapper(x, key, value, weight, image_mask,
                        eps=1e-6, clamp_value=1e-6)
```

| 参数 | 形状 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `x` | `[t, hc, dim]` | BF16 | 主激活值 |
| `key` | `[t, hc, dim]` | BF16 | key 激活值 |
| `value` | `[t, dim]` | BF16 | value 激活值（head 维广播） |
| `weight` | `[hc, dim]` | FP32 | 逐 `(head, dim)` 加权系数 |
| `image_mask` | `[t]` | BOOL | 为 True 的 token 原样返回 `x` |
| `eps` | 标量 | float | 只接受默认值 `1e-6` |
| `clamp_value` | 标量 | float | 只接受默认值 `1e-6` |

返回 `y`：`[t, hc, dim]` BF16。

## 约束说明

- `dim ≤ 8512`（`CHUNK`）；`eps`/`clamp_value` 是 kernel 内立即数，只接受 `1e-6`。
- `hc ≤ 向量核数`：token 维按 `hc` 分组轮转，起核数必须是 `hc` 的整数倍。
- 输入形状/ dtype 不匹配时 wrapper 报 `ValueError`。
- 非连续输入会被压实为连续张量后再计算（结果一致；连续输入无额外拷贝）。

## aclnn 交付形态

| 调用方式 | 说明 |
| --- | --- |
| jit 直调 | 经 `op_kernel/engram_gate.py` 的 `engram_gate_wrapper` 构造 tiling 并启动 kernel。 |
| aclnn API | `aclnnEngramGateGetWorkspaceSize` + `aclnnEngramGate`。测试桥见 `tests/aclnn_bridge/`，`pytest --mode aclnn`。 |

实现说明：

- `t`/`hc`/`dim` 是运行时 tiling 字段；tiling key 为 `ShapeId`：`0` 为通用回退，
  `1..5` 对应 5 个常用生产形状（`2/5/72 x 4 x 5120`、`5x4x5121`、`3x4x8449`）。
  常用形状把 `t/hc/dim` 变成编译期常量，避免 kernel 从 tiling 读取，共 6 个 key。
- tiling 结构为 `EngramGateTiling(t, hc, dim)`；`opParaSize=32`。
- 纯 vector kernel 在新版 pypto 中会自动生成 **AIV-only** 二进制
  （`pypto_compile._kernel_type_from_codegen` 根据 `has_cube/has_vector` 选择
  `KERNEL_TYPE_AIV_ONLY`）。kernel 显式声明 `workspace` 参数；host tiling 申报 1024 B。
- AIV-only 下 `SetBlockDim` 是 AIV block 数：host 选
  `nc = min(AIV, t·hc)`、向下取整到 `hc` 的倍数，`blockDim=nc`；kernel 内
  `nc=get_block_num()*get_subblock_num()`（AIV-only subblock=1）。
- 若使用旧版 pypto，binary 会被硬编码成 MIX_AIC；此时 host 才需要 `blockDim=nc/2`
  并按 AIC 解释 `SetBlockDim`。当前代码按 AIV-only 配置，构建时请使用带
  `_kernel_type_from_codegen` 的新 pypto。
- jit 热路径预打包 tiling 成 device uint8 tensor（`lru_cache`）并缓存 workspace 占位；
  该 kernel 只有几微秒，不能承受每次调用 2–3 ms 的 tiling H2D。
- `eps`/`clamp_value` 在 aclnn 桥内校验为默认值，非默认直接报 `ValueError`。

## 性能

`msprof` Task Duration，冷 L2 256 MiB，每 case 独立进程多次采样取中位；AIV-only aclnn
对照原 jit：

| 用例 | 原 jit | 新 aclnn（AIV-only） | 变化 |
| --- | ---: | ---: | ---: |
| t=2,4,5120 | 2.904 us | 3.028 us | +4.3% |
| t=3,4,8449 | 4.066 us | 4.214 us | +3.6% |
| t=5,4,5120 | 3.379 us | 3.444 us | +1.9% |
| t=5,4,5121 | 3.656 us | 3.757 us | +2.8% |
| t=72,4,5120 | 13.610 us | 13.671 us | +0.4% |

均为 AIV-only（`Task Type=AI_VECTOR_CORE`、`Mix Block Num=0`），全部在 5% 内。

## 目录结构

```
engram_gate/
├── op_kernel/
│   └── engram_gate.py             # delivery-form kernel + wrapper
├── op_host/
│   ├── engram_gate_def.cpp
│   ├── engram_gate_infershape.cpp
│   ├── engram_gate_tiling.cpp/.h
│   └── CMakeLists.txt
└── tests/
    ├── aclnn_bridge/
    ├── assets/
    ├── test_engram_gate.py
    └── CMakeLists.txt
```

验证：

```bash
pytest ops/pro/deepseek_v4_1/engram_gate/tests -q --mode jit
pytest ops/pro/deepseek_v4_1/engram_gate/tests -q --mode aclnn
```

测试覆盖原 5 个 SPEC 用例，以及 hc/dim 边界、全 True/单 True mask、非连续输入和
eps/clamp 契约。
