# MHCPostFwd（PyPTO-Pro）

mHC（manifold hyper-connection）后处理前向：把本层新算出的隐藏状态 `x` 与 4 份历史残差
`residual`，用两组按 token 变化的学习系数重新混合成 4 份输出。纯逐元素 + 定长小归约的
向量算子（无矩阵乘）。

## 产品支持情况

| 产品 | 是否支持 |
|---|---|
| Ascend 950PR | 支持 |

## 功能说明

对每个 token `n`（展平后的 `(b, s)`）、每个输出份 `mo ∈ [0,4)`、每个隐藏维 `h`：

```text
out[n, mo, h] = post_layer_mix[n, mo, 0] * x[n, h]
                + sum_{mi=0}^{3} comb_res_mix[n, mi, mo] * residual[n, mi, h]
```

求和只跨 `mi`，**不跨 token、不跨 hidden**；`x`/`residual`/`out` 为 BF16，
`post_layer_mix`/`comb_res_mix` 为 FP32，乘积与 4 项求和在 FP32 寄存器中累加，
落盘前一次性舍入到 BF16。

`h` 只按 `h_blk` 分块，`h_blk` 由 host tiling 按以下公式推导：

```text
h_blk = align(h if h <= 4096 else min(ceil_div(h, 2), 4096), 64)
```

`h_blk` 是编译期 tiling-key 常量（决定 UB 布局与 tile 形状），合法取值为
`64, 128, ..., 4096`（64 个），每个取值对应一个 `MhcPostKey.HBlk` 特化 kernel。

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
| `out` | 输出 | `[n0, n1, 4, h]` | BF16 | 可选。省略时按 `residual` 形状新建；传入时原地填充 |

`n0`/`n1` 任意；`mhc = 4` 固定；`h` 为 64 的倍数。

## 返回值说明

返回 `out`（`BF16[n0, n1, 4, h]`）。传入 `out` 时返回同一张量（`data_ptr` 不变），
否则返回新建张量。

## 约束说明

1. `h` 必须是 **64 的倍数**；`mhc` 固定为 4。
2. 四入一出必须**连续存储**（`is_contiguous()`）；kernel 的扁平 token 视图按显式连续 stride 构造，
   wrapper 不做 `.contiguous()` 兜底，不满足即报 `ValueError`。
3. dtype 固定：`x`/`residual`/`out` 为 BF16，`post_layer_mix`/`comb_res_mix` 为 FP32。
4. `out` 传入时形状/dtype/device 必须与 `residual` 一致。
5. wrapper 不含 host 侧形状处理；token 轴展平由核内 `pl.make_tensor` 完成；单次调用只启动一个
   `@pl.jit` kernel。
6. 参数非法时抛 `ValueError`（用 `raise` 而非 `assert`，`python -O` 下依然生效）。

## aclnn 交付形态

| 调用方式 | 说明 |
| --- | --- |
| jit 直调 | 经 `op_kernel/mhc_post_fwd.py` 的宿主入口构造 tiling 并启动 kernel。 |
| aclnn API | `aclnnMhcPostFwdGetWorkspaceSize` + `aclnnMhcPostFwd`。测试桥见 `tests/aclnn_bridge/aclnn_bridge_mhc_post_fwd.cpp`，`pytest --mode aclnn`。 |

aclnn 路径与 jit 路径的差异：

- `out` 是普通输出（kernel 只写不读）。桥在 `out=None` 时按 `residual` 形状分配；传入时
  校验 BF16/shape/device/contiguous 并直接作为输出。无需 ref/原地别名。
- `h_blk` 通过 **tiling key** 下发：host tiling 计算 `h_blk` 后设置
  `tilingKey = h_blk / 64 - 1`（0..63）。有效组合 64 个，对应 64 个 kernel。
- tiling 结构仅携带 `n_tok` 与 `h`；`h_blk` 是编译期常量，不是 tiling 字段。
- 纯 vector kernel 在当前 pypto delivery 中编成 AIV-only binary；kernel 显式声明
  `workspace` 参数（框架在 tiling 前自动插入 workspace 指针），host tiling 申报 1024 B
  非零 workspace。
- AIV-only 下 aclnn 的 launch 宽度按 AIV 核数设置；jit 路径同样按 `vector_core_num` 启动。
- kernel 的 persistent grid-stride 必须用
  `pl.get_block_num() * pl.get_subblock_num()`：AIV-only 下 `block=56 / sub=1 / workers=56`。
  旧 pypto 的 MIX binary 才是 `block=28 / sub=2`；若 host 仍按 AIC 设置 `blockDim`，
  AIV-only 只会启动 28 个 AIV，aclnn 约慢 1.5×。

## 性能

同机 56 vector core，冷 L2 256 MiB；host 3 次 × 30 样本，device 3 次 × 12 样本
（profiler kernel Duration，min/med）。下表是旧 pypto MIX binary 口径；9.3.0 AIV-only
下 host `blockDim` 已改为 AIV 数，aclnn 列需按 9.3.0 重采。

| 版本 | 1x4096x1280 | 1x4096x7168 | wrapper host |
| --- | ---: | ---: | ---: |
| 原 jit | 106.39 / 107.30 us | 423.27 / 424.53 us | 131.5 us |
| 新 jit | 107.87 / 108.48 us | 423.89 / 425.22 us | 135.9 us |
| 新 aclnn | 109.74 / 110.35 us | 420.26 / 422.15 us | — |

8 个 P0 的最差比值：jit device 1.011×、aclnn device 1.028×、host 1.039×，均在 5% 内。
jit 的 host 优化来自按 `(shape, device)` 缓存 `kernel[None, grid, key]` launcher；aclnn 的
历史修复来自 MIX 下乘 `get_subblock_num()`。当前 AIV-only 版本还要求 host 按 AIV 数设置
`blockDim`，详见 `docs/jit-to-aclnn-migration.md` §3.6/§9/§10.3。

## 目录结构

```
mhc_post_fwd/
├── op_kernel/
│   └── mhc_post_fwd.py            # delivery-form kernel（raw Ptr + OpTiling + 64 HBlk key）
├── op_host/
│   ├── mhc_post_fwd_def.cpp
│   ├── mhc_post_fwd_infershape.cpp
│   ├── mhc_post_fwd_tiling.cpp/.h
│   └── CMakeLists.txt
└── tests/
    ├── aclnn_bridge/
    ├── assets/
    └── test_mhc_post_fwd.py
```

验证：

```bash
pytest ops/pro/deepseek_v4_1/mhc_post_fwd/tests -q --mode jit
pytest ops/pro/deepseek_v4_1/mhc_post_fwd/tests -q --mode aclnn
```
