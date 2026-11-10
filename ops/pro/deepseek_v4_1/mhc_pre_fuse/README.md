# mhc_pre_fuse

## 产品支持情况

| 项目 | 说明 |
| :--- | :--- |
| 硬件平台 | Ascend 950PR |
| 使用的计算单元 | AI Vector（AIV）核，不使用 Cube |
| 输入数据类型 | `residual` 为 BF16；`gemm_out_mul`、`gemm_out_sqrsum`、`mhc_scale`、`mhc_base` 为 FP32 |
| 输出数据类型 | `post_mix`、`comb_mix` 为 FP32；`layer_input` 为 BF16 |
| 支持的形状 | 词元数 `T` 为任意正整数；隐藏维度 `H` 为 64 的倍数且 `H / 64 >= 5`；混合路数 `m` 固定为 4 |

## 功能说明

`mhc_pre_fuse` 是 mHC（m = 4 路残差混合）的前处理融合算子，把四个步骤合并成一个 NPU 向量核，一次调用同时产出三个结果：

```text
① RMSNorm 缩放        mixes = gemm_out_mul × rms          （24 路）
② 切分 mixes          pre = mixes[0:4]  / post = mixes[4:8] / comb = mixes[8:24]
③ Sinkhorn            comb 的 4×4 矩阵做行列交替归一化 → comb_mix
④ 残差预混合          layer_input = Σ_m residual[m] × pre[m]
```

`rms` 由上游给出的平方和 `gemm_out_sqrsum` 算出，算子本身不做平方和归约。

## 函数原型

```python
post_mix, comb_mix, layer_input = mhc_pre_fuse_wrapper(
    gemm_out_mul,       # FP32  [T, 24]（可带前导 split 轴，取第 0 份）
    gemm_out_sqrsum,    # FP32  [T]    （可带前导 split 轴，取第 0 份）
    mhc_scale,          # FP32  [3]
    mhc_base,           # FP32  [24]
    residual,           # BF16  [T, 4, H]
    knobs=Knobs(),      # 五个标量开关
)
```

### 参数说明

| 参数 | 形状 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `gemm_out_mul` | `[T, 24]` | FP32 | 上游 GEMM 的 24 路结果；3D 时只取第 0 份 split |
| `gemm_out_sqrsum` | `[T]` | FP32 | `residual` 的平方和；2D 时只取第 0 份 split |
| `mhc_scale` | `[3]` | FP32 | 三组缩放系数（pre/post/comb） |
| `mhc_base` | `[24]` | FP32 | 三组偏置 |
| `residual` | `[T, 4, H]` | BF16 | 4 路残差 |
| `knobs` | — | `Knobs` | 五个标量开关，见下表 |

`knobs` 字段（默认值即算子契约值）：

| 字段 | 类型 | 默认值 |
| :--- | :--- | ---: |
| `rms_eps` | float | `1e-6` |
| `mhc_pre_eps` | float | `1e-6` |
| `mhc_sinkhorn_eps` | float | `1e-6` |
| `mhc_post_mult_value` | float | `1.0` |
| `sinkhorn_repeat` | int | `10` |

### 返回值说明

| 返回值 | 形状 | 类型 |
| :--- | :--- | :--- |
| `post_mix` | `[T, 4]` | FP32 |
| `comb_mix` | `[T, 16]` | FP32 |
| `layer_input` | `[T, H]` | BF16 |

## 约束说明

- `m` 固定为 4；`H` 为 64 的倍数且 `H / 64 >= 5`；`T` 为任意正整数。
- 五个输入必须连续存放；wrapper 不做 `.contiguous()` 兜底。
- `sinkhorn_repeat >= 1`。
- 参数非法时抛 `ValueError`。

## aclnn 交付形态

| 调用方式 | 说明 |
| --- | --- |
| jit 直调 | 经 `op_kernel/mhc_pre_fuse.py` 的 `mhc_pre_fuse_wrapper` 构造 tiling 并启动 kernel。 |
| aclnn API | `aclnnMhcPreFuseGetWorkspaceSize` + `aclnnMhcPreFuse`。测试桥见 `tests/aclnn_bridge/`，`pytest --mode aclnn`。 |

实现说明：

- `T`、`H`、五个 `knobs` 都是**运行时**参数（tiling 字段 / aclnn 属性）；
- 结构参数 `(Tb, HBlk, RRem)` 由 host tiling 计算并作为 **tiling key** 下发：
  - `Tb ∈ {8,4,2,1}`：token 块大小；
  - `HBlk ∈ {512,256,192,128,64}`：hidden 分块宽度；
  - `RRem ∈ {0,1,2,3,4}`：残差 5 级流水的尾部长度。
  有效组合 100 个，每个组合编译一个 kernel。
- `h_blk` 选择规则与 jit 版一致：从宽到窄取第一个能整除 `H`、满足 `H/h_blk >= 5` 且 5 份输入 + 5 份输出流式 buffer 能放进 253952 B UB 的值。
- 纯 vector kernel 在当前 pypto delivery 中编成 AIV-only binary；kernel 显式声明
  `workspace` 参数，host tiling 申报 1024 B 非零 workspace。
- AIV-only 下 aclnn 的 launch 宽度按 AIV 核数设置；jit 路径按 `vector_core_num` 启动。
- kernel 的 persistent grid-stride 使用
  `pl.get_block_num() * pl.get_subblock_num()`：AIV-only 下 `block=56 / sub=1 / workers=56`。
  旧 pypto 的 MIX binary 才是 `block=28 / sub=2`；若 host 仍按 AIC 设置 `blockDim`，
  AIV-only 只会启动 28 个 AIV，aclnn 约慢 1.5×。
- `gemm_out_mul` 的 3D split 轴与 `gemm_out_sqrsum` 的 2D split 轴在桥内取 `[0]`，与 jit wrapper 一致。
- 输入 dtype 在 wrapper 内校验：四个 FP32 输入与 BF16 `residual` 不匹配时抛 `ValueError`。

## 性能

同机 56 vector core，冷 L2 256 MiB；host 3 次 × 30 样本，device 3 次 × 12 样本
（profiler kernel Duration，min/med）。下表是旧 pypto MIX binary 口径；9.3.0 AIV-only
下 host `blockDim` 已改为 AIV 数，aclnn 列需按 9.3.0 重采。

| 用例 | 原 jit | 新 jit | 新 aclnn |
| --- | ---: | ---: | ---: |
| 512x4096 | 29.84 / 30.21 us | 29.66 / 29.98 us | 29.63 / 30.29 us |
| 512x7168 | 49.43 / 50.02 us | 49.74 / 49.99 us | 49.56 / 50.48 us |
| 8192x4096 | 313.60 / 314.66 us | 313.38 / 314.71 us | 314.67 / 315.77 us |
| 8192x5120 | 366.16 / 367.86 us | 366.22 / 367.60 us | 366.22 / 367.40 us |
| 8192x7168 | 481.15 / 482.53 us | 480.93 / 482.43 us | 482.08 / 482.92 us |

- device 最差：新 jit 1.000×、aclnn 1.003×；host 新 jit 0.93–0.95×（新实现更快）。
- `sinkhorn_repeat` 从旧编译期常量改为运行时 tiling 后没有可测回退（512x4096 29.98 vs
  30.21 us），因此不额外增加 tiling key；100 个 `(Tb,HBlk,RRem)` key 保持不变。

## 目录结构

```
mhc_pre_fuse/
├── op_kernel/
│   └── mhc_pre_fuse.py             # delivery-form kernel + wrapper
├── op_host/
│   ├── mhc_pre_fuse_def.cpp
│   ├── mhc_pre_fuse_infershape.cpp
│   ├── mhc_pre_fuse_tiling.cpp/.h
│   └── CMakeLists.txt
└── tests/
    ├── aclnn_bridge/
    ├── assets/
    ├── test_mhc_pre_fuse.py
    └── CMakeLists.txt
```

验证：

```bash
pytest ops/pro/deepseek_v4_1/mhc_pre_fuse/tests -q --mode jit
pytest ops/pro/deepseek_v4_1/mhc_pre_fuse/tests -q --mode aclnn
```

测试除原 5 个 P0 形状外，还覆盖 `Tb/HBlk/RRem` 分支、非默认 knobs 和带前导 split 轴的输入。
