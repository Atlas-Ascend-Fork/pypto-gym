# indexer_prologue_qw

## 概述

`indexer_prologue_qw` 是 DeepSeek V4.1 indexer 前置 Q/W 投影融合算子，基于 `pypto_pro`
DSL 在 Ascend 上实现。一个 kernel 完成两条 GEMM 路径与一个向量收尾：

- **Q 路径**：`qr @ wqb^T` 的 MXFP8 GEMM（E4M3 字节、每 32 个 K 一个 E8M0 块 scale，
  FP32 累加）→ 只对末尾 `Dr=64` 列做 rotate-half RoPE → 按每 32 列一组做 MXFP4 量化
  （E2M1 就近取整、`.5` 处远离零；E8M0 对 `amax/6` 向上取整到 2 的幂），每字节打包
  两个 nibble（偶列低 4 bit）。
- **W 路径**：`softmax_scale * (x @ ww^T)` 的 BF16 GEMM（K=`dim`，FP32 累加后统一乘入）。

固定形状：`dim=5120`、`q_lora=1280`、`N=32`、`D=128`、`Dr=64`；唯一动态轴 `T`，范围
`[1, 262144]`（覆盖参考部署矩阵的 21 个 P0 `T`）。`wqb`、`ww` 需按 **FRACTAL_NZ**
提供，其余输入为 ND。

模板按 `n_tiles = ceil(T/128)` 选择：

- `n_tiles >= 32`（`SPLIT_T_TILES`）：**split-T**，一个核独占一个 `T` tile 的全部 32 个
  head，W 的 K 归约（`NWK=40` 片）交错进 Q head 循环完成；
- 否则：**split-K**，head 轴切成 `HB_K=16` 个 2-head block；当 `n_tiles <= 12`
  （`SPLIT_W_TILES`）时 W 的 K 轴也按同一 head block 切分，分片写入 workspace，向量单元
  在 barrier 后做确定性归约。

尾块（`T` 非 `BASE_M=128` 整数倍）全部在 kernel 内处理：Q 累加器保持 `[BASE_M, D]`
物理窗口，L1 load 用 `set_validshape` 夹到有效行，向量收尾按 AIV 半边夹紧 store；W 累加器
按运行期有效 M 直接裁剪 store。调用者无需补齐。

## 目录结构

```
indexer_prologue_qw/
├── README.md                       # 文档
├── op_kernel/
│   └── indexer_prologue_qw.py      # 算子 kernel（单 @pl.jit + 公开 wrapper）
└── tests/
    ├── test_indexer_prologue_qw.py # 测试入口（pytest + CLI）
    └── assets/
        ├── indexer_prologue_qw_golden_cpu.py  # 独立 CPU golden（数学 oracle）
        └── indexer_prologue_qw_precision.py   # 自包含精度比较器
```

## 接口签名

```python
# ops/pypto_pro/deepseek_v4_1/indexer_prologue_qw/op_kernel 加入 sys.path 后：
from indexer_prologue_qw import (
    indexer_prologue_qw,  # 公开入口
    SOFTMAX_SCALE,        # 默认 0.08838834764831845 (= 128**-0.5)
)

q, descale_q, w = indexer_prologue_qw(
    x,            # BF16 [T, 5120]          行数可大于 T，多出的行为死区
    qr,           # FP8E4M3FN 或 uint8 [T, 1280]
    wqb,          # FP8E4M3FN 或 uint8 [4096, 1280]   需 FRACTAL_NZ
    ww,           # BF16 [32, 5120]                    需 FRACTAL_NZ
    descale_qr,   # uint8 [T, 20, 2]          paired E8M0 块 scale
    descale_wqb,  # uint8 [4096, 20, 2]
    rope_sin,     # FP32 [T, 64]
    rope_cos,     # FP32 [T, 64]
    softmax_scale=SOFTMAX_SCALE,   # 可选，默认 128**-0.5
)
```

输出（均在 `qr.device` 上）：

| 名称 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `q` | `[T, 32, 64]` | uint8 | 打包的 E2M1 nibble（偶列低、奇列高） |
| `descale_q` | `[T, 32, 2, 2]` | uint8 | E8M0 指数，不足组用 127 填充 |
| `w` | `[T, 32]` | float32 | `softmax_scale * (x @ ww^T)` |

`T` 取自 `qr.shape[0]`；`x` 多余行（若有）被忽略。

## 设计要点

- **单 kernel、单 launch。** 所有计算在唯一一个 `@pl.jit(auto_mutex=True)` MIX kernel
  `indexer_prologue_qw_kernel` 内完成，用 `with pl.section_cube()` / `with pl.section_vector()`
  划分 CV 边界。公开 wrapper `indexer_prologue_qw` 只做三件事：`_validate_metadata` 校验
  shape/dtype、`torch.empty` 分配 `q`/`descale_q`/`w` 与 workspace、按 `_cube_cores()` 启动
  一次 kernel；**不做任何 host 侧计算或 reshape**。
- **片上 LUT 量化。** E2M1 收尾不为 host 传入任何私有输入：`ipqw_build_lut_vf` 在片内生成
  一张 `LUT_LEN=1088` 的中点桶表，`ipqw_epilogue_vf` 用截断索引即得精确的最近邻（含 `.5`
  处远离零），替代 7 次中点比较。
- **split-K / split-T 两套模板**，判据 `n_tiles`（见上）；W 的 K 归约在 split-K 下走
  workspace 确定性求和（非原子），split-T 下交织进 head 循环。
- **尾块内处理**：`qr`/scale 的 L1 载入按有效行 clamp，L0A 保持全物理窗（避免 L0C
  mis-layout），store 端 clamp。

## 精度

SPEC 采用 `pro_scheme_a` 容差策略：与独立 CPU golden
（`indexer_prologue_qw_golden_cpu.py`，纯 torch）逐块比较。`w` 用 `rtol=atol=2e-2`；
`q`/`descale_q` 为整型字节，按位比较，但需豁免源实现固有的三类**量化边界**差异（均与
ET 语义无关，只是 FP32 累加次序的 1 ulp 决定舍入方向）：

1. E2M1 相邻档位的**一步同号移动**，且 golden 商值落在中点（`0.25, 0.75, …`）的 ulp 窗口内；
2. `+0` / `-0` **有符号零对**（幅值码同为 0，仅符号位不同，商值低于第一个中点）；
3. E8M0 指数相差 **±1**，且 `amax` 落在 `6*2^k` 边界、其自身点积的累加展布之内。

上述判定结合 CPU golden 的可选调试返回（post-RoPE、量化前的 `q_fp32`）实现，位于
`tests/assets/indexer_prologue_qw_precision.py`（自包含，语义与参考比较器等价，
21/21 PASS，未放宽容差、未丢 case）。参考实现（CPU golden）、精度比较器在
`tests/assets/` 下，测试入口在 `tests/` 下，算子本体在 `op_kernel/` 下。

## 性能

设备 `Ascend950PR_9589`（A5 / `dav-3510`，cube 32 / vector 64），协议为
**NPU-graph per-call event time，10 轮 × 100 次取中位**。四个目标 case 的时延：

| T | 时延 (µs) |
|--:|--:|
| 72 | 11.288 |
| 2048 | 53.635 |
| 4096 | 69.949 |
| 262144 | 5732.903 |

21 个 P0 `T` 全部通过精度校验（`21/21`）：decode 区间（`T ≤ 2048`）时延约 8–54 µs，
`T ≥ 8192` 后转为带宽受限。

## 运行

```bash
D=ops/pypto_pro/deepseek_v4_1/indexer_prologue_qw

# CPU golden 自检（形状 / dtype / 有限性）
python $D/tests/assets/indexer_prologue_qw_golden_cpu.py

# 全量精度矩阵 + 四个目标 case 性能（默认）
python $D/tests/test_indexer_prologue_qw.py

# 精度子集 / pytest
python $D/tests/test_indexer_prologue_qw.py --t 1,72,2048 --skip-perf
pytest $D/tests -q
```

测试把逐 case 结果与性能数据写入 `$D/tests/reports/pro_pypto/validation.json`
（可用 `--out` 覆盖）。精度比较所需的 `q_fp32` 由 CPU golden 的可选调试返回提供，全程不依赖
任何仓库外路径。
