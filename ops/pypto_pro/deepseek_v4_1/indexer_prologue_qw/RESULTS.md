# indexer_prologue_qw 结果：精度与性能

测试入口 `tests/test_indexer_prologue_qw.py`（同时是 pytest 用例），独立 CPU 数学参考
`tests/assets/indexer_prologue_qw_golden_cpu.py`，精度比较器
`tests/assets/indexer_prologue_qw_precision.py`（自包含，不依赖仓库外路径）。

## 用例

| 类别 | 覆盖内容 |
|---|---|
| P0 精度矩阵 | SPEC 的 **21 个 P0 `T`**（清单即 `tests/assets/indexer_prologue_qw_golden_cpu.py` 的 `P0_CASES`），覆盖 decode 区间（`T ≤
2048`）到最长序列（`T = 262144`）；`pytest` 以 `@pytest.mark.parametrize` 逐 case 参数化 |
| 边界入口 | 显式单 case 入口，含 `T=1`（最小 T：尾块与启动开销边界）、`T=72`（decode 小 T）等 |
| 目标性能 case | `{72, 2048, 4096, 262144}`（默认）；`--perf all` 覆盖全部 21 个 P0 |
| 计时口径（两个） | ① **NPU-graph 逐调用**：冻结口径，去掉 host 启动开销、含图内设备时间（warmup 后按 10 轮 × 每次 100 次取中位）；② **eager 逐调用**：含 host
启动开销，仅作对照 |

## 精度标准

采用 SPEC 的 `pro_scheme_a` 容差策略，与独立 CPU golden 逐块比较：

- **`w`（W 路径输出）**：`rtol=atol=2e-2`（FP32 累加，容许实现间的累加次序差异）。
- **`q` / `descale_q`（Q 路径输出）**：整型字节，**按位比较**；但豁免源实现固有的三类**量化边界**
  差异（均与 ET 语义无关，只是 FP32 累加次序的 1 ulp 决定舍入方向）：
  1. E2M1 相邻档位的**一步同号移动**，且 golden 商值落在中点（`0.25, 0.75, …`）的 ulp 窗口内；
  2. `+0` / `-0` **有符号零对**（幅值码同为 0，仅符号位不同，且商值低于第一个中点）；
  3. E8M0 指数相差 **±1**，且 `amax` 落在 `6·2^k` 边界、并在其自身点积的累加展布之内。
- 上述判定借助 CPU golden 的可选调试返回（post-RoPE、量化前的 `q_fp32`）实现，因此**未放宽容差、
  未丢 case**：21/21 P0 全部通过。
- 比较器自身在 `tests/assets/indexer_prologue_qw_precision.py` 内自包含，语义与参考比较器等价。

## 性能结果

### 采集协议

- 设备 `Ascend950PR_9589`（A5 / `dav-3510`，cube 32 / vector 64）。
- 时延口径：**NPU-graph 逐调用 event 时间**（冻结口径）——预热 5 次 replay 后，每轮连续采样
  100 次、重复 10 轮，取中位；图捕获不含 host 启动开销。
- 另提供 eager 逐调用 event 时间（含 host 启动开销、warmup 3 次）作为对照。

### 绝对性能结果

| T | 时延（µs，NPU-graph 逐调用） |
|--:|--:|
| 72 | 11.288 |
| 2048 | 53.635 |
| 4096 | 69.949 |
| 262144 | 5732.903 |

decode 区间（`T ≤ 2048`）时延约 **8–54 µs**。21 个 P0 `T`
全部通过精度校验（`21/21`）。

### 理论评估对比（MFU / Bound / 利用率）

平台常量：`cube_core_cnt=32`、`vector_core_cnt=64`、`cube_freq=1650` MHz、`l1_size=524288`、
`l0_c_size=262144`、`ub_size=253952`、`[DtypeMKN] DT_FLOAT8_E4M3FN=16,32,16`。

- **算力峰值** = `M·N·K × 2 × cube_core_cnt × cube_freq` = `16×32×16 × 2 × 32 × 1.65e9`
  = **864.9 TFLOP/s**（FP8 dense）。
- **单 token 计算量** = Q 路径 `1280×4096` + W 路径 `5120×32` = 5.407 M MAC = **10.81 MFLOP**。

| T | 计算量（GFLOP） | 时延（µs） | 实测算力（TFLOP/s） | **MFU** |
|--:|--:|--:|--:|--:|
| 72 | 0.779 | 11.288 | 69.0 | **8.0%** |
| 2048 | 22.15 | 53.635 | 412.9 | **47.7%** |
| 4096 | 44.29 | 69.949 | 633.2 | **73.2%** |
| 262144 | 2834.7 | 5732.903 | 494.5 | **57.2%** |

- **Bound 情况（按 msprof 逐管线实测占比判定）**：各 `T` 上 AIC 侧占用最高的管线
  （`aic_*_time / aicore_time`）与由此判定的 Bound：

  | T | AIC 侧占用最高的管线 | VEC 侧占用最高的管线 | Bound |
  |--:|---|---|---|
  | 72 | `scalar` **0.261** | `aiv_vec` 0.111 | 启动/收尾主导（非 CUBE/VEC 计算，也非 MTE2 搬运） |
  | 2048 | **`mte2` 0.882**（`mac` 0.644） | `aiv_vec` 0.530 | **MTE2 搬运** |
  | 4096 | **`mte2` 0.898** ≈ `mac` 0.889 | `aiv_vec` 0.756 | **计算与搬运双近饱和**（均衡点，MTE2 略高） |
  | 262144 | **`mte2` 0.901**（`mac` 0.729） | `aiv_vec` 0.626 | **MTE2 搬运** |

  即 `T ≥ 2048` 时 AIC 侧占用最高的管线**始终是 MTE2（搬运）**，算子整体偏 **MTE2 搬运 bound**，
  而非 CUBE 算力 bound；`T=4096` 落在算力与搬运的双近饱和均衡点（`mac` 0.889 / `mte2` 0.898）；
  `T=72` 由启动/收尾（`scalar`）主导。VEC（AIV）侧占用最高的始终是 `vec`，全程低于 AIC 侧最高管线。
  口径：msprof kernel-only 硬件时间戳，`aic_*_ratio = aic_*_time / aicore_time`、`aiv_*_ratio = aiv_*_time / aiv_time`。
- **利用率**：上表 MFU 即 Cube 发射利用率（`计算量/时延 ÷ 平台峰值`）；向量侧（RoPE + MXFP4
  打包 + split-K 的 W 归约）与 Cube 阶段通过 CV 握手重叠：Q 的 RoPE/量化在累加器转入 UB 后立即
  开始，下一 tile 的 GEMM 同时推进；E2M1 量化用片内 LUT（`LUT_LEN=1088`）以截断索引替代 7 次中点
  比较，把收尾的比较开销降为一次查表。上表口径只含 kernel 内的设备时间，不含 host 启动。
