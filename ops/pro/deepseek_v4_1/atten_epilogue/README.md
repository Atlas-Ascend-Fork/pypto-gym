# AttenEpilogue (atten_epilogue)

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |

## 功能说明

- 算子功能：注意力输出端的融合算子。单次 launch 内完成 `o[T, 64, 512]`（bf16）到
  `out[T, 5120]`（bf16）的全部计算，四个阶段同属一个 native task，阶段间用 core-grid barrier 分隔：

  1. **M1（Vector）** inverse RoPE：对每 head 的 512 维向量按 `rope_sin`/`rope_cos` 做逆旋转，
     并回归组为 8 个 group（每 group 1024 列，对应 `woa` 的分组维度）；随后做 OCP MXFP8 量化
     （每 32 个元素一个 E8M0 块缩放），写出 GM 载荷 `aq`/`asc`。
  2. **M2（Cube）** GEMM1：`y[t, g*1024+n] = Σ_k aq[t, g, k] · woa[g, n, k]`（K = 4096，FP8 × FP8 → FP32 累加）。
  3. **M3（Vector）** 对 `y` 再做一次 MXFP8 量化，写出 GM 载荷 `yq`/`ys`。
  4. **M4（Cube）** GEMM2：`out[t, m] = Σ_k yq[t, k] · wob[m, k]`（K = 8192），以 bf16 写回。

- 计算公式：

$$
Y = \mathrm{MXQ}\!\left(\mathrm{RoPE}^{-1}(O)\right) \otimes_{\mathrm{MX}} W_{oa},\qquad
Out = \mathrm{MXQ}(Y) \otimes_{\mathrm{MX}} W_{ob}^{\mathrm{T}}
$$

  其中 $\mathrm{MXQ}$ 为 OCP MXFP8 分块量化（块长 32，E8M0 scale），$\otimes_{\mathrm{MX}}$ 为 block-scaled 矩阵乘。

- 实现为单次 launch 的 PyPTO delivery-form kernel（`op_kernel/atten_epilogue.py`）：
  raw `pl.Ptr` I/O + 末尾一个 `EpilogueTiling` 结构体，核间按 32 Cube / 64 Vector 配额切分，
  两级 GEMM 均不切 K。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| ------ | --------- | ---- | -------- | -------- |
| o | 输入 | attention 输出，`[T, 64, 512]`（64 head × 512 维）。 | BFLOAT16 | ND |
| woa | 输入 | GEMM1 权重（8 个 group），`[8, 1024, 4096]`，由调用方预先量化。 | FLOAT8_E4M3FN | ND |
| wob | 输入 | GEMM2 权重，`[5120, 8192]`，由调用方预先量化。 | FLOAT8_E4M3FN | ND |
| descale_woa | 输入 | `woa` 的块缩放（每 32 个 K 元素一个 E8M0 scale），`[8, 1024, 64, 2]`。 | FLOAT8_E8M0 | ND |
| descale_wob | 输入 | `wob` 的块缩放，`[5120, 128, 2]`。 | FLOAT8_E8M0 | ND |
| rope_sin | 输入 | inverse RoPE 的 sin 表，`[T, 64]`。 | FLOAT | ND |
| rope_cos | 输入 | inverse RoPE 的 cos 表，`[T, 64]`。 | FLOAT | ND |
| out | 输出 | 融合投影结果，`[T, 5120]`。 | BFLOAT16 | ND |

`T` 为运行期 token 数；除 `T` 外的形状均为固定维度。

## 约束说明

- `T >= 1` 的任意正整数；`o` 必须为 3 维且后两维恰为 `(64, 512)`；所有张量须为**同一设备上的连续 ND 张量**。
- 权重与 scale **不接受 FRACTAL_NZ**（按 ND 行连续搬入 L1，片上再做转置视图）；量化由调用方完成，算子不做权重量化。
- 内部把 `T` 补齐到 16 的倍数（`padded`，并行分档需要）；补齐行只存在于 GM 载荷中，**不会写入 `out`**。
- **无用户可见的辅助张量**：4 个 GM 载荷（`aq`/`asc`/`yq`/`ys`）由 kernel 用 `pl.addptr` 从框架
  workspace 中切分，host tiling 申报 `42240 * padded` 字节
  （`aq 32768 + asc 1024 + yq 8192 + ys 256`，每 padded 行）。各载荷偏移均为 512 字节整数倍。
- 支持的分档由 tiling key schema（`EpilogueKey`）约束：两种硬件布局共 6 个合法 key
  （32 核布局 `BN1=256/BN=160`，少数核布局 `BN1=160/BN=192`），签名不含 `T` 的静态取值。
  host tiling 按 `(T, GetCoreNumAic())` 选档，与 kernel 侧 `tiling_key_for` 逐位一致。
- **jit 模式不支持图捕获**：交付形态传的是 tiling 结构体，PyPTO 的 jit launcher 每次启动都会把它
  序列化进一块新的 device buffer（`_pack_tiling_arg`），该 H2D 拷贝在 GLOBAL 捕获模式下被拒绝
  （flex_attention 同理）。built-in aclnn 路径的 tiling 码流由框架构建，无此拷贝。
- 精度：与 torch_npu 原生参考**逐位一致**（`rtol=0, atol=0`），jit / aclnn 两种模式同一验收标准。

## 调用说明

| 调用方式 | 说明 |
| -------- | ---- |
| Kernel 直调（PyPTO） | `atten_epilogue(o, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos, out)`：校验入参 → 取/建缓存的扁平 workspace → 构造 `EpilogueTiling(t, padded)` → 启动 `atten_epilogue_kernel[None, cube_core_num, tiling_key_for(t, cube_core_num)](...)`。 |
| aclnn API | 已支持（run 包 built-in）。`aclnnAttenEpilogueGetWorkspaceSize` + `aclnnAttenEpilogue`；测试桥见 `tests/aclnn_bridge/aclnn_bridge_atten_epilogue.cpp`，`pytest --mode aclnn`。 |

两种模式的入口签名一致（输出由调用方预分配），因此同一组用例、同一组 golden 可在
`--mode jit` 与 `--mode aclnn` 下复用。

## 目录结构

```
atten_epilogue/
├── op_kernel/
│   └── atten_epilogue.py          # PyPTO delivery-form kernel（raw Ptr + EpilogueTiling）+ jit 宿主入口
├── op_host/
│   ├── CMakeLists.txt
│   ├── atten_epilogue_def.cpp          # 算子注册（7 输入 / 1 输出，无属性）
│   ├── atten_epilogue_infershape.cpp   # out = [T, 5120]
│   ├── atten_epilogue_tiling.h         # EpilogueTilingData（与 kernel dataclass 字节对齐）
│   └── atten_epilogue_tiling.cpp       # host tiling：校验 + workspace 申报 + tiling key + blockDim
└── tests/
    ├── CMakeLists.txt
    ├── aclnn_bridge/
    │   ├── CMakeLists.txt
    │   └── aclnn_bridge_atten_epilogue.cpp   # pybind11 桥（含 FP8 dtype 映射）
    ├── assets/
    │   ├── atten_epilogue_golden_cpu.py      # 独立 CPU/FP64 oracle
    │   ├── atten_epilogue_kernel_consts.py   # 形状与 workspace 布局常量
    │   └── atten_epilogue_precision.py       # 逐位/逐字节比较器
    ├── atten_epilogue_call.py                # 用例共享调用 helper
    └── test_atten_epilogue.py
```

## 验证

```bash
# 一键（构建 run 包 + 隔离安装 + aclnn 模式系统测试）
python3 build_ci.py -c --pkg-type run --ops atten_epilogue --stest

# 单独跑用例
pytest ops/pro/deepseek_v4_1/atten_epilogue/tests --mode jit -q     # 57 passed
pytest ops/pro/deepseek_v4_1/atten_epilogue/tests --mode aclnn -q   # 45 passed, 12 deselected
```

用例覆盖（`--mode aclnn` 跑 45 个双模式用例，`--mode jit` 额外跑 12 个 `jit_only`）：

| 用例 | 个数 | 模式 | 内容 |
| ---- | ---: | ---- | ---- |
| `test_supplied_native_exact` | 8 | 双模式 | 8 组 `(T, scale)` 逐位对齐原生参考 + 哨兵保护带（越界/补齐行未写） |
| `test_repeat_call_exact` | 5 | 双模式 | `T ∈ {1,72,256,257,2048}` 连续两次调用（第二次取负）均逐位正确 |
| `test_token_boundaries` | 26 | 双模式 | 16 行补齐边界 + 三档 profile 切换点 + 大 T（至 16384） |
| `test_dynamic_token_range` | 1 | 双模式 | `T = 1..256` 全量扫描 |
| `test_independent_cpu_projection` | 1 | 双模式 | 独立 CPU/FP64 oracle |
| `test_cpu_quantization_matches_native` | 1 | 双模式 | CPU MX 量化器与原生量化器逐字节一致 |
| `test_comparator_rejects_corruption` | 1 | 双模式 | 比较器自检（纯 CPU） |
| `test_workspace_budget_covers_payloads` | 1 | 双模式 | workspace 预算与 4 个载荷尺寸/对齐三方对账（纯 CPU） |
| `test_mode_is_dual` | 1 | 双模式 | 入口解析自检（信息性） |
| `test_gm_payloads_exact` | 8 | jit_only | 8 组 `(T, scale)` 的 `aq/asc/yq/ys` 逐字节对齐原生参考 |
| `test_workspace_cache_reuse` | 1 | jit_only | workspace 缓存复用与指针稳定 |
| `test_bounded_tiling_profiles` | 1 | jit_only | tiling key schema 可枚举且选择器不越界 |
| `test_host_contract_validation` | 1 | jit_only | 宿主入口 9 类故障注入 |
| `test_primitive_only_implementation` | 1 | jit_only | kernel 只走 tile/primitive API |

`jit_only` 用例需要直接持有 kernel 模块或 workspace（GM 载荷逐字节核对、缓存指针、
tiling key 枚举、宿主契约校验、源码检查），built-in 路径下这些对象由框架持有，故
`--mode aclnn` 自动排除（`build_ci.py --stest` 会追加 `-m "not jit_only"`）。

## 性能

以下为 **built-in（aclnn 预编译二进制）形态自身**的上板实测数据，调用链为
`aclnnAttenEpilogueGetWorkspaceSize` → `aclnnAttenEpilogue` → 预编译 `AttenEpilogue_<md5>.o`，
不含 jit 编译开销，也不含 host 侧 launch 开销（指标取的是设备侧 task duration）。

### 硬件平台与软件栈

| 项 | 值 |
| --- | --- |
| SoC | `Ascend950PR_9589`（Short SoC `Ascend950`，`NpuArch=3510`，`AIC-C-310`） |
| Cube 核 | **32**，`cube_freq = 1650 MHz` |
| Vector 核 | **64**，`vec_freq = 1650 MHz` |
| 核配额 | `Block Num = 32` / `Mix Block Num = 64`，即 `KERNEL_TYPE_MIX_AIC_1_2`（1 AIC : 2 AIV） |
| 分档 | 32 核属 `wide` 布局（`BN1=256 / BN=160`），8 个 case 共命中 3 个 tiling key |
| CANN | `Ascend-cann-950-ops 9.3.0`（`V100R001C13B036`），`ASCEND_HOME_PATH=/home/z00521416/AscendOnlyForZyt/cann-9.3.0` |
| PyPTO | `pypto 0.3.0` / `pypto_pro`（随 9.3.0 包） |
| 采集工具 | `msprof --task-time=on`，取 `op_summary_*.csv` 的 `Task Duration(us)` 列 |
| 设备 | 单卡 `npu:1`（`TILE_FWK_DEVICE_ID=1`），采集期间该卡无其它负载 |

### 采集方法（L2 冷启动 + 中位数）

- **每样本前冲刷 512 MiB L2**：`torch.ones(512*1024*1024//4, dtype=float32).sum()`，
  与被测算子同 stream 串行，保证 kernel 启动时 L2 为冷（flush task 自身中位 335.1~357.4 us，不计入结果）。
- **3 轮预热（丢弃）+ 7 轮正式样本**，每个 T 独立一次 msprof 采集。
- **取 7 轮正式样本的中位数**作为该 case 的性能数值；同时给出 min~max 与 stdev 以体现散布。
- 计时前置门禁：同一组输入下先比对 built-in 输出与原版 jit 实现输出**逐位一致**
  （原版已在测试套件中逐位对齐 torch_npu 原生参考），不一致直接拒绝计时，
  保证测的是同一份正确工作量。8 个 T 全部通过。
- 解析侧强制校验：每个计入的 task 必须**紧邻同 stream 的 flush 前导**；每 case 恰好 10 个
  flush 前导 task（3+7）；核配额必须恒为 `32/64`。任一不满足即报错退出。
- 未使用 NPUGraph：交付形态传 tiling 结构体，jit launcher 每次启动都要 H2D 拷贝，
  GLOBAL 捕获模式会拒绝；且 task duration 本就不含 host launch 开销，用不上图。

### Case shape 规格

8 个 case 只有 `T` 变化，其余维度是算子的固定契约：

| 张量 | Shape | dtype | 说明 |
| --- | --- | --- | --- |
| `o` | `[T, 64, 512]` | BF16 | N=64 head × D=512 |
| `woa` | `[8, 1024, 4096]` | FP8E4M3FN | ng=8 group × o_lora=1024 × K=4096 |
| `descale_woa` | `[8, 1024, 64, 2]` | FP8E8M0 | 每 32 个 K 元素一个 scale |
| `wob` | `[5120, 8192]` | FP8E4M3FN | dim=5120 × K=8192 |
| `descale_wob` | `[5120, 128, 2]` | FP8E8M0 | |
| `rope_sin` / `rope_cos` | `[T, 64]` | FP32 | inverse RoPE 表 |
| `out` | `[T, 5120]` | BF16 | |

权重与 scale 张量与 `T` 无关，合计约 74.3 MiB（`woa` 32 MiB + `wob` 40 MiB +
`descale_woa` 1 MiB + `descale_wob` 1.25 MiB），每个样本前都被 512 MiB flush 逐出，
因此两级 GEMM 的权重读取均为 L2 冷命中。

### 实测结果

`padded = ceil(T/16)*16`；`workspace = 42240 * padded` 字节（`aq 32768 + asc 1024 + yq 8192 + ys 256`，每 padded 行）；
`GEMM1 tiles = ng * ceil(T/BM) * ceil(1024/BN1)`，`GEMM2 tiles = ceil(T/BM) * ceil(5120/BN)`。

| T | padded | tiling key | BM/BK/VSlots/K0/BN1/BN | workspace | GEMM1 tiles | GEMM2 tiles | **task duration 中位数 (us)** | min~max (us) | stdev | cube util |
| ---: | ---: | ---: | :-- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 60 | 64/512/2/256/256/160 | 660 KiB | 32 | 32 | **55.23** | 54.90~55.86 | 0.6% | n/a |
| 33 | 48 | 60 | 64/512/2/256/256/160 | 1980 KiB | 32 | 32 | **59.21** | 58.89~59.79 | 0.5% | 96.57% |
| 72 | 80 | 41 | 128/256/2/128/256/160 | 3300 KiB | 32 | 32 | **64.61** | 63.86~65.63 | 1.1% | 96.92% |
| 128 | 128 | 41 | 128/256/2/128/256/160 | 5280 KiB | 32 | 32 | **72.96** | 72.44~73.61 | 0.6% | n/a |
| 129 | 144 | 34 | 256/256/1/128/256/160 | 5940 KiB | 32 | 32 | **71.92** | 71.77~72.79 | 0.6% | 96.93% |
| 193 | 208 | 34 | 256/256/1/128/256/160 | 8580 KiB | 32 | 32 | **82.51** | 80.53~83.35 | 1.3% | n/a |
| 255 | 256 | 34 | 256/256/1/128/256/160 | 10560 KiB | 32 | 32 | **89.57** | 88.71~91.78 | 1.1% | n/a |
| 256 | 256 | 34 | 256/256/1/128/256/160 | 10560 KiB | 32 | 32 | **88.36** | 87.54~89.35 | 0.7% | 96.46% |

- 样本数：每 case 7 个正式样本（另有 3 个预热样本被丢弃）；数值为 `msprof` 的 NPU task duration。
- 样本散布 0.5%~1.3%，中位数稳定。
- `cube util` 为 `msprof` 的 `cube_utilization(%)` 列；标 `n/a` 的是该列在当次采集中未产出
  （同一档位内 T=1 与 T=33 表现不一致，属采集缺失，不影响 task duration）。

### 观察

- **8 个 case 的 tile 数恒为 32/32**：`T ≤ 256` 时 `ceil(T/BM) = 1`，GEMM1 是
  `8 group × 1 × 4 个 BN1 分块 = 32`，GEMM2 是 `1 × ceil(5120/160) = 32`，恰好每 Cube 核一个
  输出 tile，无跨核 K 拆分、无尾核空转。因此这一段 `T` 区间内耗时增长主要来自
  `BM` 档位切换（64→128→256 时单 tile 的 M 维变大）与 L2 冷权重搬运量，而非核数利用变化。
- **T=129 (71.92 us) 与 T=128 (72.96 us) 基本持平**（差 1.4%，接近 0.6% 的样本散布量级，不宜过度归因）：
  两者 tile 数同为 32/32，但落在不同档位——T=128 走 `BM=128`（key 41，`VSlots=2`，
  C→V 半块 64 行，`_quantize_gemm1_output` 走单趟分支）；T=129 走 `BM=256`
  （key 34，`VSlots=1`，半块 128 行，量化按 `Q_ROWS=64` 分 2 个 chunk）。
  即档位切换点两侧没有性能台阶。
- **T=255 (89.57 us) 比 T=256 (88.36 us) 慢约 1.4%**：同属 `BM=256` 档、`nm=1`，
  但 255 行的 `validshape` 不对齐 16，触发尾行处理路径。
