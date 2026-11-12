# block_attn_res_update_rms_norm 结果说明

本文给出 `block_attn_res_update_rms_norm` 的精度与性能结果：用哪些用例检验、按什么标准判定、
实测数据是多少、以及这些数据说明什么。所有数值均为本机实测。设计与取舍见 [DESIGN.md](DESIGN.md)。

## 1 用例

**测试环境**

| 项目 | 取值 |
| :--- | :--- |
| 硬件平台 | Ascend 950PR_9589（32 Cube / 64 Vector），`get_platform_info()` = `PlatformInfo(soc_version='DAV_3510', vector_core_num=64)`，单卡共享设备 |
| 软件环境 | CANN 9.3.0（`compiler/version.info` 报 `Version=9.2.0`）、pypto 0.3.0（源码版）、Python 3.14.6、torch 2.13.0+cpu、torch_npu 2.13.0rc1 |
| 计时工具 | torch_npu 进程内 profiler（ACL task-based）采集 `op_summary` 的内核 `Task Duration`；本机 msprof 外部启动被驱动策略拒绝（EPERM），进程内采集与其口径一致 |

**精度用例清单**

`tests/test_block_attn_res_update_rms_norm.py` 里的用例（pytest，一个函数对应一组用例，共 21 项）：

| 组 | 用例 | 覆盖点 |
| :--- | :--- | :--- |
| shapes | `T = 1, 2, 4, 8, 32, 64, 128, 254, 512, 1024` | 正常形状、`T = 1` 的单核路径、`T` 与 64 核不整除（`T = 254`）、`T` 超过核数（`512/1024`） |
| unlisted shapes | `T = 3, 5` | 不参与特化选择的合法尺寸 |
| numeric regressions | `T = 8` 两个种子的 cancellation、`T = 8/254` 的 `ftz`、`T = 8` 的 `recip_ftz` | `dot ≈ 0` 的病态消去、denormal 输入、倒数近似相关输入 |
| routing / transitions | `T = 8 → 5 → 4 → 3 → 2 → 8`（同一 kernel 对象连跑） | `T = 3`（阈值本身）、未参与调优的合法尺寸 `5`、跨类别来回切换不留残留状态 |
| empty / validation | `T = 0`、`D != 7168`、`delta` 传 FP32、`score_eps < 0`、CPU 张量 | 空输入与入参校验（非法输入必须报错而非静默出错） |

**性能用例清单**

8 个用例，`D = 7168`，`T ∈ {1, 2, 4, 8, 32, 64, 254, 512}`，输入用固定种子 42 生成，
每次调用前把 `partial_block` 恢复成初值（该算子会原地改写它）。

## 2 精度标准

判定标准（`tests/assets/block_attn_res_update_rms_norm_precision.py`）：

- `np.isclose(candidate, reference, rtol=1e-3, atol=0)` 的不满足元素比例 ≤ `1e-3`；
- 任一元素非有限（NaN/Inf）直接判失败；
- `y` 与 CPU 参考比较，`p` 与 `partial + fp32(delta)` 比较；
- 除 `partial_block` 外的 6 个输入在调用后必须与调用前逐位相等（只读性检查）；
- 输出 shape 必须为 `[T, 7168]`、dtype 为 BF16、device 与输入一致。

cancellation 用例单独用「2 倍参考」规则：对称构造的 `p` 使 `dot ≈ 0`，任何 FP32 归约顺序
（包括 golden 自身）相对 float64 预言机都会超过 `1e-3` 的比例门限，因此要求
`kernel_bad ≤ 2 × max(golden_bad, 1)`，非有限值仍判失败。

### 2.1 结果

`TILE_FWK_DEVICE_ID=<device-id> pytest ops/pro/kimi_k3/block_attn_res_update_rms_norm/tests --mode jit -q`
→ **21 passed**（64 向量核环境，多轮复验通过）：

| 组 | 结果 |
| :--- | :--- |
| shapes（10 个 `T`） | 全部 PASS（`y`、原地 `p`、shape、只读性） |
| numeric regressions | cancellation `T = 8`：kernel_bad ≤ 2×golden_bad 全过；`ftz` / `recip_ftz` 全部 PASS |
| routing / transitions | PASS，`select_path` 在 `T = 0/1/2` 返回 0，`T = 3/4/1024` 返回 1 |
| empty / validation | `T = 0` 返回空 `[0, 7168]` BF16；5 类非法入参全部报 `ValueError` |

### 2.2 形状与边界行为

- `T = 0`：不启核，直接返回空张量；`T ≥ 1` 时起核数为 `min(64, T)`。
- `T = 1/2` 走 `Path = 0`（整行单体），`T ≥ 3` 走 `Path = 1`（分行段）。实测交叉点（两条
  路径强制各跑一遍）：`T = 2` 单体 2.574 vs 分行段 2.696、`T = 4` 单体 3.416 vs 分行段
  3.080（µs），即 `T < 3` 单体占优、`T ≥ 3` 分行段占优。
- `T = 254` 时 64 核分到 4/3 行不等（`254 = 64×3 + 62`），无尾块、无空转核，精度 PASS。
- `D != 7168`、非连续张量、跨 device、非正 epsilon 均在 host 侧报错。

## 3 性能结果

**测量方法**：每个用例单独进程，50 次预热 + 200 次受测启动，指标取内核 `Task Duration` 的
p50（单位 μs，设备端内核时间，不含预热、输入恢复和 host 调用开销），pypto 0.3.0 codegen。

### 3.1 绝对性能结果

| 用例 | 形状 | 起核数 | p10（µs） | p50（µs） | p90（µs） |
| :--- | :--- | ---: | ---: | ---: | ---: |
| T1 | `[1, 7168]` | 1 | 2.397 | 2.493 | 2.543 |
| T2 | `[2, 7168]` | 2 | 2.554 | 2.580 | 2.617 |
| T4 | `[4, 7168]` | 4 | 3.044 | 3.089 | 3.155 |
| T8 | `[8, 7168]` | 8 | 3.090 | 3.159 | 3.239 |
| T32 | `[32, 7168]` | 32 | 3.260 | 3.536 | 3.818 |
| T64 | `[64, 7168]` | 64 | 3.886 | 4.339 | 4.747 |
| T254 | `[254, 7168]` | 64 | 7.964 | 8.490 | 8.995 |
| T512 | `[512, 7168]` | 64 | 12.755 | 13.260 | 13.968 |

同一实现的复测波动 ≤ 0.4%（共享设备）。

### 3.2 理论评估对比（流量与有效吞吐）

每行固定流量：`partial` 读+写各 4·D、`delta` 读 2·D、`numerator` 读 4·D、`y` 写 2·D；
`query`（4·D）与 `gamma`（2·D）每核只搬一次。有效吞吐 = 上表总流量 ÷ p50。

| 用例 | GM 流量（MB） | p50（µs） | 有效吞吐（TB/s） |
| :--- | ---: | ---: | ---: |
| T8 | 3.67 | 3.159 | 1.16 |
| T32 | 6.42 | 3.536 | 1.82 |
| T64 | 10.09 | 4.339 | 2.33 |
| T254 | 31.88 | 8.490 | 3.76 |
| T512 | 61.47 | 13.260 | 4.64 |

小 `T` 由串行的三阶段依赖 + 两次全行归约决定；`T ≥ 254` 行重叠吃满后吞吐升到 3.8–4.6 TB/s。
`T512` 工作集约 61 MB，小于本平台 L2，稳态读多来自 L2，该吞吐不代表 HBM 带宽上限。

### 3.3 剩余优化空间

生成代码取证显示两处精确除法（`inv_den`、`rstd`）仍各占 ~60 条指令/行（合计 ~7% 的
向量发射预算）；若要继续收敛，可把它们也移到标量核，但 `inv_den` 需要 SIMT 侧标量 `exp`
（当前需 tile 级 `pl.exp` 往返）并把 phase23 拆成两个 VF，成本/风险待评估。

## 4 已知限制

- 需要 **pypto_pro 0.3.0**（依赖 `StoreDist.PACK_B32` 等宽度限定的 StoreDist 变体）。
- 在 Ascend 950PR_9589（32C/64V）+ CANN 9.3.0 上验证通过；UB 预算（峰值 229632 / 253952 B）
  与 tiling-key 接口依赖平台，换芯片需重新确认。
- `D` 固定 7168，不支持其他隐藏维；`T` 是唯一动态轴。
- 本机为共享设备，跨会话测量存在 ±2% 级环境漂移；本文件数据为单会话同 harness 结果。
