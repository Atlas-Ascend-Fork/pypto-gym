# msprof op 采集与分析

本文命令中的 `PERF_SKILL_ROOT` 为本 skill 实际目录的绝对路径，执行前按安装位置设置。

## Contents

- [前提](#prerequisites)
- [Step 1：构建算子](#build-operator)
- [Step 2：采集](#collect-profile)
- [Step 3：归档 + 统计摘要](#archive-summary)
- [Step 4：验证优化效果](#validate-optimization)
- [数据目录结构](#data-layout)
- [上板 vs 仿真选择](#target-selection)
- [注意事项](#cautions)
- [相关资源](#resources)


> 标准上板采集流程：依赖 `$ASCEND_HOME/tools/msopprof/bin/msopprof`，产出 8 份独立 CSV + 逐核 `PipeUtilization.csv`，覆盖 **构建 → 采集 → 归档 → 回归**；判定（达标、瓶颈定位）统一走[指标运用指导](metric-application-guide.md)。

---

## <a id="prerequisites"></a>前提

- 环境中存在 `$ASCEND_HOME/tools/msopprof/bin/msopprof`
- 推荐用法包含 `--warm-up` 以规避 DVFS

---

## <a id="build-operator"></a>Step 1：构建算子

> 本步仅适用 AscendC 场景（直调/aclnn 可执行文件）。PyPTO-Pro 算子不经过此构建流程：
> 官方入口是 `msprof python3 test_{op}.py`（见[性能分析入口](performance-analyze.md)
> 工具选择）；`msprof op` 仅在当前 CANN 明确支持 Python runner 时使用，入口按
> [增强指南](msprof-op-enhanced-guide.md)的包装脚本模式（无后缀可执行文件，内部
> `exec python3 test_{op}.py --case-id <id>`）准备。

**直调算子**：

```bash
cd ops/{operator_name} && mkdir -p build && cd build && cmake .. && make -j
```

**aclnn 算子**：

```bash
bash build.sh --pkg --soc=ascend910b --ops={operator_name} --vendor_name=custom -j16
./build_out/*.run --install-path=$CANN
bash build.sh --run_example {operator_name} eager cust --vendor_name=custom
```

---

## <a id="collect-profile"></a>Step 2：采集

```bash
# 基本用法
msprof op ./demo

# 推荐用法（含预热 + 指定输出）
msprof op --warm-up=10 --output=./msprof_output ./demo

# 多次运行取均值
msprof op --warm-up=10 --launch-count=5 --output=./msprof_output ./demo
```

### 关键参数

| 参数 | 说明 | 何时使用 |
|------|------|---------|
| `--warm-up=N` | 预热 N 次后再采集 | **始终建议**，避免 DVFS（动态调频）影响首次运行 |
| `--kernel-name="<name>[|<name>...]"` | 只采集名称匹配的算子；支持前缀模糊匹配、`*` 通配和 `|` 多选 | **多 kernel 场景必须指定**；不指定时只采第一个被调度的算子 |
| `--launch-count=N` | 采集算子的最大数量，默认 1 | 需统计/采集多个匹配算子时 (`1~5000`) |
| `--launch-skip-before-match=N` | 从第一个算子起跳过 N 个再开始匹配采集 | app 前面有无关算子/预热时 (`0~1000`) |
| `--output=<dir>` | 指定输出目录 | 避免结果散落 |
| 无需 `--soc-version` | 上板自动检测硬件 | — |

**多 kernel app 必须显式过滤算子**：`msprof op` 默认只采集**第一个被调度的算子**。

不指定 `--kernel-name` 时，一个 `flash_kda_wrapper` 启动的多个 kernel
（stage1/stage2 + `aclnn*` 前置算子）中，目标 kernel 不会被 dump：日志出现
`Parsing .../tmp_dump/device0 failed, which maybe empty or too many kernels`，
`OpBasicInfo.csv`/`PipeUtilization.csv` 只剩一个 `aclnn` 算子的行，逐核数据缺失，
无法计算负载不均衡率。

PyPTO-Pro 的 kernel 名是 mangled 符号（如 `_Z23flash_kda_stage2_kernelPu6__bf16...`）。
按前缀匹配填写：

- `--kernel-name="*flash_kda_stage2*"`（或 `_Z23flash_kda_stage2`）；
- `--launch-count=1`（或按需）。

通过两类日志提示确认是否命中：`Kernel ... skipped: not selected via --kernel-name`、
`No profiling data matched --kernel-name=...`。

**输出**：在指定目录或当前目录下生成 `OPPROF_{timestamp}_XXX/` 文件夹。

---

## <a id="archive-summary"></a>Step 3：归档 + 统计摘要

```bash
# 找到最新 OPPROF 目录
OPPROF_DIR=$(ls -td <output_dir>/OPPROF_* | head -1)

# 归档 CSV + 生成摘要（自动创建 docs/perf/round_NNN/）
python3 "$PERF_SKILL_ROOT/scripts/perf_summary.py" $OPPROF_DIR ops/{operator_name}
```

脚本会：

1. 在 `ops/{operator_name}/docs/perf/round_NNN/` 创建归档目录（轮次自动递增）
2. 复制全部 8 个 CSV 原始文件到归档目录
3. 生成 `summary.txt`（各指标 min/avg/max，**不做达标判定**）

归档完成即结束本文职责：达标判定、瓶颈定位与优化方向见[指标运用指导](metric-application-guide.md)的
"达标判定"节。

---

## <a id="validate-optimization"></a>Step 4：验证优化效果

每次优化后重新执行 Step 2–3，数据自动归档为 `round_NNN+1`。

```bash
# 对比两轮摘要
diff ops/{operator_name}/docs/perf/round_001/summary.txt ops/{operator_name}/docs/perf/round_002/summary.txt

# 或直接读两个 summary.txt 进行对比分析
```

**对比要点**：

1. Task Duration 是否下降
2. 瓶颈单元的 ratio 是否改善
3. 核间均衡是否改善（aiv_time min/max 差距）
4. 是否引入新的瓶颈

---

## <a id="data-layout"></a>数据目录结构

### 临时输出（采集后、归档前）

```
OPPROF_{timestamp}_XXX/
├── dump/                       # 原始性能数据（无需关注）
├── OpBasicInfo.csv             # 算子基本信息（名称、核数、总耗时、频率）
├── PipeUtilization.csv         # 各流水线单元耗时和占比（最重要）
├── ArithmeticUtilization.csv   # Cube/Vector 指令 cycle 占比和计算量
├── Memory.csv                  # 内存读写带宽和数据搬运量
├── MemoryL0.csv                # L0A/L0B/L0C 读写带宽
├── MemoryUB.csv                # UB 读写带宽（Vector/Scalar）
├── L2Cache.csv                 # L2 Cache 命中率
├── ResourceConflictRatio.csv   # Bank conflict 和资源冲突占比
└── visualize_data.bin          # MindStudio Insight 可视化文件
```

### 持久归档

```
ops/{算子名}/docs/perf/
├── round_001/
│   ├── OpBasicInfo.csv
│   ├── PipeUtilization.csv
│   ├── Memory.csv
│   ├── ResourceConflictRatio.csv
│   ├── L2Cache.csv
│   ├── ArithmeticUtilization.csv
│   ├── MemoryUB.csv
│   ├── MemoryL0.csv
│   └── summary.txt            # 统计摘要（min/avg/max，不含判定）
├── round_002/
└── ...
```

字段含义详见 [`csv_fields_reference.md`](csv_fields_reference.md)。

---

## <a id="target-selection"></a>上板 vs 仿真选择

| 维度 | 上板 (msprof op) | 仿真 (msprof op simulator) |
|------|-----------------|---------------------------|
| 需要 NPU | 是 | 否 |
| 时序精度 | 真实硬件时序 | 周期级模型估算 |
| 输出 | 8 个 CSV 文件 | CSV + trace.json |
| 指令级流水图 | 需加参数或单独仿真 | 默认输出 |
| **资源冲突数据** | **有**（ResourceConflictRatio.csv） | 无 |
| **L2 Cache** | **真实命中率** | 估算 |
| DVFS 影响 | 有（需 warm-up） | 无 |
| 适合阶段 | 性能验收、生产调优 | 早期开发、指令级调试 |

**建议**：开发阶段用仿真快速迭代，验收阶段用上板确认真实性能。

> 边界：仿真列仅适用 AscendC 场景。PyPTO-Pro 当前没有已核实的 sim-run 选项，
> 不使用该仿真路径；性能结论一律以上板数据为准（见[证据协议](evidence-protocol.md)末节）。

---

## <a id="cautions"></a>注意事项

1. **必须 warm-up**：首次运行受 DVFS 影响，耗时偏高。始终使用 `--warm-up=10`
2. **频率检查**：读取 `OpBasicInfo.csv` 的 `Current Freq` 和 `Rated Freq`，若 Current < Rated，说明芯片未满频运行
3. **MTE2/MTE3 带宽共享**：同时读写 GM 时总带宽共享，评估搬运段负载时宜按 MTE2、MTE3 合并字节量与平台带宽对照
4. **小数据量场景**：数据量很小时头开销占比会很高，这不一定是算子问题，而是数据量不足
5. **多核同地址访问**：多核同时读同一 512B 地址范围会被串行化，导致 MTE2 耗时异常
6. **多 kernel 必须 `--kernel-name`**：不指定时只采第一个调度算子，目标 kernel 的逐核
   CSV 会缺失（见[关键参数](#collect-profile)的多 kernel 说明）；采集后先确认
   `OpBasicInfo.csv` 的 `Op Name`/`Block Dim` 就是目标 kernel，再往下判读。负载均衡
   判定不在这里，走[增强指南](msprof-op-enhanced-guide.md)的 Core Occupancy /
   [指标运用指导](metric-application-guide.md)。

---

## <a id="resources"></a>相关资源

| 文件 | 内容 |
|------|------|
| [`csv_fields_reference.md`](csv_fields_reference.md) | 8 个 CSV 文件的完整字段定义和阈值 |
| [`msprof-op-enhanced-guide.md`](msprof-op-enhanced-guide.md) | Timeline / Core Occupancy / Roofline 的提取与分析用法 |
| `../../scripts/perf_summary.py` | 统计摘要生成 + CSV 归档（Step 3 调用） |
