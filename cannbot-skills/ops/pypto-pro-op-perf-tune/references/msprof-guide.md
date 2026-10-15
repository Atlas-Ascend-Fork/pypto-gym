# `msprof` 采集与分析

本文命令中的 `PERF_SKILL_ROOT` 为本 skill 实际目录的绝对路径，执行前按安装位置设置。

## Contents

- [适用场景](#scope)
- [脚本运行模式](#script-modes)
- [PyPTO-Pro 输入与输出](#pypto-io)
- [平台与工具选择](#platform-tool-selection)
- [Step 1：构建算子（如果有指定的调用方式，这一步可跳过）](#build-operator)
- [Step 2：采集](#collect-profile)
- [Step 3：归档 + 统计摘要](#archive-summary)
- [主 Bound 判定（msprof 归档）](#bound-classification)
- [数据目录结构](#data-layout)
- [通用分析纪律](#analysis-discipline)
- [注意事项](#cautions)
- [相关资源](#resources)


> 使用各 CANN 自带的 `msprof`：多组 `--aic-metrics` + sample-based `aicore.db`，得到可分析的 `op_summary_*`、`per_core_cycles.csv` 与 `summary.txt`。

---

## <a id="scope"></a>适用场景

- 目标环境以 `msprof` 为采集手段，或团队约定采用本工具链。

| 场景 | 推荐命令 | 说明 |
|---|---|---|
| PyPTO 算子性能采集 | `msprof_profile_run.sh` + `msprof_perf_summary.py` | 标准采集 7 组 aic-metrics + 深度瓶颈分析 |
| PyPTO 算子 vs golden 标杆对比 | `msprof_profile_run.sh --compare` | Golden 报告 + target-kernel Task Duration 得到参考比值 |
| 优化前后回归 | `msprof_profile_run.sh` + `msprof_perf_summary.py` | 对比独立归档 round |
| 快速候选筛选 | `msprof_profile_run.sh --quick` | 只采 Task Duration，不替代最终正式采集 |
| 批量性能测试 | `msprof_profile_run.sh --batch` | 多 NPU 并行批量测试 |

---

## <a id="script-modes"></a>脚本运行模式

本节保留不传 `--case-manifest` 时的既有兼容模式及命令语义。本 skill 性能调优使用的 compare、quick 和 batch
必须改按[证据协议](evidence-protocol.md#compare-quick-batch)执行；两种模式的 Golden、seed、逐 case
选择和输出文件不同，不能混用。

### 1. 标准采集模式（深度瓶颈分析）

按下文 [Step 2](#collect-profile) 和 [Step 3](#archive-summary)采集七指标与 sample-based 证据；
把采集命令中的 demo 替换为 `python3 test_{op}.py`，并用精确 `--op-name` 解析目标 kernel。

### 2. 对比测试模式（跨口径参考比值）

从 `GOLDEN_PERF_REPORT.md` 读取 Golden 非噪声 device-kernel E2E 总时长，用 msprof 采集 PyPTO
target-kernel Task Duration，计算 `Golden E2E / PyPTO target-kernel` 参考比值。两端计时范围不同，
该值不是同口径 optimization speedup，也不能作为正式性能对比（formal compare）。为兼容既有产物，脚本
仍使用 `speedup`、`*_speedup` 和“加速比”等旧字段名；解释时必须遵循本段语义。

行为说明：

- `--warm-up=N` 会在正式采集前真实执行 N 次，默认为 3。
- `--repeats=N` 在对比和快速模式中都生效，默认为 1。N 大于等于 3 时去掉最大值和最小值后取平均；N 小于 3 时取中位数。
- `--seed=N` 通过 `PYPTO_PERF_SEED` 传入测试脚本，同时设置 `PYTHONHASHSEED`。测试脚本应读取 `PYPTO_PERF_SEED` 来固定随机输入。
- msprof 临时数据写入算子目录下的 `.msprof/`，并使用 PID 和时间戳隔离并发任务。

```bash
bash "$PERF_SKILL_ROOT/scripts/msprof_profile_run.sh" --compare \
    --output-dir=./custom/{op} --warm-up=3 \
    --device=0 --op-name=<Op Name> --repeats=3 --seed=0
```

> **前置条件**：兼容 `--compare`、`--quick` 均需要 `custom/{op}/GOLDEN_PERF_REPORT.md`，兼容 `--batch` 的每个算子目录也需要该报告。正确性 Golden 不代表已采集性能报告；只有用户明确要求 golden 性能采集或 golden 基线对比时，才先以 `collect_golden_perf=true` 调用 `pypto-pro-golden-generate` 生成报告。若用户只要求 PyPTO kernel 自身的性能采集或瓶颈分析，使用标准模式，不要为了它自动采集 golden。
>
> **必须指定 `--op-name`**：与标准采集模式同理，PyPTO 测试脚本含多个 case + `torch.randn`，不指定会选到非目标 op。Op Name 获取方法见 [Step 3](#archive-summary)。
>
> **逐 case 限制**：该既有兼容模式没有标准的逐 case 选择协议。如果 `GOLDEN_PERF_REPORT.md` 包含多个 case，工具会明确报错，不会把一次聚合耗时复用到所有 case。请按 case 分别产出报告后再对比。

输出：`performance.json`、`performance.log`、`perf_report.md`。

### 3. 快速模式（不采集 aic-metrics）

只采集 kernel Task Duration，不采集 7 个 aic-metrics，适合快速验证优化效果。`--warm-up`、`--repeats`、`--seed` 和逐 case 限制与对比模式一致。

```bash
bash "$PERF_SKILL_ROOT/scripts/msprof_profile_run.sh" --quick \
    --output-dir=./custom/{op} --warm-up=3 \
    --device=0 --op-name=<Op Name> --repeats=3 --seed=0
```

输出：`performance.json`、`performance.log`、`perf_report.md`。

### 4. 批量并行模式（多 NPU）

扫描目录下所有算子子目录，多 NPU 并行执行对比测试：

```bash
bash "$PERF_SKILL_ROOT/scripts/msprof_profile_run.sh" --batch --base-dir=./custom --max-jobs=7 --device-start=1
```

输出：各子目录 `performance.json` + `batch_performance.log` + `batch_report.md` + `batch_summary.json`

> **使用边界**：该兼容 batch 不传逐目录 `--op-name`，仅适用于每个 runner 的目标 kernel 可唯一归属的目录；含多个 AI_CORE op 时，使用 manifest batch，并传入精确的 `--op-name` 或 `--op-name-map`。

---

## <a id="pypto-io"></a>PyPTO-Pro 输入与输出

采集脚本执行算子目录中已有、已按算子合同通过全量正确性测试的 runner；优化前后必须保留
wrapper 边界、Golden 语义与精度门槛。独立输入的要求见[主 Skill](../SKILL.md)；若项目使用
[Develop Skill](../../pypto-pro-op-develop/SKILL.md)，也须保留其测试合同。逐 case
输入、Golden 机器合同和输出证据以[证据协议](evidence-protocol.md)为准。不传 manifest 的兼容
compare/quick 输出仍按上面的模式说明生成。

---

## <a id="platform-tool-selection"></a>平台与工具选择

在加载任何平台专属模型前，先运行
[`../../pypto-pro-environment-check/scripts/get_npu_arch.py`](../../pypto-pro-environment-check/scripts/get_npu_arch.py)
或读取构建配置，记录设备型号、`NpuArch`、CANN/PyPTO 版本和 `pypto_pro.__file__`：

- 仅当结果确认 raw 输出 `3510`（`NpuArch` 值）、helper 输出 `dav-3510`、runtime 输出 `DAV_3510`
  或构建目标 `dav-c310` 时，加载
  [A5 roofline 与杠杆](general-knowledge/a5-roofline-and-levers.md)；
- 对未知平台或非 A5 平台，不应用 A5 核数、容量、带宽、频率或经验结论，直接使用该平台的
  官方资料与本次 profiler 数据；
- 平台探测失败时，将平台模型标记为不可用，但仍可继续做不依赖硬编码常量的实测瓶颈分析。

工具选择顺序：

1. 用户指定 `msprof op` / msopprof 时，走 [msprof op 指南](msprof-op-guide.md)；指定
   `msprof` 时，走本文。
2. 用户未指定时探测环境：仅 `msopprof` 可用则走 msprof op；仅 `msprof` 可用则走本文；
   两者皆可用时向用户确认或遵循项目约定；两者皆不可用时报错并检查 CANN / `ASCEND_HOME`。

完整 `test_{op}.py` 的 profile 会混入输入生成、Golden 和精度检查，不能把其中的 `aclnn*`
直接归因给 wrapper，也不能据此计算 wrapper 占比。PyPTO-Pro wrapper 合规由
[wrapper 边界合同](../../pypto-pro-op-kb/constraints/wrapper-boundary.md)裁决；profile 只提供性能
证据，不能授权越界。性能结论只使用可唯一归属到目标 kernel 或冻结 case 的字段和计时口径。

---

## <a id="build-operator"></a>Step 1：构建算子（如果有指定的调用方式，这一步可跳过）

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

原生 `msprof` 每次运行只支持一个 `--aic-metrics` 组，且 `op_summary_*.csv` 是 per-op 聚合值而非逐核。流程：

1. 在正式采集前 warm-up N 次（规避 DVFS）
2. 按顺序分别采集 7 组 `--aic-metrics`：`PipeUtilization`、`ArithmeticUtilization`、`Memory`、`MemoryL0`、`MemoryUB`、`L2Cache`、`ResourceConflictRatio`
3. 额外跑一次 `--aic-mode=sample-based`，从 `device_0/sqlite/aicore.db`（`AICoreOriginalData.task_cyc`）拿到**逐核 cycle**
4. 以 aicore_time / max_cycles 反推主频，把逐核 cycle 折算成 per-core 时长

一键脚本：

```bash
bash "$PERF_SKILL_ROOT/scripts/msprof_profile_run.sh" \
     --warm-up=3 \
     --output=./msprof_output \
     -- ./demo arg1 arg2 ...
```

**关键点**：msprof 的 `op_summary.csv` **没有** `Current Freq/Rated Freq` 和逐核 `time(us)` 字段；逐核分析必须依赖 `PROF_Sample` 下的 `aicore.db`。采集落盘目录树见文末 **数据目录结构 → 临时输出**（根路径为 `--output` 下的 `PROF_GROUP_*`）。

---

## <a id="archive-summary"></a>Step 3：归档 + 统计摘要

```bash
GROUP_DIR=$(ls -td <output_dir>/PROF_GROUP_* | head -1)

# 获取目标 kernel 的 Op Name（PyPTO 多 case 场景必须指定 --op-name）
CSV=$(ls $GROUP_DIR/PROF_PipeUtilization/*/mindstudio_profiler_output/op_summary_*.csv | head -1)
grep -i {kernel_func} "$CSV"        # {kernel_func} = @pl.jit 装饰的函数名

python3 "$PERF_SKILL_ROOT/scripts/msprof_perf_summary.py" $GROUP_DIR ops/{operator_name} --op-name=<Op Name>
```

> **重要**：PyPTO 测试脚本（`test_{op}.py`）通常包含多个 case + `torch.randn`，`op_summary` 中会有大量非目标 op（如 `StatelessNormal` 噪声注入）。**必须指定 `--op-name`** 选中目标 kernel，否则会选到非目标 op 导致结果完全错误。`--op-name` 取同名 kernel 中 Task Duration 最大的行（即最大 workload 的 case）。

脚本会：

1. 在 `ops/{operator_name}/docs/perf/round_NNN/` 创建归档目录（轮次自动递增）
2. 按目标 `Op Name` 在 7 份 `op_summary.csv` 里合并列，复制为 `op_summary_<Metric>.csv`
3. 读 `PROF_Sample/.../aicore.db` 的 `task_cyc`，根据 `aicore_time(us)` 反推主频，换算每核耗时
4. 输出 `summary.txt`，在全局统计之外附加「逐核负载均衡」段，例如：

   ```
   --- 逐核负载均衡 (sample-based aicore.db) ---
     有效核数: 32  | 主频推算: 1.651 GHz (0.6058 ns/cycle)
     min=185.93us  avg=193.13us  max=199.27us
     (max-min)/max = 6.69%  ->  达标 (<10%)
     Top-3 慢核: Core3=199.27us, ...
     Top-3 快核: Core21=185.93us, ...
     [提示] 前半段 core 均值 198.17us vs 后半段 188.10us，差距 5.99%
            疑似两簇 (NUMA / L2 slice) 负载偏斜，建议尝试 block swat / 尾轮均衡策略。
   ```

5. 同步把 `per_core_cycles.csv` 归档到同目录，方便二次分析

归档完成后即结束本 Step 的职责。下游从 `summary.txt`（含逐核负载均衡段）、`op_summary_*.csv`、`per_core_cycles.csv` 读取指标后，按 **下文「主 Bound 判定」** 推导**主 bound 档位**。

---

## <a id="bound-classification"></a>主 Bound 判定（msprof 归档）

本节适用于 **`msprof` 经 Step 2～3 得到的归档目录**（`round_NNN/`）。

### 输入与输出

| 项目 | 说明 |
|------|------|
| **输入** | 从归档中的 `op_summary_*.csv`、`summary.txt` 等抽取的**单核侧**各流水线 **busy 占 case（或 task）总时长** 的百分比 |
| **不适用** | msprof 聚合指标**无**可对照流水图的气泡时间；报告中气泡列标注「不适用」，**不得**填写气泡数值 |
| **输出** | **主 bound 档位**：`MTE2 BOUND` / `CUBE BOUND` / `VEC BOUND` / `FIXP BOUND` / `MTE3 BOUND` / `SCALAR BOUND` / **无 bound** |

各流水线 busy 占比的**抽取与列映射**以本文 Step 3 产物及 [`csv_fields_reference.md`](csv_fields_reference.md) 为准；若归档 CSV 表头与文档示例不一致，**以实际表头为准**。

### 判定规则

在已得到各流水线 busy 占比后，**从上到下**匹配**第一条成立**：

| 优先级 | Bound 类型 | 判断规则 |
|--------|-----------|----------|
| 1 | MTE2 BOUND | MTE2 busy 占 case > 80%，或（MTE2 在 8 条中占比最大且 > 70%） |
| 2 | CUBE BOUND | CUBE busy 占 case > 80%，或（CUBE 在 8 条中占比最大且 > 70%） |
| 3 | VEC BOUND | PUSHQ busy 占 case > 80% |
| 4 | FIXP BOUND | FIXP busy 占 case > 80% |
| 5 | MTE3 BOUND | MTE3 busy 占 case > 80% |
| 6 | SCALAR BOUND | SCALAR busy 占 case > 80%，或 SCALARLDST busy 占 case > 80% |
| — | **无 bound** | 以上条件均不满足 |

「占比最大」比较对象为上述单元对应的统计。

---

## <a id="data-layout"></a>数据目录结构

### 临时输出（`msprof_profile_run.sh` 的 `--output` 目录下）

```
<output_dir>/PROF_GROUP_<YYYYMMDD_HHMMSS>/
├── PROF_PipeUtilization/PROF_*/mindstudio_profiler_output/op_summary_*.csv
├── PROF_ArithmeticUtilization/...
├── PROF_Memory/...
├── PROF_MemoryL0/...
├── PROF_MemoryUB/...
├── PROF_L2Cache/...
├── PROF_ResourceConflictRatio/...
└── PROF_Sample/PROF_*/device_0/sqlite/aicore.db   ← 逐核 task_cyc
```

### 持久归档

```
ops/{算子名}/docs/perf/
├── round_001/
│   ├── op_summary_PipeUtilization.csv
│   ├── op_summary_ArithmeticUtilization.csv
│   ├── op_summary_Memory.csv
│   ├── op_summary_MemoryL0.csv
│   ├── op_summary_MemoryUB.csv
│   ├── op_summary_L2Cache.csv
│   ├── op_summary_ResourceConflictRatio.csv
│   ├── op_statistic_<Metric>.csv
│   ├── task_time_<Metric>.csv
│   ├── per_core_cycles.csv
│   └── summary.txt
└── ...
```

字段含义详见 [`csv_fields_reference.md`](csv_fields_reference.md)（按 `op_summary_*` 列名对齐同类指标）。

---

## <a id="analysis-discipline"></a>通用分析纪律

本节只提供分析证据；调优项的来源、建项时机和登记要求以[主 Skill](../SKILL.md)及
[优化项实验闭环](optimization-playbook.md)为准，不能仅凭本节文字建立优化项。

- **kernel 总耗时不是 DMA floor。** 要估计纯搬运下界，单独运行只含合同 load+store 的 sweep；
  完整 kernel 时间同时包含计算、依赖和固定开销，不能反过来当搬运下界。
- **按与参照或下界的差距排诊断优先级。** SOL 或某一 pipe ratio 无法区分并行度不足与单元素效率低；
  应把每个 case 与同一目标、工具链、shape/tiling 下可复核的 sibling reference 或 floor probe 比较。
- **说清证据类型。** sibling reference 是更简单算子的同条件实测，只用于估算量级和排序；不同算子的
  启动、融合、访存、同步和必需指令可能不同，不能据此界定任意实现或证明目标可达或不可达。floor probe 只
  界定必要数据搬运，不给语义必需计算或跨 lane 原语定价；“当前想到的杠杆”求和也只描述该清单。
  认定硬边界还须证明哪些工作不可省略以及比较方向。
- **单个 kernel 不能代表框架 roofline。** 某 kernel 的带宽、拐点或利用率只描述当时的访问
  形态；推断平台或框架限制必须有独立 probe、平台资料和其它实现的可复核证据。
- **串行依赖、历史估算和旧二进制都需排除。** 依赖链要用生成物/trace/受控 A-B 证明；结构
  改变后重新推导估计；body 修改后验证生成物身份。实验细则见
  [优化项实验闭环](optimization-playbook.md)。

更完整的 sibling/floor/证伪纪律见
[KB investigation discipline](../../pypto-pro-op-kb/references/investigation-discipline.md)。

---

## <a id="cautions"></a>注意事项

1. **必须 warm-up**：脚本默认 `--warm-up=3` 可调，避免 DVFS 影响首次运行
2. **无频率字段**：`op_summary` 没有 `Current Freq/Rated Freq`；脚本通过 `aicore_time / max_cycles` 反推主频
3. **MTE2/MTE3 带宽共享**：同时读写 GM 时总带宽共享，评估搬运段负载时宜按 MTE2、MTE3 合并字节量与平台带宽对照
4. **小数据量场景**：数据量很小时头开销占比会很高，这不一定是算子问题
5. **多核同地址访问**：多核同时读同一 512B 地址范围会被串行化，导致 MTE2 耗时异常
6. **L2 冷启（cannbench 式）**：脚本默认在每次测量前用 `scripts/l2_cache_clean.py` 跑一次大张量归约
   清空 L2，使各次测量从一致的 cache 状态起步；该刷新在独立进程、msprof 采集窗口之外执行，其 kernel
   不进入目标 `op_summary`。baseline 与 final 必须使用同一设置；与未清 L2 的历史数据对比用 `--no-l2-clean`。

---

## <a id="resources"></a>相关资源

| 文件 | 内容 |
|------|------|
| [`csv_fields_reference.md`](csv_fields_reference.md) | 字段定义和阈值（按 `op_summary_*` 列名对齐），供下游分析角色参考 |
| `../scripts/msprof_profile_run.sh` | 一键采集脚本（Step 2 调用） |
| `../scripts/msprof_perf_summary.py` | 归档 + 摘要 + 逐核负载均衡（Step 3 调用） |
