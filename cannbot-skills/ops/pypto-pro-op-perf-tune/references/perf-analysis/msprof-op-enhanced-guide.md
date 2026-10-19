# msprof op 增强指南：Timeline / Core Occupancy / Roofline

> 本文把 Timeline（Pipe 流水图）、Core Occupancy（核间负载）和 Roofline（瓶颈分析图）的
> 数据提取与分析能力收敛为调优可执行的用法说明。采集与解析实现**复用
> msopprof-visualization Skill**，本文件不重复实现；这里定义指标语义、提取路径、
> 分析结论写法和与本 Skill 证据协议的衔接。本 Skill 路径只采集并提取结构化
> digest，不生成 HTML 报告。

`VIS_SKILL_ROOT` 指已安装的 `ops-profiling/msopprof-visualization` 目录
（`$CANNBOT_CONFIG_ROOT/skills/ops-profiling/msopprof-visualization`）。

## Contents

- [能力总览](#capability-map)
- [采集入口](#collect-entry)
- [结构化提取（Agent 消费）](#agent-digest)
- [Timeline 数据提取与分析](#timeline)
- [Core Occupancy 数据提取与分析](#occupancy)
- [Roofline 数据提取与分析](#roofline)
- [与证据协议的衔接](#evidence-link)

---

## <a id="capability-map"></a>能力总览

| 能力 | 采集块 | `--aic-metrics` | 主要产物 | 分析产出 |
|------|--------|-----------------|---------|---------|
| Timeline（Pipe 流水图） | `timeline` | `PipeTimeline` | `trace.json` + `visualize_data.bin` + `OpBasicInfo.csv` | 各 Pipe 泳道事件与重叠证据 |
| Core Occupancy（核间负载） | `details` | `Occupancy` | `visualize_data.bin` + `OpBasicInfo.csv` | 逐核逐子核指标与负载离散度 |
| Roofline（瓶颈分析图） | `roofline` | `Roofline`（绑定 Default） | `visualize_data.bin` + 全套标准 CSV | 实测点、屋顶线、bound region 与原生 advice |

平台与互斥边界（官方约束，按当前 `msopprof --help` 复核）：

- `PipeTimeline` 与 `InstrTimeline` 不能同时使能；目前仅 Ascend 950PR/950DT 支持二者。
- Pipe 流水图基于采样实现，与开启核数无关，最多展示 6 个核。
- `Occupancy` 仅 Atlas A2/A3 与 950PR/950DT 支持。
- `Roofline` 与 Default 绑定，使能即同时采集标准 CSV 指标族。

## <a id="collect-entry"></a>采集入口

**首选：复用 msopprof-visualization 的管线**（自带环境解析、BasicInfo 预检、最小 replay
去重、语义校验和 `collection_manifest.json` 登记）。本 Skill 路径**只采集、不生成 HTML
报告**：固定使用 `--mode collect`，分析产物走[结构化提取](#agent-digest)：

```bash
python3 "$VIS_SKILL_ROOT/scripts/run_pipeline.py" \
  --mode collect \
  --operator-path "custom/{op}" --output "custom/{op}/reports/msopprof" \
  --app <可执行文件> --kernel-name '<精确 lowering 名或通配>' \
  --feature timeline --feature details --feature roofline
```

PyPTO-Pro runner 注意事项：

- `--app` 必须是可执行文件本身（`.sh`/`.py` 后缀不被接受）；Python 入口用包装脚本
  （无后缀、加执行位，内部 `exec python3 test_{op}.py --case-id <id>`）或 python3 绝对路径。
- `--kernel-name` 按实际 lowering 名匹配；PyPTO-Pro kernel 名带 `_Z10` 等 mangled 前缀时
  用通配（如 `*kda_kernel*`），避免过滤为空导致 preflight 失败。
- 没有 msopprof-visualization 时的最小等价命令：`msopprof --aic-metrics=PipeTimeline|Occupancy|Roofline
  --kernel-name=<name> --output=<dir> <app>`；此时语义校验、manifest 登记与 alias 责任转移到调用方。

## <a id="agent-digest"></a>结构化提取（Agent 消费）

本 Skill 路径不生成 HTML；分析只走提取脚本产出的结构化摘要。脚本通过 `sys.path`
复用 msopprof-visualization 的解析器（`visualize_common.load_collection`、
`parse_timeline`/`parse_occupancy`/`parse_roofline`），不重复实现 trace/bin 解析：

```bash
python3 "$PERF_SKILL_ROOT/scripts/extract_visual_analysis.py" \
  "custom/{op}/reports/msopprof" [--output-dir "custom/{op}/reports/msopprof"] \
  [--expect-iterations <源码循环次数>] \
  [--vis-skill-root "$VIS_SKILL_ROOT"]
```

输入是含 `collection_manifest.json` 的采集目录（已完成的 collection，拒绝缺失 manifest）；
解析器路径按 `--vis-skill-root` → `VIS_SKILL_ROOT` → `CANNBOT_CONFIG_ROOT` 下的
`skills/ops-profiling/msopprof-visualization` 顺序解析。产物：

- `VISUAL_ANALYSIS.json`：结构化摘要，字段字典见
  [增强指标参考](enhanced-metrics-reference.md)，规范约束见[证据协议](evidence-protocol.md)。含
  `blocks`（三块的 metric/status 溯源）、`timeline`（window、事件数、按 Pipe 聚合的
  busy/占比/最长事件、逐 lane 表）、`core_occupancy`（soc、布局 adapter、按子核类型分组的
  繁忙/空闲核排名、逐核逐子核指标）、`roofline`（逐 chart 逐 series 的 AI/perf/bw/ceiling/
  knee/ratio%/bound_region 与原生 advice）。未采集或解析失败的块写 `available=false` + 原因。
- `VISUAL_ANALYSIS.md`：同内容的人读表格摘要，可直接引用进 `PERFORMANCE_REPORT.md`。

提取口径与下三节一致；digest 是诊断证据，证据等级与衔接规则不变。

## <a id="timeline"></a>Timeline 数据提取与分析

**产物**：`03_timeline/OPPROF_*/trace.json`（Chrome Trace 格式）为主，`visualize_data.bin`
内嵌 `traceEvents` 为回退。

**提取语义**（与原 Skill `visualize_parsers.py` 一致）：

- 事件数组完整读入；`ph="X"` 为完整事件，直接取 `name/cname/ts/dur/args`；
  `ph="B"/"E"` 按 `(pid,tid)` 栈配对，`dur = E.ts − B.ts`。
- 泳道 lane = `(pid, tid)`；Pipe 流水图下 `pid` 是核与子核（如 `core0.cubecore`、
  `core0.veccore0`），`tid` 是 Pipe 单元：digest 统一使用
  `SCALAR/VEC/MTE1/MTE2/MTE3/MAC/FIXPIPE`（提取时从 trace 原生 `VECTOR/CUBE/FIXP` 改名）。
  一次采样可含多个核（官方上限 6 核），同一 `tid` 会出现多条 lane。
- 含上述 Pipe 标记即识别为 `pipe` 类型（"Pipe Timeline"）；含 `HCCL/AI CPU` 等标记识别为
  通算时间线。

**指标口径**：逐 lane 给 `busy` 与 `busy_pct_of_window`（lane busy ÷ 全局窗口）；
`pipes` 聚合给 `events`/`busy`/`longest_event` 与 `avg_busy_pct_per_lane`
（= busy ÷ window ÷ 该 Pipe 的 lane 数）。**禁止**用"pipe busy ÷ 单窗口"的占比——
多核采样时该值可按 lane 数虚增数倍，会把 latency bound 误读成 Pipe 饱和。

**周期判读（结合源码）**：`extract_visual_analysis.py` 在 timeline digest 中输出
`periodicity` 字段——在参考 lane（该核 busy 最大的计算 Pipe）上以 burst 聚类
（间隙 > 3×中位间隔切分）近似逐次循环迭代，给出 `period_us`、`iteration_count`、
`steady_window`、逐迭代各 Pipe 耗时（`per_iter_pipe_busy_us`）与稳定性 CV；
`status ∈ {periodic, bursty_irregular, insufficient_events}`。从 DSL 源码读取的循环
次数通过 `--expect-iterations N` 传入，与检测 burst 数对账（`source_check.match`）；
不一致时回查源码循环边界与参考 lane 选择，不倒填结论。周期成立不等于重叠成立，
重叠仍按证据协议 `pipeline_evidence` 规则判定。

**分段负载率（segment_load）**：把每个物理核的 cube 侧（`*.cubecore`）与 vector 侧
（`*.veccore*`）事件分别做并集覆盖，扫成 `cube_exposed` / `vector_exposed` / `overlapped` /
`seam` 四段。指标 `segment_load_rate` = **该段的累积时间占比**：

- 周期成立（`periodicity.status=periodic`）时，稳态窗内同签名段（侧别 + 时长聚类）聚成一类：
  `load_rate = per_iter_us × occurrences / 总窗`——**循环段按出现次数累积**，单次不长但
  出现次数最多的段累积占比最大，是优化重点；prologue/drain 段单列；
- 周期不成立时按整窗统计：`load_rate = 每核该类段总时长的跨核均值 / 总窗`（多核不累加，
  避免 >1 虚增）。

输出两个视图：`segments`（签名/类型级的累积占比，找优化重点）与 `top_exposed_instances`
（具体段区间排名，含 core/start/dur，定位窗口）。`largest_exposed_uncovered` 指向**单个最长
暴露段**；`largest_exposed_cluster` 指向累积占比最大的段类。

**分段核内流水利用率（pipe_util_pct）**：每个段行附带段窗口内各 Pipe 的利用率
（`busy ÷ 段时长`，tid 名为 `MAC/VEC/FIXPIPE/MTE1/MTE2/MTE3/SCALAR`，由 trace 原生
`CUBE/VECTOR/FIXP` 改名）。多子核侧
（每核 2 个 veccore）按 lane 求均值，不超过 100%；`overlapped` 段按侧前缀区分
（`cube:MTE2` / `vector:VEC`）。聚合口径：稳态簇取中位数，整窗行按段时长加权、
跨核求均值。暴露段内主 Pipe 利用率低（如 MAC 仅 ~50%）说明段内等待依赖
（MTE1 供数、FIXPIPE 写回、同步），指导在段内消除等待；主 Pipe 接近 100% 则该段
只能靠缩短工作量或让对侧覆盖来优化。

trace **只能证未被覆盖**（对侧 busy=0）；"无法被覆盖"需 Tile DAG/slot 依赖判定，
字段 `uncoverable=requires_dag_evidence`，按证据协议 `pipeline_evidence` 规则收口。
暴露段是优化对象：缩短它，或制造可重叠边让对侧覆盖。

**分析用途**：在唯一 target 时间窗内检查相邻 Pipe 事件的同 lane 重叠，作为搬算重叠证据；
结论必须再结合 lowering/源码 Tile DAG 与 slot/stage 映射，按[证据协议](evidence-protocol.md)
的 `--timeline` 语义写 `pipeline_evidence.status`（`proven` / `residual_not_overlapped` /
`unverified` / `not_applicable_with_dag`）。采样只覆盖最多 6 核，不能外推全部核。

## <a id="occupancy"></a>Core Occupancy 数据提取与分析

**产物**：`01_details/OPPROF_*/visualize_data.bin`（无 trace.json）。

**提取语义**：bin 为二进制容器，扫描字节流提取内嵌 JSON 记录；取记录的
`op_detail[]`（并记录 `soc`），逐核展开 `core_detail[]` 子核条目为
`(core_id, subcore_id, subcore_type, metrics)`；`_id` 结尾字段不参与指标。核布局按 SoC
schema 适配：950PR 走 `native`（8 物理核 × 3 子核 = 24 lane）；packed 布局需先按
`lanes_per_core`/`slot_order` 重组逻辑核。

**分析产出**（服务负载均衡优化，目标是找出繁忙核与空闲核）：

- **按子核类型分组**：cube 与 vector lane 分开统计，不得混在一起比（每物理核 = 1 cube +
  2 vector，不同单元的工作不可比）；指标聚合也按组给（`metrics_by_type`）。
- **组内按 busy 指标（默认 cycles）归一化利用率**：`utilization = cycles / max(cycles)`；
  利用率比最忙核低 30% 以上的 lane 为**低利用率核**。组指标为**核间负载不均衡率**
  （`imbalance_ratio_pct` = 低利用率核占比）：超过 10% 判 `imbalanced`，否则 `balanced`。
  同时给 min/median/max、极差 `diff_pct_of_max` 与 top-3 繁忙/空闲 lane。
  低利用核揭示可搬迁的余量，偏斜指向分核/尾块策略。
- **物理核总排名**：把同物理核的 cube+vector cycles 求和（`busy_sum`），排出整机最忙/最闲
  物理核。注意 cube/vector 主频可能不同，跨子核 cycles 求和是工作量代理而非时间口径，
  排名仅作定性参考。

官方 Insight 侧另有 z-score+sigmoid 归一化（0.6 阈值）的展示层判定；本文使用的是从产物
提取的原始数值，二者口径不同，引用时注明来源。

**分母必须是可用核数，不能只算已启动的 block**：核间负载不均衡率衡量的是在**可用核**
范围内的利用率，空闲的可用核按 0 利用率计入低利用率核。可用核数默认等于物理核数，但也
可能被规格/运行限制指定而小于物理核数——收口前先确认本次的可用核数，不要默认写成物理核数。
一个 launch 若只铺了 `grid < 可用核数` 的 block，"已启动核之间均衡"不等于"用满可用核"。
典型取舍：把 grid 从 28 核缩到 16 核可消除双链折返、让关键路径从 192 µs 降到 118 µs，但
12 个空闲可用核使不均衡率从 85.7% 只降到 ≈42.9%，仍属"高，需要调优"。注意 `msprof op` 的
`PipeUtilization.csv` 只对已启动的 block 落行，**用它算不均衡率会漏掉空闲核**，不能据此宣称
0%；要与可用核数对齐后再收口（浅层 CSV 只作辅助，正式值以 Occupancy 产物的全部可用核为准）。

## <a id="roofline"></a>Roofline 数据提取与分析

**产物**：`02_roofline/OPPROF_*/visualize_data.bin` + 标准 CSV 族。

**提取语义**：取内嵌记录的 `multiple_rooflines[]`；每组 `title`（如 `GM/L2`、
`Memory Unit(Cube)`、`Pipeline(Cube)`）+ `rooflines[]` 行。每行原始字段：

- `point = [x, y]`：x = 算术强度（Ops/Byte），y = 实测性能（TOPS/s）；
- `bw`：该通路理论带宽；`computility`：算力上限；`ratio`：性能百分比。

**派生与判定**：

- 拐点 `knee_intensity = computility / bw`；屋顶线 `roofline_limit = min(bw·x, computility)`。
- `ratio` 归一化：原生值可能是分数（0.14）或百分数（14.0），按 >1.5 且 ≤100 则 ÷100；
  缺失时用 `y / roofline_limit` 推。
- bound region（±2% 容差）：`x < knee·(1−2%)` → bandwidth-bound；`x > knee·(1+2%)` →
  compute-bound；其间 → ridge region。
- 原生 `advice`（如 `latency bound: pipeline caused`）直接搬运，不自行改写；无
  `multiple_rooflines` 结构时只保留实测点，**不伪造带宽与算力**。

**分析衔接**：bound region 与 ratio 共同写入[证据协议](evidence-protocol.md)的
`roofline_terminal.status`（`compute_bound` / `data_movement_bound` /
`balanced_compute_movement` 等）；注意几何区域（AI 相对拐点）与饱和程度（ratio 距屋顶
距离）是两件事——点落在 compute region 但 ratio 很低时，瓶颈在流水空隙而非算力/带宽，
应与 Timeline 重叠证据互相印证。

## <a id="evidence-link"></a>与证据协议的衔接

- 产物登记、证据等级与作为第 6 步机制核对输入的字段映射，以
  [证据协议](evidence-protocol.md)的"msopprof 诊断产物"节为准。

- 三项能力均为**诊断证据**：可用于瓶颈定位、候选排序与终态证据（`pipeline_evidence`、
  `roofline_terminal`、逐核负载），但**不能替代正式 compare 三件套**；baseline/final
  仍按[证据协议](evidence-protocol.md)的 manifest 协议逐 case 采集。
- 结论只使用可唯一归属目标 kernel 的产物；`--kernel-name` 过滤与单 launch 约束同
  证据协议的 fail-closed 规则。
- 采集目录不带 group/other 写权限；失败（超时/非零/语义空）的 replay 不得复用。
- `VISUAL_ANALYSIS.json/.md` 路径与 `collection_manifest.json` 写入
  `PERFORMANCE_REPORT.md` 的证据路径，引用轮次不得被后续改动淘汰。

## 相关资源

| 文件 | 内容 |
|------|------|
| [msprof op 指南](msprof-op-guide.md) | msopprof 采集闭环、采集与归档（达标判定与瓶颈定位走指标运用指导） |
| [证据协议](evidence-protocol.md) | manifest、compare、终态证据与归档合同 |
| [指标运用指导](metric-application-guide.md) | 指标等级分档与调优行动（基础/深度阶段） |
| [增强指标参考](enhanced-metrics-reference.md) | digest JSON 字段字典（规范约束在证据协议） |
| `../../scripts/extract_visual_analysis.py` | 采集目录 → `VISUAL_ANALYSIS.json/.md` 结构化摘要（Agent 消费） |
| `$VIS_SKILL_ROOT/references/collection-workflow.md` | 采集块映射、replay 去重规则 |
| `$VIS_SKILL_ROOT/references/visualization-contract.md` | 各页面数据合同 |
