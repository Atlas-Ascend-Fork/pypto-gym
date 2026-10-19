# 增强指标参考

> 本文件是 `VISUAL_ANALYSIS.json` 的**字段字典**（参考性质）：列出各字段的含义与取值口径。
> 规范约束（schema 身份、证据等级、使用通道、变更要求）统一由
> [证据协议](evidence-protocol.md) 的"msopprof 诊断产物"节承载，本文件不作规定。
> 产出方：`pypto-pro-op-perf-tune/scripts/extract_visual_analysis.py`；字段语义的上游来源是
> msopprof-visualization 的解析器（`visualize_common.load_collection`、
> `visualize_parsers.parse_timeline/parse_occupancy/parse_roofline`），本文件不改变上游语义。

## 1. 顶层字段

| 字段 | 类型 | 语义 |
|------|------|------|
| `schema` | string | 取值见证据协议 |
| `collection` | string | 采集目录绝对路径（含 `collection_manifest.json`） |
| `parsers_source` | string | 实际复用的解析器目录（msopprof-visualization `scripts/`） |
| `extracted_at` | string | ISO-8601 UTC 提取时间 |
| `blocks` | object | 三块采集溯源：`timeline`/`details`/`roofline` → `{metric, status}`（manifest 原文） |
| `timeline` | object | 见 §2 |
| `core_occupancy` | object | 见 §3 |
| `roofline` | object | 见 §4 |

## 2. `timeline`

不可用：`{"available": false, "reason": <string>}`；reason 区分"未采集 / manifest status≠ok /
解析失败"。可用时：

| 字段 | 语义 |
|------|------|
| `timeline_type` | `pipe` / `communication` / `generic`（按 lane 与事件名标记识别） |
| `window` | `{start, end, duration}`；单位沿用 trace 导出（当前 950PR 为 µs，见 `notes`） |
| `event_count` / `embedded_event_count` | 全量事件数 / 实际嵌入数（受 `--max-events` 截断） |
| `pipes` | 按 Pipe 名聚合：`{events, busy, longest_event, avg_busy_pct_per_lane}`；占比为 busy ÷ window ÷ 该 Pipe lane 数，多核采样不虚增 |
| `lanes` | 逐泳道 `{pid, tid, events, busy, busy_pct_of_window}`；`tid` 为 Pipe 单元名（`MAC/VEC/FIXPIPE/MTE1/MTE2/MTE3/SCALAR`，由 trace 原生 `CUBE/VECTOR/FIXP` 改名），占比按本 lane 计 |
| `periodicity` | 周期判读：`{status(periodic/bursty_irregular/insufficient_events), reference_lane, iteration_count, period_us, period_cv, burst_busy_cv, steady_window, per_iter_pipe_busy_us, source_check{expected_iterations, match}}`；burst 聚类近似迭代，`--expect-iterations` 传源码循环次数对账 |
| `segment_load` | 分段负载率：`{available, cadence, total_window_us, segments[], top_exposed_instances[], largest_exposed_uncovered, largest_exposed_cluster}`。段类型 `cube_exposed/vector_exposed/overlapped/seam`；`load_rate` 为**累积占比**（循环段 = per_iter_us × occurrences ÷ 总窗；非周期 = 跨核均值 ÷ 总窗）；每段行附 `pipe_util_pct`（**分段核内流水利用率**：段内各 Pipe busy ÷ 段时长，tid 为 `MAC/VEC/FIXPIPE/…`，多子核按 lane 均值 ≤100%，overlapped 段键带 `cube:`/`vector:` 前缀）；`largest_exposed_uncovered` 为单个最长暴露段（含 core/start/dur）；`uncoverable=requires_dag_evidence` 表示 trace 只证未覆盖 |
| `notes` | 采样边界等说明（如最多 6 核） |

## 3. `core_occupancy`

不可用同上。可用时：

| 字段 | 语义 |
|------|------|
| `soc` / `adapter` | 平台与核布局适配器（如 `Ascend950PR_9579` / `native`） |
| `raw_core_count` / `logical_core_count` / `lane_count` | 物理核、逻辑核、lane 数 |
| `balance_by_type` | 按子核类型（`cube`/`vector`…）分组：`{busy_metric, lane_count, min, median, max, diff_pct_of_max, low_utilization{definition,count,lanes[]}, imbalance_ratio_pct, verdict, verdict_rule, busiest[3], idlemost[3]}`。利用率归一到组内最忙核；比最忙核低 >30% 为低利用率核；**核间负载不均衡率** `imbalance_ratio_pct`（低利用率核占比）>10% 判 `imbalanced`。`_physical_core` 键为物理核总排名（`busy_sum` = 同核各子核求和，附 `caveat`：跨子核 cycles 求和是工作量代理而非时间口径） |
| `metrics_by_type` | 按子核类型分组的指标聚合 `{key, min, median, max}`；不做跨类型混合聚合 |
| `lanes` | 全量逐 lane `{core_id, subcore_id, subcore_type, metrics}` |

## 4. `roofline`

不可用同上。可用时：

| 字段 | 语义 |
|------|------|
| `plot_available` | 是否存在可绘制图表 |
| `advice` | profiler 原生结论文本（如 `latency bound: pipeline caused`），直接搬运不改写 |
| `charts[]` | `{title, series[]}`；series 字段：`bandwidth_name`、`compute_name`、`arithmetic_intensity_ops_per_byte`、`measured_performance_tops`、`bandwidth`、`compute_ceiling_tops`、`knee_intensity`、`ratio_percent`、`bound_region`（`bandwidth-bound region` / `compute-bound region` / `ridge region` / `unknown`） |
| `notes` | 读法提示（几何区域 ≠ 饱和程度） |

单位约定：`point`/`knee` 为 Ops/Byte 与 TOPS/s（沿用 profiler 坐标口径）；`bandwidth` 单位
随 profiler 产物原文，本文件不做归一化。

## 5. 证据等级

digest 是**诊断证据**，证据等级与使用通道以[证据协议](evidence-protocol.md)为准。
