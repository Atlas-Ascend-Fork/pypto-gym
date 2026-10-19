# 性能分析总入口

> 本文件是 PyPTO-Pro 性能分析的入口：先选工具与平台模型，再进入对应采集指南；
> 逐 case 合同与证据要求以[证据协议](evidence-protocol.md)为准，字段与阈值查
> [CSV 字段参考](csv_fields_reference.md)。

本文命令中的 `PERF_SKILL_ROOT` 为本 skill 实际目录的绝对路径，执行前按安装位置设置。

## Contents

- [工具选择](#tool-selection)
- [三份指南的分工](#guide-split)
- [分析主流程](#analysis-flow)
- [共同约束](#common-constraints)

---

## <a id="tool-selection"></a>工具选择

采集前先确认平台与可用工具：

1. **平台探测**：先运行
   [`../../../pypto-pro-environment-check/scripts/get_npu_arch.py`](../../../pypto-pro-environment-check/scripts/get_npu_arch.py)，
   仅当确认 A5（`3510`/`dav-3510`/`DAV_3510`/`dav-c310`）才加载
   [A5 roofline 与杠杆](../general-knowledge/a5-roofline-and-levers.md)；未知平台不套用 A5 常量。
2. **用户指定** `msprof` 走 [msprof 指南](msprof-guide.md)；指定 `msprof op` / msopprof 走
   [msprof op 指南](msprof-op-guide.md)。
3. **未指定时探测环境**：仅 `msopprof` 可用走 msprof op 指南；仅 `msprof` 可用走 msprof 指南；
   两者皆可用时向用户确认或遵循项目约定；两者皆不可用时报错并检查 CANN / `ASCEND_HOME`。
4. PyPTO-Pro 已核实的官方入口是 `msprof python3 test_{op}.py`；`msprof op` 仅在当前 CANN 明确
   支持 Python runner 时使用，环境不支持时回到 msprof 指南，不套用 AscendC 可执行文件流程。
5. 需要跨 Skill 的采集辅助、加速比对比或交互式 HTML 报告时，按
   [ops-profiling Skill](../../../ops-profiling/SKILL.md) 与其
   [msopprof-visualization](../../../ops-profiling/msopprof-visualization/SKILL.md) 子技能执行；
   其结果仍须按[证据协议](evidence-protocol.md)归档后才能作为调优依据。

   > 注：`ops-profiling` 未收录本仓，由安装入口在启用时从上游 `cann/cannbot-skills` 拉取；
   > 上述 sibling 链接仅在安装后布局或上游仓检出中可解析。链接失效时按
   > `$CANNBOT_CONFIG_ROOT/skills/ops-profiling`（或 `$VIS_SKILL_ROOT` 指向
   > `msopprof-visualization` 子目录）定位。

按分析需求路由：

| 分析需求 | 指南 |
|---------|------|
| 指令流水分析 | [msprof op 指南](msprof-op-guide.md) + [增强指南](msprof-op-enhanced-guide.md) |
| Core Occupancy 分析 | [增强指南](msprof-op-enhanced-guide.md) |
| Roofline Bound 判定 | [增强指南](msprof-op-enhanced-guide.md) |
| 七指标采集、逐核负载均衡、compare/quick/batch（只提取，不判定） | [msprof 指南](msprof-guide.md) |
| 主 Bound 判定（由等级与行动驱动：负载不均衡/核间流水/核内流水各 Pipe）、指标等级与行动 | [指标运用指导](metric-application-guide.md) |

## <a id="guide-split"></a>三份指南的分工

| 指南 | 工具 | 产物 | 适用场景 |
|------|------|------|---------|
| [msprof 指南](msprof-guide.md) | `msprof`（应用级） | 7 组 `op_summary_<Metric>.csv` + sample-based `aicore.db` 逐核 cycle + `summary.txt` | PyPTO-Pro 标准采集；compare / quick / batch；逐核负载均衡（只做指标提取，判定走指标运用指导） |
| [msprof op 指南](msprof-op-guide.md) | `msprof op` / `msopprof` | `OPPROF_*` 8 份独立 CSV（OpBasicInfo、PipeUtilization 等）+ `summary.txt` | 上板/仿真补充诊断的采集与归档（达标判定与瓶颈定位走指标运用指导） |
| [msprof op 增强指南](msprof-op-enhanced-guide.md) | msopprof-visualization 管线（`PipeTimeline`/`Occupancy`/`Roofline` 块） | 采集目录（`collection_manifest.json` + 分块原始产物）+ `VISUAL_ANALYSIS.json/.md`（不生成 HTML） | 指令流水分析（周期判读、分段负载率、分段核内流水利用率）；Core Occupancy 核间负载不均衡率；Roofline 实测点与 bound region |

## <a id="analysis-flow"></a>分析主流程

三份指南共有的闭环，细节分别按所选指南执行：

1. **采集**：必须 warm-up 规避 DVFS；逐 case 归属冻结 case 或精确 `Op Name`；
2. **归档**：在算子目录下生成 `docs/perf/round_NNN/` + `summary.txt`；
3. **判读**：读主导流水占比，对照阈值与算子类型预期（判定规则统一在
   [指标运用指导](metric-application-guide.md)）；
4. **瓶颈定位**：结合 CSV 字段、逐核、Roofline 模型（工具判定路由见[工具选择](#tool-selection)）
   与 Tile DAG/时间线方法定位；
5. **优化与回归**：优化后重新采集归档，对比相邻 round。

## <a id="common-constraints"></a>共同约束

- 调优闭环的 compare / quick / batch 必须按[证据协议](evidence-protocol.md)传入
  `--case-manifest`；`msprof op` 归档统计不能替代正式 compare 三件套，正式 baseline/final
  走证据协议的 compare 协议。
- 每个 profiling process 只能有一个可唯一归属的 measured target launch；同名多次 launch 时
  只有 runner marker 与 occurrence/correlation/timestamp/launch index 可复核绑定才能继续，
  否则 fail closed，禁止取最大 Task Duration。
- 所有百分比阈值只是诊断 prior，用于安排下一步实验，不用于判定完成；正式结论以 SPEC、
  同协议 `Task Duration(us)` compare 和当前平台证据为准。
- ratio 是重叠占比而非时间划分：判瓶颈看流水是否饱和与隔离扫描结果，不看谁在占比里最大。
- profile 完整 `test_{op}.py` 会混入输入生成、Golden 与精度检查，其中的 `aclnn*` 不能归因给
  wrapper；性能结论只使用可唯一归属到目标 kernel 或冻结 case 的字段。

---

## 相关资源

| 文件 | 内容 |
|------|------|
| [`msprof-guide.md`](msprof-guide.md) | `msprof` 采集、归档、指标提取与分析纪律（判定走指标运用指导） |
| [`msprof-op-guide.md`](msprof-op-guide.md) | `msprof op` / msopprof 采集与归档（判定走指标运用指导） |
| [`msprof-op-enhanced-guide.md`](msprof-op-enhanced-guide.md) | Timeline / Core Occupancy / Roofline 提取与分析（复用 msopprof-visualization） |
| [`metric-application-guide.md`](metric-application-guide.md) | 指标等级分档与调优行动指导（基础/深度两阶段） |
| [`enhanced-metrics-reference.md`](enhanced-metrics-reference.md) | `VISUAL_ANALYSIS.json` 字段字典（规范约束在证据协议） |
| [`evidence-protocol.md`](evidence-protocol.md) | 逐 case manifest、Golden、归档与时间线合同 |
| [`csv_fields_reference.md`](csv_fields_reference.md) | CSV 字段定义与阈值边界 |
| [`ops-profiling Skill`](../../../ops-profiling/SKILL.md) | 跨 Skill 采集工具、加速比对比与瓶颈速查（sibling 链接仅在安装后布局/上游检出中可解析，失效时按 `$CANNBOT_CONFIG_ROOT/skills/ops-profiling` 定位） |
| [`msopprof-visualization`](../../../ops-profiling/msopprof-visualization/SKILL.md) | 上板采集数据的交互式 HTML 性能报告（同上，失效时按 `$VIS_SKILL_ROOT` 定位） |
