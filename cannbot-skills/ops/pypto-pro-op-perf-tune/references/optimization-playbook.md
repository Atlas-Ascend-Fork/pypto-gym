# 优化项实验闭环

本页定义五类优化项如何登记、实验、重开和关闭。采集命令与归档格式见
[证据协议](evidence-protocol.md)，指标解释见[msprof 指南](msprof-guide.md)与
[CSV 字段参考](csv_fields_reference.md)。平台资料、general knowledge 和本页本身不产生优化项；
模板项只由[模板优化项索引](../templates/INDEX.md)枚举。

## 1. 冻结实验合同

第一次正式采集前，在 `PERFORMANCE_REPORT.md` 记录并冻结：

- 当前正确实现及其可恢复副本或 diff；
- device、型号、软件版本、模块路径、竞争进程和影响性能的环境变量；
- `PERFORMANCE_CASES.json`（含完整 P0 与冻结目标 case；本项须在任何调优采集或改码前确定）、输入分布、seed、warm-up、repeats、精确 `Op Name` 和计时范围；
- 完整正确性标准、quick→formal 晋级规则、稳定性判据、聚合指标及并列处理规则；
- baseline/final 的 formal 协议；quick 只用于候选筛选。

“正确候选”必须通过完整正确性；“合规候选”还必须满足冻结 SPEC、全部 selected-KB 义务、
Module/public wrapper 合同、[主 Skill](../SKILL.md)的实现约束和资源硬限制。输入实现存在已登记的 KB 缺口时仍可作为
修复起点和性能 baseline，但补齐全部义务前不能进入最终候选排名。用户性能目标、Golden 理想参考
及 Roofline/Scalar/流水的理想状态用于发现候选和披露差距，不属于合规门禁。

设备告警或复位、同卡竞争、超时、profiler 失败、目标归属不唯一、CSV 不完整或没有对应正确性
结果的样本均无效。源码修改后应确认采集使用了新生成物；发现 stale binary 时先清理证据问题，
不得把旧数据归给新实现。

## 2. 建立来源账本

第一次改代码前，枚举五类预置来源：

1. 项目 selected KB 中全部 `kind=obligation` 的原子要求；无已选 KB 时记录 0 项；
2. [通用优化手段](general-optimization-methods.md)中的全部 active/eligible item；
3. [知识卡片索引](knowledge-cards/index.md)的 Active 表中的全部 item；
4. [模板优化项索引](../templates/INDEX.md)中的全部 active/eligible item；
5. [结构性优化重审与可达目标取证](structural-optimization.md)的四类结构维度原子项；命中其 §1 触发
   条件时无条件纳入。

有 selected KB 时先按 Design Skill 的 source-first 合同核对 `DESIGN_BINDINGS.json`，再以原子 requirement
为账本项；其稳定身份使用完整原子键 `(class_id, selection_field, reference, req_id)`，不得只使用
组内唯一的 `req_id`。precondition 和 validation scope 只作为适用性与验证依据。进入自主优化后，
每个新假设在改代码前追加为第六类 `bottleneck_derived`。

每项至少记录：

| 字段 | 含义 |
|---|---|
| `item_id` / `source_kind` | 稳定 ID，以及六类来源之一 |
| `source_ref` | 权威文件、原子锚点和内容身份；自主项记录对应瓶颈证据 |
| `case_scope` | 涉及的 P0 case |
| `applicability` | 适用、不适用、未触发或待证及其依据 |
| `expected_metric` | 预计改变的工作量、搬运、依赖、冲突或固定开销 |
| `status` | 当前状态与关闭结论 |
| `evidence` | 代码位置、正确性、quick/formal 数据、机制核对和恢复点 |
| `relations` | overlap、require、enable、conflict、覆盖或重开关系 |

非终态为 `pending`、`unknown`、`enabler`、`blocked`。selected-KB 终态为
`already_implemented`、`not_triggered`、`implemented`；其余来源终态为 `accepted`、`rejected`、
`not_applicable`、`unsupported`。`rejected` 必须记录具体原因，代码恢复或切换到第 6 节规则选出的
当前最佳实现。

同一改动可以为多个来源提供实验，但每个 source id 都要有独立结论。某非 KB 项被其它项完整
覆盖时，保留该行并记录覆盖关系；只重合一部分时仍实验独特增量。实现或前提变化使旧证据失效时，
重开原 ID。账本状态描述调优实验，不得直接复制成 `KB_USAGE.json` 状态。

## 3. Baseline、瓶颈与排序

先对未修改的输入实现运行完整正确性和 JIT 预热，再按 manifest 建立逐 P0 formal
baseline。对每个 P0 记录绝对 Task Duration、AIC/AIV 时间、Vector/Cube/Scalar/MTE、分层带宽、
冲突、逐核信息，以及可获得的生成物和 Tile DAG。

对每个待处理项说明：适用位置或排除依据、预期机制、涉及 case、验证指标，以及正确性、精度、
容量、tail 和同步风险。ratio 只负责提示方向，不能单独证明硬件饱和、Roofline 或搬算重叠。
优先级只改变实验顺序，不改变必须覆盖的项目；接受结构变化后，刷新受影响 case 的瓶颈证据并
重新检查相关旧结论。

## 4. 逐项实验

按主 Skill 规定的来源顺序处理每个可能适用项：

1. `kb_selected` 缺口从当前正确版本开始修复；其它项从当前最佳正确、合规版本建立可恢复起点；
2. 记录 item id、唯一主要假设、目标指标、风险和恢复点；
3. 修改源码并运行完整正确性；失败立即恢复并记录；
4. 正确后运行同条件 quick，保留原始样本；
5. 正确落实的 `kb_selected` 缺口直接进入 formal 记录性能影响；其它候选按冻结晋级规则决定是否
   formal，未晋级则恢复当前最佳并以 `status=rejected, reason_code=quick_not_promoted` 关闭，
   有明确组合价值时暂记 `enabler`；
6. 核对预期机制；性能变化与假设不符时重新诊断，不倒填结论；
7. `kb_selected` 验证通过后保留修复并记为 `implemented`；其它项依据第 6 节决定保留或恢复，
   并关闭该 source item。

入口实现已经包含某项时，先记录当前机制证据；在合法、可恢复时用去除或替代该动作的版本作
control。两版按第 6 节比较：保留含动作版本则记 `accepted`；选择 control 则记 `rejected`，并按结果
记录 `control_faster` 或 `no_stable_gain`。control 若违反 selected-KB 义务或已登记依赖关系则不适用。

selected-KB 的真实缺口必须落实并通过该 requirement 的验证方法，不能因性能无收益而拒绝；若其与
冻结合同或当前能力确实无法同时满足，报告 `tuning_contract_blocked`。card 与 template 只按各自
index/INDEX 枚举：卡片取 Active 表，模板取 active/eligible 项；未登记文件不进入账本。知识卡片由
技术 review 把关，采用时按本节完成实验。模板只是待适配骨架，不能替代 API、正确性或性能证明。

全部 selected-KB 缺口闭合后，对当前完整修复版本重新执行 formal compare；其通过正确性与合规
检查后，作为首个可排名版本进入第 6 节。此前的 KB 中间版本只记录修复影响，不参与最终排名。

只组合不存在未消解 `conflict`，并有明确 `requires`/`enables`、共享资源或机制互补依据的项。
`enabler` 只是临时状态，指定组合接受或被证伪后必须关闭。一个 item 内可以有限搜索直接相关参数，
但不能产生没有来源 ID 的匿名实验。

## 5. 自主瓶颈优化

五类预置项全部达到终态、没有未决 enabler 后，保存当前最佳实现和一份新鲜的逐 P0 瓶颈分析，
再进入 `bottleneck_derived`。预置项未闭合或当前证据不可评估时，只补齐前置工作，不能提前建自主项。

每个自主项必须：

- 来自当前接受实现尚未尝试的具体瓶颈，并绑定相应 case、源码/生成物/DAG 和 profiler 证据；
- 与 selected KB、通用方法、active card、active template 和既有自主项去重；
- 写明可证伪机制、最小改动或有限参数域、预期指标、风险和停止条件；
- 登记后再按第 4 节实验；接受改动后刷新瓶颈证据，再提出下一项。

自主改动改变旧项的 bound、DAG、layout、容量、同步或适用前提时，重开受影响的原 ID；预置项
重开后先重新闭合，再继续自主项。目标差距以及 Scalar、等待或未重叠流水等残留状态应优先产生
候选，但在相关合法候选全部关闭后，不单独阻止交付。

## 6. 选择最佳版本

目标 case 在任何调优采集或改码前冻结；候选聚合指标在第一次正式采集前冻结。
`all_p0_no_questions` 固定对全部 P0 使用下述默认算法；其它模式若 `SPEC` 已针对冻结目标 case 记录
用户给出的可复算指标或权重，则原样使用。没有上述专用规则时，对 `optimization_target.case_ids` 逐项计算
`case_speedup = frozen_baseline_duration / candidate_duration`，取等权算术平均。非目标 P0 不进入
默认聚合，但仍须完成正确性、正式测量并逐 case 披露。

最终选择遵循一条规则：**在所有完成 formal compare 的正确、合规候选中，选择冻结聚合指标最优
的版本。** quick 只决定是否晋级，不能进入最终排名；逐 case 回退、用户目标和 Golden 理想参考
状态必须披露，但不直接否决候选。

`PERFORMANCE_REPORT.md` 保留一张可复算的候选表，覆盖仍然正确、合规的冻结 baseline 及所有按
冻结规则完成 formal compare 的正确、合规版本；至少记录候选身份、不可变的 `evaluation_order`、
关联 source item、正确性与合规证据、formal collection、逐 case duration/speedup 和聚合分数。
baseline 的顺序为 0；其它版本在第一次取得有效 formal 结果时依次编号。先按聚合分数比较；差异
落入冻结稳定性范围时，先以原始最高分为锚选出与其不可稳定区分的候选，再从中选择顺序最小者，
避免多候选逐对比较产生歧义。晋级规则、稳定性范围和并列规则不得在看到结果后修改。

选出最佳版本后，对它独立执行 final formal compare。确认结果与排名数据差异超出冻结稳定性范围
时，按同一协议重新测量受影响候选并重新选择；不得只挑有利轮次替换结果。

## 7. 完成与报告

结束前基于最终代码、最新 profiler、生成物或 Tile DAG 和完整账本再做一次候选扫描；判断流水或
Scalar 残留需要时间线时先补采。新证据使预置项结论失效时重开原 ID；其它新的合法优化项才登记为
`bottleneck_derived`。没有新项时记录扫描范围、发现及其与已关闭项、冻结合同或当前能力的对应关系。
最终验收 timeline 暴露新项时按同一规则重开闭环；代码或最佳候选变化后重新执行最终验收。

调优完成必须同时满足：

- 五类预置来源和全部已创建的自主项均已合法关闭，没有未决或需要重开的项；
- final sweep 没有新的合法候选；
- 最终代码是第 6 节规则选出的最佳正确、合规版本；
- 用户目标或 Golden 理想参考状态及残留瓶颈均如实披露；
- 最终代码重新通过完整正确性、独立 formal compare 和每个 P0 的补充 timeline；
- 报告、原始证据和最终事实记录相互一致且可复算。

`PERFORMANCE_REPORT.md` 应包含冻结合同、来源账本、baseline 与瓶颈、逐项实验、候选比较、最终
扫描、最佳版本、目标状态、残留问题和证据路径。目标未达或默认 Golden 不存在不阻止完成；证据
缺失或矛盾、仍有未关闭项、环境阻断、用户中止或冻结合同冲突时不能声称闭环完成。
