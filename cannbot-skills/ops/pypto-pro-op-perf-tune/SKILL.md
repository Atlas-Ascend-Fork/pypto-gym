---
name: pypto-pro-op-perf-tune
description: 优化并验收已通过正确性验证的 PyPTO-Pro 单 kernel 实现；冻结规格与测量协议，依次验证已选 KB、通用方法、知识卡片、模板和当前瓶颈提出的优化项，保留最佳合规实现及证据。也用于独立性能采集、比较和瓶颈分析。
---

# PyPTO-Pro 性能优化

本 Skill 定义性能调优的优化来源、执行顺序和交付边界。具体账本、实验、重开、最佳版本选择
与停止规则统一见[优化项实验闭环](references/optimization-playbook.md)。性能采集和分析按需读取：

- 标准、compare、quick、batch 采集与主 bound 判定：[msprof 指南](references/msprof-guide.md)；
- 环境支持时的 `msprof op` 补充诊断：[msprof op 指南](references/msprof-op-guide.md)；
- CSV 字段与阈值边界：[CSV 字段参考](references/csv_fields_reference.md)；
- 逐 case manifest、Golden、归档和时间线合同：[证据协议](references/evidence-protocol.md)。

用户只要求采集、比较或诊断时，进入对应参考页；需要改码调优时执行下述闭环。
命令中的 `PERF_SKILL_ROOT` 指本 Skill 的实际目录；算子目录示例可替换为项目实际路径。

本 Skill 是 `$CANNBOT_CONFIG_ROOT/references/kernel-core-principles.md` 在 A5 上的具体化：
通则给方向，本 Skill 的实测给数值。同一数值冲突时以实测为准，并按 `M5` 留痕。
调优顺序遵循 `kernel-perf-pypto-pro-a5.md` §2 的排障顺序（`G1` → `G6`），且只优化 profile 确认的绑定资源（`B4`）。

## 1. 目标与边界

保持外部功能合同和下列实现约束不变，找全、落实并关闭全部优化项；冻结 baseline
与本轮正式评价的候选只在正确、合规时纳入排名，留下冻结聚合指标最优的版本与可复算性能证据。用户数值目标
或默认 Golden 参考是理想目标，用于指引优化和披露差距，不是调优能否交付的硬门禁。

- 一个交付文件只含一个 `@pl.jit` kernel，核心计算全部在该 kernel 内；wrapper 只启动一次
  kernel，且启动不在 host 循环内。wrapper 仅做参数检查、合同允许的只读元数据读取、纯 Python
  整数推导及声明输出的 `torch.empty` 分配；完整 host 调用链均须满足该边界。
- 轮转 tile 使用 `make_tile_group` + `auto_mutex`，不得叠加手动 `sync_src`/`sync_dst`；`make_tile`
  仅用于单次 scratch。`auto_mutex` 只管理核内互斥，跨核依赖须单独同步。Vector 数值计算默认使用 VF；
  已选 KB 明确要求当前步骤使用 tile-op 时遵从原要求。
- 代码、知识落实、设计一致性和证据问题在调优闭环中修复并重新验收。环境阻断、用户中止或冻结
  输入之间存在无法合法消解的矛盾时，报告根因、证据与待解决项。
- 只修改可维护的 Python DSL 和必要的最终事实记录。`build/**`、`kernel.cpp`、trace 与指令生成物
  只读，只能作为证据。

## 2. 开始前读取与冻结

先读取当前源码、可运行的完整正确性测试、数学参考（Golden）和算子规格。规格须明确数学语义、
I/O、支持范围、精度、输入分布、性能 case 与用户目标；下文以 SPEC 统称，以 P0 指必须覆盖的
性能 case。独立使用时将用户提供并确认的规格记录到 `PERFORMANCE_REPORT.md`；已有 `SPEC.md`
则直接沿用。缺少影响正确性或支持范围的事实时先向用户补齐，再冻结和调优。

项目已建立或明确要求以下资料时，读取并交叉核对：

- `EXPLORE_REPORT.md`、`PRO_MATERIAL_INDEX.md`、`MEMORY.md` 及项目性能约束；
- `DESIGN.md`、`DESIGN_BINDINGS.json`、`module_interfaces.yaml`；
- 每个 class 的 `KB_SELECTION.json`、其中选中的每个 pattern/constraint 原文及对应
  `KB_USAGE.json`。

优化来源读取：

- [通用优化手段](references/general-optimization-methods.md)；
- [知识卡片索引](references/knowledge-cards/index.md)的 Active 表及其中全部卡片正文；
- [模板优化项索引](templates/INDEX.md)及其中全部 active/eligible 模板。

只有本节明确允许更新的源码、性能交付物和项目事实记录可补齐或纠正；其它必需输入只读。
项目要求的只读或冻结输入缺失、损坏、无法解析且无可恢复原始字节时，不得当作可选资料跳过，
不猜测、不重建，报告 `tuning_contract_blocked`。

始终冻结：

- SPEC 数学语义、I/O、支持范围、P0、输入分布、精度门与用户目标；
- Golden 数学实现和完整正确性测试语义；
- 项目已有 `KB_SELECTION.json` 的选择、引用、理由和内容哈希；
- `module_interfaces.yaml`、staged Module、公开 wrapper 的 callable/签名/I/O/optional 语义与 host
  边界合同（项目有相应产物时读取原文），以及 §1 的实现约束；wrapper body 只能在这些合同内调整；
- `PERFORMANCE_CASES.json` 及其中的目标 case，在任何调优采集或改码前冻结；
- 设备、seed、warm-up、repeats、精确 `Op Name`、计时范围和 L2 清刷设置（`--l2-clean`，默认开启）
  在首次正式采集前冻结；baseline 与 Golden 合同生成后不再修改。

调优可以更新最终 `test_{op}.py`、性能交付物，以及项目中为匹配最终接受实现所必需的
`DESIGN.md`、`DESIGN_BINDINGS.json` 和 `KB_USAGE.json`。事实记录不得删除、降级或
改写已选 KB 义务，也不得用文档修改掩盖代码未落实；未接受的实验不能写成最终事实。更新前按需
读取 [Design Skill](../pypto-pro-op-design/SKILL.md)、[Develop Skill](../pypto-pro-op-develop/SKILL.md)
及 [KB 合同](../pypto-pro-op-kb/CONTRACT.md)，只按各自产物 owner 定义的 schema 同步最终事实。

## 3. 优化项来源与顺序

优化项只有以下五类，按顺序处理。其中 `knowledge_card` 是核心来源，其余三类是补充：
`kb_selected` 补齐开发阶段已选 KB 模板、约束与样例中尚未在代码里落实的部分，`general_method` 与
`template` 补充卡片之外的机会；三者都不能代替 `knowledge_card` 的覆盖。
第一次修改代码前，按[优化项实验闭环](references/optimization-playbook.md)建立来源覆盖账本。

### 3.1 `kb_selected`：已选 KB 的未落实要点

项目没有已选 KB 时记录本来源为 0 项；已有选择时，逐 class、逐行读取
`KB_SELECTION.json.optional_patterns` 与 `required_constraints` 指向的原文，
先按 Design Skill 的 source-first 合同核对并修正 `DESIGN_BINDINGS.json`。通过 owner schema 自检后
形成本轮原子 requirement 清单；`kb_selected` 分母只包含其中 `kind=obligation` 的原子项，precondition 与
validation scope 作为适用性和验证证据保留。再把每条义务与最终代码、DESIGN/BINDINGS 和 KB_USAGE
核对：已落实、未触发和真实缺口都进入账本；前提触发但代码未落实的缺口必须最终真实落实，不能
因无性能收益而拒绝。

### 3.2 `general_method`：通用优化手段

枚举[通用优化手段](references/general-optimization-methods.md)索引中的全部 active/eligible 原子项。
明确不适用或目标/API 不支持时用证据关闭；其它可能适用项逐个进入受控实验。通用项仍要与本次
selected KB 去重，同一改动可以共享实验，但每个 source id 单独关闭。

### 3.3 `knowledge_card`：知识卡片

这是**核心来源**：按[知识卡片索引](references/knowledge-cards/index.md)的 Active 表逐卡评定，不扫描目录。

1. **逐卡通读**：读取卡片正文全文，不只依据索引摘要；明确其改写内容、加速机理与成立条件。
2. **逐卡适用性判定**：以当前 kernel 的源码、生成物与 profiler 数据对照卡片的适用条件，逐卡给出结论并写明依据，记入账本的 `applicability`。
3. **满足条件即执行实验**：适用条件成立且所需能力已确认可用时，按实验闭环完成改码、正确性验证、性能测量与机制核对；预期收益有限或 profiler 未直接呈现该现象，均不构成免于实验的理由。
4. **不执行须给出依据**：仅三种情形可跳过——机制在当前 kernel 上不成立；所需能力经证实不支持（证据不足则先补证）；同一改动已由其它 source id 完成实验，此时在 `relations` 记录关联，本条仍独立关闭。
5. **结论与预期不符时复核查证**：按现有闭环记录接受或恢复，不得倒填结论。

贡献或整理新卡片时，从[贡献指南](references/knowledge-cards/CONTRIBUTING.md)开始：Agent 按贡献者
提供的事实整理卡片，默认以 Active 提交卡片和索引变更，保留未知项与技术限制。

### 3.4 `template`：模板优化项

只按[模板优化项索引](templates/INDEX.md)枚举 active/eligible 原子项，不扫描目录或把未登记的
`.tmpl` 文件加入清单。INDEX 定义优化意图、适用门槛和稳定 ID，模板文件只提供实现骨架；退役、
负向或不 eligible 的模板不进入闭合分母。可能适用项逐个实验，其它项用明确的适用性或能力证据
关闭。采用模板前必须按当前代码适配，并重新证明正确性、机制和性能。

### 3.5 `bottleneck_derived`：当前瓶颈自主项

前四类预置来源全部闭合后，才可根据当前接受实现的 profiler 与源码、生成物或 Tile DAG 证据
提出自主项。每项必须指向尚未尝试的具体瓶颈、使用与当前代码哈希匹配的新鲜分析、与前四类和
既有自主项去重，并按同一受控闭环验证。接受改动后必须在新代码上刷新瓶颈证据，再建立下一项。

profiler、代码分析、平台资料和 general knowledge 可以共同解释新鲜瓶颈并形成假设；平台资料和
general knowledge 本身不是优化项来源，假设必须先按前述证据去重并登记为 `bottleneck_derived`，
再修改代码。自主改动使预置项结论失效时，先重开并闭合预置项。

## 4. 调优主流程

1. **确认输入与正确性**：完成上述读取，对未修改的输入实现重跑完整正确性并预热
   JIT/编译。
2. **冻结执行合同**：开始任何调优采集或改码前，从 SPEC P0 与既有测试生成
   `PERFORMANCE_CASES.json`：`cases` 覆盖全部 P0，`optimization_target` 记录本轮排名使用的非空
   目标 case 子集及选择方式。有调用方输入时核对并沿用；无效则返回调用方补齐。独立请求按用户
   指定目标选择，唯一 P0 自动选定；多个 P0 未指定时先确认，用户要求不再提问则选择全部 P0。
   无用户数值目标时，按证据协议采集并冻结一次 Golden 理想参考。未生成性能
   Golden 合同时记录 `unavailable`，不能据此否定有效的 PyPTO baseline→candidate 选择证据；已有
   合同矛盾或不可复算时必须修复证据。
3. **建立来源覆盖账本**：先按 §3.1 核对并形成 selected-KB 原子 requirement 清单，再枚举其中全部
   obligation、全部通用方法、Active 知识卡片和 active/eligible 模板优化项；账本字段和关闭状态按实验
   闭环填写。
4. **建立正式 baseline**：discovery 确认唯一、完整 lowering `Op Name`，再按 manifest 逐 case、
   逐 repeat 采集七指标 baseline 并归档原始证据。
5. **分析瓶颈并排序**：使用既有 Roofline、Vector/Cube/Scalar/MTE、分层带宽、冲突、逐核及
   Tile DAG/时间线方法判断适用性、优先级和验证指标；ratio 只负责指路。
6. **闭合预置来源**：依次处理 `kb_selected`、`general_method`、`knowledge_card`、`template`。每个
   待实验项执行“改源码 → 完整正确性 → quick → 必要时 formal compare → 机制核对 → 接受或恢复
   → 写回账本”。
7. **进入自主优化**：确认四类预置来源全部闭合，并刷新当前最佳实现的瓶颈证据；证据不足先补证，
   再根据当前瓶颈逐个建立 `bottleneck_derived`，接受改动后重新分析。
8. **闭合与停止**：关闭全部来源和已创建项，并在最终代码上执行 fresh candidate sweep。性能目标
   差距和残留瓶颈用于提出、排序候选；仍有合法未尝试项就继续，确认没有新候选时记录候选闭合与
   最佳版本选择证据。达到或未达到理想目标本身都不能提前结束或阻止交付。
9. **同步最终事实**：项目有相关记录时，按上述 owner 合同依据最终接受代码更新 DESIGN/BINDINGS/KB_USAGE，
   逐项核对引用、file/symbol、验证方法与代码一致。
10. **最终验收**：对同一最终实现连续执行完整正确性、独立 final formal compare 和每个 P0 的
    补充 timeline；若终验使预置项结论失效，返回第 6 步重开，若发现新的自主项则返回第 7 步。
    确认无新项后交付结果与可供复核的证据；中途改代码就重做最终验收。

## 5. 交付物

独立调用本 Skill 时，`PERFORMANCE_REPORT.md` 和账本必须以真实评分后最终选定的 DSL/导出为终态。
实验若被完整正确性或用户评分否决，应标为 reject 并记录回退身份，不能让旧的
KEEP/final 文字与实际交付源码相反。主 agent 随后按 `scriptor-mode.md` 整理同根
`custom/<op>/` 的 Ascriptor 交付视图和 `REPORT.md`，执行 `delivery-check`；
此段不减免下表的正式性能证据。

| 交付 | 要求 |
|---|---|
| `test_{op}.py` | 正确、合规的冻结 baseline 与本轮正式评价候选中，按冻结聚合指标选出的最佳实现 |
| `PERFORMANCE_CASES.json` | SPEC P0 与既有测试的一一映射，以及冻结的目标 case；baseline/final 共用 |
| `PERFORMANCE_REPORT.md` | 冻结合同、目标 case、来源账本、逐 case baseline/final/目标、实验、候选比较、终态和证据路径 |
| `performance.json` / `performance.log` / `perf_report.md` | final formal compare 产物；quick 不能替代 |
| `docs/perf/round_NNN/` | baseline/final 原始 CSV、measurement/collection 与逐 case timeline |
| 最终事实记录 | 项目已有或要求的 DESIGN、DESIGN_BINDINGS 和 KB_USAGE，与接受代码一致 |

验收失败时按原始证据定点修复，重新执行受影响验证和最终验收。
