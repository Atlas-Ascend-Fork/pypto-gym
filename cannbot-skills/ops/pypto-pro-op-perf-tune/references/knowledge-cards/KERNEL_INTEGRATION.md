---
type: "Reference"
title: "性能卡片与 kernel skill 的知识源接入"
description: "面向卡片维护者和工具开发者的关联格式、读取边界、候选归组与验证协议。"
status: "stable"
tags: ["pypto-pro", "knowledge-card", "integration"]
---

# 性能卡片与 kernel skill 的知识源接入

性能卡片保留在本知识库维护，`pypto-pro-kernel` 通过显式配置的目录只读接入。基础 kernel skill 可以独立使用；启用卡片不会加载性能调优 Skill 的工作流、创建来源账本或改写 kernel。卡片为优化参考文档，不能作为编译器事实、强制实现政策或当前收益证据。

## 维护职责

- 卡片维护者负责正文、`item_id`、Active 登记、适用条件、目标/API 门槛和可选关联；技术内容与关联范围由人工 review。
- Kernel skill 负责可选知识源读取、来源身份、查询、原文读取、关联校验及已有候选的归组展示。CLI 与 MCP 共用实现。
- API、物理布局、同步和完整 examples 保留各自 owner。确认正文重复后可改为引用，不复制契约，也不在 kernel skill 维护反向映射表。
- 卡片包分发必须保留索引登记的相对目录。当前安装器只复制 kernel skill，卡片目录由调用方显式提供；接入不自动安装依赖或卡片资源。

## 关联字段

向卡片现有 YAML frontmatter 添加可选 `knowledge_links`。本字段是本仓的扩展约定；没有字段的 Active 卡片仍可查询，不要求对全部卡片补齐空映射。

```yaml
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: same_mechanism
      target:
        skill: pypto-pro-kernel
        id: "contract:grouped-ownership"
      scope: "将多组独立 GEMM 的输出 tile 连续编号，避免每组重启 worker 分配。"
      boundary: "本卡示例为固定 tiles_per_group；关联不扩大到目标的所有运行时组长方案，也不证明当前收益。"
```

| 字段 | 约束与含义 |
|---|---|
| `knowledge_links.schema` | 固定为 `pypto-knowledge-links/1` |
| `relations` | 1–32 条关联；每条仅含 `kind`、`target`、`scope`、`boundary` |
| `kind` | 下表中的一种关系；同一卡片不能重复声明相同 kind、目标 skill 和目标 ID |
| `target.skill` | 稳定的 skill 名称；当前接入器选择 `pypto-pro-kernel`，其它目标 skill 会明确报告未选择 |
| `target.id` | 已存在的 `contract:<id>` 或 `document:skill/<相对路径>.md#<可选锚点>`，不能填写候选规则名、绝对路径或行号 |
| `scope` | 非空文本，说明此关联覆盖的具体交集；不是可自动求值的条件表达式 |
| `boundary` | 非空文本，说明不能互相继承的条件、范围或结论 |

卡片的整体条件仍由 `applicability`、`target_api_gate` 和正文维护。关联中不手写 SHA、验证状态或实验收益；工具读取时生成卡片、索引及目标来源的 SHA。

| 关系 | 语义 | 消费方式 |
|---|---|---|
| `same_mechanism` | 相同优化机制 | 归入已有候选组；当前源码、工作负载及具体改法的等价性仍待确认 |
| `overlaps_with` | 部分机制或组织方式重叠 | 提供关联参考，不自动合并候选 |
| `uses_contract` | 改法需要核对的技术契约 | 提供实现约束，不将卡片与契约内的优化候选判为同一改法 |
| `alternative_to` | 条件性的替代路线 | 保留条件和备选，以当前工作负载及实测选择 |

关系不自动传递：两张卡都关联同一契约，不表示它们等价。共享来源也不能共享未覆盖的验证结论。`same_mechanism` 不产生 `supported` 或 `verified` 状态。

## Active 卡片覆盖

当前 Active 卡片均有 kernel skill 关联，覆盖下表范围。完整声明以各卡片 frontmatter 为准；本表不维护第二份目标映射。新增卡片仍允许暂缺关联：找不到准确交集时保留可检索的正文，不用宽泛关键词硬配目标。

| 类别 | 当前编号 | 关联重点 |
|---|---|---|
| Vector | `vec-01`–`vec-19` | 融合与状态、Cast/容器、归约、广播、mask、布局及任务批处理 |
| Cube | `cube-01`–`cube-13` | phase、buffer、K 切分、驻留、格式、精度和完整输出网格 |
| Memory | `mem-01`–`mem-06` | valid region、批量搬运、cache 能力边界、写回及状态生命周期 |
| Pipeline | `pipe-01`–`pipe-07` | 诊断、槽位、C/V 交错、直载与片上交接 |
| Scalar | `scalar-01`–`scalar-03` | 不变 metadata、查表及每块固定开销 |
| Synchronization | `sync-01` | 同步层次、事件与阶段边界 |
| Algorithm | `algo-01` | 收缩路线比较及指数中间量范围 |

关联优先准确描述已有交集：精确机制不存在时，可以引用该技术实际需要核对的契约，并在 `boundary` 写明尚未覆盖的算法或 API 能力。`mem-03` 的 `dcci` 引用不提供 L2 bypass；`vec-08` 的隐式目标引用不证明 MERGING 支持；BF16 的数值 Cast 也不能被 UNPK/PACK 位重排替代。关联到共同的容量或布局契约不表示两张卡片重复。

### 关联示例

以下示例说明如何区分同一机制、局部重叠、契约使用和条件性替代；其它卡片按同一格式维护。

| 卡片 | 目标 | 关系与边界 |
|---|---|---|
| [cube-06](cube/cube-06-grouped-continuous-core-assign.md) | `contract:grouped-ownership` | 同一连续分配机制；固定组形状不继承动态前缀、空组和逐阶段覆盖结论 |
| [vec-09](vec/vec-09-loop-sink-into-vf.md) | `document:skill/patterns/vf-row-reduce.md#batch-independent-rows-even-when-each-row-fits` | 循环下沉与行批量组织部分重叠；不等价于整个 DMA/metadata/worker 方案，也不映射到 VF 循环分支外提规则 |
| [pipe-07](pipe/pipe-07-ub-to-l1-handoff.md) | `document:skill/api/gemm.md#transformed-operands-format-residency-and-repeated-work`；`contract:ub-l1` | 前者为重叠的驻留/交接讨论，后者为技术契约；删除 GM 往返与删除局部 ND→NZ 转换是独立改法 |
| [pipe-05](pipe/pipe-05-gm-to-l1-direct-load.md) | `document:skill/api/gemm.md#thin-row-contractions-compare-arithmetic-and-transport-paths` | 只关联原始 GM 操作数的运输备选；数据需 Vector 加工或存在有效预取重叠时，不能机械替换为直载 |

## 接入与校验

`CARDS_ROOT` 指本目录，即含 `index.md` 的知识卡片目录；`KERNEL_SKILL_ROOT` 指 kernel skill 的实际目录。路径由调用方指定，不从工作目录猜测 sibling skill。卡片适配依赖可选的 PyYAML，安装到调用检查工具的 Python 环境：

```bash
python -m pip install -r "$KERNEL_SKILL_ROOT/scripts/requirements-cards.txt"
python "$KERNEL_SKILL_ROOT/scripts/check.py" --cards-root "$CARDS_ROOT" query cube-06 --kind document --format json
python "$KERNEL_SKILL_ROOT/scripts/check.py" --cards-root "$CARDS_ROOT" query 'document:perf-cards/cube/cube-06-grouped-continuous-core-assign.md' --format json
python "$KERNEL_SKILL_ROOT/scripts/check.py" --pypto-root "$PYPTO_SRC" --cards-root "$CARDS_ROOT" query --validate --format json
python "$KERNEL_SKILL_ROOT/scripts/check.py" --cards-root "$CARDS_ROOT" suggest --source candidate.py
```

精确文档 ID 读取返回原文及 SHA；按返回的 `resolve.reference_sha256` 固定后续读取身份。全量准入还需要匹配的外部框架源码。框架缺失时不能声称全量准入通过；卡片原文读取不要求导入框架或运行 NPU。

MCP 使用相同的启动级目录配置，先按 kernel skill 的要求安装可选 MCP SDK；已有 `kernel_query`、`kernel_suggest` 复用 CLI，不新增一套工具语义：

```bash
python "$KERNEL_SKILL_ROOT/scripts/check.py" --cards-root "$CARDS_ROOT" mcp --workspace "$PROJECT_ROOT"
```

接入器只枚举 Active 表指向的文件，不扫描目录采纳卡片。校验卡片类型、状态、稳定 ID 与文件前缀，拒绝重复登记、重复 YAML 键、YAML alias、目录逃逸、未知关联格式和失效锚点。明确配置但不可读的知识源返回 `unverified`，不伪装成零卡片或 API 不支持；无效卡片会隔离，覆盖范围与原因保留在反馈中。

`coverage.documents.cards` 返回 Active 登记数、成功读取数，以及 `linked_cards`、`unlinked_cards`、`relation_count`。关联计数只覆盖已接纳且目标有效的卡片，错误另行列出。验收时同时检查读取覆盖、未关联数与错误数；全部已关联也不表示全部技术已实现静态检测或完成上板验证。

代码块、缩进代码和 HTML 注释中的表格行不构成 Active 登记；有效 Markdown 表格可带最多三列前导空格。重复的稳定 ID 会隔离所有同 ID 条目，不任意采用第一份。合法相对路径及文档目标在解析后规范化，规范化后重复的关联同样拒绝。文档覆盖的错误计数、读取统计和 manifest 包含卡片知识源，详细卡片覆盖仍可单独查看。

查询以 `document:perf-cards/<索引中的相对路径>` 标识原文，`card.item_id` 保留稳定卡片编号，`document_role=performance_card` 与契约、官方文档分开。索引、卡片或关联目标的变化会更新来源身份并使旧查询 cursor 失效。卡片来源不产生新的执行证据。

长卡片返回 partial content 时，保留相同的 `--cards-root`，将返回的带 SHA `next` 请求交给 `read --request` 或 `kernel_read` 续读，并移除父批次的 `request_index`。默认读取范围包含显式选择的卡片根目录；若使用自定义 context policy，其 `roots` 仍是最终范围，需要在其中明确允许该目录。文档对象分页 cursor 不能替代原文字节续读。

已有源码检测候选通过 `related_cards` 展示直接关联；`same_mechanism` 标为 `same_mechanism_only`，其它关系标为 `reference_only`，具体改法等价性与当前适用性保留为 `not_established`。未实现专属检测的卡片只能通过查询提供条件性参考。组间连续分配可关联现有候选；循环下沉、GM 往返消除和 GM→L1 直载不会冒充已实现的静态规则。

## 开发者验收要点

- Active 中登记的卡片可按稳定 ID 或原文 ID 读取；未登记文件、模板、Draft 和 Retired 不因目录存在而进入候选。
- 缺少可选卡片配置时，基础 CLI/MCP 行为保持独立；启用但缺依赖、文件损坏或目标失效时明确报告覆盖不足。
- `cube-06` 只增加已有候选的来源关联；`pipe-07` 的契约引用不会与局部布局转换候选合并。关联不传播适用条件或数值/性能证明。
- CLI 与 MCP 的分页、原文 SHA、候选和错误语义一致；在含空格的独立目录下仍可读取。
- 卡片更新只维护当前知识与关联，实验状态留在任务的外部或 ignored scratch，不生成仓内历史台账。
