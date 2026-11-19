---
type: "Reference"
title: "在 PyPTO-Pro 中检索和关联知识卡片"
description: "说明知识卡片如何进入 CLI 检索、如何标注与技术契约的关系，以及如何验证索引。"
status: "stable"
tags: ["pypto-pro", "performance", "knowledge-card"]
---

<a id="query-and-relate-performance-cards"></a>
# 在 PyPTO-Pro 中检索和关联知识卡片

知识卡片随本技能分发。使用命令行工具（CLI）查询时，通过 `--cards-root` 显式选中 `perf-cards` 数据源。只有登记在[有效卡片表](index.md#active-items)中的卡片会被检索。将文件放入分类目录、在示例文档里写出代码片段，都不会使它自动成为内核的技术契约。卡片说明一种优化在什么条件下可能有效，具体实验仍由[优化工作流](../workflows/optimization.md)决定。

查询工具按卡片自身的语言读取元数据与正文。

<a id="link-only-the-part-that-overlaps"></a>
## 只关联能够明确说明的部分

下面以 `cube-06` 示范关联范围和边界的填写方式。正式卡片中的 `scope` 与 `boundary` 使用英文，分别说明重合范围和该关联不能推出的结论。

```yaml
knowledge_links:
  schema: pypto-knowledge-links/1
  relations:
    - kind: same_mechanism
      target:
        skill: pypto-pro-kernel
        id: contract:grouped-ownership
      scope: Number independent GEMM output tiles continuously across group boundaries.
      boundary: Fixed group lengths do not establish the runtime prefix mapping or a speedup.
```

没有准确的关联目标时，省略 `knowledge_links`。填写时须有 1 至 32 条关系，每条恰含 `kind`、`target`、`scope`、`boundary` 四个字段，分别说明关系类型、目标、重合范围和适用边界。目标用稳定标识 `contract:<id>` 或 `document:skill/<relative-path>.md#<optional-anchor>`。绝对路径、行号和静态检查规则名不能作为目标；同一组 `(kind, skill, id)` 只能出现一次。查询工具会在读取时定位对应章节并计算来源文件的哈希。

| 关系类型 | 如何判断 | 例子 |
| --- | --- | --- |
| `same_mechanism` | 两者采用同一机制。 | [cube-06](cube/cube-06-grouped-continuous-core-assign.md) 与连续分配任务采用相同的编号方式。应用到当前形状时，仍要核实任务归属和实际收益。 |
| `overlaps_with` | 仅一部分做法相同。 | [vec-09](vec/vec-09-loop-sink-into-vf.md) 与另一种方案都把多行计算放进一个循环；两者的 DMA 搬运和元数据处理方式仍可不同。 |
| `uses_contract` | 卡片成立需要目标契约提供保证。 | [pipe-07](pipe/pipe-07-ub-to-l1-handoff.md) 依赖 UB 到 L1 的布局与数据可见性保证。消除一次 GM 往返和消除本地布局转换是两项不同的改动。 |
| `alternative_to` | 在明确的条件下，可以比较两种替代实现。 | [pipe-05](pipe/pipe-05-gm-to-l1-direct-load.md) 只在原操作数无需 Vector 变换时，与直接加载方案作对照。 |

关联不改变卡片状态，也不继承目标的适用条件或验证结论；`scope` 与 `boundary` 描述关联范围，不替代本卡的 `applicability`、`target_api_gate` 或正文。没有该字段的 Active 卡片仍可被查询；没有准确目标时省略关联，不填空列表或占位 ID。

关联不具备传递性。两张卡共用一条容量约束，仍须分别判断改法并验证收益。解释接口也要区分具体作用：缓存维护 API 不能充当 L2 旁路开关，`UNPK/PACK` 也不执行 BF16 数值转换。

<a id="接入与校验"></a>
<a id="validate-the-library"></a>
## 验证卡片库

验证前应安装卡片查询依赖，并明确选择卡片库及匹配版本的 PyPTO 源码。本节关注元数据、索引登记、关联目标及代表卡片检索；这些接入检查不替代卡片内容、正确性和真实性能收益的独立审核。具体接口参见[优化查询工具](../tooling/optimization.md#optional-performance-card-source)。

分段续读应使用返回的请求及文件内容 SHA-256（`source_sha256`），并保持 CLI 的卡片库和源码配置一致。检查报告应注明所使用的数据源和覆盖范围。新增卡片还应满足[验证规范](CONTRIBUTING.md#validation-and-evidence)和[审核规范](REVIEW.md#review-a-card)。
