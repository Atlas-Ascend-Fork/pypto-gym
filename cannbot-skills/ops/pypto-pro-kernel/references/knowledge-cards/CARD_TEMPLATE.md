---
type: "Template"
title: "性能知识卡片模板"
description: "提供可复制的英文卡片结构，填写规则统一见贡献指南。"
status: "stable"
tags: ["pypto-pro", "performance", "knowledge-card"]
---

<a id="card-template"></a>
# 性能知识卡片模板

将模板中的占位内容替换为已核实的英文卡片内容，按[格式与收录规则](CONTRIBUTING.md#card-format-and-admission)填写，并使用受测实现中的 Before/After 片段。

````markdown
---
type: "PyPTO Performance Optimization Card"
title: "<reusable mechanism>"
description: "<change and the cost it removes>"
status: "stable"
tags: ["pypto-pro", "<category>"]
item_id: "<category>-NN"
bound_hint: "<resource or dependency>"
applicability: "<observable prerequisites and scope>"
target_api_gate: "<required target and supported API, dtype or layout>"
---

# <item_id> — <mechanism>

## When to use it

<Describe the class of computations and the observable bottleneck.>

## Before

<Explain the original cost and the declarations or call context needed to understand this baseline excerpt.>

```python
```

## After

<Explain the optimization and which work it saves or adds. Keep the same computation scope as Before.>

```python
```

## Verification and limitations

<State the numerical, memory and synchronization conditions and at least one exclusion. Identify the fixed source revision, file, symbol and location of each excerpt. Link the complete comparison and raw evidence, state the measured conditions and results, and explain how the evidence tests the claimed mechanism.>

## Sources

<List the inspected references that establish the relevant contracts.>
````

可选的 `knowledge_links` 按[关联范围与边界示例](KERNEL_INTEGRATION.md#link-only-the-part-that-overlaps)填写。
