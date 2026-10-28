---
name: pypto-pro-material-explore
description: 基于已验证的 SPEC.md 探索当前 PyPTO-Pro 版本的 API、官方样例、教程与设计指南，生成可复用资料索引和带来源的可行性报告。用于 Stage 1 的 API 映射、版本约束和参考模式取证；不要修改需求语义、选择 KB、设计 Module/tile 或编写 kernel。
---

# PyPTO-Pro 资料探索

只负责回答“当前目标版本能否实现、证据在哪里、有哪些约束”。需求语义以 SPEC 为准；
KB 路由由 plan 负责，最终 Module/tile/同步方案由 Stage 3 决定。

## 输入与输出

- 输入：已通过校验的 `custom/<op>/SPEC.md` 和已装配的 `$PYPTO_DEVKIT_DIR`。
- 输出：
  - `custom/<op>/PRO_MATERIAL_INDEX.md`
  - `custom/<op>/EXPLORE_REPORT.md`
- 模板：
  - [资料索引模板](templates/pro_material_index.md)
  - [探索报告模板](templates/explore_report.md)
  - [官方样例清单](references/official_samples.md)

缓存缺失、SPEC 未冻结或官方样例清单有缺项时返回阻断证据；不得自行同步缓存或修改环境。

## 证据优先级

1. 当前 `$PYPTO_DEVKIT_DIR` 中的目标版本 API 文档；
2. `official_samples.md` 指定且缓存中存在的官方样例；
3. 当前缓存中的 Pro 编程指南、快速入门与共享简介（见下方 §C 范围）；
4. KB 中与当前 class 匹配、状态为 validated 的补充材料。

API 签名、平台常量和能力边界以前三项为准。KB 或模型记忆不能覆盖目标版本事实。

同步与 VF 的 API 说明见 `$CANNBOT_CONFIG_ROOT/references/performance-constraints.md`。
最终实现选择由 [Design Skill](../pypto-pro-op-design/SKILL.md)的“两条实现约束”判定；
本 skill 只为每个相关步骤记录目标版本证据，供 Stage 3 裁定。

## 工作流

### 1. 重建资料索引

将 `<skill-dir>`、`<缓存绝对路径>` 分别替换为本 skill 的绝对路径和编排器给定的 `PYPTO_DEVKIT_DIR` 值，保留引号；每次仅用下列固定生成器重建索引：

```bash
python "<skill-dir>/scripts/build_material_index.py" \
  --devkit "<缓存绝对路径>" \
  --output "custom/<op>/PRO_MATERIAL_INDEX.md"
```

生成器只读 devkit，失败时不覆盖已有索引；资源装配由 orchestrator 负责。

| 章节 | 来源 | 规则 |
|------|------|------|
| §A API | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/**/*.md`，总索引为 `docs/pypto_pro/api/index.md` | 全量、稳定排序 |
| §B 官方样例 | `references/official_samples.md` | 只复制清单；先核对每个清单文件存在 |
| §C 指南/教程 | `$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/**/*.md`、`$PYPTO_DEVKIT_DIR/docs/guide/quick_start/pro/**/*.md` 与 `$PYPTO_DEVKIT_DIR/docs/guide/introduction.md` | 全量列出，含各目录的 `index.md`；不扫描 Tensor 专属指南目录 |

路径写成相对缓存路径并记录计数。API 根索引、§B 清单项、§C 任一 Pro 目录及其根索引或简介缺失时生成器非零退出；
缓存中的清单外样例不写入索引，也不得参考。

### 2. 按 SPEC 分解数学步骤

把 SPEC 的公式/算法拆成原子数学步骤，但不改变表达式。对需要近似的步骤，记录近似来源、
适用区间和至少两个已知点的数值 sanity check。发现 SPEC 自相矛盾时停止并要求回到需求阶段。

### 3. 建立 API 映射与约束表

对每个原子步骤：

1. 从索引 §A 定位候选 API 并阅读全文；名称不足以定位时，只在 §A 已列文档中全文检索，
   结论写入 EXPLORE_REPORT，不手工修改索引。
2. 记录准确签名、参数单位/语义、dtype、shape、layout、MemorySpace、tile 和版本约束。
3. Vector 步骤先查目标版本是否有单一 VF API；没有时再记录有文档依据的 VF 组合；两者
   都无法完整表达语义才标记 `unsupported`。不要在本阶段做 VF/tile-op 性能选择。
4. Cube 步骤记录对应 `pl.*` API、累加语义及 layout 约束。
5. 没有可行调用链时标记 `unsupported`，附查过的文档和失败原因；不得编造 API。

同时从当前文档提取目标、UB、event_id、对齐等平台常量；每个值记录版本和来源，不预填
历史参考值。

### 4. 阅读官方样例

全量阅读索引 §B 的指定样例并按纯 Cube、纯 Vec、Cube/Vec 融合分类。对每个样例只记录
与当前算子有关或跨算子通用的证据：API 调用、tile-group、循环、分核、同步、尾块和
cross-core 边界。所有结论必须带样例路径。

可按 `pypto-pro-op-kb/examples/kernel-index.md` 补充条目，但仅允许 `validated`、路径存在且
topology/dtype/layout/platform 匹配的代码；`study` 不能作为正确性依据。

### 5. 阅读指南与教程

逐一评估索引 §C 中的 Pro 编程指南、快速入门及共享简介。记录与当前算子相关的设计
模式、适用条件和章节路径；不相关也要标记“不适用”，以证明遍历完整。

### 6. 汇总报告

使用报告模板整合：

- 公式分解与逐步骤 API 映射；
- 参数语义和版本约束；
- 官方样例与教程证据；
- 环境常量快照；
- 可行性、unsupported 阻断及可证实替代路线；
- 按 API/样例/guide 分类的证据索引。

本节是 Stage 3 的事实输入：报告可以提出 Stage 3 需解决的设计问题，但不得在这里冻结
topology、Module、tile、同步事件或 KB_SELECTION。

## 完成条件

- 上方生成器以 `--check` 检查现有 INDEX 时 exit code 为 0；
- 每个公式步骤都有可行 API 链或带完整证据的 `unsupported`；
- 所有采用的 API、平台常量、样例模式均带目标版本来源；
- 所有官方样例和 §C 文档均有适用性记录；
- EXPLORE_REPORT 的 1–10 章节完整，无未解决阻断项；
- 没有修改 SPEC，也没有执行 Stage 3/4 决策。
