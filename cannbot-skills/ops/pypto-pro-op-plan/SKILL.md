---
name: pypto-pro-op-plan
description: 将算子需求整理为 PyPTO-Pro 开发所需的规格、目标版本资料证据和 kernel 契约，结合当前 pypto-pro-kernel 的适用指导形成可验证的规划产物。
---

# PyPTO-Pro 算子规划

整理需求、实现约束和资料证据，生成彼此一致的规划产物。需求规格化与资料探索分别复用
`pypto-pro-intent-understand` 和 `pypto-pro-material-explore`。

## 输入与产物

输入：用户已授权的算子需求、目标版本 devkit 缓存、当前 `pypto-pro-kernel` 和共享
`performance-constraints.md` 的路径。已有安装可从 `$PYPTO_DEVKIT_DIR`、同级
`pypto-pro-kernel/` 和 `$CANNBOT_CONFIG_ROOT/references/performance-constraints.md` 定位。
所复用的两个 Skill 及其脚本须可加载。

使用任务指定的输出目录 `<op-dir>`，默认 `custom/<op>/`；目录名 `<op>` 与 SPEC 算子名一致。在其中生成：

| 产物 | 内容 |
|------|------|
| `SPEC.md` | 数学语义、公开接口、P0 cases |
| `PRO_MATERIAL_INDEX.md` | 本次目标版本资料目录 |
| `EXPLORE_REPORT.md` | API 可行性、约束和证据 |

调用上述 Skill 时传入本次 SPEC、输出目录、缓存和共享规则的实际路径，替换命令中的默认路径；
需求确认与资料阅读遵循各自 Skill 的规则。

## 生成与校验

复用已提供且校验有效的产物，按以下依赖补齐：先明确规格，再核实 API 可行性，最后将适用指导与待验证问题写回探索报告。

### 1. 冻结需求语义

加载 `pypto-pro-intent-understand`，按已有授权生成或更新 `SPEC.md` 并验证。

### 2. 补充 kernel 契约

在 `SPEC.md` 末尾增加 `## kernel 契约补充`，记录通用需求模板尚未表达的实现语义：

| 字段 | 记录要求 |
|------|--------------|
| 辅助张量语义 | 确认是公开输入、模型参数还是内部临时量 |
| cast 边界链 | 记录输入、累加、后处理、输出各段的语义 dtype |
| 累加/写回语义 | 确认覆盖写、跨块累加或原子累加的数学要求 |
| 目标设备 | 运行时或 build 配置已指定 target 时使用指定值；未指定时默认 A5，并在 SPEC 标注这是默认假设 |

涉及数学语义或公开接口的修订按 intent-understand 校验；硬件实现的可选方案及待核实条件记入资料报告。

### 3. 探索目标版本资料

加载 `pypto-pro-material-explore`，以包含 kernel 契约的 `SPEC.md` 为输入，重新扫描资料
索引并生成 `PRO_MATERIAL_INDEX.md` 与 `EXPLORE_REPORT.md`。结论须保持 SPEC 的数学语义；
若发现公式、接口或 P0 case 有问题，返回 intent-understand，按其规则修订和校验。

### 4. 对接当前 kernel 工作流

阅读 [pypto-pro-kernel](../pypto-pro-kernel/SKILL.md) 的当前阶段与约束，并从其
[reference index](../pypto-pro-kernel/references/index.md) 进入实际 owner。根据公式、
支持范围、dtype/layout、目标版本和数据依赖选择指导；不按算子名称猜选，也不强行归入某一模式。
融合算子的多个有效机制都要考虑；无适用 pattern 时如实说明，仍须检查通用 API、资源和数值约束。

将本 class 的具体指导放在 `EXPLORE_REPORT.md` 现有 §5/§6，保持报告模板的二级标题：
每项说明来源、适用前提、影响的设计决定或 invariant、以及待验证问题。引用采用可搬迁的
`pypto-pro-kernel/SKILL.md` 或 `pypto-pro-kernel/references/...md#section` 路径，
用行内代码或 Markdown 链接表示，章节锚点必须真实存在；无需另建知识选择 JSON、拓扑表或原文副本。

原文 owner 保持权威。规划阶段记录设计依据与开放问题；下游按真实实现继续补读相关 owner，
不受冻结参考清单限制。区分官方契约、安装源码、历史测量和本次验证，不能把旧通过记录当作
当前 target 的验收。target 未确认时保留 SPEC 的默认假设及核验风险，不能从历史样例填硬件常量。

### 5. 收尾自检

将 `<skill-dir>`、`<缓存绝对路径>` 和 `<op-dir>` 替换为实际路径，保留引号，运行只读检查器：
它复用 SPEC/索引校验器，检查报告结构、引用及样例/指南路径覆盖，以及当前 kernel 引用文件和锚点。
另按公式步骤、接口约束和当前风险核对报告证据是否充分。

```bash
python "<skill-dir>/scripts/validate_plan.py" \
  --op-dir "<op-dir>" \
  --devkit "<缓存绝对路径>"
```

检查器默认使用同级 `pypto-pro-kernel`；安装位置不同时传 `--kernel-root "<kernel-root>"`。
路径检查不能证明引用适用、API 在目标版本可用或规划已在设备验收。

输出非 0 时一次性修正全部 `[FAIL]` 后重跑。

## 完成条件

- 三类规划产物均存在；
- SPEC 校验通过，探索结论与其数学语义一致；
- INDEX 的 §A/§B/§C 与本次缓存一致；
- EXPLORE_REPORT 没有未解决的 `unsupported` 阻断；
- 报告中的当前 kernel owner 引用可解析，适用条件与本 class 的事实一致；
- 规格语义、资料结论与待验证设计问题相互一致。

返回产物路径、已完成的校验及其结果、剩余风险和需要补齐的证据。
