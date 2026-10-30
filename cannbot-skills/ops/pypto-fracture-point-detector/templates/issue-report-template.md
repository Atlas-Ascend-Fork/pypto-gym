# Issue 报告模板

本模板定义阶段③产出 `issues/issue-XX/issue.md` 的结构。字段名与必填/选填标注面向 GitCode `cann/pypto` 仓 ISSUE 规约，类型与必填字段集见 `references/issue-report.md` 的"标准字段集"；字段名保持中英双语；填充内容使用中文并遵守 `references/writing-style.md`。按 `references/issue-report.md` 的映射规则选择 Bug Report / Documentation / Requirement 三种结构之一。

---

## 结构 A：Bug Report

```markdown
[Bug-Report|缺陷反馈]: {含场景与指向的具体描述}

### Describe the current behavior / 问题描述 (Mandatory / 必填)

{现象与根因的完整说明，含复现阶段的定性修正与上游核对结论。第一句话必须是"场景 + 操作 + 现象 + 影响"的结论式概括，随后再展开细节；内部标识（错误码、枚举名、内部函数名）首次出现必须紧跟一句人话解释，见 writing-style.md 第 7 条。}

### Environment / 环境信息 (Mandatory / 必填)

- **服务器/NPU 型号**: {型号}
- **PyPTO 版本/Commit**: {commit hash}
- **CANN 版本**: {版本号}
- **Python 版本**: {版本号}
- **操作系统**: {版本号}

### Steps to reproduce the issue / 重现步骤 (Mandatory / 必填)

{复现阶段的命令序列，引用 issues/issue-XX/ 下的脚本文件名。}

1. {步骤1}
2. {步骤2}
3. {步骤3}

### Describe the expected behavior / 预期结果 (Mandatory / 必填)

{按复现阶段核实后的口径描述预期行为。}

### Related log / screenshot / 日志 / 截图 (Mandatory / 必填)

{运行日志关键片段，逐字引用；注明完整日志文件路径。}

### Special notes for this issue/备注 (Optional / 选填)

{复现状态（含未复现子现象及归因线索）、与历史记录的差异、上游核对结局、仓分类附带项、完整证据目录（issues/issue-XX/README.md）的引用。}
```

---

## 结构 B：Documentation

```markdown
[Documentation|文档反馈]: {含场景与指向的具体描述}

### Document Link（文档链接）

{问题文档的仓内路径（path:line）或在线文档地址。}

### Issues Section（问题文档片段）

{文档原文，逐字引用。}

### Existing Issues（存在的问题）

{缺失或错误的具体说明，含上游核对结局：① 上游已解决 / ② 上游至今没写 / ③ 索引失效，注明核对日期与 commit。}

### Suggested Fix / 修正建议 (Optional / 选填)

{复现阶段给出的修复方向，指向具体文件。}
```

---

## 结构 C：Requirement

```markdown
[Requirement|需求建议]: {含场景与指向的具体描述}

### Background（背景信息） (Mandatory / 必填)

{需求解决的问题、现状的不足，含影响面数据（如中招 session 数）。}

### Origin（信息来源） (Mandatory / 必填)

{断裂点分析批次：session 来源、报告路径、复现日期。}

### Benefit / Necessity（价值/作用） (Optional / 选填)

{修复后对 agent 效率或成功率的收益，可量化则量化。}

### Design（设计方案） (Optional / 选填)

{修复思路，可用伪代码；指向具体文件。}
```
