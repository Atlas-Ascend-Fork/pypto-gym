# Issue 类型映射规则

定义断裂点类型到 Issue 类型的映射关系，供阶段①报告中的"Issue 类型"字段使用。

> **阶段③的正式映射以 `references/issue-report.md` 为准**：issue 文档的标题格式（`[英文类型|中文类型]` 五前缀）、C 类断裂点的根因映射、字段填充规则均在阶段③执行，本文件只服务于阶段①的建议性标注。

## 映射规则

| 断裂点类型 | Issue 类型 | 标签 |
|------------|------------|------|
| D1, D2, D5 | Documentation | `documentation` |
| D3, D6 | Bug Report 或 Documentation | 视文档与实现哪一方错误而定 |
| A1, A2, A4, A5 | Bug Report | `bug` |
| A3 | Requirement | `enhancement` |
| E1, E2, E3, E4 | Bug Report | `bug` |
| C1, C2, C4 | 按根因映射 | 映射到根因对应的类型 |
| C5, C6 | 按根因映射 | 无明确根因时默认 Bug Report |

### C 类断裂点的 Issue 类型判定

C 类断裂点（行为模式类）的 Issue 类型根据根因判断：

- 根因指向框架 bug（如 API 返回错误导致重试）→ Bug Report
- 根因指向缺少功能或文档（如没有示例导致过度探索）→ Requirement 或 Documentation
- 根因不明确 → Bug Report（默认）

## 阶段①标题建议

阶段①报告中"建议 Issue 标题"字段直接按 `references/issue-report.md` 的标题格式生成（`[英文类型|中文类型]: 具体描述`），"具体描述"遵守 `references/writing-style.md` 第 4 条，包含场景与指向。示例：

| 断裂点类型 | 标题示例 |
|------------|----------|
| D1 | `[Documentation|文档反馈]: pypto.scatter 文档缺失` |
| D3 | `[Bug-Report|缺陷反馈]: pypto.add 文档宣称支持广播与实际行为不一致` |
| A3 | `[Requirement|需求建议]: pypto.matmul 支持 float16 输入` |
| E2 | `[Bug-Report|缺陷反馈]: F21009 编译报错缺少算子定位与修复建议` |
