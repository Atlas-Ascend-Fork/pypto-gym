# 断裂点识别报告

<!-- 聚合分类（Type-1/2/3）判定定义 —— 填每个断裂点的 {aggregate_type} 时必须严格使用以下口径，禁止自创分类标准（如"直接阻断/效率损失/信息缺失"）：
| 分类 | 定义 | 典型例子 |
|------|------|----------|
| Type-1 | pypto 框架/代码/文档/流程问题，与 pypto agent 建设直接相关 | 框架静默返回错误结果、API 行为异常、报错信息缺失/模糊/误导、文档缺失或错误、pypto-gym 工具链（lint/hook/orchestrator）缺陷 |
| Type-2 | benchmark 框架配置、调度、日志记录逻辑、超时策略等非 pypto 产出的问题 | runner 调度错误、设备池分配问题、benchmark 自身日志缺陷 |
| Type-3 | 硬件故障、集群问题、用户操作中断等外部因素 | 设备硬件故障、session 被外部中断 |
一致性自检（填写时逐条必做）：若"分类依据"文字表明该问题需要 pypto 或 pypto-gym 侧修复，则必须标 Type-1——"需要框架侧修复"与 Type-2/Type-3 相互矛盾，出现时以依据为准修正分类；Type-2/Type-3 仅用于修复责任完全在 benchmark 运行器或外部环境的问题。拿不准时标 Type-1 并在依据中注明"边界存疑"。 -->

<!-- 作弊检查结论为疑似或确认时，保留以下醒目标注；无嫌疑时删除本行 -->
> ⚠️ 作弊检查结论：{疑似|确认}。本 session 的断裂点结论可信度受污染，进入复现阶段前需人工复核。

## 1. 摘要

- **数据来源**: {data_source}
- **Session 时间**: {session_time}
- **检测时间**: {detection_time}
- **断裂点总数**: {total_count} 个
- **按优先级分布**:
  - 致命: {critical_count} 个
  - 高: {high_count} 个
  - 中: {medium_count} 个
- **按根因归属分布**:
  - 文档: {doc_count} 个
  - 框架: {framework_count} 个
  - 两者: {both_count} 个
- **按 Issue 类型分布**:
  - Bug Report: {bug_count} 个
  - Documentation: {doc_issue_count} 个
  - Requirement: {feature_count} 个
- **作弊检查结论**: {无嫌疑 / 疑似 / 确认}

## 2. 环境信息

- **CANN 版本**: {cann_version}
- **PyPTO Commit**: {pypto_commit}
- **服务器类型**: {server_type}
- **Python 版本**: {python_version}
- **操作系统**: {os_info}

## 3. 作弊检查

- **结论**: {无嫌疑 / 疑似 / 确认}
- **检查范围**: {覆盖的文件访问与写入操作数量及类型；须含仓内同族邻近算子参考实现的读取记录}

<!-- 结论为无嫌疑时仅保留以上两行；疑似或确认时补充以下证据清单 -->

### 证据

1. {操作序列证据：读取/复制行为 → 写入行为 → 雷同点，引用 session 原文并标注消息位置}

## 4. 优先修复列表

| 优先级 | 编号 | 类型 | 实体 | 对应阶段 | 解决/绕过耗时 | Issue 类型 | 置信度 |
|--------|------|------|------|---------|--------------|------------|--------|
| {priority} | {fp_id} | {fp_type} | {entity} | {stage} | {resolution_duration} | {issue_type} | {confidence} |

<!-- 按优先级排序：致命 > 高 > 中，同优先级内按置信度排序：高 > 中 -->

## 5. Session 级断裂点

<!-- 如无 Session 级断裂点（C5、C6），省略此章节 -->

### 5.1 [{fp_id}] {type_code}-{type_name}

- **类型**: {type_code}-{type_name}
- **优先级**: {priority}
- **根因归属**: 两者
- **发生时间**: {occurrence_time}
- **对应阶段**: {stage}
- **解决/绕过耗时**: {resolution_duration}
- **Issue 类型**: {issue_type}
- **建议 Issue 标题**: `{issue_title}`
- **置信度**: {confidence}
- **聚合分类**: {aggregate_type}
- **分类依据**: {aggregate_reason}

#### 问题描述

{description}

#### 证据片段

```
{evidence}
```

#### 优化建议

{suggestion}

## 6. 实体级断裂点详情

### 6.1 [{fp_id}] {type_code}-{type_name}: {entity}

- **类型**: {type_code}-{type_name}
- **优先级**: {priority}
- **根因归属**: {root_cause}
- **发生时间**: {occurrence_time}
- **对应阶段**: {stage}
- **解决/绕过耗时**: {resolution_duration}
- **Issue 类型**: {issue_type}
- **建议 Issue 标题**: `{issue_title}`
- **置信度**: {confidence}
- **可能关联**: {related_fps}
- **聚合分类**: {aggregate_type}
- **分类依据**: {aggregate_reason}

#### 问题描述

{description}

#### 证据片段

```
{evidence}
```

#### 复现步骤

1. {step1}
2. {step2}
3. {step3}

#### 期望行为

{expected}

#### 实际行为

{actual}

#### 相关链接

- {links}

#### 优化建议

1. {suggestion1}
2. {suggestion2}

---

<!-- 每个断裂点使用 --- 分隔，重复 6.x 的结构 -->
