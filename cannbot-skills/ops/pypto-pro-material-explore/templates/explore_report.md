---
schema_version: 2
op_name: {operator_name}
feasibility: {feasibility}
---

# PyPTO-Pro 资料探索报告

> **生成时间**: {timestamp}
> **全量资料索引**: `custom/{operator_name}/PRO_MATERIAL_INDEX.md`

---

<!-- REQUIRED -->
## 1. 概述

### 1.1 输入摘要

{输入内容摘要}

### 1.2 算子分类

- **计算引擎类型**: {Vector / Cube / VF / 混合}
- **判断依据**: {公式中是否含 matmul（Cube）、仅逐元素/归约（Vector）、或需 VF 指令}

### 1.3 可能会涉及的 API 类别

> 以下为常见类别提示，实际 API 以本次 PRO_MATERIAL_INDEX §A 扫描结果为准。
> Vector 规则见 [Design Skill](../../pypto-pro-op-design/SKILL.md)的“两条实现约束”；本阶段记录默认 VF 映射，KB 模板例外交 Stage 3 裁定。

| 类别 | 是否涉及 | 关键 API | 指定算子是否覆盖 |
|------|---------|----------|------------------|
| 数据搬运 | {是/否} | `load_tile` / `store_tile` / `load` / `store` / `move` | {覆盖 / 未覆盖} |
| VF 实现 | {是/否} | `vf.add` / `vf.mul` / `vf.max` / `vf.astype` / `vf.exp_sub` / `vf.muls` 等 | 覆盖（FA/lightning/vf_api） |
| 矩阵计算（Cube） | {是/否} | `matmul` / `matmul_acc` | 覆盖（matmul 类样例） |
| 系统访问 | 是 | `get_block_idx` / `get_block_num` | 覆盖 |
| 控制流 | 是 | `section_vector` / `section_cube` / `pl.range` | 覆盖 |
| 工具 | {是/否} | 动态维度声明 / `set_validshape` / `make_tile_group` | 覆盖 |

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | vf.* 调用链（vec）/ pl.* 调用链（cube） | 说明 |
|------|----------|----------|--------------------------------------|------|
| {n} | {op_type} | {math_expr} | `vf.{api1}` → `vf.{api2}`（vec）/ `pl.{api1}`（cube） | {desc} |

---

<!-- REQUIRED -->
## 3. API 文档探索

> **来源**: 探索方向 1 — 基于 `PRO_MATERIAL_INDEX.md` §A

### 3.1 API 映射结果

| 步骤 | 数学表达 | vf.* 调用链（vec）/ pl.* 调用链（cube） | 状态 | 约束满足 |
|------|----------|--------------------------------------|------|----------|
| {n} | {expr} | `vf.{api1}` → `vf.{api2}`（vec）/ `pl.{api1}`（cube） | {直接可用/需组合/不支持} | {✓/⚠/✗} |

### 3.2 替代方案

<!-- 每个 Vector 步骤均填写；无 Vector 步骤时注明不适用 -->

| 步骤 | 默认 VF 映射 | API 依据 | 目标版本 | 适用条件 | 交接状态 |
|------|---------------|----------|----------|----------|----------|
| {n} | `vf.{api}` | {API path + excerpt} | {version} | {dtype/shape/layout} | 待 Stage 3 核对已选 KB 模板 |

### 3.3 API 约束

| API | 约束项 | 要求 | 结果 | 参数语义/寄存器级行为/同名差异 |
|-----|--------|------|------|----------------------|
| {api} | {constraint} | {requirement} | {✓/⚠/✗} | {参数语义（如 offset 单位、layout 取值、matmul module 累加语义）、寄存器级行为（如 vf.astype BF16/FP32 映射）、与同名/相似 API 差异（如 vf.gather vs pl.gather）} |

### 3.4 MemorySpace 约束

| 操作 | MemorySpace | 约束 |
|------|-------------|------|
| {op} | {space} | {constraint} |

---

<!-- REQUIRED -->
## 4. 算子样例探索

> **来源**: 探索方向 2 — 基于 `PRO_MATERIAL_INDEX.md` §B（官方指定算子）
> **注意**: 官方指定算子为精选实现参考。a5 目录下其余文件不得参考。

### 4.1 全量样例参考（按 cube/vec 组成分类）

> 全量阅读索引 §B 中所有官方指定算子样例，按 cube/vec 组成分类记录参考价值。

**纯 Cube 样例**（matmul 类）：

| # | 示例路径 | 可复用点 |
|---|----------|----------|
| {n} | `{path}` | {cube tile 管理 / K 累加链 / matmul_acc module / L1/L0 布局 / set_mm_layout_transform} |

**纯 Vec 样例**（elementwise 类、vf_api 类）：

| # | 示例路径 | 可复用点 |
|---|----------|----------|
| {n} | `{path}` | {vf 指令组合 / make_tile_group 双缓冲 / auto_mutex / load_align/store_align} |

**VC 融合样例**（FA 类、lightning_indexer 类）：

| # | 示例路径 | 可复用点 |
|---|----------|----------|
| {n} | `{path}` | {section_cube→acc_to_vec→section_vector 衔接 / cross_core 流水 / 多 Module 分块} |

> **选参考原则**：优先从当前算子对应分类中提取直接可复用模式；同时从所有样例中提取通用写法参考（API 用法、分核策略、同步事件管理、尾块处理等）。

### 4.2 可复用模式

**直接可复用**（来自当前算子对应分类的样例，标注来源）：
- API 调用链**：{api_usage}（来源：`{sample_path}`）
- **Tile 配置**：{tile_config}（来源：`{sample_path}`）
- **同步策略**：{sync_pattern}（来源：`{sample_path}`）
- **循环结构**：{loop_pattern}（来源：`{sample_path}`）

**通用写法参考**（来自所有样例）：
- **API 用法**：{如 vf.gather/vf.astype/vf.reduce_* 的参数语义与调用方式}
- **分核策略**：{分核覆盖/重叠/空转检查}
- **同步事件管理**：{event_id 分配/隔离}
- **尾块处理**：{valid_shape/compact/fillpad 用法}

### 4.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| {diff} | {example} | {need} | {suggestion} |

### 4.4 高参考价值样例推荐

> 分析后对当前算子更具参考价值的样例清单，仅作推荐不否定其余样例。

| 样例路径 | 推荐理由 |
|----------|----------|
| `{path}` | {为何对当前算子参考价值高} |

---

<!-- REQUIRED -->
## 5. 教程与设计指南探索

> **来源**: 探索方向 3 — 基于 `PRO_MATERIAL_INDEX.md` §C

### 5.1 适用的设计模式

<!-- 遍历 PRO_MATERIAL_INDEX.md §C 中索引的全部教程文档（以索引实际扫描结果为准），逐行填写 -->

| 指南/教程文档（索引 §C） | 来源目录 | 设计模式 | 适用性 |
|---------------------------|----------|----------|--------|
| `{guide_path}` | guide | {模式} | {对本算子的指导意义；不相关时写“不适用”} |

### 5.2 来自教程的关键约束与建议

| 来源 | 约束/建议 | 影响 |
|------|----------|------|
| {guide} | {constraint} | {impact} |

---

<!-- REQUIRED -->
## 6. Stage 3 设计事实输入

> **综合来源**: §3 API 约束 + §4 样例参考 + §5 教程指导。本节是 Stage 3 的消费入口，只汇总事实与待裁定项，
> 不冻结 tile、Module、同步事件或 topology；这些由 Stage 3 决定。

### 6.1 Tile / 同步约束证据

| 设计问题 | 已证实约束 | 证据路径 | Stage 3 待裁定项 |
|----------|------------|----------|------------------|
| {tile_or_sync_question} | {documented_constraint} | {path + section} | {decision_needed} |

---

<!-- REQUIRED -->
## 7. 环境常量快照

> **来源**: 探索方向 1（API 文档约束）+ 探索方向 2（官方指定算子 assert）
> 从当前仓库的 API 文档和官方指定算子中提取硬件/版本相关常量。下游 Stage 3/4 全部引用本节，不再各自写死。
> 禁止预填历史参考值。每个值都必须来自本次文档遍历或官方样例扫描；无法确认时写 `unknown` 并列入风险。

| 常量 | 探测值 | 来源路径 | 备注 |
|------|--------|----------|------|
| UB 容量 | {本次探测值/unknown} | {目标版本文档或官方样例路径} | {target/version/适用条件} |
| cross_core event_id 上限 | {本次探测值/unknown} | {目标版本 API 文档路径} | {取值范围} |
| 地址对齐要求 | {本次探测值/unknown} | {目标版本 API 文档路径} | {MemorySpace/适用条件} |
| Cube tile 对齐 | {本次探测值/unknown} | {目标版本教学/文档路径} | {tile 尺寸对齐要求} |

---

<!-- REQUIRED -->
## 8. 风险评估

### 8.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| {issue} | {reason} | {suggestion} |

### 8.2 注意事项

| 注意点 | 说明 |
|--------|------|
| {warning} | {desc} |

---

<!-- REQUIRED -->
## 9. 证据索引

> **全量资料索引**: `custom/{operator_name}/PRO_MATERIAL_INDEX.md`

### 9.1 API 文档证据

| 信息 | 路径 |
|------|------|
| {api} 文档 | `{path}` |

### 9.2 算子样例证据

| 信息 | 路径 |
|------|------|
| {样例} | `{path}` |

### 9.3 指南与教程文档证据

| 信息 | 来源目录 | 路径 |
|------|----------|------|
| {指南/教程} | guide | `{path}` |

---

<!-- REQUIRED -->
## 10. 结论

- **可行性**: {可行 / 需调整 / 不可行}
- **主要问题**: {main_issue}
