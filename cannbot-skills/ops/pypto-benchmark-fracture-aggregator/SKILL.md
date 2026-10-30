---
name: pypto-benchmark-fracture-aggregator
description: >
  PyPTO Benchmark 断裂点综合分析 Skill — 对一次 benchmark 运行产出的全部算子日志，
  逐文件调用断裂点检测器生成单算子断裂点报告，再汇总产出全局综合分析报告。
  触发场景：(1) "分析这次 benchmark 的断裂点" (2) "生成断裂点综合报告"
  (3) "对这批算子做 fracture 分析" (4) 任何提到 batch/full fracture analysis 的需求。
---

# PyPTO Benchmark 断裂点综合分析

## 核心流程

- 逐算子分析 → 单算子断裂点报告 → 汇总全局综合报告
- 分析基于 `pypto-fracture-point-detector` skill 方法论，所有证据必须引用日志原文

---

## 前置步骤：获取用户输入

启动后 **必须** 向用户确认 report 路径：

```
请提供本次 benchmark 运行的 report 目录路径：
（例如 pypto-gym/benchmark_runs/Task_xxx/report）
```

该路径下预期结构：
```
report/
├── level1/
│   ├── {OpName}/
│   │   ├── pypto_run.log
│   │   ├── result.json
│   │   ├── verifier_session.md
│   │   ├── ...
│   │   └── pypto_sessions/
│   │       ├── attempt_01/
│   │       │   ├── nodes/
│   │       │   ├── root_full.md        ← 主要分析目标
│   │       │   ├── root_full.json
│   │       │   └── ...
│   │       └── attempt_02/             (可选，若存在第二次尝试)
│   │           └── root_full.md
│   └── ...
├── level2/
│   ├── Gemm_GroupNorm_Hardtanh/
│   │   └── pypto_sessions/
│   │       ├── attempt_01/
│   │       │   └── root_full.md
│   │       └── ...
│   └── ...
├── level3/
│   ├── MinGPTCausalAttention/
│   │   └── pypto_sessions/
│   │       ├── attempt_01/
│   │       │   └── root_full.md
│   │       └── ...
│   └── ...
├── level4/
│   └── ...
├── pto_case/
│   └── ...
├── summary.json       (可选，用于获取 meta 信息)
└── summary.md         (可选)
```

---

## Phase 1: 算子发现

### Step 1.1：扫描算子目录

扫描 `report_path` 下除汇总产物目录外的各 level 子目录，列出所有算子目录。

```
const ignored = new Set(["fracture-points", "logs"])
const levels = direct child directories under report_path excluding ignored
for each level:
  for each operator_dir in report_path/level/:
    record: { level, operator_name, path }
```

### Step 1.2：发现日志文件

对每个算子目录，检测 `pypto_sessions/` 下的 attempt 子目录及其中 `root_full.md`：

| 路径 | 说明 | 对应 attempt |
|------|------|-------------|
| `pypto_sessions/attempt_01/root_full.md` | 首次运行日志 | attempt_01 |
| `pypto_sessions/attempt_02/root_full.md` | 第二次运行日志（若存在） | attempt_02 |

只处理存在的 `root_full.md`。无 `pypto_sessions/` 目录或无任何 `attempt_*/` 子目录的算子：跳过该算子，在综合报告中标注"该算子无运行日志（可能为成功跳过或未执行）"。

### Step 1.3：汇总待分析清单

向用户展示待分析清单并确认：

```
发现 {N} 个算子，共 {M} 份日志文件：

| 等级 | 算子名称 | attempt_01 | attempt_02 |
|------|----------|----------|----------|
| level2 | Gemm_GroupNorm_Hardtanh | ✓ | ✓ |
| level2 | Matmul_Sigmoid_Sum | ✓ | - |
| ...

是否继续分析？(yes/no)
```

---

## Phase 2: 逐算子断裂点分析

### Step 2.1：创建工作目录

```
mkdir -p {report_path}/fracture-points
```

### Step 2.2：对每份日志执行断裂点检测

对清单中的每份日志文件，使用 task subagent 加载断裂点检测器 skill 进行分析。多份日志相互独立，**可并行启动多个 task subagent** 以加速批量分析。

**subagent 指令要点**：

- 加载 `pypto-fracture-point-detector` skill，严格遵循其方法论分析日志文件
- 输出单算子报告到 `{report_path}/fracture-points/{level}_{OpName}_attempt_{N}.md`

**每个 subagent 的要求**：

- 报告包含：摘要、环境信息、优先修复列表、Session级断裂点、实体级断裂点详情
- 每个断裂点必须包含：聚合分类（Type-1/Type-2/Type-3）和分类依据
- 所有证据片段必须引用日志原文，禁止编造
- 完成后返回简要摘要（算子名、断裂点数量、关键发现）

**边界情况**：

- **日志中无明显断裂点**：仍需生成报告，断裂点总数为 0，报告中注明
- **日志文件过大**：分段读取，先读头部和尾部定位关键信息，再针对性读取中间部分；优先关注异常/错误/超时相关行

---

## Phase 3: 综合报告生成

所有单算子报告生成完毕后，执行综合汇总。

> **强制筛选规则 — 贯穿 Phase 3 所有统计和输出**：
>
> | 分类 | 说明 | 处理 |
> |------|------|------|
> | **Type-1** | pypto 算子开发流程中 agent 碰到的框架/代码/文档/流程问题，与 pypto agent 建设直接相关（环境类问题按修复责任归入 Type-2 或 Type-3，不单列为 Type-1） | **保留，计入所有统计** |
> | **Type-2** | benchmark 框架配置、调度、日志记录逻辑、超时策略等非 pypto 产出的问题 | **剔除，不计入任何统计** |
> | **Type-3** | 硬件故障、集群问题、用户操作中断等外部因素 | **剔除，不计入任何统计** |
>
> Phase 3 产出的所有数字（断裂点总数、严重度分布、根因归属分布、Top 10 排行、建议矩阵、算子明细表）**必须仅包含 Type-1 断裂点**。分算子报告中若混有 Type-2/Type-3，在汇总阶段一律过滤剔除。

### Step 3.1：读取全部单算子报告

读取 `{report_path}/fracture-points/` 下所有刚生成的 `*_attempt*.md` 报告，提取：
- 每个报告的元数据（算子名、等级、尝试、断裂点数量、优先级分布）
- 每个断裂点的详细信息（类型、优先级、根因归属、实体、描述、证据、聚合分类、分类依据）

若某个断裂点缺少聚合分类，必须回到对应单算子报告和原始日志补判分类；无法补判时默认不计入 Type-1 统计，并在综合报告的"过滤说明"中列出。

**分类一致性校验（必做）**：逐个核对断裂点的"聚合分类"与"分类依据"是否矛盾。判定口径以 detector 报告模板内联定义为准（Type-1=pypto 框架/代码/文档/流程问题；Type-2=benchmark 运行器配置/调度/日志/超时策略；Type-3=硬件故障/外部中断）。典型矛盾信号：

- 分类依据写着"需要 PyPTO 框架侧修复""pypto-gym 工具链缺陷"等，却标了 Type-2/Type-3 → 以依据为准回判为 Type-1；
- 分类使用了定义之外的自造维度（如"直接阻断/效率损失/信息缺失"）→ 按定义重判；
- 同一问题在不同 session 的报告中分类不一 → 统一回判为同一分类，取符合定义者。

回判修正必须显式记录：在综合报告"过滤说明"中列出"分类修正记录"（断裂点编号、原分类 → 新分类、依据），禁止静默改判。修正后的分类才进入后续统计。

### Step 3.2：交叉归并共性断裂点

按根因类型（而非表面错误码）将各报告中的断裂点归并。判断两个断裂点"同根"的标准：

1. **同一实体** + **同一断裂点类型** → 同根
2. **不同算子** + **相同错误码/错误模式** + **相同根因归属** → 同根
3. **不同算子** + **相同阻塞机制**（如都是"子 agent 无时间预算"）→ 同根

归并后得到 **跨算子共性断裂点列表**，每个包含：
- 影响算子数和报告数
- 根因归属
- 置信度（多算子交叉验证 → 高置信度）

### Step 3.3：统计与分布分析

| 统计项 | 说明 |
|--------|------|
| 总算子数 / 总报告数 | 按 level 分别统计 |
| 算子开发结果分布 | pypto_failed / successfully_completed / verify_failed 等的数量和列表 |
| 每报告平均断裂点数 | 按 level 和 attempt 分别统计 |
| 断裂点严重度分布 | 致命/高/中的数量和占比 |
| 根因归属分布 | 框架/文档/两者/模型能力 的数量和占比 |
| 断裂点类型排行 | Top 10 断裂点类型及其影响范围 |

### Step 3.4：抽取共性模式并泛化为深度分析

对每个跨算子共性断裂点，根据本轮实际数据进行**具象化深度分析**（不照搬僵化模板），产出：

1. **受影响算子**（具体列表）及各自的影响程度（致命/高/中）
2. **触发机制分析**：为什么出现、不同算子中表现形式的异同、是否有算子成功规避
3. **对 Agent 的连锁影响**：如何影响后续决策、是否引发其他断裂点（如重试循环 → 时间耗尽）
4. **证据整合**：从各算子报告摘取日志片段，交叉验证是否指向同一根因

### Step 3.5：生成综合报告

输出到 `{report_path}/fracture-points/benchmark_fracture_report.md`。

报告应包含以下章节，按实际数据灵活组织：

1. **前置说明**：报告范围、筛选规则（重申上述 Type-1 强制规则）、总算子数/总报告数/总断裂点数概览
2. **全局总览**：算子开发结果分布、按等级分布、Type-1 断裂点严重度分布（致命/高/中）、按根因归属分布（PyPTO 框架/两者/模型能力/PyPTO 文档）
3. **跨算子共性断裂点深度分析**：按影响面排序，每个共性断裂点包含——影响算子列表、根因归属与置信度、普遍现象、详细触发机制、对 Agent 的阻断/连锁影响、典型日志证据片段
4. **断裂点类型统计**：Top 10 断裂点类型（数量、占比、说明）
5. **建议优先级矩阵**：P0 致命（直接导致 ≥3 算子失败）、P1 高（显著降低开发成本）、P2 中（改善体验），每项含预期收益与建议方案
6. **各算子断裂点明细表**：算子名、级别、最终状态、总断裂点数、致命/高/中分布、主要断裂类型
7. **总结**：最致命问题提炼、预期改进收益

> 数据一致性要求：综合报告中所有统计数字必须与各分算子报告严格一致；根因归属无法判定时以分算子报告中的 `根因归属` 字段为准。

### Step 3.6：生成报告索引

输出到 `{report_path}/fracture-points/README.md`，包含：
- 综合报告链接
- 单算子报告列表
- 总算子数、日志文件数、Type-1 断裂点总数
- 过滤掉的 Type-2/Type-3 数量和缺少分类的报告列表

---

## Phase 4: 校验与交付

### Step 4.1：完整性校验

确认以下各项：

- [ ] 每个有日志的算子都有对应的断裂点报告
- [ ] 综合报告中的统计数字与单算子报告一致
- [ ] **所有统计仅包含 Type-1 断裂点**，Type-2/Type-3 已正确剔除
- [ ] **分类一致性校验已执行**：分类依据与聚合分类矛盾的断裂点已回判修正，修正记录写入综合报告"过滤说明"
- [ ] 所有证据片段均为日志原文引用，无编造
- [ ] 所有断裂点的根因归属已判定
- [ ] 模型能力类断裂点已单独列出，不计入总数
- [ ] 报告文件命名符合规范

### Step 4.2：汇总产物清单

```
{report_path}/fracture-points/
├── README.md                              # 报告索引与快速概览
├── benchmark_fracture_report.md           # 全局综合报告（主产物）
├── {level}_{OpName}_attempt_01.md           # 单算子报告 × N
├── {level}_{OpName}_attempt_02.md
└── ...
```

### Step 4.3：输出摘要

向用户输出最终摘要：

```
断裂点综合分析完成。

  - 算子数: {total_operators}
  - 日志文件数: {total_logs}
  - Type-1 断裂点总数: {total_t1_fp}
  - 致命: {critical_count} / 高: {high_count} / 中: {medium_count}

产物:
  - 综合报告: {report_path}/fracture-points/benchmark_fracture_report.md
  - 单算子报告: {report_path}/fracture-points/ (共 {fp_report_count} 份)
```

---

## 参考文档

- **断裂点定义**: `pypto-fracture-point-detector/references/fracture-points.md`
- **检测规则**: `pypto-fracture-point-detector/references/detection-rules.md`
- **实体识别**: `pypto-fracture-point-detector/references/entity-patterns.md`
- **Issue 映射**: `pypto-fracture-point-detector/references/issue-mapping.md`
