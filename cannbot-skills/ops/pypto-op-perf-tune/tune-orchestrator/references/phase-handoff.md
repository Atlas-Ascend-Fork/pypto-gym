# 阶段交接摘要（PHASE_SUMMARY_F / PHASE_SUMMARY_S / PHASE_SUMMARY_I）

以下是 PHASE_SUMMARY 状态的执行规范。每个 PHASE 退出时进入对应的 PHASE_SUMMARY，⛔ 强制按以下顺序执行，禁止跳过任何一步。每个阶段的迭代上下文通过 Task subagent 隔离为结构化交接文档，确保后续阶段以干净上下文继续。

## ⛔ 阶段交接流程（强制）

PHASE 执行顺序完全确定，主 agent 只需按序列依次启动 Task subagent，每次检查达标即可：

```
固定序列：
FRONTEND → SWIMLANE_1 → INCORE_1 → SWIMLANE_2 → INCORE_2 → SWIMLANE_3 → INCORE_3
                                                                         ↓
                                                                    全部完成 → S5_REPORT
任意阶段达标 → 提前退出 → S5_REPORT
```

主 agent 的执行逻辑：

```
PHASE_QUEUE = [SWIMLANE_1, INCORE_1, SWIMLANE_2, INCORE_2, SWIMLANE_3, INCORE_3]

FRONTEND 在主会话中执行（复用 S2+S3 的性能数据）。
FRONTEND 完成后：

for phase in PHASE_QUEUE:
    1. 生成阶段交接摘要
    2. 校验摘要完整性
    3. 启动 Task(phase)，subagent_type="pypto-op-optimizer"
       ⛔ 每个 Task 只执行一个 PHASE
    4. Task 返回后，检查摘要中「达标状态」：
       ✅已达标 → break（跳出循环，进入 S5_REPORT）
       ❌未达标 → continue（启动下一个 PHASE 的 Task）

循环结束（达标提前退出 或 队列耗尽）→ S5_REPORT
```

**步骤细化**：

```
每个 PHASE 的交接：
    ↓
步骤1: 生成阶段交接摘要（按下方模板）
    ↓
步骤2: 校验摘要完整性（按下方校验清单逐项确认）
    ↓
步骤3: 启动 Task subagent
    - 使用 Task 工具，subagent_type 选择 "pypto-op-optimizer"
    - ⛔ 每个 Task 只执行一个 PHASE，禁止跨 PHASE 打包
    - 将阶段交接摘要作为 prompt 的核心上下文传入
    ↓
步骤4: Task 返回后检查达标状态
    - 摘要中达标状态 = ✅已达标 → 结束循环，进入 S5_REPORT
    - 摘要中达标状态 = ❌未达标 → 继续下一个 PHASE（回到步骤1）
```

禁止事项：
❌ 只生成摘要不启动 Task subagent
❌ 跳过摘要直接启动 Task
❌ 不在 Task prompt 中传入完整摘要就继续
❌ 在当前会话中直接执行 SWIMLANE/INCORE（必须通过 Task subagent 隔离）
❌ 将多个 PHASE 打包进同一个 Task
❌ Task 返回后不检查达标状态就继续或跳到 S5

## Task prompt 模板（两种变体）

**变体 A — 继续下一阶段（PHASE_SUMMARY_F → SWIMLANE / PHASE_SUMMARY_S → INCORE / PHASE_SUMMARY_I_{1,2}→SWIMLANE_{n+1}）**：
"""
你是 PyPTO 算子性能调优执行器。请执行以下任务：

1. 加载下一阶段子技能：`[子技能路径]`
2. 读取当前算子代码：`[算子文件路径]`
3. 根据下方阶段交接摘要中的迭代统计，重建 Todo（状态机进度、当前迭代状态、迭代轮次记录）
4. 根据下方阶段交接摘要，从 [下一阶段名] 开始继续调优

## ⛔ 外循环状态（必须遵守）

- 当前外循环轮次: [n] / 3
- 当前执行阶段: [SWIMLANE_n / INCORE_n]
- ⛔ 本 Task 只负责当前这一个 PHASE 的执行和退出
- ⛔ 退出当前 PHASE 后，生成 PHASE_SUMMARY 摘要返回给主 agent，由主 agent 决定路由（继续下一 PHASE / 达标→S5 / 外循环耗尽→S5）
- ⛔ 禁止在 Task 内部自行决定进入下一 PHASE 或进入 S5_REPORT

## 调优执行规则（⛔ 必须遵守）

- ITER 循环：ITER_START(选优化点) → ITER_MODIFY(改1个参数) → ITER_VERIFY(验精度) → ITER_MEASURE(测性能) → ITER_RECORD(更新Todo) → ITER_JUDGE(判退出)
- 精度失败/性能回退 → ITER_ROLLBACK → ITER_RECORD → ITER_JUDGE
- 单参数原则：每次只改一个参数
- 每次回复最多推进 1-2 个子状态
- 退出条件：性能达标 / 连续无提升达阈值(FRONTEND:5,SWIMLANE:8,INCORE:5) + 清单全标记 / 用户说停
- 优化点必须来自调优点清单，禁止凭直觉选题
- 每次 ITER_RECORD 后必须立即更新 Todo
- SWIMLANE/INCORE 入口必须先独立采集性能数据

## 阶段交接摘要
[粘贴完整摘要]

## 关键文件
- 算子代码: [路径]
- 测试命令: [命令]
- 环境变量: [变量列表]

请从 [下一阶段名] 的入口开始执行（SWIMLANE/INCORE 入口需先独立采集性能数据）。
执行完成后，返回 PHASE_SUMMARY 摘要（含性能数据、已采纳/失败优化、迭代记录、达标状态），由主 agent 进行路由决策。
"""

**变体 B — 结束调优（PHASE_SUMMARY_F / PHASE_SUMMARY_S_n / PHASE_SUMMARY_I_n 达标→S5_REPORT）**：
"""
你是 PyPTO 性能调优报告生成器。请执行以下任务：

1. 根据下方阶段交接摘要，生成最终调优报告

## 报告生成步骤

1. 更新 Todo：将所有状态（S1-S5）标记为 ✅
2. 输出最终状态看板（目标达成、优化总结、性能趋势）
3. 填充最终调优报告模板（性能对比、已采纳/已失败优化、最佳配置、调优记录）
4. 保存报告到算子目录下：`{op_name}_tuning_report.md`
5. 还原 debug_options：将 `@pypto.frontend.jit` 中的 `debug_options` 移除

## 阶段交接摘要

[粘贴完整摘要]

## 关键文件

- 算子代码: [路径]
- 测试命令: [命令]
- 环境变量: [变量列表]
"""

## 隔离规则（Task subagent 信息传递）

Task subagent 启动时，遵循以下信息传递规则：

| 信息类别 | 保留策略 | 说明 |
|---------|---------|------|
| 已采纳优化 | ✅ 完整保留 | 包含具体修改和代码位置 |
| 已失败优化 | ✅ 完整保留 | 避免重复尝试 |
| 约束与发现 | ✅ 完整保留 | 后续阶段必须遵守 |
| 当前代码配置 | ✅ 完整保留 | 需基于最新代码继续修改 |
| 性能趋势 | ✅ 浓缩保留 | 只保留趋势摘要，不保留每轮详细日志 |
| 中间调试日志 | ❌ 丢弃 | 如精度失败时的完整报错堆栈 |
| 失败代码完整内容 | ❌ 丢弃 | 已回退的代码不再需要 |
| 冗余性能对比细节 | ❌ 丢弃 | 只保留最终结果 |
| 子技能完整内容 | ❌ 不重新加载 | 下一阶段加载新的子技能 |

## 摘要校验清单

**⛔ 生成摘要后必须逐项确认，确认后立即启动 Task subagent：**

- [ ] 性能数据是否与实际一致？（执行时间、利用率、气泡率）
- [ ] 已采纳优化是否遗漏？（核对 Todo List 中的 ✅ 项）
- [ ] 已失败优化是否遗漏？（核对 Todo List 中的 ❌ 项）
- [ ] 约束与发现是否遗漏？（检查调优过程中的关键发现）
- [ ] 代码配置是否是最新的？（包含所有已采纳的修改）
- [ ] ⛔ Todo 完整性检查：
  - Todo 迭代轮次记录表中，所有已结束的轮次状态列是否均已填写 ✅ 或 ❌？
  - 阶段状态（S1-S5）是否与当前实际进度一致？
  - 当前迭代状态中的"当前轮次""连续无提升""失败累计"是否与轮次记录表一致？
  - ❌ 任何一项不一致 → **禁止进入 Task subagent！必须先修正 Todo**
- [ ] ⛔ 校验通过后，是否已启动 Task subagent？（未启动禁止进入下一阶段）
- [ ] ⛔ INIT 退出门控（INIT→S1 转移时必查）：
  - `run_cases.py` 已创建（`custom/<op>/<op_dir>/run_cases.py`）且 `py_compile` 通过？
  - smoke 验证通过：`run_cases.py --cases <smallest> --compile-only` → OK？
  - 交叉验证通过：`run_cases.py --cases <tuning-case-id> --perf` 的 E2E 数字 == 官方 `draw_swim_lane.py` summary 的 E2E？（一次性确认 dyn_topo 解析正确）
  - `tuning_case_set`（精度集）+ `compile_only_gap_fill`（编译期补漏）已记录到 MEMORY + Todo？
  - ❌ 任一缺失 → **禁止 dispatch S1！必须在 INIT 会话内补完**
- [ ] ⛔ S2 退出门控（S2→S3 转移时必查）：
  - tuning case swimlane 数据存在（`merged_swimlane.json` / `bubble_analysis.log`）——重量级，S3 详细分析用
  - covering set 其余 case 的 light E2E 基线已采集（`run_cases.py --cases <non-tuning> --perf` 输出行）？
  - ❌ tuning case swimlane 缺失 → 禁止用 light E2E 替代——S3 bubble/core 分析依赖完整 swimlane
- [ ] ⛔ S3 退出门控（S3→S4 转移时必查）：
  - tuning case 详细性能报告存在（`performance_analysis_report.md`）？
  - covering set per-case E2E 基线表已写入 MEMORY（tuning case from swimlane summary, 其余 from S2 light runs）→ 作为 S4 ITer 比较基线？
  - ⛔ Tile 站点清单 + A4 议程骨架已生成（机械存在性门控）：`list_tile_boundaries.py <impl> --out _debug/tile_boundaries.txt --emit-agenda _debug/a4_agenda.md` 已运行且两文件存在；报告头部 impl sha256 与当前 impl 一致（impl 后续任何结构/配置变更 → 必须重新生成）。文件缺失或 sha256 过期 → 禁止进入 S4——所有 Rule 4 站点级门控以该清单的站点集合为准，清单缺失即门控空洞（历史事故：脚本未运行 → 自创豁免无人拦截）。
- [ ] ⛔ 达标状态校验（Task 返回后首先检查）：
  - 摘要中"达标状态"为 ✅已达标 → 进入 S5_REPORT（跳过以下完整性校验——达标即止，不强制消除所有潜在精度问题）
  - 摘要中"达标状态"为 ❌未达标 → 继续以下结构变更台账校验
- [ ] ⛔ 结构变更台账校验（仅未达标时执行）：
  - §2 表格中每个已采纳项是否均标注 结构级/配置级？
  - F-1~F-21 全表标记是否完整（无 ⏳ 遗留）？
  - ⛔ 分支验证证据完整性（tile/unroll 类 ITER，措辞无关门控）：台账中每个 tile 取值/unroll_list 变更类已采纳 ITER，必须含 `run_cases.py` 的 `[case N]` 输出行（受影响分支参数 + 验证 case + 逐 case 结果）。缺 → 摘要退回。"已在 tuning case 上精度通过"不满足本门控（见 SKILL.md §5 + §1b「分支验证」）
  - ⛔ 精度集实测完整性（tile/unroll 类 ITER，措辞无关门控）：台账中每个 tile/unroll 类 ITER 是否含 `run_cases.py --cases <affected> --perf` 的 per-case `[case N] E2E XXXus` 行？分歧时是否记录了为回退 case 增设/修改的分支 + 重测数字（⛔ 禁止整体回退）？缺 → 摘要退回（判据是逐 case 数字）
  - 首轮摘要是否包含入口基线既有结构标记清单及 audited 状态？
  - 结构 delta 计算是否正确？delta 非空且未达标 → 是否已按 PHASE_SUMMARY_F 门控路由到 PHASE_FRONTEND 重跑？

## 摘要模板

```markdown
## 📋 阶段交接摘要：[阶段名] → [下一阶段名/完成]

### 0. 执行进度
- 已完成阶段: [FRONTEND, SWIMLANE_1, ...]
- 当前 PHASE: [刚完成的 PHASE 名]
- 队列剩余: [INCORE_1, SWIMLANE_2, ...]（按序列）

### 1. 性能状态
- 基准: [值] us → 当前: [值] us (累计提升 [%]%)
- 本阶段提升: [%]%
- 核心利用率: [%]%
- 气泡率: [%]%
- 调优目标: [目标] us（差距: [%]%）
- 达标状态: ✅已达标 / ❌未达标

### 2. 已采纳优化（⛔ 结构变更台账）
| # | 轮次 | 优化项 | 修改内容 | 变更类型（⛔必填：结构级/配置级） | 收益 | 代码位置 |
|---|------|--------|---------|------|---------|---------|
| 1 | F-1 | BLOCK_SIZE | 128→64 | 配置级 | -13.6% | file:line |

> **⛔ 结构变更台账（FRONTEND 摘要必填，编排器 PHASE_SUMMARY_F 门控依据）**：
> - 每个已采纳未回退优化点标注 `结构级`（循环消除/创建/合并/类型变更/外提改变了最内层循环/M轴重映射）或 `配置级`（tile/cache/pass option）
> - **首轮追加**：入口基线既有结构标记清单（如已应用的 F-19 批量化、已消除的循环等，非本轮采纳但存在于代码中）+ `audited: yes/no` 状态（本轮已重审的标记置 yes）
> - 结构 delta = 本轮结构级采纳数 +（仅首轮）未 audited 的基线结构标记数；delta 非空且未达标 → 编排器触发 FRONTEND 重跑（清除全部 DO-NOT-RETRY + 重跑完整阶段A + F- 全表重查）；delta 为空或已达标 → 正常推进
> - F-1~F-21 全表标记完整性（✅已尝试/❌已失败/❌不适用/⏳待尝试）作为台账附表一并提供
> - **⛔ tile/unroll 类已采纳 ITER 附表（分支验证 + covering set 实测，缺失 = 门控退回）**：每个 tile 取值/unroll_list 变更类已采纳 ITER 一行——受影响分支参数 | 分支验证 case+结果（精度/compile-only）| covering set 逐 case 前后 µs（case_id: before→after）| 分歧处理（无分歧 = "一致改善"；有分歧 = 增设/修改的分支 + 重测数字）

### 3. 已失败优化（⛔ 禁止重试；结构变更重跑轮次中该清单作废，须重新实测）
| # | 轮次 | 尝试项 | 失败原因 | 已回退 |
|---|------|--------|---------|--------|
| 1 | F-2 | unroll_list | 性能回退+8.6% | ✅ |

### 4. 约束与发现
- [约束/发现描述]

### 5. 当前关键代码配置
```python
@pypto.frontend.jit(
    runtime_options={...},
    pass_options={...},
    debug_options={"runtime_debug_mode": 1}
)
BLOCK_SIZE = [当前值]
TileShape配置 = [当前值]
```

### 6. 迭代统计
- 总轮次: N 轮
- 成功: X 轮 (X%)
- 失败: Y 轮 (Y%)
- 性能曲线: [持续下降 / 先降后平 / 波动]

### 7. 下一阶段建议
- 当前瓶颈: [描述]
- 建议方向: [描述]
- 需关注的性能文件: [路径]
```
