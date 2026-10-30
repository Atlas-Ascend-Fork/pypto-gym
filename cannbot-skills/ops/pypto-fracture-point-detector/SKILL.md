---
name: pypto-fracture-point-detector
description: 分析 PyPTO session 导出文件（root_full.md），识别 PyPTO 框架或文档不完善导致的断裂点，产出可转化为 Issue 的结构化报告；支持作弊检查、问题复现取证、仓分类与标准 issue 文档产出。当用户在 pypto 相关工作结束后提到"断裂点"、"识别断裂点"、"检测断裂点"、"fracture point"时触发此 skill。支持直接接收 root_full.md 文件路径进行分析，也支持被 benchmark fracture-aggregator 批量调用；三个阶段可独立调用。
---

# PyPTO 断裂点分析与 Issue 产出

分析 PyPTO session 导出文件（`root_full.md`），识别由 PyPTO 框架或文档不完善导致的"断裂点"——即 session 运行过程中的失败、重试、效率低下等问题——并沿三个阶段推进到可提交的 issue 文档：

- **阶段① 断裂点检测**：从 session 中识别断裂点，产出结构化断裂点报告（含作弊检查）。
- **阶段② 复现取证**：对候选问题编写最小复现脚本、实际运行取证、定位根因、核对上游最新状态、划分仓分类。
- **阶段③ Issue 报告产出**：将确认的问题转化为符合 GitCode 规范的 issue 文档。

三个阶段均可独立调用：阶段①接收 `root_full.md` 或当前会话；阶段②可接收本 skill 产出的或外部的断裂点报告单独执行；阶段③可接收已完成的复现目录单独执行。

## 核心概念

- **断裂点**：session 中因 pypto 框架或文档不完善导致的问题点
- **实体**：session 中涉及的 API（如 `pypto.add`）、文件、概念等对象
- **信号**：断裂点的可观察表现（如搜索失败、反复重试、操作报错）
- **置信度**：断裂点判定的可靠程度，低置信度的断裂点会被过滤掉
- **作弊检查**：检查 agent 是否抄袭现成同算子或同族邻近算子实现（含整段搬运与骨架级部分抄袭）的检测步骤；疑似或确认作弊的 session，其断裂点结论可信度受污染
- **复现状态**：问题在复现环境的确认程度，分 ✅ 复现、⚠️ 部分确认、❌ 未复现三档
- **仓分类**：按修复问题需要改动哪个仓库的代码划分修复方（pypto / pypto-gym / benchmark）

共 19 种断裂点类型，分为文档类（D1-D3, D5-D6）、API/框架类（A1-A5）、错误信息类（E1-E4）、行为模式类（C1-C2, C4-C6）。完整定义见 `references/fracture-points.md`。

## 使用场景

**场景 1：接收文件路径直接分析（阶段①）**
```
- 用户指定一个 root_full.md 文件路径进行分析
- 例如："分析这个 session 的断裂点 /path/to/root_full.md"
```

**场景 2：被 benchmark fracture-aggregator 批量调用（阶段①）**
```
- pypto-benchmark-fracture-aggregator 逐算子调用本 skill
- 提供输入：root_full.md 路径
- 提供输出：单算子断裂点报告路径
```

**场景 3：分析当前活跃 session（阶段①，兼容旧模式）**
```
- 无文件参数时，分析当前 LLM 对话历史
- 用户说："识别断裂点"或"生成断裂点报告"
```

**场景 4：对已有断裂点报告执行复现取证（阶段②）**
```
- 用户指定断裂点报告或候选问题清单
- 例如："复现这份报告里的 P0/P1 问题并取证"
```

**场景 5：对已完成的复现目录产出 issue 文档（阶段③）**
```
- 用户指定 issues/issue-XX/ 复现目录
- 例如："把这几个确认的问题写成 issue 文档"
```

## 语言风格

本 skill 的全部产出文档（断裂点报告、复现定位文档、issue 文档、汇总报告）必须遵守 `references/writing-style.md` 的写作规范：术语统一、书面技术语言、单点表述、标题含场景与指向、删除冗余、结构紧凑。证据引用逐字摘录，不改写原文。

---

## 阶段①：断裂点检测

### 步骤 0：确定数据来源

- **有文件路径参数**：读取指定的 `root_full.md` 文件作为分析数据源
- **无参数（当前会话）**：直接回顾当前对话历史作为分析数据源

`root_full.md` 是 PyPTO session 的完整导出，包含 `## User` / `## Assistant` 等标记分隔的完整对话历史，与当前会话分析使用相同的断裂点检测方法。

### 步骤 1：读取 Session 内容

读取 `root_full.md` 文件内容（或回顾当前对话历史），重点关注：
- 工具调用及其返回结果（尤其是错误和空结果）
- 用户消息中的修正、澄清、不满表达
- 搜索/读取操作的频次和模式
- 任务是否最终完成

`root_full.md` 以 `## User` / `## Assistant` 标记分隔各个消息回合。对 Assistant 回合，需关注其中的工具调用块（由 XML 标签或 Markdown 代码块包裹的 tool use）及其结果。

**成功标准**：完成回顾后，应能列出 session 中涉及的所有关键实体和操作。

### 步骤 2：时间锚点与阶段映射

建立 session 时间线，为后续每个断裂点准备"发生时间、对应阶段、解决耗时"三项数据。规则见 `references/timing-stages.md`，要点：

- **时间锚点提取**：优先 `root_full.json`（逐消息 `time.created/completed`），其次 `root_full.md`（`**Created:**` + 各 Assistant 消息 `{duration}s`，需估算并标注），当前会话用 `scripts/session_db.py` 的 `time_created`。
- **阶段映射**：优先 `.orchestrator_state.json` → 其次消息显式提及的 `Stage N` → 再其次按标准产物文件归属（`SPEC.md`→Stage1、`API_REPORT.md`→Stage2、`DESIGN.md`→Stage3、golden/impl→Stage4、`test_*.py`→Stage5、`pypto_impl.py`→Stage6、`README.md`/性能调优→Stage7）。无法判定则标注 `未知`。
- **解决耗时计算**：`resolve_end - first_signal_start`；未解决则用 session `time.updated` 并标注"未解决（截至 session 结束）"。

**成功标准**：完成映射后，时间线中的每条消息都能关联到绝对/相对时间与所属阶段（可含"未知"）。

### 步骤 3：实体识别

从对话历史中识别所有涉及的 pypto 实体。识别模式见 `references/entity-patterns.md`，主要包括：
- **API**：匹配 `pypto.xxx` 模式的 API 调用
- **文件**：涉及的 `.py`、`.md` 等文件路径
- **概念**：tile、tensor、pass、codegen、ub、gm 等框架概念

为每个实体确定复杂度等级（简单/中等/复杂），规则见 `references/entity-complexity.md`。

**成功标准**：完成识别后，应产出实体列表，每个实体标注复杂度等级。

### 步骤 4：信号检测

按实体聚合，逐一检测 10 类信号。每个信号对应一个或多个断裂点类型。检测规则和阈值见 `references/detection-rules.md`。

关键信号包括：
- **搜索失败**：搜索返回空或"未找到" → D1, D5
- **操作失败**：工具调用返回 Error/Exception/Failed → A1, A2, E1, E2
- **重复操作**：同一实体上相同操作出现 ≥3 次（简单实体） → C1
- **过度探索**：搜索/读取次数超过阈值 → C4
- **用户介入**：用户消息中包含修正或手动指导 → C6
- **任务未完成**：session 最终未达到预期目标 → C5

阈值会根据实体复杂度加权调整，复杂实体的阈值更宽松。

**成功标准**：完成检测后，应产出信号列表，每个信号标注对应断裂点类型。

### 步骤 5：断裂点判定

将检测到的信号匹配到 19 种断裂点类型，并评估置信度。

置信度评估基于 4 个条件：
1. 有明确错误信息（Error/Exception/Failed）
2. 多个信号佐证（≥2 个信号触发同一断裂点）
3. 证据完整（能清晰复现问题）
4. 用户明确反馈（表达困惑或不满）

判定规则：
- 满足 ≥2 个条件 → 高置信度 → 输出
- 满足 1 个条件 → 中置信度 → 输出
- 不满足任何条件 → 低置信度 → **剔除**

### 步骤 6：去重与合并

同一实体 + 同一断裂点类型 = 同一个断裂点。合并所有触发的信号和证据片段。

本 skill 每次分析单个 session（一份 `root_full.md` 或当前会话），不作跨 session 合并。多 session（如多次 attempt）的合并由上游 `pypto-benchmark-fracture-aggregator` 负责。

### 步骤 7：二次校验（根因归属判定）

> **目的**：排除因模型自身原因导致的误报，确保断裂点根因确实归属于框架或文档。

对去重后的候选断裂点，**必须实际读取相关文档**后执行检查。具体检查项和判定规则见 `references/detection-rules.md` 中的"二次校验规则"章节。

检查结果处理：
- 根因归属 PyPTO框架/PyPTO文档/PyPTO样例/Agent框架 → 计入断裂点总数，输出到报告正文
- 根因归属模型自身 → 标记为"模型能力"，在报告末尾单独列出，不计入断裂点总数

### 步骤 8：关联标记

相同实体的不同断裂点自动标记为关联。例如 `pypto.reshape` 同时有 D1（文档缺失）和 C1（反复重试），它们互相关联。

### 步骤 9：置信度过滤

剔除所有低置信度的断裂点。只有中/高置信度的断裂点进入最终报告。

### 步骤 10：作弊检查

检查 agent 是否抄袭现成同算子或同族邻近算子实现（其他算子目录、`custom/` 参考实现、历史 attempt 产物、参考仓库、仓内残存的同族邻近算子实现）。完整判定规则见 `references/cheat-detection.md`，要点：

- **外部实现识别**：以 session 头部元数据确定 agent 自身工作区，工作区之外或工作区内标注为参考来源的同算子实现均为外部实现；benchmark 黑名单生成场景下，同族邻近算子实现（如目标 `ds_v32_mla_prolog_quant` 时的 `mla_indexer_prolog_quant`）同样视同外部实现。
- **证据序列**：直接复制（`cp`/`cat >`）、读取全文后整段重写、分段搬运，三类操作序列构成抄袭证据。
- **边界区分**：查阅官方文档、阅读框架源码、参考 API 用法/接口模式/配置项写法、借鉴跨域算法思路后独立实现属于合法参考；整段搬运或照搬同算子/同族算子的算法结构（循环骨架、tile 配置）属于作弊。
- **结论分档**：无嫌疑 / 疑似 / 确认，疑似与确认必须列出完整证据链（session 原文引用）。

结果处理：
- 结论写入报告"作弊检查"小节；疑似或确认时在报告头部添加醒目标注，该 session 的断裂点结论可信度受污染，不直接进入阶段②的自动复现队列。

### 步骤 11：环境信息获取

优先从 `root_full.md` 头部元数据（`##` 块内的 `**Directory:**`、`**Created:**` 等字段）提取环境线索。

若有可执行环境，可尝试以下命令获取补充环境信息，**任何命令失败则对应字段标记为"未知"**，不影响报告生成：

| 信息 | 命令 |
|------|------|
| CANN 版本 | `echo $ASCEND_HOME_PATH \| grep -oP 'cann-\K[\d.]+'` |
| PyPTO Commit | `git merge-base HEAD origin/master 2>/dev/null && git log -1 --format='%H %ci' $(git merge-base HEAD origin/master) \|\| echo "Unknown"` |
| 服务器类型 | `lspci -n -D \| grep '19e5:d80[23]' \| sed 's/.*d80\([23]\).*/A\1/' 2>/dev/null \|\| echo "Unknown"` |
| Python 版本 | `python --version 2>/dev/null \|\| echo "Unknown"` |
| 操作系统 | `cat /etc/os-release 2>/dev/null \| grep PRETTY_NAME \|\| echo "Unknown"` |

### 步骤 12：报告生成

根据检测结果生成报告：

**有断裂点的情况**：
- 使用 `templates/report-template.md` 中的模板生成报告
- **被上游 benchmark 流程调用时**：输出到上游指定的路径（如 `{report_path}/fracture-points/{level}_{OpName}_attempt_{N}.md`）
- **独立使用时**：文件名 `fracture-point-YYYY-MM-DD-HHMMSS.md`，保存到当前工作目录
- 断裂点按优先级排序：致命 > 高 > 中
- 每个断裂点包含：类型、优先级、根因归属、Issue 建议、证据片段、优化建议
- 每个断裂点必须包含步骤 2 提取的三项时间/阶段字段：**发生时间**、**对应阶段**、**解决/绕过耗时**（字段格式见 `references/timing-stages.md`）
- Issue 类型映射和标题格式见 `references/issue-mapping.md`（阶段①建议性标注）与 `references/issue-report.md`（阶段③正式生成）
- 每个断裂点标注"聚合分类"（Type-1/Type-2/Type-3）和"分类依据"，分类口径以 `templates/report-template.md` 头部的内联定义为准（Type-1=pypto 框架/代码/文档/流程问题；Type-2=benchmark 运行器问题；Type-3=硬件/外部因素），禁止自创分类维度；"分类依据"表明需要 pypto/pypto-gym 侧修复的必须标 Type-1
- 无论是否有断裂点，报告均包含"作弊检查"小节

**无断裂点的情况**：
- 默认不生成报告文件，在屏幕直接输出摘要：session 基本信息、操作统计、涉及实体列表、作弊检查结论、"未检测到断裂点"提示
- 当上游流程要求产出文件（例如 benchmark 批量分析）时，仍需生成报告文件，断裂点总数为 0，并注明"未检测到断裂点"

---

## 阶段②：复现取证

对阶段①产出的候选问题（或外部断裂点报告指定的问题）逐一执行复现取证。完整操作规范见 `references/reproduction.md`。

### 步骤 R1：候选确认与目录准备

- 确定复现对象清单（如 P0/P1 全选），为每个问题创建 `issues/issue-XX/` 目录。
- 复现所需的历史资产（探针脚本、数据文件）拷贝到 issue 目录下使用，禁止改动原运行目录。

### 步骤 R2：环境准备

```bash
source ${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/set_env.sh
export TILE_FWK_DEVICE_ID=<空闲设备号>
```

采集环境快照存为 `env_info.log`。设备被占用报 `F71008` 时换设备或稍后重试，直至取得无干扰结果。

### 步骤 R3：编写最小复现脚本并运行

- 脚本最小化、可重复执行、产出明确的成功/失败信号；条件允许时带对照组。
- 文档对照型问题用 grep / find / 产物比对等静态核查命令序列代替运行脚本。
- 每次运行完整落盘日志（`run_<场景>.log`），静态核查记录存 `evidence_<场景>.log`。

### 步骤 R4：复现状态分档

按三档定性：✅ 复现 / ⚠️ 部分确认 / ❌ 未复现。部分确认逐子现象标注；未复现必须记录环境差异（pypto 构建版本、硬件型号、host 侧或设备侧行为）作为归因线索。

**报错点级台账（强制）**：判定粒度是"报错点/子现象"，不是"issue"。凡含多个子现象的问题，必须为每个子现象建立一行台账记录（现象 / 复现脚本 / 实测错误码与关键消息 / 判定 / 证据 log 路径），写入 `issues/issue-XX/README.md` 的台账表（结构见 `templates/repro-readme-template.md`）。issue 级状态由台账汇总得出，不允许绕过台账直接给 issue 级结论。

**实跑保鲜（强制）**：报错行为类问题（声称"触发某错误码/某消息"）的实跑取证必须在与提单相同的环境快照下完成。当复现环境（CANN/pypto 版本）与 session 发生时不一致，禁止引用 session 时期的报错观察值作为现象描述，必须以当前环境实跑结果为准重新定性——包括把声称的错误码改写为实际触发的错误码，或判定 ❌。

**报错行为类实跑优先（强制）**：报错行为类声称必须在当前环境实跑触发至少一次。静态代码证据只能支持"机制存在"，措辞必须显式标注"静态确认，未实跑触发"，且不计入"已复现"数量。实跑触发不出目标报错时，降级为 ⚠️（写明实际报错码）或 ❌。静态核查的合法范围仅限文档对照型问题（规则细节见 `references/reproduction.md`）。

**报错信息类问题（E1-E4 及"日志/报错是否提供足够信息"类）的输出验证**：禁止用 grep 关键字匹配判定日志是否提供了报错信息和解决方案——这是语义判断，关键字判定大概率误判。取得日志后必须由 AI 完整阅读日志原文，按 `references/reproduction.md` 的"报错信息类问题的日志判定"rubric 逐项判定（根因可读描述 / 用户源码位置 / 修复建议），每项给出日志原文引证与理由。关键字 grep 仅用于证据可回查性自检。

### 步骤 R5：上游核对

凡涉及"文档缺失""报错看不懂""示例无效"的问题，下结论前核对上游最新状态（pypto 仓最新 commit + benchmark 下载的文档快照，`sync_devkit.py` 机制，缓存在 `$PYPTO_DEVKIT_DIR` 或 `/root/.cache/pypto-devkit/`），明确归入三种结局之一：① 上游已解决 / ② 上游至今没写 / ③ 内容有了但索引失效。原定性被推翻时显式记录修正过程。

### 步骤 R6：根因定位文档

按 `templates/repro-readme-template.md` 撰写 `issues/issue-XX/README.md`：根因定位到 `path:line`，证据用日志与文档原文，与历史断言逐条对照。

### 步骤 R7：仓分类

按"修复该问题需要改动哪个仓库的代码"划分修复方：pypto（框架本体）/ pypto-gym（agent 工具链）/ benchmark（测试运行器），允许"X 为主 + Y 附带"。判定规则见 `references/repo-classification.md`。

### 步骤 R8：汇总报告

产出汇总报告：总表（编号、问题指向、仓分类、复现状态、提单建议、复现目录链接）、逐问题说明、仓分类统计、过程备注。汇总报告引用各 issue 目录的相对路径，不重复粘贴证据原文。多子现象问题的总表状态必须能回溯到该问题 README 中的报错点级台账。

**推翻即回写（强制）**：任何复现判定变更（尤其复审推翻）必须在同一步骤内更新全部下游产物——issue 目录 README、issue 草稿、汇总报告、索引文件、已生成或已提交的提单文本——并附传播清单逐项打勾写入过程备注。仅改汇总不改提单文本视为流程未完成。

---

## 阶段③：Issue 报告产出

将复现确认的问题转化为可直接提交 GitCode 的 issue 文档。完整规则见 `references/issue-report.md`，模板见 `templates/issue-report-template.md`。

### 步骤 I1：类型映射与标题

- 按断裂点类型映射 issue 类型：报错/崩溃/精度类 → Bug Report；文档缺失/错误 → Documentation；能力扩展与改进建议 → Requirement。C 类断裂点映射到其根因对应的类型。
- 标题格式 `[英文类型|中文类型]: 具体描述`，仅允许 5 种前缀（Bug-Report|缺陷反馈 / Requirement|需求建议 / Documentation|文档反馈 / Question|问题咨询 / Task|任务跟踪），具体描述含场景与指向。

### 步骤 I2：字段填充与文档生成

- Bug Report 必填：问题描述、环境信息、重现步骤、预期结果、日志。复现阶段的复现脚本直接填入"重现步骤"，运行日志原文填入"日志"字段。
- 输出 `issues/issue-XX/issue.md`；❌ 未复现的问题不生成 issue 文档。
- **报错点级门禁（强制）**：issue 文本中的每个报错点/子现象必须能映射到 R4 台账中一行非 ❌ 记录；❌ 子现象不得出现在 issue 文本的现象清单中（可在备注中说明"已排除及原因"）。仅有静态证据的报错点必须逐点标注"静态确认，未实跑触发"，且不计入"已复现"数量；复现状态为 ⚠️ 的问题必须如实写明各子现象的实测差异（含实际触发的错误码）与归因线索。
- **无脚本不写入（强制）**：issue 文本中每个报错点必须随包附带可执行复现脚本或静态核查命令序列；确实无法构造的，显式标注"session 观察，未独立复现"，并从现象计数与"复现状态"口径中剔除。
- issue 文档只写提交所需内容，复现过程细节保留在 `issues/issue-XX/README.md` 并引用其相对路径。

### 步骤 I2.5：提单前一致性关卡（强制）

issue 文档生成后、提交前，必须执行三方一致性比对并记录结果：

1. **逐点比对**：issue 文本 ↔ 复现 README 台账 ↔ 汇总报告/索引，每个报错点的复现状态措辞必须一致（如 README 自认不可复现而索引标 ✅，禁止提单）。
2. **推翻项清零**：对复审推翻或范围修正中移除的报错点，用其特征词（错误码、现象关键词）对最终提单文本做 grep 自检，必须零命中。
3. 比对与自检结果写入汇总报告过程备注；任一不通过，修正后重新过本关卡。

### 步骤 I3：汇总报告更新

在汇总报告总表中补充各问题的 issue 文档路径与提单建议，形成"断裂点报告 → 复现取证 → issue 文档"的完整链路。

---

## 重要提醒

- 证据片段必须是 session/root_full.md 或复现日志中的原文引用，不要编造或概括
- 每个断裂点的"问题描述"应清晰到能让不了解 session 上下文的人理解发生了什么
- 优化建议要具体、可操作，指向具体的文件或 API
- Session 级断裂点（C5、C6）不针对特定实体，单独列为一个章节
- 分析 `root_full.md` 时，注意识别其特有的消息分隔格式（`## User` / `## Assistant`），从中提取对话历史
- **二次校验**：必须执行步骤 7 的两项检查，验证根因是否确实归属于框架或文档。断裂点分类包含：PyPTO框架、PyPTO文档、PyPTO样例、Agent框架、模型能力。前四个是问题，模型能力是非问题（不应计入断裂点总数）。
- **作弊检查**：疑似或确认作弊的 session 必须在报告头部醒目标注，其断裂点不直接进入阶段②的自动复现队列。
- **benchmark 日志分析**：当本 skill 被 benchmark 汇总流程调用时，每个断裂点必须额外标注"聚合分类"（Type-1/Type-2/Type-3）和"分类依据"，供综合报告过滤统计使用。
- **时间数据**：发生时间/解决耗时必须取自日志或 DB 中的真实时间戳；无法获取精确值时用 md 时长累加估算并标注 `（估算）`，禁止编造时间。
- **上游核对优先**：阶段②中凡涉及"文档缺失/报错看不懂"的结论，必须先完成上游核对再定性，禁止仅凭 session 发生时的状态下结论。
- **只取证不修复**：阶段②只做复现取证与修复建议，不修改任何原始仓库与原始运行目录的代码。
- **日志质量 AI 语义判定**：报错信息类问题（E1-E4 等）的"是否提供报错信息与解决方案"判定必须由 AI 阅读日志原文按 rubric 语义判定，禁止用关键字匹配定性（规则见 `references/reproduction.md` 的"报错信息类问题的日志判定"）。
- **报错点级判定与提单门禁**：复现判定、提单门禁的粒度均为单个报错点/子现象，不是整个 issue；❌ 报错点不得进入提单文本，静态确认必须标注且不计入已复现，无脚本报错点不写入（规则见 R4、I2、I2.5）。
- **提单文档面向陌生读者**：issue 文档与复现定位文档的读者是不了解 session 上下文的仓维护者，必须遵守 `references/writing-style.md` 第 7 条：结论先行、内部标识首次出现必须翻译、证据逐行导读、复现步骤可整段照做、定性结论给理由。

## 参考文档

### 阶段①

- **断裂点定义**: `references/fracture-points.md`
- **检测规则**: `references/detection-rules.md`
- **实体识别**: `references/entity-patterns.md`
- **时间锚点与阶段映射**: `references/timing-stages.md`
- **作弊检查规则**: `references/cheat-detection.md`
- **Issue 映射（阶段①建议性标注）**: `references/issue-mapping.md`

### 阶段②

- **复现取证操作规范**: `references/reproduction.md`
- **仓分类规则**: `references/repo-classification.md`

### 阶段③

- **Issue 报告生成规则**: `references/issue-report.md`

### 通用

- **写作规范**: `references/writing-style.md`
- **断裂点报告模板**: `templates/report-template.md`
- **复现定位文档模板**: `templates/repro-readme-template.md`
- **Issue 报告模板**: `templates/issue-report-template.md`
