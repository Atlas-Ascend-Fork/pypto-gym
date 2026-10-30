---
name: pypto-session-dfx-report
description: >
  从 PyPTO benchmark 的 session 记录（root_full.json）生成算子级 DFX 可观测性分析报告。
  输出：算子信息、端到端耗时与各阶段/agent 耗时、token 消耗、性能调优情况、逐次执行轨迹
  （工具调用 + 推理/思考等全部具备耗时的事件，毫秒级，含原始 epoch 时间戳），并为可视化网页
  准备结构化数据。触发词：DFX 报告、session 分析报告、可观测性分析、生成 dfx 报告、
  分析这次 session 的耗时/token。
---

# PyPTO Session DFX 报告生成

将一次 PyPTO 算子开发的 session 记录转化为结构化 DFX 分析报告。**核心原则：一切可确定性计算的数据
由脚本生成，LLM 只负责撰写非确定性章节（算子简介与性能调优），以最小化 token 开销并保证报告格式稳定。**

## 输入

- `op_dir`：benchmark 报告目录下的算子目录，结构：
  ```
  <op_dir>/
  ├── result.json                      # 状态 / 端到端耗时 / token 总量
  ├── .skill_validate/*_task_desc.py   # 算子任务描述（公式/输入输出/目标）
  └── pypto_sessions/
      ├── attempt_01/root_full.json    # 会话导出（含 info/messages/children + time/tokens）
      └── attempt_02/root_full.json    # 每次 attempt 一个子目录（可多个）
  ```

## 流程

### 阶段 0：确认输入

- 确认 `op_dir/pypto_sessions/` 下存在至少一个 `attempt_XX/root_full.json`。
- 缺任一关键文件则停止并说明缺失项。
- `extract_dfx.py` 加载时做结构校验（`info`/`messages`/`children`/`parts` 的类型），非法时输出文件路径与问题字段并返回 1；`info.time`/`created`/`updated` 缺失时相关字段降级为 `-`，不中断。若脚本报 invalid session export，报告该文件与字段，不自行改写输入。
- 可选：verifier session（`verifier_session.md`）用于补充验证阶段说明。

### 阶段 1：脚本化数据提取（token 节约关键）

运行提取脚本（**主 agent 不要直接读 root_full.json**，其体积可达数百 MB）：

```bash
python3 {SKILL_DIR}/scripts/extract_dfx.py <op_dir> [--out <dfx_dir>]
```

脚本自动完成：
- 逐 attempt 遍历 session 树，计算每个节点（root + subagent）的墙钟/活跃耗时与 token（total/input/output/reasoning/cache-read）；
- **阶段自动分类**：按子 agent 标题正则匹配 `Stage 1-7`（兼容 `Stage5` / `stage 5` 等大小写与空格差异），
  并将 optimizer 的 `S1_SETUP`~`S5_REPORT` 统一归入 `Stage 7 性能调优`；无法判定的归入 `编排/其他`；
- 提取全部**具备具体耗时/时间戳的事件**（`start_ms/end_ms` 毫秒级 + `kind` + 类别 + 说明），
  执行轨迹不留空档：
  - **工具调用**（`kind="tool"`）：`执行命令` 的说明自动包含命令描述与命令本身，其余工具含文件路径/模式/URL 等必要细节；**说明不做长度裁剪**；
  - **推理/思考**（`kind="reasoning"`，消息 `parts[].type=="reasoning"` 且带 `time.start/end`）：模型推理时段全量保留，填满泳道图工具调用之间的空档；说明为**完整思考文本**，`dfx_data.json` 与报告 5.2 节表格均**完整保留、不截断**（表格内由 `esc()` 转义管道符、换行折叠为空格）；
  - **文本输出**（`kind="text"`）：仅当 `start < end`（真实耗时 > 0）时保留，瞬时文本输出忽略；
- 每条事件同时记录**原始 epoch 毫秒时间戳**（如 `1786269062333`），报告执行轨迹表中以独立"原始时间"列输出，便于全文检索定位；
- 按 attempt 分别输出：
  - `dfx/dfx_data.json`：结构化全量数据（可视化用，事件级，含 `kind`）
  - `dfx/summary.txt`：紧凑摘要（**LLM 主要阅读对象**，token 极小）
  - `dfx/exec_records/attempt_XX/*.md`：各节点逐事件表（毫秒级 + 原始时间）
  - `dfx/sections/{2_time,3_token,5_exec,6_data}.md`：确定性章节的预渲染 markdown
    （第 2/3 章小节按 attempt 递增编号：`2.1/2.2…`、`3.1/3.2/3.3/3.4…`，跨 attempt 不重复）

之后 LLM **只读** `dfx/summary.txt`（必须），按需读 `result.json`（小文件）与
`*.task_desc.py`（小文件）；不读任何 `root_full.json` 原始数据。

### 阶段 2：LLM 撰写报告正文（第 0/1/4 章）

按模板 `templates/report-body-template.md` 撰写 `dfx/report_body.md`，只含：

- **第 0 章 报告概览**：数据来源 `summary.txt` + `result.json`（状态/耗时/token 与脚本汇总严格一致）；
- **第 1 章 算子简要信息**：任务描述/公式/动态轴/输入输出/性能目标（来自 `*.task_desc.py`）+ 验证结果（`result.json`）；
- **第 4 章 性能调优情况**：调优结果与趋势、S1-S5 过程、已采纳/已失败优化、约束瓶颈。
  ⚠️ **4.4/4.5 表格格式约束**（dashboard 脚本解析依赖）：
  - 章节标题与表格之间必须有空行；
  - 表格第一行表头行、第二行分隔行；
  - **数据行首列必须是纯数字序号（1, 2, 3, …）**，禁止 `#1`、`基准`、`—`；
  - 4.4 列数：`| # | 阶段 | 优化项 | 修改内容 | 收益 | 开始时间 | 结束时间 |`（7 列）；
  - 4.5 列数：`| # | 阶段 | 尝试项 | 失败原因 | 开始时间 | 结束时间 |`（6 列）；
  - **开始/结束时间**为相对整个端到端任务启动时刻（0 点）的时间，精确到分钟，
    可写绝对时间（`2026-10-20 14:53`）、当日时刻（`14:53`）或相对偏移（`+40 min`）；
  - 4.2 趋势块每行建议带 `#N`（对应 4.4 标号）与 `结束 <时间>`，供趋势横轴按结束时间对齐。
  数据来源优先级：`<custom_dir>/<op>_tuning_report.md` → `pypto_sessions/attempt_XX/nodes/*Stage_7_S5_REPORT*.md`。

禁止在正文中编写第 2/3/5/6 章（脚本渲染，会在组装时注入）。

### 阶段 3：组装最终报告

```bash
python3 {SKILL_DIR}/scripts/assemble_report.py <op_dir> [--out <report.md>]
```

组装顺序固定：正文(0,1) → 第2章(时间) → 第3章(token) → 正文(4) → 第5章(执行轨迹) → 第6章(数据准备)。
输出 `<op_dir>/dfx/<op_name>_dfx_report.md`。`sections/` 下四个文件任一缺失时脚本报错并返回 1，不生成残缺报告（先重跑 `extract_dfx.py`）。

### 阶段 4：校验与交付

- [ ] 报告含全部 0-6 章，章节标题与模板一致
- [ ] 第 0/2/3 章数字与 `summary.txt`/脚本输出一致
- [ ] 第 2/3 章小节按 attempt 递增编号（`2.1/2.2…`、`3.1/3.2/3.3/3.4…`），跨 attempt 不重复
- [ ] 第 5 章按 attempt 分离（`5.x 事件/工具调用统计（attempt_XX）` / `5.y 各 subagent 逐次执行轨迹（attempt_XX）`）
- [ ] 执行记录时间精确到毫秒（`HH:MM:SS.mmm` / `0.000s`），且每条事件含**原始时间列**（epoch 毫秒，如 `1786269062333`）
- [ ] 执行轨迹含**全部具备耗时的事件**（工具调用 + 推理/思考 + 有耗时的文本输出），工具调用之间不留长空档；各事件在泳道图中都有对应矩形
- [ ] 执行轨迹说明**完整保留、不截断、无省略号**（`dfx_data.json` 与报告 5.2 节表格均完整）：执行命令含命令描述与命令本身；读写含文件路径；搜索含模式；抓取含 URL；推理/思考含完整思考文本。提示窗内的截断（省略号）仅发生在 dashboard 悬停 tooltip 展示层，不影响本报告
- [ ] 表格管道符已转义（`\|`），无破坏行
- [ ] 每节点文件位于 `dfx/exec_records/attempt_XX/`（按 attempt 分离）
- [ ] `dfx_data.json` 合法 JSON

## 报告章节固定结构（0-6）

| 章 | 标题 | 渲染方 |
|----|------|--------|
| 0 | 报告概览 | LLM（模板） |
| 1 | 算子简要信息 | LLM（模板） |
| 2 | 端到端耗时与各阶段/agent 耗时 | 脚本（sections/2_time.md） |
| 3 | 各阶段/agent/subagent 的 Token 消耗 | 脚本（sections/3_token.md） |
| 4 | 性能调优情况 | LLM（模板） |
| 5 | 执行轨迹 | 脚本（sections/5_exec.md） |
| 6 | 可视化网页数据准备 | 脚本（sections/6_data.md） |

## Token 节约要点

1. **绝不直接读 root_full.json**（数百 MB ~ GB 级）；一切统计交给 `extract_dfx.py`。
2. LLM 只读 `summary.txt`（几 KB）+ `result.json` + `task_desc.py`（小文件）。
3. 第 4 章性能调优只读 S5 节点摘要或 tuning_report，不读全部调优节点原文。
4. 逐次执行记录（可能上千条）由脚本直接渲染进 `sections/5_exec.md`，LLM 不逐一阅读。

## 参考文档

- 数据字典与信息源：`references/data-dictionary.md`
- 正文模板：`templates/report-body-template.md`
