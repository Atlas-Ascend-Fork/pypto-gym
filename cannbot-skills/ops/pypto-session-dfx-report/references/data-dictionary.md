# DFX 数据字典与信息源

## 1. 输入信息源

| 信息 | 来源 | 用途 |
|------|------|------|
| 开发/验证状态、端到端耗时 | `<op_dir>/result.json`（`pypto_status`、`verifier_status`、`pypto_duration_sec`、`verifier_duration_sec`、`started_at/finished_at`） | 第 0 章概览 |
| Token 总量 | `result.json` 的 `pypto_token_usage` / `verifier_token_usage` | 第 0 章 |
| 算子任务/公式/输入输出/目标 | `<op_dir>/.skill_validate/*_task_desc.py` | 第 1 章 |
| 各 attempt / 各 subagent 时间与 token | `pypto_sessions/attempt_XX/root_full.json`（脚本提取） | 第 2/3 章 |
| 逐次工具调用记录 | 同上（脚本提取） | 第 5 章 |
| 性能调优结果与过程 | `<custom_dir>/<op>_tuning_report.md` 或 `pypto_sessions/attempt_XX/nodes/*Stage_7_S5_REPORT*.md` 中 S5 摘要 | 第 4 章 |

## 2. dfx_data.json 结构（脚本产物）

```jsonc
{
  "attempt_01": {                        // 键 = attempt 目录名（attempt_01/attempt_02/...）
    "root_info": { "id": "...", "title": "...", "time": {"created": ms, "updated": ms} },
    "nodes": [
      {
        "order": 0,                      // 该 attempt 内 DFS 序号
        "id": "ses_...", "parent_id": null, "title": "...",
        "stage": "Stage 3 架构设计",       // 阶段分类（categorize(title)）
        "msg_count": 30,
        "duration_sec": 737.37,          // 墙钟：首条消息 created → 末条消息 completed（秒）
        "active_sec": 730.08,            // 各消息 created→completed 累加（秒）
        "tokens": { "total":..., "input":..., "output":...,
                    "reasoning":..., "cache_read":..., "cache_write":... },
        "messages": [                    // 消息级（消息内工具另列于 tools[]）
          { "id":"...", "role":"assistant", "agent":"pypto-op-architect",
            "time":{"created":ms,"completed":ms}, "tokens":{...},
            "tools":[ {"tool":"bash","status":"completed","time":{"start":ms,"end":ms},"input":{...}} ] }
        ],
        "exec_records": [                // 事件级扁平记录（可视化主数据）
          // 工具调用
          { "session_id":"...", "agent":"...", "role":"assistant", "tool":"bash",
            "kind":"tool", "status":"completed", "start_ms":ms, "end_ms":ms,
            "duration_sec":1.234, "category":"执行命令", "desc":"..." },
          // 模型推理/思考（parts[].time.start/end）
          { "session_id":"...", "agent":"...", "role":"assistant", "tool":"reasoning",
            "kind":"reasoning", "status":"completed", "start_ms":ms, "end_ms":ms,
            "duration_sec":0.932, "category":"推理/思考", "desc":"模型思考文本" }
        ]
      }
    ]
  }
}
```

### 关键字段语义

- **order**: 每个 attempt 内按 session 树深度优先遍历的顺序号，从 0 开始（root 恒为 0）；= exec_records 文件名的序号。
- **duration_sec（墙钟）**: 该 session 首条消息到末条消息的跨度，含编排器等待/并行调度时间。
- **active_sec（活跃）**: 该 session 各消息 `created→completed` 的累加，近似"模型实际工作"时间。
- **exec_records[].start_ms/end_ms**: 毫秒级时间戳（epoch ms），可视化时间线直接用；事件触发原始时间以 `start_ms` 原文保留（如 `1786269062333`），报告表格独立"原始时间"列输出，便于全文检索定位。
- **exec_records[].kind**: 事件类型，`tool`（工具调用）/ `reasoning`（模型推理/思考）/ `text`（有耗时的文本输出）。**具备具体耗时的事件全量保留**，执行轨迹（泳道图）不留工具调用之间的空档。
- **exec_records[].desc**: 事件**完整说明**（`dfx_data.json` 与报告 5.2 节表格均不截断、不加省略号），按事件补充必要细节：`bash`=命令描述：命令本身；`read/write/edit`=文件路径；`grep/glob`=匹配模式；`task`=子 agent 类型+任务；`webfetch`=URL；`todowrite`=清单项数；`reasoning`=完整模型思考文本（表格内 `esc()` 转义管道符、换行折叠为空格；dashboard 悬停 tooltip 展示层的截断省略号仅发生在页面端，报告本身完整）。
- **stage**: 由 `categorize()` 按 title 关键字归类（Stage 1-7 / 编排/其他）。

## 3. 时间与格式规范

- 触发时间（报告中 exec 表）：`HH:MM:SS.mmm`（毫秒级）。
- 原始时间（报告中 exec 表）：事件触发的 epoch 毫秒时间戳原值（如 `1786269062333`），独立一列，便于全文搜索分析。
- 事件耗时：`{sec:.3f}s`（毫秒级）。
- 聚合耗时（第 2 章表格）：秒保留 1 位小数，或换算分钟。
- 所有数字必须与脚本输出一致，禁止编造；缺失值填 `-` 或 `未知`。

## 4. 阶段分类规则（categorize）

| 命中关键字（title 小写） | 阶段 |
|--------------------------|------|
| `stage 7` | Stage 7 性能调优 |
| `stage 6` | Stage 6 E2E 门禁 |
| `stage 5` / `debug phase` / `verify phase` | Stage 5 实现与验证 |
| `stage 4` | Stage 4 模块接口/脚手架 |
| `stage 3` | Stage 3 架构设计 |
| `stage 2` | Stage 2 数学建模 |
| `stage 1` | Stage 1 规格规划 |
| 其他 | 编排/其他 |
