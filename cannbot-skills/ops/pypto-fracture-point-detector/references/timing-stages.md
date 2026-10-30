# 时间锚点与阶段映射

为每个断裂点记录三项时间/阶段信息，用于评估断裂点的成本与修复收益：

- **发生时间**：断裂点首次出现信号的时刻（绝对时间 + 相对 session 起点偏移）
- **对应阶段**：断裂点发生时处于的 pypto-op-orchestrator 工作流阶段（Stage 1-7）
- **解决或绕过耗时**：从首次出现信号到问题被解决/绕过的时长

## 时间锚点提取

### 数据源优先级

| 优先级 | 数据源 | 说明 |
|--------|--------|------|
| 1 | `root_full.json`（与 `root_full.md` 同目录） | 含逐消息 `info.time.created/completed`，精度最高 |
| 2 | `root_full.md` | 仅 session 头与消息时长，需估算 |
| 3 | 当前会话（opencode DB，`scripts/session_db.py`） | tool/text/step 均有 `time_created` |

### 1. 从 `root_full.json` 提取（精度最高）

结构示例：

```json
{
  "info": { "time": { "created": 1724052000000, "updated": 1724056000000 } },
  "messages": [
    { "info": { "role": "user",   "time": { "created": 1724052005000 } },
      "parts": [...] },
    { "info": { "role": "assistant", "time": { "created": 1724052010000, "completed": 1724052050000 } },
      "parts": [...] }
  ]
}
```

- 按 `messages[].info.time.created` 排序，即为完整时间线。
- 消息内工具调用也可从 `parts[].state` 获取更细粒度的执行时间（有则用之，无则回退到消息级）。
- 时间戳为毫秒，需转换为本地时间（与 `root_full.md` 头部的 `**Created:**` 时区保持一致）。

### 2. 从 `root_full.md` 提取（估算）

`root_full.md` 提供的线索：

- 文件头：`**Created:** {绝对时间}`、`**Updated:** {绝对时间}`，可用作 session 起点与终点。
- 每条 Assistant 消息头部：`## Assistant (Agent - model - {duration}s)`，`duration` 是该消息耗时（秒）。
- 消息顺序即时间顺序，`## User` 消息无独立时间戳。

**估算规则**：

- **相对偏移** = 起点到目标消息之间所有 Assistant 消息 `duration` 的累加和（不含 User 消息间隔），格式 `+MM:SS`。
- **绝对时间** = `Created` + 相对偏移（近似值）。
- 所有由 md 推导的时间必须标注 `（估算）`，并说明估算口径（不含用户消息间隔）。

### 3. 从当前会话提取（opencode DB）

`scripts/session_db.py` 已提供 `ToolCall.time_created`、`TextPart.time_created`、`ReasoningPart.time_created`、step marker 的 `time_created`（毫秒）。将这些对象按时间排序即构成时间线，取法同 `root_full.json`。

## 阶段映射

- 使用不同工作流时，阶段划分规则不同，以实际工作流的上下文为准。
- 无任何阶段线索 → 标注 `未知`，并在报告中注明原因（如日志被截断、只有部分 session）。

## 解决或绕过耗时计算

### 解决/绕过的判定标准

满足任一条件即视为"已解决/绕过"：

- 同一实体后续操作**成功**（工具调用无错误、编译/验证通过、得到预期结果）
- Agent 切换实现方案并继续推进（workaround）
- 用户介入提供修正后 Agent 继续推进

未满足以上条件且 session 结束 → 记为"未解决"。

### 计算公式

```
解决耗时 = resolve_end - first_signal_start

first_signal_start = 首个触发该断裂点信号的消息 time.created
resolve_end        = 解决点消息的 time.completed（或 tool 调用结束时间）
                     未解决时 = session info.time.updated
```

- 格式：`MM:SS` 或 `H:MM:SS`。
- 未解决场景：`未解决（截至 session 结束 +MM:SS）`。
- md 估算场景：在数值后追加 `（估算）`。

## 输出字段规范

| 字段 | 格式 |
|------|------|
| 发生时间 | `YYYY-MM-DD HH:MM:SS 时区（+MM:SS）`；估算时追加 `（估算）` |
| 对应阶段 | `Stage N - 阶段名称`，或 `未知` |
| 解决/绕过耗时 | `MM:SS` / `H:MM:SS`，未解决见上 |

同一断裂点可能有多个信号跨越不同消息，取**首个信号**的时间作为发生时间，取**解决点**作为耗时终点。
