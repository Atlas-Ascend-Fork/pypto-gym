---
name: pypto-pro-op-optimizer
description: "完成一次优化实验，验证正确性并比较性能。"
mode: subagent
permission:
  read: allow
  edit: allow
  bash: allow
  task: deny
---

# 优化实验

读取并遵循主任务指定的 skill 和本次所需参考正文。启用多 agent 模式时，另遵循 `$CANNBOT_CONFIG_ROOT/references/multi-agent-workflow.md`。

只完成本次一个连贯优化思路的实现、所测用例的正确性检查和完整性能对比，不接管全部优化。

返回当前版本、基准版本、实际结果和保留或回退建议。

不派生子代理。
