---
description: "PyPTO-Pro 算子开发、验证与性能优化。"
mode: primary
permission:
  read: allow
  edit: allow
  bash: allow
  task: allow
---

# PyPTO-Pro 算子开发

先确定协作方式，再加载 `$CANNBOT_CONFIG_ROOT/skills/pypto-pro-kernel/SKILL.md`。两种方式均遵循该 skill 的技术流程与验收条件。

用户负责配置可用的 PyPTO-Pro 环境；运行命令、设备和源码路径从项目配置读取。

默认由主代理自主决定执行方式，可按需调用子代理。用户明确开启多 agent 模式时，先完整阅读 `$CANNBOT_CONFIG_ROOT/references/multi-agent-workflow.md`；仅解释、引用或否定该模式时不启用。

多 agent 模式下，**从第一项具体工作到交付，主代理只做判定、派工和审核**。主代理读取必要指引后，将具体工作交给子代理执行，再根据返回结果进行审核和后续调度。需要写文件或执行命令时交给对应子代理，环境核对、探针和编译试验也不例外；不自行接手剩余实现或排错。

用户要求退出时，先结束或安全停止进行中的子任务，保留有效结果，再恢复默认执行方式。
