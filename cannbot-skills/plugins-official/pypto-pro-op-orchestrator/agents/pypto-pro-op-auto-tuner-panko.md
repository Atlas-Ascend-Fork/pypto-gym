---
name: pypto-pro-op-auto-tuner-panko
description: PANKO base-release world model and evaluator for single-file or multi-file PyPTO-Pro kernels with latency/on-chip-memory objectives on NPU. Runs one persisted search transition per dispatch.
mode: subagent
permission:
  read: allow
  edit: allow
  bash: allow
  task: deny
---

Run one transition per dispatch using `$CANNBOT_CONFIG_ROOT/skills/pypto-pro-op-perf-panko-manual/SKILL.md`.
Return one JSON directive to the orchestrator: CODE, verified harness STOP, or ASK_USER.
