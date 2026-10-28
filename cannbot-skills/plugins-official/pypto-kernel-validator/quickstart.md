# CANNBot PyPTO 算子产物校验快速入门指南

## 概述

CANNBot PyPTO Kernel Validator 用于校验一个声称由 PyPTO 开发的算子产物，判定该实现是否：

1. **真正使用 PyPTO 而非作弊** —— 脚本机械检测 + LLM 语义审阅双层反作弊
2. **精度通过** —— 对照 KernelBench 风格 task_desc 做 correctness 验证
3. **性能符合要求** —— 可选的 performance 验证

整套流程以统一的 `skill_report.json` 报告结束，是 KernelBench 桥接层评测的最终把关者。

### 与其他插件的区别

| 对比维度 | 本插件 | pypto-op-orchestrator / pypto-pro-op-orchestrator |
|---------|--------|--------------------------------------------------|
| 职责 | 校验已产出的算子产物（judge-only） | 从零开发算子 |
| 形态 | 单 subagent + 单 skill | PyPTO 为多 subagent + 状态机；Pro 默认自主决定执行方式，可显式开启多 agent 模式 |
| 典型调用方 | `benchmark/verifier_runner.py` 自动 spawn | 用户交互式发起 |
| 产物 | `<output_dir>/skill_report.json` | 算子工程；目录示例为 `custom/<op>/` |

## 一、环境搭建

### 前置条件

- 已安装 OpenCode、Claude Code、TRAE、Cursor、Copilot、CodeArts 等受支持的 AI 编程工具
- 校验目标为 KernelBench 产物时：需要 `benchmark` 模块，**该模块已从 pypto-gym 移除**，须由外部提供；不可用时 skill 记 `blocked` 而非 PASS（skill 内调用 `python -m benchmark.verifier`）
- 需要精度 / 性能验证时：已配置 CANN、torch/torch_npu 与 PyPTO 环境，NPU 可见

### OpenCode（推荐）

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym
bash cannbot-skills/plugins-official/pypto-kernel-validator/init.sh project opencode   # 项目级（默认）
```

安装后将在 `.opencode/` 下生成：

- `agents/pypto-kernel-validator.md` —— 校验 subagent
- `skills/pypto-kernel-validate` —— 校验 skill（软链接到 `cannbot-skills/ops/pypto-kernel-validate`）

### 其他工具

```bash
bash init.sh project claude     # Claude Code
bash init.sh project trae       # TRAE
bash init.sh project cursor     # Cursor
bash init.sh project copilot    # Copilot
bash init.sh project codearts   # CodeArts
bash init.sh global opencode    # 全局级
```

`init.sh` 支持通过完整路径调用，第三个参数指定目标项目路径，省略则安装到当前目录。

## 二、使用方式

### 方式一：KernelBench 桥接层自动调用（典型）

`benchmark/run_kernelbench.py` 的 verify 阶段由 `verifier_runner.py` 自动执行：

```bash
opencode run --agent pypto-kernel-validator "<结构化 prompt>"
```

无需人工干预，runner 会读取 `<output_dir>/skill_report.json` 转为 `VerifierResult`。

### 方式二：手动调度

在交互界面中给出结构化参数，调度 `pypto-kernel-validator` 子代理：

```text
请按 SKILL `pypto-kernel-validate` 校验下面这个 PyPTO 算子产物.

参数:
- op_name = <算子名>
- op_dir = <算子产物目录绝对路径>
- task_desc_file = <KernelBench task_desc.py 绝对路径>
- output_dir = <报告输出目录绝对路径>
- mode = correctness | performance | full
- device_id = 0            # 可选
- arch = ascend910b4       # 可选
- verify_timeout = 300     # 可选
```

## 三、产出物

唯一刚性产物：`<output_dir>/skill_report.json`，关键字段：

| 字段 | 说明 |
|------|------|
| `cheat_check_script` | 脚本机械检测结果 |
| `cheat_check_semantic` | LLM 语义审阅结论（S1–S9 逐项判定） |
| `correctness` / `performance` | 精度 / 性能验证结果 |
| `final_verdict` | `PASS` / `FAIL_CHEAT` / `FAIL_CORRECTNESS` / `FAIL_PERFORMANCE` / `BASELINE_FAILED` / `ERROR` |
| `failure_category` | 失败归类（PASS 时为空字符串） |

中间产物 `cheat_check_script.json`、`verify_run.json` 可选保留，便于排错。

## 四、常见问题

### Q: 手动调用时必须在 pypto-gym 仓根下吗？

skill 内 Step 1 / Step 3 调用 `python -m benchmark.verifier`，因此 cwd 需为含 `benchmark` 模块的 pypto-gym 仓根（或该模块已在 `PYTHONPATH` 中）。

### Q: `final_verdict=ERROR` 与失败的区别？

`ERROR` 表示技术性原因（CLI 异常、产物缺失、输入契约破坏）导致无法完成校验，不是算子本身的判定结论；其余 `FAIL_*` 才是对算子的实质性判定。

### Q: 如何更新？

```bash
cd pypto-gym && bash cannbot-skills/plugins-official/pypto-kernel-validator/init.sh project opencode
```
