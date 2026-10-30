# CANNBot PyPTO-Pro 算子开发快速入门指南


本入口为实验性、非正式功能。skill 上游位于 [cannbot-skills community](https://gitcode.com/cann/cannbot-skills/tree/master/community/ops/pypto-pro-kernel)，编排入口上游位于 [cannbot community](https://gitcode.com/cann/cannbot/tree/master/plugins-community/pypto-pro-op-orchestrator)。
## 概述

本插件用于原生 PyPTO-Pro 算子开发、验证、调试与性能优化。默认由主代理自主决定执行方式，可按需调用子代理；用户明确开启多 agent 模式后，才遵循该模式的协作规则。具体流程见 [pypto-pro-kernel](../../ops/pypto-pro-kernel/SKILL.md)。

### 与 PyPTO 开发的区别

| 对比维度 | PyPTO-Pro 算子开发 | PyPTO 算子开发 |
|---------|------------------|--------------|
| 适用场景 | PyPTO-Pro 框架算子开发 | PyPTO 框架算子开发 |
| 编程语言 | Python（PyPTO-Pro API） | Python（PyPTO API） |
| 开发内容 | PyPTO-Pro kernel + golden + test | PyPTO kernel + golden + test |
| 开发流程 | 默认由主代理自主决定执行方式；可显式开启多 agent 模式 | 7 阶段状态机驱动 |
| 状态管理 | 按当前任务与验证证据推进 | `.orchestrator_state.json`（`state_transition` 工具 + lint 门禁） |
| 性能调优 | 按需优化 | Stage 7 独立调优阶段 |

## 一、环境搭建

### 前置条件

先配置可用且版本匹配的 CANN、PyPTO-Pro、Python 依赖和 Ascend NPU，并在项目中说明运行命令、环境脚本、设备和设备锁。源码检查还需匹配目标安装版本的 PyPTO 源码，可用 `PYPTO_SRC` 或 `--pypto-root` 指定。

配置好 [OpenCode](https://opencode.ai/) 和模型。本插件只安装技能与入口文件，不安装运行环境。

### OpenCode（推荐）

进入插件目录运行：

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator
bash init.sh project opencode
```

默认使用当前目录。

### 其他工具

以下命令安装相同的技能和入口；多 agent 模式需要所用工具支持子代理：

```bash
bash init.sh project claude
bash init.sh project trae
bash init.sh project cursor
bash init.sh project copilot
bash init.sh project codearts
```

### 在其他目录执行

可通过完整路径调用 `init.sh`，第三个参数指定目标项目。也可在插件目录指定独立项目：

```bash
mkdir -p /path/to/my_project
bash init.sh project opencode /path/to/my_project
cd /path/to/my_project
```

### 验证安装

在工作目录检查：

```bash
opencode agent list
python .opencode/skills/pypto-pro-kernel/scripts/check.py links
```

列表应包含主入口 `pypto-pro-op-orchestrator`，以及 `pypto-pro-op-` 前缀的 `planner`、`coder`、`debugger`、`verifier`、`optimizer`、`tool` 六个子代理。这些角色可按需调用；开启多 agent 模式后按协作说明派工。`links` 只检查文档链接，不代表数值或性能验证通过。

安装器复制四个完整 skill（`pypto-pro-kernel`、`pypto-benchmark-fracture-aggregator`、`pypto-fracture-point-detector`、`pypto-session-dfx-report`），包括参考资料、工具和样例，并将仓根 `LICENSE` 复制到每个已安装 skill 的 `LICENSE`；请勿只复制 `SKILL.md`。JSONC 注释、格式和已有选项保留。若同时有 `opencode.json` 和 `opencode.jsonc`，先合并所需设置，备份移走其中一个。已有其他默认 agent 时保留原选择，用下文的 `--agent` 启动本入口。

## 二、快速上手

### 启动

在工作目录启动 OpenCode：

```bash
opencode --agent pypto-pro-op-orchestrator
```

### 开发算子示例

说明算子含义、输入输出、支持范围、精度要求和交付内容。有现成代码、评估器或性能目标时，给出路径、评价方式和预算。

```text
使用 PyPTO-Pro 开发 softmax：输入为连续二维 float16 Tensor，沿最后一维计算，
输出 shape、dtype、布局与输入一致。支持行数 1～32、列数 128～4096（含尾块）。
使用 FP32 CPU 数学参考，按 atol=1e-3、rtol=1e-3 比较；测试有界随机值和边界情况。
环境配置见 ./ENVIRONMENT.md。请交付到 custom/softmax/，并实际验证。
```

调试已有代码时，提供复现命令、错误和现有验证结果。优化任务例如：

```text
优化 ./custom/softmax/softmax.py，沿用 ./benchmark/evaluator.py 的加载、精度和计时方式。
目标是该评估器的 benchmark score。环境见 ./ENVIRONMENT.md，设备实验预算为 30 分钟。
保留最佳已验证版本，报告最终正确性和性能覆盖。
```

### 多 agent 协作（按需）

在任务中明确加入：

```text
启用多 agent 模式。使用 PyPTO-Pro 开发上述 softmax，需求和环境配置同上。
```

主代理先完整阅读[协作说明](references/multi-agent-workflow.md)，再按当前问题选择角色派工。需求与验收分析交 planner，独立的环境核对、探针和命令执行等任务交 tool；主代理负责审核、澄清和调度。六个角色不构成固定阶段，不要求每次全部调用。调用时机、输入输出和上下文交接统一见该说明。仅讲解、引用或否定该模式时不启用，用户要求退出时恢复默认执行方式。

### 交付产物示例

前述提示词的算子包如下；要求验证或性能报告时另附 `REPORT.md`：

```text
custom/softmax/
├── README.md                    # 算子规格、调用方式与测试命令
├── softmax.py                   # 原生实现与公开接口
├── softmax_golden_cpu.py        # 独立 CPU 数学参考
├── test_softmax.py              # 可执行测试
└── precision_compare.py        # 固定精度判据
```

## 三、可用技能

当前入口使用 [pypto-pro-kernel](../../ops/pypto-pro-kernel/SKILL.md) 完成原生实现、验证、调试、计时、优化和交付。协作角色与派工方式见[协作说明](references/multi-agent-workflow.md)。

### 参考资料与可选工具

可按当前问题查阅[性能原则](references/kernel-core-principles.md)、[A5 实践](references/kernel-perf-pypto-pro-a5.md)和[实现约束](references/performance-constraints.md)。

可选 lint 不自动加入工作流，参数可用 `python hooks/pypto-pro-op-lint/pypto_pro_op_lint.py --help` 查看。

## 四、常见问题

### Q: 如何查看帮助信息？

```bash
bash init.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：安装到指定项目，各项目可独立配置。
- **全局级**：将 `project` 替换为 `global`，安装到工具的全局配置目录。

### Q: 如何更新？

更新仓库后，从插件目录按原参数重跑初始化，仍指定同一项目目录。需要替换的受管文件会备份到配置目录的 `.pypto-pro-op-orchestrator-backups/`。用户文件和其他插件保留；受管文件被修改或同名文件冲突时，先报错，不覆盖。

旧模式工作目录需另建目录初始化，不自动迁移旧配置。

### Q: PyPTO 和 PyPTO-Pro 如何选择？

| 场景 | 推荐模式 |
|------|---------|
| 使用 PyPTO 框架开发算子 | PyPTO |
| 使用 PyPTO-Pro API 开发算子 | PyPTO-Pro |

---

## 总结

1. 配置可用且版本匹配的运行环境。
2. 进入插件目录运行 `init.sh`，选择项目级或全局级安装。
3. 从工作目录启动入口，给出算子需求、验收要求与交付位置。
4. 需按本插件的协作规则执行时，明确开启多 agent 模式；按实际验证结果交付。
