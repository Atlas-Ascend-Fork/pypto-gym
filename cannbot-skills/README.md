# CANNBot Skills

![License](https://img.shields.io/badge/License-CANN%20OSL%20v2.0-blue?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Ascend%20NPU-orange?style=flat-square)

📖 [PyPTO 算子开发](plugins-official/pypto-op-orchestrator/quickstart.md) · [PyPTO-Pro 算子开发](plugins-official/pypto-pro-op-orchestrator/quickstart.md) · [模型工具](plugins-official/pypto-model-tools/quickstart.md) · [算子产物校验](plugins-official/pypto-kernel-validator/quickstart.md)

---

## 📢 项目概述

**CANNBot Skills — PyPTO-Gym** 是面向 PyPTO 与 PyPTO-Pro Tile 算子开发、大模型适配和模型治理的 CANNBot Agent Skills 模块，提供 37 个顶层技能与 4 个开发路径插件。

**面向用户**：基于 PyPTO 编程框架的昇腾 NPU 算子开发者、模型适配工程师。

## 🚀 快速开始

### 前置条件

- [OpenCode](https://opencode.ai/docs) AI 编程工具
- Bash、Python 3 运行环境
- 已按 [PyPTO-Gym](https://gitcode.com/cann/pypto-gym) 文档完成环境部署

### 安装

选择所需场景后，在目标项目目录调用对应插件的安装脚本。以 PyPTO 算子开发为例：

```bash
cd /path/to/your-project
bash /path/to/pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator/init.sh \
  project opencode
opencode
```

其他场景的安装方式见页面顶部对应的快速上手指南。

### 独立 PyPTO-Pro Kernel Skill

实验性、非正式的 [pypto-pro-kernel](ops/pypto-pro-kernel/SKILL.md) 提供原生 PyPTO-Pro kernel 开发、验证、调试和优化流程，目录内包含 `references/`、`scripts/` 与 `examples/`。可按 [PyPTO-Pro quickstart](plugins-official/pypto-pro-op-orchestrator/quickstart.md) 初始化插件，或在提示词中指定其 `SKILL.md`、将整个目录复制到目标工具的 skill 目录。流程由该 skill 决定。插件入口默认由主代理自主决定执行方式，可按需调用子代理；用户明确开启多 agent 模式后，才遵循该模式的协作规则。静态检查、样例覆盖和目录一致性检查可从 skill 根目录运行：

```bash
cd /path/to/pypto-gym-toolkit/cannbot-skills/ops/pypto-pro-kernel
python scripts/check.py
```

需要核对 PyPTO API 时，用 `PYPTO_SRC` 或 `--pypto-root` 指定匹配目标安装版本的外部源码树。上板样例需要对应的 PyPTO、PyTorch、torch_npu 和 Ascend NPU 环境；该 skill 不包含仿真器。

`scripts/check.py links` 检查内置 Markdown 链接与标题引用，`explain` 分析候选源码的契约和工作量变化。可选的词法名称检查依赖 `scripts/requirements-checks.txt` 中固定版本的 Pyflakes。样例统一以 `kernel.py`、`reference.py`、`test.py` 为公开入口；Sparse Attention 的原始评估器对照额外需要 `CANN_BENCH_ROOT` 和可导入的 `kernel_eval`。

### 安装后使用

启动 OpenCode，直接以自然语言描述需求：

```text
"开发一个 Attention 融合算子"
"把 Qwen3-1.7B 模型适配到 NPU"
```

## 📐 开发路径与核心能力

| 领域 / 路径 | 适用场景 | 入口插件 |
|------------|---------|---------|
| **PyPTO 算子开发** | Stage 1–7 全流程融合算子开发与调优 | [pypto-op-orchestrator](plugins-official/pypto-op-orchestrator/AGENTS.md) |
| **PyPTO-Pro 算子开发** | 算子开发、验证与性能优化 | [pypto-pro-op-orchestrator](plugins-official/pypto-pro-op-orchestrator/AGENTS.md) |
| **模型适配** | HF 模型上 NPU、融合算子整网集成、模型格式转换 | [pypto-model-tools](plugins-official/pypto-model-tools/AGENTS.md) |
| **算子产物校验** | KernelBench 评测把关：反作弊 + 精度 + 性能统一校验 | [pypto-kernel-validator](plugins-official/pypto-kernel-validator/AGENTS.md) |

### 技能清单

**算子开发**（ops/，共 34 个顶层技能）

| 类别 | 技能 | 说明 |
|------|------|------|
| 编排入口 | `pypto-orchestration-manual` | 10-agent 团队编排入口 |
| 需求理解 | `pypto-intent-understand` | 用户需求分析、规格化与歧义澄清 |
| 方案与计划 | `pypto-op-design`、`pypto-op-plan` | 算子方案设计与实施计划 |
| 知识参考 | `pypto-op-knowledge`、`pypto-api-explore`、`pypto-docs-search`、`pypto-memory-template` | 领域知识库、API 速查、文档检索、经验模板复用 |
| 代码开发 | `pypto-op-develop`、`pypto-op-design` | 算子核心实现与工程脚手架搭建 |
| Golden 与精度 | `pypto-golden-generate`、`pypto-precision-compare`、`pypto-precision-debug` | Golden 生成、精度对比、精度问题排查 |
| 验证与检视 | `pypto-op-verify`、`pypto-op-review`、`pypto-kernel-validate` | 算子功能验证、代码检视与产物校验（反作弊 + 精度 + 性能） |
| 性能调优 | `pypto-op-perf-tune` | 算子性能采集、分析与自动调优 |
| 性能调优 | `pypto-op-perf-panko-manual` | PANKO 搜索式自动调优的 world model + evaluator 契约（由 `pypto-op-auto-tuner-panko` 读取） |
| 监控与调试 | `pypto-op-monitor`、`pypto-general-debug` | 任务进度监控与通用问题诊断 |
| 断裂点分析 | `pypto-benchmark-fracture-aggregator`、`pypto-fracture-point-detector` | Session 断裂点批量聚合、检测、复现取证与 Issue 文档产出 |
| 报告与可视化 | `pypto-session-dfx-report` | Session DFX 报告（端到端耗时、token 消耗、执行轨迹） |
| **Pro 专属** | `pypto-pro-intent-understand`、`pypto-pro-material-explore`、`pypto-pro-op-plan`、`pypto-pro-op-design`、`pypto-pro-op-develop`、`pypto-pro-op-perf-tune`、`pypto-pro-golden-generate`、`pypto-pro-environment-check`、`pypto-pro-precision-debug`、`pypto-pro-cann-delivery`、`pypto-pro-docs-search` | PyPTO-Pro 开发、精度诊断、CANN 交付与文档检索 |
| **Pro 独立技能（实验）** | [pypto-pro-kernel](ops/pypto-pro-kernel/SKILL.md) | 原生 kernel 开发、验证、调试与优化，内含参考资料、检查工具和样例 |

**模型适配**（model/，共 3 个）

| 技能 | 说明 |
|------|------|
| `hf-npu-e2e-workflow` | HuggingFace 模型到昇腾 NPU 端到端迁移 |
| `pypto-fused-op-integration` | 融合算子入网集成与端到端推理验证 |
| `pypto-convert-model` | 模型格式互转（PyTorch/ONNX/safetensors） |

> 模型插件的 `init.sh` 同时安装 8 个算子支撑 skill（`pypto-intent-understand` 等），仅由 `pypto-fused-op-integration` 按需调用，不计入独立 skill 数。

## 🔍 项目架构

### 目录结构

```
cannbot-skills/
├── ops/                  # 算子 Skills（PyPTO classic + PyPTO-Pro）
├── model/                # 模型适配与推理优化 Skills
└── plugins-official/     # 官方 Plugins（开发路径入口，含 Agents）
    ├── pypto-op-orchestrator/      # PyPTO classic 算子开发（8 Subagent + 状态机）
    ├── pypto-pro-op-orchestrator/  # PyPTO-Pro 算子开发
    ├── pypto-model-tools/          # 模型适配工具集（安装时附带 8 个算子支撑 skill）
    └── pypto-kernel-validator/     # 算子产物校验（反作弊 + 精度 + 性能，单 Subagent）
```

### 三层架构

三层架构：Plugin 编排 Agents，Agents 绑定 Skills。

- **Plugin**（应用编排层）— 通过 `AGENTS.md` 定义各 Agent 协作顺序
- **Agent**（角色执行层）— 承担方案设计、代码开发、代码检视等职责
- **Skill**（知识能力层）— 提供领域知识与工程模板

以 PyPTO classic 算子开发为例：

```
╔══════════════════════════════════════════════════════════════════════╗
║                      PLUGINS（应用编排层）                           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌────────────────────────────────────────────────────────────────┐  ║
║  │  pypto-op-orchestrator                                         │  ║
║  │  PyPTO Classic Stage 1–7 融合算子开发全流程                      │  ║
║  └──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬───────┘  ║
║         │      │      │      │      │      │      │      │          ║
╚═════════╪══════╪══════╪══════╪══════╪══════╪══════╪══════╪══════════╝
          │      │      │      │      │      │      │      │
          ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
╔══════════════════════════════════════════════════════════════════════╗
║                       AGENTS（角色执行层）                           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   ║
║  │   planner   │ │mathematician│ │  architect  │                  ║
║  │ 需求与计划  │ │  数学推导   │ │设计与接口   │                  ║
║  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   ║
║  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   ║
║  │    coder    │ │  verifier   │ │  debugger   │ │  optimizer  │   ║
║  │  代码开发   │ │  功能验证   │ │  问题诊断   │ │  性能调优   │   ║
║  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
         │ │ │           │ │ │           │ │ │           │ │ │
         ▼ ▼ ▼           ▼ ▼ ▼           ▼ ▼ ▼           ▼ ▼ ▼
╔══════════════════════════════════════════════════════════════════════╗
║                       SKILLS（知识能力层）                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─ 编排与知识类 ─────────────────────────────────────────────────┐  ║
║  │  pypto-orchestration-manual  10-agent 团队编排入口               │  ║
║  │  pypto-intent-understand     需求分析与规范化                    │  ║
║  │  pypto-op-knowledge          领域知识库                           │  ║
║  │  pypto-api-explore           API 速查与用法示例                  │  ║
║  │  pypto-docs-search           文档检索                            │  ║
║  │  pypto-memory-template       经验模板复用                        │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
║                                                                      ║
║  ┌─ 开发与工程类 ─────────────────────────────────────────────────┐  ║
║  │  pypto-op-plan               实施计划                            │  ║
║  │  pypto-op-design             方案设计                            │  ║
║  │  pypto-op-develop            代码开发                            │  ║
║  │  pypto-op-design          工程脚手架                          │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
║                                                                      ║
║  ┌─ 验证与调优类 ─────────────────────────────────────────────────┐  ║
║  │  pypto-golden-generate       Golden 生成                         │  ║
║  │  pypto-precision-compare     精度对比                            │  ║
║  │  pypto-precision-debug       精度问题排查                        │  ║
║  │  pypto-op-verify             功能验证                            │  ║
║  │  pypto-op-review             代码检视                            │  ║
║  │  pypto-op-perf-tune          性能分析与自动调优                   │  ║
║  │  pypto-op-perf-panko-manual          PANKO 搜索式自动调优契约             │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
║                                                                      ║
║  ┌─ 监控与调试类 ─────────────────────────────────────────────────┐  ║
║  │  pypto-op-monitor            任务进度监控                        │  ║
║  │  pypto-general-debug          通用问题诊断                        │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

## 🔥 最新动态

本目录随 [PyPTO-Gym 主仓](https://gitcode.com/cann/pypto-gym) 同步更新，最新变更以主仓提交记录为准。

## 💬 相关信息

- [PyPTO 主仓](https://gitcode.com/cann/pypto) — PyPTO 编程框架与文档
- [PyPTO-Gym 主仓](https://gitcode.com/cann/pypto-gym) — 融合算子样例与模型适配
- [许可证](../LICENSE) — 基于 CANN Open Software License v2.0

## 💖 免责声明

感谢您关注 CANNBot Skills — PyPTO-Gym，希望这些技能和知识能帮助您更好地进行 PyPTO 算子开发。

1. **功能满足度**：由于技术快速迭代，部分技能内容可能无法完全适用于所有场景。技能与文档持续完善中，欢迎提 Issue 或参与讨论。
2. **自动生成内容**：自动生成的代码受模型、Skills 能力、语料质量、输入指令等多因素影响，无法保证完全精准。生成代码仅作辅助研发使用，请开发者务必测试验证后再投入使用。

## 🤝 社区交流

| 渠道 | 适用场景 | 链接 |
|------|---------|------|
| GitCode Issue | Skill bug / 功能缺失 | [提交 Issue](https://gitcode.com/cann/pypto-gym/issues) |
| GitCode Discussions | 使用疑问 / 经验交流 / 功能建议 | [参与讨论](https://gitcode.com/cann/pypto-gym/discussions) |
