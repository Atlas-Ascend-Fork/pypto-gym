# CANNBot PyPTO 算子开发快速入门指南

## 概述

CANNBot PyPTO 算子开发模式适用于通过 PyPTO 开发自定义算子。由 **9 智能体团队**（orchestrator 编排者 + planner / mathematician / architect / coder / verifier / debugger / optimizer / auto-tuner-panko 8 个子代理）驱动 7 阶段状态机，覆盖从需求理解到性能调优的完整开发流程。每个阶段完成门禁校验后才能进入下一阶段，由 `state_transition` 状态机工具维护进度（`.orchestrator_state.json`），支持断点续跑与失败恢复；Stage 5+ 通过 `MEMORY.md` 协作账本记录 pass/fail 与推理过程。

### 与 PyPTO-Pro 开发的区别

| 对比维度 | PyPTO 算子开发（本模式） | PyPTO-Pro 算子开发 |
|---------|------------------------|-------------------|
| 适用场景 | PyPTO 框架算子开发 | PyPTO-Pro 框架算子开发 |
| 编程语言 | Python（PyPTO API） | Python（PyPTO-Pro API） |
| 开发内容 | PyPTO kernel + golden + test | PyPTO-Pro kernel + golden + test |
| 阶段数 | 7 阶段状态机驱动 | 5 阶段工作流 |
| 状态管理 | `.orchestrator_state.json` 状态文件 + `MEMORY.md` 协作账本 | 调度顺序隐式管理 |
| 性能调优 | Stage 7 独立调优阶段 | Stage 5：可比基线、优化循环与证据验收 |

## 一、环境搭建

### 前置条件

- 已安装 CANN Toolkit（建议 ≥ 9.0.0），具体版本配套关系请查阅 [CANN Release Notes](https://www.hiascend.com/cann/document)
- 已安装 PyPTO，版本需与 CANN 配套。通过 PyPI 安装时，CANN 与 PyPTO 版本对应关系查阅 [PyPI 安装](https://pypto.gitcode.com/install/build_and_install.html#pypi)；CANN 9.1.0 版本推荐使用源码编译安装，参阅 [源码编译安装](https://pypto.gitcode.com/install/build_and_install.html)
- 已配置 NPU 设备（支持 Ascend 910/950 PR 等芯片）
- 已安装 OpenCode、Claude Code、TRAE、Cursor、Codex、Copilot、CodeArts 等受支持的 AI 编程工具
- **仅 PANKO 自动调优需要**：`optuna >= 2.0`（`python3 -m pip install 'optuna>=2.0'`）。Tile 数值由 optuna 的 TPE 决定，缺失时 `init` 会明确拒绝而不是静默改用其它方式；不使用 PANKO 时无需安装

### 通用安装方式（init.sh）

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project <tool>   # 项目级（默认）
bash init.sh global <tool>    # 全局级
```

`<tool>` 可取 `opencode` / `claude` / `trae` / `cursor` / `codex` / `copilot` / `codearts`，省略时取默认值 `opencode`。

### 各工具安装说明

<details>
<summary>OpenCode</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project opencode   # 项目级
bash init.sh global opencode    # 全局级
```

安装后在项目根目录生成 `.opencode/` 目录与 AGENTS.md。

</details>

<details>
<summary>Claude Code</summary>

**首选：Plugin Marketplace（一键安装）**

```bash
# 注册 marketplace（首次，GitCode 仓库需完整 URL）
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git

# 安装插件
/plugin install pypto-op-orchestrator@cannbot
```

**备选：init.sh 脚本**

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project claude     # 项目级
bash init.sh global claude      # 全局级
```

</details>

<details>
<summary>TRAE</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project trae       # 项目级
bash init.sh global trae        # 全局级
```

安装后自动检测 TRAE 环境，生成 `.trae/`（TRAE IDE）、`.marscode/`（TRAE Plugin）或 `.traecli/`（TRAE CLI）目录，结构与其他工具基本一致。

</details>

<details>
<summary>Cursor</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project cursor     # 项目级
bash init.sh global cursor      # 全局级
```

安装后在项目根目录生成 `.cursor/` 目录，结构与其他工具基本一致。

</details>

<details>
<summary>Codex</summary>

Codex 的 Skill 与 Subagent 使用不同的发现目录：

- 项目级 Skill：`.agents/skills/`
- 项目级 Subagent：`.codex/agents/*.toml`
- 全局 Skill：`~/.agents/skills/`
- 全局 Subagent：`~/.codex/agents/*.toml`

#### 项目级安装

```bash
bash init.sh project codex
```

#### 全局安装

```bash
bash init.sh global codex
```

</details>

<details>
<summary>Copilot</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project copilot    # 项目级
bash init.sh global copilot     # 全局级
```

安装后在项目根目录生成 `.github/` 目录（项目级）或 `~/.copilot/` 目录（全局级），AGENTS.md 自动注入 VS Code Copilot 上下文。

</details>

<details>
<summary>CodeArts</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project codearts     # 项目级
bash init.sh global codearts      # 全局级
```

安装后在项目根目录生成 `.codeartsdoer/` 目录（项目级）或 `~/.codeartsdoer/` 目录（全局级），包含 skills/、agents/ 和 AGENTS.md。

</details>

### 在其他目录执行

`init.sh` 支持通过完整路径调用，无需先 `cd` 到插件目录。第三个参数指定目标项目路径，省略则安装到当前目录：

```bash
# 安装到当前目录
bash /path/to/pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator/init.sh project <tool>

# 安装到指定项目
bash /path/to/pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator/init.sh project <tool> /path/to/your_project_path
```

### 验证安装

```bash
# OpenCode
opencode agent list
# 应看到 pypto-op-planner / pypto-op-mathematician / pypto-op-architect
#        / pypto-op-coder / pypto-op-verifier / pypto-op-debugger / pypto-op-optimizer
#        / pypto-op-auto-tuner-panko（编排者本身以 AGENTS.md 注入）

# Claude Code
claude plugin list
# 应看到 pypto-op-orchestrator@cannbot ✔ enabled
ls .
# 应看到 CLAUDE.md 位于项目根目录

# TRAE
ls .trae/      # TRAE IDE
ls .marscode/  # TRAE Plugin（init.sh 自动检测）
ls .traecli/   # TRAE CLI（init.sh 自动检测）
# 应看到 skills/ agents/ cannbot-manifest.json
# AGENTS.md 位于项目根目录

# Cursor
ls .cursor/
# 应看到 skills/ agents/ cannbot-manifest.json
# AGENTS.md 位于项目根目录

# Copilot
ls .github/        # 项目级；全局级为 ~/.copilot/
# 应看到 skills/ agents/ cannbot-manifest.json
# AGENTS.md 位于项目根目录

# CodeArts
ls .codeartsdoer/  # 项目级；全局级为 ~/.codeartsdoer/
# 应看到 skills/ agents/ cannbot-manifest.json
# AGENTS.md 位于项目根目录
```

## 二、快速上手

### 启动

```bash
# OpenCode
opencode

# Claude Code
claude
```

> **TRAE 用户**：TRAE 通过 IDE、VS Code 插件或 CLI 启动。init.sh 会自动检测 TRAE IDE（`~/.trae-cn`）、Plugin（`~/.marscode`）或 CLI（`~/.traecli`）并安装到对应目录。安装完成后在 IDE 中直接打开项目即可。
>
> **Cursor 用户**：Cursor 通过 IDE 启动，`.cursor/` 目录中的配置会自动加载。安装完成后在 IDE 中直接打开项目即可。
>
> **Copilot / CodeArts 用户**：通过各自的 CLI 或 IDE 启动，安装目录中的 skills/agents 与项目根目录的 AGENTS.md 会自动加载。

### 开发算子示例

对于 `softmax` 这类语义明确的算子，一句描述即可启动：

```
帮我开发一个 softmax 算子，支持 float16 数据类型，shape 主要是 [1,128]、[4,2048]、[32,4096]
```

> ⚠️ **开发新算子时，请根据算子难度，参照下方描述示例，补全相关提示词信息**

以开发 `gelu` 算子为例：

```
帮我开发一个 gelu 算子（tanh 近似版），公式：
  gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

输入输出：
  - x: [B, N] float16，B 为动态轴，N ∈ {128, 2048, 4096}
  - y: 与 x 同 shape / dtype
```

把公式和输入输出规格写清楚，便于 CANNBot 准确解析开发需求。

### 核心工作流

采用 7 阶段状态机驱动，9 智能体团队协作，确保算子开发质量：

```
Stage 1: 需求规划与 API 可行性 → Stage 2: Golden 生成
    → Stage 3: 设计与模块接口 → Stage 4: 独立检查与验证准备
    → Stage 5: 按模块编码闭环 → Stage 6: 最终 E2E 验证
    → Stage 7: 性能调优
```

- **Stage 1**（planner）：完成需求规划和 API 可行性，产出 SPEC.md、API_REPORT.md
- **Stage 2**（mathematician）：生成 `<op>_golden.py` 参考实现与 GOLDEN_PERF_REPORT.md
- **Stage 3**（architect）：一次完成 DESIGN.md 和 eval/module_interfaces.yaml
- **Stage 4**（verifier）：独立检查设计与接口；多模块（L1）时准备对抗测试文件，单模块（L0）跳过
- **Stage 5**（coder、verifier、debugger）：按模块闭环，逐模块完成编码→验证→修复，产出 `modules/`、集成 `<op>_impl.py`、`test_<op>.py` 与 README.md；MEMORY.md 从 Stage 5 开始写入
- **Stage 6**（verifier）：最终 E2E 精度验证与 layout 校验
- **Stage 7**（optimizer / auto-tuner）：默认由 optimizer 做分步骤调优（性能采集、分析与迭代调优），生成 `<op>_tuning_report.md`；**用户在 initial prompt 中要求 PANKO 时**改由 auto-tuner 以 relay 循环做搜索式自动调优，产出 `optimization/search_state.json` 与 `optimization/<op>_optimization.md`。两种路径都由 verifier 回归确认精度无损

每个阶段完成门禁校验后才能进入下一阶段。支持断点续跑和失败恢复，详见 AGENTS.md。

### 产出物示例

PyPTO 算子开发模式下，CANNBot 会在 `custom/<op>/` 目录下生成以下文件：

```
custom/<op>/
├── SPEC.md                    # 需求规格
├── API_REPORT.md              # API 可行性报告
├── <op>_golden.py             # Golden 参考实现
├── GOLDEN_PERF_REPORT.md      # Golden 基准性能报告
├── DESIGN.md                  # 架构设计文档
├── <op>_impl.py               # 集成 PyPTO kernel 实现
├── test_<op>.py               # 端到端测试入口
├── README.md                  # 实现说明
├── MEMORY.md                  # 阶段协作记录（Stage 5+）
├── <op>_tuning_report.md      # 性能调优报告（Stage 7）
├── .orchestrator_state.json   # 流程状态（自动维护）
├── eval/                      # 模块接口契约与对抗测试（L1 路径）
│   ├── module_interfaces.yaml # 模块接口契约
│   ├── test_inputs.py         # 对抗测试输入
│   ├── adversarial_suite.json # 对抗测试套件
│   └── adversarial_runner.py  # 对抗测试执行器
├── modules/                   # 模块文件（L1 路径）
└── history_version/           # 版本备份
```

## 三、可用技能

| Skill | 用途 | 触发阶段 |
|-------|------|---------|
| `pypto-op-plan` | 串行组织需求理解与 API 可行性探索 | Stage 1 |
| `pypto-intent-understand` | 需求意图理解与规格生成 | Stage 1 |
| `pypto-api-explore` | API 可行性探索与分析 | Stage 1 |
| `pypto-golden-generate` | Golden 参考实现生成 | Stage 2 |
| `pypto-op-design` | 算子设计、模块划分及接口生成 | Stage 3 |
| `pypto-op-develop` | 算子代码实现 | Stage 5 |
| `pypto-op-verify` | 模块、E2E 与回归验证 | Stage 4–7 |
| `pypto-general-debug` | 通用错误定位与修复 | Stage 5 |
| `pypto-precision-debug` | 精度问题代码层排查 | Stage 5 |
| `pypto-precision-compare` | 精度中间结果对比分析 | Stage 5（辅助） |
| `pypto-op-perf-tune` | 算子性能分析与自动调优（分步骤） | Stage 7 |
| `pypto-op-perf-panko-manual` | PANKO 搜索式自动调优的 world model + evaluator 契约 | Stage 7（用户要求 PANKO 时） |
| `pypto-op-review` | 设计与实现评审 | Stage 3–7 |
| `pypto-docs-search` | 算子 API 文档、参考实现与 golden 检索 | 按需 |
| `pypto-memory-template` | MEMORY.md 协作账本模板 | Stage 5+ |
| `pypto-op-knowledge` | 算子开发知识库 | 按需 |
| `pypto-op-monitor` | 过程监控与状态追踪 | 全阶段 |
| `pypto-orchestration-manual` | 编排策略与门禁定义 | 全阶段 |

### 智能体团队

编排者 `pypto-op-orchestrator`（以 AGENTS.md/CLAUDE.md 注入，全流程唯一 owner）按 7 阶段调度 8 个子代理，自身不执行领域工作：

| Agent | 用途 | 负责阶段 |
|-------|------|---------|
| `pypto-op-planner` | 需求规划与 API 可行性 | Stage 1 |
| `pypto-op-mathematician` | Golden 参考实现 | Stage 2 |
| `pypto-op-architect` | 架构、tiling 与 loop 设计 | Stage 3 |
| `pypto-op-coder` | Kernel 实现 | Stage 5 |
| `pypto-op-verifier` | 独立裁决与检查 | Stage 4–7 |
| `pypto-op-debugger` | 失败定位与补丁建议 | Stage 5 |
| `pypto-op-optimizer` | 性能采集与调优（分步骤） | Stage 7 |
| `pypto-op-auto-tuner-panko` | PANKO 搜索式自动调优的 world model + evaluator | Stage 7（用户要求 PANKO 时） |

## 四、断点续跑与恢复

CANNBot 通过 `.orchestrator_state.json` 维护全局状态，支持：

| 场景 | 使用方式 |
|------|---------|
| 中断后继续 | 再次输入算子名，自动从上次中断处续跑 |
| 失败后重试 | 输入"继续开发 {算子名}"，从失败阶段恢复 |
| 查看状态 | 查看 `custom/<op>/.orchestrator_state.json` |

## 五、常见问题

### Q: 如何查看帮助信息？

```bash
bash init.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：适合多项目开发，每个项目可以有不同配置
- **全局**：适合单一项目，全局生效

### Q: 如何更新？

```bash
# 通用（init.sh 方式，重跑一次安装即可）
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator && bash init.sh project <tool>

# Claude Code 亦可用插件命令
/plugin update pypto-op-orchestrator@cannbot
```

### Q: PyPTO 和 PyPTO-Pro 如何选择？

| 场景 | 推荐模式 |
|------|---------|
| 使用 PyPTO 框架开发算子 | PyPTO 算子开发 |
| 使用 PyPTO-Pro API 开发算子 | PyPTO-Pro 算子开发 |
| 快速验证算子可行性 | PyPTO 算子开发 |
| 原型开发和概念验证 | PyPTO 算子开发 |

---

## 总结

1. PyPTO 算子开发模式通过 7 阶段状态机与 9 智能体团队实现端到端自动化：需求规划→Golden 生成→架构设计→模块分解→编码闭环→E2E 验证→性能调优
2. 使用 `init.sh` 脚本一键安装（支持 OpenCode / Claude Code / TRAE / Cursor / Codex / Copilot / CodeArts），Claude Code 用户也可用 `/plugin install` 一键安装
3. 各工具的 CLI 或 IDE 入口（如 `opencode`、`claude`，或 TRAE / Cursor / Codex / Copilot / CodeArts 的 IDE）即核心交互入口
4. 所有阶段通过门禁驱动，支持断点续跑与失败恢复
5. 产出物包含完整的 SPEC、API 报告、Golden 参考、架构设计、模块契约、模块实现、E2E 测试、调优报告与流程状态文件
