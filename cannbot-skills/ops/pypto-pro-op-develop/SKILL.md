---
name: pypto-pro-op-develop
description: 按既有 DESIGN 与 Module 合同完成 PyPTO-Pro 的 L0 或逐 Module staged/finalize 开发、自验证和证据交接。当前 kernel 技术规范统一由 pypto-pro-kernel 拥有；发现上游、能力或环境问题时按事实分类报告。
---

# PyPTO-Pro 算子 Kernel 实现

把已冻结的 `DESIGN.md` 翻译为一个可运行、可测试的 PyPTO-Pro kernel，并逐条落实 DESIGN 中的任务不变量。完成本地开发与自验证闭环后交给独立 verifier；本 skill 不修改 SPEC、DESIGN、Module 合同或编排状态。

## 输入与输出

先读取以下输入：

- `custom/<op>/DESIGN.md`：冻结的施工合同，包含 Module、API、tile、地址、循环、同步、尾块和测试设计。
- `custom/<op>/module_interfaces.yaml`：L1 当前 Module 的输入来源、输出与 `golden_steps`。
- `custom/<op>/EXPLORE_REPORT.md`：已核对的 API 约束和相似样例。
- `custom/<op>/PRO_MATERIAL_INDEX.md`：需要回看原文时的 API、官方样例和教程路径。
- [pypto-pro-kernel](../pypto-pro-kernel/SKILL.md)：当前唯一技术规范；本 Skill 保留旧工程 Module 开发与交接流程。

性能原则见 `$CANNBOT_CONFIG_ROOT/references/kernel-core-principles.md`。实现期直接相关的是
`P4`（Device 侧循环内零除法 / 取模，地址按 `addr += stride` 推进）、`P2`（轮转 buffer 用 TileGroup，
深度改动须复核 `T1`）、`X1`（CV 错拍不得退化为逐 tile 串行）。收尾时按 `kernel-perf-pypto-pro-a5.md` §2 的自检清单（`G1`–`G6`）自查，结论写进本轮报告；没测过的项记 `unknown`（`M1`）。

按 dispatch 交付：

| 路径 | dispatch | 本轮产物 |
|---|---|---|
| L0 | 无 `module_k` 且无 `finalize` | `custom/<op>/test_<op>.py`，附真实实现与验证证据 |
| L1 | 带 `module_k` | `custom/<op>/modules/test_<op>_module<suffix_k>.py`；附当前 staged 的验证证据 |
| L1 finalize | `finalize=true`，无 `module_k` | cleanup 尝试后完成 `custom/<op>/test_<op>.py`，附真实实现与验证证据 |

此表是既有编排器的开发/评测交接格式，保留其文件名和加载协议。另行交付 standalone package 时，按 [五文件格式](../pypto-pro-kernel/references/workflows/delivery.md#required-layout) 导出实现、测试与独立 reference/comparator；不能把两种交接格式的文件清单混用。

## 实现合同

施工时始终满足以下本阶段职责：

1. 核心计算留在 native kernel 内；入口、tiling key 与多阶段启动按 [当前 delivery owner](../pypto-pro-kernel/references/workflows/delivery.md#kernel-definitions-and-dispatch) 和 DESIGN 执行，不能把旧单 launch 规则当作通用合同。
2. 严格执行 DESIGN.md 已冻结的 Module 边界、API 序列、tile 属性/地址、循环、同步、尾块和 `vector_selection`。本轮功能或精度测试表明设计有误时上报疑似 `design_violation`，不得静默改设计。
3. buffer 角色、TileGroup 与 `auto_mutex` 的依赖范围按 [当前 allocation owner](../pypto-pro-kernel/references/api/memory-allocation.md) 和 DESIGN 执行；常驻、轮转和 scratch 分别核算，不能仅凭 Python 对象名推断同步或生命周期。
4. wrapper 按 [Host wrapper 规则](../pypto-pro-kernel/references/workflows/delivery.md#host-wrapper-boundary) 校验参数、读取允许的元数据、推导标量、用 `torch.empty` 或 `torch.empty_like` 分配输出/workspace 并启动 native entry。任何其他 Host Torch 操作（含 view）或独立 preprocessing 须有本任务对该操作的显式批准，并纳入完整调用的正确性与计时验证。
5. 测试通过 wrapper 调用 kernel；不得删改 DESIGN.md §8 的 case 来迁就实现，也不得把核心计算移到测试或 host 代码。
6. **上下文效率**：对于已加载且未发生变更的文件，尽量按需读取，避免全量重读。重复读取、冗余思考复述等无效上下文属于无效开销，会增加Token消耗，同时导致注意力稀释。

## 按需读取的资源

本表是本 skill 资源索引的唯一来源（含加载时机）。

| 资源 | 用途 | 加载时机 |
|---|---|---|
| [当前 example selection](../pypto-pro-kernel/references/workflows/authoring.md#selecting-an-example) 与 [example catalog](../pypto-pro-kernel/references/tooling/examples.md) | 按数学/轴/数值/layout 契约选择完整 kernel/test，先复跑其最窄范围再适配 | 选择纯 Vector/Cube/SIMT 代码起点时 |
| `$PYPTO_DEVKIT_DIR/pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` | **目标版本 Cube 接口/手写例外取证**——两级 K 分块、move offset、phase、尾块及 ASW 调度；仅路径存在且版本匹配时参考，不优先于当前 GEMM policy | 纯 cube 算子生成代码前 |
| [templates/impl_template.py.tmpl](templates/impl_template.py.tmpl) | kernel 文件骨架（tile 声明 + section + Module + 测试函数）——**通用**（含 CV 融合 / 多 Module / 跨核流水）；CV 融合算子当前无专用模板，走此通用骨架，按「开发流程」各步执行 | 生成代码前必读 |
| [references/debugging-methodology.md](references/debugging-methodology.md) | 调试方法论：**先确认失败可信** → 症状快查表 → 分层 review → 定位技术（最小复现/消融/单原语替换/差异属性）→ 升级切换 → **修正后验证** + 诊断纪律 | 验证失败进入 debug 状态时必读 |
| [references/vf-reduction-perf.md](references/vf-reduction-perf.md) | Vector reduction 的数值安全、正确 API 形态与性能注意事项 | 实现或调优 Vector reduction 时 |
| [references/cv-matmul-direct-buffering.md](references/cv-matmul-direct-buffering.md) | Cube 累加器在同一 launch 内被 Vector epilogue 消费时的轮转缓冲与 `auto_mutex` 形态 | 实现 CV 直连 matmul 时 |
| [templates/cv_matmul_direct_buffering.py.tmpl](templates/cv_matmul_direct_buffering.py.tmpl) | 上一行的可替换代码骨架 | 同上 |
| [templates/fp32-chain-precision-fragments.py.tmpl](templates/fp32-chain-precision-fragments.py.tmpl) | 历史精度实验片段；只供机制研究，不作为当前 API 或验收起点，实际实现按 kernel numerics 与完整示例重建 | 需理解旧精度实验时 |
| [scripts/list_idle_chip_ids.sh](scripts/list_idle_chip_ids.sh) | 查找空闲 NPU chip | 运行前按需执行 |

参考之间是互补关系：DESIGN.md 决定“实现什么”，当前 native example catalog 中契约匹配的完整 kernel/test 是“如何写”的主要起点，当前 kernel owner 规定技术约束；通用 Module 模板只提供文件组织骨架，历史模板不参与编码路由。模板和样例不是 API 或性能事实源，使用时仍须核对目标版本 API 文档；与 DESIGN.md 或当前技术规范冲突的参考片段直接弃用，只有上游合同本身无法同时落实时才按根因分流，本 skill 不自行改合同。标为 conceptual 的片段不得直接复制成交付代码。

## 开发流程

### 1. 锁定本轮范围

识别 dispatch 是否包含 `module_k` 或显式 `finalize=true`：

- **L0**：一次产出最终 `test_<op>.py`。
- **L1 Module**：只扩展当前 Module，但 staged 文件必须可独立运行，累积实现 Module 1..k。
  - 首次开发 Module k>1 时，复制上一个已验证文件生成新 suffix；重做当前 Module 时只修改当前 suffix。
  - 历史 staged 只读。
- **L1 finalize**：cleanup 尝试后由编排器显式调度。
  - cleanup 成功时校对脚本生成的最终文件；cleanup 失败时保留原始错误，以最后一个已验证 staged 为事实源重建最终文件，并只在最终文件中完成必要的交付修正，不得沿用旧 final。
  - verifier 重试时按原始证据修正现有最终文件。两种情况都只修改最终文件和本轮证据；不重开 Module、不改历史 staged、不重跑 cleanup。

L1 的 `suffix_k` 是累积序号：1 → `1`，2 → `12`，3 → `123`。文件和 wrapper 分别命名为 `test_<op>_module<suffix_k>.py` 与 `<op>_wrapper_module<suffix_k>`。确认前一 staged 文件和对应 `<op>_golden_stage<suffix_k>.py` 已存在；Module 1 没有前序文件。

### 2. 检查施工信息

读取 DESIGN.md 中的来源、具体适用条件、不变量、计划实现位置和原验证方法，核对所引用当前 owner 的范围；然后逐节检查：

- §0：I/O、动态维度、Module 和数据依赖；
- §1：API 序列及 Vector `vector_selection`；
- §2/§3：tile shape、dtype、layout、地址和各空间容量；
- §4：section 与循环；
- §5–§7：分核、流水、同步和尾块；
- §8：至少 4 个目标测试 case；
- §10：完整数据流。

L1 还要用 `module_interfaces.yaml` 核对当前 Module 的 `inputs`、`outputs`、`golden_steps` 和 section 类型。缺少关键合同，或上下游产物互相矛盾时停止编码并上报疑似 `design_violation`。

### 3. 核对全部 API

对 DESIGN.md §1 的每个 API 核对函数原型、位置/关键字参数、参数顺序、dtype、layout、MemorySpace 与目标平台支持：

1. 先读 EXPLORE_REPORT.md §3 的汇总结论；
2. 信息不足时，通过 PRO_MATERIAL_INDEX.md 定位并读取目标版本 API 原文；
3. 对不确定写法，在官方指定算子中找到同平台 working example。

形成覆盖本轮全部 API 的简短速查清单，后续直接复用。不要把编译错误直接解释为框架不支持。

### 4. 搭建 tile、section 与循环

把 DESIGN.md 的编译期常量放在 kernel 外部。逐条复制 tile 属性和地址，确认空间不重叠；按 §4–§6 搭建 section、SPMD 原语位置、循环和同步；按 §7 为整除与尾块路径预留正确的 valid-shape、填充和 store 处理。

- 纯 Vector（包括 L1 的 Vector Module）：从当前 example catalog 选择完整契约匹配的 kernel/test；归约问题可先定位 [row reduction](../pypto-pro-kernel/references/patterns/vf-row-reduce.md) 或 [outer reduction](../pypto-pro-kernel/references/patterns/vf-outer-reduce.md)。按当前 policy 使用 VF，只有明确允许的排序可用 tile compute。历史草稿不作为编码资源，不从旧实现的 wrapper 或 API 拼出未经验证的路径。
- 纯 Cube（包括 L1 的 Cube Module）：按 [当前 GEMM owner](../pypto-pro-kernel/references/patterns/gemm-desugar.md) 从 `pl.gemm`/`pl.gemm_mx` 开始，只有其明确例外才展开手写链；上表的官方 matmul 样例用于目标版本接口、tile、phase 和尾块取证，适配后仍须验证。
- CV 融合或 L1 分 Module 开发：每轮按当前 Module 的 Vector/Cube section 复用上述对应方式，在同一 kernel 内从 Module 1..k 逐步扩展。若 PRO_MATERIAL_INDEX.md §B 有与当前数据流匹配的官方 CV 融合样例，优先核对其 Module 衔接和同步写法。通用模板仅提供文件骨架，不把纯 Vector/Cube 完整算子直接拼接，也不为每个 Module 新建 kernel。

L1 的逐 Module 交付只规定开发与验证顺序，不等于运行时整段串行。连接相邻 Vector/Cube section 时，在当前 kernel owner 与 PRO_MATERIAL_INDEX.md 定位的目标版本 PyPTO-Pro 文档、源码和样例中查找与冻结 DESIGN.md 匹配的连接方式：有匹配方式就优先在本轮新 staged 文件中采用，使上下游处理不同 tile/基本块时能够重叠；没有则执行 DESIGN.md 的现有连接兜底。可以调整新文件中复制过来的前序 Module 连接实现，但不得改动历史 staged 文件，也不得改变数学、接口、支持范围或自行发明连接协议。

若本轮功能或精度测试证明 tile、地址、循环、同步或实现选择本身错误，按设计问题分流，不在代码中私自改合同。

### 5. 填写 kernel 实现

按 Module 把 DESIGN.md §1 的 API 序列翻译为代码，以 §10 校验每一步输入、输出和数据所有权。跨核同步点严格取自 §6；尾块严格取自 §7。转换、padding、索引与数值计算在 native computation 中完成。VF 局部同步按 [scratch-barrier 规则](references/vf-reduction-perf.md#scratch-barrier)核对完整依赖边，不得默认追加尾部 barrier；冻结设计缺少依赖两端、UB overlap 或 mode 证据时上报疑似 `design_violation`。

实现与任何冻结常量、算法步骤或布局不一致时，不得静默交付。先判断是抄录错误还是设计错误：前者修代码，后者上报疑似 `design_violation`。

### 6. 编写 wrapper、测试和实现证据

入口命名是硬合同：L0 与 L1 finalize 暴露 `<op>_wrapper`，签名和返回值符合算子 schema；L1 staged 暴露 `<op>_wrapper_module<suffix_k>`，输入采用 `primary_inputs`，输出当前 Module 的结果。optional 参数若可被调用方省略，必须提供相应默认值。wrapper 必须遵守实现合同 #4；若冻结 DESIGN 要求边界外操作，停止实现并上报疑似 `design_violation`。

TensorList（`is_list: true`）的参数、shape/stride、空工作与 work-item 映射执行 DESIGN.md，并核对 [当前有界集合接口](../pypto-pro-kernel/references/api/support.md#16-bounded-tensor-collections-under-a-fixed-arity-interface) 的适用边界。固定 arity 是须经目标 ABI 验证的候选，不新增单 launch 要求；不得自行发明指针打包协议或改写支持范围。

每轮静态核对当前 wrapper 的完整 host 调用链（含可达本地 helper、模块级/default/decorator 依赖），但不进入 kernel 函数体；kernel 调用前后都只允许实现合同 #4 的动作，外部调用无法确认属于允许集时不得交付。该检查不依赖 profile。

在同一文件实现 DESIGN.md §8 的全部 case：

- 至少覆盖整除、尾块和跨多 tile 等设计分支；每个 case 使用独立 `def test_...`。
- 从 `<op>_golden._get_device()` 获取设备，不硬编码卡号。
- 测试前把本 skill 的 `scripts/precision_compare.py` 复制到 `custom/<op>/`，用模板 `_assert_precision` 对比 CPU FP32 golden，不自定义阈值。
- `precision_compare` 与 `<op>_golden_cpu` 是 dev-only；import 放在函数体内，保证交付单元仅含 test 与 NPU golden 时可安全导入。
- L1 改为对比当前 `<op>_golden_stage<suffix_k>`；它同样属于 dev-only 依赖，必须在 `_assert_precision` 或测试函数体内 import，禁止顶层 import。

**实现与验证证据（Coder/Verifier 共用）**

按 DESIGN.md 的任务不变量逐项核对最终实现位置、适用条件和原验证方法，在本轮返回报告中记录真实 file/symbol、命令/输入、原始结果与结论。概念候选及历史范围不得冒充当前验证；不生成额外知识使用 JSON。

- L1 Module 只验证当前 staged 实际承载或 Module 合同分配的条件；后续 Module 的方法需要缺失依赖时，明确记录未执行的方法、原因和依赖，不能声称通过。
- L0/finalize 必须在最终公开 wrapper 上完成全部适用验证；已通过 staged 不能代替最终代码，最终修正后重做受影响验证。
- 仅源码、输入、方法和覆盖范围均未变化时复用原证据；保留命令和来源身份。同轮重做清楚区分旧结果与当前源码，不修改历史 staged。

### 7. 运行并调试

只使用当前环境运行本轮文件：

```bash
# L0 / L1 finalize
python custom/<op>/test_<op>.py

# L1
python custom/<op>/modules/test_<op>_module<suffix_k>.py
```

成功标准是全部 case 输出 PASS 且没有未解释告警。失败时读取 [debugging-methodology.md](references/debugging-methodology.md)，先确认失败可信，再按症状和层级定位；用最小复现或消融隔离根因，修复后重跑目标 case、相反分支和全量 case。

按根因返回或继续：

| 根因 | 动作 |
|---|---|
| 当前实现的代码翻译、参数或局部细节 | 修复本轮文件并重跑 |
| DESIGN 的维度、API 序列、tile、循环、同步、尾块、`vector_selection` 或 Module 合同 | 上报疑似 `design_violation`，附错误原文、最小复现和对应合同位置；不改冻结产物 |
| DESIGN 的来源、适用条件、不变量或验证方法错误、矛盾或不可执行 | 上报疑似 `design_violation`，附 owner/source、任务事实、合同位置和原始证据；仅环境阻断时报 `env_error`，不改冻结产物 |
| 已核对文档、官方样例并穷尽 DESIGN 允许路径后确认框架能力缺口 | 返回 `capability_gap`，附目标版本、原始错误、尝试路径和各自失败证据；不在 host 端绕过 |
| 导入、CANN、设备不可见或疑似 hang | 加载 `pypto-pro-environment-check` 按其流程评定并把证据交给编排器；不自行改环境 |

### 8. 交付前自检

交付前执行一次紧凑自检：

- 产物路径和 L0/L1 命名正确；L1 staged 链保留。
- module-level entry、tiling key、阶段序列与完整 wrapper 符合当前交付合同及 DESIGN。
- 实现逐项符合 DESIGN.md，Vector 使用冻结的 `vector_selection`。
- L1 只修改本轮新 staged 文件，历史 staged 文件未变；异构 Module 连接符合 DESIGN.md 和所用参考。
- 当前模式的最终/staged 代码与方法证据满足上方「实现与验证证据」。
- wrapper 的完整 host 调用链符合实现合同 #4。
- DESIGN.md §8 全部 case 已实际运行并通过；测试数据、dtype、shape、value range 未被偷换。
- dev-only import 位于函数内，交付态模块可安全导入。
- 返回运行命令、逐任务不变量的方法证据、原始结果、产物路径和分类 verdict；不声称 verifier PASS。若无需修改，明确说明并附本模式全部产物的检查证据，不得空返回或只给笼统结论。
