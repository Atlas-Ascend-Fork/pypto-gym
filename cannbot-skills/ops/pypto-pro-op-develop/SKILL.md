---
name: pypto-pro-op-develop
description: 实现、调试并自验证 PyPTO-Pro 算子 kernel。用于按冻结的 DESIGN、DESIGN_BINDINGS 和 Module 合同完成 L0（非融合的纯 Vector/纯 Cube，一次交付）或 L1（Vector/Cube 融合，逐 Module staged 后 finalize）实现与 KB_USAGE 记录；发现上游、能力或环境问题时上报有证据的分类报告，不修改上游产物或编排状态。
---

# PyPTO-Pro 算子 Kernel 实现

把已冻结的 `DESIGN.md` 翻译为一个可运行、可测试的 PyPTO-Pro kernel，并逐条落实 `DESIGN_BINDINGS.json` 中的 active requirements。完成本地开发与自验证闭环后交给独立 verifier；本 skill 不修改 SPEC、DESIGN、DESIGN_BINDINGS、Module 合同或编排状态。

## 输入与输出

先读取以下输入：

- `custom/<op>/DESIGN.md`：冻结的施工合同，包含 Module、API、tile、地址、循环、同步、尾块和测试设计。
- `custom/<op>/DESIGN_BINDINGS.json`：上游冻结的 KB 要求与设计落点（只读）。
- `custom/<op>/module_interfaces.yaml`：L1 当前 Module 的输入来源、输出与 `golden_steps`。
- `custom/<op>/EXPLORE_REPORT.md`：已核对的 API 约束和相似样例。
- `custom/<op>/PRO_MATERIAL_INDEX.md`：需要回看原文时的 API、官方样例和教程路径。
- flat 的 `custom/<op>/KB_SELECTION.json`，或 split 布局下全部 `custom/<op>/<class>/KB_SELECTION.json`，以及其中选中的所有参考。

性能原则见 `$CANNBOT_CONFIG_ROOT/references/kernel-core-principles.md`。实现期直接相关的是
`P4`（Device 侧循环内零除法 / 取模，地址按 `addr += stride` 推进）、`P2`（轮转 buffer 用 TileGroup，
深度改动须复核 `T1`）、`X1`（CV 错拍不得退化为逐 tile 串行）。收尾时按 `kernel-perf-pypto-pro-a5.md` §2 的自检清单（`G1`–`G6`）自查，结论写进本轮报告；没测过的项记 `unknown`（`M1`）。

按 dispatch 交付：

| 路径 | dispatch | 本轮产物 |
|---|---|---|
| L0 | 无 `module_k` 且无 `finalize` | `custom/<op>/test_<op>.py`，并生成或修正各 class 的 `KB_USAGE.json` |
| L1 | 带 `module_k` | `custom/<op>/modules/test_<op>_module<suffix_k>.py`；可更新对应 class 目录的 `KB_USAGE.json`，记录指向该 staged 文件 |
| L1 finalize | `finalize=true`，无 `module_k` | cleanup 尝试后完成 `custom/<op>/test_<op>.py`，并生成或修正各 class 的 `KB_USAGE.json` |

## 实现合同

施工时始终满足以下本阶段职责：

1. 一个交付文件只含一个 `@pl.jit` kernel，核心计算全部在该 kernel 内；wrapper 只启动一次 kernel，且启动不在 host 循环内。
2. 严格执行 DESIGN.md 已冻结的 Module 边界、API 序列、tile 属性/地址、循环、同步、尾块和 `vector_selection`。本轮功能或精度测试表明设计有误时上报疑似 `design_violation`，不得静默改设计。
3. 轮转 tile 使用 `make_tile_group` + `auto_mutex`；`make_tile` 只用于不参与轮转的单次 scratch。不要在 `auto_mutex` 管理的 tile 上叠加手动 `sync_src`/`sync_dst`。
4. wrapper 只做参数检查、读取 KB 约束列明的只读元数据、纯 Python 整数推导、`torch.empty` 分配当前 wrapper 合同声明的输出和一次 kernel 启动；完整边界与迁移方式见 [wrapper-boundary.md](../pypto-pro-op-kb/constraints/wrapper-boundary.md)。DESIGN、usage、`deviated` 或 profile 均不能放宽该硬约束。
5. 测试通过 wrapper 调用 kernel；不得删改 DESIGN.md §8 的 case 来迁就实现，也不得把核心计算移到测试或 host 代码。
6. **上下文效率**：对于已加载且未发生变更的文件，尽量按需读取，避免全量重读。重复读取、冗余思考复述等无效上下文属于无效开销，会增加Token消耗，同时导致注意力稀释。

## 按需读取的资源

本表是本 skill 资源索引的唯一来源（含加载时机）。

| 资源 | 用途 | 加载时机 |
|---|---|---|
| [templates/pure_vec_impl_template.py.tmpl](templates/pure_vec_impl_template.py.tmpl) | 纯 vec 通用骨架；使用前按目标 API 与 DESIGN.md 核对 | 纯 vec 算子生成代码前 |
| `$PYPTO_DEVKIT_DIR/pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` | **纯 cube 算子候选实现起点**——两级 K 分块 + move offset + 嵌套四分支 K 累加 + 尾块处理 + ASW 蛇形调度；仅在该路径存在且目标版本匹配时使用 | 纯 cube 算子生成代码前 |
| [templates/impl_template.py.tmpl](templates/impl_template.py.tmpl) | kernel 文件骨架（tile 声明 + section + Module + 测试函数）——**通用**（含 CV 融合 / 多 Module / 跨核流水）；CV 融合算子当前无专用模板，走此通用骨架，按「开发流程」各步执行 | 生成代码前必读 |
| [references/debugging-methodology.md](references/debugging-methodology.md) | 调试方法论：**先确认失败可信** → 症状快查表 → 分层 review → 定位技术（最小复现/消融/单原语替换/差异属性）→ 升级切换 → **修正后验证** + 诊断纪律 | 验证失败进入 debug 状态时必读 |
| [references/vf-reduction-perf.md](references/vf-reduction-perf.md) | Vector reduction 的数值安全、正确 API 形态与性能注意事项 | 实现或调优 Vector reduction 时 |
| [references/cv-matmul-direct-buffering.md](references/cv-matmul-direct-buffering.md) | Cube 累加器在同一 launch 内被 Vector epilogue 消费时的轮转缓冲与 `auto_mutex` 形态 | 实现 CV 直连 matmul 时 |
| [templates/cv_matmul_direct_buffering.py.tmpl](templates/cv_matmul_direct_buffering.py.tmpl) | 上一行的可替换代码骨架 | 同上 |
| [templates/fp32-chain-precision-fragments.py.tmpl](templates/fp32-chain-precision-fragments.py.tmpl) | fp32 链路的精度安全片段 | 需要与 CPU 参考逐位对齐时 |
| [templates/kb-usage-template.json](templates/kb-usage-template.json) | `KB_USAGE.json` 字段骨架与单条记录示例 | 写入或核验 usage 时 |
| [scripts/list_idle_chip_ids.sh](scripts/list_idle_chip_ids.sh) | 查找空闲 NPU chip | 运行前按需执行 |
| [KB CONTRACT](../pypto-pro-op-kb/CONTRACT.md) | KB JSON 的基础字段、路径与状态词表 | 读取 selection 或写 usage 前 |

参考之间是互补关系：DESIGN.md 决定“实现什么”，纯 Vector 模板或 Cube 官方样例提供“如何写”的主要起点，`KB_SELECTION.json` 已选参考补充必须落实的 pattern 和约束；三者不得相互替代。模板和样例不是 API 或性能事实源，使用时仍须核对目标版本 API 文档；与 DESIGN.md 或已选 KB 冲突的参考片段直接弃用，只有上游合同本身无法同时落实时才按根因分流，本 skill 不自行改合同。标为 conceptual 的片段不得直接复制成交付代码。

## 开发流程

### 1. 锁定本轮范围

识别 dispatch 是否包含 `module_k` 或显式 `finalize=true`：

- **L0**：一次产出最终 `test_<op>.py`。
- **L1 Module**：只扩展当前 Module，但 staged 文件必须可独立运行，累积实现 Module 1..k。
  - 首次开发 Module k>1 时，复制上一个已验证文件生成新 suffix；重做当前 Module 时只修改当前 suffix。
  - 历史 staged 只读。
- **L1 finalize**：cleanup 尝试后由编排器显式调度。
  - cleanup 成功时校对脚本生成的最终文件；cleanup 失败时保留原始错误，以最后一个已验证 staged 为事实源重建最终文件，并只在最终文件中完成必要的交付修正，不得沿用旧 final。
  - verifier 重试时按原始证据修正现有最终文件。两种情况都只修改最终文件和 usage；不重开 Module、不改历史 staged、不重跑 cleanup。

L1 的 `suffix_k` 是累积序号：1 → `1`，2 → `12`，3 → `123`。文件和 wrapper 分别命名为 `test_<op>_module<suffix_k>.py` 与 `<op>_wrapper_module<suffix_k>`。确认前一 staged 文件和对应 `<op>_golden_stage<suffix_k>.py` 已存在；Module 1 没有前序文件。

### 2. 检查施工信息

分别读取两份冻结合同：

- `DESIGN_BINDINGS.json`：遍历全部 binding 和 requirement，提取四元组与 `source_anchors[]`；对 active requirement（`obligation + applies`）另提取 `invariant`、`planned_location`、`verification_method`，对 validation scope 另提取验证范围、`class_evidence` 和 `verification_method`。
- `DESIGN.md`：用 `planned_location` 定位对应设计，再核对：
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

- 纯 Vector（包括 L1 的 Vector Module）：以 `pure_vec_impl_template.py.tmpl` 为主要编码参考。`vector_selection: vf` 时只保留一个匹配当前计算依赖的 Pattern，删除其他 Pattern；`vector_selection: tile_op` 时只借用模板中与当前合同一致的文件、tile/section 和测试组织结构，计算写法严格执行 DESIGN.md 和已选 KB pattern。
- 纯 Cube（包括 L1 的 Cube Module）：以上表的目标版本官方 matmul 样例为主要编码参考，使用其 tile、K 循环、phase 和尾块的已验证写法，再按 DESIGN.md 调整并落实已选 KB 约束。
- CV 融合或 L1 分 Module 开发：每轮按当前 Module 的 Vector/Cube section 复用上述对应方式，在同一 kernel 内从 Module 1..k 逐步扩展。若 PRO_MATERIAL_INDEX.md §B 有与当前数据流匹配的官方 CV 融合样例，优先核对其 Module 衔接和同步写法。通用模板仅提供文件骨架，不把纯 Vector/Cube 完整算子直接拼接，也不为每个 Module 新建 kernel。

L1 的逐 Module 交付只规定开发与验证顺序，不等于运行时整段串行。连接相邻 Vector/Cube section 时，只在 `KB_SELECTION.json` 已选参考或 PRO_MATERIAL_INDEX.md 定位的目标版本 PyPTO-Pro 文档、源码和样例中查找与冻结 DESIGN.md 匹配的连接方式：有匹配方式就优先在本轮新 staged 文件中采用，使上下游处理不同 tile/基本块时能够重叠；没有则执行 DESIGN.md 的现有连接兜底。可以调整新文件中复制过来的前序 Module 连接实现，但不得改动历史 staged 文件，也不得改变数学、接口、支持范围或自行发明连接协议。

若本轮功能或精度测试证明 tile、地址、循环、同步或实现选择本身错误，按设计问题分流，不在代码中私自改合同。

### 5. 填写 kernel 实现

按 Module 把 DESIGN.md §1 的 API 序列翻译为代码，以 §10 校验每一步输入、输出和数据所有权。跨核同步点严格取自 §6；尾块严格取自 §7。所有 dtype 转换、轴变换、padding、索引和计算都在 kernel 内完成。VF 局部同步按 [scratch-barrier 规则](references/vf-reduction-perf.md#scratch-barrier)核对完整依赖边，不得默认追加尾部 barrier；冻结设计缺少依赖两端、UB overlap 或 mode 证据时上报疑似 `design_violation`。

实现与任何冻结常量、算法步骤或布局不一致时，不得静默交付。先判断是抄录错误还是设计错误：前者修代码，后者上报疑似 `design_violation`。

### 6. 编写 wrapper、测试和 KB 使用记录

入口命名是硬合同：L0 与 L1 finalize 暴露 `<op>_wrapper`，签名和返回值符合算子 schema；L1 staged 暴露 `<op>_wrapper_module<suffix_k>`，输入采用 `primary_inputs`，输出当前 Module 的结果。optional 参数若可被调用方省略，必须提供相应默认值。wrapper 必须遵守实现合同 #4；若冻结 DESIGN 要求边界外操作，停止实现并上报疑似 `design_violation`。

TensorList（`is_list: true`）同样只能启动一次 kernel，且启动不得位于 host 循环内。参数展开、固定 arity、地址/shape 传递与 work-item 映射必须原样执行 DESIGN.md 及 `KB_SELECTION.json` 已选 TensorList pattern；本 skill 不自行发明新的打包协议或改写支持范围。

每轮静态核对当前 wrapper 的完整 host 调用链（含可达本地 helper、模块级/default/decorator 依赖），但不进入 kernel 函数体；kernel 调用前后都只允许实现合同 #4 的动作，外部调用无法确认属于允许集时不得交付。该检查不依赖 profile。

在同一文件实现 DESIGN.md §8 的全部 case：

- 至少覆盖整除、尾块和跨多 tile 等设计分支；每个 case 使用独立 `def test_...`。
- 从 `<op>_golden._get_device()` 获取设备，不硬编码卡号。
- 测试前把本 skill 的 `scripts/precision_compare.py` 复制到 `custom/<op>/`，用模板 `_assert_precision` 对比 CPU FP32 golden，不自定义阈值。
- `precision_compare` 与 `<op>_golden_cpu` 是 dev-only；import 放在函数体内，保证交付单元仅含 test 与 NPU golden 时可安全导入。
- L1 改为对比当前 `<op>_golden_stage<suffix_k>`；它同样属于 dev-only 依赖，必须在 `_assert_precision` 或测试函数体内 import，禁止顶层 import。

生成或核验 `KB_USAGE.json` 时，以 [模板](templates/kb-usage-template.json) 为骨架，基础字段、路径和状态词表遵循 [KB CONTRACT](../pypto-pro-op-kb/CONTRACT.md)；本节只规定实现环节可写的记录范围、状态子集、数量和生命周期。替换全部 `{...}`；`schema_version` 的整段占位字符串（含引号）必须换成 `topology-map.json` 当前 `contract.contract_version` 的整数，不得写死版本。

**KB usage 规范（Coder/Verifier 共用）**

- **空集**：仅当 selection 引用并集为空时，`DESIGN_BINDINGS.json.bindings` 和最终各 class 的 `KB_USAGE.json.invariants` 均为 `[]`；否则 `bindings` 不得为空。
- **范围**：仅 `obligation + applies` 生成 usage；其他 requirement 零记录。validation scope 不生成记录，但仍限制复用：conceptual/unverified 不得冒充 validated，局部结论不得外推。
- **记录**：按模板为“一个 active × 一个实现产物”复制一项。`invariant` 前缀固定为 `[{selection_field}:{req_id_json} | {source_anchors_json}]`；`req_id_json` 和 `source_anchors_json` 分别用标准库 `json.dumps(..., ensure_ascii=False)` 生成后填入，不能自行转义、重排/去重或用分隔符拼接 anchors。四元组由 usage 根级 `class_id`、记录 `reference` 及前缀中的 `selection_field`、`req_id` 还原；`source_anchors[]` 只作证据，不参与身份或拆并。`implementation.status` 默认为 `implemented`；仅上游 DESIGN 已冻结偏离时可改为 `deviated`，并在与 `implementation` 同级的 `justification` 写入同一理由。不得写 `verified` / `not_applicable`。

| 模式 | 记录要求 |
|---|---|
| L0 | 每条 active 恰好一条 final，不得有 staged |
| L1 Module | staged 可不记录；若记录，同一 active 在同一文件至多一条 |
| L1 finalize | 保留真实且已通过 module-check 的 staged，清除 stale；每条 active 恰好一条 final |

- **生命周期**：final 必须指向 `custom/<op>/test_<op>.py` 的真实 file/symbol，staged 必须指向相应 `custom/<op>/modules/test_<op>_module<suffix_k>.py` 的真实 file/symbol；两者 symbol 可不同，staged 不能替代 final。仅首次进入实现流程或上游产物变化后重入时清空全部 usage；同轮重做只清本轮产物的 stale 记录，保留其他已通过 module-check 的 staged 记录；从后续优化流程回退时保留合法 staged。

- **验证范围**：L1 Module 只检 DESIGN/Module 合同分配给当前 Module，或当前 staged 实际承载的 active 及相关 validation scope，不提前检后续 Module；L0/finalize 检全部 active 与 scope。
- **验证方法**：不得改变验证目标，输入就绪即执行；仅 L1 Module 可在原方法明确依赖尚未生成的最终 file/symbol、wrapper 或 final-only profile 时暂缓，并在本轮返回证据中记录四元组、原方法、缺失依赖和 `not_run_for_staged`。L0/finalize 不得暂缓。
- **证据复用**：仅被检实现、验证输入、方法、检查内容和覆盖范围均未变化时复用；仍按四元组记录原方法、命令/输入、原始结果和结论。已存在或本轮写入的 usage 均用 `json.load` 预检；L0/finalize 的各 class usage 必须存在。

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
| selection 的布局、唯一键、引用或适用性错误（含 optional pattern 无独立作用、required constraint 漏选），或实现依赖未选参考 | 上报疑似 `kb_selection_invalid`，附 class、引用与事实；不改 selection，也不只在 usage 中补路径 |
| DESIGN 的维度、API 序列、tile、循环、同步、尾块、`vector_selection` 或 Module 合同 | 上报疑似 `design_violation`，附错误原文、最小复现和对应合同位置；不改冻结产物 |
| selection 有效，但 Binding 的状态、不变量、`planned_location` 或 `verification_method` 错误、矛盾或不可执行 | 上报疑似 `design_violation`，附可获得的四元组、`source_anchors[]`、class 事实和合同位置；仅环境阻断时报 `env_error`，不改冻结产物 |
| 已核对文档、官方样例并穷尽 DESIGN 允许路径后确认框架能力缺口 | 返回 `capability_gap`，附目标版本、原始错误、尝试路径和各自失败证据；不在 host 端绕过 |
| 导入、CANN、设备不可见或疑似 hang | 加载 `pypto-pro-environment-check` 按其流程评定并把证据交给编排器；不自行改环境 |

### 8. 交付前自检

交付前执行一次紧凑自检：

- 产物路径和 L0/L1 命名正确；L1 staged 链保留。
- `@pl.jit` 恰好一个，wrapper 启动 kernel 恰好一次且不在循环内。
- 实现逐项符合 DESIGN.md，Vector 使用冻结的 `vector_selection`。
- L1 只修改本轮新 staged 文件，历史 staged 文件未变；异构 Module 连接符合 DESIGN.md 和所用参考。
- 当前模式的代码、usage 和方法证据满足上方「KB usage 规范」。
- wrapper 的完整 host 调用链符合实现合同 #4。
- DESIGN.md §8 全部 case 已实际运行并通过；测试数据、dtype、shape、value range 未被偷换。
- dev-only import 位于函数内，交付态模块可安全导入。
- 返回运行命令、逐四元组方法证据、原始结果、产物路径和分类 verdict；不声称 verifier PASS。若无需修改，明确说明并附本模式全部产物的检查证据，不得空返回或只给笼统结论。
