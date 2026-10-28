---
name: pypto-pro-op-design
description: 设计 PyPTO-Pro 算子的 tile 级执行方案。当 SPEC、Golden 与资料探索产物齐全，且需要确定 Module 与 API 数据流、Tile 和片上内存规划、循环与 Section、分核与同步、动态尾块和数值边界，或产出、审查有证据支撑的 DESIGN.md 时使用；同时产出 DESIGN_BINDINGS.json 与 module_interfaces.yaml。不要编写 kernel。
---

# PyPTO-Pro 迭代式方案设计

通过 9 轮迭代式约束收敛（R0-R8），**目标** 是生成可直接翻译为 kernel 代码的 DESIGN.md。

**核心原则**：
- 每个决策必须包含**结论 + 推导过程 + 证据来源**
- 力求后续 Agent 拿到 DESIGN.md 即可确定 kernel 的完整结构与关键决策；API 签名等细节仍须由 coder 以 API 文档原文为准确认（EXPLORE_REPORT 仅为派生的先行速查，不作签名权威）。设计验收通过后 DESIGN.md 成为实现合同；运行证据推翻设计时须返回 `design_violation`，由编排器重新调度设计修订
- 每轮发现的矛盾必须回溯修正前序决策，不允许累积到 R8 再处理
- 本 skill 以**思维方法指导**为主，不教具体写法——具体 API 用法、tile 配置、同步写法等请查阅 API 文档（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/`）、教学文档（`$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/`、`$PYPTO_DEVKIT_DIR/docs/guide/quick_start/pro/` 与 `$PYPTO_DEVKIT_DIR/docs/guide/introduction.md`）、官方指定算子（见 `PRO_MATERIAL_INDEX.md` §B），理解后据实设计

## 两条实现约束（设计阶段须落实）

> 本节定义本 Skill 的实现约束；`$CANNBOT_CONFIG_ROOT/references/performance-constraints.md` 提供同步与 VF 的 API 说明。进入 R0 前读取该说明，在 R3（地址分配）和 R1（API 映射）中落实以下约束：
> 1. 所有需要 buffer 切换/轮转的 tile 一律用 `make_tile_group` + `auto_mutex`，`make_tile` 仅限单次使用 scratch tile。手动同步的适用范围见[跨核同步](references/cross_core_synchronization.md)。R3 落实 buffer 管理方式，R6 落实 cross_core 同步方案。
> 2. Vector 选择按本节写入 DESIGN.md §1：已选 KB 模板明确要求当前步骤使用 `pl.*` 时按模板，否则使用 `vf.*`；本阶段不运行候选实验。仅展示 tile-op 写法、官方样例、同名 API 或性能猜测不构成例外。

## 性能原则（设计阶段须落实）

根规范：`$CANNBOT_CONFIG_ROOT/references/kernel-core-principles.md`，进入 R0 前读完（约 150 行）。
各轮须落实的编号：

| 轮次 | 落实 |
|---|---|
| R0 前 | `B1` 代价模型、`B2` 绑定资源、`B6` 目标上限——先判绑定资源，再做后续决策 |
| R2 / R3 | `T1` 逐空间容量 × 深度、`Q1`/`Q3` Cube tile、`T3` Vector `elements_per_pass` |
| R4 / R6 | `P2` 搬运与计算重叠、`X1` CV 错拍、`X2` 两级配平、`X4` 在线化归约 |
| R5 | `P3` 按工作量分核与 `η_tail`、`P4` Host 侧切分 |
| R8 | 把性能结论写进 DESIGN.md §11，每项判定标注所依据的编号 |

## 知识库

知识来源与读取边界见下方「结构化 Binding 流程（architect）」。
[pattern selector](../pypto-pro-op-kb/patterns/pattern-index.md) 仅用于确认已选 pattern 的验证
状态：`validated skeleton` 可作为代码起点，`conceptual only` 只能用于推导，不得据此扩选。

>
> **`validated skeleton` 保证代码存在，不保证它在你的工作区。** 本仓把知识库与
> 验证它的算子树拆在不同分支：pattern 页会写明产物留存在哪个分支。把它当作代码起点
> 之前，先确认能取到——`git cat-file -e <branch>:<path>`，取不到就当 `conceptual only` 用，
> 不要凭页面描述照写。
平台专属约束必须先探测目标平台。API 文档与当前环境的官方样例仍是签名和行为的权威来源。

## 输入

| 来源 | 路径 | 用途 |
|------|------|------|
| 算子规格 | `custom/<op>/SPEC.md` | 公式、shape、dtype、动态轴；末尾「kernel 契约补充」节的辅助张量暂存 / cast 边界链 / 累加语义（分别喂给 R2 tile 规划 / §1 数值边界 / R6 同步），及标注「留待 design」的 MAY-DESIGN 字段 |
| Golden 参考 | `custom/<op>/{op}_golden.py` | 函数签名参考（影响 R7.5 测试 case 规划）；增量验证模式下需暴露中间量辅助函数（如 `{op}_golden_stage1`），设计阶段应知晓此依赖 |
| 资料探索报告 | `custom/<op>/EXPLORE_REPORT.md` | API 映射与约束（§3）、相似样例与可复用模式（§4）、教程设计指导（§5）、Tile/同步策略建议（§6）、环境常量快照（§7：UB 容量/event_id 上限/对齐要求等） |
| 全量资料索引 | `custom/<op>/PRO_MATERIAL_INDEX.md` | API 文档（§A）、官方指定算子（§B）、教程（§C）的精确路径定位 |
| 冻结知识选择 | 按 KB `CONTRACT.md`「Artifact locations」发现的全部 `KB_SELECTION.json` | Planner 的只读交接产物；按本 Skill 的结构化 Binding 流程逐 class 消费全部选中引用 |
| KB 产物合同 | `$CANNBOT_CONFIG_ROOT/pypto-pro-op-kb/CONTRACT.md` | 只用于确认 `KB_SELECTION.json` 的字段与 KB 相对路径格式；设计行为由本 Skill 定义 |

## 输出

- **`custom/<op>/DESIGN.md`**，基于 [templates/design-template.md](templates/design-template.md)，并短链接到结构化 Binding；核心交付物为 §10“Tile 数据流全景图”
- **`custom/<op>/DESIGN_BINDINGS.json`**，基于 [templates/design-bindings-template.json](templates/design-bindings-template.json)，并遵守下方结构与语义合同；本 Skill 是唯一规范源，模板只提供 JSON 骨架
- **`custom/<op>/module_interfaces.yaml`**，记录 Module 机器可读契约

### `DESIGN_BINDINGS.json` 结构合同

下列字段均必填，且不得增加其他字段。

| 层级 | 固定字段 |
|---|---|
| 根对象 | `schema_version`（整数 `1`）、`bindings` |
| Binding | `class_id`、`selection_field`、`reference`、`selection_reason`、`requirements` |
| requirement | `req_id`、`kind`、`source_anchors`、`status`、`class_evidence`、`invariant`、`planned_location`、`verification_method` |

- `schema_version` 必须是整数 `1`；`bindings` 必须是数组。
- 除 `requirements` 外的四个 Binding 字段是非空字符串；`selection_field` 只能是 `optional_patterns` 或 `required_constraints`，`requirements` 是非空数组。
- requirement 的 `req_id`、`class_evidence`、`verification_method` 是非空字符串；`source_anchors` 是由非空、不重复字符串组成的非空数组。
- `kind + status` 只允许 `precondition + met`、`obligation + applies`、`obligation + not_triggered`、`validation_scope + applies`。`obligation + applies` 是活动项：每个 Binding 至少一条；其 `invariant`、`planned_location` 为非空字符串，其他项的这两个字段可为非空字符串或 `null`。

## 结构化 Binding 流程（architect）

进入 R0 前完成第 1–3 步并形成临时要求清单；第 4 步随 R0–R8 更新；交付前执行第 5 步。

1. **读取冻结 selection**：按 KB `CONTRACT.md`「Artifact locations」找到唯一合法的 flat 或 split 布局，读取所有 `KB_SELECTION.json` 及其中数量不限的 `optional_patterns`、`required_constraints`。selection 只读：只消费选中引用，不重查 KB，不增删、替换或改写。引用键是 `(class_id, selection_field, path)`；flat/split 并存、布局/`class_id`/引用无效或键重复时直接报错，不得归一化或去重后继续。API 签名和平台行为以官方文档与样例为准。
2. **先读原文，逐项盘点（source-first）**：按原文顺序扫描标题辖区、段落、列表、每行表格、代码块、注释和公式，不依赖规范关键词或告警措辞。凡删除后可能改变正确性、支持/适用前提（target/version/dtype/shape/layout）、数据流/API、tile/地址、循环位置/分核、buffer/sync/event、边界、精度/cast 顺序、对齐/容量/索引、buffer lifetime、验证/复用或性能机制的内容，都要收录；silent failure、危险边界和 fallback/条件分支也不能遗漏。仅排除历史叙述、重复原理、纯示例及不影响本 class 决策的 benchmark；清单不落盘。
3. **拆分要求并确定最窄范围**：可独立失败或验证的要求必须拆开；仅同义要求可合并，`source_anchors` 仍按原文顺序保留全部标题路径和条目短标签，不用行号。验证范围取声明明确指向的最窄语义范围。局部 `conceptual only` / `unverified` / exclusion 只覆盖其明确指向且与页面级 `validated` 重叠的范围，不向外扩散。同范围证据冲突时按 `unverified` 处理、不复用 skeleton，并在 `validation_scope` 中保留双方锚点和证据。
4. **生成 Binding 并落实设计**：按上述合同生成 `custom/<op>/DESIGN_BINDINGS.json`。Binding 键必须与 selection 引用键一一对应且不重复（exact + unique）：全部 selection 的引用并集为空时，且仅此时写 `bindings: []`；否则 `reference`、`selection_reason` 分别原样复制 `path`、`reason`。同一路径跨 class/field 时分别保留；组内 `req_id` 唯一，requirement 原子键是 `(class_id, selection_field, reference, req_id)`。

   整体必要前提用 `precondition + met`，任一不成立即为 selection 错误；互斥/可选分支不要求同时成立。未触发义务仍用 `obligation + not_triggered` 保留 class 事实；`validation_scope + applies` 的 `class_evidence` 记录最窄范围、复用结论和冲突证据（如有）。每条活动项必须给出具体且可落到代码的 `invariant`、`planned_location` 和可执行的 `verification_method`；`planned_location` 同时指向 DESIGN 决策与最终 `test_<op>.py` file/symbol，staged 位置只能补充。每个 optional pattern 须有独立设计作用；required constraint 的活动义务即使重叠也不得丢失。`DESIGN.md` 只链接 JSON，不复制 Binding 表。
5. **自检并按根因报错**：确认 JSON 可解析，并复核上述结构、键映射、唯一性和设计落实；把 requirements 与 source-first 清单双向对照，拒绝遗漏、无来源新增、错误合并、锚点丢失/不可定位、状态与 class 事实不符或活动项字段错误。

   布局/`class_id`/引用无效或键重复、整体必要前提不成立、optional pattern 无独立作用/作用重复，或 SPEC/官方资料/已选引用依赖未选 KB 路径时，报 `failure_category: kb_selection_invalid`。selection 有效，但 JSON 合同/覆盖、requirement 或设计落实错误时，报 `failure_category: design_violation`。报告包含可获得的 `class_id`、问题引用、适用时的 `source_anchors`、原因和客观证据；不得修改 selection 掩盖缺口或加入与本 class 无关的通用样板。

---

## 前置：维度契约

在切 Module 之前先确认 kernel 的输入/输出维度契约——这是后续所有轮（Tile 规划、循环结构）的前提，不属于 Module 划分本身。

按 SPEC 明确 kernel 接收的真实 rank、shape、stride 和输出形态。SPEC 要求支持 1D 或任意多维时，必须设计 kernel 内的索引/stride 映射，不能让 host 通过 reshape 等张量操作把输入归一化，也不能默默只实现 2D。产出填入 DESIGN.md §0 的维度契约栏（真实输入 + kernel 内适配规则）。

---

## 迭代设计流程

设计是**问题驱动**的迭代，不是线性填表。每一轮聚焦一个核心问题（回溯原则见核心原则第 3 条）。

### R0：Module 划分

**核心问题**：整体计算流可以分解为几个 Module？每个 Module 做什么？

> 📌 **权威依据（必读，官方标准）**：读取[Module划分](references/module_partitioning.md)。Module划分分两步：先按数据依赖确定边界，再按Section边界继续拆分。

**输出**：

- Module 列表（Module 1/2/3...），每个标注：目的、输入依赖、输出、涉及的 Section
- 如果只有一个 Module，标注"单 Module"并说明为什么不需要拆分
- **`module_interfaces.yaml` 契约**（机器可读，single source of truth）——产出到 `custom/<op>/module_interfaces.yaml`，包含以下字段：
  - `module_count`：Module 总数
  - `is_fusion`：是否为融合算子（同时含 cube 和 vec section → `true`）。按 R0 定义，`is_fusion=true` 隐含 `module_count >= 2`（Cube 和 Vector 必须划分到不同 Module）；本 Skill 只记录设计事实，不决定后续编排路径
  - `has_cross_core`：是否涉及 cross_core 跨核流水（来自 §6，信息记录用，不影响分流判据）
  - `modules[]`：每个 Module 的 `id` / `name` / `description` / `section`（`cube` 或 `vector`）/ `golden_steps`（该 Module 对应的数学步骤列表，供 mathematician 切分 golden 用）/ `inputs`（source 为 `primary` 或 `module_<j>`，`j < 当前 id`）/ `outputs` / `golden_stage_fn`
  - `final_outputs`：每个 golden 返回值对应到产出 Module
  - `composition_verification`：atol / rtol / seeds / shapes
  - 骨架由脚本生成：`python ./scripts/gen_module_interfaces.py custom/<op>/<op>_golden_cpu.py --spec custom/<op>/SPEC.md --op <op> --design custom/<op>/DESIGN.md > custom/<op>/module_interfaces.yaml`，自动填 `schema_version` / `op` / `primary_inputs` / `composition_verification`，architect 填标 `TODO` 的判断部分
  - 产出后须自验：`python ./scripts/validate_module_yaml.py custom/<op>/module_interfaces.yaml --json`，返回 `"status": "PASS"` 才算完成

---

### R1：API 映射

**核心问题**：每个数学步骤具体用哪些 API？在 R0 的 Module 划分基础上，将 API 调用链细化到 Module 内部每一步操作。

> 📌 **权威依据（必读，官方标准）**：读取[API映射与Tile规划](references/api_mapping_and_tile_planning.md)中的“API映射”一节。数学步骤拆分、Cube/Vector执行域、完整调用链和数值安全边界按该节执行；单个接口的签名和约束以对应API参考页为准。Vector实现层级按本Skill“两条实现约束”中的KB模板选择规则确定。

**DESIGN.md §1须记录的设计内容**：
- 按R0 Module划分展开的完整API调用链，包括数据搬运、dtype/layout转换、广播、临时空间和尾块操作
- 每个API的输入、输出及临时Tile，以及dtype、shape、MemorySpace、layout和地址重叠限制
- 每个Vector步骤唯一的`vector_selection`；已选KB模板明确要求使用`pl.*`时选择`tile_op`，否则选择`vf`
- 涉及非线性函数、窄dtype量级增长或长轴归约时的数值安全边界和处理方案

> **A5 L0C→UB数据通路**：在950/A5上，Vector需要继续处理Cube结果时，先查阅目标版本`memory_data_movement/move.md`和`basic_data_structures/AccToVecMode.md`，确认`pl.move`是否支持当前MemorySpace、dtype、layout、shape和尾块。支持时优先从Acc(L0C)直接搬到Vec(UB)：两个Vector subblock能够独立处理时，沿非归约轴拆分并尽量均衡负载；存在归约依赖时，使用单个Vector subblock，或在R5/R6明确设计部分结果的合并与同步。不支持时改用GM workspace通路。
>
> DESIGN.md §1须记录所选数据通路、`AccToVecMode`、两个Vector subblock的数据范围和尾块处理；片上容量在R3统一校验。

EXPLORE_REPORT.md §3只用于提供候选映射。逐项核对API参考页后再确定调用链；接口无法满足算子语义时，回退本轮重新选择API。同步API在R6设计。

**输出**：DESIGN.md §1（引用`api_mapping_and_tile_planning.md`，给出Module级API调用序列和逻辑Tile清单）

---

### R2：Tile 规划

**核心问题**：需要哪些 tile？每个 tile 的 shape、dtype、layout 是什么？

> 📌 **权威依据（必读，官方标准）**：读取[API映射与Tile规划](references/api_mapping_and_tile_planning.md)中的“Tile规划”一节。Tile的shape、dtype、MemorySpace、layout、有效形状、缓冲深度以及`make_tile`/`make_tile_group`的使用范围全部照该节执行，与经验推断冲突时以该节为准。

**DESIGN.md §2须记录的设计内容**：
- 关键常量表，包括所有Tile尺寸和公式使用的派生常量
- R1调用链中全部输入、输出、跨迭代状态和临时Tile的属性表：变量名、用途、shape、dtype、MemorySpace、layout、`valid_shape`、缓冲数和单槽字节数
- 每个Tile使用`make_tile`或`make_tile_group`的结论；缓冲槽位数量按流水重叠关系确定
- TileGroup的`depth`、逐Tile mutex配置和访问方式；采用多ID时写清每个Tile对应的ID组，不配置`mutex_ids`时记录需要手动插入的核内跨Pipe同步；采用`group[i]`时记录下标表达式和`[0, depth)`有界依据
- Tile总占用的初步估算；容量不足时回查Tile尺寸、缓冲数量和API临时空间，实际地址与逐空间容量在R3确认

**输出**：DESIGN.md §2（引用`api_mapping_and_tile_planning.md`，填写关键常量和Tile属性表）

---

### R3：片上空间布局

**核心问题**：每块 tile 放在哪个内存空间的哪个地址？如何分配管理？

> 📌 **权威依据（必读）**：读取[片上空间布局](references/onchip_memory_layout.md)，按文档中的统一流程完成地址分配、对齐、复用和逐空间容量检查。该文档同时说明A5双AIV的UB核算、L0C→UB通路超限时的处理、MX scale地址映射和L1 Bank检查；容量值以当前`EXPLORE_REPORT.md` §7为准。

**输出**：DESIGN.md §3。按模板填写逐槽位地址表、逐空间容量结果和地址复用依据。布局失败时按参考文档给出的顺序回退R2或R1。

---

### R4：循环与 Section 结构

**核心问题**：R0确定的Module如何放入Section，循环嵌套、跨Tile状态和分核信息应如何组织？

> 📌 **权威依据（必读）**：读取[循环与Section结构设计](references/loop_design.md)。R4不重新划分Module，负责把R0的Module落实到Section与循环结构。融合算子还须读取[CV融合算子手动预加载流水设计](references/cv_fusion_pipeline.md)和`$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/advanced_programming/auto_parallel_pipeline.md`：满足自动流水文档的全部“使用约束”时推荐自动流水；任一约束不满足时，记录具体条款并采用手动流水。两种方式都要设计任务编号、阶段关系、启动时序、稳定运行时序、最后一个任务进入后的剩余阶段执行时序，以及缓冲生命周期。

> **CV并行流水必须在Stage 3/R4完成设计，并在Stage 4实现。** Stage 5不得新增流水或切换自动/手动方案；实测表明必须修改流水设计时，返回Stage 3修订DESIGN。

**输出**：填入模板 §4：
- 参考样例路径与可复用结构点
- 本算子的Section代码结构、结果单元、跨Tile状态生命周期、动态循环上界、各Module内的循环嵌套和分核信息的获取位置
- `is_fusion=true`时补充第一阶段每次交给下一阶段的数据范围、产生下一份数据的循环索引、`task_id`递增位置、交替阶段链、候选预加载轮数、逐阶段delay计算表，以及启动、稳定运行和末尾剩余阶段的执行时序；使用上下文缓冲时补充字段、深度和索引
- `is_fusion=true`时记录`pipeline_impl=auto/manual`及选择依据；具体配置和同步字段按模板填写
- `is_fusion=true`时逐一列出阶段链中的每一处 Cube↔Vector 交接，分别检查同一逻辑 Block 内 1 个 Cube 与 2 个 Vector subblock 的数据分配，并记录两个 subblock 各自处理的范围和工作量
- 引用`loop_design.md`，说明采用的Section和循环组织方式

---

### R5：分核策略

**核心问题**：work item 如何分配到各物理核？

> 📌 **权威依据（必读，官方标准）**：`$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md`。分核策略全部照该文档执行，与经验推断冲突时以该文档为准。

**本轮须在 DESIGN.md §5 落实的产出**：
- 分核方案
- host 侧 `num_cores` 计算式
- **launch 次数**：交付必须恰好启动 1 次 kernel；写明唯一 kernel 与 wrapper 调用点。若现有设计做不到，记录客观证据并返回 `failure_category: design_violation`，不得把多 launch 作为交付 fallback
- A5 CV 融合路径中两个 Vector subblock 的分工：切分轴、各自处理的索引范围、尾块归属和工作量；没有使用两个 subblock 时说明接口、数据量或依赖方面的原因

> **A5的1 Cube : 2 Vector分工**：逐一检查每处Cube↔Vector交接。两个Vector subblock能够独立处理时，沿非归约轴均衡切分，并记录各自的数据范围和尾块；存在归约依赖时，优先改用非归约轴切分或由一个subblock完成归约。确需拆分归约轴时，在GM保存部分结果，并在R6设计同步和最终合并。

> **TensorList 输入（`is_list: true`）：launch 次数必须是 1。**
> 逐元素 launch 是这类算子最常见也最贵的设计错误——它们的基线是一次融合调用，
> per-tensor launch 把基线特意消除的开销又加了回来，list 越长差距越大。
> §5 要写清楚冻结合同中的有限 `L_MAX`、多个 TensorList 形参间的长度关系，
> 以及每槽有限元素数上限 `N_MAX` 对所有 `DT_INT32` 派生式的安全性；
> 并令 `B_MAX=L_MAX`，只生成一组
> `B_MAX` 个固定槽位，每个 TensorList 形参在每个槽位展开为独立 `Ptr` 参数。
> wrapper 校验真实槽位，以 `n_i=0` 填充其余槽位；
> 地址值不得进入 tiling 数据，核内遍历各固定槽位的数据块，
> 并且 `num_cores` 按**所有元素的 tile 总数**推导，不按 list 长度推导。
> 冻结合同若允许所有真实槽位的 `n_i` 同时为 0，§5 还须写清零总任务策略：
> 使用已在目标验证的 `block_dim=1` 空工作单次启动，否则返回 `design_violation`；不得传入
> `block_dim=0` 或跳过启动。

> **适用边界**：该固定槽位模式仅适用于连续且 rank 不改变计算语义的逐元素算子。
> 设计要写成对 rank 不敏感的形式——按总元素数切 tile，而不是按 `[M, N]` 之类的
> 固定维数写死。同一 dtype 下不同 rank 的 case 应当能共用同一份实现；非连续输入
> 或 rank 相关计算须另行给出有限且经目标验证的 shape/stride ABI。

**输出**：DESIGN.md §5（引用 multi_core_partitioning_and_Tiling.md，填入上述产出）

---

### R6：核间同步（cross_core）

**核心问题**：Cube与Vector之间，或不同Block/subblock之间存在数据依赖时，如何使用自动流水同步或手动跨核事件及多槽缓冲，让Cube与Vector同时处理不同编号的数据？

> 📌 **权威依据（必读，官方标准）**：读取[跨核同步](references/cross_core_synchronization.md)，融合算子同时读取[CV融合算子手动预加载流水设计](references/cv_fusion_pipeline.md)。流水方式沿用R4结论：自动流水配置TileGroup的`fwd_ids`/`bwd_ids`并复核`pipeline_generated.py`；手动流水按共享槽位显式放置`set_cross_core`/`wait_cross_core`。两种方式都要说明槽位的初始可写状态、稳定复用顺序和最后一批数据的消费完成条件。

**DESIGN.md §6须记录的设计内容**：
- 是否涉及cross_core；Cube与Vector之间无数据依赖，并且不存在需要`INTER_BLOCK`、`INTER_SUBBLOCK`或`UNICAST_BLOCK`处理的依赖时，填写“不涉及cross_core”。不能仅凭Section数量判定
- `is_fusion=true`时记录所选自动或手动预加载流水：Cube/Vector之间传递数据的连续编号、各阶段延迟、预加载轮数、最后一个新任务进入后两侧需要继续执行的轮数、跨核TileGroup的就绪/释放关系和初始可写状态；自动方式记录`fwd_ids`/`bwd_ids`及生成代码复核结果，手动方式记录实际事件位置；使用上下文缓冲时同时记录其深度与索引
- 生产者、消费者和共享数据；共享数据是多槽TileGroup时，补充缓冲深度和两侧的槽位访问表达式
- 同步方案表：自动方式记录方向、共享缓冲、槽位表达式、`fwd_ids`/`bwd_ids`和生成代码中的set/wait；手动方式记录方向、共享缓冲、槽位表达式、set/wait位置、pipe、`sync_mode`和`event_id`
- 循环复用缓冲时的正向与反向同步，以及最后一次消费完成后生产者侧的等待位置
- 两个Vector subblock之间存在结果依赖时，记录GM部分结果、最终合并者和workspace复用条件；若使用`INTER_SUBBLOCK`屏障，具体参与者和调用规则以目标版本同步文档及官方样例为准

**完成设计后校验**：先核对生产者、消费者和就绪/释放事件是否按同一表达式选择物理槽位，再逐槽位、逐轮次核对set/wait是否一一对应。检查动态下标是否始终落在`[0, depth)`，以及零次、一次、整除和尾块分支中的事件是否都能配对，并确认生产者结束前已经等待最后一次消费完成。任一项不满足时，重新设计本轮同步方案。

EXPLORE_REPORT §4中的官方指定算子用于核对完整调用方式，不能代替权威文档中的同步规则。

**输出**：DESIGN.md §6（引用`cross_core_synchronization.md`，填写同步点表和event_id分配表）

---

### R7：尾块处理

**核心问题**：如何处理维度不整除 tile 尺寸的尾块？

> 📌 **权威依据（必读，官方标准）**：`$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/tile_based_python_programming/tail_block_handling.md`。尾块的完整机制全部照该文档执行，与经验推断冲突时以该文档为准。核心模型：**Tile的物理shape固定，有效shape随当前块变化**。

**本轮须在 DESIGN.md §7 落实的产出**：将该文档的尾块机制落到本算子的循环与Section结构中（填入 §7 尾块处理方案）。

**输出**：DESIGN.md §7（引用 tail_block_handling.md 落地尾块代码）

---

### R7.5：目标测试 case 规划

**核心问题**：本设计要交给 develop 验证的具体测试 shape 是哪几个？

设计阶段必须**基于已确定的 tile 切分（R2 tile 尺寸）与尾块方案（R7）**，把测试 case 连同具体 shape 一次性确定下来，写入 DESIGN.md §8，供 develop 直接实现——develop 开工前即已知这些 case，不再自行重算。
**流程**：

1. **确定 case 集合**：
   - 若 SPEC.md / 用户已给出目标 case，则**在其基础上追加**下述 4 个基础 case。
   - 若无用户指定 case，则**直接确定为下述 4 个基础 case，不询问用户**。
2. **4 个基础 case（按 tile 切分推导具体 shape，取值以 R2 tile 尺寸为基准）**：

   | case | 覆盖场景 | shape 取法（示例） |
   |------|---------|-------------------|
   | 全整除 | 所有 tile 维度均整除 | `[TILE_A, TILE_B]` |
   | 单轴尾块 | 一个 tile 维度存在尾块 | `[TILE_A + 尾块余数, TILE_B]` |
   | 双轴尾块 | 两个 tile 维度均存在尾块 | `[TILE_A + 尾块余数, TILE_B - 尾块余数]` |
   | 跨多 tile + 尾块 | 动态轴跨越 2–3 个 tile 并带尾块，验证跨 tile 循环与状态持久化 | `[2~3 × TILE_A + 尾块余数, TILE_B - 尾块余数]`（取触发多 tile 迭代的最小规模即可，不必放大，避免拖慢编译/执行或触发 OOM） |

3. **逐个验证 design 可适配**：对每个 case，核对当前 design（R2 tile 规划、R4 循环边界、R7 尾块处理、归约方案等）能否正确处理该 shape。**若发现某 case 适配不了，必须回溯对应轮次具体解决**（如尾块处理缺陷回 R7、循环边界错误回 R4、归约轴超单 tile 回 R0），**不得反过来删改 case 迁就 design**。
4. **例外**：若算子凑不出 4 个有区分度的 case，则以覆盖到的边界类别（整除 / 尾块 / 跨多 tile）为准，不为凑数量制造冗余用例；此时在 DESIGN.md §8 与 MEMORY.md 说明实际可覆盖的 case 数与原因。

**输出**：DESIGN.md §8 的"目标测试 case"表（case 名 / 具体 shape / 覆盖场景 / design 适配确认），至少 4 个（单轴算子按例外处理）。
---

### R8：综合评估

**核心问题**：整体设计是否准确、可靠、泛化？

**流程**：

> 评估时每项检查须标注 ✅（通过）或 ❌（不通过）；orchestrator 以评估表中无 ❌ 作为设计通过判据。

| 维度 | 检查项 | 结果 | 不通过时的处理 |
|------|--------|------|----------------|
| **准确性** | API 调用链是否完整实现了数学公式的每一步 | | 回到 R1 补充 |
| | §4 已参照官方样例确定循环结构（含参考样例路径与结构说明） | | 回到 R4 补充 |
| | 数据依赖是否正确（Module 顺序、sync 位置） | | 回到 R0 或 R4 调整 |
| | CV融合是否依据自动流水“使用约束”选择自动或手动方案，并完整设计任务编号、阶段关系、缓冲、启动时序、稳定运行时序和最后一个任务进入后的剩余阶段执行时序 | | 回到 R4/R6 重构流水 |
| | A5 CV融合是否逐一检查Cube↔Vector交接的1:2分工；存在归约依赖时是否给出部分结果与合并方案 | | 回到 R4/R5/R6 修正 |
| | dtype 选择是否能保证精度（如 matmul 累加用 FP32） | | 回到 R2 调整 |
| | 归约类 API 的 `[M,1]`/`[1,N]` 输出已设合适的 `layout` | | 回到 R2 补 layout |
| **泛化性** | 目标测试 case（≥4，单轴算子按例外）已按 tile 切分确定具体 shape，且逐个验证 design 可适配（R7.5 已完成） | | 回到 R7.5 补充 / 回溯适配不了的轮次 |
| | 是否正确处理了尾块 | | 回到 R7 补充 |
| | 循环边界是否正确（ceiling division、valid_m/valid_n 计算） | | 回到 R4 修正 |
| | 超越函数（exp/log/sqrt 等）在目标 dtype 范围内无溢出（§1 数值安全边界已分析） | | 回到 R1 补溢出防护 |
| | 窄 dtype（fp16/bf16）下的平方、同量级相乘、长轴累加，其**中间值**量级上界在该 dtype 范围内；若不在，升位宽的 `vf.astype` 已落在产生增长的那一步之前而非归约之前（§1 数值安全边界已分析） | | 回到 R1 调整 cast 位置 / 回到 R2 改累加 dtype |
| | 跨 tile 状态是否正确初始化和持久化 | | 回到 R0 或 R1 修正 |
| | cross_core 同步方案是否正确（存在跨执行域或跨Block/subblock依赖时）：自动或手动预加载流水的阶段延迟、上下文槽位、就绪/释放关系、初始可写状态、同步点和event_id是否参照权威文档与当前实现；自动方式是否复核生成代码 | | 回到 R4/R6 修正 |
| **一致性** | R0-R7 各轮输出是否存在矛盾（如 API 需要的 tile 在 R2 中缺失） | | 回溯到矛盾产生的轮次修正 |
| | 证据链是否完整（每个决策都有来源） | | 补充缺失的文档引用或官方指定算子路径 |
| | R3 片上地址范围与逐空间容量检查是否全部通过 | | 回到 R3 重排地址 / R2 缩 tile |
| **性能** | §11 已写代价模型与绑定资源（`B1`/`B2`）、目标上限（`B6`）、分核 `η_tail`（`P3`）；CV 融合已错拍（`X1`） | | 回到对应轮次 |
| **条件性检查** | 若 §6 填“不涉及cross_core”，确认不存在Cube↔Vector或跨Block/subblock的数据依赖 | | 回到 R0 重新评估 |

**迭代规则**：
- 发现问题数 ≤ 3，修复后重新走 R8
- 发现问题数 > 3，回到问题最早出现的轮次，重新过后续轮
- 最多 5 次完整 R0-R8 迭代

**输出**：评估结论 + 修改记录（如有）。

---

### Tile 数据流全景图

R8 评估通过后，将 R0-R7 各轮的分散产出串成一张全景图（填入模板 §10）。这张图应能单页展示算子中所有数据流的全貌。

**绘制方法**：

1. **起点**：输入张量
2. **数据搬运**：标注 `load_tile` 、 `store_tile` 与流水线（MTE2 / MTE3）
3. **数据流向**：用箭头 `→` 连接每个操作，箭头标注 API 名。同一块 tile 被多个操作串联使用时用 `├─` 表示分支
4. **跨 Module 持久化**：tile 在某个 Module 中写入、在后续 Module 中读取的，用 `---` 虚线表示数据跨越 Module 边界。标注 tile 变量名和地址
5. **输出**：输出张量

**验证**：全景图中每一块 tile 和每一个操作都必须能在 R3 的地址映射表、R1 的 API 序列中找到对应条目。缺失或矛盾则回溯修正。

---

## 设计原则

1. **每个决策必须有证据**：API 文档引用、官方指定算子路径、教学文档、数学推导至少占其一
2. **Tile 数据流图是给 coder 的施工合同**：coder 拿到 DESIGN.md 应能确定 kernel 的完整结构与关键决策；API 签名等细节须以 API 文档原文为准确认（EXPLORE_REPORT 仅为派生的先行速查，不作签名权威）；运行验证暴露设计失误时返回 `design_violation`，由编排器重新调度设计修订并验收
3. **地址分配精确到字节**：不写"大约"、"若干"
4. **实现约束不可违背**（见上方「实现约束」节）
5. **Vector 选择必须可复核**：`vector_selection` 字段完整；`tile_op` 有已选 KB 模板的明确要求，否则为 `vf`
6. **参考官方指定算子优先于自行设计，拥有最高优先级**：遇到同步策略、tile 尺寸、Vector 指令组合等决策时，优先查阅官方指定算子（PRO_MATERIAL_INDEX §B）中相似者，复用成熟模式；无相似时以 API 文档 / 教学文档为准

## 强制规则：wrapper 边界（每个 class 都适用）

公开 callable 是实现边界的一部分。host 侧张量操作可能被下发成真实的 device kernel
（`aclnnInplaceCopy_CastAiCore_Cast`、
`..._TransposeAiCore_Transpose`、`..._SliceAiCore_Slice`、`aclnnCat_ConcatD_ConcatD`
等）并计入耗时；纯 view 即使不下发 device kernel，也仍越过交付边界。

> **cast、slice、transpose、pad、concat 以及任何数据形状/dtype 处理，
> 必须放进 `@pl.jit` kernel 内部。wrapper 只做参数校验、读取 KB 约束列明的只读
> 元数据、纯 Python 整数推导、`torch.empty` 分配当前 wrapper 合同声明的输出和一次 kernel 启动。**

必读 [`pypto-pro-op-kb/constraints/wrapper-boundary.md`](../pypto-pro-op-kb/constraints/wrapper-boundary.md)，
它是 `required_constraints` 中的全局约束，不占用可选 pattern 名额。

### 设计阶段必须完成的事

1. **kernel 接口按真实输入定义**，不要为了 kernel 好写而要求 host 先归一化。
   kernel 应当直接接收原始 dtype、原始 layout、原始 rank。
2. `DESIGN.md` 的 **Wrapper 边界外操作**正文必须只有 `空`。DESIGN、usage、`deviated` 或
   profile 都不能授权边界外操作；无法迁入 kernel 时须附证据上报
   `design_violation`，不得交付放宽边界的设计。
3. 需要的轴变换用 **stride/offset 索引**在 tile 循环里表达，不要用
   `movedim` / `permute` / `contiguous`。
4. dtype 转换在 **tile load / store 时**用 `pl.cast`（tile 级）或
   `vf.astype`（寄存器级）完成，不要在 host 上 `.to()` 整个张量。
   （没有 `vf.cast` 这个 API。）
5. 尾块用 `pl.set_validshape`，不要在 host 上 pad 到 tile 整数倍。

### 反面样例（真实生成产物）

```python
def op_wrapper(input_tensor, dim=-1, ...):
    x_fp32 = input_tensor.to(torch.float32)              # 计入耗时
    x_transposed = x_fp32.movedim(dim, -1).contiguous()  # 计入耗时
    ...
    op_kernel(x_2d, y_2d, ...)                           # 真正的计算
    y = y_transposed.movedim(-1, dim).contiguous()       # 计入耗时
    return y.to(out_dtype)                               # 计入耗时
```

一次 kernel 启动，外面包了四个被计时的 device 算子。kernel 被写成只接受「规范化的 FP32 连续 2-D 输入」，于是 host 被迫去生产它——这份便利按全价计费。
wrapper 占比过半的 class 并不罕见，把这些搬进 kernel 通常是该算子最大的一根杠杆。

注意方向：wrapper 也**不得**承担算子的**算术**（那是作弊，实现验收的 anti-cheat
会判 FAIL）。要求是「形状/dtype 处理进 kernel」，不是「计算搬到 host」。
