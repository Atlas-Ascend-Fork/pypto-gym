# PyPTO-Pro 两条性能强制

> **地位**：`kernel-core-principles.md` 在 PyPTO-Pro API 上的具体化；冲突时以根规范为准并上报。
>
> **用途**：PyPTO-Pro 工作流的性能约束。被 `pypto-pro-material-explore`（探索阶段 API 映射）、`pypto-pro-op-design`（设计阶段落实）、`pypto-pro-op-develop`（实现阶段遵守）和 `pypto-pro-op-perf-tune`（性能优化阶段保持铁律）共同引用。各 skill 不重复定义，统一指向本文件。
>
> **安装路径**：源码位于 `plugins-official/pypto-pro-op-orchestrator/references/`，由 `init.sh` 软链到 `.opencode/references/`。各 skill 引用时写「见 `.opencode/references/performance-constraints.md`」。

以下两条是性能约定，探索阶段的 API 映射与样例参考、设计阶段的 tile 规划、实现阶段的 kernel 编码以及 Stage 5 优化均须遵守。否则产出的算子性能可能不可接受。

---

## 强制 1：buffer 管理用 `make_tile_group` + `auto_mutex`

> 依据：`P2`（搬运与计算重叠，默认双缓冲）。手动同步不足导致死锁，过度则每个 tile 排空流水；`auto_mutex` 由框架保证二者皆不发生。深度改动须复核 `T1`。

| 项目 | 规则 |
|------|------|
| buffer 轮转 tile（含 double buffer） | `make_tile_group` 创建，`auto_mutex=True` 自动管理轮转与互斥 |
| `make_tile` 合法用途 | 仅单次 scratch tile（写入一次、读取一次、不参与轮转循环，如一次性中间结果暂存、不迭代的归约标量结果）|
| 禁止 | `make_tile` + 手动 `sync_src`/`sync_dst` 管理轮转 |

**同步分工**（auto_mutex 管核内；跨核由已选的自动或手动 CV 流水机制管理，命名空间独立）：

| 范围 | 机制 | 标识符 | 说明 |
|------|------|--------|------|
| 核内 pipe 间 | `auto_mutex=True`（框架自动插 `mutex_lock`/`mutex_unlock`） | `mutex_id` ∈ [0,31] | 覆盖 `make_tile_group` tile 的 buffer 互斥；禁止在其上叠加手动 `sync_src`/`sync_dst`，否则依赖环导致 AICore timeout 死锁 |
| 跨核（AIC↔AIV），自动流水 | 跨核TileGroup的`fwd_ids`/`bwd_ids`，框架生成`set_cross_core`/`wait_cross_core` | `event_id` ∈ [0,16) | 适用约束和配置以目标版本`auto_parallel_pipeline.md`为准，须复核生成代码 |
| 跨核（AIC↔AIV），手动流水 | 显式`set_cross_core`/`wait_cross_core` | `event_id` ∈ [0,16) | auto_mutex 不覆盖跨核依赖；事件位置和pipe按实际数据操作设计 |

> `mutex_id` 与 `event_id` 是两个独立命名空间（FA 样例中数值重叠共存无报错），不构成冲突。

---

## <a id="强制-2vector-数值计算用-vf-手写"></a>强制 2：Vector 默认使用 VF

> 依据：`V2`（逐元素链单趟融合）。`vf.*` 的中间值留在寄存器，不占 UB，省下的容量用于更大的 tile。

Vector 数值计算按以下静态规则选择实现，不比较 VF 与 tile-op 性能：

| 条件 | 选择规则 | 必要证据 |
|------|----------|----------|
| `KB_SELECTION.json` 已选中的 KB pattern/template 明确要求当前具体步骤使用 tile-op `pl.*` | 使用该模板要求的 `pl.*` | KB 路径、明确要求的原文/片段、目标版本和适用条件 |
| 其他情况 | 使用 `vf.*` | 目标版本 VF API 或可行 VF 组合的依据 |

“明确要求”必须直接约束当前步骤的实现方式。KB 只展示 tile-op 写法、官方样例使用 `pl.*`、
存在对应 `pl.*` API，或推测 tile-op 更快/更简洁，都不构成例外。Stage 1 记录默认 VF 映射
并完成 `KB_SELECTION.json`；Stage 3 核对已选 KB 模板并冻结唯一实现；Stage 4 只执行 DESIGN，不再生成
双候选或触发额外选择验证。若冻结方案无法实现，按既有 `capability_gap` 流程处理，不在
Stage 4 自行切换实现层级。

| 项目 | 规则 |
|------|------|
| 默认实现 | `vf.*` 指令手写，在 `section_vector()` 内通过 `@pl.vector_function` 执行 |
| tile-op 例外 | 仅在已选 KB pattern/template 明确要求当前步骤时使用对应 `pl.*` API |
| 禁止 | 无明确 KB 模板要求时使用 tile-op，或在 Stage 4 改写 DESIGN 已冻结的实现层级 |
| 不受此限（数据搬运与控制流） | `pl.load`/`pl.store`/`pl.load_tile`/`pl.store_tile`/`pl.range`/`pl.get_block_idx` 等 |
| 等价写法① | `@pl.vector_function` 装饰器（通常模块级，从 `section_vector()` 调用，见 FA；也可工厂函数内定义，见 lightning）|
| 等价写法② | `@pl.inline` + `with pl.section_vf():` 块（见 vf API 文档示例）|

> **vf.\* 在 VF 寄存器中完成计算，寄存器不占 UB 空间**——VF 路径的 UB 占用不会高于 Memory 级路径。UB 超限不是放弃 VF 的理由（UB 超限处理见 design SKILL R3 重排地址 / R2 缩 tile）。
