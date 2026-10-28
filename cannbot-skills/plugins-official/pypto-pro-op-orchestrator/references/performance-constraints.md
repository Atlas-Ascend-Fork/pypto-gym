# 同步与 Vector API 参考

本文归纳核内与跨核同步机制、VF 写法和寄存器使用，不另设实现选择优先级或工作流程。API 的适用条件须核对目标版本文档与源码。

源码位于 `plugins-official/pypto-pro-op-orchestrator/references/`；`init.sh` 复制到工具配置目录的 `references/`，OpenCode 项目安装中为 `.opencode/references/`。

## TileGroup 与自动互斥

使用 `auto_mutex` 管理核内 buffer 时，轮转 tile 由 `make_tile_group` 创建；单独的 `make_tile` 不会获得这套自动同步。手动同步仍须分别核对真实依赖与 slot 生命周期，不能仅凭装饰器已启用认定同步完整。

| 项目 | 使用说明 |
|------|------|
| 自动管理的轮转 tile（含 double buffer） | `make_tile_group` 创建，`auto_mutex=True` 管理轮转与互斥 |
| 独立 `make_tile` | 适用于不依赖 TileGroup 自动同步的存储；访问的保序须有依据 |
| TileGroup 上的手动同步 | 不对同一自动管理依赖重复叠加 `sync_src`/`sync_dst` |

**同步分工**（auto_mutex 管核内；跨核由已选的自动或手动 CV 流水机制管理，命名空间独立）：

| 范围 | 机制 | 标识符 | 说明 |
|------|------|--------|------|
| 核内 pipe 间 | `auto_mutex=True`（框架自动插 `mutex_lock`/`mutex_unlock`） | `mutex_id` ∈ [0,31] | 覆盖 `make_tile_group` tile 的 buffer 互斥；勿为同一依赖重复叠加手动 `sync_src`/`sync_dst`，避免依赖环 |
| 跨核（AIC↔AIV），自动流水 | 跨核TileGroup的`fwd_ids`/`bwd_ids`，框架生成`set_cross_core`/`wait_cross_core` | `event_id` ∈ [0,16) | 适用约束和配置以目标版本`auto_parallel_pipeline.md`为准，须复核生成代码 |
| 跨核（AIC↔AIV），手动流水 | 显式`set_cross_core`/`wait_cross_core` | `event_id` ∈ [0,16) | auto_mutex 不覆盖跨核依赖；事件位置和pipe按实际数据操作设计 |

> `mutex_id` 与 `event_id` 是两个独立命名空间（FA 样例中数值重叠共存无报错），不构成冲突。

---

## <a id="强制-2vector-数值计算用-vf-手写"></a>Vector 函数与寄存器

`vf.*` 的中间值位于寄存器；Tile 的 UB 存储与寄存器状态需分别计入资源分析。存在同名 `pl.*` API 不能证明有同名、同参数的 VF 操作，须查目标版本 VF 定义。

| 项目 | 使用说明 |
|------|------|
| VF 执行 | 在 `section_vector()` 内调用 `@pl.vector_function` 定义的函数 |
| 搬运与控制流 | `pl.load`/`pl.store`/`pl.load_tile`/`pl.store_tile`/`pl.range`/`pl.get_block_idx` 等不属于 VF 寄存器数值指令 |
| 等价写法① | `@pl.vector_function` 装饰器（通常模块级，从 `section_vector()` 调用，见 FA；也可工厂函数内定义，见 lightning）|
| 等价写法② | `@pl.inline` + `with pl.section_vf():` 块（见 vf API 文档示例）|

寄存器中间值不直接占用 UB，不代表完整 VF 路径的 UB 占用一定更低。容量分析还需计入输入输出 Tile、搬运布局、缓冲深度与实际 lowering。
