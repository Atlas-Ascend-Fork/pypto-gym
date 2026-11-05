# 性能调优模板优化项索引

本目录是性能调优的独立预置优化项来源。`INDEX.md` 是模板项生命周期和枚举的唯一入口；文件
存在本身不使模板生效。调优时只登记下表中 `lifecycle=active && optimization_eligible=yes` 的原子项，
使用 `source_kind=template` 和表内稳定 `item_id`，并记录 INDEX 行与模板文件的运行时内容哈希。
`active` 表示必须在本专项调优中评估并关闭，不表示必须采用。当前 kernel owner 的规则和机制参考不因此变成模板优化项；用户明确指定的项目约束按主 Skill §3.1 核对。

**当前计数：active/eligible item 为 6。**

## Active atomic items

| item_id | 模板与原子优化意图 | lifecycle | optimization_eligible | bound_hint | hard_applicability_gates | target/api_gate |
|---|---|---|---|---|---|---|
| `template-cube-output-wave-reuse` | [`cube-output-wave-reuse.py.tmpl`](cube-output-wave-reuse.py.tmpl)：同一 partition/K block 的多个输出复用一份 left Tile | active | yes | Cube / 搬运 / 流水 | 至少两个输出共享完全相同的 left operand；每个固定 partition 内的 q 链独立开闭；每个输出的 K 顺序、FP32 partition reduction tree 和 GM owner 不变；Acc/Right/Left 容量足够 | 当前 Cube、Acc phase、TileGroup 与 parser 能力经目标环境核对 |
| `template-cube-shared-left-output-pair` | [`cube-shared-left-output-pair.py.tmpl`](cube-shared-left-output-pair.py.tmpl)：相邻输出对复用同一组 left residual | active | yes | Cube / 搬运 | 存在完整相邻输出对且 left residual 完全相同；单 residual 使用一次 `Final` 专门路径；逐输出累加顺序、odd tail 与唯一写者不变；双 Acc/RHS 及相关 TileGroup 容量足够 | 显式 Tile handle、精确 phase 链和目标 parser 行为经核对 |
| `template-stage-task-flatten` | [`stage-task-flatten.py.tmpl`](stage-task-flatten.py.tmpl)：把多阶段逻辑任务展开到物理核，减少空核与调度长尾 | active | yes | 调度 / Scalar / 流水 | 已有多阶段单 kernel 且任务映射可调整；各 section 按目标 engine 语义独立还原 physical id；每阶段唯一且完整覆盖；所有 launched AIC/AIV 以相同顺序和次数无条件到达 barrier；tail、数值 dataflow 与 reduction order 不变 | 目标版本 block/subblock 语义经验证；使用保守单 AIV 路径 |
| `template-tiling-key-resource-specialization` | [`tiling-key-resource-specialization.py.tmpl`](tiling-key-resource-specialization.py.tmpl)：按有限 host metadata 编译期专门化资源与热路径 | active | yes | Scalar / 资源 / 调度 | 分类只依赖合同允许的 host metadata，类别有限，入口与阶段序列遵守当前 delivery owner，全部 key 保持同一公共合同 | TilingKey 字段、调用语法和 key-specific IR 消枝在目标版本验证 |
| `template-typed-tilegroup-stage-reuse` | [`typed-tilegroup-stage-reuse.py.tmpl`](typed-tilegroup-stage-reuse.py.tmpl)：互斥硬阶段复用同一物理片上 arena | active | yes | 片上容量 / 搬运 | 不同类型 TileGroup 的 live interval 确实互斥；地址、容量、对齐与 mutex id 各自合法且不冲突；所有 launched lane 无条件到达 hard barrier | 目标 memory space、TileType move 规则和 MIX barrier 行为经验证 |
| `template-vf-broadcast-dot-panel` | [`vf-broadcast-dot-panel.py.tmpl`](vf-broadcast-dot-panel.py.tmpl)：用 VF scalar broadcast 对 panel 执行 dot 累加 | active | yes | Vector / Scalar / 搬运 | 算子存在对应 panel dot，layout/stride/mask/tail 可表达；`PANEL_K` 可整除 `GROUP_K`，或另有显式 K-tail 路径；冻结精度允许该累加与补偿方案 | 仅目标 A5；broadcast load、三地址 VF 与多行循环须做生产形态探针 |

同一改动可以共享实验，但每个 template item 必须单独关闭并记录 overlap/require/conflict。模板只是
待适配骨架，不保证可独立编译；采用前必须核对当前 API、dtype/layout、容量、tail、同步、wrapper
合同与用户明确的项目约束；模板中的普通 Python helper 占位必须按当前 frontend 要求内联或改写。最终
必须通过完整正确性、机制核对及当前设备同协议实测。

实现与验证时可按对应 item 读取以下辅助资料；它们不产生新的 source item：

- `template-stage-task-flatten`：[多阶段任务展平指南](../references/general-knowledge/stage-task-flatten.md)；
- `template-cube-output-wave-reuse`、`template-cube-shared-left-output-pair`：
  [多阶段 Matmul A/B 协议](../references/general-knowledge/staged-matmul-ab-protocol.md)。

不得扫描本目录补充来源，也不得把未登记文件或模板内部的可选变体匿名加入账本。确有
独立原子优化价值的新模板，须先与当前 kernel 已有机制、通用方法、active 卡片和
active 模板去重或说明独特增量，再以新稳定 ID 加入本 INDEX，之后才能作为预置项参与调优。
