# 通用优化手段

本页汇总历史性能调优材料中经逐行对照当前 `pypto-pro-op-kb` 后仍未被完整覆盖的独特优化
增量。它是性能调优的一个独立预置优化项来源，不属于知识卡片库，也不替代已选 KB 合同。

执行者必须逐行枚举下表中 `lifecycle=active && optimization_eligible=yes` 的 item，并使用
`source_kind=general_method` 登记。明确不适用时以 hard-gate 证据关闭；可能适用或现有证据
不足以排除时进入受控实验。方法标题、历史经验或理论估计都不能代替当前算子的正确性、机制和
同协议性能证据。

正文中的 formal compare 证据只适用于按冻结规则晋级的候选；进入实验但 quick 未晋级的项按
playbook 以 `quick_not_promoted` 和 quick 原始证据关闭。

本页只整理优化动作及其边界。采集、字段解释和瓶颈分析仍以
[证据协议](perf-analysis/evidence-protocol.md)、[msprof 指南](perf-analysis/msprof-guide.md)、
[msprof op 指南](perf-analysis/msprof-op-guide.md)和 [CSV 字段参考](perf-analysis/csv_fields_reference.md)为准，本文不改写
这些通用技巧。

## 原子项索引

`source_ref` 应记录本文路径、item 锚点和运行时读取到的内容哈希。`source_anchor` 只标识整理
这些方法所依据的历史性能调优原文，不是当前设备的性能证据。

| item_id | 方法与锚点 | lifecycle | optimization_eligible | bound_hint | source_anchor |
|---|---|---|---|---|---|
| `general-01` | [为 fold 保留独立工作 Tile](#general-01) | active | yes | VEC / 访存 / 流水 | `SKILL.md@700fabe` §二.7；`optimization-playbook.md@700fabe` §5 |
| `general-02` | [按 value_range 门控重关联](#general-02) | active | yes | VEC / 精度 / 算法 | `SKILL.md@700fabe` §三 |
| `general-03` | [TopK 截断与分层树形归并](#general-03) | active | yes | 算法 / VEC / Scalar / 访存 | `optimization-playbook.md@700fabe` §6 |
| `general-04` | [有限精确域 histogram / threshold](#general-04) | active | yes | 算法 / VEC / 访存 | `optimization-playbook.md@700fabe` §6 |
| `general-05` | [删除不会被消费的 candidate 链](#general-05) | active | yes | 算法 / VEC / Scalar / 访存 | `optimization-playbook.md@700fabe` §6 |
| `general-06` | [跳过无效 lane 的非必要工作](#general-06) | active | yes | 算法 / VEC / 访存 | `optimization-playbook.md@700fabe` §6 |
| `general-07` | [合并语义等价的重复工作](#general-07) | active | yes | 算法 / VEC / Scalar / 访存 | `optimization-playbook.md@700fabe` §6 |
| `general-08` | [block_dim / owner / tail 联合调优](#general-08) | active | yes | Launch / 调度 / Scalar / 访存 | `optimization-playbook.md@700fabe` §7 |
| `general-09` | [提升热循环中的恒定分支](#general-09) | active | yes | Scalar / 调度 / VEC / Cube | `optimization-playbook.md@700fabe` §4 |
| `general-10` | [提升一般循环不变量 setup](#general-10) | active | yes | Scalar / 调度 / 访存 | `optimization-playbook.md@700fabe` §4 |
| `general-11` | [删除没有消费者的同步边](#general-11) | active | yes | 同步 / Scalar / 流水 | `optimization-playbook.md@700fabe` §4/§5 |
| `general-12` | [一个 task 批处理多个独立 work item](#general-12) | active | yes | Launch / Scalar / 调度 / 访存 | `optimization-playbook.md@700fabe` §4 |
| `general-13` | [增大单个逻辑 item 的合法 Tile/strip](#general-13) | active | yes | Scalar / 调度 / 访存 / 流水 | `optimization-playbook.md@700fabe` §4 |
| `general-15` | [消除数据并行路径的 Scalar 往返与循环](#general-15) | active | yes | Scalar / VEC / 访存 | `optimization-playbook.md@700fabe` §4 |
| `general-16` | [静态化有界小控制与归约拓扑](#general-16) | active | yes | Scalar / VEC / Cube / 精度 | `optimization-playbook.md@700fabe` §4/§6/§7 |
| `general-17` | [按消费者所需域缩小计算范围](#general-17) | active | yes | 算法 / VEC / Scalar / 访存 | `optimization-playbook.md@700fabe` §6 |
| `general-18` | [按已证明值域窄化内部表示](#general-18) | active | yes | Scalar / VEC / 访存 / 精度 | `optimization-playbook.md@700fabe` §6 |
| `general-19` | [消除非合同边界的中间落盘与回读](#general-19) | active | yes | 访存 / 流水 / 同步 | `optimization-playbook.md@700fabe` §6/§7 |

## 方法正文

<a id="general-01"></a>
### `general-01`：为 fold 保留独立工作 Tile

- **适用**：输入轮转槽被原地 fold/归约状态长期占用，导致本应重叠的 `load(N+1)` 与
  `fold(N)` 在同一监控 lane 串行；live-range 又证明两种角色需同时存活。
- **动作**：把短生命周期输入槽与长生命周期 fold 状态分成独立角色，使下一拍能够覆盖已经
  退休的输入槽。先画 live-range 和地址表，再决定用独立 Tile、TileGroup 角色或合法寄存器
  状态；不得为了“独立”无条件增加一次 UB 复制。
- **边界**：每核只有一个 work item、当前已合法重叠、容量/slot/mutex 不足，或 API 只能原地
  fold 时不适用。不得用地址重叠或重复 mutex id 伪造空间。
- **关闭证据**：完整覆盖单拍、多拍、槽位回绕、aligned/tail；同时记录代码/生成物 live-range、
  容量变化、formal compare，以及同一 target task、同一 lane 上可映射到相邻 work item 的
  timeline。
- **KB 边界**：[buffer-reuse-lifetime.md](../../pypto-pro-op-kb/patterns/buffer-reuse-lifetime.md)
  已规定角色生命周期、轮转槽和别名正确性；本项只保留“合法别名仍可能因 fold 长生命周期
  破坏流水”的性能增量。

<a id="general-02"></a>
### `general-02`：按 `value_range` 门控重关联

- **适用**：候选通过重分组、树形 fold 或跨行合并改变浮点运算次序，而误差随输入值域和中间
  动态范围变化；冻结合同或既有合法元数据能证明 range class。
- **动作**：只在可证明值域满足现有精度预算时启用重关联快路；未证明的范围使用原运算顺序。
  shape、dtype tier 或少量随机样本不能代替值域证明。
- **边界**：不得修改 SPEC 精度门或输入分布，不得在 wrapper 扫描 Tensor 内容，也不得臆造
  TilingKey/host 取值。无法合法区分 range class 时保留通用顺序。
- **关闭证据**：逐个可达 range class 验证边界值、特殊值、随机值和全部 P0，记录值域合同、
  误差统计、依赖层数与 formal compare；NaN/Inf、signed zero、溢出/下溢和 tie 语义按 SPEC
  覆盖。
- **KB 边界**：[precision.md](../../pypto-pro-op-kb/constraints/precision.md) 已规定精度合同；
  本项只保留“重关联快路必须由 value range 而非 shape tier 门控”的优化决策。

<a id="general-03"></a>
### `general-03`：TopK 截断与分层树形归并

- **适用**：最终只消费 K 个 value/index，当前却完整排序 N 个元素，或每层归并保留超过 K 个
  候选；能够证明每块局部 TopK 之外的元素不可能进入全局 TopK。
- **动作**：每个分块最多保留 K 个候选，分层或树形合并后在每一级重新截断到 K。比较 key
  必须完整编码升/降序、稳定性、tie-break、NaN/Inf、signed zero 与原始 index 语义。
- **边界**：需要完整排序/完整 rank/所有等值项、K 接近 N，或局部截断无法保持完整顺序语义时
  不适用。结构中的 `local_topk`、`merge` 只是算法描述，不是 PyPTO-Pro API。
- **关闭证据**：记录候选数、比较/归并层数、中间存储和字节；完整覆盖 K=1、K 接近 N、尾块、
  全相等、边界 tie、NaN/Inf、signed zero，并分别回归升序与降序，再做同协议 formal compare。
- **KB 边界**：当前 KB 没有通用 TopK 截断合同；scan 类 pattern 不能证明 TopK 截断与
  tie-break。

<a id="general-04"></a>
### `general-04`：有限精确域的 histogram / threshold 替代

- **适用**：SPEC 证明输入属于可一一映射的小型有限精确域，域大小远小于输入规模，计数表可
  容纳，且前缀计数能精确恢复要求的 value/index/tie 结果。
- **动作**：用一次输入遍历建立精确计数，再遍历域定位 threshold 并按原合同恢复输出；计数器
  位置、更新 owner、跨核归并与扫描必须单独设计。
- **边界**：连续浮点域、未知动态范围、近似分桶、计数溢出、缺少合法并发更新能力，或无法
  恢复稳定 index/tie 时不适用。不得从观测样本缩窄 SPEC 合法域，不得静默 clamp 越界值，
  也不得假设存在任意索引原子加法或 histogram API。
- **关闭证据**：记录 N、域大小、counter dtype/容量、初始化与归并成本；覆盖域边界、极端偏斜、
  最大计数、tail、重复 threshold 和完整输出语义，并与精确 Golden 和 formal compare 闭环。
- **KB 边界**：当前 KB 没有通用有限域 histogram/threshold pattern；精度、owner、同步和 tail
  约束仍分别有效。

<a id="general-05"></a>
### `general-05`：删除不会被消费的 candidate 链

- **适用**：完整 def-use/可达性证明 candidate、sentinel、value/index pair 或 scratch 的初始化、
  更新、搬运和存储不会到达任何可观察输出，且生成物仍保留对应工作。
- **动作**：整段删除已证明 dead 的生产链；删除范围同时包含它独占的中间搬运、descriptor 和
  事件，而不是顺便修改候选算法或 tie 规则。
- **边界**：rare/tail/空输入/全无效/NaN/tie/fallback 路径仍可能消费，或初始化值承担 identity、
  padding、安全地址时不能删。当前 case 未命中不是不可达证明。
- **关闭证据**：覆盖所有公共合同路径和边界，确认生成物中的整条生产链、字节和 live range
  实际消失，再做完整正确性与 formal compare。
- **KB 边界**：当前 KB 没有通用 dead-candidate 性能 pattern；已选 precision/tail/owner 合同
  仍决定 sentinel、fallback 与可达路径。

<a id="general-06"></a>
### `general-06`：跳过无效 lane 的非必要工作

- **适用**：physical Tile 大于真实 valid extent；padding/tail lane 在完成必要 neutralization 和
  安全寻址后仍参与 compare、scan、merge、candidate 更新或中间搬运，且当前能力能表达真实
  valid 域。
- **动作**：保留正确性必需的 identity/predicate/安全地址，只把后续计算和搬运裁到 valid lane。
  这不是删除 tail mask；gather/scatter 必须在形成地址前屏蔽无效 lane。
- **边界**：固定宽度指令即使 masked 仍完整发射、固定网络必须包含 padding，或额外 mask/setup
  大于节省时拒绝。不能少处理合法元素或只在最终 store 裁剪。
- **关闭证据**：覆盖 valid=0（若合法）、1、VL−1、VL、VL+1、最大 tail、aligned 与多 tile；
  由生成物/实测证明指令或搬运确实减少，并完成 formal compare。
- **KB 边界**：[tail-validshape.md](../../pypto-pro-op-kb/constraints/tail-validshape.md) 已定义正确
  有效域；本项只保留满足该合同后继续删除无效 lane 下游工作的性能增量。

<a id="general-07"></a>
### `general-07`：合并语义等价的重复工作

- **适用**：两条生产链对相同输入、范围、dtype、layout、mask、运算次序、rounding 与 tie/NaN
  语义重复执行 scan、reduction 或其它中间构造，结果也未被第一个消费者改写。
- **动作**：只计算一次，并在合法 owner/stage/可见性和生命周期内共享给全部消费者；每个账本
  项必须指出具体被合并的生产链。
- **边界**：任何语义字段不同、共享会导致 spill/容量冲突/流水退化，或编译器已完成 CSE 时不
  实施。不得合并随机状态、owner-local 状态、可变 descriptor 或不同 tail mask。
- **关闭证据**：记录删除的 pass、指令、descriptor、字节及共享 live range；覆盖所有消费者和
  tail/dtype/layout 边界，生成物确认重复链减少，再做 formal compare。
- **KB 边界**：KB 有若干 mask、gather 中间项、TileGroup/Cube operand 等特定复用实例；本项只
  保留跨形态的“先证明完全等价，再共享具体重复生产链”的通用增量。

<a id="general-08"></a>
### `general-08`：`block_dim / owner / tail` 联合调优

- **适用**：参与核数、owner 粒度与尾核负担互相影响，出现空核、长尾或明显逐核离散；只改单个
  `block_dim` 的结果不能解释或不稳定。
- **动作**：把 `(blocks, owner_grain, tail_policy)` 视为一个完整候选。每个候选先证明所有 work
  item 恰好一次覆盖，owner 之间不相交；若存在重叠，则必须证明它是合同允许且实现合法的
  reduction，再做完整正确性和 all-P0 测量。
- **边界**：先核对当前版本的 launch 校验与 `pl.get_block_num()` 语义；wrapper 只能基于允许的
  shape 元数据计算 launch 参数。不得破坏 barrier 参与者、subblock 映射或公共边界。
- **关闭证据**：逐候选记录实际 launched blocks、每核 work/bytes/descriptor、最慢核、离散度和
  tail；覆盖 task 数小于/等于/大于 block 数及不能整除的边界，任一 P0 不能被 geomean 掩盖。
- **KB 边界**：[pypto-pro-launch-block-dim.md](../../pypto-pro-op-kb/references/pypto-pro-launch-block-dim.md)
  已提供版本事实、安全门和单变量核数验证；本项只保留 blocks/owner/tail 三者联合作为一个
  性能候选的增量。

<a id="general-09"></a>
### `general-09`：提升热循环中的恒定分支

- **适用**：每行、每 rank 或每 tile 重复判断的条件对整个 owner、stage 或有限 shape class
  恒定，且生成物仍在热区保留 branch、共同 setup 或未命中路径。
- **动作**：在合法 metadata 上只分类一次，分流到有限的专用路径，使内层循环不再求值该条件，
  并确认未使用路径与活跃值确实从生成物消失。
- **边界**：数据相关或迭代携带状态不能外提；不得臆造 parser/TilingKey 能力、在 wrapper 读取
  Tensor，或删除未证明 class 的 fallback。专门化数量必须有界。
- **关闭证据**：覆盖每个 mode、分类边界、owner/stage、tail 和全部 P0；记录 branch/setup、
  Scalar/control、代码体积、资源压力和 formal compare。
- **KB 边界**：wrapper、shape、owner 与 TilingKey 资料提供合法 dispatch 门；本项只保留把当前
  已证明恒定的热循环分支提升到外层的通用性能动作。

<a id="general-10"></a>
### `general-10`：提升一般循环不变量 setup

- **适用**：shape product、axis 规范化、基址/步长、layout 或 descriptor 参数在一批 work item
  内恒定，却在热循环重复构造；生成物中 Scalar/地址/descriptor 成本可见。
- **动作**：把可证明且当前 API 可表达的不变量提升到复用范围外，只在循环内更新真实变化的
  offset、tail 等字段。没有可复用 descriptor 对象时只提升合法标量参数。
- **边界**：per-item valid shape、tail、offset 或 owner 不能当作不变量；复用必须服从元素/字节
  单位、layout、异步消费和 API 生命周期，且不能因 live range 扩大引发 spill/容量/流水退化。
- **关闭证据**：覆盖 shape/axis/dtype、首尾 item、owner 边界、tail 与槽位回绕；用生成物证明
  setup 真正移出循环，并记录 Scalar/SCALARLDST、descriptor、资源和 formal compare。
- **KB 边界**：KB 已覆盖 full-mask、部分地址/TileGroup 等具体提升实例；本项不重复这些动作，
  只保留其它经逐字段证明的一般 shape/axis/base/stride/descriptor 不变量提升。

<a id="general-11"></a>
### `general-11`：删除没有消费者的同步边

- **适用**：一条 hot-path wait/barrier 所保护的资源已没有后续消费者；能把该边映射到具体
  producer、consumer、资源和退休点。
- **动作**：每次只 ablate 一条已证明无消费者的逻辑边，重新建立 dependency DAG。ratio 下降、
  “看起来重复”或未复现 race 都不是证明。
- **边界**：跨 section/engine/subblock、scratch store→load、同址 scatter、回绕覆盖和官方 ST/API
  要求的同步全部保留；不把 auto-mutex 外推成其它可见性保证。
- **关闭证据**：逐边记录原同步、消费者证明、生成物位置和 ablation diff；覆盖单拍、多拍、槽位
  回绕、aligned/tail，检查 wait、同-lane overlap、错误/超时与 formal compare。出现 hang、stale
  value 或设备告警立即恢复。
- **KB 边界**：[sync-stitch.md](../../pypto-pro-op-kb/constraints/sync-stitch.md) 已规定同步正确性和
  受管理依赖边；本项只保留“证明无消费者后逐边删除”的独特性能动作，不重复删除已被
  auto-managed dependency 覆盖的通用指导。

<a id="general-12"></a>
### `general-12`：一个 task 批处理多个独立 work item

- **适用**：每行或每个小 batch 的有效工作很少，却各自重复建立 task、descriptor、Tile/VF
  setup 或同步；多个 work item 彼此独立、布局兼容，能由同一 owner 成组处理，并且成组后不会
  恶化访问形态。
- **动作**：把少量独立 work item 放入一个 owner task，公共 setup 每组只做一次；目标能力
  允许时可使用合法二维 Tile，否则在一个 owner task 内循环。批大小、布局和搬运形式必须由
  当前容量、并行度与实测确定。
- **边界**：行间存在前缀、递推、跨行归约、随机状态或可见顺序依赖，批处理会造成同址写、
  owner 冲突、碎片化搬运、容量超限或并行度不足时不适用。最后一组真实 item 数、每项 tail、
  GM offset 与 valid shape 必须分别正确。
- **关闭证据**：扫描少量有依据的批大小，记录 task/descriptor/VF setup 数、每 task 工作量、
  逐核负载、最慢核、容量与 formal compare；覆盖 item 数小于、等于、大于批大小，不能整除、
  单 item、最大 tail、dtype/layout 和 owner 边界，并证明每个逻辑 item 恰好处理一次。
- **KB 边界**：[vec-row-reduce-broadcast.md](../../pypto-pro-op-kb/patterns/vec-row-reduce-broadcast.md)
  已给出独立行归约的批行模式；本项不复制该特定 pattern，只保留可跨拓扑验证的通用
  independent work-item/task batching 增量。

<a id="general-13"></a>
### `general-13`：增大单个逻辑 item 的合法 Tile/strip

- **适用**：同一逻辑 item 被切成大量小 tile，每段重复 descriptor、mask、VF/Cube setup、轮转
  或同步，固定成本随 tile 数增长并成为主要开销；相邻分段可合法合并，容量仍能保留必要的
  轮转槽和 live roles。
- **动作**：保持同一 item 与 owner 不变，在对齐、物理 Tile、内存、live-range 和真实访问形态
  允许的范围内扩大 strip，用有限候选搜索固定开销、并行度、tail、资源和访问拐点之间的甜点。
- **边界**：当前 Tile 已受容量、slot、寄存器、Mat/Cube geometry 或合法 VL 上限约束，扩大后
  会降低并行度、制造长尾、恶化 burst/bank、破坏流水或改变分段发布/递推语义时不适用。
  valid shape、tail neutral、offset 单位和全部 load/store/reduction 语义必须随新宽度重算。
- **关闭证据**：记录 tile/setup/descriptor/sync 次数、每拍字节、容量/slot/寄存器、逐核工作和
  formal compare；覆盖小于一 tile、恰好一 tile、多个整 tile、余数 1、最大余数、槽位回绕和
  全部 P0。结构变化使历史参数判断失效时，重开受影响候选。
- **KB 边界**：[pypto-pro-framework-findings.md](../../pypto-pro-op-kb/references/pypto-pro-framework-findings.md)
  已覆盖“DMA 太窄时加宽 column tile”的具体方法，[tiling.md](../../pypto-pro-op-kb/constraints/tiling.md)
  规定合法性门；本项不复制窄 DMA 指导，只保留由 descriptor/mask/VF/Cube setup、轮转或同步
  固定成本主导时，减少单 item tile 数以摊薄固定成本的独特增量。

<a id="general-15"></a>
### `general-15`：消除数据并行路径的 Scalar 往返与循环

- **适用**：compare/select/reduce/merge 本可数据并行，生成物却仍有 get/set-value、Tile 结果经
  Scalar 再广播，或 per-element/per-row/per-rank Scalar loop。
- **动作**：在当前公开 API 允许时，把同一工作留在 Tile 或 VF 数据流中，并只保留语义必需的
  reduction；不改变 owner、数学语义或输出合同。
- **边界与证据**：若并行化改变浮点归约树或运算次序，必须按冻结精度合同重新证明精度，并在
  适用时遵守 `general-02`、`general-16` 的边界；存在真实串行依赖、目标 API 不支持，或额外
  cast/workspace/sync 抵消收益时不实施。覆盖全部 P0 和 tail，用生成物证明往返/循环消失，并记录
  Scalar、Vector 与 formal compare。
- **KB 边界**：[vec.md](../../pypto-pro-op-kb/constraints/vec.md)负责合法实现 surface；本项只保留
  “消除本可并行的数据路径中的 Scalar 往返”这一通用性能动作。

<a id="general-16"></a>
### `general-16`：静态化有界小控制与归约拓扑

- **适用**：冻结合同证明 loop trip 或网络拓扑固定且很小，动态回边、反复 VF 启停或单 accumulator
  依赖链成为主要成本。
- **动作**：把 `(网络结构, accumulator 数, merge fan-in, unroll)` 作为一个有界候选，选择最小合法
  静态结构；数学工作量、owner 与输出语义不变。
- **边界与证据**：动态范围、代码体积、寄存器/片上容量或浮点次序风险不能被证明时不实施。记录
  依赖深度、生成物、spill/容量、精度边界和 formal compare。
- **KB 边界**：[precision.md](../../pypto-pro-op-kb/constraints/precision.md)的 deep-K 累加和
  [fixed-arity pattern](../../pypto-pro-op-kb/patterns/vec-tensorlist-fixed-arity.md)优先；本项只覆盖
  未被已选 KB 完整规定的通用控制/归约拓扑。

<a id="general-17"></a>
### `general-17`：按消费者所需域缩小计算范围

- **适用**：SPEC 与完整 def-use 证明消费者只需要完整 sort/scan/reduction 域的严格子集，且省略
  部分不可能影响任何可观察输出。
- **动作**：把输入域、中间归并域或归约域裁到该必要子集，同时保留原顺序、tie、NaN/Inf、index
  与 tail 语义。
- **边界与证据**：不得从样本推断合法域；TopK、dead candidate、无效 lane 和重复生产链分别归
  `general-03/05/06/07`。记录等价证明、工作量/字节变化、边界正确性与 formal compare。
- **KB 边界**：已选 scan/sort/reduction pattern 的完整语义优先；本项不复制其具体算法或 API。

<a id="general-18"></a>
### `general-18`：按已证明值域窄化内部表示

- **适用**：内部 key、index 或 value 的完整可达范围可由冻结合同或允许的 host metadata 证明，且
  当前宽度造成可测的 Scalar、Vector、寄存器或搬运成本。
- **动作**：只把内部表示改为目标 API 支持的最窄安全宽度；公开输入输出 dtype 与数值合同不变。
- **边界与证据**：不得扫描 Tensor、按样本缩窄或绕过溢出/精度门；已选 KB 的具体位宽规则优先。
  覆盖范围边界和特殊值，核对生成物、容量/字节、正确性及 formal compare。
- **KB 边界**：gather/scatter 与 precision 页面已有的具体宽度要求不在本项重复。

<a id="general-19"></a>
### `general-19`：消除非合同边界的中间落盘与回读

- **适用**：仍被消费的中间值先写 GM 或额外 UB staging 再读回，但 producer/consumer 的 live range、
  dtype、owner、同步与容量证明允许直接衔接或合法片上驻留。
- **动作**：删除该非必要 store/load 或 staging copy，保持数据流、精度和公开边界不变。
- **边界与证据**：公开输出、已选 precision 要求的 GM 中间、合法 fusion/engine 边界、容量或同步
  不允许时不得删除。记录 live-range/DAG、实际字节和指令、timeline、完整正确性与 formal compare。
- **KB 边界**：本项不替代 memory-layout、sync、tiling 和 wrapper 合同，也不重复
  `general-05` 的 dead chain 或 `general-07` 的重复生产链。

## 使用边界

- 索引中的 active item 都是待验证候选，不是无条件代码要求；逐项评估并按
  [实验闭环](optimization-playbook.md)关闭。
- 同一改动可以共享实验，但每个 source id 保留独立结论；结构变化使旧结论失效时重开。
- 平台页和 general knowledge 不是预置来源；它们可辅助解释新鲜瓶颈并形成自主假设，但假设必须
  先按[实验闭环](optimization-playbook.md)登记为 `bottleneck_derived`，不能扫描这些资料扩充预置
  分母。模板项由[模板优化项索引](../templates/INDEX.md)作为独立来源枚举。
