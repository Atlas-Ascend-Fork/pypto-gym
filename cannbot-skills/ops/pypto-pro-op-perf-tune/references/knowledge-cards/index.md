---
okf_version: "0.2"
---

# 性能优化知识卡片库

本目录是一个采用 [Open Knowledge Format（OKF）v0.2](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md)
组织的 PyPTO-Pro 性能优化方法库，供性能调优时查阅和实验。卡片说明适用条件、改法、技术限制与验证方法，
内容质量在合入前由技术人员把关。

本文件是卡片清单的**唯一入口**，调优时只从 Active 表获取候选，不通过扫描目录自动采用卡片。
实际卡片按稳定类别放入子目录。已内置 16 张 VF/VEC 卡片，位于 `vec/`，编号为 `vec-01` 至
`vec-16`；11 张 Cube 卡片，位于 `cube/`，编号为 `cube-01` 至 `cube-11`；5 张 MEM 卡片，位于
`mem/`，编号为 `mem-01` 至 `mem-05`；5 张 PIPE 卡片，位于 `pipe/`，编号为 `pipe-01`
至 `pipe-05`；3 张 SCALAR 卡片，位于 `scalar/`，编号为 `scalar-01` 至
`scalar-03`；另有 1 张跨引擎卡片，
位于 `cross-engine/`，编号为 `cross-engine-01`。以上卡片均以 `status=stable` 登记到 Active 表。

## Active items

新卡默认登记到本表，合入前由技术人员 review。调优时按本表列出候选，
由使用者结合卡片描述、当前算子和目标能力判断适用性。

| item_id | 卡片与锚点 | status | bound_hint | applicability | target/api_gate |
|---|---|---|---|---|---|
| `vec-01` | [Vec Tile / VF 内融合，消除 GM 往返](vec/vec-01-ub-fusion.md) | `stable` | `VEC、MTE2、MTE3` | 相邻 Vector 链的中间结果无外部消费者，生成物仍有中间 GM 回写再读入，且 Vec 容量可容纳融合 live set | 仅限已核验 VF、TileGroup 与 auto_mutex 的 Ascend 950PR 或 950DT 工具链；其它 SoC 重新查表并验证 |
| `vec-02` | [减少、批量或就近执行 Cast](vec/vec-02-reduce-cast.md) | `stable` | `VEC` | 存在冗余 Cast、可批量或就近转换的机会，或不必要的转换中转 Tile，且 dtype、rounding、布局与值域允许相应改写 | 仅限 Ascend 950PR 或 950DT；Tile 级 pl.cast 需当前 ST 支持，UNPK/PACK 须有明确分布模式与端到端 value ST |
| `vec-03` | [`vf.mul_dst_add` 融合乘加](vec/vec-03-fused-instr.md) | `stable` | `VEC` | 存在 x 乘 weight 加 bias，编译结果未自动融合，中间乘积无其他消费者且 dtype 与 mask 兼容 | 仅在当前 Ascend 950PR 或 950DT PyPTO-Pro 公开 Vf.mul_dst_add 语义和生成 vmadd 均已复核时采用 |
| `vec-04` | [标量 round-trip 改为寄存器广播与批量计算](vec/vec-04-scalar-roundtrip-vectorize.md) | `stable` | `VEC、SCALAR` | 每行归约后存在标量往返或不必要的 Tile store→load；消费者可用同 VF full、跨阶段单 B32 广播，或已通过八哨兵 value ST 的 BLK 路径 | 仅限 Ascend 950PR 或 950DT；Vf.full 与 BRC_B32 可作保守路径，BLK 和 stride-zero 多行布局须另证 |
| `vec-05` | [消除中间 Vec Tile 暂存（寄存器内直算）](vec/vec-05-eliminate-ub-staging.md) | `stable` | `VEC、MTE2、MTE3` | 中间 Vec Tile 仅供紧邻 VF 步骤使用且无跨循环、输出或 bank-conflict 角色，生成物存在 store-load | 仅限 Ascend 950PR 或 950DT；使用已核验的 Vf.full、reduce、shuffle、mem_bar 与 Cast，布局能力须补 value ST |
| `vec-06` | [不变量广播一次 + 多行 VL merge](vec/vec-06-broadcast-once-vl-merge.md) | `stable` | `VEC、MTE2` | 外提时数据须在复用循环内不变；VL merge 时须证明 R、packed 布局及 Tile、tiling、mask、dispatch 一致 | 仅限 Ascend 950PR 或 950DT；BRC_B32 是保守路径；BLK 须八哨兵 value ST；VL merge 须完整 packed/蝶式布局与尾 mask value ST |
| `vec-07` | [`vf.reduce_*` 硬件树形归约](vec/vec-07-low-latency-reduce.md) | `stable` | `VEC` | 当前归约由标量循环或线性依赖链实现，结果只需单值或可广播，且 mask 与 dtype 满足 reduce 接口 | 仅限 Ascend 950PR 或 950DT；reduce、full 与 FIRST_ELEMENT 的语义、对齐和 dtype 支持均须核验 |
| `vec-08` | [归约累加链多路展开](vec/vec-08-multi-accumulator-unroll.md) | `stable` | `VEC` | 结合性能指标及源码、生成物或 trace 判断单累加器 RAW 链是瓶颈且 spill 不是主因，归约长度足以摊销多累加器与最终合并 | 仅限 Ascend 950PR 或 950DT；示例所需 mul_add_dst MERGING 当前文档不支持，采用该示例前须另证；另核验 reduce、动态 pl.range 与展开度 |
| `vec-09` | [外层循环下沉进 VF](vec/vec-09-loop-sink-into-vf.md) | `stable` | `VEC、SCALAR` | kernel 按行重复调用同一 VF，行间无依赖，setup 跨行不变且合并后的寄存器与 Tile 生命周期可容纳 | 仅限 Ascend 950PR 或 950DT；确认 vector_function 内动态 pl.range、offset 与逐行 mask 语义后采用 |
| `vec-10` | [小定长循环显式展开](vec/vec-10-small-loop-unroll.md) | `stable` | `VEC` | trip count 编译期已知且很小，展开后的数据、offset 与 mask 等价 | 仅限 Ascend 950PR 或 950DT；只使用已核验 VF 基础算术，是否源码展开由当前 parser、生成物和 TilingKey 决定 |
| `vec-11` | [减少超越函数计算量（PyPTO-Pro 能力门控）](vec/vec-11-reduce-compute-lut.md) | `stable` | `VEC` | 超越函数主导热点，且存在以下机会之一：重复调用复用、公开融合接口或有当前 API、编译产物和精度证据的 LUT/近似方案 | 仅限 Ascend 950PR 或 950DT；采用目标版本支持的公开超越函数或融合接口，所需 precision/LUT 能力未明确支持时记录 capability gap |
| `vec-12` | [寄存器溢出时按条件拆分 VF](vec/vec-12-preg-split.md) | `stable` | `VEC` | 目标路径生成物或 trace 出现 spill/reload，分支条件可证明，且专用 VF 能真正删除整段逻辑和活跃值 | 仅限 Ascend 950PR 或 950DT；通过 TilingKey 或合法控制流分流，RegTraitNumTwo 仅作后端风险模型 |
| `vec-13` | [VF 尾块统一 mask](vec/vec-13-vf-tail-unified-mask.md) | `stable` | `VEC、SCALAR` | full 与 tail 代码体重复，逐拍 active 可由 total 减 offset 精确重算，offset 与 mask 使用同一元素粒度，且不违反已选 KB 中前提成立的 full-mask 外提义务 | 仅限 Ascend 950PR 或 950DT；须核验 vf.update_mask 不回写 Python 标量，lane 常量按 dtype 与生成物确定 |
| `vec-14` | [对齐分段 Tile 布局](vec/vec-14-aligned-split-copy.md) | `stable` | `MTE2、MTE3` | 非 32B 段起点在生成物或 trace 中造成对齐退化，且独立 Tile 及 padding 可被 Vec 容量容纳 | 仅限 Ascend 950PR 或 950DT；须核验 pl.load/store 元素 offset、Tile physical/valid shape、TileGroup 地址与对齐合同 |
| `vec-15` | [布局先行 + 64-lane 向量树化，消除跨 lane 归约与标量 pack](vec/vec-15-layout-first-vector-tree.md) | `stable` | `VEC` | VF 热点沿某一轴做归约（在线 softmax 行 max/段和等），分数矩阵布局可翻转为列=query（cube 侧把 Q·Kᵀ 改写为 K·Q̃ 类形式并按 N 维拆分搬运），每条 64-lane load 覆盖一个归约行 × 64 个查询，归约结果无跨行交叉消费且翻布局后 live set 可被容量容纳 | 仅限 Ascend 950PR 或 950DT；依赖已核验的 vf.load_align/vf.store_align、vf.max/muls/add 树、vf.reduce_max/reduce_sum 与 AccToVecMode.DualModeSplitN；store_unalign tracker 语义须按当前版本复核 |
| `vec-16` | [量化/统计类 epilogue 直接产出消费者最终物理布局，省去中转 repack pass](vec/vec-16-produce-final-layout-skip-repack.md) | `stable` | `MTE3` | 量化/归约类中间结果的内部紧凑布局与下游 GM 契约 pad/分组布局不同，当前用逐行/逐元素二次搬运重排，且计算阶段本身可以自由选择直接按目标 stride/order 产出 | 仅限 Ascend 950PR 实测；依赖已核验的 pl.store/pl.load 的 order 跨轴映射；能否直接产出目标布局由具体计算的向量化实现决定，需逐个验证 |
| `cube-01` | [matmul `phase=` 细粒度 M↔FixPipe 流水（unit_flag 硬件握手）](cube/cube-01-unitflag-fine-grained-fixpipe.md) | `stable` | `MAC、FIXPIPE` | Cube kernel 的 K 循环累加器经 pl.store/store_tile 直接写回 GM（无 Acc→Vec epilogue），profiling 显示 FIXPIPE bound 或 MMAD 与搬出整段串行，且 L0C 被完整输出块占满无法开双缓冲 | 仅限 Ascend 950PR 或 950DT；phase= 仅在 drain 为带 phase= 的 pl.store/store_tile 时合法；pl.move 是否携带 phase 形参随版本变化，Acc→Vec 配对合法性须按目标版本重新验证，验证前禁用维持 |
| `cube-02` | [L1 bank 冲突规避（ping/pong 分居前后半 L1）](cube/cube-02-l1-bank-half-split.md) | `stable` | `MTE1` | K 循环 matmul 的 L1 操作数 buffer 数 ≥2（ping/pong），profiling 显示 MTE1 bound 或 MMAD 断流，且单个 buffer 数据总量不超过半个 L1 | 仅限 Ascend 950PR 或 950DT；依赖 make_tile_group 的 addrs 列表显式指定各 buffer 地址与 auto_mutex 轮转；L1 容量与 bank 边界按目标 ini 复核 |
| `cube-03` | [StreamK / Split-K：K 维跨核切分 + 原子累加部分和](cube/cube-03-streamk-atomic-k-split.md) | `stable` | `MAC` | matmul 的 M/N tile 数明显小于可用核数且 K 足够长；输出 dtype 支持原子累加；FP32 部分和精度预算允许 | 仅限 Ascend 950PR 或 950DT；依赖 pl.store/store_tile 的 atomic=AtomicAdd（目的区域须预先初始化）与 kernel[stream, block_dim] 启动 |
| `cube-04` | [FullLoad：小侧操作数全量驻留 L1](cube/cube-04-fullload-resident-l1.md) | `stable` | `MTE2` | matmul 一侧矩阵字节数 ≤ 可用 L1 预算（预留对侧流式与轮转空间），对侧循环次数 ≥2，且当前为 MTE2 bound | 仅限 Ascend 950PR 或 950DT；依赖 pl.load 单次大搬运与 pl.move 的 offset 切片语义（源 tile 大于目的 tile 时按元素偏移读） |
| `cube-05` | [MTE2 预取（显式 ping/pong 组 + 消费后回填）](cube/cube-05-mte2-preload-explicit-pingpong.md) | `stable` | `MTE2` | K 循环 matmul 已有双缓冲语义但流水仍见 MTE2 空泡（搬运发射滞后于数据依赖解除），且 k_blocks ≥ 2 | 仅限 Ascend 950PR 或 950DT；依赖独立 tile group + auto_mutex 的同组内 pipe 排序；动态奇偶分支选择句柄的形式须按当前 parser 复核 |
| `cube-06` | [GroupedMatmul 组间连续核分配（全局线性 tile 空间）](cube/cube-06-grouped-continuous-core-assign.md) | `stable` | `MAC` | 单 kernel 承载多组 matmul（MoE 分组、batch matmul 等），各组 tile 数不被核数整除，组数较多 | 仅限 Ascend 950PR 或 950DT；依赖 pl.get_block_idx/pl.get_block_num、三维 GM 张量的组维寻址与单次 launch |
| `cube-07` | [权重 NZ 离线预打包（GM NZ 声明 + NZ→NZ 纯搬运）](cube/cube-07-weight-nz-prepack.md) | `stable` | `MTE2` | 推理场景权重固定、可离线预处理；matmul 为 MTE2 bound 且权重搬运占比高；原始内轴不对齐时随路转换代价更高、收益更明显 | 仅限 Ascend 950PR 或 950DT；GM Tensor 仅支持 ND/NZ 两种声明，NZ 要求调用方已按 NZ 物理排布 packing 且按对齐后容量分配；NZ 搬运不支持降序 order 转置 |
| `cube-08` | [同一 L1 地址提供两种布局](cube/cube-08-dual-view-l1-tile-group.md) | `stable` | `MTE1` | 同一份数据要按原布局和转置布局供两条 Cube 输入路径使用，且两边都能同形搬运 | 同地址 TileGroup、NZ/ZN 视角、L0 搬运和共享地址 mutex 合同需在目标版本复核 |
| `cube-09` | [Cube 串行链操作数降精度：fp32→bf16（acc 保持 fp32 累加）](cube/cube-09-bf16-operand-downcast.md) | `stable` | `MAC` | kernel 由多次 pl.matmul 串行链主导且操作数为 fp32，当前设备同规模对比测试显示 bf16 单次调用成本显著更低，精度合同允许操作数按低精度写回且 golden 容差门内可复验 | Ascend 950PR 实测；依赖已核验的 pl.TileType(Mat/Left/Right, NZ)、pl.cast(RoundMode)、pl.move、pl.insert 与 fp32 acc 语义；写回必须两步法（cast 到 bf16 ND + move 到 bf16 NZ）；其它 SoC 须重做同规模对比测试 |
| `cube-10` | [matmul N 外 / K 内循环重构（cube-bound + 小侧操作数常驻 L1）](cube/cube-10-ktail-n-split-copyout.md) | `stable` | `MAC、SCALAR、MTE1` | cube-only matmul，K 循环逐块交替累加两个 N 子块（每轮 2 个 MMAD、写两个不同 L0C 地址的 Acc）；profiler 显示 cube-bound（MAC 占用高、MTE2 占用低）；小侧操作数（如权重 B）可常驻 L1 复用（cube-04 Fullload 形态） | 仅限 Ascend 950PR 或 950DT；依赖 phase= 直排 store 与 pl.move 的 offset 切片（cube-04）；MTE2-bound / 操作数流式载入时该重构为负收益 |
| `cube-11` | [SWAT 调度：Serpentine 行窗口 + 自适应尾块](cube/cube-11-swat-serpentine-tail-scheduling.md) | `stable` | `MAC、MTE2` | matmul：机制 A 需 B 是访存大头且工作集 > 可共享 L2（固定列 B tile 复用距离超 L2）；机制 B 尾薄片拖慢行窗口；机制 C tile 数不整除核数致末轮空闲核 | 仅限 Ascend 950PR 或 950DT；机制 A 已核验但两形状未测出收益（B 驻留 L2），机制 B/C（变尺寸 tile/二维子块）合法性未核验，采用前须另证 |
| `mem-01` | [对齐连续主路径 + 隔离尾块](mem/mem-01-aligned-main-tail-split.md) | `stable` | `MTE2、MTE3` | 同一算子同时覆盖对齐与非对齐规格，主循环每个 tile 都走动态 valid_shape/保守搬运路径，且主区可切出静态 shape 的对齐完整 Tile | 仅限 Ascend 950PR 或 950DT；依赖 TileType 静态 shape、valid_shape=[-1,-1] 与 set_validshape |
| `mem-02` | [按 UB 预算批量搬行](mem/mem-02-batched-row-copy.md) | `stable` | `MTE2、MTE3` | 循环中反复按单行或单个小张量搬入/写出，相邻行在 GM 连续排布且下游同阶段消费，UB 预算可容纳多行 Tile | 仅限 Ascend 950PR 或 950DT；依赖 pl.load/pl.store 的二维 Tile 整块搬运与 make_tile_group 轮转 |
| `mem-03` | [一次性数据绕过 L2（能力门控）](mem/mem-03-l2-bypass-one-shot.md) | `stable` | `MTE2` | 大权重/scale/输入块只被完整消费一次且可由 shape/schedule 谓词证明无后续复用，热数据 L2 命中被流式读挤压 | 能力门控：PyPTO-Pro 当前版本无 L2 cache hint/bypass 公开 API；`pl.system.dcci` 仅做缓存清理失效，不构成 bypass |
| `mem-04` | [连续完整结果一次性直写](mem/mem-04-direct-contiguous-output.md) | `stable` | `MTE3` | 结果在 UB 中已按输出地址连续排布却仍按行/段逐个搬出，且后续无中间变换、结果只需写入最终 GM 区间 | 仅限 Ascend 950PR 或 950DT；依赖 pl.store 整块搬运与 pl.move 的 UB→UB offset 读 |
| `mem-05` | [Identity/Pure-Copy 快路径路由](mem/mem-05-identity-pure-copy.md) | `stable` | `SCALAR` | 算子存在退化语义（identity reshape、单段 split、参数使变换退化为 copy），且该条件可由 shape/axis/layout/属性在 tiling 阶段完整证明 | 仅限 Ascend 950PR 或 950DT；依赖 pl.load/pl.store 二维整块搬运与 host 侧模板路由 |
| `pipe-01` | [无 bound 手段清单](pipe/pipe-01-no-bound-checklist.md) | `stable` | `PIPELINE` | 各硬件单元利用率均不高、无明显单一瓶颈，trace 有流水气泡与等待区间、搬移-计算 overlap 不足 | 仅限 Ascend 950PR 或 950DT；各手段分别依赖 make_tile_group 深度轮转、auto_mutex、整块搬运与 VF 融合等已核验能力 |
| `pipe-02` | [三阶段顺序重排（计算发射优先 + 写出延后一拍）](pipe/pipe-02-stage-order-rescheduling.md) | `stable` | `PIPELINE` | TileGroup 双缓冲已开但 trace 显示搬入-计算-写出仍串行（写回等待刚发出的计算、搬运发射反压计算发射），且每核 tile 数足够形成流水 | 仅限 Ascend 950PR 或 950DT；依赖 make_tile_group 的 group[i] 运行时索引与 auto_mutex 依赖 |
| `pipe-03` | [C/V 交错流水](pipe/pipe-03-cv-pipeline.md) | `stable` | `PIPELINE` | Cube 和 Vector 存在相邻轮次的独立工作，但当前流水图显示整段串行 | Cube/Vector 分区、TileGroup.next()、独立 buffer 和跨核事件需在目标版本复核 |
| `pipe-04` | [共享地址 Tile Group 用显式共享计数器管理 Buffer 下标](pipe/pipe-04-shared-buffer-counter.md) | `stable` | `PIPELINE` | 同一地址空间（相同 addrs 与 mutex_ids）被多组 tile group 共享，且不同流水线阶段（QK/PV、drain、按位图跳块）对各组调用次数不一致 | 仅限 Ascend 950PR 或 950DT；仅用 pl.make_tile_group 下标取用 group[idx] 与 Python 标量计数器，已在当前工具链 kernel 核验，其它 SoC 重新查表并验证 |
| `pipe-05` | [消费点 GM→L1 直载：替换经 UB 的 NZ 转换路径](pipe/pipe-05-gm-to-l1-direct-load.md) | `stable` | `PIPELINE` | 串行链上存在仅为把 GM 数据搬入 L1 而设的 vec 暂存段（GM→UB→NZ 转换→L1），数据只被紧邻 cube 阶段消费、无需 vec 侧加工，串行链同步主导且上游恒写满全部行；vector 空闲且搬运可提前发射（与 cube 计算重叠）时保留 UB 路径更优 | Ascend 950PR 实测；依赖已核验的 pl.load（GM→Mat/L1 直达 + order 轴映射 + set_validshape 尾块语义）与 Mat/NZ TileType；尾块整行直载必须确认上游写满全部行，否则越界读未定义内存 |
| `scalar-01` | [Scalar bound 手段清单](scalar/scalar-01-scalar-bound-checklist.md) | `stable` | `SCALAR` | scalar 泳道忙而 VECTOR 空闲、总计算量极小或 shape 很小，循环内存在可外提的冗余标量计算（div/mod、重复 offset 推导） | 仅限 Ascend 950PR 或 950DT；依赖 pl.range 循环标量表达式与 tile 基础算术 |
| `scalar-02` | [热循环查表与守卫的 load-store 减载](scalar/scalar-02-lut-guard-load-shed.md) | `stable` | `SCALARLDST、SCALAR` | 热循环内存在 getval 结果再作为下一个 getval 偏移的链式查表，或每轮读取 state 类守卫标量；表项 launch 后不变且表很小，守卫条件首轮后恒成立 | 仅限 Ascend 950PR 或 950DT；须核验 pl.getval/pl.setval 动态偏移与运行期标量分支 |
| `scalar-03` | [小 tile 固定开销与粒度修正](scalar/scalar-03-small-tile-granularity.md) | `stable` | `SCALAR` | 总计算量小或 shape 小，块数远大于每块工作量（如行块高取 1~2）；scalar 占比高、发射率低，热点在逐块固定标量开销而非数据搬运或计算本身 | 仅限已核验 pl.make_tile_group/TileType、pl.range、pl.min、pl.set_validshape 的 Ascend 950PR 或 950DT 工具链 |
| `cross-engine-01` | [UB 直接交给 L1，少一次 GM 往返](cross-engine/cross-engine-01-ub-to-l1-handoff.md) | `stable` | `VEC、MTE2、MTE3` | Vector 的结果紧接着由 Cube 消费，当前存在关键路径上的 GM 写回和读回 | UB 到 L1 的布局转换、insert、valid shape 和跨核事件需在目标版本复核 |

## ID 与状态规则

- 按方法主题选择类别，优先沿用已有分类；新增类别使用简短英文名。
- `item_id` 使用 `<类别>-<两位递增序号>`，文件名使用 `<item_id>-<短名>.md`，放入类别目录。
  编号从已登记条目中该类别的最大编号递增，新类别从 `01` 开始；已登记 ID 保持不变且不复用。
- `general-*` 由[通用优化手段](../general-optimization-methods.md)独占。
- 修改卡片时同步索引中的 ID、状态、适用条件和 API 限制，方便查阅和 review。
- 索引按 `status` 分组：`stable` 为 Active，`draft` 为 Draft，`deprecated` 为 Retired；有条目时列出对应分组。
- 退役卡保留稳定 ID、历史链接和退役原因，编号不复用。
- 调优时按现有账本记录卡片 ID、文件位置和本轮结论；Draft、Retired 不作为候选。

字段、来源和状态说明见 [OKF 格式约定](PROFILE.md)。

## 贡献入口

第一次贡献先按[贡献指南](CONTRIBUTING.md)准备事实包，并使用其中可复制的提示词让 Agent 生成
卡片，默认以 Active 提交；[带注释示例](examples/annotated-card.md)展示了证据不足时如何保留语义而不补猜。
内容按[卡片模板](CARD_TEMPLATE.md)组织，合入前交技术人员 review。
采用卡片调优的执行流程见[性能调优 Skill](../../SKILL.md)。

`templates/` 是独立的模板优化项来源，其生效项由
[模板优化项索引](../../templates/INDEX.md)单独枚举，与本索引的知识卡片分别计数。

## Bundle resources

* [OKF 格式约定](PROFILE.md) - 卡片字段、可选参考资料和状态的简明说明。
* [贡献指南](CONTRIBUTING.md) - 准备事实包、驱动 Agent 生成卡片和提交人工评审的入口。
* [卡片模板](CARD_TEMPLATE.md) - 新建知识卡时使用的结构与默认字段。
* [带注释示例](examples/annotated-card.md) - 展示事实不足时如何保留语义和待确认项。
