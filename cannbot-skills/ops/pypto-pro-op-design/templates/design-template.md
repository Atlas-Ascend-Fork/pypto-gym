## 算子名称

- **op_name**: {operator_name}
- **设计时间**: {timestamp}
- **基于 SPEC**: `custom/{op}/SPEC.md`
- **基于 EXPLORE_REPORT**: `custom/{op}/EXPLORE_REPORT.md`

---

## Knowledge Bindings

> 完整、机器可读的 requirement 清单保存在生成后与本文同目录的 `DESIGN_BINDINGS.json`。
> 本文只记录 R0–R8 的实际设计决策；每条活动 requirement 的 `planned_location` 必须准确指向
> 对应决策及最终 `test_<op>.py` file/symbol，不复制 Binding 分组或 requirement 表格。

---

## §0 Module 划分（R0 输出）

> 依据[Module划分](../references/module_partitioning.md)，先按数据依赖确定Module边界，再按Section边界继续拆分。R0记录每个Module的Section归属，具体Section代码结构在§4确定。

### 维度契约

- **kernel 接收与处理维度**：{逐项列出 SPEC 要求的 rank/shape 与 kernel 内逻辑视图}
- **kernel 内维度适配规则**（SPEC 要求多维/1D 时填写）：

| SPEC 输入 shape | kernel 接收 | kernel 内索引/stride 映射 | 输出形态 |
|-----------------|-------------|---------------------------|----------|
| {如 1D [L]} | {原始 [L]} | {直接按 L 索引，不做 host reshape} | {SPEC 要求形态} |
| {2D [M,N]} | {原始 [M,N]} | {按原始 stride/offset 访问} | {SPEC 要求形态} |

### Wrapper 边界外操作

空

### 划分依据

{数据依赖边界 + 各步骤的Section归属 + 按Section边界继续拆分的结果；量化场景写明使用V流水还是FIX路径，以及Scaling参数缓冲的准备位置；若因UB等片上空间不足继续拆分，写明超限对象和拆分位置}

### Module 列表

| Module | 目的 | 输入依赖 | 输出 | Section |
|-------|------|---------|------|---------|
| Module 1 | {purpose} | {dependency} | {output} | {section_vector / section_cube} |
| Module 2 | ... | ... | ... | ... |

> 若只有一个 Module，标注"单 Module"并说明为什么不需要拆分。

### 归约轴容量结论

> 含归约算子时必填；无归约时填"不涉及归约"。

- **归约轴是否可能超单 tile**：{是 / 否，依据 SPEC 动态轴范围}
- **多Tile归约方案**：{常规多遍 / 在线统计加输出两遍 / 其他；逐遍写明遍历范围、状态和依据；单Tile装得下则写“不涉及”}

### Module 级数据流

```
{Module 间的数据流向示意；标注中间数据所在空间、传递API，以及Cube/Vector之间的同步方式}
```

---

## §1 API 映射（R1 输出）

> 按[本 Skill](../SKILL.md)的“两条实现约束”填写。每个 Vector 步骤冻结唯一实现：
>
> ```
> vector_selection:
>   step:                  <本 §1 中的哪一步>
>   implementation:        vf | tile_op
>   api_sequence:          <vf.* 或 pl.* API 序列>
>   decision_reason:       default_vf | kb_template_required
>   kb_template_evidence:  <default_vf 写 n/a；tile_op 写已选 KB 路径、明确要求的原文/片段>
>   target_version:        <目标软件/框架版本>
>   applicable_conditions: <dtype/shape/layout/算子条件>
> ```
### Module 1 API 调用序列

```
Module 1 — {名称}:
  for j in range(n_tile_num):
    {API 调用}
    ...
```

{每个 API 标注来源（API 文档路径或 EXPLORE_REPORT §4 定位的官方指定算子）}

### Module 2 API 调用序列

```
...
```

### 超越函数数值安全边界

> 条件性：仅当 API 链含 exp/log/sqrt/reciprocal/tanh 等超越/非线性函数时填写。

| API | 输入理论范围 | 目标 dtype 上限 | 是否溢出 | 防护措施 |
|-----|-------------|----------------|----------|----------|
| `{超越函数 API}` | {如 x 无上界，x>11 时 exp 溢出} | fp16 ≈ 65504 | exp(11.09)≈65504 → +inf → NaN | 先减最大值再 exp（或归一化省去分母）

---

## §2 Tile 规划（R2 输出）

### 2.1 关键常量定义

列出所有在 kernel 中使用的编译期常量。涉及 Cube 的 tile 尺寸必须满足本次资料探索确认的对齐要求，具体以 `custom/{op}/EXPLORE_REPORT.md` §7 的记录为准。

```python
# ── 算子关键常量（coder 必须逐字使用） ──
TS = {S_tile}           # S 方向 tile 尺寸
TD = {D_tile}           # D 方向 tile 尺寸
TS_HALF = {S_half}      # subblock 半尺寸 (如有 dual_split)
SCALE = 1.0 / sqrt({D_logical})  # 缩放因子（若算子有 scale 步骤）
```

> **注意**：若某维度实际值固定且 < tile 尺寸（如 D=64 对齐到 TD=128），tile shape 用 TD 声明，运行时通过 §7 的 `set_validshape` 限制有效区域。公式常量（如 SCALE）若有依赖，必须使用实际维度值而非 tile 尺寸。

### Tile 属性表

每个 tile 的 shape 由其所参与的 API 操作数要求决定：

| 用途 | 变量名 | shape | dtype | 内存空间 | layout | 大小 | 备注 |
|------|--------|-------|-------|---------|--------|------|------|
| 输入暂存 | `tile_a` | `[64,128]` | FP32 | UB(Vec) | `—` | 32768 | ... |
| ... | ... | ... | ... | {Vec/Mat/Left/Right/Acc} | {— 或具体 layout} | ... | {...} |

> 内存空间取 `Vec`(UB)/`Mat`(L1)/`Left`(L0A)/`Right`(L0B)/`Acc`(L0C)；`—` 表示未显式指定layout。`Vec`没有统一的默认layout，是否需要填写以及允许哪些值由具体API决定；其他空间的默认值也要按目标架构核对`TileType.md`。

### TileGroup槽位访问表

> 所有TileGroup都记录`depth`和逐Tile mutex配置，避免漏掉单Tile多ID以及不配置mutex元数据的情况。运行时下标不会自动取模，采用`group[i]`时必须给出`i`始终位于`[0, depth)`的依据。

| TileGroup | depth | 每个Tile的mutex ID | 访问方式 | 槽位表达式 | 同步及有界/游标说明 |
|---|---:|---|---|---|---|
| `{group_name}` | {1/2/N} | {如 `[[0, 1], [2, 3]]` / 不配置} | {`current()` / `next()` / `group[i]`} | {如 `task_idx % 2`} | {多ID用途；不配置时的手动同步；下标范围；与`next()`混用时的游标状态} |

### tile_dims stride 注意事项

> 仅当使用了 `load_tile` + `tile_dims=[d0, d1]`（覆盖多维度）时填写。大 stride 可能影响性能，具体阈值以官方指定算例经验为准。

| load_tile/store_tile | tile_dims | d0 在张量布局中的 stride | 备注 |
|----------------------|-----------|------------------------|------|
| {tile_a, x, [i,j]} | {如 [0,1]} | {计算值} | {是否需关注} |

---

## §3 片上空间布局（R3 输出）

> tile 按 `target_memory` 落到不同片上空间，**每个空间独立寻址、独立限容**——地址各自从 `0x00000` 起算，同一 addr 在不同空间是不同物理位置（证据 `matmul.md` 调用示例：L0A/L0B/L0C 同用 `addr=0x0000` 互不冲突）。纯 vector 算子只有 UB 一节；含 cube/matmul 的算子须补 L1(Mat)/L0A(Left)/L0B(Right)/L0C(Acc) 各节——`matmul` 操作数空间为硬性约束：`lhs`→L0A、`rhs`→L0B、`dst`→L0C（证据 `matmul.md` 参数类型表）。

### 片上地址映射表（按内存空间分节）

| 内存空间 | 用途 | 变量名 | shape | dtype | layout | 起始字节 | 每槽字节数 | 槽位数 | 生命周期 | 结束字节（不含） | 备注 |
|---------|------|--------|-------|-------|--------|---------|-----------:|------:|----------|------------------|------|
| UB(Vec) | ... | ... | ... | ... | {— 或具体 layout} | ... | ... | ... | {轮转/驻留/临时} | ... | {...} |
| L1(Mat) | ... | ... | ... | ... | {— 默认} | ... | ... | ... | {轮转/驻留/临时} | ... | {含 cube 时填} |
| L0C(Acc) | ... | ... | ... | ... | {— 默认} | ... | ... | ... | {轮转/驻留/临时} | ... | {含 cube 时填} |

> 地址使用半开区间`[起始字节, 结束字节)`。连续槽位的结束字节为`起始字节 + 每槽字节数 × 槽位数`；槽位不连续时逐槽列出起止字节。生命周期重叠的 tile 地址不得相交。

**各空间最高地址上界**（逐空间列出，无对应 tile 的空间可省略）:
- UB(Vec): {max(结束字节（不含）)} / {EXPLORE_REPORT §7 UB 容量} = {百分比}
- L1(Mat) / L0A / L0B / L0C（如有 cube）: {max(结束字节（不含）)} / {EXPLORE_REPORT §7 对应容量} = {百分比}——容量取 §7 探测记录，§7 未记录则回退 material-explore 补测，不臆测

### 分配方式选择

> **性能强制**：所有需要 buffer 切换/轮转的 tile 一律用 `make_tile_group` + `auto_mutex`；`make_tile` 仅限单次使用 scratch tile（不参与 buffer 轮转）。禁止用 `make_tile` + 手动 `sync_src`/`sync_dst` 管理 buffer 轮转。

- **方案**: make_tile_group + auto_mutex（TileGroup提供多槽buffer，通过`next()`或显式下标选择槽位；auto_mutex根据mutex信息管理core内跨Pipe依赖）；若有单次使用 scratch tile 用 make_tile
- **依据算子**: {EXPLORE_REPORT §4 定位的最相似官方指定算子路径}
- **理由**: {基于 R0 Module 划分，说明槽位选择方式，以及auto_mutex如何覆盖本算子的core内跨Pipe依赖}

### double buffer 地址规划

> `make_tile_group` 的 buffer 数 > 1 时地址占用按倍数放大，须在此显式记录。buffer 数取值参照官方指定算子中相似算子的实际配置。

| 项目 | 结论 | 受影响 tile/地址 |
|------|------|-----------------|
| buffer 数 | {1 / 2 / ...} | {tile 名单} |
| PONG 地址 | {无 / 预留} | {地址范围} |
| 槽位访问 | {`next()` / `group[slot_idx]`} | {`slot_idx`的定义与范围} |

---

## §4 循环与 Section 结构（R4 输出）

> 依据[循环与Section结构设计](../references/loop_design.md)，把§0确定的Module放入具体Section代码结构，再组织循环和跨Tile状态，并确定分核信息的获取位置。若API数据通路与§0记录的Section归属冲突，回到R0修正Module划分。

### 参考样例

- **主要参考样例**: {EXPLORE_REPORT.md §4 定位的官方指定算子路径}
- **可复用结构点**: {该样例中可复用的Section / 循环 / 分核信息获取方式}
- **补充参考**（如有）: {`../pypto-pro-material-explore/references/official_samples.md` 清单中其他参考样例路径及参考点}

### 本算子结构说明

- **Module / Section对应关系**: {每个Module放在哪个Section；相邻同域Module是否共用一个Section}
- **结果单元**: {一次完整计算产出的输出范围；如一个输出Tile、一个Softmax行块或一个Matmul `[M_tile, N_tile]`块}
- **跨Tile状态生命周期**: {状态名称、初始化位置、更新方式和使用位置；不涉及则填“不涉及”}
- **Section结构**: {section_vector / section_cube的数量、顺序，以及循环相对Section的位置}
- **循环嵌套**: {参照样例说明 M-tile / N-tile 等循环嵌套关系}
- **动态循环上界**: {如 `n_tiles = (N + TILE_N - 1) // TILE_N`；静态shape则写固定值及来源}
- **分核信息**: {核编号、核数及其获取位置；混合Kernel的Vector Section还要写明AIV核数的计算方式}

> 同步点见 §6（R6），尾块处理见 §7（R7）。

### CV 并行流水设计（`is_fusion=true`时必填）

> 当前框架提供自动 CV 并行流水功能。读取`$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/advanced_programming/auto_parallel_pipeline.md`并逐项核对“使用约束”：全部满足时推荐自动流水；任一不满足时，记录具体条款并按[CV融合算子手动预加载流水设计](../references/cv_fusion_pipeline.md)采用手动流水。两种方式都先确定第一阶段每次交给下一阶段的数据范围，以及哪些循环索引会产生下一份数据，再为这些数据分配连续编号，并展示稳定运行时Cube与Vector同时处理不同编号数据的时序。流水结构必须在编排 Stage 3 冻结，不得留到 Stage 5 才引入。

- **流水实现**: {`auto` / `manual`；逐项列出目标版本自动流水“使用约束”的核对结果；选择manual时写明不满足的具体条款}
- **自动流水配置**: {不涉及 / stage函数及顺序；跨核TileGroup的`fwd_ids`/`bwd_ids`；串行精度验证配置（本次指定版本为`PipelineConfig(preload=0)`，版本变化时以缓存文档为准）；候选与初始`preload`；`pipeline_generated.py`复核位置}
- **流水编号**: {第一阶段每次交给下一阶段的数据范围；共同确定该数据位置的循环索引；`task_id`递增位置；Cube/Vector两侧的编号关系}
- **阶段链**: {按声明顺序列出阶段及所属Cube/Vector执行域；相邻同执行域计算的合并方式}
- **逐交接的1:2分工**: {逐一列出每一处Cube↔Vector交接；该处subblock 0和1分别处理的数据范围、尾块和工作量；存在依赖时引用§5/§6的GM合并与同步方案}
- **阶段延迟表**: {`preload`取值；每个阶段所属Section、同执行域上一个阶段、delay计算式、执行条件和处理的数据编号}
- **上下文循环缓冲（按需）**: {迭代信息能直接由`task_id`推导时填“不使用”；否则记录字段、深度、当前写槽、各阶段读槽表达式和信息生存期}
- **预加载轮数与缓冲深度**: {候选预加载轮数、计划采用的轮数和性能依据；各共享数据与状态缓冲的深度及推导过程}
- **流水启动、稳定运行和末尾剩余阶段**: {两侧的启动条件；最后一个新任务进入后各自还需执行的轮数；画出稳定运行时不同编号数据的Cube/Vector重叠}
- **槽位的初始可写状态**: {Vector Section主循环前发送的释放事件，或首次写使用的单独启动路径}

---

## §5 分核策略（R5 输出）

> 📌 **权威依据（必读，官方标准）**：`$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md`（分核方式、负载均衡、核数设置和tiling传参以此为准）。

### 分核方式

- **方案**: strided loop —— {扁平切 `pl.range(core_id, m_tiles*n_tiles, num_cores)` / 二维切 外 `range(core_id, m_tiles, num_cores)`+内 `range(0, n_tiles, 1)`} + {选择理由}
- **host 侧 block_dim**: {仅Vector Kernel使用`vector_core_num`，Cube或混合Kernel使用`core_num`}；`total_tasks > 0` 时 `block_dim = min(max_blocks, total_tasks)`；`total_tasks = 0` 时仅允许经目标验证的 `block_dim=1` 空工作单次启动，否则报 `design_violation`，禁止 `block_dim=0` 或跳过启动
- **交付 launch 合同**: 恰好 1 次；唯一 kernel `{kernel_symbol}` 由 `{wrapper_symbol}` 在 host 循环外调用一次；若做不到，记录证据并返回 `failure_category: design_violation`，不得填写多 launch fallback
- **TensorList ABI（不涉及则填 N/A）**: {冻结合同中的有限 `L_MAX` 及 `B_MAX=L_MAX`；多个 TensorList 形参间的长度关系；每槽有限元素数上限 `N_MAX` 对所有 `DT_INT32` 派生式的安全性；只生成一组 `B_MAX` 个固定槽位，每个 TensorList 形参逐槽展开为独立 Ptr 参数；wrapper 对真实槽位的校验和 `n_i=0` 填充；连续且 rank 无关的适用依据，或非连续/rank 相关情形所需的独立有界 ABI；地址值不进入 tiling 数据}
- **A5 CV 的两个 Vector subblock 分工**: {不涉及 / 切分轴；subblock 0和1各自的索引范围、尾块归属和工作量；未同时使用两个subblock的原因}
- **两个 Vector subblock 的结果依赖**: {无依赖，可独立处理 / 存在归约等依赖：为何不能改切非归约轴、GM部分结果的shape与地址、最终合并者；同步细节见§6}

---

## §6 核间同步（R6 输出）

> 带非空`mutex_ids`的TileGroup由`auto_mutex`管理执行区内部的跨Pipe依赖；未配置`mutex_ids`时，按§2记录的数据路径手动插入核内同步。Cube与Vector之间的数据交接使用`set_cross_core`/`wait_cross_core`，具体规则见[跨核同步](../references/cross_core_synchronization.md)。
> 满足目标版本`auto_parallel_pipeline.md`全部“使用约束”时推荐自动流水，由框架生成核间同步并复核`pipeline_generated.py`；否则按[CV融合算子手动预加载流水设计](../references/cv_fusion_pipeline.md)显式设计同步。手动流水的编号错位、通用多阶段delay计算、预加载和末尾剩余阶段执行规则也见该文档。跨核同步须对应共享缓冲区的实际第一次读、最后一次写和槽位复用位置。
>
> **条件性**：Cube与Vector之间有数据依赖，或存在需要`INTER_BLOCK`、`INTER_SUBBLOCK`、`UNICAST_BLOCK`处理的依赖时填写；否则填“不涉及cross_core”。Section数量本身不是判断依据。

### cross_core 涉及判定

- **是否涉及 cross_core**: {是 / 不涉及}
- **依据**: {依赖是否跨Cube/Vector执行域或跨Block/subblock；若跨Block/subblock，说明为何不能用普通核内同步或算法重构处理}
- **流水实现**: {`auto` / `manual`；列出阶段延迟表、最后一个新任务进入后两侧还需执行的轮数和稳定运行时序的引用位置；自动方式补充`fwd_ids`/`bwd_ids`及生成代码复核结果，手动方式补充事件位置；使用上下文缓冲时补充其深度}

> 若"不涉及"，以下各表填"不涉及"或省略。

### 共享数据与槽位映射

| 共享数据/TileGroup | depth或槽位数 | 生产者/Section | 消费者/Section | 生产者访问 | 消费者访问 | 就绪/释放事件 | 槽位一致性依据 |
|---|---:|---|---|---|---|---|---|
| `{group_name}` | {2/N} | {Cube/...} | {Vector/...} | {`group[slot_idx]`} | {`group[slot_idx]`} | {就绪事件ID；复用时的释放事件ID} | {`slot_idx = task_idx % depth`及范围证明} |
| `{workspace_name}` | {1/N} | {Cube/...} | {Vector/...} | {地址/offset表达式} | {地址/offset表达式} | {逐任务的就绪/释放事件，或单次使用依据} | {同一逻辑块映射到同一地址} |

> `group[i]`不读取也不推进轮转游标。若一侧或两侧使用`next()`，必须说明初始游标、调用次数和分支路径为何仍选中同一物理槽位。

### 同步点

> 按数据方向和物理槽位记录就绪/释放事件；同步点规则以参考文档为准，官方指定算子用于核对完整调用方式。

| 事件组 | 共享缓冲/槽位表达式 | 方向 | set位置/pipe | wait位置/pipe | `sync_mode` | `event_id` | 复用条件 |
|---|---|---|---|---|---|---:|---|
| `{READY}` | `{group[slot_idx]}` | {生产者→消费者} | {最后一次写后；pipe按紧邻的写操作确定} | {第一次读前；pipe按紧邻的读操作确定} | {INTRA_BLOCK/...} | {ID或表达式} | {旧信号已被wait消费} |
| `{RELEASE}` | `{group[slot_idx]}` | {消费者→生产者} | {最后一次读后；pipe按紧邻的读操作确定} | {下一次覆盖前；pipe按紧邻的写操作确定} | {INTRA_BLOCK/...} | {ID或表达式} | {旧信号已被wait消费} |

### event_id 分配表

> event_id取值范围为`[0, 16)`。记录ID对应的数据方向和物理槽位，动态表达式的全部运行时取值也必须在该范围内。

| 流水组 | 方向 | Tile槽位 | event_id范围或表达式 | 复用条件 |
|---|---|---|---|---|
| {流水组1} | {Cube→Vector} | {`slot_idx`} | {`READY_BASE + slot_idx`} | {对应wait已消费} |
| {流水组1} | {Vector→Cube} | {`slot_idx`} | {`RELEASE_BASE + slot_idx`} | {对应wait已消费} |

---

## §7 尾块处理（R7 输出）

> 📌 **权威依据（必读，官方标准）**：`$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/tile_based_python_programming/tail_block_handling.md`（尾块处理以此为准）。

### 尾块处理方案

> 在 §7 填入以下尾块处理代码：

```python
# ceiling division 计算 tile 数（必须向上取整，用 N//TILE 直接整除会漏掉尾块）
m_tile_num = (M + TS - 1) // TS
n_tile_num = (N + TD - 1) // TD

# 循环内计算尾块有效尺寸并告知硬件（set_validshape 有状态，每轮都要重设）
valid_m = pl.min(M - m_off, TS)       # 满 tile = TS, 尾块 = 余数
valid_n = pl.min(N - n_off, TD)
pl.set_validshape(tile_a, [valid_m, valid_n])  # 运行时告知硬件
```

- **compact**: {`None`/`1`/`2`，按数据通路和API约束说明依据；普通Vec ND尾块通常不需要`compact`}
- **无效区域是否会被读取**: {否→不填充；是→按计算语义选择`pad`并执行`fillpad`；矩阵尾块是否填充由具体数据路径和API约束决定}

---

## §8 目标测试 case（R7.5 输出 → 交付 develop）

> 基于 R2 tile 尺寸与 R7 尾块方案确定的具体测试 shape，develop 直接实现这些 case，不再自行重算。
> 若 SPEC/用户已有目标 case，在其基础上追加下述基础 case；无用户指定 case 时直接确定，不询问用户。
> 至少 4 个（单轴算子按 R7.5 例外处理，并在此说明实际 case 数与原因）。

| case 名 | 具体 shape | 覆盖场景 | design 适配确认 |
|---------|-----------|---------|----------------|
| `test_{op}_aligned` | {如 `[TILE_A, TILE_B]`} | 全整除 | ✅ / {不适配→回溯的轮次} |
| `test_{op}_tail` | {如 `[TILE_A + 尾块余数, TILE_B]`} | 单轴尾块 | ✅ / ... |
| `test_{op}_tail2d` | {如 `[TILE_A + 尾块余数, TILE_B - 尾块余数]`} | 双轴尾块 | ✅ / ... |
| `test_{op}_multitile` | {如 `[2~3 × TILE_A + 尾块余数, TILE_B - 尾块余数]`} | 跨多 tile + 尾块（最小规模，勿放大） | ✅ / ... |
| {如用户指定的额外 case} | {shape} | {场景} | ✅ / ... |

---

## §9 综合评估（R8 输出）

### 准确性检查

| 检查项 | 结果 | 证据 |
|--------|------|------|
| API 调用链完整覆盖数学公式 | ✅ / ❌ | {映射验证} |
| §4 已参照官方样例确定循环与Section结构（含参考样例路径与结构说明） | ✅ / ❌ | {回 R4 补充} |
| 数据依赖正确（Module 顺序 + sync 点） | ✅ / ❌ | {依赖分析} |
| dtype 精度满足要求 | ✅ / ❌ | {FP32 matmul 累加 / ...} |
| 归约类 API 的 `[M,1]`/`[1,N]` 输出已设 `layout` | ✅ / ❌ | {回 R2 补 layout} |
| Acc tile 物理 M×N×dtype_bytes ≥ fractal（FP32 ≥ 1024 bytes；动态轴含极小维度时尤需检查） | ✅ / ❌ | {回 R2 pad tile shape} |

### 泛化性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 目标测试 case（≥4，单轴算子按例外）已按 tile 切分确定具体 shape，且逐个验证 design 可适配（见 §8「目标测试 case」表） | ✅ / ❌ | {回 R7.5 补充 / 回溯适配不了的轮次} |
| 是否正确处理了尾块（M/N 尾块） | ✅ / ❌ | {ceiling division + set_validshape 设计} |
| 归约轴可能超单Tile时，已给出覆盖完整归约轴的方案，并说明遍历次数、跨Tile状态及依据 | ✅ / ❌ | {回R0重设归约方案} |
| 循环边界正确 | ✅ / ❌ | {valid_m/valid_n 计算验证} |
| 超越函数在 dtype 范围内无溢出 | ✅ / ❌ | {引用 §1 数值安全边界} |
| 跨 tile 状态初始化/持久化正确 | ✅ / ❌ | {expands 恒等值 / muls 拷贝；回 R0 或 R1 修正} |
| cross_core 同步方案正确（存在跨执行域或跨Block/subblock依赖时：同步点位置与 event_id 隔离参照权威文档和官方指定算子） | ✅ / ❌ | {回 R6 修正} |

### 一致性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| R0-R7 输出无矛盾 | ✅ / ❌ | {交叉验证} |
| 所有决策有证据支撑 | ✅ / ❌ | {证据链检查} |
| R3 片上地址范围与逐空间容量检查 | ✅ / ❌ | 规则见[片上空间布局](../references/onchip_memory_layout.md)；证据见本文 §3 的地址表、最高地址上界和容量来源；❌ 时回 R3 重排地址 / R2 缩 tile |
| `tile_dims` 使用时已关注大 stride 对性能的影响 | ✅ / ❌ | {回 R2 调整布局} |
| 条件性检查（若 §6 填“不涉及cross_core”，确认不存在Cube↔Vector或跨Block/subblock的数据依赖） | ✅ / ❌ | {R0 重新评估} |

### 评估结论

- **整体**: {通过 / 需修改}
- **限制条件**: {如不支持尾块、需 M 整除完整 M-tile 尺寸等}
- **修改记录**: {修改内容、轮次、原因}

---

## §10 Tile 数据流全景图

将 R0-R7 各轮产出的片段串联为一张完整的 tile 级数据流图。标注每一块 tile 的流向：从哪里读取、经过哪些操作转换、写入哪里。

```
{按实际算子的 tile 流向绘制}

GM ─[load_tile]→ tile_a ─→ {操作} → {输出tile} → ...
                                     ↓
                              {操作}({输出tile}, {持久化tile})
                                     ↓
                              {持久化tile} → tile_a → {操作} → ... → tile_out
                                                                  ↓
GM ←[store_tile]─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

### 图例

| 标记 | 含义 |
|------|------|
| `[load_tile]` | MTE2 搬运：GM → UB |
| `[store_tile]` | MTE3 搬运：UB → GM |
| `→` | V 流水线操作，标注 API 名 |
| `---` | 数据跨 Module 持久化（tile 在不同 Module 间不被覆盖） |
| `├─` | 同一 Module 内分支（同一数据被多次使用） |

---

## §11 性能结论（R8 输出）

> 依据 `$CANNBOT_CONFIG_ROOT/references/kernel-core-principles.md`，每项判定标注所依据的编号，
> 便于复核推导。该文件是指导性参考，不是门禁；无对应 lint，也不需要额外产出物。

### 11.1 代价模型（`B1`）

| case | bytes_min | FLOPs | AI | t_mte | t_cube | t_vec | t_est | 绑定资源 |
|------|-----------|-------|----|-------|--------|-------|-------|---------|
| {目标 case} | | | | | | | | |

### 11.2 Roofline 判定（`B2`/`B3`）

{执行流水、拐点、CB/BB 结论；上限来源为实测证据路径或 `unknown`}

### 11.3 目标上限（`B6`）

{验收口径下的可达上限及推导；目标超出上限时写明差距与原因}

### 11.4 切分、容量与分核（`T1`/`P3`）

{tile 参数与缓冲深度、逐空间容量合计、block_dim 与 η_tail}

### 11.5 例外（`M4`）

{偏离的编号、理由、实测代价；无则写「无」}
