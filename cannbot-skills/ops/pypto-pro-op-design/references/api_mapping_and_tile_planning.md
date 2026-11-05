# API映射与Tile规划

本文包含以下内容：

- API映射：分解数学步骤、核对API约束、明确逻辑Tile并分析数值安全边界。
- Tile规划：确定Tile属性、容量和缓冲深度，并选择`make_tile`或`make_tile_group`。

从数学公式确定每步的执行域，再规划 Tile 的 shape、dtype、内存空间和 layout。Vector 计算遵守 [当前 authoring policy](../../pypto-pro-kernel/SKILL.md#constraints-and-workspace-policy)，不按模板或性能猜测放宽 tile compute 例外。API 映射负责语义与接口可行性，Tile 规划负责实际存储，最后落实地址布局。

API映射完成后规划Tile，Tile规格和缓冲数量确定后分配片上地址。地址容量不足时，依次处理地址空洞、重复分配和可以复用的地址区间，再缩小Tile或减少缓冲槽位。若容量仍然不足，且主要空间由接口要求的临时Tile或格式转换占用，则重新选择API调用链。

本文先介绍数学步骤到API调用链的映射方法，再说明如何根据接口约束规划Tile。接口签名、参数顺序、dtype、shape、layout、内存空间以及产品支持范围，以`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/`中的当前文档为准。设计前应同时阅读该目录下的`basic_data_structures/TileType.md`、`MemorySpace.md`和`TensorLayout.md`。

## API映射

### 1. 分解数学步骤并建立初步映射

先将数学公式拆成输入、输出和依赖关系明确的计算步骤，再为每一步选择API。拆分时需要标出归约轴、广播方向和中间计算精度。以稳定Softmax为例，其核心数据依赖如下：

```text
x ── reduce_max(N) ── max
│                       │
└──────── subtract ── exp ── reduce_sum(N) ── sum
                                  │
                  exp ───────── divide ── out
```

根据上述数据流，`max`和`sum`的shape均为`[M, 1]`，后续计算需要将它们沿N轴广播；`exp`在输入减去行最大值后执行。如果N轴跨越多个Tile，每个Tile的局部归约结果不能直接作为完整N轴的归约结果。此时可以分多次遍历，依次求全局最大值、指数和与最终输出；也可以使用经过数学推导和API验证的在线归约算法。两种实现都要写明跨Tile保存的状态及其更新公式。

数学分解还应明确中间精度。输入和输出dtype来自算子规格，中间dtype则需要结合误差要求和API能力确定。例如FP16输入并不意味着所有中间结果都应使用FP16：长轴归约和矩阵乘累加通常需要检查是否存在FP32累加路径，最终再按输出契约完成cast。

### 2. 核对API文档与约束

按当前 policy 与目标接口选择执行域：VF 通过 `@pl.vector_function` 中的 `vf.load*`、`vf.*` 和 `vf.store*` 组织寄存器计算；允许的排序与 SIMT 路径分别核对其契约。Cube 默认从 [GEMM owner](../../pypto-pro-kernel/references/patterns/gemm-desugar.md) 开始。下述手写搬运/矩阵乘链用于解释物理数据通路，仅在该 owner 允许的例外下作为实现方案，不替代默认入口。

Cube路径尤其受硬件数据通路约束。`pl.matmul(dst, lhs, rhs)`要求`lhs`位于Left（L0A）、`rhs`位于Right（L0B）、`dst`位于Acc（L0C），所以一次矩阵乘通常展开为：

```text
GM ── load ──> Mat(L1)
Mat(L1) ── move ──> Left(L0A) / Right(L0B)
Left × Right ── matmul或matmul_acc ──> Acc(L0C)
Acc(L0C) ── store或move ──> GM或Vec(UB)
```

VF路径也有清晰的层次。外层Kernel负责GM与UB之间的搬运，VF函数负责UB与寄存器之间的数据交换以及寄存器内计算：

```text
GM ── pl.load ──> Vec(UB) ── vf.load* ──> Register File
Register File ── vf.* ──> Register File ── vf.store* ──> Vec(UB)
Vec(UB) ── pl.store ──> GM
```

每个候选API都要核对参数顺序和写入语义、输入输出dtype、操作数shape关系、允许的MemorySpace和layout、尾块处理方式、输入输出能否重叠，以及目标产品是否支持当前参数组合。接口的功能和限制以对应API参考页为准，样例用于了解多个接口的组合方式。

数学公式通常省略数据搬运和数据表示转换，API调用链还要补充以下内容：

- GM、L1、L0、UB和寄存器之间的数据搬运。
- dtype转换、layout转换和归约结果的广播。
- 动态尾块使用的`set_validshape`、VF mask或`fillpad`。
- Cube与Vector之间传递中间结果所需的数据通路，例如GM workspace或`move`支持的片上通路；具体选择以接口和目标产品支持范围为准。

例如，`fillpad`当前只支持Vec Tile，不能用于Mat、Left或Right操作数。K维分块时，首个分块使用`matmul`，后续分块使用`matmul_acc`进行累加。完成映射后，从每个原始输入出发都应能沿调用链追踪到最终输出，所有搬运和转换都必须明确写出。

API映射阶段不需要分配cross-core事件号，但应记录API自身带来的顺序要求。例如VF函数内部的局部依赖可能需要`vf.mem_bar`，使用`phase`的矩阵乘涉及M与FIX之间的`unit_flag`配对，跨Cube和Vector Section的数据流则需要在后续阶段确定手动cross-core协议。

### 3. 明确API操作数与逻辑Tile

调用链确认后，为每个API标出输入和输出Tile。归约状态、API要求的临时Tile以及中间cast或layout转换也应列入调用链，不能只记录原始输入和最终输出。本节建立API操作数与逻辑Tile之间的对应关系；Tile的物理shape、缓冲数量和地址在后续规划中确定。

### 4. 分析数值安全边界

接口可以调用，并不代表数值方案正确。设计中出现指数、对数、平方根、除法、倒数、低精度cast或长轴归约时，应说明输入范围、目标dtype、可能的异常结果以及处理依据。

稳定Softmax使用`exp(x - max)`是典型例子：减去行最大值后，指数输入不会出现无界的正值，从而避免正向溢出。对数需要确认输入是否可能为零或负数，平方根要考虑舍入后出现微小负值，除法要证明分母不会为零，长轴求和则要评估低精度累积误差。防护措施必须来自算子定义或明确规格；不能为了通过测试，给普通除法擅自加入`epsilon`或改变边界语义。

如果使用VF接口实现非线性计算，应以对应VF API的精度和范围说明为准。例如可以检查是否存在直接表达稳定模式的`vf.exp_sub`，但不能仅凭名称推断它与目标公式等价。

## Tile规划

API调用链确定后，为每个输入、输出和临时量建立一个逻辑Tile。多个逻辑Tile复用同一片上地址时，相关Tile需要使用相同的`mutex_id`。

逻辑Tile的shape、dtype、内存空间和排布通过`TileType`写入Kernel，缓冲深度通过`make_tile`或`make_tile_group`确定。接口定义见`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/basic_data_structures/TileType.md`。每个Tile需要明确以下内容：

- `shape`由API的操作数关系、遍历方向、对齐和分形要求共同决定。当前`TileType`只支持二维编译期常量shape；动态输入通过固定物理Tile配合有效形状处理。
- `dtype`同时服从输入输出契约、API支持范围、中间累加精度和cast链，不能简单理解为“由用户输入决定”。
- `target_memory`由数据通路决定。Vector通常使用Vec；Cube路径使用Mat、Left、Right和Acc；量化路径还可能使用Scaling。
- `layout`首先满足具体API要求，其次才考虑`TileType`按内存空间和架构给出的默认值。归约结果的ND/DN语义、Cube的NZ/ZN分形以及转置搬入都需要单独核对。
- `valid_shape`描述Tile的有效区域；缺省时后端按`[-1, -1]`生成动态模板，并以物理shape作为初始值。动态尾块建议显式声明动态维度并逐块调用`set_validshape`。`pad`和`compact`只在对应计算语义或硬件路径需要时设置，具体方法见`$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/tile_based_python_programming/tail_block_handling.md`。
- 缓冲深度由数据流水和容量共同决定。双缓冲或N-buffer必须按所有槽位计算物理占用。

例如，一个处理FP16二维动态尾块的Vec Tile可写为：

```python
src_type = pl.TileType(
    shape=[64, 128],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
    valid_shape=[-1, -1],
)
```

`src_type`声明了Tile的物理shape、dtype、内存空间，并允许在运行时更新有效形状。确定缓冲深度和地址后，使用`make_tile`或`make_tile_group`创建Tile。

普通非分形Tile单槽位的字节数等于`shape`各维长度的乘积乘以每个元素的字节数，TileGroup的占用还要乘以槽位数。分形layout、接口规定的特殊存储规格以及显式`size`按对应API计算，不能套用普通公式。容量计算使用物理`shape`，不能使用尾块的`valid_shape`。`compact`只改变接口解释Tile排布的方式，不会自动增加缓冲区大小；RowPlusOne等需要额外行的路径，必须把额外行直接计入`shape`。

Tile shape的选择有稳定的优先顺序。首先满足API的维数、M/K/N关系、layout、分形和对齐等硬约束；随后确保固定物理shape能够容纳单个任务在目标动态范围内可能出现的最大有效窗口；再统计同一时刻并存的输入、输出、状态、临时量和多缓冲副本；最后评估并行度与搬运开销。Tile不需要覆盖完整动态Tensor。Tile过大可能使任务数少于可用核数，Tile过小则会增加循环和搬运次数。此处可以估算资源需求，精确容量仍要等地址规划展开全部缓冲槽位后确认。

容量超限按以下顺序处理：先整理地址排布并删除重复分配，再调整Tile尺寸、缓冲数量和中间量保留时间，最后重新选择占用更少临时资源的API路径。

### 创建Tile与缓冲轮转

片上Tile通过`make_tile`或`make_tile_group`创建。接口定义见`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/resource_management/`中的对应API文档。常规Kernel使用`make_tile_group`和`@pl.jit(auto_mutex=True)`，由框架根据mutex元数据管理核内跨Pipe依赖。

#### `make_tile_group`的使用范围

`make_tile_group`用于创建输入、输出、跨迭代状态以及需要轮转的片上Tile。单缓冲和多缓冲均使用该接口，缓冲槽位数量根据Tile的使用周期和流水重叠关系确定。

Cube Section中的Tile统一使用`make_tile_group`。Cube计算包含GM到L1、L1到L0、矩阵计算和结果搬出等多个流水阶段，Tile通常需要携带mutex信息，配合`auto_mutex=True`生成核内同步。只有一个缓冲槽位时，也使用单槽TileGroup，并通过`current()`或`group[0]`取得Tile。

TileGroup的槽位数由`depth`确定，`mutex_ids`描述每个Tile携带的mutex ID：

- 显式填写`depth=N`时，group包含N个Tile。`mutex_ids`非空时，其外层长度必须等于N。
- 省略`depth`时，必须提供非空`mutex_ids`，框架用`len(mutex_ids)`推导槽位数。
- `mutex_ids=None`或`mutex_ids=[]`时无法推导槽位数，因此必须填写`depth`。

`mutex_ids`的外层元素与Tile一一对应，内层元素是该Tile携带的全部mutex ID。扁平列表等价于每个Tile各带一个ID。

下表都省略了`depth`，因此Tile数量等于`mutex_ids`的外层元素个数：

| 配置 | 对应的Tile数 | 每个Tile的mutex ID |
|---|---:|---|
| `mutex_ids=[0]` | 1 | Tile 0：`[0]` |
| `mutex_ids=[0, 1]` | 2 | Tile 0：`[0]`；Tile 1：`[1]` |
| `mutex_ids=[[0, 1]]` | 1 | Tile 0：`[0, 1]` |
| `mutex_ids=[[0, 2], [1, 3]]` | 2 | Tile 0：`[0, 2]`；Tile 1：`[1, 3]` |

一个Tile可以携带多个ID。访问该Tile时，`auto_mutex`会带上这一组ID，用于同一物理Tile需要同时受多组mutex关系约束的场景；增加内层ID数量不会增加Tile槽位。组内每个Tile携带的ID数量必须相同，同一Tile内不能出现重复ID，不同Tile之间可以复用ID。ID范围、`depth`与地址列表长度等约束以`make_tile_group.md`为准。

不传`mutex_ids`或传空列表时，group仍可按`depth`创建和访问Tile，但这些Tile没有mutex元数据，即使Kernel启用了`auto_mutex=True`，框架也不会为它们插入mutex同步。相关核内跨Pipe依赖必须按实际数据路径手工插入`sync_src`/`sync_dst`等同步；cross-core同步仍按跨Section规则单独处理。

#### 选择TileGroup槽位

TileGroup提供轮转访问和显式下标访问：

| 写法 | 游标变化 | 用途 |
|---|---|---|
| `group.next()` | 前进一格 | 按固定顺序轮转缓冲 |
| `group.current()` | 不变 | 继续使用当前槽位 |
| `group.previous()` | 不变 | 取得当前槽位的前一个槽位 |
| `group[i]` | 不读取也不修改游标 | 直接选择第`i`个槽位 |

`group[i]`的`i`可以是整数常量，也可以是循环变量等整数标量表达式。常量越界会在解析时报错；运行时下标不会自动按槽位数取模，设计必须保证它始终位于`[0, depth)`。轮转访问需要回绕时框架会处理，显式下标则要写成`group[index % depth]`之类的有界表达式。

下标选中的Tile仍携带对应槽位的mutex元数据，`auto_mutex`会按实际槽位处理核内跨Pipe依赖。显式下标适合以下场景：同一轮要同时指明当前槽位和预取槽位；生产者与消费者需要按同一个`slot_idx`访问共享缓冲；或者控制流不能只靠一次`next()`表达。`group[i]`与`next()`可以混用，但下标访问不会推进游标，不能据此推断后续`next()`选中的槽位。

下面用两个双缓冲group说明单ID和多ID配置。`source_group`的每个Tile带一个ID，`output_group`的每个Tile带两个ID；同一个`slot`选中的源Tile与目标Tile共同参与`move`，`auto_mutex`据此处理这次操作涉及的全部mutex ID。每个Tile是否需要多个ID由实际互斥关系决定，不要为了双缓冲固定套用多ID配置。

```python
tile_type = pl.TileType(
    shape=[128, 128],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
)

# 两个Tile，每个Tile一个mutex ID；省略depth时由外层长度推导为2
source_group = pl.make_tile_group(
    type=tile_type,
    addrs=0x0000,
    mutex_ids=[0, 1],
)

# 两个Tile，每个Tile两个mutex ID；增加内层ID不会增加Tile数量
output_group = pl.make_tile_group(
    type=tile_type,
    addrs=0x8000,
    mutex_ids=[[0, 2], [1, 3]],
)

with pl.section_vector():
    slot = index % 2
    source_tile = source_group[slot]
    output_tile = output_group[slot]
    pl.load(source_tile, source, [row, 0])
    pl.move(output_tile, source_tile)
    pl.store(output, output_tile, [row, 0])

# 两个Tile，不配置mutex元数据；相关跨Pipe依赖需要手工同步
manual_sync_db = pl.make_tile_group(
    type=tile_type,
    addrs=0x10000,
    mutex_ids=None,
    depth=2,
)
```

#### `make_tile`的使用范围

分配方式由 [当前 allocation/lifetimes owner](../../pypto-pro-kernel/references/api/memory-allocation.md) 决定，不能把“单 Tile”一概等同于单次 scratch，或把 TileGroup 等同于所有依赖都已同步。逐项说明常驻/轮转/临时角色、实际地址区间、首次写入、最后读取、复用位置及所需的跨 Pipe/跨核或同 VF 排序；`auto_mutex` 只覆盖其确实管理的访问。

若某个受支持接口确实要求独立临时空间，按该接口的实际 shape/dtype/别名合同规划。例如合同要求 `[64,128]` FP32 的临时 Tile 时，它占 `64*128*4=32768` 字节；基址 `0x18000` 对应 `[0x18000,0x20000)`。只有生命周期确实不重叠，且旧消费者已完成时才能把该区间给另一角色。这是资源算例，不表示当前 VF select 需要这个临时 Tile，也不授权使用历史 tile-Vector select 代码。

具体条件选择及 mask 的当前接口见 [VF owner](../../pypto-pro-kernel/references/api/vf.md#25-combining-masks-with-existing-comparisons-and-selection)，不要把旧 Tile mask/temporary 契约套到 MaskReg 或仅替换 API 前缀。

设计文档需要记录API调用链和Tile清单。API调用链按执行顺序写明计算步骤、所用接口、输入输出Tile、关键约束和对应的API参考页。Tile清单记录变量名、shape、dtype、MemorySpace、layout、有效形状、缓冲数、单槽字节数和使用范围。多槽TileGroup还要记录使用`next()`轮转还是用`group[i]`显式选择；采用下标时，写明下标表达式及其有界依据。每个Tile操作数都应能在清单中找到对应条目，每个Tile也应注明用于哪个计算步骤。

提交设计前，沿API调用链检查以下内容：

- 搬运、dtype转换和layout转换是否完整。
- 归约输出shape和广播方向是否与公式一致。
- 每个动态轴是否同时覆盖满块和尾块。
- 非线性计算和低精度累加是否完成数值分析。
- TileGroup的全部槽位是否计入片上容量。
- TileGroup的`depth`与`mutex_ids`外层长度是否一致；多ID配置是否逐Tile列全且组内各Tile的ID数量一致。
- 不配置`mutex_ids`时，相关核内跨Pipe依赖是否明确给出手工同步。
- `group[i]`的所有运行时取值是否都在`[0, depth)`，与`next()`混用时是否单独核对了游标状态。

## 官方指定算子示例

下面这些示例来自PyPTO Pro工作流维护的官方指定算子清单，路径均相对于`$PYPTO_DEVKIT_DIR`。实际设计时先检查`PRO_MATERIAL_INDEX.md` §B，只引用其中当前可用、且与目标算子执行域和数据通路相近的文件。

| 示例 | 适合核对的内容 |
|---|---|
| `pro_ops/element_wise/test_add.py` | Vec Tile、逐元素API、二维Tile遍历和`make_tile_group`双缓冲 |
| `pro_ops/matmul/test_matmul_8k_example.py` | Mat、Left、Right、Acc之间的数据通路，以及Cube TileGroup的组织方式 |
| `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` | 动态M/N/K的Tile shape、K分块、`move` offset、尾块有效形状和`phase`配套要求 |
| `pro_ops/vf_api/test_softmax_tile_group_vf.py` | GM→UB→寄存器→UB→GM的VF调用链、mask和动态归约轴 |
| `pro_ops/vf_api/test_layernorm_tile_group_vf.py` | VF归约、FP32中间计算、gamma/beta Tile和动态有效形状 |
| `pro_ops/fa/test_fa_with_mask.py` | Cube/Vector复合数据流、N-buffer、mask Tile和多组中间Tile的生命周期 |

示例用于核对多个API和Tile资源如何组合，不能替代单个API参考页，也不能直接照搬其中的Tile尺寸、地址、layout、`mutex_id`或缓冲深度。示例与当前API文档不一致时，以当前API文档为准。
