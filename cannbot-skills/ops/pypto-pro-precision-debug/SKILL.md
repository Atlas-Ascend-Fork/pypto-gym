---
name: pypto-pro-precision-debug
description: 基于 PyPTO Pro Kernel 的代码审查、设备侧 dump 数据与编译产物定位算子精度问题。用于 @pl.jit 算子最终输出不一致、局部 Shape 或尾块错误、单核正确但多核错误、结果错位、误差随归约长度放大、结果偶发不稳定等场景；先核对失败用例和代码数据路径，再按需通过 pl.dump_data、pl.printf 对齐 GM Tensor、Vec Tile、Acc（L0C）Tile 与写回结果，结合 kernel.cpp 指令级审查和错误规律分析确定首错位置及根因。
---

# PyPTO Pro 精度定位

本技能先核对失败用例与 Kernel 数据路径，再按需使用设备侧 dump 和编译产物，定位 `@pl.jit` 算子精度问题的第一个出错环节并确定根因。

## 一、排查思路

### 遇到精度问题时的完整流程

1. **确认失败与比较基准**：固定原始失败用例、输入、TilingKey、实际运行配置及误差摘要；核对 Golden 的 dtype、运算顺序、layout 和比较阈值，确认输出已在设备同步后读取。记录无 dump 时的原始结果。详见“dump 定位流程 / Step 1”。
2. **沿实际数据路径审查代码**：从出错输出反推可能写入该区域的 Core、循环迭代和 Tile，优先检查相关路径上的输入映射、offset、shape、layout、有效区域、计算、累加、同步及写回；若有多个 Core 可能写同一区域，也要检查写入范围是否重叠。把代码与实际调用的 API 文档、已通过的同类用例对照。若发现可以从代码和 API 约束直接证明的错误，记录具体位置与错误机制，做最小修复，然后进入第 5 步验证。
3. **无法确认根因时定位首错**：按“dump 定位流程 / Step 2～4”缩小观察范围，在 GM 输入、load 后、计算后、store 前和 GM 输出中选择实际可观察的边界，与对应的中间 Golden 对齐。找到相邻的“前一处正确、后一处错误”，再缩小到具体搬运、计算或写回；不能直接 dump 的 L1/L0A/L0B 用特殊输入和 `kernel.cpp` 间接定位。若 dump 使错误消失，保留无 dump 基线，转向同步、Tile 复用及地址冲突排查，不把有 dump 的正确结果当作修复成功。
4. **按证据查速查表并验证假设**：确定首错环节，或已形成可验证的代码假设后，只阅读“已知问题速查表”中与该数据路径、API 和错误现象相关的条目及所引文档。表项是候选原因，不是逐项修改清单。对每个候选说明它如何导致当前错误，用数据、代码或生成指令确认；证据不足时继续定位。
5. **最小修复与回归**：只改已确认的根因；有关键 dump 时先确认首错位置已正确，再移除本次定位临时添加的调试代码、确保重新编译并复测原始失败用例，最后覆盖相关 Shape、尾块、dtype、TilingKey 和多核配置。若原用例仍失败或首错位置转移，回到第 2～4 步；报告已确认的范围及尚未确认的部分。详见“修复与验证”。

代码审查能直接证明问题时可以跳过 dump；不能确认时必须继续取证。无论走哪条路径，都要用原失败用例验证修复效果。

### 精度问题的两类根因

Kernel 侧的常见错误可从以下两类切入。先确认 Golden、比较阈值与 Kernel 的 dtype 和运算顺序一致，再用错误规律形成待验证的假设。

**逻辑错误**——确定性错误，同一输入每次运行错误位置和数值完全一致。大部分稳定复现的精度问题属于此类。

- 搬运参数错误：load/store/move 的 offset 计算错误，Tile/Tensor 的 shape、stride、order（转置）设置错误
- 数据类型处理错误：累加前过早 cast 回低精度（如 FP16）、Vector 侧手动累加用了 FP16 而非 FP32、rounding mode 不符
- layout 不匹配：先核对源 Tensor 的实际存放方式与 `load(order=...)` 是否匹配，再核对 L1→L0A/L0B 的 `move` 和 matmul。转置搬入 L1 支持 DN→NZ 和 DN→ZN，不能仅因 L1 Tile 使用 NZ 就判错；L0A 的目的 layout 为 NZ，L0B 的目的 layout 为 ZN。具体组合还须满足对应 API 的 dtype 和形状约束（见 `$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/cube_computation.md`、`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/memory_data_movement/load.md`、`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/memory_data_movement/move.md`）
- 尾块排布错误：若首错出现在尾块搬运、矩阵计算或 Acc 搬出后，核对该路径各 Tile 的 `valid_shape`、`layout` 和 `compact` 如何决定有效数据的物理排布；确认需要按有效窗口紧凑排布时再配置 `compact=1`（见 `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/basic_data_structures/TileType.md`）
- Tiling 或 Core 分片错误：`pl.range` 的起始/步长/边界计算错误、把 Host 请求的 `block_dim` 当成 `pl.get_block_num()` 返回的实际逻辑 Block 数、尾块未均衡
- API 参数错误：归约方向（`dim`）、mask、shape 不满足对应 API 约束、误把专用 API 的广播语义套用到普通 tile-tile 操作、`scale` 随路量化比例配置
- 初始化遗漏：K 维累加首块误用 `matmul_acc`（应在已有值上累加）而非 `matmul`（覆盖写）、Tile 未写入即被读取

**同步问题**——错误位置或数值随运行变化，或加入 dump 后现象消失。但部分同步问题在固定输入和调度下也可能稳定复现，不能仅凭"稳定"排除。

- 缺少流水同步：依赖操作之间未插入对应流水的同步（MTE2→V、MTE3→V、M→FIX、FIX→V 等）
- Tile 生命周期错误：Tile 在前一个消费者读取前被复用写入
- mutex 使用错误：TileGroup 缓冲区复用的依赖管理缺失、轮转错位或 ID 冲突
- 跨核 GM 写冲突：多个 Core 向重叠的 GM 地址写入

**判断信号**：怀疑同步问题时，用同一输入重复运行并比较错误位置和数值。有变化提示同步问题；完全一致只能提示逻辑错误的可能性较高，不能排除同步缺陷。加入 `pl.dump_data` 后精度恢复，提示执行时序或编译产物受到调试代码影响，应优先核查同步与缓冲区复用（详见后文“dump 对执行时序的影响”）。这些信号用于选取验证手段，不能单独作为修改代码的依据。

### dump 对执行时序的影响

`pl.dump_data` 会增加运行开销，生成的调试代码还会插入额外的流水同步屏障。加入 dump 后精度恢复，应先检查原 Kernel 的同步、依赖、mutex 和缓冲区复用，也要比较生成代码是否变化；这个现象本身不能证明根因。

遇到这种情况，保留"无 dump 错误、有 dump 正确"的两份结果，然后撤掉 dump，检查对应位置前后的同步和地址复用。也可以临时增加一个 GM 输出，把少量中间结果写回 Host 比较，降低设备打印对执行时序的影响。

## 二、调试工具

### dump 能力矩阵

| 数据位置 | `MemorySpace` | 是否支持 `pl.dump_data` | 用法 |
|---|---|---|---|
| GM Tensor | Tensor | 支持 | 直接传 Tensor，可打印全量或窗口 |
| UB | `Vec` | 支持 | 直接传 Tile，可打印全量或二维窗口 |
| L1 | `Mat` | 不支持 | `pl.dump_data` 无法直接打印 |
| L0A | `Left` | 不支持 | `pl.dump_data` 无法直接打印 |
| L0B | `Right` | 不支持 | `pl.dump_data` 无法直接打印 |
| L0C | `Acc` | 支持 | 必须提供 GM Tensor 作为 `workspace` |

显式使用 `section_vector()`/`section_cube()` 时，GM Tensor 可在两种 section 中打印；Vec Tile 须在 Vector section 中打印，Acc Tile 须在 Cube section 中打印。`dump_data` 和 `printf` 的当前文档仅列 Ascend 950PR/950DT 为支持设备，使用前核对目标设备。

`pl.dump_data` 可直接打印 GM Tensor 和 Vec Tile；Acc 通过 GM workspace 中转；Mat、Left、Right 无法直接打印。具体约束见 `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/Utils-API/debugging/dump_data.md`。后文的“间接定位”是无法直接 dump 时的排查方法，不是 L1/L0A/L0B 的 dump 能力。

`pl.dump_data` 的结果直接输出到终端。

### `pl.dump_data` 用法

#### 打印 GM Tensor

```python
pl.dump_data(gm_tensor, loc=True)
pl.dump_data(gm_tensor, offsets=[m_off, n_off], shapes=[8, 8], loc=True)
```

窗口打印（提供 `offsets`/`shapes`）的约束：

- `offsets` 和 `shapes` 须同时提供，长度等于 Tensor rank；
- `shapes` 中的静态值须大于 0；
- 若 Tensor 通过 `make_tensor` 构造了带 stride 的视图，最内维 stride 须为静态 `1`；

GM Tensor 不使用 `workspace`。

#### 打印 Vec Tile

```python
pl.dump_data(vec_tile, loc=True)
pl.dump_data(vec_tile, offsets=[0, 0], shapes=[8, 8], loc=True)
```

`offsets` 和 `shapes` 可以包含循环变量、`pl.get_block_idx()`、运行时标量或动态 Shape 表达式。

#### 打印 Acc（L0C）Tile

```python
pl.dump_data(acc_tile, workspace=workspace, loc=True)
pl.dump_data(
    acc_tile,
    offsets=[16, 16],
    shapes=[8, 8],
    workspace=workspace,
    loc=True,
)
```

`workspace` 约束：

- 类型为 GM `pl.Tensor`；
- dtype 与 Acc Tile 一致；
- 存储空间须容纳完整物理 Tile。当前 CCE lowering 即使只打印窗口，也会先把完整物理 Acc Tile 搬到 workspace；不要只按窗口 `shapes` 分配；
- 须与 Kernel 中其他数据缓冲区不重叠（框架不做编译期检查，需开发者自行确保）。

Kernel 参数是 `pl.Ptr` 时，先用 `pl.make_tensor` 构造 GM Tensor 再作为 workspace 传入。

## 三、dump 定位流程

### Step 1：固定复现用例

精度定位的前提是一个可复现的失败用例。记录以下信息，后续每次 dump 都基于同一用例对照：

- **复现命令**：完整的 Kernel 调用脚本或测试命令
- **输入信息**：Shape、dtype、layout；如果输入是随机生成的，记录随机种子
- **运行配置**：TilingKey、`block_dim`；如果 Kernel 按 TilingKey 走不同代码分支，记录命中的是哪个 Key
- **Golden 来源**：CPU 参考实现的代码位置，或预先生成的期望数据文件路径
- **误差摘要**：输出与 Golden 对比后的 `rtol`/`atol` 阈值、NaN/Inf 个数、最大绝对误差、最大相对误差、首个错误元素的索引
- **复现稳定性**：需要判断时序影响时，用同一输入重复运行并记录错误位置和数值是否一致

**先确认 Golden 本身正确**。先按算子预期语义确定输出 Golden 和容差；再核对 CPU 参考实现与 Kernel 的 dtype、运算顺序和 layout。FP16/BF16 等低精度场景还要检查在哪里转为 FP32、以何种精度累加以及何时转换输出。必要时另算一份与 Kernel 当前精度路径对齐的中间参考，用来判断差异来自实现错误还是合理舍入；不要把 Kernel 当前的错误精度路径直接当作验收标准。

**区分逻辑错误与同步问题**：错误位置和数值随运行变化，优先检查同步、缓冲区复用和写冲突；完全一致仍不能排除固定调度下的同步缺陷。

Kernel 异步启动后，在读取输出前完成同步：

```python
kernel[None, block_dim](*args)
torch.npu.synchronize()
```

定位期间保持输入和比较阈值不变。每次新增一个 dump 后，结果才能与原始失败用例直接对照。

### Step 2：确定观察范围

大 case 出错时，可以先尝试缩小 shape 到单个基本块（如 `[16, 16]`），并显式以 `block_dim=1` 启动，看能否复现。如果单基本块、单 Block 能复现，直接按 Step 3 逐阶段 dump 定位；如果复现不了，需要回到原始出错的 case，推导首个错误位置可能由哪些 Core、循环迭代和 Tile 写入，然后针对这些位置做 dump。

推导步骤：

1. **定位首个错误索引**：Golden 对比结果中通常给出首个错误元素的线性索引。例如输出 shape 为 `[2048, 2048]`，首个错误索引为 `131200`，换算到二维坐标是 `[64, 128]`（`131200 // 2048 = 64`，`131200 % 2048 = 128`）。
2. **映射到 Tile**：根据 Kernel 的 Tile shape 反推该坐标属于哪个 Tile。例如 Tile shape 为 `[16, 16]`，则 `[64, 128]` 落在行方向第 4 个 Tile（`64 // 16 = 4`）、列方向第 8 个 Tile（`128 // 16 = 8`）。
3. **映射到循环迭代**：根据 Kernel 的 `pl.range` 循环结构，反推该 Tile 对应的循环变量值。例如 `for m in pl.range(0, 2048, 16)`，行方向 Tile 4 对应 `m = 64`；`for n in pl.range(0, 2048, 16)`，列方向 Tile 8 对应 `n = 128`。
4. **映射到 Core**：根据多核分片方式反推。如果用 `pl.range(core_id, total_tiles, num_cores)` 跨步分配，通过 Tile 的全局序号和 `num_cores` 计算原本负责该 Tile 的 Core；若怀疑跨核覆盖，还要检查其他 Core 的写回范围。
5. **只 dump 相关核的目标迭代**：在 `if` 条件中过滤 Core ID 和循环变量，只打印出错 Tile 的数据窗口（如 `8×8` 或 `16×16`）。有多个可能写入者时分别检查。

用 `if` 条件同时过滤 Core ID 和循环变量，只打印目标核的目标迭代：

```python
core_id = pl.get_block_idx()
if core_id == target_core and m == target_m and n == target_n:
    pl.printf("core=%d, m=%d, n=%d\n", core_id, m, n, loc=True)
    pl.dump_data(vec_tile, offsets=[0, 0], shapes=[8, 8], loc=True)
```

`pl.printf` 打印 Core、循环变量、GM offset 等上下文信息，方便在日志中定位；`pl.dump_data` 打印实际数据用于和 Golden 对照。

### Step 3：关键阶段 dump

按 Kernel 的实际执行顺序列出中间结果，选择 2～4 个关键位置加入 dump。多核或循环场景按 Step 2 过滤相关 Core 和迭代；单核或无循环场景只需保留适用的过滤条件。

Vector Kernel 通常按以下顺序检查：

```text
GM 输入
→ load 后的 Vec Tile
→ 关键计算后的 Vec Tile
→ store 前的 Vec Tile（若与计算后不是同一 Tile）
→ GM 输出
```

Cube Kernel 可直接观察的位置较少：

```text
GM 输入 A/B
→ [L1、L0A、L0B 无法直接 dump]
→ matmul 后的 Acc Tile
→ GM 输出
```

CV Kernel（Cube + Vector 混合 Kernel）可按以下边界检查：

```text
Cube 的 GM 输入
→ Acc Tile
→ Cube 写回的中转 GM 区域
→ Vector load 后的 Vec Tile
→ Vector 计算结果
→ 最终 GM 输出
```

例如 Vector Kernel 先比较 load 后、核心计算后和 store 后；如果 load 后正确而核心计算后错误，下一轮只细查这段计算。

### Step 4：Golden 对齐并逐级缩小

中间 Golden 必须与 Kernel 该位置的语义一致，包括：

- dtype 转换发生的位置；
- 矩阵转置和 layout；
- padding 值和有效 Shape；
- 归约的累加 dtype、矩阵乘的累加 dtype；
- 当前 Core 和当前 Tile 对应的数据范围。

记录每个 dump 的结果：

| 位置 | 实际数据 | Golden | 结论 |
|---|---|---|---|
| load 后 Tile | dump 打印的数值摘要 | 对应输入窗口 | 正确/错误 |
| 计算后 Tile | dump 打印的数值摘要 | 中间 Golden | 正确/错误 |
| store 后 GM | dump 打印的数值摘要 | 输出 Golden | 正确/错误 |

找到相邻的"一处正确、一处错误"后，在两者之间继续增加一个 dump。重复这个过程，直到范围缩小到一次 load、move、计算、cast 或 store。

### 辅助 A：分析 dump 数据的错误规律

拿到 dump 数据后，观察错误的分布规律。不同类型的错误通常对应不同的根因：

| 错误现象 | 典型根因 | 检查方向 |
|---|---|---|
| 全零 | Tile 从未被写入、store 到错误地址、累加器未初始化即被读取、计算被编译器删掉 | store offset、Tile 地址分配、kernel.cpp 中对应指令是否被删掉 |
| 全为同一值 | `expands` 传错值、专用广播 API 的模式或维度错误、累加器未清零 | 对应 API 的广播参数、累加器清零逻辑 |
| 数据看起来是转置的 | load 的 order 参数错误、ND/NZ layout 混淆、Mat 与 Left/Right 的 layout 配置不一致 | load order、TileType layout、move 搬运方向 |
| 整体偏移固定值 | offset 计算差一个常数、Tile 基址错误、stride 差一倍 | GM offset 公式、Tile 基址、stride 计算 |
| 每隔 N 个元素出错 | stride 或 block size 配置错误、mask 覆盖范围不符 | Tensor stride、Tile shape、mask 设置 |
| 误差随 K 或归约长度增大 | Vector 侧手动累加用了 FP16 而非 FP32、累加前过早 cast 回低精度、累加顺序导致误差累积 | 改用 FP32 累加、推迟 cast、检查 matmul_acc 累加路径 |
| 仅特定行/列出错 | `pl.range` 跨步分片边界、尾块、valid_shape 未设置导致 padding 读入越界 | `pl.range` 起始/步长、尾块处理、valid_shape |
| 正确与错误块交替出现 | 跨核 GM 写冲突、Tile group 轮转错位、mutex 缺失 | Core 间地址重叠、Tile group depth 和轮转、mutex ID |
| 前几轮正确，后续错误 | 地址递增错误、累加器未在迭代间清零、Tile 被提前复用 | 循环内地址递增、累加器清零、Tile 生命周期 |
| 数值像随机垃圾 | 缓冲区重叠、读取了未初始化内存、地址完全错误 | Tile 地址分配、workspace 是否与数据缓冲区重叠 |
| 符号翻转或数值翻倍 | 计算符号错误（sub 写成 add）、重复计算、`scale` 随路量化比例配置错误 | API 选择（add/sub）、循环是否多跑一轮、`scale` 配置 |
| 加入 dump 后恢复正常 | dump 插入的 pipe_barrier 带来了额外同步，使原本缺失的同步暂时生效 | 撤掉 dump，检查该位置前后的流水依赖和 Tile 复用 |

观察到规律后，用 `pl.printf` 打印相关变量（offset、stride、循环变量、`pl.get_block_idx()`）验证假设。

### 辅助 B：常见现象速查

| 现象 | 优先检查 |
|---|---|
| GM 输入已经错误 | Host 侧调用 Kernel 时参数顺序、dtype、shape、指针是否正确，Host 预处理逻辑是否有误 |
| GM 正确，Vec load 后错误 | load offset、Tensor stride、layout、尾块 padding、有效 Shape |
| Vec 输入正确，计算后错误 | API 参数、dtype、mask、shape 不满足对应 API 约束、误用广播语义、归约方向、类型转换 |
| 前几轮正确，后续循环错误 | 地址递增、Tile 清零、累加器初始化、Tile 复用、同步和 mutex |
| store 前正确，GM 输出错误 | store offset、有效 Shape、写回 layout、重复写和越界 |
| 单核正确，多核错误 | `block_dim`、`pl.range` 跨步分片参数、把 Host 请求核数当作实际核数、混合 Kernel 工作单元数、跨核覆盖 |
| 只有尾块错误 | 尾块长度、padding、有效 Shape、Tile layout、compact 和 store 范围 |
| 错误呈转置或按固定块大小分布 | ND/NZ、Left/Right layout、stride 和转置配置 |
| 误差随 K 或归约长度增大 | Vector 侧手动累加用了 FP16、累加前过早 cast、运算顺序和误差阈值 |
| 每次运行错误位置不同 | 同步、mutex、未初始化数据、Tile 生命周期和写冲突 |
| 全零或全同一值 | Tile 未写入、store 地址错误、expands 传错值、累加器未初始化 |
| 数值像随机垃圾 | 缓冲区重叠、workspace 与数据区冲突、地址完全错误 |
| 结果符号翻转或翻倍 | API 选错（add/sub）、循环多跑一轮、scale 配置错误 |
| 仅特定 TilingKey 出错 | 该 Key 选择的代码路径、特化参数、分支条件 |
| 量化/反量化结果异常 | `scale` 随路量化比例 dtype/shape、dequant 路径、fixpipe 量化配置 |
| ND↔NZ 转换后出错 | 转换前后的 shape/stride、layout 标记、搬运 order |
| AtomicAdd 累加结果不对 | 原子写冲突、Core 间 GM 覆盖区域、目标 GM 区域未初始化为零 |
| Golden 本身有误 | CPU 参考实现的 dtype、运算顺序、layout 与 Kernel 不一致 |

### 辅助 C：L1、L0A、L0B 不可直接 dump 时的间接定位

Cube 数据路径为 `GM → L1(Mat) → L0A/L0B(Left/Right) → L0C(Acc)`。`pl.dump_data` 无法直接查看中间的 L1、L0A 和 L0B，因此按下面的方法缩小范围。

**1. 先确认 GM 中的输入数据是否正确**

打印 A、B 的目标窗口，同时打印对应的 GM offset、shape、stride 和转置配置。先排除 Host 传入的数据本身、地址计算和 GM 视图的问题。

**2. 打印 matmul 后的 Acc**

使用带 workspace 的 `pl.dump_data(acc_tile, ...)`。如果 GM A/B 正确而 Acc 错误，优先检查以下环节，同时保留同步和片上地址复用等候选原因：

- GM 到 L1 的 `pl.load`；
- L1 到 L0A/L0B 的 `pl.move`；
- Left/Right 的 layout 或转置配置；
- `pl.matmul` 的 M/N/K、dtype、累加方式（`matmul` 覆盖写 vs `matmul_acc` 累加）或有效 Shape。

**3. 通过构造特殊输入，缩小 A/B 数据路径的问题范围**

Acc = A × B。L1→L0A 和 L1→L0B 这两段搬运都无法直接 dump，但可以通过构造特殊输入来间接判断哪边出错：保持 tile shape、dtype、layout 和搬运参数不变，只改变输入数值。

- **把 B 设为单位矩阵**：在维度与有效区域允许时，预期 Acc = A × I = A。如果 Acc 与预期不一致，重点检查 A 的 GM→L1→L0A 路径，同时核对 B 的单位矩阵是否正确搬入以及 matmul 配置；不能仅凭这次对比断定 L1→L0A 出错。
- **把 A 设为单位矩阵**：在维度与有效区域允许时，预期 Acc = I × B = B。如果 Acc 与预期不一致，重点检查 B 的 GM→L1→L0B 路径，同时核对 A 的单位矩阵是否正确搬入以及 matmul 配置；不能仅凭这次对比断定 L1→L0B 出错。
- **使用递增值、行号或列号模式**作为输入，便于识别转置错误、分块错位和 stride 错误。

如果两组实验都正常、但恢复原始 A/B 后异常，只能说明两种特殊输入未触发故障；继续检查数据相关的搬运、尾块、matmul 参数、累加和同步，不排除 A/B 路径。

**4. 检查生成代码和同类工作样例**

在对应 JIT 目录的 `kernel.cpp` 中核对 TLOAD、TMOV 和 TMATMUL 的 shape、offset、layout、有效 Shape 及源/目的 Tile。再与目标版本中已上板通过、数据路径相同的 Matmul 用例逐项比较。

这个方法不能直接显示 L1/L0A/L0B 的内容，但可以把问题缩小到左搬运、右搬运或 matmul 本身。输出中应标注为"间接定位"。

### 辅助 D：检查编译生成的 kernel.cpp

dump 只能看数据，看不到指令。当 dump 缩小了范围但仍无法确认根因时，读编译产物 `kernel.cpp` 可以看到指令级细节。

JIT 编译后，未设置 `ASCEND_WORK_PATH` 时在当前运行目录的 `build/` 下搜索 `**/tk_*/kernel.cpp`；设置后则从 `${ASCEND_WORK_PATH}/PYPTO_PRO/build/` 搜索。目录名可能包含 kernel 名、arch、静态签名、Device、Rank、datatype hash 和 TilingKey 等信息，不要依赖固定目录字符串；未使用 TilingKey 时为 `tk_none/`。同一产物目录还包含 Host Launcher 源码 `call_kernel.cpp` 和带 hash 的 `call_kernel_{hash}.so`。当怀疑 `block_dim`、TilingKey 分发或参数打包时，检查 `call_kernel.cpp`。详见 `$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/compilation_and_execution/JIT_compilation.md`。

重点检查项：

**指令序列**：确认 TLOAD、TMOV、TMATMUL、VADD、VCAST 等指令的排列顺序与预期数据流一致。编译器可能合并或删除操作——例如将 cast 合并到 store 的 fixpipe 路径，或删掉被判定为无效的计算。

**搬运参数**：逐条核对 TLOAD/TMOV 的 offset、shape、stride、layout（ND/NZ/ZN）、源和目的 Tile。前端的 `offsets=[m, n]` 编译后会变成具体的字节或元素偏移，确认转换符合预期。

**同步指令**：不要只按固定的底层函数名判断同步是否存在。`make_tile` 的跨 Pipe 依赖来自开发者显式调用的 `pl.system.sync_src`/`sync_dst`；`make_tile_group` 仅在配置 mutex 元数据并开启 `auto_mutex` 后，才由框架为可识别的数据依赖插入同步；使用 `phase` 时还须核对 `AccPhase`/`STPhase` 配对。结合生成代码检查以下关键依赖是否成立：

- MTE2（GM→UB）完成后才能在 V 流水读取 → 需要 MTE2→V 同步
- MTE3（UB→GM）完成后才能在 V 流水复用该 UB → 需要 MTE3→V 同步
- M（matmul）完成后 FIX（fixpipe）才能读 L0C → 需要 M→FIX 同步
- FIX（L0C→UB/GM）完成后 V 才能读取搬出的数据 → 需要 FIX→V 同步；FIX 完成后 M 才能复用同一块 L0C 写入新结果 → 需要 FIX→M 同步
- Cube 流水（MTE2→MTE1→M→FIX）与 Vector 流水（MTE2→V→MTE3）之间的跨流水依赖

**Tile 地址分配**：确认各 Tile 的 UB/L1/L0C 地址不发生非预期重叠；地址重叠负例见 [L1 地址重叠负例](reference/test_l1_buffer_address_overlap.py)。当前 `pl.make_tile(tile_type, addr=...)` 只有 `tile_type` 和关键字参数 `addr`，不接收 `size`；地址占用由 `TileType` 的物理 shape 和 dtype 推导，`valid_shape` 不缩小分配范围。`make_tile_group` 可接收单个基地址并自动排列各槽位，也可接收地址列表显式指定每块 Tile；两种方式都要按实际物理占用和对齐要求检查。

**Tile group 轮转**：根据 `make_tile_group` 实际创建的 Tile 数量，检查缓冲区轮转、Tile 复用和 mutex 同步是否符合数据依赖。

**编译器优化导致的行为变化**：对比 PyPTO Pro 源码和生成的 CCE 代码，确认编译器没有做开发者预期之外的优化。常见问题：循环展开导致寄存器拷贝错误、算子融合改变了累加顺序、编译器认为无用而删掉了实际必要的操作。

典型配合方式：dump 定位到某步出错 → 打开 `kernel.cpp` 找到该步对应的指令 → 核对参数和同步 → 用 `pl.printf` 打印可疑变量验证假设。

## 四、已知问题速查表

定位到第一个出错环节，或已形成可验证的代码假设后，只查与该数据路径和 API 相关的条目。确认代码或数据证据后再修改。

提供了部分常见精度错误负例。

| 检查项 | 问题现象 | 规避方法 | 依据 |
|---|---|---|---|
| set_validshape 顺序 | 尾块数据错误、越界读入 | 尾块搬入前先调用 `pl.set_validshape`，再执行 `pl.load`；load 后才设置不会改变这次搬运，不能补救已经搬错或越界读取的数据 | `$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/vector_computation/tile_computation.md`、`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/memory_data_movement/load.md` |
| 输出 Tile 漏设 valid_shape | 越界写或写回无效数据 | 输入 Tile、计算结果 Tile 和写回 Tile 对同一逻辑区域使用一致的 valid_shape | `$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/vector_computation/tile_computation.md`；负例：[输出 Tile 未限定有效区域](reference/test_output_tile_missing_valid_shape.py) |
| pad 未实际填充 | 归约、softmax 或矩阵计算把尾块无效区的残留值带入结果 | `TileType(pad=...)` 只声明填充值，不会写入无效区域。后续操作会读取物理 Tile 的无效区时，使用带动态 `valid_shape` 的源 Tile 搬入有效数据，再通过 `pl.fillpad` 写入配置了 `pad` 的目的 Tile；求和填 zero，最大值或 softmax 填 min，最小值填 max | `$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/vector_computation/tile_computation.md`、`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/memory_data_movement/load.md`；负例：[声明 pad 但未 fillpad](reference/test_pad_declared_without_fillpad.py) |
| Tile 未初始化即读取 | 随机垃圾值混入计算 | `make_tile`/`make_tile_group` 创建的是裸缓冲区，不自动初始化；读取前确认相应物理范围已由 `pl.load`、`pl.move`、`pl.fillpad` 或计算等操作写入 | `$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/tile_creation_and_operations.md`、`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/resource_management/make_tile.md` |
| reinterpret 状态遗漏 | 重声明后的尾块越界，或新旧 TileGroup 交替出现错块 | `pl.reinterpret` 只创建共享原缓冲区的新视图，不搬运或转换数据；新视图不继承原 Tile 的 `valid_shape`，须重新设置。对 TileGroup 重声明后，新旧对象共享轮转状态，任一方调用 `next()` 都会推进同一组缓冲区 | `$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/tile_creation_and_operations.md`、`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/resource_management/reinterpret.md`；负例：[视图未重设有效区域](reference/test_reinterpret_view_missing_valid_shape.py) |
| make_tile 缺少同步 | 跨流水数据竞争、AICore 异常或 Kernel 卡住 | `make_tile` 不携带供 `auto_mutex` 使用的 TileGroup mutex 元数据，跨 Pipe 依赖须按实际数据流显式配对 `pl.system.sync_src`/`pl.system.sync_dst`；`make_tile_group` 只有在配置非空 `mutex_ids` 且开启 `@pl.jit(auto_mutex=True)` 时，框架才会为可识别的数据依赖插入同步 | `$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/tile_creation_and_operations.md`、`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/synchronization/index.md` |
| 把 block_dim 当作实际核数 | 限核后部分任务遗漏，或混合 Kernel 的 Vector 任务划分错误 | `block_dim` 是 Host 申请上界，实际逻辑 Block 数可能更小。Kernel 按逻辑 Block 跨步分片时，使用 `pl.get_block_num()` 作为步长；MIX Kernel 中协作处理同一任务的 Cube/Vector 使用共同的 `core_id = pl.get_block_idx() // pl.get_subblock_num()`，Vector 再用 `pl.get_subblock_idx()` 分担核内数据。若 Vector 核各自处理独立任务，才按实际 Vector 工作单元数设计分片 | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/system_variables/get_block_num.md`、`$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/tiling/multi_core_tiling.md` |
| FP16 归约精度 | 大规模数据归约误差偏大 | FP16 Tile 归约受有限精度累加、硬件累加顺序和输出舍入影响；VF `reduce_sum` 的 FP16 累加精度也是 FP16。对精度敏感的场景显式转为 FP32 路径，并按设备累加顺序判断 Golden | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/tile_computation/math_functions/sum.md`、`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/reg_computation/reduction/reduce_sum.md`；负例：[FP16 归约按 FP32 精度比较](reference/test_fp16_reduction_expected_fp32_precision.py) |
| VF 融合乘加误用 | 分步乘加舍入误差偏大，或融合后结果缺少原加数 | `vf.mul_add_dst` 计算 `dst = src0 * src1 + dst`，单指令完成乘加可避免中间乘积舍入；调用前必须把 dst 初始化为要累加的值，并确认 dtype 和 mask 满足接口约束 | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/reg_computation/composite_computation/mul_add_dst.md`；负例：[目标寄存器加数错误](reference/test_vf_mul_add_dst_wrong_addend.py) |
| load offsets/order 错误 | 搬入错误批次、错误维度或转置后的数据 | `offsets` 长度必须等于源 Tensor rank，并给出各维绝对元素偏移；`order` 是两个互不重复的编译期维度索引，决定 Tile 两维对应的 Tensor 维度及是否转置。普通 Tensor 省略 `order` 时默认选择最后两维正序；若源数据按转置方式存放而计算需要原逻辑方向，应设置相应的转置 `order`。NZ Tensor 只能从最后两维正序搬运，且 shape、valid_shape 和 N 方向 offset 满足分形对齐 | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/memory_data_movement/load.md`；负例：[转置搬入漏设 order](reference/test_transposed_load_missing_order.py) |
| 连续 load 复用同一地址 | 后一次搬入覆盖前一次搬入，结果偶发或整块错误 | 开启 `auto_mutex` 时，如果连续两个 `pl.load` 写同一 UB/L1 地址，且中间没有操作读取前一次数据，在两次 load 之间调用 `pl.system.bar_mte2()`；如果前一次数据仍需使用，先完成读取或复制再复用地址 | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/memory_data_movement/load.md`、`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/synchronization/bar_mte2.md`；负例：[前次数据未读取就复用地址](reference/test_load_reuses_address_before_read.py) |
| sync_all 参与核或次数不一致 | Kernel 超时或死锁 | 当前 `sync_all` 仅支持 HARD 模式，不使用 workspace。纯 Vector 使用 `AIV_ONLY`，纯 Cube 使用 `AIC_ONLY`；MIX 模式下 AIC 与 AIV 必须一一对应调用。所有参与核须以相同顺序执行相同次数的 `sync_all`，不能把调用放在只有部分核会进入的分支中 | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/synchronization/sync_all.md` |
| `vf.gather`/`vf.scatter` 索引错误 | gather 越界读取，或 scatter 重复索引导致结果不确定 | 此条仅针对寄存器计算 API。`vf.gather` 的 Tile→reg 索引必须落在 Tile 有效地址范围内；`vf.scatter` 的索引值必须唯一。Tile API `pl.gather`/`pl.scatter` 是另一组接口，应查各自约束，不能套用本条 | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/reg_computation/data_movement/gather.md`、`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/reg_computation/data_movement/scatter.md` |
| phase 配对与收尾 | Fixpipe 读到未完成的 L0C 数据，或 Kernel 卡死 | matmul 系列一旦配置 `pl.AccPhase`，对应的 `store`、`store_tile` 或 Acc→Vec `move` 也要配置 `pl.STPhase`，不能混用 phase 与自动软件同步。对同一 L0C，最后一次 matmul 写和最后一次 Fixpipe 读都使用 `Final` | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/cube_computation/phase.md` |
| matmul_acc K 维累加 | 首个 K 块含随机值、后续块未正确累加或结果布局错误 | 首块用 `pl.matmul` 初始化 L0C，后续块用四参数形式 `pl.matmul_acc(dst, acc, lhs, rhs)`；dst 与 acc 的 shape、dtype 必须一致。FP32/INT32 Acc 的 fractal 默认是 1024，显式设置时应与该格式一致。只有启用 phase 时，才按非末块 `AccPhase.Partial`、末块 `AccPhase.Final` 并与 `STPhase` 配对。`set_mm_layout_transform` 只用于 matmul 与 Fixpipe 并行访问同一块 L0C、需要切换 Fixpipe 读出方向的场景；结果搬出后关闭，串行执行或使用不同 L0C 时无需调用 | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/cube_computation/matmul_acc.md`、`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/cube_computation/set_mm_layout_transform.md`；负例：[首块累加旧结果](reference/test_matmul_acc_reuses_stale_accumulator.py) |
| Vec 尾块 layout/compact | ND 尾块正确，而 NZ/ZN 尾块从非满分形开始出现数据错位 | 先核对 Vec Tile 的 `layout`、`valid_shape` 及后续搬运对物理跨度的要求。UB ND 逐元素尾块通常用 `valid_shape` 控制范围，不需要 `compact`；NZ/ZN 尾块只有在数据路径要求按有效窗口连续排布时才配 `compact=1`。`compact=2` 用于明确采用 RowPlusOne 排布的 UB Tile，不能当作普通尾块修复 | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/basic_data_structures/TileType.md`、`$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/vector_computation/tile_computation.md` |
| Cube 尾块 compact | GM 输入正确，但尾块在 L1→L0A/L0B、matmul 或 Acc 搬出后错位；使用 phase 时尾块可能卡死 | 按参与尾块的各 Tile 设置实际 `valid_shape`，核对 `move`、matmul、Acc 搬出对有效跨度的解释；需要按有效窗口紧凑搬运的 L0A/L0B、Acc 路径使用 `compact=1`，但全量搬运、拼接等路径不能一律套用。L1/Mat 上配置 `compact` 与否不影响结果；使用 phase 且 L0C 存在尾块时，Acc 必须配置 `compact=1` | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/basic_data_structures/TileType.md`、`$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/vector_computation/tile_computation.md`；负例：[尾块缺少 compact](reference/test_cube_tail_missing_compact.py) |

## 五、排障

### dump 没有输出

依次检查：

1. Kernel 调用后是否执行了同步；
2. Core 和循环过滤条件是否命中；
3. `pl.printf` 标记是否出现；
4. 设备日志的输出位置和级别；
5. 生成的 `kernel.cpp` 中是否有对应打印代码。

### 编译失败

检查：

- 是否尝试直接 dump Mat、Left 或 Right；
- `offsets` 和 `shapes` 是否成对且 rank 一致；
- Tensor 窗口的最内维 stride 是否为静态 `1`；
- Tile 窗口是否为二维；
- Acc workspace 的类型、dtype 和容量是否正确。

### `pl.pto_assert` 的行为限制

`pl.pto_assert` 在条件不满足时仅通过设备侧打印记录信息，**不中止 Kernel、不在 Host 侧抛异常**。即使断言放在访问之前，条件不满足时后续代码仍会执行，不能用它阻止越界访问；如果断言放在访问之后，也无法撤销已经发生的错误。需要中止 Kernel 时使用 `pl.trap()`。

## 六、修复与验证

1. 保留关键 dump，确认原来第一个出错的位置已经正确。
2. 移除本次定位临时添加的 `pl.dump_data`、`pl.printf`、`pl.pto_assert` 和 `pl.trap`；保留 Kernel 原有且仍需要的代码。
3. 确保 Kernel 重新编译。
4. 重新运行原始失败用例。
5. 恢复生产 `block_dim`，覆盖目标 Shape、尾块、dtype 和全部 TilingKey。
6. 重复运行，确认结果稳定。

测试数据、失败 case 和比较阈值保持不变。

## 七、输出模板

```markdown
## 精度定位结论

### 失败用例
- 复现命令：
- Shape / dtype / layout：
- TilingKey / block_dim：
- 比较阈值和误差摘要：

### 定位证据
- 代码位置：
- 代码审查直接确认时：违反的 API 约束、错误机制及与失败输出的关系：
- 使用 dump 定位时：
  - Core / 循环迭代 / 数据窗口：
  - 前一个正确结果：
  - 当前错误结果：
  - 对应的中间 Golden：

### 原因
- 涉及的搬运或计算：
- 参数或状态错误：
- 为什么会产生当前错误：

### 修复与回归
- 修改内容：
- 原始失败用例结果：
- 多 Shape / dtype / TilingKey / 多核回归结果：
```

如果问题仍处于 `GM → L1 → L0A/L0B → L0C` 这段不可直接观察的范围，明确列出已确认正确的 GM 输入、错误的 Acc 输出、A/B 搬运路径的排查结果和下一步验证计划。
