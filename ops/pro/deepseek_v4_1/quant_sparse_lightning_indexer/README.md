# QuantSparseLightningIndexer (quant_sparse_lightning_indexer)

MXFP4 稀疏索引算子，以 **built-in 形态**交付：kernel 在构建期由 `asc_opc` 预编译成二进制，
随 custom run 包安装，运行期通过生成的 `aclnnQuantSparseLightningIndexer` 接口调用。
算子读取每个 Query 指定的候选 Key 块，计算逐 head 加权相关性，返回候选集合中的 token TopK。
候选块由调用方提供，每块含 8 个 token；算子在这些 token 中选取 TopK，不生成候选块，
也不计算后续 Attention 输出。

## 产品支持情况

| 产品                                                     | 是否支持 |
| -------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |

## 功能说明

- 算子功能：Q 的逻辑维度固定 `[T,32,128]`、K 为 `[K,1,128]`；MXFP4 由 uint8 打包
  （一字节两个 FP4 E2M1，**先低 4 位后高 4 位**），E8M0 scale 每 32 个连续逻辑元素一组、
  指数为 `字节值 - 127`。对一个 Query 与候选集合中可见的 K token：

```text
dequant(x) = E2M1(低 4 位优先) * 2^(E8M0 - 127)
qk[h] = BF16(dot(dequant(Q[h]), dequant(K)))       # Cube MXFP4 矩阵乘，交接时窄化为 BF16
w[h]  = BF16(W[h])
s     = 0
for h = 0..31:
    s = BF16(s + max(qk[h], 0) * w[h])             # 每步融合乘加结果都舍入为 BF16
indices = unordered_topk_indices(s, topk)
```

- 五个设备阶段（A 候选取址与搬运 / B QK 点积 / C QK 交接 / D head 归约与键编码 /
  E TopK 与输出）同属一个 native task，每次公开调用只发射 **1 个** kernel。
  阶段划分、核间/核内切分、UB 与 L1/L0 布局、GM 环与跨核事件编号见 [DESIGN.md](DESIGN.md)。
- 可见性规则：设 batch 内 Query 行号为 `r`，实际 Q/K 长度为 `Uq`/`Uk`，压缩比例为 `c`，
  余数为 `res`。`r >= Uq` 时整行无效；`mask_mode = 0` 时可见长度为 `Uk`；`mask_mode = 3`
  （causal 压缩）时为 `min(Uk, max(0, (Uk*c + res - Uq + r + 1)//c))`。允许参与 TopK 选择的集合是
  「候选 8-token 块」与「该可见区间」的交集：只有落在交集内的 token 才可能进入 TopK，
  未被选中的候选块和不可见 token 不参与输出选择。有效 token 不足 `topk` 时才补无效项。
- 输出索引是 batch 内的逻辑 token 编号，不是候选数组位置或 PA 物理地址；若该行
  `output_idx_offset` 为 10000，则 token 40 输出为 10040，补齐位置仍为 `-1`。
  相同分数允许选择任意满足 TopK 边界的 token，**不保证输出排序**。

## 函数原型

入口签名的位置顺序与 `op_host` 的注册顺序逐参数一致（15 输入 → 2 输出 → 9 属性），
输出由调用方预分配。同一份调用代码既可用于 `op_kernel` 的宿主入口，也可用于
生成的 aclnn 接口：

```python
import torch

def quant_sparse_lightning_indexer(
    q, k, w, descale_q, descale_k,
    candidate_block_indices, candidate_block_length,
    cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k,
    block_table, output_idx_offset, metadata,
    indices, values,
    topk, candidate_block_size=8, quant_mode=1, max_seqlen_q=-1,
    mask_mode=0, cmp_ratio=1, layout_q="TND", layout_k="TND", return_value=False,
) -> tuple[torch.Tensor, torch.Tensor]: ...
```

对应的 C 接口（`op_api/include/aclnnop/aclnn_quant_sparse_lightning_indexer.h`，由 op_build 生成）：

```c
aclnnStatus aclnnQuantSparseLightningIndexerGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *w,
    const aclTensor *descaleQ, const aclTensor *descaleKOptional,
    const aclTensor *candidateBlockIndices, const aclTensor *candidateBlockLength,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional, const aclTensor *blockTableOptional,
    const aclTensor *outputIdxOffsetOptional, const aclTensor *metadata,
    int64_t topk, int64_t candidateBlockSize, int64_t quantMode,
    int64_t maxSeqlenQ, int64_t maskMode, int64_t cmpRatio,
    char *layoutQOptional, char *layoutKOptional, bool returnValue,
    const aclTensor *indicesOut, const aclTensor *valuesOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnQuantSparseLightningIndexer(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

`tests/` 里的 `_load_operators()` 在此签名之上再包一层公开调用约定
（属性用 keyword-only 传入、输出在内部按 `[T,1,topk]` 分配、返回 `(indices, values)` 二元组），
因此 62 个用例在 `--mode jit` 与 `--mode aclnn` 下共用同一套调用代码。

## 参数说明

| 参数名                  | 输入/输出/属性 | 描述                                                                                           | 数据类型 | 数据格式 |
| ----------------------- | -------------- | ---------------------------------------------------------------------------------------------- | -------- | -------- |
| q                       | 输入           | 打包 MXFP4 Query，`[T,32,64]`（每 head 128 个逻辑元素）。                                    | UINT8    | ND       |
| k                       | 输入           | 打包 K。PA：`[P, S/8, 544]`（每记录 8 token 的 512B K + 32B scale）；TND：`[K,1,64]`。     | UINT8    | ND       |
| w                       | 输入           | 逐 Query / 逐 head 权重，`[T,32]`，kernel 内转 BF16，允许负数。                              | FLOAT    | ND       |
| descale_q               | 输入           | Q 的 E8M0 scale，`[T,32,2,2]`（4 个 scale 依次对应特征 0–31/32–63/64–95/96–127）。       | UINT8    | ND       |
| descale_k               | 可选输入       | TND 必填 `[K,1,2,2]`；PA 必须缺省（scale 已在 544B 记录内）。                                | UINT8    | ND       |
| candidate_block_indices | 输入           | batch 内逻辑 8-token 块号，`[T,1,2048]`；有效前缀不重复，后缀被忽略。                        | INT32    | ND       |
| candidate_block_length  | 输入           | 每行有效前缀长度 `[0,2048]`，`[T,1]`。                                                     | INT32    | ND       |
| cu_seqlens_q            | 可选输入       | 从 0 开始的单调累积 Query 边界，`[B+1]`；TND 必填，PA 缺省按单 batch。                       | INT32    | ND       |
| cu_seqlens_k            | 可选输入       | TND 必填的 K 累积边界，`[B+1]`；PA 不使用。                                                  | INT32    | ND       |
| seqused_q / seqused_k   | 可选输入       | 实际长度 `[B]`；缺省时用累积边界之差（PA 的 K 长度用页表容量）。                             | INT32    | ND       |
| cmp_residual_k          | 可选输入       | `mask_mode=3` 且 `cmp_ratio != 1` 时必填 `[B]`，其余必须缺省。                           | INT32    | ND       |
| block_table             | 可选输入       | PA 逻辑页→物理页映射 `[B,max_pages]`；单 batch 可缺省；TND 必须缺省。                       | INT32    | ND       |
| output_idx_offset       | 可选输入       | 仅加到有效输出索引上，`[T,1]`；缺省即 0。                                                    | INT32    | ND       |
| metadata                | 输入           | **必填**的 LI/LD 任务调度表 `[1024]`，由调用方生成；算子不生成也不校验其内容。         | INT32    | ND       |
| indices                 | 输出           | batch 内 token 索引加 `output_idx_offset`，`[T,1,topk]`；无效项 `-1`。                   | INT32    | ND       |
| values                  | 输出           | 对应分数，`[T,1,topk]`；无效项 `-inf`。`return_value=false` 时不写入（缓冲区仍须存在）。 | BFLOAT16 | ND       |
| topk                    | 必选属性       | 实际输出数量，`[1,8192]`。                                                                   | INT      | -        |
| candidate_block_size    | 可选属性       | 固定 `8`，默认 8。                                                                           | INT      | -        |
| quant_mode              | 可选属性       | 固定 `1`，默认 1。                                                                           | INT      | -        |
| max_seqlen_q            | 可选属性       | 保留参数，默认 `-1`。                                                                        | INT      | -        |
| mask_mode               | 可选属性       | `0`（无 mask）/ `3`（causal 压缩），默认 0。                                               | INT      | -        |
| cmp_ratio               | 可选属性       | 压缩比例 `[1,128]`，仅 `mask_mode=3` 生效，默认 1。                                        | INT      | -        |
| layout_q                | 可选属性       | Q 布局，固定 `"TND"`。                                                                       | STRING   | -        |
| layout_k                | 可选属性       | `"TND"`（默认）/ `"PA_BBND"`。                                                             | STRING   | -        |
| return_value            | 可选属性       | 是否写出分数，默认 `false`。                                                                 | BOOL     | -        |

`descale_q` / `descale_k` 注册为 **UINT8**：E8M0 本身就是原始指数字节，按字节类型注册可让
aclnn 接口不涉及子字节 dtype，kernel 内再用 `pl.make_ptr(ptr, DT_FP8E8M0)` 重解释为 scale 视图。
调用方若持有 `float8_e8m0fnu` 视图，测试桥会自动 `view(uint8)` 归一。

### 返回值说明

返回 `(indices, values)` 二元组。

| 输出 | dtype | shape | 说明 |
| :--- | :--- | :--- | :--- |
| `indices` | int32 | `[T,1,topk]` | batch 内 token 索引加 `output_idx_offset`；无效项为 `-1` |
| `values` | BF16 | `return_value=True` 时 `[T,1,topk]`；`False` 时为 `numel()==0` 的空视图 | 对应分数；无效项为 `-inf` |

两个输出都是注册输出，必须由调用方（或测试适配层）预先按 `[T,1,topk]` 分配好再传入。
`return_value=False` 时 `values` 缓冲区**仍然必须存在且形状完整**——`WriteValues=0`
特化只是不往里写，host 侧不会因此省掉这块分配；返回值里它被切成 0 长度视图，
使省略分数输出这一语义在返回值上可见。

两个输出都不做任何设备侧预置：`-1` / `-inf` 哨兵全部由 kernel 写出，
因此每次公开调用的设备开销就是那一个 kernel。`topk` 大于实际有效数量时尾部按哨兵补齐；
输出顺序不保证按分数排序。

## 约束说明

**输入与布局**

1. 全部张量必须在同一 NPU 上、dtype 与 shape 如上表；除 `k` 与 TND 的 `descale_k` 可在
   axis 0 追加 padding 外，所有张量必须连续，内层 strides 保持 ND 连续，且物理行不得重叠
   （`k` 的 axis-0 stride 会被显式校验）。
2. `layout_q` 固定 `"TND"`；`layout_k` 为 `"PA_BBND"` 时 `descale_k` 必须缺省、
   `cu_seqlens_k` 不使用、`block_table` 可用；为 `"TND"` 时 `descale_k` 与 `cu_seqlens_k` 必填、
   `block_table` 必须缺省。
3. PA 页大小 `S = k.shape[1] * 8` 必须落在 `[16,1024]` 且为 16 的倍数；页数与 `T` 必须为正。
4. `mask_mode = 3` 且 `cmp_ratio != 1` 时 `cmp_residual_k` 必填；否则必须缺省。
   省略 `seqused_q` / `seqused_k` 时按累积边界之差回退，PA 的 K 长度回退为页表容量。
5. `candidate_block_indices` 的有效前缀（长度由 `candidate_block_length` 给出）不得含重复块号，
   且必须是 batch 内合法块号；后缀内容被忽略。调用方负责序列长度、页表与候选前缀的**内容**
   合法性：累积边界应覆盖张量范围，实际长度不能超过对应容量，页表项必须指向有效物理页，
   候选块号应落在该 batch 的合法 K 范围内。`cmp_residual_k` 表示压缩后的余数，通常满足
   `0 <= res < cmp_ratio`。输出 offset 应保证有效索引加偏移后仍可用 int32 表示且不与 `-1` 冲突。
   算子不把这些输入读回 host，也不逐项检查这些内容。

**接口与调度**

6. `quant_mode` 必须为 1，`candidate_block_size` 必须为 8；`1 <= topk <= 8192`，
   `mask_mode ∈ {0,3}`，`1 <= cmp_ratio <= 128`，`max_seqlen_q >= -1`。
7. `metadata` 必填，为 1024 个 int32、连续、与 `q` 同设备；其 LI/LD 记录必须与实际的序列长度、
   候选长度、mask 参数一致，且生成时使用的 worker 数必须等于本次调用的
   `min(32, cube_core_num)`。算子只校验它的 shape、dtype 与设备，不生成也不校验其内容，
   调用方需保证它完整覆盖所有输出行（包括空行）。
8. TopK 容量按 `16,32,64,128,256,512,1024,2048,4096,8192` 归档选择编译特化
   （例如 `topk=257/383/511/512` 共用容量 512），输出仍严格按实际 `topk` 截断。
   tiling key 只包含布局、是否写分数、TND 连续性类别、成对 DMA 开关与 TopK 容量档位，
   不含任何 shape 或 case 名称。

**数值与执行**

9. QK 交接与逐 head 融合乘加均按 BF16 舍入（见「功能说明」的公式），这是数值契约的一部分，
   不是放宽容差；NaN 输入不属于已验证范围。
10. host 侧只做形状/属性校验、工作区申报与输出校验、容量档位选择与一次 kernel 启动；
    输出存储相互独立，不跨调用共享工作区。
11. 在确定性等级为 3 的环境中，PA 的成对 DMA 通路会被关闭（改为逐记录搬运）以适配框架的
    批间一致性要求；打分公式保持相同，同分项仍按无序 TopK 约定处理，设备耗时可能上升。
12. 模块通过 `_check_constant` 校验 tile 与块数关系、环深度的 2 的幂约束、整记录尺寸关系
    以及跨核事件 id 的范围和冲突，违反时抛 `RuntimeError`，在 `python -O` 下仍有效。
    UB 布局函数另计算总占用及是否放得下的标志，但没有对全部 UB/L0 容量做统一的导入期强制校验；
    具体占用见 [DESIGN.md](DESIGN.md)。

**built-in 形态的附加约束**

- **`k` 与 `descale_k` 在 def.cpp 里不带 `AutoContiguous()`，但 built-in 路径下 kernel 只会看到
  紧凑 stride**：aclnn/nnopbase 在 launch 前会把非连续输入 `aclnnContiguous` 拷成连续
  （实测 host tiling 申报 5,767,168 B，而 `aclnnQuantSparseLightningIndexerGetWorkspaceSize`
  返回 22,545,408 B，差值即 16 MiB 系统预留 + 这块连续化拷贝缓冲）。因此 host tiling
  **按紧凑值申报** `rt_tnd_k_stride=64` / `rt_tnd_s_stride=4` / `rt_key_stride=blocks_per_page*544`，
  `TndStrideMode` 在 built-in 路径下**恒为 3**；它仍调 `GetInputStride` / `GetOptionalInputStride`，
  但只用于校验 axis-0 行不重叠，不用于下发给 kernel。
  按调用方声明的 padded stride 下发是**错的**——那会让 kernel 用错误的 pitch 遍历那份连续副本，
  表现为 indices 正确而每个分数都错。
  推论：`TndStrideMode` 的 0/1/2 三个特化经 aclnn 接口不可达。`QsliKey.is_valid` 保留全部
  120 个合法组合（`op_kernel` 的宿主入口直接把调用方张量交给 kernel、不经过这次拷贝，
  因此仍会选到它们），而预编译进 `.o` 的默认只是 aclnn 可达的 60 个。
  其余输入仍要求连续（`AutoContiguous()`）。
- **无用户可见的辅助张量**：5 个工作区缓冲区
  （`stg_k` / `stg_s` / `key_ws` / `part_key` / `part_pos`）共用**一块框架 workspace**，
  kernel 内用 `pl.addptr` 按 `ws_layout()` 切分。host tiling 申报的字节数为
  `stage_rows*64 + stage_rows*4 + num_workers*MAX_TILES*2*256*2 + 2*(128*TopKCapacity*2)`，
  其中 `stage_rows = num_workers*STAGE_DEPTH*TILE_N`。
- **`num_workers = min(32, GetCoreNumAic())`**，同时作为 `blockDim`；工作区与事件设计按最多 32 worker 预留。
- **`PairMin` 由确定性等级决定**：host tiling 读 `GetDeterministicLevel()`，等级 3 时取 0
  （逐记录 DMA，满足批间一致性），否则取 544（成对记录 DMA）。与 jit 侧
  `_batch_consistency_enabled()` 逐位一致。
- **tiling key**：`QsliKey` 五字段 `WriteValues / IsPA / TopKCapacity / TndStrideMode / PairMin`。
  原笛卡尔积 320 个组合中，`IsPA=1` 时 `TndStrideMode` 不被读取、`IsPA=0` 时 `PairMin` 不被读取，
  故新增 `is_valid` 做**无损剪枝**至 `2 * (10*1*2 + 10*4*1) = 120` 个合法 key（全部编入 `.o`）。
  key 只含布局/是否写分数/TND 连续性类别/成对 DMA 开关/TopK 容量档，**不含任何 shape 或 case 名称**。
- `metadata` 的生成 worker 数必须等于本次调用的 `min(32, cube_core_num)`；序列、候选长度或 mask
  改变后必须重新生成。算子不把它读回 host，也不校验其内容。

## 调用说明

| 调用方式             | 说明                                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Kernel 直调（PyPTO） | `quant_sparse_lightning_indexer(q, k, w, descale_q, descale_k, candidate_block_indices, candidate_block_length, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, block_table, output_idx_offset, metadata, indices, values, topk, ...)`：校验 → 取/建扁平 workspace → 构造 `QsliTiling`（22 字段）→ 启动 `quant_sparse_lightning_indexer_kernel[None, num_workers, key](...)`。 |
| aclnn API            | 已支持（run 包 built-in）。`aclnnQuantSparseLightningIndexerGetWorkspaceSize` + `aclnnQuantSparseLightningIndexer`；测试桥见 `tests/aclnn_bridge/aclnn_bridge_quant_sparse_lightning_indexer.cpp`，`pytest --mode aclnn`。                                                                                                                                                                        |

两种模式的入口签名**逐参数一致**（15 输入 + 2 输出 + 9 属性，输出由调用方预分配），
因此同一组用例、同一组 golden 可在 `--mode jit` 与 `--mode aclnn` 下复用；
`tests/` 里的 `_load_operators()` 适配层在此之上再包一层公开调用约定（keyword-only 属性、
内部分配输出、返回 `(indices, values)`），所以 62 个用例在两种模式下写法一致。

## 目录结构

```
quant_sparse_lightning_indexer/
├── op_kernel/
│   └── quant_sparse_lightning_indexer.py   # @pl.jit kernel（raw Ptr + QsliTiling）+ host 入口
├── op_host/
│   ├── CMakeLists.txt                      # 含 QSLI_DEV_TILING_KEYS_ONLY 开关（只编 8 个代表 key）
│   ├── quant_sparse_lightning_indexer_def.cpp          # 15 输入(8 可选) / 2 输出 / 9 属性
│   ├── quant_sparse_lightning_indexer_infershape.cpp   # indices/values = [T,1,topk]
│   ├── quant_sparse_lightning_indexer_tiling.h         # QsliTilingData（22 个 int64，与 kernel dataclass 字节对齐）
│   └── quant_sparse_lightning_indexer_tiling.cpp       # host tiling：校验 + stride 读取 + rt_* 派生 + key + workspace + blockDim
└── tests/
    ├── CMakeLists.txt
    ├── aclnn_bridge/
    │   ├── CMakeLists.txt
    │   └── aclnn_bridge_quant_sparse_lightning_indexer.cpp   # pybind11 桥（含 E8M0→uint8 归一）
    ├── assets/
    │   ├── quant_sparse_lightning_indexer_golden_cpu.py      # 独立 CPU 参考（MXFP4 解码 + 同舍入链）
    │   └── quant_sparse_lightning_indexer_precision.py       # 无序 TopK 比较器（rtol=1/128, atol=2.5e-5）
    └── test_quant_sparse_lightning_indexer.py                # 62 个用例
```

## 验证

```bash
# 一键（构建 run 包 + 隔离安装 + aclnn 模式系统测试）
export CXXFLAGS="-I$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])')"
python3 build_ci.py -c --pkg-type run --stest

# 单独跑用例
pytest ops/pro/deepseek_v4_1/quant_sparse_lightning_indexer/tests --mode jit -q
pytest ops/pro/deepseek_v4_1/quant_sparse_lightning_indexer/tests --mode aclnn -q
```

精度判据：独立 CPU 参考实现 + 无序 TopK 比较器（`rtol=1/128`、`atol=2.5e-5`），
逐项检查无效项哨兵、有效数量、索引唯一性、允许集合归属、选中位置分数一致性与 TopK 边界。

`jit_only` 用例（4 个）需要直接改写 kernel 模块内部量，built-in 路径下这些对象由框架/预编译二进制持有：

| 用例                                 | 为什么只能 jit                                                                                           |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `test_pa_address_and_dma_dispatch` | monkeypatch `MAX_ADDRESS32_BYTES=0` 与 `_batch_consistency_enabled`，强制 64 位地址与单记录 DMA 路径 |
| `test_dynamic_shape_reuse`         | 拦截 `kernel.__getitem__` 记录 tiling key，并检查 jit 编译缓存项数不增长                               |
| `test_runtime_attributes_reuse`    | 同上（验证运行期属性不改变 tiling key）                                                                  |
| `test_no_sugar_dependency`         | 从语言模块移除 `gemm/gemm_mx/mem_space` 后重新导入 kernel 源文件                                       |

## 性能

上板性能数据（22 个测量点：11 个设备场景 × `return_value=False/True`）统一记录在
**[RESULTS.md](RESULTS.md)「性能结果 → 绝对性能结果」**，含完整的硬件平台与软件栈、
采集口径（每样本前 512 MiB 冷 L2 驱逐、3 轮预热丢弃 + 7 轮正式样本取中位数、
`msprof --task-time=on` 取 NPU task duration）、场景 shape 规格、逐点数值与判读。
本文不重复性能数据，避免两处漂移。

一句话结论：主性能场景 `typical`（`T=72`、每行 2048 候选块、cube util 96.24%、stdev 1.0%~1.5%）
实测 **76.709 µs**（`return_value=false`）/ **77.981 µs**（`true`）。
