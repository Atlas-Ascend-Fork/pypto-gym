# QuantSparseLightningIndexer（PyPTO-Pro）

`quant_sparse_lightning_indexer` 使用原生 PyPTO-Pro 实现 MXFP4 稀疏索引：读取每个 Query 指定的候选 Key 块，计算逐 head 加权相关性，返回候选集合中的 token TopK。候选块由调用方提供，每块含 8 个 token；算子在这些 token 中选取 TopK，并不生成候选块，也不计算后续 Attention 输出。接口、数值语义与支持范围以本目录的 kernel 和测试为依据。

## 产品支持情况

| 项 | 取值 |
| :--- | :--- |
| 芯片 | Ascend 950PR / 950DT（`NpuArch.DAV_3510`，kernel 装饰器 `arch="a5"`） |
| 能力依赖 | `matmul_mx`（MXFP4 + E8M0 分组量化）、`MemorySpace.ScaleLeft/ScaleRight`、`AccToVecMode.SingleModeVec0/Vec1`（Acc→UB 窄化）、VF 累计直方图（`vf.histograms`）、非对齐压缩存储（`vf.squeeze_store_unalign`）、寄存器 `vf.gather`。这些能力在目标工具链中仅按 950 系列登记支持 |
| 分核数 | 运行期由 `pypto_pro.runtime.platform.get_platform_info().cube_core_num` 取得，上限 `min(32, cube_core_num)`，启动核数随平台获取，工作区与事件设计按最多 32 个 worker 预留 |
| 实测环境 | Ascend950PR_9589，32 Cube / 64 AIV，L2 = 128 MiB；CANN 9.3.0、PyPTO 0.2.1、PyTorch 2.12.0+cpu、torch_npu 2.12.0.post2、Python 3.12 |
| 测试标注 | `pytestmark = pytest.mark.soc("950")`；不满足 SoC 标记的用例由仓库 conftest 统一跳过 |
| 依赖 | PyPTO、torch、torch_npu、NumPy、pytest |

## 功能说明

本文中的 Query（Q）是待检索的查询，Key（K）是被检索的序列元素，head 是 Q 的头维度。`T` 表示所有 batch 拼接后的 Query 行数；K 的每个 token 只有一个 head，被 32 个 Query head 共同使用。TND 布局按 token、head、特征维度组织数据；PA_BBND 布局通过页表访问分页存放的 K，记录格式见参数表。

host 指 CPU 侧调用代码；MIX kernel 同时使用 AIC（Cube 矩阵计算单元）和 AIV（向量计算单元）。本文的 worker 是一个 AIC 及其两个 AIV，tile 是一次处理的 512 个候选 token（64 个候选块）；候选在原 K 中不必连续。

一个公开的 host 函数和一个模块级 `@pl.jit(auto_mutex=True)` 的 MIX kernel 服务全部输入；每次公开调用在设备侧只发射一个 kernel（无附带的填充/清零 task，输出哨兵在 kernel 内写出）。batch 数、Q/K 长度、候选长度、页大小、压缩比例与实际 TopK 都是运行时参数，没有按 case 名称或具体 shape 的分发。

Q 的逻辑维度固定为 `[T,32,128]`，K 的逻辑维度为 `[K,1,128]`；MXFP4 由 uint8 打包，一个字节装两个 FP4 E2M1 元素，**先解码低 4 位，再解码高 4 位**。E8M0 scale 每 32 个连续逻辑元素一组，指数为 `字节值 - 127`。

对于一个 Query 与候选集合中可见的 K token，计算：

```text
dequant(x) = E2M1(低 4 位优先) * 2^(E8M0 - 127)      # 每 32 个元素共享一个 E8M0
qk[h] = BF16(dot(dequant(Q[h]), dequant(K)))             # Cube MXFP4 矩阵乘，交接时窄化为 BF16
w[h]  = BF16(W[h])                                       # FP32 权重在 kernel 内转 BF16
s     = 0
for h = 0..31:
    s = BF16(s + max(qk[h], 0) * w[h])                   # 每一步融合乘加结果都舍入为 BF16
indices = unordered_topk_indices(s, topk)
```

上式中乘法与加法由 `vf.mul_add_dst` 融合计算，再将结果舍入为 BF16，乘积不单独舍入。ReLU 仅作用于逐 head 的 QK 点积，权重允许为负，因此最后的分数也可能为负。这里不包含 softmax、Attention 的缩放系数或 head 平均。

QK 交接与每一步融合乘加的结果均舍入为 BF16；相同分数允许选择任意满足 TopK 边界的 token，**不保证输出排序**。NaN 输入不属于已验证范围。

可见性规则：设 batch 内 Query 行号为 `r`，实际 Q/K 长度为 `Uq/Uk`，压缩比例为 `c`，余数为 `res`。`r >= Uq` 时整行无效；`mask_mode = 0` 时可见长度为 `Uk`；`mask_mode = 3`（causal 压缩）时为 `min(Uk, max(0, (Uk*c + res - Uq + r + 1)//c))`。允许参与 TopK 选择的集合是「候选 8-token 块」与「该可见区间」的交集：只有落在交集内的 token 才可能进入 TopK，未被选中的候选块和不可见 token 不参与输出选择。有效 token 不足 `topk` 时才补无效项。

例如，候选前缀为 `[5, 1]`，可见长度为 43，则允许集合为 token `40..42` 和 `8..15`，共 11 项。即使其他块有更大的分数，也不能入选。输出索引是 batch 内的逻辑 token 编号，不是候选数组位置或 PA 物理地址；若该行 offset 为 10000，则 token 40 输出为 10040，补齐位置仍为 `-1`。

## 函数原型

```python
import torch

def quant_sparse_lightning_indexer(
    q, k, w, descale_q, candidate_block_indices, candidate_block_length,
    cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None,
    cmp_residual_k=None, block_table=None, output_idx_offset=None, metadata=None,
    *, descale_k=None, topk, candidate_block_size, quant_mode,
    max_seqlen_q=-1, mask_mode=0, cmp_ratio=1,
    layout_q="TND", layout_k="TND", return_value=False,
) -> tuple[torch.Tensor, torch.Tensor]: ...
```

`topk`、`candidate_block_size`、`quant_mode` 是必须显式提供的 keyword-only 参数。全部张量须位于同一 NPU。

### 参数说明

| 参数 | dtype | shape / 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `q` | uint8 | `[T,32,64]` | 按每字节两个 FP4 元素打包的 MXFP4 Query，`T>0`；每行 32 个 head，每 head 128 个逻辑元素打包为 64 字节 |
| `w` | FP32 | `[T,32]` | 逐 Query / 逐 head 的加权系数，kernel 内转 BF16，允许负数 |
| `descale_q` | uint8 / float8_e8m0fnu | `[T,32,2,2]` | Q 的 E8M0 scale；最后两维展平为 4 个 scale，依次对应特征 0–31、32–63、64–95、96–127 |
| `k`（PA） | uint8 | `[P,S/8,544]` | 物理页数为 `P`；每条记录含 8 个 token 的 512 字节 K 数据与 32 字节 scale；页大小 `S` 为 `[16,1024]` 内 16 的倍数 |
| `k`（TND） | uint8 | `[K,1,64]` | 各 batch 拼接后的打包 K 数据 |
| `descale_k` | uint8 / float8_e8m0fnu | `[K,1,2,2]` | TND 必填；PA 布局必须为 `None`（K scale 已在 544 字节记录内） |
| `candidate_block_indices` | int32 | `[T,1,2048]` | batch 内逻辑 8-token 块号；有效前缀不重复，后缀被忽略 |
| `candidate_block_length` | int32 | `[T,1]` | 每行有效前缀长度 `[0,2048]`；PA 也接受 `[T]` |
| `cu_seqlens_q` | int32 | `[B+1]` | 从 0 开始的单调累积 Query 边界；TND 必填；PA 布局省略时按单 batch 处理 |
| `cu_seqlens_k` | int32 | `[B+1]` | TND 必填的 K 累积边界；PA 布局不使用，建议传 `None` |
| `seqused_q` / `seqused_k` | int32 | `[B]` | 实际长度；省略时使用累积边界之差，PA 的 K 长度使用页表容量 |
| `cmp_residual_k` | int32 | `[B]` | `mask_mode=3` 且 `cmp_ratio != 1` 时必填；其余情况必须为 `None` |
| `block_table` | int32 | `[B,max_pages]` | PA 逻辑页到物理页的映射；单 batch 可省略并使用顺序物理页；TND 必须为 `None` |
| `output_idx_offset` | int32 | `[T,1]` | 仅加到有效输出索引上；省略即为 0 |
| `metadata` | int32 | `[1024]` | 必填的任务调度表；LI 记录分配连续 tile 区间，LD 记录分配跨核分片行的归并任务；每记录 8 个 int32，记录号从 0 计数，LD 从记录 36 开始 |
| `topk` | int | `[1,8192]` | 实际输出数量，不足时补无效项 |
| `candidate_block_size` | int | `8` | 固定取值 |
| `quant_mode` | int | `1` | 固定取值 |
| `mask_mode` | int | `0 / 3` | 无 mask / causal 压缩 |
| `cmp_ratio` | int | `[1,128]` | 压缩比例，仅在 `mask_mode=3` 时生效 |
| `layout_q` | str | `"TND"` | Q 布局固定 |
| `layout_k` | str | `"TND" / "PA_BBND"` | K 布局 |
| `max_seqlen_q` | int | 默认 `-1` | 保留的接口参数，实际范围由序列张量定义 |
| `return_value` | bool | 默认 `False` | 是否输出分数 |

PA 的 544 字节记录按以下顺序存放：前 512 字节是 8 个 token 各自的 64 字节 packed K；后 32 字节是这 8 个 token 各自的 4 个 E8M0 scale。逻辑块号 `b` 对应 token `8*b..8*b+7`。设页大小为 `S`，则逻辑页号为 `b//(S/8)`，页内记录号为 `b%(S/8)`，通过 `block_table` 获取物理页后即可读取整条记录。TND 则先用 `cu_seqlens_k` 定位 batch 起点，再访问相应 token 的 K 和独立 scale。

`metadata` 的生成 worker 数必须等于 `min(32, get_platform_info().cube_core_num)`；序列、候选长度或 mask 改变后必须重新生成。测试中的 `_make_metadata` 调用 `_schedule_cpu` 构造该输入，并将结果拷贝到 NPU；这一步会读取序列及候选长度到 CPU，不属于公开算子，也不计入设备性能窗口。公开函数不会自动生成或校验调度表内容，调用方需要保证它完整覆盖所有输出行，包括空行。

### 返回值说明

返回 `(indices, values)` 二元组。

| 输出 | dtype | shape | 说明 |
| :--- | :--- | :--- | :--- |
| `indices` | int32 | `[T,1,topk]` | batch 内 token 索引加 `output_idx_offset`；无效项为 `-1` |
| `values` | BF16 | `return_value=True` 时 `[T,1,topk]`；`False` 时为 `numel()==0` 的空张量 | 对应的分数；无效项为 `-inf` |

`return_value=False` 时 host 仍分配完整的 `out_val` 缓冲区并传入 kernel，但 `WriteValues=0` 特化不写公开分数，最终只返回其长度为 0 的视图；因此省略分数输出并不代表省掉该缓冲区的分配。

两个输出都由 host 用 `torch.empty` 分配、不做任何设备侧预置：`-1` / `-inf` 哨兵全部由 kernel 写出，因此每次公开调用的设备开销就是那一个 kernel。`topk` 大于实际有效数量时，尾部按上述哨兵补齐；输出顺序不保证按分数排序。

## 约束说明

**输入与布局**

1. 全部张量必须在同一 NPU 上、dtype 与 shape 如上表；除 `k` 与 TND 的 `descale_k` 可在 axis 0 追加 padding 外，所有张量必须连续，内层 strides 保持 ND 连续，且物理行不得重叠（`k` 的 axis-0 stride 会被显式校验）。
2. `layout_q` 固定 `"TND"`；`layout_k` 为 `"PA_BBND"` 时 `descale_k` 必须为 `None`、`cu_seqlens_k` 不使用、`block_table` 可用；为 `"TND"` 时 `descale_k` 与 `cu_seqlens_k` 必填、`block_table` 必须为 `None`。
3. PA 页大小 `S = k.shape[1] * 8` 必须落在 `[16,1024]` 且为 16 的倍数；页数与 `T` 必须为正。
4. `mask_mode = 3` 且 `cmp_ratio != 1` 时 `cmp_residual_k` 必填；否则必须为 `None`。省略 `seqused_q/seqused_k` 时的回退规则见参数表。
5. `candidate_block_indices` 的有效前缀（长度由 `candidate_block_length` 给出）不得含重复块号，且必须是 batch 内合法块号；后缀内容被忽略。调用方负责序列长度、页表与候选前缀的**内容**合法性：累积边界应覆盖张量范围，实际长度不能超过对应容量，页表项必须指向有效物理页，候选块号应落在该 batch 的合法 K 范围内。`cmp_residual_k` 表示压缩后的余数，通常满足 `0 <= res < cmp_ratio`。输出 offset 应保证有效索引加偏移后仍可用 int32 表示且不与 `-1` 冲突。算子不把这些输入读回 CPU，也不逐项检查这些内容。

**接口与调度**

6. `quant_mode` 必须为 1，`candidate_block_size` 必须为 8；`1 <= topk <= 8192`，`mask_mode ∈ {0,3}`，`1 <= cmp_ratio <= 128`，`max_seqlen_q >= -1`。
7. `metadata` 必填，为 1024 个 int32、连续、与 `q` 同设备；其 LI/LD 记录必须与实际的序列长度、候选长度、mask 参数一致，且生成时使用的 worker 数必须等于本次调用的 `min(32, cube_core_num)`。
8. TopK 容量按 `16,32,64,128,256,512,1024,2048,4096,8192` 归档选择编译特化（例如 `topk=257/383/511/512` 共用容量 512），输出仍严格按实际 `topk` 截断。tiling key 只包含布局、是否写分数、TND 连续性类别、成对 DMA 开关与 TopK 容量档位，不含任何 shape 或 case 名称。

**数值与执行**

9. QK 交接与逐 head 融合乘加均按 BF16 舍入（见「功能说明」的公式），这是数值契约的一部分，不是放宽容差；NaN 输入不属于已验证范围。
10. host 只做形状/属性校验、工作区与输出分配、容量档位选择与一次 kernel 启动；输出存储相互独立，不跨调用共享工作区。
11. 在确定性等级为 3 的环境中，PA 的成对 DMA 通路会被关闭（改为逐记录搬运）以适配框架的批间一致性要求；打分公式保持相同，同分项仍按无序 TopK 约定处理，设备耗时可能上升。
12. 模块通过 `_check_constant` 校验 tile 与块数关系、环深度的 2 的幂约束、整记录尺寸关系以及跨核事件 id 的范围和冲突，违反时抛 `RuntimeError`，在 `python -O` 下仍有效。UB 布局函数另计算总占用及 `_fits` 标志，但没有对全部 UB/L0 容量做统一的导入期强制校验；具体占用见 `DESIGN.md`。

## 调用示例

仓库自带的输入与 metadata 辅助函数可直接构造一次完整调用（`sparse=True` 表示走候选块稀疏路径）：

```python
import runpy

TEST = ("ops/pypto_pro/deepseek_v4_1/quant_sparse_lightning_indexer/tests/"
        "test_quant_sparse_lightning_indexer.py")
test = runpy.run_path(TEST)

data = test["_make_inputs"]("PA_BBND", 128, 3, sparse=True)
indices, values = test["_benchmark_call"](data, return_value=True)()
```

`_benchmark_call` 返回一个已绑定全部实参的可调用对象。若已经按参数表准备好输入和 metadata，也可直接调用公开函数；下面展示不返回分数的 PA 调用形式：

```python
import sys

sys.path.insert(0, "ops/pypto_pro/deepseek_v4_1/quant_sparse_lightning_indexer/op_kernel")
from quant_sparse_lightning_indexer import quant_sparse_lightning_indexer

indices, values = quant_sparse_lightning_indexer(
    q, k, w, descale_q, candidate_ids, candidate_lengths,
    cu_seqlens_q=cu_seqlens_q, seqused_q=seqused_q, seqused_k=seqused_k,
    cmp_residual_k=cmp_residual_k, block_table=block_table,
    output_idx_offset=output_idx_offset, metadata=metadata,
    descale_k=None, topk=512, candidate_block_size=8, quant_mode=1,
    mask_mode=3, cmp_ratio=2, layout_q="TND", layout_k="PA_BBND",
    return_value=False,
)
```

运行测试与采集性能（均在仓库根目录执行，需先加载 CANN 与 PyPTO 环境并选择目标 NPU；测试脚本读取 `TILE_FWK_DEVICE_ID`，未设置时使用逻辑设备 0）：

```bash
# 全量功能用例（也可直接用 pytest）
python ops/pypto_pro/deepseek_v4_1/quant_sparse_lightning_indexer/tests/test_quant_sparse_lightning_indexer.py
pytest ops/pypto_pro/deepseek_v4_1/quant_sparse_lightning_indexer/tests -q

# 冷 L2 设备耗时采集；省略 --cases 覆盖全部 11 个设备场景
python ops/pypto_pro/deepseek_v4_1/quant_sparse_lightning_indexer/tests/test_quant_sparse_lightning_indexer.py \
  --benchmark --cases typical --values false --repeats 20
```

例如可先执行 `export TILE_FWK_DEVICE_ID=0` 选择设备。直接运行测试文件时，逐 case 打印一行判定；`--list` 只列出与 `pytest --collect-only -q` 完全一致的用例 id。精度判据、用例清单与绝对性能结果见同级目录的 `RESULTS.md`，实现结构与流水设计见 `DESIGN.md`。
