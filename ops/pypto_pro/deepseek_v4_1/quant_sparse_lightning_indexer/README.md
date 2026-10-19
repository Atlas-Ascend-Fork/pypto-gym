# QuantSparseLightningIndexer（PyPTO-Pro）

`quant_sparse_lightning_indexer` 使用原生 PyPTO-Pro 实现 MXFP4 稀疏索引：读取每个 Query 指定的候选 Key 块，计算逐 head 加权相关性，返回候选集合中的 token TopK。接口契约、数值语义与支持范围全部由本仓库的实现与测试定义。

## 产品支持情况

| 项 | 取值 |
| :--- | :--- |
| 芯片 | Ascend 950PR / 950DT（`NpuArch.DAV_3510`，kernel 装饰器 `arch="a5"`） |
| 能力依赖 | `matmul_mx`（MXFP4 + E8M0 分组量化）、`MemorySpace.ScaleLeft/ScaleRight`、`AccToVecMode.SingleModeVec0/Vec1`（Acc→UB 窄化）、VF 累计直方图（`vf.histograms`）、非对齐压缩存储（`vf.squeeze_store_unalign`）、寄存器 `vf.gather`。这些能力在目标工具链中仅按 950 系列登记支持 |
| 分核数 | 运行期由 `pypto_pro.runtime.platform.get_platform_info().cube_core_num` 取得，上限 `min(32, cube_core_num)`，代码内不硬编码 |
| 实测环境 | Ascend950PR_9589，32 Cube / 64 AIV，L2 = 128 MiB；CANN 9.3.0、PyPTO 0.2.1、PyTorch 2.12.0+cpu、torch_npu 2.12.0.post2、Python 3.12 |
| 测试标注 | `pytestmark = pytest.mark.soc("950")`；不满足 SoC 标记的用例由仓库 conftest 统一跳过 |
| 依赖 | PyPTO、torch、torch_npu、NumPy、pytest |

## 功能说明

一个公开的 host 函数 + 一个模块级 `@pl.jit(auto_mutex=True)` 的 MIX kernel 服务全部输入；每次公开调用在设备侧只发射一个 kernel（无附带的填充/清零 task，输出哨兵在 kernel 内写出）。batch 数、Q/K 长度、候选长度、页大小、压缩比例与实际 TopK 都是运行时参数，没有按 case 名称或具体 shape 的分发。

Q 的逻辑维度固定为 `[T,32,128]`，K 的逻辑维度为 `[K,1,128]`；MXFP4 由 uint8 打包，一个字节装两个 FP4 E2M1 元素，**低 nibble 先于高 nibble** 解码。E8M0 scale 每 32 个连续逻辑元素一组，指数为 `字节值 - 127`。

对于一个 Query 与候选集合中可见的 K token，计算：

```text
dequant(x) = E2M1(低 nibble 优先) * 2^(E8M0 - 127)      # 每 32 个元素共享一个 E8M0
qk[h] = BF16(dot(dequant(Q[h]), dequant(K)))             # Cube MXFP4 矩阵乘，交接时窄化为 BF16
w[h]  = BF16(W[h])                                       # FP32 权重在 kernel 内转 BF16
s     = 0
for h = 0..31:
    s = BF16(s + max(qk[h], 0) * w[h])                   # 每一步融合乘加结果都舍入为 BF16
indices = unordered_topk_indices(s, topk)
```

QK 交接与每一步 FMA 的结果均舍入为 BF16；相同分数允许选择任意满足 TopK 边界的 token，**不保证输出排序**。NaN 输入不属于已验证范围。

可见性规则：设 batch 内 Query 行号为 `r`，实际 Q/K 长度为 `Uq/Uk`，压缩比例为 `c`，余数为 `res`。`r >= Uq` 时整行无效；`mask_mode = 0` 时可见长度为 `Uk`；`mask_mode = 3`（causal 压缩）时为 `min(Uk, max(0, (Uk*c + res - Uq + r + 1)//c))`。参与计算的集合是「候选 8-token 块」与「该可见区间」的交集：只有落在交集内的 token 才可能进入 TopK，其余按无效项输出。

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
| `q` | uint8 | `[T,32,64]` | packed MXFP4 Query，`T>0`；每行 32 个 head，每 head 128 个逻辑元素打包为 64 字节 |
| `w` | FP32 | `[T,32]` | 逐 Query / 逐 head 合议权重，kernel 内转 BF16，允许负数 |
| `descale_q` | uint8 / float8_e8m0fnu | `[T,32,2,2]` | Q 的 E8M0 scale，每 32 个逻辑元素一组 |
| `k`（PA） | uint8 | `[P,S/8,544]` | 每条记录含 8 个 K 的 512 字节 payload 与 32 字节 scale；页大小 `S` 为 `[16,1024]` 内 16 的倍数 |
| `k`（TND） | uint8 | `[K,1,64]` | batch 拼接的 packed K |
| `descale_k` | uint8 / float8_e8m0fnu | `[K,1,2,2]` | TND 必填；PA 布局必须为 `None`（K scale 已在 544 字节记录内） |
| `candidate_block_indices` | int32 | `[T,1,2048]` | batch 内逻辑 8-token 块号；有效前缀不重复，后缀被忽略 |
| `candidate_block_length` | int32 | `[T,1]` | 每行有效前缀长度 `[0,2048]`；PA 也接受 `[T]` |
| `cu_seqlens_q` | int32 | `[B+1]` | 从 0 开始的单调累积 Query 边界；PA 布局省略时按单 batch 处理 |
| `cu_seqlens_k` | int32 | `[B+1]` | TND 必填的 K 累积边界；PA 布局不使用 |
| `seqused_q` / `seqused_k` | int32 | `[B]` | 实际长度；省略时使用累积边界之差，PA 的 K 长度使用页表容量 |
| `cmp_residual_k` | int32 | `[B]` | `mask_mode=3` 且 `cmp_ratio != 1` 时必填；其余情况必须为 `None` |
| `block_table` | int32 | `[B,max_pages]` | PA 逻辑页到物理页的映射；单 batch 可省略并使用顺序物理页；TND 必须为 `None` |
| `output_idx_offset` | int32 | `[T,1]` | 仅加到有效输出索引上；省略即为 0 |
| `metadata` | int32 | `[1024]` | 必填，LI/LD 调度表：每记录 8 个 int32，LI 从记录 0 开始、LD 从记录 36 开始 |
| `topk` | int | `[1,8192]` | 实际输出数量，不足时补无效项 |
| `candidate_block_size` | int | `8` | 固定取值 |
| `quant_mode` | int | `1` | 固定取值 |
| `mask_mode` | int | `0 / 3` | 无 mask / causal 压缩 |
| `cmp_ratio` | int | `[1,128]` | 压缩比例，仅在 `mask_mode=3` 时生效 |
| `layout_q` | str | `"TND"` | Q 布局固定 |
| `layout_k` | str | `"TND" / "PA_BBND"` | K 布局 |
| `max_seqlen_q` | int | 默认 `-1` | 保留的接口参数，实际范围由序列张量定义 |
| `return_value` | bool | 默认 `False` | 是否输出分数 |

`metadata` 的生成 worker 数必须等于 `min(32, get_platform_info().cube_core_num)`；序列、候选长度或 mask 改变后必须重新生成。测试目录提供的 CPU 调度 fixture 只用于构造该输入，不属于设备算子。

### 返回值说明

返回 `(indices, values)` 二元组。

| 输出 | dtype | shape | 说明 |
| :--- | :--- | :--- | :--- |
| `indices` | int32 | `[T,1,topk]` | batch 内 token 索引加 `output_idx_offset`；无效项为 `-1` |
| `values` | BF16 | `return_value=True` 时 `[T,1,topk]`；`False` 时为 `numel()==0` 的空张量 | 对应的分数；无效项为 `-inf` |

两个输出都由 host 用 `torch.empty` 分配、不做任何设备侧预置：`-1` / `-inf` 哨兵全部由 kernel 写出，因此每次公开调用的设备开销就是那一个 kernel。`topk` 大于实际有效数量时，尾部按上述哨兵补齐；输出顺序不保证按分数排序。

## 约束说明

**输入与布局**

1. 全部张量必须在同一 NPU 上、dtype 与 shape 如上表；除 `k` 与 TND 的 `descale_k` 可在 axis 0 追加 padding 外，所有张量必须连续，内层 strides 保持 ND 连续，且物理行不得重叠（`k` 的 axis-0 stride 会被显式校验）。
2. `layout_q` 固定 `"TND"`；`layout_k` 为 `"PA_BBND"` 时 `descale_k` 与 `cu_seqlens_k` 必须为 `None`、`block_table` 可用；为 `"TND"` 时 `descale_k` 与 `cu_seqlens_k` 必填、`block_table` 必须为 `None`。
3. PA 页大小 `S = k.shape[1] * 8` 必须落在 `[16,1024]` 且为 16 的倍数；页数与 `T` 必须为正。
4. `mask_mode = 3` 且 `cmp_ratio != 1` 时 `cmp_residual_k` 必填；否则必须为 `None`。省略 `seqused_q/seqused_k` 时的回退规则见参数表。
5. `candidate_block_indices` 的有效前缀（长度由 `candidate_block_length` 给出）不得含重复块号，且必须是 batch 内合法块号；后缀内容被忽略。调用方负责序列长度、页表与候选前缀的**内容**合法性，算子不把这些输入读回 CPU。

**接口与调度**

6. `quant_mode` 必须为 1，`candidate_block_size` 必须为 8；`1 <= topk <= 8192`，`mask_mode ∈ {0,3}`，`1 <= cmp_ratio <= 128`，`max_seqlen_q >= -1`。
7. `metadata` 必填，为 1024 个 int32、连续、与 `q` 同设备；其 LI/LD 记录必须与实际的序列长度、候选长度、mask 参数一致，且生成时使用的 worker 数必须等于本次调用的 `min(32, cube_core_num)`。
8. TopK 容量按 `16,32,64,128,256,512,1024,2048,4096,8192` 归档选择编译特化（例如 `topk=257/383/511/512` 共用容量 512），输出仍严格按实际 `topk` 截断。tiling key 只包含布局、是否写分数、TND 连续性类别、成对 DMA 开关与 TopK 容量档位，不含任何 shape 或 case 名称。

**数值与执行**

9. QK 交接与逐 head 融合乘加均按 BF16 舍入（见「功能说明」的公式），这是数值契约的一部分，不是放宽容差；NaN 输入不属于已验证范围。
10. host 只做形状/属性校验、工作区与输出分配、容量档位选择与一次 kernel 启动；输出存储相互独立，不跨调用共享工作区。
11. 在确定性等级为 3 的环境中，成对 DMA 通路会被关闭（改为逐记录搬运）以保证批间一致性；该模式下数值结果不变、设备耗时上升。
12. kernel 内的模块级不变量（跨核事件 id 预算、UB/L0 占用、GM staging 环尺寸）在导入期由 `_check_constant` 校验，违反时抛 `RuntimeError`；因此即使在 `python -O` 下也不会被静默跳过。

## 调用示例

仓库自带的输入与 metadata fixture 可直接构造一次完整调用（`sparse=True` 表示走候选块稀疏路径）：

```python
import runpy

TEST = ("ops/pypto_pro/deepseek_v4_1/quant_sparse_lightning_indexer/tests/"
        "test_quant_sparse_lightning_indexer.py")
test = runpy.run_path(TEST)

data = test["_make_inputs"]("PA_BBND", 128, 3, sparse=True)
indices, values = test["_benchmark_call"](data, return_value=True)()
```

`_benchmark_call` 返回的是一个已绑定全部实参的可调用对象，等价于直接调用公开函数：

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

运行测试与采集性能（均在仓库根目录执行，需先加载 CANN 与 PyPTO 环境并选择目标 NPU；默认设备读取 `NPU_DEVICE_ID`，其次 `TILE_FWK_DEVICE_ID`，都未设置时使用逻辑设备 0）：

```bash
# 全量功能用例（也可直接用 pytest）
python ops/pypto_pro/deepseek_v4_1/quant_sparse_lightning_indexer/tests/test_quant_sparse_lightning_indexer.py
pytest ops/pypto_pro/deepseek_v4_1/quant_sparse_lightning_indexer/tests -q

# 冷 L2 设备耗时采集；省略 --cases 覆盖全部 11 个设备场景
python ops/pypto_pro/deepseek_v4_1/quant_sparse_lightning_indexer/tests/test_quant_sparse_lightning_indexer.py \
  --benchmark --cases typical --values false --repeats 20
```

直接运行测试文件时，逐 case 打印一行判定；`--list` 只列出与 `pytest --collect-only -q` 完全一致的用例 id。精度判据、用例清单与绝对性能结果见同级目录的 `RESULTS.md`，实现结构与流水设计见 `DESIGN.md`。
