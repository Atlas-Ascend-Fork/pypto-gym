# mixed_quant_sparse_flash_mla（PyPTO-Pro）

`mixed_quant_sparse_flash_mla`（MQSMLA，混合量化稀疏 Flash MLA）是用 PyPTO-Pro 实现的稀疏
Flash MLA 融合算子。对 BF16 Query、FP8（E4M3FN）原始 KV 与可选 FP4（E2M1）压缩 KV 执行稀疏
Attention，**单 kernel** 融合反量化、QK、在线 Softmax（含 per-head sink）、PV 与输出写回，
并支持整行分核与 metadata 指定的 FlashDecode（FD）分片归约。全部 21 个张量维均为动态，每个
tilingkey 只编译一份产物；共 13 条 P0 用例，精度 13/13 通过。接口契约、数值语义与支持范围由本
目录的实现与测试定义。

## 产品支持情况

| 项目 | 说明 |
| :--- | :--- |
| 硬件平台 | Ascend 950PR（`NpuArch=3510`）；Cube / Vector 核数运行时取自设备属性（`Ascend950PR_9579` 为 28/56，`Ascend950PR_9589` 为 32/64） |
| 使用的计算单元 | Cube 与 AI Vector 融合在单个 MIX kernel 内：Cube 承担 QK/PV 矩阵乘，Vector 承担反量化、在线 softmax、累加与写回 |
| 软件环境 | CANN 9.2.0、PyPTO-Pro DSL（`pypto_pro`）、torch / torch_npu、Python 3.12 |
| 设备侧 kernel 名 | `mixed_quant_sparse_flash_mla_kernel` |
| 输入数据类型 | `q` 为 BF16；`ori_kv` / `cmp_kv` 为 `uint8`（FP8 / FP4 的编码字节，非数值类型）；索引、页表、长度、`metadata` 为 int32；`sinks` 为 FP32 |
| 输出数据类型 | `out` 为 BF16；`lse` 为 FP32 |
| 支持的形状 | `T1 ∈ [1,4096]`、`B ∈ [1,1024]`、`K1`/`K2` `∈ [1,4096]`、页数 `∈ [1,65536]`、`bs`/`bt` `∈ [1,1024]` |
| 分核数 | 由 `metadata`（host 侧规划）给定；host 规划器缺省读取当前设备 `cube_core_num`（`torch.npu.get_device_properties`），不硬编码平台核数，上限 `AIC_CORE_MAX_NUM=36` |
| 测试标注 | `pytestmark = pytest.mark.soc("950")`；不满足 SoC 标注的用例由仓库 conftest 统一跳过 |
| 依赖 | pypto_pro、torch、torch_npu、NumPy、pytest |

不接受的输入由 wrapper 直接拒绝并给出错误信息，不做静默降级：shape/dtype/stride 与上表不符、
`cu_seqlens_q` 不是合法前缀和、`layout_q` / `layout_kv` 不是固定取值、`quant_mode != 1`、
`metadata` 不是本算子 host 入口的产出。

## 功能说明

一个公开 host 函数加一个模块级 `@pl.jit(auto_mutex=True)` 的 MIX kernel 服务全部输入；每次
公开调用在设备侧只发射一个 kernel。两种输入组合：

| 模式 | 输入组合 |
| :--- | :--- |
| 原始 KV 稀疏注意力 | `ori_kv`、`ori_sparse_indices`、`ori_block_table` |
| 混合 KV 稀疏注意力 | 同时提供原始侧与压缩侧的 KV、稀疏索引与页表 |

对每个 query token `t` 与 head `h`，按稀疏索引经 block table 收集两侧有效列的反量化 KV
（原始侧列在前、压缩侧列在后，保持索引顺序）合并为 `X`，同一 `X` 既作 Key 也作 Value：

```text
z_j = softmax_scale * q[t,h,:] · X_j
Z   = exp(sinks_h) + Σ_j exp(z_j)          # sink 只进分母，不贡献 Value
O   = (Σ_j exp(z_j) · X_j) / Z              # 分子分母均 FP32 累加
LSE = log Z
```

两侧有效长度均为 0 的空行输出**严格为 0**、`lse` **严格等于**该 head 的 `sinks`。

**反量化**：原始侧为 FP8 E4M3FN group32（每 token 512 数据字节 + 16 个 BF16 scale = 544B）；
压缩侧为 FP4 E2M1 group16（256 打包字节 + 32 个 BF16 scale = 320B，偶数位置在低 nibble）。
反量化值 = 量化值（FP32）× scale（BF16→FP32）后再舍入到 BF16。

**在线 softmax**：按 128-token tile 推进，并跨原始侧 / 压缩侧连续累计；每个 tile 先以
`alpha = exp(m_old − m_new)` 缩放历史输出；概率 P 舍入 BF16 后参与 PV。每个 tile 内超出该行
有效前缀的无效列：地址 clamp 到该行最后一个有效索引，softmax 排除该列，P 写 0。PV 的 K 维
按 16 的倍数扩展。

**KV 页地址**：`idx → page = idx // bs`、`off = idx % bs`、`phys = block_table[b, page]`，
字节地址 `= phys × kv.stride(0) + off × row_bytes`，因此支持页间 padding（axis-0 非连续）。

**编译形态**：单 kernel 单 launch，`with pl.section_cube()` / `with pl.section_vector()` 划分
CV 边界并靠核间事件握手。tilingkey 只承载 4 个 1-bit 条件（`HAS_ORI_LEN` / `HAS_CMP_LEN` /
`HAS_CMP` / `FD_ENABLE`，即可选输入存在性与 FlashDecode），所有数值参数（`softmax_scale`、
strides、`b_num`、`k`、`bs`、`bt`）走运行时标量；条件在编译期折叠，被门控的分支整段不编译
（如 FD 关闭时 FD 归约段不存在）。缺省的可选输入直接传 `None`，由运行时空指针判断处理，无需
占位张量。

## 函数原型

```python
# op_kernel 加入 sys.path 后：
from mixed_quant_sparse_flash_mla import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)

metadata = mixed_quant_sparse_flash_mla_metadata(       # -> int32[1024]
    ori_topk_length, cmp_topk_length,                   # int32 [T1, 1]（每行有效 TopK 长度）
    cu_seqlens_q=cu_seqlens_q,                          # int32 [B+1]（缺省单 batch [0, T1]）
    num_heads_q=64, num_heads_kv=1, head_dim=512,
    quant_mode=1, has_cmp_kv=True,
)

mixed_quant_sparse_flash_mla(
    q, *, ori_kv, ori_sparse_indices, ori_block_table, cu_seqlens_q,
    sinks, metadata, quant_mode=1, softmax_scale=None,
    cmp_kv=None, cmp_sparse_indices=None, cmp_block_table=None,
    ori_topk_length=None, cmp_topk_length=None,
    seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None,
    layout_q="TND", layout_kv="PA_BBND", return_softmax_lse=False,
    out=None, lse=None, block_dim=None,                      # 返回 {"out": out, "lse": lse}
)
```

`mixed_quant_sparse_flash_mla_metadata` 是 host 侧公开入口，按 `cu_seqlens_q` 与每行长度做
整行分核规划，产出主 kernel 消费的 `int32[1024]` 描述；它读取的是长度张量的数值（D2H），
不属于主 kernel wrapper 的边界。全部张量须位于同一 NPU。

### 参数说明

`T1` 为展平后的 query token 数，`B` 为 batch 数，`K1` / `K2` 为两侧 TopK，`bs` 为每页 slot
数，`bt` 为块表宽度。下表均为逻辑形状：

| 参数 | dtype | shape | 说明 |
| :--- | :--- | :--- | :--- |
| `q` | BF16 | `[T1, 64, 512]` | Query（TND） |
| `ori_kv` | uint8 | `[pages, bs, 1, 544]` | FP8 E4M3FN group32 原始 KV 编码字节 |
| `ori_sparse_indices` | int32 | `[T1, 1, K1]` | 稀疏索引；合法前缀之外的列被忽略，可为 -1 |
| `ori_block_table` | int32 | `[B, bt]` | 逻辑页 → 物理页 |
| `ori_topk_length` | int32 / None | `[T1, 1]` | 每行有效 TopK；省略 = 全 K 有效 |
| `sinks` | FP32 | `[64]` | 每 head 的 sink |
| `metadata` | int32 | `[1024]` | `mixed_quant_sparse_flash_mla_metadata` 的输出 |
| `cmp_kv` | uint8 / None | `[pages, bs, 1, 320]` | FP4 E2M1 group16 压缩 KV；省略 = 关闭压缩侧 |
| `cmp_sparse_indices` | int32 / None | `[T1, 1, K2]` | 压缩侧稀疏索引 |
| `cmp_block_table` | int32 / None | `[B, bt]` | 压缩侧逻辑页 → 物理页 |
| `cmp_topk_length` | int32 / None | `[T1, 1]` | 压缩侧每行有效 TopK；省略 = 全 K 有效 |
| `cu_seqlens_q` | int32 | `[B+1]` | Query 前缀和，`cu[0]=0`、末值 `T1` |
| `seqused_q` / `seqused_ori_kv` / `seqused_cmp_kv` | int32 / None | — | 仅做类型 / 形状校验，不参与计算 |
| `softmax_scale` | float / None | 标量 | 缺省 `1/sqrt(512)`；运行期标量，改数值不触发重编译 |
| `return_softmax_lse` | bool | 标量 | 为 `False` 时 `lse` 仍在 kernel 内计算，但由 wrapper 分配后丢弃 |
| `out` / `lse` | BF16 / FP32 | `[T1, 64, 512]` / `[1, T1, 64]` | 未传时由 wrapper `torch.empty` 分配 |
| `block_dim` | int / None | 标量 | 缺省用 `metadata` 规划出的核数 |

### 返回值说明

返回 `{"out": out, "lse": lse}` 字典；`out` / `lse` 调用方预分配时原地写入，未传时由 wrapper
用 `torch.empty` 分配。

| 输出 | dtype | shape | 说明 |
| :--- | :--- | :--- | :--- |
| `out` | `torch.bfloat16` | `[T1, 64, 512]` | 稀疏注意力输出 |
| `lse` | `torch.float32` | `[1, T1, 64]` | 每 token 每 head 的 `logsumexp`（`LSE = log Z`） |

## 约束说明

**输入与布局**

1. 全部张量须位于同一 NPU，dtype 与 shape 如上表；`ori_kv` / `cmp_kv` 以 `uint8` 承载 FP8 /
   FP4 编码字节，调用方应通过 `.view(...)` 提供字节视图，不能用数值类型转换替代字节重解释。
2. `cu_seqlens_q` 是合法 int32 前缀和：`cu[0]==0`、非降、`cu[-1]==T1`。
3. 每行 `ori_topk_length ∈ [0, K1]`、`cmp_topk_length ∈ [0, K2]`；合法稀疏索引
   `∈ [0, s_side)`（`s_side` 为该 batch 的 KV 窗口），**合法前缀之外**的列被忽略、可为 -1。
   越界索引属非法输入，kernel 不做值校验。
4. `ori_kv` / `cmp_kv` 的 axis-0 stride 分别 `≥ bs×544` / `bs×320`（**允许页间 padding**），
   内层维度必须连续；`block_table` 把逻辑页映射到物理页（`[0, pages)`）。
5. `metadata` 必须是本算子 host 入口产出的 `int32[1024]`，且与调用时的块数一致。
6. `layout_q` 固定 `"TND"`、`layout_kv` 固定 `"PA_BBND"`、`quant_mode` 固定 `1`。

**数值与执行**

7. sink 只进分母、不贡献 Value；两侧有效长度均为 0 的行输出严格为 0，`lse` 严格等于 `sinks`。
8. 反量化值 = 量化值（FP32）× scale（BF16→FP32）后舍入到 BF16；概率 P 同样舍入 BF16 后参与
   PV，分子分母按 FP32 累加。这些舍入位置是数值契约的组成部分，见「功能说明」。
9. 无效列（超出该行有效前缀）的地址 clamp 到该行最后一个有效索引，softmax 排除该列、P 写 0。
10. host 侧公开 wrapper 只做参数校验（shape / dtype / stride 元数据）、`torch.empty` 分配合同
    输出、派生 Python 标量并发射一次 kernel；**不做任何设备侧计算或数据搬运**。FD 的 partial
    归约缓冲区由 wrapper 分配的 workspace 承载。
11. kernel 恒计算 LSE；`return_softmax_lse=False` 只是把它丢弃（额外代价为一次小的 GM 写）。

**支持范围**

12. 精度只在 `tests/` 的 13 条 P0 具体 shape 上验证；本算子不声明任意 `T1`/`B`/`K`/页尺寸组合的
    覆盖、不声明 `quant_mode != 1` 或其他 layout、不声明 FD 归约的执行路径（metadata 契约字段
    保留，本机 uniform 规划下为 0）、不声明确定性 level-3 逐行拷贝模式。

## 调用示例

```python
import sys

sys.path.insert(0, "ops/pypto_pro/deepseek_v4_1/mixed_quant_sparse_flash_mla/op_kernel")
from mixed_quant_sparse_flash_mla import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)

metadata = mixed_quant_sparse_flash_mla_metadata(
    ori_topk_length, cmp_topk_length,
    cu_seqlens_q=cu_seqlens_q, num_heads_q=64, num_heads_kv=1, head_dim=512,
    quant_mode=1, has_cmp_kv=True,
)                                                        # host 侧规划，设备侧不执行

result = mixed_quant_sparse_flash_mla(
    q, ori_kv=ori_kv, ori_sparse_indices=ori_sparse_indices,
    ori_block_table=ori_block_table, cu_seqlens_q=cu_seqlens_q,
    sinks=sinks, metadata=metadata, quant_mode=1,
    cmp_kv=cmp_kv, cmp_sparse_indices=cmp_sparse_indices,
    cmp_block_table=cmp_block_table,
    ori_topk_length=ori_topk_length, cmp_topk_length=cmp_topk_length,
)
out, lse = result["out"], result["lse"]
```

运行测试（均在仓库根目录执行，需先加载 CANN 与 PyPTO 环境）：

```bash
D=ops/pypto_pro/deepseek_v4_1/mixed_quant_sparse_flash_mla

# 全部 P0 case（精度矩阵）；或子集
python $D/tests/test_mixed_quant_sparse_flash_mla.py
python $D/tests/test_mixed_quant_sparse_flash_mla.py --cases mixed_decode,decode_1

pytest $D/tests -q
```

直接运行测试文件时逐 case 打印一行判定，并把结果写入 `--out` 指定的 JSON（默认
`tests/reports/pro_pypto/validation.json`，目录自动创建）。精度判据、用例清单、绝对性能结果
与优化记录见同级目录的 [`RESULTS.md`](RESULTS.md)，实现结构与流水设计见
[`DESIGN.md`](DESIGN.md)。
