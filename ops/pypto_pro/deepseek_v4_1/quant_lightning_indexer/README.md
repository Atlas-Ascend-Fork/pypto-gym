# quant_lightning_indexer

`quant_lightning_indexer` 是 deepseek_v4_1 里 MXFP4 量化的 Lightning Indexer：给每个 query
在一段 key 里算出相关性分数，挑出分数最高的若干个 key 的索引。用 PyPTO-Pro 在昇腾 950 上以
**单 kernel** 融合实现。

- 实现（kernel + host wrapper）：`op_kernel/quant_lightning_indexer.py`
- 测试与精度基准：`tests/`（CPU golden 与精度判定脚本在 `tests/assets/`）
- 设计说明：`DESIGN.md`；用例、精度标准与性能结果：`RESULTS.md`

## 产品支持情况

| 项目 | 内容 |
| :--- | :--- |
| 昇腾产品 | Ascend 950PR（SoC `Ascend950PR_9589`，NPU 架构 `3510`） |
| 执行单元 | 32 个 Cube 核 + 64 个 Vector 核（平台配置 `cube_vector_combine=split`） |
| CANN 版本 | CANN Toolkit 9.3.0（平台配置 `Ascend950PR_9589.ini`） |
| 算子开发方式 | PyPTO-Pro kernel DSL（`pypto_pro.language`） |
| 配套软件栈 | PyTorch 2.8.0 + torch-npu 2.8.0.post4，Python 3.10 |
| 接口约束 | Q 只支持 TND 布局；K 支持 TND 与 PA_BBND（page 为 16 的倍数、不超过 1024） |

kernel 按 arch 3510 编译，启动核数在运行时取自平台信息。其它昇腾产品未做编译与验证。

## 功能说明

先约定几个记号。**MXFP4** 是一种 4 bit 浮点格式，本算子用到它的两种配套编码：
`E2M1` 是数据码（1 位符号 + 2 位指数 + 1 位尾数，能表示
`{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`），`E8M0` 是只存 8 位指数的**块缩放因子**
（实际值是 `2^(E8M0 − 127)`）。两者的打包规则是：一个字节里放两个 `E2M1`（低四位在前），
每 32 个逻辑元素共用一个 `E8M0`。下文里 `Tq` 是 query 的总行数，`Lq` / `Lk` 是单个序列的
query / key 有效长度，`D = 128` 是 head 维度（打包成 `E2M1` 之后是 64 字节）。

一次调用完成四件事：

1. **反量化 Q/K**：把按半字节打包的 `E2M1` 还原成数值，再乘上各自的 `E8M0` 块缩放因子。
2. **多头相关打分**：`q_f[Tq, 32, 128] @ k_f[Lk, 128]^T`，只在这里做一次 BF16 舍入，得到
   `qk[Tq, 32, Lk]`。
3. **稀疏 / 压缩 causal mask**：按 `mask_mode` 求出每个 query 能看到多少个 key，再对 32 个
   head 逐 head 累加折叠（`relu(qk[:, h]) * w[:, h]`），得到分数面 `score[Tq, Lk]`。
4. **精确 Top-K**：输出每个 query 分数最高的 `topk` 个 token 索引；可选同时返回这些位置的分数，
   以及 8-token 候选块（供后续稀疏注意力挑选）。

分数怎么算是一条**数值契约**，不是可以自由优化的目标：

```text
q_f = decode_mxfp4(q, descale_q)                     # [Tq, 32, 128]
k_f = decode_mxfp4(k, descale_k)                     # [Lk, 128]
qk  = bf16(q_f @ k_f^T).float()                      # [Tq, 32, Lk]，只在这里舍入一次
w_b = bf16(w).float()
acc_0 = 0
acc_{h+1} = bf16(acc_h + relu(qk[:, h]) * w_b[:, h]).float()   # h = 0..31，逐步舍入
score = acc_32
```

**head 0..31 的逐步 BF16 舍入顺序是契约的一部分**：每一步都要按 BF16 舍一次再继续，
不能合并成一次 FP32 归约，否则得到的分数不满足上面这条契约。

每个 query 能看到多少 key 由 `mask_mode` 决定（批次内第 `t` 个 query；`t ≥ Lq` 时为 0）：

```text
mask_mode = 0:  visible = Lk
mask_mode = 3:  visible = min(Lk, max(0, (cmp_ratio * Lk + residual − Lq + t + 1)
                                       // cmp_ratio))
```

`Lq` / `Lk` 由 `seqused_q` / `seqused_k` 给出，不传时取张量的存储长度。

Top-K **不要求按分数排序**；出现同分时允许返回不同的索引集合，但必须满足精确 Top-K 的边界、
索引合法性与唯一性。开启候选块时，每 8 个连续 token 组成一块：完整的块取块内可见 token 的
最大分数；末尾那个不完整的有效块置 `+inf`，保证它优先被选中。候选数量为
`min(candidate_topk_blocks, ceil(visible / 8))`，候选里放的是序列内的**块号**，不叠加
`output_idx_offset`。

## 函数原型

```python
indices, values, candidates, candidate_lengths = quant_lightning_indexer_wrapper(
    q, k, w, descale_q, descale_k,                   # 5 个必需张量
    cu_seqlens_q=None, cu_seqlens_k=None,            # 可选：序列前缀
    seqused_q=None, seqused_k=None,                  # 可选：有效长度
    cmp_residual_k=None,                             # 条件必需：压缩 causal 的余量
    block_table=None,                                # 可选：PA 的物理页映射
    output_idx_offset=None,                          # 可选：索引偏移
    metadata=None,                                   # 接口兼容项，不参与计算
    *, topk, quant_mode,                             # 仅关键字，必需
    max_seqlen_q=-1, mask_mode=0, cmp_ratio=1,
    layout_q="TND", layout_k="TND", return_value=False,
    candidate_topk_blocks=-1, candidate_block_size=-1,
)
```

本算子的 kernel 与公开 wrapper 在 `op_kernel/quant_lightning_indexer.py`，把该目录加入 `sys.path`
后导入：

```python
from quant_lightning_indexer import quant_lightning_indexer_wrapper
```

### 参数说明

| 参数 | 类型 / shape | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `q` | UINT8 `[Tq, 32, 64]` | 是 | 打包的 `E2M1`，连续存储 |
| `k` | UINT8 `[Tk, 1, 64]`（TND）、UINT8 `[P, S, 1, 64]`（PA_BBND） | 是 | PA 下 `S` 为 16 的倍数且 `16 ≤ S ≤ 1024` |
| `w` | FP32 `[Tq, 32]` | 是 | 逐 head 权重，允许负值，连续存储 |
| `descale_q` | UINT8 `[Tq, 32, 2, 2]` | 是 | `E8M0` 编码；每个 head 4 个缩放因子，`[2, 2]` 是它们的物理排布 |
| `descale_k` | UINT8 `[Tk, 1, 2, 2]`（TND）、UINT8 `[P, S, 1, 2, 2]`（PA_BBND） | 是 | 与 `k` 的布局一一对应 |
| `cu_seqlens_q` | INT32 `[B+1]` | 否 | Q 的累积长度；不传表示单批次 |
| `cu_seqlens_k` | INT32 `[B+1]` | 条件必需 | TND 的多批次必须传，PA 不用 |
| `seqused_q` | INT32 `[B]` | 否 | Q 的实际有效长度，允许 0 |
| `seqused_k` | INT32 `[B]` | 否 | K 的实际有效长度，允许 0 |
| `cmp_residual_k` | INT32 `[B]` | 条件必需 | `mask_mode=3 && cmp_ratio!=1` 时必须传，缺省时必须不传；这是调用方的前置条件，取值范围 `[0, cmp_ratio)` |
| `block_table` | INT32 `[B, W]` | 条件必需 | PA 的逻辑页到物理页映射；PA 多批次必须传，TND 不传 |
| `output_idx_offset` | INT32 `[Tq, 1]` | 否 | 加到有效 token 索引上，不得与 `-1` 哨兵冲突 |
| `metadata` | INT32 `[1024]` | 否 | 接口兼容项，不参与计算；传入时只校验 dtype / 形状 / 设备 / 连续性 |
| `topk` | int | 是 | `1 ≤ topk ≤ 8192`；等于 512 时保留编译期特化 |
| `quant_mode` | int | 是 | 只能取 1，表示 MXFP4 |
| `max_seqlen_q` | int | 否 | `≤ 0` 时取 `Tq`；用于按最大 query 数预推分核表 |
| `mask_mode` | int | 否 | 0（全可见）或 3（压缩 causal） |
| `cmp_ratio` | int | 否 | `1 ≤ cmp_ratio ≤ 128`；只在 `mask_mode=3` 时起作用 |
| `layout_q` | str | 否 | 只能是 `"TND"` |
| `layout_k` | str | 否 | `"TND"` 或 `"PA_BBND"` |
| `return_value` | bool | 否 | `True` 时额外返回分数，`False` 时 `values` 是空张量 |
| `candidate_topk_blocks` | int | 否 | 关闭候选时为 `-1`；开启时为正数，即候选容量 |
| `candidate_block_size` | int | 否 | 关闭候选时为 `-1`；开启时必须为 8 |

### 返回值说明

返回 `(indices, values, candidates, candidate_lengths)` 四个张量：

| 返回值 | 类型 / shape | 说明 |
| :--- | :--- | :--- |
| `indices` | INT32 `[Tq, 1, topk]` | 有效 token 索引叠加 `output_idx_offset`；无效槽为 `-1` |
| `values` | BF16 `[Tq, 1, topk]`，或空张量 `[0]` | `return_value=True` 时是这些位置的分数，无效槽为 `-inf` |
| `candidates` | INT32 `[Tq, 1, C]`，或空张量 `[0]` | `C = candidate_topk_blocks`；块号不叠加 offset，无效槽为 `-1` |
| `candidate_lengths` | INT32 `[Tq, 1]`，或空张量 `[0]` | 每行实际候选数量 `min(C, ceil(visible / 8))` |

`indices` / `values` 每行的前 `min(topk, visible)` 个槽是有效结果，顺序不作要求。

`indices` 里放的是**各序列内的逻辑位置**——PA 布局下也是逻辑位置，不是物理页地址；
`candidates` 里放的是同一条 key 序列上的逻辑块号。

## 约束说明

- `q` / `w` / `descale_q` 以及其它非 K 张量必须**连续存储**，且与 `q` 在同一张 NPU 上。
- `k` / `descale_k` 只允许在 K 轴（PA 是 page 轴、TND 是 token 轴）上留存储 padding：
  - TND：`k.stride()[1:] == (64, 1)`、`descale_k.stride()[1:] == (4, 2, 1)`；
    `k` 的 token stride ≥ 64、`descale_k` 的 ≥ 4。
  - PA：`k.stride()[1:] == (64, 64, 1)`、`descale_k.stride()[1:] == (4, 4, 2, 1)`；
    page stride 不小于一个完整 page（K 是 `S × 64`、scale 是 `S × 4`），且分别是 64 / 4 的倍数。
- `mask_mode=3 && cmp_ratio!=1` 与 `cmp_residual_k` 必须同时出现或同时缺省。
- 多批次 PA 必须传 `block_table`；多批次 TND 必须传 `cu_seqlens_k`；TND 不允许传 `block_table`。
- 空 query / 空 key 用**有效长度 0** 表达，而不是用零容量的存储：`k.shape[0]` 必须大于 0，
  `Tq` 与批次数也必须大于 0。
- 关闭候选时 `candidate_topk_blocks = candidate_block_size = -1`；开启时
  `candidate_topk_blocks > 0` 且 `candidate_block_size = 8`。
- 候选块的归约在「通用路径、`max_k ≤ 16384`、候选容量不超过 8192」三条同时成立时融合进主
  kernel；其余情况走分步路径（先归约块内最大分、再单独做一次选择），结果一致，多一次 launch。
- wrapper 只检查 host 侧看得到的 shape / dtype / stride，**不回读设备数据做校验**；
  唯一一次回读是把序列数组取回 host 用来预计算分核表（见 `DESIGN.md`）。
- 四个输出张量由 wrapper 内部用 `torch.empty` 分配，调用方不持有它们的生命周期控制权。

## 调用示例

以下示例假定 `torch_npu` 已初始化，张量放在当前默认 NPU 设备上。`descale` 全填 127 表示块缩放
因子取 `2^0 = 1`，读到的分数没有额外放大。

**示例 1：TND 单批次。**

```python
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "ops/pypto_pro/deepseek_v4_1/quant_lightning_indexer/op_kernel")

from quant_lightning_indexer import quant_lightning_indexer_wrapper  # noqa: E402

dev = torch.device("npu")
Tq, Lk, HALF_DIM = 6, 1024, 64          # D = 128 的逻辑维打包后是 64 字节
q = torch.randint(0, 256, (Tq, 32, HALF_DIM), dtype=torch.uint8, device=dev)
k = torch.randint(0, 256, (Lk, 1, HALF_DIM), dtype=torch.uint8, device=dev)
w = torch.rand((Tq, 32), dtype=torch.float32, device=dev)
descale_q = torch.full((Tq, 32, 2, 2), 127, dtype=torch.uint8, device=dev)
descale_k = torch.full((Lk, 1, 2, 2), 127, dtype=torch.uint8, device=dev)

indices, values, candidates, lengths = quant_lightning_indexer_wrapper(
    q, k, w, descale_q, descale_k,
    cu_seqlens_q=torch.tensor([0, Tq], dtype=torch.int32, device=dev),
    seqused_q=torch.tensor([Tq], dtype=torch.int32, device=dev),
    seqused_k=torch.tensor([Lk], dtype=torch.int32, device=dev),
    topk=512, quant_mode=1, mask_mode=0, layout_k="TND", return_value=True,
)
torch.npu.synchronize()
print(indices.shape, values.shape, candidates.shape, lengths.shape)
# torch.Size([6, 1, 512]) torch.Size([6, 1, 512]) torch.Size([0]) torch.Size([0])
```

**示例 2：PA_BBND 布局 + 压缩 causal + 候选块。** 沿用示例 1 的 `q` / `w` / `descale_q`。

```python
page, pages = 128, 8
k = torch.randint(0, 256, (pages, page, 1, HALF_DIM), dtype=torch.uint8, device=dev)
descale_k = torch.full((pages, page, 1, 2, 2), 127, dtype=torch.uint8, device=dev)
block_table = torch.arange(pages, dtype=torch.int32, device=dev).reshape(1, pages)

indices, values, candidates, lengths = quant_lightning_indexer_wrapper(
    q, k, w, descale_q, descale_k,
    cmp_residual_k=torch.tensor([0], dtype=torch.int32, device=dev),
    block_table=block_table,
    topk=512, quant_mode=1, mask_mode=3, cmp_ratio=2,
    layout_k="PA_BBND", return_value=True,
    candidate_topk_blocks=2048, candidate_block_size=8,
)
torch.npu.synchronize()
print(candidates.shape, lengths.shape)
# torch.Size([6, 1, 2048]) torch.Size([6, 1])
```

**示例 3：与 CPU 参考实现对拍。**

`tests/assets/` 下的 CPU 参考实现与精度判定脚本可以和 kernel 拼成一个自校验用例：参考实现造
输入并给出期望分数，判定脚本负责比对。

```python
import os
import sys

import torch
import torch_npu  # noqa: F401

OP = "ops/pypto_pro/deepseek_v4_1/quant_lightning_indexer"
sys.path[:0] = [os.path.join(OP, "tests", "assets"), os.path.join(OP, "op_kernel")]

import quant_lightning_indexer_golden_cpu as golden
import quant_lightning_indexer_precision as verify
from quant_lightning_indexer import quant_lightning_indexer_wrapper

dev = torch.device("npu")
args, kwargs, refs, allowed = golden.make_inputs(
    golden.CASES["pa64_causal_candidates"], dev)

indices, values, candidates, lengths = quant_lightning_indexer_wrapper(
    *args, **kwargs, return_value=True)
torch.npu.synchronize()
print(indices.shape, candidates.shape)     # [7, 1, 512] [7, 1, 2048]

# 参考实现只给出分数面；按 kernel 实际选中的索引取值，无效槽为 -inf。
want = golden.expected_scores(indices.cpu(), refs)
print(verify.check_precision(values.cpu(), want))
```

索引类的输出不能像分数那样逐元素比较，必须按集合语义判定（`verify.check_topk`）；四路输出的
完整判定见 `RESULTS.md` 的「精度标准」。
