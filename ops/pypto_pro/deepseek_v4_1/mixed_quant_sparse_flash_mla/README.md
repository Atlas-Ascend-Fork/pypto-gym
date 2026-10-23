# mixed_quant_sparse_flash_mla

## 概述

`mixed_quant_sparse_flash_mla`（MQSMLA，混合量化稀疏 Flash MLA）基于 `pypto_pro`
DSL 在 Ascend 950 上实现。对 BF16 Query、FP8 原始 KV 与可选 FP4 压缩 KV 执行稀疏
Attention，**单 kernel** 融合反量化、QK、在线 Softmax（含每 head sink）、PV 与输出
写回，并支持整行分核与 metadata 指定的 FlashDecode（FD）分片归约。

| 模式 | 输入组合 |
| :--- | :--- |
| 原始 KV 稀疏注意力 | `ori_kv`、`ori_sparse_indices`、`ori_block_table` |
| 混合 KV 稀疏注意力 | 同时提供原始侧与压缩侧的 KV、索引与页表 |

对每个 Query head，按稀疏索引经块表收集两侧有效列的反量化 KV，合并为 $X$（同一 $X$
既作 Key 也作 Value）：令 $z_j=\text{softmax\_scale}\,q^TX_j$，则

```text
Z = exp(sinks_h) + Σ_j exp(z_j)   O = (Σ_j exp(z_j)·X_j) / Z   LSE = log Z
```

sink 只入分母、不贡献 Value；两侧有效长度均为 0 时输出严格为 0、LSE 严格等于该 head
的 sink。反量化：ORI = FP8 E4M3FN group32（512 数据字节 + 16 个 BF16 scale = 544B）；
CMP = FP4 E2M1 group16（256 打包字节 + 32 个 BF16 scale = 320B，偶数位置在低 nibble）。

## 目录结构

```
mixed_quant_sparse_flash_mla/
├── README.md                       # 文档
├── DESIGN.md                       # 设计（tile/数据流/同步）
├── SPEC.md                         # 机器合同（p0_cases / 形状 / 容差）
├── op_kernel/
│   └── mixed_quant_sparse_flash_mla.py  # 内核 + host metadata 规划器 + 公开 wrapper
└── tests/
    ├── test_mixed_quant_sparse_flash_mla.py     # 测试入口（pytest + CLI）
    └── assets/
        ├── mixed_quant_sparse_flash_mla_golden_cpu.py  # 独立 CPU golden（数学 oracle）
        └── mixed_quant_sparse_flash_mla_precision.py   # 自包含精度比较器（方案 A）
```

## 接口签名

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

| 参数 | dtype | shape | 说明 |
| :--- | :--- | :--- | :--- |
| `q` | BF16 | `[T1, 64, 512]` | Query（TND） |
| `ori_kv` | uint8 | `[pages, bs, 1, 544]` | FP8 E4M3FN group32 原始 KV |
| `ori_sparse_indices` | int32 | `[T1, 1, K1]` | 稀疏索引；合法前缀外的列忽略（可为 -1） |
| `ori_block_table` | int32 | `[B, bt]` | 逻辑页 → 物理页 |
| `ori_topk_length` | int32 / None | `[T1, 1]` | 每行有效 TopK（省略 = 全 K 有效） |
| `sinks` | FP32 | `[64]` | 每 head sink |
| `metadata` | int32 | `[1024]` | `..._metadata` 的输出 |
| `cmp_kv` | uint8 / None | `[pages, bs, 1, 320]` | FP4 E2M1 group16 压缩 KV（省略 = 关闭 cmp 臂） |
| `cmp_sparse_indices` / `cmp_block_table` / `cmp_topk_length` | int32 / None | 同上 | 压缩侧 |
| `cu_seqlens_q` | int32 | `[B+1]` | Query 前缀和，`cu[0]=0`、末值 `T1` |
| `out` / `lse` | BF16 / FP32 | `[T1, 64, 512]` / `[1, T1, 64]` | 未传时由 wrapper `torch.empty` 分配 |

`seqused_*` 仅做类型/形状校验（与参考一致，不参与计算）。KV 允许页间 padding
（axis-0 非连续，页内维度连续）。

## 设计要点

- **单 kernel、单 launch。** 全部计算在唯一一个 `@pl.jit(auto_mutex=True)` MIX kernel
  `mixed_quant_sparse_flash_mla_kernel` 内，用 `with pl.section_cube()` / `with
  pl.section_vector()` 划分 CV 边界并靠核间事件握手。公开 wrapper 只做参数校验、
  `torch.empty` 分配、派生运行时标量并发射一次；**不做任何 host 侧数据运算**。
- **tilingkey 只承载条件。** `MqsmlaTilingKey` 仅 4 个 1-bit 条件：`HAS_ORI_LEN` /
  `HAS_CMP_LEN` / `HAS_CMP` / `FD_ENABLE`（可选输入存在性 + FlashDecode）。所有数值参数
  （`softmax_scale` / `strides` / `b_num` / `k` / `bs` / `bt`）走运行时标量；条件在编译期
  折叠，被门控分支消除（FD 关闭时 FD 归约整段不编译）。
- **全动态维。** 21 个张量维均为 `pl.DYNAMIC`，**每个 tilingkey 只编译 1 份**产物
  （形状无关），13 条 P0 只产生 3 个二进制。
- **可选输入 `pl.Ptr`。** 缺省输入直接传 `None`（运行时空指针判断），无需占位张量。
- **gather 双缓冲。** staging 池按 group 奇偶乒乓，消除 `stage(g+1)` 等 `dequant(g)` 的
  WAR，使 MTE2 与向量并行。
- **尾块/空行。** 行内按 128-token tile 处理，尾 tile 的无效列地址 clamp、softmax 排除、
  P 写 0；空行两 AIV 直接写 `out=0`、`lse=sinks`。

## 精度

SPEC `pro_scheme_a`：与独立 CPU golden（`tests/assets/mixed_quant_sparse_flash_mla_golden_cpu.py`，
纯 torch，直接从合同字节池解码反量化）逐 case 比对；附加准出：LSE `rtol=atol=1e-5`、
空行输出严格 0 且 LSE 严格等于 sinks、输出有限、污染门（无残留 NaN）。比较器
`tests/assets/mixed_quant_sparse_flash_mla_precision.py` 自包含。**13/13 P0 PASS**。

## 性能

设备 `Ascend950PR_9579`（28 Cube / 56 Vector），同机 msprof `task duration`，warmup+15
取中位（µs）：

| case | T1 | 时延 |
| :--- | --: | --: |
| mixed_decode | 2 | 10.6 |
| empty_and_tail | 8 | 11.4 |
| multirow_42 | 42 | 18.6 |
| selective_fd_72 | 72 | 40.4 |
| decode_1 | 32 | 26.2 |
| decode_6 | 192 | 101.3 |
| decode_6_long_s2 | 72 | 43.0 |

## 运行

```bash
D=ops/pypto_pro/deepseek_v4_1/mixed_quant_sparse_flash_mla

# 全部 P0 case（精度矩阵；写出 tests/reports/pro_pypto/validation.json）
python $D/tests/test_mixed_quant_sparse_flash_mla.py

# 子集
python $D/tests/test_mixed_quant_sparse_flash_mla.py --cases mixed_decode,decode_1

# pytest
pytest $D/tests -q
```

测试用 `tests/assets` 的 CPU golden 造输入并生成 metadata，在设备上调用公开入口后
与 golden 逐 case 比对，全程不依赖任何仓库外路径。
