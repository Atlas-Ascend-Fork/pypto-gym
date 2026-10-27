# mixed_quant_sparse_flash_mla 结果记录（PyPTO-Pro）

对应实现：`op_kernel/mixed_quant_sparse_flash_mla.py`（5076 行，SHA-256
`464bd4102b0a4da0cc1b8c718a9f633713b21da8fcb739d9c51e09e85ddb00e5`）。精度与性能数据均在
CANN 9.2 环境下、设备 `Ascend950PR_9579`（28 Cube / 56 Vector）实测。接口与约束见
[`README.md`](README.md)，结构与流水设计见 [`DESIGN.md`](DESIGN.md)。

## 用例

13 条 P0，全部为真机精度用例（`tests/test_mixed_quant_sparse_flash_mla.py`，`--cases` 可子集、
`--list` 列出）。除 `ori_only` 外均启用压缩侧（cmp）支路；`K1` 为原始侧 TopK，`K2` 为压缩侧
TopK；`s_ori`/`s_cmp` 为两侧 KV 窗口长度（页数 × `bs`）。用例名与 `--list` 输出一致：

| case | T1 | K1 | K2 | s_ori | s_cmp | 覆盖点 |
| :--- | --: | --: | --: | --: | --: | :--- |
| mixed_decode | 2 | 128 | 129 | 257 | 193 | 基线：两侧齐备、非整 tile 尾块 |
| ori_only | 2 | 128 | — | 257 | — | 仅原始侧（`cmp_kv=None`，cmp 分支被编译期消除） |
| empty_and_tail | 8 | 128 | 129 | 257 | 193 | 逐行长度不一（含空行）+ 尾块 |
| padded_pages | 5 | 128 | 129 | 257 | 193 | `bs_ori=64`/`bs_cmp=32`、KV 页间 padding（axis-0 非连续） |
| single_token_k | 4 | 1 | 1 | 257 | 193 | 每行仅 1 列有效（`k` 边界） |
| narrow_k_tail | 4 | 17 | 65 | 257 | 193 | 窄 K + 尾块（非 128 倍数） |
| cmp_only_rows | 3 | 128 | 129 | 257 | 193 | 原始侧整侧为 0（`zero_ori`） |
| multirow_42 | 42 | 128 | 129 | 257 | 193 | 多行分核（T1 不是核数整数倍） |
| selective_fd_72 | 72 | 128 | 512 | 257 | 769 | 长压缩侧 + 选择性 FlashDecode 规划 |
| omit_lengths | 2 | 128 | 129 | 257 | 193 | 省略 `ori_topk_length`/`cmp_topk_length`（全 K 有效） |
| decode_1 | 32 | 128 | 512 | 32768 | 16384 | 生产规模长 KV 窗口、完整块表 |
| decode_6 | 192 | 128 | 512 | 32768 | 16384 | 最大用例：192 行 × 完整块表 |
| decode_6_long_s2 | 72 | 128 | 512 | 131072 | 65536 | 长序列：块表宽 1024（页数上界附近） |

## 精度标准

**独立参考实现**：`tests/assets/mixed_quant_sparse_flash_mla_golden_cpu.py`，纯 torch 实现
算子契约的全部数学——直接从合同字节池解码反量化（FP8 E4M3FN group32 / FP4 E2M1 group16）、
128-token 分块在线 softmax（含 per-head sink）、BF16 概率舍入、FP32 累加、FP64 `logsumexp`
求 LSE；不读取设备输出、不与 kernel 共享中间量。13 条 P0 的输入参数以
`_DEFAULT_PARAMS` + `_CASE_OVERRIDES` 内联在该文件中。

**比较器**：`tests/assets/mixed_quant_sparse_flash_mla_precision.py`（自包含），判据
`pro_scheme_a`，阈值冻结、不可由测试输入覆盖。逐项检查：

1. **out**（BF16，按 FP32 比较）：按方案 A 误差判据逐元素比对；
2. **lse**（FP32）：`rtol=atol=1e-5` 直接比较；空行（两侧有效长度均为 0）要求输出**严格为 0**、
   LSE **严格等于** `sinks`；
3. **有限性与污染门**：输出必须全部有限，且无残留 NaN（输出初始化为 NaN，未写回区域可被检出）。

**结论：13/13 P0 PASS**（CANN 9.2 全量）。逐 case 判定写入
`tests/reports/pro_pypto/validation.json`。

## 性能结果

### 绝对性能结果

设备 `Ascend950PR_9579`（28 Cube / 56 Vector），msprof `task duration`，每个 case 独立进程
`warmup 2 + repeat 10`，取后 10 次中位（µs）：

| case | T1 | 时延 (µs) |
| :--- | --: | --: |
| cmp_only_rows | 3 | 6.27 |
| ori_only | 2 | 6.64 |
| single_token_k | 4 | 6.85 |
| narrow_k_tail | 4 | 7.94 |
| mixed_decode | 2 | 9.44 |
| omit_lengths | 2 | 9.46 |
| padded_pages | 5 | 9.99 |
| empty_and_tail | 8 | 10.38 |
| multirow_42 | 42 | 16.07 |
| decode_1 | 32 | 23.72 |
| selective_fd_72 | 72 | 33.76 |
| decode_6_long_s2 | 72 | 35.56 |
| decode_6 | 192 | 80.75 |

**测量协议**（固定，改动前后必须一致）：`warmup 2 + repeat 10` 取后 10 次中位；单轮全量 sweep
的绝对值有 ±1% 抖动，因此**任何改动都走同会话交错 A/B**（base/cand 交替、≥2 轮取中位），
只有 ≥1% 且稳定的收益（或纯删死代码不回归）才保留。

### 瓶颈画像（decode_6）

瓶颈在 **Vector**：`vec` ≈ 58µs 是占用最高的 AIV 管道，且 `wall ≈ vec`；`mte2` ≈ 63µs
（散列 KV staging）、`scalar` ≈ 39µs、`mte3` ≈ 25µs、`fixpipe` ≈ 24µs。cannsim 主循环 trace
佐证：**无全局气泡**（各管道 busy 区间并集 100%），MTE2 占用 86%、PUSHQ 81%，VF push 的 52%
来自 4 个反量化 site（VF 密度已触底）；MTE2 空闲的 12.9µs 中 6.8µs 是等 vec——即
**是 vec 拖住 MTE2，不是反过来**。

