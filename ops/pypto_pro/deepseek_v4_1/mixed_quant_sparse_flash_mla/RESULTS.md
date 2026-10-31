# mixed_quant_sparse_flash_mla 结果记录（PyPTO-Pro）

对应实现：`op_kernel/mixed_quant_sparse_flash_mla.py`（4565 行，SHA-256
`8762c22c32eb1b80e9ac74633c0fd728ee96d38b1035c1374604ebf5b66f632a`）。精度与性能数据均在
CANN 9.3.0 环境下、设备 `Ascend950PR_9589`（32 Cube / 64 Vector）实测；分核数由运行期
`cube_core_num` 决定（本机 32），不再固定 28。接口与约束见 [`README.md`](README.md)，
结构与流水设计见 [`DESIGN.md`](DESIGN.md)。

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

**结论：13/13 P0 PASS**（CANN 9.3.0、32 核全量）。逐 case 判定写入
`tests/reports/pro_pypto/validation.json`。

## 性能结果

### 绝对性能结果

设备 `Ascend950PR_9589`（32 Cube / 64 Vector），分核数由运行期 `cube_core_num` 规划；msprof
`task duration`，每个 case 独立进程 `warmup 2 + repeat 12`，取后 10 次中位（µs）：

| case | T1 | 时延 (µs) | 同机 blocks=28 (µs) |
| :--- | --: | --: | --: |
| cmp_only_rows | 3 | 6.45 | 6.45 |
| single_token_k | 4 | 6.88 | 6.95 |
| ori_only | 2 | 7.07 | 7.05 |
| narrow_k_tail | 4 | 7.58 | 7.65 |
| omit_lengths | 2 | 9.63 | 9.72 |
| mixed_decode | 2 | 9.68 | 9.71 |
| empty_and_tail | 8 | 9.91 | 9.92 |
| padded_pages | 5 | 9.96 | 10.06 |
| multirow_42 | 42 | 15.40 | 15.04 |
| **decode_1** | 32 | **15.51** | 21.58 |
| **selective_fd_72** | 72 | **31.94** | 32.86 |
| **decode_6_long_s2** | 72 | **33.09** | 34.46 |
| **decode_6** | 192 | **68.35** | 77.16 |

**FD 归约优化**：两条 72 行用例启用 2-way FlashDecode（各 24 split）。归约核取"在本核最后一行
结束"的 producer，其 O chunk 仍在片上，只从 GM 读回远端那一半（`fd_tail` 按地址别名复用主循环
staging 区），归约开销从 ~3.8µs 降到 ~2.4µs（同会话 A/B）；当前 `selective_fd_72` 31.94µs、
`decode_6_long_s2` 33.09µs。

**核数敏感性**：收益集中在"行数 > 28 且能被 32 整除得更整齐"的用例——`decode_1`（32 行，1 波
vs 2 波）21.58→15.51µs（−28%），`decode_6`（192 行，6 波 vs 7 波）77.16→68.35µs（−11%）；
`selective_fd_72`、`decode_6_long_s2` 分别 −2.8% / −4%，其余用例在 ±3% 抖动内。`decode_1` 在
28 核时还会触发 20 路 FlashDecode 切分，32 核下临界路径已均衡、FD 关闭，走 uniform 整行路径。

**测量协议**（固定，改动前后必须一致）：`warmup 2 + repeat 12` 取后 10 次中位；单轮全量 sweep
的绝对值有 ±1% 抖动，因此**任何改动都走同会话交错 A/B**（base/cand 交替、≥2 轮取中位），
只有 ≥1% 且稳定的收益（或纯删死代码不回归）才保留。

### 瓶颈画像（decode_6）

32 核（block_dim=32，mix=64）下瓶颈在 **AIV 侧**：`mte2` 77.3%、`vec` 62.0%、`scalar`
46.3%、`mte3` 27.7%；AIC 侧 `mac` 51.6%、`mte1` 40.9%、`fixpipe` 22.9%、`scalar` 25.5%，
`aicore_time` ≈ 86.2µs 与 `aiv_time` ≈ 82.1µs 基本齐平。与本轮改造前（`mte2` 70.3%、`vec`
62.5%）相比，staging 管道占用提高约 7pp——建表与 dequant 的 tile 级同步解耦后搬运更连续，
这也是 decode_6 从 71.2µs 降到 68.3µs 的来源。cannsim 主循环 trace（28 核会话，kernel 主体
未改）佐证：**无全局气泡**（各管道 busy 区间并集 100%），VF push 的 52% 来自 4 个反量化
site（VF 密度已触底）；MTE2 空闲段主要在等 vec——即**是 vec 拖住 MTE2，不是反过来**。

