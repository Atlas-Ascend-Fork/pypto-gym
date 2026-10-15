# IndexerPrologueK 结果说明（RESULT）

> 实现：`op_kernel/indexer_prologue_k.py`；用例：`tests/test_indexer_prologue_k.py`；
> 参考/精度基准（golden）：`tests/assets/indexer_prologue_k_golden.py`。

## 1. 用例文件

`tests/test_indexer_prologue_k.py`（共 60 例）。CPU 用例（golden / MXFP4 / cache ABI）无需设备始终可跑；
NPU 端到端用例在无设备时自动跳过（`@requires_npu`）。用例矩阵与规格：

| 用例 | 规格（T, H, D, Dr, blocks, block_size, mode/G, scale）| 覆盖点 |
|---|---|---|
| `test_projection_m_tail` | T∈{1,15,16,17,33}, H=512, D=128 | 投影 M 尾块（`valid_m`）|
| `test_projection_shape_fallbacks` | 16, (H,D)∈{(48,128),(128,96),(64,32),(512,128)} | `base_n`/`base_k` 回退路径 |
| `test_projection_axes_pick_aligned_bases` | — | host：`projection_axes` 的类别选择 |
| `test_device_encoder_matches_reference_on_random` | 6, 512, 128, 64, 2, 64, mode0+scale | 随机端到端 |
| `test_device_encoder_shapes` | 5, 512, (D,Dr)∈{(128,64),(32,16),(96,32)}, 2, 64, mode0+scale | 形状泛化 |
| `test_norm_rope_boundaries` | 13, 512, 128, Dr∈{2,16,62,64,122,128}, 2, 128, mode0+scale | RoPE 短尾/全尾 |
| `test_norm_rope_small_head_dim` | 16, 64, 32, 16, 2, 128, mode0+scale | 小头维 |
| `test_duplicate_index_last_token_wins_mode0/1` | 8, 512, 128, 64, 2, 128, mode0 / mode1(G=4) | 重复 index「最后写入」|
| `test_out_of_order_indices` | 10, 512, 128, 64, 2, 128, mode0+scale | 乱序 index |
| `test_skip_all_tokens_leaves_cache_untouched` | 5, 512, 128, 64, 2, 128, mode0+scale（全 -1）| 全跳过不改写 |
| `test_mode1_combined_rows` | 16, 512, 128, 64, 2, 128, mode1 G∈{2,4,8,16} | 组合行布局 |
| `test_untouched_cache_region_sentinel` | 4, 512, 128, 64, 2, 64, mode0+scale, idx=[0,5,70,-1] | 未命中 slot 哨兵 |
| `test_token_boundaries` | T∈{1,15,16,17}, 512, 128, 64, 2, 128, mode0+scale | token 边界 |
| `test_shape_generalisation` | 16, (H,D,Dr)∈{(48,128,64),(128,96,32),(64,32,16),(512,128,122)}, 2, 128, mode0+scale | 综合泛化 |
| `test_mode0_without_scale_cache` | 9, 512, 128, 64, 2, 128, mode0（无 scale）| 不写 scale cache |
| `test_public_api_returns_same_tensor` | 4, 512, 128, 64, 2, 64, mode0+scale | 原地更新并返回同一 tensor |
| `test_large_reference_shape[_mode1]` | 4, 7168, 512, 510, 2, 64, mode0+scale / mode1(G=4) | 模型大规格 |
| `test_fused_matches_two_kernel_dispatch` | 4096, 512, 128, 64, 32, 128, mode0+scale / mode1(G=8) | 融合与两 kernel 两条路径逐字节一致 |

## 2. 精度标准

- **逐字节相等**：NPU 端到端输出与 `tests/assets/indexer_prologue_k_golden.py`（bit-exact CPU 参考）
  用 `torch.equal` 比较，`k_cache` 与 `k_scale_cache` 均要求完全一致。
- **三处 BF16 发布边界**与参考实现严格一致：GEMM 输出、RMSNorm 输出、尾部 RoPE 输出。
- **MXFP4** 使用参考实现的 exponent 位操作（`safe_amax·(1/6)` 倒数乘而非除法）+ E2M1 阈值链 +
  符号位/nibble 顺序，因此 scale 字节与 E2M1 码可逐位比较。
- **大 T 的个别 tie**：极少数元素落在 E2M1 同号相邻码的阈值 tie 上，可能与参考相差 ±1 码（参考容许，
  非语义差异）。小/中 T 与 golden 逐字节一致。

## 3. 性能数据

环境：Ascend950PR（cube/vector 32/64）；CANN 9.3.0；模型规格 `H=7168, D=512, Dr=510, block=128`
（`value_bytes=256`、`scale_bytes=16`）；`torch.npu.Event` 取中位；派发阈值 `FUSED_T_MAX=2048`。

| case | T | storage_mode | 派发 | `k_cache` 形状 | Pro (ms) |
|---|---|---|---|---|---|
| decode | 16 | 0 | fused | `(1, 128, 1, 256)` | 0.781 |
| wide | 512 | 0 | fused | `(4, 128, 1, 256)` | 0.675 |
| prefill | 4096 | 0 | two | `(32, 128, 1, 256)` | 1.637 |
| combined | 4096 | 1（G=8）| two | `(32, 16, 1, 2176)` | 1.467 |

融合相对两 kernel 的收益（同进程对比，H=7168/D=512）：`T ≤ 2048` 约 **1.3–2.1×**
（T=64 2.08×、T=16 1.82×、T=512 1.76×）；`T ≥ 4096` 融合约 **慢 8–15%**（T=4096 0.92×、T=8192 0.88×），
故 `> FUSED_T_MAX` 走两 kernel。

## 4. 理论评估与对比

### 4.1 计算量与访存量（模型规格 H=7168, D=512, 每 token）

| 项 | 公式 | 每 token | T=4096 |
|---|---|---|---|
| 投影 FLOPs | `2·D·H` | 7.34 MFLOP | 3.0 GFLOP |
| 投影读 latent | `H·2B` | 14.3 KB | 58.7 MB |
| 投影读 wk（T 无关）| `D·H·2B` | — | 7.3 MB |
| 中间 projected | `D·2B`(写) + `D·2B`(读) | 2.0 KB | 8.2 MB |
| 后处理写 cache | `(D/2 + D/32) B` | 272 B | 1.1 MB |
| 后处理算术（norm/rope/mxfp4）| `~3·D` | ~1.5 K ops | ~6 M ops |

### 4.2 分段耗时构成与 bound

- **投影**：`m_tiles = ceil(T/16)`、`n_tiles = ceil(D/128) = 4`，K 循环 `ceil(H/128)=56` 次。
  - 小 T（m_tiles < AIC 数）时 tile 数不足以填满 32 个 Cube 核，且单 tile 内是 56 步串行
    K 累加 → **延迟受限**，耗时与 T 基本无关（≈0.44 ms）。
  - 大 T 时 tile 数填满核、并行充分 → 耗时随 T 近线性增长（T=4096≈0.83 ms、T=8192≈1.41 ms）。
- **后处理**：块级 owner 遍历 + 每行 SIMT 依赖链，小/中 T 由**每 kernel 固定开销**主导
  （≈0.55 ms，T=1..2048 近乎不变）；随 T 增大由 header/owner 扫描与行处理共同决定。
- **实测佐证**：`single ≈ burst`（单次与连发均 ≈0.5 ms），说明该固定量是**真实设备耗时**而非 launch 间隙；
  host 侧查询（`get_effective_core_counts`）≈0.0005 ms，可忽略。

### 4.3 激活核数（后处理）敏感性

同一份已编译后处理 kernel，仅改变 launch 的 vector block 数（=激活核数）：

| T | 1 核 | 2 | 5 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 5 | 0.555 | 0.550 | 0.554 | 0.550 | 0.549 | 0.578 | 0.550 |
| 512 | 1.278 | 0.909 | 0.674 | 0.622 | 0.584 | 0.593 | 0.680 |
| 8192 | 13.913 | 7.114 | 3.064 | 2.196 | 1.249 | 0.890 | 0.743 |

→ **大 T 激活核数是主导因素**（T=8192 从 1 核 13.9 ms → 64 核 0.74 ms，≈19×）；**小 T 与核数无关**（固定开销主导）。

### 4.4 理论小结

- 小/中 T：两阶段耗时几乎与 T 无关，均由**每 kernel 固定开销**主导 → 单 kernel 融合（省一次下发）
  收益最直接；`T ≤ 2048` 实测约 1.3–2.1×。
- 大 T：耗时随 T 增长、由计算/访存与 owner 扫描共同决定，激活核数主导 → 融合的全局屏障无收益，
  故走两 kernel。
- 综合：按 `T` 阈值派发可同时拿到小 T 的固定开销收益与大 T 的并行/带宽收益。

## 5. 数据来源

- 实现/设计：`op_kernel/indexer_prologue_k.py`、`DESIGN.md`。
- 精度基准：`tests/assets/indexer_prologue_k_golden.py`；用例：`tests/test_indexer_prologue_k.py`（60 passed）。
- 性能采集脚本（会话产物，未入库）：`proj/post/e2e` 分段、`grid sweep`、`fused vs two`。
