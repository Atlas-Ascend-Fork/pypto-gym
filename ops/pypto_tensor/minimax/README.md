# MiniMax MoE / MSA Fused Operators (M2.7 / M3)

基于 PyPTO 框架实现的 MiniMax M2.7 与 M3 文本骨干网络融合算子，运行于 Ascend NPU。BF16 输入输出，FP32 累加。

## 产品支持情况

- Ascend 910B：支持
- Ascend 950 / 950PR：支持

## 文件说明

| 文件 | 说明 |
|------|------|
| `minimax_grouped_gemm_impl.py` | MoE Grouped GEMM Kernel 实现（M2.7 silu / M3 swigluoai 共用） |
| `minimax_m3_msa_indexer_impl.py` | M3 MSA lightning-INDEXER Kernel 实现（block 选择） |
| `minimax_m3_msa_sparse_attention_impl.py` | M3 MSA block-sparse DECODE attention Kernel 实现 |
| `msa_main_branch_impl.py` | M3 MSA Main Branch GQA-batched flash attention Kernel 实现 |

## 算子总览

| 算子 | 入口函数 | 适用模型 | 说明 |
|------|---------|---------|------|
| `minimax_moe_grouped_gemm` | `minimax_moe_grouped_gemm()` | M2.7 / M3 | All-experts single grouped GEMM，通过 `activation` 参数区分变体：`"silu"`（M2.7，H=3072/I=1536）或 `"swigluoai"`（M3，H=6144/I=3072），各有独立的 UB-fitting tile 默认值。 |
| `minimax_m3_msa_indexer` | `minimax_m3_msa_indexer()` | M3 | MSA lightning indexer：对每个 decode query 的 4 个 index-query head 打分，选出 top-k key block。 |
| `minimax_m3_msa_sparse_decode` | `minimax_m3_msa_sparse_decode()` | M3 | MSA block-sparse decode attention：仅在 indexer 选中的 key block 上做 flash attention。 |
| `msa_main_branch` | `msa_main_branch()` | M3 | MSA Main Branch：GQA-batched flash attention，16 个 Q head 共享 1 个 KV head，batch 进 cube M 轴。 |

所有 tile 旋钮和 swigluoai alpha/limit 均可通过环境变量覆盖（`PYPTO_VEC_TILE`、`PYPTO_CUBE_NBUFFER`、`PYPTO_VEC_NBUFFER`、`PYPTO_L1_REUSE`、`PYPTO_MM*`、`PYPTO_SWIGLU_ALPHA`、`PYPTO_SWIGLU_LIMIT`）。Grouped GEMM 通过 `USE_PTO_GROUPED_GEMM` 开关 opt-in 启用。

---

## 算子一：minimax_moe_grouped_gemm

### 算法概述

MiniMax MoE expert FFN 的融合 grouped GEMM：一次 kernel 调用完成所有 expert 的 `mm1 → activation → mm2`。`pypto.loop` 遍历 expert，`pypto.loop_unroll` 遍历 token，UB-fitting vector tile。

两个 MiniMax 变体仅 expert activation 不同，通过 `activation` 参数选择：

| 变体 | activation | 激活公式 | tile 默认值 (VEC_TILE / CUBE_NBUF / VEC_NBUF / L1) |
|------|-----------|---------|---------------------------------------------------|
| M2.7 | `"silu"` | `SiLU(gate) * up` | 128 / 2 / 2 / 2 |
| M3 | `"swigluoai"` | `(clamp(up) + 1) * (gate * sigmoid(alpha * gate))` | 256 / 4 / 1 / 3 |

swigluoai 参数（M3 config）：`alpha = 1.702`，`limit = 7.0`（env: `PYPTO_SWIGLU_ALPHA` / `PYPTO_SWIGLU_LIMIT`）。

### 循环结构

```
EXPERT_LOOP          — 遍历 expert，偏移从 expert_cumsum 动态获取
  LOOP_TOKEN (unroll)— 每个 expert 的 token 按 unroll_list 分块
    mm1: tile_x @ w13_e          → gate_up [tile_batch, 2*I] FP32
    activation (silu/swigluoai)  → cast BF16
    mm2: sw @ w2_e               → down [tile_batch, H] FP32 → cast BF16
    assemble → result
```

### 权重布局

```
F.linear 约定（输入）:
  gate_up_proj [E, 2*I, H]   — gate||up weights
  down_proj    [E, H,   I]   — down weights

convert_minimax_weights 转换后（direct matmul 格式）:
  w13_flat [E*H, 2*I] BF16   — flattened gate||up
  w2_flat  [E*I, H]   BF16   — flattened down
```

### Kernel 签名

```python
minimax_moe_grouped_gemm(
    sorted_tokens,   # [N_total, H]            BF16  — tokens pre-sorted by expert
    weights,         # (w13_flat, w2_flat)     BF16  — converted weights
    expert_cumsum,   # [E+1]                   INT32 — cumulative token counts per expert
    result,          # [N_total, H]            BF16  — output buffer
    dims,            # MoeDims(num_experts, hidden_size, intermediate_size, activation)
)
```

### Dtype 转换流程

| 阶段 | 操作 | Dtype |
|------|------|-------|
| 输入 | sorted_tokens / w13_flat / w2_flat | BF16 |
| mm1 | tile_x @ w13_e | BF16 → FP32 (out_dtype=FP32) |
| activation | silu / swigluoai | FP32 全程 |
| 中间 cast | activation 输出 → BF16 | FP32 → BF16 |
| mm2 | sw @ w2_e | BF16 → FP32 (out_dtype=FP32) |
| 输出 cast | down → result | FP32 → BF16 |

---

## 算子二：minimax_m3_msa_indexer

### 算法概述

M3 MiniMax Sparse Attention 的 lightning indexer（decode 步骤，单 query）。4 个 index-query head 对单条 (MQA) index key 全序列打分，每 128 token block 做 max-pool，再跨 4 head 取 max，最后 top-k 选出 16 个 block（含强制保留的 local block）。

PyPTO kernel 负责计算量大的 BF16 score matmul（O(ctx) 部分）；block max-pool / head-max / top-k / local 强制在 host 端 eager torch 上执行。

### 关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `NIDX` | 4 | `sparse_num_index_heads` |
| `NPAD` | 16 | cube M 轴 16 对齐，4 head 零填充到 16 |
| `D` | 128 | `sparse_index_dim` |
| `BY` | 128 | `sparse_block_size` |
| `TOPK` | 16 | `sparse_topk_blocks` |
| `LOCAL` | 1 | `sparse_local_block` |

### Kernel 签名

```python
minimax_m3_msa_indexer(
    idx_q,   # [NIDX, D]        BF16 — index-query heads (post norm + RoPE)
    idx_k,   # [nb*BY, D]       BF16 — single index key over all keys (post norm + RoPE)
    nb,      # int              — number of 128-token key blocks
)
# Returns: [1, min(TOPK, nb)] INT32 — selected block ids
```

### Block 选择逻辑

```
1. score matmul:  idx_q_pad [NPAD, D] @ idx_k [nb*BY, D]^T  → scores [NPAD, nb*BY] FP32
2. block max-pool: scores[:NIDX].view(NIDX, nb, BY).amax(-1) → [NIDX, nb]
3. head max:      .amax(0)                                  → [nb] block scores
4. top-k:         blk[:nb-LOCAL].topk(TOPK-LOCAL)           → top-(TOPK-LOCAL) non-local blocks
5. local 强制:    arange(nb-LOCAL, nb)                      → LOCAL 个最近 block
6. concat → [1, TOPK]

短上下文保护: nb <= TOPK 时返回 arange(nb)，避免 topk(k) 越界
```

---

## 算子三：minimax_m3_msa_sparse_decode

### 算法概述

M3 MSA block-sparse decode attention（Q seq-len = 1）。GQA group（`Hq // Hkv = 16` 个 query head 共享 1 个 KV head）batch 进 cube M 轴，对 indexer 选中的 key block 做 online-softmax flash attention，按 `NTILE` 宽的 chunk 迭代。当前（部分）block 通过 `valid_mask` 列掩码。

### 关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `HQ` | 64 | query head 数 |
| `HKV` | 4 | KV head 数 |
| `GROUP` | 16 | `HQ // HKV`，cube M 轴 |
| `D` | 128 | head_dim |
| `BY` | 128 | KV block size |
| `NTILE` | 512 | online-softmax chunk 宽度（env: `MSA_NTILE`） |
| `SCALE` | 1/√128 | attention scale |

### 循环结构

```
outer_loop (B*HKV, static unroll)
  nchunk_loop (sel // NTILE, static unroll)
    S = q_block @ k_ch^T * scale            [GROUP, NTILE] BF16 → FP32
    mask: valid_mask 列掩码（部分 block 置 LARGE_NEG）
    online softmax: m_c → exp → l_c → p_bf16
    O = p_bf16 @ v_ch                        [GROUP, D] FP32
    累加器更新 (mi, li, oi)
  out = oi / li → cast BF16 → assemble
```

### Kernel 签名

```python
minimax_m3_msa_sparse_decode(
    q,          # [B, HQ, D]           BF16 — decode query (post norm + RoPE)
    k_blocks,   # [B, HKV, nb, BY, D]  BF16 — paged KV cache keys
    v_blocks,   # [B, HKV, nb, BY, D]  BF16 — paged KV cache values
    block_ids,  # [B, topk]            INT  — indexer-selected block ids
    seq_len,    # int                  — total KV length
)
# Returns: [B, HQ, D] BF16 attention output
```

### 性能说明

对于 M3 decode 形状，原生 `npu_fused_infer_attention_score`（paged block-sparse）目前比此 PyPTO kernel 快约 5x。此 kernel 为 PyPTO-native MSA 路径，原生 op 为性能目标。

---

## 算子四：msa_main_branch

### 算法概述

M3 MSA Main Branch：GQA-batched flash attention。16 个 Q head 共享 1 个 KV head，batch 进 cube M 维（per query-block tile），K/V/mask 每个 chunk 加载一次并被 16 个 Q head 复用，大幅降低 HBM 带宽。

循环顺序 `h_kv → n_block → g(inner chunk) → c(kv chunk)`，使 `oi` 保持片上（单 head 一次），K/V/mask 在外层循环加载并跨所有 16 个 Q head 复用。

参考：MiniMax M3 Technical Report (arXiv:2606.13392v2), Equation 8。

### 关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `HQ` | 64 | query head 数 |
| `HKV` | 4 | KV head 数 |
| `GROUP` | 16 | `HQ // HKV` |
| `D` | 128 | head_dim |
| `BK` | 128 | KV block size |
| `TOPK` | 16 | selected KV block 数 |
| `_MAX_N` | 2048 | 最大 query 序列长度 |
| `_NTILE` | 512 | KV chunk 宽度（env: `MSA_NTILE`） |
| `_NQ_TILE` | 128 | query-block tile（env: `MSA_NQ_TILE`） |

### 循环结构

```
LOOP_HG (total_heads = HKV * GROUP)
  c_loop (nchunks = kv_len // NTILE)
    raw = q_all @ k_ch^T                    [MAX_N, NTILE] → FP32
    scaled = raw * scale
    masked = scaled + mask_ch               (causal mask 预计算于 host)
    online softmax: m_c → exp → l_c → p_cast
    pv = p_cast @ v_ch                       [MAX_N, dh] FP32
    累加器更新 (mi, li, oi)
  out = oi / li → assemble
```

### Causal Mask

所有 causal 逻辑在 host 端预计算进 `block_mask [num_blocks, topk, bk, bk]`，kernel 内无动态条件分支（PyPTO `AssignMemoryType` pass 要求）。掩码规则：
- `kv_seq > qb`：`MASK_NEG`（block 不可达）
- `kv_seq == qb`：lower-triangular causal mask
- `kv_seq < qb`：`0.0`（全注意力，无掩码）

### Kernel 签名

```python
msa_main_branch(hq, hkv, dh, bk, topk)(query, key_blocks, value_blocks, block_mask, output)
# query:       [N, HQ, D]               — decode query
# key_blocks:  [topk*bk, HKV, D]        — gathered KV blocks
# value_blocks:[topk*bk, HKV, D]        — gathered KV blocks
# block_mask:  [bk*topk, kv_len] FP32   — 预计算 causal mask (transposed+reshaped)
# output:      [N, HQ, D] FP32
```

### Dtype 支持

通过 `MSA_DTYPE` 环境变量选择：`fp32`（默认）/ `fp16` / `bf16`。

---

## 测试用例

### minimax_moe_grouped_gemm（M2.7 / silu）

| 用例 | E | H | I | counts | 说明 |
|------|---|---|---|--------|------|
| case_001 | 8 | 3072 | 1536 | [0,1,2,4,8,16,32,0] | 混合 token 数，含零 token expert |

```bash
export TILE_FWK_DEVICE_ID=0
python3 tests/ops/minimax_m27/test_minimax_m27_grouped_gemm.py
# 或 pytest
python3 -m pytest tests/ops/minimax_m27/test_minimax_m27_grouped_gemm.py
```

### minimax_moe_grouped_gemm（M3 / swigluoai）

| 用例 | E | H | I | counts | 说明 |
|------|---|---|---|--------|------|
| case_smoke_edge | 4 | 256 | 128 | [1,0,3,4] | 不均匀路由 + swigluoai clamp 覆盖 |

```bash
python3 tests/ops/minimax_m3/test_minimax_m3_grouped_gemm.py
python3 -m pytest tests/ops/minimax_m3/test_minimax_m3_grouped_gemm.py
```

### minimax_m3_msa_indexer + sparse_decode

| 用例 | nb | 说明 |
|------|----|------|
| test_indexer_selection_identical_to_torch[8] | 8 | nb≤TOPK 短上下文保护 |
| test_indexer_selection_identical_to_torch[17] | 17 | 最短 sparse 路径 |
| test_e2e_indexer_plus_attention | 17 | indexer→attention 端到端（skip，需空闲 die） |
| test_attention_forward_pypto_msa_matches_native_paged_decode | — | HF attention 集成（skip） |

```bash
python3 -m pytest tests/ops/minimax_m3/test_minimax_m3_msa_pypto.py -v
```

### msa_main_branch

| 配置 | 值 |
|------|-----|
| HQ | 64 |
| HKV | 4 |
| TOPK | 16 |
| N | 2048 |
| HEAD_DIM | 128 |
| BK | 128 |

```bash
export TILE_FWK_DEVICE_ID=0
python3 tests/ops/minimax_m3/test_msa.py

# 带泳道图采集
COLLECT_SWIMLANE=1 python3 tests/ops/minimax_m3/test_msa.py

# msprof 性能采集
python3 tests/ops/minimax_m3/profile_msa.py
python3 tests/ops/minimax_m3/profile_golden_msa.py
```

### 精度校验

- Grouped GEMM：`numpy.testing.assert_allclose`，rtol=0.008~0.015，atol=0.008~0.5（视变体和 clamp 覆盖而定）
- MSA indexer：选中的 block id 集合与 torch reference **完全一致**（identical）
- MSA sparse decode / main branch：max_diff < 5e-2 / atol_abs=1e-3, atol_rel=1e-3

## 运行方式

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# M2.7 Grouped GEMM (silu)
python3 -m pytest tests/ops/minimax_m27/

# M3 Grouped GEMM (swigluoai) + MSA indexer/decode
python3 -m pytest tests/ops/minimax_m3/

# MSA Main Branch
python3 tests/ops/minimax_m3/test_msa.py
```

## 依赖

- Python 3.x
- PyTorch + torch_npu
- PyPTO (`pypto` 包)
- NumPy
- pytest（可选，用于 skip 标记和参数化）
