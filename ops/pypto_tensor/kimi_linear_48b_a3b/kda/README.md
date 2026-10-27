# KDA — Kimi Delta Attention (chunk + fused_decode) PyPTO kernels

Fused PyPTO Ascend-NPU implementation of the **Kimi Delta Attention (KDA)** linear-attention
core used by `KimiDeltaAttention` in Kimi-Linear-48B-A3B-Instruct. KDA is a refined gated
delta rule with a **per-channel (fine-grained) log-gate** `g ∈ [-5, 0]`.

Two kernels:

| File | Op | Path | Default |
|------|----|------|---------|
| `kda_chunk_impl.py` | `kda_chunk_wrapper` | prefill (`mode == 'chunk'`) | dispatched when `USE_PTO_KDA` |
| `kda_fused_decode_impl.py` | `kda_fused_decode_step` | decode | called from `kda_decode_graph.py` |

The wrapper mirrors the upstream `fla.ops.kda.chunk_kda` signature and is `@allow_in_graph`
(torch.compile / aclgraph capture compatible).

## Math (per (b,h), state `S = [V,K]`, state_v_first)

```
q,k L2-normalized: x / (sqrt(sum(x*x)) + 1e-6)
g       = log-space gate (lower_bound(-5) * sigmoid(...)),  alpha = exp(g)
S'      = S * alpha                       # Diag(alpha) decay on the K axis
pred    = S' @ k                          # [V]
delta   = beta * (v - pred)               # [V]
S_new   = S' + delta ⊗ k                  # outer product -> [V,K]
o       = scale * (S_new @ q)             # [V],  scale = K**-0.5
```

The **chunk** kernel runs this recurrence in blocks: cumulative gate `G = cumsum(g)`,
pairwise decay `exp(G_i - G_j)` for `j <= i`, triangular inverse `(I + A_kk)^-1` (via
nilpotent doubling), and inter-/intra-chunk state carry. See the `kda_chunk_impl.py`
module docstring for the per-subchunk equations.

### Numerical stability — subchunk = 16
The real KDA gate accumulates to `G ≈ -640` over a 128-step chunk; factorizing the
intra-chunk decay as `q·exp(G)` / `k·exp(-G)` overflows fp32 (`exp(+640) = inf → NaN`).
The chunk kernel processes each chunk in **subchunks of `SUB = 16` steps**, carrying the
recurrent state across subchunks. Within a 16-step subchunk the cumulative-gate range is at
most `16*5 = 80`, so `exp(80) ≈ 5.5e34 < 3.4e38` stays in fp32 range and never overflows.
Mathematically identical to `chunk_size = 16`.

### Kernel dtype — FP32 (intentional, ≠ model bf16)
The kernel computes in **FP32**, not the model's native bfloat16. This is a deliberate
stability choice, not an oversight: the per-channel log-gate's `exp(±G)` factorization
needs FP32 headroom (see above), and bf16 (~3 mantissa bits) would lose the delta-rule
state. The q/k/v/g casts are done **host-side** (`.float()`) so there is no dependency on
NPU-side bf16↔fp32 `aclnnCast`; the output is cast back to the input dtype by the wrapper.

## Shapes / constraints

| Param | Value |
|-------|-------|
| heads `H` | 32 (validated 4 / 32) |
| head_k_dim `K` | 128 |
| head_v_dim `V` | 128 |
| chunk subchunk `SUB` | 16 |
| `use_qk_l2norm_in_kernel` | `True` (L2 norm done on host before packing) |
| dtype | q/k/v/g bf16/fp16/fp32 in, fp32 state, out cast back to input dtype |

FLA layout: `q,k,g [B,T,H,128]`, `v [B,T,H,128]`, `beta [B,T,H]`, `state [B,H,V,K]` fp32.
Out-of-envelope shapes (`K != 128`, `V != 128`,
`use_qk_l2norm_in_kernel=False`) raise `NotImplementedError`; the modeling layer then falls
back to the torch chunk/recurrent path.

**Single-NPU binding.** PyPTO binds a JIT kernel to one NPU per process (the device of the
first launch). For a model sharded across NPUs via `device_map`, the wrapper raises
`NotImplementedError` for any KDA layer whose tensors are on a different NPU, so those layers
fall back to torch and only the bound device's KDA layers run on PyPTO. Full multi-NPU PyPTO
coverage requires one process per NPU (pipeline/tensor parallel).

## Accuracy

Checked against the `_naive_recurrent_kda` golden (and the `vec_chunk_kda` torch chunked
op) by the tests in `tests/ops/kimi_linear_48b_a3b/`, on Ascend 910B3:

- chunk: `B ∈ {1,2}`, `H ∈ {4,32}`, `T ∈ {64,128,300,512,1000}`, realistic gate `g=-5·rand`
  AND worst-case `g=-5·ones` (cumulative `≈ -640`) — **max abs error ~6.1e-5** (bf16,
  vs the fp32 naive-recurrent golden; well within the 6e-3 test tolerance).

Test inputs match the real model's dtypes (probed from `KimiDeltaAttention.forward`, see
`tests/.../test_cases.json` `probe`): q/k/v bf16, **g and beta fp32** (`fused_kda_gate` /
`.float().sigmoid()`), `initial_state` fp32.

> **验证范围说明（scenario-B 限制）：** 被替换的原算子是 `fla.ops.kda.{chunk_kda,
> fused_recurrent_kda}`（Triton/CUDA 实现）。`fla`/Triton 在 Ascend 环境**不可用**
> （`modeling_kimi.py` 的 `ImportError` 分支即据此回退到 `kimi_fla_compat` 的纯 torch
> 实现）。因此**无法**在本硬件上直接对比 kernel 与**真实 fla 融合算子**的输出。
>
> 替代基准采用**两个相互独立的参考实现**交叉校验，而非单一“算法等价重写”——
> 这一点很关键，可避免“参考与 kernel 共享同一算法误解、却互相吻合”的循环验证盲区：
> 1. **`_naive_recurrent_kda`（主 golden）**——逐时间步朴素递归，直接照 KDA 定义式
>    （per-channel 对数门 `g`、`alpha=exp(g)`、delta-rule 外积更新、q/k L2norm、`scale=K**-0.5`，
>    见 `kda_chunk_impl.py` 模块 docstring 与 Kimi-Linear KDA 数学定义）实现，**不含**任何
>    分块/矩阵求逆技巧。kernel 的精度即是对拍它（chunk 6.1e-5）。
> 2. **`vec_chunk_kda`（第二参考）**——分块矩阵算法（subchunk=16，nilpotent 三角求逆），
>    与 kernel 同构但纯 torch；它独立地对拍 `_naive_recurrent_kda`（≤6e-5）。
>
> 由于①是定义式逐步递归、②是分块算法，二者**结构独立**：kernel 同时吻合两者，已排除
> “实现误解被参考掩盖”的主要风险。**残余风险**仅剩一项且已界定：①②均是 KDA 数学的
> **本仓库转写**，未与上游 CUDA `fla.ops.kda` kernel 逐位对拍。**关闭方式**（需 CUDA 机器、
> 非本 Ascend 环境）：在一台 GPU 上跑一次 `vec_chunk_kda` vs 真实 `fla.ops.kda.chunk_kda`
> 的差异并记录于此；在拿到该数据前，本集成按“已披露的已知差距”对待。

## Performance

The fused chunk kernel reduces KDA prefill cost vs the torch chunked op, with the gain
growing with sequence length. Decode (T=1 recurrent) is roughly parity, so it is left on the
upstream torch path; PyPTO covers the prefill/chunk path only.

## ACLGraph / 图捕获

The chunk kernel is registered with `torch.library` as `pypto::kda_chunk_kimi` (Meta + NPU
keys; see the registration block at the bottom of `kda_chunk_impl.py`, mirroring
`phi_3_mini_4k_instruct/rms_norm`). The Meta ("fake") impl gives a graph tracer the output
shape/dtype rule without launching the kernel. `kda_chunk_pypto` is the capturable entry point
(`torch.ops.pypto.kda_chunk_kimi`); the default eager dispatch still uses `kda_chunk_wrapper`,
so this adds capture-readiness without touching the verified eager path.

The wrappers do **not** call `torch_npu.npu.synchronize()` — the kernel enqueues on the
current stream and downstream ops are stream-ordered after it, so a device-wide sync is
unnecessary (and would abort aclgraph capture). Verified safe: op tests are bit-identical
with the sync removed, and the full-model bench keeps identical KDA coverage.

**Verified** (`tests/ops/kimi_linear_48b_a3b/test_kda_aclgraph.py`, on NPU):
- registered NPU op == eager wrapper, **bit-identical** (chunk);
- Meta shape inference correct under `FakeTensorMode`;
- the op composes into a **`torch.compile(fullgraph=True)`** graph with **no graph break**
  (`aot_eager` + `eager`, bit-exact);
- **real torchair aclgraph capture** (`torch_npu.dynamo.torchair`, `get_npu_backend`,
  `mode="reduce-overhead"`) **captures + replays bit-exact** (`diff=0`) — this is the
  npugraph/stream-capture path, which needs **no GE converter**.

**Remaining for whole-model aclgraph:** the KDA op itself is capture-ready (above). Running
the *entire* Kimi model under a single torchair `reduce-overhead` graph additionally
requires every other op (MoE routing, attention, conv) to be capture-clean; that end-to-end
model capture is the next step and is not yet validated here. (torchair's separate GE-graph
mode is a different path that *would* need a per-op GE converter — not pursued, since the
npugraph capture above already works for this fused kernel.)

## Files

| File | Description |
|------|-------------|
| `kda_chunk_impl.py` | Chunk/prefill kernel + `kda_chunk_wrapper` (`@allow_in_graph`) + `torch.library` reg (`pypto::kda_chunk_kimi`, Meta+NPU) + `kda_chunk_pypto` |
| `_device_guard.py` | Shared single-NPU bind guard (partial-coverage fallback) |
| `__init__.py` | Package init — re-exports `kda_chunk_wrapper` (eager) and `kda_chunk_pypto` (capturable) |
| `README.md` | This document |

## 状态

| 维度 | 状态 |
|------|------|
| 单算子精度 | ✅ chunk 测试通过（PRECISION_PASS）— `tests/ops/kimi_linear_48b_a3b/test_kda_chunk.py`，max abs err 6.1e-5 |
| 整网集成 | ✅ 已接入 — `USE_PTO_KDA` 开关 + `sys.modules` 注入 + `NotImplementedError` → 上游 torch fallback；端到端可跑通，PyPTO 运行间结果可复现（逐字节一致）。与 torch 参考 **非** bit-exact（kernel 与 torch 参考**均为 fp32 计算 + bf16 I/O**，差异仅来自实现/算子顺序的舍入，单算子 ~6.1e-5；greedy argmax 在 token 级可能分叉） |
| ACLGraph | ✅ 算子级已验证 — chunk kernel `torch.library` 注册（`pypto::kda_chunk_kimi`，Meta+NPU），且 wrapper 去除了多余的 `torch_npu.npu.synchronize()`（kernel 走 stream 顺序，去同步后算子测试逐位一致、整网 bench KDA 覆盖不变且更快）。`test_kda_aclgraph.py` 验证：注册 op == wrapper 逐位一致、Meta 推导正确、`torch.compile(fullgraph)` 无 graph break、**真实 torchair aclgraph（reduce-overhead）capture+replay 逐位一致（diff=0，无需 GE converter）**。整网 aclgraph（其余算子也需 capture-clean）为后续工作 |
| 性能调优 | ✅ chunk nested-64 已调优（subchunk=16，prefill 1.05–1.26x）。详见上级 `modeling/transformers/kimi_linear_48b_a3b/README.md` 性能对比 |

---

## KDA Fused Decode Kernel

**Full KimiDeltaAttention layer fusion for decode:**

Combines 11 operations into one kernel:

**Input projections (3 matmuls):**
1. q_proj: [B, H] @ [H, P]^T → [B, P]
2. k_proj: [B, H] @ [H, P]^T → [B, P]
3. v_proj: [B, H] @ [H, P]^T → [B, P]

**KDA core (7 ops):**
4. Causal depthwise conv1d (K=4) + SiLU
5. q/k L2 normalization  
6. Decay gate: `lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))`
7. Update weight: `beta = sigmoid(b_raw)`
8. Delta-rule state update with per-channel gate
9. RMSNorm (head_dim)
10. Sigmoid output gate

**Output projection (1 matmul):**
11. o_proj: [B, P] @ [P, H]^T → [B, H]

### Performance

**Full model decode benchmark (27 layers, 20 KDA, 4 NPUs):**

|           | Baseline (torch) | Fused (PyPTO)     | operator gain |
|-----------|------------------|-------------------|---------------|
| **eager** | 88.58 ms/token   | **73.30 ms/token**| **−17%**      |
| **graph** | 29.80 ms/token   | **16.38 ms/token**| **−45%**      |


### Accuracy

**Verified against torch golden (`_naive_recurrent_kda`):**

- Fused decode: max abs error ~6e-3
- Graph capture: verified (state restored correctly)

### Graph Capture

- `kda_fused_decode_step`: allocates nothing, raises nothing
- All intermediates in caller-owned `KdaFusedBuffers`
- In-place state update
- Compatible with `torch.npu.graph`

### Usage

```bash
MODEL_PATH=/data/models/Kimi-Linear-48B-A3B-Instruct \
  torchrun --nproc_per_node=4 bench_kimi_multinpu.py --mode decode \
    --pypto --graph --route uniform --report-file perf.json 
```

### Test

```bash
export ASCEND_RT_VISIBLE_DEVICES=6
python3 -m pytest tests/ops/kimi_linear_48b_a3b/test_kda_fused_decode.py
```

### Key Features

✅ **45% latency reduction** for decode path  
✅ **All 20 KDA layers accelerated** (pipeline parallel)  
✅ **Numerical accuracy maintained** (~6e-3 max error)  
✅ **Graph capture compatible**
