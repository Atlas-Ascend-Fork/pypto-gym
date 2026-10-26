# KDA — Kimi Delta Attention (fused_prefill + fused_decode) PyPTO kernels

Fused PyPTO Ascend-NPU implementation of the **Kimi Delta Attention (KDA)** linear-attention
core used by `KimiDeltaAttention` in Kimi-Linear-48B-A3B-Instruct. KDA is a refined gated
delta rule with a **per-channel (fine-grained) log-gate** `g ∈ [-5, 0]`.

Two kernels:

| File | Op | Path | Default |
|------|----|------|---------|
| `kda_fused_prefill_impl.py` | `kda_fused_prefill` | prefill (`mode == 'chunk'`) | dispatched when `USE_PTO_KDA` |
| `kda_fused_decode_impl.py` | `kda_fused_decode_step` | decode | called from `kda_decode_graph.py` |

Both wrappers are `@allow_in_graph` and stream-ordered (NPU-graph capturable). The
reference FLA/torch path (`chunk_kda` / `fused_recurrent_kda` via `kimi_fla_compat`) is
preserved as the fallback: out-of-envelope shapes raise `NotImplementedError` and the
modeling layer falls back.

## Fused prefill (`kda_fused_prefill_impl.py`)

The single PyPTO prefill operator: one fused PyPTO kernel covers the whole
per-layer KDA attention core between the projections and `o_proj` —
`conv1d(k=4)+silu → l2norm → gate transform → beta sigmoid → WY-chunked gated
delta rule → o_norm (RMSNorm × sigmoid gate) → bf16 out`.

## Numerical stability

A naive exponential factorization can overflow in fp32 for long decay ranges.
The fused kernel uses 16-row pivot bands to keep intra-chunk decay exponents within a safe range,
while cross-chunk decay remains non-positive and is safe by construction.

## Shapes / constraints

| Param | Value |
|-------|-------|
| heads `H` | 32 (validated 4 / 32) |
| head_k_dim `K` | 128 |
| head_v_dim `V` | 128 |
| WY chunk | 128 tokens (static) |
| q/k/v | bf16 `[B,T,H,128]` pre-conv projections (host packs to fp32) |
| g_raw / g_gate / b_raw | projection outputs, any float dtype (bf16 in the model) |
| state dtype | fp32 `[B,H,V,K]` |
| output dtype | bf16 in-kernel; wrapper casts back to the input dtype if needed |
 
Unsupported shapes or execution modes raise `NotImplementedError`, and
`_try_kda_fused_prefill` falls back to the reference torch prefill path.

**Single-NPU binding.** PyPTO binds a JIT kernel to one NPU per process (the device of the
first launch). For a model sharded across NPUs via `device_map`, the wrapper raises
`NotImplementedError` for any KDA layer whose tensors are on a different NPU, so those layers
fall back to torch and only the bound device's KDA layers run on PyPTO. Full multi-NPU PyPTO
coverage requires one process per NPU (pipeline/tensor parallel).

## Accuracy

Checked against the torch composition the operator fuses (conv1d+silu → gate →
beta → `_naive_recurrent_kda` core → o_norm) by
`tests/ops/kimi_linear_48b_a3b/test_kda_fused_prefill.py`, on Ascend 910:
 
- `B ∈ {1,2}`, `H ∈ {4,32}`, `T ∈ {128, 300, 512}` — exact chunk, non-%128
  tail and multi-chunk state carry; the remaining cases cover H=32, B>1,
  worst-case gate (`g ≈ -5` everywhere), no-initial-state and the
  continuing-prefill conv-cache path. All compared against an **all-fp32
  golden** (fp32 conv+silu, exact naive recurrence, fp32 o_norm):
  dO ≤ 1e-2 (≈1 bf16 ulp — the single final round-to-bf16, i.e. the
  format floor), dS ≤ 1e-5, conv-state **bit-exact** (it seeds decode).
 
Test inputs match the real model's dtypes: q/k/v bf16 (pre-conv projections),
g_raw/g_gate/b_raw bf16 (projection outputs), dt_bias/A_log/o_norm.weight fp32
parameters, initial_state fp32.

> **验证范围说明（scenario-B 限制）：** 被替换的原算子是 `fla.ops.kda.{chunk_kda,
> fused_recurrent_kda}`（Triton/CUDA 实现）。`fla`/Triton 在 Ascend 环境**不可用**
> （`modeling_kimi.py` 的 `ImportError` 分支即据此回退到 `kimi_fla_compat` 的纯 torch
> 实现）。因此**无法**在本硬件上直接对比 kernel 与**真实 fla 融合算子**的输出。
>
> The substitute baseline is `_naive_recurrent_kda` — a plain per-timestep recurrence
> straight from the KDA defining equations, with no chunking or matrix inversion, so it
> is structurally independent of the kernel's blocked algorithm. Cross-check:
> dS ≤ 3.9e-6, dO ≤ 8.8e-3 (≈1 bf16 ulp, the format floor).
>
> **Residual risk:** the golden is this repo's own transcription of the KDA math and
> has not been bit-compared with the upstream CUDA `fla.ops.kda` kernel. **To close it**
> (needs a CUDA machine): run `kimi_fla_compat.vec_chunk_kda` vs the real
> `fla.ops.kda.chunk_kda` once on a GPU and record the result here; until then this is
> a disclosed, known gap.

## Graph capture
 
The fused wrappers do **not** call `torch_npu.npu.synchronize()` — the kernel enqueues
on the current stream and downstream ops are stream-ordered after it, so a device-wide
sync is unnecessary (and would abort NPU-graph capture). The prefill dispatch lives at
the fused-subgraph level (`_try_kda_fused_prefill` in `modeling_kimi.py`); its
try/except resolves to the fused branch during warmup + capture, so a captured stage
bakes in the fused op. Capture+replay of the operator is regression-tested in
`test_kda_fused_prefill.py::test_kda_fused_prefill_graph_capture` (bit-exact vs
eager); the full-model `--graph` bench captures every KDA layer through this
path. The decode stage is captured via `kda_decode_graph.py`'s
`StaticDecodeCache` + `kda_fused_decode_step`.

## Files

| File | Description |
|------|-------------|
| `kda_fused_prefill_impl.py` | Fused prefill operator — `kda_fused_prefill` (conv+silu, gate, beta, WY-chunk KDA core, o_norm; one kernel) |
| `kda_fused_decode_impl.py` | Decode kernel + `kda_fused_decode` (eager, guarded) + `kda_fused_decode_step` (capture-safe) |
| `_device_guard.py` | Single-NPU binding guard shared by both wrappers |
| `__init__.py` | Package init — re-exports `kda_fused_prefill` (prefill) and `kda_fused_decode`/`kda_fused_decode_step` (decode) |

## 状态

| 维度 | 状态 |
|------|------|
| 单算子精度 | ✅ `tests/ops/kimi_linear_48b_a3b/test_kda_fused_prefill.py` — vs 全 fp32 golden: dO ≤ 8.8e-3 (≈1 bf16 ulp), dS ≤ 3.9e-6, conv-state bit-exact |
| 整网集成 | ✅ `USE_PTO_KDA` + `sys.modules` injection + fused-subgraph dispatch (`_try_kda_fused_prefill`) with `NotImplementedError` → torch fallback |
| ACLGraph | ✅ no device-wide sync; NPUGraph capture+replay regression-tested in `test_kda_fused_prefill.py`; full-model `--graph` bench captures every KDA layer |
| 性能 | Benchmarked with `bench_kimi_multinpu.py --mode prefill --pypto [--graph]` |

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
