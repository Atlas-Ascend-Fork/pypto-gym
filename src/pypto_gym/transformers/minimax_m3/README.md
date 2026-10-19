# MiniMax-M3 PyPTO Kernel Integration

Routes the MiniMax-M3 **text-backbone** MoE expert FFN through the PyPTO fused grouped-GEMM
operator on Ascend 910B. MiniMax-M3 is a native vision-language MoE model (~428B total / ~23B
activated); only the text MoE expert FFN is accelerated here (128 routed experts, top-4, sigmoid +
`e_score_correction_bias` routing, **swigluoai** activation). Values verified against
the MiniMax-M3 checkpoint `config.json` (`text_config`) and the real expert weight names
(`language_model.model.layers.N.block_sparse_moe.experts.*.{w1,w2,w3}`, BF16).

Like `minimax_m27`, the text backbone is defined **in-repo** (`MiniMaxM3ForCausalLM`, no
`trust_remote_code`) — the official HF checkpoint ships only config + processors (served via
vLLM / sglang), and `transformers` has no native `minimax_m3_vl` arch, so we vendor it. The same
module also exposes the PyPTO patch (`patch_moe`) and the FP8-streaming loader (`load_model`).
Structure is validated against the real checkpoint: the loader's key remap matches 22665/22665
text tensors with 0 missing / 0 unexpected.

## Integration scope

| Operation | PyPTO integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| Routed expert FFN (grouped GEMM) | Yes | Per-expert eager loop | All experts in one kernel (`minimax_moe_grouped_gemm` (`activation="swigluoai"`)); FP32 accumulation, **swigluoai** in the vector stage |
| Router gating | No | host MoE block | sigmoid + `e_score_correction_bias` top-4; `routed_scaling_factor=2.0` applied host-side |
| Shared expert | No | host (`shared_experts.{gate,up,down}_proj`) | always-on, untouched by the patch |
| RMSNorm / qk-norm | No | host | Gemma-style `(1+w)` norm, per-head qk-norm |
| Attention (MSA) | **Modelled** | dense GQA (short ctx) | MiniMax Sparse Attention: lightning indexer (`index_{q,k}_{proj,norm}`) selects top-`sparse_topk_blocks` key blocks/query. Numerically identical to dense GQA for short ctx (≤ `topk·block`), block-sparse for long ctx. |

`patch_minimax_m3_moe(block, streaming=...)` rebinds the expert container's `forward` to the PyPTO
path when `USE_PTO_GROUPED_GEMM` is enabled — a single cumsum-indexed grouped GEMM over all experts,
eliminating the per-expert loop. `patch_moe(model)` walks every block that passes the
`is_minimax_m3_moe` duck-type check and patches it. Two weight paths: **prebuilt** (dequant all
experts to BF16 flats once, MoE memory ~1x) and **streaming** (experts kept on the host, only the
routed experts dequantized per forward, so one die holds ~one layer's active experts).

## Switch variable

`USE_PTO_GROUPED_GEMM` (in `ops/pypto_tensor/minimax/__init__.py`) — the single switch
gating whether the MoE FFN routes through the fused kernel (default `False`; the entry scripts enable
it per run). Tile / activation parameters are env-exposed (`PYPTO_VEC_TILE`, `PYPTO_CUBE_NBUFFER`,
`PYPTO_VEC_NBUFFER`, `PYPTO_MM1_*`, `PYPTO_MM2_*`, `PYPTO_SWIGLU_ALPHA`, `PYPTO_SWIGLU_LIMIT`);
`PYPTO_VEC_TILE=256` (the M3-tuned default) keeps the swigluoai/cast vector tile within the 910B
192 KB UB (M3's H=6144 makes the cap mandatory).

## Loading & quantization

`load_model(model_path, device, use_pypto, streaming, max_layers)` builds the in-repo backbone on
meta, loads the checkpoint (stripping the `language_model.` prefix; skipping vision / projector /
MSA-indexer / MTP tensors), keeps the routed experts FP8 on the host, and patches the MoE blocks.

The full BF16 checkpoint (~856 GB) does **not** fit one 910B die, so the benchmark uses an **FP8
weight-only** checkpoint (e.g. `MiniMaxAI/MiniMax-M3-MXFP8`): the kernel is BF16-compute, so
weight-only FP8 (block-dequant to BF16 via the existing `_dequant_fp8_block`) halves expert memory
with no activation-quant needed — there is no INT8 kernel, so W8A8's `A8` would buy nothing here.
The grouped-GEMM kernel and its precision test (`tests/ops/minimax_m3/`) are independent of the
loader and validate on the NPU today; an on-NPU E2E parity run is the remaining step.
