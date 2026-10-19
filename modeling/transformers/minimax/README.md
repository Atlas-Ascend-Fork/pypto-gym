# MiniMax (M2.7 / M3) — NPU Migration

MiniMax M2.7 and M3 share one bench/runner set, selected by `--variant {m27,m3}`.
The MoE expert FFN is routed through the shared PyPTO fused grouped-GEMM operator
(routed experts kept FP8 on the host, dequantized per layer / streamed).

## Model Info

| Field | M2.7 (`--variant m27`) | M3 (`--variant m3`) |
|------|------|-----|
| hidden_size (H) | 3072 | 6144 |
| moe_intermediate_size (I) | 1536 | 3072 |
| num_experts (E) | 256 | 128 |
| top_k | 8 | 4 |
| Expert activation | SiLU-SwiGLU (`silu`) | clamped GLU (`swigluoai`) |
| Code source | in-repo (no trust_remote_code) | in-repo (no trust_remote_code) |
| Weight precision | FP8 block-quant (streamed dequant to BF16) | FP8 (MXFP8; streamed dequant to BF16) |

## Environment

| Field | Version |
|------|------|
| torch_npu | 2.10 |
| transformers | 4.57.1 |
| pypto | 0.2.1 |
| pto-isa | v9.1.0 |
| CANN | 9.1.0 |
| NPU | Ascend 910B3 |

## Usage

```bash
# --- M2.7 ---
export MODEL_PATH=/path/to/MiniMax-M2.7
python3 ../download_hf_model.py --model-id MiniMaxAI/MiniMax-M2.7 --output-dir "$MODEL_PATH"
python3 ../runtime_patch.py --model-family minimax_m27 --model-path "$MODEL_PATH"

# --- M3 ---
export MODEL_PATH=/path/to/MiniMax-M3
python3 ../download_hf_model.py --model-id MiniMaxAI/MiniMax-M3 --output-dir "$MODEL_PATH"
python3 ../runtime_patch.py --model-family minimax_m3 --model-path "$MODEL_PATH"

# inference (variant selects the model + activation)
MODEL_PATH=/path/to/MiniMax-M2.7 python3 scripts/ask_minimax.py --variant m27 --device <NPU>
MODEL_PATH=/path/to/MiniMax-M3   python3 scripts/ask_minimax.py --variant m3  --device <NPU> --use_pypto

# E2E generation benchmark (single die)
MODEL_PATH=/path/to/MiniMax-M2.7 python3 scripts/bench_minimax.py --variant m27 --device <NPU> --max-layers 2
MODEL_PATH=/path/to/MiniMax-M3   python3 scripts/bench_minimax.py --variant m3  --device <NPU> --max-layers 6 --use_pypto

# NPUGraph capture/replay decode benchmark
MODEL_PATH=/path/to/MiniMax-M3 python3 scripts/bench_minimax.py --variant m3 --graph --use_pypto

# shell driver (variant via VARIANT env or first positional arg)
MODEL_PATH=/path/to/MiniMax-M3 VARIANT=m3 DEVICE=0 bash bench_minimax.sh
```

> `--model-path` overrides `$MODEL_PATH`. For M3, layers 0-2 are dense and layer 3+ are MoE, so
> `--max-layers >= 4` is needed to exercise the grouped-GEMM path.

## Archive Mapping

| Source | Target (`{pypto_gym_repo}/`) | Action |
|---|---|---|
| `ask_minimax.py` / `bench_minimax.py` / `bench_minimax.sh` | `modeling/transformers/minimax/` | overwrite |
| `modeling_minimax_m27.py` | `src/pypto_gym/transformers/minimax_m27/` | overwrite |
| `modeling_minimax_m3.py` | `src/pypto_gym/transformers/minimax_m3/` | overwrite |
| grouped-GEMM / MSA operators | `ops/pypto_tensor/minimax/` | overwrite |
| operator tests | `tests/ops/minimax_m27/`, `tests/ops/minimax_m3/` | new |
