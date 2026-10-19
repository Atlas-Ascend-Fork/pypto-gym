# LLaDA2.0-mini MoE — NPU Migration

## Model Info

| Field | Value |
|------|-----|
| HuggingFace | inclusionAI/LLaDA2.0-mini |
| Architecture | Diffusion-LM Mixture-of-Experts |
| Key dims | H=2048, I=512, E=256 |
| Precision | BF16 weights, FP32 accumulation |
| Generation | Block-wise masked diffusion |

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
export MODEL_PATH=/path/to/LLaDA2.0-mini

# download weights
python3 ../download_hf_model.py \
    --model-id inclusionAI/LLaDA2.0-mini \
    --output-dir "$MODEL_PATH"

# patch the checkpoint to the in-repo model definition
python3 ../runtime_patch.py \
    --model-family llada2_moe \
    --model-path "$MODEL_PATH"

# inference (baseline)
python3 scripts/ask_LLaDA2-mini.py --model-path "$MODEL_PATH" --device <NPU>

# inference (PyPTO fused ops)
python3 scripts/ask_LLaDA2-mini.py --model-path "$MODEL_PATH" --device <NPU> --use_pypto

# benchmark (baseline vs PyPTO)
MODEL_PATH="$MODEL_PATH" DEVICE=0 bash bench_LLaDA2-mini.sh
```

## Archive Mapping

| Source | Target (`{pypto_gym_repo}/`) | Action |
|---|---|---|
| `ask_*.py` / `bench_*.py` / `bench_*.sh` | `modeling/transformers/llada2_moe/` | overwrite |
| `configuration_llada2_moe.py` / `modeling_llada2_moe.py` | `src/pypto_gym/transformers/llada2_moe/` | overwrite |
| fused operators (`*_impl.py`) | `ops/pypto_tensor/llada2_moe/` | overwrite |
| operator tests | `tests/ops/llada2_moe/` | new |
