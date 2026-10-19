# Gemma-4-31B-it — NPU Migration

## Model Info

| Field | Value |
|------|-----|
| HuggingFace | google/gemma-4-31b-it |
| Architecture | Gemma4ForConditionalGeneration (VLM; text-only inference supported) |
| Params | ~31B |
| Precision | bfloat16 |

## Environment

| Field | Version |
|------|------|
| torch_npu | 2.10 |
| transformers | 5.12.0 |
| pypto | 0.2.1 |
| pto-isa | v9.1.0 |
| CANN | 9.1.0 |
| NPU | Ascend 910B3 |

## Usage

```bash
export MODEL_PATH=/path/to/gemma-4-31b-it

# download weights
python3 ../download_hf_model.py \
    --model-id google/gemma-4-31b-it \
    --output-dir "$MODEL_PATH"

# patch the checkpoint to the in-repo model definition
python3 ../runtime_patch.py \
    --model-family gemma4_31b_it \
    --model-path "$MODEL_PATH"

# inference (baseline)
python3 scripts/ask_Gemma-4-31B-it.py --model-path "$MODEL_PATH" --device <NPU>

# inference (PyPTO fused ops)
python3 scripts/ask_Gemma-4-31B-it.py --model-path "$MODEL_PATH" --device <NPU> --use_pypto

# benchmark (baseline vs PyPTO)
DEVICE=0 MODEL_PATH="$MODEL_PATH" bash bench_Gemma-4-31B-it.sh
```

## Archive Mapping

| Source (`{model_dir}/`) | Target (`{pypto_gym_repo}/`) | Action |
|---|---|---|
| `ask_Gemma-4-31B-it.py`, `bench_Gemma-4-31B-it.sh` | `modeling/transformers/gemma4_31b_it/` | overwrite |
| `configuration_gemma4.py`, `modeling_gemma4.py` | `src/pypto_gym/transformers/gemma4_31b_it/` | overwrite |
| `pto_kernels/` (excl. golden + test) | `ops/pypto_tensor/gemma4_31b_it/` | overwrite |
| `pto_kernels/*/test/` | `tests/ops/gemma4_31b_it/` | new |
