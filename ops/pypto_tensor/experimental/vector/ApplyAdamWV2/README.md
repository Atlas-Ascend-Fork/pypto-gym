# apply_adam_w_v2

PyPTO custom kernel implementing one AdamW optimizer step (with bias correction
and decoupled weight decay) on Ascend NPU.

```
m_t   = beta1 * m + (1 - beta1) * g
v_t   = beta2 * v + (1 - beta2) * g * g
m_hat = m_t / (1 - beta1**t)
v_hat = v_t / (1 - beta2**t)
update = m_hat / (sqrt(v_hat) + eps) + weight_decay * w
w_new  = w - lr * update
```


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## Files

- `SPEC.md` — operator requirements, shapes, dtypes, tolerance.
- `API_REPORT.md` — PyPTO API mapping and feasibility analysis.
- `DESIGN.md` — kernel design (tiling, loop, precision routing).
- `apply_adam_w_v2_golden.py` — pure-PyTorch reference implementation.
- `apply_adam_w_v2_impl.py` — PyPTO kernels + host wrapper.
- `test_apply_adam_w_v2.py` — precision test (level0 fp32, level1 bf16).
- `test_cases.json` — test case configuration.

## Shape & dtype

- `weight`, `grad`: `[7168, K]`, `bfloat16` or `float32` (K dynamic, 2048-24576).
- `m`, `v`: `[7168, K]`, `float32`.
- Outputs match input dtypes; m/v outputs always fp32.

## Usage

```python
from apply_adam_w_v2_impl import apply_adam_w_v2_wrapper

w_new, m_new, v_new = apply_adam_w_v2_wrapper(
    weight, grad, m, v,
    beta1=0.9, beta2=0.999, lr=1e-3,
    weight_decay=0.01, eps=1e-8, step=1,
)
```

The wrapper precomputes `bc1=1-beta1**step`, `bc2=1-beta2**step`,
`one_m_b1=1-beta1`, `one_m_b2=1-beta2` on the host and dispatches to either
`apply_adam_w_v2_kernel_fp32` or `apply_adam_w_v2_kernel_bf16` depending on
the dtype of `weight`.

## Running the precision test

```
cd <repo-root>
source env_setup.sh
python3 tests/ops/experimental/vector/ApplyAdamWV2/test_apply_adam_w_v2.py
```

Pass tolerance: `atol=1e-4`, `rtol=0.0078125`. The test prints
`[PRECISION_PASS]` (exit 0) or `[PRECISION_FAIL]` (exit 1).

## Implementation notes

- Single K-axis loop with `pypto.loop`; `valid_shape=[7168, (K-k_off).min(N_TILE)]`
  handles tail blocks (no-op when K is a multiple of `N_TILE`).
- `pypto.set_vec_tile_shapes(1, 1024)` configures vector tiling.
- bf16 path casts to fp32 at entry (`pypto.cast(..., DT_FP32)`) and casts the
  weight result back to bf16 just before `pypto.assemble`. m_new / v_new always
  stay in fp32.
- Three outputs are written via three independent `pypto.assemble` calls in the
  same kernel iteration.
- `step` only enters the kernel through the precomputed `bc1` / `bc2` floats —
  no device-side `pow`.

## 2026-06-11 Update

- Kernel now supports dynamic `M` and dynamic `K` 2D tensors, not only `M=7168`.
- Added small/tail shape verification levels `level5` to `level7`.
- Large-M strategy is retained for `[7168, K]` network-style cases to avoid regressing existing performance coverage.
