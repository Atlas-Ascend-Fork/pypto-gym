# kda_flash fused operators

PyPTO fused-operator implementations for Key-Delta-Attention inference. Each operator
lives in its own subdirectory with an `_impl.py` exporting a `*_wrapper`
function and a `README.md` describing the contract.


## 产品支持情况

- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## Operators

| Operator | Subdirectory | Status |
|----------|--------------|--------|
| `fused_recurrent_kda` | [`fused_recurrent_kda/`](fused_recurrent_kda/) | Available |
| `chunk_kda` | [`chunk_kda/`](chunk_kda/) | Available |

## Common shape constraints

| Symbol | Meaning | Value |
|--------|---------|-------|
| `D`   | Per-head dimension              | 128 |
| `H`   | Head count                       | Dynamic |
| `K`   | Key dimension (= D)              | 128 |
| `V`   | Value dimension (= D)            | 128 |

## Integration

The kernels are wired into the target model through the monkey-patch hook
pattern — see each operator's README for details.
