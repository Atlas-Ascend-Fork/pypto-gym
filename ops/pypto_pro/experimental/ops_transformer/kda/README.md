# KDA (Kernelized Decay Attention) Kernels

KDA 算子的 5 个 PyPTO-Pro block kernel 实现，覆盖 6 个计算 stage（gate_cumsum 与 kkt 融合为一个 kernel）。

## Kernel 列表

| Stage | 文件 | Kernel | Wrapper | 描述 |
|---|---|---|---|---|
| 1+2 | `gate_kkt_kda_impl.py` | `gate_kkt_kda_kernel` | `run_gate_kkt_kda` | 融合 gate_cumsum + kkt_kda，Phase 3 M-split |
| 3 | `inversion_kda_impl.py` | `inversion_kda_cube_kernel` | `inversion_kda_cube` | Neumann 级数矩阵求逆 (I+L)^{-1} |
| 4 | `wy_kda_impl.py` | `wy_kda_kernel` | `wy_kda_block` | WY 表示：u = A_inv @ V_scaled, w = A_inv @ K_eff |
| 5 | `chunk_h_kda_impl.py` | `chunk_h_kda_kernel` | `run_chunk_h_kda` | 顺序递归状态传递 S_new = exp(g)*S + k_rest^T @ v_corr |
| 6 | `chunk_o_kda_impl.py` | `chunk_o_kda_kernel` | `run_chunk_o_kda` | 输出计算 o = (q*exp(g_cs))@S + tril(q_eff@k_eff^T)@v_corr |

## Pipeline 流程

```
gate_kkt_kda → inversion_kda → wy_kda → chunk_h_kda → chunk_o_kda
   (g→g_cs→L)    ((I+L)^{-1})    (u, w)     (S, v_corr)      (o)
```

## 常量

| 常量 | 值 | 说明 |
|---|---|---|
| C | 128 | Chunk size |
| K | 128 | Key 维度 |
| V | 128 | Value 维度 |
| HC | 64 | Half chunk (sub-block 分片) |
| HV | 4 | Value heads 数 |

## 测试

测试文件位于 `tests/ops/pypto_pro/experimental/ops_transformer/kda/`：

- `test_gate_kkt_kda.py` — Stage 1+2 单 kernel 测试
- `test_inversion_kda_cube.py` — Stage 3 单 kernel 测试
- `test_wy_kda.py` — Stage 4 单 kernel 测试
- `test_chunk_h_kda.py` — Stage 5 单 kernel 测试
- `test_chunk_o_kda.py` — Stage 6 单 kernel 测试
- `test_kda_pro_e2e.py` — 端到端 pipeline 测试
- `ref_kda.py` — CPU golden 参考实现 (RefKDA 类)
- `kda_test_config.py` — 共享测试配置
